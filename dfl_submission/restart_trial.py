"""Tune and confirm set-piece restart anchors on external DFL matches (DECISIONS.md D4)."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from dfl_submission import augment, decoder_trial, motion_trial, restarts, retrain_trial
from dfl_submission.external_only import require_external_only
from dfl_submission.temporal_trial import Match
from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN, write_json
from dfl_transfer_data import digest, timestamp
from evaluation import possession_metrics as pm


PROTOCOL = "DECISIONS.md D4"
EVENT_SOURCES = Path(__file__).with_name("dfl_event_sources.json")
DFL_RESTARTS = {"ThrowIn", "CornerKick", "GoalKick", "FreeKick", "KickOff"}
PROFILES = {"moderate": (.5, 1, 2.5, .1), "heavy": (.75, 2, 5, .25)}
DRAWS = 3
GRID = tuple(dict(method=method, seconds=seconds, strength=strength)
             for method, strengths in (("evidence", (.5, 1, 2)), ("transition", (.5, 1)))
             for seconds, strength in product((2, 4), strengths))
MIN_GAIN, MAX_LOSS, HEAVY_LOSS = .003, .01, .005


def event_times(folder, suffix):
    match = f"DFL-MAT-J03{suffix}"
    source = json.loads(EVENT_SOURCES.read_text())
    record, = [item for item in source["files"] if item["name"].endswith(f"_{match}.xml")]
    path = folder / record["name"]
    if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
        raise ValueError(f"DFL event file identity mismatch: {path}")
    times = []
    for event in ET.parse(path).iter("Event"):
        if event.get("MatchId") != match:
            raise ValueError("DFL event body match identity differs")
        for action in event:
            if action.tag not in DFL_RESTARTS:
                continue
            if action.tag == "KickOff" and action.get("GameSection") in ("firstHalf", "secondHalf"):
                continue
            times.append(timestamp(event.get("EventTime")))
    return np.asarray(sorted(times))


def simulate(times, start_time, length, profile, rng):
    loss, deviation, limit, false_share = profile
    frames = np.rint((times - start_time) * restarts.FPS).astype(int)
    frames = frames[(frames >= 0) & (frames < length)]
    frames = frames[rng.random(len(frames)) >= loss]
    noise = rng.normal(0, deviation, len(frames)).clip(-limit, limit)
    frames += np.rint(noise * restarts.FPS).astype(int)
    frames = frames[(frames >= 0) & (frames < length)]
    false = rng.integers(0, length, round(false_share * len(frames)))
    return np.unique(np.r_[frames, false])


def blend_package(paths, blend):
    report = json.loads((blend / "external_validation.json").read_text())
    if report["passed"] is not True:
        raise ValueError("D4 requires the confirmed D3 blend")
    return retrain_trial.frozen_candidate(*paths, blend)


def prepare(dfl, events, split, models, weight):
    forecasts = defaultdict(list)
    anchors = {}
    split_matches = (DEVELOPMENT,) if split == "development" else HOLDOUT
    times = {match: event_times(events, match) for match in split_matches}
    for half in augment.guard_halves(dfl, None, split, raw=True):
        length = len(half.labels)
        if half.view == "clean":
            index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(half.match)
            base_seed = 740000 + 1000 * index + 100 * half.index
            profiles = {
                name: tuple(simulate(times[half.match], half.start_time, length, profile,
                                     np.random.default_rng(base_seed + 10 * p + draw))
                            for draw in range(DRAWS))
                for p, (name, profile) in enumerate(PROFILES.items())
            }
            if not any(len(draw) for draw in profiles["moderate"]):
                raise ValueError(f"No in-half simulated restart tags for {half.match} half {half.index + 1}")
            anchors[half.match, half.index] = length, profiles
        _, prediction = retrain_trial.forecast_pair(models, weight, half.positions, half.ball, half.teams, half.pitch)
        forecasts[half.source, half.match, half.view].append((half.labels, prediction))
        print(f"Forecast {half.key}", flush=True)
    matches = {}
    for key, parts in forecasts.items():
        first, second = parts
        truth = np.r_[first[0], pm.X, second[0]]
        matches[key] = Match(truth, pm.valid_runs(truth), [first[1], second[1]])
    return matches, anchors, decoder_trial.targets(matches)


def predictions(matches, anchors, profile, draw, previous, recipe, setting):
    adjustments = {key: restarts.decoder_adjustment(length, profiles[profile][draw], **setting)
                   for key, (length, profiles) in anchors.items()}
    decoded = {}
    for key, match in matches.items():
        _, name, _ = key
        halves = [motion_trial.decode(forecast, previous, recipe, restart=adjustments[name, i])
                  for i, forecast in enumerate(match.halves)]
        decoded[key] = np.r_[halves[0], pm.X, halves[1]]
    return decoded


def accumulate(scores, predictions_by_view, targets):
    for name, by_view in targets.items():
        for key, prediction in predictions_by_view.items():
            truth, runs = by_view[key]
            spells = pm.spell_scores(truth, prediction, runs)
            confusion = pm.scored_confusion(pm.confusion(truth, prediction))
            scores[name][key].append((spells["mean_f1"], pm.macro_f1(confusion)))


def summarise(scores):
    by_view = {(match, view): np.mean(values, axis=0) for (_, match, view), values in scores.items()}
    by_match = {}
    for match in sorted({match for match, _ in by_view}):
        clean, masked, carried, swap = (by_view[match, view]
                                       for view in ("clean", "masked", "dead_carried", "team_swap"))
        spell, macro = (masked + carried) / 2
        by_match[match] = {"primary": float(spell), "macro": float(macro),
                           "standard": float((clean[0] + 2 * masked[0]) / 3), "team_swap": float(swap[0])}
    summary = {metric: float(np.mean([row[metric] for row in by_match.values()]))
               for metric in ("primary", "macro", "standard", "team_swap")}
    return summary | {"matches": by_match}


def evaluate(prepared, previous, recipe, setting=None):
    matches, anchors, targets = prepared
    if setting is None:
        scores = {name: defaultdict(list) for name in targets}
        accumulate(scores, decoder_trial.decoded(matches, previous, recipe), targets)
        summary = {name: summarise(rows) for name, rows in scores.items()}
        return {profile: summary for profile in PROFILES}
    result = {}
    for profile in PROFILES:
        scores = {name: defaultdict(list) for name in targets}
        for draw in range(DRAWS):
            decoded = predictions(matches, anchors, profile, draw, previous, recipe, setting)
            accumulate(scores, decoded, targets)
        result[profile] = {name: summarise(rows) for name, rows in scores.items()}
    return result


def mean_score(result, name, metric="primary"):
    return float(np.mean([result[profile][name][metric] for profile in PROFILES]))


def compare(reference, result):
    primary = decoder_trial.PRIMARY
    by_convention = {name: mean_score(result, name) - mean_score(reference, name)
                     for name in decoder_trial.CONVENTIONS}
    by_profile = {profile: {key: result[profile][primary][key] - reference[profile][primary][key]
                            for key in ("primary", "standard", "team_swap")}
                  for profile in PROFILES}
    by_match = {match: float(np.mean([
        result[profile][primary]["matches"][match]["primary"]
        - reference[profile][primary]["matches"][match]["primary"] for profile in PROFILES
    ])) for match in result["moderate"][primary]["matches"]}
    passed = (by_convention[primary] >= MIN_GAIN and min(by_match.values()) >= -MAX_LOSS
              and by_profile["heavy"]["primary"] >= -HEAVY_LOSS
              and all(row[key] >= -MAX_LOSS for row in by_profile.values() for key in ("standard", "team_swap"))
              and all(by_convention[name] >= -MAX_LOSS for name in decoder_trial.CONFIRMATION_GUARDS))
    return {"conventions": by_convention, "profiles": by_profile, "matches": by_match, "passed": passed}


def run_trial(dfl, events, split, paths, blend):
    now, models, frozen = blend_package(paths, blend)
    prepared = prepare(dfl, events, split, models, frozen["weight"])
    return prepared, now[0][2]["recipe"], frozen["recipe"]


def report(label, reference, trial):
    primary = decoder_trial.PRIMARY
    before = mean_score(reference, primary)
    after = mean_score(trial["scores"], primary)
    passed = trial["gates"]["passed"]
    print(f"{label} {trial['setting']}: {before:.4f} -> {after:.4f} ({after - before:+.4f})"
          f" {'passed' if passed else 'failed'}", flush=True)


def tune(dfl, events, paths, blend, output):
    require_external_only()
    output.mkdir(parents=True, exist_ok=False)
    prepared, previous, recipe = run_trial(dfl, events, "development", paths, blend)
    reference = evaluate(prepared, previous, recipe)
    trials = []
    for setting in GRID:
        scores = evaluate(prepared, previous, recipe, setting)
        trial = {"setting": setting, "scores": scores, "gates": compare(reference, scores)}
        trials.append(trial)
        report("Development", reference, trial)
    eligible = [trial for trial in trials if trial["gates"]["passed"]]
    primary = decoder_trial.PRIMARY

    def rank(trial):
        return mean_score(trial["scores"], primary), mean_score(trial["scores"], primary, "macro")

    selected = max(eligible, key=rank, default=None)
    write_json(output / "development.json", {"protocol": PROTOCOL, "reference": reference,
                                             "trials": trials, "selected": selected})
    if selected is not None:
        write_json(output / "recipe.json", {"protocol": PROTOCOL, "setting": selected["setting"],
                                            "blend_sha256": digest(blend / "recipe.json")})


def confirm(dfl, events, paths, blend, output):
    require_external_only()
    result = output / "external_validation.json"
    marker = output / "external_validation.started"
    if result.exists() or marker.exists():
        raise ValueError("D4 holdout confirmation is limited to one attempt")
    development = json.loads((output / "development.json").read_text())
    selected = development["selected"]
    if selected is None or selected["gates"]["passed"] is not True:
        raise ValueError("D4 requires a candidate that passed development before confirmation")
    frozen = json.loads((output / "recipe.json").read_text())
    if (frozen["setting"] != selected["setting"]
            or frozen["blend_sha256"] != digest(blend / "recipe.json")):
        raise ValueError("Expected the D4 candidate frozen after passing development")
    prepared, previous, recipe = run_trial(dfl, events, "holdout", paths, blend)
    marker.touch(exist_ok=False)
    reference = evaluate(prepared, previous, recipe)
    scores = evaluate(prepared, previous, recipe, frozen["setting"])
    trial = {"setting": frozen["setting"], "scores": scores, "gates": compare(reference, scores)}
    write_json(result, {"protocol": PROTOCOL, "reference": reference, "selected": trial})
    marker.unlink()
    report("Holdout", reference, trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tune", "confirm"))
    parser.add_argument("--dfl", required=True, type=Path, help="Directory of raw DFL position and match-info XML")
    parser.add_argument("--events", required=True, type=Path, help="Directory of raw DFL event XML")
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--fallback", type=Path, default=Path("."))
    args = parser.parse_args()
    package = args.fallback / "dfl_submission"
    paths = (args.fallback, *(package / name for name in ("seed_models", "ensemble_recipe",
                                                         "decoder_recipe", "retrain_models")))
    action = tune if args.action == "tune" else confirm
    action(args.dfl, args.events, paths, package / "blend_recipe", args.trial)
