"""Fit and evaluate the external-only adaptation proxy in DECISIONS.md D5."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dfl_submission import decoder_trial, external_benchmark, joint_decoder, models, robust, train
from dfl_submission.external_only import require_external_only
from dfl_submission.robust_features import invariant
from dfl_transfer_data import SECTIONS, digest
from evaluation import possession_metrics as pm


PROTOCOL = "DECISIONS.md D5"
REFERENCE = Path(__file__).with_name("robust_models")
STRIDE = 5
WINDOW = 60 * 25
SPEEDS = tuple(f"speed_{step}_{direction}" for step in (3, 12, 30) for direction in (0, 1)) + (
    "speed_mean_12", "speed_mean_50",
)
SETTINGS = ("speed_alignment", "matched_corruption", "self_training_90", "self_training_97")
THRESHOLDS = dict(zip(SETTINGS[2:], (.90, .97)))
MINIMUM_GAIN, MATCH_LOSS, CONVENTION_LOSS = .010, .020, .010


def tracking(item):
    return tuple(item[key] for key in ("positions", "ball", "teams", "pitch"))


def tracking_masks(positions, ball, teams):
    seen = np.isfinite(positions).all(axis=2)
    return np.column_stack([np.isfinite(ball).all(axis=1),
                            seen[:, teams == 0].sum(axis=1), seen[:, teams == 1].sum(axis=1)])


def matched_tracking(positions, ball, teams, donors, rng):
    schedule = []
    for start in range(0, len(ball), WINDOW):
        length = min(WINDOW, len(ball) - start)
        donor = donors[rng.integers(len(donors))]
        offset = rng.integers(len(donor) - length + 1)
        schedule.append(donor[offset:offset + length])
    mask = np.concatenate(schedule)
    positions = positions.astype(np.float32, copy=True)
    ball = ball.astype(np.float32, copy=True)
    for team in (0, 1):
        slots = rng.permutation(np.flatnonzero(teams == team))
        seen = np.isfinite(positions[:, slots]).all(axis=2)
        keep = seen & (np.cumsum(seen, axis=1) <= mask[:, team + 1, None])
        positions[:, slots] = np.where(keep[..., None], positions[:, slots], np.nan)
    ball[mask[:, 0] == 0] = np.nan
    positions += rng.normal(0, .2, positions.shape).astype(np.float32)
    ball += rng.normal(0, .2, ball.shape).astype(np.float32)
    return positions, ball


def features(positions, ball, teams, pitch):
    return tuple(frame.iloc[::STRIDE].reset_index(drop=True)
                 for frame in robust.features(positions, ball, teams, pitch))


def heads(owner, dead, extra):
    return owner.join(extra), dead.join(invariant(extra))


def add_source(rows, inputs, labels, match_index):
    live = labels < pm.D
    known = labels != pm.X
    for head, frame, select, positive in zip(rows, inputs, (live, known), (pm.H, pm.D)):
        head.append((frame.loc[select].copy(), labels[select] == positive, np.full(select.sum(), match_index)))


def probabilities(checkpoints, inputs):
    owner, dead = inputs
    return models.ownership_probability(checkpoints[0], owner), checkpoints[1].predict_proba(dead)[:, 1]


def pseudo_labels(home, dead, threshold):
    live = dead <= 1 - threshold
    known = live | (dead >= threshold)
    owner = live & ((home <= 1 - threshold) | (home >= threshold))
    return (owner, home[owner] >= threshold), (known, dead[known] >= threshold)


def speed_quantiles(views):
    pooled = pd.concat(views, ignore_index=True)
    quantiles = {}
    for name in SPEEDS:
        speed = pooled[name].to_numpy()
        positive = speed[np.isfinite(speed) & (speed > 0)]
        if len(positive):
            lower, median, upper = np.quantile(np.log1p(positive.astype(float)), (.25, .50, .75))
            if upper > lower:
                quantiles[name] = (median, upper - lower)
    return quantiles


def speed_alignment(source, target):
    return {name: {"target_median": target[name][0], "source_median": source[name][0],
                   "scale": float(np.clip(source[name][1] / target[name][1], .5, 2))}
            for name in SPEEDS if name in source and name in target}


def aligned(extra, alignment):
    extra = extra.copy()
    for name, mapping in alignment.items():
        speed = extra[name].to_numpy(copy=True)
        positive = np.isfinite(speed) & (speed > 0)
        logged = np.log1p(speed[positive].astype(float))
        moved = (logged - mapping["target_median"]) * mapping["scale"] + mapping["source_median"]
        speed[positive] = np.maximum(0, np.expm1(moved))
        extra[name] = speed
    return extra


def reference():
    frozen = json.loads((REFERENCE / "recipe.json").read_text())["models"]
    return robust.verified_models(REFERENCE, frozen), frozen


def target_training(skill, teacher):
    donors, speeds = [], []
    pseudo = {name: ([], []) for name in THRESHOLDS}
    for match, index, path in external_benchmark.skillcorner_paths(skill, "train"):
        with np.load(path, allow_pickle=False) as item:
            positions, ball, teams, pitch = tracking(item)
        donors.append(tracking_masks(positions, ball, teams))
        owner, dead, extra = features(positions, ball, teams, pitch)
        speeds.append(extra.loc[:, SPEEDS].copy())
        inputs = heads(owner, dead, extra)
        home, stoppage = probabilities(teacher, inputs)
        for name, threshold in THRESHOLDS.items():
            for rows, frame, (select, labels) in zip(pseudo[name], inputs, pseudo_labels(home, stoppage, threshold)):
                rows.append((frame.loc[select].copy(), labels))
        print(f"Prepared unlabelled SkillCorner {match} half {index + 1}", flush=True)
    return donors, speed_quantiles(speeds), pseudo


def source_training(dfl, donors):
    original, matched, speeds = ([], []), ([], []), []
    manifest = json.loads(train.SOURCE_MANIFEST.read_text())
    for match_index, match in enumerate(train.TRAIN):
        halves = train.dfl_halves(dfl, match, manifest)
        for index, section in enumerate(SECTIONS):
            half = halves[section]
            labels = pm.label_codes(half["labels"])[::STRIDE]
            teams, pitch = half["teams"], half["pitch"]
            for view, positions, ball in robust.views(half, match_index, index):
                owner, dead, extra = features(positions, ball, teams, pitch)
                inputs = heads(owner, dead, extra)
                add_source(original, inputs, labels, match_index)
                if view == "clean":
                    for rows, source in zip(matched, original):
                        rows.append(source[-1])
                else:
                    speeds.append(extra.loc[:, SPEEDS].copy())
            for view_index in range(2):
                rng = np.random.default_rng(810000 + 100 * match_index + 10 * index + view_index)
                positions, ball = matched_tracking(half["positions"], half["ball"], teams, donors, rng)
                inputs = heads(*features(positions, ball, teams, pitch))
                add_source(matched, inputs, labels, match_index)
            print(f"Prepared DFL {match} half {index + 1}", flush=True)
    return original, matched, speed_quantiles(speeds)


def fit_student(source, pseudo, *, owner):
    source_frames, source_targets, groups = zip(*source)
    target_frames, target_targets = zip(*pseudo)
    source_labels, target_labels = np.concatenate(source_targets), np.concatenate(target_targets)
    counts = np.bincount(target_labels.astype(int), minlength=2)
    if not counts.all():
        raise ValueError("Both pseudo-classes are required")
    size = len(source_labels) + len(target_labels)
    source_weight = robust.source_weights(source_labels, np.concatenate(groups)) * (.75 * size / len(source_labels))
    target_weight = .25 * size / (2 * counts[target_labels.astype(int)])
    inputs = pd.concat((*source_frames, *target_frames), ignore_index=True)
    labels = np.r_[source_labels, target_labels]
    return robust.fit_weighted(inputs, labels, np.r_[source_weight, target_weight], owner=owner)


def fit(dfl, skill, trial):
    require_external_only()
    trial.mkdir(parents=True, exist_ok=False)
    teacher, reference_hashes = reference()
    donors, target_speeds, pseudo = target_training(skill, teacher)
    original, matched, source_speeds = source_training(dfl, donors)
    fitted = {
        "protocol": PROTOCOL,
        "reference": reference_hashes,
        "dfl_manifest_sha256": digest(train.SOURCE_MANIFEST),
        "skill_manifest_sha256": digest(skill / "manifest.json"),
        "alignment": speed_alignment(source_speeds, target_speeds),
        "settings": {},
    }
    for name in SETTINGS[1:]:
        folder = trial / name
        counts = {}
        if name in THRESHOLDS:
            for filename, rows in zip(models.MODEL_FILES, pseudo[name]):
                counts[filename] = np.bincount(np.concatenate([y for _, y in rows]).astype(int),
                                               minlength=2).tolist()
        hashes = {}
        if all(all(count) for count in counts.values()):
            folder.mkdir()
            for index, filename in enumerate(models.MODEL_FILES):
                if name == "matched_corruption":
                    model, _ = robust.fit(*zip(*matched[index]), owner=index == 0)
                else:
                    model = fit_student(original[index], pseudo[name][index], owner=index == 0)
                model.save_model(folder / filename)
                hashes[filename] = digest(folder / filename)
                print(f"Fitted {name}/{filename}", flush=True)
        fitted["settings"][name] = {"models": hashes, "pseudo_class_counts": counts}
    train.write_json(trial / "fit.json", fitted)


def load(skill, trial, settings):
    fitted = json.loads((trial / "fit.json").read_text())
    teacher, hashes = reference()
    if (fitted["protocol"] != PROTOCOL or hashes != fitted["reference"]
            or digest(skill / "manifest.json") != fitted["skill_manifest_sha256"]
            or digest(train.SOURCE_MANIFEST) != fitted["dfl_manifest_sha256"]):
        raise ValueError("External inputs differ from the fitting record")
    checkpoints = {"reference": teacher}
    for name in settings:
        if name == "speed_alignment":
            checkpoints[name] = teacher
        else:
            hashes = fitted["settings"][name]["models"]
            if hashes:
                checkpoints[name] = robust.verified_models(trial / name, hashes)
    return fitted, checkpoints


def decode(checkpoint, inputs, length):
    frames, steps = np.arange(length), np.arange(0, length, STRIDE)
    home, dead = (np.interp(frames, steps, p).astype(np.float32)
                  for p in probabilities(checkpoint, inputs))
    edges = np.zeros(length, dtype=np.float32)
    return joint_decoder.decode(home, dead, edges, edges, cutoff=.53, owner_cost=6, dead_cost=2, discount=0)


def score_match(truth, predictions):
    scores = {name: {} for name in predictions}
    for convention, seconds in decoder_trial.CONVENTIONS.items():
        labels = decoder_trial.relabel(truth, seconds)
        runs = pm.valid_runs(labels)
        for name, prediction in predictions.items():
            spells = pm.spell_scores(labels, prediction, runs)
            confusion = pm.scored_confusion(pm.confusion(labels, prediction))
            scores[name][convention] = {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion)}
    return scores


def evaluate(skill, split, fitted, checkpoints):
    labels, forecasts = defaultdict(list), {name: defaultdict(list) for name in checkpoints}
    for match, index, path in external_benchmark.skillcorner_paths(skill, split):
        with np.load(path, allow_pickle=False) as item:
            positions, ball, teams, pitch = tracking(item)
            labels[match].append(pm.label_codes(item["labels"]))
        owner, dead, extra = features(positions, ball, teams, pitch)
        original = heads(owner, dead, extra)
        for name, checkpoint in checkpoints.items():
            inputs = heads(owner, dead, aligned(extra, fitted["alignment"])) if name == "speed_alignment" else original
            forecasts[name][match].append(decode(checkpoint, inputs, len(ball)))
        print(f"Decoded {split} SkillCorner {match} half {index + 1}", flush=True)

    scores = {name: {} for name in checkpoints}
    for match, (first, second) in labels.items():
        truth = np.r_[first, pm.X, second]
        predictions = {name: np.r_[parts[match][0], pm.X, parts[match][1]] for name, parts in forecasts.items()}
        for name, metrics in score_match(truth, predictions).items():
            scores[name][str(match)] = metrics
    return {
        name: {"matches": matches, "means": {
            convention: {metric: float(np.mean([result[convention][metric] for result in matches.values()]))
                         for metric in ("spell_f1", "macro_f1")}
            for convention in decoder_trial.CONVENTIONS}}
        for name, matches in scores.items()
    }


def comparison(base, challenger, *, minimum_gain=MINIMUM_GAIN):
    gains = {name: score["spell_f1"] - base["means"][name]["spell_f1"]
             for name, score in challenger["means"].items()}
    primary = decoder_trial.PRIMARY
    matches = {match: scores[primary]["spell_f1"] - base["matches"][match][primary]["spell_f1"]
               for match, scores in challenger["matches"].items()}
    passed = (gains[primary] >= minimum_gain and min(matches.values()) >= -MATCH_LOSS
              and all(gains[name] >= -CONVENTION_LOSS for name in decoder_trial.CONFIRMATION_GUARDS))
    return {"gains": gains, "primary_match_gains": matches, "passed": passed}


def tune(skill, trial):
    require_external_only()
    output = trial / "development.json"
    if output.exists():
        raise ValueError("Development scoring already ran")
    fitted, checkpoints = load(skill, trial, SETTINGS)
    scores = evaluate(skill, "development", fitted, checkpoints)
    results = {name: comparison(scores["reference"], scores[name]) for name in SETTINGS if name in scores}
    eligible = [name for name in SETTINGS if name in results and results[name]["passed"]]
    selected = max(eligible, key=lambda name: (scores[name]["means"][decoder_trial.PRIMARY]["spell_f1"],
                                              scores[name]["means"][decoder_trial.PRIMARY]["macro_f1"]), default=None)
    train.write_json(output, {"protocol": PROTOCOL, "fit_sha256": digest(trial / "fit.json"),
                              "scores": scores, "comparisons": results, "selected": selected})
    print(f"Development selected: {selected or 'none'}", flush=True)


def confirm(skill, trial):
    require_external_only()
    development = json.loads((trial / "development.json").read_text())
    selected = development["selected"]
    if selected not in SETTINGS or not development["comparisons"][selected]["passed"]:
        raise ValueError("No setting passed development")
    if development["fit_sha256"] != digest(trial / "fit.json"):
        raise ValueError("Fitted recipe differs from development")
    fitted, checkpoints = load(skill, trial, (selected,))
    if selected not in checkpoints:
        raise ValueError("Selected checkpoint is unavailable")
    output = trial / "external_validation.json"
    if output.exists():
        raise ValueError("Confirmation already ran")
    with (trial / "confirmation_started").open("x") as attempt:
        attempt.write(selected + "\n")
    scores = evaluate(skill, "holdout", fitted, checkpoints)
    result = comparison(scores["reference"], scores[selected])
    train.write_json(output, {"protocol": PROTOCOL, "fit_sha256": development["fit_sha256"], "selected": selected,
                              "scores": scores, "comparison": result})
    print(f"Confirmation {'passed' if result['passed'] else 'failed'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("fit", "tune", "confirm"):
        command = commands.add_parser(action)
        command.add_argument("--skill", required=True, type=Path)
        command.add_argument("--trial", required=True, type=Path)
        if action == "fit":
            command.add_argument("--dfl", required=True, type=Path, help="Pinned DFL XML directory")
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.dfl, args.skill, args.trial)
    elif args.action == "tune":
        tune(args.skill, args.trial)
    else:
        confirm(args.skill, args.trial)
