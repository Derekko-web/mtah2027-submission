"""Fit a per-match ownership adaptation and test it on external SkillCorner 2020."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from dfl_submission import (
    adaptation_trial, decoder_trial, external_benchmark, joint_decoder, long_context,
    models, motion_trial, retrain_trial, skillcorner_2020, train,
)
from dfl_submission.build import tactical_labels, tactical_tracking_halves, write_csv
from dfl_submission.external_only import require_external_only
from dfl_transfer_data import digest
from dfl_transfer_features import swap
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
BLEND = PACKAGE / "blend_recipe"
PROTOCOL = "DECISIONS.md D6"
CONFIDENCE = .90
REPLAY_STRIDE = 25
ROUNDS = 40
REPLAY_WEIGHT = .75
MULTIPLIERS = (.25, .5, 1.0)
PRIMARY_GAIN = .005


def read_json(path):
    return json.loads(path.read_text())


def reference():
    if read_json(BLEND / "external_validation.json")["passed"] is not True:
        raise ValueError("Expected the verified D3 ensemble")
    original, packages, frozen = retrain_trial.frozen_candidate(
        ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe",
        PACKAGE / "retrain_models", BLEND,
    )
    return original[0], packages, frozen


def prepare_replay(dfl, skill, output):
    require_external_only()
    if output.exists():
        raise FileExistsError(output)
    sources = read_json(PACKAGE / "long_models" / "fit.json")
    for path, key in ((train.SOURCE_MANIFEST, "dfl_manifest_sha256"),
                      (skill / "manifest.json", "skill_manifest_sha256")):
        if digest(path) != sources[key]:
            raise ValueError("Replay inputs must match the original long-context fit")
    rows, targets, groups = [], [], []
    for half in external_benchmark.halves(dfl, skill, "train", raw=True):
        frames, owner, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        labels = half.labels[frames]
        use = (frames % REPLAY_STRIDE == 0) & (labels < pm.D)
        rows.append(owner.loc[use].copy())
        targets.append(labels[use] == pm.H)
        groups.append(np.full(use.sum(), f"{half.source}:{half.match}"))
        print(f"Prepared replay {half.key}", flush=True)
    inputs, labels = pd.concat(rows, ignore_index=True), np.concatenate(targets).astype(int)
    weights = long_context.source_weights(labels, np.concatenate(groups))
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent) as temporary:
        np.savez_compressed(temporary, inputs=inputs.to_numpy(dtype=np.float32),
                            columns=inputs.columns.to_numpy(dtype=str), labels=labels, weights=weights)
        temporary.flush()
        # Publish only a complete replay and preserve any concurrently published copy.
        output.hardlink_to(temporary.name)
    print(f"Prepared {len(labels):,} external replay rows", flush=True)


def load_replay(path):
    with np.load(path, allow_pickle=False) as replay:
        return pd.DataFrame(replay["inputs"], columns=replay["columns"]), replay["labels"], replay["weights"]


@dataclass
class Forecast:
    frames: np.ndarray
    features: pd.DataFrame
    tree_home: np.ndarray
    signals: tuple


def forecast(tracking, teacher):
    original, packages, frozen = teacher
    previous = original[2]["recipe"]
    # Capture the unmodified tree signal at its native sample points, then use the
    # same D3 blend and arithmetic as the submitted decoder.
    current, mixed = retrain_trial.forecast_pair(packages, frozen["weight"], *tracking)
    frames, owner, _ = long_context.features(*tracking)
    tree_home = current[0][0][0][frames]
    return Forecast(frames, owner, tree_home, motion_trial.blend_signals(mixed, previous, frozen["recipe"]))


def fit_match(halves, replay, checkpoint):
    frames, targets = [], []
    for half in halves:
        home, dead, *_ = half.signals
        (use, labels), _ = adaptation_trial.pseudo_labels(home[half.frames], dead[half.frames], CONFIDENCE)
        frames.append(half.features.loc[use])
        targets.append(labels)
    pseudo = np.concatenate(targets).astype(int)
    counts = np.bincount(pseudo, minlength=2)
    if not counts.all():
        return None, counts.tolist()
    source, truth, source_weights = replay
    if not source.columns.equals(halves[0].features.columns):
        raise ValueError("Replay and target feature columns differ")
    total = len(truth) + len(pseudo)
    weights = np.r_[source_weights * (REPLAY_WEIGHT * total / len(truth)),
                    (1 - REPLAY_WEIGHT) * total / (2 * counts[pseudo])]
    inputs = pd.concat([source, *frames], ignore_index=True)
    labels = np.r_[truth, pseudo]
    student = XGBClassifier(**dict(long_context.TREE_PARAMS, n_estimators=ROUNDS))
    student.fit(pd.concat([inputs, swap(inputs)], ignore_index=True),
                np.r_[labels, 1 - labels], sample_weight=np.tile(weights, 2), xgb_model=checkpoint)
    return student, counts.tolist()


def decode(half, adjustment, multiplier, recipe):
    home, dead, dead_edge, owner_edge = half.signals
    if adjustment is not None and multiplier:
        home = np.clip(home + multiplier * adjustment, 0, 1)
    return joint_decoder.decode(home, dead, dead_edge, owner_edge, cutoff=recipe["cutoff"],
                                owner_cost=recipe["owner_cost"], dead_cost=recipe["dead_cost"],
                                discount=recipe["discount"])


def predict_match(tracking, teacher, replay, multipliers):
    halves = [forecast(half, teacher) for half in tracking]
    original, _, frozen = teacher
    student, counts = fit_match(halves, replay, original[0][0])
    predictions = {weight: [] for weight in multipliers}
    for half in halves:
        adjustment = None
        if student is not None:
            residual = models.ownership_probability(student, half.features) - half.tree_home
            adjustment = np.interp(np.arange(len(half.signals[0])), half.frames, residual).astype(np.float32)
        for weight in multipliers:
            predictions[weight].append(decode(half, adjustment, weight, frozen["recipe"]))
    return predictions, counts


def score_match(labels, predictions):
    truth = np.r_[labels[0], pm.X, labels[1]]
    predicted = {str(weight): np.r_[parts[0], pm.X, parts[1]] for weight, parts in predictions.items()}
    results = {weight: {} for weight in predicted}
    for name, seconds in decoder_trial.CONVENTIONS.items():
        target = decoder_trial.relabel(truth, seconds)
        runs = pm.valid_runs(target)
        for weight, states in predicted.items():
            spells = pm.spell_scores(target, states, runs)
            confusion = pm.scored_confusion(pm.confusion(target, states))
            macro = np.mean([pm.per_class_prf(confusion, team)[2] for team in (pm.H, pm.A)])
            results[weight][name] = {"spell_f1": spells["mean_f1"], "macro_f1": float(macro)}
    return results


def evaluate(raw, matches, teacher, replay, grid):
    multipliers = (0.0, *grid)
    scores = {str(weight): {"matches": {}} for weight in multipliers}
    target_records = {}
    for match in matches:
        tracking = skillcorner_2020.read_tracking(raw, match)
        predictions, counts = predict_match(tracking, teacher, replay, multipliers)
        labels = skillcorner_2020.read_ownership(raw, match)
        target_records[str(match)] = {"pseudo_away_home": counts, "fallback": not all(counts),
                                      "label_coverage": float(np.mean(np.concatenate(labels) < pm.D))}
        for weight, result in score_match(labels, predictions).items():
            scores[weight]["matches"][str(match)] = result
        print(f"Scored external SkillCorner 2020 match {match}", flush=True)
    for result in scores.values():
        result["means"] = {
            name: {metric: float(np.mean([m[name][metric] for m in result["matches"].values()]))
                   for metric in ("spell_f1", "macro_f1")}
            for name in decoder_trial.CONVENTIONS
        }
    return scores, target_records


def identity(replay):
    return {
        "protocol": PROTOCOL, "teacher_sha256": digest(BLEND / "recipe.json"),
        "replay_sha256": digest(replay), "skillcorner_revision": skillcorner_2020.REVISION,
        "confidence": CONFIDENCE, "rounds": ROUNDS, "replay_weight": REPLAY_WEIGHT,
        "replay_stride": REPLAY_STRIDE,
    }


def tune(raw, replay, trial):
    require_external_only()
    trial.mkdir(parents=True, exist_ok=False)
    frozen = identity(replay)
    scores, targets = evaluate(raw, skillcorner_2020.DEVELOPMENT, reference(), load_replay(replay), MULTIPLIERS)
    comparisons = {str(weight): adaptation_trial.comparison(scores["0.0"], scores[str(weight)],
                                                            minimum_gain=PRIMARY_GAIN)
                   for weight in MULTIPLIERS}
    eligible = [weight for weight in MULTIPLIERS if comparisons[str(weight)]["passed"]]
    primary = decoder_trial.PRIMARY
    selected = max(eligible, key=lambda weight: (
        scores[str(weight)]["means"][primary]["spell_f1"],
        scores[str(weight)]["means"][primary]["macro_f1"],
    ), default=None)
    train.write_json(trial / "development.json", {
        "identity": frozen, "scores": scores, "targets": targets, "comparisons": comparisons, "selected": selected,
    })
    print(f"Development selected multiplier: {selected if selected is not None else 'none'}", flush=True)


def selection(replay, trial):
    development = trial / "development.json"
    report = read_json(development)
    selected = report["selected"]
    if selected not in MULTIPLIERS or not report["comparisons"][str(selected)]["passed"]:
        raise ValueError("No D6 setting passed external development")
    if report["identity"] != identity(replay):
        raise ValueError("Replay or teacher differs from the development run")
    return report["identity"], selected, digest(development)


def confirm(raw, replay, trial):
    require_external_only()
    frozen, selected, development_sha256 = selection(replay, trial)
    output = trial / "external_validation.json"
    if output.exists():
        raise ValueError("D6 confirmation already ran")
    teacher, replay_rows = reference(), load_replay(replay)
    with (trial / "confirmation_started").open("x") as attempt:
        attempt.write(development_sha256 + "\n")
    scores, targets = evaluate(raw, skillcorner_2020.HOLDOUT, teacher, replay_rows, (selected,))
    comparison = adaptation_trial.comparison(scores["0.0"], scores[str(selected)], minimum_gain=PRIMARY_GAIN)
    train.write_json(output, {"identity": frozen, "selected": selected, "development_sha256": development_sha256,
                              "scores": scores, "targets": targets, "comparison": comparison})
    print(f"Confirmation {'passed' if comparison['passed'] else 'failed'}", flush=True)


def build(replay, trial, output, *, organiser_approved):
    if not organiser_approved:
        raise ValueError("D6 competition adaptation requires written organiser permission")
    frozen, selected, development_sha256 = selection(replay, trial)
    confirmation = read_json(trial / "external_validation.json")
    if (confirmation["comparison"]["passed"] is not True or confirmation["identity"] != frozen
            or confirmation["selected"] != selected or confirmation["development_sha256"] != development_sha256):
        raise ValueError("D6 competition adaptation requires a matching, passed external confirmation")
    teacher, replay_rows = reference(), load_replay(replay)
    halves = list(tactical_tracking_halves(centered=True, pitch=teacher[2]["pitch"]))
    predictions, _ = predict_match([tracking for _, _, tracking in halves], teacher, replay_rows, (selected,))
    labels = tactical_labels((span, codes) for (_, span, _), codes in zip(halves, predictions[selected], strict=True))
    write_csv(output, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare-replay")
    prepare.add_argument("--dfl", required=True, type=Path, help="Pinned DFL training XML")
    prepare.add_argument("--skill", required=True, type=Path, help="Pinned 2024–25 SkillCorner cache")
    prepare.add_argument("--replay", required=True, type=Path)
    for action in ("tune", "confirm", "build"):
        command = commands.add_parser(action)
        command.add_argument("--replay", required=True, type=Path)
        command.add_argument("--trial", required=True, type=Path)
        if action != "build":
            command.add_argument("--legacy-raw", required=True, type=Path, help="SkillCorner 2020 JSON cache")
        else:
            command.add_argument("--out", required=True, type=Path)
            command.add_argument("--organiser-approved", action="store_true",
                                 help="Assert written organiser permission for label-free competition fitting")
    args = parser.parse_args()
    if args.action == "prepare-replay":
        prepare_replay(args.dfl, args.skill, args.replay)
    elif args.action == "tune":
        tune(args.legacy_raw, args.replay, args.trial)
    elif args.action == "confirm":
        confirm(args.legacy_raw, args.replay, args.trial)
    else:
        build(args.replay, args.trial, args.out, organiser_approved=args.organiser_approved)
