"""Train, validate and build a tracking-robust possession model using external DFL only."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from dfl_submission import models as shared
from dfl_submission.build import write_csv
from dfl_submission.decoder import decode
from dfl_submission.robust_features import extras, invariant, stress
from dfl_submission.train import (
    DEAD_PARAMS, DEVELOPMENT, HOLDOUT, SOURCE_MANIFEST, TRAIN, dfl_halves, write_json,
)
from dfl_transfer_data import SECTIONS, digest
from dfl_transfer_features import corrupt, swap
from dfl_transfer_teacher import PARAMS
from evaluation import possession_metrics as pm


CONTROL = Path(__file__).with_name("models")
VIEWS = ("clean", "masked", "stress")
WEIGHTS = (1, 2, 2)
BLENDS = (0.0, 0.5, 1.0)
GRID = tuple(product((.4, .5, .6), (25, 50, 75), (25, 50, 75)))


def views(half, match_index, half_index):
    positions, ball = half["positions"], half["ball"]
    seed = 270000 + 100 * match_index + half_index
    yield "clean", positions, ball
    yield "masked", *corrupt(positions, ball, seed)
    yield "stress", *stress(positions, ball, seed + 100000)


def features(positions, ball, teams, pitch):
    blocks = [(0, len(ball))]
    owner = shared.ownership_features(positions, ball, teams, blocks)
    dead = shared.dead_features(positions, ball, teams, blocks)
    extra = extras(positions, ball, teams, pitch)
    return owner, dead, extra


def source_weights(y, group):
    weight = np.empty(len(y))
    for match in range(len(TRAIN)):
        for label in (0, 1):
            member = (group == match) & (y == label)
            if not member.any():
                raise ValueError("DFL training match lacks both classes")
            weight[member] = len(y) / (2 * len(TRAIN) * member.sum())
    return weight


def fit_weighted(inputs, y, weight, *, owner):
    if owner:
        inputs = pd.concat([inputs, swap(inputs)], ignore_index=True)
        y = np.r_[y, 1 - y]
        weight = np.tile(weight, 2)
    params = dict(PARAMS if owner else DEAD_PARAMS, max_depth=4, n_estimators=260)
    model = XGBClassifier(**params)
    model.fit(inputs, y, sample_weight=weight)
    return model


def fit(frames, targets, groups, *, owner):
    inputs = pd.concat(frames, ignore_index=True)
    y = np.concatenate(targets)
    group = np.concatenate(groups)
    return fit_weighted(inputs, y, source_weights(y, group), owner=owner), len(y) * (2 if owner else 1)


def fit_models(raw, manifest):
    owner, dead = ([], [], []), ([], [], [])
    for match_index, suffix in enumerate(TRAIN):
        halves = dfl_halves(raw, suffix, manifest)
        for half_index, section in enumerate(SECTIONS):
            half = halves[section]
            labels = half["labels"]
            sampled = np.arange(len(labels)) % 5 == 0
            known, live = sampled & (labels != "X"), sampled & np.isin(labels, ["H", "A"])
            for view, positions, ball in views(half, match_index, half_index):
                owner_base, dead_base, extra = features(positions, ball, half["teams"], half["pitch"])
                owner[0].append(owner_base.loc[live].join(extra.loc[live]))
                owner[1].append((labels[live] == "H").astype(int))
                owner[2].append(np.full(live.sum(), match_index))
                dead[0].append(dead_base.loc[known].join(invariant(extra.loc[known])))
                dead[1].append((labels[known] == "D").astype(int))
                dead[2].append(np.full(known.sum(), match_index))
                print(f"Prepared {suffix} {section} {view}", flush=True)
    owner_model, owner_count = fit(*owner, owner=True)
    dead_model, dead_count = fit(*dead, owner=False)
    return (owner_model, dead_model), {"paired_owner_rows": owner_count, "dead_rows": dead_count}


def predict(control, robust, positions, ball, teams, pitch):
    owner, dead, extra = features(positions, ball, teams, pitch)
    return (
        shared.ownership_probability(control[0], owner), control[1].predict_proba(dead)[:, 1],
        shared.ownership_probability(robust[0], owner.join(extra)),
        robust[1].predict_proba(dead.join(invariant(extra)))[:, 1],
    )


def match_predictions(raw, manifest, suffix, control, robust):
    match_index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(suffix)
    halves = dfl_halves(raw, suffix, manifest)
    predictions = {view: [] for view in VIEWS}
    for half_index, section in enumerate(SECTIONS):
        half = halves[section]
        labels = pm.label_codes(half["labels"])
        for view, positions, ball in views(half, match_index, half_index):
            probs = predict(control, robust, positions, ball, half["teams"], half["pitch"])
            predictions[view].append((labels, probs))
            print(f"Predicted {suffix} {section} {view}", flush=True)
    return predictions


def decode_blend(probabilities, recipe):
    old_owner, old_dead, new_owner, new_dead = probabilities
    owner_weight, dead_weight = recipe["owner_weight"], recipe["dead_weight"]
    return decode((1 - owner_weight) * old_owner + owner_weight * new_owner,
                  (1 - dead_weight) * old_dead + dead_weight * new_dead,
                  dead_cutoff=recipe["dead_cutoff"], dead_radius=recipe["dead_radius"],
                  owner_radius=recipe["owner_radius"])


def evaluate(predictions, recipe):
    result = {}
    for view, (first, second) in predictions.items():
        labels = np.r_[first[0], pm.X, second[0]]
        decoded = np.r_[decode_blend(first[1], recipe), pm.X, decode_blend(second[1], recipe)]
        confusion = pm.scored_confusion(pm.confusion(labels, decoded))
        spells = pm.spell_scores(labels, decoded, pm.valid_runs(labels))
        _, dead_recall, _ = pm.per_class_prf(confusion, pm.D)
        result[view] = {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion),
                        "dead_recall": dead_recall, "spells": spells["n_pred"]}
    return result


def rank(metrics, metric="spell_f1"):
    return sum(w * metrics[view][metric] for view, w in zip(VIEWS, WEIGHTS)) / sum(WEIGHTS)


def verified_models(folder, hashes):
    for name in shared.MODEL_FILES:
        if digest(folder / name) != hashes[name]:
            raise ValueError(f"Checkpoint identity mismatch: {folder / name}")
    return shared.load_models(folder)


def train(raw, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    control_frozen = json.loads((CONTROL / "recipe.json").read_text())
    control = verified_models(CONTROL, control_frozen["models"])
    reference = dict(control_frozen["recipe"], owner_weight=0, dead_weight=0)
    models, counts = fit_models(raw, manifest)
    hashes = {}
    for model, name in zip(models, shared.MODEL_FILES):
        model.save_model(output / name)
        hashes[name] = digest(output / name)
    robust = verified_models(output, hashes)
    dev = match_predictions(raw, manifest, DEVELOPMENT, control, robust)
    trials = []
    for owner_weight, dead_weight in product(BLENDS, repeat=2):
        if owner_weight == dead_weight == 0:
            continue
        for cutoff, dead_radius, owner_radius in GRID:
            recipe = dict(owner_weight=owner_weight, dead_weight=dead_weight, dead_cutoff=cutoff,
                          dead_radius=dead_radius, owner_radius=owner_radius)
            trials.append({"recipe": recipe, "metrics": evaluate(dev, recipe)})
    selected = max(trials, key=lambda trial: (rank(trial["metrics"]), rank(trial["metrics"], "macro_f1")))
    frozen = dict(models=hashes, control_models=control_frozen["models"], recipe=selected["recipe"])
    write_json(output / "recipe.json", frozen)
    print(f"Frozen recipe: {selected['recipe']}", flush=True)
    results = {DEVELOPMENT: {"control": evaluate(dev, reference), "robust": selected["metrics"]}}
    for suffix in HOLDOUT:
        predictions = match_predictions(raw, manifest, suffix, control, robust)
        results[suffix] = {"control": evaluate(predictions, reference),
                           "robust": evaluate(predictions, selected["recipe"])}
    gains = {match: rank(scores["robust"]) - rank(scores["control"]) for match, scores in results.items()}
    passed = all(gain > 0 for gain in gains.values()) and all(
        results[match]["robust"][view]["spell_f1"] >= results[match]["control"][view]["spell_f1"] - .02
        for match in HOLDOUT for view in VIEWS)
    write_json(output / "external_validation.json", {
        "counts": counts, "development": DEVELOPMENT, "held_out": HOLDOUT, "train": TRAIN,
        "dataset_manifest_sha256": digest(SOURCE_MANIFEST), "view_weights": dict(zip(VIEWS, WEIGHTS)),
        "matches": results, "development_search": trials, "weighted_gains": gains, "holdout_passed": passed,
        "gate": "Positive weighted SpellF1 gain on all three evaluation matches; no holdout view loses more than .02",
    })
    print(f"External comparison passed: {passed}, weighted gains: {gains}", flush=True)


def build(folder, output):
    from dfl_submission.build import tactical_halves

    report = json.loads((folder / "external_validation.json").read_text())
    if report["holdout_passed"] is not True:
        raise ValueError("Robust model failed its external comparison")
    frozen = json.loads((folder / "recipe.json").read_text())
    control = verified_models(CONTROL, frozen["control_models"])
    robust = verified_models(folder, frozen["models"])
    def decode_half(positions, ball, teams, pitch):
        return decode_blend(predict(control, robust, positions, ball, teams, pitch), frozen["recipe"])

    labels = tactical_halves(decode_half, centered=True)
    write_csv(output, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--dfl-raw", required=True, type=Path)
    fitting.add_argument("--out", required=True, type=Path)
    inference = commands.add_parser("build")
    inference.add_argument("--model-dir", required=True, type=Path)
    inference.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    if args.action != "build":
        require_external_only()
    if args.action == "train":
        train(args.dfl_raw, args.out)
    else:
        build(args.model_dir, args.out)
