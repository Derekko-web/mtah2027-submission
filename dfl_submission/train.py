"""Fit and tune on checksum-pinned DFL XML, then validate on held-out DFL matches."""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from dfl_submission.decoder import decode
from dfl_submission.models import (
    MODEL_FILES, dead_features, load_models, ownership_features, predict,
)
from dfl_transfer_data import SECTIONS, digest, parse_match
from dfl_transfer_features import corrupt
from dfl_transfer_teacher import PARAMS, fit as fit_ownership
from evaluation import possession_metrics as pm

SOURCE_MANIFEST = Path(__file__).with_name("dfl_sources.json")
TRAIN = ("WMX", "WN1", "WOH", "WOY")
DEVELOPMENT = "WPY"
HOLDOUT = ("WQQ", "WR9")
VIEWS = ("clean", "masked")
DEAD_PARAMS = dict(PARAMS, max_depth=3, n_estimators=240, min_child_weight=50)
GRID = tuple(dict(dead_cutoff=cutoff, dead_radius=dead_radius, owner_radius=owner_radius)
             for cutoff, dead_radius, owner_radius in product(
                 (0.35, 0.5, 0.65), (0, 12, 25, 50), (12, 25, 50)))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dfl_halves(raw, suffix, manifest):
    match = f"DFL-MAT-J03{suffix}"
    files = {}
    for record in manifest["files"]:
        if not record["name"].endswith(f"_{match}.xml"):
            continue
        path = raw / record["name"]
        if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            raise ValueError(f"DFL file identity mismatch: {path}")
        kind = "info" if "matchinformation" in path.name else "positions"
        files[kind] = path
    print(f"Parsing {match}", flush=True)
    halves, _ = parse_match(files["info"], files["positions"], match)
    return halves


def views(half, match_index, half_index):
    positions, ball = half["positions"], half["ball"]
    yield "clean", positions, ball
    yield "masked", *corrupt(positions, ball, 270000 + 100 * match_index + half_index)


def fit_models(raw, manifest):
    owner_x, owner_y, owner_groups = [], [], []
    dead_x, dead_y, dead_groups = [], [], []
    for match_index, suffix in enumerate(TRAIN):
        halves = dfl_halves(raw, suffix, manifest)
        for half_index, section in enumerate(SECTIONS):
            half = halves[section]
            labels, teams = half["labels"], half["teams"]
            blocks = [(0, len(labels))]
            sample = np.arange(len(labels)) % 5 == 0
            live = sample & np.isin(labels, ["H", "A"])
            known = sample & (labels != "X")
            for view, positions, ball in views(half, match_index, half_index):
                owner_x.append(ownership_features(positions, ball, teams, blocks).loc[live].copy())
                owner_y.append((labels[live] == "H").astype(int))
                owner_groups.append(np.full(live.sum(), match_index))
                dead_x.append(dead_features(positions, ball, teams, blocks).loc[known].copy())
                dead_y.append((labels[known] == "D").astype(int))
                dead_groups.append(np.full(known.sum(), match_index))
                print(f"Prepared {suffix} {section} {view}", flush=True)

    owner_targets = np.concatenate(owner_y)
    ownership, _ = fit_ownership(pd.concat(owner_x, ignore_index=True), owner_targets,
                                 np.concatenate(owner_groups))
    dead_targets = np.concatenate(dead_y)
    groups = np.concatenate(dead_groups)
    weights = np.empty(len(groups))
    for group in range(len(TRAIN)):
        for label in (0, 1):
            members = (groups == group) & (dead_targets == label)
            if not members.any():
                raise ValueError("Every DFL training match needs both live and dead frames")
            weights[members] = len(groups) / (2 * len(TRAIN) * members.sum())
    dead = XGBClassifier(**DEAD_PARAMS)
    dead.fit(pd.concat(dead_x, ignore_index=True), dead_targets, sample_weight=weights)
    return (ownership, dead), {"ownership_paired_rows": 2 * len(owner_targets), "dead_rows": len(groups)}


def match_predictions(raw, manifest, suffix, models):
    match_index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(suffix)
    halves = dfl_halves(raw, suffix, manifest)
    result = {view: [] for view in VIEWS}
    for half_index, section in enumerate(SECTIONS):
        half = halves[section]
        labels = pm.label_codes(half["labels"])
        for view, positions, ball in views(half, match_index, half_index):
            home, dead, raw_prediction = predict(*models, positions, ball, half["teams"])
            result[view].append((labels, home, dead, raw_prediction))
            print(f"Predicted {suffix} {section} {view}", flush=True)
    return result


def score(halves, recipe=None):
    first, second = halves
    labels = np.r_[first[0], pm.X, second[0]]
    predictions = [raw if recipe is None else decode(home, dead, **recipe)
                   for _, home, dead, raw in halves]
    prediction = np.r_[predictions[0], pm.X, predictions[1]]
    confusion = pm.scored_confusion(pm.confusion(labels, prediction))
    spells = pm.spell_scores(labels, prediction, pm.valid_runs(labels))
    dead_precision, dead_recall, _ = pm.per_class_prf(confusion, pm.D)
    return {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion),
            "dead_precision": dead_precision, "dead_recall": dead_recall,
            "matched_at_0.5": spells["per_iou"][0.5]["n_matched"],
            "predicted_spells": spells["n_pred"], "true_spells": spells["n_gt"]}


def evaluate(views_by_name, recipe):
    return {name: score(halves, recipe) for name, halves in views_by_name.items()}


def tuning_rank(trial):
    clean, masked = (trial["metrics"][view] for view in VIEWS)
    return (clean["spell_f1"] + 2 * masked["spell_f1"],
            clean["macro_f1"] + 2 * masked["macro_f1"])


def train(raw, output):
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    models, counts = fit_models(raw, manifest)
    for model, name in zip(models, MODEL_FILES):
        model.save_model(output / name)
    models = load_models(output)

    dev = match_predictions(raw, manifest, DEVELOPMENT, models)
    trials = [{"recipe": recipe, "metrics": evaluate(dev, recipe)} for recipe in GRID]
    selected = max(trials, key=tuning_rank)
    frozen = {"recipe": selected["recipe"], "models": {name: digest(output / name) for name in MODEL_FILES}}
    write_json(output / "recipe.json", frozen)
    print(f"Frozen decoder: {selected['recipe']}", flush=True)

    results = {DEVELOPMENT: {"selected": selected["metrics"], "raw": evaluate(dev, None)}}
    for suffix in HOLDOUT:
        predictions = match_predictions(raw, manifest, suffix, models)
        results[suffix] = {"selected": evaluate(predictions, frozen["recipe"]),
                           "raw": evaluate(predictions, None)}
    # Holdout can reject the DFL model but cannot choose another decoder.
    passed = all(results[match]["selected"][view]["spell_f1"] > results[match]["raw"][view]["spell_f1"]
                 for match in HOLDOUT for view in VIEWS)
    report = {"splits": {"train": TRAIN, "development": DEVELOPMENT, "held_out": HOLDOUT},
              "counts": counts, "dataset_manifest_sha256": digest(SOURCE_MANIFEST),
              "tuning": {"candidates": trials, "selection": "1/3 clean + 2/3 masked SpellF1, then weighted MacroF1"},
              "holdout_gate": "Selected SpellF1 exceeds raw on both views of each held-out match",
              "holdout_passed": passed, "matches": results}
    write_json(output / "external_validation.json", report)
    print(f"External holdout gate passed: {passed}", flush=True)
    if not passed:
        raise SystemExit("Frozen DFL recipe failed the external gate. Use the raw fallback.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl-raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="A new model output directory")
    args = parser.parse_args()
    train(args.dfl_raw, args.out)
