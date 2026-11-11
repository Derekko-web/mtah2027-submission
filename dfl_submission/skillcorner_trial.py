"""Fit SkillCorner heads and select a cross-dataset blend on external matches."""

import argparse
from itertools import product
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from dfl_submission import models as shared
from dfl_submission import skillcorner
from dfl_submission.build import write_csv
from dfl_submission.robust import CONTROL, decode_blend, evaluate, verified_models
from dfl_submission.robust_features import extras, invariant, mean
from dfl_submission.train import DEAD_PARAMS, DEVELOPMENT, HOLDOUT, SOURCE_MANIFEST, dfl_halves, views, write_json
from dfl_transfer_data import SECTIONS, digest
from dfl_transfer_features import swap
from dfl_transfer_teacher import PARAMS
from evaluation import possession_metrics as pm


def inputs(positions, ball, teams, pitch):
    extra = extras(positions, ball, teams, pitch)
    available = np.isfinite(positions).all(axis=2)
    seen = np.isfinite(ball).all(axis=1)
    fields = {"seen": seen}
    for team, prefix in enumerate(("h__", "a__")):
        xy = positions[:, teams == team]
        count = available[:, teams == team].sum(axis=1).astype(float)
        distance = np.linalg.norm(xy - ball[:, None], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        closest = distance.min(axis=1)
        closest[~np.isfinite(closest)] = np.nan
        fields[prefix + "closest"] = closest
        fields[prefix + "count"] = count
        for radius in (12, 50):
            fields[prefix + f"count_{radius}"] = mean(count, radius)
            fields[prefix + f"closest_{radius}"] = mean(closest, radius)
        for lag in (5, 25):
            speed = np.linalg.norm(xy[lag:] - xy[:-lag], axis=2) * (25 / lag)
            supported = np.isfinite(speed) & (speed < 12)
            n_speed = supported.sum(axis=1)
            speed_mean = np.divide(np.where(supported, speed, 0).sum(axis=1), n_speed,
                                   out=np.full(len(speed), np.nan), where=n_speed > 0)
            speed_mean = np.r_[np.full(lag, np.nan), speed_mean]
            for radius in (0, 25, 75):
                fields[prefix + f"player_motion_{lag}_{radius}"] = mean(speed_mean, radius)
    for radius in (12, 50, 100):
        fields[f"ball_support_{radius}"] = mean(seen.astype(float), radius)
    owner = extra.join(pd.DataFrame(fields, dtype=np.float32))
    return owner, invariant(owner)


def skill_halves(folder, match):
    for half in (1, 2):
        with np.load(folder / f"{match}_{half}.npz", allow_pickle=False) as parsed:
            yield {key: parsed[key] for key in parsed.files}


def fit(rows, *, owner):
    features, targets, groups = map(list, zip(*rows))
    inputs = pd.concat(features, ignore_index=True)
    y, group = np.concatenate(targets), np.concatenate(groups)
    weight = np.empty(len(y))
    for match in skillcorner.TRAIN:
        for label in (0, 1):
            selected = (group == match) & (y == label)
            if not selected.any():
                raise ValueError(f"SkillCorner training match lacks label support: {match}")
            weight[selected] = len(y) / (2 * len(skillcorner.TRAIN) * selected.sum())
    if owner:
        inputs = pd.concat([inputs, swap(inputs)], ignore_index=True)
        y, weight = np.r_[y, 1 - y], np.tile(weight, 2)
    params = dict(PARAMS if owner else DEAD_PARAMS, max_depth=4, n_estimators=300, n_jobs=3)
    model = XGBClassifier(**params)
    model.fit(inputs, y, sample_weight=weight)
    return model, len(y)


def fit_models(folder):
    owner, dead = [], []
    for match in skillcorner.TRAIN:
        for half_index, half in enumerate(skill_halves(folder, match), 1):
            labels = half["labels"]
            sampled = np.arange(len(labels)) % 5 == 0
            known, live = sampled & (labels != "X"), sampled & np.isin(labels, ["H", "A"])
            of, df = inputs(half["positions"], half["ball"], half["teams"], half["pitch"])
            owner.append((of.loc[live].copy(), labels[live] == "H", np.full(live.sum(), match)))
            dead.append((df.loc[known].copy(), labels[known] == "D", np.full(known.sum(), match)))
            print(f"Prepared SkillCorner {match} half {half_index}", flush=True)
    owner_model, owner_rows = fit(owner, owner=True)
    dead_model, dead_rows = fit(dead, owner=False)
    return (owner_model, dead_model), {"paired_owner_rows": owner_rows, "dead_rows": dead_rows}


def predict(control, models, positions, ball, teams, pitch):
    old_owner, old_dead, _ = shared.predict(*control, positions, ball, teams)
    owner, dead = inputs(positions, ball, teams, pitch)
    return (old_owner, old_dead, shared.ownership_probability(models[0], owner),
            models[1].predict_proba(dead)[:, 1])


def score_match(halves, control, models, *, match_index=None):
    result = {}
    for half_index, half in enumerate(halves):
        labels = pm.label_codes(half["labels"])
        if match_index is None:
            observed = [("observed", half["positions"], half["ball"])]
        else:
            observed = views(half, match_index, half_index)
        for view, positions, ball in observed:
            probabilities = predict(control, models, positions, ball, half["teams"], half["pitch"])
            result.setdefault(view, []).append((labels, probabilities))
    return result


def dfl_predictions(raw, suffix, control, models):
    from dfl_submission.train import TRAIN as DFL_TRAIN

    manifest = json.loads(SOURCE_MANIFEST.read_text())
    halves = dfl_halves(raw, suffix, manifest)
    return score_match((halves[half] for half in SECTIONS), control, models,
                       match_index=(*DFL_TRAIN, DEVELOPMENT, *HOLDOUT).index(suffix))


def skill_predictions(folder, matches, control, models):
    result = {}
    for match in matches:
        print(f"Scoring SkillCorner {match}", flush=True)
        result[str(match)] = score_match(skill_halves(folder, match), control, models)
    return result


def compare(predictions, reference, recipe):
    return {match: {"control": evaluate(p, reference), "selected": evaluate(p, recipe)}
            for match, p in predictions.items()}


def skill_average(results, key):
    return float(np.mean([result[key]["observed"]["spell_f1"] for result in results.values()]))


def dfl_average(result, key):
    return (result[key]["clean"]["spell_f1"] + 2 * result[key]["masked"]["spell_f1"]) / 3


def train(folder, raw, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["revision"] != skillcorner.REVISION:
        raise ValueError("SkillCorner source revision differs")
    for name, expected in manifest["files"].items():
        if skillcorner.identity(folder / "raw" / name) != tuple(expected):
            raise ValueError(f"SkillCorner input identity differs: {name}")
    for match in (*skillcorner.TRAIN, *skillcorner.DEVELOPMENT, *skillcorner.HOLDOUT):
        for half in (1, 2):
            name = f"{match}_{half}.npz"
            if skillcorner.identity(folder / name) != tuple(manifest["parsed"][name]):
                raise ValueError(f"SkillCorner parsed half identity differs: {name}")
    frozen = json.loads((CONTROL / "recipe.json").read_text())
    control = verified_models(CONTROL, frozen["models"])
    reference = dict(frozen["recipe"], owner_weight=0, dead_weight=0)
    models, counts = fit_models(folder)
    hashes = {}
    for model, name in zip(models, shared.MODEL_FILES):
        model.save_model(output / name)
        hashes[name] = digest(output / name)
    models = verified_models(output, hashes)
    development = skill_predictions(folder, skillcorner.DEVELOPMENT, control, models)
    dfl_dev = dfl_predictions(raw, DEVELOPMENT, control, models)
    skill_control = skill_average({match: {"control": evaluate(p, reference)} for match, p in development.items()}, "control")
    dfl_control = dfl_average({"control": evaluate(dfl_dev, reference)}, "control")
    trials = []
    for owner, dead, cutoff, radius in product((.25, .5, .75, 1.0), (.25, .5, .75, 1.0), (.4, .5, .6), (25, 50, 75)):
        recipe = dict(owner_weight=owner, dead_weight=dead, dead_cutoff=cutoff, dead_radius=radius, owner_radius=radius)
        skill = skill_average({match: {"selected": evaluate(p, recipe)} for match, p in development.items()}, "selected")
        dfl = dfl_average({"selected": evaluate(dfl_dev, recipe)}, "selected")
        trials.append(dict(recipe=recipe, skillcorner=skill, dfl=dfl))
    allowed = [trial for trial in trials if trial["dfl"] >= dfl_control - .02]
    if not allowed:
        raise ValueError("No blend stayed within .02 of DFL control on development")
    chosen = max(allowed, key=lambda trial: (trial["skillcorner"], trial["dfl"]))
    recipe = chosen["recipe"]
    write_json(output / "recipe.json", dict(recipe=recipe, models=hashes, control_models=frozen["models"]))
    print(f"Frozen cross-source blend {recipe}, development {chosen}", flush=True)
    holdout = skill_predictions(folder, skillcorner.HOLDOUT, control, models)
    dfl_holdout = {suffix: dfl_predictions(raw, suffix, control, models) for suffix in HOLDOUT}
    skill_holdout = compare(holdout, reference, recipe)
    dfl_held = compare(dfl_holdout, reference, recipe)
    skill_gain = skill_average(skill_holdout, "selected") - skill_average(skill_holdout, "control")
    dfl_gains = {match: dfl_average(value, "selected") - dfl_average(value, "control") for match, value in dfl_held.items()}
    passed = (chosen["skillcorner"] > skill_control and skill_gain > .01
              and sum(result["selected"]["observed"]["spell_f1"] > result["control"]["observed"]["spell_f1"]
                      for result in skill_holdout.values()) >= 3 and all(gain >= -.02 for gain in dfl_gains.values()))
    write_json(output / "external_validation.json", {
        "skillcorner_revision": skillcorner.REVISION, "skillcorner_manifest": manifest,
        "fit_matches": skillcorner.TRAIN, "counts": counts, "development_search": trials,
        "skillcorner_dev": compare(development, reference, recipe), "skillcorner_holdout": skill_holdout,
        "dfl_dev": compare({DEVELOPMENT: dfl_dev}, reference, recipe), "dfl_holdout": dfl_held,
        "skillcorner_holdout_gain": skill_gain, "dfl_holdout_gains": dfl_gains, "holdout_passed": passed,
        "gate": "SkillCorner gains on development and >=3/4 holdouts, mean holdout gain >.01; both DFL holdouts within .02",
    })
    print(f"Cross-source comparison passed: {passed}; SkillCorner gain {skill_gain:.4f}, DFL {dfl_gains}", flush=True)


def build(folder, output):
    from dfl_submission.build import tactical_halves

    report = json.loads((folder / "external_validation.json").read_text())
    if report["holdout_passed"] is not True:
        raise ValueError("Cross-source blend failed external validation")
    frozen = json.loads((folder / "recipe.json").read_text())
    control = verified_models(CONTROL, frozen["control_models"])
    models = verified_models(folder, frozen["models"])
    def decode_half(positions, ball, teams, pitch):
        return decode_blend(predict(control, models, positions, ball, teams, pitch), frozen["recipe"])

    labels = tactical_halves(decode_half, centered=True)
    write_csv(output, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--skillcorner", required=True, type=Path)
    fitting.add_argument("--dfl-raw", required=True, type=Path)
    fitting.add_argument("--out", required=True, type=Path)
    inference = commands.add_parser("build")
    inference.add_argument("--model-dir", required=True, type=Path)
    inference.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "train":
        if os.environ.get("MTAH_DATA_DIR") is None:
            raise ValueError("Set MTAH_DATA_DIR to a nonexistent directory for external-only fitting")
        if Path(os.environ["MTAH_DATA_DIR"]).exists():
            raise ValueError("External-only fitting requires the competition bundle to be unavailable")
        train(args.skillcorner, args.dfl_raw, args.out)
    else:
        build(args.model_dir, args.out)
