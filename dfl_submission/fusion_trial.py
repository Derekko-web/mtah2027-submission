"""Evaluate a blend of the richer DFL heads and the SkillCorner heads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path

import numpy as np

from dfl_submission import models as shared, robust, skillcorner, skillcorner_trial as skill
from dfl_submission.build import tactical_halves, write_csv
from dfl_submission.robust_features import invariant
from dfl_submission.train import DEVELOPMENT, HOLDOUT, SOURCE_MANIFEST, TRAIN, dfl_halves, views, write_json
from dfl_transfer_data import SECTIONS, digest
from evaluation import possession_metrics as pm


def load_pair(dfl_folder, skill_folder):
    dfl_frozen = json.loads((dfl_folder / "recipe.json").read_text())
    skill_frozen = json.loads((skill_folder / "recipe.json").read_text())
    if dfl_frozen["recipe"]["owner_weight"] != 1 or dfl_frozen["recipe"]["dead_weight"] != 1:
        raise ValueError("Fusion expects the standalone richer DFL heads")
    return (robust.verified_models(dfl_folder, dfl_frozen["models"]),
            robust.verified_models(skill_folder, skill_frozen["models"])), dfl_frozen, skill_frozen


def predict(models, positions, ball, teams, pitch):
    dfl_models, skill_models = models
    owner, dead, extra = robust.features(positions, ball, teams, pitch)
    dfl_home = shared.ownership_probability(dfl_models[0], owner.join(extra))
    dfl_dead = dfl_models[1].predict_proba(dead.join(invariant(extra)))[:, 1]
    owner, dead = skill.inputs(positions, ball, teams, pitch)
    return dfl_home, dfl_dead, shared.ownership_probability(skill_models[0], owner), skill_models[1].predict_proba(dead)[:, 1]


def external_halves(folder, suffix, *, raw):
    if raw:
        manifest = json.loads(SOURCE_MANIFEST.read_text())
        return dfl_halves(folder, suffix, manifest)
    hashes = json.loads((folder / "manifest.json").read_text())["cache_sha256"]
    result = {}
    for half in SECTIONS:
        path = folder / f"DFL-MAT-J03{suffix}_{half}.npz"
        if digest(path) != hashes[path.name]:
            raise ValueError(f"DFL parsed half identity differs: {path}")
        with np.load(path, allow_pickle=False) as source:
            result[half] = {name: source[name] for name in ("labels", "positions", "ball", "teams", "pitch")}
    return result


def external_predictions(models, dfl_folder, skill_folder, suffixes, matches, *, raw):
    skill_hashes = json.loads((skill_folder / "manifest.json").read_text())
    if skill_hashes["revision"] != skillcorner.REVISION:
        raise ValueError("SkillCorner source revision differs")
    dfl_result, skill_result = {}, {}
    for suffix in suffixes:
        index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(suffix)
        dfl_result[suffix] = {"clean": [], "masked": []}
        halves = external_halves(dfl_folder, suffix, raw=raw)
        for i, half in enumerate(SECTIONS):
            rows = halves[half]
            labels = pm.label_codes(rows["labels"])
            for view, positions, ball in views(rows, index, i):
                p = predict(models, positions, ball, rows["teams"], rows["pitch"])
                dfl_result[suffix][view].append((labels, p))
        print(f"Scored DFL {suffix}", flush=True)
    for match in matches:
        skill_result[str(match)] = {"observed": []}
        for i, half in enumerate(skill.skill_halves(skill_folder, match), 1):
            name = f"{match}_{i}.npz"
            if skillcorner.identity(skill_folder / name) != tuple(skill_hashes["parsed"][name]):
                raise ValueError(f"SkillCorner parsed half identity differs: {name}")
            labels = pm.label_codes(half["labels"])
            p = predict(models, half["positions"], half["ball"], half["teams"], half["pitch"])
            skill_result[str(match)]["observed"].append((labels, p))
        print(f"Scored SkillCorner {match}", flush=True)
    return dfl_result, skill_result


def tune(dfl_model, skill_model, dfl_data, skill_data, output, *, raw):
    output.mkdir(parents=True, exist_ok=False)
    models, dfl_frozen, skill_frozen = load_pair(dfl_model, skill_model)
    reference = dict(dfl_frozen["recipe"], owner_weight=0, dead_weight=0)
    dfl_dev, skill_dev = external_predictions(models, dfl_data, skill_data, (DEVELOPMENT,), skillcorner.DEVELOPMENT, raw=raw)
    dfl_control = skill.dfl_average({"control": robust.evaluate(dfl_dev[DEVELOPMENT], reference)}, "control")
    trials = []
    for owner, dead, cutoff, radius in product((0, .25, .5, .75), (.25, .5, .75, 1), (.45, .5, .55), (35, 50, 65)):
        recipe = dict(owner_weight=owner, dead_weight=dead, dead_cutoff=cutoff, dead_radius=radius, owner_radius=radius)
        dfl_score = skill.dfl_average({"selected": robust.evaluate(dfl_dev[DEVELOPMENT], recipe)}, "selected")
        skill_score = skill.skill_average({match: {"selected": robust.evaluate(p, recipe)}
                                          for match, p in skill_dev.items()}, "selected")
        trials.append(dict(recipe=recipe, dfl=dfl_score, skillcorner=skill_score))
    admissible = [trial for trial in trials if trial["dfl"] >= dfl_control - .01]
    if not admissible:
        raise ValueError("Fusion candidates exceeded the DFL development loss bound")
    chosen = max(admissible, key=lambda trial: trial["dfl"] + trial["skillcorner"])
    recipe = chosen["recipe"]
    frozen = dict(recipe=recipe, dfl_models=dfl_frozen["models"], skillcorner_models=skill_frozen["models"])
    write_json(output / "recipe.json", frozen)
    write_json(output / "development.json", dict(chosen=chosen, trials=trials))
    print(f"Frozen fusion: {chosen}", flush=True)
    confirm(dfl_model, skill_model, dfl_data, skill_data, output, raw=raw)


def confirm(dfl_model, skill_model, dfl_data, skill_data, output, *, raw):
    frozen = json.loads((output / "recipe.json").read_text())
    models, dfl_frozen, skill_frozen = load_pair(dfl_model, skill_model)
    if frozen["dfl_models"] != dfl_frozen["models"] or frozen["skillcorner_models"] != skill_frozen["models"]:
        raise ValueError("Frozen fusion checkpoints differ")
    if (output / "external_validation.json").exists():
        raise ValueError("Fusion validation already exists")
    recipe = frozen["recipe"]
    reference = dict(dfl_frozen["recipe"], owner_weight=0, dead_weight=0)
    tasks = [((match,), ()) for match in HOLDOUT] + [((), (match,)) for match in skillcorner.HOLDOUT]
    dfl_held, skill_held = {}, {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(external_predictions, models, dfl_data, skill_data, dfl, skill, raw=raw)
                   for dfl, skill in tasks]
        for job in pending:
            dfl, other = job.result()
            dfl_held.update(dfl)
            skill_held.update(other)
    dfl_results = skill.compare(dfl_held, reference, recipe)
    skill_results = skill.compare(skill_held, reference, recipe)
    gains = {match: skill.dfl_average(result, "selected") - skill.dfl_average(result, "control")
             for match, result in dfl_results.items()}
    skill_gain = skill.skill_average(skill_results, "selected") - skill.skill_average(skill_results, "control")
    passed = bool(all(gain >= -.02 for gain in gains.values()) and skill_gain > .02
                  and skill_gain + np.mean(list(gains.values())) > 0)
    write_json(output / "external_validation.json", dict(holdout_passed=passed,
               dfl_holdout=dfl_results, skillcorner_holdout=skill_results, dfl_holdout_gains=gains,
               skillcorner_holdout_gain=skill_gain, chosen=recipe,
               gate="Both DFL holdouts within .02 of richer DFL; SkillCorner mean gain above .02 and positive combined gain"))
    print(f"Fusion comparison passed: {passed}; DFL {gains}; SkillCorner gain {skill_gain:.4f}", flush=True)


def build(dfl_model, skill_model, fusion, output):
    report = json.loads((fusion / "external_validation.json").read_text())
    if report["holdout_passed"] is not True:
        raise ValueError("Fusion failed its external gate")
    frozen = json.loads((fusion / "recipe.json").read_text())
    models = (robust.verified_models(dfl_model, frozen["dfl_models"]),
              robust.verified_models(skill_model, frozen["skillcorner_models"]))

    def decode(positions, ball, teams, pitch):
        return robust.decode_blend(predict(models, positions, ball, teams, pitch), frozen["recipe"])

    write_csv(output, tactical_halves(decode, centered=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    fitting = commands.add_parser("tune")
    confirmation = commands.add_parser("confirm")
    for command in (fitting, confirmation):
        external = command.add_mutually_exclusive_group(required=True)
        external.add_argument("--dfl-data", type=Path, help="Verified DFL half caches")
        external.add_argument("--dfl-raw", type=Path, help="Original checksum-pinned DFL XML files")
        command.add_argument("--skill-data", type=Path, required=True)
    inference = commands.add_parser("build")
    inference.add_argument("--fusion", type=Path, required=True)
    for command in (fitting, confirmation, inference):
        command.add_argument("--dfl-model", type=Path, required=True)
        command.add_argument("--skill-model", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    if args.action != "build":
        require_external_only()
    if args.action in ("tune", "confirm"):
        action = tune if args.action == "tune" else confirm
        action(args.dfl_model, args.skill_model, args.dfl_raw or args.dfl_data, args.skill_data, args.out,
               raw=args.dfl_raw is not None)
    else:
        build(args.dfl_model, args.skill_model, args.fusion, args.out)
