"""Compare player-motion and joint-decoding challengers on external matches."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from dfl_submission import (
    adaptive_decoder, context_trial, external_benchmark, joint_decoder, long_context,
    motion_cpu, motion_features, temporal_trial,
)
from dfl_submission.build import tactical_halves, write_csv
from dfl_submission.train import write_json
from evaluation import possession_metrics as pm


SINGLES = tuple(f"motion_{step}.npz" for step in motion_cpu.STEPS)
VARIANTS = {name: (name,) for name in SINGLES} | {"late_mean": SINGLES[-2:]}
REFERENCE = {"method": "reference"}


def fallback(folder):
    package = folder / "dfl_submission"
    trial = package / "temporal_recipe"
    frozen = json.loads((trial / "recipe.json").read_text())
    report = json.loads((trial / "external_validation.json").read_text())
    reference, prior, _ = temporal_trial.fallback(folder)
    if report["passed"] is not True or frozen["fallback"] != prior:
        raise ValueError("Expected the verified temporal blend and its original XGBoost heads")
    checkpoints = temporal_trial.verified(package / "temporal_models", frozen)
    return reference, checkpoints, frozen, report


def forecast(reference, checkpoints, positions, ball, teams, pitch):
    trees, neural, _, _ = reference
    baseline = temporal_trial.forecast(trees, neural, positions, ball, teams, pitch)
    candidates = {}
    if checkpoints:
        frames, features, _ = long_context.features(positions, ball, teams, pitch)
        context = frames, features.to_numpy(dtype=np.float32), features.columns.tolist(), len(ball)
        players = motion_features.features(positions, ball, teams, frames)
        candidates = {name: motion_cpu.predict(model, context, players) for name, model in checkpoints.items()}
        if all(name in candidates for name in VARIANTS["late_mean"]):
            first, second = (candidates[name] for name in VARIANTS["late_mean"])
            candidates["late_mean"] = tuple((a + b) / 2 for a, b in zip(first, second))
    return baseline, candidates


def predictions(dfl, skill, split, reference, checkpoints, *, raw=False):
    records = defaultdict(list)
    for half in external_benchmark.halves(dfl, skill, split, raw=raw):
        p = forecast(reference, checkpoints, half.positions, half.ball, half.teams, half.pitch)
        records[half.source, half.match, half.view].append((half.labels, p))
        print(f"Predicted {split} {half.key}", flush=True)
    matches = {}
    for key, halves in records.items():
        if len(halves) != 2:
            raise ValueError(f"External match requires two halves: {key}")
        truth = np.r_[halves[0][0], pm.X, halves[1][0]]
        matches[key] = temporal_trial.Match(truth, pm.valid_runs(truth), [part[1] for part in halves])
    return matches


def blend_signals(predicted, previous, recipe):
    baseline, candidates = predicted
    trees, neural = baseline
    old_home, old_dead, dead_edge, owner_edge = neural[previous["variant"]]
    old_home = context_trial.blend(trees[0], old_home, previous["owner_mix"])
    old_dead = context_trial.blend(trees[1], old_dead, previous["dead_mix"])
    if recipe["variant"] is not None:
        home, dead, de, oe = candidates[recipe["variant"]]
        old_home = context_trial.blend(old_home, home, recipe["owner_mix"])
        old_dead = context_trial.blend(old_dead, dead, recipe["dead_mix"])
        dead_edge = context_trial.blend(dead_edge, de, recipe["dead_mix"])
        owner_edge = context_trial.blend(owner_edge, oe, recipe["owner_mix"])
    return old_home, old_dead, dead_edge, owner_edge


def decode(predicted, previous, recipe, *, restart=None):
    if restart is not None and recipe["method"] != "joint":
        raise ValueError("Restart anchors require the joint decoder")
    if recipe["method"] == "reference":
        return temporal_trial.decode(predicted[0], previous)
    signals = blend_signals(predicted, previous, recipe)
    decoder = joint_decoder if recipe["method"] == "joint" else adaptive_decoder
    extra = {} if restart is None else {"restart": restart}
    return decoder.decode(*signals, cutoff=recipe["cutoff"],
                          dead_cost=recipe["dead_cost"], owner_cost=recipe["owner_cost"], discount=recipe["discount"],
                          **extra)


def scores(matches, previous, recipe):
    result = {}
    for key, match in matches.items():
        prediction = np.r_[decode(match.halves[0], previous, recipe), pm.X, decode(match.halves[1], previous, recipe)]
        spells = pm.spell_scores(match.truth, prediction, match.runs)
        cm = pm.scored_confusion(pm.confusion(match.truth, prediction))
        result[key] = {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(cm), "spells": spells["n_pred"]}
    return result


def grade(matches, previous, recipe):
    result = scores(matches, previous, recipe)
    return {"recipe": recipe, "spell_f1": context_trial.aggregate(result), "macro_f1": context_trial.aggregate(result, "macro_f1")}


def search(matches, previous, variants):
    base = grade(matches, previous, REFERENCE)
    coarse = []
    for variant, owner, dead, cutoff in product((*variants, None), (0, .5, 1), (0, .5, 1), (.53, .57, .61)):
        if (owner == dead == 0) != (variant is None):
            continue
        recipe = dict(variant=variant, method="separate", owner_mix=owner, dead_mix=dead, cutoff=cutoff,
                      dead_cost=2, owner_cost=2, discount=.8)
        coarse.append(grade(matches, previous, recipe))
    shortlist = {}
    for item in sorted(coarse, key=context_trial.rank, reverse=True):
        recipe = item["recipe"]
        identity = tuple(recipe[key] for key in ("variant", "owner_mix", "dead_mix"))
        shortlist.setdefault(identity, recipe)
    trials = [base, *coarse]
    for recipe, method, (dead_cost, owner_cost), discount in product(
        list(shortlist.values())[:6], ("separate", "joint"), ((1, 1), (2, 2), (3, 2), (2, 3), (4, 3)), (.55, .8),
    ):
        settings = dict(recipe, method=method, dead_cost=dead_cost, owner_cost=owner_cost, discount=discount)
        if settings != recipe:
            trials.append(grade(matches, previous, settings))
    eligible = [trial for trial in trials if all(trial["spell_f1"][source] >= base["spell_f1"][source] - .02
                                                for source in ("dfl", "skillcorner"))]
    selected = max(eligible, key=context_trial.rank)
    passed = selected["spell_f1"]["combined"] >= base["spell_f1"]["combined"] + .003
    return {"reference": base, "selected": selected, "trials": trials, "passed": passed}


def tune(dfl, skill, reference_folder, models_folder, output, *, raw=False):
    output.mkdir(parents=True, exist_ok=False)
    reference = fallback(reference_folder)
    fit = json.loads((models_folder / "fit.json").read_text())
    checkpoints = {name: motion_cpu.load(models_folder, name) for name in SINGLES}
    matches = predictions(dfl, skill, "development", reference, checkpoints, raw=raw)
    development = search(matches, reference[2]["recipe"], VARIANTS)
    selected = development["selected"]
    recipe = selected["recipe"]
    variant = recipe.get("variant")
    hashes = {name: fit["models"][name] for name in VARIANTS[variant]} if variant is not None else {}
    write_json(output / "recipe.json", {"fallback": reference[2], "models": hashes, "normalization": fit["normalization"],
                                       "recipe": recipe})
    write_json(output / "development.json", development)
    print(f"Frozen motion development comparison {development['reference']['spell_f1']} -> {selected}", flush=True)


def verified(folder, frozen):
    fit = json.loads((folder / "fit.json").read_text())
    if fit["normalization"] != frozen["normalization"] or any(fit["models"][key] != value for key, value in frozen["models"].items()):
        raise ValueError("Frozen player-motion checkpoint identity differs")
    return {key: motion_cpu.load(folder, key) for key in frozen["models"]}


def confirm(dfl, skill, reference_folder, models_folder, output, *, raw=False):
    development = json.loads((output / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("Challenger did not pass external development")
    frozen = json.loads((output / "recipe.json").read_text())
    reference = fallback(reference_folder)
    if reference[2] != frozen["fallback"]:
        raise ValueError("Temporal reference differs from the frozen recipe")
    checkpoints = verified(models_folder, frozen)
    matches = predictions(dfl, skill, "holdout", reference, checkpoints, raw=raw)
    report_comparison(matches, reference, frozen, output)


def report_comparison(matches, reference, frozen, output):
    previous = reference[2]["recipe"]
    baseline, challenger = scores(matches, previous, REFERENCE), scores(matches, previous, frozen["recipe"])
    base, selected = context_trial.aggregate(baseline), context_trial.aggregate(challenger)
    if any(abs(base[key] - reference[3]["selected"][key]) > 1e-10 for key in base):
        raise ValueError(f"Verified 0.5818 reference score did not reproduce: {base}")
    gain = {key: selected[key] - base[key] for key in base}
    passed = gain["combined"] >= .005 and min(gain["dfl"], gain["skillcorner"]) >= -.02
    report = {"passed": passed, "reference": base, "selected": selected, "gain": gain,
              "reference_matches": context_trial.serialize(baseline), "selected_matches": context_trial.serialize(challenger),
              "gate": "Combined gain at least .005, neither dataset loses more than .02"}
    write_json(output / "external_validation.json", report)
    print(json.dumps({key: report[key] for key in ("passed", "reference", "selected", "gain")}), flush=True)


def build(reference_folder, models_folder, trial, output):
    if json.loads((trial / "external_validation.json").read_text())["passed"] is not True:
        raise ValueError("Challenger failed external confirmation")
    frozen = json.loads((trial / "recipe.json").read_text())
    reference = fallback(reference_folder)
    if reference[2] != frozen["fallback"]:
        raise ValueError("Temporal fallback identity differs")
    checkpoints = verified(models_folder, frozen)

    def decode_half(positions, ball, teams, pitch):
        p = forecast(reference, checkpoints, positions, ball, teams, pitch)
        return decode(p, reference[2]["recipe"], frozen["recipe"])

    write_csv(output, tactical_halves(decode_half, centered=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("tune", "confirm", "build"):
        command = commands.add_parser(action)
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--trial", required=True, type=Path)
        command.add_argument("--fallback", required=True, type=Path)
        if action != "build":
            command.add_argument("--dfl", required=True, type=Path)
            command.add_argument("--dfl-raw", action="store_true")
            command.add_argument("--skill", required=True, type=Path)
        else:
            command.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    if args.action != "build":
        require_external_only()
    if args.action == "build":
        build(args.fallback, args.model, args.trial, args.out)
    else:
        action = tune if args.action == "tune" else confirm
        action(args.dfl, args.skill, args.fallback, args.model, args.trial, raw=args.dfl_raw)
