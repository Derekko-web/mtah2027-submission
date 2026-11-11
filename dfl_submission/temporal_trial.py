"""Compare temporal models and decoders on external matches, then build the frozen winner."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np

from dfl_submission import adaptive_decoder, context_trial, external_benchmark, long_context, models, temporal_cpu
from dfl_submission.build import tactical_halves, write_csv
from dfl_submission.train import write_json
from evaluation import possession_metrics as pm


SINGLES = tuple(f"temporal_{step}.npz" for step in temporal_cpu.STEPS)
VARIANTS = {filename: (filename,) for filename in SINGLES} | {"late_mean": SINGLES[-2:]}


@dataclass
class Match:
    truth: np.ndarray
    runs: list
    halves: list


def fallback(folder):
    package = folder / "dfl_submission"
    trial = package / "context_recipe"
    frozen = json.loads((trial / "recipe.json").read_text())
    report = json.loads((trial / "external_validation.json").read_text())
    if report["passed"] is not True or frozen["recipe"]["owner_mix"] != 1 or frozen["recipe"]["dead_mix"] != 1:
        raise ValueError("Expected the verified standalone extended-context model")
    model, fit = long_context.load(package / "long_models")
    if fit["models"] != frozen["long_context"]:
        raise ValueError("Fallback checkpoints differ from its frozen recipe")
    return model, frozen, report


def forecast(reference, checkpoints, positions, ball, teams, pitch):
    frames, owner, dead = long_context.features(positions, ball, teams, pitch)
    return forecast_features(reference, checkpoints, frames, owner, dead, len(ball))


def forecast_features(reference, checkpoints, frames, owner, dead, length):
    baseline = None
    if reference is not None:
        home_p = models.ownership_probability(reference[0], owner)
        dead_p = reference[1].predict_proba(dead)[:, 1]
        timeline = np.arange(length)
        baseline = tuple(np.interp(timeline, frames, p).astype(np.float32) for p in (home_p, dead_p))
    features = frames, owner.to_numpy(dtype=np.float32), owner.columns.tolist(), length
    neural = {name: temporal_cpu.predict(model, features) for name, model in checkpoints.items()}
    if all(name in neural for name in VARIANTS["late_mean"]):
        first, second = (neural[name] for name in VARIANTS["late_mean"])
        neural["late_mean"] = tuple((a + b) / 2 for a, b in zip(first, second))
    return baseline, neural


def predictions(dfl, skill, split, reference, checkpoints, *, raw=False):
    collected = defaultdict(list)
    for half in external_benchmark.halves(dfl, skill, split, raw=raw):
        predicted = forecast(reference, checkpoints, half.positions, half.ball, half.teams, half.pitch)
        collected[half.source, half.match, half.view].append((half.labels, predicted))
        print(f"Predicted {split} {half.key}", flush=True)
    result = {}
    for key, halves in collected.items():
        if len(halves) != 2:
            raise ValueError(f"External match requires two halves: {key}")
        truth = np.r_[halves[0][0], pm.X, halves[1][0]]
        result[key] = Match(truth, pm.valid_runs(truth), [halves[0][1], halves[1][1]])
    return result


def decode(predictions, recipe):
    baseline, neural = predictions
    variant = recipe["variant"]
    if variant is None:
        old_home, old_dead = baseline
        new_home, new_dead = old_home, old_dead
    else:
        new_home, new_dead, dead_edge, owner_edge = neural[variant]
        old_home, old_dead = baseline if baseline is not None else (new_home, new_dead)
    if recipe["method"] == "adaptive":
        return adaptive_decoder.decode(
            context_trial.blend(old_home, new_home, recipe["owner_mix"]),
            context_trial.blend(old_dead, new_dead, recipe["dead_mix"]), dead_edge, owner_edge,
            cutoff=recipe["dead_cutoff"], dead_cost=recipe["dead_cost"], owner_cost=recipe["owner_cost"],
            discount=recipe["discount"],
        )
    return context_trial.decode_recipe((old_home, old_dead, new_home, new_dead), recipe)


def metrics(matches, recipe):
    result = {}
    for key, match in matches.items():
        prediction = np.r_[decode(match.halves[0], recipe), pm.X, decode(match.halves[1], recipe)]
        score = pm.spell_scores(match.truth, prediction, match.runs)
        cm = pm.scored_confusion(pm.confusion(match.truth, prediction))
        result[key] = {"spell_f1": score["mean_f1"], "macro_f1": pm.macro_f1(cm), "spells": score["n_pred"]}
    return result


def grade(matches, recipe):
    result = metrics(matches, recipe)
    return {"recipe": recipe, "spell_f1": context_trial.aggregate(result), "macro_f1": context_trial.aggregate(result, "macro_f1")}


def baseline_recipe(frozen):
    return dict(frozen["recipe"], method="centered", variant=None, owner_mix=0, dead_mix=0)


def tune(dfl, skill, reference_folder, models_folder, output, *, raw=False):
    output.mkdir(parents=True, exist_ok=False)
    reference, frozen, _ = fallback(reference_folder)
    fit = json.loads((models_folder / "fit.json").read_text())
    checkpoints = {name: temporal_cpu.load(models_folder, name) for name in SINGLES}
    matches = predictions(dfl, skill, "development", reference, checkpoints, raw=raw)
    base = grade(matches, baseline_recipe(frozen))
    trials = [base]
    radii = ((10, 10), (25, 25), (40, 25), (25, 40))
    for variant, owner, dead, cutoff, (dead_radius, owner_radius) in product(
        (*VARIANTS, None), (0, .5, 1), (0, .5, 1), (.50, .57, .64), radii,
    ):
        if (owner == dead == 0) != (variant is None):
            continue
        recipe = dict(variant=variant, method="centered", owner_mix=owner, dead_mix=dead, dead_cutoff=cutoff,
                      dead_radius=dead_radius, owner_radius=owner_radius, island_frames=40, island_confidence=.82)
        trials.append(grade(matches, recipe))
    finalists = sorted(trials, key=context_trial.rank, reverse=True)[:8]
    for item, frames in product(finalists, (0, 20)):
        trials.append(grade(matches, dict(item["recipe"], island_frames=frames)))
    unique = {}
    for item in sorted((item for item in trials if item["recipe"]["variant"] is not None),
                       key=context_trial.rank, reverse=True):
        recipe = item["recipe"]
        key = tuple(recipe[name] for name in ("variant", "owner_mix", "dead_mix", "dead_cutoff"))
        unique.setdefault(key, recipe)
    for recipe, dead_cost, owner_cost, discount in product(list(unique.values())[:3], (2, 5, 10), (2, 5, 10), (0, .8)):
        settings = dict(recipe, method="adaptive", dead_cost=dead_cost, owner_cost=owner_cost, discount=discount)
        trials.append(grade(matches, settings))
    allowed = [trial for trial in trials if all(trial["spell_f1"][source] >= base["spell_f1"][source] - .02
                                               for source in ("dfl", "skillcorner"))]
    selected = max(allowed, key=context_trial.rank)
    recipe = selected["recipe"]
    variant = recipe["variant"]
    model_hashes = {name: fit["models"][name] for name in VARIANTS[variant]} if variant is not None else {}
    write_json(output / "recipe.json", {"fallback": frozen, "neural": model_hashes,
               "normalization": fit["normalization"], "recipe": recipe})
    passed = selected["spell_f1"]["combined"] >= base["spell_f1"]["combined"] + .003
    write_json(output / "development.json", {"reference": base, "selected": selected, "trials": trials, "passed": passed})
    print(f"Frozen temporal development comparison {base['spell_f1']} -> {selected}", flush=True)


def verified(models_folder, frozen):
    fit = json.loads((models_folder / "fit.json").read_text())
    if fit["normalization"] != frozen["normalization"] or any(fit["models"][name] != hash_value
                                                               for name, hash_value in frozen["neural"].items()):
        raise ValueError("Frozen temporal checkpoints differ")
    return {name: temporal_cpu.load(models_folder, name) for name in frozen["neural"]}


def confirm(dfl, skill, reference_folder, models_folder, output, *, raw=False):
    development = json.loads((output / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("Temporal recipe did not pass external development")
    recipe = json.loads((output / "recipe.json").read_text())
    reference, frozen, report = fallback(reference_folder)
    if frozen != recipe["fallback"]:
        raise ValueError("Reference recipe differs from development")
    checkpoints = verified(models_folder, recipe)
    matches = predictions(dfl, skill, "holdout", reference, checkpoints, raw=raw)
    reference_match, selected_match = metrics(matches, baseline_recipe(frozen)), metrics(matches, recipe["recipe"])
    base, selected = context_trial.aggregate(reference_match), context_trial.aggregate(selected_match)
    if any(abs(value - report["selected"][key]) > 1e-10 for key, value in base.items()):
        raise ValueError(f"Verified 0.5645 reference score did not reproduce: {base}")
    gain = {key: selected[key] - base[key] for key in base}
    passed = gain["combined"] >= .005 and min(gain["dfl"], gain["skillcorner"]) >= -.02
    result = {"passed": passed, "reference": base, "selected": selected, "gain": gain,
              "reference_matches": context_trial.serialize(reference_match), "selected_matches": context_trial.serialize(selected_match),
              "gate": "Combined gain at least .005, neither dataset loses more than .02"}
    write_json(output / "external_validation.json", result)
    print(json.dumps({key: result[key] for key in ("passed", "reference", "selected", "gain")}), flush=True)


def build(reference_folder, models_folder, trial, output):
    report = json.loads((trial / "external_validation.json").read_text())
    if report["passed"] is not True:
        raise ValueError("Temporal challenger failed external confirmation")
    frozen = json.loads((trial / "recipe.json").read_text())
    recipe = frozen["recipe"]
    checkpoints = verified(models_folder, frozen)
    reference = None
    if recipe["owner_mix"] != 1 or recipe["dead_mix"] != 1:
        if reference_folder is None:
            raise ValueError("This blend requires --fallback")
        reference, fallback_recipe, _ = fallback(reference_folder)
        if fallback_recipe != frozen["fallback"]:
            raise ValueError("Fallback identity differs")

    def decode_half(positions, ball, teams, pitch):
        return decode(forecast(reference, checkpoints, positions, ball, teams, pitch), recipe)

    write_csv(output, tactical_halves(decode_half, centered=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("tune", "confirm", "build"):
        command = commands.add_parser(action)
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--trial", required=True, type=Path)
        command.add_argument("--fallback", required=action != "build", type=Path)
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
