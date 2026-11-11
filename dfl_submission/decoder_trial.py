"""Retune the joint decoder's switch costs against controlled-possession external targets.

The protocol is fixed in DECISIONS.md (D1) before any scoring: tune on the external development
matches, confirm once on the external holdout matches, then build with the frozen costs.
"""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from dfl_submission import context_trial, ensemble_trial, motion_trial
from dfl_submission.build import tactical_halves, write_csv
from dfl_submission.conventions import collapse_brief_changes
from dfl_submission.train import write_json
from evaluation import possession_metrics as pm


CONVENTIONS = {"raw": None, "aba1.75": 1.75, "aba3": 3.0, "aba4": 4.0, "aba4.5": 4.5}
PRIMARY = "aba3"
DEVELOPMENT_GUARDS = ("aba1.75", "aba4.5")
CONFIRMATION_GUARDS = ("aba1.75", "aba4", "aba4.5")
OWNER_COSTS = (2, 3, 4, 6, 8, 12, 16)
DEAD_COSTS = (2, 3, 4)
GUARD = .01
MINIMUM_GAIN = .005


def relabel(truth, seconds):
    if seconds is None:
        return truth
    return collapse_brief_changes(truth, seconds)


def targets(matches):
    """Truth arrays per convention. The half separator is X, so no merge crosses halves."""
    result = {}
    for name, seconds in CONVENTIONS.items():
        result[name] = {}
        for key, match in matches.items():
            truth = relabel(match.truth, seconds)
            result[name][key] = (truth, pm.valid_runs(truth))
    return result


def decoded(matches, previous, recipe):
    return {key: np.r_[motion_trial.decode(match.halves[0], previous, recipe), pm.X,
                       motion_trial.decode(match.halves[1], previous, recipe)]
            for key, match in matches.items()}


def grade(predictions, truths):
    records = {}
    for key, prediction in predictions.items():
        truth, runs = truths[key]
        spells = pm.spell_scores(truth, prediction, runs)
        confusion = pm.scored_confusion(pm.confusion(truth, prediction))
        records[key] = {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion), "spells": spells["n_pred"]}
    return {"spell_f1": context_trial.aggregate(records), "macro_f1": context_trial.aggregate(records, "macro_f1"),
            "matches": {"/".join(key): value for key, value in records.items()}}


def evaluate(matches, previous, recipe, truths):
    predictions = decoded(matches, previous, recipe)
    return {"recipe": recipe, "conventions": {name: grade(predictions, truths[name]) for name in CONVENTIONS}}


def combined(trial, name, metric="spell_f1"):
    return trial["conventions"][name][metric]["combined"]


def load(fallback, models, ensemble):
    reference, checkpoints, frozen, variants = ensemble_trial.load(fallback, models, ensemble)
    return reference, checkpoints, frozen, variants


def tune(dfl, skill, fallback, models, ensemble, output, *, raw=False):
    output.mkdir(parents=True, exist_ok=False)
    reference, checkpoints, frozen, variants = load(fallback, models, ensemble)
    previous = reference[2]["recipe"]
    matches = ensemble_trial.predictions(dfl, skill, "development", reference, checkpoints, variants, raw=raw)
    truths = targets(matches)
    base = evaluate(matches, previous, frozen["recipe"], truths)
    trials = [evaluate(matches, previous, dict(frozen["recipe"], owner_cost=owner, dead_cost=dead), truths)
              for owner, dead in product(OWNER_COSTS, DEAD_COSTS)]
    eligible = [trial for trial in trials
                if all(combined(trial, name) >= combined(base, name) - GUARD for name in DEVELOPMENT_GUARDS)]
    selected = max(eligible, key=lambda trial: (combined(trial, PRIMARY), combined(trial, PRIMARY, "macro_f1")))
    passed = combined(selected, PRIMARY) > combined(base, PRIMARY)
    write_json(output / "development.json", {"protocol": "DECISIONS.md D1", "reference": base, "selected": selected,
                                             "trials": trials, "passed": passed})
    write_json(output / "recipe.json", {"ensemble": frozen, "recipe": selected["recipe"]})
    print(f"Development {PRIMARY}: {combined(base, PRIMARY):.4f} -> {combined(selected, PRIMARY):.4f} "
          f"with owner_cost={selected['recipe']['owner_cost']} dead_cost={selected['recipe']['dead_cost']}", flush=True)


def frozen_trial(fallback, models, ensemble, trial):
    reference, checkpoints, frozen, variants = load(fallback, models, ensemble)
    recipe = json.loads((trial / "recipe.json").read_text())
    if recipe["ensemble"] != frozen:
        raise ValueError("The frozen ensemble differs from the one the decoder was tuned on")
    return reference, checkpoints, frozen, variants, recipe["recipe"]


def confirm(dfl, skill, fallback, models, ensemble, trial, *, raw=False):
    development = json.loads((trial / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("The decoder retune did not improve the development primary score")
    if (trial / "external_validation.json").exists():
        raise ValueError("Confirmation already ran; the holdout is scored once")
    reference, checkpoints, frozen, variants, recipe = frozen_trial(fallback, models, ensemble, trial)
    previous = reference[2]["recipe"]
    matches = ensemble_trial.predictions(dfl, skill, "holdout", reference, checkpoints, variants, raw=raw)
    truths = targets(matches)
    base = evaluate(matches, previous, frozen["recipe"], truths)
    challenger = evaluate(matches, previous, recipe, truths)
    gain = {name: combined(challenger, name) - combined(base, name) for name in CONVENTIONS}
    passed = gain[PRIMARY] >= MINIMUM_GAIN and all(gain[name] >= -GUARD for name in CONFIRMATION_GUARDS)
    write_json(trial / "external_validation.json", {"protocol": "DECISIONS.md D1", "reference": base,
                                                    "selected": challenger, "gain": gain, "passed": passed})
    print("Holdout gains: " + ", ".join(f"{name} {value:+.4f}" for name, value in gain.items())
          + f" -> {'passed' if passed else 'failed'}", flush=True)


def build(fallback, models, ensemble, trial, output):
    report = json.loads((trial / "external_validation.json").read_text())
    reference, checkpoints, frozen, variants, recipe = frozen_trial(fallback, models, ensemble, trial)
    if report["passed"] is not True:
        raise ValueError("The decoder retune did not pass confirmation; build the frozen ensemble instead")

    def decode_half(positions, ball, teams, pitch):
        predicted = ensemble_trial.forecasts(reference, checkpoints, variants, positions, ball, teams, pitch)
        return motion_trial.decode(predicted, reference[2]["recipe"], recipe)

    write_csv(output, tactical_halves(decode_half, centered=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("tune", "confirm", "build"):
        command = commands.add_parser(action)
        command.add_argument("--fallback", required=True, type=Path)
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--ensemble", required=True, type=Path)
        command.add_argument("--trial", required=True, type=Path)
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
    if args.action == "tune":
        tune(args.dfl, args.skill, args.fallback, args.model, args.ensemble, args.trial, raw=args.dfl_raw)
    elif args.action == "confirm":
        confirm(args.dfl, args.skill, args.fallback, args.model, args.ensemble, args.trial, raw=args.dfl_raw)
    else:
        build(args.fallback, args.model, args.ensemble, args.trial, args.out)
