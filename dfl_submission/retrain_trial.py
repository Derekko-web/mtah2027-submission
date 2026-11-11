"""Compare the shortcut-free retrained models with the current models (DECISIONS.md D2).

Both are decoded with the current decoder recipe; only the candidate's stoppage cutoff is
re-selected on the external development matches. Guard views test a carried dead ball and
cross-team identity swaps. The holdout is scored once.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from dfl_submission import augment, context_trial, decoder_trial, ensemble_trial, long_context, motion_trial, temporal_cpu
from dfl_submission.build import METADATA_PITCH, tactical_halves, write_csv
from dfl_submission.conventions import collapse_brief_changes
from dfl_submission.temporal_trial import Match
from dfl_submission.train import write_json
from dfl_transfer_data import digest
from evaluation import possession_metrics as pm


STANDARD = ("clean", "masked", "observed")
GUARDS = ("dead_carried", "team_swap")
CUTOFFS = (.45, .49, .53, .57, .61)
CARRIED_GAIN, STANDARD_LOSS, SWAP_LOSS = .03, .01, .01
FOLDERS = ("long_models", "temporal_models", "seed_models")


def current(fallback, models, ensemble, decoder):
    reference, checkpoints, frozen, variants, recipe = decoder_trial.frozen_trial(fallback, models, ensemble, decoder)
    return reference, checkpoints, variants, recipe, frozen


def candidate(folder, current_package):
    reference, checkpoints, variants, recipe, frozen = current_package
    trees, _ = long_context.load(folder / "long_models")
    original = {name: temporal_cpu.load(folder / "temporal_models", name) for name in reference[1]}
    seeds = {name: temporal_cpu.load(folder / "seed_models", name) for name in checkpoints}
    return (trees, original, reference[2], reference[3]), seeds, variants


def mix(current_value, retrained_value, weight):
    """Average two forecast structures leaf by leaf; weight 1 keeps the retrained models only."""
    if weight == 1:
        return retrained_value
    if current_value is None:
        return None
    if isinstance(current_value, dict):
        return {key: mix(current_value[key], retrained_value[key], weight) for key in current_value}
    if isinstance(current_value, (tuple, list)):
        return tuple(mix(a, b, weight) for a, b in zip(current_value, retrained_value))
    return ((1 - weight) * current_value + weight * retrained_value).astype(np.float32)


def model_hashes(folder):
    return {name: json.loads((folder / name / "fit.json").read_text()) | {"fit_sha256": digest(folder / name / "fit.json")}
            for name in FOLDERS}


def forecast_pair(packages, weight, positions, ball, teams, pitch):
    predicted = {name: ensemble_trial.forecasts(reference, checkpoints, variants, positions, ball, teams, pitch)
                 for name, (reference, checkpoints, variants) in packages.items()}
    return predicted["current"], mix(predicted["current"], predicted["candidate"], weight)


def forecasts(halves, packages, weight):
    records = defaultdict(lambda: defaultdict(list))
    for half in halves:
        current_value, candidate_value = forecast_pair(packages, weight, half.positions, half.ball, half.teams, half.pitch)
        for name, predicted in (("current", current_value), ("candidate", candidate_value)):
            records[name][half.source, half.match, half.view].append((half.labels, predicted))
        print(f"Forecast {half.key}", flush=True)
    result = {}
    for name, matches in records.items():
        result[name] = {}
        for key, halves_of_match in matches.items():
            truth = np.r_[halves_of_match[0][0], pm.X, halves_of_match[1][0]]
            result[name][key] = Match(truth, pm.valid_runs(truth), [part[1] for part in halves_of_match])
    return result


def score(matches, previous, recipe, seconds):
    records = {}
    for key, match in matches.items():
        truth = collapse_brief_changes(match.truth, seconds) if seconds is not None else match.truth
        prediction = np.r_[motion_trial.decode(match.halves[0], previous, recipe), pm.X,
                           motion_trial.decode(match.halves[1], previous, recipe)]
        spells = pm.spell_scores(truth, prediction, pm.valid_runs(truth))
        confusion = pm.scored_confusion(pm.confusion(truth, prediction))
        records[key] = {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion), "spells": spells["n_pred"]}
    return records


def summarise(records):
    standard = {key: value for key, value in records.items() if key[2] in STANDARD}
    summary = {"standard": {metric: context_trial.aggregate(standard, metric)["combined"] for metric in ("spell_f1", "macro_f1")}}
    for view in GUARDS:
        values = [value["spell_f1"] for key, value in records.items() if key[2] == view]
        summary[view] = float(np.mean(values))
    summary["matches"] = {"/".join(key): value for key, value in records.items()}
    return summary


def evaluate(matches, previous, recipe):
    return {"recipe": recipe, "conventions": {name: summarise(score(matches, previous, recipe, seconds))
                                              for name, seconds in decoder_trial.CONVENTIONS.items()}}


def primary(trial, part="standard", metric="spell_f1"):
    value = trial["conventions"][decoder_trial.PRIMARY][part]
    return value[metric] if isinstance(value, dict) else value


def gates(base, challenger):
    changes = {"dead_carried": primary(challenger, "dead_carried") - primary(base, "dead_carried"),
               "standard": primary(challenger) - primary(base),
               "team_swap": primary(challenger, "team_swap") - primary(base, "team_swap")}
    passed = (changes["dead_carried"] >= CARRIED_GAIN and changes["standard"] >= -STANDARD_LOSS
              and changes["team_swap"] >= -SWAP_LOSS)
    return changes, passed


def packages(fallback, models, ensemble, decoder, folder):
    now = current(fallback, models, ensemble, decoder)
    return now, {"current": now[:3], "candidate": candidate(folder, now)}


def tune(dfl, skill, fallback, models, ensemble, decoder, folder, output, *, raw=False, weight=1.0, protocol="DECISIONS.md D2"):
    output.mkdir(parents=True, exist_ok=False)
    now, models_by_name = packages(fallback, models, ensemble, decoder, folder)
    previous, recipe = now[0][2]["recipe"], now[3]
    matches = forecasts(augment.guard_halves(dfl, skill, "development", raw=raw), models_by_name, weight)
    base = evaluate(matches["current"], previous, recipe)
    trials = [evaluate(matches["candidate"], previous, dict(recipe, cutoff=cutoff)) for cutoff in CUTOFFS]
    selected = max(trials, key=lambda trial: (primary(trial), primary(trial, metric="macro_f1")))
    changes, passed = gates(base, selected)
    write_json(output / "development.json", {"protocol": protocol, "weight": weight, "reference": base, "trials": trials,
                                             "selected": selected, "changes": changes, "passed": passed})
    write_json(output / "recipe.json", {"decoder": {"ensemble": now[4], "recipe": recipe}, "models": model_hashes(folder),
                                       "recipe": selected["recipe"], "pitch": list(METADATA_PITCH), "weight": weight,
                                       "protocol": protocol})
    print(f"Development cutoff {selected['recipe']['cutoff']}: " + ", ".join(f"{k} {v:+.4f}" for k, v in changes.items())
          + f" -> {'passed' if passed else 'failed'}", flush=True)


def frozen_candidate(fallback, models, ensemble, decoder, folder, trial):
    frozen = json.loads((trial / "recipe.json").read_text())
    now, models_by_name = packages(fallback, models, ensemble, decoder, folder)
    if frozen["decoder"] != {"ensemble": now[4], "recipe": now[3]}:
        raise ValueError("The current decoder recipe differs from the one the retrain was compared against")
    if model_hashes(folder) != frozen["models"]:
        raise ValueError("Retrained model identity differs from the frozen recipe")
    return now, models_by_name, frozen


def confirm(dfl, skill, fallback, models, ensemble, decoder, folder, trial, *, raw=False):
    development = json.loads((trial / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("The retrained models did not pass the development gates")
    if (trial / "external_validation.json").exists():
        raise ValueError("Confirmation already ran; the holdout is scored once")
    now, models_by_name, frozen = frozen_candidate(fallback, models, ensemble, decoder, folder, trial)
    previous = now[0][2]["recipe"]
    matches = forecasts(augment.guard_halves(dfl, skill, "holdout", raw=raw), models_by_name, frozen.get("weight", 1.0))
    base = evaluate(matches["current"], previous, now[3])
    challenger = evaluate(matches["candidate"], previous, frozen["recipe"])
    changes, passed = gates(base, challenger)
    write_json(trial / "external_validation.json", {"protocol": frozen.get("protocol", "DECISIONS.md D2"), "reference": base,
                                                    "selected": challenger, "changes": changes, "passed": passed})
    print("Holdout: " + ", ".join(f"{k} {v:+.4f}" for k, v in changes.items())
          + f" -> {'passed' if passed else 'failed'}", flush=True)


def build(fallback, models, ensemble, decoder, folder, trial, output):
    report = json.loads((trial / "external_validation.json").read_text())
    if report["passed"] is not True:
        raise ValueError("The retrained models did not pass confirmation; build the decoder recipe instead")
    now, models_by_name, frozen = frozen_candidate(fallback, models, ensemble, decoder, folder, trial)
    previous = models_by_name["current"][0][2]["recipe"]

    def decode_half(positions, ball, teams, pitch):
        _, predicted = forecast_pair(models_by_name, frozen.get("weight", 1.0), positions, ball, teams, pitch)
        return motion_trial.decode(predicted, previous, frozen["recipe"])

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen["pitch"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("tune", "confirm", "build"):
        command = commands.add_parser(action)
        command.add_argument("--fallback", required=True, type=Path)
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--ensemble", required=True, type=Path)
        command.add_argument("--decoder", required=True, type=Path)
        command.add_argument("--retrained", required=True, type=Path)
        command.add_argument("--trial", required=True, type=Path)
        if action == "tune":
            command.add_argument("--weight", type=float, default=1.0, help="Share of the retrained models in the blend")
            command.add_argument("--protocol", default="DECISIONS.md D2")
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
    shared = (args.fallback, args.model, args.ensemble, args.decoder, args.retrained)
    if args.action == "tune":
        tune(args.dfl, args.skill, *shared, args.trial, raw=args.dfl_raw, weight=args.weight, protocol=args.protocol)
    elif args.action == "confirm":
        confirm(args.dfl, args.skill, *shared, args.trial, raw=args.dfl_raw)
    else:
        build(*shared, args.trial, args.out)
