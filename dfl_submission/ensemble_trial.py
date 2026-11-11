"""Test an independent boundary-weighted seed and shifted sampling grids on external matches."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from dfl_submission import context_trial, external_benchmark, long_context, motion_trial, temporal_cpu, temporal_trial
from dfl_submission.build import tactical_halves, write_csv
from dfl_submission.train import write_json
from evaluation import possession_metrics as pm


EARLY = "temporal_1200.npz"
LATE = "temporal_1800.npz"
VARIANTS = {"seed_1200": (EARLY,), "seed_1800": (LATE,), "seed_late_mean": (EARLY, LATE),
            "phase_reference": (), "seed_phase": (LATE,)}
PHASES = (0, 2, 4)


def blend_reference(prediction, recipe):
    trees, temporal = prediction
    home, dead, de, oe = temporal[recipe["variant"]]
    return (context_trial.blend(trees[0], home, recipe["owner_mix"]),
            context_trial.blend(trees[1], dead, recipe["dead_mix"]), de, oe)


def mean(parts):
    return tuple(np.mean(np.stack(values), axis=0) for values in zip(*parts))


def forecasts(reference, checkpoints, variants, positions, ball, teams, pitch):
    trees, original, frozen, _ = reference
    recipes = frozen["recipe"]
    phases = PHASES if "phase_reference" in variants or "seed_phase" in variants else PHASES[:1]
    phase_reference, phase_seed = [], []
    start_reference, seeds = None, {}
    for offset in phases:
        frames, owner, dead = long_context.features(positions[offset:], ball[offset:], teams, pitch)
        length = len(ball) - offset
        context = frames, owner.to_numpy(dtype=np.float32), owner.columns.tolist(), length
        if offset == 0 or "phase_reference" in variants:
            predicted = temporal_trial.forecast_features(trees, original, frames, owner, dead, length)
            if offset == 0:
                start_reference = predicted
            if "phase_reference" in variants:
                phase_reference.append(tuple(np.pad(p, (offset, 0), mode="edge") for p in blend_reference(predicted, recipes)))
        if offset == 0:
            seeds = {name: temporal_cpu.predict(checkpoint, context) for name, checkpoint in checkpoints.items()}
        if "seed_phase" in variants:
            shifted = seeds[LATE] if offset == 0 else temporal_cpu.predict(checkpoints[LATE], context)
            phase_seed.append(tuple(np.pad(p, (offset, 0), mode="edge") for p in shifted))

    candidates = {}
    for name in variants:
        if name == "phase_reference":
            candidates[name] = mean(phase_reference)
        elif name == "seed_phase":
            candidates[name] = mean(phase_seed)
        else:
            parts = [seeds[key] for key in VARIANTS[name]]
            candidates[name] = parts[0] if len(parts) == 1 else mean(parts)
    return start_reference, candidates


def predictions(dfl, skill, split, reference, checkpoints, variants, *, raw=False):
    records = defaultdict(list)
    for half in external_benchmark.halves(dfl, skill, split, raw=raw):
        predicted = forecasts(reference, checkpoints, variants, half.positions, half.ball, half.teams, half.pitch)
        records[half.source, half.match, half.view].append((half.labels, predicted))
        print(f"Predicted {split} {half.key}", flush=True)
    matches = {}
    for key, halves in records.items():
        if len(halves) != 2:
            raise ValueError(f"External match requires two halves: {key}")
        truth = np.r_[halves[0][0], pm.X, halves[1][0]]
        matches[key] = temporal_trial.Match(truth, pm.valid_runs(truth), [part[1] for part in halves])
    return matches


def tune(dfl, skill, reference_folder, models, output, *, raw=False):
    output.mkdir(parents=True, exist_ok=False)
    reference = motion_trial.fallback(reference_folder)
    fit = json.loads((models / "fit.json").read_text())
    checkpoints = {name: temporal_cpu.load(models, name) for name in (EARLY, LATE)}
    matches = predictions(dfl, skill, "development", reference, checkpoints, VARIANTS, raw=raw)
    development = motion_trial.search(matches, reference[2]["recipe"], VARIANTS)
    recipe = development["selected"]["recipe"]
    files = VARIANTS[recipe["variant"]] if recipe.get("variant") is not None else ()
    write_json(output / "recipe.json", {"fallback": reference[2], "models": {name: fit["models"][name] for name in files},
                                       "normalization": fit["normalization"], "recipe": recipe})
    write_json(output / "development.json", development)
    print(f"Frozen ensemble development comparison {development['reference']['spell_f1']} -> {development['selected']}", flush=True)


def load(reference_folder, models, output):
    frozen = json.loads((output / "recipe.json").read_text())
    reference = motion_trial.fallback(reference_folder)
    fit = json.loads((models / "fit.json").read_text())
    if (reference[2] != frozen["fallback"] or fit["normalization"] != frozen["normalization"]
            or any(fit["models"][key] != value for key, value in frozen["models"].items())):
        raise ValueError("Frozen ensemble checkpoint identity differs")
    checkpoints = {name: temporal_cpu.load(models, name) for name in frozen["models"]}
    selected = frozen["recipe"].get("variant")
    return reference, checkpoints, frozen, (selected,) if selected is not None else ()


def confirm(dfl, skill, reference_folder, models, output, *, raw=False):
    if json.loads((output / "development.json").read_text())["passed"] is not True:
        raise ValueError("Ensemble did not pass external development")
    reference, checkpoints, frozen, variants = load(reference_folder, models, output)
    matches = predictions(dfl, skill, "holdout", reference, checkpoints, variants, raw=raw)
    motion_trial.report_comparison(matches, reference, frozen, output)


def build(reference_folder, models, trial, output):
    if json.loads((trial / "external_validation.json").read_text())["passed"] is not True:
        raise ValueError("Ensemble failed external confirmation")
    reference, checkpoints, frozen, variants = load(reference_folder, models, trial)

    def decode_half(positions, ball, teams, pitch):
        p = forecasts(reference, checkpoints, variants, positions, ball, teams, pitch)
        return motion_trial.decode(p, reference[2]["recipe"], frozen["recipe"])

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
