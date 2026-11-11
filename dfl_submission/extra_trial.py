"""D14: SkillCorner 2019–20 ownership labels as a third training source (DECISIONS.md D14).

`prepare` turns the five 2019–20 development matches into prepared halves with ownership-only
targets and combines them with each set's original prepared halves. The ownership trees and the
networks are then refit with `aligned_trial trees` and `aligned_trial networks` on the combined
folders. `tune` applies the no-loss development gates against D3, `confirm` scores the four
never-scored 2019–20 matches and PFF confirmation set C once, and `build` writes the competition
file with confirmed D14 models.
"""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np

from dfl_submission import aligned_trial, diagnosis
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "extra_recipe"
MODELS = PACKAGE / "extra_models"
PROTOCOL = "DECISIONS.md D14"
SOURCE = "skillcorner2020"
KNOWN = 4
LOSS, GAIN = .01, .01
PRIMARY = "aba3"


def ownership_targets(labels):
    """Network targets with the stoppage and stoppage-boundary losses switched off (no dead-ball label)."""
    from dfl_submission import temporal_network
    result = temporal_network.targets(labels)
    result[:, KNOWN] = 0
    return result


def prepare_extra(raw, output):
    from dfl_submission import long_context, skillcorner_2020
    output.mkdir(parents=True, exist_ok=False)
    records, columns = [], None
    for match in skillcorner_2020.DEVELOPMENT:
        tracking = skillcorner_2020.read_tracking(raw, match)
        owners = skillcorner_2020.read_ownership(raw, match)
        for index, ((positions, ball, teams, pitch), labels) in enumerate(zip(tracking, owners)):
            if len(labels) != len(ball):
                raise ValueError(f"Labels and tracking differ in length for {match} half {index + 1}")
            frames, table, _ = long_context.features(positions, ball, teams, pitch)
            if columns is None:
                columns = table.columns.tolist()
            elif columns != table.columns.tolist():
                raise ValueError("External feature schema differs")
            name = f"{SOURCE}_{match}_observed_{index}.npz"
            np.savez_compressed(output / name, values=table.to_numpy(dtype=np.float32),
                                targets=ownership_targets(labels[frames]))
            records.append({"file": name, "source": SOURCE, "match": str(match), "sha256": digest(output / name)})
            print(f"Prepared SkillCorner 2020 {match} half {index + 1}", flush=True)
    aligned_trial.write_json(output / "manifest.json", {"columns": columns, "records": records,
                                                        "skillcorner_revision": skillcorner_2020.REVISION})


def combine(original, extra, output):
    """One prepared folder with a set's original halves followed by the 2019–20 halves."""
    first, second = (json.loads((folder / "manifest.json").read_text()) for folder in (original, extra))
    if first["columns"] != second["columns"]:
        raise ValueError("The 2019–20 halves have a different feature schema")
    output.mkdir(parents=True, exist_ok=False)
    for folder in (original, extra):
        for record, path in aligned_trial.verified_records(folder):
            copyfile(path, output / record["file"])
    aligned_trial.write_json(output / "manifest.json", dict(first, records=first["records"] + second["records"],
                                                            source_manifest=digest(original / "manifest.json"),
                                                            extra_manifest=digest(extra / "manifest.json")))
    print(f"Combined {len(first['records'])} + {len(second['records'])} prepared halves into {output}", flush=True)


def development_passed(change):
    return (change["primary"] >= -LOSS and all(value >= -LOSS for value in change["guards"].values())
            and change["pff"] >= -LOSS)


def tune(dfl, skill, parsed, *, raw, d3_cache, cache=None, trial=RECIPE):
    require_external_only()
    from dfl_submission import competition_adaptation, refine_trial, vendor_trial
    output = trial / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D14 development is scored once")
    d3_model = competition_adaptation.reference()
    recipe = d3_model[2]["recipe"]
    games = vendor_trial.development()
    reference_halves = diagnosis.development_halves(dfl, skill, parsed, d3_model, raw=raw, cache=d3_cache)
    checks = diagnosis.reproduces(aligned_trial.decoded(reference_halves, recipe), recipe)
    if not all(checks.values()):
        raise ValueError(f"D14 pipeline check failed {checks}; no record was written")
    base = aligned_trial.evaluate(reference_halves, recipe)
    halves = aligned_trial.aligned_halves(dfl, skill, parsed, games, "development", aligned_trial.teacher(MODELS),
                                          raw=raw, cache=cache, models=MODELS)
    candidate = aligned_trial.evaluate(halves, recipe)
    change = refine_trial.changes(base, candidate)
    passed = development_passed(change)
    identity = aligned_trial.aligned_identity(MODELS)
    aligned_trial.write_new(output, {"protocol": PROTOCOL, "reference": base, "candidate": candidate,
                                     "reproduction": checks, "changes": change, "passed": passed, "models": identity,
                                     "teacher_sha256": diagnosis.teacher_identity(),
                                     "inputs": vendor_trial.inputs(parsed, games)})
    print(f"D14 development: N = 3 standard {change['primary']:+.5f}, "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f", PFF {change['pff']:+.4f} -> {'passed' if passed else 'failed'}", flush=True)
    return passed


def legacy_scores(raw, matches, models):
    """Mean scores of each model over the 2019–20 matches, as D6 scores them."""
    from dfl_submission import competition_adaptation, decoder_trial, skillcorner_2020
    recipe = models["d3"][2]["recipe"]
    per_match = {name: {} for name in models}
    for match in matches:
        tracking = skillcorner_2020.read_tracking(raw, match)
        labels = skillcorner_2020.read_ownership(raw, match)
        predictions = {name: [diagnosis.decode(diagnosis.half_signals(model, *half)[0], recipe) for half in tracking]
                       for name, model in models.items()}
        for name, result in competition_adaptation.score_match(labels, predictions).items():
            per_match[name][str(match)] = result
        print(f"Scored SkillCorner 2020 match {match}", flush=True)
    return {name: {"matches": scores, "means": {
        convention: {metric: float(np.mean([match[convention][metric] for match in scores.values()]))
                     for metric in ("spell_f1", "macro_f1")} for convention in decoder_trial.CONVENTIONS}}
            for name, scores in per_match.items()}


def confirm(raw, parsed, *, trial=RECIPE):
    require_external_only()
    from dfl_submission import competition_adaptation, skillcorner_2020, vendor_trial
    development = json.loads((trial / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("D14 failed its development gates; D3 stays and nothing is confirmed")
    output = trial / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D14 confirmation is scored once")
    if development["models"] != aligned_trial.aligned_identity(MODELS):
        raise ValueError("The D14 models differ from the ones development scored")
    models = {"d3": competition_adaptation.reference(), "candidate": aligned_trial.teacher(MODELS)}
    legacy = legacy_scores(raw, skillcorner_2020.HOLDOUT, models)
    games = vendor_trial.split_games("confirm_c")
    recipe = models["d3"][2]["recipe"]
    vendor = {}
    for name, model in models.items():
        halves = list(diagnosis.pff_halves(parsed, games, model))
        matches = diagnosis.pff_record(diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe)))
        vendor[name] = {"means": vendor_trial.means(matches), "matches": matches}
    change = {"legacy_primary": legacy["candidate"]["means"][PRIMARY]["spell_f1"] - legacy["d3"]["means"][PRIMARY]["spell_f1"],
              "pff": vendor["candidate"]["means"][PRIMARY]["spell_f1"] - vendor["d3"]["means"][PRIMARY]["spell_f1"]}
    passed = change["legacy_primary"] >= GAIN and change["pff"] >= -LOSS
    aligned_trial.write_new(output, {"protocol": PROTOCOL, "legacy": legacy, "pff": vendor, "changes": change,
                                     "passed": passed, "models": development["models"],
                                     "inputs": vendor_trial.inputs(parsed, games)})
    print(f"D14 confirmation: 2019–20 N = 3 {change['legacy_primary']:+.5f}, PFF set C {change['pff']:+.4f} "
          f"-> {'passed' if passed else 'failed'}", flush=True)
    return passed


def build(output, trial=RECIPE):
    from dfl_submission import motion_trial, retrain_trial
    from dfl_submission.build import tactical_halves, write_csv
    report = json.loads((trial / "confirmation.json").read_text())
    if report["passed"] is not True or report["models"] != aligned_trial.aligned_identity(MODELS):
        raise ValueError("The D14 models did not pass confirmation or differ from the confirmed ones")
    blend = json.loads((PACKAGE / "blend_recipe" / "recipe.json").read_text())
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    packages = {"current": retrain_trial.candidate(MODELS / "current", now),
                "candidate": retrain_trial.candidate(MODELS / "augmented", now)}
    previous = now[0][2]["recipe"]

    def decode_half(positions, ball, teams, pitch):
        _, mixed = retrain_trial.forecast_pair(packages, blend["weight"], positions, ball, teams, pitch)
        return diagnosis.decode(motion_trial.blend_signals(mixed, previous, blend["recipe"]), blend["recipe"])

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=blend["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--legacy-raw", type=Path, required=True)
    preparation.add_argument("--extra", type=Path, required=True, help="folder for the 2019–20 prepared halves")
    preparation.add_argument("--original", type=Path, nargs=2, required=True, metavar=("CURRENT", "AUGMENTED"))
    preparation.add_argument("--out", type=Path, required=True, help="parent folder for the two combined folders")
    tuning = commands.add_parser("tune")
    for flag in ("--dfl", "--skill", "--parsed", "--d3-cache"):
        tuning.add_argument(flag, type=Path, required=True)
    tuning.add_argument("--dfl-raw", action="store_true")
    tuning.add_argument("--cache", type=Path)
    confirmation = commands.add_parser("confirm")
    confirmation.add_argument("--legacy-raw", type=Path, required=True)
    confirmation.add_argument("--parsed", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "prepare":
        prepare_extra(args.legacy_raw, args.extra)
        for name, original in zip(aligned_trial.SETS, args.original):
            combine(original, args.extra, args.out / name)
    elif args.command == "tune":
        tune(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d3_cache=args.d3_cache, cache=args.cache)
    else:
        confirm(args.legacy_raw, args.parsed)


if __name__ == "__main__":
    main()
