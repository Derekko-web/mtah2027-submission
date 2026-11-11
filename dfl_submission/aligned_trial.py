"""D13: both model sets refit on controlled-possession targets (DECISIONS.md D13).

`retarget` rewrites a prepared training-sequence folder with targets computed after collapsing
brief possession changes, after checking that the stored targets are reproduced from the reloaded
25 Hz labels. `trees` fits a long-context ownership tree from a prepared folder and copies the
unchanged stoppage tree; run on an original folder it must reproduce the published tree byte for
byte. `networks` trains and exports the two temporal networks of one set. `tune` scores the
ownership-cost and stoppage-cutoff grid against D3 on the development matches, `confirm` scores the
selected setting once on the holdout and PFF confirmation set B, and `build` writes the competition
file with confirmed aligned models.
"""

import argparse
from itertools import product
import json
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "aligned_recipe"
MODELS = PACKAGE / "aligned_models"
PROTOCOL = "DECISIONS.md D13"
CONVENTION = 3.0
SETS = {"current": PACKAGE, "augmented": PACKAGE / "retrain_models"}
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
FOLDERS = ("long_models", "temporal_models", "seed_models")
OWNER_COSTS = (2, 3, 4, 6, 8)
CUTOFFS = (.49, .53, .57)
GAIN = .01
LIVE, HOME = 5, 1


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D13 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)


# -- training inputs -------------------------------------------------------------------------------

def record_key(name):
    source, match, view, index = Path(name).stem.split("_")
    return source, match, view, int(index)


def training_labels(dfl, skill, *, raw):
    """25 Hz labels of every external training half, by (source, match, half index)."""
    from dfl_submission import external_benchmark, fusion_trial
    from dfl_submission.train import TRAIN
    from dfl_transfer_data import SECTIONS
    labels = {}
    for match in TRAIN:
        halves = fusion_trial.external_halves(dfl, match, raw=raw)
        for index, name in enumerate(SECTIONS):
            labels["dfl", match, index] = pm.label_codes(halves[name]["labels"])
        print(f"Loaded labels of DFL {match}", flush=True)
    for match, index, path in external_benchmark.skillcorner_paths(skill, "train"):
        with np.load(path, allow_pickle=False) as item:
            labels["skillcorner", str(match), index] = pm.label_codes(item["labels"])
    return labels


def verified_records(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    for record in manifest["records"]:
        path = folder / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Prepared sequence hash differs: {path}")
        yield record, path


def retarget(source, labels, output, convention=CONVENTION):
    """Copy prepared inputs and recompute their targets under `convention` (D13 pipeline check 1)."""
    from dfl_submission import long_context, temporal_network
    manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for record, path in verified_records(source):
        origin, match, _, index = record_key(record["file"])
        with np.load(path, allow_pickle=False) as item:
            values, stored = item["values"], item["targets"]
        full = labels[origin, match, index]
        frames = np.arange(0, len(full), long_context.STRIDE)
        if len(frames) != len(values) or not np.array_equal(temporal_network.targets(full[frames]), stored):
            raise ValueError(f"Stored targets are not reproduced from the reloaded labels: {record['file']}")
        aligned = temporal_network.targets(temporal_network.convention_labels(full, convention)[frames])
        np.savez_compressed(output / record["file"], values=values, targets=aligned)
        records.append(dict(record, sha256=digest(output / record["file"])))
    write_json(output / "manifest.json", dict(manifest, records=records, convention=convention,
                                              source_manifest=digest(source / "manifest.json")))
    print(f"Retargeted {len(records)} prepared halves into {output}", flush=True)


def ownership_rows(folder):
    import pandas as pd
    columns = json.loads((folder / "manifest.json").read_text())["columns"]
    rows = []
    for record, path in verified_records(folder):
        with np.load(path, allow_pickle=False) as item:
            values, target = item["values"], item["targets"]
        live = target[:, LIVE].astype(bool)
        rows.append((pd.DataFrame(values[live], columns=columns), target[live, HOME].astype(bool),
                     np.full(live.sum(), f"{record['source']}:{record['match']}")))
    return rows


def fit_trees(source, reference, output):
    """The ownership tree from prepared inputs and targets, with the reference stoppage tree copied."""
    from dfl_submission import long_context
    model, count = long_context.fit_head(ownership_rows(source), ownership=True)
    output.mkdir(parents=True, exist_ok=False)
    model.save_model(output / "ownership.ubj")
    copyfile(reference / "dead.ubj", output / "dead.ubj")
    fitted = json.loads((reference / "fit.json").read_text())
    if digest(output / "dead.ubj") != fitted["models"]["dead.ubj"]:
        raise ValueError("The reference stoppage tree differs from its fit record")
    manifest = json.loads((source / "manifest.json").read_text())
    write_json(output / "fit.json", dict(fitted, models={"ownership.ubj": digest(output / "ownership.ubj"),
                                                         "dead.ubj": fitted["models"]["dead.ubj"]},
                                         rows=dict(fitted["rows"], **{"ownership.ubj": count}),
                                         prepared_manifest_sha256=digest(source / "manifest.json"),
                                         convention=manifest.get("convention"),
                                         stoppage_tree_from=str(reference.relative_to(ROOT))))
    print(f"Fitted the ownership tree on {count:,} rows into {output}", flush=True)
    return digest(output / "ownership.ubj")


def check_trees(source, reference, scratch):
    """D13 pipeline check 2: the original prepared inputs reproduce the published ownership tree."""
    published = json.loads((reference / "fit.json").read_text())["models"]["ownership.ubj"]
    refit = fit_trees(source, reference, scratch)
    print(f"Ownership tree refit {'reproduces' if refit == published else 'DIFFERS FROM'} {reference}", flush=True)
    return refit == published


def train_networks(source, scratch, output, *, device="mps"):
    """Train and export the set's two temporal networks with the published seeds and boundary weights."""
    for folder, (seed, boost) in NETWORKS.items():
        raw = scratch / folder
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "train", "--source", str(source),
                        "--out", str(raw), "--device", device, "--seed", str(seed), "--boundary-boost", str(boost)],
                       check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                        "--out", str(output / folder)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- scoring ---------------------------------------------------------------------------------------

def model_hashes(folder):
    return {name: json.loads((folder / name / "fit.json").read_text()) | {"fit_sha256": digest(folder / name / "fit.json")}
            for name in FOLDERS}


def aligned_identity(models=MODELS):
    return {name: model_hashes(models / name) for name in SETS}


def teacher(models=MODELS):
    """D3's reference structure and blend recipe with both model sets replaced by the aligned ones."""
    from dfl_submission import competition_adaptation, retrain_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    packages = {"current": retrain_trial.candidate(models / "current", now),
                "candidate": retrain_trial.candidate(models / "augmented", now)}
    return reference, packages, frozen


def grid(recipe):
    return [dict(recipe, owner_cost=owner, cutoff=cutoff) for owner, cutoff in product(OWNER_COSTS, CUTOFFS)]


def decoded(halves, recipe):
    return diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe))


def evaluate(halves, recipe):
    from dfl_submission import vendor_trial
    external, vendor = diagnosis.split_sources(decoded(halves, recipe))
    result = {"recipe": recipe}
    if external:
        result["external"] = diagnosis.external_record(external, recipe)
    if vendor:
        matches = diagnosis.pff_record(vendor)
        result["pff"] = {"means": vendor_trial.means(matches), "matches": matches}
    return result


def aligned_halves(dfl, skill, parsed, games, split, model, *, raw, cache, models=MODELS):
    from dfl_submission import vendor_trial
    identity = {"aligned_models": aligned_identity(models), "teacher_sha256": diagnosis.teacher_identity(), "split": split,
                "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"),
                "pff": vendor_trial.inputs(parsed, games)}

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw)
        yield from diagnosis.pff_halves(parsed, games, model)

    return diagnosis.collect(cache, identity, produce)


def describe(recipe):
    return f"owner_cost {recipe['owner_cost']}, cutoff {recipe['cutoff']}"


def tune(dfl, skill, parsed, *, raw, d3_cache, cache=None, trial=RECIPE):
    require_external_only()
    from dfl_submission import refine_trial, vendor_trial
    output = trial / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D13 development is scored once")
    d3 = diagnosis.teacher_identity()
    from dfl_submission import competition_adaptation
    d3_model = competition_adaptation.reference()
    recipe = d3_model[2]["recipe"]
    games = vendor_trial.development()
    reference_halves = diagnosis.development_halves(dfl, skill, parsed, d3_model, raw=raw, cache=d3_cache)
    checks = diagnosis.reproduces(decoded(reference_halves, recipe), recipe)
    if not all(checks.values()):
        raise ValueError(f"D13 pipeline check failed {checks}; no record was written")
    base = evaluate(reference_halves, recipe)
    halves = aligned_halves(dfl, skill, parsed, games, "development", teacher(), raw=raw, cache=cache)
    entries = []
    for setting in grid(recipe):
        result = evaluate(halves, setting)
        change = refine_trial.changes(base, result)
        eligible = refine_trial.passed(change, gain=GAIN)
        entries.append({"setting": setting, "eligible": eligible, "changes": change, "trial": result})
        print(f"D13 {describe(setting)}: N = 3 standard {refine_trial.standard(result, 'aba3'):.5f} "
              f"({change['primary']:+.5f}), " + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items())
              + ", " + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
              + f", PFF {change['pff']:+.4f} -> {'eligible' if eligible else 'not eligible'}", flush=True)
    selected = refine_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "reference": base, "reproduction": checks, "entries": entries,
                       "selected": selected, "teacher_sha256": d3, "models": aligned_identity(),
                       "inputs": vendor_trial.inputs(parsed, games)})
    if selected is not None:
        write_new(trial / "recipe.json", {"protocol": PROTOCOL, "recipe": selected, "models": aligned_identity(),
                                          "teacher_sha256": d3, "weight": d3_model[2]["weight"],
                                          "pitch": d3_model[2]["pitch"]})
    print(f"D13 development selected {describe(selected) if selected else 'no setting; D3 stays'}", flush=True)
    return selected


def confirm(dfl, skill, parsed, *, raw, d3_cache=None, cache=None, trial=RECIPE):
    require_external_only()
    from dfl_submission import competition_adaptation, refine_trial, vendor_trial
    development = json.loads((trial / "development.json").read_text())
    if development["selected"] is None:
        raise ValueError("No setting was eligible on development; D3 stays and nothing is confirmed")
    output = trial / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D13 confirmation is scored once")
    frozen = json.loads((trial / "recipe.json").read_text())
    if frozen["teacher_sha256"] != diagnosis.teacher_identity() or frozen["models"] != aligned_identity():
        raise ValueError("The D3 recipe or the aligned models differ from the ones D13 development used")
    games = vendor_trial.split_games("confirm_b")
    d3_model = competition_adaptation.reference()
    recipe = d3_model[2]["recipe"]
    identity = {"teacher_sha256": diagnosis.teacher_identity(), "split": "holdout+confirm_b", "dfl_raw": raw,
                "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}

    def produce():
        yield from diagnosis.external_halves(dfl, skill, "holdout", d3_model, raw=raw)
        yield from diagnosis.pff_halves(parsed, games, d3_model)

    reference_halves = diagnosis.collect(d3_cache, identity, produce)
    base = evaluate(reference_halves, recipe)
    d3_holdout = json.loads((PACKAGE / "blend_recipe" / "external_validation.json").read_text())["selected"]
    if vendor_trial.canonical(base["external"]) != d3_holdout:
        raise ValueError("D3 does not reproduce its holdout record; no record was written")
    halves = aligned_halves(dfl, skill, parsed, games, "holdout", teacher(), raw=raw, cache=cache)
    challenger = evaluate(halves, frozen["recipe"])
    change = refine_trial.changes(base, challenger)
    result = refine_trial.passed(change, gain=GAIN)
    write_new(output, {"protocol": PROTOCOL, "recipe": frozen["recipe"], "reproduction": {"d3_holdout": True},
                       "reference": base, "selected": challenger, "changes": change, "passed": result,
                       "models": frozen["models"], "inputs": vendor_trial.inputs(parsed, games)})
    print(f"D13 confirmation {describe(frozen['recipe'])}: N = 3 {change['primary']:+.5f}, "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f", PFF set B {change['pff']:+.4f} -> {'passed' if result else 'failed'}", flush=True)
    return result


def build(output, trial=RECIPE):
    """The competition file with the confirmed aligned models and costs."""
    from dfl_submission import motion_trial, retrain_trial
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((trial / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("The aligned models did not pass confirmation; build D3 instead")
    frozen = json.loads((trial / "recipe.json").read_text())
    if frozen["models"] != aligned_identity():
        raise ValueError("Aligned model identity differs from the confirmed recipe")
    blend = PACKAGE / "blend_recipe"
    if digest(blend / "recipe.json") != frozen["teacher_sha256"]:
        raise ValueError("The D3 recipe differs from the one D13 was confirmed against")
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    packages = {"current": retrain_trial.candidate(MODELS / "current", now),
                "candidate": retrain_trial.candidate(MODELS / "augmented", now)}
    previous = now[0][2]["recipe"]

    def decode_half(positions, ball, teams, pitch):
        _, mixed = retrain_trial.forecast_pair(packages, frozen["weight"], positions, ball, teams, pitch)
        return diagnosis.decode(motion_trial.blend_signals(mixed, previous, frozen["recipe"]), frozen["recipe"])

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("retarget")
    step.add_argument("--source", type=Path, required=True)
    step.add_argument("--out", type=Path, required=True)
    trees = commands.add_parser("trees")
    trees.add_argument("--source", type=Path, required=True)
    trees.add_argument("--set", choices=tuple(SETS), required=True)
    trees.add_argument("--out", type=Path, required=True)
    trees.add_argument("--check", action="store_true", help="refit from original inputs and compare with the published tree")
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    networks.add_argument("--out", type=Path, required=True)
    for name in ("retarget", "tune", "confirm"):
        command = step if name == "retarget" else commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        if name != "retarget":
            command.add_argument("--parsed", type=Path, required=True)
            command.add_argument("--d3-cache", type=Path)
            command.add_argument("--cache", type=Path)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "retarget":
        retarget(args.source, training_labels(args.dfl, args.skill, raw=args.dfl_raw), args.out)
    elif args.command == "trees":
        reference = SETS[args.set] / "long_models"
        if args.check:
            if not check_trees(args.source, reference, args.out):
                raise SystemExit("D13 pipeline check 2 failed")
        else:
            fit_trees(args.source, reference, args.out)
    elif args.command == "networks":
        train_networks(args.source, args.scratch, args.out)
    else:
        action = tune if args.command == "tune" else confirm
        action(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d3_cache=args.d3_cache, cache=args.cache)


if __name__ == "__main__":
    main()
