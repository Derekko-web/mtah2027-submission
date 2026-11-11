"""D16: last-touch and possession-chain features (DECISIONS.md D16).

`prepare` re-extracts every external training half of one model set, checks that the 162 existing
columns equal the stored prepared sequences exactly, and appends the touch columns. `trees` refits
both XGBoost heads of one set from a prepared folder; with `--check` on an original folder it must
reproduce the published trees byte for byte. `gate` scores the tree-only refit (new trees, standing
networks). `networks` trains the set's two networks on the extended inputs. `tune` scores the full
refit's grid against D3 on the development matches, `confirm` scores the selected setting once on
the holdout and PFF confirmation set B, and `build` writes the competition file.
"""

import argparse
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from dfl_submission import diagnosis, scale_trial
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from dfl_submission.robust_features import invariant


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "touch_recipe"
MODELS = PACKAGE / "touch_models"
PROTOCOL = "DECISIONS.md D16"
SETS = ("current", "augmented")
OWNER_COSTS = (3, 4, 6, 8)
CUTOFFS = (.49, .53, .57)
GAIN, TREE_GATE = .01, .003
PRIMARY = "aba3"
DEAD, HOME, KNOWN, LIVE = 0, 1, 4, 5


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D16 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)


def standing(scale_recipe=scale_trial.RECIPE):
    """The network setting and decoder recipe standing after D15: D15's if it passed confirmation, else D3's (None)."""
    confirmation = scale_recipe / "confirmation.json"
    if confirmation.exists() and json.loads(confirmation.read_text())["passed"] is True:
        frozen = json.loads((scale_recipe / "recipe.json").read_text())
        return (frozen["setting"]["width"], frozen["setting"]["steps"]), frozen["recipe"]
    return scale_trial.REFERENCE, None


def published_manifest(model_set):
    return json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]


# -- training inputs -------------------------------------------------------------------------------

def training_halves(model_set, dfl, skill, *, raw):
    from dfl_submission import augment, external_benchmark
    if model_set == "current":
        return external_benchmark.halves(dfl, skill, "train", raw=raw)
    return augment.training_halves(dfl, skill, raw=raw)


def extend(source, halves, output):
    """Append the touch columns to a prepared folder, after checking its stored columns are re-extracted exactly."""
    from dfl_submission import aligned_trial, long_context, touch_features
    manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    records, columns = [], None
    for (record, path), half in zip(aligned_trial.verified_records(source), halves, strict=True):
        if record["file"] != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match the prepared record {record['file']}")
        frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        with np.load(path, allow_pickle=False) as item:
            values, targets = item["values"], item["targets"]
        if table.columns.tolist() != manifest["columns"] or not np.array_equal(
                table.to_numpy(dtype=np.float32), values, equal_nan=True):
            raise ValueError(f"Re-extracted features differ from the stored prepared sequence: {record['file']}")
        touch = touch_features.features(half.positions, half.ball, half.teams, frames)
        columns = manifest["columns"] + touch.columns.tolist()
        np.savez_compressed(output / record["file"], values=np.hstack([values, touch.to_numpy(dtype=np.float32)]),
                            targets=targets)
        records.append(dict(record, sha256=digest(output / record["file"])))
        print(f"Extended {record['file']}", flush=True)
    write_json(output / "manifest.json", dict(manifest, columns=columns, records=records, touch_features=PROTOCOL,
                                              base_manifest=digest(source / "manifest.json")))


def prepare(model_set, source, dfl, skill, output, *, raw):
    if digest(source / "manifest.json") != published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    extend(source, training_halves(model_set, dfl, skill, raw=raw), output)


# -- trees -----------------------------------------------------------------------------------------

def stoppage_rows(folder):
    from dfl_submission import aligned_trial
    columns = json.loads((folder / "manifest.json").read_text())["columns"]
    rows = []
    for record, path in aligned_trial.verified_records(folder):
        with np.load(path, allow_pickle=False) as item:
            values, target = item["values"], item["targets"]
        known = target[:, KNOWN].astype(bool)
        rows.append((invariant(pd.DataFrame(values, columns=columns)).loc[known].reset_index(drop=True),
                     target[known, DEAD].astype(bool), np.full(known.sum(), f"{record['source']}:{record['match']}")))
    return rows


def fit_trees(source, model_set, output):
    """Both long-context heads from a prepared folder, as `long_context.train` fits them."""
    from dfl_submission import aligned_trial, long_context
    reference = scale_trial.TREES[model_set]
    output.mkdir(parents=True, exist_ok=False)
    hashes, counts = {}, {}
    for filename, rows, ownership in (("ownership.ubj", aligned_trial.ownership_rows, True),
                                      ("dead.ubj", stoppage_rows, False)):
        model, counts[filename] = long_context.fit_head(rows(source), ownership=ownership)
        model.save_model(output / filename)
        hashes[filename] = digest(output / filename)
        print(f"Fitted {filename} on {counts[filename]:,} rows", flush=True)
    manifest = json.loads((source / "manifest.json").read_text())
    fitted = json.loads((reference / "fit.json").read_text())
    write_json(output / "fit.json", dict(fitted, models=hashes, rows=counts, prepared_manifest_sha256=digest(source / "manifest.json"),
                                         touch_features=manifest.get("touch_features")))
    return hashes


def check_trees(source, model_set, scratch):
    """D16 pipeline check: the original prepared folder refits both published trees byte for byte."""
    published = json.loads((scale_trial.TREES[model_set] / "fit.json").read_text())["models"]
    refit = fit_trees(source, model_set, scratch)
    result = {name: refit[name] == published[name] for name in published}
    print(f"Tree refit of the {model_set} set reproduces the published trees: {result}", flush=True)
    return all(result.values())


# -- networks --------------------------------------------------------------------------------------

def train_networks(source, model_set, scratch, *, setting=None, models=MODELS, device="mps"):
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("touch_features") != PROTOCOL or manifest.get("base_manifest") != published_manifest(model_set):
        raise ValueError(f"{source} is not the extended prepared folder of the {model_set} set")
    width, steps = setting if setting is not None else standing()[0]
    for folder, (seed, boost) in scale_trial.NETWORKS.items():
        raw = scratch / model_set / folder
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "train", "--source", str(source),
                        "--out", str(raw), "--device", device, "--seed", str(seed), "--boundary-boost", str(boost),
                        "--width", str(width), "--steps", str(steps)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                        "--out", str(models / "networks" / model_set / folder)], check=True, cwd=ROOT,
                       env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- forecasts with the extended inputs ------------------------------------------------------------

def table(positions, ball, teams, pitch):
    """The long-context tables with the touch columns appended to the ownership table."""
    from dfl_submission import long_context, touch_features
    frames, owner, _ = long_context.features(positions, ball, teams, pitch)
    owner = owner.join(touch_features.features(positions, ball, teams, frames))
    return frames, owner, invariant(owner)


def tree_view(frame, model):
    return frame[model.get_booster().feature_names]


def network_context(frames, owner, checkpoint, length):
    columns = list(checkpoint[1])
    return frames, owner[columns].to_numpy(dtype=np.float32), columns, length


def forecast_features(reference, checkpoints, frames, owner, dead, length):
    """`temporal_trial.forecast_features`, with each model given exactly the columns it was fitted on."""
    from dfl_submission import models, temporal_cpu, temporal_trial
    baseline = None
    if reference is not None:
        home_p = models.ownership_probability(reference[0], tree_view(owner, reference[0]))
        dead_p = reference[1].predict_proba(tree_view(dead, reference[1]))[:, 1]
        timeline = np.arange(length)
        baseline = tuple(np.interp(timeline, frames, p).astype(np.float32) for p in (home_p, dead_p))
    neural = {name: temporal_cpu.predict(model, network_context(frames, owner, model, length))
              for name, model in checkpoints.items()}
    if all(name in neural for name in temporal_trial.VARIANTS["late_mean"]):
        first, second = (neural[name] for name in temporal_trial.VARIANTS["late_mean"])
        neural["late_mean"] = tuple((a + b) / 2 for a, b in zip(first, second))
    return baseline, neural


def forecasts(reference, checkpoints, variants, positions, ball, teams, pitch):
    """`ensemble_trial.forecasts` on the extended tables; identical to it for models without touch columns."""
    from dfl_submission import ensemble_trial, temporal_cpu
    trees, original, frozen, _ = reference
    recipes = frozen["recipe"]
    phases = ensemble_trial.PHASES if "phase_reference" in variants or "seed_phase" in variants else ensemble_trial.PHASES[:1]
    late = ensemble_trial.LATE
    phase_reference, phase_seed = [], []
    start_reference, seeds = None, {}
    for offset in phases:
        frames, owner, dead = table(positions[offset:], ball[offset:], teams, pitch)
        length = len(ball) - offset
        if offset == 0 or "phase_reference" in variants:
            predicted = forecast_features(trees, original, frames, owner, dead, length)
            if offset == 0:
                start_reference = predicted
            if "phase_reference" in variants:
                phase_reference.append(tuple(np.pad(p, (offset, 0), mode="edge")
                                             for p in ensemble_trial.blend_reference(predicted, recipes)))
        if offset == 0:
            seeds = {name: temporal_cpu.predict(checkpoint, network_context(frames, owner, checkpoint, length))
                     for name, checkpoint in checkpoints.items()}
        if "seed_phase" in variants:
            shifted = seeds[late] if offset == 0 else temporal_cpu.predict(
                checkpoints[late], network_context(frames, owner, checkpoints[late], length))
            phase_seed.append(tuple(np.pad(p, (offset, 0), mode="edge") for p in shifted))
    candidates = {}
    for name in variants:
        if name == "phase_reference":
            candidates[name] = ensemble_trial.mean(phase_reference)
        elif name == "seed_phase":
            candidates[name] = ensemble_trial.mean(phase_seed)
        else:
            parts = [seeds[key] for key in ensemble_trial.VARIANTS[name]]
            candidates[name] = parts[0] if len(parts) == 1 else ensemble_trial.mean(parts)
    return start_reference, candidates


def half_signals(teacher, positions, ball, teams, pitch):
    """`diagnosis.half_signals` with the extended forecasts."""
    from dfl_submission import motion_trial, retrain_trial
    reference, packages, frozen = teacher
    previous = reference[2]["recipe"]
    outputs = tuple(forecasts(*packages[name], positions, ball, teams, pitch) for name in ("current", "candidate"))
    mixed = retrain_trial.mix(*outputs, frozen["weight"])
    labels = motion_trial.decode(mixed, previous, frozen["recipe"])
    signals = tuple(motion_trial.blend_signals(mixed, previous, frozen["recipe"]))
    if not np.array_equal(diagnosis.decode(signals, frozen["recipe"]), labels):
        raise ValueError("Decoding the cached inputs does not reproduce the labels")
    return signals, labels


# -- candidates ------------------------------------------------------------------------------------

def tree_folders(models=MODELS):
    return {key: models / "trees" / key for key in SETS}


def network_folders(stage, models=MODELS):
    """The standing networks for the tree-only stage, the retrained ones for the full refit."""
    if stage == "trees":
        return scale_trial.network_folders(standing()[0])
    return {key: models / "networks" / key for key in SETS}


def teacher(stage, models=MODELS):
    from dfl_submission import competition_adaptation, retrain_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    trees, networks = tree_folders(models), network_folders(stage, models)
    packages = {role: scale_trial.package(trees[key], networks[key], now) for role, key in scale_trial.ROLES.items()}
    return reference, packages, frozen


def identity(stage, models=MODELS):
    trees, networks = tree_folders(models), network_folders(stage, models)
    return {"stage": stage,
            "trees": {key: digest(trees[key] / "fit.json") for key in SETS},
            "networks": {key: {folder: digest(networks[key] / folder / "fit.json") for folder in scale_trial.NETWORKS}
                         for key in SETS}}


def stage_halves(dfl, skill, parsed, games, split, stage, *, raw, cache, models=MODELS):
    from dfl_submission import vendor_trial
    key = {"touch_models": identity(stage, models), "teacher_sha256": diagnosis.teacher_identity(), "split": split,
           "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}
    model = teacher(stage, models)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw, signals_of=half_signals)
        yield from diagnosis.pff_halves(parsed, games, model, signals_of=half_signals)

    return diagnosis.collect(cache, key, produce)


# -- scoring ---------------------------------------------------------------------------------------

def grid(recipe):
    return [dict(recipe, owner_cost=owner, cutoff=cutoff) for owner, cutoff in product(OWNER_COSTS, CUTOFFS)]


def skillcorner_part(trial):
    matches = trial["external"]["conventions"][PRIMARY]["matches"]
    return float(np.mean([value["spell_f1"] for key, value in matches.items()
                          if key.startswith("skillcorner/") and key.endswith("/observed")]))


def changes(base, trial):
    from dfl_submission import refine_trial
    return dict(refine_trial.changes(base, trial), skillcorner=skillcorner_part(trial) - skillcorner_part(base))


def eligible(change):
    """D16's gates: D15's, plus no drop on the SkillCorner part of the primary or on PFF development."""
    from dfl_submission import refine_trial
    return refine_trial.passed(change, gain=GAIN) and change["skillcorner"] >= 0 and change["pff"] >= 0


def describe(recipe):
    return f"owner_cost {recipe['owner_cost']}, cutoff {recipe['cutoff']}"


def line(prefix, result, change, verdict):
    from dfl_submission import refine_trial
    return (f"{prefix}: N = 3 standard {refine_trial.standard(result, PRIMARY):.5f} ({change['primary']:+.5f}), "
            f"SkillCorner {change['skillcorner']:+.4f}, "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items()) + f", PFF {change['pff']:+.4f} -> {verdict}")


def reference_halves(dfl, skill, parsed, *, raw, d3_cache, reference_cache=None):
    """The reference's development inputs and recipe: D3's, or D15's if D15 was adopted.

    D3 must reproduce its development record and the D7 measurement either way; an adopted D15
    model must also reproduce its own development record.
    """
    from dfl_submission import aligned_trial, competition_adaptation, vendor_trial
    d3_model = competition_adaptation.reference()
    d3_recipe = d3_model[2]["recipe"]
    d3_halves = diagnosis.development_halves(dfl, skill, parsed, d3_model, raw=raw, cache=d3_cache)
    checks = diagnosis.reproduces(aligned_trial.decoded(d3_halves, d3_recipe), d3_recipe)
    if not all(checks.values()):
        raise ValueError(f"D16 pipeline check failed {checks}; no record was written")
    setting, recipe = standing()
    if recipe is None:
        return d3_model, d3_recipe, d3_halves, checks
    halves = scale_trial.setting_halves(dfl, skill, parsed, vendor_trial.development(), "development", setting, raw=raw,
                                        cache=reference_cache)
    record = json.loads((scale_trial.RECIPE / "development.json").read_text())
    expected = next(entry["trial"] for entry in record["entries"] if entry["setting"] == record["selected"])
    checks["d15"] = vendor_trial.canonical(aligned_trial.evaluate(halves, recipe)) == expected
    if not checks["d15"]:
        raise ValueError(f"D16 pipeline check failed {checks}; no record was written")
    return d3_model, recipe, halves, checks


def segmentation(halves, recipe):
    """D11's ownership-change matching and recall by spell length, for the clean DFL view and SkillCorner."""
    report = diagnosis.measure_all(diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe)))
    return {name: {group: report["groups"][name][group] for group in ("dfl_clean", "skillcorner", "pff")}
            for name in report["groups"]}


def score_grid(halves, base, recipe, *, gate):
    from dfl_submission import aligned_trial
    entries = []
    for setting in grid(recipe):
        result = aligned_trial.evaluate(halves, setting)
        change = changes(base, result)
        allowed = None if gate else eligible(change)
        entries.append({"setting": setting, "eligible": allowed, "changes": change, "trial": result})
        verdict = "scored" if gate else "eligible" if allowed else "not eligible"
        print(line(f"D16 {'tree-only' if gate else 'full'} {describe(setting)}", result, change, verdict), flush=True)
    return entries


def gate(dfl, skill, parsed, *, raw, d3_cache, reference_cache=None, cache=None, trial=RECIPE, models=MODELS):
    """The tree-only refit: if its best primary does not gain TREE_GATE over D3, the networks are not retrained."""
    require_external_only()
    from dfl_submission import aligned_trial, vendor_trial
    output = trial / "tree_gate.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; the D16 tree gate is scored once")
    _, recipe, base_halves, checks = reference_halves(dfl, skill, parsed, raw=raw, d3_cache=d3_cache,
                                                     reference_cache=reference_cache)
    base = aligned_trial.evaluate(base_halves, recipe)
    games = vendor_trial.development()
    halves = stage_halves(dfl, skill, parsed, games, "development", "trees", raw=raw, cache=cache, models=models)
    entries = score_grid(halves, base, recipe, gate=True)
    best = max(entry["changes"]["primary"] for entry in entries)
    passed = best >= TREE_GATE
    write_new(output, {"protocol": PROTOCOL, "reference": base, "reproduction": checks, "entries": entries,
                       "best_primary_change": best, "passed": passed, "models": identity("trees", models),
                       "segmentation": {"reference": segmentation(base_halves, recipe),
                                        "tree_only_at_reference_recipe": segmentation(halves, recipe)},
                       "teacher_sha256": diagnosis.teacher_identity(), "inputs": vendor_trial.inputs(parsed, games)})
    print(f"D16 tree gate: best primary change {best:+.5f} -> {'passed; train the networks' if passed else 'failed; the reference stays'}",
          flush=True)
    return passed


def tune(dfl, skill, parsed, *, raw, d3_cache, reference_cache=None, cache=None, trial=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, refine_trial, vendor_trial
    if json.loads((trial / "tree_gate.json").read_text())["passed"] is not True:
        raise ValueError("The tree-only refit did not pass its gate; the networks are not retrained or scored")
    output = trial / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D16 development is scored once")
    d3_model, recipe, base_halves, checks = reference_halves(dfl, skill, parsed, raw=raw, d3_cache=d3_cache,
                                                            reference_cache=reference_cache)
    base = aligned_trial.evaluate(base_halves, recipe)
    games = vendor_trial.development()
    halves = stage_halves(dfl, skill, parsed, games, "development", "full", raw=raw, cache=cache, models=models)
    entries = score_grid(halves, base, recipe, gate=False)
    selected = refine_trial.select(entries)
    best = max(entries, key=lambda entry: entry["changes"]["primary"])["setting"]
    write_new(output, {"protocol": PROTOCOL, "reference": base, "reproduction": checks, "entries": entries,
                       "selected": selected, "models": identity("full", models),
                       "segmentation": {"reference": segmentation(base_halves, recipe),
                                        "full_at_best_primary": dict(segmentation(halves, best), setting=best)},
                       "teacher_sha256": diagnosis.teacher_identity(), "inputs": vendor_trial.inputs(parsed, games)})
    if selected is not None:
        write_new(trial / "recipe.json", {"protocol": PROTOCOL, "recipe": selected, "models": identity("full", models),
                                          "teacher_sha256": diagnosis.teacher_identity(), "weight": d3_model[2]["weight"],
                                          "pitch": d3_model[2]["pitch"]})
    print(f"D16 development selected {describe(selected) if selected else 'no setting; the reference stays'}", flush=True)
    return selected


def confirm(dfl, skill, parsed, *, raw, d3_cache=None, reference_cache=None, cache=None, trial=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, competition_adaptation, vendor_trial
    development = json.loads((trial / "development.json").read_text())
    if development["selected"] is None:
        raise ValueError("No setting was eligible on development; D3 stays and nothing is confirmed")
    output = trial / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D16 confirmation is scored once")
    frozen = json.loads((trial / "recipe.json").read_text())
    if frozen["teacher_sha256"] != diagnosis.teacher_identity() or frozen["models"] != identity("full", models):
        raise ValueError("The D3 recipe or the D16 models differ from the ones D16 development used")
    games = vendor_trial.split_games("confirm_b")
    setting, recipe = standing()
    if recipe is None:
        d3_model = competition_adaptation.reference()
        recipe = d3_model[2]["recipe"]
        key = {"teacher_sha256": diagnosis.teacher_identity(), "split": "holdout+confirm_b", "dfl_raw": raw,
               "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}

        def produce():
            yield from diagnosis.external_halves(dfl, skill, "holdout", d3_model, raw=raw)
            yield from diagnosis.pff_halves(parsed, games, d3_model)

        base = aligned_trial.evaluate(diagnosis.collect(d3_cache, key, produce), recipe)
        expected = json.loads((PACKAGE / "blend_recipe" / "external_validation.json").read_text())["selected"]
    else:
        halves = scale_trial.setting_halves(dfl, skill, parsed, games, "holdout", setting, raw=raw, cache=reference_cache)
        base = aligned_trial.evaluate(halves, recipe)
        expected = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]["external"]
    if vendor_trial.canonical(base["external"]) != expected:
        raise ValueError("The reference does not reproduce its holdout record; no record was written")
    halves = stage_halves(dfl, skill, parsed, games, "holdout", "full", raw=raw, cache=cache, models=models)
    challenger = aligned_trial.evaluate(halves, frozen["recipe"])
    change = changes(base, challenger)
    result = eligible(change)
    write_new(output, {"protocol": PROTOCOL, "recipe": frozen["recipe"], "reproduction": {"d3_holdout": True},
                       "reference": base, "selected": challenger, "changes": change, "passed": result,
                       "models": frozen["models"], "inputs": vendor_trial.inputs(parsed, games)})
    print(line(f"D16 confirmation {describe(frozen['recipe'])} (PFF = set B)", challenger, change,
               "passed" if result else "failed"), flush=True)
    return result


def build(output, trial=RECIPE, models=MODELS):
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((trial / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("The D16 models did not pass confirmation; build D3 instead")
    frozen = json.loads((trial / "recipe.json").read_text())
    if frozen["models"] != identity("full", models):
        raise ValueError("D16 model identity differs from the confirmed recipe")
    reference, packages, d3 = teacher("full", models)
    model = (reference, packages, dict(d3, recipe=frozen["recipe"]))

    def decode_half(positions, ball, teams, pitch):
        return half_signals(model, positions, ball, teams, pitch)[1]

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("prepare")
    step.add_argument("--set", choices=SETS, required=True)
    step.add_argument("--source", type=Path, required=True)
    step.add_argument("--out", type=Path, required=True)
    trees = commands.add_parser("trees")
    trees.add_argument("--set", choices=SETS, required=True)
    trees.add_argument("--source", type=Path, required=True)
    trees.add_argument("--out", type=Path, required=True)
    trees.add_argument("--check", action="store_true", help="refit from original inputs and compare with the published trees")
    networks = commands.add_parser("networks")
    networks.add_argument("--set", choices=SETS, required=True)
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    for name in ("prepare", "gate", "tune", "confirm"):
        command = step if name == "prepare" else commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        if name != "prepare":
            command.add_argument("--parsed", type=Path, required=True)
            command.add_argument("--d3-cache", type=Path)
            command.add_argument("--reference-cache", type=Path, help="the adopted D15 model's inputs, if D15 was adopted")
            command.add_argument("--cache", type=Path)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "prepare":
        prepare(args.set, args.source, args.dfl, args.skill, args.out, raw=args.dfl_raw)
    elif args.command == "trees":
        if args.check:
            if not check_trees(args.source, args.set, args.out):
                raise SystemExit("D16 tree pipeline check failed")
        else:
            fit_trees(args.source, args.set, args.out)
    elif args.command == "networks":
        train_networks(args.source, args.set, args.scratch)
    else:
        action = {"gate": gate, "tune": tune, "confirm": confirm}[args.command]
        action(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d3_cache=args.d3_cache,
               reference_cache=args.reference_cache, cache=args.cache)


if __name__ == "__main__":
    main()
