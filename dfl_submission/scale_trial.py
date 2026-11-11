"""D15: network width and training length (DECISIONS.md D15).

`networks` trains and exports the two temporal networks of one model set at one width and length.
`tune` scores the network-setting x ownership-cost grid against D3 on the development matches,
after checking that D3 reproduces its development record and the D7 measurement. `confirm` scores
the selected setting once on the DFL/SkillCorner holdout and PFF confirmation set A, and `build`
writes the competition file with confirmed networks. The XGBoost trees are D3's throughout.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from dfl_submission import diagnosis
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from dfl_submission.temporal_cpu import STEPS


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "scale_recipe"
MODELS = PACKAGE / "scale_models"
PROTOCOL = "DECISIONS.md D15"
TREES = {"current": PACKAGE / "long_models", "augmented": PACKAGE / "retrain_models" / "long_models"}
PUBLISHED = {"current": PACKAGE, "augmented": PACKAGE / "retrain_models"}
ROLES = {"current": "current", "candidate": "augmented"}
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
SETTINGS = ((48, 1800), (48, 5400), (96, 1800), (96, 5400))
REFERENCE = SETTINGS[0]
OWNER_COSTS = (4, 6, 8)
GAIN, EARLY = .01, .003
INFERENCE_NAME = f"temporal_{STEPS[-1]}.npz"


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D15 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def name(setting):
    return f"w{setting[0]}_s{setting[1]}"


def network_folders(setting, models=MODELS):
    """Per model set, D3's published networks for the reference setting, otherwise the retrained ones."""
    return dict(PUBLISHED) if tuple(setting) == REFERENCE else {key: models / name(setting) / key for key in TREES}


# -- training --------------------------------------------------------------------------------------

def train_networks(source, model_set, setting, scratch, *, models=MODELS, device="mps"):
    """Train and export one set's two networks at `setting`, with D3's seeds, boundary weights and inputs."""
    setting = tuple(setting)
    if setting == REFERENCE or setting not in SETTINGS:
        raise ValueError(f"{setting} is not a D15 candidate setting")
    expected = json.loads((PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
    if digest(source / "manifest.json") != expected:
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    width, steps = setting
    for folder, (seed, boost) in NETWORKS.items():
        raw = scratch / name(setting) / model_set / folder
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "train", "--source", str(source),
                        "--out", str(raw), "--device", device, "--seed", str(seed), "--boundary-boost", str(boost),
                        "--width", str(width), "--steps", str(steps)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                        "--out", str(models / name(setting) / model_set / folder)], check=True, cwd=ROOT,
                       env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- the candidate models --------------------------------------------------------------------------

def final_snapshot(folder):
    """The last snapshot of an exported network, the one D3 uses at its own length."""
    fit = json.loads((folder / "fit.json").read_text())
    filename = f"temporal_{fit.get('steps', STEPS[-1])}.npz"
    if filename not in fit["models"]:
        raise ValueError(f"{folder} has no final snapshot {filename}")
    return filename


def package(trees, networks, now):
    """One model set as `retrain_trial.candidate` builds it, from D3's trees and each network's last snapshot."""
    from dfl_submission import long_context, temporal_cpu
    reference, checkpoints, variants, _, _ = now
    if set(reference[1]) != {INFERENCE_NAME} or set(checkpoints) != {INFERENCE_NAME}:
        raise ValueError("D3 is expected to use only the last snapshot of each network")
    forest, _ = long_context.load(trees)
    original = {INFERENCE_NAME: temporal_cpu.load(networks / "temporal_models", final_snapshot(networks / "temporal_models"))}
    seeds = {INFERENCE_NAME: temporal_cpu.load(networks / "seed_models", final_snapshot(networks / "seed_models"))}
    return (forest, original, reference[2], reference[3]), seeds, variants


def teacher(setting, models=MODELS):
    """D3's reference structure and blend recipe with both sets' networks replaced by `setting`'s."""
    from dfl_submission import competition_adaptation, retrain_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    folders = network_folders(setting, models)
    packages = {role: package(TREES[key], folders[key], now) for role, key in ROLES.items()}
    return reference, packages, frozen


def identity(setting, models=MODELS):
    folders = network_folders(setting, models)
    return {"setting": list(setting),
            "networks": {key: {folder: digest(folders[key] / folder / "fit.json") for folder in NETWORKS} for key in TREES},
            "trees": {key: digest(TREES[key] / "fit.json") for key in TREES}}


def setting_halves(dfl, skill, parsed, games, split, setting, *, raw, cache, models=MODELS):
    from dfl_submission import vendor_trial
    key = {"scale_models": identity(setting, models), "teacher_sha256": diagnosis.teacher_identity(), "split": split,
           "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}
    model = teacher(setting, models)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw)
        yield from diagnosis.pff_halves(parsed, games, model)

    return diagnosis.collect(cache, key, produce)


# -- scoring ---------------------------------------------------------------------------------------

def label(setting):
    return f"width {setting['width']}, steps {setting['steps']}, owner_cost {setting['owner_cost']}"


def eligible(setting, change):
    """D3's own networks are never eligible: at other costs that is a decoder-only change (D1)."""
    from dfl_submission import refine_trial
    return (setting["width"], setting["steps"]) != REFERENCE and refine_trial.passed(change, gain=GAIN)


def summary_line(prefix, result, change, verdict):
    from dfl_submission import refine_trial
    return (f"{prefix}: N = 3 standard {refine_trial.standard(result, 'aba3'):.5f} ({change['primary']:+.5f}), "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items()) + f", PFF {change['pff']:+.4f} -> {verdict}")


def tune(dfl, skill, parsed, *, raw, d3_cache, cache_root=None, trial=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, competition_adaptation, refine_trial, vendor_trial
    output = trial / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D15 development is scored once")
    d3_model = competition_adaptation.reference()
    recipe = d3_model[2]["recipe"]
    games = vendor_trial.development()
    reference_halves = diagnosis.development_halves(dfl, skill, parsed, d3_model, raw=raw, cache=d3_cache)
    checks = diagnosis.reproduces(aligned_trial.decoded(reference_halves, recipe), recipe)
    if not all(checks.values()):
        raise ValueError(f"D15 pipeline check failed {checks}; no record was written")
    base = aligned_trial.evaluate(reference_halves, recipe)
    entries = []
    for setting in SETTINGS:
        if setting == REFERENCE:
            halves = reference_halves
        else:
            halves = setting_halves(dfl, skill, parsed, games, "development", setting, raw=raw, models=models,
                                    cache=None if cache_root is None else cache_root / name(setting))
        for cost in OWNER_COSTS:
            point = {"width": setting[0], "steps": setting[1], "owner_cost": cost}
            candidate = dict(recipe, owner_cost=cost)
            result = base if setting == REFERENCE and candidate == recipe else aligned_trial.evaluate(halves, candidate)
            change = refine_trial.changes(base, result)
            allowed = eligible(point, change)
            entries.append({"setting": point, "eligible": allowed, "changes": change, "trial": result})
            print(summary_line(f"D15 {label(point)}", result, change, "eligible" if allowed else "not eligible"), flush=True)
    selected = refine_trial.select(entries)
    new_at_d3_cost = [entry["changes"]["primary"] for entry in entries
                      if (entry["setting"]["width"], entry["setting"]["steps"]) != REFERENCE
                      and entry["setting"]["owner_cost"] == recipe["owner_cost"]]
    write_new(output, {"protocol": PROTOCOL, "reference": base, "reproduction": checks, "entries": entries,
                       "selected": selected, "teacher_sha256": diagnosis.teacher_identity(),
                       "models": {name(setting): identity(setting, models) for setting in SETTINGS},
                       "best_new_setting_change_at_d3_cost": max(new_at_d3_cost),
                       "continue_d15": max(new_at_d3_cost) >= EARLY, "inputs": vendor_trial.inputs(parsed, games)})
    if selected is not None:
        setting = (selected["width"], selected["steps"])
        write_new(trial / "recipe.json", {"protocol": PROTOCOL, "setting": selected,
                                          "recipe": dict(recipe, owner_cost=selected["owner_cost"]),
                                          "models": identity(setting, models), "teacher_sha256": diagnosis.teacher_identity(),
                                          "weight": d3_model[2]["weight"], "pitch": d3_model[2]["pitch"]})
    print(f"D15 development selected {label(selected) if selected else 'no setting; D3 stays'}", flush=True)
    return selected


def frozen_recipe(trial, models):
    frozen = json.loads((trial / "recipe.json").read_text())
    setting = (frozen["setting"]["width"], frozen["setting"]["steps"])
    if frozen["teacher_sha256"] != diagnosis.teacher_identity() or frozen["models"] != identity(setting, models):
        raise ValueError("The D3 recipe or the D15 networks differ from the ones D15 development used")
    return frozen, setting


def confirm(dfl, skill, parsed, *, raw, d3_cache=None, cache=None, trial=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, competition_adaptation, refine_trial, vendor_trial
    development = json.loads((trial / "development.json").read_text())
    if development["selected"] is None:
        raise ValueError("No setting was eligible on development; D3 stays and nothing is confirmed")
    output = trial / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D15 confirmation is scored once")
    frozen, setting = frozen_recipe(trial, models)
    d3_model = competition_adaptation.reference()
    recipe = d3_model[2]["recipe"]
    reference_halves, games = refine_trial.holdout_halves(dfl, skill, parsed, d3_model, raw=raw, cache=d3_cache)
    base = aligned_trial.evaluate(reference_halves, recipe)
    d3_holdout = json.loads((PACKAGE / "blend_recipe" / "external_validation.json").read_text())["selected"]
    if vendor_trial.canonical(base["external"]) != d3_holdout:
        raise ValueError("D3 does not reproduce its holdout record; no record was written")
    halves = setting_halves(dfl, skill, parsed, games, "holdout", setting, raw=raw, cache=cache, models=models)
    challenger = aligned_trial.evaluate(halves, frozen["recipe"])
    change = refine_trial.changes(base, challenger)
    result = refine_trial.passed(change, gain=GAIN)
    write_new(output, {"protocol": PROTOCOL, "setting": frozen["setting"], "recipe": frozen["recipe"],
                       "reproduction": {"d3_holdout": True}, "reference": base, "selected": challenger,
                       "changes": change, "passed": result, "models": frozen["models"],
                       "inputs": vendor_trial.inputs(parsed, games)})
    print(summary_line(f"D15 confirmation {label(frozen['setting'])} (PFF = set A)", challenger, change,
                       "passed" if result else "failed"), flush=True)
    return result


def build(output, trial=RECIPE, models=MODELS):
    """The competition file with the confirmed D15 networks and ownership cost."""
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((trial / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("The D15 networks did not pass confirmation; build D3 instead")
    frozen, setting = frozen_recipe(trial, models)
    reference, packages, d3 = teacher(setting, models)
    model = (reference, packages, dict(d3, recipe=frozen["recipe"]))

    def decode_half(positions, ball, teams, pitch):
        return diagnosis.half_signals(model, positions, ball, teams, pitch)[1]

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--set", choices=tuple(TREES), required=True)
    networks.add_argument("--width", type=int, required=True)
    networks.add_argument("--steps", type=int, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    for name_ in ("tune", "confirm"):
        command = commands.add_parser(name_)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--d3-cache", type=Path)
        command.add_argument("--cache", type=Path, help="tune: one cache folder per setting below it")
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "networks":
        train_networks(args.source, args.set, (args.width, args.steps), args.scratch)
    elif args.command == "tune":
        tune(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d3_cache=args.d3_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d3_cache=args.d3_cache, cache=args.cache)


if __name__ == "__main__":
    main()
