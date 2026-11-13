"""D34: optimisation hyperparameters by cross-validation (DECISIONS.md D34).

`check` confirms that the D34 trainer at its defaults reproduced D20's fold-0 network bit for bit.
`stage-a` trains one network per fold for each A1 setting (and the A2 combination when the rule calls
for it), decodes each alone on its held-out training halves as D22 Stage A did, and selects a point.
`networks` retrains one model set's two networks at the selected hyperparameters, `develop` scores them
once on development against D15, `confirm` once on the reused holdout and PFF confirmation set B, and
`build` writes the competition file. The XGBoost trees are D3's and inference stays NumPy only.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from dfl_submission import diagnosis
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "hyper_recipe"
MODELS = PACKAGE / "hyper_models"
PROTOCOL = "DECISIONS.md D34"
WIDTH, STEPS = 96, 5400
DEFAULTS = {"lr_scale": 1., "batch": 16, "weight_decay": .01, "dropout": .08}
A1 = (("lr_scale", 2.), ("lr_scale", .5), ("batch", 32), ("batch", 64), ("weight_decay", .1), ("dropout", .2),
      ("dropout", 0.))
OWNER_COSTS = (8, 10, 12)
ANCHOR_SETTING = {"width": 96, "steps": 5400}
ANCHOR_BEST = .71409                 # D22 Stage A, width 96 and 5,400 steps at cost 10
STAGE_A_GAIN = .008
GAIN, LOSS = .01, .01
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
SETS = ("current", "augmented")
FLAGS = {"lr_scale": "--lr-scale", "batch": "--batch", "weight_decay": "--weight-decay", "dropout": "--dropout"}


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D34 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def name(setting):
    """A stable folder name: the changed hyperparameters in key order, or `default`."""
    parts = [f"{key}{setting[key]:g}" for key in DEFAULTS if setting.get(key, DEFAULTS[key]) != DEFAULTS[key]]
    return "_".join(parts) or "default"


def full(setting):
    return dict(DEFAULTS, **setting)


def a1_settings():
    return [{key: value} for key, value in A1]


def train_command(source, output, setting, *, seed, boost, device="mps"):
    command = [sys.executable, "-m", "dfl_submission.hyper_network", "--source", str(source), "--out", str(output),
               "--device", device, "--seed", str(seed), "--boundary-boost", str(boost), "--width", str(WIDTH),
               "--steps", str(STEPS)]
    for key, value in full(setting).items():
        if value != DEFAULTS[key]:
            command += [FLAGS[key], str(int(value) if key == "batch" else value)]
    return command


# -- the pipeline check ----------------------------------------------------------------------------

def anchor_run(fold, cv_root):
    from dfl_submission import zoo
    return cv_root / "tcn" / zoo.name(ANCHOR_SETTING) / f"fold{fold}"


def check(scratch, cv_root, output=RECIPE / "check.json"):
    require_external_only()
    reference = json.loads((anchor_run(0, cv_root) / "fit.json").read_text())
    trained = json.loads((scratch / "fit.json").read_text())
    for key in ("seed", "boundary_boost", "width", "steps", "train_manifest", "training_device", "torch"):
        if trained[key] != reference[key]:
            raise ValueError(f"The check run differs from D20's fold-0 run in {key}")
    if any(key in trained for key in DEFAULTS):
        raise ValueError("The check run must use the default hyperparameters")
    filename = f"temporal_{STEPS}.pt"
    expected, actual = reference["models"][filename], digest(scratch / filename)
    passed = expected == actual
    write_new(output, {"protocol": PROTOCOL, "expected": expected, "actual": actual, "passed": passed,
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D34 trainer check: {'bit for bit' if passed else 'DIFFERS'} ({actual[:12]} against {expected[:12]})",
          flush=True)
    return passed


# -- Stage A ---------------------------------------------------------------------------------------

def train_folds(setting, source, cv_root, scratch):
    """One network per fold at `setting`, each on the other three folds' prepared halves."""
    from dfl_submission import zoo
    if json.loads((RECIPE / "check.json").read_text())["passed"] is not True:
        raise ValueError("The D34 trainer check did not pass; nothing is trained")
    seed, boost = zoo.CV_NETWORK
    runs = []
    for fold in range(zoo.FOLDS):
        folder = zoo.fold_folder(source, fold, cv_root / "folds" / str(fold))
        run = scratch / name(setting) / f"fold{fold}"
        if not (run / "fit.json").exists():
            if run.exists():
                shutil.rmtree(run)
            subprocess.run(train_command(folder, run, setting, seed=seed, boost=boost), check=True, cwd=ROOT)
        runs.append(run)
    return runs


def score_setting(runs, source, labels, recipe):
    """Fold networks decoded alone on their held-out training halves, at every ownership cost."""
    from dfl_submission import triple_trial, zoo
    halves = []
    for fold, run in enumerate(runs):
        part, _ = zoo.fold_halves(run, source, fold, labels)
        halves += part
    return {cost: triple_trial.score_point(halves, dict(recipe, owner_cost=cost)) for cost in OWNER_COSTS}


def primary(trial, metric="spell_f1"):
    return trial["standard"]["aba3"][metric]


def combination(entries):
    """D34's A2 setting: each hyperparameter's best A1 level whose best primary beats the anchor's best."""
    best = {}
    for entry in entries:
        (key, value), = entry["setting"].items()
        score = max(primary(trial) for trial in entry["trials"].values())
        if score > ANCHOR_BEST and (key not in best or score > best[key][1]):
            best[key] = (value, score)
    return {key: value for key, (value, _) in best.items()} if len(best) >= 2 else None


def stage_a_select(entries):
    """Highest primary, then N = 3 macroF1, then grid order (settings in order, costs ascending)."""
    points = [(entry["setting"], cost, trial) for entry in entries for cost, trial in sorted(entry["trials"].items())]
    index, (setting, cost, trial) = max(enumerate(points), key=lambda item: (primary(item[1][2]),
                                                                             primary(item[1][2], "macro_f1"), -item[0]))
    return {"setting": setting, "owner_cost": cost, "primary": primary(trial),
            "gain_over_anchor_best": primary(trial) - ANCHOR_BEST}


def stage_a(dfl, skill, source, cv_root, scratch, *, raw, output=RECIPE / "stage_a.json"):
    require_external_only()
    from dfl_submission import aligned_trial, triple_trial, zoo
    if output.exists():
        raise FileExistsError(f"{output} exists; D34 Stage A is scored once")
    zoo.check_original_source(source)
    labels = aligned_trial.training_labels(dfl, skill, raw=raw)
    recipe = triple_trial.d15_recipe()
    anchor = score_setting([anchor_run(fold, cv_root) for fold in range(zoo.FOLDS)], source, labels, recipe)
    stored = json.loads((PACKAGE / "triple_recipe" / "stage_a.json").read_text())
    expected = {entry["point"]["owner_cost"]: entry["trial"]["standard"] for entry in stored["entries"]
                if (entry["point"]["width"], entry["point"]["steps"]) == (WIDTH, STEPS)}
    if any(json.loads(json.dumps(anchor[cost]["standard"])) != expected[cost] for cost in OWNER_COSTS):
        raise ValueError("D20's fold networks do not reproduce D22's Stage A anchor; nothing was written")
    if abs(max(primary(trial) for trial in anchor.values()) - ANCHOR_BEST) > 5e-6:
        raise ValueError("The anchor's best primary differs from the rule's 0.71409")
    print("D34 Stage A anchor reproduces D22: " + ", ".join(f"cost {c} {primary(t):.5f}" for c, t in anchor.items()),
          flush=True)
    entries = []
    for setting in a1_settings():
        runs = train_folds(setting, source, cv_root, scratch)
        trials = score_setting(runs, source, labels, recipe)
        entries.append({"stage": "A1", "setting": setting, "trials": trials,
                        "runs": {str(i): digest(run / "fit.json") for i, run in enumerate(runs)}})
        print(f"D34 Stage A {name(setting)}: " + ", ".join(f"cost {c} {primary(t):.5f}" for c, t in trials.items()),
              flush=True)
    combined = combination(entries)
    if combined is not None:
        runs = train_folds(combined, source, cv_root, scratch)
        trials = score_setting(runs, source, labels, recipe)
        entries.append({"stage": "A2", "setting": combined, "trials": trials,
                        "runs": {str(i): digest(run / "fit.json") for i, run in enumerate(runs)}})
        print(f"D34 Stage A A2 {name(combined)}: " + ", ".join(f"cost {c} {primary(t):.5f}" for c, t in trials.items()),
              flush=True)
    selected = stage_a_select(entries)
    proceeds = selected["gain_over_anchor_best"] >= STAGE_A_GAIN
    write_new(output, {"protocol": PROTOCOL, "stage": "A", "anchor": {str(c): t for c, t in anchor.items()},
                       "anchor_best": ANCHOR_BEST, "entries": [dict(e, trials={str(c): t for c, t in e["trials"].items()})
                                                              for e in entries],
                       "combination": combined, "selected": selected, "proceeds": proceeds, "recipe": recipe,
                       "source_manifest": digest(source / "manifest.json"),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D34 Stage A selected {name(selected['setting'])} at cost {selected['owner_cost']}: "
          f"{selected['primary']:.5f} ({selected['gain_over_anchor_best']:+.5f} over the anchor's best) -> "
          f"{'Stage B' if proceeds else 'stop; D15 stays'}", flush=True)
    return proceeds


def standing(recipe=RECIPE):
    record = json.loads((recipe / "stage_a.json").read_text())
    if not record["proceeds"]:
        raise ValueError("D34 Stage A did not clear its bar; D15 stays and nothing more is trained")
    return record["selected"]["setting"]


# -- Stage B ---------------------------------------------------------------------------------------

def network_folders(setting, models=MODELS):
    return {key: models / name(setting) / key for key in SETS}


def train_networks(source, model_set, scratch, *, models=MODELS):
    from dfl_submission import scale_trial
    setting = standing()
    expected = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
    if digest(source / "manifest.json") != expected:
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for folder, (seed, boost) in NETWORKS.items():
        output = network_folders(setting, models)[model_set] / folder
        if (output / "fit.json").exists():
            continue
        raw = scratch / name(setting) / model_set / folder
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            subprocess.run(train_command(source, raw, setting, seed=seed, boost=boost), check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
        fit = json.loads((output / "fit.json").read_text())
        if (fit["seed"], fit["boundary_boost"], fit["train_manifest"]) != (seed, boost, expected) or \
                {key: fit.get(key, DEFAULTS[key]) for key in DEFAULTS} != full(setting):
            raise ValueError(f"{output} was not trained as D34 prescribes")


def teacher(setting, owner_cost=None, models=MODELS):
    from dfl_submission import competition_adaptation, objective_trial, retrain_trial, scale_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    folders = network_folders(setting, models)
    packages = {role: scale_trial.package(scale_trial.TREES[key], folders[key], now)
                for role, key in scale_trial.ROLES.items()}
    if owner_cost is not None:
        _, d15 = objective_trial.d15()
        frozen = dict(frozen, recipe=dict(d15["recipe"], owner_cost=owner_cost))
    return reference, packages, frozen


def identity(setting, models=MODELS):
    from dfl_submission import scale_trial
    folders = network_folders(setting, models)
    return {"setting": setting,
            "networks": {key: {folder: digest(folders[key] / folder / "fit.json") for folder in NETWORKS} for key in SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in SETS}}


def candidate_halves(dfl, skill, parsed, games, split, setting, *, raw, cache, models=MODELS):
    from dfl_submission import vendor_trial
    key = {"hyper_models": identity(setting, models), "teacher_sha256": diagnosis.teacher_identity(), "split": split,
           "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"),
           "pff": vendor_trial.inputs(parsed, games)}
    model = teacher(setting, models=models)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw)
        yield from diagnosis.pff_halves(parsed, games, model)

    return diagnosis.collect(cache, key, produce)


def variant(setting):
    return f"D34 hyperparameters {name(setting)}"


def recipe_at(cost):
    from dfl_submission import objective_trial
    _, d15 = objective_trial.d15()
    return dict(d15["recipe"], owner_cost=cost)


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D34 development is scored once")
    setting = standing(recipe_folder)
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    folder = cache_root / name(setting)
    halves = candidate_halves(dfl, skill, parsed, games, "development", setting, raw=raw,
                              cache=folder / "development", models=models)
    model = teacher(setting, models=models)
    _, metrica = ms.reference_halves("D15", cache=folder / "metrica", model=model, signals_of=diagnosis.half_signals,
                                     variant=variant(setting))
    views = mv.view_halves(cache=folder / "d24", signals_of=lambda _, *tracking: diagnosis.half_signals(model, *tracking),
                           variant=variant(setting))
    entries = []
    for cost in OWNER_COSTS:
        candidate = recipe_at(cost)
        trial = aligned_trial.evaluate(halves, candidate)
        change = refine_trial.changes(base, trial)
        veto = ms.veto(objective_trial.decoded(metrica, candidate), reference="D15")
        view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
        d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
        d24["change"] = d24["candidate"] - d24["reference"]
        allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24["change"])
        entries.append({"point": {"setting": setting, "owner_cost": cost}, "eligible": allowed, "changes": change,
                        "trial": trial, "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                        "d24": dict(d24, views=view_scores)})
        print(objective_trial.summary_line(f"D34 development {name(setting)} cost {cost}", trial, change,
                                           f", Metrica {veto['changes']['primary']:+.4f} (veto "
                                           f"{'passed' if veto['changes']['passed'] else 'failed'}), D24 {d24['change']:+.4f}",
                                           "eligible" if allowed else "not eligible"), flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "setting": setting, "costs": list(OWNER_COSTS), "recipe": recipe,
                       "reference": base, "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected, "models": identity(setting, models),
                       "teacher_sha256": diagnosis.teacher_identity(),
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D34 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE, models=MODELS):
    development = json.loads((recipe_folder / "development.json").read_text())
    point = development["selected"]
    if point is None:
        raise ValueError("No D34 point was eligible on development; D15 stays and nothing is confirmed")
    if development["models"] != identity(point["setting"], models):
        raise ValueError("The D34 networks differ from the ones development scored")
    return development, point


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, objective_trial, refine_trial, scale_trial, vendor_trial
    _, point = frozen(recipe_folder, models)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D34 confirmation is scored once")
    setting_d15, frozen_d15 = objective_trial.d15()
    recipe = frozen_d15["recipe"]
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting_d15, raw=raw,
                                           cache=d15_holdout_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full_record = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full_record["external"]) != stored["external"]
            or vendor_trial.canonical(full_record["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    external = [half for half in reference if half.source != "pff"]
    set_b, games_b = objective_trial.set_b_halves(parsed, cache_root / "d15_set_b")
    base = aligned_trial.evaluate(external + set_b, recipe)
    candidate = candidate_halves(dfl, skill, parsed, games_b, "holdout", point["setting"], raw=raw,
                                 cache=cache_root / name(point["setting"]) / "holdout", models=models)
    trial = aligned_trial.evaluate(candidate, recipe_at(point["owner_cost"]))
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": recipe_at(point["owner_cost"]),
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": identity(point["setting"], models),
                       "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line(f"D34 confirmation {name(point['setting'])} cost {point['owner_cost']} "
                                       "(PFF = set B)", trial, change, "", "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the confirmed D34 networks and cost in D15's structure, then D27's rule."""
    from dfl_submission import kickoff, objective_trial
    from dfl_submission.build import write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D34 did not pass confirmation; build D27 instead (`kickoff build`)")
    _, point = frozen(recipe_folder, models)
    _, frozen_d15 = objective_trial.d15()
    labels = kickoff.model_labels(teacher(point["setting"], point["owner_cost"], models), frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--scratch", type=Path, required=True)
    checking.add_argument("--cv", type=Path, required=True, help="D20's cross-validation folder")
    first = commands.add_parser("stage-a")
    first.add_argument("--dfl", type=Path, required=True)
    first.add_argument("--dfl-raw", action="store_true")
    first.add_argument("--skill", type=Path, required=True)
    first.add_argument("--source", type=Path, required=True, help="the original set's prepared sequences")
    first.add_argument("--cv", type=Path, required=True)
    first.add_argument("--scratch", type=Path, required=True)
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--set", choices=SETS, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    for command_name in ("develop", "confirm"):
        command = commands.add_parser(command_name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        if command_name == "develop":
            command.add_argument("--d15-cache", type=Path, required=True)
        else:
            command.add_argument("--d15-holdout-cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "check":
        check(args.scratch, args.cv)
    elif args.command == "stage-a":
        stage_a(args.dfl, args.skill, args.source, args.cv, args.scratch, raw=args.dfl_raw)
    elif args.command == "networks":
        train_networks(args.source, args.set, args.scratch)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                cache_root=args.cache)


if __name__ == "__main__":
    main()
