"""D38: beyond D34's combination: longer, more regularised, larger (DECISIONS.md D38).

`stage-a` decodes D34's A2 fold networks (the anchor) and one fold network per B1 setting on their
held-out training halves, as D34 did. `networks` retrains one model set's two networks at the selected
setting, `develop` caches the candidate's member outputs (D35's method) and scores the tree-share x cost
grid once against D15, `confirm` scores the selected point once on the reused holdout and PFF
confirmation set B, and `build` writes the competition file. Inference stays NumPy only.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from dfl_submission import diagnosis, hyper_trial as ht, mix_trial as mt
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "fourth_recipe"
MODELS = PACKAGE / "fourth_models"
PROTOCOL = "DECISIONS.md D38"
BASE = {"lr_scale": 1., "batch": 16, "weight_decay": .01, "dropout": .08, "width": 96, "steps": 5400}
ANCHOR = dict(BASE, lr_scale=2., batch=32, dropout=.2)
B1 = ({"steps": 10800}, {"dropout": .3}, {"batch": 64}, {"lr_scale": 3.}, {"width": 128})
STAGE_A_COSTS = (10, 12, 14, 16)
ANCHOR_RECORD = {8: .71548, 10: .72034, 12: .72160}       # D34 Stage A, A2
STAGE_A_GAIN = .003
OWNER_MIXES = (.5, 1.)
OWNER_COSTS = (6, 8, 10, 12)
GAIN = .01
FLAGS = {"lr_scale": "--lr-scale", "batch": "--batch", "weight_decay": "--weight-decay", "dropout": "--dropout",
         "width": "--width", "steps": "--steps"}


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D38 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def settings():
    return [dict(ANCHOR, **change) for change in B1]


def name(setting):
    return (f"lr{setting['lr_scale']:g}_b{setting['batch']}_d{setting['dropout']:g}_wd{setting['weight_decay']:g}"
            f"_w{setting['width']}_s{setting['steps']}")


def train_command(source, output, setting, *, seed, boost, device="mps"):
    command = [sys.executable, "-m", "dfl_submission.hyper_network", "--source", str(source), "--out", str(output),
               "--device", device, "--seed", str(seed), "--boundary-boost", str(boost)]
    for key, flag in FLAGS.items():
        value = setting[key]
        command += [flag, str(int(value)) if key in ("batch", "width", "steps") else str(value)]
    return command


def anchor_runs():
    return [ROOT / "data" / "d34" / "cv" / ht.name({"lr_scale": 2., "batch": 32, "dropout": .2}) / f"fold{fold}"
            for fold in range(4)]


# -- Stage A ---------------------------------------------------------------------------------------

def train_folds(setting, source, cv_root, scratch):
    from dfl_submission import zoo
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


def score_runs(runs, source, labels, recipe, costs):
    from dfl_submission import triple_trial, zoo
    halves = []
    for fold, run in enumerate(runs):
        part, _ = zoo.fold_halves(run, source, fold, labels)
        halves += part
    return {cost: triple_trial.score_point(halves, dict(recipe, owner_cost=cost)) for cost in costs}


def stage_a(dfl, skill, source, cv_root, scratch, *, raw, output=RECIPE / "stage_a.json"):
    require_external_only()
    from dfl_submission import aligned_trial, triple_trial, zoo
    if output.exists():
        raise FileExistsError(f"{output} exists; D38 Stage A is scored once")
    zoo.check_original_source(source)
    labels = aligned_trial.training_labels(dfl, skill, raw=raw)
    recipe = triple_trial.d15_recipe()
    anchor = score_runs(anchor_runs(), source, labels, recipe, (8, *STAGE_A_COSTS))
    for cost, value in ANCHOR_RECORD.items():
        if abs(ht.primary(anchor[cost]) - value) > 5e-6:
            raise ValueError(f"D34's A2 fold networks do not reproduce their cost-{cost} score; nothing was written")
    print("D38 Stage A anchor reproduces D34's A2: " + ", ".join(f"cost {c} {ht.primary(t):.5f}"
                                                              for c, t in anchor.items()), flush=True)
    anchor = {cost: anchor[cost] for cost in STAGE_A_COSTS}
    entries = [{"setting": ANCHOR, "anchor": True, "trials": anchor}]
    for setting in settings():
        runs = train_folds(setting, source, cv_root, scratch)
        trials = score_runs(runs, source, labels, recipe, STAGE_A_COSTS)
        entries.append({"setting": setting, "anchor": False, "trials": trials,
                        "runs": {str(i): digest(run / "fit.json") for i, run in enumerate(runs)}})
        print(f"D38 Stage A {name(setting)}: " + ", ".join(f"cost {c} {ht.primary(t):.5f}" for c, t in trials.items()),
              flush=True)
    points = [(entry, cost, trial) for entry in entries for cost, trial in sorted(entry["trials"].items())]
    index, (entry, cost, trial) = max(enumerate(points), key=lambda item: (
        ht.primary(item[1][2]), ht.primary(item[1][2], "macro_f1"), -item[0]))
    anchor_best = max(ht.primary(t) for t in anchor.values())
    selected = {"setting": entry["setting"], "owner_cost": cost, "primary": ht.primary(trial),
                "gain_over_anchor_best": ht.primary(trial) - anchor_best}
    proceeds = not entry["anchor"] and selected["gain_over_anchor_best"] >= STAGE_A_GAIN
    write_new(output, {"protocol": PROTOCOL, "stage": "A", "anchor_best": anchor_best,
                       "entries": [dict(e, trials={str(c): t for c, t in e["trials"].items()}) for e in entries],
                       "selected": selected, "proceeds": proceeds, "recipe": recipe,
                       "source_manifest": digest(source / "manifest.json"),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D38 Stage A selected {name(selected['setting'])} at cost {cost}: {selected['primary']:.5f} "
          f"({selected['gain_over_anchor_best']:+.5f} over the anchor's best) -> "
          f"{'Stage B' if proceeds else 'stop; D15 stays'}", flush=True)
    return proceeds


def standing(recipe=RECIPE):
    record = json.loads((recipe / "stage_a.json").read_text())
    if not record["proceeds"]:
        raise ValueError("D38 Stage A did not clear its bar; D15 stays")
    return record["selected"]["setting"]


# -- Stage B ---------------------------------------------------------------------------------------

def network_folders(setting, models=MODELS):
    return {key: models / name(setting) / key for key in ht.SETS}


def train_networks(source, model_set, scratch, *, models=MODELS):
    from dfl_submission import scale_trial
    setting = standing()
    expected = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
    if digest(source / "manifest.json") != expected:
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for folder, (seed, boost) in ht.NETWORKS.items():
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
        found = {"lr_scale": fit.get("lr_scale", 1.), "batch": fit.get("batch", 16),
                 "weight_decay": fit.get("weight_decay", .01), "dropout": fit.get("dropout", .08),
                 "width": fit.get("width", 48), "steps": fit.get("steps", 1800)}
        if (fit["seed"], fit["boundary_boost"], fit["train_manifest"]) != (seed, boost, expected) or found != setting:
            raise ValueError(f"{output} was not trained as D38 prescribes")


def teacher(setting, models=MODELS):
    from dfl_submission import competition_adaptation, retrain_trial, scale_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    folders = network_folders(setting, models)
    packages = {role: scale_trial.package(scale_trial.TREES[key], folders[key], now)
                for role, key in scale_trial.ROLES.items()}
    return reference, packages, frozen


def identity(setting, models=MODELS):
    from dfl_submission import scale_trial
    folders = network_folders(setting, models)
    return {"protocol": PROTOCOL, "setting": setting,
            "networks": {key: {folder: digest(folders[key] / folder / "fit.json") for folder in ht.NETWORKS}
                         for key in ht.SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in ht.SETS},
            "teacher_sha256": diagnosis.teacher_identity()}


def external_members(model, key, dfl, skill, parsed, games, split, *, raw, cache):
    from dfl_submission import augment, pff, vendor_trial

    def produce():
        for half in augment.guard_halves(dfl, skill, split, raw=raw):
            yield (diagnosis.cache_name(half.source, half.match, half.view, half.index).removesuffix(".npz"),
                   {"source": half.source, "match": half.match, "view": half.view, "index": half.index},
                   mt.member_outputs(model, half.positions, half.ball, half.teams, half.pitch), half.labels)
            print(f"Members {half.key}", flush=True)
        for game in games:
            for index, half in enumerate(pff.load(parsed, game)):
                yield (f"pff_{game}_observed_{index}", {"source": "pff", "match": str(game), "view": "observed",
                                                         "index": index}, mt.member_outputs(model, *pff.tracking(half)),
                       half["labels"])
            print(f"Members PFF {game}", flush=True)

    return mt.collect(cache, dict(key, split=split, dfl_raw=raw, skill_manifest_sha256=digest(skill / "manifest.json"),
                                  pff=vendor_trial.inputs(parsed, games)), produce)


def metrica_members(model, key, cache):
    from dfl_submission import metrica_sample as ms, metrica_views as mv

    def produce():
        for game in ms.GAMES:
            for index, half in enumerate(ms.load(ms.PARSED, game)):
                yield (f"metrica_{game}_observed_{index}", {"source": "metrica", "match": str(game), "view": "observed",
                                                             "index": index},
                       mt.member_outputs(model, *ms.tracking(half)), half["labels"])
                for view, (positions, ball) in mv.build_views(half, game, index).items():
                    yield (f"metrica_{game}_{view}_{index}", {"source": "metrica", "match": str(game), "view": view,
                                                               "index": index},
                           mt.member_outputs(model, positions, ball, half["teams"], half["pitch"]), half["labels"])
            print(f"Members Metrica game {game}", flush=True)

    return mt.collect(cache, dict(key, metrica=ms.inputs(ms.PARSED), d24=mv.identity(ms.PARSED)), produce)


def mixes(owner_mix):
    return {"owner_mix": owner_mix, "dead_mix": .5}


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D38 development is scored once")
    setting = standing(recipe_folder)
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    model, key = teacher(setting, models), identity(setting, models)
    external = external_members(model, key, dfl, skill, parsed, games, "development", raw=raw,
                                cache=cache_root / name(setting) / "development")
    metrica = metrica_members(model, key, cache_root / name(setting) / "metrica")
    entries = []
    for owner_mix in OWNER_MIXES:
        halves = mt.half_signals(external, model, mixes(owner_mix), recipe)
        observed = mt.half_signals([m for m in metrica if m[0]["view"] == "observed"], model, mixes(owner_mix), recipe)
        views = mt.half_signals([m for m in metrica if m[0]["view"] != "observed"], model, mixes(owner_mix), recipe)
        for cost in OWNER_COSTS:
            candidate = ht.recipe_at(cost)
            trial = aligned_trial.evaluate(halves, candidate)
            change = refine_trial.changes(base, trial)
            veto = ms.veto(objective_trial.decoded(observed, candidate), reference="D15")
            view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
            d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
            d24["change"] = d24["candidate"] - d24["reference"]
            allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24["change"])
            entries.append({"point": {"setting": setting, "owner_mix": owner_mix, "owner_cost": cost},
                            "eligible": allowed, "changes": change, "trial": trial,
                            "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                            "d24": dict(d24, views=view_scores)})
            print(objective_trial.summary_line(f"D38 development owner_mix {owner_mix:g} cost {cost}", trial, change,
                                               f", Metrica {veto['changes']['primary']:+.4f} (veto "
                                               f"{'passed' if veto['changes']['passed'] else 'failed'}), "
                                               f"D24 {d24['change']:+.4f}", "eligible" if allowed else "not eligible"),
                  flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "setting": setting, "grid": {"owner_mix": list(OWNER_MIXES),
                       "owner_cost": list(OWNER_COSTS)}, "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected, "models": key,
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D38 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE, models=MODELS):
    development = json.loads((recipe_folder / "development.json").read_text())
    point = development["selected"]
    if point is None:
        raise ValueError("No D38 point was eligible on development; D15 stays and nothing is confirmed")
    if development["models"] != identity(point["setting"], models):
        raise ValueError("The D38 networks differ from the ones development scored")
    return point


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, objective_trial, refine_trial, scale_trial, vendor_trial
    point = frozen(recipe_folder, models)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D38 confirmation is scored once")
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
    model, key = teacher(point["setting"], models), identity(point["setting"], models)
    members = external_members(model, key, dfl, skill, parsed, games_b, "holdout", raw=raw,
                               cache=cache_root / name(point["setting"]) / "holdout")
    candidate_recipe = ht.recipe_at(point["owner_cost"])
    trial = aligned_trial.evaluate(mt.half_signals(members, model, mixes(point["owner_mix"]), candidate_recipe),
                                   candidate_recipe)
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": candidate_recipe,
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": key, "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line(f"D38 confirmation {point['owner_mix']:g}/{point['owner_cost']} (PFF = set B)",
                                       trial, change, "", "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the confirmed D38 networks, tree share and cost in D15's structure, then D27's rule."""
    from dfl_submission import kickoff, objective_trial
    from dfl_submission.build import write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D38 did not pass confirmation; build D27 instead (`kickoff build`)")
    point = frozen(recipe_folder, models)
    reference, packages, d3 = teacher(point["setting"], models)
    first = dict(reference[2], recipe=dict(reference[2]["recipe"], **mixes(point["owner_mix"])))
    model = ((*reference[:2], first, *reference[3:]), packages, dict(d3, recipe=ht.recipe_at(point["owner_cost"])))
    _, frozen_d15 = objective_trial.d15()
    labels = kickoff.model_labels(model, frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("stage-a")
    first.add_argument("--dfl", type=Path, required=True)
    first.add_argument("--dfl-raw", action="store_true")
    first.add_argument("--skill", type=Path, required=True)
    first.add_argument("--source", type=Path, required=True)
    first.add_argument("--cv", type=Path, required=True)
    first.add_argument("--scratch", type=Path, required=True)
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--set", choices=ht.SETS, required=True)
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
    if args.command == "stage-a":
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
