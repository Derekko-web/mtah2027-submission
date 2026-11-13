"""D41: the best-evidenced candidate to confirmation (DECISIONS.md D41).

Two candidates: D36's networks at cost 8 (its development row is reused from D36's record) and new
networks at learning rate x 2, batch 32, dropout 0.3 at costs 8, 10 and 12. `networks` trains one set's
two new networks, `develop` scores the new points once and selects under D41's gates, `confirm` scores
the selected point once on the reused holdout and PFF confirmation set B, and `build` writes the
competition file. The model plumbing is `hyper_trial`'s.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dfl_submission import diagnosis, hyper_trial as ht, second_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "sixth_recipe"
MODELS = PACKAGE / "sixth_models"
PROTOCOL = "DECISIONS.md D41"
SETTING = {"lr_scale": 2., "batch": 32, "dropout": .3}
OWNER_COSTS = (8, 10, 12)
D36_COST = 8
DEVELOPMENT_GAIN, LOSS, CONFIRMATION_GAIN = .005, .01, .01


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D41 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def candidate_models(candidate):
    """(setting, models folder) of a candidate name."""
    return (st.SETTING, st.MODELS) if candidate == "D36" else (SETTING, MODELS)


def train_networks(source, model_set, scratch, *, models=MODELS):
    from dfl_submission import scale_trial
    expected = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
    if digest(source / "manifest.json") != expected:
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for folder, (seed, boost) in ht.NETWORKS.items():
        output = ht.network_folders(SETTING, models)[model_set] / folder
        if (output / "fit.json").exists():
            continue
        raw = scratch / ht.name(SETTING) / model_set / folder
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            subprocess.run(ht.train_command(source, raw, SETTING, seed=seed, boost=boost), check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
        fit = json.loads((output / "fit.json").read_text())
        if (fit["seed"], fit["boundary_boost"], fit["train_manifest"]) != (seed, boost, expected) or \
                {key: fit.get(key, ht.DEFAULTS[key]) for key in ht.DEFAULTS} != ht.full(SETTING):
            raise ValueError(f"{output} was not trained as D41 prescribes")


def eligible(change, metrica_passed, d24_change):
    from dfl_submission import refine_trial
    return bool(refine_trial.passed(change, gain=DEVELOPMENT_GAIN) and metrica_passed and d24_change >= -LOSS)


def d36_entry():
    """D36's cost-8 development row, from its record, checked against D36's networks."""
    record = json.loads((st.RECIPE / "development.json").read_text())
    if record["models"] != ht.identity(st.SETTING, st.MODELS):
        raise ValueError("D36's networks differ from the ones its development scored")
    entry = next(e for e in record["entries"] if e["point"]["owner_cost"] == D36_COST)
    return {"point": {"candidate": "D36", "owner_cost": D36_COST}, "changes": entry["changes"], "trial": entry["trial"],
            "metrica": entry["metrica"], "d24": entry["d24"], "reference": record["reference"],
            "eligible": eligible(entry["changes"], entry["metrica"]["changes"]["passed"], entry["d24"]["change"])}


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D41 development is scored once")
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d36 = d36_entry()
    if json.loads(json.dumps(d36.pop("reference"))) != json.loads(json.dumps(base)):
        raise ValueError("D36's record was scored against a different D15 reference")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    folder = cache_root / ht.name(SETTING)
    halves = ht.candidate_halves(dfl, skill, parsed, games, "development", SETTING, raw=raw,
                                 cache=folder / "development", models=models)
    model = ht.teacher(SETTING, models=models)
    variant = f"D41 hyperparameters {ht.name(SETTING)}"
    _, metrica = ms.reference_halves("D15", cache=folder / "metrica", model=model, signals_of=diagnosis.half_signals,
                                     variant=variant)
    views = mv.view_halves(cache=folder / "d24", signals_of=lambda _, *tracking: diagnosis.half_signals(model, *tracking),
                           variant=variant)
    entries = [d36]
    print(objective_trial.summary_line("D41 development D36 cost 8 (from D36's record)", d36["trial"], d36["changes"],
                                       "", "eligible" if d36["eligible"] else "not eligible"), flush=True)
    for cost in OWNER_COSTS:
        candidate = ht.recipe_at(cost)
        trial = aligned_trial.evaluate(halves, candidate)
        change = refine_trial.changes(base, trial)
        veto = ms.veto(objective_trial.decoded(metrica, candidate), reference="D15")
        view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
        d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
        d24["change"] = d24["candidate"] - d24["reference"]
        allowed = eligible(change, veto["changes"]["passed"], d24["change"])
        entries.append({"point": {"candidate": "D41", "owner_cost": cost}, "eligible": allowed, "changes": change,
                        "trial": trial, "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                        "d24": dict(d24, views=view_scores)})
        print(objective_trial.summary_line(f"D41 development dropout 0.3 cost {cost}", trial, change,
                                           f", Metrica {veto['changes']['primary']:+.4f} (veto "
                                           f"{'passed' if veto['changes']['passed'] else 'failed'}), D24 {d24['change']:+.4f}",
                                           "eligible" if allowed else "not eligible"), flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected,
                       "models": {"D36": ht.identity(st.SETTING, st.MODELS), "D41": ht.identity(SETTING, models)},
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D41 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE, models=MODELS):
    development = json.loads((recipe_folder / "development.json").read_text())
    point = development["selected"]
    if point is None:
        raise ValueError("No D41 point was eligible on development; D15 stays and nothing is confirmed")
    setting, folder = candidate_models(point["candidate"])
    if development["models"][point["candidate"]] != ht.identity(setting, folder if point["candidate"] == "D36" else models):
        raise ValueError("The selected networks differ from the ones development scored")
    return point


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, objective_trial, refine_trial, scale_trial, vendor_trial
    point = frozen(recipe_folder, models)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D41 confirmation is scored once")
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
    setting, folder = candidate_models(point["candidate"])
    folder = folder if point["candidate"] == "D36" else models
    candidate = ht.candidate_halves(dfl, skill, parsed, games_b, "holdout", setting, raw=raw,
                                    cache=cache_root / f"{point['candidate']}_holdout", models=folder)
    trial = aligned_trial.evaluate(candidate, ht.recipe_at(point["owner_cost"]))
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=CONFIRMATION_GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": ht.recipe_at(point["owner_cost"]),
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": ht.identity(setting, folder),
                       "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line(f"D41 confirmation {point} (PFF = set B)", trial, change, "",
                                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the confirmed D41 point in D15's structure, then D27's kick-off rule."""
    from dfl_submission import kickoff, objective_trial
    from dfl_submission.build import write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D41 did not pass confirmation; build D27 instead (`kickoff build`)")
    point = frozen(recipe_folder, models)
    setting, folder = candidate_models(point["candidate"])
    folder = folder if point["candidate"] == "D36" else models
    _, frozen_d15 = objective_trial.d15()
    labels = kickoff.model_labels(ht.teacher(setting, point["owner_cost"], folder), frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
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
    if args.command == "networks":
        train_networks(args.source, args.set, args.scratch)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                cache_root=args.cache)


if __name__ == "__main__":
    main()
