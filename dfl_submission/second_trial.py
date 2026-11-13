"""D36: D34's combined hyperparameters taken to development (DECISIONS.md D36).

The setting is fixed by the rule (learning rate x 2, batch 32, dropout 0.2). `networks` trains one model
set's two networks with `hyper_network`, `develop` scores the cost grid once on development against
D15, `confirm` scores the selected point once on the reused holdout and PFF confirmation set B, and
`build` writes the competition file. The model plumbing is `hyper_trial`'s.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dfl_submission import diagnosis, hyper_trial as ht
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "second_recipe"
MODELS = PACKAGE / "second_models"
PROTOCOL = "DECISIONS.md D36"
SETTING = {"lr_scale": 2., "batch": 32, "dropout": .2}
OWNER_COSTS = (8, 10, 12, 14)
GAIN = .01


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D36 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


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
            raise ValueError(f"{output} was not trained as D36 prescribes")


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D36 development is scored once")
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    folder = cache_root / ht.name(SETTING)
    halves = ht.candidate_halves(dfl, skill, parsed, games, "development", SETTING, raw=raw,
                                 cache=folder / "development", models=models)
    model = ht.teacher(SETTING, models=models)
    variant = f"D36 hyperparameters {ht.name(SETTING)}"
    _, metrica = ms.reference_halves("D15", cache=folder / "metrica", model=model, signals_of=diagnosis.half_signals,
                                     variant=variant)
    views = mv.view_halves(cache=folder / "d24", signals_of=lambda _, *tracking: diagnosis.half_signals(model, *tracking),
                           variant=variant)
    entries = []
    for cost in OWNER_COSTS:
        candidate = ht.recipe_at(cost)
        trial = aligned_trial.evaluate(halves, candidate)
        change = refine_trial.changes(base, trial)
        veto = ms.veto(objective_trial.decoded(metrica, candidate), reference="D15")
        view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
        d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
        d24["change"] = d24["candidate"] - d24["reference"]
        allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24["change"])
        entries.append({"point": {"setting": SETTING, "owner_cost": cost}, "eligible": allowed, "changes": change,
                        "trial": trial, "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                        "d24": dict(d24, views=view_scores)})
        print(objective_trial.summary_line(f"D36 development cost {cost}", trial, change,
                                           f", Metrica {veto['changes']['primary']:+.4f} (veto "
                                           f"{'passed' if veto['changes']['passed'] else 'failed'}), D24 {d24['change']:+.4f}",
                                           "eligible" if allowed else "not eligible"), flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "setting": SETTING, "costs": list(OWNER_COSTS), "recipe": recipe,
                       "reference": base, "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected, "models": ht.identity(SETTING, models),
                       "teacher_sha256": diagnosis.teacher_identity(),
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D36 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE, models=MODELS):
    development = json.loads((recipe_folder / "development.json").read_text())
    point = development["selected"]
    if point is None:
        raise ValueError("No D36 point was eligible on development; D15 stays and nothing is confirmed")
    if development["models"] != ht.identity(SETTING, models):
        raise ValueError("The D36 networks differ from the ones development scored")
    return point


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, objective_trial, refine_trial, scale_trial, vendor_trial
    point = frozen(recipe_folder, models)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D36 confirmation is scored once")
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
    candidate = ht.candidate_halves(dfl, skill, parsed, games_b, "holdout", SETTING, raw=raw,
                                    cache=cache_root / ht.name(SETTING) / "holdout", models=models)
    trial = aligned_trial.evaluate(candidate, ht.recipe_at(point["owner_cost"]))
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": ht.recipe_at(point["owner_cost"]),
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": ht.identity(SETTING, models),
                       "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line(f"D36 confirmation cost {point['owner_cost']} (PFF = set B)", trial, change, "",
                                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the confirmed D36 networks and cost in D15's structure, then D27's rule."""
    from dfl_submission import kickoff, objective_trial
    from dfl_submission.build import write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D36 did not pass confirmation; build D27 instead (`kickoff build`)")
    point = frozen(recipe_folder, models)
    _, frozen_d15 = objective_trial.d15()
    labels = kickoff.model_labels(ht.teacher(SETTING, point["owner_cost"], models), frozen_d15["pitch"])
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
