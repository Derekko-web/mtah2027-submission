"""D37: D36's training values with spell weighting and touch inputs (DECISIONS.md D37).

Variants (all at D36's learning rate x 2, batch 32, dropout 0.2): `plain` is D36's networks, reused;
`spell` adds D33's spell weight at alpha 0.5; `touch` trains on D16's extended inputs; `touch_spell`
does both. `check` confirms that `hyper_network` with the spell options at their defaults still
reproduces D20's fold-0 checkpoint. `networks` trains one set's networks for the new variants,
`develop` scores the grid once on development against D15, `confirm` once on the reused holdout and
PFF confirmation set B, and `build` writes the competition file.
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
RECIPE = PACKAGE / "third_recipe"
MODELS = PACKAGE / "third_models"
PROTOCOL = "DECISIONS.md D37"
VARIANTS = {"plain": {"alpha": 0., "touch": False}, "spell": {"alpha": .5, "touch": False},
            "touch": {"alpha": 0., "touch": True}, "touch_spell": {"alpha": .5, "touch": True}}
OWNER_COSTS = (8, 10)
GAIN = .01
TOUCH_PROTOCOL = "DECISIONS.md D16"


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D37 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def check(scratch, cv_root, output=RECIPE / "check.json"):
    """`hyper_network` with the spell options at alpha 0 and edge 0.2 reproduces D20's fold-0 checkpoint."""
    require_external_only()
    reference = json.loads((ht.anchor_run(0, cv_root) / "fit.json").read_text())
    trained = json.loads((scratch / "fit.json").read_text())
    for key in ("seed", "boundary_boost", "width", "steps", "train_manifest", "training_device", "torch"):
        if trained[key] != reference[key]:
            raise ValueError(f"The check run differs from D20's fold-0 run in {key}")
    filename = f"temporal_{ht.STEPS}.pt"
    expected, actual = reference["models"][filename], digest(scratch / filename)
    passed = expected == actual and "alpha" not in trained
    write_new(output, {"protocol": PROTOCOL, "expected": expected, "actual": actual, "passed": passed,
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D37 trainer check: {'bit for bit' if passed else 'DIFFERS'} ({actual[:12]} against {expected[:12]})",
          flush=True)
    return passed


def network_folders(variant, models=MODELS):
    if variant == "plain":
        return ht.network_folders(st.SETTING, st.MODELS)
    return {key: models / variant / key for key in ht.SETS}


def expected_source(model_set, touch):
    """The prepared-sequence manifest a variant's set must train on."""
    from dfl_submission import scale_trial
    published = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
    return published if not touch else ("touch", published)


def check_source(source, model_set, touch):
    manifest = json.loads((source / "manifest.json").read_text())
    published = expected_source(model_set, False)
    if touch:
        if manifest.get("touch_features") != TOUCH_PROTOCOL or manifest.get("base_manifest") != published:
            raise ValueError(f"{source} is not D16's extended prepared folder of the {model_set} set")
    elif digest(source / "manifest.json") != published:
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")


def train_networks(source, touch_source, model_set, scratch, *, models=MODELS):
    if json.loads((RECIPE / "check.json").read_text())["passed"] is not True:
        raise ValueError("The D37 trainer check did not pass; nothing is trained")
    check_source(source, model_set, False)
    check_source(touch_source, model_set, True)
    for variant, spec in VARIANTS.items():
        if variant == "plain":
            continue
        folder_source = touch_source if spec["touch"] else source
        for folder, (seed, boost) in ht.NETWORKS.items():
            output = network_folders(variant, models)[model_set] / folder
            if (output / "fit.json").exists():
                continue
            raw = scratch / variant / model_set / folder
            if not (raw / "fit.json").exists():
                if raw.exists():
                    raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                command = ht.train_command(folder_source, raw, st.SETTING, seed=seed, boost=boost)
                if spec["alpha"]:
                    command += ["--alpha", str(spec["alpha"])]
                subprocess.run(command, check=True, cwd=ROOT)
            subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                            "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
            fit = json.loads((output / "fit.json").read_text())
            if (fit["seed"], fit["boundary_boost"], fit["train_manifest"], fit.get("alpha", 0.)) != \
                    (seed, boost, digest(folder_source / "manifest.json"), spec["alpha"]) or \
                    {key: fit.get(key, ht.DEFAULTS[key]) for key in ht.DEFAULTS} != ht.full(st.SETTING):
                raise ValueError(f"{output} was not trained as D37 prescribes")


def teacher(variant, owner_cost=None, models=MODELS):
    from dfl_submission import competition_adaptation, retrain_trial, scale_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    folders = network_folders(variant, models)
    packages = {role: scale_trial.package(scale_trial.TREES[key], folders[key], now)
                for role, key in scale_trial.ROLES.items()}
    if owner_cost is not None:
        frozen = dict(frozen, recipe=ht.recipe_at(owner_cost))
    return reference, packages, frozen


def signals_of(variant):
    from dfl_submission import touch_trial
    return touch_trial.half_signals if VARIANTS[variant]["touch"] else diagnosis.half_signals


def identity(variant, models=MODELS):
    from dfl_submission import scale_trial
    folders = network_folders(variant, models)
    return {"variant": variant, "setting": st.SETTING,
            "networks": {key: {folder: digest(folders[key] / folder / "fit.json") for folder in ht.NETWORKS}
                         for key in ht.SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in ht.SETS}}


def candidate_halves(dfl, skill, parsed, games, split, variant, *, raw, cache, models=MODELS):
    from dfl_submission import vendor_trial
    if variant == "plain":
        return ht.candidate_halves(dfl, skill, parsed, games, split, st.SETTING, raw=raw, cache=cache, models=st.MODELS)
    key = {"third_models": identity(variant, models), "teacher_sha256": diagnosis.teacher_identity(), "split": split,
           "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"),
           "pff": vendor_trial.inputs(parsed, games)}
    model, forecast = teacher(variant, models=models), signals_of(variant)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw, signals_of=forecast)
        yield from diagnosis.pff_halves(parsed, games, model, signals_of=forecast)

    return diagnosis.collect(cache, key, produce)


def metrica_and_views(variant, folder, models=MODELS):
    from dfl_submission import metrica_sample as ms, metrica_views as mv
    model, forecast = teacher(variant, models=models), signals_of(variant)
    name = f"D36 hyperparameters {ht.name(st.SETTING)}" if variant == "plain" else f"D37 {variant}"
    _, metrica = ms.reference_halves("D15", cache=folder / "metrica", model=model, signals_of=forecast, variant=name)
    views = mv.view_halves(cache=folder / "d24", signals_of=lambda _, *tracking: forecast(model, *tracking),
                           variant=name)
    return metrica, views


def cache_folder(cache_root, variant):
    return Path("data/d36/cache") / ht.name(st.SETTING) if variant == "plain" else cache_root / variant


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D37 development is scored once")
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    entries = []
    for variant in VARIANTS:
        folder = cache_folder(cache_root, variant)
        halves = candidate_halves(dfl, skill, parsed, games, "development", variant, raw=raw,
                                  cache=folder / "development", models=models)
        metrica, views = metrica_and_views(variant, folder, models)
        for cost in OWNER_COSTS:
            candidate = ht.recipe_at(cost)
            trial = aligned_trial.evaluate(halves, candidate)
            change = refine_trial.changes(base, trial)
            veto = ms.veto(objective_trial.decoded(metrica, candidate), reference="D15")
            view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
            d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
            d24["change"] = d24["candidate"] - d24["reference"]
            allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24["change"])
            entries.append({"point": {"variant": variant, "owner_cost": cost}, "eligible": allowed, "changes": change,
                            "trial": trial, "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                            "d24": dict(d24, views=view_scores)})
            print(objective_trial.summary_line(f"D37 development {variant} cost {cost}", trial, change,
                                               f", Metrica {veto['changes']['primary']:+.4f} (veto "
                                               f"{'passed' if veto['changes']['passed'] else 'failed'}), "
                                               f"D24 {d24['change']:+.4f}", "eligible" if allowed else "not eligible"),
                  flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "variants": VARIANTS, "costs": list(OWNER_COSTS), "recipe": recipe,
                       "reference": base, "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected,
                       "models": {variant: identity(variant, models) for variant in VARIANTS},
                       "teacher_sha256": diagnosis.teacher_identity(),
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D37 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE, models=MODELS):
    development = json.loads((recipe_folder / "development.json").read_text())
    point = development["selected"]
    if point is None:
        raise ValueError("No D37 point was eligible on development; D15 stays and nothing is confirmed")
    if development["models"][point["variant"]] != identity(point["variant"], models):
        raise ValueError("The D37 networks differ from the ones development scored")
    return point


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, objective_trial, refine_trial, scale_trial, vendor_trial
    point = frozen(recipe_folder, models)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D37 confirmation is scored once")
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
    candidate = candidate_halves(dfl, skill, parsed, games_b, "holdout", point["variant"], raw=raw,
                                 cache=cache_root / f"{point['variant']}_holdout", models=models)
    trial = aligned_trial.evaluate(candidate, ht.recipe_at(point["owner_cost"]))
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": ht.recipe_at(point["owner_cost"]),
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": identity(point["variant"], models),
                       "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line(f"D37 confirmation {point} (PFF = set B)", trial, change, "",
                                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the confirmed D37 point in D15's structure, then D27's kick-off rule."""
    from dfl_submission import kickoff, objective_trial
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D37 did not pass confirmation; build D27 instead (`kickoff build`)")
    point = frozen(recipe_folder, models)
    _, frozen_d15 = objective_trial.d15()
    model, forecast = teacher(point["variant"], point["owner_cost"], models), signals_of(point["variant"])

    def decode_half(positions, ball, teams, pitch):
        return forecast(model, positions, ball, teams, pitch)[1]

    labels = tactical_halves(decode_half, centered=True, pitch=frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--scratch", type=Path, required=True)
    checking.add_argument("--cv", type=Path, required=True)
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--touch-source", type=Path, required=True)
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
    if args.command == "check":
        check(args.scratch, args.cv)
    elif args.command == "networks":
        train_networks(args.source, args.touch_source, args.set, args.scratch)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                cache_root=args.cache)


if __name__ == "__main__":
    main()
