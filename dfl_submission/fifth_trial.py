"""D40: D36's networks with the tree share re-selected (DECISIONS.md D40).

`develop` caches the member outputs of D36's model (D35's method), checks that at D15's mixes they
reproduce D36's cached decoder inputs exactly, then scores the mix x cost grid once against D15.
`confirm` scores the selected point once on the reused holdout and PFF confirmation set C, and `build`
writes the competition file. Nothing is trained.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import diagnosis, fourth_trial as ft, hyper_trial as ht, mix_trial as mt, second_trial as st
from dfl_submission.external_only import require_external_only


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "fifth_recipe"
PROTOCOL = "DECISIONS.md D40"
OWNER_MIXES = (.5, .75, 1.)
OWNER_COSTS = (6, 8, 10)
D36_CACHE = ROOT / "data" / "d36" / "cache" / ht.name(st.SETTING)
D36_VARIANT = f"D36 hyperparameters {ht.name(st.SETTING)}"
GAIN = .01


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D40 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def model():
    return ht.teacher(st.SETTING, models=st.MODELS)


def key():
    return {"protocol": PROTOCOL, "networks": ht.identity(st.SETTING, st.MODELS),
            "teacher_sha256": diagnosis.teacher_identity()}


def same_inputs(fresh, cached):
    lookup = {(h.source, h.match, h.view, h.index): h for h in cached}
    if len(lookup) != len(fresh):
        return False
    for half in fresh:
        other = lookup.get((half.source, half.match, half.view, half.index))
        if other is None or not np.array_equal(half.labels, other.labels) or \
                not all(np.array_equal(x, y) for x, y in zip(half.signals, other.signals)):
            return False
    return True


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D40 development is scored once")
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    teacher, identity = model(), key()
    external = ft.external_members(teacher, identity, dfl, skill, parsed, games, "development", raw=raw,
                                   cache=cache_root / "development")
    metrica = ft.metrica_members(teacher, identity, cache_root / "metrica")
    d36_dev = ht.candidate_halves(dfl, skill, parsed, games, "development", st.SETTING, raw=raw,
                                  cache=D36_CACHE / "development", models=st.MODELS)
    _, d36_metrica = ms.reference_halves("D15", cache=D36_CACHE / "metrica", model=teacher,
                                         signals_of=diagnosis.half_signals, variant=D36_VARIANT)
    d36_views = mv.view_halves(cache=D36_CACHE / "d24", signals_of=lambda _, *t: diagnosis.half_signals(teacher, *t),
                               variant=D36_VARIANT)
    reference_mix = ft.mixes(.5)
    observed = [m for m in metrica if m[0]["view"] == "observed"]
    viewed = [m for m in metrica if m[0]["view"] != "observed"]
    if not (same_inputs(mt.half_signals(external, teacher, reference_mix, recipe), d36_dev)
            and same_inputs(mt.half_signals(observed, teacher, reference_mix, recipe), d36_metrica)
            and same_inputs(mt.half_signals(viewed, teacher, reference_mix, recipe), d36_views)):
        raise ValueError("D40 members at D15's mixes do not reproduce D36's cached inputs; nothing was written")
    print("D40 members reproduce D36's development, Metrica and D24 inputs exactly", flush=True)
    entries = []
    for owner_mix in OWNER_MIXES:
        mixes = ft.mixes(owner_mix)
        halves = mt.half_signals(external, teacher, mixes, recipe)
        metrica_halves = mt.half_signals(observed, teacher, mixes, recipe)
        views = mt.half_signals(viewed, teacher, mixes, recipe)
        for cost in OWNER_COSTS:
            candidate = ht.recipe_at(cost)
            trial = aligned_trial.evaluate(halves, candidate)
            change = refine_trial.changes(base, trial)
            veto = ms.veto(objective_trial.decoded(metrica_halves, candidate), reference="D15")
            view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
            d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
            d24["change"] = d24["candidate"] - d24["reference"]
            allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24["change"])
            entries.append({"point": {"owner_mix": owner_mix, "owner_cost": cost}, "eligible": allowed,
                            "changes": change, "trial": trial,
                            "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                            "d24": dict(d24, views=view_scores)})
            print(objective_trial.summary_line(f"D40 development owner_mix {owner_mix:g} cost {cost}", trial, change,
                                               f", Metrica {veto['changes']['primary']:+.4f}, D24 {d24['change']:+.4f}",
                                               "eligible" if allowed else "not eligible"), flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "grid": {"owner_mix": list(OWNER_MIXES), "owner_cost": list(OWNER_COSTS)},
                       "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True, "d36_members": True},
                       "entries": entries, "selected": selected, "identity": identity,
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D40 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE):
    development = json.loads((recipe_folder / "development.json").read_text())
    if development["selected"] is None:
        raise ValueError("No D40 point was eligible on development; D15 stays and nothing is confirmed")
    if development["identity"] != key():
        raise ValueError("D36's networks differ from the ones D40 development used")
    return development["selected"]


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, objective_trial, refine_trial, scale_trial, vendor_trial
    point = frozen(recipe_folder)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D40 confirmation is scored once")
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
    games_c = vendor_trial.split_games("confirm_c")
    model_d15 = scale_trial.teacher(setting_d15)
    set_key = {"protocol": PROTOCOL, "scale_models": scale_trial.identity(setting_d15),
               "teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games_c)}
    set_c = diagnosis.collect(cache_root / "d15_set_c", set_key, lambda: diagnosis.pff_halves(parsed, games_c, model_d15))
    base = aligned_trial.evaluate(external + set_c, recipe)
    teacher = model()
    members = ft.external_members(teacher, key(), dfl, skill, parsed, games_c, "holdout", raw=raw,
                                  cache=cache_root / "holdout")
    candidate = ht.recipe_at(point["owner_cost"])
    trial = aligned_trial.evaluate(mt.half_signals(members, teacher, ft.mixes(point["owner_mix"]), candidate), candidate)
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": candidate, "reproduction": {"d15_holdout": True},
                       "reference": base, "trial": trial, "changes": change, "passed": passed, "identity": key(),
                       "inputs": {"set_c": vendor_trial.inputs(parsed, games_c)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line(f"D40 confirmation {point} (PFF = set C)", trial, change, "",
                                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE):
    """The competition file: D36's networks with the confirmed mix and cost, then D27's kick-off rule."""
    from dfl_submission import kickoff, objective_trial
    from dfl_submission.build import write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D40 did not pass confirmation; build D27 instead (`kickoff build`)")
    point = frozen(recipe_folder)
    reference, packages, d3 = model()
    first = dict(reference[2], recipe=dict(reference[2]["recipe"], **ft.mixes(point["owner_mix"])))
    teacher = ((*reference[:2], first, *reference[3:]), packages, dict(d3, recipe=ht.recipe_at(point["owner_cost"])))
    _, frozen_d15 = objective_trial.d15()
    labels = kickoff.model_labels(teacher, frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
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
    if args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                cache_root=args.cache)


if __name__ == "__main__":
    main()
