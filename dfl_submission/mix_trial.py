"""D35: the tree share of each model set, re-selected for D15's networks (DECISIONS.md D35).

D15's decoder inputs are blends: in each model set the trees' ownership (and stoppage) output is mixed
with the seed-42 network's at `owner_mix` (`dead_mix`) 0.5, that mix with the phase-averaged seed-97
network at 0.5, and the two sets at 0.5. The first-level mixes were chosen in the original ensemble,
when the networks were 48 wide and trained for 1,800 steps. `members` caches every member's outputs
(both sets already mixed, leaf by leaf, as the build mixes them) on the development matches, the
Metrica games and the D24 views; `develop` re-blends them over the grid, decodes and scores once against
D15; `confirm` scores the selected point once on the reused holdout and PFF confirmation set C; `build`
writes the competition file. Nothing is trained, and at owner_mix = dead_mix = 0.5 the blend is D15's
exactly.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "mix_recipe"
PROTOCOL = "DECISIONS.md D35"
OWNER_MIXES = (.25, .5, .75, 1.)      # the seed-42 network's share against the trees, ownership
DEAD_MIXES = (.25, .5, .75, 1.)       # the same for the stoppage output
OWNER_COSTS = (6, 8, 10)
REFERENCE = {"owner_mix": .5, "dead_mix": .5, "owner_cost": 8}
GAIN, LOSS = .01, .01


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D35 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def points():
    return [{"owner_mix": a, "dead_mix": b, "owner_cost": c} for a in OWNER_MIXES for b in DEAD_MIXES for c in OWNER_COSTS]


# -- member outputs --------------------------------------------------------------------------------

def flatten(value, prefix="m"):
    """[(path, array)] of a nested forecast structure; None leaves are kept as markers."""
    if value is None:
        return [(prefix + "=none", None)]
    if isinstance(value, dict):
        return [item for key in sorted(value) for item in flatten(value[key], f"{prefix}/d:{key}")]
    if isinstance(value, (tuple, list)):
        return [item for i, part in enumerate(value) for item in flatten(part, f"{prefix}/t{i}")]
    return [(prefix, np.asarray(value))]


def unflatten(items):
    """The inverse of `flatten` for the structures forecasts produce (dicts and tuples)."""
    tree = {}
    for path, array in items:
        parts = path.split("/")[1:]
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        last = parts[-1]
        if last.endswith("=none"):
            node[last[:-5]] = None
        else:
            node[last] = array

    def build(node):
        if not isinstance(node, dict):
            return node
        keys = list(node)
        if all(key.startswith("t") for key in keys):
            return tuple(build(node[f"t{i}"]) for i in range(len(keys)))
        return {key[2:]: build(node[key]) for key in sorted(keys)}

    return build(tree)


def d15_model():
    from dfl_submission import scale_trial, zoo
    setting, frozen = zoo.d15_setting()
    reference, packages, d3 = scale_trial.teacher(setting)
    return reference, packages, dict(d3, recipe=frozen["recipe"])


def member_outputs(model, positions, ball, teams, pitch):
    """Both sets' member outputs for one half, mixed leaf by leaf at D3's set weight, as the build mixes them."""
    from dfl_submission import retrain_trial, vendor_trial
    _, _, frozen = model
    return retrain_trial.mix(*vendor_trial.model_outputs(model, positions, ball, teams, pitch), frozen["weight"])


def signals(mixed, model, point):
    """D15's four decoder inputs re-blended with the point's first-level mixes."""
    from dfl_submission import motion_trial
    reference, _, frozen = model
    previous = dict(reference[2]["recipe"], owner_mix=point["owner_mix"], dead_mix=point["dead_mix"])
    return tuple(motion_trial.blend_signals(mixed, previous, frozen["recipe"]))


def recipe_at(cost):
    from dfl_submission import zoo
    return dict(zoo.d15_setting()[1]["recipe"], owner_cost=cost)


def save_members(folder, key, mixed, labels, extra):
    items = flatten(mixed)
    arrays = {f"a{i}": array for i, (_, array) in enumerate(items) if array is not None}
    path = folder / f"{key}.npz"
    np.savez_compressed(path, labels=np.asarray(labels), **arrays)
    return dict(extra, file=path.name, paths=[p for p, _ in items], sha256=digest(path))


def load_members(folder, entry):
    path = folder / entry["file"]
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Cached member outputs differ from the manifest: {path.name}")
    with np.load(path, allow_pickle=False) as data:
        items = [(p, None if p.endswith("=none") else data[f"a{i}"]) for i, p in enumerate(entry["paths"])]
        return unflatten(items), data["labels"]


def collect(folder, identity, produce):
    """[(entry, mixed, labels)] from `folder` when its identity matches, otherwise computed and cached."""
    manifest = folder / "manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["identity"] != identity:
            raise ValueError(f"{folder} holds member outputs for different models or matches")
        return [(entry, *load_members(folder, entry)) for entry in stored["halves"]]
    folder.mkdir(parents=True, exist_ok=False)
    result, entries = [], []
    for key, extra, mixed, labels in produce():
        entry = save_members(folder, key, mixed, labels, extra)
        entries.append(entry)
        result.append((entry, mixed, np.asarray(labels)))
    manifest.write_text(json.dumps({"identity": identity, "halves": entries}, indent=2, sort_keys=True) + "\n")
    return result


def identity(**extra):
    from dfl_submission import scale_trial, zoo
    return {"protocol": PROTOCOL, "scale_models": scale_trial.identity(zoo.d15_setting()[0]),
            "teacher_sha256": diagnosis.teacher_identity()} | extra


def external_members(dfl, skill, parsed, games, split, *, raw, cache):
    from dfl_submission import augment, pff, vendor_trial
    model = d15_model()

    def produce():
        for half in augment.guard_halves(dfl, skill, split, raw=raw):
            mixed = member_outputs(model, half.positions, half.ball, half.teams, half.pitch)
            yield (diagnosis.cache_name(half.source, half.match, half.view, half.index).removesuffix(".npz"),
                   {"source": half.source, "match": half.match, "view": half.view, "index": half.index},
                   mixed, half.labels)
            print(f"Members {half.key}", flush=True)
        for game in games:
            for index, half in enumerate(pff.load(parsed, game)):
                mixed = member_outputs(model, *pff.tracking(half))
                yield (f"pff_{game}_observed_{index}", {"source": "pff", "match": str(game), "view": "observed",
                                                         "index": index}, mixed, half["labels"])
            print(f"Members PFF {game}", flush=True)

    key = identity(split=split, dfl_raw=raw, skill_manifest_sha256=digest(skill / "manifest.json"),
                   pff=vendor_trial.inputs(parsed, games))
    return collect(cache, key, produce)


def metrica_members(cache):
    from dfl_submission import metrica_sample as ms, metrica_views as mv
    model = d15_model()

    def produce():
        for game in ms.GAMES:
            for index, half in enumerate(ms.load(ms.PARSED, game)):
                yield (f"metrica_{game}_observed_{index}", {"source": "metrica", "match": str(game), "view": "observed",
                                                             "index": index},
                       member_outputs(model, *ms.tracking(half)), half["labels"])
                for view, (positions, ball) in mv.build_views(half, game, index).items():
                    yield (f"metrica_{game}_{view}_{index}", {"source": "metrica", "match": str(game), "view": view,
                                                               "index": index},
                           member_outputs(model, positions, ball, half["teams"], half["pitch"]), half["labels"])
            print(f"Members Metrica game {game}", flush=True)

    return collect(cache, identity(metrica=ms.inputs(ms.PARSED), d24=mv.identity(ms.PARSED)), produce)


def half_signals(items, model, point, recipe_for_labels):
    """`diagnosis.HalfSignals` for cached member outputs at `point` (labels decoded at `recipe_for_labels`)."""
    halves = []
    for entry, mixed, labels in items:
        values = signals(mixed, model, point)
        halves.append(diagnosis.HalfSignals(entry["source"], entry["match"], entry["view"], entry["index"], labels,
                                            values, diagnosis.decode(values, recipe_for_labels)))
    return halves


# -- scoring ---------------------------------------------------------------------------------------

def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D35 development is scored once")
    reference_halves, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    d24_base = mv.d24_mean(d24_reference)
    model = d15_model()
    games = vendor_trial.development()
    external = external_members(dfl, skill, parsed, games, "development", raw=raw, cache=cache_root / "development")
    metrica = metrica_members(cache_root / "metrica")
    # Pipeline check: at D15's mixes the re-blended inputs equal D15's cached inputs exactly.
    for halves_now, cached in ((half_signals(external, model, REFERENCE, recipe), reference_halves),
                               (half_signals([m for m in metrica if m[0]["view"] == "observed"], model, REFERENCE, recipe),
                                d19_halves),
                               (half_signals([m for m in metrica if m[0]["view"] != "observed"], model, REFERENCE, recipe),
                                d24_reference_halves)):
        lookup = {(h.source, h.match, h.view, h.index): h for h in cached}
        for half in halves_now:
            other = lookup[(half.source, half.match, half.view, half.index)]
            if not (np.array_equal(half.labels, other.labels)
                    and all(np.array_equal(x, y) for x, y in zip(half.signals, other.signals))):
                raise ValueError(f"D35 members at D15's mixes do not reproduce D15's inputs for {half.key}; nothing written")
    print("D35 members reproduce D15's development, Metrica and D24 inputs exactly", flush=True)
    entries = []
    for a in OWNER_MIXES:
        for b in DEAD_MIXES:
            mixes = {"owner_mix": a, "dead_mix": b}
            halves = half_signals(external, model, mixes, recipe)
            observed = half_signals([m for m in metrica if m[0]["view"] == "observed"], model, mixes, recipe)
            views = half_signals([m for m in metrica if m[0]["view"] != "observed"], model, mixes, recipe)
            for cost in OWNER_COSTS:
                point = dict(mixes, owner_cost=cost)
                candidate = recipe_at(cost)
                trial = aligned_trial.evaluate(halves, candidate)
                change = refine_trial.changes(base, trial)
                veto = ms.veto(objective_trial.decoded(observed, candidate), reference="D15")
                view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
                d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
                d24["change"] = d24["candidate"] - d24["reference"]
                allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24["change"])
                entries.append({"point": point, "eligible": allowed, "changes": change, "trial": trial,
                                "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                                "d24": dict(d24, views=view_scores)})
                print(objective_trial.summary_line(f"D35 development owner_mix {a:g}, dead_mix {b:g}, cost {cost}",
                                                   trial, change, f", Metrica {veto['changes']['primary']:+.4f}, "
                                                   f"D24 {d24['change']:+.4f}", "eligible" if allowed else "not eligible"),
                      flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "grid": points(), "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True, "members": True},
                       "entries": entries, "selected": selected, "identity": identity(),
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D35 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE):
    development = json.loads((recipe_folder / "development.json").read_text())
    if development["selected"] is None:
        raise ValueError("No D35 point was eligible on development; D15 stays and nothing is confirmed")
    if development["identity"] != identity():
        raise ValueError("D15's models differ from the ones D35 development used")
    return development["selected"]


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, refine_trial, scale_trial, vendor_trial, zoo
    point = frozen(recipe_folder)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D35 confirmation is scored once")
    setting, frozen_d15 = zoo.d15_setting()
    recipe = frozen_d15["recipe"]
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting, raw=raw,
                                           cache=d15_holdout_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full["external"]) != stored["external"]
            or vendor_trial.canonical(full["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    games_c = vendor_trial.split_games("confirm_c")
    model = d15_model()
    members = external_members(dfl, skill, parsed, games_c, "holdout", raw=raw, cache=cache_root / "holdout")
    base_halves = half_signals(members, model, REFERENCE, recipe)
    external_now = [h for h in base_halves if h.source != "pff"]
    lookup = {(h.source, h.match, h.view, h.index): h for h in reference if h.source != "pff"}
    for half in external_now:
        other = lookup[(half.source, half.match, half.view, half.index)]
        if not all(np.array_equal(x, y) for x, y in zip(half.signals, other.signals)):
            raise ValueError("D35 members do not reproduce D15's holdout inputs; no record was written")
    base = aligned_trial.evaluate(base_halves, recipe)
    trial = aligned_trial.evaluate(half_signals(members, model, point, recipe_at(point["owner_cost"])),
                                   recipe_at(point["owner_cost"]))
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "reproduction": {"d15_holdout": True},
                       "reference": base, "trial": trial, "changes": change, "passed": passed,
                       "inputs": {"set_c": vendor_trial.inputs(parsed, games_c)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    from dfl_submission import objective_trial
    print(objective_trial.summary_line(f"D35 confirmation {point} (PFF = set C)", trial, change, "",
                                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE):
    """The competition file: D15 with the confirmed mixes and cost, then D27's kick-off rule."""
    from dfl_submission import kickoff, zoo
    from dfl_submission.build import write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D35 did not pass confirmation; build D27 instead (`kickoff build`)")
    point = frozen(recipe_folder)
    reference, packages, d3 = d15_model()
    first = dict(reference[2], recipe=dict(reference[2]["recipe"], owner_mix=point["owner_mix"],
                                          dead_mix=point["dead_mix"]))
    model = ((*reference[:2], first, *reference[3:]), packages, dict(d3, recipe=recipe_at(point["owner_cost"])))
    labels = kickoff.model_labels(model, zoo.d15_setting()[1]["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("members", "develop", "confirm"):
        command = commands.add_parser(command_name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        if command_name == "develop":
            command.add_argument("--d15-cache", type=Path, required=True)
        if command_name == "confirm":
            command.add_argument("--d15-holdout-cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "members":
        from dfl_submission import vendor_trial
        external_members(args.dfl, args.skill, args.parsed, vendor_trial.development(), "development",
                         raw=args.dfl_raw, cache=args.cache / "development")
        metrica_members(args.cache / "metrica")
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                cache_root=args.cache)


if __name__ == "__main__":
    main()
