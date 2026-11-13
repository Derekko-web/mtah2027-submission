"""D33: spell-balanced network training (DECISIONS.md D33).

`check` confirms that the D33 trainer at alpha 0 and edge 0.2 retrained D15's original-set seed-42
network bit for bit. `networks` trains and exports one model set's two networks for every new
objective. `develop` scores the objective x decoder grid once on development against D15 (with the
PFF, Metrica and D24 vetoes), `confirm` the selected point once on the reused holdout and PFF
confirmation set B, and `build` writes the competition file: the confirmed point, then D27's rule.
The XGBoost trees are D3's throughout, and inference stays NumPy only.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dfl_submission import diagnosis
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "objective_recipe"
MODELS = PACKAGE / "objective_models"
PROTOCOL = "DECISIONS.md D33"
OBJECTIVES = ((0., .2), (0., .6), (.5, .2), (.5, .6), (1., .2), (1., .6))     # (alpha, edge), in grid order
REFERENCE = (0., .2)                                                          # D15's own networks
DECODERS = ((8, .8), (8, .9), (10, .8), (10, .9), (12, .8), (12, .9))         # (ownership cost, discount)
WIDTH, STEPS = 96, 5400
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
SETS = ("current", "augmented")
GAIN, LOSS = .01, .01


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D33 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def name(objective):
    alpha, edge = objective
    return f"a{alpha:g}_e{edge:g}"


def points():
    return [{"alpha": alpha, "edge": edge, "owner_cost": cost, "discount": discount}
            for alpha, edge in OBJECTIVES for cost, discount in DECODERS]


def objective_of(point):
    return point["alpha"], point["edge"]


def d15():
    from dfl_submission import zoo
    return zoo.d15_setting()


def point_recipe(point):
    _, frozen = d15()
    return dict(frozen["recipe"], owner_cost=point["owner_cost"], discount=point["discount"])


# -- training --------------------------------------------------------------------------------------

def published_checkpoint():
    """D15's original-set seed-42 network: the folder and the hash of its last raw checkpoint."""
    from dfl_submission import scale_trial
    folder = scale_trial.network_folders(d15()[0])["current"] / "temporal_models"
    fit = json.loads((folder / "fit.json").read_text())
    return fit, fit["original_checkpoints"][f"temporal_{STEPS}.pt"]


def check(scratch, output=RECIPE / "check.json"):
    """D33's pipeline check: the trainer at alpha 0 and edge 0.2 reproduces D15's checkpoint bit for bit."""
    require_external_only()
    fit, expected = published_checkpoint()
    trained = json.loads((scratch / "fit.json").read_text())
    for key in ("seed", "boundary_boost", "width", "steps", "train_manifest", "training_device", "torch"):
        if trained[key] != fit[key]:
            raise ValueError(f"The check run differs from D15's in {key}: {trained[key]} against {fit[key]}")
    if (trained["alpha"], trained["edge"]) != REFERENCE:
        raise ValueError("The check run must use alpha 0 and edge 0.2")
    actual = digest(scratch / f"temporal_{STEPS}.pt")
    passed = actual == expected
    write_new(output, {"protocol": PROTOCOL, "expected": expected, "actual": actual, "passed": passed,
                       "fit": {key: trained[key] for key in sorted(trained) if key != "columns"},
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D33 trainer check: {'bit for bit' if passed else 'DIFFERS'} ({actual[:12]} against {expected[:12]})",
          flush=True)
    return passed


def network_folders(objective, models=MODELS):
    """{set: folder holding temporal_models and seed_models}: D15's published networks for the reference."""
    from dfl_submission import scale_trial
    if tuple(objective) == REFERENCE:
        return scale_trial.network_folders(d15()[0])
    return {key: models / name(objective) / key for key in SETS}


def train_networks(source, model_set, scratch, *, models=MODELS, device="mps", objectives=None):
    """Train and export one set's two networks for every new objective, as D15 trained them."""
    from dfl_submission import scale_trial
    if json.loads((RECIPE / "check.json").read_text())["passed"] is not True:
        raise ValueError("The D33 trainer check did not pass; nothing is trained")
    expected = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
    if digest(source / "manifest.json") != expected:
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for objective in objectives or OBJECTIVES:
        if tuple(objective) == REFERENCE:
            continue
        alpha, edge = objective
        for folder, (seed, boost) in NETWORKS.items():
            output = network_folders(objective, models)[model_set] / folder
            if (output / "fit.json").exists():
                continue
            raw = scratch / name(objective) / model_set / folder
            if not (raw / "fit.json").exists():
                if raw.exists():
                    raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                subprocess.run([sys.executable, "-m", "dfl_submission.objective_network", "--source", str(source),
                                "--out", str(raw), "--device", device, "--seed", str(seed), "--boundary-boost",
                                str(boost), "--width", str(WIDTH), "--steps", str(STEPS), "--alpha", str(alpha),
                                "--edge", str(edge)], check=True, cwd=ROOT)
            subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                            "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
            fit = json.loads((output / "fit.json").read_text())
            if (fit["seed"], fit["boundary_boost"], fit["train_manifest"], fit["alpha"], fit["edge"]) != \
                    (seed, boost, expected, alpha, edge):
                raise ValueError(f"{output} was not trained as D33 prescribes")


# -- the candidate models --------------------------------------------------------------------------

def teacher(objective, point=None, models=MODELS):
    """D15's structure with both sets' networks replaced by `objective`'s, decoded at `point` if given."""
    from dfl_submission import competition_adaptation, retrain_trial, scale_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    folders = network_folders(objective, models)
    packages = {role: scale_trial.package(scale_trial.TREES[key], folders[key], now)
                for role, key in scale_trial.ROLES.items()}
    if point is not None:
        frozen = dict(frozen, recipe=point_recipe(point))
    return reference, packages, frozen


def identity(objective, models=MODELS):
    from dfl_submission import scale_trial
    folders = network_folders(objective, models)
    return {"objective": list(objective),
            "networks": {key: {folder: digest(folders[key] / folder / "fit.json") for folder in NETWORKS} for key in SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in SETS}}


def variant(objective):
    return f"D33 objective {name(objective)}"


def candidate_halves(dfl, skill, parsed, games, split, objective, *, raw, cache, models=MODELS):
    from dfl_submission import vendor_trial
    key = {"objective_models": identity(objective, models), "teacher_sha256": diagnosis.teacher_identity(),
           "split": split, "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"),
           "pff": vendor_trial.inputs(parsed, games)}
    model = teacher(objective, models=models)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw)
        yield from diagnosis.pff_halves(parsed, games, model)

    return diagnosis.collect(cache, key, produce)


def metrica_halves(objective, cache, models=MODELS):
    from dfl_submission import metrica_sample as ms
    model = teacher(objective, models=models)
    _, halves = ms.reference_halves("D15", cache=cache, model=model, signals_of=diagnosis.half_signals,
                                    variant=variant(objective))
    return halves


def d24_halves(objective, cache, models=MODELS):
    from dfl_submission import metrica_views as mv
    model = teacher(objective, models=models)
    return mv.view_halves(cache=cache, signals_of=lambda _, *tracking: diagnosis.half_signals(model, *tracking),
                          variant=variant(objective))


def decoded(halves, recipe):
    """{game: [labels of half 0, labels of half 1]} of Metrica halves, decoded at `recipe`."""
    result = {}
    for half in sorted(halves, key=lambda item: (int(item.match), item.index)):
        result.setdefault(int(half.match), []).append(diagnosis.decode(half.signals, recipe))
    return result


# -- scoring ---------------------------------------------------------------------------------------

def eligible(change, metrica_passed, d24_change):
    from dfl_submission import refine_trial
    return bool(refine_trial.passed(change, gain=GAIN) and metrica_passed and d24_change >= -LOSS)


def select(entries):
    """The eligible point with the highest primary, then N = 3 standard macroF1, then grid order."""
    from dfl_submission import refine_trial
    chosen = [(i, entry) for i, entry in enumerate(entries) if entry["eligible"]]
    if not chosen:
        return None
    return max(chosen, key=lambda item: (refine_trial.standard(item[1]["trial"], refine_trial.PRIMARY),
                                         refine_trial.standard(item[1]["trial"], refine_trial.PRIMARY, "macro_f1"),
                                         -item[0]))[1]["point"]


def label(point):
    return f"alpha {point['alpha']:g}, edge {point['edge']:g}, owner_cost {point['owner_cost']}, discount {point['discount']:g}"


def summary_line(prefix, trial, change, extra, verdict):
    from dfl_submission import refine_trial
    return (f"{prefix}: N = 3 standard {refine_trial.standard(trial, 'aba3'):.5f} ({change['primary']:+.5f}), "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items()) + f", PFF {change['pff']:+.4f}"
            + extra + f" -> {verdict}")


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D33 development is scored once")
    reference_halves, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")                     # stops unless D15 reproduces its D19 record
    d24_reference, d24_reference_halves, truths = mv.reference_scores()   # and its D24 record
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    entries = []
    for objective in OBJECTIVES:
        if objective == REFERENCE:
            halves, metrica, views = reference_halves, d19_halves, d24_reference_halves
        else:
            folder = cache_root / name(objective)
            halves = candidate_halves(dfl, skill, parsed, games, "development", objective, raw=raw,
                                      cache=folder / "development", models=models)
            metrica = metrica_halves(objective, folder / "metrica", models)
            views = d24_halves(objective, folder / "d24", models)
        for cost, discount in DECODERS:
            point = {"alpha": objective[0], "edge": objective[1], "owner_cost": cost, "discount": discount}
            candidate = point_recipe(point)
            trial = aligned_trial.evaluate(halves, candidate)
            change = refine_trial.changes(base, trial)
            veto = ms.veto(decoded(metrica, candidate), reference="D15")
            view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
            d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
            d24["change"] = d24["candidate"] - d24["reference"]
            allowed = eligible(change, veto["changes"]["passed"], d24["change"])
            entries.append({"point": point, "eligible": allowed, "changes": change, "trial": trial,
                            "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                            "d24": dict(d24, views=view_scores)})
            print(summary_line(f"D33 development {label(point)}", trial, change,
                               f", Metrica {veto['changes']['primary']:+.4f} (veto "
                               f"{'passed' if veto['changes']['passed'] else 'failed'}), D24 {d24['change']:+.4f}",
                               "eligible" if allowed else "not eligible"), flush=True)
    selected = select(entries)
    write_new(output, {"protocol": PROTOCOL, "objectives": [list(o) for o in OBJECTIVES],
                       "decoders": [list(d) for d in DECODERS], "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected,
                       "models": {name(o): identity(o, models) for o in OBJECTIVES},
                       "teacher_sha256": diagnosis.teacher_identity(),
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D33 development selected {label(selected) if selected else 'no point; D15 stays'}", flush=True)
    return selected


def frozen(recipe_folder=RECIPE, models=MODELS):
    development = json.loads((recipe_folder / "development.json").read_text())
    point = development["selected"]
    if point is None:
        raise ValueError("No D33 point was eligible on development; D15 stays and nothing is confirmed")
    if development["models"][name(objective_of(point))] != identity(objective_of(point), models):
        raise ValueError("The D33 networks differ from the ones development scored")
    return development, point


def set_b_halves(parsed, cache):
    """D15's decoder inputs on PFF confirmation set B, computed once and cached by identity."""
    from dfl_submission import scale_trial, vendor_trial
    games = vendor_trial.split_games("confirm_b")
    setting, _ = d15()
    key = {"protocol": PROTOCOL, "scale_models": scale_trial.identity(setting),
           "teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games)}
    model = scale_trial.teacher(setting)
    return diagnosis.collect(cache, key, lambda: diagnosis.pff_halves(parsed, games, model)), games


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, refine_trial, scale_trial, vendor_trial
    development, point = frozen(recipe_folder, models)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D33 confirmation is scored once")
    setting, frozen_d15 = d15()
    recipe = frozen_d15["recipe"]
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting, raw=raw,
                                           cache=d15_holdout_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full["external"]) != stored["external"]
            or vendor_trial.canonical(full["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    external = [half for half in reference if half.source != "pff"]
    set_b, games_b = set_b_halves(parsed, cache_root / "d15_set_b")
    base = aligned_trial.evaluate(external + set_b, recipe)
    candidate = candidate_halves(dfl, skill, parsed, games_b, "holdout", objective_of(point), raw=raw,
                                 cache=cache_root / name(objective_of(point)) / "holdout", models=models)
    trial = aligned_trial.evaluate(candidate, point_recipe(point))
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": point_recipe(point),
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": identity(objective_of(point), models),
                       "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(summary_line(f"D33 confirmation {label(point)} (PFF = set B)", trial, change, "",
                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the confirmed D33 point in D15's structure, then D27's kick-off rule."""
    from dfl_submission import kickoff
    from dfl_submission.build import write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D33 did not pass confirmation; build D27 instead (`kickoff build`)")
    _, point = frozen(recipe_folder, models)
    _, frozen_d15 = d15()
    labels = kickoff.model_labels(teacher(objective_of(point), point, models), frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--scratch", type=Path, required=True, help="the check run's raw output folder")
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--set", choices=SETS, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    networks.add_argument("--only", nargs=2, type=float, action="append", metavar=("ALPHA", "EDGE"))
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
        check(args.scratch)
    elif args.command == "networks":
        objectives = None if args.only is None else [tuple(pair) for pair in args.only]
        if objectives is not None and any(tuple(o) not in OBJECTIVES for o in objectives):
            raise ValueError(f"--only must name grid objectives {OBJECTIVES}")
        train_networks(args.source, args.set, args.scratch, objectives=objectives)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                cache_root=args.cache)


if __name__ == "__main__":
    main()
