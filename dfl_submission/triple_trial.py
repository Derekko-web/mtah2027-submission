"""D22: finish D15's search and average three seeds per network slot (DECISIONS.md D22).

`stage-a` decodes D20's 24 fold networks on their held-out training matches and scores spellF1 over
the setting x ownership-cost grid, on the training matches only, after checking that the fold
inputs reproduce D20's cross-validation losses and the prepared targets. `networks` trains the
extra seeds of one model set at the Stage A setting. `develop` scores the three-seed candidate once
on development against D15 (with the PFF and Metrica vetoes), `confirm` once on the reused holdout
and PFF confirmation set C, and `build` writes the competition file. Inference stays NumPy only.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dfl_submission import zoo
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "triple_recipe"
MODELS = PACKAGE / "triple_models"
PROTOCOL = "DECISIONS.md D22"
SETTINGS = zoo.GRIDS["tcn"]                        # width 96 or 192 x 5,400, 10,800 or 19,800 steps
OWNER_COSTS = (8, 10, 12)
ANCHOR = {"width": 96, "steps": 5400, "owner_cost": 8}
STAGE_A_GAIN = .005
GAIN = .005
NETWORKS = zoo.NETWORKS                            # slot: (seed, boundary weight), in each model set
SETS = zoo.SETS
EXTRA_SEEDS = (1000, 2000)                         # a slot of seed s adds s + 1000 and s + 2000
METRICS = ("spell_f1", "macro_f1")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D22 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def setting_name(setting):
    return f"w{setting['width']}_s{setting['steps']}"


def d15_recipe():
    _, frozen = zoo.d15_setting()
    return frozen["recipe"]


# -- Stage A: fold networks on their held-out training matches -------------------------------------

def stage_a_points():
    return [dict(setting, owner_cost=cost) for setting in SETTINGS for cost in OWNER_COSTS]


def score_matches(decoded):
    """Per match and convention, spellF1 and macroF1 of already decoded (truth, prediction) pairs."""
    from dfl_submission import decoder_trial, diagnosis
    return {key: {name: diagnosis.score(diagnosis.target(truth, seconds), prediction)
                  for name, seconds in decoder_trial.CONVENTIONS.items()}
            for key, (truth, prediction, _) in decoded.items()}


def standard(per_match):
    """The standard combined aggregate per convention: DFL 1/3 clean + 2/3 masked, SkillCorner mean."""
    from dfl_submission import context_trial, decoder_trial
    result = {}
    for name in decoder_trial.CONVENTIONS:
        records = {key: {metric: value[name][metric] for metric in METRICS} for key, value in per_match.items()
                   if key[0] == "skillcorner" or key[2] in ("clean", "masked")}
        result[name] = {metric: context_trial.aggregate(records, metric)["combined"] for metric in METRICS}
    return result


def score_point(halves, recipe):
    from dfl_submission import diagnosis
    per_match = score_matches(diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe)))
    return {"standard": standard(per_match), "matches": {"|".join(key): value for key, value in per_match.items()}}


def stage_a_primary(trial, metric="spell_f1"):
    return trial["standard"]["aba3"][metric]


def stage_a_select(entries):
    """The highest primary, ties to N = 3 macroF1 then grid order; it replaces the anchor only by +0.005."""
    best = max(enumerate(entries), key=lambda item: (stage_a_primary(item[1]["trial"]),
                                                      stage_a_primary(item[1]["trial"], "macro_f1"), -item[0]))[1]
    anchor = next(entry for entry in entries if entry["point"] == ANCHOR)
    gain = stage_a_primary(best["trial"]) - stage_a_primary(anchor["trial"])
    replaces = best["point"] != ANCHOR and gain >= STAGE_A_GAIN
    return best["point"], gain, replaces, (best["point"] if replaces else dict(ANCHOR))


def check_cv_record(record, setting, fold, run, losses):
    """D22 pipeline check 1 and the run identity, against D20's cross-validation record."""
    entry = next(item for item in record["entries"] if item["setting"] == setting)
    if digest(run / "fit.json") != entry["runs"][str(fold)]["fit_sha256"]:
        raise ValueError(f"{run} is not the run D20 recorded for fold {fold}")
    expected = {key: value for key, value in entry["matches"].items()
                if key in {f"{source}:{match}" for source, match in losses}}
    found = json.loads(json.dumps({f"{source}:{match}": value for (source, match), value in losses.items()}))
    if found != expected or len(expected) != len(losses):
        raise ValueError(f"Fold {fold} of {zoo.name(setting)} does not reproduce D20's held-out log-losses")


def stage_a(dfl, skill, source, cv_root, *, raw, output=RECIPE / "stage_a.json"):
    require_external_only()
    from dfl_submission import aligned_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D22 Stage A is scored once")
    record = json.loads((zoo.RECIPE / "cv_tcn.json").read_text())
    if record["source_manifest"] != digest(source / "manifest.json") or record["grid"] != list(SETTINGS):
        raise ValueError("The prepared sequences or the grid differ from D20's temporal-network cross-validation")
    labels = aligned_trial.training_labels(dfl, skill, raw=raw)
    recipe = d15_recipe()
    entries = []
    for setting in SETTINGS:
        halves = []
        for fold in range(zoo.FOLDS):
            run = cv_root / "tcn" / zoo.name(setting) / f"fold{fold}"
            fold_part, losses = zoo.fold_halves(run, source, fold, labels)
            check_cv_record(record, setting, fold, run, losses)
            halves += fold_part
            print(f"D22 Stage A {zoo.name(setting)} fold {fold}: inputs checked", flush=True)
        for cost in OWNER_COSTS:
            point = dict(setting, owner_cost=cost)
            trial = score_point(halves, dict(recipe, owner_cost=cost))
            entries.append({"point": point, "trial": trial})
            print(f"D22 Stage A {setting_name(setting)} cost {cost}: N = 3 standard {stage_a_primary(trial):.5f}, "
                  + ", ".join(f"{name} {values['spell_f1']:.4f}" for name, values in trial["standard"].items()),
                  flush=True)
    best, gain, replaces, standing = stage_a_select(entries)
    write_new(output, {"protocol": PROTOCOL, "stage": "A", "grid": stage_a_points(), "anchor": ANCHOR,
                       "recipe": recipe, "entries": entries, "best": best, "best_gain_over_anchor": gain,
                       "replaces_anchor": replaces, "standing": standing,
                       "checks": {"cv_losses": True, "prepared_targets": True, "fit_hashes": True},
                       "source_manifest": digest(source / "manifest.json"),
                       "cv_record_sha256": digest(zoo.RECIPE / "cv_tcn.json"),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D22 Stage A: best {best} ({gain:+.5f} over the anchor); standing {standing}", flush=True)
    return standing


def standing_point(recipe=RECIPE):
    record = json.loads((recipe / "stage_a.json").read_text())
    return record["standing"]


# -- Stage B: the networks of each slot ------------------------------------------------------------

def is_d15_setting(point):
    return (point["width"], point["steps"]) == zoo.d15_setting()[0]


def slot_seeds(seed, point):
    """The seeds of a slot, in averaging order: its own seed, then s + 1000 and s + 2000."""
    return (seed, *(seed + extra for extra in EXTRA_SEEDS))


def slot_folders(model_set, point, models=MODELS):
    """{slot: [network folders in seed order]}: D15's published network first when the setting is D15's."""
    from dfl_submission import scale_trial
    result = {}
    for slot, (seed, _) in NETWORKS.items():
        folders = []
        for index, value in enumerate(slot_seeds(seed, point)):
            if index == 0 and is_d15_setting(point):
                folders.append(scale_trial.network_folders(zoo.d15_setting()[0])[model_set] / slot)
            else:
                folders.append(models / setting_name(point) / model_set / f"{slot}_seed{value}")
        result[slot] = folders
    return result


def train_networks(source, model_set, scratch, *, point=None, models=MODELS, device="mps"):
    """Train and export the networks a model set still needs at the standing setting (D22 Stage B)."""
    from dfl_submission import scale_trial
    point = standing_point() if point is None else point
    expected = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
    if digest(source / "manifest.json") != expected:
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for slot, folders in slot_folders(model_set, point, models).items():
        seed, boost = NETWORKS[slot]
        for index, (value, folder) in enumerate(zip(slot_seeds(seed, point), folders)):
            if (index == 0 and is_d15_setting(point)) or (folder / "fit.json").exists():
                continue  # D15's published network, or a network already exported
            raw = scratch / setting_name(point) / model_set / folder.name
            if not (raw / "fit.json").exists():
                subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "train", "--source", str(source),
                                "--out", str(raw), "--device", device, "--seed", str(value), "--boundary-boost",
                                str(boost), "--width", str(point["width"]), "--steps", str(point["steps"])],
                               check=True, cwd=ROOT)
            subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                            "--out", str(folder)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
            fit = json.loads((folder / "fit.json").read_text())
            if (fit["seed"], fit["boundary_boost"], fit["train_manifest"]) != (value, boost, expected):
                raise ValueError(f"{folder} was not trained as D22 prescribes")


def package(trees, slots, now):
    """`scale_trial.package` with each slot the mean of its networks (one network passes through unchanged)."""
    from dfl_submission import long_context, scale_trial
    reference, checkpoints, variants, _, _ = now
    inference = scale_trial.INFERENCE_NAME
    if set(reference[1]) != {inference} or set(checkpoints) != {inference}:
        raise ValueError("D3 is expected to use only the last snapshot of each network")
    forest, _ = long_context.load(trees)

    def slot(folders):
        return zoo.SlotMean([zoo.load_checkpoint(folder, scale_trial.final_snapshot(folder)) for folder in folders])

    return (forest, {inference: slot(slots["temporal_models"])}, reference[2], reference[3]), \
        {inference: slot(slots["seed_models"])}, variants


def teacher(point=None, models=MODELS, *, single=False):
    """D15's structure with three-seed slots at `point` (or D15's own single networks when `single`)."""
    from dfl_submission import competition_adaptation, retrain_trial, scale_trial
    point = standing_point() if point is None else point
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    packages = {}
    for role, key in scale_trial.ROLES.items():
        slots = slot_folders(key, point, models)
        if single:
            slots = {slot: folders[:1] for slot, folders in slots.items()}
        packages[role] = package(scale_trial.TREES[key], slots, now)
    return reference, packages, dict(frozen, recipe=dict(d15_recipe(), owner_cost=point["owner_cost"]))


def identity(point=None, models=MODELS):
    from dfl_submission import scale_trial
    point = standing_point() if point is None else point
    return {"point": point,
            "networks": {key: {slot: [digest(folder / "fit.json") for folder in folders]
                               for slot, folders in slot_folders(key, point, models).items()} for key in SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in SETS}}


def candidate_halves(dfl, skill, parsed, games, split, *, raw, cache, point=None, models=MODELS):
    from dfl_submission import diagnosis, vendor_trial
    key = {"triple": identity(point, models), "teacher_sha256": diagnosis.teacher_identity(), "split": split,
           "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}
    model = teacher(point, models)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw, signals_of=zoo.half_signals)
        yield from diagnosis.pff_halves(parsed, games, model, signals_of=zoo.half_signals)

    return diagnosis.collect(cache, key, produce)


def metrica_inputs(cache, point=None, models=MODELS):
    """The candidate's decoder inputs on the three Metrica sample games, cached by identity."""
    from dfl_submission import diagnosis, metrica_sample as ms
    key = {"triple": identity(point, models), "teacher_sha256": diagnosis.teacher_identity(), "inputs": ms.inputs(ms.PARSED)}
    model = teacher(point, models)

    def produce():
        for game in ms.GAMES:
            for index, half in enumerate(ms.load(ms.PARSED, game)):
                signals, labels = zoo.half_signals(model, *ms.tracking(half))
                yield diagnosis.HalfSignals("metrica", str(game), "observed", index, half["labels"], signals, labels)
            print(f"Forecast the D22 candidate on Metrica game {game}", flush=True)

    halves = diagnosis.collect(cache, key, produce)
    return {game: tuple(half.signals for half in halves if half.match == str(game)) for game in ms.GAMES}


# -- development, confirmation and build -----------------------------------------------------------

def summary_line(prefix, trial, change, extra, verdict):
    from dfl_submission import refine_trial
    return (f"{prefix}: N = 3 standard {refine_trial.standard(trial, 'aba3'):.5f} ({change['primary']:+.5f}), "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items()) + f", PFF {change['pff']:+.4f}"
            + extra + f" -> {verdict}")


def develop(dfl, skill, parsed, *, raw, d15_cache, cache, metrica_cache, recipe=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, refine_trial, vendor_trial
    output = recipe / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D22 development is scored once")
    point = standing_point(recipe)
    _, d15, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)  # stops unless reproduced
    candidate_recipe = dict(d15, owner_cost=point["owner_cost"])
    games = vendor_trial.development()
    halves = candidate_halves(dfl, skill, parsed, games, "development", raw=raw, cache=cache, point=point, models=models)
    trial = aligned_trial.evaluate(halves, candidate_recipe)
    change = refine_trial.changes(base, trial)
    veto = ms.veto(metrica_inputs(metrica_cache, point, models), reference="D15", recipe=candidate_recipe)
    passed = refine_trial.passed(change, gain=GAIN) and veto["changes"]["passed"]
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": candidate_recipe, "reference": base,
                       "reproduction": {"d15_development": True}, "trial": trial, "changes": change,
                       "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                       "passed": passed, "identity": identity(point, models),
                       "teacher_sha256": zoo.diagnosis_identity(), "inputs": vendor_trial.inputs(parsed, games),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(summary_line("D22 development", trial, change, f", Metrica {veto['changes']['primary']:+.4f}",
                       "eligible" if passed else "not eligible"), flush=True)
    return passed


def frozen(recipe=RECIPE, models=MODELS):
    development = json.loads((recipe / "development.json").read_text())
    if development["identity"] != identity(development["point"], models):
        raise ValueError("The D22 networks differ from the ones development scored")
    return development


def confirm(dfl, skill, parsed, *, raw, d15_cache, cache, recipe=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, refine_trial, scale_trial, vendor_trial
    development = frozen(recipe, models)
    if development["passed"] is not True:
        raise ValueError("The D22 candidate failed development; D15 stays and nothing is confirmed")
    output = recipe / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D22 confirmation is scored once")
    setting, d15 = zoo.d15_setting()
    games = vendor_trial.split_games("confirm_c")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games, "holdout", setting, raw=raw, cache=d15_cache)
    base = aligned_trial.evaluate(reference, d15["recipe"])
    expected = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]["external"]
    if vendor_trial.canonical(base["external"]) != expected:
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    halves = candidate_halves(dfl, skill, parsed, games, "holdout", raw=raw, cache=cache, point=development["point"],
                              models=models)
    challenger = aligned_trial.evaluate(halves, development["recipe"])
    change = refine_trial.changes(base, challenger)
    passed = refine_trial.passed(change, gain=GAIN)
    write_new(output, {"protocol": PROTOCOL, "point": development["point"], "recipe": development["recipe"],
                       "reproduction": {"d15_holdout": True}, "reference": base, "selected": challenger,
                       "changes": change, "passed": passed, "identity": development["identity"],
                       "inputs": vendor_trial.inputs(parsed, games), "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(summary_line("D22 confirmation (PFF = set C)", challenger, change, "", "passed" if passed else "failed"),
          flush=True)
    return passed


def build(output, recipe=RECIPE, models=MODELS):
    """The competition file with the confirmed three-seed networks and cost."""
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((recipe / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("The D22 candidate did not pass confirmation; build D15 instead")
    development = frozen(recipe, models)
    model = teacher(development["point"], models)
    _, frozen_d15 = zoo.d15_setting()

    def decode_half(positions, ball, teams, pitch):
        return zoo.half_signals(model, positions, ball, teams, pitch)[1]

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen_d15["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("stage-a")
    first.add_argument("--dfl", type=Path, required=True)
    first.add_argument("--dfl-raw", action="store_true")
    first.add_argument("--skill", type=Path, required=True)
    first.add_argument("--source", type=Path, required=True, help="the original set's prepared sequences")
    first.add_argument("--cv", type=Path, required=True, help="D20's cross-validation scratch folder")
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--set", choices=SETS, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    for name in ("develop", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--d15-cache", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        if name == "develop":
            command.add_argument("--metrica-cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "stage-a":
        stage_a(args.dfl, args.skill, args.source, args.cv, raw=args.dfl_raw)
    elif args.command == "networks":
        train_networks(args.source, args.set, args.scratch)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache=args.cache,
                metrica_cache=args.metrica_cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache=args.cache)


if __name__ == "__main__":
    main()
