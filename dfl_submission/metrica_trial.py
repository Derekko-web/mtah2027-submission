"""D30: Metrica's sample games as a training source (DECISIONS.md D30).

`prepare` writes one model set's Metrica training halves: each parsed half of the three games in
the views that set's DFL halves are trained on, with D30's seeds, as `source = "dfl"` records so
that the games count as DFL matches in the trees' weights and the networks' batches. `fit` combines
a set's stored training folder with the halves of a fold's games and refits that set's trees and
networks at D15's setting. `check` runs the pipeline checks (no Metrica half reproduces D15's trees
and networks; this module's held-out scoring reproduces D15's D19 and D24 records). `logo` scores
each fold on its held-out game and on development once; `confirm` scores the final fit once on the
reused holdout and PFF set B; `build` writes the competition file with the final fit and D27's rule.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
from shutil import copyfile
import subprocess
import sys
import time

import numpy as np

from dfl_submission import augment
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from dfl_submission.robust_features import stress
from dfl_transfer_features import corrupt


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "metrica_fit_recipe"
MODELS = PACKAGE / "metrica_models"
DATA = ROOT / "data" / "d30"
PROTOCOL = "DECISIONS.md D30"
SEEDS = {"mask": 870000, "stress": 880000, "augment": 890000}
SETS = ("current", "augmented")
GAMES = (1, 2, 3)
FOLD_GAMES = {"none": (), "fold1": (2, 3), "fold2": (1, 3), "fold3": (1, 2), "final": GAMES}
D24_VIEWS = ("masked", "handled", "team_swap")
LOGO_VIEWS = ("clean", *D24_VIEWS)
LOGO_TARGETS = ("raw", "aba1.75", "aba3", "aba4", "aba4.5", "control")
OTHER_TARGETS = ("aba1.75", "aba4", "aba4.5", "control")
PRIMARY = "aba3"
GAIN, LOSS, HALF_LOSS = .01, .01, .005


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D30 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def seed(kind, game, index):
    return SEEDS[kind] + 10 * game + index


# -- training halves -------------------------------------------------------------------------------

def training_views(half, game, index, model_set):
    """[(view, positions, ball)] of one parsed half in the views `model_set` trains its DFL halves on."""
    if model_set == "current":   # external_benchmark.halves on a training split
        positions, ball = half["positions"], half["ball"]
        return [("clean", positions, ball),
                ("masked", *corrupt(positions, ball, seed("mask", game, index))),
                ("stress", *stress(positions, ball, seed("stress", game, index)))]
    if model_set == "augmented":  # augment.training_halves
        rng = np.random.default_rng(seed("augment", game, index))
        positions = np.asarray(half["positions"], dtype=np.float32)
        ball = augment.handle_dead_ball(half["labels"], positions, half["ball"], rng)
        masked_positions, masked_ball = augment.wide_masking(positions, ball, rng)
        masked_positions = augment.swap_teams(masked_positions, half["teams"], rng, every=augment.TRAIN_SWAP_EVERY,
                                              frames=augment.training_frames)
        stress_positions, stress_ball = stress(positions, ball, seed("stress", game, index))
        stress_positions = augment.swap_teams(stress_positions, half["teams"], rng, every=augment.TRAIN_SWAP_EVERY,
                                              frames=augment.training_frames)
        return [("clean", positions, ball), ("masked", masked_positions, masked_ball),
                ("stress", stress_positions, stress_ball)]
    raise ValueError(f"Unknown model set {model_set!r}; expected one of {SETS}")


def prepare_halves(items, output, model_set, *, inputs):
    """Prepared sequences of (game, index, half) items, as `final_trial.prepare_sequences` writes them."""
    from dfl_submission import long_context
    from dfl_submission.temporal_network import targets
    output.mkdir(parents=True, exist_ok=False)
    records, columns = [], None
    for game, index, half in items:
        labels = np.asarray(half["labels"])
        for view, positions, ball in training_views(half, game, index, model_set):
            frames, table, _ = long_context.features(positions, ball, half["teams"], half["pitch"])
            if columns is None:
                columns = table.columns.tolist()
            elif columns != table.columns.tolist():
                raise ValueError("Metrica feature schema differs between halves")
            name = f"metrica_{game}_{view}_{index}.npz"
            np.savez_compressed(output / name, values=table.to_numpy(dtype=np.float32), targets=targets(labels[frames]))
            records.append({"file": name, "source": "dfl", "vendor": "metrica", "match": f"metrica{game}",
                            "game": int(game), "sha256": digest(output / name)})
            print(f"Prepared Metrica {model_set} game {game} half {index + 1} view {view}", flush=True)
    write_json(output / "manifest.json", {"protocol": PROTOCOL, "set": model_set, "columns": columns,
                                          "records": records, "inputs": inputs, "seeds": SEEDS})


def prepare(model_set, output, parsed=None):
    require_external_only()
    from dfl_submission import metrica_sample as ms
    parsed = ms.PARSED if parsed is None else parsed
    items = ((game, index, half) for game in GAMES for index, half in enumerate(ms.load(parsed, game)))
    prepare_halves(items, output, model_set, inputs=ms.inputs(parsed))


def link(source, target):
    """A hard link where the file system allows one (the data are read only), otherwise a copy."""
    try:
        os.link(source, target)
    except OSError:
        copyfile(source, target)


def combine(original, metrica, output, games):
    """A set's stored records followed by the prepared Metrica records of `games`, in prepared order."""
    from dfl_submission import aligned_trial
    first, second = (json.loads((folder / "manifest.json").read_text()) for folder in (original, metrica))
    if first["columns"] != second["columns"]:
        raise ValueError("The Metrica halves have a different feature schema from the stored halves")
    chosen = [record for record in second["records"] if record["game"] in tuple(games)]
    output.mkdir(parents=True, exist_ok=False)
    for record, path in aligned_trial.verified_records(original):
        link(path, output / record["file"])
    files = {record["file"]: path for record, path in aligned_trial.verified_records(metrica)}
    for record in chosen:
        link(files[record["file"]], output / record["file"])
    write_json(output / "manifest.json", dict(first, records=first["records"] + chosen, protocol=PROTOCOL,
                                              source_manifest=digest(original / "manifest.json"),
                                              metrica_manifest=digest(metrica / "manifest.json"),
                                              metrica_games=list(games)))
    print(f"Combined {len(first['records'])} stored + {len(chosen)} Metrica halves into {output}", flush=True)


# -- fitting ---------------------------------------------------------------------------------------

def stored_manifest(model_set):
    from dfl_submission import scale_trial
    return json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]


def fit(stored, metrica_root, fold, model_set, *, data=DATA, models=MODELS, device="mps"):
    """One set of one fold: the combined folder, both trees, then both networks at D15's setting."""
    require_external_only()
    from dfl_submission import scale_trial, touch_trial, zoo
    if digest(stored / "manifest.json") != stored_manifest(model_set):
        raise ValueError(f"{stored} is not the folder the {model_set} set was trained on")
    source = data / "combined" / fold / model_set
    if not (source / "manifest.json").exists():
        if source.exists():
            shutil.rmtree(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        combine(stored, metrica_root / model_set, source, FOLD_GAMES[fold])
    out = models / fold / model_set
    trees = out / "long_models"
    if not (trees / "fit.json").exists():
        if trees.exists():
            shutil.rmtree(trees)
        out.mkdir(parents=True, exist_ok=True)
        touch_trial.fit_trees(source, model_set, trees)
    (width, steps), _ = zoo.d15_setting()
    for slot, (seed_, boost) in scale_trial.NETWORKS.items():
        folder = out / slot
        if (folder / "fit.json").exists():
            continue
        raw = data / "scratch" / fold / model_set / slot
        if raw.exists():
            shutil.rmtree(raw)
        raw.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "train", "--source", str(source),
                        "--out", str(raw), "--device", device, "--seed", str(seed_), "--boundary-boost", str(boost),
                        "--width", str(width), "--steps", str(steps)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                        "--out", str(folder)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
        fitted = json.loads((folder / "fit.json").read_text())
        if (fitted["seed"], fitted["boundary_boost"], fitted["train_manifest"]) != (seed_, boost, digest(source / "manifest.json")):
            raise ValueError(f"{folder} was not trained as D30 prescribes")
        shutil.rmtree(raw)
    print(f"D30 fit {fold} {model_set} complete: {identity(models / fold)[model_set]}", flush=True)


def identity(folder):
    from dfl_submission import scale_trial
    result = {}
    for key in SETS:
        parts = {"trees": folder / key / "long_models"} | {slot: folder / key / slot for slot in scale_trial.NETWORKS}
        result[key] = {name: digest(path / "fit.json") if (path / "fit.json").exists() else None
                       for name, path in parts.items()}
    return result


def same_arrays(reference, candidate):
    """{file: True if every array of the reference .npz is equal in the candidate's} for every .npz."""
    result = {}
    for path in sorted(Path(reference).glob("*.npz")):
        other = Path(candidate) / path.name
        if not other.exists():
            result[path.name] = False
            continue
        with np.load(path, allow_pickle=False) as a, np.load(other, allow_pickle=False) as b:
            result[path.name] = (set(a.files) == set(b.files)
                                 and all(np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind == "f")
                                         for name in a.files))
    return result


def teacher(folder):
    """D15's structure, blend and recipe with both sets' trees and networks taken from `folder`."""
    from dfl_submission import competition_adaptation, retrain_trial, scale_trial, zoo
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    packages = {role: scale_trial.package(folder / key / "long_models", folder / key, now)
                for role, key in scale_trial.ROLES.items()}
    _, d15 = zoo.d15_setting()
    return reference, packages, dict(frozen, recipe=d15["recipe"])


# -- scoring ---------------------------------------------------------------------------------------

def held_out_scores(model, game, parsed=None):
    """{view: {target: score}} of a model on one Metrica game: D19's clean view and D24's views."""
    from dfl_submission import diagnosis, metrica_sample as ms, metrica_views as mv
    parsed = ms.PARSED if parsed is None else parsed
    halves = ms.load(parsed, game)
    labels = {view: [] for view in LOGO_VIEWS}
    for index, half in enumerate(halves):
        views = {"clean": (half["positions"], half["ball"]), **mv.build_views(half, game, index)}
        for view in LOGO_VIEWS:
            positions, ball = views[view]
            labels[view].append(diagnosis.half_signals(model, positions, ball, half["teams"], half["pitch"])[1])
        print(f"Forecast Metrica game {game} half {index + 1} in {len(LOGO_VIEWS)} views", flush=True)
    truth = ms.targets(halves)
    return {view: ms.score(truth, parts) for view, parts in labels.items()}


def d15_records():
    """{game: {view: {target: score}}} of D15 from its D19 and D24 records."""
    from dfl_submission import metrica_sample as ms, metrica_views as mv
    clean = json.loads(ms.record_path("D15").read_text())["matches"]
    views = json.loads(mv.RECORD.read_text())["views_report"]
    return {str(game): {"clean": clean[str(game)]} | {view: views[view]["matches"][str(game)] for view in D24_VIEWS}
            for game in GAMES}


def mean_score(scores, game, views, target):
    return float(np.mean([scores[game][view][target]["spell_f1"] for view in views]))


def logo_gates(reference, candidate):
    """D30's leave-one-game-out conditions 1-3 on {game: {view: {target: score}}} of D15 and the folds."""
    games = sorted(reference)
    per_game = {game: mean_score(candidate, game, D24_VIEWS, PRIMARY) - mean_score(reference, game, D24_VIEWS, PRIMARY)
                for game in games}
    d24_change = float(np.mean(list(per_game.values())))
    clean_change = float(np.mean([candidate[g]["clean"][PRIMARY]["spell_f1"] for g in games])
                         - np.mean([reference[g]["clean"][PRIMARY]["spell_f1"] for g in games]))
    other = {target: float(np.mean([mean_score(candidate, g, LOGO_VIEWS, target) for g in games])
                           - np.mean([mean_score(reference, g, LOGO_VIEWS, target) for g in games]))
             for target in OTHER_TARGETS}
    result = {"d24_per_game": per_game, "d24_change": d24_change, "gain": d24_change >= GAIN,
              "no_game_lower": all(value >= 0 for value in per_game.values()),
              "clean_change": clean_change, "clean_not_lower": clean_change >= 0,
              "other_targets": other, "other_targets_passed": all(value >= -LOSS for value in other.values())}
    result["passed"] = bool(result["gain"] and result["no_game_lower"] and result["clean_not_lower"]
                            and result["other_targets_passed"])
    return result


def development_guard(change):
    """D30 condition 4, per fold model on development."""
    return bool(change["primary"] >= -HALF_LOSS and all(value >= -LOSS for value in change["guards"].values())
                and change["pff"] >= -LOSS)


def confirmation_passed(change):
    """D30's confirmation of the final fit: the holdout and PFF set B."""
    return bool(change["primary"] >= -HALF_LOSS and all(value >= -LOSS for value in change["guards"].values())
                and change["pff"] >= -HALF_LOSS)


def model_halves(dfl, skill, parsed, games, split, model, models_identity, *, raw, cache):
    """A model's decoder inputs on the DFL/SkillCorner `split` and PFF `games`, cached by identity."""
    from dfl_submission import diagnosis, vendor_trial
    key = {"protocol": PROTOCOL, "models": models_identity, "teacher_sha256": diagnosis.teacher_identity(),
           "split": split, "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"),
           "pff": vendor_trial.inputs(parsed, games)}

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw)
        yield from diagnosis.pff_halves(parsed, games, model)

    return diagnosis.collect(cache, key, produce)


# -- the pipeline checks ---------------------------------------------------------------------------

def check(stored, metrica_root, *, data=DATA, output=RECIPE / "check.json"):
    """No Metrica half reproduces D15's trees and networks; held-out scoring reproduces D15's records."""
    require_external_only()
    from dfl_submission import metrica_sample as ms, scale_trial, zoo
    if output.exists():
        raise FileExistsError(f"{output} exists; D30's pipeline checks are recorded once")
    models = data / "check_models"
    for key in SETS:
        fit(stored[key], metrica_root, "none", key, data=data, models=models)
    folders = scale_trial.network_folders(zoo.d15_setting()[0])
    trees, networks = {}, {}
    for key in SETS:
        published = json.loads((scale_trial.TREES[key] / "fit.json").read_text())["models"]
        trees[key] = {name: digest(models / "none" / key / "long_models" / name) == value for name, value in published.items()}
        networks[key] = {slot: same_arrays(folders[key] / slot, models / "none" / key / slot)
                         for slot in scale_trial.NETWORKS}
    reference = d15_records()
    d15 = ms.reference_model("D15")
    scoring = {}
    for game in GAMES:
        scores = held_out_scores(d15, game)
        scoring[str(game)] = json.loads(json.dumps(scores)) == reference[str(game)]
    passed = (all(all(v.values()) for v in trees.values())
              and all(all(all(arrays.values()) and arrays for arrays in v.values()) for v in networks.values())
              and all(scoring.values()))
    write_new(output, {"protocol": PROTOCOL, "trees": trees, "networks": networks, "scoring": scoring,
                       "passed": passed, "metrica": {key: digest(metrica_root / key / "manifest.json") for key in SETS},
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D30 pipeline checks: trees {trees}, networks "
          f"{ {key: {slot: all(arrays.values()) for slot, arrays in v.items()} for key, v in networks.items()} }, "
          f"scoring {scoring} -> {'passed' if passed else 'FAILED'}", flush=True)
    if not passed:
        raise ValueError("D30's pipeline checks failed; nothing is scored")
    return passed


def check_passed(path=RECIPE / "check.json"):
    return path.exists() and json.loads(path.read_text())["passed"] is True


# -- leave one game out ----------------------------------------------------------------------------

def logo(dfl, skill, parsed, *, raw, d15_cache, cache_root=DATA / "cache", models=MODELS, output=RECIPE / "logo.json"):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, refine_trial, vendor_trial, zoo
    if output.exists():
        raise FileExistsError(f"{output} exists; D30's leave-one-game-out is scored once")
    if not check_passed():
        raise ValueError("D30's pipeline checks have not passed; nothing is scored")
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)  # stops unless reproduced
    ms.reference_scores("D15")                                                           # and its D19 record
    mv.reference_scores()                                                                # and its D24 record
    reference = d15_records()
    games = vendor_trial.development()
    candidate, development, identities = {}, {}, {}
    for game in GAMES:
        fold = f"fold{game}"
        identities[fold] = identity(models / fold)
        if any(value is None for parts in identities[fold].values() for value in parts.values()):
            raise ValueError(f"{models / fold} is not fully fitted")
        model = teacher(models / fold)
        candidate[str(game)] = held_out_scores(model, game)
        halves = model_halves(dfl, skill, parsed, games, "development", model, identities[fold], raw=raw,
                              cache=cache_root / fold / "development")
        trial = aligned_trial.evaluate(halves, recipe)
        change = refine_trial.changes(base, trial)
        development[fold] = {"changes": change, "passed": development_guard(change), "trial": trial}
        print(f"D30 {fold}: held-out game {game} D24 N = 3 "
              f"{mean_score(candidate, str(game), D24_VIEWS, PRIMARY):.5f} "
              f"(D15 {mean_score(reference, str(game), D24_VIEWS, PRIMARY):.5f}); development N = 3 standard "
              f"{change['primary']:+.5f}, guards " + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
              + f", PFF {change['pff']:+.4f} -> {'ok' if development[fold]['passed'] else 'FAILED'}", flush=True)
    gates = logo_gates(reference, candidate)
    passed = bool(gates["passed"] and all(entry["passed"] for entry in development.values()))
    write_new(output, {"protocol": PROTOCOL, "recipe": recipe, "reference": reference, "candidate": candidate,
                       "gates": gates, "development": development, "development_reference": base,
                       "passed": passed, "models": identities,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D30 leave one game out: D24 change {gates['d24_change']:+.5f} per game "
          + ", ".join(f"{g} {v:+.4f}" for g, v in gates["d24_per_game"].items())
          + f"; clean {gates['clean_change']:+.5f}; other targets "
          + ", ".join(f"{k} {v:+.4f}" for k, v in gates["other_targets"].items())
          + f" -> {'passed' if passed else 'failed'}", flush=True)
    return passed


# -- confirmation and build ------------------------------------------------------------------------

def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root=DATA / "cache", models=MODELS,
            output=RECIPE / "confirmation.json"):
    require_external_only()
    from dfl_submission import aligned_trial, diagnosis, refine_trial, scale_trial, vendor_trial, zoo
    if json.loads((RECIPE / "logo.json").read_text())["passed"] is not True:
        raise ValueError("D30 failed leave one game out; nothing is confirmed")
    if output.exists():
        raise FileExistsError(f"{output} exists; D30's confirmation is scored once")
    setting, frozen = zoo.d15_setting()
    recipe = frozen["recipe"]
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting, raw=raw,
                                           cache=d15_holdout_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full["external"]) != stored["external"]
            or vendor_trial.canonical(full["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    external = [half for half in reference if half.source != "pff"]
    games_b = vendor_trial.split_games("confirm_b")
    d15_model = scale_trial.teacher(setting)
    d15_model = (d15_model[0], d15_model[1], dict(d15_model[2], recipe=recipe))
    set_b = diagnosis.collect(cache_root / "d15_set_b",
                              {"protocol": PROTOCOL, "scale_models": scale_trial.identity(setting),
                               "teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games_b)},
                              lambda: diagnosis.pff_halves(parsed, games_b, d15_model))
    base = aligned_trial.evaluate(external + list(set_b), recipe)
    final = identity(models / "final")
    model = teacher(models / "final")
    halves = model_halves(dfl, skill, parsed, games_b, "holdout", model, final, raw=raw, cache=cache_root / "final_holdout")
    trial = aligned_trial.evaluate(halves, recipe)
    change = refine_trial.changes(base, trial)
    passed = confirmation_passed(change)
    write_new(output, {"protocol": PROTOCOL, "recipe": recipe, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": final, "reproduction": {"d15_holdout": True},
                       "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D30 confirmation (PFF = set B): N = 3 standard {change['primary']:+.5f}, "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f", PFF {change['pff']:+.4f} -> {'passed' if passed else 'failed'}", flush=True)
    return passed


def confirmed_teacher(models=MODELS, recipe_folder=RECIPE):
    """The final fit, after checking that it passed confirmation and is the model confirmation scored."""
    confirmation = json.loads((recipe_folder / "confirmation.json").read_text())
    if confirmation["passed"] is not True:
        raise ValueError("D30 did not pass confirmation")
    if confirmation["models"] != identity(models / "final"):
        raise ValueError("The D30 final models differ from the ones confirmation scored")
    return teacher(models / "final")


def build(output):
    """The competition file: D30's final fit, then D27's kick-off rule."""
    from dfl_submission import round8
    round8.build(output, metrica=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare")
    preparing.add_argument("--set", choices=SETS, required=True)
    preparing.add_argument("--out", type=Path, default=DATA / "metrica")
    for name in ("check", "fit"):
        command = commands.add_parser(name)
        command.add_argument("--current", type=Path, required=True, help="the stored original-set training folder")
        command.add_argument("--augmented", type=Path, required=True, help="the stored augmented-set training folder")
        command.add_argument("--metrica", type=Path, default=DATA / "metrica")
        if name == "fit":
            command.add_argument("--fold", choices=tuple(FOLD_GAMES), required=True)
            command.add_argument("--set", choices=SETS, required=True)
    for name in ("logo", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--d15-cache" if name == "logo" else "--d15-holdout-cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "prepare":
        prepare(args.set, args.out / args.set)
    elif args.command in ("check", "fit"):
        stored = {"current": args.current, "augmented": args.augmented}
        if args.command == "check":
            check(stored, args.metrica)
        else:
            fit(stored[args.set], args.metrica, args.fold, args.set,
                models=DATA / "check_models" if args.fold == "none" else MODELS)
    elif args.command == "logo":
        logo(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache)


if __name__ == "__main__":
    main()
