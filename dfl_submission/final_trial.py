"""D23: the standing recipe refit on all 27 labelled external matches (DECISIONS.md D23).

`evidence` decodes D20's fold networks (trained on 12 matches) and the standing recipe's matching
network (trained on 16) alone on development and records whether the 16-match network wins by the
rule's margin. `prepare` writes both model sets' prepared sequences for every labelled match and
checks that the 16 training matches reproduce the stored sequences exactly. `fit` refits the trees
and networks on them. `check` scores the standing and the refit model once on three sources neither
was trained on, and `build` writes the competition file. Inference stays NumPy only.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from dfl_submission import zoo
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "final_recipe"
MODELS = PACKAGE / "final_models"
PROTOCOL = "DECISIONS.md D23"
EVIDENCE_GAIN = .005
CHECK_LOSS = .01
SETS = zoo.SETS
NETWORKS = zoo.NETWORKS
STORED = {"current": "train_data", "augmented": "sequences_augmented"}


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D23 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


# -- the standing recipe ---------------------------------------------------------------------------

def standing():
    """The submission when D23 starts: D22's three-seed model if it was confirmed, otherwise D15."""
    from dfl_submission import triple_trial
    confirmation = triple_trial.RECIPE / "confirmation.json"
    if confirmation.exists() and json.loads(confirmation.read_text())["passed"] is True:
        development = triple_trial.frozen()
        point = development["point"]
        slots = {key: triple_trial.slot_folders(key, point) for key in SETS}
        return {"name": "D22", "point": point, "recipe": development["recipe"], "slots": slots}
    setting, frozen = zoo.d15_setting()
    from dfl_submission import scale_trial
    folders = scale_trial.network_folders(setting)
    slots = {key: {slot: [folders[key] / slot] for slot in NETWORKS} for key in SETS}
    point = {"width": setting[0], "steps": setting[1], "owner_cost": frozen["recipe"]["owner_cost"]}
    return {"name": "D15", "point": point, "recipe": frozen["recipe"], "slots": slots}


def slot_seeds(model_set, slot, standing_model):
    return [json.loads((folder / "fit.json").read_text())["seed"] for folder in standing_model["slots"][model_set][slot]]


def model(trees, slots, recipe):
    """D15's structure with the given trees and network slots, decoded at `recipe`."""
    from dfl_submission import competition_adaptation, retrain_trial, scale_trial, triple_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    packages = {role: triple_trial.package(trees[key], slots[key], now) for role, key in scale_trial.ROLES.items()}
    return reference, packages, dict(frozen, recipe=recipe)


def standing_model(standing_record=None):
    from dfl_submission import scale_trial
    record = standing() if standing_record is None else standing_record
    return model(dict(scale_trial.TREES), record["slots"], record["recipe"])


def candidate_folders(record, models=MODELS):
    trees = {key: models / key / "long_models" for key in SETS}
    slots = {key: {slot: [models / key / f"{slot}_seed{seed}" for seed in slot_seeds(key, slot, record)]
                   for slot in NETWORKS} for key in SETS}
    return trees, slots


def candidate_model(record=None, models=MODELS):
    record = standing() if record is None else record
    trees, slots = candidate_folders(record, models)
    return model(trees, slots, record["recipe"])


def identity(trees, slots):
    return {"trees": {key: digest(folder / "fit.json") for key, folder in trees.items()},
            "networks": {key: {slot: [digest(folder / "fit.json") for folder in folders] for slot, folders in value.items()}
                         for key, value in slots.items()}}


# -- evidence: does a network trained on more matches decode better? -------------------------------

def network_signals(networks, positions, ball, teams, pitch):
    """Each network's four 25 Hz outputs for one half, from one feature extraction."""
    from dfl_submission import long_context, temporal_cpu
    frames, owner, _ = long_context.features(positions, ball, teams, pitch)
    values, columns, length = owner.to_numpy(dtype=np.float32), owner.columns.tolist(), len(ball)
    result = {}
    for name, (kind, network) in networks.items():
        if kind == "fold":
            family, weights, norm, expected = network
            if columns != expected:
                raise ValueError("Feature columns differ from the ones the fold network was trained on")
            probabilities = zoo.network_probabilities(family, weights, norm, values)
            if len(frames) != len(probabilities[0]) or not np.array_equal(frames, np.arange(0, length, zoo.FEATURE_STRIDE)):
                raise ValueError("Unexpected feature frames")
            result[name] = zoo.fold_signals(probabilities, length)
        else:
            result[name] = temporal_cpu.predict(network, (frames, values, columns, length))
    return result


def evidence(dfl, skill, cv_root, *, raw, output=RECIPE / "evidence.json"):
    require_external_only()
    from dfl_submission import diagnosis, external_benchmark, scale_trial, temporal_cpu, triple_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D23's evidence is scored once")
    record = standing()
    point, recipe = record["point"], record["recipe"]
    setting = {"steps": point["steps"], "width": point["width"]}
    cv = json.loads((zoo.RECIPE / "cv_tcn.json").read_text())
    entry = next(item for item in cv["entries"] if item["setting"] == setting)
    networks = {}
    for fold in range(zoo.FOLDS):
        run = cv_root / "tcn" / zoo.name(setting) / f"fold{fold}"
        if digest(run / "fit.json") != entry["runs"][str(fold)]["fit_sha256"]:
            raise ValueError(f"{run} is not the run D20 recorded")
        columns = json.loads((run / "fit.json").read_text())["columns"]
        networks[f"fold{fold}"] = ("fold", (*zoo.final_weights(run), columns))
    matching = record["slots"]["current"]["temporal_models"][0]
    fit = json.loads((matching / "fit.json").read_text())
    if (fit["seed"], fit["boundary_boost"]) != NETWORKS["temporal_models"]:
        raise ValueError(f"{matching} is not the slot the fold networks match")
    networks["sixteen"] = ("exported", temporal_cpu.load(matching, scale_trial.final_snapshot(matching)))
    halves = {name: [] for name in networks}
    for half in external_benchmark.halves(dfl, skill, "development", raw=raw):
        if half.view not in zoo.SCORED_VIEWS:
            continue
        outputs = network_signals(networks, half.positions, half.ball, half.teams, half.pitch)
        for name, signals in outputs.items():
            halves[name].append(diagnosis.HalfSignals(half.source, half.match, half.view, half.index,
                                                      np.asarray(half.labels), signals, None))
        print(f"Forecast {half.key} with {len(networks)} networks", flush=True)
    trials = {name: triple_trial.score_point(parts, recipe) for name, parts in halves.items()}
    primaries = {name: triple_trial.stage_a_primary(trial) for name, trial in trials.items()}
    folds = float(np.mean([primaries[f"fold{fold}"] for fold in range(zoo.FOLDS)]))
    gain = primaries["sixteen"] - folds
    proceed = gain >= EVIDENCE_GAIN
    write_new(output, {"protocol": PROTOCOL, "standing": record["name"], "point": point, "recipe": recipe,
                       "primaries": primaries, "fold_mean": folds, "gain": gain, "proceed": proceed,
                       "trials": {name: trial["standard"] for name, trial in trials.items()},
                       "networks": {"folds": {str(k): entry["runs"][str(k)]["fit_sha256"] for k in range(zoo.FOLDS)},
                                    "sixteen": digest(matching / "fit.json")},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print("D23 evidence: " + ", ".join(f"{name} {value:.5f}" for name, value in primaries.items())
          + f"; 16-match network {gain:+.5f} over the fold mean -> {'proceed' if proceed else 'stop'}", flush=True)
    return proceed


def evidence_passed(recipe=RECIPE):
    path = recipe / "evidence.json"
    return path.exists() and json.loads(path.read_text())["proceed"] is True


# -- prepared sequences for every labelled match ---------------------------------------------------

def compare_prepared(stored, new):
    """D23 pipeline check: every stored training half is reproduced exactly by the `all` split."""
    old_manifest = json.loads((stored / "manifest.json").read_text())
    new_manifest = json.loads((new / "manifest.json").read_text())
    if old_manifest["columns"] != new_manifest["columns"]:
        return {"columns": False}
    found = {record["file"]: record for record in new_manifest["records"]}
    result = {}
    for record in old_manifest["records"]:
        if record["file"] not in found:
            result[record["file"]] = False
            continue
        with np.load(stored / record["file"], allow_pickle=False) as a, np.load(new / record["file"], allow_pickle=False) as b:
            result[record["file"]] = (np.array_equal(a["values"], b["values"], equal_nan=True)
                                      and np.array_equal(a["targets"], b["targets"]))
    return result


def prepare_sequences(dfl, skill, output, *, raw, augment):
    """`temporal_network.prepare` over the `all` split: every labelled match with the training views.

    The loop is restated here so that `temporal_network.py` stays the code D20's cloud runs used.
    """
    from dfl_submission import augment as augmentation, external_benchmark, long_context
    from dfl_submission.temporal_network import targets
    from dfl_submission.train import SOURCE_MANIFEST, write_json
    output.mkdir(parents=True, exist_ok=False)
    halves = (augmentation.training_halves(dfl, skill, raw=raw, split="all") if augment
              else external_benchmark.halves(dfl, skill, "all", raw=raw))
    records, columns = [], None
    for half in halves:
        frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        if columns is None:
            columns = table.columns.tolist()
        elif columns != table.columns.tolist():
            raise ValueError("External feature schema differs")
        name = f"{half.key}.npz"
        np.savez_compressed(output / name, values=table.to_numpy(dtype=np.float32), targets=targets(half.labels[frames]))
        records.append({"file": name, "source": half.source, "match": half.match, "sha256": digest(output / name)})
        print(f"Prepared temporal sequence {half.key}", flush=True)
    write_json(output / "manifest.json", {"columns": columns, "records": records, "dfl_manifest": digest(SOURCE_MANIFEST),
                                           "skill_manifest": digest(skill / "manifest.json"), "split": "all"}
               | ({"augmented": True} if augment else {}))


def prepare(dfl, skill, stored_root, output_root, *, raw, record=RECIPE / "prepare_check.json"):
    """Both sets' `all` sequences, then the pipeline check against the stored training sequences."""
    require_external_only()
    if record.exists():
        raise FileExistsError(f"{record} exists; D23's prepare check is recorded once")
    if not evidence_passed():
        raise ValueError("D23's evidence did not pass; the standing recipe is not refit")
    checks = {}
    for key, augment in (("current", False), ("augmented", True)):
        folder = output_root / key
        if not (folder / "manifest.json").exists():
            if folder.exists():
                shutil.rmtree(folder)
            prepare_sequences(dfl, skill, folder, raw=raw, augment=augment)
        stored = stored_root[key]
        expected = json.loads(((PACKAGE if key == "current" else PACKAGE / "retrain_models") / "temporal_models"
                               / "fit.json").read_text())["train_manifest"]
        if digest(stored / "manifest.json") != expected:
            raise ValueError(f"{stored} is not the {key} set's stored training sequences")
        compared = compare_prepared(stored, folder)
        manifest = json.loads((folder / "manifest.json").read_text())
        matches = sorted({f"{r['source']}:{r['match']}" for r in manifest["records"]})
        checks[key] = {"halves": compared, "passed": bool(compared) and all(compared.values()),
                       "stored_manifest": expected, "manifest": digest(folder / "manifest.json"),
                       "records": len(manifest["records"]), "matches": matches}
        print(f"D23 prepare check, {key} set: {sum(compared.values())} of {len(compared)} stored halves reproduced; "
              f"{len(manifest['records'])} halves from {len(matches)} matches", flush=True)
    passed = all(check["passed"] for check in checks.values())
    write_new(record, {"protocol": PROTOCOL, "checks": checks, "passed": passed,
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    if not passed:
        raise ValueError("D23 pipeline check failed: the `all` split does not reproduce the stored sequences")
    return checks


def prepared_folder(output_root, key, record=RECIPE / "prepare_check.json"):
    checked = json.loads(record.read_text())
    if checked["passed"] is not True or digest(output_root / key / "manifest.json") != checked["checks"][key]["manifest"]:
        raise ValueError(f"{output_root / key} is not the prepared folder that passed D23's check")
    return output_root / key


# -- the refit -------------------------------------------------------------------------------------

def fit(output_root, scratch, *, models=MODELS, device="mps"):
    """Trees and networks of both sets on every labelled match, at the standing recipe."""
    require_external_only()
    from dfl_submission import touch_trial
    record = standing()
    point = record["point"]
    for key in SETS:
        source = prepared_folder(output_root, key)
        trees = models / key / "long_models"
        if not (trees / "fit.json").exists():
            if trees.exists():
                shutil.rmtree(trees)
            touch_trial.fit_trees(source, key, trees)
        for slot, (seed_, boost) in NETWORKS.items():
            for seed in slot_seeds(key, slot, record):
                folder = models / key / f"{slot}_seed{seed}"
                if (folder / "fit.json").exists():
                    continue
                raw = scratch / key / folder.name
                if not (raw / "fit.json").exists():
                    if raw.exists():
                        shutil.rmtree(raw)
                    subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "train", "--source", str(source),
                                    "--out", str(raw), "--device", device, "--seed", str(seed), "--boundary-boost",
                                    str(boost), "--width", str(point["width"]), "--steps", str(point["steps"])],
                                   check=True, cwd=ROOT)
                subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", "export", "--source", str(raw),
                                "--out", str(folder)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
                fitted = json.loads((folder / "fit.json").read_text())
                if (fitted["seed"], fitted["boundary_boost"], fitted["train_manifest"]) != (seed, boost, digest(source / "manifest.json")):
                    raise ValueError(f"{folder} was not trained as D23 prescribes")
    print(f"D23 refit complete: {identity(*candidate_folders(record, models))}", flush=True)


# -- the independent checks ------------------------------------------------------------------------

def pff_scores(parsed, games, teacher, recipe):
    from dfl_submission import diagnosis, vendor_trial
    halves = list(diagnosis.pff_halves(parsed, games, teacher, signals_of=zoo.half_signals))
    matches = diagnosis.pff_record(diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe)))
    return {"means": vendor_trial.means(matches), "matches": matches}


def metrica_scores(teacher, recipe):
    from dfl_submission import diagnosis, metrica_sample as ms
    matches = {}
    for game in ms.GAMES:
        halves = ms.load(ms.PARSED, game)
        labels = [diagnosis.decode(zoo.half_signals(teacher, *ms.tracking(half))[0], recipe) for half in halves]
        matches[str(game)] = ms.score(ms.targets(halves), labels)
        print(f"Scored Metrica game {game}", flush=True)
    return {"means": ms.means(matches), "matches": matches}


def legacy_scores(raw, matches, teacher, recipe):
    """Mean scores over the 2019–20 SkillCorner matches, as D6 scores them (ownership only)."""
    from dfl_submission import competition_adaptation, decoder_trial, diagnosis, skillcorner_2020
    per_match = {}
    for match in matches:
        tracking = skillcorner_2020.read_tracking(raw, match)
        labels = skillcorner_2020.read_ownership(raw, match)
        predictions = {"model": [diagnosis.decode(zoo.half_signals(teacher, *half)[0], recipe) for half in tracking]}
        per_match[str(match)] = competition_adaptation.score_match(labels, predictions)["model"]
        print(f"Scored SkillCorner 2020 match {match}", flush=True)
    return {"matches": per_match, "means": {
        convention: {metric: float(np.mean([match[convention][metric] for match in per_match.values()]))
                     for metric in ("spell_f1", "macro_f1")} for convention in decoder_trial.CONVENTIONS}}


def adopted(changes):
    """D23's rule: no source drops by more than 0.01 and the mean change is not negative."""
    mean_change = float(np.mean(list(changes.values())))
    return min(changes.values()) >= -CHECK_LOSS and mean_change >= 0, mean_change


def check(parsed, legacy_raw, *, models=MODELS, output=RECIPE / "check.json"):
    """The standing and the refit model, once each, on three sources neither was trained on."""
    require_external_only()
    from dfl_submission import metrica_sample as ms, skillcorner_2020, vendor_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D23's independent checks are scored once")
    record = standing()
    recipe = record["recipe"]
    trees, slots = candidate_folders(record, models)
    games = vendor_trial.split_games("untouched")
    teachers = {"standing": standing_model(record), "candidate": candidate_model(record, models)}
    if record["name"] == "D15":
        ms.reference_scores("D15")  # stops unless D15 reproduces its Metrica record from its cache
    scores = {}
    for name, teacher in teachers.items():
        scores[name] = {"pff_untouched": pff_scores(parsed, games, teacher, recipe),
                        "metrica": metrica_scores(teacher, recipe),
                        "skillcorner_2020": legacy_scores(legacy_raw, skillcorner_2020.HOLDOUT, teacher, recipe)}
        print(f"D23 scored the {name} model on the three independent sources", flush=True)
    if record["name"] == "D15":
        expected = json.loads(ms.record_path("D15").read_text())["matches"]
        if json.loads(json.dumps(scores["standing"]["metrica"]["matches"])) != expected:
            raise ValueError("The standing D15 model does not reproduce its Metrica record; no record was written")
    else:
        from dfl_submission import triple_trial
        expected = triple_trial.frozen()["metrica"]["means"]
        if json.loads(json.dumps(scores["standing"]["metrica"]["means"])) != expected:
            raise ValueError("The standing D22 model does not reproduce its Metrica means; no record was written")
    changes = {source: scores["candidate"][source]["means"]["aba3"]["spell_f1"]
               - scores["standing"][source]["means"]["aba3"]["spell_f1"] for source in scores["standing"]}
    passed, mean_change = adopted(changes)
    write_new(output, {"protocol": PROTOCOL, "standing": record["name"], "recipe": recipe, "scores": scores,
                       "changes": changes, "mean_change": mean_change, "passed": passed,
                       "identity": identity(trees, slots), "inputs": vendor_trial.inputs(parsed, games),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print("D23 check: " + ", ".join(f"{source} {value:+.4f}" for source, value in changes.items())
          + f", mean {mean_change:+.4f} -> {'adopted' if passed else 'not adopted'}", flush=True)
    return passed


def build(output, models=MODELS, recipe=RECIPE):
    """The competition file with the refit model."""
    from dfl_submission.build import tactical_halves, write_csv
    checked = json.loads((recipe / "check.json").read_text())
    if checked["passed"] is not True:
        raise ValueError("The D23 refit was not adopted; build the standing recipe instead")
    record = standing()
    if checked["identity"] != identity(*candidate_folders(record, models)) or checked["recipe"] != record["recipe"]:
        raise ValueError("The D23 models differ from the ones its checks scored")
    teacher = candidate_model(record, models)
    _, frozen = zoo.d15_setting()

    def decode_half(positions, ball, teams, pitch):
        return zoo.half_signals(teacher, positions, ball, teams, pitch)[1]

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("evidence")
    first.add_argument("--dfl", type=Path, required=True)
    first.add_argument("--dfl-raw", action="store_true")
    first.add_argument("--skill", type=Path, required=True)
    first.add_argument("--cv", type=Path, required=True)
    preparing = commands.add_parser("prepare")
    preparing.add_argument("--dfl", type=Path, required=True)
    preparing.add_argument("--dfl-raw", action="store_true")
    preparing.add_argument("--skill", type=Path, required=True)
    preparing.add_argument("--current", type=Path, required=True, help="the stored original-set training sequences")
    preparing.add_argument("--augmented", type=Path, required=True, help="the stored augmented-set training sequences")
    preparing.add_argument("--out", type=Path, required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--prepared", type=Path, required=True)
    fitting.add_argument("--scratch", type=Path, required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--parsed", type=Path, required=True)
    checking.add_argument("--legacy", type=Path, required=True, help="the 2019–20 SkillCorner release")
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "evidence":
        evidence(args.dfl, args.skill, args.cv, raw=args.dfl_raw)
    elif args.command == "prepare":
        prepare(args.dfl, args.skill, {"current": args.current, "augmented": args.augmented}, args.out, raw=args.dfl_raw)
    elif args.command == "fit":
        fit(args.prepared, args.scratch)
    else:
        check(args.parsed, args.legacy)


if __name__ == "__main__":
    main()
