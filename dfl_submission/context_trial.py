"""Tune on external development matches, confirm against the verified fallback, then build."""

import argparse
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np

from dfl_submission import external_benchmark as benchmark, long_context
from dfl_submission.build import tactical_halves, write_csv
from dfl_submission.decoder import decode
from dfl_submission.train import write_json
from dfl_transfer_data import digest
from evaluation import possession_metrics as pm


def blend(old, new, weight):
    if weight == 0:
        return old
    if weight == 1:
        return new
    return (1 - weight) * old + weight * new


def suppress_islands(labels, home, dead, max_length, confidence):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    states = labels[edges[:-1]]
    selected = (states[:-2] == states[2:]) & (np.diff(edges)[1:-1] <= max_length)
    revised = labels.copy()
    for j in np.flatnonzero(selected) + 1:
        start, stop = edges[j:j + 2]
        state = states[j]
        evidence = (home if state == pm.H else 1 - home if state == pm.A else dead)[start:stop]
        if evidence.max() < confidence:
            revised[start:stop] = states[j - 1]
    return revised


def decode_recipe(probabilities, recipe):
    old_home, old_dead, new_home, new_dead = probabilities
    home = blend(old_home, new_home, recipe["owner_mix"])
    dead = blend(old_dead, new_dead, recipe["dead_mix"])
    labels = decode(home, dead, dead_cutoff=recipe["dead_cutoff"], dead_radius=recipe["dead_radius"],
                    owner_radius=recipe["owner_radius"])
    if recipe["island_frames"]:
        labels = suppress_islands(labels, home, dead, recipe["island_frames"], recipe["island_confidence"])
    return labels


def baseline_recipe(reference):
    return {key: reference["recipe"][key] for key in ("dead_cutoff", "dead_radius", "owner_radius")} | {
        "owner_mix": 0, "dead_mix": 0, "island_frames": 0, "island_confidence": .8,
    }


def predictions(dfl, skill, split, cache, model_folder, *, raw=False):
    checkpoint, fitted = long_context.load(model_folder)
    reference = json.loads((cache / "manifest.json").read_text())
    if reference["split"] != split or reference["skill_manifest"] != digest(skill / "manifest.json"):
        raise ValueError("External reference cache split or source differs")
    matches = defaultdict(list)
    for half in benchmark.halves(dfl, skill, split, raw=raw):
        with np.load(cache / f"{half.key}.npz", allow_pickle=False) as source:
            if not np.array_equal(half.labels, source["labels"]):
                raise ValueError(f"External reference truth differs for {half.key}")
            old_home, old_dead = source["home"], source["dead"]
        home, dead = long_context.predict(checkpoint, half.positions, half.ball, half.teams, half.pitch)
        matches[half.source, half.match, half.view].append((half.labels, (old_home, old_dead, home, dead)))
        print(f"Context predictions {half.key}", flush=True)
    return matches, reference["reference"], fitted


def score(matches, recipe):
    records = {}
    for key, halves in matches.items():
        truth = np.r_[halves[0][0], pm.X, halves[1][0]]
        result = np.r_[decode_recipe(halves[0][1], recipe), pm.X, decode_recipe(halves[1][1], recipe)]
        spelling = pm.spell_scores(truth, result, pm.valid_runs(truth))
        confusion = pm.scored_confusion(pm.confusion(truth, result))
        records[key] = {"spell_f1": spelling["mean_f1"], "macro_f1": pm.macro_f1(confusion),
                        "spells": spelling["n_pred"]}
    return records


def aggregate(records, metric="spell_f1"):
    dfl = {match for source, match, _ in records if source == "dfl"}
    dfl_score = float(np.mean([(records["dfl", match, "clean"][metric]
                               + 2 * records["dfl", match, "masked"][metric]) / 3 for match in sorted(dfl)]))
    skill = float(np.mean([value[metric] for (source, _, _), value in records.items() if source == "skillcorner"]))
    return {"dfl": dfl_score, "skillcorner": skill, "combined": (dfl_score + skill) / 2}


def candidate(matches, recipe):
    result = score(matches, recipe)
    return {"recipe": recipe, "spell_f1": aggregate(result), "macro_f1": aggregate(result, "macro_f1")}


def rank(trial):
    return trial["spell_f1"]["combined"], trial["macro_f1"]["combined"]


def tune(dfl, skill, cache, model, output, *, raw=False):
    output.mkdir(parents=True, exist_ok=False)
    matches, reference, fitted = predictions(dfl, skill, "development", cache, model, raw=raw)
    base = candidate(matches, baseline_recipe(reference))
    trials = []
    for owner, dead, threshold, dead_radius, owner_radius in product(
        (0, .35, .7, 1), (0, .35, .7, 1), (.43, .5, .57), (25, 50, 70), (25, 50, 70),
    ):
        recipe = dict(owner_mix=owner, dead_mix=dead, dead_cutoff=threshold, dead_radius=dead_radius,
                      owner_radius=owner_radius, island_frames=0, island_confidence=.8)
        trials.append(candidate(matches, recipe))
    shortlist = sorted(trials, key=rank, reverse=True)[:8]
    for trial, frames, confidence in product(shortlist, (12, 25, 40), (.68, .82)):
        recipe = dict(trial["recipe"], island_frames=frames, island_confidence=confidence)
        trials.append(candidate(matches, recipe))
    allowed = [trial for trial in trials if all(trial["spell_f1"][source] >= base["spell_f1"][source] - .02
                                               for source in ("dfl", "skillcorner"))]
    selected = max(allowed, key=rank)
    improved = selected["spell_f1"]["combined"] >= base["spell_f1"]["combined"] + .003
    write_json(output / "development.json", {"reference": base, "selected": selected, "trials": trials, "passed": improved})
    write_json(output / "recipe.json", {"reference": reference, "long_context": fitted["models"], "recipe": selected["recipe"]})
    print(f"Frozen candidate development comparison {base['spell_f1']} -> {selected}", flush=True)


def serialize(records):
    return {"/".join(key): value for key, value in records.items()}


def confirm(dfl, skill, cache, model, folder, fallback, *, raw=False):
    development = json.loads((folder / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("Candidate did not pass external development")
    frozen = json.loads((folder / "recipe.json").read_text())
    matches, reference, fitted = predictions(dfl, skill, "holdout", cache, model, raw=raw)
    if frozen["reference"] != reference or frozen["long_context"] != fitted["models"]:
        raise ValueError("Holdout inference did not use the frozen model files")
    base, result = score(matches, baseline_recipe(reference)), score(matches, frozen["recipe"])
    base_score, result_score = aggregate(base), aggregate(result)
    expected = json.loads((fallback / "validation/selection.json").read_text())["candidates"]["dfl_skillcorner_fusion"]
    if any(abs(base_score[source] - expected[f"{source}_spell_f1"]) > 1e-10
           for source in ("dfl", "skillcorner", "combined")):
        raise ValueError(f"Fallback scores did not reproduce: {base_score}")
    gain = {source: result_score[source] - base_score[source] for source in base_score}
    passed = gain["combined"] >= .005 and min(gain["dfl"], gain["skillcorner"]) >= -.02
    report = {"reference": base_score, "selected": result_score, "gain": gain, "passed": passed,
              "reference_matches": serialize(base), "selected_matches": serialize(result),
              "gate": "Combined gain at least .005, neither dataset loses more than .02"}
    write_json(folder / "external_validation.json", report)
    print(json.dumps({"passed": passed, "reference": base_score, "selected": result_score, "gain": gain}), flush=True)


def build(fallback, model_folder, folder, output):
    report = json.loads((folder / "external_validation.json").read_text())
    if report["passed"] is not True:
        raise ValueError("Candidate failed external validation")
    frozen = json.loads((folder / "recipe.json").read_text())
    checkpoints, fitted = long_context.load(model_folder)
    if frozen["long_context"] != fitted["models"]:
        raise ValueError("Inference checkpoint identity differs")
    recipe = frozen["recipe"]
    standalone = recipe["owner_mix"] == recipe["dead_mix"] == 1
    if not standalone:
        if fallback is None:
            raise ValueError("This blend requires --fallback")
        reference, reference_models = benchmark.reference(fallback)
        if frozen["reference"] != reference:
            raise ValueError("Fallback checkpoint identity differs")

    def decode_half(positions, ball, teams, pitch):
        challenger = long_context.predict(checkpoints, positions, ball, teams, pitch)
        original = challenger if standalone else benchmark.reference_probabilities(
            reference_models, reference["recipe"], positions, ball, teams, pitch)
        return decode_recipe((*original, *challenger), recipe)

    write_csv(output, tactical_halves(decode_half, centered=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("tune", "confirm", "build"):
        command = commands.add_parser(action)
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--trial", required=True, type=Path)
        if action != "tune":
            command.add_argument("--fallback", required=action == "confirm", type=Path)
        if action != "build":
            command.add_argument("--dfl", required=True, type=Path)
            command.add_argument("--dfl-raw", action="store_true")
            command.add_argument("--skill", required=True, type=Path)
            command.add_argument("--cache", required=True, type=Path)
        else:
            command.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    if args.action != "build":
        require_external_only()
    if args.action == "build":
        build(args.fallback, args.model, args.trial, args.out)
    elif args.action == "tune":
        tune(args.dfl, args.skill, args.cache, args.model, args.trial, raw=args.dfl_raw)
    else:
        confirm(args.dfl, args.skill, args.cache, args.model, args.trial, args.fallback, raw=args.dfl_raw)
