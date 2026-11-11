"""D7 and D8: frozen D3 on PFF tracking, a vendor none of its models has seen.

`sanity` scores the organisers' smooth50 baseline on the PFF development games. It checks the
reader's alignment and team orientation without using our models, and must pass before `measure`.
`measure` scores frozen D3 on the same games, against the raw provider labels and every
controlled-possession convention. Neither command makes a decision; see DECISIONS.md D7.

`tune` scores D8's grid of model-set weights on the PFF development games and the DFL/SkillCorner
guard views, after checking that the D3 weight reproduces the D7 and D3 records. `confirm` scores
the selected weight once on PFF confirmation set A. See DECISIONS.md D8.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_submission import decoder_trial, pff
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm
from starter import baseline


PACKAGE = Path(__file__).resolve().parent
RECIPE = PACKAGE / "vendor_recipe"
PROTOCOL = "DECISIONS.md D7"
SANITY_RANGE = (0.15, 0.40)
BASELINE_WINDOW = 50

D8_PROTOCOL = "DECISIONS.md D8"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
D3_WEIGHT = 0.5
PRIMARY = decoder_trial.PRIMARY
OTHER_CONVENTIONS = tuple(name for name in decoder_trial.CONVENTIONS if name not in ("raw", PRIMARY))
GAIN, CONVENTION_LOSS, MATCH_LOSS = .02, .01, .03
GUARD_LOSS = {"standard": .02, "dead_carried": .01, "team_swap": .01}
D3_DEVELOPMENT = PACKAGE / "blend_recipe" / "development.json"


def split_games(slot, sources=pff.SOURCES):
    pinned = json.loads(sources.read_text())["files"]
    return pff.split({int(Path(name).name.split(".")[0]) for name in pinned})[slot]


def development(sources=pff.SOURCES):
    return split_games("development", sources)


def baseline_codes(positions, ball, teams):
    """The starter baseline's first step: the team of the visible player nearest the visible ball."""
    distance = np.square(positions - ball[:, None]).sum(axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest = distance.argmin(axis=1)
    seen = np.isfinite(distance[np.arange(len(distance)), nearest])
    codes = np.where(teams[nearest] == 0, baseline.HOME, baseline.AWAY)
    return np.where(seen, codes, baseline.NOBODY).astype(np.int8)


def baseline_prediction(half):
    codes = baseline_codes(half["positions"], half["ball"], half["teams"])
    return pm.label_codes(baseline.smooth(codes, BASELINE_WINDOW))


def score_match(labels, predictions):
    """spellF1 and macroF1 over both halves, for the raw labels and every convention."""
    truth = np.r_[labels[0], pm.X, labels[1]]
    predicted = np.r_[predictions[0], pm.X, predictions[1]]
    result = {}
    for name, seconds in decoder_trial.CONVENTIONS.items():
        target = decoder_trial.relabel(truth, seconds)
        spells = pm.spell_scores(target, predicted, pm.valid_runs(target))
        confusion = pm.scored_confusion(pm.confusion(target, predicted))
        result[name] = {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion), "spells": spells["n_pred"]}
    return result


def means(matches):
    return {name: {metric: float(np.mean([match[name][metric] for match in matches.values()]))
                   for metric in ("spell_f1", "macro_f1")}
            for name in decoder_trial.CONVENTIONS}


def sane(value):
    return SANITY_RANGE[0] <= value <= SANITY_RANGE[1]


def coverage(labels):
    labels = np.concatenate(labels)
    return {pm.LABELS[code]: float(np.mean(labels == code)) for code in (pm.H, pm.A, pm.D, pm.X)}


def inputs(parsed, games):
    manifest = json.loads((parsed / "manifest.json").read_text())
    return {"sources_sha256": digest(pff.SOURCES), "parsed": {
        name: manifest["parsed"][name] for game in games for name in (f"{game}_1.npz", f"{game}_2.npz")}}


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D7 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def record_attempt(path, attempt):
    """Append one sanity attempt; D7 keeps every attempt, failed or passed."""
    record = (json.loads(path.read_text()) if path.exists() else
              {"protocol": PROTOCOL, "check": "smooth50 raw spellF1 mean", "range": SANITY_RANGE, "attempts": []})
    record["attempts"].append(attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def gate_passed(path, expected_inputs):
    """The latest attempt passed, on exactly the parsed inputs about to be measured."""
    latest = json.loads(path.read_text())["attempts"][-1]
    return latest["passed"] is True and latest["inputs"] == expected_inputs


def sanity(parsed, output=RECIPE / "sanity.json"):
    require_external_only()
    games = development()
    matches, targets = {}, {}
    for game in games:
        halves = pff.load(parsed, game)
        labels = [half["labels"] for half in halves]
        matches[str(game)] = score_match(labels, [baseline_prediction(half) for half in halves])
        targets[str(game)] = coverage(labels)
        print(f"Scored the smooth50 baseline on PFF {game}", flush=True)
    raw = means(matches)["raw"]["spell_f1"]
    record_attempt(output, {"mean_raw_spell_f1": raw, "passed": sane(raw), "matches": matches,
                            "label_shares": targets, "reader_sha256": digest(Path(pff.__file__)),
                            "inputs": inputs(parsed, games)})
    print(f"Reader sanity gate: mean raw spellF1 {raw:.4f}, {'passed' if sane(raw) else 'FAILED'}")
    return sane(raw)


def d3_prediction(half, teacher):
    from dfl_submission import competition_adaptation
    forecast = competition_adaptation.forecast(pff.tracking(half), teacher)
    return competition_adaptation.decode(forecast, None, 0, teacher[2]["recipe"])


def measure(parsed, output=RECIPE / "d7_measurement.json", gate=RECIPE / "sanity.json"):
    require_external_only()
    from dfl_submission import competition_adaptation
    games = development()
    if not gate_passed(gate, inputs(parsed, games)):
        raise ValueError("The latest reader sanity attempt did not pass on these parsed inputs")
    teacher = competition_adaptation.reference()
    matches = {}
    for game in games:
        halves = pff.load(parsed, game)
        matches[str(game)] = score_match([half["labels"] for half in halves],
                                         [d3_prediction(half, teacher) for half in halves])
        print(f"Scored frozen D3 on PFF {game}", flush=True)
    summary = means(matches)
    write_new(output, {"protocol": PROTOCOL, "candidate": "D3", "decision": None, "means": summary,
                       "matches": matches, "teacher_sha256": digest(competition_adaptation.BLEND / "recipe.json"),
                       "inputs": inputs(parsed, games)})
    for name, values in summary.items():
        print(f"D3 on PFF development, {name}: spellF1 {values['spell_f1']:.5f}, macroF1 {values['macro_f1']:.5f}")
    return summary


def pff_changes(base, challenger):
    """D8's PFF comparison of one weight with D3, from per-match scores."""
    before, after = means(base), means(challenger)
    return {"primary": after[PRIMARY]["spell_f1"] - before[PRIMARY]["spell_f1"],
            "conventions": {name: after[name]["spell_f1"] - before[name]["spell_f1"] for name in OTHER_CONVENTIONS},
            "worst_match": min(challenger[game][PRIMARY]["spell_f1"] - base[game][PRIMARY]["spell_f1"]
                               for game in base)}


def pff_passed(changes):
    return (changes["primary"] >= GAIN and changes["worst_match"] >= -MATCH_LOSS
            and all(value >= -CONVENTION_LOSS for value in changes["conventions"].values()))


def guard_changes(base, challenger):
    """D8's DFL/SkillCorner comparison of one weight with D3, against the N = 3 targets."""
    from dfl_submission import retrain_trial
    return {part: retrain_trial.primary(challenger, part) - retrain_trial.primary(base, part) for part in GUARD_LOSS}


def guard_passed(changes):
    return all(changes[part] >= -limit for part, limit in GUARD_LOSS.items())


def select(entries):
    """The eligible weight with the highest PFF primary, ties broken by macroF1; None if none is eligible."""
    eligible = [entry for entry in entries if entry["eligible"]]
    if not eligible:
        return None
    return max(eligible, key=lambda entry: (entry["pff"]["means"][PRIMARY]["spell_f1"],
                                            entry["pff"]["means"][PRIMARY]["macro_f1"]))["weight"]


def canonical(value):
    return json.loads(json.dumps(value))


def reproduction(pff_matches, guard_trial, d7=RECIPE / "d7_measurement.json", d3=D3_DEVELOPMENT):
    """Whether D8's run at the D3 weight reproduces the D7 measurement and the D3 development record exactly."""
    return {"d7": canonical(pff_matches) == json.loads(d7.read_text())["matches"],
            "d3": canonical(guard_trial) == json.loads(d3.read_text())["selected"]}


def model_outputs(teacher, positions, ball, teams, pitch):
    """Outputs of the current and the retrained model sets for one half, before D3's blend."""
    from dfl_submission import ensemble_trial
    _, packages, _ = teacher
    return tuple(ensemble_trial.forecasts(*packages[name], positions, ball, teams, pitch)
                 for name in ("current", "candidate"))


def blend_decode(outputs, weight, teacher):
    """One half's labels at one weight, mixed and decoded exactly as `retrain_trial.build` does."""
    from dfl_submission import motion_trial, retrain_trial
    reference, _, frozen = teacher
    return motion_trial.decode(retrain_trial.mix(*outputs, weight), reference[2]["recipe"], frozen["recipe"])


def pff_trials(parsed, games, teacher, weights):
    """Every game's scores at every weight; the model sets run once per half."""
    labels, predicted = {}, {weight: {} for weight in weights}
    for game in games:
        halves = pff.load(parsed, game)
        labels[str(game)] = [half["labels"] for half in halves]
        outputs = [model_outputs(teacher, *pff.tracking(half)) for half in halves]
        for weight in weights:
            predicted[weight][str(game)] = [blend_decode(output, weight, teacher) for output in outputs]
        print(f"Scored PFF {game} at every weight", flush=True)
    return {weight: {game: score_match(labels[game], predicted[weight][game]) for game in labels} for weight in weights}


def guard_trials(dfl, skill, teacher, weights, *, raw):
    """DFL/SkillCorner development scores at every weight, with the D2 guard views."""
    from collections import defaultdict
    from dfl_submission import augment, retrain_trial
    from dfl_submission.temporal_trial import Match
    reference, _, frozen = teacher
    records = defaultdict(list)
    for half in augment.guard_halves(dfl, skill, "development", raw=raw):
        outputs = model_outputs(teacher, half.positions, half.ball, half.teams, half.pitch)
        records[half.source, half.match, half.view].append((half.labels, outputs))
        print(f"Forecast {half.key}", flush=True)
    trials = {}
    for weight in weights:
        matches = {}
        for key, halves in records.items():
            truth = np.r_[halves[0][0], pm.X, halves[1][0]]
            matches[key] = Match(truth, pm.valid_runs(truth), [retrain_trial.mix(*outputs, weight) for _, outputs in halves])
        trials[weight] = retrain_trial.evaluate(matches, reference[2]["recipe"], frozen["recipe"])
    return trials


def tune(parsed, dfl, skill, *, raw=False, output=RECIPE / "d8_development.json", gate=RECIPE / "sanity.json"):
    require_external_only()
    from dfl_submission import competition_adaptation
    if output.exists():
        raise FileExistsError(f"{output} exists; D8 development is scored once")
    games = development()
    if not gate_passed(gate, inputs(parsed, games)):
        raise ValueError("The latest reader sanity attempt did not pass on these parsed inputs")
    teacher = competition_adaptation.reference()
    pff_scores = pff_trials(parsed, games, teacher, WEIGHTS)
    guards = guard_trials(dfl, skill, teacher, WEIGHTS, raw=raw)
    checks = reproduction(pff_scores[D3_WEIGHT], guards[D3_WEIGHT])
    if not all(checks.values()):
        raise ValueError(f"D8 pipeline check failed {checks}; no record was written")
    entries = []
    for weight in WEIGHTS:
        changes = {"pff": pff_changes(pff_scores[D3_WEIGHT], pff_scores[weight]),
                   "guards": guard_changes(guards[D3_WEIGHT], guards[weight])}
        eligible = weight != D3_WEIGHT and pff_passed(changes["pff"]) and guard_passed(changes["guards"])
        entries.append({"weight": weight, "eligible": eligible, "changes": changes, "guards": guards[weight],
                        "pff": {"means": means(pff_scores[weight]), "matches": pff_scores[weight]}})
        print(f"D8 weight {weight}: PFF N = 3 spellF1 {means(pff_scores[weight])[PRIMARY]['spell_f1']:.5f} "
              f"({changes['pff']['primary']:+.5f}), worst match {changes['pff']['worst_match']:+.4f}, "
              + ", ".join(f"{name} {value:+.4f}" for name, value in changes["pff"]["conventions"].items())
              + "; guards " + ", ".join(f"{name} {value:+.4f}" for name, value in changes["guards"].items())
              + f" -> {'eligible' if eligible else 'not eligible'}", flush=True)
    selected = select(entries)
    write_new(output, {"protocol": D8_PROTOCOL, "weights": list(WEIGHTS), "reference_weight": D3_WEIGHT,
                       "reproduction": checks, "entries": entries,
                       "selected": None if selected is None else {"weight": selected},
                       "teacher_sha256": digest(competition_adaptation.BLEND / "recipe.json"),
                       "inputs": inputs(parsed, games)})
    print(f"D8 development selected {'no weight; D3 stays' if selected is None else f'weight {selected}'}")
    return selected


def confirm(parsed, output=RECIPE / "d8_confirmation.json", development_record=RECIPE / "d8_development.json"):
    require_external_only()
    record = json.loads(development_record.read_text())
    if record["selected"] is None:
        raise ValueError("No weight was eligible on development; D3 stays and nothing is confirmed")
    if output.exists():
        raise FileExistsError(f"{output} exists; D8 confirmation is scored once")
    from dfl_submission import competition_adaptation
    if digest(competition_adaptation.BLEND / "recipe.json") != record["teacher_sha256"]:
        raise ValueError("The D3 recipe differs from the one D8 development used")
    weight = record["selected"]["weight"]
    games = split_games("confirm_a")
    teacher = competition_adaptation.reference()
    scores = pff_trials(parsed, games, teacher, (D3_WEIGHT, weight))
    changes = pff_changes(scores[D3_WEIGHT], scores[weight])
    passed = pff_passed(changes)
    write_new(output, {"protocol": D8_PROTOCOL, "weight": weight, "reference_weight": D3_WEIGHT, "changes": changes,
                       "passed": passed, "reference": {"means": means(scores[D3_WEIGHT]), "matches": scores[D3_WEIGHT]},
                       "selected": {"means": means(scores[weight]), "matches": scores[weight]},
                       "teacher_sha256": record["teacher_sha256"], "inputs": inputs(parsed, games)})
    print(f"D8 confirmation, weight {weight}: PFF N = 3 {changes['primary']:+.5f}, worst match "
          f"{changes['worst_match']:+.4f}, " + ", ".join(f"{k} {v:+.4f}" for k, v in changes["conventions"].items())
          + f" -> {'passed' if passed else 'failed'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("sanity", "measure", "tune", "confirm"))
    parser.add_argument("--parsed", type=Path, required=True, help="folder written by `dfl_submission.pff prepare`")
    parser.add_argument("--dfl", type=Path, help="DFL XML folder (tune only)")
    parser.add_argument("--dfl-raw", action="store_true", help="read the raw DFL position files (tune only)")
    parser.add_argument("--skill", type=Path, help="prepared SkillCorner folder (tune only)")
    args = parser.parse_args()
    if args.command == "tune":
        if args.dfl is None or args.skill is None:
            parser.error("tune needs --dfl and --skill")
        tune(args.parsed, args.dfl, args.skill, raw=args.dfl_raw)
    else:
        {"sanity": sanity, "measure": measure, "confirm": confirm}[args.command](args.parsed)


if __name__ == "__main__":
    main()
