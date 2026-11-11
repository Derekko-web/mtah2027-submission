"""D7: measure the frozen D3 submission on PFF tracking, a vendor none of its models has seen.

`sanity` scores the organisers' smooth50 baseline on the PFF development games. It checks the
reader's alignment and team orientation without using our models, and must pass before `measure`.
`measure` scores frozen D3 on the same games, against the raw provider labels and every
controlled-possession convention. Neither command makes a decision; see DECISIONS.md D7.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_submission import decoder_trial, pff
from dfl_submission.external_only import require_external_only
from dfl_transfer_data import digest
from evaluation import possession_metrics as pm
from starter import baseline


PACKAGE = Path(__file__).resolve().parent
RECIPE = PACKAGE / "vendor_recipe"
PROTOCOL = "DECISIONS.md D7"
SANITY_RANGE = (0.15, 0.40)
BASELINE_WINDOW = 50


def development(sources=pff.SOURCES):
    pinned = json.loads(sources.read_text())["files"]
    return pff.split({int(Path(name).name.split(".")[0]) for name in pinned})["development"]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("sanity", "measure"))
    parser.add_argument("--parsed", type=Path, required=True, help="folder written by `dfl_submission.pff prepare`")
    args = parser.parse_args()
    (sanity if args.command == "sanity" else measure)(args.parsed)


if __name__ == "__main__":
    main()
