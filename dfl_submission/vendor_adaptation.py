"""D32: label-free per-match adaptation of D15, proved on PFF (DECISIONS.md D32).

For one match, D15 forecasts both halves; its own confident outputs become pseudo-labels for the
original set's ownership and stoppage trees, each continued for 40 rounds with replay from its
training rows; the students' change from the original trees, times a multiplier, is added to D15's
final ownership and stoppage probabilities. `replay` builds the stoppage replay (D6's ownership
replay is reused). `develop` adapts and scores the eight PFF development matches once, `confirm`
PFF set C once, `determinism` writes one match's adapted labels for the cross-platform check, and
`build` applies the confirmed step to the competition match only with written organiser approval.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
import pandas as pd

from dfl_submission.external_only import require_external_only


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "vendor_adaptation_recipe"
DATA = ROOT / "data" / "d32"
OWNER_REPLAY = ROOT / "data" / "d6_replay.npz"
DEAD_REPLAY = DATA / "stoppage_replay.npz"
PROTOCOL = "DECISIONS.md D32"
CONFIDENCE, ROUNDS, REPLAY_WEIGHT, REPLAY_STRIDE = .90, 40, .75, 25
MULTIPLIERS = (.25, .5, 1.0)
GAIN, MATCH_LOSS, LOSS = .01, .02, .01
PRIMARY, OTHERS = "aba3", ("aba1.75", "aba4", "aba4.5")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D32 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def digest(path):
    from dfl_submission.pff import digest as file_digest
    return file_digest(path)


# -- pieces ----------------------------------------------------------------------------------------

def student_weights(replay_weights, pseudo):
    """Replay keeps 75% of the weight as its own shares; each pseudo-label class gets half of the rest."""
    counts = np.bincount(pseudo, minlength=2)
    total = len(replay_weights) + len(pseudo)
    return np.r_[replay_weights * (REPLAY_WEIGHT * total / len(replay_weights)),
                 (1 - REPLAY_WEIGHT) * total / (2 * counts[pseudo])]


def continue_tree(checkpoint, inputs, labels, weights, params, rounds=ROUNDS):
    from xgboost import XGBClassifier
    student = XGBClassifier(**dict(params, n_estimators=rounds))
    student.fit(inputs, labels, sample_weight=weights, xgb_model=checkpoint)
    return student


def adjusted(signals, residual_home, residual_dead, multiplier):
    """D15's four decoder inputs with the students' changes added to ownership and stoppage."""
    home, dead, dead_edge, owner_edge = signals
    if residual_home is not None and multiplier:
        home = np.clip(home + multiplier * residual_home, 0, 1).astype(np.asarray(signals[0]).dtype)
    if residual_dead is not None and multiplier:
        dead = np.clip(dead + multiplier * residual_dead, 0, 1).astype(np.asarray(signals[1]).dtype)
    return home, dead, dead_edge, owner_edge


def eligible(changes):
    return bool(changes["primary"] >= GAIN and changes["worst_match"] >= -MATCH_LOSS
                and all(value >= -LOSS for value in changes["conventions"].values()))


def select(entries):
    """The eligible multiplier with the highest mean N = 3, then macroF1, then grid order; None if none."""
    order = list(MULTIPLIERS)
    candidates = [m for m, entry in entries.items() if entry["eligible"]]
    if not candidates:
        return None
    return max(candidates, key=lambda m: (entries[m]["primary"], entries[m]["macro"], -order.index(m)))


# -- one match -------------------------------------------------------------------------------------

@dataclass
class Half:
    frames: np.ndarray
    owner: pd.DataFrame
    dead: pd.DataFrame
    tree_home: np.ndarray
    tree_dead: np.ndarray
    signals: tuple
    labels: np.ndarray


def teacher():
    from dfl_submission import metrica_sample as ms
    return ms.reference_model("D15")


def forest(model):
    """The original set's (ownership tree, stoppage tree), as D15 runs them."""
    return model[1]["current"][0][0]


def forecast_half(model, tracking):
    from dfl_submission import diagnosis, long_context, models
    signals, labels = diagnosis.half_signals(model, *tracking)
    frames, owner, dead = long_context.features(*tracking)
    trees = forest(model)
    return Half(frames, owner, dead, models.ownership_probability(trees[0], owner),
                trees[1].predict_proba(dead)[:, 1], tuple(signals), np.asarray(labels))


def load_replay(path):
    with np.load(path, allow_pickle=False) as replay:
        return pd.DataFrame(replay["inputs"], columns=replay["columns"]), replay["labels"], replay["weights"]


def fit_match(halves, replays, trees):
    """{head: student or None} and pseudo-label counts per head for one match."""
    from dfl_submission import adaptation_trial, long_context
    from dfl_transfer_features import swap
    rows = {"ownership": ([], []), "stoppage": ([], [])}
    for half in halves:
        home, dead = half.signals[0][half.frames], half.signals[1][half.frames]
        (owned, owner_targets), (known, dead_targets) = adaptation_trial.pseudo_labels(home, dead, CONFIDENCE)
        rows["ownership"][0].append(half.owner.loc[owned])
        rows["ownership"][1].append(owner_targets)
        rows["stoppage"][0].append(half.dead.loc[known])
        rows["stoppage"][1].append(dead_targets)
    students, counts = {}, {}
    for head, checkpoint, mirror in (("ownership", trees[0], True), ("stoppage", trees[1], False)):
        frames, targets = rows[head]
        pseudo = np.concatenate(targets).astype(int)
        counts[head] = np.bincount(pseudo, minlength=2).tolist()
        if not all(counts[head]):
            students[head] = None
            continue
        source, truth, source_weights = replays[head]
        table = pd.concat(frames, ignore_index=True)
        if not source.columns.equals(table.columns):
            raise ValueError(f"The {head} replay and the match's feature columns differ")
        weights = student_weights(source_weights, pseudo)
        inputs, labels = pd.concat([source, table], ignore_index=True), np.r_[truth, pseudo]
        if mirror:
            inputs, labels, weights = pd.concat([inputs, swap(inputs)], ignore_index=True), np.r_[labels, 1 - labels], np.tile(weights, 2)
        students[head] = continue_tree(checkpoint, inputs, labels, weights, long_context.TREE_PARAMS)
    return students, counts


def residuals(half, students):
    from dfl_submission import models
    timeline = np.arange(len(half.signals[0]))
    result = []
    for head, original, probability in (
            ("ownership", half.tree_home, lambda s: models.ownership_probability(s, half.owner)),
            ("stoppage", half.tree_dead, lambda s: s.predict_proba(half.dead)[:, 1])):
        student = students[head]
        result.append(None if student is None else
                      np.interp(timeline, half.frames, probability(student) - original).astype(np.float32))
    return result


def predict_match(trackings, model, replays, multipliers):
    """{multiplier: [labels per half]} (0 is D15 itself) and the pseudo-label counts."""
    from dfl_submission import diagnosis
    recipe = model[2]["recipe"]
    halves = [forecast_half(model, tracking) for tracking in trackings]
    students, counts = fit_match(halves, replays, forest(model))
    predictions = {m: [] for m in multipliers}
    for half in halves:
        home_change, dead_change = residuals(half, students)
        for m in multipliers:
            labels = np.asarray(diagnosis.decode(adjusted(half.signals, home_change, dead_change, m), recipe))
            if m == 0 and not np.array_equal(labels, half.labels):
                raise ValueError("Multiplier 0 does not reproduce D15's labels")
            predictions[m].append(labels)
    return predictions, counts


# -- replay ----------------------------------------------------------------------------------------

def replay(dfl, skill, output=DEAD_REPLAY):
    """The stoppage replay: the original set's training halves at frames divisible by 25 with a label."""
    require_external_only()
    from dfl_submission import external_benchmark, long_context, train
    from evaluation import possession_metrics as pm
    if output.exists():
        raise FileExistsError(output)
    sources = json.loads((PACKAGE / "long_models" / "fit.json").read_text())
    for path, key in ((train.SOURCE_MANIFEST, "dfl_manifest_sha256"), (skill / "manifest.json", "skill_manifest_sha256")):
        if digest(path) != sources[key]:
            raise ValueError("Replay inputs must match the original long-context fit")
    rows, targets, groups = [], [], []
    for half in external_benchmark.halves(dfl, skill, "train", raw=True):
        frames, _, dead = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        labels = half.labels[frames]
        use = (frames % REPLAY_STRIDE == 0) & (labels != pm.X)
        rows.append(dead.loc[use].copy())
        targets.append(labels[use] == pm.D)
        groups.append(np.full(use.sum(), f"{half.source}:{half.match}"))
        print(f"Prepared stoppage replay {half.key}", flush=True)
    inputs, labels = pd.concat(rows, ignore_index=True), np.concatenate(targets).astype(int)
    weights = long_context.source_weights(labels, np.concatenate(groups))
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".npz") as temporary:
        np.savez_compressed(temporary, inputs=inputs.to_numpy(dtype=np.float32),
                            columns=inputs.columns.to_numpy(dtype=str), labels=labels, weights=weights)
        temporary.flush()
        output.hardlink_to(temporary.name)
    print(f"Prepared {len(labels):,} stoppage replay rows", flush=True)


def replays():
    from dfl_submission import competition_adaptation
    stored = json.loads((competition_adaptation.PACKAGE / "competition_adaptation_recipe" / "development.json").read_text())
    if digest(OWNER_REPLAY) != stored["identity"]["replay_sha256"]:
        raise ValueError(f"{OWNER_REPLAY} is not D6's ownership replay")
    return {"ownership": load_replay(OWNER_REPLAY), "stoppage": load_replay(DEAD_REPLAY)}


def identity():
    return {"protocol": PROTOCOL, "owner_replay_sha256": digest(OWNER_REPLAY), "dead_replay_sha256": digest(DEAD_REPLAY),
            "confidence": CONFIDENCE, "rounds": ROUNDS, "replay_weight": REPLAY_WEIGHT, "multipliers": list(MULTIPLIERS)}


# -- development and confirmation ------------------------------------------------------------------

def match_scores(parsed, games, model, rows, multipliers, cache):
    """{game: {multiplier: scores}} and pseudo-label counts, one adaptation per game, cached per game."""
    from dfl_submission import pff, vendor_trial
    scores, counts = {}, {}
    for game in games:
        path = cache / f"{game}.json"
        if path.exists():
            stored = json.loads(path.read_text())
            if stored["identity"] == identity() and stored["multipliers"] == [str(m) for m in multipliers]:
                scores[str(game)], counts[str(game)] = stored["scores"], stored["counts"]
                continue
        halves = pff.load(parsed, game)
        started = time.time()
        predictions, found = predict_match([pff.tracking(half) for half in halves], model, rows, multipliers)
        labels = [half["labels"] for half in halves]
        scores[str(game)] = {str(m): vendor_trial.score_match(labels, predictions[m]) for m in multipliers}
        counts[str(game)] = found
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"identity": identity(), "multipliers": [str(m) for m in multipliers],
                                    "scores": scores[str(game)], "counts": found}, sort_keys=True))
        print(f"D32 adapted PFF {game} in {time.time() - started:.0f} s: pseudo-labels {found}, N = 3 "
              + ", ".join(f"m={m} {scores[str(game)][str(m)][PRIMARY]['spell_f1']:.4f}" for m in multipliers), flush=True)
    return scores, counts


def changes(scores, m):
    games = sorted(scores)
    per = {g: scores[g][str(m)][PRIMARY]["spell_f1"] - scores[g]["0"][PRIMARY]["spell_f1"] for g in games}
    mean = lambda name, metric="spell_f1": float(np.mean([scores[g][str(m)][name][metric] for g in games]))
    base = lambda name, metric="spell_f1": float(np.mean([scores[g]["0"][name][metric] for g in games]))
    return {"primary": mean(PRIMARY) - base(PRIMARY), "per_match": per, "worst_match": min(per.values()),
            "conventions": {name: mean(name) - base(name) for name in OTHERS}}, mean(PRIMARY), mean(PRIMARY, "macro_f1")


def develop(parsed, output=RECIPE / "development.json"):
    require_external_only()
    from dfl_submission import scale_trial, vendor_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D32 development is scored once")
    games = vendor_trial.development()
    record = json.loads((scale_trial.RECIPE / "development.json").read_text())
    expected = next(e["trial"] for e in record["entries"] if e["setting"] == record["selected"])["pff"]["matches"]
    multipliers = (0, *MULTIPLIERS)
    scores, counts = match_scores(parsed, games, teacher(), replays(), multipliers, DATA / "develop")
    if vendor_trial.canonical({g: s["0"] for g, s in scores.items()}) != expected:
        raise ValueError("D15 (multiplier 0) does not reproduce its PFF development record; no record was written")
    entries = {}
    for m in MULTIPLIERS:
        change, primary, macro = changes(scores, m)
        entries[m] = {"changes": change, "eligible": eligible(change), "primary": primary, "macro": macro}
        print(f"D32 development m = {m}: N = 3 {change['primary']:+.5f}, worst match {change['worst_match']:+.4f}, "
              + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items())
              + f" -> {'eligible' if entries[m]['eligible'] else 'not eligible'}", flush=True)
    selected = select(entries)
    write_new(output, {"protocol": PROTOCOL, "identity": identity(), "scores": scores, "counts": counts,
                       "entries": {str(m): e for m, e in entries.items()}, "selected": selected,
                       "reproduction": {"d15_pff_development": True}, "inputs": vendor_trial.inputs(parsed, games),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D32 development selected {'m = ' + str(selected) if selected is not None else 'nothing; D32 stops'}", flush=True)
    return selected


def selected_multiplier(path=RECIPE / "development.json"):
    record = json.loads(path.read_text())
    if record["selected"] is None:
        raise ValueError("No D32 multiplier was eligible on development; D32 stopped")
    if record["identity"] != identity():
        raise ValueError("The D32 replays differ from the ones development used")
    return record["selected"]


def confirm(parsed, output=RECIPE / "confirmation.json"):
    require_external_only()
    from dfl_submission import vendor_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D32 confirmation is scored once")
    m = selected_multiplier()
    games = vendor_trial.split_games("confirm_c")
    scores, counts = match_scores(parsed, games, teacher(), replays(), (0, m), DATA / "confirm")
    change, primary, _ = changes(scores, m)
    passed = bool(change["primary"] >= GAIN)
    write_new(output, {"protocol": PROTOCOL, "identity": identity(), "multiplier": m, "scores": scores, "counts": counts,
                       "changes": change, "passed": passed, "inputs": vendor_trial.inputs(parsed, games),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D32 confirmation (PFF set C) m = {m}: N = 3 {change['primary']:+.5f} -> {'passed' if passed else 'failed'}",
          flush=True)
    return passed


def determinism(parsed, output):
    """The adapted labels of the first PFF development match at the selected multiplier, and their SHA-256."""
    require_external_only()
    from dfl_submission import pff, vendor_trial
    m = selected_multiplier()
    game = vendor_trial.development()[0]
    halves = pff.load(parsed, game)
    predictions, _ = predict_match([pff.tracking(half) for half in halves], teacher(), replays(), (m,))
    data = np.concatenate([np.asarray(part, dtype=np.int8) for part in predictions[m]]).tobytes()
    Path(output).write_bytes(data)
    print(f"D32 determinism PFF {game} m = {m}: {hashlib.sha256(data).hexdigest()}", flush=True)


def competition_labels(model, pitch, merge_ball=None, approval=RECIPE / "organiser_approval.json"):
    """The competition match adapted at the confirmed multiplier; only with recorded written approval."""
    from dfl_submission.build import tactical_labels, tactical_tracking_halves
    if not approval.exists() or json.loads(approval.read_text()).get("approved") is not True:
        raise ValueError("D32 needs the organisers' written yes, recorded in " + str(approval))
    if json.loads((RECIPE / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D32 did not pass confirmation")
    m = selected_multiplier()
    halves = list(tactical_tracking_halves(centered=True, pitch=pitch, merge_ball=merge_ball))
    predictions, _ = predict_match([tracking for _, _, tracking in halves], model, replays(), (m,))
    return tactical_labels((span, codes) for (_, span, _), codes in zip(halves, predictions[m], strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    replaying = commands.add_parser("replay")
    replaying.add_argument("--dfl", type=Path, required=True)
    replaying.add_argument("--skill", type=Path, required=True)
    for name in ("develop", "confirm", "determinism"):
        command = commands.add_parser(name)
        command.add_argument("--parsed", type=Path, required=True)
        if name == "determinism":
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require_external_only()
    if args.command == "replay":
        replay(args.dfl, args.skill)
    elif args.command == "develop":
        develop(args.parsed)
    elif args.command == "confirm":
        confirm(args.parsed)
    else:
        determinism(args.parsed, args.out)


if __name__ == "__main__":
    main()
