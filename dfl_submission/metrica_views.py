"""D24: the Metrica sample matches under competition-like tracking (DECISIONS.md D24).

D19 scores the three sample games on corrected tracking, where the ball is missing exactly when it
is dead. Three degraded views of the same games move them towards the competition's automatic
tracking:
- `masked`: the DFL masked degradation (`dfl_transfer_features.corrupt`), unchanged;
- `handled`: D2's dead-ball handling (`augment.handle_dead_ball`), then the same masking;
- `team_swap`: the handled view with D2's guard swaps (`augment.swap_teams`).

`measure` scores frozen D15 on every view against D19's targets, once, after D15 and D3 reproduce
their D19 records, and writes `metrica_recipe/d24_measurement.json`. It selects nothing. A
candidate's D24 mean (`d24_mean`) is the gate quantity D25 and D26 use.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from dfl_submission import augment, diagnosis, metrica_sample as ms
from dfl_submission.external_only import require_external_only
from dfl_transfer_features import corrupt
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "DECISIONS.md D24"
VIEWS = ("masked", "handled", "team_swap")
SEEDS = {"mask": 640000, "handle": 650000, "swap": 660000}
CACHE = ms.CACHE / "d24"
RECORD = ms.RECIPE / "d24_measurement.json"
BRIEF_FRAMES = 25                 # a stoppage shorter than one second
LIVE_HIDDEN = (.30, .50)          # the masked view hides the ball on this share of live frames
OFFSET_TARGETS = ("raw", "aba3")
BASELINE_TARGETS = ("raw", "aba3")


def seed(kind, game, index):
    return SEEDS[kind] + 10 * game + index


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D24 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


# -- views -----------------------------------------------------------------------------------------

def build_views(half, game, index):
    """{view: (positions, ball)} for one parsed D19 half of `game` (1-3), half `index` (0 or 1)."""
    positions = np.asarray(half["positions"], dtype=np.float32)
    masked = corrupt(positions, half["ball"], seed("mask", game, index))
    carried = augment.handle_dead_ball(half["labels"], positions, half["ball"],
                                       np.random.default_rng(seed("handle", game, index)))
    handled = corrupt(positions, carried, seed("mask", game, index))
    swapped = augment.swap_teams(handled[0], half["teams"], np.random.default_rng(seed("swap", game, index)),
                                 every=augment.GUARD_SWAP_EVERY, frames=lambda _: augment.GUARD_SWAP_FRAMES)
    return {"masked": masked, "handled": handled, "team_swap": (swapped, handled[1])}


def seen(ball):
    return np.isfinite(ball).all(axis=1)


def visibility(labels, views):
    """Frame counts behind D24's view checks for one half."""
    live, dead = np.isin(labels, (pm.H, pm.A)), labels == pm.D
    return {"live": int(live.sum()), "dead": int(dead.sum()),
            "masked_live_hidden": int((live & ~seen(views["masked"][1])).sum()),
            "masked_dead_seen": int((dead & seen(views["masked"][1])).sum()),
            "handled_dead_seen": int((dead & seen(views["handled"][1])).sum())}


def view_check(rows):
    """D24's view checks for one game, pooled over its halves."""
    live, dead = sum(row["live"] for row in rows), sum(row["dead"] for row in rows)
    hidden = sum(row["masked_live_hidden"] for row in rows) / live if live else 0.0
    masked = sum(row["masked_dead_seen"] for row in rows) / dead if dead else 0.0
    handled = sum(row["handled_dead_seen"] for row in rows) / dead if dead else 0.0
    return {"masked_live_hidden": hidden, "masked_dead_seen": masked, "handled_dead_seen": handled,
            "passed": bool(LIVE_HIDDEN[0] <= hidden <= LIVE_HIDDEN[1] and handled > masked)}


# -- decoder inputs --------------------------------------------------------------------------------

def identity(parsed, reference="D15"):
    return {"protocol": PROTOCOL, "reference": ms.reference_identity(reference, parsed), "views": list(VIEWS),
            "seeds": SEEDS, "augment_sha256": ms.digest(Path(augment.__file__)),
            "corrupt_sha256": ms.digest(ROOT / "experiments" / "dfl_transfer_features.py")}


def view_halves(parsed=ms.PARSED, cache=CACHE, reference="D15", signals_of=None, variant=None):
    """The reference's decoder inputs and labels on every view of every half, cached by identity.

    `signals_of`, if given, replaces `diagnosis.half_signals` (same arguments) to forecast a variant
    of the reference, such as D28's ball fill. A variant names itself in `variant`, which joins the
    cache identity, and needs a cache of its own.
    """
    key = identity(parsed, reference)
    if signals_of is not None:
        if variant is None or cache is None or Path(cache) == CACHE:
            raise ValueError("A variant forecast needs a `variant` and a cache of its own")
        key = dict(key, variant=variant)

    def produce():
        model = ms.reference_model(reference)
        forecast = diagnosis.half_signals if signals_of is None else signals_of
        for game in ms.GAMES:
            for index, half in enumerate(ms.load(parsed, game)):
                for view, (positions, ball) in build_views(half, game, index).items():
                    signals, labels = forecast(model, positions, ball, half["teams"], half["pitch"])
                    yield diagnosis.HalfSignals("metrica", str(game), view, index, half["labels"], signals, labels)
                print(f"Forecast {reference} on the D24 views of Metrica game {game} half {index + 1}", flush=True)

    return diagnosis.collect(cache, key, produce)


# -- scoring ---------------------------------------------------------------------------------------

def grouped(halves, predict):
    """{view: {game: [prediction of half 0, prediction of half 1]}}."""
    result = {}
    for half in sorted(halves, key=lambda item: (item.view, int(item.match), item.index)):
        result.setdefault(half.view, {}).setdefault(int(half.match), []).append(np.asarray(predict(half)))
    return result


def view_scores(halves, truths, predict):
    """{view: {game: {target: score}}} of `predict(half)` against D19's targets."""
    return {view: {str(game): ms.score(truths[game], parts) for game, parts in games.items()}
            for view, games in grouped(halves, predict).items()}


def d24_mean(scores, views=VIEWS):
    """Mean over the views of the mean N = 3 spellF1 over the games: the gate quantity of D25 and D26."""
    return float(np.mean([np.mean([match[ms.PRIMARY]["spell_f1"] for match in scores[view].values()])
                          for view in views]))


def brief_stoppages(labels, runs, limit=BRIEF_FRAMES):
    """Stoppages shorter than `limit` frames: all, and interior (a team on both sides)."""
    found = {"all": 0, "interior": 0}
    parts = pm.segments(labels, runs)
    for i, (start, stop, code) in enumerate(parts):
        if code != pm.D or stop - start >= limit:
            continue
        found["all"] += 1
        before, after = (parts[i - 1] if i else None), (parts[i + 1] if i + 1 < len(parts) else None)
        if (before is not None and after is not None and before[1] == start and after[0] == stop
                and before[2] in pm.POSSESSION_LABELS and after[2] in pm.POSSESSION_LABELS):
            found["interior"] += 1
    return found


def joined(parts):
    return np.r_[parts[0], pm.X, parts[1]].astype(np.int8)


def stoppage_counts(predictions, truths):
    """Sub-second stoppages per game and summed, predicted and true."""
    games = {}
    for game, parts in predictions.items():
        truth = truths[game]["raw"]
        runs = pm.valid_runs(truth)
        games[str(game)] = {"predicted": brief_stoppages(joined(parts), runs), "true": brief_stoppages(truth, runs)}
    total = {side: {kind: sum(games[game][side][kind] for game in games) for kind in ("all", "interior")}
             for side in ("predicted", "true")}
    return {"games": games, **total}


def offset_report(predictions, truths):
    """Signed boundary offsets by transition type, pooled and per game, against the raw and N = 3 targets."""
    report = {}
    for name in OFFSET_TARGETS:
        per_game = {game: diagnosis.offsets(truths[game][name], joined(parts)) for game, parts in predictions.items()}
        report[name] = {"pooled": {kind: diagnosis.offset_summary([item[kind] for item in per_game.values()])
                                   for kind in diagnosis.TYPES},
                        "games": {str(game): {kind: diagnosis.offset_summary([item[kind]]) for kind in diagnosis.TYPES}
                                  for game, item in per_game.items()}}
    return report


def measurements(halves, truths, predict):
    """Scores, sub-second stoppages and boundary offsets of every view."""
    scores = view_scores(halves, truths, predict)
    predictions = grouped(halves, predict)
    return {view: {"matches": scores[view], "means": ms.means(scores[view]),
                   "stoppages": stoppage_counts(predictions[view], truths),
                   "offsets": offset_report(predictions[view], truths)}
            for view in scores}


def baseline_scores(parsed):
    """The organisers' smooth50 baseline on every view, reported only."""
    from dfl_submission.vendor_trial import baseline_prediction
    result = {view: {} for view in VIEWS}
    for game in ms.GAMES:
        halves = ms.load(parsed, game)
        truth = ms.targets(halves)
        built = [build_views(half, game, index) for index, half in enumerate(halves)]
        for view in VIEWS:
            parts = [baseline_prediction({"positions": views[view][0], "ball": views[view][1], "teams": half["teams"]})
                     for half, views in zip(halves, built)]
            scored = ms.score(truth, parts)
            result[view][str(game)] = {name: scored[name] for name in BASELINE_TARGETS}
    return {view: {"matches": games, "means": {name: float(np.mean([match[name]["spell_f1"] for match in games.values()]))
                                               for name in BASELINE_TARGETS}}
            for view, games in result.items()}


def measure(parsed=ms.PARSED, cache=CACHE, output=RECORD):
    """D24: frozen D15 on the three degraded views, once (DECISIONS.md D24)."""
    require_external_only()
    if output.exists():
        raise FileExistsError(f"{output} exists; D24 is measured once")
    if not ms.gate_passed(ms.RECIPE / "sanity.json", ms.inputs(parsed)):
        raise ValueError("The latest D19 reader attempt did not pass on these parsed inputs")
    _, clean_halves, truths = ms.reference_scores("D15", parsed)   # stops unless D15 reproduces its D19 record
    ms.reference_scores("D3", parsed)                                # and D3 its own
    rows = {game: [visibility(half["labels"], build_views(half, game, index))
                   for index, half in enumerate(ms.load(parsed, game))] for game in ms.GAMES}
    checks = {str(game): view_check(items) for game, items in rows.items()}
    print("D24 view checks: " + ", ".join(
        f"game {game} live hidden {item['masked_live_hidden']:.3f}, dead seen masked {item['masked_dead_seen']:.3f} "
        f"handled {item['handled_dead_seen']:.3f}" for game, item in checks.items()), flush=True)
    if not all(item["passed"] for item in checks.values()):
        raise ValueError(f"D24 view checks failed {checks}; only the view code may be fixed, nothing was written")
    halves = view_halves(parsed, cache)
    clean = [replace(half, view="clean") for half in clean_halves]
    report = measurements(halves + clean, truths, lambda half: half.d3)
    scores = {view: report[view]["matches"] for view in report}
    record = {"protocol": PROTOCOL, "candidate": "D15", "decision": None, "views": list(VIEWS), "seeds": SEEDS,
              "view_checks": checks, "visibility": {str(game): items for game, items in rows.items()},
              "reproduction": {"d15_d19": True, "d3_d19": True}, "views_report": report,
              "d24_mean": d24_mean(scores), "clean_mean": d24_mean(scores, ("clean",)),
              "baseline": baseline_scores(parsed), "recipe": ms.reference_recipe("D15"),
              "identity": identity(parsed), "inputs": ms.inputs(parsed)}
    write_new(output, record)
    for view in (*VIEWS, "clean"):
        entry = report[view]
        print(f"D24 {view}: " + ", ".join(f"{name} {values['spell_f1']:.4f}" for name, values in entry["means"].items())
              + f"; sub-second stoppages predicted {entry['stoppages']['predicted']} true {entry['stoppages']['true']}",
              flush=True)
    print(f"D24 mean (N = 3 spellF1 over the three views) {record['d24_mean']:.5f}; clean {record['clean_mean']:.5f}")
    return record


def check_record(halves, truths, record=RECORD):
    """The view scores of cached labels, after checking they reproduce the D24 record exactly."""
    scores = view_scores(halves, truths, lambda half: half.d3)
    stored = json.loads(Path(record).read_text())["views_report"]
    if json.loads(json.dumps(scores)) != {view: stored[view]["matches"] for view in VIEWS}:
        raise ValueError("The cached labels do not reproduce the D24 record")
    return scores


def reference_scores(parsed=ms.PARSED, cache=CACHE, record=RECORD):
    """D15's cached view inputs, D19 targets and view scores, checked against the D24 record."""
    halves = view_halves(parsed, cache)
    truths = {game: ms.targets(ms.load(parsed, game)) for game in ms.GAMES}
    return check_record(halves, truths, record), halves, truths


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure",))
    parser.add_argument("--parsed", type=Path, default=ms.PARSED)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    require_external_only()
    measure(args.parsed, args.cache)


if __name__ == "__main__":
    main()
