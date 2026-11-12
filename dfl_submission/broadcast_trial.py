"""D31: the broadcast ball as a second source (DECISIONS.md D31).

`calibrate` sets the aligner's two constants on harness recordings of the DFL training halves.
`develop` merges harness broadcast balls into D15's development views (DFL WPY masked, dead ball
carried and team swap; D24's Metrica views), thins the recovery to 10-40% of the hidden frames,
forecasts frozen D15 and picks r*. `confirm` repeats it once on WQQ and WR9 at r*. `check` is the
competition check Derek runs outside the AI session: it prints PASS, or FAIL with the failed names.
`build` writes the competition file with the merged ball, then D15 and D27's rule.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import broadcast_ball as bb, two_camera as tc
from dfl_submission.external_only import require_external_only


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "broadcast_recipe"
DATA = ROOT / "data" / "d31"
PROTOCOL = "DECISIONS.md D31"
GRID = {"max_cost": (1.0, 1.5, 2.0, 3.0), "min_margin": (.25, .5, 1.0, 2.0)}
WRONG_SHARE = .01
LEVELS = (.1, .2, .3, .4)
GAIN = .01
SEEDS = {"calibrate": 900000, "develop": 910000, "confirm": 920000}
DFL_VIEWS = ("masked", "dead_carried", "team_swap")
D24_VIEWS = ("masked", "handled", "team_swap")
COVERAGE, AGREEMENT_FRAMES, AGREEMENT = .30, 2000, (1.5, 5.0)
PRIMARY = "aba3"
# D31 and its second attempt D31a: constant-setting seeds, grid, cost and record prefix.
ATTEMPTS = {"": {"prefix": "", "seed": SEEDS["calibrate"], "grid": GRID, "trim": 0.0, "translate": False},
            "a": {"prefix": "a_", "seed": 905000, "grid": {"max_cost": (.75, 1.0, 1.5, 2.0, 3.0),
                                                            "min_margin": (.25, .5, 1.0, 2.0)},
                  "trim": bb.TRIM, "translate": True}}
CURRENT = "a"


def record(name, attempt=CURRENT):
    return RECIPE / f"{ATTEMPTS[attempt]['prefix']}{name}.json"


def cost_options(constants):
    """The aligner's cost options for a constants record (D31's records have none)."""
    return {"trim": constants.get("trim", 0.0), "translate": constants.get("translate", False)}


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D31 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def plain(results):
    """Stretch results as JSON-safe dicts."""
    return [{key: (None if isinstance(value, float) and not np.isfinite(value) else value) for key, value in r.items()}
            for r in results]


# -- rules -----------------------------------------------------------------------------------------

def select_constants(entries):
    """Most accepted frames among settings with at most 1% wrong; then larger margin, then smaller cost."""
    eligible = [e for e in entries if e["accepted"] > 0 and e["wrong"] <= WRONG_SHARE * e["accepted"]]
    if not eligible:
        return None
    best = max(eligible, key=lambda e: (e["accepted"], e["min_margin"], -e["max_cost"]))
    return {"max_cost": best["max_cost"], "min_margin": best["min_margin"]}


def thin(view_ball, broadcast_ball, results, level, rng):
    """The view's ball with recovery cut to `level` of its missing frames, stretch by stretch in `rng` order."""
    hidden = int((~np.isfinite(view_ball).all(axis=1)).sum())
    target = int(np.floor(level * hidden))
    accepted = [r for r in results if r["accepted"]]
    ball, kept = np.array(view_ball, copy=True), 0
    for index in rng.permutation(len(accepted)):
        if kept >= target:
            break
        merged, filled = bb.merge(ball, broadcast_ball, [accepted[index]])
        take = np.flatnonzero(filled)[:target - kept]
        ball[take] = merged[take]
        kept += len(take)
    return ball, {"hidden": hidden, "target": target, "kept": kept, "shortfall": target - kept}


def select_level(gains):
    """r*: the smallest level whose gain is at least GAIN on both groups, or None."""
    for level in sorted(gains):
        if gains[level]["dfl"] >= GAIN and gains[level]["metrica"] >= GAIN:
            return level
    return None


def confirmation_passed(per_match):
    values = list(per_match.values())
    return bool(np.mean(values) >= GAIN and min(values) >= 0)


def criteria(report, graded, r_star):
    """The competition check's failed criteria (names only)."""
    if not report.get("teams", False):
        return ["teams"]
    failed = []
    if report["mapped"][graded].mean() < COVERAGE:
        failed.append("coverage")
    distances = report["distances"][graded]
    distances = distances[np.isfinite(distances)]
    if (len(distances) < AGREEMENT_FRAMES or np.median(distances) > AGREEMENT[0]
            or np.percentile(distances, 90) > AGREEMENT[1]):
        failed.append("agreement")
    missing = graded & report["missing"]
    if not missing.any() or report["filled"][missing].mean() < r_star:
        failed.append("recovery")
    return failed


# -- the merge as the build uses it ----------------------------------------------------------------

def merge_arrays(positions, ball, teams, broadcast_positions, broadcast_ball, broadcast_teams, broadcast_valid, pitch,
                 constants, report=None):
    """The tactical ball (uncentred, as the build has it) with missing frames filled from the broadcast.

    Broadcast arrays are centred. Frames that are not filled keep their exact values.
    """
    centre = np.asarray(pitch, dtype=np.float64) / 2
    tactical_ball = np.asarray(ball, dtype=np.float64) - centre
    results = bb.align(np.asarray(positions, dtype=np.float64) - centre, teams, broadcast_positions, broadcast_teams,
                       broadcast_valid, constants, **cost_options(constants))
    merged, filled = bb.merge(tactical_ball, broadcast_ball, results)
    out = np.array(ball, copy=True)
    out[filled] = (merged[filled] + centre).astype(out.dtype)
    if report is not None:
        n = len(out)
        mapped, distances = np.zeros(n, dtype=bool), np.full(n, np.nan)
        seen_tactical = np.isfinite(tactical_ball).all(axis=1)
        seen_broadcast = np.isfinite(broadcast_ball).all(axis=1)
        for result in sorted((r for r in results if r["accepted"]), key=lambda r: r["start"]):
            frames = np.arange(result["start"], result["stop"])
            targets = frames + result["offset"]
            inside = (targets >= 0) & (targets < n)
            frames, targets = frames[inside], targets[inside]
            mapped[targets] = True
            both = seen_broadcast[frames] & seen_tactical[targets] & np.isnan(distances[targets])
            shown = (bb.orient(np.asarray(broadcast_ball)[frames[both]], result["orientation"])
                     - np.asarray(result.get("shift", (0.0, 0.0)), dtype=np.float64))
            distances[targets[both]] = np.linalg.norm(shown - tactical_ball[targets[both]], axis=1)
        report.update(results=results, mapped=mapped, filled=filled, missing=~np.isfinite(ball).all(axis=1),
                      distances=distances)
    return out


def merger(constants, report=None):
    """A `merge_ball` hook for `build.tactical_tracking_halves`: reads the broadcast files of the bundle."""
    def merge_ball(positions, ball, teams, pitch):
        from broadcast_alignment import PITCH, load_source, reconcile_teams
        from games import require
        from starter.baseline import team_names
        broadcast = load_source("broadcast")
        (meta,) = require(["atd_meta"])
        try:
            broadcast_teams = reconcile_teams(team_names(meta), broadcast["names"], broadcast["teams"])
        except ValueError:
            if report is None:
                raise
            report["teams"] = False
            return ball
        pitch = np.asarray(pitch, dtype=np.float64)
        scale = (pitch / PITCH).astype(np.float32)
        valid = np.asarray(broadcast["valid"], dtype=bool)
        broadcast_positions = broadcast["positions"] * scale
        broadcast_positions[~valid] = np.nan
        broadcast_ball = broadcast["ball"] * scale
        seen = (np.isfinite(broadcast_ball).all(axis=1) & (broadcast_ball >= -5).all(axis=1)
                & (broadcast_ball <= pitch + 5).all(axis=1) & valid)
        broadcast_ball[~seen] = np.nan
        centre = pitch / 2
        out = merge_arrays(positions, ball, teams, broadcast_positions - centre, broadcast_ball - centre,
                           broadcast_teams, valid, pitch, constants, report)
        if report is not None:
            report["teams"] = True
        return out
    return merge_ball


# -- harness bookkeeping ---------------------------------------------------------------------------

def dfl_halves(dfl, split, *, raw, guard):
    """{(match, view, index): half} of the DFL halves of a split (the guard views if `guard`)."""
    from dfl_submission import augment, external_benchmark
    halves = augment.guard_halves(dfl, None, split, raw=raw) if guard else external_benchmark.halves(dfl, None, split, raw=raw)
    return {(half.match, half.view, half.index): half for half in halves if half.source == "dfl"}


def cached_labels(halves, recipe, key):
    """{key(half): labels} of cached halves, decoded at `recipe` from their signals.

    D15's development and holdout caches store labels decoded at D3's recipe; only their signals are D15's.
    """
    from dfl_submission import diagnosis
    return {key(half): np.asarray(diagnosis.decode(half.signals, recipe)) for half in halves}


def forecast(model, positions, ball, teams, pitch):
    from dfl_submission import diagnosis
    return np.asarray(diagnosis.half_signals(model, positions, ball, teams, pitch)[1])


def dfl_targets(first, second):
    from dfl_submission.decoder_trial import CONVENTIONS, relabel
    from evaluation import possession_metrics as pm
    raw = np.r_[np.asarray(first), pm.X, np.asarray(second)].astype(np.int8)
    return {name: relabel(raw, seconds) for name, seconds in CONVENTIONS.items()}


def spell(truth, predictions):
    from dfl_submission import metrica_sample as ms
    return ms.score(truth, predictions)


def recording_summary(recording, results):
    accepted = [r for r in results if r["accepted"]]
    counts = tc.frame_counts(results, recording["shows"], recording["orientation"])
    return {"stretches": len(results), "accepted_stretches": len(accepted), **counts,
            "offset": recording["offset"], "orientation": recording["orientation"]}


def merged_views(model, recording, views, constants, seed, levels, cached=None):
    """Base and thinned-merge labels of one half's views, with alignment and thinning summaries."""
    out = {}
    for view, (positions, ball, teams, pitch) in views.items():
        base = forecast(model, positions, ball, teams, pitch)
        if cached is not None and not np.array_equal(base, cached[view]):
            raise ValueError(f"D15's labels without the merge differ from its cache on view {view}; nothing is scored")
        results = bb.align(positions, teams, recording["positions"], recording["teams"], recording["valid"], constants,
                           **cost_options(constants))
        full, filled = bb.merge(ball, recording["ball"], results)
        entry = {"base": base, "levels": {}, "summary": recording_summary(recording, results),
                 "recovered": int(filled.sum()), "hidden": int((~np.isfinite(ball).all(axis=1)).sum())}
        for level in levels:
            thinned, info = thin(ball, recording["ball"], results, level, np.random.default_rng(seed + 1))
            entry["levels"][level] = {"labels": forecast(model, positions, thinned, teams, pitch), "info": info}
        out[view] = entry
        print(f"D31 merged view {view}: recovered {entry['recovered']} of {entry['hidden']} hidden frames; "
              + ", ".join(f"r={level} kept {e['info']['kept']}" for level, e in entry["levels"].items()), flush=True)
    return out


# -- calibrate -------------------------------------------------------------------------------------

def calibrate(dfl, *, raw, attempt=CURRENT):
    require_external_only()
    from dfl_submission.train import TRAIN
    settings = ATTEMPTS[attempt]
    output = record("constants", attempt)
    options = {"trim": settings["trim"], "translate": settings["translate"]}
    if output.exists():
        raise FileExistsError(f"{output} exists; D31's constants are set once")
    halves = dfl_halves(dfl, "train", raw=raw, guard=False)
    rows = []
    for m, match in enumerate(TRAIN):
        for index in (0, 1):
            truth, tactical = halves[(match, "clean", index)], halves[(match, "masked", index)]
            seed = settings["seed"] + 10 * m + index
            recording = tc.simulate(truth.positions, truth.ball, truth.teams, seed)
            started = time.time()
            results = bb.align(tactical.positions, tactical.teams, recording["positions"], recording["teams"],
                               recording["valid"], **options)
            rows.append({"match": match, "index": index, "seed": seed, "results": results,
                         "shows": recording["shows"], "orientation": recording["orientation"],
                         "offset": recording["offset"], "usable_frames": int(sum(r["stop"] - r["start"] for r in results))})
            print(f"D31 calibration {match} half {index + 1}: {len(results)} stretches aligned in "
                  f"{time.time() - started:.0f} s", flush=True)
    entries = []
    for max_cost in settings["grid"]["max_cost"]:
        for min_margin in settings["grid"]["min_margin"]:
            constants = {"max_cost": max_cost, "min_margin": min_margin}
            accepted = wrong = 0
            for row in rows:
                decided = [dict(r, accepted=bb.decide(r, constants)) if r["offset"] is not None else dict(r, accepted=False)
                           for r in row["results"]]
                counts = tc.frame_counts(decided, row["shows"], row["orientation"])
                accepted, wrong = accepted + counts["accepted"], wrong + counts["wrong"]
            entries.append({**constants, "accepted": accepted, "wrong": wrong})
            print(f"D31 constants max_cost {max_cost} min_margin {min_margin}: accepted {accepted}, wrong {wrong}", flush=True)
    selected = select_constants(entries)
    if selected is not None and attempt:
        selected = dict(selected, **options)
    write_new(output, {"protocol": PROTOCOL + ("a" if attempt else ""), "grid": settings["grid"], "entries": entries,
                       "selected": selected, "options": options,
                       "halves": [{key: row[key] for key in ("match", "index", "seed", "orientation", "offset", "usable_frames")}
                                  | {"results": plain(row["results"])} for row in rows],
                       "set_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D31 constants: {selected if selected else 'none qualifies; D31 stops'}", flush=True)
    return selected


def constants_selected(attempt=CURRENT):
    selected = json.loads(record("constants", attempt).read_text())["selected"]
    if selected is None:
        raise ValueError("No D31 constants qualified; D31 stopped")
    return selected


# -- develop ---------------------------------------------------------------------------------------

def develop(dfl, skill, parsed, *, raw, d15_cache, attempt=CURRENT):
    require_external_only()
    from dfl_submission import metrica_sample as ms, metrica_views as mv, zoo
    output = record("development", attempt)
    if output.exists():
        raise FileExistsError(f"{output} exists; D31 development is scored once")
    constants = constants_selected(attempt)
    cached_dev, recipe, _ = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)   # stops unless reproduced
    _, d24_cached, truths = mv.reference_scores()                                                 # and D24
    model = ms.reference_model("D15")
    dfl_cache = cached_labels([h for h in cached_dev if h.source == "dfl"], recipe, lambda h: (h.view, h.index))
    d24_cache = cached_labels(d24_cached, recipe, lambda h: (int(h.match), h.view, h.index))
    halves = dfl_halves(dfl, "development", raw=raw, guard=True)
    match = next(key[0] for key in halves)
    dfl_out, metrica_out = {}, {}
    for index in (0, 1):
        truth = halves[(match, "clean", index)]
        seed = SEEDS["develop"] + 100 * 0 + 10 * 0 + index
        recording = tc.simulate(truth.positions, truth.ball, truth.teams, seed)
        views = {view: (halves[(match, view, index)].positions, halves[(match, view, index)].ball,
                        halves[(match, view, index)].teams, halves[(match, view, index)].pitch) for view in DFL_VIEWS}
        dfl_out[index] = merged_views(model, recording, views, constants, seed, LEVELS,
                                      cached={view: dfl_cache[(view, index)] for view in DFL_VIEWS})
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            seed = SEEDS["develop"] + 100 * 1 + 10 * game + index
            recording = tc.simulate(half["positions"], half["ball"], half["teams"], seed)
            built = mv.build_views(half, game, index)
            views = {view: (built[view][0], built[view][1], half["teams"], half["pitch"]) for view in D24_VIEWS}
            metrica_out[(game, index)] = merged_views(model, recording, views, constants, seed, LEVELS,
                                                      cached={view: d24_cache[(game, view, index)] for view in D24_VIEWS})
    truth_dfl = dfl_targets(halves[(match, "clean", 0)].labels, halves[(match, "clean", 1)].labels)

    def dfl_scores(pick):
        return {view: spell(truth_dfl, [pick(dfl_out[i][view]) for i in (0, 1)]) for view in DFL_VIEWS}

    def d24_scores(pick):
        return {view: {str(game): ms.score(truths[game], [pick(metrica_out[(game, i)][view]) for i in (0, 1)])
                       for game in ms.GAMES} for view in D24_VIEWS}

    base = {"dfl": dfl_scores(lambda e: e["base"]), "d24": d24_scores(lambda e: e["base"])}
    entries, gains = {}, {}
    for level in LEVELS:
        merged = {"dfl": dfl_scores(lambda e: e["levels"][level]["labels"]),
                  "d24": d24_scores(lambda e: e["levels"][level]["labels"])}
        dfl_gain = float(np.mean([merged["dfl"][v][PRIMARY]["spell_f1"] - base["dfl"][v][PRIMARY]["spell_f1"]
                                  for v in DFL_VIEWS]))
        metrica_gain = mv.d24_mean(merged["d24"]) - mv.d24_mean(base["d24"])
        gains[level] = {"dfl": dfl_gain, "metrica": metrica_gain}
        entries[str(level)] = {"scores": merged, "gains": gains[level],
                               "thinning": {"dfl": {f"{v}_{i}": dfl_out[i][v]["levels"][level]["info"]
                                                    for i in (0, 1) for v in DFL_VIEWS},
                                            "metrica": {f"{g}_{v}_{i}": metrica_out[(g, i)][v]["levels"][level]["info"]
                                                        for (g, i) in metrica_out for v in D24_VIEWS}}}
        print(f"D31 development r = {level}: DFL {dfl_gain:+.5f}, Metrica D24 {metrica_gain:+.5f}", flush=True)
    r_star = select_level(gains)
    summaries = {"dfl": {f"{v}_{i}": {k: dfl_out[i][v][k] for k in ("summary", "recovered", "hidden")}
                         for i in (0, 1) for v in DFL_VIEWS},
                 "metrica": {f"{g}_{v}_{i}": {k: metrica_out[(g, i)][v][k] for k in ("summary", "recovered", "hidden")}
                             for (g, i) in metrica_out for v in D24_VIEWS}}
    write_new(output, {"protocol": PROTOCOL, "constants": constants, "recipe": recipe, "base": base,
                       "base_d24_mean": mv.d24_mean(base["d24"]), "levels": entries, "r_star": r_star,
                       "alignment": summaries, "reproduction": {"d15_development": True, "d15_d24": True,
                                                                "d15_labels_without_merge": True},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D31 development: r* = {r_star if r_star is not None else 'none; D31 stops'}", flush=True)
    return r_star


def r_star_selected(attempt=CURRENT):
    value = json.loads(record("development", attempt).read_text())["r_star"]
    if value is None:
        raise ValueError("No D31 level gained enough on development; D31 stopped")
    return value


# -- confirm ---------------------------------------------------------------------------------------

def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, attempt=CURRENT):
    require_external_only()
    from dfl_submission import metrica_sample as ms, scale_trial, vendor_trial, zoo
    from dfl_submission.train import HOLDOUT
    output = record("confirmation", attempt)
    if output.exists():
        raise FileExistsError(f"{output} exists; D31 confirmation is scored once")
    constants, r_star = constants_selected(attempt), r_star_selected(attempt)
    setting, frozen = zoo.d15_setting()
    cached = scale_trial.setting_halves(dfl, skill, parsed, vendor_trial.split_games("confirm_a"), "holdout", setting,
                                        raw=raw, cache=d15_holdout_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    from dfl_submission import aligned_trial
    full = aligned_trial.evaluate(cached, frozen["recipe"])
    if vendor_trial.canonical(full["external"]) != stored["external"]:
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    labels = cached_labels([h for h in cached if h.source == "dfl"], frozen["recipe"], lambda h: (h.match, h.view, h.index))
    model = ms.reference_model("D15")
    halves = dfl_halves(dfl, "holdout", raw=raw, guard=True)
    per_match, scores, summaries = {}, {}, {}
    for m, match in enumerate(HOLDOUT):
        out = {}
        for index in (0, 1):
            truth = halves[(match, "clean", index)]
            seed = SEEDS["confirm"] + 10 * m + index
            recording = tc.simulate(truth.positions, truth.ball, truth.teams, seed)
            views = {view: (halves[(match, view, index)].positions, halves[(match, view, index)].ball,
                            halves[(match, view, index)].teams, halves[(match, view, index)].pitch) for view in DFL_VIEWS}
            out[index] = merged_views(model, recording, views, constants, seed, (r_star,),
                                      cached={view: labels[(match, view, index)] for view in DFL_VIEWS})
        truth = dfl_targets(halves[(match, "clean", 0)].labels, halves[(match, "clean", 1)].labels)
        base = {view: spell(truth, [out[i][view]["base"] for i in (0, 1)]) for view in DFL_VIEWS}
        merged = {view: spell(truth, [out[i][view]["levels"][r_star]["labels"] for i in (0, 1)]) for view in DFL_VIEWS}
        per_match[match] = float(np.mean([merged[v][PRIMARY]["spell_f1"] - base[v][PRIMARY]["spell_f1"] for v in DFL_VIEWS]))
        scores[match] = {"base": base, "merged": merged}
        summaries[match] = {f"{v}_{i}": {k: out[i][v][k] for k in ("summary", "recovered", "hidden")}
                            | {"thinning": out[i][v]["levels"][r_star]["info"]} for i in (0, 1) for v in DFL_VIEWS}
        print(f"D31 confirmation {match}: {per_match[match]:+.5f}", flush=True)
    passed = confirmation_passed(per_match)
    write_new(output, {"protocol": PROTOCOL, "constants": constants, "r_star": r_star, "per_match": per_match,
                       "mean": float(np.mean(list(per_match.values()))), "passed": passed, "scores": scores,
                       "alignment": summaries, "reproduction": {"d15_holdout": True, "d15_labels_without_merge": True},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D31 confirmation: mean {np.mean(list(per_match.values())):+.5f} -> {'passed' if passed else 'failed'}", flush=True)
    return passed


# -- competition check and build -------------------------------------------------------------------

def confirmed(attempt=CURRENT):
    stored = json.loads(record("confirmation", attempt).read_text())
    if stored["passed"] is not True:
        raise ValueError("D31 did not pass confirmation")
    return stored["constants"], stored["r_star"]


def check():
    """Derek's competition check: prints PASS, or FAIL with the names of the failed criteria only."""
    from dfl_submission import zoo
    from dfl_submission.build import tactical_tracking_halves
    from games import GAME, half_video_blocks, source_frames
    constants, r_star = confirmed()
    _, frozen = zoo.d15_setting()
    report = {}
    for _ in tactical_tracking_halves(centered=True, pitch=frozen["pitch"], merge_ball=merger(constants, report)):
        pass
    graded = np.zeros(source_frames(), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    failed = criteria(report, graded, r_star)
    print("PASS" if not failed else "FAIL: " + ", ".join(failed), flush=True)
    return not failed


def confirmed_merger(path=RECIPE / "a_competition_check.json"):
    """The build's ball merge, after D31's confirmation and Derek's recorded competition check PASS."""
    constants, _ = confirmed()
    if not path.exists() or json.loads(path.read_text()).get("result") != "PASS":
        raise ValueError("D31's competition check has not been recorded as PASS")
    return merger(constants)


def build(output):
    """The competition file: the merged ball, then D15, then D27's kick-off rule."""
    from dfl_submission import round8
    round8.build(output, broadcast=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--dfl", type=Path, required=True)
    calibration.add_argument("--dfl-raw", action="store_true")
    calibration.add_argument("--attempt", choices=tuple(ATTEMPTS), default=CURRENT)
    for name in ("develop", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--d15-cache" if name == "develop" else "--d15-holdout-cache", type=Path, required=True)
        command.add_argument("--attempt", choices=tuple(ATTEMPTS), default=CURRENT)
    commands.add_parser("check")
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "check":
        check()
        return
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "calibrate":
        calibrate(args.dfl, raw=args.dfl_raw, attempt=args.attempt)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, attempt=args.attempt)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                attempt=args.attempt)


if __name__ == "__main__":
    main()
