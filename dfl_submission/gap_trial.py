"""D28: fill short ball dropouts (DECISIONS.md D28).

The candidate is frozen D15 with one change before the models see a half: each run of frames with
a missing ball that is at most K frames long, and has a seen ball on the frames just before and
after it in the same half, is filled on the straight line between those two sightings. Nothing is
retrained and the decoder is unchanged.

`coverage` records, before any scoring, how many missing-ball frames each K fills per development
group. `develop` forecasts D15 with each K once on the development matches (DFL WPY in four views,
four SkillCorner matches, eight PFF matches), the Metrica games (D19) and the D24 views, after D15
reproduces each of their records, and selects at most one K. `confirm` scores the selected K once on
the reused DFL/SkillCorner holdout and PFF confirmation set B. `build` writes the competition file:
the fill, then D15, then D27.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.build import write_csv
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "gap_recipe"
CACHE = ROOT / "data" / "d28"
PROTOCOL = "DECISIONS.md D28"
GRID = (6, 12, 25)                     # frames: 0.24, 0.48 and 1 s; nothing longer (D28)
GAIN, LOSS = .005, .01


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D28 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


# -- the fill --------------------------------------------------------------------------------------

def seen(ball):
    """Frames whose ball has both coordinates."""
    return np.isfinite(ball).all(axis=1)


def fill(ball, limit):
    """A copy of one half's ball with every interior missing run of at most `limit` frames filled linearly."""
    ball = np.asarray(ball)
    result = np.array(ball, copy=True)
    if limit <= 0 or len(ball) == 0:
        return result
    missing = ~seen(ball)
    edges = np.flatnonzero(np.diff(np.r_[0, missing.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        length = stop - start
        if start == 0 or stop == len(ball) or length > limit:
            continue
        before, after = ball[start - 1].astype(np.float64), ball[stop].astype(np.float64)
        weight = (np.arange(1, length + 1, dtype=np.float64) / (length + 1))[:, None]
        result[start:stop] = before + (after - before) * weight
    return result


def filler(limit):
    """The fill as a step on one half's ball, for `kickoff.d15_labels(prepare=...)`."""
    return lambda ball: fill(ball, limit)


def signals_with(limit):
    """A `signals_of` hook: D15's decoder inputs and labels for a half whose ball was filled first."""
    def signals_of(teacher, positions, ball, teams, pitch):
        return diagnosis.half_signals(teacher, positions, fill(ball, limit), teams, pitch)
    return signals_of


def variant(limit):
    """What a K's forecasts depend on beyond D15's own identity."""
    code = hashlib.sha256(inspect.getsource(fill).encode()).hexdigest()
    return {"protocol": PROTOCOL, "limit": int(limit), "fill_sha256": code}


# -- coverage --------------------------------------------------------------------------------------

def counts(ball, live, limits=GRID):
    """Missing-ball frames of one half, all and live, and how many each limit fills."""
    missing = ~seen(ball)
    live = np.asarray(live, dtype=bool)
    filled = {limit: missing & seen(fill(ball, limit)) for limit in limits}
    return {"frames": int(len(ball)), "live": int(live.sum()), "missing": int(missing.sum()),
            "live_missing": int((missing & live).sum()),
            "filled": {str(limit): int(mask.sum()) for limit, mask in filled.items()},
            "live_filled": {str(limit): int((mask & live).sum()) for limit, mask in filled.items()}}


def summarise(rows):
    """Pooled counts of several halves, with the share of missing frames each limit fills."""
    total = {key: sum(row[key] for row in rows) for key in ("frames", "live", "missing", "live_missing")}
    limits = list(rows[0]["filled"]) if rows else []
    for key in ("filled", "live_filled"):
        total[key] = {limit: sum(row[key][limit] for row in rows) for limit in limits}
    total["share"] = {limit: total["filled"][limit] / total["missing"] if total["missing"] else 0.0 for limit in limits}
    total["live_share"] = {limit: total["live_filled"][limit] / total["live_missing"] if total["live_missing"] else 0.0
                           for limit in limits}
    return total


def is_live(labels):
    return np.isin(np.asarray(labels), pm.POSSESSION_LABELS)


def coverage(dfl, skill, parsed, *, raw, recipe_folder=RECIPE):
    """D28's coverage record: per development group, the missing-ball frames each K fills. No scoring."""
    require_external_only()
    from dfl_submission import augment, metrica_sample as ms, metrica_views as mv, pff, vendor_trial
    output = recipe_folder / "coverage.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D28 coverage is recorded once")
    rows = {}
    for half in augment.guard_halves(dfl, skill, "development", raw=raw):
        group = half.source if half.source != "dfl" else f"dfl_{half.view}"
        rows.setdefault(group, []).append(counts(half.ball, is_live(half.labels)))
    games = vendor_trial.development()
    for game in games:
        for half in pff.load(parsed, game):
            rows.setdefault("pff", []).append(counts(half["ball"], is_live(half["labels"])))
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            rows.setdefault("metrica_clean", []).append(counts(half["ball"], is_live(half["labels"])))
            for view, (_, ball) in mv.build_views(half, game, index).items():
                rows.setdefault(f"d24_{view}", []).append(counts(ball, is_live(half["labels"])))
    groups = {group: summarise(items) for group, items in rows.items()}
    write_new(output, {"protocol": PROTOCOL, "grid": list(GRID), "groups": groups,
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                                  "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json")},
                       "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    for group, total in groups.items():
        print(f"D28 coverage {group}: missing {total['missing']} of {total['frames']} frames "
              f"(live {total['live_missing']} of {total['live']}); filled share "
              + ", ".join(f"K={limit} {share:.3f}" for limit, share in total["share"].items())
              + "; live " + ", ".join(f"K={limit} {share:.3f}" for limit, share in total["live_share"].items()),
              flush=True)
    return groups


# -- forecasts -------------------------------------------------------------------------------------

def candidate_halves(dfl, skill, parsed, games, split, setting, limit, *, raw, cache):
    """D15's decoder inputs with the `limit` fill on the DFL/SkillCorner `split` and PFF `games`, cached."""
    from dfl_submission import scale_trial, vendor_trial
    key = {"variant": variant(limit), "scale_models": scale_trial.identity(setting),
           "teacher_sha256": diagnosis.teacher_identity(), "split": split, "dfl_raw": raw,
           "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}
    model = scale_trial.teacher(setting)
    hook = signals_with(limit)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw, signals_of=hook)
        yield from diagnosis.pff_halves(parsed, games, model, signals_of=hook)

    return diagnosis.collect(cache, key, produce)


def decoded(halves, recipe):
    """{game: [labels of half 0, labels of half 1]} of Metrica halves, decoded at `recipe`."""
    result = {}
    for half in sorted(halves, key=lambda item: (int(item.match), item.index)):
        result.setdefault(int(half.match), []).append(diagnosis.decode(half.signals, recipe))
    return result


# -- gates -----------------------------------------------------------------------------------------

def eligible(change, metrica_passed=None, d24_change=None):
    """D28's conditions; development also passes the Metrica veto and the D24 change."""
    from dfl_submission import refine_trial
    result = refine_trial.passed(change, gain=GAIN)
    if metrica_passed is not None:
        result = result and bool(metrica_passed)
    if d24_change is not None:
        result = result and d24_change >= 0
    return bool(result)


def select(entries):
    """The eligible limit with the highest primary change, the smaller limit on a tie; None if none."""
    candidates = [entry for entry in entries if entry["eligible"]]
    if not candidates:
        return None
    return max(candidates, key=lambda entry: (entry["changes"]["primary"], -entry["limit"]))["limit"]


def summary_line(prefix, trial, change, extra, verdict):
    from dfl_submission import refine_trial
    return (f"{prefix}: N = 3 standard {refine_trial.standard(trial, 'aba3'):.5f} ({change['primary']:+.5f}), "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items()) + f", PFF {change['pff']:+.4f}"
            + extra + f" -> {verdict}")


# -- development, confirmation and build -----------------------------------------------------------

def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root=CACHE, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, refine_trial, scale_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D28 development is scored once")
    if not (recipe_folder / "coverage.json").exists():
        raise ValueError("D28 records its coverage before any scoring; run `coverage` first")
    halves, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)  # stops unless reproduced
    _, d19_halves, _ = ms.reference_scores("D15")                     # stops unless D15 reproduces its D19 record
    d24_reference, d24_halves, truths = mv.reference_scores()         # and its D24 record
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    setting, frozen = zoo.d15_setting()
    games = vendor_trial.development()
    d24_base = mv.d24_mean(d24_reference)
    entries = []
    for limit in GRID:
        folder = cache_root / f"k{limit}"
        candidate = candidate_halves(dfl, skill, parsed, games, "development", setting, limit, raw=raw,
                                     cache=folder / "development")
        trial = aligned_trial.evaluate(candidate, recipe)
        change = refine_trial.changes(base, trial)
        _, metrica = ms.reference_halves("D15", cache=folder / "metrica", signals_of=signals_with(limit),
                                         variant=variant(limit))
        veto = ms.veto(decoded(metrica, recipe), reference="D15")
        views = mv.view_halves(cache=folder / "d24", signals_of=signals_with(limit), variant=variant(limit))
        view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, recipe))
        d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
        d24["change"] = d24["candidate"] - d24["reference"]
        allowed = eligible(change, veto["changes"]["passed"], d24["change"])
        entries.append({"limit": limit, "eligible": allowed, "changes": change, "trial": trial,
                        "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                        "d24": dict(d24, views=view_scores)})
        print(summary_line(f"D28 development K = {limit}", trial, change,
                           f", Metrica {veto['changes']['primary']:+.4f} (veto {'passed' if veto['changes']['passed'] else 'failed'})"
                           f", D24 {d24['change']:+.4f}", "eligible" if allowed else "not eligible"), flush=True)
    selected = select(entries)
    write_new(output, {"protocol": PROTOCOL, "grid": list(GRID), "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected, "models": frozen["models"],
                       "teacher_sha256": diagnosis.teacher_identity(),
                       "variants": {str(limit): variant(limit) for limit in GRID},
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D28 development selected {'K = ' + str(selected) if selected is not None else 'no limit; D15 stays'}",
          flush=True)
    return selected


def set_b_halves(parsed, setting, cache):
    """D15's decoder inputs on PFF confirmation set B, computed once and cached by identity."""
    from dfl_submission import scale_trial, vendor_trial
    games = vendor_trial.split_games("confirm_b")
    identity = {"protocol": PROTOCOL, "scale_models": scale_trial.identity(setting),
                "teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games)}
    model = scale_trial.teacher(setting)
    return diagnosis.collect(cache, identity, lambda: diagnosis.pff_halves(parsed, games, model)), games


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root=CACHE, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, refine_trial, scale_trial, vendor_trial, zoo
    development = json.loads((recipe_folder / "development.json").read_text())
    if development["selected"] is None:
        raise ValueError("No D28 limit was eligible on development; D15 stays and nothing is confirmed")
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D28 confirmation is scored once")
    limit = development["selected"]
    setting, frozen = zoo.d15_setting()
    recipe = frozen["recipe"]
    if development["recipe"] != recipe or development["variants"][str(limit)] != variant(limit):
        raise ValueError("D15's recipe or the fill differs from the one D28 development used")
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting, raw=raw,
                                           cache=d15_holdout_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full["external"]) != stored["external"]
            or vendor_trial.canonical(full["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    external = [half for half in reference if half.source != "pff"]
    set_b, games_b = set_b_halves(parsed, setting, cache_root / "d15_set_b")
    base = aligned_trial.evaluate(external + set_b, recipe)
    candidate = candidate_halves(dfl, skill, parsed, games_b, "holdout", setting, limit, raw=raw,
                                 cache=cache_root / f"k{limit}" / "holdout")
    trial = aligned_trial.evaluate(candidate, recipe)
    change = refine_trial.changes(base, trial)
    result = eligible(change)
    write_new(output, {"protocol": PROTOCOL, "limit": limit, "variant": variant(limit), "recipe": recipe,
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": result, "models": scale_trial.identity(setting),
                       "inputs": {"set_b": vendor_trial.inputs(parsed, games_b)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(summary_line(f"D28 confirmation K = {limit} (PFF = set B)", trial, change, "",
                       "passed" if result else "failed"), flush=True)
    return result


def build(output, recipe_folder=RECIPE):
    """The competition file: the confirmed fill, then D15, then D27's kick-off rule."""
    from dfl_submission import kickoff
    confirmation = json.loads((recipe_folder / "confirmation.json").read_text())
    if confirmation["passed"] is not True:
        raise ValueError("D28 did not pass confirmation; build D27 instead")
    labels = kickoff.d15_labels(prepare=filler(confirmation["limit"]))
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("coverage", "develop", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        if name == "develop":
            command.add_argument("--d15-cache", type=Path, required=True)
        if name == "confirm":
            command.add_argument("--d15-holdout-cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "coverage":
        coverage(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache)


if __name__ == "__main__":
    main()
