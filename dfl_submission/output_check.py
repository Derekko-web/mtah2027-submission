"""C2: a check of the final file without ground truth (DECISIONS.md C2).

`ranges` computes three statistics on D15's predictions for 28 external match views and writes their
plausible ranges once; `ranges-d43` and `ranges-d48` do the same for the models that replaced D15
(C2's amendments), and `ranges-d53` for the D53 fallback (not the default; `check --ranges` names
its file if the fallback ships). `check` computes the same statistics on a competition file and prints PASS, or
FAIL with the names of the statistics outside their ranges (by default the newest ranges, D48's). It
never prints a number, so the statistics of the competition file reach no AI tool.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm


PACKAGE = Path(__file__).resolve().parent
D15_RANGES = PACKAGE / "output_check_ranges.json"        # C2 as first declared (D15, then D27)
D43_RANGES = PACKAGE / "output_check_ranges_d43.json"   # C2 re-declared for D43
RANGES = PACKAGE / "output_check_ranges_d48.json"       # C2 re-declared for D48 (the stack), read by `check`
PROTOCOL = "DECISIONS.md C2"
FPS = 25
MINUTE = 60 * FPS
NINETY = 90 * MINUTE
BRIEF_FRAMES = 25
LOW, HIGH = .5, 1.5
STATISTICS = ("possessions_per_minute", "dead_share", "sub_second_stoppages_per_90")


def statistics(codes):
    """C2's three statistics of one label sequence, counted inside its own non-X runs."""
    codes = np.asarray(codes)
    runs = pm.valid_runs(codes)
    graded = sum(stop - start for start, stop in runs)
    if not graded:
        return {name: float("nan") for name in STATISTICS}  # no graded frame: outside every range
    parts = pm.segments(codes, runs)
    spells = sum(code in pm.POSSESSION_LABELS for _, _, code in parts)
    brief = sum(code == pm.D and stop - start < BRIEF_FRAMES for start, stop, code in parts)
    return {"possessions_per_minute": spells / (graded / MINUTE), "dead_share": float(np.sum(codes == pm.D)) / graded,
            "sub_second_stoppages_per_90": brief / (graded / NINETY)}


def ranges_from(values):
    """[half the smallest, 1.5 x the largest] of each statistic over the external match views."""
    return {name: [LOW * min(item[name] for item in values), HIGH * max(item[name] for item in values)]
            for name in STATISTICS}


def failed(stats, ranges):
    return sorted(name for name in STATISTICS if not ranges[name][0] <= stats[name] <= ranges[name][1])


def check(csv, ranges=RANGES):
    """PASS, or FAIL with the failed statistics' names. Prints no number."""
    outside = failed(statistics(pm.label_codes(pm.read_label_csv(csv))), json.loads(Path(ranges).read_text())["ranges"])
    print("PASS" if not outside else "FAIL: " + ", ".join(outside))
    return not outside


def external_predictions(dfl, skill, parsed, *, raw, d15_cache):
    """{view name: D15's labels over both halves} for the 28 external match views C2 names."""
    from dfl_submission import diagnosis, metrica_sample as ms, metrica_views as mv, stoppage_trial, zoo
    halves, recipe, _ = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)  # stops unless reproduced
    _, d19_halves, _ = ms.reference_scores("D15")
    _, d24_halves, _ = mv.reference_scores()
    predict = stoppage_trial.predictor(recipe)
    found = {}
    for group in (halves, d19_halves, d24_halves):
        for key, (_, prediction, _) in diagnosis.matches(group, predict).items():
            found["/".join(key)] = prediction
    return found


def write_ranges(dfl, skill, parsed, *, raw, d15_cache, output=D15_RANGES):
    from dfl_submission.external_only import require_external_only
    require_external_only()
    if output.exists():
        raise FileExistsError(f"{output} exists; C2's ranges are written once")
    predictions = external_predictions(dfl, skill, parsed, raw=raw, d15_cache=d15_cache)
    if len(predictions) != 28:
        raise ValueError(f"C2 names 28 external match views, found {len(predictions)}")
    values = {name: statistics(codes) for name, codes in sorted(predictions.items())}
    ranges = ranges_from(list(values.values()))
    output.write_text(json.dumps({"protocol": PROTOCOL, "candidate": "D15", "factors": [LOW, HIGH], "ranges": ranges,
                                  "external": values}, indent=2, sort_keys=True) + "\n")
    for name in STATISTICS:
        print(f"C2 {name}: external {min(v[name] for v in values.values()):.4f} to "
              f"{max(v[name] for v in values.values()):.4f}; range {ranges[name][0]:.4f} to {ranges[name][1]:.4f}")
    return ranges


def d43_predictions(dfl, skill, parsed, *, raw):
    """{view name: the player-set model's labels at cost 10} for the same 28 external match views (C2's D43 amendment)."""
    from dfl_submission import diagnosis, hyper_trial as ht, metrica_sample as ms, metrica_views as mv, player_trial
    from dfl_submission import stoppage_trial, vendor_trial
    cache = PACKAGE.parent / "data" / "d42" / "cache"
    halves = player_trial.candidate_halves(dfl, skill, parsed, vendor_trial.development(), "development", raw=raw,
                                           cache=cache / "development")
    model = player_trial.teacher()
    _, metrica = ms.reference_halves("D15", cache=cache / "metrica", model=model, signals_of=player_trial.half_signals,
                                     variant="D42 player set")
    views = mv.view_halves(cache=cache / "d24", signals_of=lambda _, *t: player_trial.half_signals(model, *t),
                           variant="D42 player set")
    predict = stoppage_trial.predictor(ht.recipe_at(player_trial.D43_COST))
    found = {}
    for group in (halves, metrica, views):
        for key, (_, prediction, _) in diagnosis.matches(group, predict).items():
            found["/".join(key)] = prediction
    return found


def write_d43_ranges(dfl, skill, parsed, *, raw, output=D43_RANGES):
    from dfl_submission.external_only import require_external_only
    require_external_only()
    if output.exists():
        raise FileExistsError(f"{output} exists; C2's D43 ranges are written once")
    predictions = d43_predictions(dfl, skill, parsed, raw=raw)
    if len(predictions) != 28:
        raise ValueError(f"C2 names 28 external match views, found {len(predictions)}")
    values = {name: statistics(codes) for name, codes in sorted(predictions.items())}
    ranges = ranges_from(list(values.values()))
    output.write_text(json.dumps({"protocol": PROTOCOL + " (amendment for D43)", "candidate": "D43 player-set model, cost 10",
                                  "factors": [LOW, HIGH], "ranges": ranges, "external": values}, indent=2, sort_keys=True) + "\n")
    for name in STATISTICS:
        print(f"C2 (D43) {name}: external {min(v[name] for v in values.values()):.4f} to "
              f"{max(v[name] for v in values.values()):.4f}; range {ranges[name][0]:.4f} to {ranges[name][1]:.4f}")
    return ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    computed = commands.add_parser("ranges")
    computed.add_argument("--dfl", type=Path, required=True)
    computed.add_argument("--dfl-raw", action="store_true")
    computed.add_argument("--skill", type=Path, required=True)
    computed.add_argument("--parsed", type=Path, required=True)
    computed.add_argument("--d15-cache", type=Path, required=True)
    later = commands.add_parser("ranges-d43")
    later.add_argument("--dfl", type=Path, required=True)
    later.add_argument("--dfl-raw", action="store_true")
    later.add_argument("--skill", type=Path, required=True)
    later.add_argument("--parsed", type=Path, required=True)
    stack = commands.add_parser("ranges-d48")
    stack.add_argument("--dfl", type=Path, required=True)
    stack.add_argument("--dfl-raw", action="store_true")
    stack.add_argument("--skill", type=Path, required=True)
    stack.add_argument("--parsed", type=Path, required=True)
    fallback = commands.add_parser("ranges-d53")
    fallback.add_argument("--dfl", type=Path, required=True)
    fallback.add_argument("--dfl-raw", action="store_true")
    fallback.add_argument("--skill", type=Path, required=True)
    fallback.add_argument("--parsed", type=Path, required=True)
    checked = commands.add_parser("check")
    checked.add_argument("csv", type=Path)
    checked.add_argument("--ranges", type=Path, default=RANGES, help="a ranges file (default: D48's)")
    args = parser.parse_args()
    if args.command == "ranges":
        write_ranges(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, output=D15_RANGES)
    elif args.command == "ranges-d43":
        write_d43_ranges(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, output=D43_RANGES)
    elif args.command == "ranges-d48":
        from dfl_submission import output_check_d48
        output_check_d48.write_d48_ranges(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, output=RANGES)
    elif args.command == "ranges-d53":
        from dfl_submission import output_check_d53
        output_check_d53.write_d53_ranges(args.dfl, args.skill, args.parsed, raw=args.dfl_raw,
                                          output=output_check_d53.RANGES)
    else:
        raise SystemExit(0 if check(args.csv, args.ranges) else 1)


if __name__ == "__main__":
    main()
