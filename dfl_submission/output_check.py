"""C2: a check of the final file without ground truth (DECISIONS.md C2).

`ranges` computes three statistics on D15's predictions for 28 external match views and writes their
plausible ranges once. `check` computes the same statistics on a competition file and prints PASS,
or FAIL with the names of the statistics outside their ranges. It never prints a number, so the
statistics of the competition file reach no AI tool.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm


PACKAGE = Path(__file__).resolve().parent
RANGES = PACKAGE / "output_check_ranges.json"
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


def write_ranges(dfl, skill, parsed, *, raw, d15_cache, output=RANGES):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    computed = commands.add_parser("ranges")
    computed.add_argument("--dfl", type=Path, required=True)
    computed.add_argument("--dfl-raw", action="store_true")
    computed.add_argument("--skill", type=Path, required=True)
    computed.add_argument("--parsed", type=Path, required=True)
    computed.add_argument("--d15-cache", type=Path, required=True)
    checked = commands.add_parser("check")
    checked.add_argument("csv", type=Path)
    args = parser.parse_args()
    if args.command == "ranges":
        write_ranges(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache)
    else:
        raise SystemExit(0 if check(args.csv) else 1)


if __name__ == "__main__":
    main()
