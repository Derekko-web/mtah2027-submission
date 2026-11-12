"""C2 on synthetic labels: the statistics, the ranges and a check that prints no number."""

import contextlib
import io
import json

import numpy as np

from dfl_submission import output_check as oc
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def test_statistics_count_inside_graded_runs():
    codes = labels((X, 100), (H, 1400), (D, 10), (A, 1490), (X, 50), (A, 1500), (D, 100), (H, 1500))
    stats = oc.statistics(codes)
    graded = 1400 + 10 + 1490 + 1500 + 100 + 1500
    assert stats["possessions_per_minute"] == 4 / (graded / 1500)  # A after X starts a new run, so a new spell
    assert stats["dead_share"] == 110 / graded
    assert stats["sub_second_stoppages_per_90"] == 1 / (graded / 135000)


def test_ranges_halve_the_smallest_and_widen_the_largest():
    values = [{"possessions_per_minute": 2.0, "dead_share": .3, "sub_second_stoppages_per_90": 0.0},
              {"possessions_per_minute": 4.0, "dead_share": .4, "sub_second_stoppages_per_90": 10.0}]
    ranges = oc.ranges_from(values)
    assert ranges["possessions_per_minute"] == [1.0, 6.0]
    assert ranges["dead_share"][0] == .15 and abs(ranges["dead_share"][1] - .6) < 1e-12
    assert ranges["sub_second_stoppages_per_90"] == [0.0, 15.0]


def test_failed_names_the_statistics_outside_their_ranges():
    ranges = {"possessions_per_minute": [1.0, 6.0], "dead_share": [.15, .6], "sub_second_stoppages_per_90": [0.0, 15.0]}
    ok = {"possessions_per_minute": 3.0, "dead_share": .3, "sub_second_stoppages_per_90": 2.0}
    assert oc.failed(ok, ranges) == []
    assert oc.failed(dict(ok, dead_share=.95, possessions_per_minute=.5), ranges) == ["dead_share", "possessions_per_minute"]


def test_check_prints_pass_or_fail_and_no_number(tmp_path):
    codes = labels((X, 10), (H, 3000), (D, 500), (A, 3000), (X, 10))
    csv = tmp_path / "file.csv"
    csv.write_text("frame,label\n" + "".join(f"{i},{pm.LABELS[c]}\n" for i, c in enumerate(codes)))
    stats = oc.statistics(codes)
    wide = {name: [0.0, value * 2 + 1] for name, value in stats.items()}
    narrow = dict(wide, dead_share=[.9, 1.0])
    for ranges, expected in ((wide, "PASS"), (narrow, "FAIL: dead_share")):
        path = tmp_path / "ranges.json"
        path.write_text(json.dumps({"ranges": ranges}))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = oc.check(csv, path)
        assert out.getvalue().strip() == expected and result is (expected == "PASS")
        assert not any(character.isdigit() for character in out.getvalue())
