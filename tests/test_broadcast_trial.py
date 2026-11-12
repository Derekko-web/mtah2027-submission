"""D31's selection rules, thinning, competition criteria and merge hook on synthetic data."""

import numpy as np

from dfl_submission import broadcast_ball as bb, broadcast_trial as bt, two_camera as tc
from tests.test_broadcast_ball import TEAMS, match


def entry(cost, margin, accepted, wrong):
    return {"max_cost": cost, "min_margin": margin, "accepted": accepted, "wrong": wrong}


def test_constants_maximise_accepted_frames_with_at_most_one_percent_wrong():
    entries = [entry(3.0, .25, 10000, 500), entry(2.0, .5, 9000, 90), entry(1.5, .5, 8000, 10),
               entry(2.0, 1.0, 9000, 20), entry(1.0, 2.0, 0, 0)]
    assert bt.select_constants(entries) == {"max_cost": 2.0, "min_margin": 1.0}


def test_constants_tie_goes_to_larger_margin_then_smaller_cost():
    entries = [entry(2.0, .5, 9000, 0), entry(1.5, .5, 9000, 0), entry(3.0, .25, 9000, 0)]
    assert bt.select_constants(entries) == {"max_cost": 1.5, "min_margin": .5}


def test_no_qualifying_constants_stop_d31():
    assert bt.select_constants([entry(3.0, .25, 1000, 11), entry(1.0, 2.0, 0, 0)]) is None


def test_the_grid_is_the_declared_one():
    assert bt.GRID == {"max_cost": (1.0, 1.5, 2.0, 3.0), "min_margin": (.25, .5, 1.0, 2.0)}
    assert bt.LEVELS == (.1, .2, .3, .4) and bt.GAIN == .01 and bt.WRONG_SHARE == .01


def stretch(start, stop, offset, accepted=True):
    return {"start": start, "stop": stop, "offset": offset, "orientation": 0, "accepted": accepted}


def test_thinning_keeps_stretches_in_seeded_order_and_cuts_the_last_one():
    view = np.full((100, 2), np.nan)
    view[::2] = 0.                                     # 50 frames with a missing ball
    broadcast = np.ones((100, 2))
    results = [stretch(0, 20, 0), stretch(40, 60, 0), stretch(70, 90, 0), stretch(90, 100, 0, accepted=False)]
    ball, info = bt.thin(view, broadcast, results, .3, np.random.default_rng(5))
    assert info == {"hidden": 50, "target": 15, "kept": 15, "shortfall": 0}
    filled = np.flatnonzero(np.isfinite(ball).all(axis=1) & ~np.isfinite(view).all(axis=1))
    assert len(filled) == 15
    order = np.random.default_rng(5).permutation(3)
    first = [stretch(0, 20, 0), stretch(40, 60, 0), stretch(70, 90, 0)][order[0]]
    assert ((filled >= first["start"]) & (filled < first["stop"])).sum() == 10   # the first in order is kept whole
    assert np.array_equal(ball[np.isfinite(view).all(axis=1)], view[np.isfinite(view).all(axis=1)])


def test_thinning_records_a_shortfall_when_recovery_is_short():
    view = np.full((100, 2), np.nan)
    ball, info = bt.thin(view, np.ones((100, 2)), [stretch(0, 10, 0)], .4, np.random.default_rng(0))
    assert info == {"hidden": 100, "target": 40, "kept": 10, "shortfall": 30}


def test_r_star_is_the_smallest_level_gaining_a_point_on_both_groups():
    gains = {.1: {"dfl": .02, "metrica": .005}, .2: {"dfl": .011, "metrica": .01}, .3: {"dfl": .03, "metrica": .03},
             .4: {"dfl": .04, "metrica": .04}}
    assert bt.select_level(gains) == .2
    assert bt.select_level({.1: {"dfl": .0, "metrica": .02}, .2: {"dfl": .009, "metrica": .02}}) is None


def test_confirmation_needs_a_point_on_average_and_no_match_lower():
    assert bt.confirmation_passed({"WQQ": .015, "WR9": .006})
    assert not bt.confirmation_passed({"WQQ": .03, "WR9": -.0001})
    assert not bt.confirmation_passed({"WQQ": .012, "WR9": .007})


def report(n=10000, mapped=.5, recovered=.3, agreement=(1.0, 4.0), count=3000, teams=True):
    rng = np.random.default_rng(0)
    missing = np.zeros(n, dtype=bool)
    missing[::2] = True
    filled = np.zeros(n, dtype=bool)
    filled[np.flatnonzero(missing)[:int(recovered * missing.sum())]] = True
    covered = np.zeros(n, dtype=bool)
    covered[:int(mapped * n)] = True
    median, p90 = agreement
    low, top = count // 2, count * 15 // 100
    values = np.r_[np.full(low, median * .99), np.full(count - low - top, median * 1.01), np.full(top, p90)]
    distances = np.full(n, np.nan)
    distances[:count] = rng.permutation(values)
    return {"teams": teams, "mapped": covered, "filled": filled, "missing": missing, "distances": distances}


def test_competition_criteria_pass_and_name_each_failure():
    graded = np.ones(10000, dtype=bool)
    assert bt.criteria(report(), graded, .2) == []
    assert bt.criteria(report(mapped=.29), graded, .2) == ["coverage"]
    assert bt.criteria(report(agreement=(1.6, 4.0)), graded, .2) == ["agreement"]
    assert bt.criteria(report(agreement=(1.0, 5.2)), graded, .2) == ["agreement"]
    assert bt.criteria(report(count=1999), graded, .2) == ["agreement"]
    assert bt.criteria(report(recovered=.19), graded, .2) == ["recovery"]
    assert bt.criteria(report(teams=False), graded, .2) == ["teams"]


def test_competition_criteria_count_graded_frames_only():
    graded = np.zeros(10000, dtype=bool)
    graded[5000:] = True                               # every mapped frame is outside the graded ones
    assert "coverage" in bt.criteria(report(mapped=.5), graded, .2)


def test_merge_hook_fills_only_missing_balls_and_reports_its_coverage():
    from tests.test_broadcast_ball import broadcast_view
    pitch = np.array([105.0, 68.0])
    centred = match(3000, seed=6)
    positions = (centred + pitch / 2).astype(np.float32)          # uncentred, as the build has them
    ball = positions[:, 3].copy()
    ball[1000:1400] = np.nan
    broadcast = broadcast_view(centred, np.arange(3000), 1)
    broadcast_ball = bb.orient(centred[:, 3] + .5, 1)
    found = {}
    merged = bt.merge_arrays(positions, ball, TEAMS, broadcast, broadcast_ball, TEAMS, np.ones(3000, dtype=bool),
                             pitch, {"max_cost": 3.0, "min_margin": .25}, report=found)
    seen = np.isfinite(ball).all(axis=1)
    assert np.array_equal(merged[seen], ball[seen]) and merged.dtype == np.float32
    assert set(found) >= {"results", "mapped", "filled", "missing", "distances"}
    assert found["filled"].sum() > 300 and not found["filled"][seen].any()
    assert np.abs(merged[found["filled"]] - (positions[found["filled"], 3] + .5)).max() < 1e-3
    both = np.isfinite(found["distances"])
    assert both.sum() > 1000 and not both[~seen].any() and np.allclose(found["distances"][both], np.sqrt(.5), atol=1e-3)


def synthetic_source(frames=200):
    rng = np.random.default_rng(1)
    positions = rng.uniform(0, 100, (frames, 22, 2)).astype(np.float32)
    ball = rng.uniform(0, 60, (frames, 2)).astype(np.float32)
    ball[::4] = np.nan
    ball[5] = [500., 500.]                                    # outside the pitch: masked by the build
    return {"positions": positions, "ball": ball, "teams": TEAMS, "valid": np.ones(frames, dtype=bool)}


def tracking_halves(merge_ball=None):
    import broadcast_alignment
    from dfl_submission import build
    originals = broadcast_alignment.load_source, build.half_video_blocks
    broadcast_alignment.load_source = lambda source: synthetic_source()
    build.half_video_blocks = lambda game: [(1, 10, 99), (2, 120, 189)]
    try:
        options = {} if merge_ball is None else {"merge_ball": merge_ball}
        return list(build.tactical_tracking_halves(centered=True, pitch=(105.0, 68.0), **options))
    finally:
        broadcast_alignment.load_source, build.half_video_blocks = originals


def test_merge_ball_hook_sees_the_masked_uncentred_ball_and_feeds_the_halves():
    calls = {}

    def hook(positions, ball, teams, pitch):
        calls.update(ball=ball.copy(), pitch=np.asarray(pitch))
        out = ball.copy()
        out[np.isnan(out).any(axis=1)] = [10., 20.]
        return out

    plain_halves = tracking_halves()
    merged_halves = tracking_halves(hook)
    assert np.isnan(calls["ball"][5]).all() and np.isnan(calls["ball"][::4]).all()
    assert np.array_equal(calls["pitch"], [105.0, 68.0])
    for (_, span, (positions, ball, _, _)), (_, span2, (positions2, ball2, _, _)) in zip(plain_halves, merged_halves):
        assert span == span2 and np.array_equal(positions, positions2, equal_nan=True)
        seen = np.isfinite(ball).all(axis=1)
        assert np.array_equal(ball[seen], ball2[seen])
        assert np.allclose(ball2[~seen], np.array([10., 20.]) - np.array([105.0, 68.0]) / 2)


def test_cached_labels_are_decoded_at_d15s_recipe_not_the_stored_ones():
    from dfl_submission import diagnosis
    rng = np.random.default_rng(0)
    signals = tuple(rng.uniform(0, 1, 500).astype(np.float32) for _ in range(4))
    recipe = {"cutoff": .5, "dead_cost": 4.0, "owner_cost": 8.0, "discount": .9}
    half = diagnosis.HalfSignals("dfl", "WPY", "masked", 0, np.zeros(500, dtype=np.int8), signals,
                                 np.zeros(500, dtype=np.int8))      # stored labels from another recipe
    cached = bt.cached_labels([half], recipe, lambda h: (h.view, h.index))
    assert np.array_equal(cached[("masked", 0)], diagnosis.decode(signals, recipe))


def test_second_attempt_settings_are_the_declared_ones():
    first, second = bt.ATTEMPTS[""], bt.ATTEMPTS["a"]
    assert first == {"prefix": "", "seed": 900000, "grid": bt.GRID, "trim": 0.0, "translate": False}
    assert second == {"prefix": "a_", "seed": 905000, "grid": {"max_cost": (.75, 1.0, 1.5, 2.0, 3.0),
                                                                "min_margin": (.25, .5, 1.0, 2.0)},
                      "trim": .25, "translate": True}


def test_merge_hook_removes_the_stretch_shift_when_constants_translate():
    from tests.test_broadcast_ball import broadcast_view
    pitch = np.array([105.0, 68.0])
    centred = match(3000, seed=11)
    positions = (centred + pitch / 2).astype(np.float32)
    ball = positions[:, 3].copy()
    ball[1000:1400] = np.nan
    bias = np.array([1.5, -1.0])
    broadcast = broadcast_view(centred, np.arange(3000), 0) + bias.astype(np.float32)
    broadcast_ball = centred[:, 3] + bias
    constants = {"max_cost": 3.0, "min_margin": .25, "trim": .25, "translate": True}
    found = {}
    merged = bt.merge_arrays(positions, ball, TEAMS, broadcast, broadcast_ball, TEAMS, np.ones(3000, dtype=bool),
                             pitch, constants, report=found)
    filled = found["filled"]
    assert filled.sum() > 300
    assert np.abs(merged[filled] - positions[filled, 3]).max() < .3
    both = np.isfinite(found["distances"])
    assert np.median(found["distances"][both]) < .3
