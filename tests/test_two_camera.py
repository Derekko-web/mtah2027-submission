"""D31's synthetic two-camera harness on synthetic truth: timeline, camera, ball error and determinism."""

import numpy as np

from dfl_submission import broadcast_ball as bb, two_camera as tc
from tests.test_broadcast_ball import TEAMS, match


def truth(frames=40000, seed=0):
    positions = match(frames, seed)
    ball = positions[:, 3].copy()
    return positions, ball


def test_parameters_are_the_ones_d31_fixes():
    assert tc.OFFSET_RANGE == 3000 and tc.LIVE_MIN == 125 and tc.LIVE_MEAN == 900
    assert tc.BREAKS == (("cut", .5), ("replay", .3), ("slow", .2))
    assert tc.CUT_FRAMES == (50, 500) and tc.REPLAY_FRAMES == (75, 200) and tc.REPLAY_LAG == (125, 375)
    assert tc.WIPE_PROBABILITY == .5 and tc.WIPE_FRAMES == 25
    assert tc.WINDOW == 50 and tc.SMOOTHING == 25 and tc.DROP == (.1, 25) and tc.BIAS == .5 and tc.NOISE == .3
    assert tc.BALL_RUNS == (100, 25) and tc.BALL_ERROR == (1.5, 5.0)


def test_live_frames_show_the_tactical_frame_at_one_offset():
    positions, ball = truth()
    recording = tc.simulate(positions, ball, TEAMS, seed=7)
    live = recording["kind"] == tc.KINDS["live"]
    frames = np.flatnonzero(live)
    assert live.mean() > .5
    assert np.array_equal(recording["shows"][frames], frames + recording["offset"])
    assert abs(recording["offset"]) <= 3000 and recording["orientation"] in range(4)
    assert recording["valid"][frames].all()


def test_cuts_and_wipes_have_no_homography_and_no_players():
    positions, ball = truth()
    recording = tc.simulate(positions, ball, TEAMS, seed=8)
    off = np.isin(recording["kind"], (tc.KINDS["cut"], tc.KINDS["wipe"]))
    assert off.any() and not recording["valid"][off].any()
    assert np.isnan(recording["positions"][off]).all() and np.isnan(recording["ball"][off]).all()


def test_replays_show_earlier_frames_at_normal_or_half_speed():
    positions, ball = truth()
    recording = tc.simulate(positions, ball, TEAMS, seed=9)
    for name, step in (("replay", 1.0), ("slow", .5)):
        frames = np.flatnonzero(recording["kind"] == tc.KINDS[name])
        assert len(frames)
        steps = np.diff(recording["shows"][frames])
        assert np.isclose(np.median(steps), step)
        assert (recording["shows"][frames] < frames + recording["offset"]).all()


def test_broadcast_players_are_the_oriented_truth_inside_the_window():
    positions, ball = truth()
    recording = tc.simulate(positions, ball, TEAMS, seed=10)
    frames = np.flatnonzero(recording["kind"] == tc.KINDS["live"])
    shown = bb.orient(recording["positions"][frames], recording["orientation"])
    true = positions[recording["shows"][frames].astype(int)]
    seen = np.isfinite(shown).all(axis=2)
    assert .2 < seen.mean() < .9
    error = np.linalg.norm(shown - true, axis=2)[seen]
    assert np.median(error) < 1.5 and np.percentile(error, 99) < 4


def test_broadcast_ball_error_has_the_declared_median_and_90th_percentile():
    positions, ball = truth(80000, seed=1)
    recording = tc.simulate(positions, ball, TEAMS, seed=11)
    frames = np.flatnonzero((recording["kind"] == tc.KINDS["live"]) & np.isfinite(recording["ball"]).all(axis=1))
    shown = bb.orient(recording["ball"][frames], recording["orientation"])
    error = np.linalg.norm(shown - ball[recording["shows"][frames].astype(int)], axis=1)
    assert abs(np.median(error) - 1.5) < .1 and abs(np.percentile(error, 90) - 5.0) < .4


def test_unseen_true_ball_is_unseen_on_the_broadcast():
    positions, ball = truth()
    ball[::3] = np.nan
    recording = tc.simulate(positions, ball, TEAMS, seed=12)
    frames = np.flatnonzero(recording["kind"] == tc.KINDS["live"])
    hidden = ~np.isfinite(ball[recording["shows"][frames].astype(int)]).all(axis=1)
    assert np.isnan(recording["ball"][frames[hidden]]).all()


def test_simulation_is_deterministic_by_seed():
    positions, ball = truth(20000)
    a, b = tc.simulate(positions, ball, TEAMS, seed=3), tc.simulate(positions, ball, TEAMS, seed=3)
    c = tc.simulate(positions, ball, TEAMS, seed=4)
    for key in ("positions", "ball", "shows"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["shows"], c["shows"], equal_nan=True)


def test_wrong_frames_count_offsets_more_than_two_frames_off_or_a_wrong_orientation():
    shows = np.r_[np.arange(100, 110), 300 + np.arange(10) / 2].astype(float)
    results = [{"start": 0, "stop": 10, "offset": 102, "orientation": 1, "accepted": True},
               {"start": 10, "stop": 20, "offset": 292, "orientation": 1, "accepted": True}]
    counts = tc.frame_counts(results, shows, orientation=1)
    # first stretch: mapped 102..111 vs shown 100..109, 2 off: right. Second: mapped 302..311 against
    # shown 300, 300.5, ..., 304.5 rounded half up (300, 301, 301, 302, ...): only the first two are within 2.
    assert counts == {"accepted": 20, "wrong": 8}
    assert tc.frame_counts(results, shows, orientation=0) == {"accepted": 20, "wrong": 20}
