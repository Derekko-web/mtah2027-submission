"""D16 last-touch and possession-chain features on synthetic tracking."""

import numpy as np

from dfl_submission import touch_features

FPS = 25


def scene(length=500):
    """Home player at the origin, away player at (20, 0); the ball comes in at 10 m/s, turns at frame 200
    half a metre from the home player, and leaves at 10 m/s at a right angle."""
    positions = np.array([[[0.0, 0.0], [20.0, 0.0]]] * length, dtype=np.float32)
    teams = np.array([0, 1])
    t = np.arange(length, dtype=np.float32)
    ball = np.zeros((length, 2), dtype=np.float32)
    before = t <= 200
    ball[before, 0] = -.5 - 10 * (200 - t[before]) / FPS
    ball[before, 1] = .0
    ball[~before, 0] = -.5
    ball[~before, 1] = 10 * (t[~before] - 200) / FPS
    return positions, ball, teams


def test_turn_next_to_a_home_player_is_a_home_touch_only():
    positions, ball, teams = scene()
    frames, touching = touch_features.touches(positions, ball, teams)
    assert len(frames) == 1 and abs(frames[0] - 200) <= 1 and touching[0] == 0
    table = touch_features.features(positions, ball, teams, np.arange(0, len(ball), 5))
    after = table.iloc[45]   # frame 225, one second after the touch
    assert after["h__touches_past_3"] == 1 and after["a__touches_past_3"] == 0
    assert np.isclose(after["h__touch_since"], 1.0, atol=.05) and after["a__touch_since"] == touch_features.CAP
    assert np.isclose(after["h__touch_run_past"], 1.0, atol=.05) and after["a__touch_run_past"] == 0
    early = table.iloc[30]   # frame 150, two seconds before it
    assert np.isclose(early["h__touch_until"], 2.0, atol=.05) and early["h__touches_future_3"] == 1
    assert table["h__touch_evidence_5"].max() > 4 and np.nanmax(table["a__touch_evidence_5"]) < .01


def test_swapping_teams_swaps_the_touch_features():
    positions, ball, teams = scene()
    frames = np.arange(0, len(ball), 5)
    table = touch_features.features(positions, ball, teams, frames)
    swapped = touch_features.features(positions, ball, 1 - teams, frames)
    for name in table:
        other = "a__" + name[3:] if name.startswith("h__") else "h__" + name[3:] if name.startswith("a__") else name
        np.testing.assert_array_equal(table[name].to_numpy(), swapped[other].to_numpy())


def test_every_team_column_has_a_mirror_and_unseen_ball_is_missing():
    positions, ball, teams = scene()
    ball[300:340] = np.nan
    table = touch_features.features(positions, ball, teams, np.arange(0, len(ball), 5))
    names = set(table)
    assert all(("a__" + name[3:]) in names for name in names if name.startswith("h__"))
    assert all(("h__" + name[3:]) in names for name in names if name.startswith("a__"))
    assert table.iloc[62].isna().all() and not table.iloc[50].isna().all()
    assert all(str(dtype) == "float32" for dtype in table.dtypes)


def test_runs_of_same_team_touches():
    frames = np.array([100, 150, 200, 300])
    teams = np.array([0, 0, 0, 1])
    timeline = np.arange(0, 400)
    runs = touch_features.chain_features(frames, teams, timeline)
    at = lambda name, frame: runs[name][frame]
    assert np.isclose(at("h__touch_run_past", 175), 75 / FPS) and np.isclose(at("h__touch_run_future", 175), 25 / FPS)
    assert at("a__touch_run_past", 175) == 0 and np.isclose(at("a__touch_run_future", 250), 50 / FPS)
    assert at("h__touch_run_future", 250) == 0 and np.isclose(at("h__touch_run_past", 250), 150 / FPS)
    assert at("h__touches_past_3", 210) == 2 and at("h__touches_past_9", 210) == 3 and at("h__touches_future_9", 50) == 3
    assert at("a__touches_past_3", 299) == 0 and at("a__touches_past_3", 300) == 1
