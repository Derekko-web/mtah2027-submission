"""D39 pressure features on synthetic tracks (dfl_submission/pressure_features.py)."""

import numpy as np

from dfl_submission import pressure_features as pf


def scene(length=40):
    ball = np.zeros((length, 2))
    ball[:, 0] = np.arange(length) * .2             # ball moving at 5 m/s along x
    positions = np.full((length, 4, 2), np.nan)
    positions[:, 0] = ball + [1.0, 0]               # home carrier, 1 m ahead, moving with the ball
    positions[:, 1, 0], positions[:, 1, 1] = 20.0, 0.  # home team-mate, static far away
    positions[:, 2] = np.c_[ball[:, 0] + 8 - np.arange(length) * .1, np.zeros(length)]  # away closing in
    positions[:, 3] = [[-30., 0.]] * length          # away static far away
    return positions, ball, np.array([0, 0, 1, 1])


def test_columns_are_paired_and_counts_follow_the_radii():
    positions, ball, teams = scene()
    table = pf.features(positions, ball, teams, np.arange(0, 40, 5))
    assert sorted(table.columns) == sorted(p + n for p in pf.PREFIXES for n in
                                           ("second_closest", "count_5", "count_10", "approach_3", "nearest_speed",
                                            "nearest_alignment"))
    assert (table["h__count_5"] == 1).all() and (table["a__count_5"].iloc[-1] == 1)


def test_motion_columns():
    positions, ball, teams = scene()
    table = pf.features(positions, ball, teams, np.arange(0, 40, 5))
    later = table.iloc[1:]
    assert np.isnan(table["h__nearest_speed"].iloc[0])           # no past at frame 0
    assert np.allclose(later["h__nearest_speed"], 5.0, atol=1e-5)
    assert np.allclose(later["h__nearest_alignment"], 1.0, atol=1e-5)
    # Away: the closing player approaches at 2.5 m/s, the static one recedes at 5 m/s; both are among its 3 nearest.
    assert np.allclose(later["a__approach_3"], (2.5 - 5.0) / 2, atol=1e-4)
    assert np.allclose(later["h__second_closest"], 20 - (np.arange(5, 40, 5) * .2), atol=1e-4)


def test_unseen_ball_gives_missing_values():
    positions, ball, teams = scene()
    ball[10:20] = np.nan
    table = pf.features(positions, ball, teams, np.array([10, 15, 20, 25]))
    assert table.iloc[:2].isna().all().all()
    assert np.isnan(table["h__nearest_speed"].iloc[2])          # the ball five frames earlier is unseen
    assert table["h__count_5"].iloc[3] == 1


def test_swapping_teams_swaps_the_pairs():
    positions, ball, teams = scene()
    frames = np.arange(0, 40, 5)
    table = pf.features(positions, ball, teams, frames)
    swapped = pf.features(positions, ball, 1 - teams, frames)
    for name in ("count_10", "approach_3", "nearest_speed"):
        assert np.array_equal(table["h__" + name].to_numpy(), swapped["a__" + name].to_numpy(), equal_nan=True)
