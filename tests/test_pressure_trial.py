"""D39 plumbing (dfl_submission/pressure_trial.py)."""

import numpy as np

from dfl_submission import pressure_trial as pt, second_trial as st


def test_constants():
    assert pt.OWNER_COSTS == (8, 10) and pt.PROTOCOL == "DECISIONS.md D39"
    assert pt.network_folders() == {key: pt.MODELS / key for key in ("current", "augmented")}


def test_table_appends_the_twelve_columns_to_the_long_context_table():
    from dfl_submission import long_context
    rng = np.random.default_rng(3)
    length = 600
    positions = rng.normal(0, 20, size=(length, 22, 2)).cumsum(axis=0) * .01
    ball = positions[:, 0] + rng.normal(0, .5, size=(length, 2))
    ball[100:160] = np.nan
    teams = np.array([0] * 11 + [1] * 11)
    pitch = np.array([105., 68.])
    frames, owner, dead = pt.table(positions, ball, teams, pitch)
    base_frames, base, _ = long_context.features(positions, ball, teams, pitch)
    assert np.array_equal(frames, base_frames)
    assert owner.columns.tolist()[:len(base.columns)] == base.columns.tolist()
    assert len(owner.columns) == len(base.columns) + 12
    assert all(column in dead.columns for column in ("count_5_min", "count_10_max"))
    assert st.SETTING == {"lr_scale": 2., "batch": 32, "dropout": .2}
