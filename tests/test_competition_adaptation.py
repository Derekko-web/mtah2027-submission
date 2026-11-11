from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from dfl_submission import competition_adaptation as d6, motion_trial, retrain_trial, skillcorner_2020
from evaluation import possession_metrics as pm


def test_legacy_tracking_does_not_require_possession_fields():
    metadata = {"home_team": {"id": 1}, "away_team": {"id": 2},
                "players": [{"trackable_object": 13, "team_id": 1}, {"trackable_object": 14, "team_id": 2}],
                "ball": {"trackable_object": 55}, "pitch_length": 105, "pitch_width": 68}
    rows = [{"period": period, "frame": frame, "data": [
        {"trackable_object": 13, "x": 1, "y": 2},
        {"trackable_object": 55, "x": frame, "y": 0},
        {"group_name": "away team", "x": 4, "y": 5},
    ]} for period in (1, 2) for frame in (0, 2)]
    for players, ball, teams, _ in skillcorner_2020.tracking(metadata, rows):
        assert np.isnan(players[:, 1]).all()
        np.testing.assert_array_equal(teams, [0, 1])
        np.testing.assert_allclose(ball[:, 0], [0, np.nan, np.nan, np.nan, np.nan, 2])
    for i, row in enumerate(rows):
        row["possession"] = {"group": "home team" if i == 0 else None}
    first, second = skillcorner_2020.ownership(rows)
    np.testing.assert_array_equal(first, [pm.H, pm.H, pm.H, pm.X, pm.X, pm.X])
    np.testing.assert_array_equal(second, np.full(6, pm.X))


def test_ownership_adaptation_continues_a_frozen_tree_and_falls_back_on_single_class():
    truth = np.repeat([0, 1], 50)
    inputs = pd.DataFrame({"h__test": truth, "a__test": 1 - truth}, dtype=np.float32)
    teacher = XGBClassifier(n_estimators=2, n_jobs=1).fit(inputs, truth)
    frames = np.arange(len(truth))
    dead = np.zeros(len(truth), dtype=np.float32)
    replay = inputs, truth, np.ones(len(truth))
    forecast = d6.Forecast(frames, inputs, truth, (np.where(truth, .95, .05), dead))
    student, counts = d6.fit_match([forecast, forecast], replay, teacher)
    assert counts == [100, 100]
    assert student.get_booster().num_boosted_rounds() == 2 + d6.ROUNDS
    assert teacher.get_booster().num_boosted_rounds() == 2
    forecast.signals = (np.full(len(truth), .95), dead)
    assert d6.fit_match([forecast, forecast], replay, teacher) == (None, [0, 200])


def test_d6_zero_adjustment_decodes_as_d3():
    rng = np.random.default_rng(2020)
    positions = np.cumsum(rng.normal(0, .05, (180, 4, 2)), axis=0).astype(np.float32)
    tracking = positions, positions[:, 0].copy(), np.array([0, 0, 1, 1]), np.array([105, 68])
    teacher = d6.reference()
    original, packages, frozen = teacher
    _, predicted = retrain_trial.forecast_pair(packages, frozen["weight"], *tracking)
    expected = motion_trial.decode(predicted, original[2]["recipe"], frozen["recipe"])
    half = d6.forecast(tracking, teacher)
    np.testing.assert_array_equal(d6.decode(half, np.ones(len(expected)), 0.0, frozen["recipe"]), expected)


def test_tactical_build_requires_approval_and_passed_confirmation_before_loading_tracking():
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        with patch.object(d6, "tactical_tracking_halves") as load_tracking:
            try:
                d6.build(folder / "replay.npz", folder, folder / "out.csv", organiser_approved=False)
            except ValueError as error:
                assert "written organiser permission" in str(error)
            else:
                raise AssertionError("Build accepted missing organiser permission")
            with patch.object(d6, "selection", return_value=({}, .5, "development")):
                (folder / "external_validation.json").write_text('{"comparison":{"passed":false}}')
                try:
                    d6.build(folder / "replay.npz", folder, folder / "out.csv", organiser_approved=True)
                except ValueError as error:
                    assert "passed external confirmation" in str(error)
                else:
                    raise AssertionError("Build accepted a failed external confirmation")
            load_tracking.assert_not_called()
