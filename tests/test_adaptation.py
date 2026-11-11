from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from dfl_submission import adaptation_trial as d5


def test_target_fitting_accepts_tracking_without_labels():
    rng = np.random.default_rng(17)
    positions = np.cumsum(rng.normal(0, .1, (500, 4, 2)), axis=0).astype(np.float32)
    ball = positions[:, 0].copy()
    teams = np.array([0, 0, 1, 1])
    teacher, _ = d5.reference()
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / "tracking-only.npz"
        np.savez(source, positions=positions, ball=ball, teams=teams, pitch=np.array([105, 68]))
        with patch.object(d5.external_benchmark, "skillcorner_paths", return_value=[(1, 0, source)]) as paths:
            donors, speeds, pseudo = d5.target_training(folder, teacher)
        paths.assert_called_once_with(folder, "train")
    np.testing.assert_array_equal(donors[0], d5.tracking_masks(positions, ball, teams))
    assert set(speeds) == set(d5.SPEEDS)
    assert set(pseudo) == set(d5.THRESHOLDS)


def test_corruption_replays_counts_without_inventing_observations():
    positions = np.zeros((4, 4, 2), dtype=np.float32)
    ball = np.zeros((4, 2), dtype=np.float32)
    teams = np.array([0, 0, 1, 1])
    positions[1, 0] = np.nan
    ball[2] = np.nan
    donor = np.array([[1, 1, 2], [1, 2, 0], [1, 1, 1], [0, 0, 2]])
    masked_players, masked_ball = d5.matched_tracking(positions, ball, teams, [donor], np.random.default_rng(2))
    expected = np.minimum(donor, d5.tracking_masks(positions, ball, teams))
    np.testing.assert_array_equal(d5.tracking_masks(masked_players, masked_ball, teams), expected)
