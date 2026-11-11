import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from dfl_submission.augment import handle_dead_ball, swap_teams, wide_masking  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402


def synthetic(runs=40, live=200, dead=150, players=6):
    labels = np.concatenate([np.r_[np.full(live, pm.H), np.full(dead, pm.D)] for _ in range(runs)]).astype(np.int8)
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(0, .1, (len(labels), players, 2)), axis=0).astype(np.float32)
    ball = positions[:, 0].copy()
    for start in range(live, len(labels), live + dead):
        ball[start:start + dead] = ball[start]
    return labels, positions, ball


def test_dead_ball_is_no_longer_always_frozen():
    labels, positions, ball = synthetic()
    result = handle_dead_ball(labels, positions, ball, np.random.default_rng(1))
    dead = labels == pm.D
    moving = np.r_[False, np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-3]
    assert np.array_equal(result[~dead], ball[~dead])
    assert moving[dead].mean() > .3


def test_some_dead_runs_lose_the_ball_and_some_keep_it():
    labels, positions, ball = synthetic()
    result = handle_dead_ball(labels, positions, ball, np.random.default_rng(2))
    dead = labels == pm.D
    assert 0 < (~np.isfinite(result[dead]).all(axis=1)).mean() < .5


def test_wide_masking_visibility_varies_by_seed():
    labels, positions, ball = synthetic()
    shares = [np.isfinite(wide_masking(positions, ball, np.random.default_rng(seed))[1]).all(axis=1).mean()
              for seed in range(8)]
    assert .15 < min(shares) and max(shares) < .97 and max(shares) - min(shares) > .1


def test_swaps_exchange_home_and_away_tracks():
    _, positions, _ = synthetic()
    teams = np.array([0, 0, 0, 1, 1, 1])
    swapped = swap_teams(positions, teams, np.random.default_rng(3), every=200, frames=lambda _: 100)
    changed = ~np.isclose(swapped, positions).all(axis=2)
    assert changed.any()
    assert np.allclose(np.sort(swapped, axis=1), np.sort(positions, axis=1))
