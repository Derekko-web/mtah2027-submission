import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfl_submission.conventions import collapse_brief_changes  # noqa: E402
from evaluation import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.asarray(pm.H).dtype) for code, length in parts])


def test_brief_opponent_touch_is_merged():
    result = collapse_brief_changes(labels((H, 100), (A, 20), (H, 100)), seconds=1)
    assert (result == H).all()


def test_long_opponent_spell_is_kept():
    original = labels((H, 100), (A, 30), (H, 100))
    assert np.array_equal(collapse_brief_changes(original, seconds=1), original)


def test_restart_possessor_and_dead_ball_are_kept():
    original = labels((H, 100), (D, 50), (A, 10), (H, 100))
    assert np.array_equal(collapse_brief_changes(original, seconds=1), original)


def test_shortest_interior_merges_first():
    result = collapse_brief_changes(labels((H, 100), (A, 10), (H, 5), (A, 100)), seconds=1)
    assert np.array_equal(result, labels((H, 100), (A, 115)))


def test_abstentions_split_live_segments():
    original = labels((H, 100), (X, 1), (A, 10), (H, 100))
    assert np.array_equal(collapse_brief_changes(original, seconds=1), original)
