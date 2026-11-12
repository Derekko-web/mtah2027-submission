import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfl_submission.conventions import collapse_brief_changes, remove_brief_stoppages  # noqa: E402
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


# -- D25: no stoppage shorter than one second --------------------------------------------------------

def test_brief_stoppage_between_the_same_team_disappears():
    result = remove_brief_stoppages(labels((H, 50), (D, 10), (H, 50)))
    assert (result == H).all()


def test_brief_stoppage_between_different_teams_is_split_in_half():
    result = remove_brief_stoppages(labels((H, 50), (D, 10), (A, 50)))
    assert np.array_equal(result, labels((H, 55), (A, 55)))
    odd = remove_brief_stoppages(labels((H, 50), (D, 11), (A, 50)))
    assert np.array_equal(odd, labels((H, 55), (A, 56)))  # the earlier team takes the floor of half


def test_stoppages_of_one_second_or_more_are_untouched():
    for length in (25, 30):
        original = labels((H, 50), (D, length), (A, 50))
        assert np.array_equal(remove_brief_stoppages(original), original)


def test_stoppages_at_a_half_edge_or_next_to_x_are_untouched():
    for original in (labels((D, 10), (H, 50)), labels((H, 50), (D, 10)),
                     labels((H, 50), (X, 1), (D, 10), (A, 50)), labels((H, 50), (D, 10), (X, 3), (A, 50))):
        assert np.array_equal(remove_brief_stoppages(original), original)


def test_every_brief_stoppage_in_a_half_is_removed_and_the_input_kept():
    original = labels((H, 40), (D, 5), (A, 40), (D, 20), (A, 40), (D, 40), (H, 40))
    result = remove_brief_stoppages(original)
    assert np.array_equal(result, labels((H, 42), (A, 103), (D, 40), (H, 40)))
    assert np.array_equal(original, labels((H, 40), (D, 5), (A, 40), (D, 20), (A, 40), (D, 40), (H, 40)))
