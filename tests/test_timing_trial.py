"""D26 plumbing on synthetic data: the shift, the robust-shift rule, offsets and leave-one-game-out."""

import numpy as np

from dfl_submission import timing_trial as tt
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def test_shift_moves_changes_between_teams_earlier():
    assert np.array_equal(tt.shift_changes(labels((H, 50), (A, 50)), 3), labels((H, 47), (A, 53)))
    assert np.array_equal(tt.shift_changes(labels((H, 50), (A, 2), (H, 50)), 3), labels((H, 47), (A, 2), (H, 53)))


def test_shift_leaves_stops_restarts_and_abstentions_alone():
    for original in (labels((H, 50), (D, 10), (A, 50)), labels((H, 50), (X, 1), (A, 50))):
        assert np.array_equal(tt.shift_changes(original, 3), original)


def test_shift_never_moves_a_boundary_past_its_neighbour():
    assert np.array_equal(tt.shift_changes(labels((D, 10), (A, 2), (H, 50)), 3), labels((D, 10), (A, 1), (H, 51)))
    assert np.array_equal(tt.shift_changes(labels((H, 2), (A, 50)), 3), labels((H, 1), (A, 51)))
    original = labels((H, 40), (A, 40))
    assert np.array_equal(tt.shift_changes(original, 0), original)
    assert np.array_equal(original, labels((H, 40), (A, 40)))


def test_robust_shift_needs_every_median_at_two_frames_and_rounds_the_smallest():
    assert tt.robust_shift([3, 4, 6, 3]) == 3
    assert tt.robust_shift([2.5, 4]) == 3
    assert tt.robust_shift([2, 9]) == 2
    assert tt.robust_shift([3, 1.5]) == 0
    assert tt.robust_shift([3, None]) == 0


def test_ownership_median_per_game():
    truth = np.r_[labels((H, 100), (A, 100)), X, labels((A, 100), (H, 100))].astype(np.int8)
    prediction = np.r_[labels((H, 104), (A, 96)), X, labels((A, 102), (H, 98))].astype(np.int8)
    assert tt.ownership_median(truth, prediction) == 3.0
    assert tt.ownership_median(truth, truth) == 0.0
    assert tt.ownership_median(labels((H, 50), (D, 50)), labels((H, 50), (D, 50))) is None


def test_leave_one_game_out_estimates_each_shift_from_the_other_games():
    medians = {1: [3, 3, 3, 3], 2: [4, 4, 4, 4], 3: [1, 3, 3, 3]}
    seen = {}

    def change(game, shift):
        seen[game] = shift
        return {1: .004, 2: .002, 3: .006}[game] if shift else 0.0

    result = tt.leave_one_game_out(medians, change)
    assert seen == {1: 0, 2: 0, 3: 3} and result["shifts"] == {"1": 0, "2": 0, "3": 3}
    assert result["changes"] == {"1": 0.0, "2": 0.0, "3": .006} and abs(result["mean"] - .002) < 1e-12
    assert result["passed"] is False
    medians = {1: [3, 3], 2: [4, 4], 3: [3, 5]}
    assert tt.leave_one_game_out(medians, change)["passed"] is True
    assert tt.leave_one_game_out(medians, lambda game, shift: -1e-9 if game == 2 else .01)["passed"] is False


def test_guards_allow_a_loss_of_up_to_one_hundredth():
    ok = {"primary": -0.01, "dead_carried": -0.01, "team_swap": 0.0, "pff": -0.0099}
    assert tt.guards_passed(ok)
    assert not tt.guards_passed(dict(ok, team_swap=-0.0101))
