"""Check player-set symmetry and the joint three-state decoder."""

from itertools import product
import unittest

import numpy as np

from dfl_submission import joint_decoder, motion_features


class MotionTests(unittest.TestCase):
    def test_joint_decoder_matches_exhaustive_paths(self):
        emissions = np.array([[.1, -.5, .7], [.6, .1, -.5], [-.3, .2, .5], [.2, .5, -.2]])
        dead_cost = np.array([0., .6, .1, .4])
        owner_cost = np.array([0., .1, .8, .2])

        def objective(path):
            value = emissions[np.arange(len(path)), path].sum()
            for i in range(1, len(path)):
                if path[i] != path[i - 1]:
                    value -= dead_cost[i] if 2 in (path[i], path[i - 1]) else owner_cost[i]
            return value

        best = max(objective(np.asarray(path)) for path in product(range(3), repeat=len(emissions)))
        self.assertAlmostEqual(objective(joint_decoder.path(emissions, dead_cost, owner_cost)), best)

    def test_player_tensors_follow_team_swap_and_player_permutation(self):
        rng = np.random.default_rng(13)
        positions = rng.normal(size=(60, 27, 2))
        ball = rng.normal(size=(60, 2))
        ball[10:34] = np.nan
        positions[20:27, :12] = np.nan
        teams = np.repeat([0, 1], [12, 15])
        frames = np.arange(0, 60, 5)
        result = motion_features.features(positions, ball, teams, frames)
        swapped = motion_features.features(positions, ball, 1 - teams, frames)
        np.testing.assert_array_equal(swapped, result[:, ::-1])
        permutation = rng.permutation(len(teams))
        permuted = motion_features.features(positions[:, permutation], ball, teams[permutation], frames)
        np.testing.assert_array_equal(permuted, result)

    def test_absent_tracking_has_zero_player_tokens(self):
        result = motion_features.features(np.full((20, 2, 2), np.nan), np.full((20, 2), np.nan),
                                          np.array([0, 1]), np.arange(0, 20, 5))
        self.assertTrue(np.all(result == 0))
