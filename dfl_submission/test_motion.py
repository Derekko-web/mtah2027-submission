"""Check player-set symmetry and the joint three-state decoder."""

from itertools import product
import unittest

import numpy as np

from dfl_submission import joint_decoder, motion_features, restarts


class MotionTests(unittest.TestCase):
    def test_joint_decoder_matches_exhaustive_paths(self):
        emissions = np.array([[.1, -.5, .7], [.6, .1, -.5], [-.3, .2, .5], [.2, .5, -.2]])
        dead_cost = np.array([0., .6, .1, .4])
        owner_cost = np.array([0., .1, .8, .2])

        for restart_cost in (dead_cost, np.array([0., 0., 0., 0.])):
            with self.subTest(restart_cost=restart_cost):
                def objective(path):
                    value = emissions[np.arange(len(path)), path].sum()
                    for i in range(1, len(path)):
                        if path[i] == path[i - 1]:
                            continue
                        if path[i - 1] == 2:
                            value -= restart_cost[i]
                        elif path[i] == 2:
                            value -= dead_cost[i]
                        else:
                            value -= owner_cost[i]
                    return value

                best = max(objective(np.asarray(path)) for path in product(range(3), repeat=len(emissions)))
                decoded = joint_decoder.path(emissions, dead_cost, owner_cost, restart_cost)
                self.assertAlmostEqual(objective(decoded), best)

    def test_restart_evidence_stays_local_to_each_half(self):
        args = dict(method="evidence", seconds=2, strength=1)
        bias, discount = restarts.decoder_adjustment(151, [75], **args)
        np.testing.assert_array_equal(bias[[4, 5, 10, 15, 20, 25, 26]], [0, 0, 1, 0, -1, 0, 0])
        self.assertFalse(discount.any())
        bias, _ = restarts.decoder_adjustment(101, [0], **args)
        self.assertFalse((bias > 0).any())
        self.assertIsNone(restarts.decoder_adjustment(101, [], **args))

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
