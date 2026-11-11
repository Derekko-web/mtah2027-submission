"""Checks for the longer tracking context and frozen decoder behavior."""

import unittest

import numpy as np
import pandas as pd

from dfl_submission.context_trial import suppress_islands
from dfl_submission.long_context import features, interval_stats
from dfl_transfer_features import swap


class LongContextTests(unittest.TestCase):
    def test_directional_windows_stop_at_input_edges(self):
        values = np.array([1.0, np.nan, 5.0])
        past, past_coverage = interval_stats(values, 1, False)
        future, future_coverage = interval_stats(values, 1, True)
        np.testing.assert_array_equal(past, [1, 1, 5])
        np.testing.assert_array_equal(future, [1, 5, 5])
        np.testing.assert_array_equal(past_coverage, [1, .5, .5])
        np.testing.assert_array_equal(future_coverage, [.5, .5, 1])

    def test_pooled_features_are_team_symmetric(self):
        rng = np.random.default_rng(4)
        players = rng.normal(size=(210, 8, 2))
        ball = rng.normal(size=(210, 2))
        ball[30:53] = np.nan
        teams = np.repeat([0, 1], 4)
        _, owner, dead = features(players, ball, teams, [105, 68])
        _, other_owner, other_dead = features(players, ball, 1 - teams, [105, 68])
        pd.testing.assert_frame_equal(swap(owner), other_owner)
        pd.testing.assert_frame_equal(dead, other_dead)

    def test_short_island_requires_matching_neighbors_and_weak_evidence(self):
        labels = np.array([0, 0, 2, 0, 0, 1, 0, 0, 2, 1, 1])
        home = np.array([.9, .9, .7, .9, .9, .1, .9, .9, .7, .1, .1])
        dead = np.where(labels == 2, .7, .1)
        got = suppress_islands(labels, home, dead, 2, .8)
        np.testing.assert_array_equal(got, [0, 0, 0, 0, 0, 1, 0, 0, 2, 1, 1])
