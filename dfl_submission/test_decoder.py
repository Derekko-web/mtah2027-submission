"""Boundary and sequence-scoring tests for the submission decoder."""

import unittest

import numpy as np

from dfl_submission.decoder import centered_mean, decode
from dfl_submission.train import score


class DecoderTests(unittest.TestCase):
    def test_centered_windows_shrink_at_half_boundaries(self):
        result = centered_mean(np.array([0.0, 0.0, 0.9]), 1, 0, 3)
        np.testing.assert_allclose(result, [0.0, 0.3, 0.45])

    def test_ownership_windows_do_not_cross_stoppage(self):
        home = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
        dead = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(decode(home, dead, dead_cutoff=0.5, dead_radius=0,
                                             owner_radius=50), [0, 2, 1, 1, 1])

    def test_external_scoring_separates_halves_and_scores_all_classes(self):
        half = (np.array([0, 0, 2, 1]), np.array([1.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 0.0]), np.array([0, 0, 2, 1]))
        result = score([half, half], dict(dead_cutoff=0.5, dead_radius=0, owner_radius=0))
        self.assertEqual(result["spell_f1"], 1.0)
        self.assertEqual(result["macro_f1"], 1.0)
        self.assertEqual(result["true_spells"], 4)
