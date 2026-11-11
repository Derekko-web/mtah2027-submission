"""Regression tests for temporal context and evidence-aware switches."""

from itertools import product
import unittest

import numpy as np
import torch

from dfl_submission import adaptive_decoder, temporal_cpu, temporal_network
from evaluation import possession_metrics as pm


class TemporalTests(unittest.TestCase):
    def test_switch_costs_match_exhaustive_paths(self):
        evidence = np.array([-1.0, 2.0, 1.5, -3.0, .2, -1.0])
        costs = np.array([0, .5, 2, 1, 5, 0])

        def objective(path):
            return evidence @ path - costs[1:] @ (path[1:] != path[:-1])

        best = max(objective(np.asarray(path)) for path in product((0, 1), repeat=len(evidence)))
        self.assertAlmostEqual(objective(adaptive_decoder.binary_path(evidence, costs)), best)

    def test_network_halo_preserves_the_interior(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        net = temporal_network.Network(3).eval()
        halo = temporal_network.HALO
        inputs = torch.randn(2, 2 * halo + 50, 6)
        with torch.inference_mode():
            full = net(inputs)[:, halo + 10:halo + 30]
            chunk = net(inputs[:, 10:2 * halo + 30])[:, halo:halo + 20]
        torch.testing.assert_close(full, chunk)

    def test_stoppages_do_not_supply_ownership_boundary_targets(self):
        labels = np.array([pm.H, pm.H, pm.D, pm.D, pm.A, pm.A, pm.X])
        encoded = temporal_network.targets(labels)
        self.assertTrue(encoded[:, 2].any())
        self.assertFalse(encoded[:, 3].any())

    def test_exported_network_matches_pytorch(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        net = temporal_network.Network(3).eval()
        inputs = torch.randn(2, 151, 6)
        weights = {name: value.numpy() for name, value in net.state_dict().items()}
        with torch.inference_mode():
            reference = net(inputs).numpy()
        np.testing.assert_allclose(temporal_cpu.forward(inputs.numpy(), weights), reference, atol=2e-5, rtol=1e-5)
