"""D33 spell-balanced loss weights (dfl_submission/objective_network.py)."""

import numpy as np
import torch

from dfl_submission import objective_network as on, temporal_network as tn
from evaluation import possession_metrics as pm


H, A, D, X = pm.H, pm.A, pm.D, pm.X


def codes(*runs):
    return np.concatenate([np.full(n, label, dtype=np.int8) for label, n in runs])


def test_sequence_labels_round_trip_the_stored_targets():
    labels = codes((X, 3), (D, 4), (H, 10), (A, 6), (D, 2), (A, 5), (X, 2))
    assert np.array_equal(on.sequence_labels(tn.targets(labels)), labels)


def test_inconsistent_targets_are_refused():
    targets = tn.targets(codes((H, 5), (D, 5)))
    targets[0, 5] = 0
    try:
        on.sequence_labels(targets)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_possession_lengths_follow_the_merged_runs():
    # 5 fps: H 4 s, A 1 s (brief, merged into H), H 2 s -> one H run of 7 s; then D; then A 1.2 s.
    labels = codes((H, 20), (A, 5), (H, 10), (D, 5), (A, 6))
    lengths = on.possession_lengths(labels)
    assert np.allclose(lengths[:35], 7.0)
    assert np.isnan(lengths[35:40]).all()
    assert np.allclose(lengths[40:], 1.2)


def test_a_change_of_three_seconds_or_more_is_kept():
    labels = codes((H, 20), (A, 15), (H, 10))
    lengths = on.possession_lengths(labels)
    assert np.allclose(lengths[:20], 4.0) and np.allclose(lengths[20:35], 3.0) and np.allclose(lengths[35:], 2.0)


def test_raw_weights():
    lengths = np.array([np.nan, 1.0, 2.0, 5.0, 10.0, 30.0])
    assert np.allclose(on.raw_weights(lengths, 0), 1)
    assert np.allclose(on.raw_weights(lengths, 1), [1, 5, 5, 2, 1, 1])
    assert np.allclose(on.raw_weights(lengths, .5), [1, 5 ** .5, 5 ** .5, 2 ** .5, 1, 1])


def test_spell_weights_have_mean_one_on_live_frames_per_source():
    first = tn.targets(codes((H, 50), (A, 10), (D, 8), (H, 100)))
    second = tn.targets(codes((A, 12), (H, 70), (X, 4)))
    groups = {"dfl": [(None, first), (None, second)], "skillcorner": [(None, second)]}
    weights = on.spell_weights(groups, 1.0)
    for source, sequences in groups.items():
        live = np.concatenate([labels[:, 5] > .5 for _, labels in sequences])
        values = np.concatenate(weights[source])
        assert abs(values[live].mean() - 1) < 1e-6
        assert np.all(values[~live] == 1)
    # Shorter possessions weigh more per frame.
    dfl_first = weights["dfl"][0]
    assert dfl_first[55] > dfl_first[0] == dfl_first[100]


def test_alpha_zero_gives_unit_weights():
    groups = {"dfl": [(None, tn.targets(codes((H, 20), (A, 3), (H, 9))))]}
    assert np.all(on.spell_weights(groups, 0)["dfl"][0] == 1)


def test_loss_matches_temporal_network_at_the_defaults():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(2, 40, 4, generator=generator)
    labels = tn.targets(codes((X, 2), (D, 8), (H, 12), (A, 9), (D, 3), (H, 6)))
    labels = torch.from_numpy(np.stack([labels, labels[::-1].copy()]))
    weighted = torch.cat([labels, torch.ones(2, 40, 1)], dim=-1)
    for boost in (.5, 1.4):
        expected = tn.loss(logits, labels, boost)
        assert torch.equal(on.loss(logits, weighted, boost), expected)
        assert torch.equal(on.loss(logits, weighted, boost, alpha=0., edge=.2), expected)


def test_loss_uses_the_spell_weight_and_edge_weight():
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(1, 30, 4, generator=generator)
    labels = torch.from_numpy(tn.targets(codes((H, 10), (A, 10), (H, 10))))[None]
    ones = torch.cat([labels, torch.ones(1, 30, 1)], dim=-1)
    heavy = torch.cat([labels, torch.full((1, 30, 1), 3.)], dim=-1)
    assert not torch.equal(on.loss(logits, heavy, .5, alpha=1.), on.loss(logits, ones, .5, alpha=1.))
    assert not torch.equal(on.loss(logits, ones, .5, edge=.6), on.loss(logits, ones, .5))
