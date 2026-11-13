"""D51's auxiliary ball targets, batches, loss and trainer on synthetic fixtures."""

import json

import numpy as np
import torch  # before scipy's OpenMP (via player_set), as in test_player_set.py; CPU training crashes otherwise

from dfl_submission import hidden_ball as hb, hidden_ball_trial as hbt, hidden_views as hv, player_set
from dfl_submission.external_benchmark import Half
from dfl_submission.player_set import CORE, HALO
from evaluation import possession_metrics as pm


def tracking(n=6000, players=22, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-50, 50, (n, players, 2)).astype(np.float32)
    ball = rng.uniform(-50, 50, (n, 2)).astype(np.float32)
    return positions, ball


def half(source, view, match, index=0, n=6000):
    positions, ball = tracking(n, seed=index + 7)
    labels = np.where(np.arange(n) % 1000 < 900, pm.H, pm.D).astype(np.int8)
    teams = np.array([0] * 11 + [1] * 11)
    return Half(source, match, view, index, labels, positions, ball, teams, np.array([105.0, 68.0]))


# -- targets ---------------------------------------------------------------------------------------

def test_view_with_ball_re_derives_the_stored_view_from_its_seeds():
    from dfl_submission.train import TRAIN
    clean = half("dfl", "clean", TRAIN[1])
    for model_set, swaps in (("current", True), ("augmented", True), ("augmented", False)):
        positions, view_ball, source_ball, visible = hb.view_with_ball(model_set, clean, swaps=swaps)
        view, _ = hv.training_view(model_set, clean, swaps=swaps)
        assert np.array_equal(view.positions, positions, equal_nan=True)
        assert view.ball.tobytes() == view_ball.tobytes()
        assert np.array_equal(np.isfinite(view_ball).all(axis=1), visible & np.isfinite(source_ball).all(axis=1))
        assert np.array_equal(source_ball[visible], view_ball[visible])
    assert hb.view_with_ball("current", half("dfl", "masked", TRAIN[1])) is None
    observed = half("skillcorner", "observed", "2017", 1)
    observed.ball[50:90] = np.nan
    positions, view_ball, source_ball, visible = hb.view_with_ball("current", observed, skill_order=4)
    view, _ = hv.training_view("current", observed, skill_order=4)
    assert positions is observed.positions and view.ball.tobytes() == view_ball.tobytes()
    assert np.isnan(source_ball[50:90]).all()


def test_targets_are_the_hidden_ball_offset_from_the_visible_players_mean():
    positions = np.zeros((6, 4, 2), dtype=np.float32)
    positions[:, :, 0] = [[0, 2, 4, 6]] * 6           # mean x 3, mean y 0
    positions[3, :] = np.nan                          # no player visible
    positions[4, 3] = np.nan                          # mean x of the first three: 2
    source = np.array([[13, 4], [13, 4], [13, 4], [13, 4], [12, -10], [np.nan, np.nan]], dtype=np.float32)
    view = source.copy()
    view[[1, 2, 3, 4, 5]] = np.nan                    # frame 0 shows the ball
    offset = hb.targets(positions, view, source, np.arange(6))
    assert np.isnan(offset[0]).all()                  # the view shows the ball
    assert np.allclose(offset[1], [1.0, .4]) and np.allclose(offset[2], [1.0, .4])
    assert np.isnan(offset[3]).all()                  # no visible player
    assert np.allclose(offset[4], [1.0, -1.0])
    assert np.isnan(offset[5]).all()                  # the source has no ball either
    assert offset.dtype == np.float32


def test_targets_match_the_token_reference_point():
    positions, ball = tracking(400)
    view = ball.copy()
    view[100:300] = np.nan
    frames = np.arange(0, 400, 5)
    offset = hb.targets(positions, view, ball, frames)
    tokens = player_set.tokens(positions, view, np.array([0] * 11 + [1] * 11), np.array([105., 68.]), frames)
    hidden = (frames >= 100) & (frames < 300)
    assert np.isfinite(offset[hidden]).all() and np.isnan(offset[~hidden]).all()
    # A token holds the player's offset from the reference point: the ball target minus it is the ball's
    # offset from that player, for any player.
    reference = positions[frames[hidden]].astype(np.float64).mean(axis=1)
    assert np.allclose(offset[hidden], (ball[frames[hidden]] - reference) / 10, atol=1e-5)
    # Where the ball is hidden the tokens' offsets are taken from that same reference point.
    rows = np.flatnonzero(hidden)
    nearest = tokens[rows, 0, 0, :2].astype(np.float64) * 10 + reference
    home = positions[frames[rows]][:, :11].astype(np.float64)
    assert np.all(np.min(np.linalg.norm(home - nearest[:, None], axis=2), axis=1) < 1e-2)


# -- batches, loss and network ---------------------------------------------------------------------

def groups(columns=6, length=400, seed=0, with_aux=True):
    rng = np.random.default_rng(seed)
    three, four = {}, {}
    for source in ("dfl", "skillcorner"):
        three[source], four[source] = [], []
        for i in range(3):
            values = rng.normal(size=(length, columns)).astype(np.float32)
            labels = rng.integers(0, 2, (length, 6)).astype(np.float32)
            tok = rng.normal(size=(length, 2, player_set.PER_TEAM, player_set.TOKEN)).astype(np.float16)
            aux = None
            if with_aux and i == 2:
                aux = np.full((length, 2), np.nan, dtype=np.float32)
                aux[100:300] = rng.normal(size=(200, 2))
            three[source].append((values, labels, tok))
            four[source].append((values, labels, tok, aux))
    return three, four


def test_batch_draws_exactly_as_d42_and_adds_aligned_targets():
    three, four = groups()
    permutation, location, scale = np.arange(6)[::-1].copy(), np.zeros(6, np.float32), np.ones(6, np.float32)
    first, second = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(4):
        reference = player_set.batch(three, first, permutation, location, scale, 8)
        ours = hb.batch(four, second, permutation, location, scale, 8)
        assert all(np.array_equal(a.numpy(), b.numpy()) for a, b in zip(reference, ours[:4]))
        assert ours[4].shape == (8, CORE, 2)
    assert first.bit_generator.state == second.bit_generator.state


def test_aux_loss_is_huber_on_target_frames_only():
    import torch
    predicted = torch.tensor([[[0., 0.], [3., 0.], [5., 5.]]])
    target = torch.tensor([[[.5, 0.], [0., 0.], [float("nan"), float("nan")]]])
    # Huber(delta 1): 0.5 * 0.25 = 0.125 and 0; 3 - 0.5 = 2.5 and 0; mean over four values.
    assert abs(float(hb.aux_loss(predicted, target)) - (.125 + 0 + 2.5 + 0) / 4) < 1e-6
    empty = torch.full((1, 3, 2), float("nan"))
    value = hb.aux_loss(predicted.requires_grad_(), empty)
    assert value.item() == 0.0
    value.backward()
    assert torch.equal(predicted.grad, torch.zeros_like(predicted))


def test_hidden_layer_reproduces_the_network_forward_and_the_head_leaves_the_random_stream():
    import torch
    torch.manual_seed(3)
    net = player_set.torch_network(6, 16, .2).eval()
    values = torch.randn(2, 180, 12)
    token_values = torch.randn(2, 180, 2, player_set.PER_TEAM, player_set.TOKEN)
    mask = (torch.rand(2, 180, 2, player_set.PER_TEAM) > .3).float()
    with torch.inference_mode():
        assert torch.equal(net.outputs(hb.hidden_layer(net, values, token_values, mask)), net(values, token_values, mask))
    torch.manual_seed(11)
    expected = torch.rand(5)
    torch.manual_seed(11)
    hb.ball_head(16)
    assert torch.equal(torch.rand(5), expected)


def test_trainer_at_weight_zero_trains_exactly_as_d42(tmp_path):
    three, four = groups(columns=6, length=420, seed=2)
    permutation, location, scale = np.arange(6), np.zeros(6, np.float32), np.ones(6, np.float32)
    manifest = {"columns": [f"c{i}" for i in range(6)], "records": []}
    folders = {}
    for name in ("source", "tokens", "aux"):
        folders[name] = tmp_path / name
        folders[name].mkdir()
        (folders[name] / "manifest.json").write_text("{}")
    original_ps, original_hb = player_set.training_data, hb.training_data
    player_set.training_data = lambda source, tokens: (three, manifest, permutation, location, scale)
    hb.training_data = lambda source, tokens, aux: (four, manifest, permutation, location, scale)
    try:
        player_set.train(folders["source"], folders["tokens"], tmp_path / "d42", "cpu", seed=4, boundary_boost=.5,
                         width=16, steps=3)
        hb.train(folders["source"], folders["tokens"], folders["aux"], tmp_path / "lb0", "cpu", seed=4,
                 boundary_boost=.5, aux_weight=0., width=16, steps=3)
        hb.train(folders["source"], folders["tokens"], folders["aux"], tmp_path / "lb", "cpu", seed=4,
                 boundary_boost=.5, aux_weight=.5, width=16, steps=3)
    finally:
        player_set.training_data, hb.training_data = original_ps, original_hb
    d42, lb0, lb = (json.loads((tmp_path / name / "fit.json").read_text()) for name in ("d42", "lb0", "lb"))
    assert d42["models"] == lb0["models"]
    assert lb["models"] != d42["models"]                 # the auxiliary term changes the shared network
    assert set(lb["ball"]["heads"]) == {"ball_1.pt", "ball_2.pt", "ball_3.pt"}
    assert lb["ball"]["aux_weight"] == .5 and lb0["ball"]["aux_weight"] == 0.


# -- the trial's helpers ---------------------------------------------------------------------------

def entry(cost, guard, pff, composite, groups=None):
    from dfl_submission import hidden_trial as h49
    values = dict.fromkeys(h49.GROUPS, .7) | {"pff": pff} | (groups or {})
    return {"owner_cost": cost, "guards": {"mean": guard}, "groups": values, "composite": composite}


def test_replaces_compares_best_costs_against_l_and_groups_against_d43():
    d43 = entry(10, .67, .66, .714)
    l_entries = [entry(8, .715, .684, .710), entry(10, .715, .687, .713), entry(12, .717, .685, .714)]
    better = [entry(8, .73, .69, .712), entry(10, .72, .69, .713)]
    decision = hbt.replaces(l_entries, d43, better)
    assert decision["l_cost"] == 10 and decision["selected_cost"] == 8
    assert decision["conditions"] == {"guard_pff_mean": True, "composite": True, "groups": True}
    assert decision["replaces_l"] is True
    small = [entry(10, .72, .69, .713)]
    assert hbt.replaces(l_entries, d43, small)["conditions"]["guard_pff_mean"] is False
    harmed = [entry(8, .74, .70, .713, {"metrica": .68})]
    assert hbt.replaces(l_entries, d43, harmed)["conditions"]["groups"] is False


def test_other_trainings_excludes_this_process():
    import os
    assert hbt.other_trainings() == hbt.other_trainings(own=-1) - (1 if "--seed" in " ".join(__import__("sys").argv) else 0)
    assert isinstance(hbt.other_trainings(own=os.getpid()), int)
