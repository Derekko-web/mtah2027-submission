"""D42 player-set network: tokens, NumPy/PyTorch parity, swap symmetry (dfl_submission/player_set.py)."""

import numpy as np
import torch

from dfl_submission import player_set as ps


def scene(length=30):
    rng = np.random.default_rng(2)
    positions = rng.uniform(-30, 30, size=(length, 6, 2))
    positions[:, 5] = np.nan                                   # one away slot never visible
    ball = positions[:, 0] + .5
    ball[10:15] = np.nan
    return positions, ball, np.array([0, 0, 0, 1, 1, 1]), np.array([105., 68.])


def test_tokens_shape_order_and_missing_values():
    positions, ball, teams, pitch = scene()
    frames = np.arange(0, 30, 5)
    result = ps.tokens(positions, ball, teams, pitch, frames)
    assert result.shape == (6, 2, ps.PER_TEAM, ps.TOKEN) and result.dtype == np.float32
    # Home's nearest player at a seen frame is slot 0, half a metre from the ball.
    assert np.isclose(result[0, 0, 0, 2], np.hypot(.5, .5) / 10, atol=1e-6)
    assert np.isfinite(result[0, 0, :3]).all() and np.isnan(result[0, 0, 3:]).all()
    assert np.isfinite(result[0, 1, :2]).all() and np.isnan(result[0, 1, 2:]).all()
    # Distances are sorted; the ball flag is 0 where the ball is unseen (frame 10).
    assert np.all(np.diff(result[0, 0, :3, 2]) >= 0)
    assert (result[2, :, :2, 7] == 0).all() and (result[0, :, :2, 7] == 1).all()


def random_weights(columns=5, width=8, seed=0):
    torch.manual_seed(seed)
    net = ps.torch_network(columns, width, .2).eval()
    for parameter in net.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    return net, {name: tensor.detach().numpy() for name, tensor in net.state_dict().items()}


def test_numpy_forward_matches_torch():
    net, weights = random_weights()
    rng = np.random.default_rng(5)
    values = rng.normal(size=(2, 140, 10)).astype(np.float32)
    token_values = rng.normal(size=(2, 140, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    mask = (rng.uniform(size=(2, 140, 2, ps.PER_TEAM)) > .4).astype(np.float32)
    mask[0, :7, 0] = 0
    with torch.inference_mode():
        reference = net(torch.from_numpy(values), torch.from_numpy(token_values), torch.from_numpy(mask)).numpy()
    np.testing.assert_allclose(ps.forward(values, token_values, mask, weights), reference, atol=2e-5, rtol=1e-5)


def test_predict_is_swap_symmetric():
    _, weights = random_weights(columns=4)
    rng = np.random.default_rng(9)
    location, scale = np.zeros(4, np.float32), np.ones(4, np.float32)
    permutation = np.array([1, 0, 3, 2])
    checkpoint = (weights, ["h__a", "a__a", "h__b", "a__b"], location, scale, permutation)
    frames = np.arange(0, 200, 5)
    values = rng.normal(size=(len(frames), 4)).astype(np.float32)
    player_tokens = rng.normal(size=(len(frames), 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    player_tokens[:, :, 8:] = np.nan
    home, dead, _, _ = ps.predict(checkpoint, frames, values, checkpoint[1], player_tokens, 200)
    swapped = ps.predict(checkpoint, frames, values[:, permutation], checkpoint[1], player_tokens[:, ::-1], 200)
    np.testing.assert_allclose(swapped[0], 1 - home, atol=1e-5)
    np.testing.assert_allclose(swapped[1], dead, atol=1e-5)
