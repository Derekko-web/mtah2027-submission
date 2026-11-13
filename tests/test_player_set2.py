"""D44: configurable player-set tokens and network (dfl_submission/player_set2.py)."""

import numpy as np
import torch

from dfl_submission import player_set as ps, player_set2 as ps2


def scene(length=30):
    rng = np.random.default_rng(4)
    positions = rng.uniform(-30, 30, size=(length, 40, 2))
    positions[:, 35:] = np.nan
    ball = positions[:, 0] + .5
    ball[10:15] = np.nan
    return positions, ball, np.array([0] * 20 + [1] * 20), np.array([105., 68.])


def test_default_tokens_equal_player_set_tokens():
    positions, ball, teams, pitch = scene()
    frames = np.arange(0, 30, 5)
    assert np.array_equal(ps2.tokens(positions, ball, teams, pitch, frames),
                          ps.tokens(positions, ball, teams, pitch, frames), equal_nan=True)


def test_variant_token_shapes_and_relative_velocity():
    positions, ball, teams, pitch = scene()
    frames = np.arange(0, 30, 5)
    wide = ps2.tokens(positions, ball, teams, pitch, frames, dict(ps2.DEFAULT, per_team=16))
    assert wide.shape == (6, 2, 16, 8) and np.isfinite(wide[0, 0, :16]).all() and np.isnan(wide[0, 1, 15:]).all()
    relative = ps2.tokens(positions, ball, teams, pitch, frames, dict(ps2.DEFAULT, relative=True))
    assert relative.shape == (6, 2, 11, 10)
    assert np.array_equal(relative[..., :8], ps.tokens(positions, ball, teams, pitch, frames), equal_nan=True)
    assert (relative[2:4, :, :, 8:][np.isfinite(relative[2:4, :, :, 8:])] == 0).all()   # ball unseen at frame 10 or 15


def test_numpy_inference_matches_the_configurable_network():
    for config in (dict(ps2.DEFAULT, hidden=64), dict(ps2.DEFAULT, relative=True, per_team=16)):
        torch.manual_seed(1)
        net = ps2.torch_network(5, 8, .2, config).eval()
        weights = {k: v.detach().numpy() for k, v in net.state_dict().items()}
        rng = np.random.default_rng(2)
        values = rng.normal(size=(1, 140, 10)).astype(np.float32)
        tok = rng.normal(size=(1, 140, 2, config["per_team"], ps2.token_width(config))).astype(np.float32)
        mask = (rng.uniform(size=(1, 140, 2, config["per_team"])) > .3).astype(np.float32)
        with torch.inference_mode():
            reference = net(torch.from_numpy(values), torch.from_numpy(tok), torch.from_numpy(mask)).numpy()
        np.testing.assert_allclose(ps.forward(values, tok, mask, weights), reference, atol=2e-5, rtol=1e-5)
