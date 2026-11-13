"""D52 seven-block player-set network: six-block equality with player_set / stack_set, seven-block parity and export."""

import json

import numpy as np
import torch

from dfl_submission import context_set as cs, player_set as ps, stack_set as ss, temporal_network
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


STACK = ss.config_of(direction=True, mirror="y")
SIGN_MAP = {"columns": ["h__x", "a__x", "h__y", "a__y"], "signs": {m: [1, 1, 1, 1] for m in ("x", "y", "xy")}}


def weights_of(network, seed=0):
    torch.manual_seed(seed)
    for parameter in network.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    return {name: tensor.detach().numpy() for name, tensor in network.state_dict().items()}


def inputs(length=300, columns=4, seed=5):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(2, length, columns * 2)).astype(np.float32)
    token_values = rng.normal(size=(2, length, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    mask = (rng.uniform(size=(2, length, 2, ps.PER_TEAM)) > .4).astype(np.float32)
    mask[0, :7, 0] = 0
    return values, token_values, mask


def test_six_blocks_are_player_set_exactly():
    torch.manual_seed(3)
    reference = ps.torch_network(4, 8, .2)
    torch.manual_seed(3)
    candidate = cs.torch_network(4, 8, .2, cs.SIX)
    a, b = reference.state_dict(), candidate.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[key], b[key]) for key in a)
    weights = weights_of(reference)
    values, token_values, mask = inputs()
    expected = ps.forward(values, token_values, mask, weights)
    np.testing.assert_array_equal(cs.forward(values, token_values, mask, weights), expected)
    np.testing.assert_array_equal(cs.forward(values, token_values, mask, weights, cs.SIX), expected)
    assert cs.dilations_of(weights) == cs.SIX


def test_seven_blocks_numpy_matches_torch():
    network = cs.torch_network(4, 8, .2, cs.SEVEN).eval()
    weights = weights_of(network, seed=1)
    assert cs.dilations_of(weights) == cs.SEVEN and cs.halo(cs.SEVEN) == 127 and cs.halo(cs.SIX) == ps.HALO
    values, token_values, mask = inputs(length=2 * 127 + 40)
    with torch.inference_mode():
        reference = network(torch.from_numpy(values), torch.from_numpy(token_values), torch.from_numpy(mask)).numpy()
    np.testing.assert_allclose(cs.forward(values, token_values, mask, weights), reference, atol=2e-5, rtol=1e-5)
    # The seventh block reaches 64 steps further: an input change 120 steps away moves the output only with it.
    moved = values.copy()
    moved[:, 200] += 3
    six = weights_of(cs.torch_network(4, 8, .2, cs.SIX).eval(), seed=1)
    for w, reach in ((six, False), (weights, True)):
        delta = np.abs(cs.forward(moved, token_values, mask, w) - cs.forward(values, token_values, mask, w))[:, 80]
        assert (delta.max() > 0) == reach


def test_predict_with_six_blocks_is_stack_predict():
    weights = weights_of(ps.torch_network(4, 8, .2).eval(), seed=2)
    rng = np.random.default_rng(9)
    columns = SIGN_MAP["columns"]
    checkpoint = (weights, columns, rng.normal(size=4).astype(np.float32), np.ones(4, np.float32), np.array([1, 0, 3, 2]))
    frames = np.arange(0, 400, 5)
    values = rng.normal(size=(len(frames), 4)).astype(np.float32)
    player_tokens = rng.normal(size=(len(frames), 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    player_tokens[:, :, 9:] = np.nan
    for config, sign in ((STACK, -1), (STACK, 1), (ss.DEFAULT, None), (ss.config_of(mirror="xy"), None)):
        expected = ss.predict(checkpoint, frames, values, columns, player_tokens, 400, config=config, sign=sign,
                              sign_map=SIGN_MAP)
        result = cs.predict(checkpoint, frames, values, columns, player_tokens, 400, config=config, sign=sign,
                            sign_map=SIGN_MAP)
        for a, b in zip(expected, result, strict=True):
            if config is ss.DEFAULT:
                np.testing.assert_allclose(a, b, atol=1e-6)        # stack_set takes its one-mirror shortcut here
            else:
                np.testing.assert_array_equal(a, b)


def prepared(tmp_path, halves=3, length=460):
    folder, token_folder = tmp_path / "prepared", tmp_path / "tokens"
    folder.mkdir()
    token_folder.mkdir()
    rng = np.random.default_rng(3)
    records, token_records = [], {}
    for index in range(halves):
        labels = np.where(np.arange(length) % 90 < 45, pm.H, pm.A).astype(np.int8)
        labels[200:230] = pm.D
        name = f"dfl_M{index}_clean_0.npz"
        np.savez_compressed(folder / name, values=rng.normal(size=(length, 4)).astype(np.float32),
                            targets=temporal_network.targets(labels))
        records.append({"file": name, "source": "dfl", "match": f"M{index}", "sha256": digest(folder / name)})
        tokens = rng.normal(size=(length, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float16)
        tokens[:, :, 9:] = np.nan
        np.savez_compressed(token_folder / name, tokens=tokens)
        token_records[name] = digest(token_folder / name)
    (folder / "manifest.json").write_text(json.dumps({"columns": SIGN_MAP["columns"], "records": records}))
    (token_folder / "manifest.json").write_text(json.dumps({"base_manifest": digest(folder / "manifest.json"),
                                                            "records": token_records}))
    signs = {record["file"]: (-1 if i == 1 else 1) for i, record in enumerate(records)}
    return folder, token_folder, signs


def test_six_block_batch_is_stack_batch(tmp_path):
    folder, token_folder, signs = prepared(tmp_path)
    groups, manifest, permutation, location, scale = ss.training_data(folder, token_folder, signs=signs)
    column_signs = {m: np.ones(4, np.float32) for m in ("identity", "x", "y", "xy")}
    expected = ss.batch(groups, np.random.default_rng(4), np.random.default_rng([4, 4747]), permutation, location, scale,
                        8, STACK, column_signs=column_signs)
    result = cs.batch(groups, np.random.default_rng(4), np.random.default_rng([4, 4747]), permutation, location, scale,
                      8, STACK, column_signs=column_signs, dilations=cs.SIX)
    for a, b in zip(expected, result, strict=True):
        assert torch.equal(a, b)
    seven = cs.batch(groups, np.random.default_rng(4), np.random.default_rng([4, 4747]), permutation, location, scale,
                     8, STACK, column_signs=column_signs)
    assert seven[0].shape[1] == ps.CORE + 2 * 127 and seven[3].shape[1] == ps.CORE


def test_six_block_training_is_stack_training_and_seven_blocks_export(tmp_path):
    folder, token_folder, signs = prepared(tmp_path)
    options = dict(seed=11, boundary_boost=.5, config=STACK, signs=signs, sign_map=SIGN_MAP, width=8, steps=6,
                   batch_size=4)
    ss.train(folder, token_folder, tmp_path / "stack", "cpu", **options)
    cs.train(folder, token_folder, tmp_path / "six", "cpu", dilations=cs.SIX, **options)
    stack = json.loads((tmp_path / "stack" / "fit.json").read_text())
    six = json.loads((tmp_path / "six" / "fit.json").read_text())
    assert stack["models"] == six["models"] and six["dilations"] == list(cs.SIX)
    cs.train(folder, token_folder, tmp_path / "seven", "cpu", **options)
    seven = json.loads((tmp_path / "seven" / "fit.json").read_text())
    assert seven["dilations"] == list(cs.SEVEN) and seven["models"] != six["models"]
    cs.export(tmp_path / "seven", tmp_path / "seven_export")
    exported = json.loads((tmp_path / "seven_export" / "fit.json").read_text())
    assert max(exported["export_max_logit_difference"].values()) < 2e-5
    from dfl_submission import player_trial
    checkpoint = player_trial.PlayerCheckpoint(tmp_path / "seven_export")
    assert cs.dilations_of(checkpoint.checkpoint[0]) == cs.SEVEN
    frames = np.arange(0, 600, 5)
    rng = np.random.default_rng(0)
    out = cs.predict(checkpoint.checkpoint, frames, rng.normal(size=(len(frames), 4)).astype(np.float32),
                     list(checkpoint.columns), rng.normal(size=(len(frames), 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32),
                     600, config=STACK, sign=1, sign_map=SIGN_MAP)
    assert len(out) == 4 and all(len(item) == 600 and np.isfinite(item).all() for item in out)
    # The six-block export of the same run type is player_set's export.
    cs.export(tmp_path / "six", tmp_path / "six_export")
    ps.export(tmp_path / "six", tmp_path / "six_player_export")
    a = json.loads((tmp_path / "six_export" / "fit.json").read_text())["models"]
    b = json.loads((tmp_path / "six_player_export" / "fit.json").read_text())["models"]
    assert a == b
