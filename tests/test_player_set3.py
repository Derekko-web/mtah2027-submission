"""D47 player-set trainer and inference: mirrors, token noise, switches-off equality (dfl_submission/player_set3.py)."""

import json

import numpy as np
import torch

from dfl_submission import player_set as ps, player_set3 as p3, temporal_network
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


def scene(length=30):
    rng = np.random.default_rng(2)
    positions = rng.uniform(-30, 30, size=(length, 6, 2))
    positions[:, 5] = np.nan
    ball = positions[:, 0] + .5
    ball[10:15] = np.nan
    return positions, ball, np.array([0, 0, 0, 1, 1, 1]), np.array([105., 68.])


def test_token_sign_maps():
    assert np.flatnonzero(p3.TOKEN_SIGNS["x"] < 0).tolist() == [0, 3, 5]
    assert np.flatnonzero(p3.TOKEN_SIGNS["y"] < 0).tolist() == [1, 4, 6]
    assert np.flatnonzero(p3.TOKEN_SIGNS["xy"] < 0).tolist() == [0, 1, 3, 4, 5, 6]
    assert (p3.TOKEN_SIGNS["identity"] == 1).all()


def test_tokens_from_mirrored_tracking_are_sign_flips():
    positions, ball, teams, pitch = scene()
    frames = np.arange(0, 30, 5)
    base = ps.tokens(positions, ball, teams, pitch, frames)
    for mirror in ("x", "y", "xy"):
        mirrored_positions, mirrored_ball = p3.mirror_tracking(positions, ball, mirror)
        assert mirrored_positions.dtype == positions.dtype
        result = ps.tokens(mirrored_positions, mirrored_ball, teams, pitch, frames)
        np.testing.assert_array_equal(result, base * p3.TOKEN_SIGNS[mirror])
    unchanged, _ = p3.mirror_tracking(positions, ball, "identity")
    np.testing.assert_array_equal(unchanged, positions)


def test_column_signs_need_matching_columns():
    sign_map = {"columns": ["a", "b"], "signs": {"x": [1, -1], "y": [1, 1], "xy": [1, -1]}}
    assert p3.column_signs(sign_map, "x", ["a", "b"]).tolist() == [1, -1]
    assert p3.column_signs(None, "identity", ["a", "b"]).tolist() == [1, 1]
    for bad in ((sign_map, "x", ["b", "a"]), (None, "x", ["a", "b"])):
        try:
            p3.column_signs(*bad)
        except ValueError:
            continue
        raise AssertionError("column_signs accepted a mismatched or missing sign map")


def test_noise_keeps_masked_tokens_masked_and_recomputes_distance():
    rng = np.random.default_rng(4)
    window = rng.normal(size=(400, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    window[..., 2] = np.hypot(window[..., 0], window[..., 1])
    window[:, 1, 7:] = np.nan
    window[50:60, 0] = np.nan
    half_pitch = (52.5, 34.)
    result = p3.noisy_tokens(window, np.random.default_rng(1), half_pitch)
    absent = ~np.isfinite(window).all(axis=-1)
    kept = np.isfinite(result).all(axis=-1)
    assert not kept[absent].any()                                   # masked tokens stay masked
    assert (np.isnan(result) == ~kept[..., None]).all()             # a token is wholly present or absent
    dropped = 1 - kept[~absent].mean()
    assert abs(dropped - p3.MASK_PROBABILITY) < .015
    np.testing.assert_allclose(result[kept][:, 2], np.hypot(result[kept][:, 0], result[kept][:, 1]), rtol=1e-6)
    shift_x = (result[..., 0] - window[..., 0])[kept] * p3.OFFSET_SCALE
    assert abs(shift_x.std() - p3.POSITION_SD) < .02
    np.testing.assert_allclose((result[..., 5] - window[..., 5])[kept], shift_x / half_pitch[0], atol=2e-6)
    shift_y = (result[..., 1] - window[..., 1])[kept] * p3.OFFSET_SCALE
    np.testing.assert_allclose((result[..., 6] - window[..., 6])[kept], shift_y / half_pitch[1], atol=2e-6)
    velocity = (result[..., 3] - window[..., 3])[kept] * p3.VELOCITY_SCALE
    assert abs(velocity.std() - p3.VELOCITY_SD) < .03
    np.testing.assert_array_equal(result[..., 7][kept], window[..., 7][kept])   # the ball flag is untouched


def random_weights(columns=4, width=8, seed=0):
    torch.manual_seed(seed)
    net = ps.torch_network(columns, width, .2).eval()
    for parameter in net.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    return {name: tensor.detach().numpy() for name, tensor in net.state_dict().items()}


def symmetric_tokens(frames, rng):
    """Tokens whose four mirrors are the same set of players in each team (permuted)."""
    base = rng.normal(size=(frames, 2, 2, ps.TOKEN)).astype(np.float32)
    parts = [base * p3.TOKEN_SIGNS[mirror] for mirror in p3.MIRRORS]
    tokens = np.full((frames, 2, ps.PER_TEAM, ps.TOKEN), np.nan, np.float32)
    tokens[:, :, :8] = np.concatenate(parts, axis=2)
    return tokens


def test_mirror_averaged_inference_equals_plain_for_mirror_symmetric_input():
    weights = random_weights()
    rng = np.random.default_rng(9)
    columns = ["h__a", "a__a", "h__b", "a__b"]
    checkpoint = (weights, columns, np.zeros(4, np.float32), np.ones(4, np.float32), np.array([1, 0, 3, 2]))
    sign_map = {"columns": columns, "signs": {"x": [1, 1, -1, -1], "y": [1, 1, 1, 1], "xy": [1, 1, -1, -1]}}
    frames = np.arange(0, 200, 5)
    values = rng.normal(size=(len(frames), 4)).astype(np.float32)
    values[:, 2:] = 0                                               # sign-changing columns at their mirror point
    player_tokens = symmetric_tokens(len(frames), rng)
    plain = ps.predict(checkpoint, frames, values, columns, player_tokens, 200)
    mirrored = p3.predict(checkpoint, frames, values, columns, player_tokens, 200, mirrored=True, sign_map=sign_map)
    for a, b in zip(plain, mirrored, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6)
    same = p3.predict(checkpoint, frames, values, columns, player_tokens, 200)
    for a, b in zip(plain, same, strict=True):
        np.testing.assert_array_equal(a, b)
    # An asymmetric input changes under the mirror average, and the swap symmetry still holds.
    player_tokens = rng.normal(size=player_tokens.shape).astype(np.float32)
    values = rng.normal(size=values.shape).astype(np.float32)
    plain = ps.predict(checkpoint, frames, values, columns, player_tokens, 200)
    mirrored = p3.predict(checkpoint, frames, values, columns, player_tokens, 200, mirrored=True, sign_map=sign_map)
    assert np.abs(plain[0] - mirrored[0]).max() > 1e-4
    swapped = p3.predict(checkpoint, frames, values[:, [1, 0, 3, 2]], columns, player_tokens[:, ::-1], 200,
                         mirrored=True, sign_map=sign_map)
    np.testing.assert_allclose(swapped[0], 1 - mirrored[0], atol=1e-5)
    np.testing.assert_allclose(swapped[1], mirrored[1], atol=1e-5)


def prepared_folder(tmp_path, halves=3, length=420):
    """A tiny prepared-sequence folder with player tokens, in the layout `player_set.training_data` reads."""
    folder, token_folder = tmp_path / "prepared", tmp_path / "tokens"
    folder.mkdir()
    token_folder.mkdir()
    rng = np.random.default_rng(3)
    records, token_records, pitches = [], {}, {}
    for index in range(halves):
        labels = np.where(np.arange(length) % 90 < 45, pm.H, pm.A).astype(np.int8)
        labels[200:230] = pm.D
        values = rng.normal(size=(length, 4)).astype(np.float32)
        name = f"dfl_M{index}_clean_0.npz"
        np.savez_compressed(folder / name, values=values, targets=temporal_network.targets(labels))
        records.append({"file": name, "source": "dfl", "match": f"M{index}", "sha256": digest(folder / name)})
        tokens = rng.normal(size=(length, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float16)
        tokens[:, :, 9:] = np.nan
        np.savez_compressed(token_folder / name, tokens=tokens)
        token_records[name] = digest(token_folder / name)
        pitches[name] = [105. + index, 68.]
    (folder / "manifest.json").write_text(json.dumps({"columns": ["h__x", "a__x", "h__y", "a__y"], "records": records}))
    (token_folder / "manifest.json").write_text(json.dumps({"base_manifest": digest(folder / "manifest.json"),
                                                            "records": token_records}))
    (tmp_path / "pitches.json").write_text(json.dumps(pitches))
    return folder, token_folder, tmp_path / "pitches.json"


def test_switches_off_batch_is_player_set_batch(tmp_path):
    folder, token_folder, _ = prepared_folder(tmp_path)
    groups, _, permutation, location, scale = ps.training_data(folder, token_folder)
    augment = np.random.default_rng([5, p3.AUGMENT_STREAM])
    before = augment.bit_generator.state
    expected = ps.batch(groups, np.random.default_rng(5), permutation, location, scale, 8)
    result = p3.batch(groups, np.random.default_rng(5), augment, permutation, location, scale, 8, p3.DEFAULT)
    for a, b in zip(expected, result, strict=True):
        assert torch.equal(a, b)
    assert augment.bit_generator.state == before


def test_augmented_batch_keeps_the_window_stream(tmp_path):
    folder, token_folder, pitch_file = prepared_folder(tmp_path)
    groups, manifest, permutation, location, scale = ps.training_data(folder, token_folder)
    signs = {m: np.ones(4, np.float32) for m in p3.MIRRORS}
    pitches = p3.load_pitches(pitch_file, manifest)
    assert pitches["dfl"][1] == (53., 34.)
    plain = ps.batch(groups, np.random.default_rng(6), permutation, location, scale, 8)
    for config in ({"seed_offset": 0, "mirror": True, "noise": False}, {"seed_offset": 0, "mirror": False, "noise": True}):
        rng = np.random.default_rng(6)
        result = p3.batch(groups, rng, np.random.default_rng([6, p3.AUGMENT_STREAM]), permutation, location, scale, 8,
                          config, signs=signs, pitches=pitches)
        assert torch.equal(result[0], plain[0]) and torch.equal(result[3], plain[3])   # same windows, swaps, labels
        if config["mirror"]:
            assert torch.equal(result[2], plain[2])
            ratio = (result[1] / torch.where(plain[1] == 0, torch.ones_like(plain[1]), plain[1]))[plain[2] > 0]
            assert torch.all((ratio.abs() - 1).abs() < 1e-6)
            flipped = (ratio < 0).float().mean().item()
            assert 0 < flipped < 1
        else:
            assert (result[2] <= plain[2]).all() and (result[2] < plain[2]).any()   # noise only removes tokens


def test_switches_off_training_equals_player_set_train(tmp_path):
    folder, token_folder, _ = prepared_folder(tmp_path)
    options = dict(seed=11, boundary_boost=.5, width=8, steps=6, batch_size=4)
    ps.train(folder, token_folder, tmp_path / "reference", "cpu", **options)
    p3.train(folder, token_folder, tmp_path / "candidate", "cpu", config=p3.DEFAULT, **options)
    reference = json.loads((tmp_path / "reference" / "fit.json").read_text())
    candidate = json.loads((tmp_path / "candidate" / "fit.json").read_text())
    assert reference["models"] == candidate["models"]
    for name in reference["models"]:
        a = torch.load(tmp_path / "reference" / name, weights_only=True)
        b = torch.load(tmp_path / "candidate" / name, weights_only=True)
        assert all(torch.equal(a[key], b[key]) for key in a)
    assert candidate["seed"] == 11 and candidate["d47"]["mirror"] is False and candidate["d47"]["noise"] is False
    # A seed offset changes the run, and is recorded as the run's seed.
    p3.train(folder, token_folder, tmp_path / "offset", "cpu", config=dict(p3.DEFAULT, seed_offset=1000), **options)
    offset = json.loads((tmp_path / "offset" / "fit.json").read_text())
    assert offset["seed"] == 1011 and offset["models"] != reference["models"]


def test_augmented_training_runs_and_records_its_switches(tmp_path):
    folder, token_folder, pitch_file = prepared_folder(tmp_path)
    sign_map = {"columns": ["h__x", "a__x", "h__y", "a__y"],
                "signs": {"x": [1, 1, 1, 1], "y": [1, 1, 1, 1], "xy": [1, 1, 1, 1]}}
    p3.train(folder, token_folder, tmp_path / "fn", "cpu", seed=11, boundary_boost=.5, width=8, steps=6, batch_size=4,
             config={"seed_offset": 0, "mirror": True, "noise": True}, sign_map=sign_map, pitch_file=pitch_file)
    fit = json.loads((tmp_path / "fn" / "fit.json").read_text())
    assert fit["d47"]["mirror"] and fit["d47"]["noise"] and fit["d47"]["pitch_file"] == digest(pitch_file)
    assert set(fit["d47"]["sign_map"]) == set(p3.MIRRORS)
