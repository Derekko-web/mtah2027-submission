"""D50 attacking sign, canonical tokens, trainer and inference (dfl_submission/direction_tokens.py)."""

import json

import numpy as np
import torch

from dfl_submission import direction_tokens as dt, player_set as ps, temporal_network
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


def formation(length=60, home_x=-20., away_x=20., per_team=11, extra=0, seed=0):
    """Two teams around their own x, with a few missing players; `extra` adds visible bench players."""
    rng = np.random.default_rng(seed)
    count = per_team + extra
    home = rng.normal([home_x, 0], [8, 15], size=(length, count, 2))
    away = rng.normal([away_x, 0], [8, 15], size=(length, count, 2))
    positions = np.concatenate([home, away], axis=1)
    teams = np.r_[np.zeros(count, int), np.ones(count, int)]
    ball = positions[:, 3] + .5
    return positions, ball, teams, np.array([105., 68.])


def test_sign_follows_the_away_team_side():
    positions, _, teams, _ = formation()
    result = dt.attacking(positions, teams)
    assert result["sign"] == 1 and result["first_sign"] == 1 and result["agrees"]
    assert result["d"] > 30 and result["frames"] == 12 and result["first_frame"] == 0
    flipped = positions.copy()
    flipped[..., 0] *= -1
    assert dt.attacking(flipped, teams)["sign"] == -1
    # Swapping the team labels swaps the sign.
    assert dt.attacking(positions, 1 - teams)["sign"] == -1


def test_frames_need_seven_visible_players_per_team():
    positions, _, teams, _ = formation()
    positions[:10, 4:11] = np.nan               # home keeps 4 visible players for the first 10 frames
    result = dt.attacking(positions, teams)
    assert result["first_frame"] == 10 and result["frames"] == 10   # long-context frames 10, 15, ..., 55
    positions[:, 5:11] = np.nan                  # home never has seven visible players
    result = dt.attacking(positions, teams)
    assert result["sign"] is None and result["frames"] == 0 and not result["agrees"]


def test_first_frame_disagreement_is_reported():
    positions, _, teams, _ = formation(length=100)
    positions[:6, :11, 0] += 60                  # home starts on the far side, then crosses back
    result = dt.attacking(positions, teams)
    assert result["sign"] == 1 and result["first_sign"] == -1 and not result["agrees"]


def test_d_is_the_mean_of_per_frame_team_mean_differences():
    positions, _, teams, _ = formation(length=20)
    positions[5, 0] = np.nan
    frames = dt.long_context_frames(20)
    now = positions[frames]
    expected = np.mean([np.nanmean(now[i, 11:, 0]) - np.nanmean(now[i, :11, 0]) for i in range(len(frames))])
    assert abs(dt.attacking(positions, teams)["d"] - expected) < 1e-12


def test_tokens_give_the_same_sign_and_see_at_most_eleven_players():
    positions, ball, teams, pitch = formation(length=200, seed=4)
    positions[40:80, 2:5] = np.nan
    frames = dt.long_context_frames(200)
    tokens = ps.tokens(positions, ball, teams, pitch, frames)
    tracked, stored = dt.attacking(positions, teams, frames), dt.attacking_from_tokens(tokens, frames)
    assert tracked["sign"] == stored["sign"] and tracked["first_sign"] == stored["first_sign"]
    assert abs(stored["d"] * pitch[0] / 2 - tracked["d"]) < 1e-3          # value 5 is x / half-length
    assert tracked["max_visible"] == [11, 11]
    # With 13 visible players a team, the tokens keep 11 and the two means can differ.
    positions, ball, teams, pitch = formation(length=50, extra=2, seed=5)
    assert dt.attacking(positions, teams)["max_visible"] == [13, 13]
    tokens = ps.tokens(positions, ball, teams, pitch, dt.long_context_frames(50))
    assert np.isfinite(tokens).all(axis=-1).sum(axis=-1).max() == ps.PER_TEAM


def test_canonical_tokens_flip_only_the_x_values_exactly():
    rng = np.random.default_rng(1)
    tokens = rng.normal(size=(30, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float16)
    tokens[:, 1, 8:] = np.nan
    assert dt.canonical(tokens, 1) is tokens
    flipped = dt.canonical(tokens, -1)
    assert flipped.dtype == np.float16
    np.testing.assert_array_equal(flipped[..., [0, 3, 5]], -tokens[..., [0, 3, 5]])
    np.testing.assert_array_equal(flipped[..., [1, 2, 4, 6, 7]], tokens[..., [1, 2, 4, 6, 7]])
    np.testing.assert_array_equal(dt.canonical(flipped, -1), tokens)
    assert np.flatnonzero(dt.X_SIGNS < 0).tolist() == [0, 3, 5]
    for bad in (0, 2):
        try:
            dt.canonical(tokens, bad)
        except ValueError:
            continue
        raise AssertionError("canonical accepted a sign other than +1 or -1")


def test_tokens_of_mirrored_tracking_are_canonical_alike():
    """A half and its x mirror (teams attacking the other way) give the same canonical tokens."""
    positions, ball, teams, pitch = formation(length=100, seed=2)
    frames = dt.long_context_frames(100)
    mirrored, mirrored_ball = positions.copy(), ball.copy()
    mirrored[..., 0] *= -1
    mirrored_ball[..., 0] *= -1
    a = ps.tokens(positions, ball, teams, pitch, frames)
    b = ps.tokens(mirrored, mirrored_ball, teams, pitch, frames)
    sign_a, sign_b = dt.attacking(positions, teams)["sign"], dt.attacking(mirrored, teams)["sign"]
    assert (sign_a, sign_b) == (1, -1)
    np.testing.assert_array_equal(dt.canonical(a, sign_a), dt.canonical(b, sign_b))


def test_swapped_window_negates_x_after_reversing_the_teams():
    rng = np.random.default_rng(3)
    window = rng.normal(size=(12, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    np.testing.assert_array_equal(dt.swapped(window, False), window[:, ::-1])
    result = dt.swapped(window, True)
    np.testing.assert_array_equal(result[:, 0], window[:, 1] * dt.X_SIGNS)
    np.testing.assert_array_equal(result[:, 1], window[:, 0] * dt.X_SIGNS)


def test_column_factors():
    assert dt.column_factors(None, -1) is None
    assert dt.column_factors([False, False], -1) is None
    np.testing.assert_array_equal(dt.column_factors([True, False, True], -1), [-1, 1, -1])
    np.testing.assert_array_equal(dt.column_factors([True, False, True], 1), [1, 1, 1])


def random_weights(columns=4, width=8, seed=0):
    torch.manual_seed(seed)
    net = ps.torch_network(columns, width, .2).eval()
    for parameter in net.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    return {name: tensor.detach().numpy() for name, tensor in net.state_dict().items()}


def inference_case(seed=9):
    weights = random_weights()
    rng = np.random.default_rng(seed)
    columns = ["h__a", "a__a", "h__b", "a__b"]
    checkpoint = (weights, columns, np.zeros(4, np.float32), np.ones(4, np.float32), np.array([1, 0, 3, 2]))
    frames = np.arange(0, 200, 5)
    values = rng.normal(size=(len(frames), 4)).astype(np.float32)
    player_tokens = rng.normal(size=(len(frames), 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    player_tokens[:, 0, 9:] = np.nan
    player_tokens[3:6, 1] = np.nan
    return checkpoint, columns, frames, values, player_tokens


def test_plain_inference_is_player_set_predict():
    checkpoint, columns, frames, values, player_tokens = inference_case()
    expected = ps.predict(checkpoint, frames, values, columns, player_tokens, 200)
    result = dt.predict(checkpoint, frames, values, columns, player_tokens, 200, sign=1, negate=False)
    for a, b in zip(expected, result, strict=True):
        np.testing.assert_array_equal(a, b)


def test_candidate_inference_is_invariant_to_the_attacking_direction_and_swap_symmetric():
    checkpoint, columns, frames, values, player_tokens = inference_case()
    forward = dt.predict(checkpoint, frames, values, columns, player_tokens, 200, sign=1)
    # The same half seen with the home team attacking -x: the tokens' x values flip and the sign is -1.
    mirrored = player_tokens * dt.X_SIGNS
    backward = dt.predict(checkpoint, frames, values, columns, mirrored, 200, sign=-1)
    for a, b in zip(forward, backward, strict=True):
        np.testing.assert_array_equal(a, b)
    # Reversing the team order (and the columns) mirrors home and keeps the rest: the swapped half's home
    # team is the old away team, which attacks -x in the old frame, so its sign is the opposite.
    swapped = dt.predict(checkpoint, frames, values[:, [1, 0, 3, 2]], columns, player_tokens[:, ::-1], 200, sign=-1)
    np.testing.assert_allclose(swapped[0], 1 - forward[0], atol=1e-6)
    for a, b in zip(forward[1:], swapped[1:], strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)
    # The negation matters: without it the canonical inference differs.
    plain = dt.predict(checkpoint, frames, values, columns, player_tokens, 200, sign=1, negate=False)
    assert np.abs(plain[0] - forward[0]).max() > 1e-4


def test_x_odd_columns_follow_the_sign_and_the_swap():
    checkpoint, columns, frames, values, player_tokens = inference_case()
    odd = np.array([False, False, True, True])
    forward = dt.predict(checkpoint, frames, values, columns, player_tokens, 200, sign=1, odd=odd)
    flipped_values = values * np.array([1, 1, -1, -1], np.float32)
    backward = dt.predict(checkpoint, frames, flipped_values, columns, player_tokens * dt.X_SIGNS, 200, sign=-1, odd=odd)
    for a, b in zip(forward, backward, strict=True):
        np.testing.assert_array_equal(a, b)
    swapped = dt.predict(checkpoint, frames, values[:, [1, 0, 3, 2]], columns, player_tokens[:, ::-1], 200, sign=-1,
                         odd=odd)
    np.testing.assert_allclose(swapped[0], 1 - forward[0], atol=1e-6)


def prepared_folder(tmp_path, halves=3, length=420):
    """A tiny prepared-sequence folder with player tokens, in the layout `player_set.training_data` reads."""
    folder, token_folder = tmp_path / "prepared", tmp_path / "tokens"
    folder.mkdir()
    token_folder.mkdir()
    rng = np.random.default_rng(3)
    records, token_records = [], {}
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
    (folder / "manifest.json").write_text(json.dumps({"columns": ["h__x", "a__x", "h__y", "a__y"], "records": records}))
    (token_folder / "manifest.json").write_text(json.dumps({"base_manifest": digest(folder / "manifest.json"),
                                                            "records": token_records}))
    return folder, token_folder, [record["file"] for record in records]


def test_check_batch_is_player_set_batch(tmp_path):
    folder, token_folder, names = prepared_folder(tmp_path)
    plain = ps.training_data(folder, token_folder)
    groups, _, permutation, location, scale = dt.training_data(folder, token_folder, {name: 1 for name in names})
    expected = ps.batch(plain[0], np.random.default_rng(5), permutation, location, scale, 8)
    result = dt.batch(groups, np.random.default_rng(5), permutation, location, scale, 8, negate=False)
    for a, b in zip(expected, result, strict=True):
        assert torch.equal(a, b)


def test_candidate_batch_keeps_the_window_stream_and_flips_x(tmp_path):
    folder, token_folder, names = prepared_folder(tmp_path)
    plain = ps.training_data(folder, token_folder)
    signs = {names[0]: -1, names[1]: 1, names[2]: -1}
    groups, _, permutation, location, scale = dt.training_data(folder, token_folder, signs)
    for (_, _, raw), (_, _, canonical), name in zip(plain[0]["dfl"], groups["dfl"], names, strict=True):
        np.testing.assert_array_equal(canonical, dt.canonical(raw, signs[name]))
    expected = ps.batch(plain[0], np.random.default_rng(6), permutation, location, scale, 12)
    result = dt.batch(groups, np.random.default_rng(6), permutation, location, scale, 12, negate=True)
    assert torch.equal(result[0], expected[0]) and torch.equal(result[2], expected[2]) and torch.equal(result[3], expected[3])
    ratio = (result[1] / torch.where(expected[1] == 0, torch.ones_like(expected[1]), expected[1]))[expected[2] > 0]
    assert torch.all((ratio.abs() - 1).abs() < 1e-6)
    assert (ratio < 0).any() and (ratio > 0).any()
    untouched = result[1][..., [1, 2, 4, 6, 7]] == expected[1][..., [1, 2, 4, 6, 7]]
    assert torch.all(untouched)
    try:
        dt.training_data(folder, token_folder, {names[0]: 1})
    except ValueError:
        return
    raise AssertionError("training_data accepted signs that do not cover the records")


def test_check_training_equals_player_set_train(tmp_path):
    folder, token_folder, names = prepared_folder(tmp_path)
    options = dict(seed=11, boundary_boost=.5, width=8, steps=6, batch_size=4)
    ps.train(folder, token_folder, tmp_path / "reference", "cpu", **options)
    dt.train(folder, token_folder, tmp_path / "check", "cpu", signs={name: 1 for name in names}, negate=False, **options)
    reference = json.loads((tmp_path / "reference" / "fit.json").read_text())
    check = json.loads((tmp_path / "check" / "fit.json").read_text())
    assert reference["models"] == check["models"]
    for name in reference["models"]:
        a = torch.load(tmp_path / "reference" / name, weights_only=True)
        b = torch.load(tmp_path / "check" / name, weights_only=True)
        assert all(torch.equal(a[key], b[key]) for key in a)
    assert check["d50"]["negate_on_swap"] is False and check["d50"]["negative_records"] == []
    # The candidate run differs, records its signs, and exports through player_set's checked export.
    dt.train(folder, token_folder, tmp_path / "candidate", "cpu", signs={names[0]: -1, names[1]: 1, names[2]: 1},
             negate=True, **options)
    candidate = json.loads((tmp_path / "candidate" / "fit.json").read_text())
    assert candidate["models"] != reference["models"]
    assert candidate["d50"]["negative_records"] == [names[0]] and candidate["d50"]["negate_on_swap"] is True
    ps.export(tmp_path / "candidate", tmp_path / "export")
    exported = json.loads((tmp_path / "export" / "fit.json").read_text())
    assert max(exported["export_max_logit_difference"].values()) < 2e-5


def test_read_signs_checks_the_folders(tmp_path):
    folder, token_folder, names = prepared_folder(tmp_path)
    path = tmp_path / "signs.json"
    path.write_text(json.dumps({"base_manifest": digest(folder / "manifest.json"),
                                "token_manifest": digest(token_folder / "manifest.json"),
                                "records": {name: {"sign": -1 if i else 1} for i, name in enumerate(names)}}))
    assert dt.read_signs(path, folder, token_folder) == {names[0]: 1, names[1]: -1, names[2]: -1}
    path.write_text(json.dumps({"base_manifest": "0" * 64, "token_manifest": digest(token_folder / "manifest.json"),
                                "records": {}}))
    try:
        dt.read_signs(path, folder, token_folder)
    except ValueError:
        return
    raise AssertionError("read_signs accepted signs for a different prepared folder")
