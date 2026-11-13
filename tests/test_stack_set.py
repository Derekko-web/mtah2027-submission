"""D48 stack trainer and inference: each switch alone reduces to the module it reuses (dfl_submission/stack_set.py)."""

import json

import numpy as np
import torch

from dfl_submission import direction_tokens as dt, player_set as ps, player_set3 as p3, stack_set as ss, temporal_network
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


OPTIONS = dict(seed=11, boundary_boost=.5, width=8, steps=6, batch_size=4)


def prepared_folder(tmp_path, halves=4, length=420):
    """A tiny prepared-sequence folder with tokens, auxiliary targets and pitch sizes."""
    folder, token_folder, aux_folder = tmp_path / "prepared", tmp_path / "tokens", tmp_path / "aux"
    for path in (folder, token_folder, aux_folder):
        path.mkdir()
    rng = np.random.default_rng(3)
    records, token_records, aux_records, pitches = [], {}, {}, {}
    for index in range(halves):
        labels = np.where(np.arange(length) % 90 < 45, pm.H, pm.A).astype(np.int8)
        labels[200:230] = pm.D
        values = rng.normal(size=(length, 4)).astype(np.float32)
        source = "dfl" if index < 2 else "skillcorner"
        name = f"{source}_M{index // 2}_clean_{index % 2}.npz"
        np.savez_compressed(folder / name, values=values, targets=temporal_network.targets(labels))
        records.append({"file": name, "source": source, "match": f"M{index // 2}", "sha256": digest(folder / name)})
        tokens = rng.normal(size=(length, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float16)
        tokens[:, :, 9:] = np.nan
        np.savez_compressed(token_folder / name, tokens=tokens)
        token_records[name] = digest(token_folder / name)
        if index % 2:
            offset = rng.normal(size=(length, 2)).astype(np.float32)
            offset[::3] = np.nan
            np.savez_compressed(aux_folder / name, offset=offset)
            aux_records[name] = digest(aux_folder / name)
        pitches[name] = [105. + index, 68.]
    (folder / "manifest.json").write_text(json.dumps({"columns": ["h__x", "a__x", "h__y", "a__y"], "records": records}))
    (token_folder / "manifest.json").write_text(json.dumps({"base_manifest": digest(folder / "manifest.json"),
                                                            "records": token_records}))
    (aux_folder / "manifest.json").write_text(json.dumps({"base_manifest": digest(folder / "manifest.json"),
                                                          "records": aux_records}))
    (tmp_path / "pitches.json").write_text(json.dumps(pitches))
    names = [record["file"] for record in records]
    return folder, token_folder, aux_folder, tmp_path / "pitches.json", names


def same_runs(a, b):
    first, second = (json.loads((path / "fit.json").read_text()) for path in (a, b))
    if first["models"] != second["models"]:
        return False
    for name in first["models"]:
        x, y = torch.load(a / name, weights_only=True), torch.load(b / name, weights_only=True)
        if not all(torch.equal(x[key], y[key]) for key in x):
            return False
    return True


SIGN_MAP = {"columns": ["h__x", "a__x", "h__y", "a__y"], "signs": {"x": [1, 1, 1, 1], "y": [1, 1, 1, 1], "xy": [1, 1, 1, 1]}}


def test_defaults_are_player_set_train(tmp_path):
    folder, tokens, _, _, _ = prepared_folder(tmp_path)
    ps.train(folder, tokens, tmp_path / "d42", "cpu", **OPTIONS)
    ss.train(folder, tokens, tmp_path / "stack", "cpu", **OPTIONS)
    assert same_runs(tmp_path / "d42", tmp_path / "stack")


def test_direction_alone_is_direction_tokens_train(tmp_path):
    folder, tokens, _, _, names = prepared_folder(tmp_path)
    signs = {name: (-1 if i % 2 else 1) for i, name in enumerate(names)}
    dt.train(folder, tokens, tmp_path / "d50", "cpu", signs=signs, negate=True, **OPTIONS)
    ss.train(folder, tokens, tmp_path / "stack", "cpu", config=ss.config_of(direction=True), signs=signs, **OPTIONS)
    assert same_runs(tmp_path / "d50", tmp_path / "stack")


def test_seed_offset_mirror_and_noise_alone_are_player_set3_train(tmp_path):
    folder, tokens, _, pitch_file, _ = prepared_folder(tmp_path)
    cases = [({"seed_offset": 1000, "mirror": False, "noise": False}, ss.config_of(seed_offset=1000)),
             ({"seed_offset": 0, "mirror": True, "noise": False}, ss.config_of(mirror="xy")),
             ({"seed_offset": 0, "mirror": False, "noise": True}, ss.config_of(noise=True))]
    for index, (d47, stack) in enumerate(cases):
        p3.train(folder, tokens, tmp_path / f"d47_{index}", "cpu", config=d47, sign_map=SIGN_MAP if d47["mirror"] else None,
                 pitch_file=pitch_file if d47["noise"] else None, **OPTIONS)
        ss.train(folder, tokens, tmp_path / f"stack_{index}", "cpu", config=stack,
                 sign_map=SIGN_MAP if stack["mirror"] else None, pitch_file=pitch_file if stack["noise"] else None, **OPTIONS)
        assert same_runs(tmp_path / f"d47_{index}", tmp_path / f"stack_{index}"), d47
    fit = json.loads((tmp_path / "stack_0" / "fit.json").read_text())
    assert fit["seed"] == 1011 and fit["stack"]["seed_offset"] == 1000


def test_aux_alone_is_hidden_ball_train(tmp_path):
    from dfl_submission import hidden_ball as hb
    folder, tokens, aux, _, _ = prepared_folder(tmp_path)
    hb.train(folder, tokens, aux, tmp_path / "d51", "cpu", **OPTIONS)
    ss.train(folder, tokens, tmp_path / "stack", "cpu", config=ss.config_of(aux=True), aux_folder=aux, **OPTIONS)
    assert same_runs(tmp_path / "d51", tmp_path / "stack")
    d51 = json.loads((tmp_path / "d51" / "fit.json").read_text())["ball"]["heads"]
    stack = json.loads((tmp_path / "stack" / "fit.json").read_text())["stack"]["aux"]["heads"]
    assert d51 == stack


def test_direction_puts_the_aux_target_in_the_attacking_frame(tmp_path):
    folder, tokens, aux, _, names = prepared_folder(tmp_path)
    signs = {name: -1 for name in names}
    plain, *_ = ss.training_data(folder, tokens, aux_folder=aux)
    canonical, *_ = ss.training_data(folder, tokens, signs=signs, aux_folder=aux)
    for key in plain:
        for (_, _, tok, target), (_, _, ctok, ctarget) in zip(plain[key], canonical[key], strict=True):
            np.testing.assert_array_equal(ctok, dt.canonical(tok, -1))
            if target is None:
                assert ctarget is None
                continue
            np.testing.assert_array_equal(ctarget[:, 0], -target[:, 0])
            np.testing.assert_array_equal(ctarget[:, 1], target[:, 1])


def test_batch_swap_negates_the_aux_x_with_direction(tmp_path):
    folder, tokens, aux, _, names = prepared_folder(tmp_path)
    signs = {name: 1 for name in names}
    groups, _, permutation, location, scale = ss.training_data(folder, tokens, signs=signs, aux_folder=aux)
    plain = ss.batch(groups, np.random.default_rng(4), None, permutation, location, scale, 16, ss.config_of(aux=True))
    framed = ss.batch(groups, np.random.default_rng(4), None, permutation, location, scale, 16,
                      ss.config_of(aux=True, direction=True))
    assert torch.equal(plain[0], framed[0]) and torch.equal(plain[3], framed[3])
    present = torch.isfinite(plain[4]).all(dim=-1)
    ratio = framed[4][..., 0][present] / plain[4][..., 0][present]
    assert set(torch.unique(torch.sign(ratio)).tolist()) == {-1.0, 1.0}         # swapped windows only
    assert torch.equal(framed[4][..., 1][present], plain[4][..., 1][present])


def test_validate():
    for bad in ({"direction": True, "mirror": "xy"}, {"mirror": "x"}, {"seed_offset": 5}):
        try:
            ss.config_of(**bad)
        except ValueError:
            continue
        raise AssertionError(f"config_of accepted {bad}")
    assert ss.mirrors_of(ss.config_of(direction=True, mirror="y")) == ("identity", "y")
    try:
        ss.validate(dict(ss.DEFAULT, extra=1))
    except ValueError:
        return
    raise AssertionError("validate accepted an unknown switch")


def test_other_half_and_fallback():
    assert ss.other_half("pff_3852_observed_1.npz") == "pff_3852_observed_0.npz"
    assert ss.other_half("dfl_WOH_long_hidden_0.npz") == "dfl_WOH_long_hidden_1.npz"
    entries = {"a_0.npz": {"sign": 1}, "a_1.npz": {"sign": None}, "b_0.npz": {"sign": None}, "b_1.npz": {"sign": None},
               "c_0.npz": {"sign": -1}, "c_1.npz": {"sign": 1}}
    signs, fallbacks = ss.with_fallback(entries, ss.other_half)
    assert signs == {"a_0.npz": 1, "a_1.npz": -1, "b_0.npz": 1, "b_1.npz": 1, "c_0.npz": -1, "c_1.npz": 1}
    assert sorted(fallbacks) == ["a_1.npz", "b_0.npz", "b_1.npz"]


def random_weights(columns=4, width=8, seed=0):
    torch.manual_seed(seed)
    net = ps.torch_network(columns, width, .2).eval()
    for parameter in net.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    return {name: tensor.detach().numpy() for name, tensor in net.state_dict().items()}


def inference_case(seed=9):
    rng = np.random.default_rng(seed)
    columns = ["h__a", "a__a", "h__b", "a__b"]
    checkpoint = (random_weights(), columns, np.zeros(4, np.float32), np.ones(4, np.float32), np.array([1, 0, 3, 2]))
    frames = np.arange(0, 200, 5)
    values = rng.normal(size=(len(frames), 4)).astype(np.float32)
    player_tokens = rng.normal(size=(len(frames), 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    player_tokens[:, 0, 9:] = np.nan
    return checkpoint, columns, frames, values, player_tokens


def test_predict_reduces_to_the_modules_it_reuses():
    checkpoint, columns, frames, values, tokens = inference_case()
    pairs = [(ps.predict(checkpoint, frames, values, columns, tokens, 200),
              ss.predict(checkpoint, frames, values, columns, tokens, 200)),
             (dt.predict(checkpoint, frames, values, columns, tokens, 200, sign=-1),
              ss.predict(checkpoint, frames, values, columns, tokens, 200, config=ss.config_of(direction=True), sign=-1)),
             (p3.predict(checkpoint, frames, values, columns, tokens, 200, mirrored=True, sign_map={
                 "columns": columns, "signs": {"x": [1] * 4, "y": [1] * 4, "xy": [1] * 4}}),
              ss.predict(checkpoint, frames, values, columns, tokens, 200, config=ss.config_of(mirror="xy"),
                         sign_map={"columns": columns, "signs": {"x": [1] * 4, "y": [1] * 4, "xy": [1] * 4}}))]
    for expected, result in pairs:
        for a, b in zip(expected, result, strict=True):
            np.testing.assert_array_equal(a, b)
    try:
        ss.predict(checkpoint, frames, values, columns, tokens, 200, config=ss.config_of(direction=True))
    except ValueError:
        return
    raise AssertionError("predict accepted `direction` without a sign")


def test_y_mirror_with_direction_is_direction_and_y_invariant():
    checkpoint, columns, frames, values, tokens = inference_case()
    sign_map = {"columns": columns, "signs": {"x": [1] * 4, "y": [1] * 4, "xy": [1] * 4}}
    config = ss.config_of(direction=True, mirror="y")
    forward = ss.predict(checkpoint, frames, values, columns, tokens, 200, config=config, sign=1, sign_map=sign_map)
    backward = ss.predict(checkpoint, frames, values, columns, tokens * dt.X_SIGNS, 200, config=config, sign=-1,
                          sign_map=sign_map)
    flipped = ss.predict(checkpoint, frames, values, columns, tokens * p3.TOKEN_SIGNS["y"], 200, config=config, sign=1,
                         sign_map=sign_map)
    for a, b, c in zip(forward, backward, flipped, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(a, c, atol=1e-6)
    swapped = ss.predict(checkpoint, frames, values[:, [1, 0, 3, 2]], columns, tokens[:, ::-1], 200, config=config,
                         sign=-1, sign_map=sign_map)
    np.testing.assert_allclose(swapped[0], 1 - forward[0], atol=1e-6)


def test_read_signs_applies_the_fallback(tmp_path):
    folder, tokens, _, _, names = prepared_folder(tmp_path)
    path = tmp_path / "signs.json"
    records = {name: {"sign": None if name.endswith("_1.npz") and name.startswith("dfl") else -1} for name in names}
    path.write_text(json.dumps({"base_manifest": digest(folder / "manifest.json"),
                                "token_manifest": digest(tokens / "manifest.json"), "records": records}))
    signs = ss.read_signs(path, folder, tokens)
    assert signs["dfl_M0_clean_1.npz"] == 1 and signs["dfl_M0_clean_0.npz"] == -1


def tracking(seed=2, length=100):
    rng = np.random.default_rng(seed)
    home, away = rng.normal([-20, 3], [8, 12], size=(length, 11, 2)), rng.normal([20, -3], [8, 12], size=(length, 11, 2))
    positions = np.concatenate([home, away], axis=1)
    positions[30:40, 2] = np.nan
    ball = positions[:, 4] + .5
    ball[50:60] = np.nan
    return positions, ball, np.r_[np.zeros(11, int), np.ones(11, int)], np.array([105., 68.])


def test_in_the_attacking_frame_a_y_mirror_flips_the_y_values_and_an_x_mirror_changes_nothing():
    positions, ball, teams, pitch = tracking()
    frames = dt.long_context_frames(len(ball))
    sign = dt.attacking(positions, teams)["sign"]
    base = dt.canonical(ps.tokens(positions, ball, teams, pitch, frames), sign)
    for mirror in ("x", "y", "xy"):
        mirrored, mirrored_ball = p3.mirror_tracking(positions, ball, mirror)
        mirrored_sign = dt.attacking(mirrored, teams)["sign"]
        assert mirrored_sign == (-sign if "x" in mirror else sign)
        result = dt.canonical(ps.tokens(mirrored, mirrored_ball, teams, pitch, frames), mirrored_sign)
        expected = base * p3.TOKEN_SIGNS["y"] if "y" in mirror else base
        np.testing.assert_array_equal(result, expected)


def test_y_mirror_batch_with_direction_flips_only_y_in_about_half_the_windows(tmp_path):
    folder, tokens, _, _, names = prepared_folder(tmp_path)
    groups, _, permutation, location, scale = ss.training_data(folder, tokens, signs={name: -1 for name in names})
    plain = ss.batch(groups, np.random.default_rng(8), None, permutation, location, scale, 64,
                     ss.config_of(direction=True))
    augment = np.random.default_rng([8, p3.AUGMENT_STREAM])
    signs = {mirror: np.ones(4, np.float32) for mirror in p3.MIRRORS}
    mirrored = ss.batch(groups, np.random.default_rng(8), augment, permutation, location, scale, 64,
                        ss.config_of(direction=True, mirror="y"), column_signs=signs)
    assert torch.equal(plain[0], mirrored[0]) and torch.equal(plain[2], mirrored[2]) and torch.equal(plain[3], mirrored[3])
    x_values, y_values = [0, 2, 3, 5, 7], [1, 4, 6]
    assert torch.equal(plain[1][..., x_values], mirrored[1][..., x_values])          # x never changes
    present = plain[2] > 0
    ratio = mirrored[1][..., 1] / torch.where(plain[1][..., 1] == 0, torch.ones_like(plain[1][..., 1]), plain[1][..., 1])
    flipped = torch.stack([(ratio[i][present[i]] < 0).all() for i in range(len(ratio))])
    same = torch.stack([(ratio[i][present[i]] > 0).all() for i in range(len(ratio))])
    assert torch.all(flipped | same)                                                 # whole windows flip
    assert 20 <= int(flipped.sum()) <= 44                                            # about half of 64
    for index in y_values:
        assert torch.equal(mirrored[1][..., index].abs(), plain[1][..., index].abs())
