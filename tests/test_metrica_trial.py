"""D30 plumbing on synthetic data: Metrica training views, prepared halves, combined folders and gates."""

import json

import numpy as np

from dfl_submission import augment, metrica_trial as mt
from dfl_submission.robust_features import stress
from dfl_transfer_features import corrupt
from tests.test_metrica_views import synthetic_half


def test_original_set_views_are_dfl_training_views_with_d30_seeds():
    half = synthetic_half(1)
    views = mt.training_views(half, 2, 1, "current")
    assert [name for name, _, _ in views] == ["clean", "masked", "stress"]
    np.testing.assert_array_equal(views[0][1], half["positions"])
    np.testing.assert_array_equal(views[0][2], half["ball"])
    masked = corrupt(half["positions"], half["ball"], 870000 + 10 * 2 + 1)
    stressed = stress(half["positions"], half["ball"], 880000 + 10 * 2 + 1)
    for got, expected in zip(views[1][1:], masked):
        np.testing.assert_array_equal(got, expected)
    for got, expected in zip(views[2][1:], stressed):
        np.testing.assert_array_equal(got, expected)


def test_augmented_set_views_follow_augment_training_halves_order():
    half = synthetic_half(2)
    views = mt.training_views(half, 3, 0, "augmented")
    assert [name for name, _, _ in views] == ["clean", "masked", "stress"]
    rng = np.random.default_rng(890000 + 10 * 3 + 0)
    positions = np.asarray(half["positions"], dtype=np.float32)
    ball = augment.handle_dead_ball(half["labels"], positions, half["ball"], rng)
    masked_positions, masked_ball = augment.wide_masking(positions, ball, rng)
    masked_positions = augment.swap_teams(masked_positions, half["teams"], rng, every=augment.TRAIN_SWAP_EVERY,
                                          frames=augment.training_frames)
    stress_positions, stress_ball = stress(positions, ball, 880000 + 10 * 3 + 0)
    stress_positions = augment.swap_teams(stress_positions, half["teams"], rng, every=augment.TRAIN_SWAP_EVERY,
                                          frames=augment.training_frames)
    expected = [(positions, ball), (masked_positions, masked_ball), (stress_positions, stress_ball)]
    for (_, got_positions, got_ball), (want_positions, want_ball) in zip(views, expected):
        np.testing.assert_array_equal(got_positions, want_positions)
        np.testing.assert_array_equal(got_ball, want_ball)


def test_unknown_model_set_is_refused():
    try:
        mt.training_views(synthetic_half(), 1, 0, "other")
    except ValueError:
        return
    raise AssertionError("an unknown model set was accepted")


def prepared_metrica(tmp_path, model_set="current"):
    items = [(game, index, synthetic_half(10 * game + index)) for game in (1, 2, 3) for index in (0, 1)]
    folder = tmp_path / f"metrica_{model_set}"
    mt.prepare_halves(items, folder, model_set, inputs={"test": True})
    return folder


def test_prepared_metrica_halves_join_the_dfl_source_and_name_their_vendor(tmp_path):
    folder = prepared_metrica(tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["protocol"] == mt.PROTOCOL and manifest["set"] == "current" and manifest["inputs"] == {"test": True}
    records = manifest["records"]
    assert len(records) == 3 * 2 * 3
    assert {record["source"] for record in records} == {"dfl"}
    assert {record["vendor"] for record in records} == {"metrica"}
    assert {record["match"] for record in records} == {"metrica1", "metrica2", "metrica3"}
    assert records[0]["file"] == "metrica_1_clean_0.npz"
    with np.load(folder / records[0]["file"], allow_pickle=False) as item:
        assert item["values"].shape[1] == len(manifest["columns"])
        assert item["targets"].shape[0] == item["values"].shape[0]


def original_folder(tmp_path, columns):
    folder = tmp_path / "original"
    folder.mkdir()
    records = []
    for name in ("dfl_WMX_clean_0.npz", "skillcorner_1_observed_0.npz"):
        np.savez_compressed(folder / name, values=np.ones((40, len(columns)), dtype=np.float32),
                            targets=np.zeros((40, 6), dtype=np.float32))
        from dfl_submission.pff import digest
        records.append({"file": name, "source": name.split("_")[0], "match": name.split("_")[1],
                        "sha256": digest(folder / name)})
    (folder / "manifest.json").write_text(json.dumps({"columns": columns, "records": records, "split": "train"}))
    return folder


def test_combine_keeps_the_original_records_first_and_only_the_chosen_games(tmp_path):
    metrica = prepared_metrica(tmp_path)
    columns = json.loads((metrica / "manifest.json").read_text())["columns"]
    original = original_folder(tmp_path, columns)
    mt.combine(original, metrica, tmp_path / "fold2", games=(1, 3))
    manifest = json.loads((tmp_path / "fold2" / "manifest.json").read_text())
    files = [record["file"] for record in manifest["records"]]
    assert files[:2] == ["dfl_WMX_clean_0.npz", "skillcorner_1_observed_0.npz"]
    assert {record["match"] for record in manifest["records"][2:]} == {"metrica1", "metrica3"}
    assert len(files) == 2 + 2 * 2 * 3
    assert manifest["metrica_games"] == [1, 3] and manifest["columns"] == columns
    from dfl_submission import aligned_trial
    assert len(list(aligned_trial.verified_records(tmp_path / "fold2"))) == len(files)


def test_combine_with_no_game_keeps_exactly_the_original_records(tmp_path):
    metrica = prepared_metrica(tmp_path)
    columns = json.loads((metrica / "manifest.json").read_text())["columns"]
    original = original_folder(tmp_path, columns)
    mt.combine(original, metrica, tmp_path / "none", games=())
    combined = json.loads((tmp_path / "none" / "manifest.json").read_text())
    stored = json.loads((original / "manifest.json").read_text())
    assert combined["records"] == stored["records"] and combined["columns"] == stored["columns"]


def test_combine_refuses_a_different_feature_schema(tmp_path):
    metrica = prepared_metrica(tmp_path)
    original = original_folder(tmp_path, ["a", "b"])
    try:
        mt.combine(original, metrica, tmp_path / "bad", games=(1,))
    except ValueError:
        return
    raise AssertionError("a different schema was combined")


def test_same_arrays_compares_every_array_of_every_npz(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        np.savez(tmp_path / name / "temporal_5400.npz", w=np.arange(4.0), b=np.ones(2))
        np.savez(tmp_path / name / "normalization.npz", location=np.zeros(3))
    assert mt.same_arrays(tmp_path / "a", tmp_path / "b") == {"normalization.npz": True, "temporal_5400.npz": True}
    np.savez(tmp_path / "b" / "temporal_5400.npz", w=np.arange(4.0) + 1e-7, b=np.ones(2))
    assert mt.same_arrays(tmp_path / "a", tmp_path / "b")["temporal_5400.npz"] is False
    (tmp_path / "b" / "normalization.npz").unlink()
    assert mt.same_arrays(tmp_path / "a", tmp_path / "b")["normalization.npz"] is False


def scores(value):
    """{game: {view: {target: {"spell_f1": value, ...}}}} for every held-out game, view and target."""
    return {game: {view: {target: {"spell_f1": value, "macro_f1": .9} for target in mt.LOGO_TARGETS}
                   for view in mt.LOGO_VIEWS} for game in ("1", "2", "3")}


def test_logo_passes_on_a_gain_of_a_point_on_every_game():
    reference = scores(.70)
    candidate = scores(.70)
    for game in ("1", "2", "3"):
        for view in ("masked", "handled", "team_swap"):
            candidate[game][view]["aba3"]["spell_f1"] = .7101
    gates = mt.logo_gates(reference, candidate)
    assert gates["passed"] and abs(gates["d24_change"] - .0101) < 1e-12


def test_logo_fails_when_one_game_is_lower_even_with_a_large_mean_gain():
    reference = scores(.70)
    candidate = scores(.70)
    for view in ("masked", "handled", "team_swap"):
        candidate["1"][view]["aba3"]["spell_f1"] = .76
        candidate["2"][view]["aba3"]["spell_f1"] = .76
        candidate["3"][view]["aba3"]["spell_f1"] = .6999
    gates = mt.logo_gates(reference, candidate)
    assert gates["d24_change"] > .01 and not gates["no_game_lower"] and not gates["passed"]


def test_logo_fails_when_the_clean_view_is_lower():
    reference = scores(.70)
    candidate = scores(.72)
    for game in ("1", "2", "3"):
        candidate[game]["clean"]["aba3"]["spell_f1"] = .6999
    gates = mt.logo_gates(reference, candidate)
    assert gates["d24_change"] > .01 and not gates["clean_not_lower"] and not gates["passed"]


def test_logo_fails_when_another_target_drops_more_than_a_point():
    reference = scores(.70)
    candidate = scores(.72)
    for game in ("1", "2", "3"):
        for view in mt.LOGO_VIEWS:
            candidate[game][view]["control"]["spell_f1"] = .6899
    gates = mt.logo_gates(reference, candidate)
    assert gates["other_targets"]["control"] < -.01 and not gates["passed"]


def change(primary=0., carried=0., swap=0., pff=0.):
    return {"primary": primary, "conventions": {"aba1.75": 0., "aba4": 0., "aba4.5": 0.},
            "guards": {"dead_carried": carried, "team_swap": swap}, "pff": pff}


def test_fold_development_guard_allows_half_a_point_on_the_standard_views():
    assert mt.development_guard(change(-.005, -.01, -.01, -.01))
    assert not mt.development_guard(change(-.0051))
    assert not mt.development_guard(change(carried=-.0101))
    assert not mt.development_guard(change(pff=-.0101))


def test_confirmation_allows_half_a_point_on_pff_set_b():
    assert mt.confirmation_passed(change(-.005, -.01, -.01, -.005))
    assert not mt.confirmation_passed(change(pff=-.0051))
    assert not mt.confirmation_passed(change(primary=-.0051))
    assert not mt.confirmation_passed(change(swap=-.0101))
