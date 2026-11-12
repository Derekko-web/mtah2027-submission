"""D23 plumbing on synthetic data: the standing recipe, the prepare check and the adoption rule."""

import json

import numpy as np

from dfl_submission import final_trial, scale_trial, zoo


def test_standing_recipe_is_d15_until_d22_is_confirmed():
    record = final_trial.standing()
    if (final_trial.PACKAGE / "triple_recipe" / "confirmation.json").exists():
        return  # after D22's confirmation the standing recipe depends on its outcome
    assert record["name"] == "D15" and record["point"] == {"width": 96, "steps": 5400, "owner_cost": 8}
    folders = scale_trial.network_folders(zoo.d15_setting()[0])
    assert record["slots"]["augmented"]["seed_models"] == [folders["augmented"] / "seed_models"]
    trees, slots = final_trial.candidate_folders(record, final_trial.MODELS)
    assert slots["current"]["temporal_models"] == [final_trial.MODELS / "current" / "temporal_models_seed42"]
    assert slots["augmented"]["seed_models"] == [final_trial.MODELS / "augmented" / "seed_models_seed97"]
    assert trees["current"] == final_trial.MODELS / "current" / "long_models"


def prepared(folder, records):
    folder.mkdir()
    entries = []
    for name, values in records.items():
        np.savez_compressed(folder / name, values=values, targets=np.ones((len(values), 6), dtype=np.float32))
        entries.append({"file": name, "source": "dfl", "match": name.split("_")[1]})
    (folder / "manifest.json").write_text(json.dumps({"columns": ["a", "b"], "records": entries}))
    return folder


def test_prepare_check_needs_every_stored_half_exactly(tmp_path):
    a = np.arange(10, dtype=np.float32).reshape(5, 2)
    b = a.copy()
    b[1, 1] = np.nan
    stored = prepared(tmp_path / "stored", {"dfl_WMX_clean_0.npz": a, "dfl_WMX_clean_1.npz": b})
    same = prepared(tmp_path / "same", {"dfl_WMX_clean_0.npz": a, "dfl_WMX_clean_1.npz": b.copy(),
                                        "dfl_WPY_clean_0.npz": a + 1})
    assert final_trial.compare_prepared(stored, same) == {"dfl_WMX_clean_0.npz": True, "dfl_WMX_clean_1.npz": True}
    changed = b.copy()
    changed[0, 0] += 1e-6
    other = prepared(tmp_path / "other", {"dfl_WMX_clean_0.npz": a, "dfl_WMX_clean_1.npz": changed})
    assert final_trial.compare_prepared(stored, other)["dfl_WMX_clean_1.npz"] is False
    missing = prepared(tmp_path / "missing", {"dfl_WMX_clean_0.npz": a})
    assert final_trial.compare_prepared(stored, missing)["dfl_WMX_clean_1.npz"] is False


def test_adoption_needs_no_drop_over_a_point_and_a_non_negative_mean():
    assert final_trial.adopted({"pff": .004, "metrica": -.002, "legacy": .0})[0]
    assert not final_trial.adopted({"pff": .03, "metrica": -.0101, "legacy": .02})[0]
    assert not final_trial.adopted({"pff": .002, "metrica": -.005, "legacy": .0})[0]
    assert final_trial.adopted({"pff": 0., "metrica": 0., "legacy": 0.}) == (True, 0.)
