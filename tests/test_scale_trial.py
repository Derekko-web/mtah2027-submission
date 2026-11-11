"""D15 trial plumbing: candidate packages, snapshots, eligibility and training inputs."""

import json

import numpy as np

from dfl_submission import retrain_trial, scale_trial


def now():
    root = scale_trial.ROOT
    package = scale_trial.PACKAGE
    return retrain_trial.current(root, package / "seed_models", package / "ensemble_recipe", package / "decoder_recipe")


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if hasattr(a, "get_booster"):
        return bytes(a.get_booster().save_raw()) == bytes(b.get_booster().save_raw())
    if hasattr(a, "save_raw"):
        return bytes(a.save_raw()) == bytes(b.save_raw())
    return a == b


def test_reference_setting_loads_exactly_the_published_d3_model_sets():
    current = now()
    for key, published in scale_trial.PUBLISHED.items():
        folders = scale_trial.network_folders(scale_trial.REFERENCE)
        assert folders[key] == published
        assert same(scale_trial.package(scale_trial.TREES[key], folders[key], current),
                    retrain_trial.candidate(published, current))


def test_final_snapshot_follows_the_training_length(tmp_path):
    folder = tmp_path / "net"
    folder.mkdir()
    (folder / "fit.json").write_text(json.dumps({"models": {"temporal_600.npz": "", "temporal_1200.npz": "",
                                                            "temporal_1800.npz": ""}}))
    assert scale_trial.final_snapshot(folder) == "temporal_1800.npz"
    (folder / "fit.json").write_text(json.dumps({"steps": 5400, "models": {"temporal_1800.npz": "", "temporal_3600.npz": "",
                                                                           "temporal_5400.npz": ""}}))
    assert scale_trial.final_snapshot(folder) == "temporal_5400.npz"


def change(primary):
    return {"primary": primary, "conventions": {"aba1.75": 0, "aba4": 0, "aba4.5": 0},
            "guards": {"dead_carried": 0, "team_swap": 0}, "pff": 0}


def test_only_new_networks_are_eligible():
    assert not scale_trial.eligible({"width": 48, "steps": 1800, "owner_cost": 4}, change(.05))
    assert scale_trial.eligible({"width": 96, "steps": 1800, "owner_cost": 6}, change(.01))
    assert not scale_trial.eligible({"width": 96, "steps": 1800, "owner_cost": 6}, change(.009))
    failing_guard = dict(change(.02), pff=-.011)
    assert not scale_trial.eligible({"width": 48, "steps": 5400, "owner_cost": 8}, failing_guard)


def test_training_refuses_the_reference_setting_and_foreign_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}")
    for setting in (scale_trial.REFERENCE, (64, 1800)):
        try:
            scale_trial.train_networks(source, "current", setting, tmp_path / "scratch", models=tmp_path / "models")
        except ValueError as error:
            assert "not a D15 candidate" in str(error)
        else:
            raise AssertionError("expected a refusal")
    try:
        scale_trial.train_networks(source, "current", (96, 1800), tmp_path / "scratch", models=tmp_path / "models")
    except ValueError as error:
        assert "not the prepared-sequence folder" in str(error)
    else:
        raise AssertionError("expected a refusal")
    assert not (tmp_path / "scratch").exists()
