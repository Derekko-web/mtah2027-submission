"""D13 retargeting and tree inputs on synthetic prepared folders."""

import json

import numpy as np

from dfl_submission import aligned_trial, temporal_network
from dfl_submission.conventions import collapse_brief_changes
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def prepared(folder, halves):
    """A prepared folder in the layout of `temporal_network.prepare`."""
    folder.mkdir()
    records = []
    for (source, match, view, index), full in halves.items():
        frames = np.arange(0, len(full), 5)
        name = f"{source}_{match}_{view}_{index}.npz"
        values = np.arange(len(frames) * 2, dtype=np.float32).reshape(-1, 2)
        np.savez_compressed(folder / name, values=values, targets=temporal_network.targets(full[frames]))
        records.append({"file": name, "source": source, "match": match, "sha256": digest(folder / name)})
    (folder / "manifest.json").write_text(json.dumps({"columns": ["h__x", "a__x"], "records": records}))


def test_record_key():
    assert aligned_trial.record_key("dfl_WMX_stress_1.npz") == ("dfl", "WMX", "stress", 1)
    assert aligned_trial.record_key("skillcorner_2006363_observed_0.npz") == ("skillcorner", "2006363", "observed", 0)


def test_aligned_targets_equal_raw_targets_without_brief_spells():
    full = labels((D, 100), (H, 300), (A, 300), (D, 50), (H, 200))
    assert np.array_equal(temporal_network.convention_labels(full, 3.0), full)
    assert temporal_network.convention_labels(full, None) is full


def test_retarget_collapses_ownership_and_keeps_stoppage_targets(tmp_path):
    brief = labels((H, 300), (A, 40), (H, 300), (D, 100), (A, 200))
    plain = labels((A, 400), (D, 50), (H, 400))
    halves = {("dfl", "WMX", "clean", 0): brief, ("dfl", "WMX", "masked", 0): brief,
              ("skillcorner", "1", "observed", 1): plain}
    source = tmp_path / "source"
    prepared(source, halves)
    full = {("dfl", "WMX", 0): brief, ("skillcorner", "1", 1): plain}
    aligned_trial.retarget(source, full, tmp_path / "aligned")
    manifest = json.loads((tmp_path / "aligned" / "manifest.json").read_text())
    assert manifest["convention"] == 3.0 and manifest["source_manifest"] == digest(source / "manifest.json")
    with np.load(tmp_path / "aligned" / "dfl_WMX_clean_0.npz") as item:
        frames = np.arange(0, len(brief), 5)
        expected = temporal_network.targets(collapse_brief_changes(brief, 3.0)[frames])
        assert np.array_equal(item["targets"], expected)
        original = temporal_network.targets(brief[frames])
        # Stoppage, stoppage-boundary, known and live columns never change.
        assert np.array_equal(item["targets"][:, [0, 2, 4, 5]], original[:, [0, 2, 4, 5]])
        assert not np.array_equal(item["targets"][:, 1], original[:, 1])
    with np.load(tmp_path / "aligned" / "skillcorner_1_observed_1.npz") as item, \
            np.load(source / "skillcorner_1_observed_1.npz") as before:
        assert np.array_equal(item["targets"], before["targets"]) and np.array_equal(item["values"], before["values"])


def test_retarget_refuses_labels_that_do_not_reproduce_the_stored_targets(tmp_path):
    full = labels((H, 300), (A, 300))
    prepared(tmp_path / "source", {("dfl", "WMX", "clean", 0): full})
    try:
        aligned_trial.retarget(tmp_path / "source", {("dfl", "WMX", 0): labels((A, 600))}, tmp_path / "aligned")
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched labels must stop the retarget")


def test_ownership_rows_use_live_frames_and_home_targets(tmp_path):
    full = labels((D, 50), (H, 100), (A, 100))
    prepared(tmp_path / "source", {("dfl", "WMX", "clean", 0): full})
    rows = aligned_trial.ownership_rows(tmp_path / "source")
    (table, home, groups), = rows
    assert list(table.columns) == ["h__x", "a__x"]
    assert len(table) == 40 and home.sum() == 20 and set(groups) == {"dfl:WMX"}


def test_grid_order():
    settings = aligned_trial.grid({"owner_cost": 6, "cutoff": .53, "dead_cost": 2})
    assert len(settings) == 15
    assert settings[0] == {"owner_cost": 2, "cutoff": .49, "dead_cost": 2}
    assert settings[-1] == {"owner_cost": 8, "cutoff": .57, "dead_cost": 2}
