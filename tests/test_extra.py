"""D14 ownership-only targets and folder combination on synthetic inputs."""

import json

import numpy as np

from dfl_submission import aligned_trial, extra_trial
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def folder(path, source, columns, count):
    path.mkdir()
    records = []
    for index in range(count):
        name = f"{source}_{index + 1}_observed_{index}.npz"
        np.savez_compressed(path / name, values=np.zeros((3, len(columns)), np.float32), targets=np.zeros((3, 6), np.float32))
        records.append({"file": name, "source": source, "match": str(index + 1), "sha256": digest(path / name)})
    (path / "manifest.json").write_text(json.dumps({"columns": columns, "records": records}))


def test_ownership_targets_switch_off_the_stoppage_losses():
    labels = np.array([X, H, H, A, A, X, A], dtype=np.int8)
    targets = extra_trial.ownership_targets(labels)
    assert (targets[:, 4] == 0).all()
    assert targets[:, 5].tolist() == [0, 1, 1, 1, 1, 0, 1]
    assert targets[:, 1].tolist() == [0, 1, 1, 0, 0, 0, 0]
    assert targets[:, 3].tolist() == [0, 0, 1, 1, 1, 0, 0]
    assert (targets[:, [0, 2]] == 0).all()


def test_combine_appends_the_extra_halves(tmp_path):
    folder(tmp_path / "original", "dfl", ["h__x", "a__x"], 2)
    folder(tmp_path / "extra", extra_trial.SOURCE, ["h__x", "a__x"], 1)
    extra_trial.combine(tmp_path / "original", tmp_path / "extra", tmp_path / "combined")
    manifest = json.loads((tmp_path / "combined" / "manifest.json").read_text())
    assert [record["source"] for record in manifest["records"]] == ["dfl", "dfl", extra_trial.SOURCE]
    assert len(list(aligned_trial.verified_records(tmp_path / "combined"))) == 3
    rows = aligned_trial.ownership_rows(tmp_path / "combined")
    assert [set(groups) for _, _, groups in rows if len(groups)] == []


def test_combine_refuses_a_different_schema(tmp_path):
    folder(tmp_path / "original", "dfl", ["h__x", "a__x"], 1)
    folder(tmp_path / "extra", extra_trial.SOURCE, ["h__y", "a__y"], 1)
    try:
        extra_trial.combine(tmp_path / "original", tmp_path / "extra", tmp_path / "combined")
    except ValueError:
        pass
    else:
        raise AssertionError("A different feature schema must be refused")


def test_development_gates_require_no_loss_only():
    change = {"primary": -.009, "conventions": {"aba1.75": -.05}, "guards": {"dead_carried": -.01, "team_swap": .02},
              "pff": -.01}
    assert extra_trial.development_passed(change)
    assert not extra_trial.development_passed(dict(change, primary=-.011))
    assert not extra_trial.development_passed(dict(change, guards={"dead_carried": -.02, "team_swap": 0}))
    assert not extra_trial.development_passed(dict(change, pff=-.02))
