"""D15: network width and training-length options (DECISIONS.md D15) on synthetic data."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from dfl_submission import temporal_cpu, temporal_network
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm

ROOT = Path(__file__).resolve().parents[1]


def isolated(name):
    """Run a torch check of this module in a fresh process.

    The torch wheel brings its own OpenMP runtime; mixing it with XGBoost's in one test process can
    hang or crash, so no torch code runs in the shared test process.
    """
    code = f"import sys; sys.path[:0] = ['.', 'experiments']; from tests.test_scale import {name}; {name}()"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                            env=dict(os.environ, OMP_NUM_THREADS="1"))
    assert result.returncode == 0, result.stderr[-2000:]


def check_published_shapes():
    columns = len(json.loads((ROOT / "dfl_submission" / "temporal_models" / "fit.json").read_text())["columns"])
    with np.load(ROOT / "dfl_submission" / "temporal_models" / "temporal_1800.npz") as published:
        expected = {name: published[name].shape for name in published.files}
    for net in (temporal_network.Network(columns), temporal_network.Network(columns, width=temporal_cpu.WIDTH)):
        assert {name: tuple(value.shape) for name, value in net.state_dict().items()} == expected


def test_default_network_has_the_published_checkpoint_shapes():
    isolated("check_published_shapes")


def test_default_schedule_is_unchanged_and_longer_runs_stretch_it():
    assert temporal_network.snapshots(1800) == temporal_cpu.STEPS == (600, 1200, 1800)
    assert temporal_network.snapshots(5400) == (1800, 3600, 5400)
    for step in (1, 2, 900, 1799, 1800):
        old = .0002 + .0013 * .5 * (1 + math.cos(math.pi * (step - 1) / 1800))
        assert temporal_network.learning_rate(step, 1800) == old
    assert math.isclose(temporal_network.learning_rate(2701, 5400), temporal_network.learning_rate(901, 1800))
    assert temporal_network.learning_rate(5400, 5400) > .0002


def check_wider_export():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(5)
    net = temporal_network.Network(3, width=96).eval()
    assert net.state_dict()["project.weight"].shape == (96, 6, 1)
    inputs = torch.randn(2, 151, 6)
    weights = {name: value.numpy() for name, value in net.state_dict().items()}
    with torch.inference_mode():
        reference = net(inputs).numpy()
    np.testing.assert_allclose(temporal_cpu.forward(inputs.numpy(), weights), reference, atol=2e-5, rtol=1e-5)


def test_wider_network_exports_to_the_numpy_forward():
    isolated("check_wider_export")


def prepared(folder, halves=2, length=320):
    folder.mkdir()
    rng = np.random.default_rng(3)
    records = []
    for index in range(halves):
        labels = np.where(np.arange(length) % 90 < 45, pm.H, pm.A).astype(np.int8)
        labels[200:230] = pm.D
        values = rng.normal(size=(length, 2)).astype(np.float32)
        name = f"dfl_M{index}_clean_0.npz"
        np.savez_compressed(folder / name, values=values, targets=temporal_network.targets(labels))
        records.append({"file": name, "source": "dfl", "match": f"M{index}", "sha256": digest(folder / name)})
    (folder / "manifest.json").write_text(json.dumps({"columns": ["h__x", "a__x"], "records": records}))
    return folder


def network_cli(tmp_path, *arguments):
    """The trainer's CLI in a fresh process: torch's OpenMP can deadlock after XGBoost's in one process."""
    subprocess.run([sys.executable, "-m", "dfl_submission.temporal_network", *map(str, arguments)], check=True, cwd=ROOT,
                   capture_output=True, env=dict(os.environ, MTAH_DATA_DIR=str(tmp_path / "absent"), OMP_NUM_THREADS="1"))


def test_width_and_steps_are_recorded_and_exported(tmp_path):
    source = prepared(tmp_path / "source")
    network_cli(tmp_path, "train", "--source", source, "--out", tmp_path / "wide", "--seed", 3, "--width", 8, "--steps", 6)
    fit = json.loads((tmp_path / "wide" / "fit.json").read_text())
    assert fit["width"] == 8 and fit["steps"] == 6
    assert sorted(fit["models"]) == ["temporal_2.pt", "temporal_4.pt", "temporal_6.pt"]
    network_cli(tmp_path, "export", "--source", tmp_path / "wide", "--out", tmp_path / "wide_cpu")
    weights, *_ = temporal_cpu.load(tmp_path / "wide_cpu", "temporal_6.npz")
    assert weights["project.weight"].shape == (8, 4, 1)


def test_default_training_record_has_no_new_keys(tmp_path):
    assert temporal_network.options(temporal_cpu.WIDTH, temporal_cpu.STEPS[-1]) == {}
    assert temporal_network.options(96, 5400) == {"width": 96, "steps": 5400}
    source = prepared(tmp_path / "source")
    network_cli(tmp_path, "train", "--source", source, "--out", tmp_path / "default", "--seed", 3, "--steps", 3)
    fit = json.loads((tmp_path / "default" / "fit.json").read_text())
    assert "width" not in fit and fit["steps"] == 3
