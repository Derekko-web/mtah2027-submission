"""D35 plumbing: the member-output cache round trip and the grid (dfl_submission/mix_trial.py)."""

import json

import numpy as np

from dfl_submission import mix_trial as mt


def structure():
    rng = np.random.default_rng(1)
    arr = lambda: rng.uniform(size=7).astype(np.float32)
    return ((arr(), arr()), {"temporal_1800.npz": (arr(), arr(), arr(), arr()), "late_mean": None}), \
        {"seed_phase": (arr(), arr(), arr(), arr())}


def same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return isinstance(b, tuple) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return np.array_equal(a, b) and a.dtype == b.dtype


def test_flatten_round_trip():
    value = structure()
    assert same(mt.unflatten(mt.flatten(value)), value)


def test_save_and_load_round_trip(tmp_path):
    value = structure()
    labels = np.arange(7, dtype=np.int8)
    entry = mt.save_members(tmp_path, "dfl_X_clean_0", value, labels, {"source": "dfl"})
    loaded, loaded_labels = mt.load_members(tmp_path, json.loads(json.dumps(entry)))
    assert same(loaded, value) and np.array_equal(loaded_labels, labels)


def test_grid():
    points = mt.points()
    assert len(points) == 48 and mt.REFERENCE in points
    assert points[0] == {"owner_mix": .25, "dead_mix": .25, "owner_cost": 6}
