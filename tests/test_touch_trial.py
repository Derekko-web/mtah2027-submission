"""D16 trial plumbing on synthetic tracking: faithful forecasts, the extension check and the gates."""

import json

import numpy as np
import pandas as pd

from dfl_submission import ensemble_trial, long_context, touch_trial
from dfl_submission.external_benchmark import Half
from dfl_submission.pff import digest
from dfl_submission.robust_features import invariant
from evaluation import possession_metrics as pm


def synthetic_half(seed, length=1500, match="M1"):
    rng = np.random.default_rng(seed)
    start = rng.uniform([-45, -30], [45, 30], size=(22, 2))
    positions = (start + np.cumsum(rng.normal(0, .15, size=(length, 22, 2)), axis=0)).astype(np.float32)
    ball = (np.cumsum(rng.normal(0, .5, size=(length, 2)), axis=0) % 40 - 20).astype(np.float32)
    ball[400:430] = np.nan
    positions[700:760, 3] = np.nan
    labels = np.where(np.arange(length) % 300 < 150, pm.H, pm.A).astype(np.int8)
    labels[900:960] = pm.D
    return Half("dfl", match, "clean", 0, labels, positions, ball, np.repeat([0, 1], 11), np.array([105., 68.]))


def test_extended_forecasts_equal_the_published_forecasts_for_d3():
    from dfl_submission import competition_adaptation
    _, packages, _ = competition_adaptation.reference()
    half = synthetic_half(1)
    for name in ("current", "candidate"):
        expected = ensemble_trial.forecasts(*packages[name], half.positions, half.ball, half.teams, half.pitch)
        result = touch_trial.forecasts(*packages[name], half.positions, half.ball, half.teams, half.pitch)
        assert same(expected, result)


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True)
    return a == b


def stored_targets(labels):
    """The dead, home, known and live columns of `temporal_network.targets`, without importing torch.

    Loading torch's OpenMP runtime next to XGBoost's in one test process can crash.
    """
    zero = np.zeros(len(labels), dtype=bool)
    return np.column_stack([labels == pm.D, labels == pm.H, zero, zero, labels != pm.X, labels < pm.D]).astype(np.float32)


def prepared(folder, halves):
    folder.mkdir()
    records, columns = [], None
    for half in halves:
        frames, owner, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        columns = owner.columns.tolist()
        name = f"{half.key}.npz"
        np.savez_compressed(folder / name, values=owner.to_numpy(dtype=np.float32),
                            targets=stored_targets(half.labels[frames]))
        records.append({"file": name, "source": half.source, "match": half.match, "sha256": digest(folder / name)})
    (folder / "manifest.json").write_text(json.dumps({"columns": columns, "records": records}))
    return folder


def test_extension_checks_the_stored_columns_and_appends_the_touch_columns(tmp_path):
    halves = [synthetic_half(2, match="M1"), synthetic_half(3, match="M2")]
    source = prepared(tmp_path / "source", halves)
    touch_trial.extend(source, halves, tmp_path / "extended")
    manifest = json.loads((tmp_path / "extended" / "manifest.json").read_text())
    base = json.loads((source / "manifest.json").read_text())["columns"]
    assert manifest["columns"][:len(base)] == base and len(manifest["columns"]) == len(base) + 22
    assert manifest["base_manifest"] == digest(source / "manifest.json")
    with np.load(tmp_path / "extended" / manifest["records"][0]["file"]) as item, \
            np.load(source / manifest["records"][0]["file"]) as original:
        assert np.array_equal(item["values"][:, :len(base)], original["values"], equal_nan=True)
        assert np.array_equal(item["targets"], original["targets"])
    altered = [halves[0], synthetic_half(4, match="M2")]
    try:
        touch_trial.extend(source, altered, tmp_path / "altered")
    except ValueError as error:
        assert "differ from the stored" in str(error)
    else:
        raise AssertionError("expected the re-extraction check to fail")


def test_stoppage_rows_are_the_invariant_known_rows(tmp_path):
    half = synthetic_half(5)
    source = prepared(tmp_path / "source", [half])
    (rows, target, keys), = touch_trial.stoppage_rows(source)
    frames, owner, dead = long_context.features(half.positions, half.ball, half.teams, half.pitch)
    known = half.labels[frames] != pm.X
    pd.testing.assert_frame_equal(rows, invariant(pd.DataFrame(owner.to_numpy(dtype=np.float32), columns=owner.columns))
                                  .loc[known].reset_index(drop=True))
    assert np.array_equal(target, half.labels[frames][known] == pm.D) and set(keys) == {"dfl:M1"}


def test_d16_gates_are_stricter_on_skillcorner_and_pff():
    change = {"primary": .012, "conventions": {"aba1.75": 0, "aba4": 0, "aba4.5": 0},
              "guards": {"dead_carried": -.005, "team_swap": 0}, "pff": 0, "skillcorner": 0}
    assert touch_trial.eligible(change)
    assert not touch_trial.eligible(dict(change, pff=-.0001))
    assert not touch_trial.eligible(dict(change, skillcorner=-.0001))
    assert not touch_trial.eligible(dict(change, primary=.0099))


def test_skillcorner_part_uses_only_the_standard_skillcorner_matches():
    trial = {"external": {"conventions": {"aba3": {"matches": {
        "skillcorner/1/observed": {"spell_f1": .6}, "skillcorner/2/observed": {"spell_f1": .8},
        "dfl/WPY/clean": {"spell_f1": .1}, "dfl/WPY/team_swap": {"spell_f1": .2}}}}}}
    assert np.isclose(touch_trial.skillcorner_part(trial), .7)


def test_standing_follows_the_d15_confirmation(tmp_path):
    from dfl_submission import scale_trial
    assert touch_trial.standing(tmp_path) == (scale_trial.REFERENCE, None)
    recipe = {"owner_cost": 8, "cutoff": .53}
    (tmp_path / "recipe.json").write_text(json.dumps({"setting": {"width": 96, "steps": 5400, "owner_cost": 8},
                                                      "recipe": recipe}))
    (tmp_path / "confirmation.json").write_text(json.dumps({"passed": False}))
    assert touch_trial.standing(tmp_path) == (scale_trial.REFERENCE, None)
    (tmp_path / "confirmation.json").write_text(json.dumps({"passed": True}))
    assert touch_trial.standing(tmp_path) == ((96, 5400), recipe)
