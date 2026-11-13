"""D37 plumbing: variants, folders and source checks (dfl_submission/third_trial.py)."""

import json

from dfl_submission import hyper_trial as ht, second_trial as st, third_trial as tt


def test_variants_and_grid():
    assert list(tt.VARIANTS) == ["plain", "spell", "touch", "touch_spell"]
    assert tt.VARIANTS["spell"] == {"alpha": .5, "touch": False} and tt.VARIANTS["touch_spell"]["touch"]
    assert tt.OWNER_COSTS == (8, 10)


def test_plain_reuses_d36s_networks():
    assert tt.network_folders("plain") == ht.network_folders(st.SETTING, st.MODELS)
    assert tt.network_folders("touch") == {key: tt.MODELS / "touch" / key for key in ("current", "augmented")}


def test_touch_source_must_be_d16s_extension(tmp_path):
    published = tt.expected_source("current", False)
    (tmp_path / "manifest.json").write_text(json.dumps({"touch_features": "DECISIONS.md D16", "base_manifest": published}))
    tt.check_source(tmp_path, "current", True)
    (tmp_path / "manifest.json").write_text(json.dumps({"touch_features": "DECISIONS.md D16", "base_manifest": "x"}))
    try:
        tt.check_source(tmp_path, "current", True)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")
