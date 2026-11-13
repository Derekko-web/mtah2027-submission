"""D54: three confirmed models on the four 2019–20 SkillCorner confirmation matches (dfl_submission/fresh_trial.py)."""

import hashlib
import json

import numpy as np

from dfl_submission import fresh_trial as fr, skillcorner_2020 as sc
from evaluation import possession_metrics as pm


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(values):
    """A {matches, means} score block with the given N = 3 spellF1 per match."""
    matches = {str(match): {name: {"spell_f1": value, "macro_f1": value / 2} for name in fr.CONVENTIONS}
               for match, value in values.items()}
    return {"matches": matches, "means": fr.means(matches)}


def test_split_costs_and_records():
    assert fr.MATCHES == tuple(sc.HOLDOUT) == (3442, 3518, 3749, 4039)
    assert fr.DEVELOPMENT == tuple(sc.DEVELOPMENT) and not set(fr.MATCHES) & set(fr.DEVELOPMENT)
    assert fr.COSTS == {"D43": 10, "stack": 14, "fallback": 14} and fr.PRIMARY == "aba3"
    for name, file in fr.CONFIRMATIONS.items():
        record = json.loads((fr.RECIPE / file).read_text())
        assert record["passed"] is True and record["owner_cost"] == fr.COSTS[name]
        assert record["levers"]["config"]["direction"] is (name == "stack")


def test_d6_row_is_the_reproduction_target():
    stored = json.loads(fr.D6_RECORD.read_text())["scores"]["0.0"]
    assert round(stored["means"]["aba3"]["spell_f1"], 5) == .62374 and round(stored["means"]["raw"]["spell_f1"], 5) == .54793
    assert sorted(stored["matches"]) == [str(match) for match in fr.DEVELOPMENT]


def test_compare_scores_is_exact():
    base = block({2068: .6, 2269: .5})
    assert fr.compare_scores(base, json.loads(json.dumps(base))) == []
    as_numpy = {"matches": {m: {c: {k: np.float64(v) for k, v in s.items()} for c, s in row.items()}
                            for m, row in base["matches"].items()}, "means": base["means"]}
    assert fr.compare_scores(as_numpy, base) == []
    changed = json.loads(json.dumps(base))
    changed["matches"]["2269"]["aba4"]["spell_f1"] += 1e-12
    assert fr.compare_scores(changed, base) == ["match 2269"]
    changed = json.loads(json.dumps(base))
    changed["means"]["raw"]["macro_f1"] += 1e-12
    assert fr.compare_scores(changed, base) == ["means"]
    missing = json.loads(json.dumps(base))
    del missing["matches"]["2068"]
    assert "match 2068" in fr.compare_scores(missing, base)


def test_means_and_coverage():
    scores = block({1: .2, 2: .4, 3: .9})
    assert np.isclose(scores["means"]["aba3"]["spell_f1"], .5) and np.isclose(scores["means"]["raw"]["macro_f1"], .25)
    labels = [np.array([pm.H, pm.A, pm.X, pm.X], dtype=np.int8), np.array([pm.A, pm.X], dtype=np.int8)]
    assert fr.label_coverage(labels) == .5


def test_reading_rules():
    scores = {"D43": block({1: .6, 2: .6}), "stack": block({1: .65, 2: .56}), "fallback": block({1: .63, 2: .61})}
    result = fr.reading(scores)
    stack, fallback = result["stack"], result["fallback"]
    assert np.isclose(stack["primary_change"], .005) and stack["supported"] and not stack["clearly_supported"]
    assert np.isclose(stack["match_changes"]["1"], .05) and np.isclose(stack["match_changes"]["2"], -.04)
    assert np.isclose(fallback["primary_change"], .02) and fallback["supported"] and fallback["clearly_supported"]
    assert set(fallback["convention_changes"]) == set(fr.CONVENTIONS)
    lower = fr.reading({"D43": block({1: .6}), "stack": block({1: .59}), "fallback": block({1: .6})})
    assert not lower["stack"]["supported"] and lower["fallback"]["supported"] and not lower["fallback"]["clearly_supported"]


def row(half, sign, first, fallback=None, used=None):
    return {"half": half, "group": fr.GROUP, "sign": sign, "first_sign": first, "agrees": sign is not None and sign == first,
            "d": None if sign is None else float(sign), "frames": 0 if sign is None else 9,
            "first_frame": None if sign is None else 0, "max_visible": [8, 8], "max_visible_all_frames": [10, 10],
            "sign_used": sign if used is None else used, "fallback": fallback}


def test_sign_report_counts_fallbacks_and_disagreements():
    table = {"a": row("skillcorner_2020/3442/observed/0", 1, 1), "b": row("skillcorner_2020/3442/observed/1", -1, 1),
             "c": row("skillcorner_2020/3518/observed/0", None, None,
                      {"other_half": ["skillcorner_2020", "3518", "observed", 1], "other_half_sign": 1}, used=-1),
             "d": row("skillcorner_2020/3518/observed/1", 1, 1),
             "e": row("skillcorner_2020/3749/observed/0", 1, 1), "f": row("skillcorner_2020/3749/observed/1", 1, 1)}
    report = fr.sign_report(table)
    assert report["fallback_count"] == 1 and report["fallbacks"] == ["skillcorner_2020/3518/observed/0"]
    assert report["first_frame_disagrees"] == ["skillcorner_2020/3442/observed/1"] and report["signed_count"] == 5
    assert report["same_sign_matches"] == ["3749"]
    assert report["halves"]["skillcorner_2020/3518/observed/0"]["sign_used"] == -1


def test_sign_table_fallback_through_the_stack_rule():
    from dfl_submission import stack_trial as st
    rng = np.random.default_rng(3)
    teams = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    pitch = np.array([105., 68.])
    seen = rng.normal(size=(50, 22, 2)).astype(np.float32)
    seen[:, teams == 1, 0] += 10            # away to the right of home: sign +1
    hidden = np.full((50, 22, 2), np.nan, dtype=np.float32)
    ball = np.zeros((50, 2), dtype=np.float32)
    items = [(fr.GROUP, fr.half_key(3442, 0), seen, ball, teams, pitch),
             (fr.GROUP, fr.half_key(3442, 1), hidden, ball + 1, teams, pitch)]
    report = fr.sign_report(st.sign_table(items))
    first, second = report["halves"]["skillcorner_2020/3442/observed/0"], report["halves"]["skillcorner_2020/3442/observed/1"]
    assert first["sign"] == 1 and first["agrees"] and first["fallback"] is None
    assert second["sign"] is None and second["sign_used"] == -1 and report["fallback_count"] == 1


def test_forecast_cache_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    results = {name: (tuple(rng.random(20).astype(np.float32) for _ in range(fr.SIGNALS)),
                      rng.integers(0, 3, 20).astype(np.int8)) for name in fr.MODELS}
    path = tmp_path / "3442_0.npz"
    key = {"models": {"stack": 1}, "code": "x"}
    fr.save_forecast(path, key, results, True)
    loaded, check = fr.load_forecast(path, key, with_check=True)
    assert check is True and not (tmp_path / "3442_0.part.npz").exists()
    for name in fr.MODELS:
        assert all(np.array_equal(a, b) for a, b in zip(results[name][0], loaded[name][0], strict=True))
        assert np.array_equal(results[name][1], loaded[name][1])
    try:
        fr.load_forecast(path, {"models": {"stack": 2}, "code": "x"})
    except ValueError:
        pass
    else:
        raise AssertionError("a cache under a different identity was accepted")


def make_networks(root):
    copied, identity = {}, {}
    for offset in (0, 1):
        for model_set in ("current",):
            for folder in ("temporal_models",):
                path = root / str(offset) / model_set / folder
                path.mkdir(parents=True)
                (path / "fit.json").write_text(json.dumps({"offset": offset}))
                (path / "temporal_8100.npz").write_bytes(bytes([offset]) * 10)
                copied[f"{offset}/{model_set}/{folder}"] = {"files": {p.name: sha(p) for p in sorted(path.iterdir())}}
                identity.setdefault(str(offset), {}).setdefault(model_set, {})[folder] = sha(path / "fit.json")
    return {"copied": copied}, {"identity": {"networks": identity}}


def test_network_checks_refuse_a_changed_or_missing_file(tmp_path):
    adoption, confirmation = make_networks(tmp_path)
    options = dict(offsets=(0, 1), sets=("current",), networks=("temporal_models",),
                   folder_of=lambda offset, model_set, folder, root: root / str(offset) / model_set / folder)
    assert fr.network_checks(tmp_path, adoption, confirmation, **options) == []
    (tmp_path / "1" / "current" / "temporal_models" / "temporal_8100.npz").write_bytes(b"changed")
    assert fr.network_checks(tmp_path, adoption, confirmation, **options) == [
        "1/current/temporal_models: files differ from the adoption copy"]
    adoption["copied"]["1/current/temporal_models"]["files"]["temporal_8100.npz"] = sha(
        tmp_path / "1" / "current" / "temporal_models" / "temporal_8100.npz")
    confirmation["identity"]["networks"]["0"]["current"]["temporal_models"] = "0" * 64
    assert fr.network_checks(tmp_path, adoption, confirmation, **options) == [
        "0/current/temporal_models: fit.json differs from the confirmed identity"]


def test_code_check_allows_only_the_locked_driver(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    for name in ("a.py", "driver.py"):
        (package / name).write_text(f"# {name}\n")
    confirmation_path = tmp_path / "confirmation.json"
    confirmation = {"code": {name: sha(package / name) for name in ("a.py", "driver.py")}}
    confirmation_path.write_text(json.dumps(confirmation))
    lock_path = tmp_path / "lock.json"
    lock = ("driver.py", lock_path, "driver_sha256")
    assert fr.code_check(confirmation, confirmation_path, lock, package) == (True, [])
    (package / "driver.py").write_text("# changed\n")
    assert fr.code_check(confirmation, confirmation_path, lock, package) == (False, ["driver.py"])
    lock_path.write_text(json.dumps({"passed": True, "confirmation_sha256": sha(confirmation_path),
                                     "driver_sha256": sha(package / "driver.py")}))
    assert fr.code_check(confirmation, confirmation_path, lock, package) == (True, ["driver.py"])
    (package / "a.py").write_text("# changed too\n")
    assert fr.code_check(confirmation, confirmation_path, lock, package) == (False, ["a.py", "driver.py"])


def refuses(call, kind):
    try:
        call()
    except kind:
        return True
    return False


def test_steps_refuse_without_their_predecessor(tmp_path):
    raw = tmp_path / "raw"
    record = tmp_path / "fresh.json"
    assert refuses(lambda: fr.prepare(raw, scratch=tmp_path, record=record), SystemExit)
    (tmp_path / "reproduction.json").write_text(json.dumps({"passed": False}))
    assert refuses(lambda: fr.prepare(raw, scratch=tmp_path, record=record), SystemExit)
    assert refuses(lambda: fr.score(raw, scratch=tmp_path, output=record), SystemExit)
    record.write_text("{}")
    assert refuses(lambda: fr.prepare(raw, scratch=tmp_path, record=record), FileExistsError)
    assert refuses(lambda: fr.score(raw, scratch=tmp_path, output=record), FileExistsError)
    assert refuses(lambda: fr.reproduce(raw, output=tmp_path / "reproduction.json"), FileExistsError)
    (tmp_path / "prepared.json").write_text("{}")
    (tmp_path / "score_started").write_text("x")
    assert refuses(lambda: fr.score(raw, scratch=tmp_path, output=tmp_path / "other.json"), SystemExit)


def test_write_new_once(tmp_path):
    path = tmp_path / "deep" / "record.json"
    fr.write_new(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert refuses(lambda: fr.write_new(path, {"a": 2}), FileExistsError)


def test_summary_lists_every_match_and_reading():
    scores = {"D43": block({3442: .6, 3518: .5}), "stack": block({3442: .62, 3518: .5}), "fallback": block({3442: .6, 3518: .52})}
    record = {"matches": [3442, 3518], "coverage": {"3442": .61, "3518": .7}, "scores": scores,
              "reading": fr.reading(scores)}
    text = fr.summary(record)
    assert "| 3442 | 0.610 | 0.6000 | 0.6200 | 0.6000 |" in text and "mean raw" in text
    assert "stack: N = 3 change +0.0100" in text and "fallback: N = 3 change +0.0100" in text
