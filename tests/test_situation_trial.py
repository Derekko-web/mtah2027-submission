"""D55's trial helpers on synthetic fixtures, plus the identity checks of the real adopted stack."""

import json
from pathlib import Path

import numpy as np

from dfl_submission import diagnosis, situation_trial as st55, situation_views as sv
from dfl_submission.pff import digest
from dfl_transfer_features import corrupt
from evaluation import possession_metrics as pm


def refused(call, kind=SystemExit):
    try:
        call()
    except kind:
        return True
    return False


def test_agreement_counts_frames_where_target_and_prediction_are_live():
    target = np.array([pm.H, pm.H, pm.A, pm.A, pm.D, pm.H, pm.A, pm.H], dtype=np.int8)
    prediction = np.array([pm.H, pm.A, pm.A, pm.D, pm.H, pm.H, pm.H, pm.H], dtype=np.int8)
    visible = np.array([True, False, False, False, False, True, False, True])
    result = st55.agreement(target, prediction, visible)
    # hidden: frames 1 (H vs A), 2 (A vs A), 6 (A vs H); frame 3 is predicted dead, frame 4's target is dead
    assert result["hidden"] == {"frames": 3, "agree": 1, "agreement": 1 / 3, "target_live_frames": 4}
    assert result["seen"] == {"frames": 3, "agree": 3, "agreement": 1.0, "target_live_frames": 3}


def test_half_targets_follow_the_joined_match_target():
    first = np.full(600, pm.H, dtype=np.int8)
    first[200:240] = pm.A                  # a brief A inside H: merged under N = 3
    first[400:] = pm.A
    second = np.full(500, pm.A, dtype=np.int8)
    second[100:150] = pm.H                 # 2 s of H inside A: merged
    second[300:] = pm.D
    one, two = st55.half_targets(first, second)
    assert len(one) == 600 and len(two) == 500
    assert (one[:400] == pm.H).all() and (one[400:] == pm.A).all()
    assert (two[:300] == pm.A).all() and (two[300:] == pm.D).all()
    _, joined = st55.truth_of(first, second)
    assert np.array_equal(joined, np.r_[one, pm.X, two])


class FakeHalf:
    def __init__(self, index, labels, prediction):
        self.index, self.labels, self.signals = index, labels, (prediction,)


def test_view_scores_score_the_joined_match_and_hidden_frames():
    original = diagnosis.decode
    diagnosis.decode = lambda signals, recipe: signals[0]
    try:
        labels = [np.repeat(np.array([pm.H, pm.A, pm.D, pm.H], dtype=np.int8), 200) for _ in range(2)]
        predictions = [labels[0].copy(), labels[1].copy()]
        predictions[1][250:330] = pm.H
        balls = [np.zeros((800, 2), dtype=np.float32) for _ in range(2)]
        balls[1][240:300] = np.nan
        halves = [FakeHalf(0, labels[0], predictions[0]), FakeHalf(1, labels[1], predictions[1])]
        result = st55.view_scores(halves, balls, "stack")
    finally:
        diagnosis.decode = original
    truth, target = st55.truth_of(*labels)
    prediction = np.r_[predictions[0], pm.X, predictions[1]].astype(np.int8)
    assert result["owner_cost"] == 14
    assert result["spell_f1"] == float(diagnosis.score(target, prediction)["spell_f1"]) and result["spell_f1"] < 1
    assert result["agreement"]["hidden"] == {"frames": 60, "agree": 10, "agreement": 10 / 60, "target_live_frames": 60}
    assert refused(lambda: st55.view_scores(halves[::-1], balls, "D43"), ValueError)


def test_reading_applies_d55_s_thresholds():
    def scores(view_value, random_value, duplicates):
        result = {view: {"spell_f1": view_value} for view in sv.SITUATIONS}
        result |= {f"{view}_random": {"spell_f1": random_value} for view in sv.SITUATIONS}
        return result | {"duplicates": {"spell_f1": duplicates}}

    check = {"standing": {"scores": {"D43": {"masked": {"spell_f1": .70}}, "stack": {"masked": {"spell_f1": .72}}}}}
    weak = st55.reading({"D43": scores(.60, .65, .69), "stack": scores(.70, .72, .70)}, check)
    assert np.isclose(weak["stack_gap_air_or_crowd"], -.02) and weak["real_weakness"]
    assert weak["duplicates_call_for_own_rule"] and "D56 trains" in weak["conclusion"]
    calm = st55.reading({"D43": scores(.60, .65, .69), "stack": scores(.7, .719, .7005)}, check)
    assert not calm["real_weakness"] and not calm["duplicates_call_for_own_rule"]
    assert calm["conclusion"] == "no training follows from D55"
    assert np.isclose(calm["gaps"]["D43"]["crowd"], -.05) and np.isclose(calm["duplicates_minus_masked"]["D43"], -.01)


def synthetic_match(n=2000, players=22, seed=5):
    rng = np.random.default_rng(seed)
    existing, height = {}, {}
    for i, section in enumerate(("firstHalf", "secondHalf")):
        positions = rng.uniform(-30, 30, (n, players + 4, 2)).astype(np.float32)
        positions[:, players:] = np.nan                      # substitutes on the teamsheet, never on the pitch
        ball = rng.uniform(-30, 30, (n, 2)).astype(np.float32)
        positions[::7, :4] = ball[::7, None, :] + np.float32(.5)   # crowds every seventh frame
        labels = np.array(["H"] * (n // 4) + ["A"] * (n // 4) + ["D"] * (n // 4) + ["A"] * (n - 3 * (n // 4)))
        labels[300 + i:330 + i] = "A"
        existing[section] = {"positions": positions, "ball": ball, "labels": labels,
                             "teams": np.r_[np.zeros(13, np.int8), np.ones(13, np.int8)], "pitch": np.array([105.0, 68.0])}
        z = rng.uniform(0, 1.2, n)
        z[1000:1060] = 3.0
        z[200:210] = 2.0
        height[section] = {"z": z}
    return existing, height


def test_build_views_makes_the_nine_views_as_d55_defines_them():
    existing, height = synthetic_match()
    views, notes = st55.build_views(existing, height)
    assert sorted(views) == sorted([*sv.SITUATIONS, *(f"{v}_random" for v in sv.SITUATIONS), "duplicates"])
    labels = [pm.label_codes(existing[s]["labels"]) for s in ("firstHalf", "secondHalf")]
    targets = st55.half_targets(*labels)
    for i, section in enumerate(("firstHalf", "secondHalf")):
        item = existing[section]
        masks = sv.situation_masks(item["positions"], item["ball"], height[section]["z"], targets[i])
        assert masks["air"].sum() == 70 and masks["crowd"][::7].all()
        assert len(range(0, 2000, 7)) <= masks["crowd"].sum() <= len(range(0, 2000, 7)) + 10
        assert notes["changes"][i] == 1                      # H -> A at 500; the brief A inside H is merged
        for view in sv.SITUATIONS:
            _, vp, vb, teams, _ = views[view][i]
            _, rp, rb, _, _ = views[f"{view}_random"][i]
            hidden, placed = np.isnan(vb).any(axis=1), np.isnan(rb).any(axis=1)
            assert np.array_equal(hidden, masks[view])
            assert hidden.sum() == placed.sum() and sorted(sv.runs_of(hidden)[1]) == sorted(sv.runs_of(placed)[1])
            assert np.array_equal(vp, rp, equal_nan=True)
            assert notes["seeds"][f"{view}/{i}"]["view"] == sv.SEEDS[view] + 400 + i
        masked_positions, masked_ball = corrupt(item["positions"], item["ball"], 270000 + 400 + i)
        _, dp, db, dteams, _ = views["duplicates"][i]
        assert np.array_equal(db, masked_ball, equal_nan=True)
        assert np.array_equal(dp[:, :26], masked_positions, equal_nan=True) and dp.shape[1] == 28
        assert dteams.tolist() == item["teams"].tolist() + [0, 1]
        assert 200 <= notes["duplicates"][i]["frames"] < 325
    again, _ = st55.build_views(*synthetic_match())
    for name in views:
        for i in (0, 1):
            assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(views[name][i][1:3], again[name][i][1:3]))


def test_sign_summary_reports_disagreements_fallbacks_and_same_sign_views():
    table = {"k1": {"half": "dfl/WPY/air/0", "sign": 1, "agrees": True, "fallback": None, "sign_used": 1, "d": 4.0},
             "k2": {"half": "dfl/WPY/air/1", "sign": -1, "agrees": False, "fallback": None, "sign_used": -1, "d": -3.0},
             "k3": {"half": "dfl/WPY/crowd/0", "sign": None, "agrees": None, "fallback": "other half", "sign_used": 1,
                    "d": None},
             "k4": {"half": "dfl/WPY/crowd/1", "sign": 1, "agrees": True, "fallback": None, "sign_used": 1, "d": 2.5}}
    summary = st55.sign_summary(table)
    assert summary["first_frame_disagrees"] == ["dfl/WPY/air/1"] and summary["fallbacks"] == ["dfl/WPY/crowd/0"]
    assert summary["same_sign_views"] == ["dfl/WPY/crowd"] and summary["smallest_abs_d_m"] == 2.5


def test_forecast_views_caches_by_identity(tmp_path):
    calls = []

    def produce_one(positions, ball, teams, pitch):
        calls.append(len(ball))
        home = np.full(len(ball), .9, dtype=np.float32)
        return (home, home, home, home), np.full(len(ball), pm.H, dtype=np.int8)

    labels = np.full(10, pm.H, dtype=np.int8)
    views = {"air": {i: (labels, np.zeros((10, 2, 2)), np.zeros((10, 2)), np.zeros(2), np.array([105, 68]))
                     for i in (0, 1)}}
    first = st55.forecast_views(views, "D43", produce_one, tmp_path / "d43", {"a": 1})
    second = st55.forecast_views(views, "D43", produce_one, tmp_path / "d43", {"a": 1})
    assert len(calls) == 2 and [h.view for h in first] == ["air", "air"] and [h.index for h in second] == [0, 1]
    assert refused(lambda: st55.forecast_views(views, "D43", produce_one, tmp_path / "d43", {"a": 2}), ValueError)


def test_written_once_and_the_check_gate(tmp_path):
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    assert refused(lambda: st55.check(None, None, None, output=existing), FileExistsError)
    assert refused(lambda: st55.measure(None, output=existing), FileExistsError)
    assert refused(lambda: st55.checked(tmp_path / "missing.json"))
    good = {"passed": True, "code": st55.code_identity(),
            "confirmation_sha256": digest(st55.RECIPE / "confirmation.json"),
            "development_t_sha256": digest(st55.RECIPE / "development_t.json")}
    path = tmp_path / "check.json"
    path.write_text(json.dumps(good))
    assert st55.checked(path)["passed"] is True
    for change in ({"passed": False}, {"code": dict(good["code"], **{"stack_trial.py": "0" * 64})},
                   {"confirmation_sha256": "0" * 64}):
        path.write_text(json.dumps(dict(good, **change)))
        assert refused(lambda: st55.checked(path))


def test_the_adopted_stack_passes_its_identity_checks_and_a_wrong_lock_refuses(tmp_path):
    config, cost, checks = st55.stack_checks()
    assert cost == 14 and config["direction"] is True and config["mirror"] == "y"
    assert all(value is True for key, value in checks.items() if key != "code_changed_since_confirmation")
    lock = json.loads((st55.RECIPE / "build_identity.json").read_text())
    wrong = tmp_path / "lock.json"
    wrong.write_text(json.dumps(dict(lock, stack_trial_sha256="0" * 64)))
    assert refused(lambda: st55.stack_checks(lock=("stack_trial.py", wrong, "stack_trial_sha256")))
    assert all((Path(st55.PACKAGE) / name).exists() for name in st55.MODULES)
