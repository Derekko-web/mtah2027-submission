"""D50 trial: training count, sign-check summaries, column decision, composite and carry rule (dfl_submission/direction_trial.py)."""

import json

import numpy as np

from dfl_submission import direction_tokens as dt, direction_trial as trial


PY = "/opt/homebrew/bin/Python"


def test_trainings_count_trainer_processes_once():
    rows = [
        (10, 1, f"{PY} -m dfl_submission.screen_trial networks --candidate S1000 --set current"),
        (11, 10, f"{PY} -m dfl_submission.player_set3 --source a --tokens b --out c --seed 42"),
        (12, 1, f"{PY} -m dfl_submission.screen_trial networks --candidate S1000 --set augmented"),   # exporting
        (13, 1, f"{PY} -m dfl_submission.player_set train --source a --tokens b"),
        (14, 1, "/bin/zsh -c python -m dfl_submission.player_set train"),                            # a shell, not python
        (15, 1, f"{PY} -m dfl_submission.diagnosis_d43 measure-hidden --parsed x"),
        (16, 1, f"{PY} -m dfl_submission.direction_trial networks --set current"),                    # this process
        (17, 1, f"{PY} -m dfl_submission.screen_trial score --candidates N"),
    ]
    assert trial.trainings(rows, own=16) == 3            # 11, 13 and the exporting run 12
    assert trial.trainings(rows, own=None) == 4
    assert trial.trainings([], own=None) == 0
    assert trial.trainings([(20, 1, f"{PY} -m dfl_submission.direction_tokens train --check")]) == 1


def entry(sign, first_sign, d, visible=(11, 11), any_frame=None, frames=100, **extra):
    result = {"sign": sign, "first_sign": first_sign, "d": d, "agrees": sign == first_sign if sign else False,
              "frames": frames, "max_visible": list(visible),
              "max_visible_all_frames": list(any_frame if any_frame else visible)}
    return result | extra


def test_part_summary_passes_and_reports():
    entries = {"a": entry(1, 1, 30.), "b": entry(-1, -1, -12.5), "c": entry(1, 1, 4., visible=(11, 12), any_frame=(12, 12))}
    result = trial.part_summary(entries, False)
    assert result["passed"] and result["halves"] == 3 and result["negative_signs"] == 1
    assert result["smallest_abs_d_m"] == 4. and result["smallest_abs_d_half"] == "c"
    assert result["over_per_team_long_context"] == {"c": [11, 12]} and list(result["over_per_team_any_frame"]) == ["c"]


def test_part_summary_fails_on_disagreement_or_no_frame():
    assert not trial.part_summary({"a": entry(1, -1, 3.)}, False)["passed"]
    none = {"sign": None, "first_sign": None, "d": None, "agrees": False, "frames": 0, "max_visible": [6, 11],
            "max_visible_all_frames": [6, 11]}
    result = trial.part_summary({"a": entry(1, 1, 3.), "b": none}, False)
    assert not result["passed"] and result["no_qualifying_frame"] == ["b"]


def test_training_summary_needs_equal_tokens_and_the_same_token_sign():
    token = {"frames": 100, "d": .5, "sign": 1, "first_sign": 1, "agrees": True}
    good = entry(1, 1, 26., tokens_equal=True, token=token)
    assert trial.part_summary({"a": good}, True)["passed"]
    assert not trial.part_summary({"a": dict(good, tokens_equal=False)}, True)["passed"]
    assert not trial.part_summary({"a": dict(good, token=dict(token, sign=-1))}, True)["passed"]


def test_half_entry_counts_players_on_every_frame():
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(23, 26, 2))
    positions[:, [11, 12, 24, 25]] = np.nan
    positions[7, 11] = 3.                          # a twelfth home player seen on one frame between long-context frames
    teams = np.r_[np.zeros(13, int), np.ones(13, int)]
    result = trial.half_entry(positions, teams)
    assert result["max_visible"] == [11, 11] and result["max_visible_all_frames"] == [12, 11]


def test_x_odd_flags_come_from_the_x_mirror():
    parity = {"sign_map": {"columns": ["h__a", "a__a", "speed"], "signs": {"x": [-1, -1, 1], "y": [1, 1, 1]}}}
    assert trial.x_odd(parity, ["h__a", "a__a", "speed"]) == [True, True, False]
    try:
        trial.x_odd(parity, ["speed", "h__a", "a__a"])
    except ValueError:
        return
    raise AssertionError("x_odd accepted a sign map on other columns")


def test_freeze_columns_with_and_without_a_parity_record(tmp_path):
    source = tmp_path / "prepared"
    source.mkdir()
    columns = ["h__a", "a__a", "speed"]
    (source / "manifest.json").write_text(json.dumps({"columns": columns}))
    missing = tmp_path / "parity.json"
    flags = trial.freeze_columns(source, output=tmp_path / "none.json", parity_path=missing)
    record = json.loads((tmp_path / "none.json").read_text())
    assert flags == [False] * 3 and record["status"] == "no parity record yet" and record["parity_sha256"] is None
    missing.write_text(json.dumps({"passed": True, "neither": {"x": []},
                                   "sign_map": {"columns": columns, "signs": {"x": [-1, -1, 1]}}}))
    assert trial.freeze_columns(source, output=tmp_path / "read.json", parity_path=missing) == [True, True, False]
    assert json.loads((tmp_path / "read.json").read_text())["x_odd_columns"] == ["h__a", "a__a"]
    missing.write_text(json.dumps({"passed": True, "neither": {"x": []},
                                   "sign_map": {"columns": columns, "signs": {"x": [-1, 1, 1]}}}))
    try:
        trial.freeze_columns(source, output=tmp_path / "bad.json", parity_path=missing)
    except ValueError:
        return
    raise AssertionError("freeze_columns accepted x-odd columns not closed under the team swap")


def development_trial(values):
    """A minimal trial with the fields `group_scores` reads."""
    matches = {f"dfl/WPY/{view}": {"spell_f1": values[i]} for i, view in enumerate(("clean", "masked", "dead_carried", "team_swap"))}
    matches |= {f"skillcorner/{i}/observed": {"spell_f1": v} for i, v in enumerate(values[4])}
    return {"external": {"conventions": {"aba3": {"matches": matches}}},
            "pff": {"matches": {str(i): {} for i in range(8)}, "means": {"aba3": {"spell_f1": values[5]}}}}


def test_group_scores_is_d47s_composite():
    trial_record = development_trial([.8, .7, .6, .65, [.7, .72, .74, .76], .66])
    groups, composite = trial.group_scores(trial_record, {"aba3": {"spell_f1": .79}}, .7)
    assert groups == {"dfl_clean": .8, "dfl_masked": .7, "dfl_dead_carried": .6, "dfl_team_swap": .65,
                      "skillcorner": float(np.mean([.7, .72, .74, .76])), "pff": .66, "metrica": .79, "d24": .7}
    assert abs(composite - np.mean(list(groups.values()))) < 1e-15
    assert list(groups) == list(trial.GROUPS)


def scored(cost, composite, groups):
    return {"owner_cost": cost, "composite": composite, "groups": groups}


def test_carry_rule():
    base_groups = {name: .7 for name in trial.GROUPS}
    base = scored(10, .7, base_groups)
    up = dict(base_groups, pff=.72)
    entries = [scored(8, .7, base_groups), scored(10, .7025, up), scored(12, .701, base_groups)]
    result = trial.carry(entries, base)
    assert result["owner_cost"] == 10 and result["passed"] and abs(result["composite_change"] - .0025) < 1e-12
    harmed = dict(up, dfl_clean=.689)                      # one group down by more than 0.01
    result = trial.carry([scored(8, .705, harmed), scored(10, .7, base_groups)], base)
    assert result["owner_cost"] == 8 and not result["passed"] and result["worst_group"] == "dfl_clean"
    # Ties go to the lower cost; a lower composite fails even with no group harmed.
    assert trial.carry([scored(12, .7, base_groups), scored(8, .7, base_groups)], base)["owner_cost"] == 8
    assert not trial.carry([scored(8, .6999, base_groups)], base)["passed"]


def test_brief_keeps_the_report_fields():
    full = {"owner_cost": 8, "groups": {}, "composite": 1., "primary": .5, "trial": {}, "agreement": {}}
    assert set(trial.brief(full)) == {"owner_cost", "groups", "composite", "primary", "agreement"}


def test_x_signs_match_d47s_x_mirror():
    from dfl_submission import player_set3
    np.testing.assert_array_equal(dt.X_SIGNS, player_set3.TOKEN_SIGNS["x"])


def test_tracking_key_follows_content_shape_and_dtype():
    positions = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    ball, teams, pitch = np.zeros((3, 2), np.float32), np.array([0, 0, 1, 1]), np.array([105., 68.])
    key = trial.tracking_key(positions, ball, teams, pitch)
    assert key == trial.tracking_key(positions.copy(), ball, teams, pitch)
    changed = positions.copy()
    changed[1, 2, 0] += 1
    assert key != trial.tracking_key(changed, ball, teams, pitch)
    assert key != trial.tracking_key(positions.astype(np.float64), ball, teams, pitch)


def test_relaxed_sign_counts_frames_with_one_player_and_restores_the_threshold():
    rng = np.random.default_rng(1)
    positions = np.concatenate([rng.normal([-20, 0], 5, size=(40, 11, 2)), rng.normal([20, 0], 5, size=(40, 11, 2))], axis=1)
    positions[:, 3:11] = np.nan                     # three home players visible
    teams = np.r_[np.zeros(11, int), np.ones(11, int)]
    assert dt.attacking(positions, teams)["sign"] is None
    relaxed = trial.relaxed_sign(positions, teams)
    assert relaxed["sign"] == 1 and relaxed["frames"] == 8
    assert dt.MIN_PLAYERS == 7


def sign_parts(missing=(), disagree=()):
    part = {"passed": True, "no_qualifying_frame": [], "first_frame_disagrees": []}
    parts = {"training_current": dict(part), "training_augmented": dict(part), "pff": dict(part, no_qualifying_frame=list(missing),
                                                                                        passed=not missing),
             "skillcorner": dict(part, first_frame_disagrees=list(disagree), passed=not disagree)}
    return {"passed": not missing and not disagree, "parts": parts}


def test_scoring_signs_needs_a_decision_for_every_sign_less_half(tmp_path):
    original = trial.RECIPE
    trial.RECIPE = tmp_path
    try:
        (tmp_path / "signs.json").write_text(json.dumps(sign_parts()))
        assert trial.scoring_signs() == {}
        (tmp_path / "signs.json").write_text(json.dumps(sign_parts(missing=["pff/9/observed/1"])))
        for gap in (None, {"signs_sha256": "0" * 64, "halves": {"pff/9/observed/1": {"sign": -1, "tracking_key": "k"}}}):
            if gap is not None:
                (tmp_path / "sign_gap.json").write_text(json.dumps(gap))
            try:
                trial.scoring_signs()
            except ValueError:
                continue
            raise AssertionError("scoring_signs accepted a sign-less half without a matching decision")
        from dfl_submission.pff import digest
        (tmp_path / "sign_gap.json").write_text(json.dumps({"signs_sha256": digest(tmp_path / "signs.json"),
                                                             "halves": {"pff/9/observed/1": {"sign": -1, "tracking_key": "k"}}}))
        assert trial.scoring_signs() == {"k": -1}
        (tmp_path / "signs.json").write_text(json.dumps(sign_parts(disagree=["skillcorner/1/observed/0"])))
        try:
            trial.training_signs_passed()
        except ValueError:
            return
        raise AssertionError("a first-frame disagreement did not stop D50")
    finally:
        trial.RECIPE = original


def test_fallback_sign_is_the_other_half_negated_or_plus_one():
    assert trial.fallback_sign(1) == -1 and trial.fallback_sign(-1) == 1
    assert trial.fallback_sign(None) == 1


def test_sign_gap_applies_both_fallbacks(tmp_path):
    """End to end on two PFF games: game 1 has one signed half, game 2 has none."""
    from dfl_submission import pff
    from dfl_submission.pff import digest
    rng = np.random.default_rng(3)

    def half(home_visible):
        positions = np.concatenate([rng.normal([-20, 0], 5, size=(30, 11, 2)), rng.normal([20, 0], 5, size=(30, 11, 2))],
                                   axis=1).astype(np.float32)
        positions[:, home_visible:11] = np.nan
        return {"positions": positions, "ball": positions[:, 12].copy(), "teams": np.r_[np.zeros(11, int), np.ones(11, int)],
                "pitch": np.array([105., 68.])}

    games = {1: [half(11), half(4)], 2: [half(5), half(3)]}
    unsigned = {"sign": None, "d": None}
    development = {"pff/1/observed/0": {"sign": 1, "d": 40.}, "pff/1/observed/1": unsigned,
                   "pff/2/observed/0": unsigned, "pff/2/observed/1": unsigned}
    missing = ["pff/1/observed/1", "pff/2/observed/0", "pff/2/observed/1"]
    originals = (trial.RECIPE, trial.SCRATCH, pff.load)
    trial.RECIPE, trial.SCRATCH = tmp_path / "recipe", tmp_path / "scratch"
    trial.RECIPE.mkdir()
    trial.SCRATCH.mkdir()
    pff.load = lambda parsed, game: games[game]
    try:
        (trial.RECIPE / "signs.json").write_text(json.dumps(
            {"passed": False, "parts": {"pff": {"no_qualifying_frame": missing, "first_frame_disagrees": [], "passed": False}}}))
        (trial.SCRATCH / "signs_development.json").write_text(json.dumps({"halves": development}))
        result = trial.sign_gap("other_half", tmp_path)
        assert {key: value["sign"] for key, value in result.items()} == {
            "pff/1/observed/1": -1, "pff/2/observed/0": 1, "pff/2/observed/1": 1}
        tracked = games[1][1]
        assert result["pff/1/observed/1"]["tracking_key"] == trial.tracking_key(
            tracked["positions"], tracked["ball"], tracked["teams"], tracked["pitch"])
        record = json.loads((trial.RECIPE / "sign_gap.json").read_text())
        assert record["rule"] == "other_half" and record["signs_sha256"] == digest(trial.RECIPE / "signs.json")
    finally:
        trial.RECIPE, trial.SCRATCH, pff.load = originals
