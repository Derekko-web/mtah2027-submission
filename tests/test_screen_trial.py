"""D47 screening harness: parity classification, group scores, the no-harm rule, records (dfl_submission/screen_trial.py)."""

import json

import numpy as np

from dfl_submission import screen_trial as st


def test_column_comparison_classifies_unchanged_sign_and_neither():
    rng = np.random.default_rng(0)
    stored = rng.normal(size=(50, 4)).astype(np.float32)
    stored[3, 1] = np.nan
    mirrored = stored.copy()
    mirrored[:, 1] *= -1                                     # a pure sign change (NaN stays NaN)
    mirrored[:, 2] = stored[:, 2] + 5e-5                     # within the tolerance
    mirrored[:, 3] = np.abs(stored[:, 3])                    # neither
    same, sign, d_same, d_flip = st.column_comparison(mirrored, stored)
    assert same.tolist() == [True, False, True, False]
    assert sign.tolist() == [False, True, False, False]
    assert d_flip[1] == 0 and d_same[2] < 1e-4
    moved = mirrored.copy()
    moved[5, 0] = np.nan                                     # a changed missing pattern is neither
    assert not st.column_comparison(moved, stored)[0][0]


def part(name, unchanged, sign):
    count = len(unchanged["x"])
    zeros = {m: [0.] * count for m in st.MIRRORED}
    return {"set": name, "columns": [f"c{i}" for i in range(count)], "halves": 3, "sequences_manifest": name,
            "pitches_sha256": name, "unchanged": unchanged, "sign_change": sign, "max_unchanged_difference": zeros,
            "max_sign_difference": zeros, "token_max_difference": {m: 0. for m in st.MIRRORED}}


def test_parity_record_sign_map_and_failure(tmp_path):
    unchanged = {"x": [True, False, True], "y": [True, True, True], "xy": [True, False, True]}
    sign = {"x": [False, True, False], "y": [False, False, False], "xy": [False, True, False]}
    paths = []
    for name in st.SETS:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(part(name, unchanged, sign)))
        paths.append(path)
    assert st.parity_record(paths, output=tmp_path / "parity.json")
    record = json.loads((tmp_path / "parity.json").read_text())
    assert record["sign_map"]["signs"] == {"x": [1, -1, 1], "y": [1, 1, 1], "xy": [1, -1, 1]}
    assert record["sign_change_columns"]["x"] == ["c1"] and record["tokens"]["passed"]
    # A column that is a sign change in one set and unchanged in the other is neither.
    other = part("augmented", {"x": [True, True, True], "y": [True, True, True], "xy": [True, True, True]},
                 {m: [False] * 3 for m in st.MIRRORED})
    paths[1].write_text(json.dumps(other))
    assert not st.parity_record(paths, output=tmp_path / "parity_failed.json")
    failed = json.loads((tmp_path / "parity_failed.json").read_text())
    assert failed["neither"]["x"] == ["c1"] and failed["sign_map"]["signs"]["x"][1] == 0
    try:
        st.parity_record(paths, output=tmp_path / "parity.json")
    except FileExistsError:
        pass
    else:
        raise AssertionError("parity record written twice")


def trial(clean=.78, skill=(.7, .72, .68, .74), pff=.66):
    matches = {"dfl/WPY/clean": clean, "dfl/WPY/masked": .72, "dfl/WPY/dead_carried": .64, "dfl/WPY/team_swap": .69}
    matches.update({f"skillcorner/{i}/observed": value for i, value in enumerate(skill)})
    return {"external": {"conventions": {"aba3": {"matches": {k: {"spell_f1": v} for k, v in matches.items()},
                                                  "standard": {"spell_f1": .73}}}},
            "pff": {"means": {"aba3": {"spell_f1": pff}}, "matches": {str(i): {} for i in range(8)}}}


def test_group_scores_and_composite():
    groups, composite = st.group_scores(trial(), {"aba3": {"spell_f1": .78}}, .70)
    assert list(groups) == list(st.GROUPS)
    assert np.isclose(groups["skillcorner"], .71) and groups["dfl_clean"] == .78 and groups["d24"] == .70
    assert np.isclose(composite, np.mean([.78, .72, .64, .69, .71, .66, .78, .70]))
    broken = trial(skill=(.7, .72, .68))
    try:
        st.group_scores(broken, {"aba3": {"spell_f1": .78}}, .70)
    except ValueError:
        pass
    else:
        raise AssertionError("group_scores accepted three SkillCorner matches")


def entry(cost, composite, **changes):
    groups = {name: .7 + changes.get(name, 0.) for name in st.GROUPS}
    return {"owner_cost": cost, "composite": composite, "groups": groups}


def test_no_harm_rule():
    base = {"composite": .7, "groups": {name: .7 for name in st.GROUPS}}
    # Best cost by composite is taken (ties to the lower cost); a group down by more than 0.01 fails it.
    entries = [entry(8, .701), entry(10, .705, pff=-.011), entry(12, .705, pff=-.02)]
    screened = st.no_harm(entries, base)
    assert screened["owner_cost"] == 10 and not screened["passed"] and screened["worst_group"] == "pff"
    entries = [entry(8, .700, metrica=-.0099), entry(10, .699)]
    screened = st.no_harm(entries, base)
    assert screened["owner_cost"] == 8 and screened["passed"]        # composite equal to D43's passes
    assert not st.no_harm([entry(10, .6999)], base)["passed"]


def test_candidate_record_and_write_once(tmp_path):
    base = {"composite": .7, "groups": {name: .7 for name in st.GROUPS}, "primary": .73}
    d43_trial = {"external": {"conventions": {name: {"standard": {"spell_f1": .73}, "dead_carried": .64, "team_swap": .69}
                                              for name in ("aba3", "aba1.75", "aba4", "aba4.5")}},
                 "pff": {"means": {"aba3": {"spell_f1": .66}}}}
    trial_entry = dict(entry(10, .71, d24=.01), primary=.74, trial=d43_trial)
    record = st.candidate_record("M3", [trial_entry], base, d43_trial, {"candidate": "M3"}, {"development": 32}, {})
    assert record["screen"]["passed"] and record["entries"][0]["primary_change"] == .74 - .73
    assert record["entries"][0]["changes"]["primary"] == 0 and record["reference"]["composite"] == .7
    seed_record = st.candidate_record("S1000", [dict(trial_entry)], base, d43_trial, {}, {}, {})
    assert seed_record["screen"]["passed"] is None and not seed_record["lever"]
    st.write_new(tmp_path / "M3.json", record)
    try:
        st.write_new(tmp_path / "M3.json", record)
    except FileExistsError:
        pass
    else:
        raise AssertionError("record written twice")


def test_tracking_key_separates_halves_and_candidates_are_complete():
    positions = np.zeros((10, 4, 2))
    ball = np.zeros((10, 2))
    teams, pitch = np.array([0, 0, 1, 1]), np.array([105., 68.])
    key = st.tracking_key(positions, ball, teams, pitch)
    assert key == st.tracking_key(positions.copy(), ball.copy(), teams, pitch)
    moved = positions.copy()
    moved[0, 0, 0] = 1
    assert key != st.tracking_key(moved, ball, teams, pitch)
    assert key != st.tracking_key(positions.astype(np.float32), ball, teams, pitch)
    assert set(st.LEVERS) | {"S1000", "S2000"} == set(st.CANDIDATES)
    assert st.CANDIDATES["M3"]["slots"] == ("D43", "S1000", "S2000") and 6 in st.CANDIDATES["M3"]["costs"]
    assert all(st.CANDIDATES[c]["costs"][-3:] == (8, 10, 12) for c in st.CANDIDATES)
    assert {c for c, spec in st.CANDIDATES.items() if spec["mirrored"]} == set(st.MIRROR_FAMILY)
    assert st.TRAINED["S1000"]["seed_offset"] == 1000 and st.TRAINED["FN"] == {"seed_offset": 0, "mirror": True, "noise": True}
