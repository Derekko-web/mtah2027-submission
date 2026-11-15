"""D63's development run (dfl_submission/edge_trial.py): readings, substitution, reported items and the flow."""

import json
import subprocess

import numpy as np

from dfl_submission import diagnosis, edge_cost as ec, edge_trial as et, joint_decoder
from evaluation import possession_metrics as pm


RECIPE = {"cutoff": .53, "dead_cost": 2, "owner_cost": 14, "discount": .8, "method": "joint"}
H, A, D, X = pm.H, pm.A, pm.D, pm.X


def refused(call, error=Exception):
    try:
        call()
    except error:
        return True
    return False


def zero_changes(**overrides):
    changes = {"composite": 0.0, "primary": 0.0, "selection": 0.0, "guard_mean": 0.0, "guard_dfl": 0.0,
               "guard_metrica": 0.0, "groups": {name: 0.0 for name in et.GROUPS},
               "conventions": {name: 0.0 for name in et.CONVENTIONS}, "control": {"metrica": 0.0, "d24": 0.0}}
    for path, value in overrides.items():
        if "." in path:
            parent, name = path.split(".", 1)          # "conventions.aba1.75": the convention's name keeps its dot
            changes[parent][name] = value
        else:
            changes[path] = value
    return changes


def test_rule_constants():
    assert et.candidates()[0] == (1.0, .75) and len(et.candidates()) == 15 and (1.0, 1.0) not in et.candidates()
    assert len(set(et.candidates())) == 15 and all(p in ec.FACTORS and q in ec.FACTORS for p, q in et.candidates())
    assert et.key_of(.5, 1.0) == "0.5/1" and et.key_of(1, .25) == "1/0.25"
    assert (et.GROUP_LOSS, et.GUARD_LOSS, et.CONVENTION_LOSS, et.PFF_LOSS, et.CONTROL_LOSS) == (.01, .005, .01, .005, .01)
    assert (et.SELECTION_GAIN, et.METRICA_D24_GAIN) == (.008, .008) and et.BOUNDED == ("aba1.75", "aba4", "aba4.5")
    assert len(et.GROUPS) == 8 and len(et.REPORT_GROUPS) == 12 and et.COST == 14


def test_conditions_hold_at_zero_and_at_each_line():
    assert et.conditions(zero_changes())["holds"]
    at_lines = zero_changes(**{"groups.dfl_masked": -.01, "guard_mean": -.005, "conventions.aba4": -.01,
                               "groups.pff": -.005, "control.metrica": -.01, "control.d24": -.01})
    reading = et.conditions(at_lines)
    assert reading["holds"] and reading["largest_group_fall_name"] == "dfl_masked"


def test_each_condition_fails_on_its_own():
    cases = {"a_composite": {"composite": -1e-9}, "b_groups": {"groups.metrica": -.0101},
             "c_guard_mean": {"guard_mean": -.0051}, "d_conventions": {"conventions.aba1.75": -.0101},
             "e_pff": {"groups.pff": -.0051}, "f_control": {"control.d24": -.0101}}
    for name, overrides in cases.items():
        reading = et.conditions(zero_changes(**overrides))
        assert not reading["holds"] and reading["failed"] == [name], (name, reading["failed"])
    # aba3 is not one of (d)'s conventions
    assert et.conditions(zero_changes(**{"conventions.aba3": -.5}))["holds"]
    # (f) needs both control means
    assert not et.conditions(zero_changes(**{"control.metrica": -.02}))["holds"]


def item(rho_pre, rho_post, selection, holds=True):
    return {"rho_pre": rho_pre, "rho_post": rho_post, "summary": {"selection": selection},
            "reading": {"holds": holds}, "changes": zero_changes(selection=selection - .7)}


def test_selection_and_tie_rule():
    items = {"0.25/0.25": item(.25, .25, .75), "0.5/1": item(.5, 1.0, .74), "1/0.5": item(1.0, .5, .80, holds=False)}
    assert et.selected(items) == "0.25/0.25"                     # the best eligible, not the best overall
    items = {"0.5/0.25": item(.5, .25, .75), "0.25/1": item(.25, 1.0, .75), "0.5/0.75": item(.5, .75, .75)}
    assert et.selected(items) == "0.5/0.75"                      # tie: rho_pre nearer 1, then rho_post nearer 1
    items = {"0.25/1": item(.25, 1.0, .75), "0.75/0.25": item(.75, .25, .75)}
    assert et.selected(items) == "0.75/0.25"
    assert et.selected({"0.5/0.5": item(.5, .5, .9, holds=False)}) is None


def test_gate_lines():
    passing = et.gate(zero_changes(selection=.008, **{"groups.metrica": .006, "groups.d24": .010}))
    assert passing["passed"] and abs(passing["numbers"]["metrica_d24_mean"] - .008) < 1e-15
    assert not et.gate(zero_changes(selection=.0079, **{"groups.metrica": .02, "groups.d24": .02}))["passed"]
    assert not et.gate(zero_changes(selection=.02, **{"groups.metrica": .02, "groups.d24": -.0045}))["passed"]


def test_reading_outcomes():
    items = {"0.5/1": dict(item(.5, 1.0, .75), changes=zero_changes(selection=.01, **{"groups.metrica": .01,
                                                                                     "groups.d24": .01}))}
    decision = et.reading(items)
    assert decision["selected"] == "0.5/1" and decision["passed"] and decision["eligible"] == ["0.5/1"]
    assert decision["selected_factors"] == {"rho_pre": .5, "rho_post": 1.0}
    items["0.5/1"]["changes"] = zero_changes(selection=.01)
    assert not et.reading(items)["passed"]
    decision = et.reading({"0.5/1": item(.5, 1.0, .75, holds=False)})
    assert decision["selected"] is None and decision["gate"] is None and not decision["passed"]


def short_spell_signals():
    home = np.r_[np.full(300, .9), np.full(70, .33), np.full(50, .5)].astype(np.float32)
    dead = np.r_[np.full(370, .001), np.full(50, .95)].astype(np.float32)
    zeros = np.zeros(420, dtype=np.float32)
    return home, dead, zeros, zeros


def test_the_substitution_is_used_and_restored():
    signals = short_spell_signals()
    original_decode, original_joint = diagnosis.decode, joint_decoder.decode
    standing = diagnosis.decode(signals, RECIPE)
    with et.decoding_with(.5, 1.0, RECIPE) as calls:
        recovered = diagnosis.decode(signals, RECIPE)
        assert refused(lambda: joint_decoder.decode(*signals, cutoff=.53, dead_cost=2, owner_cost=14, discount=.8),
                       RuntimeError)
        assert refused(lambda: diagnosis.decode(signals, dict(RECIPE, owner_cost=8)), ValueError)
        assert refused(lambda: diagnosis.decode(signals, RECIPE, step=1), ValueError)
    assert calls["count"] == 1 and (recovered == A).sum() > 50 and not (standing == A).any()
    assert diagnosis.decode is original_decode and joint_decoder.decode is original_joint
    with et.decoding_with(1.0, 1.0, RECIPE):
        assert np.array_equal(diagnosis.decode(signals, RECIPE), standing)
    try:
        with et.decoding_with(.25, .25, RECIPE):
            raise KeyError("inside")
    except KeyError:
        pass
    assert diagnosis.decode is original_decode and joint_decoder.decode is original_joint


def fake_row(value, control=None, raw=None):
    """A row with the structure `summary_of` reads."""
    control = value if control is None else control
    raw = value if raw is None else raw
    games = {str(game): {"aba3": {"spell_f1": value}, "control": {"spell_f1": control}, "raw": {"spell_f1": raw}}
             for game in (1, 2, 3)}
    matches = {"dfl/WPY/clean": {"spell_f1": raw}, "dfl/WPY/masked": {"spell_f1": raw},
               "skillcorner/1/observed": {"spell_f1": raw}, "skillcorner/2/observed": {"spell_f1": raw - .1}}
    conventions = {name: {"standard": {"spell_f1": value}} for name in et.CONVENTIONS}
    conventions["raw"] = {"matches": matches, "dead_carried": raw, "team_swap": raw, "standard": {"spell_f1": raw}}
    return {"composite": value, "primary": value, "selection": value, "groups": {name: value for name in et.GROUPS},
            "guards": {"mean": value, "a_dfl": value, "b_metrica": value},
            "metrica_matches": games, "d24_views": {view: games for view in et.D24_VIEWS},
            "trial": {"external": {"conventions": conventions}, "pff_means": {"raw": {"spell_f1": raw}}}}


def test_summary_reads_control_and_raw_targets():
    row = fake_row(.7, control=.6, raw=.5)
    row["d24_views"] = {"handled": {"1": {"control": {"spell_f1": .3}, "raw": {"spell_f1": .5}}},
                        "masked": {"1": {"control": {"spell_f1": .6}, "raw": {"spell_f1": .5}},
                                   "2": {"control": {"spell_f1": .8}, "raw": {"spell_f1": .5}}},
                        "team_swap": {"1": {"control": {"spell_f1": .9}, "raw": {"spell_f1": .5}}}}
    summary = et.summary_of(row, {name: .7 for name in et.CONVENTIONS})
    assert abs(summary["control"]["metrica"] - .6) < 1e-15
    assert abs(summary["control"]["d24"] - (.3 + .7 + .9) / 3) < 1e-12        # mean over views of game means
    assert abs(summary["raw"]["skillcorner"] - .45) < 1e-12 and summary["raw"]["pff"] == .5
    changes = et.change(summary, et.summary_of(fake_row(.7, control=.6, raw=.5) | {"d24_views": row["d24_views"]},
                                               {name: .7 for name in et.CONVENTIONS}))
    assert changes["control"]["d24"] == 0 and changes["raw"]["standard_combined"] == 0


def test_match_items_count_added_possessions_and_their_match():
    truth = np.array([H] * 100 + [A] * 30 + [D] * 20 + [X] + [A] * 80, dtype=np.int8)
    standing = np.array([H] * 130 + [D] * 20 + [X] + [A] * 80, dtype=np.int8)
    candidate = np.array([H] * 102 + [A] * 28 + [D] * 20 + [X] + [A] * 80, dtype=np.int8)
    items = et.match_items(truth, standing, candidate)
    assert items["predicted_standing"] == 2 and items["predicted_candidate"] == 3
    assert items["added"] == 1 and items["added_matching_truth"] == 1 and items["removed"] == 0
    assert items["added_lengths_frames"] == [28]
    lengths = sorted(items["true_lengths"].tolist())
    assert lengths == [30, 80, 100]
    wrong = np.array([H] * 102 + [D] * 2 + [A] * 6 + [H] * 20 + [D] * 20 + [X] + [A] * 80, dtype=np.int8)
    items = et.match_items(truth, standing, wrong)
    assert items["added"] >= 1 and items["added_matching_truth"] == 0
    same = et.match_items(truth, standing, standing)
    assert same["added"] == 0 and same["removed"] == 0


def synthetic_halves(count):
    rng = np.random.default_rng(3)
    halves = []
    for i in range(count):
        length = 1200 + 35 * i
        home = (1 / (1 + np.exp(-np.cumsum(rng.normal(0, .35, length))))).astype(np.float32)
        dead = (1 / (1 + np.exp(-np.cumsum(rng.normal(0, .25, length))))).astype(np.float32)
        zeros = np.zeros(length, dtype=np.float32)
        signals = (home, dead, zeros, zeros)
        labels = diagnosis.decode(signals, RECIPE)
        halves.append(diagnosis.HalfSignals("dfl", "WPY", "clean", i % 2, labels, signals, labels))
    return halves


def test_pass_checks_on_synthetic_halves():
    halves = synthetic_halves(4) + [diagnosis.HalfSignals("dfl", "X", "clean", 0, None, short_spell_signals(), None)]
    result = et.pass_checks(halves, RECIPE)
    assert result["passed"] and result["pass_one_equals_joint_decoder"] == 5


def test_decoded_halves_and_window_counts():
    half = diagnosis.HalfSignals("dfl", "X", "clean", 0, None, short_spell_signals(), None)
    decoded = et.decoded_halves([half], RECIPE, .5, 1.0)[et.half_id(half)]
    first, second, counts = decoded
    assert np.array_equal(first, diagnosis.decode(half.signals, RECIPE)) and (second == A).sum() > 50
    assert counts["steps"] == 84 and counts["changed_steps"] > 10
    assert counts["window_steps"] == 15 and counts["live_window_steps"] == 15


def test_modules_unchanged_reads_git(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (tmp_path / "dfl_submission").mkdir()
    (tmp_path / "dfl_submission" / "module.py").write_text("A = 1\n")
    (tmp_path / "DECISIONS.md").write_text("rules\n")
    git("add", ".")
    git("commit", "-q", "-m", "start")
    (tmp_path / "DECISIONS.md").write_text("rules and D63\n")
    (tmp_path / "dfl_submission" / "new.py").write_text("B = 2\n")
    result = et.modules_unchanged(tmp_path)
    assert result["passed"] and result["tracked_changes"] == ["DECISIONS.md"]
    assert result["untracked"] == ["dfl_submission/new.py"]
    (tmp_path / "dfl_submission" / "module.py").write_text("A = 2\n")
    result = et.modules_unchanged(tmp_path)
    assert not result["passed"] and result["guarded_changes"] == ["dfl_submission/module.py"]


def test_write_new_writes_once(tmp_path):
    path = tmp_path / "records" / "one.json"
    et.write_new(path, {"a": 1})
    assert json.loads(path.read_text()) == {"a": 1}
    assert refused(lambda: et.write_new(path, {"a": 2}), FileExistsError)


def test_develop_end_to_end_on_synthetic_halves(tmp_path):
    """The whole `develop` flow with its loaders and scorer replaced: checks, the fifteen candidates, both records."""
    from dfl_submission import diagnosis_stack as ds, stack_trial as st
    halves = synthetic_halves(4) + [diagnosis.HalfSignals("dfl", "S", "clean", 0, None, short_spell_signals(), None),
                                    diagnosis.HalfSignals("dfl", "S", "clean", 1, None, short_spell_signals(), None)]
    parts = {"development": halves[:4], "metrica": halves[4:], "d24": [], "guards": []}

    def fake_recomputed(development, metrica, d24, guards, recipe, truths, visibility):
        # The away share of the short-spell halves rises when the spell before the stop is recovered.
        away = float(np.mean([np.mean(diagnosis.decode(h.signals, recipe) == A) for h in metrica]))
        rest = float(np.mean([np.mean(diagnosis.decode(h.signals, recipe) == h.labels) for h in development]))
        return fake_row(.7 + away / 10 - (1 - rest) / 100)

    def fake_reported(scored_parts, recipe, rho_pre, rho_post):
        return {group: {"added": 0 if (rho_pre, rho_post) == (1.0, 1.0) else 1, "removed": 0,
                        "length_recall": {}} for group in et.REPORT_GROUPS}

    cache_root = tmp_path / "cache"
    for part in et.PARTS:
        (cache_root / "stack" / part).mkdir(parents=True)
        (cache_root / "stack" / part / "manifest.json").write_text("{}")
    (tmp_path / "d58.json").write_text("{}")
    saved = {(et, "inputs"): et.inputs, (et, "reported"): et.reported, (et, "d58_check"): et.d58_check,
             (et, "modules_unchanged"): et.modules_unchanged, (et, "D58_RECORD"): et.D58_RECORD,
             (ds, "recomputed_row"): ds.recomputed_row, (ds, "compare_row"): ds.compare_row,
             (ds, "stored_row"): ds.stored_row, (st, "wait_for_scoring"): st.wait_for_scoring}
    try:
        et.inputs = lambda *args, **kwargs: (None, {}, {"stack": "synthetic"}, dict(RECIPE), parts, {}, {})
        et.reported = fake_reported
        et.d58_check = lambda items: {"passed": True}
        et.modules_unchanged = lambda: {"passed": True, "tracked_changes": [], "guarded_changes": []}
        et.D58_RECORD = tmp_path / "d58.json"
        ds.recomputed_row = fake_recomputed
        ds.compare_row = lambda row, stored: {"whole_row": abs(row["composite"] - .7) < 1e-12}
        ds.stored_row = lambda record, cost: {}
        st.wait_for_scoring = lambda *args, **kwargs: None
        output = tmp_path / "records" / "d63_development.json"
        decision = et.develop(tmp_path, tmp_path, tmp_path, cache_root=cache_root, output=output)
        assert refused(lambda: et.develop(tmp_path, tmp_path, tmp_path, cache_root=cache_root, output=output),
                       FileExistsError)
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)
    record = json.loads(output.read_text())
    stored = json.loads((output.parent / "d63_decision.json").read_text())
    assert record["passed_checks"] and record["checks"]["passes"]["passed"]
    assert record["checks"]["decode_calls_per_row"] == 6
    assert record["grid"]["candidates"] == [et.key_of(p, q) for p, q in et.candidates()]
    assert set(record["candidates"]) == set(record["grid"]["candidates"])
    assert record["sides"]["rho_pre_alone"] == ["0.75/1", "0.5/1", "0.25/1"]
    # only factors that reach the short spell before the stop (rho_pre at most 0.5) raise the composite
    raised = {key for key, item in record["candidates"].items() if item["changes"]["composite"] > 0}
    assert raised == {key for key, item in record["candidates"].items() if item["rho_pre"] <= .5}
    assert stored["selected"] == decision["selected"] == "0.5/1"          # tie among them: nearer 1, rho_pre first
    assert stored["development_sha256"] and stored["gate"]["numbers"]["selection"] > 0
    assert raised <= set(stored["eligible"]) and stored["passed"] == decision["passed"]
