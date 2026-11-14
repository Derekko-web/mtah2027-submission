"""D62's development helpers and rule readings on synthetic halves and hand-made score tables."""

import copy

import numpy as np

from dfl_submission import diagnosis, phase_evidence as pe, phase_trial as pt
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X
RECIPE = {"cutoff": .53, "dead_cost": 2, "owner_cost": 14, "discount": .8}


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def changes(composite=0., pff=0., guard=0., groups=None, conventions=None):
    """One profile's averaged changes in the structure `phase_trial.summary_of` gives."""
    group_values = {name: 0. for name in pt.GROUPS}
    group_values["pff"] = pff
    group_values |= groups or {}
    convention_values = {name: 0. for name in pt.CONVENTIONS} | (conventions or {})
    return {"composite": composite, "primary": 0., "selection": 0., "groups": group_values, "guard_mean": guard,
            "guard_dfl": guard, "guard_metrica": guard, "conventions": convention_values,
            "pff_agreement_unseen": 0., "pff_agreement_seen": 0.}


def table(overrides=None):
    """{weight: {profile: changes}}: every weight harmless under Poor and Carry, +0.02 under Moderate."""
    result = {}
    for weight in pe.WEIGHTS:
        key = pt.weight_key(weight)
        result[key] = {"Good": changes(.01, .03, .03), "Moderate": changes(.002, .02, .02),
                       "Poor": changes(-.001, -.002, -.001), "Carry": changes(-.002, -.004, -.004)}
    for (key, profile), value in (overrides or {}).items():
        result[key][profile] = value
    return result


SHARES = {"Good": .9, "Moderate": .8, "Poor": .6, "Carry": .79}


def test_the_rule_s_constants():
    assert (pt.COMPOSITE_LOSS, pt.PFF_LOSS, pt.GUARD_LOSS, pt.GROUP_LOSS, pt.CONVENTION_LOSS) == (.003, .005, .005, .01, .01)
    assert (pt.PFF_GAIN, pt.GUARD_GAIN) == (.010, .010)
    assert pt.SAFETY_PROFILES == ("Poor", "Carry") and pt.GATE_PROFILE == "Moderate"
    assert pt.THETA_PROFILES == ("Moderate", "Poor") and pt.BOUNDED == ("aba1.75", "aba4", "aba4.5")
    assert pt.COST == 14 and pt.PARTS == ("development", "metrica", "d24", "guards")
    assert len(pt.GROUPS) == 8


def test_all_safe_the_largest_weight_is_selected_and_the_gate_passes():
    decision = pt.reading(table(), SHARES)
    assert decision["safe_weights"] == ["0.5", "1.0", "2.0"]
    assert decision["selected_weight"] == "2.0"
    assert decision["gate"]["passed"] and decision["passed"]
    assert decision["gate"]["numbers"] == {"pff": .02, "guard_mean": .02, "composite": .002}
    assert abs(decision["theta"] - .7) < 1e-12
    assert decision["good_reported"]["2.0"]["composite"] == .01


def test_a_failing_bound_under_carry_makes_a_weight_unsafe():
    bad = changes(-.0031, 0., 0.)                       # composite falls by more than 0.003
    decision = pt.reading(table({("2.0", "Carry"): bad}), SHARES)
    assert decision["safe_weights"] == ["0.5", "1.0"] and decision["selected_weight"] == "1.0"
    assert decision["safety"]["2.0"]["why_not"] == ["Carry: composite"]
    assert decision["safety"]["2.0"]["Carry"]["numbers"]["composite"] == -.0031
    assert decision["passed"]


def test_each_bound_is_read_at_its_own_line():
    cases = {"pff": changes(pff=-.0051), "guard_mean": changes(guard=-.0051),
             "groups": changes(groups={"metrica": -.0101}), "conventions": changes(conventions={"aba4.5": -.0101})}
    for name, bad in cases.items():
        result = pt.bounded(bad)
        assert not result["holds"] and [k for k, ok in result["conditions"].items() if not ok] == [name]
    at_the_line = changes(-.003 + 1e-12, -.005 + 1e-12, -.005 + 1e-12, {"d24": -.01 + 1e-12}, {"aba1.75": -.01 + 1e-12})
    assert pt.bounded(at_the_line)["holds"]
    assert pt.bounded(changes(conventions={"aba3": -.5}))["holds"]          # N = 3 is not among the bounded three
    worst = pt.bounded(changes(groups={"skillcorner": -.004, "pff": -.002}))["numbers"]
    assert worst["largest_group_fall"] == -.004 and worst["largest_group_fall_name"] == "skillcorner"


def test_selection_ignores_the_gain_and_poor_counts_as_much_as_carry():
    overrides = {("2.0", "Poor"): changes(groups={"dfl_masked": -.02}),
                 ("1.0", "Moderate"): changes(.01, .05, .05)}
    decision = pt.reading(table(overrides), SHARES)
    assert decision["selected_weight"] == "1.0" and decision["safety"]["2.0"]["why_not"] == ["Poor: groups"]
    overrides = {("2.0", "Poor"): changes(groups={"dfl_masked": -.02})}
    decision = pt.reading(table({**overrides, ("1.0", "Carry"): changes(pff=-.006)}), SHARES)
    assert decision["safe_weights"] == ["0.5"] and decision["selected_weight"] == "0.5"


def test_no_safe_weight_means_no_gate_and_nothing_ships():
    bad = changes(-.01, -.01, -.01)
    decision = pt.reading(table({(pt.weight_key(w), "Poor"): bad for w in pe.WEIGHTS}), SHARES)
    assert decision["selected_weight"] is None and decision["gate"] is None and not decision["passed"]
    assert "no weight is safe" in decision["outcome"]
    assert abs(decision["theta"] - .7) < 1e-12


def test_a_failing_gate():
    for moderate, failed in ((changes(.002, .0099, .02), "pff"), (changes(.002, .02, .0099), "guard_mean"),
                             (changes(-1e-9, .02, .02), "composite")):
        decision = pt.reading(table({("2.0", "Moderate"): moderate}), SHARES)
        assert decision["selected_weight"] == "2.0" and not decision["passed"]
        assert [k for k, ok in decision["gate"]["conditions"].items() if not ok] == [failed]
        assert "fails the gate" in decision["outcome"]
    # the gate reads only the selected weight: a passing smaller weight does not rescue it
    decision = pt.reading(table({("2.0", "Moderate"): changes(0., 0., 0.)}), SHARES)
    assert not decision["passed"] and decision["selected_weight"] == "2.0"
    exactly = pt.gate(changes(0., .010, .010))
    assert exactly["passed"]


def test_average_and_change_keep_the_structure():
    first, second = changes(.01, .02, .03), changes(.03, .04, .05)
    mean = pt.average([first, second])
    assert mean["composite"] == .02 and mean["groups"]["pff"] == .03 and mean["guard_mean"] == .04
    delta = pt.change(mean, first)
    assert abs(delta["composite"] - .01) < 1e-15 and abs(delta["conventions"]["aba4"]) == 0


def test_theta_is_the_midpoint_of_moderate_and_poor():
    assert pt.theta({"Moderate": .9, "Poor": .7, "Good": 1., "Carry": 0.}) == .8


def test_groups_and_the_scored_order():
    def half(source, match, view, index):
        return diagnosis.HalfSignals(source, match, view, index, np.zeros(3, dtype=np.int8), (), np.zeros(3))
    parts = {"development": [half("dfl", "WPY", "masked", 0), half("skillcorner", "1", "observed", 0)],
             "metrica": [half("metrica", "1", "observed", 0)], "d24": [half("metrica", "1", "team_swap", 0)],
             "guards": [half("dfl", "WPY", "long_hidden", 0), half("metrica", "2", "long_hidden", 1)]}
    order = pt.half_order(parts)
    assert [item["group"] for item in order] == ["dfl_masked", "skillcorner", "metrica", "d24_team_swap", "guard_dfl",
                                                 "guard_metrica"]
    assert [item["index"] for item in order] == list(range(6))
    assert [part for part, _ in pt.ordered(parts)] == ["development"] * 2 + ["metrica", "d24", "guards", "guards"]
    assert refused(lambda: pt.group_of("other", parts["metrica"][0]), ValueError)


def synthetic_halves(seed=0):
    """Two DFL halves and one guard half with decoder inputs that follow their labels, with hidden-ball runs."""
    rng = np.random.default_rng(seed)
    result = []
    for index, (source, view) in enumerate((("dfl", "clean"), ("dfl", "clean"), ("dfl", "long_hidden"))):
        truth = labels((D, 100), (H, 400), (A, 300), (D, 50), (H, 600), (A, 500), (D, 100))
        home = np.where(truth == H, .8, .2).astype(np.float32) + rng.normal(0, .05, len(truth)).astype(np.float32)
        home = np.clip(home, .01, .99).astype(np.float32)
        dead = np.where(truth == D, .9, .05).astype(np.float32)
        signals = (home, dead, np.zeros(len(truth), dtype=np.float32), np.zeros(len(truth), dtype=np.float32))
        result.append(diagnosis.HalfSignals(source, "WPY", view, index % 2, truth, signals,
                                            diagnosis.decode(signals, RECIPE)))
    seen = [np.ones(len(h.labels), dtype=bool) for h in result]
    for visible in seen:
        visible[600:1400] = False
    return result, seen


def test_check_two_passes_on_synthetic_halves():
    halves, seen = synthetic_halves()
    pairs = [("development", h) for h in halves]
    targets = [diagnosis.target(h.labels, 3.) for h in halves]
    masks = [pe.far_mask(s) for s in seen]
    standing = [diagnosis.decode(h.signals, RECIPE) for h in halves]
    result = pt.check_two(pairs, targets, seen, masks, standing, RECIPE)
    assert result["passed"] and all(result.values())


def test_a_nonzero_weight_changes_the_decode_only_through_masked_tagged_frames():
    halves, seen = synthetic_halves(1)
    half = halves[0]
    target = diagnosis.target(half.labels, 3.)
    mask = pe.far_mask(seen[0])
    wrong = np.where(target == H, pe.AWAY, np.where(target == A, pe.HOME, 0)).astype(np.int8)
    standing = diagnosis.decode(half.signals, RECIPE)
    flipped = diagnosis.decode(pe.shifted_signals(half.signals, wrong, mask, 4.), RECIPE)
    assert not np.array_equal(flipped, standing)
    assert np.array_equal(diagnosis.decode(pe.shifted_signals(half.signals, wrong, np.zeros_like(mask), 4.), RECIPE),
                          standing)


def test_shifted_parts_touch_only_home():
    halves, seen = synthetic_halves(2)
    parts = {"development": halves[:2], "metrica": [], "d24": [], "guards": halves[2:]}
    assert pt.shifted_parts(parts, None, None, 1.) is parts
    states = [np.ones(len(h.labels), dtype=np.int8) for h in halves]
    masks = [pe.far_mask(s) for s in seen]
    shifted = pt.shifted_parts(parts, states, masks, 1.)
    for original, new, mask in zip(halves, shifted["development"] + shifted["guards"], masks):
        assert np.array_equal(new.signals[0][~mask], original.signals[0][~mask])
        assert np.all(new.signals[0][mask] > original.signals[0][mask])
        assert all(a is b for a, b in zip(new.signals[1:], original.signals[1:]))
        assert new.labels is original.labels and (new.source, new.match, new.view, new.index) == \
            (original.source, original.match, original.view, original.index)


def test_counts_and_share_tables():
    target = labels((H, 10), (D, 5), (A, 10))
    seen = np.ones(25, dtype=bool)
    seen[2:20] = False
    mask = np.zeros(25, dtype=bool)
    mask[5:18] = True
    state = labels((1, 8), (0, 4), (1, 6), (-1, 7)).astype(np.int8)
    counts = pt.state_counts(target, state, mask, [(0, 8, 1), (12, 18, 1), (18, 25, -1)])
    # live: 0-9 and 15-24 (20); tagged live: 0-7, 15-24 (18), right: 0-7 and 18-24 (15)
    # acting: 5-7, 12-17 (9); acting live: 5-7, 15-17 (6); right: 5-7 (3)
    assert counts == {"live": 20, "tagged_live": 18, "tagged_live_right": 15, "acting": 9, "acting_live": 6,
                      "acting_live_right": 3, "phases": 3}
    frames = pt.frame_shares(target, seen, mask)
    assert frames == {"frames": 25, "live": 20, "unseen": 18, "far": 13, "far_live": 8}
    order = [{"group": "pff"}, {"group": "pff"}, {"group": "metrica"}]
    shares = pt.share_table([counts, counts, dict(counts, acting_live=0)], order)
    assert shares["pff"]["acting_share_of_live"] == 12 / 40 and shares["metrica"]["acting_share_of_live"] == 0
    assert shares["all"]["live"] == 60 and shares["all"]["acting_live"] == 12
    far = pt.far_table([frames, frames, frames], order)
    assert far["pff"]["far_share"] == 26 / 50 and far["all"]["far_share_of_live"] == 24 / 60
    assert pt.pooled([[3, 5], [1, 5]]) == {"numerator": 4, "denominator": 10, "share": .4}
    assert pt.pooled([[0, 0]])["share"] is None


def test_develop_refuses_to_overwrite_its_records(tmp_path):
    output = tmp_path / "d62_development.json"
    output.write_text("{}")
    assert refused(lambda: pt.develop(tmp_path, tmp_path, tmp_path, output=output), FileExistsError)
    output.unlink()
    (tmp_path / "d62_decision.json").write_text("{}")
    assert refused(lambda: pt.develop(tmp_path, tmp_path, tmp_path, output=output), FileExistsError)


def test_write_new_writes_once(tmp_path):
    path = tmp_path / "record.json"
    pt.write_new(path, {"a": 1})
    assert refused(lambda: pt.write_new(path, {"a": 2}), FileExistsError)
    assert path.read_text() == '{\n  "a": 1\n}\n'


def test_the_reading_does_not_mutate_its_table():
    original = table()
    kept = copy.deepcopy(original)
    pt.reading(original, SHARES)
    assert original == kept


def test_develop_end_to_end_on_synthetic_halves(tmp_path):
    """The whole `develop` flow with its loaders replaced by synthetic halves: checks, grid, readings, both records."""
    import json
    from dfl_submission import diagnosis_stack as ds, stack_trial as st
    halves, seen = synthetic_halves(3)
    parts = {"development": halves[:2], "metrica": [], "d24": [], "guards": halves[2:]}
    visible = {"development": seen[:2], "metrica": [], "d24": [], "guards": seen[2:]}

    def fake_row(scored_parts, recipe, truths, pff_seen):
        accuracy = [float(np.mean(diagnosis.decode(h.signals, recipe) == h.labels))
                    for part in pt.PARTS for h in scored_parts[part]]
        value = float(np.mean(accuracy))
        groups = {name: value for name in pt.GROUPS}
        row = {"composite": value, "primary": value, "selection": value, "groups": groups,
               "guards": {"mean": accuracy[-1], "a_dfl": accuracy[-1], "b_metrica": accuracy[-1]},
               "pff_visibility": {"agreement_unseen": {"agreement": value}, "agreement_seen": {"agreement": value}}}
        return row, {name: value for name in pt.CONVENTIONS}

    cache_root = tmp_path / "cache"
    for part in pt.PARTS:
        (cache_root / "stack" / part).mkdir(parents=True)
        (cache_root / "stack" / part / "manifest.json").write_text("{}")
        (tmp_path / "d59" / part).mkdir(parents=True)
        (tmp_path / "d59" / part / "manifest.json").write_text("{}")
    saved = {(pt, "inputs"): pt.inputs, (pt, "row_of"): pt.row_of, (pt, "rebuilt_visibility"): pt.rebuilt_visibility,
             (pt, "d59_seen_check"): pt.d59_seen_check, (pt, "D59_MEMBERS"): pt.D59_MEMBERS,
             (ds, "compare_row"): ds.compare_row, (ds, "stored_row"): ds.stored_row,
             (st, "wait_for_scoring"): st.wait_for_scoring}
    try:
        pt.inputs = lambda *args, **kwargs: (None, {}, {"stack": "synthetic"}, dict(RECIPE), parts, {}, {})
        pt.row_of = fake_row
        pt.rebuilt_visibility = lambda *args, **kwargs: (visible, {part: 0 for part in pt.PARTS})
        pt.d59_seen_check = lambda *args, **kwargs: {part: {"halves": len(parts[part]), "seen_and_labels_equal": True}
                                                     for part in pt.PARTS}
        pt.D59_MEMBERS = tmp_path / "d59"
        ds.compare_row = lambda row, stored: {"whole_row": True}
        ds.stored_row = lambda record, cost: {}
        st.wait_for_scoring = lambda *args, **kwargs: None
        output = tmp_path / "records" / "d62_development.json"
        decision = pt.develop(tmp_path, tmp_path, tmp_path, cache_root=cache_root, output=output)
    finally:
        for (module, name), value in saved.items():
            setattr(module, name, value)
    record = json.loads(output.read_text())
    stored = json.loads((output.parent / "d62_decision.json").read_text())
    assert record["passed_checks"] and record["check_2"]["passed"]
    assert [item["index"] for item in record["half_order"]] == [0, 1, 2]
    assert set(record["draws"]) == {"0.5", "1.0", "2.0"}
    assert all(len(record["draws"][w][name]) == 3 for w in record["draws"] for name in pe.PROFILE_ORDER)
    assert set(record["acting"]["per_profile"]["Carry"]) == {"dfl_clean", "guard_dfl", "all"}
    assert stored["selected_weight"] == decision["selected_weight"]
    assert stored["development_sha256"] and abs(stored["theta"] - pt.theta(stored["agreement_shares"])) < 1e-15
    # the same draws for every weight: the agreement counts are recorded once per profile and draw
    assert len(record["agreement"]["counts"]["Moderate"]) == 3
    # Carry acts on every masked live frame whose last sighting saw a team; Moderate on fewer
    carry, moderate = (record["acting"]["per_profile"][name]["all"]["acting_live"] for name in ("Carry", "Moderate"))
    assert carry > moderate
    assert refused(lambda: pt.develop(tmp_path, tmp_path, tmp_path, cache_root=cache_root, output=output),
                   FileExistsError)
