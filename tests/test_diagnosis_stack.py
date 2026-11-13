"""D58 stoppage measurements on synthetic label sequences and ball visibility."""

import numpy as np

from dfl_submission import diagnosis, diagnosis_d43 as d46, diagnosis_stack as d58
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def flags(*parts):
    return np.concatenate([np.full(length, value, dtype=bool) for value, length in parts])


def half(source, match, view, index, truth, decoded=None):
    decoded = truth if decoded is None else decoded
    home = np.where(decoded == H, .9, .1)
    signals = (home, np.zeros(len(truth)), np.zeros(len(truth)), np.zeros(len(truth)))
    return diagnosis.HalfSignals(source, match, view, index, truth, signals, decoded)


def test_stoppage_runs_stay_inside_the_graded_halves():
    codes = labels((H, 5), (D, 3), (X, 1), (D, 4), (A, 6))
    assert d58.stoppage_runs(codes, pm.valid_runs(codes)) == [(5, 8), (9, 13)]


def test_best_cover_takes_the_largest_overlap_and_the_earliest_on_ties():
    assert d58.best_cover((10, 20), [(0, 12), (15, 30)]) == (1, 5)
    assert d58.best_cover((10, 20), [(5, 15), (15, 25)]) == (0, 5)
    assert d58.best_cover((10, 20), [(0, 5), (25, 30)]) == (None, 0)
    assert d58.best_cover((10, 20), []) == (None, 0)


def test_found_needs_one_predicted_stoppage_covering_at_least_half():
    truth = labels((H, 50), (D, 20), (A, 50))
    exactly_half = labels((H, 60), (D, 10), (A, 50))
    just_under = labels((H, 61), (D, 9), (A, 50))
    split = labels((H, 50), (D, 6), (H, 2), (D, 6), (A, 56))       # two pieces of 6: together 12 of 20
    visible = np.ones(len(truth), dtype=bool)
    assert d58.stoppage_items(truth, exactly_half, visible)[0][0]["found"]
    assert not d58.stoppage_items(truth, just_under, visible)[0][0]["found"]
    item = d58.stoppage_items(truth, split, visible)[0][0]
    assert not item["found"] and item["found_union"]


def test_real_needs_at_least_half_inside_one_true_stoppage_and_hidden_needs_80_percent():
    truth = labels((H, 50), (D, 20), (A, 50), (D, 10), (H, 20))
    prediction = labels((H, 60), (D, 20), (A, 50), (D, 10), (H, 10))   # 10 of 20 inside, then 0 of 10
    visible = flags((True, 130), (False, 10), (True, 10))
    _, predicted = d58.stoppage_items(truth, prediction, visible)
    assert [item["real"] for item in predicted] == [True, False]
    # The second predicted stoppage (frames 130-139) has its ball hidden on all of its 10 frames.
    assert predicted[1]["hidden_frames"] == 10 and predicted[1]["mostly_hidden"]
    assert d58.mostly_hidden(8, 10) and not d58.mostly_hidden(7, 10)


def test_offsets_are_predicted_minus_true():
    truth = labels((H, 50), (D, 30), (A, 50))
    prediction = labels((H, 45), (D, 40), (A, 45))      # stops 5 frames early, restarts 5 frames late
    true_items, _ = d58.stoppage_items(truth, prediction, np.ones(len(truth), dtype=bool))
    assert (true_items[0]["stop_offset"], true_items[0]["restart_offset"]) == (-5, 5)
    late = labels((H, 60), (D, 25), (A, 45))
    true_items, _ = d58.stoppage_items(truth, late, np.ones(len(truth), dtype=bool))
    assert (true_items[0]["stop_offset"], true_items[0]["restart_offset"]) == (10, 5)


def test_frame_reading_partitions_the_full_oracle():
    truth = labels((H, 40), (D, 20), (A, 40), (D, 15), (H, 35))
    prediction = labels((H, 30), (D, 5), (H, 10), (D, 12), (A, 53), (D, 10), (H, 30))
    home = np.where(prediction == H, .9, .1)
    true_items, predicted_items = d58.stoppage_items(truth, prediction, np.ones(len(truth), dtype=bool))
    oracles = d58.one_sided_oracles(truth, prediction, home, true_items, predicted_items)
    full = diagnosis.oracle_stoppage(truth, prediction, home)
    assert np.array_equal(oracles["full"], full)
    frame = oracles["frame"]
    assert np.array_equal(np.where(truth == D, frame["add"], frame["remove"]), full)
    assert np.array_equal(frame["add"] == D, (prediction == D) | (truth == D))
    assert not np.any(frame["remove"][truth != D] == D)


def test_stoppage_reading_touches_only_missed_and_invented_stoppages():
    # True stoppages: 40-59 (missed), 100-114 (found, predicted 95-114). Predicted 30-34 is invented.
    truth = labels((H, 40), (D, 20), (A, 40), (D, 15), (H, 35))
    prediction = labels((H, 30), (D, 5), (H, 25), (A, 35), (D, 20), (H, 35))
    home = np.where(prediction == H, .9, .1)
    true_items, predicted_items = d58.stoppage_items(truth, prediction, np.ones(len(truth), dtype=bool))
    assert [item["found"] for item in true_items] == [False, True]
    assert [item["real"] for item in predicted_items] == [False, True]
    oracles = d58.one_sided_oracles(truth, prediction, home, true_items, predicted_items)
    add, remove = oracles["stoppage"]["add"], oracles["stoppage"]["remove"]
    assert np.all(add[40:60] == D)
    assert np.array_equal(np.delete(add, np.s_[40:60]), np.delete(prediction, np.s_[40:60]))
    assert np.all(remove[30:35] == H)
    assert np.array_equal(np.delete(remove, np.s_[30:35]), np.delete(prediction, np.s_[30:35]))
    # The found stoppage's early stop (frames 95-99) is corrected by the full oracle only.
    assert np.all(oracles["full"][95:100] == A) and np.all(add[95:100] == D) and np.all(remove[95:100] == D)


def test_an_invented_stoppage_keeps_its_true_dead_frames_when_removed():
    truth = labels((H, 50), (D, 4), (A, 50))
    prediction = labels((H, 44), (D, 12), (A, 48))        # 4 of 12 inside the true stoppage: invented
    home = np.where(prediction == H, .9, .1)
    true_items, predicted_items = d58.stoppage_items(truth, prediction, np.ones(len(truth), dtype=bool))
    assert not predicted_items[0]["real"] and true_items[0]["found"]
    remove = d58.one_sided_oracles(truth, prediction, home, true_items, predicted_items)["stoppage"]["remove"]
    assert np.array_equal(remove, diagnosis.oracle_stoppage(truth, prediction, home))
    assert np.all(remove[50:54] == D) and not np.any(remove[44:50] == D) and not np.any(remove[54:56] == D)


def test_length_bins_follow_the_rule():
    fps = diagnosis.FPS
    assert [d58.length_bin(n) for n in (1, 3 * fps - 1, 3 * fps, 10 * fps - 1, 10 * fps, 30 * fps - 1, 30 * fps)] == [
        "under_3s", "under_3s", "3-10s", "3-10s", "10-30s", "10-30s", "over_30s"]


def test_by_length_counts_found_real_and_hidden_invented():
    true_items = [{"length": 50, "found": True}, {"length": 50, "found": False}, {"length": 300, "found": True}]
    predicted_items = [{"length": 40, "real": True, "mostly_hidden": False},
                       {"length": 20, "real": False, "mostly_hidden": True},
                       {"length": 20, "real": False, "mostly_hidden": False},
                       {"length": 800, "real": False, "mostly_hidden": True}]
    table = d58.by_length(true_items, predicted_items)
    assert (table["under_3s"]["true"], table["under_3s"]["found"], table["under_3s"]["found_share"]) == (2, 1, .5)
    assert (table["under_3s"]["predicted"], table["under_3s"]["real"], table["under_3s"]["invented"]) == (3, 1, 2)
    assert table["under_3s"]["invented_hidden_share"] == .5
    assert table["10-30s"]["found_share"] == 1.0 and table["10-30s"]["predicted"] == 0
    assert table["10-30s"]["real_share"] is None and table["over_30s"]["invented_hidden_share"] == 1.0
    assert table["all"]["true"] == 3 and table["all"]["invented"] == 3 and table["all"]["invented_mostly_hidden"] == 2
    assert table["all"]["missed_frames"] == 50 and table["all"]["invented_frames"] == 840


def scored_fixture():
    truth = labels((H, 300), (D, 100), (A, 300), (D, 60), (H, 240))
    miss = labels((H, 300), (D, 10), (H, 90), (A, 300), (D, 60), (H, 240))     # the first stoppage missed
    invent = labels((H, 150), (D, 40), (H, 110), (D, 100), (A, 300), (D, 60), (H, 240))
    halves = []
    for source, view, match in (("dfl", "clean", "WPY"), ("dfl", "masked", "WPY"), ("skillcorner", "observed", "1"),
                                ("pff", "observed", "1"), ("guard_metrica", "long_hidden", "1"),
                                ("d24_masked", "masked", "1")):
        for index, decoded in enumerate((miss, invent)):
            halves.append(half(source, match, view, index, truth, decoded))
    return halves


def test_measure_stoppages_agrees_with_d21_and_groups_guard_and_d24_views_apart():
    halves = scored_fixture()
    scored = diagnosis.matches(halves)
    groups_names = ("dfl_clean", "dfl_masked", "skillcorner", "pff", "guard_metrica", "d24_masked")
    visible = {key: np.ones(len(truth), dtype=bool) for key, (truth, _, _) in scored.items()}
    groups, combined = d58.measure_stoppages(scored, visible, groups=groups_names)
    report = diagnosis.measure_all(scored, groups=groups_names, vendors={name: (name,) for name in groups_names},
                                   name="stack", with_shifts=False)
    for name in groups_names:
        entry, oracle = report["groups"][d58.TARGET][name], groups[name]["oracle"]["spell_f1_mean"]
        assert oracle["decoded"] == entry["spell_f1_mean"] and oracle["full"] == entry["oracle"]["stoppage"]
        assert groups[name]["counts"] == {"true": 4, "predicted": 5}
        assert groups[name]["by_length"]["all"]["found"] == 3 and groups[name]["by_length"]["all"]["invented"] == 1
        # Adding the missed stoppage (half 0) and removing the invented one (half 1) each help.
        gain = groups[name]["oracle"]["gain"]
        assert gain["stoppage_add"] > 0 and gain["stoppage_remove"] > 0 and gain["full"] > 0
        assert np.isclose(groups[name]["oracle"]["edges"]["stoppage"],
                          gain["full"] - gain["stoppage_add"] - gain["stoppage_remove"])
    assert combined["full"]["combined"] == report["combined"][d58.TARGET]["oracle_stoppage"]["combined"]


def test_guard_halves_get_their_own_groups():
    halves = d58.guard_grouped([half("dfl", "WPY", "long_hidden", 0, labels((H, 5))),
                                half("metrica", "2", "long_hidden", 1, labels((A, 5)))])
    assert [item.source for item in halves] == ["guard_dfl", "guard_metrica"]
    assert [item.group for item in halves] == ["guard_dfl", "guard_metrica"]


def gains(full, add, remove):
    return {"oracle": {"gain": {"full": full, "stoppage_add": add, "stoppage_remove": remove,
                                "frame_add": add, "frame_remove": remove},
                       "edges": {"stoppage": full - add - remove, "frame": full - add - remove}}}


def test_rule_reading_needs_the_line_and_two_thirds_on_one_side():
    groups = {name: gains(.06, .045, .01) for name in d58.REALISTIC}
    verdict = d58.rule_reading(groups, reading="stoppage", average="groups")
    assert verdict["result"]["worth"] and verdict["result"]["sides_at_two_thirds"] == ["add"]
    assert verdict["outcome"] == "a stoppage rule is written for the missed side"
    assert np.isclose(verdict["result"]["edges"], .005) and np.isclose(verdict["per_group"]["pff"]["edges"], .005)
    groups = {name: gains(.06, .03, .03) for name in d58.REALISTIC}
    assert d58.rule_reading(groups, reading="stoppage", average="groups")["outcome"] == "round 13 ends with the adopted stack"
    groups = {name: gains(.03, .03, 0.) for name in d58.REALISTIC}
    verdict = d58.rule_reading(groups, reading="frame", average="groups")
    assert not verdict["result"]["worth"] and verdict["result"]["rule_for"] is None


def test_two_thirds_is_exact():
    groups = {name: gains(.06, .04, 0.) for name in d58.REALISTIC}
    assert d58.rule_reading(groups, reading="stoppage", average="groups")["result"]["sides_at_two_thirds"] == ["add"]


def test_family_average_weights_the_d24_views_and_the_guards_as_one_each():
    groups = {name: gains(.0, .0, .0) for name in d58.REALISTIC}
    for name in d46.D24_GROUPS:
        groups[name] = gains(.07, .07, 0.)
    by_groups = d58.rule_reading(groups, reading="stoppage", average="groups")["result"]["full"]
    by_families = d58.rule_reading(groups, reading="stoppage", average="families")["result"]["full"]
    assert np.isclose(by_groups, .07 * 3 / 7) and np.isclose(by_families, .07 / 4)
    verdict = d58.rule_reading(groups, reading="stoppage", average="groups")
    assert set(verdict["supplementary"]) == {"note", "stoppage/families", "frame/groups", "frame/families"}
    assert (d58.READING, d58.AVERAGE) == ("stoppage", "families")
    verdict = d58.rule_reading(groups, reading=d58.READING, average=d58.AVERAGE)
    assert set(verdict["supplementary"]) == {"note", "stoppage/groups", "frame/groups", "frame/families"}


def test_the_reading_must_be_named_before_anything_is_decoded(tmp_path):
    for reading, average in (("both", "families"), ("stoppage", "views")):
        try:
            d58.measure(tmp_path, tmp_path, tmp_path, raw=True, output=tmp_path / "d58.json", reading=reading,
                        average=average)
        except ValueError as error:
            assert "named" in str(error)
        else:
            raise AssertionError("measure ran without a named reading")
    groups = {name: gains(.06, .045, .01) for name in d58.REALISTIC}
    try:
        d58.rule_reading(groups, reading="stoppage", average=None)
    except ValueError:
        pass
    else:
        raise AssertionError("rule_reading read without a named average")


def test_compare_row_reports_each_part():
    stored = {"composite": .7, "guards": {"mean": .73}, "owner_cost": 14}
    assert all(d58.compare_row(dict(stored), stored).values())
    checks = d58.compare_row(dict(stored, composite=.71), stored)
    assert not checks["composite"] and not checks["whole_row"] and checks["guards"]


def test_row_group_means_reads_the_twelve_groups():
    row = {"groups": {name: .5 for name in ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap",
                                            "skillcorner", "pff", "metrica", "d24")},
           "d24_views": {view: {"1": {"aba3": {"spell_f1": .6}}, "2": {"aba3": {"spell_f1": .8}}}
                         for view in ("masked", "handled", "team_swap")},
           "guards": {"a_dfl": .75, "b_metrica": .72}}
    means = d58.row_group_means(row)
    assert set(means) == set(d58.GROUPS)
    assert np.isclose(means["d24_handled"], .7) and means["guard_metrica"] == .72


def test_adopted_networks_compares_fit_digests():
    identity = {"networks": {"0": {"current": {"seed_models": "a", "temporal_models": "b"}}}}
    adoption = {"copied": {"0/current/seed_models": {"files": {"fit.json": "a"}},
                           "0/current/temporal_models": {"files": {"fit.json": "b"}}}}
    assert d58.adopted_networks(identity, adoption)
    adoption["copied"]["0/current/temporal_models"]["files"]["fit.json"] = "c"
    assert not d58.adopted_networks(identity, adoption)
