"""D59's tree-share blend, member cache, rule and refit helpers on synthetic fixtures."""

import json

import numpy as np

from dfl_submission import motion_trial, stack_trial as st, tree_share as ts, tree_share_trial as tt
from dfl_submission.pff import digest


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


PREVIOUS = {"variant": "temporal_1800.npz", "owner_mix": .5, "dead_mix": .5}
RECIPE = {"variant": "seed_phase", "owner_mix": .5, "dead_mix": .5}


def members(n=400, seed=0, with_refit=True):
    rng = np.random.default_rng(seed)
    arrays = lambda count: tuple(rng.random(n).astype(np.float32) for _ in range(count))
    visible = np.ones(n, dtype=bool)
    visible[100:250] = False
    return {"seen": visible, "trees_d3": arrays(2), "trees_r": arrays(2) if with_refit else None, "late": arrays(4),
            "seed": arrays(4)}


def test_without_h_the_blend_is_motion_trial_s():
    m = members()
    for tree_set in ("d3", "r"):
        mixed = ts.structure(m, tree_set)
        expected = motion_trial.blend_signals(mixed, PREVIOUS, RECIPE)
        fresh = ts.signals(m, tree_set, None, PREVIOUS, RECIPE)
        assert all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(fresh, expected))
    assert refused(lambda: ts.structure(members(with_refit=False), "r"), ValueError)


def test_h_changes_only_the_home_input_on_unseen_frames():
    m = members()
    base = ts.signals(m, "d3", None, PREVIOUS, RECIPE)
    for share in ts.SHARES:
        fresh = ts.signals(m, "d3", share, PREVIOUS, RECIPE)
        assert fresh[0].dtype == np.float32
        assert np.array_equal(fresh[0][m["seen"]], base[0][m["seen"]])
        assert all(np.array_equal(a, b) for a, b in zip(fresh[1:], base[1:]))
        trees, slot, seed_home = m["trees_d3"][0], m["late"][0], m["seed"][0]
        unseen = ~m["seen"]
        expected = .5 * (share * trees + (1 - share) * slot) + .5 * seed_home
        assert np.allclose(fresh[0][unseen], expected[unseen], atol=1e-6)
    zero = ts.signals(m, "d3", 0.0, PREVIOUS, RECIPE)
    assert np.array_equal(zero[0][~m["seen"]], (.5 * m["late"][0] + .5 * m["seed"][0])[~m["seen"]])
    # a share of 0.5 on unseen frames is the reference exactly
    half = ts.signals(m, "d3", .5, PREVIOUS, RECIPE)
    assert all(np.array_equal(a, b) for a, b in zip(half, base))


def test_seen_follows_the_ball_s_x_and_y():
    ball = np.zeros((6, 2), dtype=np.float32)
    ball[1, 0] = np.nan
    ball[3] = np.nan
    assert ts.seen(ball).tolist() == [True, False, True, False, True, True]


def test_member_cache_round_trip_and_identity(tmp_path):
    halves = [ts.MemberHalf("dfl", "WPY", "clean", i, np.arange(400) % 3, members(seed=i, with_refit=i == 0)) for i in (0, 1)]
    stored = ts.collect(tmp_path / "part", {"a": 1}, lambda: iter(halves))
    loaded = ts.collect(tmp_path / "part", {"a": 1}, lambda: (_ for _ in ()).throw(AssertionError("recomputed")))
    for original, again in zip(stored, loaded):
        assert (again.source, again.match, again.view, again.index) == (original.source, original.match, original.view,
                                                                         original.index)
        for field in ts.FIELDS:
            if original.members[field] is None:
                assert again.members[field] is None
            else:
                assert all(np.array_equal(a, b) for a, b in zip(original.members[field], again.members[field]))
        assert np.array_equal(again.members["seen"], original.members["seen"])
    assert refused(lambda: ts.collect(tmp_path / "part", {"a": 2}, lambda: iter(halves)), ValueError)
    path = tmp_path / "part" / json.loads((tmp_path / "part" / "manifest.json").read_text())["halves"][0]["file"]
    np.savez_compressed(path, labels=np.zeros(3))
    assert refused(lambda: ts.collect(tmp_path / "part", {"a": 1}, lambda: iter(halves)), ValueError)


def item(cost, composite, guard, pff=.70, shift=0.0, unseen=.85):
    groups = {name: .72 + shift for name in st.GROUPS}
    groups["pff"] = pff
    result = {"owner_cost": cost, "composite": composite, "guards": {"mean": guard, "a_dfl": guard, "b_metrica": guard},
              "groups": groups, "primary": .73, "situation_d": .75,
              "pff_visibility": {"agreement_unseen": {"agreement": unseen}, "under_20%_seen_spells": {"recall_iou0.5": .2}}}
    return dict(result, selection=(composite + guard) / 2)


def full(values):
    return {"trial": {"external": {"conventions": {name: {"standard": {"spell_f1": value}} for name, value in values.items()}}}}


CONV = {"aba3": .73, "aba1.75": .70, "aba4": .74, "aba4.5": .74}


def test_judge_applies_d59_s_five_conditions():
    reference = item(14, .728, .737)
    good = item(12, .740, .745, pff=.711)
    result = ts.judge(good, full(CONV), reference, full(CONV))
    assert result["conditions"] == {"selection": True, "pff": True, "composite": True, "groups": True, "conventions": True}
    assert result["holds"] and np.isclose(result["changes"]["selection"], (.740 + .745 - .728 - .737) / 2)
    assert not ts.judge(item(12, .740, .745, pff=.7099), full(CONV), reference, full(CONV))["conditions"]["pff"]
    assert not ts.judge(item(12, .7279, .765, pff=.711), full(CONV), reference, full(CONV))["conditions"]["composite"]
    low = item(12, .740, .745, pff=.711)
    low["groups"]["dfl_team_swap"] = .72 - .0101
    assert not ts.judge(low, full(CONV), reference, full(CONV))["conditions"]["groups"]
    worse = dict(CONV, **{"aba4.5": .74 - .0101})
    assert not ts.judge(good, full(worse), reference, full(CONV))["conditions"]["conventions"]
    small = item(12, .735, .737, pff=.711)
    assert not ts.judge(small, full(CONV), reference, full(CONV))["conditions"]["selection"]


def test_best_cost_and_decision():
    pairs = [(None, item(12, .74, .74)), (None, item(14, .745, .735)), (None, item(16, .73, .73))]
    assert ts.best(pairs)[1]["owner_cost"] == 12                      # a tie in selection goes to the lower cost
    results = {"H(0)": {"holds": True, "selection": .75}, "R": {"holds": True, "selection": .76},
               "RH(0)": {"holds": False, "selection": .80}}
    assert ts.decision(results) == "R"
    assert ts.decision({"R": {"holds": False, "selection": .9}}) is None
    assert set(ts.CANDIDATES) == {"H(0.25)", "H(0)", "R", "RH(0.25)", "RH(0)"} and ts.COSTS == (12, 14, 16)


def test_refit_folders_check_the_record(tmp_path):
    record = tmp_path / "d59_refit.json"
    folders = {}
    for key in st.SETS:
        folder = tmp_path / "trees" / "refit" / key
        folder.mkdir(parents=True)
        (folder / "fit.json").write_text(json.dumps({"models": {"ownership.ubj": key}}))
        folders[key] = folder
    record.write_text(json.dumps({"passed": True, "refits": {key: {"ownership.ubj": key} for key in st.SETS},
                                  "sources": {key: {"refit_fit_sha256": digest(folders[key] / "fit.json")} for key in st.SETS}}))
    assert tt.refit_folders(record, tmp_path / "trees") == folders
    (folders["current"] / "fit.json").write_text(json.dumps({"models": {"ownership.ubj": "other"}}))
    assert refused(lambda: tt.refit_folders(record, tmp_path / "trees"), ValueError)
    record.write_text(json.dumps({"passed": False}))
    assert refused(lambda: tt.refit_folders(record, tmp_path / "trees"), SystemExit)


def test_records_once(tmp_path):
    ts.write_new(tmp_path / "r.json", {"a": 1})
    assert refused(lambda: ts.write_new(tmp_path / "r.json", {}), FileExistsError)
    tt.write_new(tmp_path / "s.json", {"a": 1})
    assert refused(lambda: tt.write_new(tmp_path / "s.json", {}), FileExistsError)


def test_far_from_sighting_on_the_25_hz_grid():
    visible = np.zeros(1000, dtype=bool)
    visible[:100] = True                       # seen until frame 99
    visible[700:710] = True                    # a short sighting inside the half
    far = tt.far_from_sighting(visible)
    assert tt.GAP == 75
    assert not far[:100].any() and not far[700:710].any()          # seen frames are 0 frames from a sighting
    assert not far[100:175].any()                                  # within 3 s after the last sighting (1-75 frames)
    assert far[175]                                                # 76 frames after it
    assert far[400]                                                # deep inside the long gap
    assert not far[625:700].any() and far[624]                     # within 3 s before the next sighting, then not
    assert not far[710:785].any() and far[785:].all()              # after the short sighting to the end of the half
    gap = np.zeros(400, dtype=bool)
    gap[0] = gap[-1] = True
    distance_ok = tt.far_from_sighting(gap)
    assert not distance_ok[:76].any() and not distance_ok[-76:].any() and distance_ok[76:-76].all()
    assert tt.far_from_sighting(np.zeros(300, dtype=bool)).all()   # no sighting in the half: farther than 3 s
    assert not tt.far_from_sighting(np.ones(300, dtype=bool)).any()
    assert tt.far_from_sighting(np.zeros(0, dtype=bool)).shape == (0,)


def test_ht_changes_home_only_far_from_sightings():
    m = members(n=600)
    m["seen"] = np.ones(600, dtype=bool)
    m["seen"][100:400] = False                 # one 12 s gap
    base = ts.signals(m, "d3", None, PREVIOUS, RECIPE)
    for share in ts.SHARES:
        fresh = tt.timed_signals(m, "d3", share, PREVIOUS, RECIPE)
        far = tt.far_from_sighting(m["seen"])
        assert far.sum() == 300 - 2 * 75
        assert np.array_equal(fresh[0][~far], base[0][~far])        # short-gap and seen frames keep 0.5
        assert all(np.array_equal(a, b) for a, b in zip(fresh[1:], base[1:]))
        expected = .5 * (share * m["trees_d3"][0] + (1 - share) * m["late"][0]) + .5 * m["seed"][0]
        assert np.allclose(fresh[0][far], expected[far], atol=1e-6)
        h = ts.signals(m, "d3", share, PREVIOUS, RECIPE)            # H changes the whole gap, HT only its interior
        assert not np.array_equal(h[0][100:175], fresh[0][100:175])
    assert set(tt.TIMED) == {"HT(0.25)", "HT(0)", "RHT(0.25)", "RHT(0)"}
    assert tt.TIMED["RHT(0)"] == ("r", 0.0)
