"""D46 measurements on synthetic label sequences and ball visibility."""

import json

import numpy as np

from dfl_submission import diagnosis, diagnosis_d43 as d46
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


def test_frames_since_seen_counts_from_the_last_seen_frame():
    visible = np.array([False, False, True, False, False, True, False])
    since = d46.frames_since_seen(visible)
    assert np.isinf(since[:2]).all()
    assert since[2:].tolist() == [0, 1, 2, 0, 1]


def test_time_since_seen_bins_follow_the_rule():
    since = np.array([0, 1, 25, 26, 75, 76, np.inf])
    assert d46.since_bin(since).tolist() == ["0", "up_to_1s", "up_to_1s", "1-3s", "1-3s", "over_3s", "over_3s"]


def test_split_oracles_partition_the_full_ownership_oracle():
    truth = labels((H, 20), (A, 20), (D, 10), (H, 20))
    prediction = labels((H, 30), (A, 10), (D, 12), (A, 18))
    visible = flags((True, 25), (False, 20), (True, 10), (False, 15))
    oracles = d46.split_oracles(truth, prediction, visible)
    full = diagnosis.oracle_ownership(truth, prediction)
    assert np.array_equal(oracles["full"], full)
    assert np.array_equal(np.where(visible, oracles["seen"], oracles["unseen"]), full)
    # Each restricted oracle keeps the decoded labels where it does not apply.
    assert np.array_equal(oracles["unseen"][visible], prediction[visible])
    assert np.array_equal(oracles["seen"][~visible], prediction[~visible])
    # Predicted stoppages stay stoppages under every oracle.
    for result in oracles.values():
        assert np.array_equal(result == D, prediction == D)


def test_unseen_oracle_recovers_an_error_made_where_the_ball_is_hidden():
    truth = labels((H, 100), (A, 100), (H, 100))
    prediction = labels((H, 300))
    visible = flags((True, 100), (False, 100), (True, 100))
    record = d46.visibility_match(truth, prediction, visible, d46.frames_since_seen(visible))
    assert record["oracle"]["unseen"] == record["oracle"]["full"] == 1.0
    assert record["oracle"]["seen"] == record["oracle"]["decoded"] < 1.0
    assert record["oracle_changed"] == {"frames": 100, "unseen": 100}


def test_agreement_is_split_by_visibility_and_time_since_seen():
    truth = labels((H, 50), (D, 10), (A, 40))
    prediction = labels((H, 40), (A, 10), (D, 10), (A, 20), (H, 20))
    visible = flags((True, 45), (False, 55))
    record = d46.visibility_match(truth, prediction, visible, d46.frames_since_seen(visible))
    assert record["target_live"] == 90
    assert record["target_live_unseen"] == 5 + 40
    assert record["agreement"]["seen"] == {"frames": 45, "agree": 40}
    assert record["agreement"]["unseen"] == {"frames": 45, "agree": 20}
    assert record["by_since"]["0"] == {"frames": 45, "agree": 40}
    # Frames 45-69 are 1-25 frames after the last seen frame (44), 70-99 are 26-55 frames after it.
    assert record["by_since"]["up_to_1s"] == {"frames": 15, "agree": 10}
    assert record["by_since"]["1-3s"] == {"frames": 30, "agree": 10}
    assert record["by_since"]["over_3s"] == {"frames": 0, "agree": 0}


def test_recall_by_seen_share_uses_the_metric_matching():
    truth = labels((H, 100), (A, 100), (H, 100))
    prediction = labels((H, 300))
    visible = flags((True, 100), (False, 90), (True, 110))
    record = d46.visibility_match(truth, prediction, visible, d46.frames_since_seen(visible))
    shares = [share for share, _, _ in record["spells"]]
    assert shares == [1.0, .1, 1.0]
    # The single predicted H spell matches neither H spell at IoU 0.5 (IoU 1/3), and A is never predicted.
    assert [hit for _, _, hit in record["spells"]] == [False, False, False]
    lengths, matched, _ = diagnosis.spell_lengths(truth, prediction)
    assert lengths.tolist() == [length for _, length, _ in record["spells"]]
    assert matched[pm.SPELL_IOUS.index(.5)].tolist() == [hit for _, _, hit in record["spells"]]
    by_share, crossed = d46.seen_share_recall([(1.0, 100, True), (.1, 100, False), (.5, 30, True), (.85, 200, False)])
    assert by_share["80%+"] == {"count": 2, "recall_iou0.5": .5}
    assert by_share["under_20%"] == {"count": 1, "recall_iou0.5": 0.0}
    assert by_share["20-80%"] == {"count": 1, "recall_iou0.5": 1.0}
    assert crossed["20-80%"]["0-1.75s"] == {"count": 1, "recall_iou0.5": 1.0}
    assert crossed["80%+"]["5-10s"] == {"count": 1, "recall_iou0.5": 0.0}


def test_attach_stops_when_a_rebuilt_view_differs_from_the_cache():
    truth = labels((H, 10), (A, 10))
    halves = [half("pff", "1", "observed", 0, truth)]
    visible = np.ones(20, dtype=bool)
    result, frames = d46.attach(halves, {("pff", "1", "observed", 0): (truth, visible)})
    assert result[("pff", "1", "observed", 0)] is visible and frames == {"pff/1/observed/0": 20}
    for rebuilt in ((truth[:-1], visible[:-1]), (labels((A, 10), (A, 10)), visible)):
        try:
            d46.attach(halves, {("pff", "1", "observed", 0): rebuilt})
        except ValueError:
            continue
        raise AssertionError("A differing rebuilt view was accepted")


def test_d24_views_become_groups_of_their_own():
    truth = labels((H, 60), (A, 60))
    decoded = labels((H, 70), (A, 50))
    halves = []
    for view in ("observed", *d46.D24_VIEWS):
        for index in (0, 1):
            halves.append(half("metrica", "1", view, index, truth, decoded))
    metrica = [item for item in halves if item.view == "observed"]
    grouped = metrica + d46.d24_grouped([item for item in halves if item.view != "observed"])
    scored = diagnosis.matches(grouped)
    groups = ("metrica", *d46.D24_GROUPS)
    report = diagnosis.measure_all(scored, groups=groups, vendors={group: (group,) for group in groups}, name="d43",
                                   with_shifts=False)
    assert set(report["groups"]["aba3"]) == set(groups)
    assert all(report["groups"]["aba3"][group]["matches"] == 1 for group in groups)
    visibility = {("metrica", "1", view, index): (truth, np.ones(120, dtype=bool))
                  for view in ("observed", *d46.D24_VIEWS) for index in (0, 1)}
    attached, _ = d46.attach(grouped, d46.renamed(visibility))
    visible = d46.joined(grouped, attached, False)
    assert set(visible) == set(scored)
    assert all(len(visible[key]) == len(scored[key][0]) and not visible[key][120] for key in visible)
    since = d46.joined(grouped, {key: d46.frames_since_seen(value) for key, value in attached.items()}, np.inf)
    result, combined = d46.measure_visibility(scored, visible, since, groups=groups)
    assert combined is None
    for group in groups:
        assert result[group]["split_oracle"]["spell_f1_mean"]["decoded"] == report["groups"]["aba3"][group]["spell_f1_mean"]
        assert result[group]["split_oracle"]["spell_f1_mean"]["full"] == report["groups"]["aba3"][group]["oracle"]["ownership"]
        assert result[group]["target_live_unseen_share"] == 0.0
        # The ball is always seen, so the unseen-only oracle changes nothing.
        assert result[group]["split_oracle"]["gain"]["unseen"] == 0.0


def test_degraded_shares_report_each_group_and_the_pools():
    def entry(decoded, full, unseen):
        gain = {"full": full - decoded, "unseen": unseen - decoded, "seen": 0.0}
        return {"split_oracle": {"gain": gain, "unseen_share_of_full_gain": d46.ratio(gain["unseen"], gain["full"])}}
    groups = {"dfl_masked": entry(.70, .90, .80), "skillcorner": entry(.70, .80, .72), "pff": entry(.60, .60, .60),
              "d24_masked": entry(.70, .90, .85), "d24_handled": entry(.70, .90, .85), "d24_team_swap": entry(.70, .90, .85)}
    shares = d46.degraded_shares(groups)
    assert abs(shares["per_group"]["dfl_masked"] - .5) < 1e-12
    assert abs(shares["per_group"]["skillcorner"] - .2) < 1e-12
    assert shares["per_group"]["pff"] is None and shares["at_least_half"]["pff"] is False
    assert abs(shares["d24_mean"] - .75) < 1e-12
    assert abs(shares["pooled_groups"] - (.1 + .02 + 0 + .45) / (.2 + .1 + 0 + .6)) < 1e-12
    assert abs(shares["pooled_families"] - (.1 + .02 + 0 + .15) / (.2 + .1 + 0 + .2)) < 1e-12


def test_compare_row_names_the_part_that_differs():
    stored = {"point": {"owner_cost": 10}, "eligible": False, "changes": {"primary": .01},
              "trial": {"external": {"a": 1.0}, "pff": {"b": 2.0}}, "metrica": {"m": 1}, "d24": {"c": .5}}
    same = json.loads(json.dumps(stored))
    assert all(d46.compare_row(same, stored).values())
    changed = json.loads(json.dumps(stored))
    changed["trial"]["pff"]["b"] = 2.0000000001
    checks = d46.compare_row(changed, stored)
    assert checks["trial_pff"] is False and checks["whole_row"] is False and checks["trial_external"] is True


def test_cached_refuses_to_forecast_without_a_cache(tmp_path):
    try:
        d46.cached(tmp_path / "missing", {"protocol": "x"})
    except ValueError as error:
        assert "forecasts nothing" in str(error)
    else:
        raise AssertionError("A missing cache was accepted")
    folder = tmp_path / "cache"
    folder.mkdir()
    (folder / "manifest.json").write_text(json.dumps({"identity": {"protocol": "y"}, "halves": []}))
    try:
        d46.cached(folder, {"protocol": "x"})
    except ValueError:
        pass
    else:
        raise AssertionError("A cache with a different identity was accepted")
    assert d46.cached(folder, {"protocol": "y"}) == []


def test_write_new_writes_once(tmp_path):
    path = tmp_path / "d46.json"
    d46.write_new(path, {"a": 1})
    try:
        d46.write_new(path, {"a": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("D46 was written twice")
    assert json.loads(path.read_text()) == {"a": 1}


# -- D46a ------------------------------------------------------------------------------------------

def test_hidden_run_lengths_mark_every_frame_of_a_run_with_its_length():
    visible = flags((False, 3), (True, 2), (False, 4), (True, 1), (False, 1))
    assert d46.hidden_run_lengths(visible).tolist() == [3, 3, 3, 0, 0, 4, 4, 4, 4, 0, 1]
    assert d46.hidden_run_lengths(np.ones(5, dtype=bool)).tolist() == [0] * 5


def test_run_bins_follow_the_rule():
    lengths = np.array([1, 25, 26, 75, 76, 250, 251, 10_000])
    assert d46.run_bin(lengths).tolist() == ["up_to_1s", "up_to_1s", "1-3s", "1-3s", "3-10s", "3-10s",
                                             "over_10s", "over_10s"]


def test_visible_players_counts_finite_slots():
    positions = np.zeros((3, 4, 2))
    positions[0, 1] = np.nan
    positions[1, :, 0] = np.nan
    positions[2, 3, 1] = np.nan
    assert d46.visible_players(positions).tolist() == [3, 0, 3]


def test_hidden_match_bins_unseen_frames_by_their_run():
    truth = labels((H, 300), (A, 300))
    prediction = labels((H, 260), (A, 40), (H, 20), (A, 280))
    visible = flags((True, 250), (False, 300), (True, 30), (False, 20))
    lengths = d46.hidden_run_lengths(visible)
    record = d46.hidden_match(truth, prediction, visible, lengths)
    assert record["by_run"]["over_10s"] == {"unseen_live": 300, "frames": 300, "agree": 240}
    assert record["by_run"]["up_to_1s"] == {"unseen_live": 20, "frames": 20, "agree": 20}
    assert record["by_run"]["1-3s"]["unseen_live"] == record["by_run"]["3-10s"]["unseen_live"] == 0
    summary = d46.summarise_hidden([record])
    assert summary["unseen_live_frames"] == 320
    assert summary["share_by_run"]["over_10s"] == 300 / 320
    assert summary["run_length_s"]["median"] == 300 / 25
    assert summary["agreement_by_run"]["over_10s"] == {"frames": 300, "agreement": 240 / 300}
    assert "players" not in summary


def test_hidden_match_splits_unseen_frames_and_spells_by_visible_players():
    truth = labels((H, 100), (A, 100), (H, 100))
    prediction = labels((H, 100), (H, 100), (H, 100))
    visible = flags((True, 100), (False, 100), (True, 100))
    players = np.r_[np.full(150, 20), np.full(150, 10)]
    record = d46.hidden_match(truth, prediction, visible, d46.hidden_run_lengths(visible), players)
    assert record["players"]["unseen_live"] == 100 and record["players"]["unseen_live_many"] == 50
    assert record["players"]["agreement"]["many"] == {"frames": 50, "agree": 0}
    assert record["players"]["agreement"]["few"] == {"frames": 50, "agree": 0}
    summary = d46.summarise_hidden([record])
    assert summary["players"]["unseen_live_share_many"] == .5
    # The one under-20%-seen spell (A, never seen) has a mean of 15 visible players: "few", and it is missed.
    assert summary["players"]["under_20%_seen_spell_recall"] == {"many": {"count": 0, "recall_iou0.5": None},
                                                                 "few": {"count": 1, "recall_iou0.5": 0.0}}
