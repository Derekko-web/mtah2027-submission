"""D11 measurements on synthetic label sequences."""

import json

import numpy as np

from dfl_submission import diagnosis
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def test_nearest_fill_prefers_the_earlier_value_on_ties():
    values = np.array([H, D, D, A], dtype=np.int8)
    filled = diagnosis.nearest_fill(values, values != D)
    assert filled.tolist() == [H, H, A, A]
    values = np.array([D, D, A, D, D, D, H], dtype=np.int8)
    assert diagnosis.nearest_fill(values, values != D).tolist() == [A, A, A, A, A, H, H]


def test_oracle_stoppage_uses_true_dead_frames_and_keeps_predicted_teams():
    truth = labels((H, 10), (D, 10), (A, 10))
    prediction = labels((H, 8), (D, 4), (H, 6), (D, 2), (A, 10))
    home = np.full(len(truth), .2)
    result = diagnosis.oracle_stoppage(truth, prediction, home)
    assert result[10:20].tolist() == [D] * 10
    assert result[:10].tolist() == [H] * 10
    assert result[20:].tolist() == [A] * 10


def test_oracle_stoppage_falls_back_to_the_home_signal_when_a_live_segment_was_predicted_dead():
    truth = labels((D, 5), (H, 10), (D, 5))
    prediction = labels((D, 20))
    home = np.r_[np.full(5, .1), np.full(10, .7), np.full(5, .1)]
    result = diagnosis.oracle_stoppage(truth, prediction, home)
    assert result.tolist() == [D] * 5 + [H] * 10 + [D] * 5


def test_oracle_ownership_keeps_predicted_stoppages_and_uses_true_teams():
    truth = labels((H, 10), (D, 5), (A, 10))
    prediction = labels((A, 12), (D, 3), (H, 10))
    result = diagnosis.oracle_ownership(truth, prediction)
    # Predicted-live frames where the truth is dead take the truth's most recent team.
    assert result.tolist() == [H] * 12 + [D] * 3 + [A] * 10
    early = labels((D, 4), (A, 6))
    assert diagnosis.oracle_ownership(early, labels((H, 10))).tolist() == [A] * 10


def test_offsets_are_signed_and_typed():
    truth = labels((H, 100), (A, 100), (D, 100), (H, 100))
    prediction = labels((H, 103), (A, 95), (D, 104), (H, 98))
    result = diagnosis.offsets(truth, prediction)
    assert result["ownership"]["offsets"] == [3]
    assert result["stop"]["offsets"] == [-2]
    assert result["restart"]["offsets"] == [2]
    assert all(result[kind]["missed"] == result[kind]["spurious"] == 0 for kind in diagnosis.TYPES)


def test_offsets_count_missed_and_spurious_transitions():
    truth = labels((H, 200), (A, 200))
    prediction = labels((H, 50), (D, 50), (H, 300))
    result = diagnosis.offsets(truth, prediction)
    assert result["ownership"]["missed"] == 1
    assert result["stop"]["spurious"] == 1 and result["restart"]["spurious"] == 1


def test_shift_rule_needs_three_vendors_to_agree_and_two_frames():
    def vendors(dfl, skill, pff):
        return {name: {kind: {"median": value} for kind in diagnosis.TYPES}
                for name, value in (("dfl", dfl), ("skillcorner", skill), ("pff", pff))}

    assert diagnosis.shifts(vendors(4, 3, 9))["stop"]["shift"] == -4  # (4 + 3) / 2 = 3.5 rounds away from zero
    assert diagnosis.shifts(vendors(-2, -3, -1))["stop"]["shift"] == 3
    assert diagnosis.shifts(vendors(4, 3, -1))["stop"]["shift"] is None
    assert diagnosis.shifts(vendors(1, 1, 5))["stop"]["shift"] is None
    assert diagnosis.shifts(vendors(0, 4, 4))["stop"]["shift"] is None
    assert diagnosis.round_half_away(1.5) == 2 and diagnosis.round_half_away(-2.5) == -3


def test_shift_rule_rounds_one_and_a_half_frames_up_to_two():
    result = diagnosis.shifts({name: {kind: {"median": value} for kind in diagnosis.TYPES}
                               for name, value in (("dfl", 1), ("skillcorner", 2), ("pff", 1))})
    assert result["restart"]["shift"] == -2


def test_brief_interior_spells_are_counted_and_matched():
    truth = labels((H, 200), (A, 60), (H, 200))
    prediction = labels((H, 200), (A, 60), (H, 100), (A, 20), (H, 80))
    lengths, matched, brief = diagnosis.spell_lengths(truth, prediction)
    assert lengths.tolist() == [200, 60, 200]
    # The H spell between the two A spells is interior too, but at 4 s it is not brief.
    assert sorted(brief) == [(20, False), (60, True), (100, True)]
    record = {"brief": brief}
    summary = diagnosis.brief_summary([record])
    assert summary["under_1.75s"] == {"count": 1, "matched_iou0.5": 0}
    assert summary["under_3s"] == {"count": 2, "matched_iou0.5": 1}
    assert matched.shape == (len(pm.SPELL_IOUS), 3)


def test_cache_round_trip_and_identity_check(tmp_path):
    half = diagnosis.HalfSignals("dfl", "WPY", "clean", 0, labels((H, 5), (A, 5)),
                                 tuple(np.linspace(0, 1, 10, dtype=np.float32) for _ in range(4)),
                                 labels((H, 6), (A, 4)))
    other = diagnosis.HalfSignals("dfl", "WPY", "clean", 1, labels((A, 10)), half.signals, labels((A, 10)))
    cache = tmp_path / "cache"
    first = diagnosis.collect(cache, {"teacher": "x"}, lambda: iter([half, other]))
    second = diagnosis.collect(cache, {"teacher": "x"}, lambda: iter(()))
    assert len(first) == len(second) == 2
    assert np.array_equal(second[0].d3, half.d3) and second[0].signals[0].dtype == np.float32
    try:
        diagnosis.collect(cache, {"teacher": "y"}, lambda: iter(()))
    except ValueError:
        pass
    else:
        raise AssertionError("A cache for other inputs must be refused")
    manifest = json.loads((cache / "manifest.json").read_text())
    assert [entry["index"] for entry in manifest["halves"]] == [0, 1]
    scored = diagnosis.matches(second)
    truth, prediction, home = scored["dfl", "WPY", "clean"]
    assert len(truth) == 21 and truth[10] == X and np.isnan(home[10])


def test_measurements_run_on_a_small_match():
    truth = np.r_[labels((H, 300), (D, 100), (A, 300)), X, labels((A, 200), (H, 40), (A, 200))]
    prediction = np.r_[labels((H, 305), (D, 95), (A, 300)), X, labels((A, 440))]
    home = np.where(prediction == H, .8, .2).astype(float)
    record = diagnosis.measure_match(truth, prediction, home)
    assert record["spells"]["n_gt"] == 5 and record["spells"]["n_pred"] == 3
    assert record["oracle"]["ownership"] >= record["spells"]["spell_f1"]
    summary = diagnosis.summarise_group([record])
    assert summary["offsets"]["stop"]["median"] == 5
    assert summary["stoppages"]["true"]["count"] == 1
