"""D56's candidate helpers on synthetic fixtures: the selection score, the rule, the training commands, D55's view scores."""

import json

import numpy as np

from dfl_submission import diagnosis, situation_fit as f56, situation_trial as st55, stack_trial as st
from evaluation import possession_metrics as pm


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


def entry(cost, composite, guard, d, pff=.68, shift=0.0, a=.7):
    groups = {name: .7 + shift for name in f56.GROUPS}
    groups["pff"] = pff
    item = {"owner_cost": cost, "composite": composite, "guards": {"mean": guard}, "situation_d": d, "situation_a": a,
            "groups": groups, "primary": .73}
    return dict(item, selection_d56=f56.selection(item))


def test_selection_is_the_mean_of_composite_guard_mean_and_the_d_view():
    item = entry(10, .72, .70, .74)
    assert np.isclose(item["selection_d56"], (.72 + .70 + .74) / 3)


def test_holds_applies_d56_s_three_conditions_at_the_best_costs():
    base = [entry(c, .72, .70, .74 - .001 * i, pff=.69) for i, c in enumerate(f56.COSTS)]   # base best: cost 8
    good = [entry(8, .72, .70, .74), entry(10, .73, .72, .77, pff=.69)]                      # +0.02 / 3 * ... at cost 10
    result = f56.holds(good, base)
    assert result["cost"] == 10 and result["base_cost"] == 8
    assert np.isclose(result["changes"]["selection"], (.73 + .72 + .77 - .72 - .70 - .74) / 3)
    assert result["conditions"] == {"selection": True, "pff": True, "groups": True} and result["holds"]
    low_pff = [entry(10, .73, .72, .77, pff=.6899)]
    assert not f56.holds(low_pff, base)["conditions"]["pff"]
    group_loss = [dict(entry(10, .73, .72, .77, pff=.69))]
    group_loss[0]["groups"] = dict(group_loss[0]["groups"], dfl_team_swap=.7 - .0101)
    assert not f56.holds(group_loss, base)["conditions"]["groups"]
    small = [entry(10, .722, .702, .742, pff=.69)]
    assert not f56.holds(small, base)["conditions"]["selection"]
    ties = [entry(8, .73, .72, .77, pff=.69), entry(12, .73, .72, .77, pff=.69)]
    assert f56.holds(ties, base)["cost"] == 8
    results = {"S": dict(f56.holds(good, base), selection=.75), "SL": dict(f56.holds(good, base), selection=.76)}
    assert f56.decision(results) == "SL"
    assert f56.decision({"S": dict(results["S"], holds=False)}) is None


def test_training_command_is_length_trial_on_d56_s_folders(tmp_path):
    original = f56.prepared
    f56.prepared = lambda model_set, recipe=None: {"passed": True}
    try:
        command = f56.command_for("SL", "augmented", "seed_models", scratch=tmp_path)
    finally:
        f56.prepared = original
    assert command[1:4] == ["-m", "dfl_submission.length_trial", "train"]
    values = dict(zip(command[4::2], command[5::2]))
    assert values["--source"].endswith("data/d56/sequences_augmented") and values["--tokens"].endswith("data/d56/tokens_augmented")
    assert values["--signs"].endswith("data/d56/signs_augmented.json") and values["--sign-map"] == str(st.PARITY)
    assert values["--steps"] == "10800" and values["--blocks"] == "6"
    assert (values["--seed"], values["--boundary-boost"]) == tuple(str(v) for v in st.NETWORKS["seed_models"])
    assert json.loads(values["--config"]) == dict(st.levers()["config"], seed_offset=0)
    assert values["--out"] == str(tmp_path / "SL" / "raw" / "augmented" / "seed_models")
    assert len(f56.jobs(tmp_path)) == 8 and f56.jobs(tmp_path)[0] == ("S", "current", "temporal_models")
    done = f56.export_folder("S", "current", "temporal_models", tmp_path)
    done.mkdir(parents=True)
    (done / "fit.json").write_text("{}")
    assert len(f56.jobs(tmp_path)) == 7


def test_prepared_refuses_folders_that_changed(tmp_path):
    record = tmp_path / "d56_prepare_current.json"
    record.write_text(json.dumps({"passed": True, "sequences_manifest": "0" * 64, "tokens_manifest": "0" * 64,
                                  "signs_file": "0" * 64}))
    assert refused(lambda: f56.prepared("current", recipe=tmp_path))


class FakeHalf:
    def __init__(self, view, index, labels):
        self.view, self.index, self.signals = view, index, (labels,)


def test_situation_scores_decode_both_views_at_the_cost():
    original = diagnosis.decode
    seen = []

    def decode(signals, recipe):
        seen.append(recipe["owner_cost"])
        return signals[0]

    diagnosis.decode = decode
    try:
        labels = np.repeat(np.array([pm.H, pm.A, pm.D, pm.H], dtype=np.int8), 300)
        wrong = labels.copy()
        wrong[100:200] = pm.A
        tracking = {(name, i): (labels, None, None, None, None) for name in f56.VIEWS.values() for i in (0, 1)}
        halves = [FakeHalf("air_or_crowd", 1, labels), FakeHalf("air_or_crowd", 0, labels),
                  FakeHalf("air", 0, wrong), FakeHalf("air", 1, labels)]
        scores = f56.situation_scores(halves, tracking, 12)
    finally:
        diagnosis.decode = original
    assert set(seen) == {12}
    assert scores["situation_d"] == 1.0
    _, target = st55.truth_of(labels, labels)
    expected = diagnosis.score(target, np.r_[wrong, pm.X, labels].astype(np.int8))["spell_f1"]
    assert scores["situation_a"] == float(expected) < 1.0
    assert refused(lambda: f56.situation_scores(halves[:1], tracking, 12), ValueError)


def test_records_are_written_once(tmp_path):
    exists = tmp_path / "exists.json"
    exists.write_text("{}")
    assert refused(lambda: f56.base(None, None, None, output=exists), FileExistsError)
    assert refused(lambda: f56.write_new(exists, {}), FileExistsError)
    assert f56.CANDIDATES == {"S": {"steps": 8100}, "SL": {"steps": 10800}} and f56.COSTS == (8, 10, 12, 14)
    assert (f56.SELECTION_MARGIN, f56.GROUP_LOSS) == (.008, .01)
    assert f56.VIEWS == {"situation_d": "air_or_crowd", "situation_a": "air"}
