"""D57's candidate helpers on synthetic fixtures: the training command, the queue's limits, the air view's scoring, the decision."""

import json

import numpy as np

from dfl_submission import diagnosis, height_fit as hf, situation_fit as sf, stack_trial as st
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


def test_training_command_is_s_s_with_d57_s_folders(tmp_path):
    original = sf.prepared
    sf.prepared = lambda model_set, recipe=None: {"passed": True}
    try:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                s_command = sf.command_for("S", model_set, folder, scratch=tmp_path / "d56")
                command = hf.command_for(model_set, folder, scratch=tmp_path / "d57")
                values, s_values = dict(zip(command[4::2], command[5::2])), dict(zip(s_command[4::2], s_command[5::2]))
                assert command[:4] == s_command[:4] and list(values) == list(s_values)
                differing = {key for key in values if values[key] != s_values[key]}
                assert differing == {"--source", "--tokens", "--signs", "--out"}
                assert values["--source"].endswith(f"data/d57/sequences_{model_set}")
                assert values["--tokens"].endswith(f"data/d57/tokens_{model_set}")
                assert values["--signs"].endswith(f"data/d57/signs_{model_set}.json")
                assert values["--out"] == str(tmp_path / "d57" / "SZ" / "raw" / model_set / folder)
                assert values["--steps"] == "8100" and values["--blocks"] == "6"
    finally:
        sf.prepared = original


def test_jobs_and_own_limit(tmp_path):
    assert hf.jobs(tmp_path) == [(s, f) for s in st.SETS for f in st.NETWORKS] and len(hf.jobs(tmp_path)) == 4
    done = hf.export_folder("current", "temporal_models", tmp_path)
    done.mkdir(parents=True)
    (done / "fit.json").write_text("{}")
    assert hf.jobs(tmp_path)[0] == ("current", "seed_models") and len(hf.jobs(tmp_path)) == 3
    assert hf.own_limit(tmp_path) == 1
    (tmp_path / "d56_decision.json").write_text("{}")
    assert hf.own_limit(tmp_path) == 2


class FakeProcess:
    live = []

    def __init__(self, command, **_):
        self.command, self.calls = command, 0
        FakeProcess.live.append(self)

    def poll(self):
        self.calls += 1
        if self.calls < 3:
            return None
        if self in FakeProcess.live:
            FakeProcess.live.remove(self)
        return 0


def test_queue_keeps_to_the_machine_and_own_limits(tmp_path):
    import subprocess
    originals = (subprocess.Popen, subprocess.run, hf.prepared, hf.command_for)
    exports, peaks, outside, own = [], [], {"count": 1}, {"limit": 1}
    starts = []
    subprocess.Popen = lambda command, **k: (starts.append((len(FakeProcess.live), own["limit"], outside["count"])),
                                             FakeProcess(command))[1]
    subprocess.run = lambda command, **_: exports.append(command)
    hf.prepared = lambda model_set: {"passed": True}
    hf.command_for = lambda model_set, folder, scratch: ["train", model_set, folder]

    def running():
        peaks.append(len(FakeProcess.live))
        return outside["count"] + len(FakeProcess.live)

    def sleep(_):
        if len(exports) >= 1:
            own["limit"], outside["count"] = 2, 0         # D56's decision is written and the other trainings end

    FakeProcess.live = []
    try:
        hf.queue(scratch=tmp_path / "networks", running_trainings=running, own=lambda: own["limit"], sleep=sleep)
    finally:
        subprocess.Popen, subprocess.run, hf.prepared, hf.command_for = originals
    assert len(exports) == 4 and all(command[1:4] == ["-m", "dfl_submission.player_set", "export"] for command in exports)
    assert len(starts) == 4 and all(before < limit and before + other < 2 for before, limit, other in starts)
    assert starts[0] == (0, 1, 1) and max(peaks) <= 2
    assert (tmp_path / "train_SZ_current_temporal_models.log").exists()


def test_queue_waits_while_two_trainings_run(tmp_path):
    import subprocess
    originals = (subprocess.Popen, hf.prepared)
    subprocess.Popen = lambda *a, **k: (_ for _ in ()).throw(AssertionError("started while two trainings run"))
    hf.prepared = lambda model_set: {"passed": True}
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] > 3:
            raise StopIteration

    try:
        assert refused(lambda: hf.queue(scratch=tmp_path / "networks", running_trainings=lambda: 2, own=lambda: 2,
                                        sleep=sleep), StopIteration)
    finally:
        subprocess.Popen, hf.prepared = originals


class Half:
    def __init__(self, source, match, index, labels, predicted):
        self.source, self.match, self.view, self.index = source, match, "observed", index
        self.labels, self.signals = labels, (predicted, None, None, None)

    @property
    def key(self):
        return self.source, self.match, self.view


def test_match_scores_score_the_skillcorner_matches_as_the_group_does():
    original = diagnosis.decode
    seen = []

    def decode(signals, recipe):
        seen.append(recipe["owner_cost"])
        return signals[0]

    labels = np.repeat(np.array([pm.H, pm.A, pm.D, pm.H, pm.A], dtype=np.int8), 400)
    wrong = labels.copy()
    wrong[150:420] = pm.A
    halves = [Half("skillcorner", "2", 0, labels, wrong), Half("skillcorner", "2", 1, labels, labels),
              Half("skillcorner", "1", 0, labels, labels), Half("skillcorner", "1", 1, labels, labels),
              Half("pff", "9", 0, labels, wrong), Half("pff", "9", 1, labels, wrong)]
    diagnosis.decode = decode
    try:
        scores = hf.match_scores(halves, {"owner_cost": 12})
    finally:
        diagnosis.decode = original
    assert set(seen) == {12} and list(scores) == ["2", "1"]         # the halves' order, PFF left out
    truth = np.r_[labels, pm.X, labels].astype(np.int8)
    expected = diagnosis.score(diagnosis.target(truth, 3.0), np.r_[wrong, pm.X, labels].astype(np.int8))["spell_f1"]
    assert scores["2"] == float(expected) < 1.0 and scores["1"] == 1.0
    assert refused(lambda: hf.match_scores(halves[:1], {"owner_cost": 8}), ValueError)


def entry(cost, composite, guard, d, pff=.68):
    groups = {name: .7 for name in hf.GROUPS}
    groups["pff"] = pff
    item = {"owner_cost": cost, "composite": composite, "guards": {"mean": guard}, "situation_d": d, "situation_a": .7,
            "groups": groups, "primary": .73}
    return dict(item, selection_d56=sf.selection(item))


def write_records(folder, candidates):
    base = [entry(c, .72, .70, .74, pff=.69) for c in hf.COSTS]
    (folder / "d56_base.json").write_text(json.dumps({"entries": base, "passed": True}))
    base_sha = digest(folder / "d56_base.json")
    for name, entries in candidates.items():
        rule = sf.holds(entries, base)
        prefix = "d57" if name == "SZ" else "d56"
        (folder / f"{prefix}_{name}.json").write_text(json.dumps({"entries": entries, "rule": rule, "base_sha256": base_sha}))
    (folder / "d56_decision.json").write_text(json.dumps({"taken": None}))


def test_decide_takes_the_highest_holding_selection(tmp_path):
    write_records(tmp_path, {"S": [entry(10, .73, .72, .77, pff=.69)], "SL": [entry(10, .722, .705, .745, pff=.69)],
                             "SZ": [entry(12, .74, .73, .78, pff=.70)]})
    assert hf.decide(recipe=tmp_path) == "SZ"
    record = json.loads((tmp_path / "d57_decision.json").read_text())
    assert record["results"]["SL"]["holds"] is False and record["results"]["S"]["holds"] is True
    assert refused(lambda: hf.decide(recipe=tmp_path), FileExistsError)


def test_decide_with_a_low_pff_and_none_holding(tmp_path):
    write_records(tmp_path, {"S": [entry(10, .73, .72, .77, pff=.69)], "SL": [entry(10, .70, .70, .74)],
                             "SZ": [entry(12, .74, .73, .78, pff=.6899)]})
    assert hf.decide(recipe=tmp_path) == "S"
    other = tmp_path / "second"
    other.mkdir()
    write_records(other, {"S": [entry(10, .70, .70, .74)], "SL": [entry(10, .70, .70, .74)], "SZ": [entry(10, .70, .70, .74)]})
    assert hf.decide(recipe=other) is None


def test_decide_refuses_until_d56_s_records_exist_and_on_inconsistent_records(tmp_path):
    write_records(tmp_path, {"S": [entry(10, .73, .72, .77, pff=.69)], "SL": [entry(10, .70, .70, .74)],
                             "SZ": [entry(12, .74, .73, .78, pff=.70)]})
    (tmp_path / "d56_decision.json").unlink()
    assert refused(lambda: hf.decide(recipe=tmp_path), SystemExit)
    (tmp_path / "d56_decision.json").write_text(json.dumps({"taken": "S"}))
    record = json.loads((tmp_path / "d57_SZ.json").read_text())
    record["rule"]["holds"] = not record["rule"]["holds"]
    (tmp_path / "d57_SZ.json").write_text(json.dumps(record))
    assert refused(lambda: hf.decide(recipe=tmp_path), ValueError)
    record["rule"]["holds"] = not record["rule"]["holds"]
    record["base_sha256"] = "0" * 64
    (tmp_path / "d57_SZ.json").write_text(json.dumps(record))
    assert refused(lambda: hf.decide(recipe=tmp_path), ValueError)


def test_prepared_refuses_folders_that_changed(tmp_path):
    (tmp_path / "d57_prepare_current.json").write_text(json.dumps({"passed": True, "sequences_manifest": "0" * 64,
                                                                   "tokens_manifest": "0" * 64, "signs_file": "0" * 64}))
    assert refused(lambda: hf.prepared("current", recipe=tmp_path, scratch=tmp_path))


def test_records_are_written_once(tmp_path):
    hf.write_new(tmp_path / "r.json", {"a": 1})
    assert refused(lambda: hf.write_new(tmp_path / "r.json", {"a": 2}), FileExistsError)
    assert hf.NAME == "SZ" and hf.STEPS == sf.CANDIDATES["S"]["steps"] and hf.COSTS == (8, 10, 12, 14)
