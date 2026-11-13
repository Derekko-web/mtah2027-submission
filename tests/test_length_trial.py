"""D52 harness: T's trainer call is the stack's, the GPU schedule, D52's rule (dfl_submission/length_trial.py)."""

import argparse
import json
import re
import sys

import numpy as np

from dfl_submission import context_set as cs, length_trial as lt, stack_set as ss, stack_trial as st
from dfl_submission.pff import digest


def folders(tmp_path):
    source, tokens = tmp_path / "prepared", tmp_path / "tokens"
    source.mkdir()
    tokens.mkdir()
    records = [{"file": f"dfl_M{i}_clean_{h}.npz", "source": "dfl"} for i in range(2) for h in (0, 1)]
    (source / "manifest.json").write_text(json.dumps({"columns": ["h__x", "a__x"], "records": records}))
    (tokens / "manifest.json").write_text(json.dumps({"base_manifest": digest(source / "manifest.json"), "records": {}}))
    signs = tmp_path / "signs.json"
    entries = {record["file"]: {"sign": 1 if i % 2 else -1} for i, record in enumerate(records)}
    entries[records[0]["file"]]["sign"] = None                      # D50's fallback: the other half's sign negated
    signs.write_text(json.dumps({"base_manifest": digest(source / "manifest.json"),
                                 "token_manifest": digest(tokens / "manifest.json"), "records": entries}))
    parity = tmp_path / "parity.json"
    parity.write_text(json.dumps({"passed": True, "sign_map": {"columns": ["h__x", "a__x"],
                                                               "signs": {m: [1, 1] for m in ("x", "y", "xy")}}}))
    return source, tokens, signs, parity


class Patch:
    """A minimal monkeypatch (the test runner has no pytest fixtures besides tmp_path)."""

    def __init__(self):
        self.saved = []

    def setattr(self, owner, name, value):
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def undo(self):
        for owner, name, value in reversed(self.saved):
            setattr(owner, name, value)


def captured(monkeypatch, module):
    calls = []
    monkeypatch.setattr(module, "train", lambda *args, **kwargs: calls.append((args, kwargs)))
    return calls


def test_trainer_call_at_5400_steps_is_the_stacks(tmp_path):
    monkeypatch = Patch()
    try:
        source, tokens, signs, parity = folders(tmp_path)
        config = json.dumps(dict(ss.DEFAULT, direction=True, mirror="y"))
        argv = ["train", "--source", str(source), "--tokens", str(tokens), "--out", str(tmp_path / "out"), "--seed", "42",
                "--boundary-boost", "0.5", "--config", config, "--signs", str(signs), "--sign-map", str(parity)]
        stack_calls = captured(monkeypatch, ss)
        monkeypatch.setattr(sys, "argv", ["stack_set", *argv])
        ss.main()
        (stack_args, stack_kwargs), = stack_calls
        for steps, blocks, module in ((5400, 6, ss), (8100, 6, ss), (5400, 7, cs)):
            calls = captured(monkeypatch, module)
            args = lt.parser_for_train(argparse.ArgumentParser()).parse_args(
                argv[1:] + ["--steps", str(steps), "--blocks", str(blocks)])
            lt.train_one(args)
            (call_args, call_kwargs), = calls
            assert call_args == stack_args
            expected = dict(stack_kwargs, steps=steps)
            if module is cs:
                expected.pop("aux_folder")
                expected["dilations"] = cs.SEVEN
            assert call_kwargs == expected
            assert ("steps" in stack_kwargs) is False                   # the stack's call relies on 5,400 by default
    finally:
        monkeypatch.undo()


def test_training_commands_are_counted_and_the_queue_is_not():
    trainer = ("/usr/bin/python3.12 -m dfl_submission.length_trial train --source a --tokens b --out c --seed 42 "
               "--boundary-boost 0.5 --config {} --steps 8100 --blocks 6")
    queue = "/usr/bin/python3.12 -m dfl_submission.length_trial queue --external /data"
    rows = [(10, 1, trainer), (11, 1, queue)]
    assert st.trainings(rows) == 1
    assert st.TRAINER.search(trainer) and not st.TRAINER.search(queue) and not st.IN_PROCESS.search(queue)


def test_schedule_rule():
    assert lt.allowed(0, 0, False) and lt.allowed(1, 0, False)
    assert not lt.allowed(1, 1, False)                                # one of D52's at a time while the stack trains
    assert lt.allowed(1, 1, True) and not lt.allowed(2, 1, True)       # two once it has finished, never three in all
    assert not lt.allowed(2, 0, True)


def test_jobs_order_and_skips_exported(tmp_path):
    order = lt.jobs(scratch=tmp_path)
    assert order[0] == ("T", "current", "temporal_models") and order[-1] == ("X", "augmented", "seed_models")
    assert len(order) == 8 and [job[0] for job in order] == ["T"] * 4 + ["X"] * 4
    done = lt.export_folder("T", "current", "temporal_models", tmp_path)
    done.mkdir(parents=True)
    (done / "fit.json").write_text("{}")
    assert ("T", "current", "temporal_models") not in lt.jobs(scratch=tmp_path)
    assert lt.CANDIDATES == {"T": {"steps": 8100, "blocks": 6}, "X": {"steps": 5400, "blocks": 7}}
    assert lt.COSTS == (8, 10, 12, 14)


def item(cost, composite, guard, **changes):
    groups = {name: .7 + changes.get(name, 0.) for name in st.GROUPS}
    return {"owner_cost": cost, "composite": composite, "guards": {"mean": guard, "a_dfl": guard, "b_metrica": guard},
            "groups": groups, "primary": .73}


def test_rule_holds_and_decision():
    base = [item(8, .72, .70), item(10, .72, .71), item(12, .70, .70), item(14, .70, .69)]   # best: cost 10, 0.715
    good = [item(8, .73, .72, pff=-.004), item(10, .72, .70), item(12, .72, .70), item(14, .72, .70)]
    result = lt.holds(good, base)
    assert result["cost"] == 8 and result["base_cost"] == 10 and result["holds"]
    assert np.isclose(result["changes"]["selection"], (.73 + .72) / 2 - .715)
    short = [item(8, .72, .72)]                                      # selection 0.72 = base + 0.005 < +0.008
    assert not lt.holds(short, base)["conditions"]["selection"]
    pff = [item(8, .73, .73, pff=-.006)]
    assert lt.holds(pff, base)["conditions"] == {"selection": True, "pff": False, "groups": True}
    group = [item(8, .73, .73, metrica=-.011)]
    assert not lt.holds(group, base)["conditions"]["groups"]
    better = [item(8, .74, .74)]
    results = {"T": lt.holds(good, base), "X": lt.holds(better, base)}
    assert lt.decision(results) == "X"
    assert lt.decision({"T": lt.holds(short, base), "X": lt.holds(pff, base)}) is None
    tie = [item(8, .725, .72), item(10, .725, .72)]
    assert lt.best(tie)["owner_cost"] == 8
