"""Round 8's switchable build (DECISIONS.md C3) with the model and data steps replaced by stand-ins."""

import numpy as np

from dfl_submission import broadcast_trial, kickoff, metrica_trial, round8


BLOCKS = [(1, 10, 9999), (2, 12000, 19999)]   # halves further apart than D27's 5 minutes


def patched(test):
    """Run `test(calls)` with kickoff's data access and the round 8 models replaced by recorders."""
    calls = []
    labels = np.full(20010, "X", dtype="<U1")
    labels[10:10000] = "H"
    labels[12000:20000] = "A"
    originals = (kickoff.d15_labels, kickoff.model_labels, kickoff.tag_path, kickoff.half_blocks, kickoff.read_tags,
                 kickoff.read_instances, metrica_trial.confirmed_teacher, broadcast_trial.confirmed_merger,
                 round8.d15_model)
    kickoff.d15_labels = lambda prepare=None: calls.append(("d15",)) or labels.copy()
    kickoff.model_labels = lambda model, pitch, prepare=None, merge_ball=None: (
        calls.append(("model", model, merge_ball)) or labels.copy())
    kickoff.tag_path = lambda: "tags.xml"
    kickoff.half_blocks = lambda: BLOCKS
    kickoff.read_tags = lambda path: [(12100, 12105)]
    kickoff.read_instances = lambda path: [{"code": "goals", "type": None, "start": 40, "end": 50},
                                           {"code": "set pieces", "type": "kick off (after goal)", "start": 70, "end": 75}]
    metrica_trial.confirmed_teacher = lambda: "metrica model"
    broadcast_trial.confirmed_merger = lambda: "merge hook"
    round8.d15_model = lambda: "d15 model"
    try:
        return test(calls)
    finally:
        (kickoff.d15_labels, kickoff.model_labels, kickoff.tag_path, kickoff.half_blocks, kickoff.read_tags,
         kickoff.read_instances, metrica_trial.confirmed_teacher, broadcast_trial.confirmed_merger,
         round8.d15_model) = originals


def test_all_switches_off_is_exactly_the_d27_build():
    def test(calls):
        result = round8.labels()
        assert calls == [("d15",)]
        expected = kickoff.apply(kickoff.d15_labels(), [(12100, 12105)], BLOCKS)
        assert np.array_equal(result, expected)
        assert (result[12000:12100] == "D").all() and (result[10:12000] != "D").all()
    patched(test)


def test_goal_switch_applies_d29_after_d27():
    def test(calls):
        result = round8.labels(goals=True)
        assert (result[50:70] == "D").all() and (result[12000:12100] == "D").all()
        assert (result[10:50] == "H").all() and (result[70:10000] == "H").all()
    patched(test)


def test_metrica_and_broadcast_switches_choose_the_model_and_the_ball_merge():
    def test(calls):
        round8.labels(metrica=True)
        round8.labels(broadcast=True)
        round8.labels(metrica=True, broadcast=True)
        assert calls == [("model", "metrica model", None), ("model", "d15 model", "merge hook"),
                         ("model", "metrica model", "merge hook")]
    patched(test)


def test_the_switches_are_the_four_round_8_steps():
    assert round8.SWITCHES == ("metrica", "broadcast", "adapt", "goals")
