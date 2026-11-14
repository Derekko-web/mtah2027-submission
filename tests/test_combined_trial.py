"""D60's training commands, trainer check, network folders and development gate on synthetic fixtures."""

import json

import numpy as np

from dfl_submission import combined_trial as ct, height_fit as hf, stack_trial as st


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


def test_training_command_is_the_stack_s_trainer_on_d57_s_folders(tmp_path):
    original = hf.prepared
    hf.prepared = lambda model_set, recipe=None, scratch=None: {"passed": True}
    try:
        command = ct.train_command(1000, "augmented", "seed_models", tmp_path / "out")
        reference = st.steps_command("S", "T", tmp_path / "out", *st.NETWORKS["seed_models"],
                                     dict(ct.config(0), seed_offset=1000), "augmented")
    finally:
        hf.prepared = original
    values, expected = dict(zip(command[4::2], command[5::2])), dict(zip(reference[4::2], reference[5::2]))
    assert command[1:4] == ["-m", "dfl_submission.stack_trial", "train-steps"] and list(values) == list(expected)
    assert {key for key in values if values[key] != expected[key]} == {"--source", "--tokens", "--signs"}
    assert values["--source"].endswith("data/d57/sequences_augmented") and values["--tokens"].endswith("data/d57/tokens_augmented")
    assert values["--signs"].endswith("data/d57/signs_augmented.json") and values["--steps"] == "8100"
    assert json.loads(values["--config"])["seed_offset"] == 1000 and values["--sign-map"] == str(st.PARITY)


def test_config_and_folders():
    assert ct.config(2000) == dict(ct.config(0), seed_offset=2000)
    assert ct.export_folder(0, "current", "seed_models") == hf.export_folder("current", "seed_models")
    assert ct.export_folder(1000, "current", "seed_models").parts[-4:] == ("export", "1000", "current", "seed_models")
    assert ct.TRAINED == (1000, 2000) and ct.COSTS == (12, 14, 16) and ct.STEPS == 8100


def test_jobs_put_the_check_first_and_skip_finished_ones(tmp_path):
    original = hf.prepared
    hf.prepared = lambda model_set, recipe=None, scratch=None: {"passed": True}
    try:
        names = [job[0] for job in ct.jobs(tmp_path)]
        assert names[0] == "check" and len(names) == 9
        (tmp_path / "check" / "current" / "temporal_models").mkdir(parents=True)
        (tmp_path / "check" / "current" / "temporal_models" / "fit.json").write_text("{}")
        done = ct.export_folder(1000, "current", "temporal_models", tmp_path)
        done.mkdir(parents=True)
        (done / "fit.json").write_text("{}")
        names = [job[0] for job in ct.jobs(tmp_path)]
        assert names[0] == "1000/current/seed_models" and len(names) == 7
    finally:
        hf.prepared = original


def fit(models, **changes):
    base = {key: 1 for key in st.FIT_KEYS} | {"seed": 42, "models": models}
    return base | changes


def test_check_record_compares_bit_for_bit(tmp_path):
    original = hf.raw_folder
    reference = tmp_path / "sz"
    reference.mkdir()
    (reference / "fit.json").write_text(json.dumps(fit({"temporal_8100.pt": "a"})))
    hf.raw_folder = lambda model_set, folder, scratch=None: reference
    try:
        check = ct.check_folder(tmp_path / "net")
        check.mkdir(parents=True)
        (check / "fit.json").write_text(json.dumps(fit({"temporal_8100.pt": "a"})))
        assert ct.check_record(tmp_path / "net", output=tmp_path / "c1.json")["passed"] is True
        (check / "fit.json").write_text(json.dumps(fit({"temporal_8100.pt": "b"})))
        assert ct.check_record(tmp_path / "net", output=tmp_path / "c2.json")["passed"] is False
        (check / "fit.json").write_text(json.dumps(fit({"temporal_8100.pt": "a"}, steps=5400)))
        record = ct.check_record(tmp_path / "net", output=tmp_path / "c3.json")
        assert record["passed"] is False and record["differing_fit_keys"] == ["steps"]
        assert ct.check_record(tmp_path / "net", output=tmp_path / "c1.json")["passed"] is True   # written once, then read
    finally:
        hf.raw_folder = original


def item(cost, pff, composite=.728, guard=.737, shift=0.0):
    groups = {name: .72 + shift for name in st.GROUPS}
    groups["pff"] = pff
    return {"owner_cost": cost, "composite": composite, "guards": {"mean": guard}, "groups": groups, "primary": .735,
            "selection": (composite + guard) / 2}


def full(values=None):
    values = values or {}
    names = {"aba3": .735, "aba1.75": .70, "aba4": .74, "aba4.5": .74} | values
    return {"trial": {"external": {"conventions": {k: {"standard": {"spell_f1": v}} for k, v in names.items()}}}}


def test_gate_takes_the_highest_pff_cost_and_applies_d60_s_conditions():
    reference = item(14, .6958)
    items = [item(12, .7090, composite=.724), item(14, .7080), item(16, .7090, composite=.7231)]
    result = ct.gate(items, [full(), full(), full()], reference, full())
    assert result["selected_cost"] == 12                               # ties in PFF go to the lower cost
    assert np.isclose(result["changes"]["pff"], .0132) and result["conditions"]["pff"]
    assert result["conditions"]["composite"] and result["passed"]
    low = ct.gate([item(12, .7077)], [full()], reference, full())
    assert not low["conditions"]["pff"]                                # +0.0119 < +0.012
    composite = ct.gate([item(12, .71, composite=.7229)], [full()], reference, full())
    assert not composite["conditions"]["composite"]                    # -0.0051
    guard = ct.gate([item(12, .71, guard=.7319)], [full()], reference, full())
    assert not guard["conditions"]["guard_mean"]
    group = item(12, .71)
    group["groups"]["dfl_team_swap"] = .72 - .0101
    assert not ct.gate([group], [full()], reference, full())["conditions"]["groups"]
    assert not ct.gate([item(12, .71)], [full({"aba4": .74 - .0101})], reference, full())["conditions"]["conventions"]


def test_records_once(tmp_path):
    ct.write_new(tmp_path / "r.json", {"a": 1})
    assert refused(lambda: ct.write_new(tmp_path / "r.json", {}), FileExistsError)
