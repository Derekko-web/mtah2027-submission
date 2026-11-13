"""D53 fallback: the stack without the attacking frame (dfl_submission/fallback_trial.py)."""

import json

import numpy as np
import torch

from dfl_submission import fallback_trial as ft, player_set as ps, player_set3 as p3, stack_set as ss, stack_trial as st


def test_the_candidate_is_the_stack_without_the_attacking_frame():
    assert ft.CONFIG == dict(ss.DEFAULT, mirror="xy") and ft.CONFIG["direction"] is False
    assert ss.mirrors_of(ft.CONFIG) == p3.MIRRORS and ft.STEPS == 8100


def test_training_command_takes_no_signs_and_the_mirror_map():
    command = ft.train_command("source", "tokens", "out", 97, 1.4, 2000)
    assert command[1:4] == ["-m", "dfl_submission.stack_trial", "train-steps"]
    assert "--signs" not in command
    assert command[command.index("--sign-map") + 1] == str(st.PARITY)
    assert command[command.index("--steps") + 1] == "8100"
    assert json.loads(command[command.index("--config") + 1]) == dict(ft.CONFIG, seed_offset=2000)
    assert command[command.index("--seed") + 1] == "97" and command[command.index("--boundary-boost") + 1] == "1.4"


def test_inference_is_d47s_mirror_average_without_a_sign():
    torch.manual_seed(0)
    net = ps.torch_network(4, 8, .2).eval()
    for parameter in net.parameters():
        torch.nn.init.normal_(parameter, std=.3)
    weights = {name: tensor.detach().numpy() for name, tensor in net.state_dict().items()}
    columns = ["h__a", "a__a", "h__b", "a__b"]
    checkpoint = (weights, columns, np.zeros(4, np.float32), np.ones(4, np.float32), np.array([1, 0, 3, 2]))
    rng = np.random.default_rng(1)
    frames = np.arange(0, 100, 5)
    values = rng.normal(size=(len(frames), 4)).astype(np.float32)
    tokens = rng.normal(size=(len(frames), 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    sign_map = {"columns": columns, "signs": {"x": [1] * 4, "y": [1] * 4, "xy": [1] * 4}}
    expected = p3.predict(checkpoint, frames, values, columns, tokens, 100, mirrored=True, sign_map=sign_map)
    result = ss.predict(checkpoint, frames, values, columns, tokens, 100, config=ft.CONFIG, sign_map=sign_map)
    for a, b in zip(expected, result, strict=True):
        np.testing.assert_array_equal(a, b)


def test_unsigned_table_accepts_every_half():
    table = ft.Unsigned()
    row = table.get("any tracking key")
    assert row["sign_used"] == 1 and row["fallback"] is None and table.get("another") == row


def test_paths_variant_and_gate_summary(tmp_path):
    assert ft.export_folder(1000, "current", "seed_models", tmp_path) == tmp_path / "export" / "1000" / "current" / "seed_models"
    assert ft.raw_folder(0, "augmented", "temporal_models", tmp_path) == tmp_path / "raw" / "0" / "augmented" / "temporal_models"
    variant = ft.variant()
    assert variant["levers"] == ft.LEVERS and variant["export"] is ft.export_folder and variant["extra"]["steps"] == 8100
    summary = ft.gate_summary({"passed": True, "selected_cost": 12, "selection_score": .73})
    assert summary == {"gate_passed": True, "selected_cost": 12, "selection_score": .73}


def test_lever_record_is_written_once(tmp_path):
    output = tmp_path / "levers_d53.json"
    ft.record_levers(output)
    record = json.loads(output.read_text())
    assert record["config"] == ft.CONFIG and record["steps"] == 8100 and record["data"] == "d49"
    assert set(record["trainer_checks"]) == set(ft.TRAINER_CHECKS) and "No new trainer check" in record["note"]
    try:
        ft.record_levers(output)
    except FileExistsError:
        return
    raise AssertionError("record_levers wrote twice")
