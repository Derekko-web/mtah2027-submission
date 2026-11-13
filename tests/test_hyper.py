"""D34: the hyperparameter trainer's defaults and the Stage A plumbing (hyper_network, hyper_trial)."""

import numpy as np
import torch

from dfl_submission import hyper_network as hn, hyper_trial as ht, temporal_network as tn
from evaluation import possession_metrics as pm


def groups():
    rng = np.random.default_rng(0)
    labels = np.concatenate([np.full(200, pm.H), np.full(150, pm.A), np.full(50, pm.D)]).astype(np.int8)
    make = lambda: (rng.normal(size=(400, 6)).astype(np.float32), tn.targets(labels))
    return {"dfl": [make(), make()], "skillcorner": [make()]}


def test_batch_of_sixteen_draws_exactly_temporal_networks_windows():
    data = groups()
    permutation = np.array([1, 0, 2, 3, 5, 4])
    location, scale = np.zeros(6, np.float32), np.ones(6, np.float32)
    a = tn.batch(data, np.random.default_rng(5), permutation, location, scale)
    b = hn.batch(data, np.random.default_rng(5), permutation, location, scale, 16)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    big = hn.batch(data, np.random.default_rng(5), permutation, location, scale, 32)
    assert big[0].shape[0] == 32 and big[1].shape[0] == 32


def test_learning_rate_scale():
    for step in (1, 900, 5400):
        assert hn.learning_rate(step, 5400) == tn.learning_rate(step, 5400)
        assert hn.learning_rate(step, 5400, 2.) == 2 * tn.learning_rate(step, 5400)


def test_options_record_only_changed_values():
    assert hn.options(lr_scale=1., batch=16, weight_decay=.01, dropout=.08) == {}
    assert hn.options(lr_scale=2., batch=16, weight_decay=.01, dropout=0.) == {"lr_scale": 2., "dropout": 0.}


def test_names_and_commands():
    assert [ht.name(s) for s in ht.a1_settings()] == ["lr_scale2", "lr_scale0.5", "batch32", "batch64",
                                                      "weight_decay0.1", "dropout0.2", "dropout0"]
    assert ht.name({}) == "default" and ht.name({"batch": 32, "lr_scale": 2.}) == "lr_scale2_batch32"
    command = ht.train_command("src", "out", {"batch": 64, "dropout": 0.}, seed=42, boost=.5)
    assert command[command.index("--batch") + 1] == "64" and command[command.index("--dropout") + 1] == "0.0"
    assert "--lr-scale" not in command and "--weight-decay" not in command


def trial(value, macro=.9):
    return {"standard": {"aba3": {"spell_f1": value, "macro_f1": macro}}}


def test_combination_needs_two_hyperparameters_beating_the_anchor():
    entries = [{"setting": {"lr_scale": 2.}, "trials": {8: trial(.715), 10: trial(.712)}},
               {"setting": {"lr_scale": .5}, "trials": {8: trial(.716)}},
               {"setting": {"batch": 32}, "trials": {8: trial(.70)}},
               {"setting": {"dropout": .2}, "trials": {12: trial(.7142)}}]
    assert ht.combination(entries) == {"lr_scale": .5, "dropout": .2}
    assert ht.combination(entries[:3]) is None


def test_stage_a_selection_and_bar():
    entries = [{"setting": {"lr_scale": 2.}, "trials": {8: trial(.72), 10: trial(.7225, .9), 12: trial(.7225, .95)}},
               {"setting": {"batch": 32}, "trials": {8: trial(.7225, .95)}}]
    selected = ht.stage_a_select(entries)
    assert selected["setting"] == {"lr_scale": 2.} and selected["owner_cost"] == 12
    assert abs(selected["gain_over_anchor_best"] - (.7225 - ht.ANCHOR_BEST)) < 1e-12
    assert selected["gain_over_anchor_best"] >= ht.STAGE_A_GAIN
