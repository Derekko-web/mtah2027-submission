"""D36 constants and plumbing (dfl_submission/second_trial.py)."""

from dfl_submission import hyper_trial as ht, second_trial as st


def test_setting_is_d34s_combination():
    assert st.SETTING == {"lr_scale": 2., "batch": 32, "dropout": .2}
    assert ht.name(st.SETTING) == "lr_scale2_batch32_dropout0.2"
    assert ht.full(st.SETTING) == {"lr_scale": 2., "batch": 32, "weight_decay": .01, "dropout": .2}


def test_costs_and_folders():
    assert st.OWNER_COSTS == (8, 10, 12, 14)
    assert ht.network_folders(st.SETTING, st.MODELS) == {key: st.MODELS / "lr_scale2_batch32_dropout0.2" / key
                                                         for key in ("current", "augmented")}


def test_train_command_carries_the_three_changes():
    command = ht.train_command("src", "out", st.SETTING, seed=97, boost=1.4)
    for flag, value in (("--lr-scale", "2.0"), ("--batch", "32"), ("--dropout", "0.2"), ("--seed", "97")):
        assert command[command.index(flag) + 1] == value
    assert "--weight-decay" not in command
