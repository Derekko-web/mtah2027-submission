"""D38 plumbing: the anchor, the B1 grid and the commands (dfl_submission/fourth_trial.py)."""

from dfl_submission import fourth_trial as ft


def test_anchor_is_d34s_combination():
    assert ft.ANCHOR == {"lr_scale": 2., "batch": 32, "weight_decay": .01, "dropout": .2, "width": 96, "steps": 5400}
    assert ft.anchor_runs()[0].parent.name == "lr_scale2_batch32_dropout0.2"


def test_b1_changes_one_value_each():
    settings = ft.settings()
    assert len(settings) == 5
    for setting, change in zip(settings, ft.B1):
        differing = {key for key in ft.ANCHOR if setting[key] != ft.ANCHOR[key]}
        assert differing == set(change)


def test_names_are_unique_and_commands_carry_every_value():
    names = [ft.name(s) for s in ft.settings()]
    assert len(set(names)) == 5 and ft.name(ft.ANCHOR) == "lr2_b32_d0.2_wd0.01_w96_s5400"
    command = ft.train_command("src", "out", ft.settings()[0], seed=42, boost=.5)
    assert command[command.index("--steps") + 1] == "10800" and command[command.index("--batch") + 1] == "32"
    assert command[command.index("--width") + 1] == "96" and command[command.index("--lr-scale") + 1] == "2.0"


def test_stage_b_grid():
    assert ft.OWNER_MIXES == (.5, 1.) and ft.OWNER_COSTS == (6, 8, 10, 12)
    assert ft.mixes(1.) == {"owner_mix": 1., "dead_mix": .5}
