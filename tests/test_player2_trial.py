"""D44 plumbing: settings, token folders and training commands (dfl_submission/player2_trial.py)."""

from dfl_submission import player2_trial as p


def test_settings_change_one_thing_each():
    settings = p.settings()
    assert [name for name, _ in settings] == ["anchor", "V1", "V2", "V3", "V4"]
    anchor = settings[0][1]
    assert anchor == {"per_team": 11, "hidden": 32, "relative": False, "lr_scale": 2., "batch": 32, "dropout": .2}
    changed = {name: {k for k in anchor if setting[k] != anchor[k]} for name, setting in settings[1:]}
    assert changed == {"V1": {"per_team"}, "V2": {"hidden"}, "V3": {"relative"}, "V4": {"lr_scale", "batch", "dropout"}}


def test_token_folders_reuse_d42s_tokens_when_the_token_options_match():
    settings = dict(p.settings())
    assert p.token_folder(settings["anchor"]) == p.D42_TOKENS == p.token_folder(settings["V2"]) == p.token_folder(settings["V4"])
    assert p.token_folder(settings["V1"]).name == "tokens_current_p16"
    assert p.token_folder(settings["V3"]).name == "tokens_current_p11_rel"


def test_train_command():
    settings = dict(p.settings())
    command = p.train_command("src", "tok", "out", settings["V3"], seed=42, boost=.5)
    assert "--relative" in command and command[command.index("--per-team") + 1] == "11"
    command = p.train_command("src", "tok", "out", settings["V4"], seed=97, boost=1.4)
    assert command[command.index("--batch") + 1] == "16" and command[command.index("--lr-scale") + 1] == "1.0"
    assert "--relative" not in command
