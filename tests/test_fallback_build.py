"""D53 fallback build path: identity checks, no attacking sign, `player_trial build`'s labels with every switch off.

Kept apart from `test_fallback_trial.py` (which loads torch): the build path loads D3's trees (XGBoost), and the two
OpenMP runtimes must not share a process.
"""

import json

import numpy as np

from dfl_submission import fallback_trial as ft, stack_set as ss, stack_trial as st


def fallback_fixture(tmp_path):
    """A recipe folder, a package folder and a models root that pass `fallback_trial.confirmed_build`."""
    from dfl_submission import scale_trial
    from dfl_submission.pff import digest
    recipe, package, models = tmp_path / "recipe", tmp_path / "package", tmp_path / "models"
    for folder in (recipe, package, models):
        folder.mkdir(parents=True)
    for name in (*ft.BUILD_MODULES, "fallback_trial.py"):
        (package / name).write_text(f"# {name}\n")
    levers = {"config": dict(ft.CONFIG), "steps": 8100, "offsets": [0, 1000, 2000]}
    (recipe / ft.LEVERS).write_text(json.dumps(levers))
    networks = {}
    for offset in st.OFFSETS:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                path = st.stack_model_folder(offset, model_set, folder, models)
                path.mkdir(parents=True)
                (path / "fit.json").write_text(json.dumps({"network": [offset, model_set, folder]}))
                networks.setdefault(str(offset), {}).setdefault(model_set, {})[folder] = digest(path / "fit.json")
    identity = {"config": dict(ft.CONFIG), "networks": networks, "levers_sha256": digest(recipe / ft.LEVERS),
                "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in st.SETS},
                "stack_set_sha256": digest(package / "stack_set.py"), "player_set_sha256": digest(package / "player_set.py"),
                "direction_tokens_sha256": digest(package / "direction_tokens.py")}
    confirmation = {"passed": True, "owner_cost": 14, "levers": levers, "levers_sha256": digest(recipe / ft.LEVERS),
                    "identity": identity, "code": {name: digest(package / name) for name in ft.BUILD_MODULES}}
    (recipe / "confirmation_d53.json").write_text(json.dumps(confirmation))
    (recipe / ft.BUILD_IDENTITY).write_text(json.dumps({"passed": True,
                                                        "confirmation_sha256": digest(recipe / "confirmation_d53.json"),
                                                        "fallback_trial_sha256": digest(package / "fallback_trial.py")}))
    return recipe, package, models, confirmation


def refusal(confirmation, recipe, package, models):
    try:
        ft.confirmed_build(confirmation, models_root=models, recipe=recipe, package=package)
    except st.BuildRefused as error:
        return str(error)
    return None


def test_fallback_build_identity_checks(tmp_path):
    recipe, package, models, confirmation = fallback_fixture(tmp_path)
    config, cost = ft.confirmed_build(confirmation, models_root=models, recipe=recipe, package=package)
    assert config == ft.CONFIG and cost == 14
    framed = json.loads(json.dumps(confirmation))
    framed["levers"]["config"]["direction"] = True
    framed["levers"]["config"]["mirror"] = "y"
    assert "no attacking frame" in refusal(framed, recipe, package, models)
    other = json.loads(json.dumps(confirmation))
    other["levers"]["config"]["mirror"] = None
    assert "lever record" in refusal(other, recipe, package, models)
    (package / "stack_trial.py").write_text("# the forecaster changed\n")
    assert "the code differs" in refusal(confirmation, recipe, package, models)
    recipe, package, models, confirmation = fallback_fixture(tmp_path / "net")
    (st.stack_model_folder(2000, "current", "temporal_models", models) / "fit.json").write_text("{}")
    assert "a network differs" in refusal(confirmation, recipe, package, models)
    recipe, package, models, confirmation = fallback_fixture(tmp_path / "lock")
    (package / "fallback_trial.py").write_text("# changed driver\n")
    assert "build driver" in refusal(confirmation, recipe, package, models)
    (recipe / ft.BUILD_IDENTITY).unlink()
    assert "no build identity" in refusal(confirmation, recipe, package, models)
    assert refusal(dict(confirmation, passed=False), recipe, package, models).startswith("D53 did not pass")


def synthetic_half(length, seed, flip=False):
    rng = np.random.default_rng(seed)
    teams = np.array([0] * 11 + [1] * 11)
    start = np.zeros((22, 2))
    start[:11, 0], start[11:, 0] = rng.uniform(-40, -2, 11), rng.uniform(2, 40, 11)
    start[:, 1] = rng.uniform(-30, 30, 22)
    positions = start[None] + np.cumsum(rng.normal(0, .12, size=(length, 22, 2)), axis=0)
    positions[rng.uniform(size=(length, 22)) < .03] = np.nan
    if flip:
        positions[..., 0] *= -1
    holder = np.repeat(rng.integers(0, 22, size=length // 100 + 1), 100)[:length]
    ball = positions[np.arange(length), holder] + .5
    ball[300:420] = np.nan
    return positions, ball, teams, np.array([105., 68.])


def run_both(tmp_path, config, *, refuse_signs=True):
    """(labels of `player_trial build`, labels of `fallback_trial build`, tables) on a synthetic two-half match."""
    from dfl_submission import build as build_module, direction_tokens as dt, kickoff, player_trial
    halves = [(slice(30, 2030), synthetic_half(2000, 1)), (slice(2100, 3900), synthetic_half(1800, 2, flip=True))]
    written, tables = [], []
    real_forecaster = st.Forecaster

    class Recording(real_forecaster):
        def __init__(self, models, table, **options):
            tables.append(table)
            super().__init__(models, table, **options)

    saved = [(build_module, name, getattr(build_module, name)) for name in ("tactical_tracking_halves", "source_frames",
                                                                           "write_csv")]
    saved += [(kickoff, name, getattr(kickoff, name)) for name in ("tag_path", "half_blocks", "constrained")]
    saved += [(ft, "confirmed_build", ft.confirmed_build), (st, "stack_models", st.stack_models),
              (st, "Forecaster", st.Forecaster), (dt, "attacking", dt.attacking)]
    real_read = st.read
    saved += [(st, "read", st.read), (ft, "read", ft.read), (player_trial, "json", player_trial.json)]
    try:
        build_module.tactical_tracking_halves = lambda **options: ((index + 1, span, tracking)
                                                                   for index, (span, tracking) in enumerate(halves))
        build_module.source_frames = lambda: 3950
        build_module.write_csv = lambda output, labels: written.append(np.asarray(labels).copy())
        kickoff.tag_path = lambda: "tags"
        kickoff.half_blocks = lambda: "blocks"
        kickoff.constrained = lambda labels, tags, blocks: labels
        ft.confirmed_build = lambda confirmation, **options: (dict(config), 10)
        ft.read = lambda path: {} if str(path).endswith("confirmation_d53.json") else real_read(path)
        st.stack_models = lambda *args, **options: {"stack": st.d43_model()}
        st.Forecaster = Recording
        if refuse_signs:
            def no_sign(*args, **options):
                raise AssertionError("An attacking sign was computed")
            dt.attacking = no_sign
        player_trial.build(tmp_path / "standing.csv")
        ft.build(tmp_path / "fallback.csv")
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)
    return written[0], written[1], tables


def test_fallback_build_with_switches_off_equals_player_trial_build_and_computes_no_sign(tmp_path):
    standing, fallback, tables = run_both(tmp_path, ss.DEFAULT)
    assert standing.shape == (3950,) and set(standing[:30]) == {"X"} and np.array_equal(standing, fallback)
    (table,) = tables
    assert isinstance(table, ft.Unsigned)
