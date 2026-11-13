"""D48 stack trial: GPU share, sign table, selection score and gate (dfl_submission/stack_trial.py)."""

import json

import numpy as np

from dfl_submission import direction_tokens as dt, stack_trial as st


PY = "/opt/homebrew/bin/Python"


def test_trainings_count_every_trainer_once():
    rows = [
        (10, 1, f"{PY} -m dfl_submission.stack_set train --source a --tokens b --out c --seed 42 --boundary-boost 0.5"),
        (11, 1, f"{PY} -m dfl_submission.screen_trial networks --candidate N --set current"),
        (12, 11, f"{PY} -m dfl_submission.player_set3 --source a --tokens b --out c --seed 42 --boundary-boost 0.5"),
        (13, 1, f"{PY} -m dfl_submission.hidden_ball train --source a --seed 97"),
        (14, 1, f"{PY} -m dfl_submission.stack_trial check --name seed --external x"),            # parent, not a training
        (15, 1, f"{PY} -m dfl_submission.hidden_trial networks --set augmented"),                  # exporting
        (16, 1, "/bin/zsh -c .venv/bin/python -m dfl_submission.stack_set train --seed 42"),       # a shell
        (17, 1, f"{PY} -m dfl_submission.stack_trial develop --dfl x"),
    ]
    assert st.trainings(rows) == 4               # 10, 12, 13 and the exporting `networks` run 15


def test_wait_for_gpu_needs_the_d49_file_and_a_free_slot(tmp_path):
    original = st.running_trainings
    try:
        st.running_trainings = lambda: 1
        done = tmp_path / "TRAINING_DONE"
        done.touch()
        st.wait_for_gpu("a test", done=done, poll=0)          # returns at once
        counts = iter([2, 2, 1])
        st.running_trainings = lambda: next(counts)
        st.wait_for_gpu("a test", done=done, poll=0)          # waits until one finishes
    finally:
        st.running_trainings = original


def formation(seed, home_visible=11, flip=False, length=40):
    rng = np.random.default_rng(seed)
    home, away = rng.normal([-20, 0], 5, size=(length, 11, 2)), rng.normal([20, 0], 5, size=(length, 11, 2))
    positions = np.concatenate([home, away], axis=1).astype(np.float32)
    if flip:
        positions[..., 0] *= -1
    positions[:, home_visible:11] = np.nan
    return positions, positions[:, 15].copy(), np.r_[np.zeros(11, int), np.ones(11, int)], np.array([105., 68.])


def test_sign_table_applies_the_fallback_and_keys_by_tracking():
    items = [("pff", ("pff", "1", "observed", 0), *formation(1)),
             ("pff", ("pff", "1", "observed", 1), *formation(2, home_visible=4)),
             ("pff", ("pff", "2", "observed", 0), *formation(3, home_visible=3)),
             ("pff", ("pff", "2", "observed", 1), *formation(4, home_visible=5)),
             ("dfl_clean", ("dfl", "WPY", "clean", 0), *formation(5, flip=True)),
             ("dfl_clean", ("dfl", "WPY", "clean", 1), *formation(6))]
    table = st.sign_table(items)
    used = {row["half"]: row["sign_used"] for row in table.values()}
    assert used == {"pff/1/observed/0": 1, "pff/1/observed/1": -1, "pff/2/observed/0": 1, "pff/2/observed/1": 1,
                    "dfl/WPY/clean/0": -1, "dfl/WPY/clean/1": 1}
    fallbacks = sorted(row["half"] for row in table.values() if row["fallback"])
    assert fallbacks == ["pff/1/observed/1", "pff/2/observed/0", "pff/2/observed/1"]
    assert table[st.tracking_key(*items[4][2:])]["half"] == "dfl/WPY/clean/0"


def test_sign_table_refuses_one_tracking_with_two_signs():
    positions, ball, teams, pitch = formation(7, home_visible=4)
    other_positions, other_ball, _, _ = formation(8, flip=True)
    items = [("a", ("x", "1", "v", 0), positions, ball, teams, pitch),
             ("a", ("x", "1", "v", 1), *formation(9)),
             ("b", ("y", "2", "v", 0), positions, ball, teams, pitch),         # same tracking, other match
             ("b", ("y", "2", "v", 1), other_positions, other_ball, teams, pitch)]
    try:
        st.sign_table(items)
    except ValueError:
        return
    raise AssertionError("sign_table accepted one tracking with two signs")


def entry(sign, first, d, **extra):
    return {"sign": sign, "first_sign": first, "d": d, "agrees": sign is not None and sign == first,
            "max_visible_all_frames": [11, 11]} | extra


def test_summarise():
    token = {"sign": 1}
    good = {"a": entry(1, 1, 20., tokens_equal=True, token=token), "b": entry(None, None, None, tokens_equal=True,
                                                                              token={"sign": None})}
    result = st.summarise(good, True)
    assert result["passed"] and result["fallback_halves"] == ["b"] and result["smallest_abs_d_m"] == 20.
    assert not st.summarise({"a": entry(1, -1, 3.)}, False)["passed"]
    assert not st.summarise({"a": entry(1, 1, 3., tokens_equal=False, token=token)}, True)["passed"]


def item(cost, composite, guard, groups, conventions=(0., 0., 0.)):
    standard = {name: {"standard": {"spell_f1": .7 + value}} for name, value in zip(("aba1.75", "aba4", "aba4.5"), conventions)}
    trial = {"external": {"conventions": {"aba3": {"standard": {"spell_f1": .7}, "dead_carried": .6, "team_swap": .6},
                                          **standard}},
             "pff": {"means": {"aba3": {"spell_f1": groups["pff"]}}}}
    return {"owner_cost": cost, "composite": composite, "guards": {"mean": guard}, "groups": groups, "trial": trial}


def test_selection_score_and_gate():
    base_groups = {name: .7 for name in st.GROUPS}
    base = item(10, .7139, .6748, base_groups)
    assert abs(st.selection(base) - .69435) < 1e-12
    good_groups = dict(base_groups, pff=.72, d24=.72)
    entries = [item(8, .72, .70, good_groups), item(10, .725, .71, good_groups), item(12, .72, .69, good_groups)]
    decision = st.gate(entries, base)
    assert decision["selected_cost"] == 10 and decision["passed"]
    assert abs(decision["changes"]["selection"] - (st.selection(entries[1]) - st.selection(base))) < 1e-12
    # Each condition can fail on its own.
    small_pff = [item(10, .725, .71, dict(good_groups, pff=.71))]
    assert not st.gate(small_pff, base)["conditions"]["pff"]
    harmed = [item(10, .725, .71, dict(good_groups, dfl_clean=.689))]
    assert not st.gate(harmed, base)["conditions"]["groups"]
    convention = [item(10, .725, .71, good_groups, conventions=(0., -.011, 0.))]
    assert not st.gate(convention, base)["conditions"]["conventions"]
    lower = [item(10, .7138, .75, good_groups)]
    decision = st.gate(lower, base)
    assert decision["conditions"]["selection"] and not decision["conditions"]["composite"]
    # Ties go to the lower cost.
    assert st.gate([item(14, .72, .70, good_groups), item(8, .72, .70, good_groups)], base)["selected_cost"] == 8


def test_holdout_long_hidden_seeds_are_new():
    from dfl_submission import hidden_views as hv
    used = set(hv.SEEDS.values())
    assert st.HOLDOUT_GUARD_SEED not in used
    spans = sorted(used | {st.HOLDOUT_GUARD_SEED})
    assert all(b - a >= 10000 for a, b in zip(spans, spans[1:]))     # no overlap with any D49 seed range


def test_data_folders():
    current, tokens = st.data_folders("d42", "current", "/external")
    assert str(current) == "/external/external_temporal/train_data" and tokens.name == "tokens_current"
    augmented, _ = st.data_folders("d42", "augmented", "/external")
    assert augmented.parts[-2:] == ("d13", "sequences_augmented")
    hidden, hidden_tokens = st.data_folders("d49", "augmented", "/external")
    assert hidden.parts[-2:] == ("d49", "sequences_augmented") and hidden_tokens.name == "tokens_augmented"
    try:
        st.data_folders("d50", "current", "/external")
    except ValueError:
        return
    raise AssertionError("data_folders accepted an unknown data folder")


def test_record_levers_and_networks_need_the_checks(tmp_path):
    original = st.RECIPE
    st.RECIPE = tmp_path
    try:
        st.record_levers(note="test", output=tmp_path / "levers.json")
        record = json.loads((tmp_path / "levers.json").read_text())
        assert record["config"] == dict(st.ss.DEFAULT, direction=True) and record["data"] == "d49"
        try:
            st.record_levers(output=tmp_path / "levers.json")
        except FileExistsError:
            pass
        else:
            raise AssertionError("record_levers wrote twice")
        try:
            st.train_networks("/external", scratch=tmp_path)
        except ValueError:
            return
        raise AssertionError("train_networks ran without the pipeline checks")
    finally:
        st.RECIPE = original


def test_summary_prints_every_model_and_the_gate(tmp_path):
    base_groups = {name: .7 for name in st.GROUPS}

    def full(cost, composite, guard, groups):
        result = item(cost, composite, guard, groups)
        result["guards"] |= {"a_dfl": guard - .01, "b_metrica": guard + .01}
        return result

    base = full(10, .7139, .6748, base_groups)
    reference = {"entries": {str(cost): full(cost, .7139, .6748, base_groups) for cost in st.COSTS}}
    good = dict(base_groups, pff=.72)
    entries = {name: [full(cost, .72, .70, good) for cost in st.COSTS] for name in st.MODEL_NAMES}
    development = {"reference": dict(base, selection=st.selection(base)), "entries": entries,
                   "decision": st.gate(entries["stack"], base)}
    (tmp_path / "reference.json").write_text(json.dumps(reference))
    (tmp_path / "development.json").write_text(json.dumps(development))
    text = st.summary(tmp_path)
    table = [line for line in text.splitlines() if line.startswith("| ")]
    assert len(table) == 1 + 4 * (1 + len(st.MODEL_NAMES))
    assert "Selected cost 8" in text and "Gate PASSED" in text


def first_decision(tmp_path, selection=0.7265477, passed=True, cost=14):
    (tmp_path / "development.json").write_text(json.dumps({"decision": {"passed": passed, "selected_cost": cost,
                                                                        "selection_score": selection}}))


def test_t_choice_follows_the_amendment(tmp_path):
    original = st.RECIPE
    st.RECIPE = tmp_path
    try:
        first_decision(tmp_path)
        better = st.t_choice({"passed": True, "selected_cost": 12, "selection_score": .7300})
        assert better["stack"] == "8100" and better["owner_cost"] == 12
        failed = st.t_choice({"passed": False, "selected_cost": 12, "selection_score": .7400})
        assert failed["stack"] == "5400" and failed["owner_cost"] == 14
        lower = st.t_choice({"passed": True, "selected_cost": 10, "selection_score": .7200})
        assert lower["stack"] == "5400" and lower["owner_cost"] == 14
        # Between the rounded 0.7265 and the exact 5,400-step score: the exact score decides, and both are recorded.
        between = st.t_choice({"passed": True, "selected_cost": 14, "selection_score": .72651})
        assert between["stack"] == "5400" and between["selection_at_least_0.7265"] and not between["selection_at_least_5400"]
        first_decision(tmp_path, cost=12)
        try:
            st.t_choice({"passed": True, "selected_cost": 12, "selection_score": .73})
        except ValueError:
            return
        raise AssertionError("t_choice accepted a 5,400-step decision the amendment does not name")
    finally:
        st.RECIPE = original


def test_t_export_reuses_d52_for_offset_zero(tmp_path):
    assert st.t_export(0, "current", "temporal_models") == st.D52_T / "export" / "current" / "temporal_models"
    assert st.t_export(1000, "augmented", "seed_models", tmp_path) == tmp_path / "export" / "1000" / "augmented" / "seed_models"
    assert st.t_raw(0, "augmented", "seed_models") == st.D52_T / "raw" / "augmented" / "seed_models"
    assert st.T_OFFSETS == (1000, 2000) and st.T_STEPS == 8100


def test_model_identity_default_keys_are_unchanged():
    import inspect
    signature = inspect.signature(st.model_identity)
    assert signature.parameters["export"].default is st.export_folder
    assert signature.parameters["levers_name"].default == "levers.json" and signature.parameters["extra"].default is None


def test_confirmation_stack_follows_the_choice(tmp_path):
    original = st.RECIPE
    st.RECIPE = tmp_path
    try:
        first_decision(tmp_path)
        target = st.confirmation_stack()
        assert target["stack"] == "5400" and target["cost"] == 14 and target["levers"] == "levers.json"
        (tmp_path / "development_t.json").write_text(json.dumps({"amendment_choice": {"stack": "5400", "owner_cost": 14}}))
        assert st.confirmation_stack()["stack"] == "5400"
        first_decision(tmp_path, passed=False)
        try:
            st.confirmation_stack()
        except ValueError:
            return
        raise AssertionError("confirmation_stack accepted a stack whose gate did not pass")
    finally:
        st.RECIPE = original


def test_holdout_seeds_are_new_and_distinct():
    from dfl_submission import hidden_views as hv
    seeds = [st.holdout_seed(match, index) for match in ("WQQ", "WR9") for index in (0, 1)]
    assert len(set(seeds)) == 4
    assert not set(seeds) & {hv.guard_dfl_seed(match, index) for match in hv.DFL_ORDER for index in (0, 1)}
    assert all(st.HOLDOUT_GUARD_SEED <= seed < st.HOLDOUT_GUARD_SEED + 10000 for seed in seeds)


def test_long_hidden_scores_are_per_match_n3_spell_f1():
    from dfl_submission import diagnosis, hyper_trial as ht
    from evaluation import possession_metrics as pm
    rng = np.random.default_rng(5)
    recipe = ht.recipe_at(10)
    halves = []
    for match in ("WQQ", "WR9"):
        for index in (0, 1):
            labels = np.where((np.arange(3000) // 250) % 2 == 0, pm.H, pm.A).astype(np.int8)
            labels[1000:1100] = pm.D
            home = np.clip((labels == pm.H) * .8 + rng.uniform(0, .2, 3000), 0, 1).astype(np.float32)
            dead = ((labels == pm.D) * .9 + .05).astype(np.float32)
            edge = np.full(3000, .05, np.float32)
            signals = (home, dead, edge, edge)
            halves.append(diagnosis.HalfSignals("dfl", match, "long_hidden", index, labels, signals,
                                                diagnosis.decode(signals, recipe)))
    result = st.long_hidden_scores(halves, recipe)
    expected = {}
    for key, (truth, prediction, _) in diagnosis.matches(halves, lambda h: diagnosis.decode(h.signals, recipe)).items():
        expected[key[1]] = diagnosis.score(diagnosis.target(truth, 3.0), prediction)["spell_f1"]
    assert result["matches"] == expected and abs(result["mean"] - np.mean(list(expected.values()))) < 1e-12
    assert all(isinstance(value, float) for value in result["matches"].values())


def build_fixture(tmp_path):
    """A recipe folder, a package folder and a models root that pass `confirmed_build`."""
    from dfl_submission import scale_trial
    from dfl_submission.pff import digest
    recipe, package, models = tmp_path / "recipe", tmp_path / "package", tmp_path / "models"
    for folder in (recipe, package, models):
        folder.mkdir(parents=True)
    for name in (*st.BUILD_MODULES, "stack_trial.py"):
        (package / name).write_text(f"# {name}\n")
    config = dict(st.ss.DEFAULT, direction=True, mirror="y")
    levers = {"config": config, "steps": 8100, "offsets": [0, 1000, 2000]}
    (recipe / st.T_LEVERS).write_text(json.dumps(levers))
    networks = {}
    for offset in st.OFFSETS:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                path = st.stack_model_folder(offset, model_set, folder, models)
                path.mkdir(parents=True)
                (path / "fit.json").write_text(json.dumps({"network": [offset, model_set, folder]}))
                networks.setdefault(str(offset), {}).setdefault(model_set, {})[folder] = digest(path / "fit.json")
    identity = {"config": config, "networks": networks, "levers_sha256": digest(recipe / st.T_LEVERS),
                "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in st.SETS},
                "stack_set_sha256": digest(package / "stack_set.py"), "player_set_sha256": digest(package / "player_set.py"),
                "direction_tokens_sha256": digest(package / "direction_tokens.py")}
    confirmation = {"passed": True, "stack": "8100", "owner_cost": 14, "levers": levers,
                    "levers_sha256": digest(recipe / st.T_LEVERS), "identity": identity,
                    "code": {name: digest(package / name) for name in st.BUILD_MODULES}}
    (recipe / "confirmation.json").write_text(json.dumps(confirmation))
    (recipe / st.BUILD_IDENTITY).write_text(json.dumps({"passed": True, "confirmation_sha256": digest(recipe / "confirmation.json"),
                                                        "stack_trial_sha256": digest(package / "stack_trial.py")}))
    return recipe, package, models, confirmation


def refused(confirmation, recipe, package, models):
    try:
        st.confirmed_build(confirmation, models_root=models, recipe=recipe, package=package)
    except st.BuildRefused as error:
        return str(error)
    return None


def test_confirmed_build_passes_and_reads_the_confirmed_levers(tmp_path):
    recipe, package, models, confirmation = build_fixture(tmp_path)
    config, cost = st.confirmed_build(confirmation, models_root=models, recipe=recipe, package=package)
    assert config == confirmation["levers"]["config"] and cost == 14
    assert config["mirror"] == "y" and config["direction"] is True


def test_confirmed_build_refuses_every_identity_difference(tmp_path):
    recipe, package, models, confirmation = build_fixture(tmp_path)
    assert refused(dict(confirmation, passed=False), recipe, package, models).startswith("D48 did not pass")
    other = json.loads(json.dumps(confirmation))
    other["levers"]["config"]["mirror"] = None
    assert "lever record" in refused(other, recipe, package, models)
    (st.stack_model_folder(1000, "augmented", "seed_models", models) / "fit.json").write_text("{}")
    assert "a network differs" in refused(confirmation, recipe, package, models)
    recipe, package, models, confirmation = build_fixture(tmp_path / "trees")
    other = json.loads(json.dumps(confirmation))
    other["identity"]["trees"]["current"] = "0" * 64
    assert "trees" in refused(other, recipe, package, models)
    (package / "player_set3.py").write_text("# changed\n")
    assert "the code differs" in refused(confirmation, recipe, package, models)
    recipe, package, models, confirmation = build_fixture(tmp_path / "lock")
    (package / "stack_trial.py").write_text("# changed driver\n")
    assert "build driver" in refused(confirmation, recipe, package, models)
    (recipe / st.BUILD_IDENTITY).unlink()
    assert "no build identity" in refused(confirmation, recipe, package, models)


def sign_half(seed, home_visible=11, flip=False, early_cross=False):
    positions, ball, teams, pitch = formation(seed, home_visible=home_visible, flip=flip, length=100)
    if early_cross:
        positions[:6, :11, 0] += 60
    return positions, ball, teams, pitch


def test_build_signs_fail_closed_without_numbers():
    assert st.build_signs([sign_half(1), sign_half(2, flip=True)]) == [1, -1]
    cases = {"no sign of its own": [sign_half(1), sign_half(2, home_visible=4, flip=True)],
             "disagrees with its first qualifying frame": [sign_half(1, early_cross=True), sign_half(2, flip=True)],
             "the same sign": [sign_half(1), sign_half(2)],
             "two graded halves": [sign_half(1)]}
    for expected, halves in cases.items():
        try:
            st.build_signs(halves)
        except st.BuildRefused as error:
            assert expected in str(error) and not any(ch.isdigit() for ch in str(error)), str(error)
            continue
        raise AssertionError(f"build_signs accepted: {expected}")


def test_build_table_records_no_fallback():
    halves = [sign_half(1), sign_half(2, flip=True)]
    table = st.build_table(halves)
    assert {row["half"]: (row["sign_used"], row["fallback"]) for row in table.values()} == {"0": (1, None), "1": (-1, None)}
    assert table[st.tracking_key(*halves[0])]["half"] == "0"


def test_build_loads_the_networks_before_reading_the_tracking(tmp_path):
    """`build.tactical_tracking_halves` imports the bundle reader from `experiments/`, which the networks' imports put
    on the module path (as `player_trial build`'s teacher does), so the networks are loaded first."""
    from dfl_submission import build as build_module, kickoff
    calls = []
    saved = [(st, "confirmed_build", st.confirmed_build), (st, "stack_models", st.stack_models), (st, "read", st.read),
             (build_module, "tactical_tracking_halves", build_module.tactical_tracking_halves)]
    try:
        st.confirmed_build = lambda confirmation, **options: (dict(st.ss.DEFAULT, direction=True, mirror="y"), 14)
        st.read = lambda path: {"sign_map": None} if str(path).endswith("parity.json") else {}
        st.stack_models = lambda *args, **options: calls.append("models") or {}

        def halves(**options):
            import broadcast_alignment  # noqa: F401 (the bundle reader must be importable when the tracking is read)
            calls.append("tracking")
            raise RuntimeError("stop after the order is known")
            yield  # pragma: no cover

        build_module.tactical_tracking_halves = halves
        try:
            st.build(tmp_path / "out.csv")
        except RuntimeError:
            pass
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)
    assert calls == ["models", "tracking"]


def tracked(length, flip=False, before=0, mixed=False, seed=11):
    """Synthetic tracking whose first `before` frames have the teams on the wrong sides (or mixed)."""
    positions, ball, teams, pitch = formation(seed, flip=flip, length=length)
    if before:
        if mixed:
            rng = np.random.default_rng(seed + 1)
            positions[:before, :, 0] = rng.uniform(-30, 30, size=positions[:before, :, 0].shape)
        else:
            positions[:before, :, 0] *= -1
    return positions, ball, teams, pitch


def test_kickoff_starts_pair_tags_as_d27_does():
    from dfl_submission import kickoff
    # Halves longer than D27's five-minute window and far apart, as the graded halves are.
    halves = [(1, slice(100, 9100), None), (2, slice(20000, 29000), None)]
    blocks = [(1, 100, 9099), (2, 20000, 28999)]
    tags = [(90, 120), (20050, 20060)]
    starts = st.kickoff_starts(halves, tags, blocks)
    assert starts == [0, 50]
    dead = kickoff.apply(np.full(30000, "H"), tags, blocks)                 # D27's own pairing
    for (half, span, _), start in zip(halves, starts, strict=True):
        assert int((dead[span] == "D").sum()) == start
    cases = {"no single kick-off tag": ([(90, 120)], blocks, halves),
             "no single kick-off tag ": ([(90, 120), (20050, 20060), (20070, 20080)], blocks, halves),
             "no single kick-off tag  ": ([(90, 120), (20000 + kickoff.LIMIT + 1, 20000 + kickoff.LIMIT + 9)], blocks,
                                          halves),
             "does not match its half block": (tags, [(1, 100, 9099), (2, 20000, 29500)], halves),
             "lies after the half": ([(90, 120), (20050, 20060)], [(1, 100, 9099), (2, 20000, 20010)],
                                     [(1, slice(100, 9100), None), (2, slice(20000, 20011), None)])}
    for expected, (case_tags, case_blocks, case_halves) in cases.items():
        try:
            st.kickoff_starts(case_halves, case_tags, case_blocks)
        except st.BuildRefused as error:
            assert expected.strip() in str(error) and not any(ch.isdigit() for ch in str(error)), str(error)
            continue
        raise AssertionError(f"kickoff_starts accepted: {expected}")


def test_kickoff_signs_read_each_half_from_its_kick_off():
    for mixed in (False, True):
        first, second = tracked(200), tracked(200, flip=True, before=30, mixed=mixed, seed=12)
        if not mixed:
            try:
                st.build_signs([first, second])
            except st.BuildRefused as error:
                assert "first qualifying frame" in str(error)
            else:
                raise AssertionError("The whole-half form accepted wrong-side frames before the kick-off")
        assert st.build_signs([first, second], [0, 30]) == [1, -1]
        table = st.build_table([first, second], [0, 30])
        assert {row["half"]: row["sign_used"] for row in table.values()} == {"0": 1, "1": -1}


def test_kickoff_start_zero_is_the_whole_half_bit_for_bit():
    """External halves start at their kick-off (frame 0): for them the kick-off form is the current function."""
    for seed in range(4):
        positions, ball, teams, pitch = tracked(150, flip=bool(seed % 2), seed=20 + seed)
        positions[::7, seed] = np.nan
        whole = dt.attacking(positions, teams)
        assert dt.attacking(positions, teams, st.kickoff_frames(len(positions), 0)) == whole
    halves = [tracked(180, seed=30), tracked(160, flip=True, seed=31)]
    assert st.build_signs(halves) == st.build_signs(halves, [0, 0])
    assert st.build_table(halves) == st.build_table(halves, [0, 0])
    assert np.array_equal(st.kickoff_frames(23, 0), dt.long_context_frames(23))
    assert st.kickoff_frames(23, 7).tolist() == [10, 15, 20]


def test_build_reads_each_sign_from_the_tagged_kick_off(tmp_path):
    """D48's second adoption amendment: the build's sign starts at each half's tag; a half without one is refused."""
    from dfl_submission import build as build_module, kickoff
    first, second = tracked(9000, seed=40), tracked(9000, flip=True, before=30, seed=41)
    halves = [(1, slice(0, 9000), first), (2, slice(20000, 29000), second)]
    try:
        st.build_signs([first, second])
    except st.BuildRefused as error:
        assert "first qualifying frame" in str(error)
    else:
        raise AssertionError("The whole-half form accepted wrong-side frames before the kick-off")
    tables, tag_reads = [], []
    tags = [[(0, 5), (20030, 20035)]]

    class Recorder:
        def __init__(self, models, table, *, owner_cost=None, with_d43=True):
            tables.append(table)

        def forecast(self, positions, ball, teams, pitch):
            return {"stack": (None, np.zeros(len(ball), dtype=np.int8))}

    saved = [(st, name, getattr(st, name)) for name in ("confirmed_build", "stack_models", "read", "Forecaster")]
    saved += [(build_module, name, getattr(build_module, name)) for name in ("tactical_tracking_halves", "source_frames",
                                                                           "write_csv")]
    saved += [(kickoff, name, getattr(kickoff, name)) for name in ("tag_path", "half_blocks", "read_tags", "constrained")]
    try:
        st.confirmed_build = lambda confirmation, **options: (dict(st.ss.DEFAULT, direction=True, mirror="y"), 14)
        st.read = lambda path: {"sign_map": None} if str(path).endswith("parity.json") else {}
        st.stack_models = lambda *args, **options: {}
        st.Forecaster = Recorder
        build_module.tactical_tracking_halves = lambda **options: iter(halves)
        build_module.source_frames = lambda: 30000
        build_module.write_csv = lambda output, labels: None
        kickoff.tag_path = lambda: "tags"
        kickoff.half_blocks = lambda: [(1, 0, 8999), (2, 20000, 28999)]
        kickoff.read_tags = lambda path: tag_reads.append(path) or tags[0]
        kickoff.constrained = lambda labels, path, blocks: labels
        st.build(tmp_path / "on.csv")
        assert tag_reads == ["tags"]
        (table,) = tables
        assert {row["half"]: row["sign_used"] for row in table.values()} == {"0": 1, "1": -1}
        tags[0] = [(0, 5)]
        try:
            st.build(tmp_path / "untagged.csv")
        except st.BuildRefused as error:
            assert "no single kick-off tag" in str(error)
        else:
            raise AssertionError("The build accepted a graded half without its kick-off tag")
        assert len(tables) == 1
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)
