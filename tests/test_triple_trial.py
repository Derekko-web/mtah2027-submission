"""D22 plumbing on synthetic data: slot means, Stage A scoring and selection, slot folders, D15 parity."""

import numpy as np

from dfl_submission import retrain_trial, triple_trial, zoo
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


class FixedNetwork(zoo.Checkpoint):
    """A checkpoint whose four outputs are fixed arrays, for testing the slot mean."""

    def __init__(self, outputs):
        self.outputs = tuple(np.asarray(value, dtype=np.float32) for value in outputs)

    def predict(self, features):
        return self.outputs


def outputs(seed, length=40):
    rng = np.random.default_rng(seed)
    return tuple(rng.uniform(size=length).astype(np.float32) for _ in range(4))


def test_mean_of_one_structure_is_unchanged_and_of_three_is_summed_in_order():
    a, b, c = outputs(1), outputs(2), outputs(3)
    assert retrain_trial.mean([a]) is a
    result = retrain_trial.mean([a, b, c])
    for index in range(4):
        expected = ((np.array(a[index], copy=True) + b[index]) + c[index]) / np.float32(3)
        assert result[index].dtype == np.float32 and np.array_equal(result[index], expected.astype(np.float32))
    nested = retrain_trial.mean([{"x": (a[0], None)}, {"x": (b[0], None)}])
    assert np.array_equal(nested["x"][0], ((a[0] + b[0]) / np.float32(2)).astype(np.float32))


def test_slot_of_one_network_returns_its_outputs_unchanged():
    single = FixedNetwork(outputs(4))
    features = (np.arange(0, 40, 5), np.zeros((8, 2), dtype=np.float32), ["a", "b"], 40)
    assert all(np.array_equal(x, y) for x, y in zip(zoo.predict(zoo.SlotMean([single]), features), single.outputs))
    three = zoo.SlotMean([single, FixedNetwork(outputs(5)), FixedNetwork(outputs(6))])
    averaged = zoo.predict(three, features)
    expected = [((x + y) + z) / np.float32(3) for x, y, z in zip(outputs(4), outputs(5), outputs(6))]
    assert all(np.array_equal(x, y.astype(np.float32)) for x, y in zip(averaged, expected))
    try:
        zoo.SlotMean([])
    except ValueError:
        pass
    else:
        raise AssertionError("An empty slot must be refused")


def fake_stage_a_trial(primary, macro=.5):
    return {"standard": {"aba3": {"spell_f1": primary, "macro_f1": macro}}}


def stage_a_entries(values):
    return [{"point": point, "trial": fake_stage_a_trial(value)}
            for point, value in zip(triple_trial.stage_a_points(), values)]


def test_stage_a_grid_is_the_rule():
    points = triple_trial.stage_a_points()
    assert len(points) == 18 and points[0] == triple_trial.ANCHOR
    assert points[:3] == [{"width": 96, "steps": 5400, "owner_cost": c} for c in (8, 10, 12)]
    assert points[-1] == {"width": 192, "steps": 19800, "owner_cost": 12}


def test_stage_a_replaces_the_anchor_only_by_at_least_half_a_point():
    values = [.70] + [.70] * 17
    values[4] = .7049  # width 96, 10,800 steps, cost 10
    best, gain, replaces, standing = triple_trial.stage_a_select(stage_a_entries(values))
    assert best == {"width": 96, "steps": 10800, "owner_cost": 10} and not replaces and standing == triple_trial.ANCHOR
    values[4] = .7050
    best, gain, replaces, standing = triple_trial.stage_a_select(stage_a_entries(values))
    assert replaces and standing == best and abs(gain - .005) < 1e-12
    values = [.70] * 18
    best, _, replaces, standing = triple_trial.stage_a_select(stage_a_entries(values))
    assert best == triple_trial.ANCHOR and not replaces  # ties go to grid order


def half(source, match, view, index, truth, home):
    signals = (np.asarray(home, dtype=np.float32), np.where(truth == D, .9, .1).astype(np.float32),
               np.zeros(len(truth), dtype=np.float32), np.zeros(len(truth), dtype=np.float32))
    from dfl_submission import diagnosis
    return diagnosis.HalfSignals(source, match, view, index, truth, signals, None)


def test_stage_a_scores_the_standard_aggregate_of_decoded_halves():
    truth = np.r_[np.full(400, H), np.full(150, D), np.full(400, A)].astype(np.int8)
    good = np.where(truth == H, .9, .1)
    bad = np.full(len(truth), .9)
    halves = []
    for match in ("WMX", "WN1"):
        for view, home in (("clean", good), ("masked", bad)):
            halves += [half("dfl", match, view, i, truth, home) for i in (0, 1)]
    halves += [half("skillcorner", "7", "observed", i, truth, good) for i in (0, 1)]
    recipe = dict(triple_trial.d15_recipe())
    trial = triple_trial.score_point(halves, recipe)
    clean = trial["matches"]["dfl|WMX|clean"]["aba3"]["spell_f1"]
    masked = trial["matches"]["dfl|WMX|masked"]["aba3"]["spell_f1"]
    skill = trial["matches"]["skillcorner|7|observed"]["aba3"]["spell_f1"]
    assert clean > masked
    assert abs(trial["standard"]["aba3"]["spell_f1"] - ((clean + 2 * masked) / 3 + skill) / 2) < 1e-12


def test_cv_record_check_needs_the_recorded_run_and_losses(tmp_path):
    run = tmp_path / "fold0"
    run.mkdir()
    (run / "fit.json").write_text("{}")
    from dfl_submission.pff import digest
    setting = {"steps": 5400, "width": 96}
    losses = {("dfl", "WMX"): {"ownership": .25, "stoppage": .03}}
    record = {"entries": [{"setting": setting, "runs": {"0": {"fit_sha256": digest(run / "fit.json")}},
                           "matches": {"dfl:WMX": {"ownership": .25, "stoppage": .03},
                                       "dfl:WN1": {"ownership": .3, "stoppage": .04}}}]}
    triple_trial.check_cv_record(record, setting, 0, run, losses)
    for broken in ({("dfl", "WMX"): {"ownership": .2500001, "stoppage": .03}},):
        try:
            triple_trial.check_cv_record(record, setting, 0, run, broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Losses that differ from D20's record must be refused")
    (run / "fit.json").write_text('{"other": 1}')
    try:
        triple_trial.check_cv_record(record, setting, 0, run, losses)
    except ValueError:
        pass
    else:
        raise AssertionError("A run other than the recorded one must be refused")


def test_slot_folders_put_d15s_network_first_at_its_own_setting(tmp_path):
    from dfl_submission import scale_trial
    anchor = triple_trial.slot_folders("augmented", triple_trial.ANCHOR, tmp_path)
    published = scale_trial.network_folders(zoo.d15_setting()[0])["augmented"]
    assert anchor["temporal_models"][0] == published / "temporal_models"
    assert [folder.name for folder in anchor["temporal_models"][1:]] == ["temporal_models_seed1042", "temporal_models_seed2042"]
    assert [folder.name for folder in anchor["seed_models"][1:]] == ["seed_models_seed1097", "seed_models_seed2097"]
    other = triple_trial.slot_folders("current", {"width": 96, "steps": 10800, "owner_cost": 8}, tmp_path)
    assert [folder.name for folder in other["seed_models"]] == ["seed_models_seed97", "seed_models_seed1097",
                                                                "seed_models_seed2097"]
    assert all(folder.parent == tmp_path / "w96_s10800" / "current" for folder in other["temporal_models"])


def synthetic_tracking(seed, length=1500):
    rng = np.random.default_rng(seed)
    start = rng.uniform([-45, -30], [45, 30], size=(22, 2))
    positions = (start + np.cumsum(rng.normal(0, .15, size=(length, 22, 2)), axis=0)).astype(np.float32)
    ball = (np.cumsum(rng.normal(0, .5, size=(length, 2)), axis=0) % 40 - 20).astype(np.float32)
    ball[400:430] = np.nan
    return positions, ball, np.repeat([0, 1], 11), np.array([105., 68.])


def test_single_network_slots_reproduce_d15_exactly():
    from dfl_submission import diagnosis, scale_trial
    tracking = synthetic_tracking(9)
    ours = zoo.half_signals(triple_trial.teacher(triple_trial.ANCHOR, single=True), *tracking)
    model = scale_trial.teacher(zoo.d15_setting()[0])
    reference, packages, frozen = model
    theirs = diagnosis.half_signals((reference, packages, dict(frozen, recipe=triple_trial.d15_recipe())), *tracking)
    assert np.array_equal(ours[1], theirs[1])
    assert all(np.array_equal(a, b) for a, b in zip(ours[0], theirs[0]))
