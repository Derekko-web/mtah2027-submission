"""Review of the stack's competition build path (`stack_trial build`) against `player_trial build`, on synthetic matches.

No competition file is read: the bundle readers, the kick-off tag reader and the CSV writer are replaced
by stand-ins, and the tracking is synthetic. D43's published networks and D3's trees (external models in
the repository) stand in for the stack's networks where only the inference path is under test.
"""

import json

import numpy as np

from dfl_submission import build as build_module, direction_tokens as dt, kickoff, player_set3 as p3, player_trial
from dfl_submission import retrain_trial, stack_set as ss, stack_trial as st


class Patch:
    """A minimal monkeypatch (the test runner provides no pytest fixtures besides tmp_path)."""

    def __init__(self):
        self.saved = []

    def setattr(self, owner, name, value):
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def undo(self):
        for owner, name, value in reversed(self.saved):
            setattr(owner, name, value)


def synthetic_half(length=2400, seed=0, few_players=False):
    """Centred tracking (positions, ball, teams, pitch) as `build.tactical_tracking_halves` yields it."""
    rng = np.random.default_rng(seed)
    teams = np.array([0] * 11 + [1] * 11 + [0, 1])                 # two slots that are never visible
    count = len(teams)
    start = np.zeros((count, 2))
    start[:11, 0], start[11:22, 0] = rng.uniform(-40, -2, 11), rng.uniform(2, 40, 11)   # home towards +x
    start[:22, 1] = rng.uniform(-30, 30, 22)
    positions = start[None] + np.cumsum(rng.normal(0, .12, size=(length, count, 2)), axis=0)
    positions[:, 22:] = np.nan
    positions[rng.uniform(size=(length, count)) < .03] = np.nan
    if few_players:
        positions[:, 6:11] = np.nan                                 # six visible home players at most
    holder = np.repeat(rng.integers(0, 22, size=length // 100 + 1), 100)[:length]
    ball = positions[np.arange(length), holder] + .5
    ball[300:420] = np.nan
    ball[1500:1530] = np.nan
    return positions, ball, teams, np.array([105., 68.])


def mirrored(half, axes):
    positions, ball, teams, pitch = half
    positions, ball = positions.copy(), ball.copy()
    for axis in axes:
        positions[..., axis] *= -1
        ball[..., axis] *= -1
    return positions, ball, teams, pitch


def same(first, second):
    signals_a, labels_a = first
    signals_b, labels_b = second
    return np.array_equal(labels_a, labels_b) and all(np.array_equal(a, b) for a, b in zip(signals_a, signals_b, strict=True))


def test_switches_off_the_stack_forecaster_is_player_trials_half_signals():
    half = synthetic_half()
    standing = player_trial.half_signals(player_trial.teacher(10), *half)
    forecaster = st.Forecaster({"stack": st.d43_model()}, {st.tracking_key(*half): {"sign_used": 1}}, owner_cost=10,
                               with_d43=False)
    assert same(forecaster.forecast(*half)["stack"], standing)


def run_builds(halves, frames, patch, tmp_path, *, stack_models, config, forecaster=None):
    """(labels written by `player_trial build`, labels written by `stack_trial build`) on synthetic halves."""
    written, constrained = [], []

    def tracking_halves(*, centered=False, pitch=None, merge_ball=None):
        assert centered is True and np.array_equal(pitch, [105., 68.]) and merge_ball is None
        for index, (span, tracking) in enumerate(halves):
            yield index + 1, span, tracking

    def constrain(labels, tags, blocks):
        constrained.append((tags, blocks))
        return labels

    patch.setattr(build_module, "tactical_tracking_halves", tracking_halves)
    patch.setattr(build_module, "source_frames", lambda: frames)
    patch.setattr(build_module, "write_csv", lambda output, labels: written.append(np.asarray(labels).copy()))
    patch.setattr(kickoff, "tag_path", lambda: "tags")
    # D48's second adoption amendment: the stack's build reads each half's kick-off tag; here each half is tagged at its
    # first frame (as an external half is), where the kick-off form equals the whole-half rule bit for bit.
    blocks = [(index + 1, span.start, span.stop - 1) for index, (span, _) in enumerate(halves)]
    patch.setattr(kickoff, "half_blocks", lambda: blocks)
    patch.setattr(kickoff, "read_tags", lambda path: [(span.start, span.start + 5) for span, _ in halves])
    patch.setattr(kickoff, "LIMIT", 25)       # the synthetic halves are far shorter than D27's five-minute window
    patch.setattr(kickoff, "constrained", constrain)
    real_read = st.read
    patch.setattr(st, "read", lambda path: {"passed": True, "owner_cost": 10, "stack": "test"}
                  if str(path).endswith("confirmation.json") else real_read(path))
    patch.setattr(st, "levers", lambda: {"config": config})
    # D48's adoption amendment: the build reads its switches and cost only through the identity checks, which these
    # synthetic runs stand in for (they are tested in tests/test_stack_trial.py).
    patch.setattr(st, "confirmed_build", lambda confirmation, **options: (dict(config), 10))
    patch.setattr(st, "stack_models", stack_models)
    if forecaster is not None:
        patch.setattr(st, "Forecaster", forecaster)
    player_trial.build(tmp_path / "standing.csv")
    st.build(tmp_path / "stack.csv")
    assert constrained == [("tags", blocks)] * 2
    return written


def test_build_with_switches_off_writes_player_trial_builds_labels(tmp_path):
    first, second = synthetic_half(2000, 1), mirrored(synthetic_half(1800, 2), (0,))     # teams change ends
    halves = [(slice(30, 2030), first), (slice(2100, 3900), second)]
    patch = Patch()
    try:
        standing, stack = run_builds(halves, 3950, patch, tmp_path, config=dict(ss.DEFAULT),
                                     stack_models=lambda config, root, **options: {"stack": st.d43_model()})
    finally:
        patch.undo()
    assert standing.shape == (3950,) and set(standing[:30]) == {"X"} and set(standing[3900:]) == {"X"}
    assert set(standing[30:2030]) <= {"H", "A", "D"} and np.array_equal(standing, stack)


def stack_like(config, sign_map):
    """D43's networks as each of the three seeds of every slot, with the stack's inference switches."""
    folders = player_trial.network_folders()
    return st.stack_models(config, None, export=lambda offset, key, folder, root: folders[key] / folder,
                           sign_map=sign_map, names=("stack",))


def test_attacking_frame_and_y_mirror_make_the_forecast_independent_of_orientation():
    config = ss.config_of(direction=True, mirror="y")
    sign_map = json.loads(st.PARITY.read_text())["sign_map"]
    models = stack_like(config, sign_map)
    half = synthetic_half(2200, 3)
    variants = {name: mirrored(half, axes) for name, axes in (("plain", ()), ("x", (0,)), ("y", (1,)), ("xy", (0, 1)))}
    signs = {name: dt.attacking(tracking[0], tracking[2])["sign"] for name, tracking in variants.items()}
    assert signs == {"plain": 1, "x": -1, "y": 1, "xy": -1}           # home attacks +x in the plain half
    table = {st.tracking_key(*tracking): {"sign_used": signs[name]} for name, tracking in variants.items()}
    forecaster = st.Forecaster(models, table, owner_cost=14, with_d43=False)
    results = {name: forecaster.forecast(*tracking)["stack"] for name, tracking in variants.items()}
    # A half whose home team attacks towards -x gives exactly what its mirror image towards +x gives.
    assert all(same(results["plain"], results[name]) for name in ("x", "y", "xy"))
    # Without the attacking frame the x mirror is visible to the networks (the test can tell the difference).
    plain = ss.config_of(mirror="y")
    unsigned = st.Forecaster(stack_like(plain, sign_map), table, owner_cost=14, with_d43=False)
    assert not same(unsigned.forecast(*variants["plain"])["stack"], unsigned.forecast(*variants["x"])["stack"])


def test_training_swap_and_mirror_equal_inferences_order():
    rng = np.random.default_rng(4)
    tokens = rng.normal(size=(50, 2, 11, 8)).astype(np.float32)
    tokens[:, 1, 7:] = np.nan
    for negate in (True, False):
        trained = dt.swapped(tokens, negate) * p3.TOKEN_SIGNS["y"]          # `stack_set.batch`: swap, then mirror
        inferred = dt.swapped(tokens * p3.TOKEN_SIGNS["y"], negate)         # `stack_set.predict`: mirror, then swap
        np.testing.assert_array_equal(trained, inferred)
    flipped = dt.swapped(tokens, True)
    finite = np.isfinite(tokens[:, ::-1])
    np.testing.assert_array_equal(flipped[..., [0, 3, 5]][finite[..., [0, 3, 5]]],
                                  -tokens[:, ::-1][..., [0, 3, 5]][finite[..., [0, 3, 5]]])
    np.testing.assert_array_equal(flipped[..., [1, 2, 4, 6, 7]][finite[..., [1, 2, 4, 6, 7]]],
                                  tokens[:, ::-1][..., [1, 2, 4, 6, 7]][finite[..., [1, 2, 4, 6, 7]]])
    assert np.flatnonzero(p3.TOKEN_SIGNS["y"] < 0).tolist() == [1, 4, 6]   # the y mirror: y offset, velocity, position


def test_build_refuses_a_half_without_a_sign_of_its_own(tmp_path):
    """D48's adoption amendment (fail closed): a half with six visible home players at most has no sign, so the build
    refuses before any forecast; the review's original version of this test checked the fallback this replaces."""
    first = synthetic_half(1600, 5)
    second = mirrored(synthetic_half(1600, 6, few_players=True), (0,))
    assert dt.attacking(first[0], first[2])["sign"] == 1 and dt.attacking(second[0], second[2])["sign"] is None
    halves = [(slice(0, 1600), first), (slice(1650, 3250), second)]
    tables = []

    class Recorder:
        def __init__(self, models, table, *, owner_cost=None, with_d43=True):
            tables.append((table, owner_cost, with_d43))

        def forecast(self, positions, ball, teams, pitch):
            return {"stack": (None, np.zeros(len(ball), dtype=np.int8))}

    patch = Patch()
    try:
        run_builds(halves, 3300, patch, tmp_path, config=ss.config_of(direction=True, mirror="y"),
                   stack_models=lambda config, root, **options: {"stack": st.d43_model()}, forecaster=Recorder)
    except st.BuildRefused as error:
        assert "no sign of its own" in str(error)
    else:
        raise AssertionError("The build accepted a half without a sign of its own")
    finally:
        patch.undo()
    assert tables == []


def test_three_seeds_are_averaged_per_slot_in_offset_order_before_the_mixes():
    loaded = []

    class Fake:
        def __init__(self, folder, config, sign_map=None):
            self.key, self.config = str(folder), config
            loaded.append(self.key)

        def predict(self, frames, owner, player_tokens, length, sign):
            rng = np.random.default_rng(abs(hash((self.key, len(frames)))) % 2 ** 32)
            return tuple(rng.uniform(.1, .9, length).astype(np.float32) for _ in range(4))

    captured = []
    patch = Patch()
    real_mix = retrain_trial.mix
    try:
        patch.setattr(st, "Network", Fake)
        patch.setattr(retrain_trial, "mix", lambda *values: captured.append(values) or real_mix(*values))
        models = st.stack_models(ss.DEFAULT, "root", export=lambda o, k, f, r: f"{r}/{o}/{k}/{f}", names=("stack",))
        slot = models["stack"]["current"]["temporal_models"]
        assert [net.key for net in slot] == ["root/0/current/temporal_models", "root/1000/current/temporal_models",
                                             "root/2000/current/temporal_models"]
        half = synthetic_half(1500, 7)
        forecaster = st.Forecaster(models, {st.tracking_key(*half): {"sign_used": 1}}, owner_cost=10, with_d43=False)
        forecaster.forecast(*half)
    finally:
        patch.undo()
    current, candidate, weight = captured[0]                       # the top-level call (mix recurses)
    from dfl_submission import ensemble_trial, long_context, zoo
    frames, _, _ = long_context.features(half[0], half[1], half[2], half[3])
    expected = retrain_trial.mean([net.predict(frames, None, None, len(half[1]), 1) for net in slot])
    original = current[0][1][ensemble_trial.LATE]
    assert all(np.array_equal(a, b) for a, b in zip(original, expected, strict=True))
    reversed_order = retrain_trial.mean([net.predict(frames, None, None, len(half[1]), 1) for net in slot[::-1]])
    assert not all(np.array_equal(a, b) for a, b in zip(original, reversed_order, strict=True))   # the order is fixed
    assert zoo.SlotMean.predict.__code__.co_names.count("mean") == 1                              # SlotMean: the same mean
