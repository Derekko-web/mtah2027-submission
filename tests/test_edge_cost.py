"""D63's two-pass decode (dfl_submission/edge_cost.py)."""

import numpy as np

from dfl_submission import edge_cost as ec, joint_decoder
from evaluation import possession_metrics as pm


RECIPE = {"cutoff": .53, "dead_cost": 2, "owner_cost": 14, "discount": .8}
H, A, D = pm.H, pm.A, pm.D


def random_signals(seed, length):
    rng = np.random.default_rng(seed)
    # Slowly varying probabilities, so the decode has spells and stoppages of every length.
    def smooth(scale):
        walk = np.cumsum(rng.normal(0, scale, length))
        return (1 / (1 + np.exp(-walk))).astype(np.float32)
    home, dead = smooth(.35), smooth(.25)
    dead_edge = (rng.random(length) < .02).astype(np.float32) * rng.random(length).astype(np.float32)
    owner_edge = (rng.random(length) < .03).astype(np.float32) * rng.random(length).astype(np.float32)
    return home, dead, dead_edge, owner_edge


def standing(signals, recipe=RECIPE):
    return joint_decoder.decode(*signals, cutoff=recipe["cutoff"], dead_cost=recipe["dead_cost"],
                                owner_cost=recipe["owner_cost"], discount=recipe["discount"])


def test_pass_one_equals_the_standing_decode_on_random_inputs():
    recipes = [RECIPE, dict(RECIPE, owner_cost=8), dict(RECIPE, cutoff=.4, dead_cost=3, discount=.5)]
    for seed in range(12):
        for length in (5, 7, 333, 2001, 4003):
            signals = random_signals(seed, length)
            for recipe in recipes:
                expected = standing(signals, recipe)
                assert np.array_equal(ec.to_frames(ec.pass_one(signals, recipe), length), expected)
                emissions, dead_steps, owner_steps = ec.recipe_terms(signals, recipe)
                assert emissions.shape == (len(range(0, length, 5)), 3)
                assert owner_steps.dtype == np.float32 and dead_steps.dtype == np.float32


def test_both_factors_at_one_give_pass_one_exactly():
    for seed in range(12):
        signals = random_signals(100 + seed, 3001)
        first, second, scale = ec.two_pass_steps(signals, RECIPE, 1.0, 1.0)
        assert np.array_equal(first, second) and np.all(scale == 1)
        assert np.array_equal(ec.two_pass(signals, RECIPE, 1.0, 1.0), standing(signals))


def test_windows_of_a_half_live_throughout():
    labels = np.full(40, H, dtype=np.int8)
    assert ec.stops_and_restarts(labels) == ([40], [0])
    before, after = ec.windows(labels)
    assert np.flatnonzero(before).tolist() == list(range(25, 40))
    assert np.flatnonzero(after).tolist() == list(range(1, 16))


def test_windows_when_the_half_starts_and_ends_dead():
    labels = np.array([D] * 3 + [H] * 30 + [D] * 4, dtype=np.int8)
    stops, restarts = ec.stops_and_restarts(labels)
    assert stops == [0, 33] and restarts == [3]
    before, after = ec.windows(labels)
    assert np.flatnonzero(before).tolist() == list(range(18, 33))       # t = 0 has no steps before it
    assert np.flatnonzero(after).tolist() == list(range(4, 19))


def test_windows_clip_and_overlap():
    labels = np.array([D] * 5 + [A] * 10 + [D] * 5 + [H] * 30, dtype=np.int8)   # 50 steps, ends live
    stops, restarts = ec.stops_and_restarts(labels)
    assert stops == [0, 15, 50] and restarts == [5, 20]
    before, after = ec.windows(labels)
    assert np.flatnonzero(before).tolist() == list(range(0, 15)) + list(range(35, 50))
    assert np.flatnonzero(after).tolist() == list(range(6, 36))
    scale = ec.factors(labels, .5, .25)
    expected = np.ones(50)
    expected[0:15] = .5
    expected[35:50] = .5
    expected[6:36] = .25                                                    # the smaller factor where both apply
    assert np.array_equal(scale, expected)
    scale = ec.factors(labels, .25, .75)
    expected = np.ones(50)
    expected[6:36] = .75
    expected[0:15] = .25
    expected[35:50] = .25
    assert np.array_equal(scale, expected)


def test_windows_of_short_and_empty_halves():
    assert ec.stops_and_restarts(np.array([], dtype=np.int8)) == ([], [])
    before, after = ec.windows(np.array([A, A, A], dtype=np.int8))
    assert before.tolist() == [True, True, True] and after.tolist() == [False, True, True]
    before, after = ec.windows(np.array([D, D], dtype=np.int8))
    assert not before.any() and not after.any()
    labels = np.array([H] * 3 + [D] + [A] * 2 + [D] + [H] * 40, dtype=np.int8)
    stops, restarts = ec.stops_and_restarts(labels)
    assert stops == [3, 6, 47] and restarts == [0, 4, 7]


def short_spell_before_a_stop():
    """300 frames clearly home, 70 frames (2.8 s, 14 steps) mildly away, then 50 frames of stoppage.

    The live frames are clearly not a stoppage (0.001), so the standing decode cannot change team through a one-step
    stoppage either (restart 2 + stoppage 2 + about 7 of stoppage evidence)."""
    home = np.r_[np.full(300, .9), np.full(70, .33), np.full(50, .5)].astype(np.float32)
    dead = np.r_[np.full(370, .001), np.full(50, .95)].astype(np.float32)
    zeros = np.zeros(420, dtype=np.float32)
    return home, dead, zeros, zeros


def test_a_short_opposing_spell_before_a_stop_is_recovered_only_at_a_low_factor():
    signals = short_spell_before_a_stop()
    standing_labels = standing(signals)
    assert not (standing_labels == A).any() and not (standing_labels[:365] == D).any()
    assert (standing_labels[380:] == D).all()
    assert np.array_equal(ec.two_pass(signals, RECIPE, 1.0, 1.0), standing_labels)
    assert np.array_equal(ec.two_pass(signals, RECIPE, 1.0, .25), standing_labels)   # not after a restart
    assert np.array_equal(ec.two_pass(signals, RECIPE, .75, 1.0), standing_labels)   # 10.5 still above its evidence
    for rho in (.5, .25):
        labels = ec.two_pass(signals, RECIPE, rho, 1.0)
        away = np.flatnonzero(labels == A)
        assert 55 <= len(away) <= 75 and away.min() >= 290 and away.max() < 380
        assert np.array_equal(labels[away.max() + 1:] == D, np.ones(len(labels) - away.max() - 1, dtype=bool))
        first, second, scale = ec.two_pass_steps(signals, RECIPE, rho, 1.0)
        assert np.array_equal(first, standing_labels[::5])
        changed = np.flatnonzero(first != second)
        assert scale[changed].max() == rho


def test_pass_two_changes_only_the_ownership_cost_inside_the_windows():
    signals = random_signals(7, 2500)
    emissions, dead_steps, owner_steps = ec.recipe_terms(signals, RECIPE)
    first = ec.viterbi(emissions, dead_steps, owner_steps)
    before, after = ec.windows(first)
    scale = ec.factors(first, .5, .25)
    scaled = owner_steps * scale.astype(owner_steps.dtype)
    assert scaled.dtype == owner_steps.dtype
    outside = ~(before | after)
    assert np.array_equal(scaled[outside], owner_steps[outside])
    assert np.array_equal(scaled[before & ~after], owner_steps[before & ~after] * np.float32(.5))
    assert np.array_equal(scaled[after], owner_steps[after] * np.float32(.25))
    _, second, _ = ec.two_pass_steps(signals, RECIPE, .5, .25)
    assert np.array_equal(second, ec.viterbi(emissions, dead_steps, scaled))


def test_grid():
    assert ec.FACTORS == (1.0, .75, .5, .25) and ec.WINDOW == 15
