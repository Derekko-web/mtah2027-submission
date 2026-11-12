"""D32 on synthetic data: weights, adjustments, eligibility and a tiny continued tree."""

import numpy as np
import pandas as pd

from dfl_submission import vendor_adaptation as va


def test_constants_are_d6s():
    assert va.CONFIDENCE == .90 and va.ROUNDS == 40 and va.REPLAY_WEIGHT == .75
    assert va.MULTIPLIERS == (.25, .5, 1.0) and va.GAIN == .01 and va.MATCH_LOSS == .02 and va.LOSS == .01


def test_weights_give_replay_three_quarters_and_each_pseudo_class_an_eighth():
    replay = np.array([1., 1., 2.])                     # sums to its length after scaling below
    replay = replay * len(replay) / replay.sum()
    pseudo = np.array([0, 1, 1, 1])
    weights = va.student_weights(replay, pseudo)
    total = len(replay) + len(pseudo)
    assert np.isclose(weights[:3].sum(), .75 * total)
    assert np.isclose(weights[3:][pseudo == 0].sum(), .125 * total)
    assert np.isclose(weights[3:][pseudo == 1].sum(), .125 * total)


def test_zero_multiplier_leaves_the_teacher_signals_unchanged():
    signals = tuple(np.linspace(0, 1, 10, dtype=np.float32) for _ in range(4))
    residual = np.full(10, .3, dtype=np.float32)
    assert all(np.array_equal(a, b) for a, b in zip(va.adjusted(signals, residual, residual, 0.0), signals))


def test_adjustment_moves_home_and_stoppage_and_clips():
    home = np.array([.1, .5, .95], dtype=np.float32)
    dead = np.array([.05, .5, .9], dtype=np.float32)
    edges = np.zeros(3, dtype=np.float32)
    out = va.adjusted((home, dead, edges, edges), np.array([-.2, .1, .2]), np.array([.1, -.6, .2]), .5)
    assert np.allclose(out[0], [0., .55, 1.]) and np.allclose(out[1], [.1, .2, 1.])
    assert out[2] is edges and out[3] is edges
    assert va.adjusted((home, dead, edges, edges), None, None, 1.0)[0] is home


def entry(primary, conventions=(0., 0., 0.), worst=0.):
    return {"changes": {"primary": primary, "conventions": dict(zip(("aba1.75", "aba4", "aba4.5"), conventions)),
                        "worst_match": worst}}


def test_eligibility_needs_a_point_no_match_below_two_and_conventions_within_one():
    assert va.eligible(entry(.01, (-.01, -.01, -.01), -.02)["changes"])
    assert not va.eligible(entry(.0099)["changes"])
    assert not va.eligible(entry(.03, worst=-.0201)["changes"])
    assert not va.eligible(entry(.03, (-.0101, 0, 0))["changes"])


def test_selection_takes_the_best_eligible_multiplier():
    entries = {.25: dict(entry(.012), eligible=True, primary=.64, macro=.9),
               .5: dict(entry(.015), eligible=True, primary=.645, macro=.9),
               1.0: dict(entry(.02), eligible=False, primary=.65, macro=.9)}
    assert va.select(entries) == .5
    assert va.select({.25: dict(entry(0), eligible=False, primary=.6, macro=.9)}) is None


def test_continued_tree_starts_from_the_checkpoint():
    from xgboost import XGBClassifier
    rng = np.random.default_rng(0)
    inputs = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    labels = (inputs["a"] > 0).to_numpy().astype(int)
    base = XGBClassifier(n_estimators=5, max_depth=2, random_state=0).fit(inputs, labels)
    student = va.continue_tree(base, inputs, labels, np.ones(400), {"n_estimators": 5, "max_depth": 2, "random_state": 0},
                               rounds=3)
    assert student.get_booster().num_boosted_rounds() == 8
