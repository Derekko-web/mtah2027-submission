"""D12 decoder refinements on synthetic inputs."""

from unittest.mock import patch

import numpy as np

from dfl_submission import joint_decoder, refine_trial
from dfl_submission.conventions import collapse_brief_changes
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X
RECIPE = {"cutoff": .53, "dead_cost": 2, "owner_cost": 6, "discount": .8}


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def block_constant(rng, blocks, low, high):
    return np.repeat(rng.uniform(low, high, blocks), joint_decoder.STRIDE)


def test_one_frame_decoder_equals_five_frame_decoder_on_block_constant_evidence():
    rng = np.random.default_rng(3)
    for _ in range(20):
        blocks = 400
        # Constant boundary signals keep every switch cost constant. With costs that vary by block, a
        # 1-frame switch can sit one frame inside the cheaper block, so equality holds only here.
        edges = rng.uniform(0, 1, 2)
        signals = (block_constant(rng, blocks, .02, .98), block_constant(rng, blocks, .05, .95),
                   np.full(5 * blocks, edges[0]), np.full(5 * blocks, edges[1]))
        # Without the 17-frame smoothing the evidence itself is constant inside each 5-frame block.
        with patch.object(joint_decoder, "centered_mean", lambda values, *_: values):
            coarse = joint_decoder.decode(*signals, **RECIPE)
            fine = joint_decoder.decode(*signals, **RECIPE, step=1)
        assert np.array_equal(coarse, fine)
        assert len(np.unique(coarse)) > 1


def test_one_frame_stoppages_last_at_least_five_frames():
    rng = np.random.default_rng(5)
    signals = tuple(rng.uniform(.01, .99, 3000) for _ in range(4))
    fine = joint_decoder.decode(*signals, **RECIPE, step=1)
    edges = np.flatnonzero(np.diff(np.r_[False, fine == D, False].astype(np.int8)))
    lengths = edges[1::2] - edges[::2]
    interior = [(start, length) for start, length in zip(edges[::2], lengths) if start > 0 and start + length < 3000]
    assert interior and min(length for _, length in interior) >= joint_decoder.STRIDE


def test_minimum_stoppage_path_matches_the_plain_path_when_stoppages_are_long():
    rng = np.random.default_rng(6)
    evidence = np.column_stack([np.repeat(rng.normal(0, 1, 60), 20) for _ in range(3)])
    costs = np.full(len(evidence), 2.0), np.full(len(evidence), 3.0)
    plain = joint_decoder.path(evidence, *costs)
    edges = np.flatnonzero(np.diff(np.r_[False, plain == D, False].astype(np.int8)))
    assert (edges[1::2] - edges[::2]).min() >= 5
    assert np.array_equal(joint_decoder.path(evidence, *costs, min_dead=5), plain)


def test_step_five_is_the_default_decoder():
    rng = np.random.default_rng(4)
    signals = tuple(rng.uniform(.01, .99, 1003) for _ in range(4))
    assert np.array_equal(joint_decoder.decode(*signals, **RECIPE), joint_decoder.decode(*signals, **RECIPE, step=5))
    fine = joint_decoder.decode(*signals, **RECIPE, step=1)
    assert len(fine) == 1003 and set(np.unique(fine)) <= {H, A, D}


def test_output_collapse_matches_the_convention_cases():
    cases = [labels((H, 100), (A, 20), (H, 100)), labels((H, 100), (A, 30), (H, 100)),
             labels((H, 100), (D, 50), (A, 10), (H, 100)), labels((H, 100), (A, 10), (H, 5), (A, 100))]
    for case in cases:
        for seconds in (1, 1.75, 3.0):
            with patch.object(refine_trial.diagnosis, "decode", lambda *_, **__: case.copy()):
                result = refine_trial.refine(None, RECIPE, {"step": 5, "collapse": seconds, "shift": None})
            assert np.array_equal(result, collapse_brief_changes(case, seconds))
    with patch.object(refine_trial.diagnosis, "decode", lambda *_, **__: cases[0].copy()):
        assert (refine_trial.refine(None, RECIPE, {"step": 5, "collapse": 1, "shift": None}) == H).all()
        assert np.array_equal(refine_trial.refine(None, RECIPE, refine_trial.REFERENCE), cases[0])


def test_shift_moves_each_transition_type_by_its_own_amount():
    original = labels((H, 50), (A, 50), (D, 50), (H, 50))
    result = refine_trial.shift_boundaries(original, {"ownership": -3, "restart": 4})
    assert np.array_equal(result, labels((H, 47), (A, 53), (D, 54), (H, 46)))
    later = refine_trial.shift_boundaries(original, {"stop": 2})
    assert np.array_equal(later, labels((H, 50), (A, 52), (D, 48), (H, 50)))
    assert np.array_equal(refine_trial.shift_boundaries(original, None), original)


def test_shift_never_removes_a_segment():
    original = labels((H, 50), (A, 4), (H, 50))
    result = refine_trial.shift_boundaries(original, {"ownership": -10})
    # The A spell can lose at most (4 - 1) // 2 = 1 frame at either end; each H spell up to 24.
    assert np.array_equal(result, labels((H, 40), (A, 13), (H, 51)))
    result = refine_trial.shift_boundaries(original, {"ownership": 10})
    assert np.array_equal(result, labels((H, 51), (A, 13), (H, 40)))
    tiny = labels((H, 50), (A, 1), (H, 50))
    assert np.array_equal(refine_trial.shift_boundaries(tiny, {"ownership": 5}), labels((H, 50), (A, 6), (H, 45)))


def test_grid_and_reference():
    assert len(refine_trial.grid(None)) == 6
    settings = refine_trial.grid({"stop": -3})
    assert len(settings) == 12 and settings[0] == refine_trial.REFERENCE
    assert settings[1] == {"step": 5, "collapse": None, "shift": {"stop": -3}}


def entry(setting, primary, macro=.9, eligible=True):
    trial = {"external": {"conventions": {"aba3": {"standard": {"spell_f1": primary, "macro_f1": macro}}}}}
    return {"setting": setting, "eligible": eligible, "trial": trial}


def test_selection_order_and_gates():
    first, second, third = ({"step": s, "collapse": None, "shift": None} for s in (1, 2, 3))
    assert refine_trial.select([entry(first, .70), entry(second, .71), entry(third, .72, eligible=False)]) == second
    assert refine_trial.select([entry(first, .70, .9), entry(second, .70, .91)]) == second
    assert refine_trial.select([entry(first, .70), entry(second, .70)]) == first
    assert refine_trial.select([entry(first, .70, eligible=False)]) is None
    change = {"primary": .006, "conventions": {"aba1.75": -.009, "aba4": 0, "aba4.5": .01},
              "guards": {"dead_carried": -.01, "team_swap": .0}, "pff": -.005}
    assert refine_trial.passed(change)
    assert not refine_trial.passed(dict(change, primary=.004))
    assert not refine_trial.passed(dict(change, pff=-.011))
    assert not refine_trial.passed(dict(change, guards={"dead_carried": -.02, "team_swap": 0}))
    assert not refine_trial.passed(dict(change, conventions={"aba1.75": -.011, "aba4": 0, "aba4.5": 0}))
