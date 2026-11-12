"""D25 plumbing on synthetic data: the gates, the post-processed decoder and the stoppage counts."""

import numpy as np

from dfl_submission import aligned_trial, diagnosis, stoppage_trial as st
from dfl_submission.conventions import remove_brief_stoppages
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def change(primary=0.0, conventions=0.0, guards=(0.0, 0.0), pff=0.0):
    return {"primary": primary, "conventions": {"aba1.75": conventions, "aba4": 0.0, "aba4.5": 0.0},
            "guards": {"dead_carried": guards[0], "team_swap": guards[1]}, "pff": pff}


def test_no_harm_gates_on_development():
    assert st.passed(change(), metrica=0.0, d24=0.0)
    assert st.passed(change(primary=-0.002, conventions=-0.005, guards=(-0.01, 0.01)), metrica=0.0, d24=0.0)
    assert not st.passed(change(primary=-0.0021), metrica=0.0, d24=0.0)
    assert not st.passed(change(conventions=-0.0051), metrica=0.0, d24=0.0)
    assert not st.passed(change(guards=(-0.01, 0.009)), metrica=0.0, d24=0.0)
    assert not st.passed(change(pff=-1e-9), metrica=0.0, d24=0.0)
    assert not st.passed(change(), metrica=-1e-9, d24=0.0)
    assert not st.passed(change(), metrica=0.0, d24=-1e-9)


def test_confirmation_gates_leave_out_the_metrica_conditions():
    assert st.passed(change())
    assert not st.passed(change(pff=-0.001))


def test_predictor_applies_the_post_step_after_decoding():
    decoded = labels((H, 50), (D, 10), (A, 50))
    monkeypatch_decode = lambda signals, recipe, **options: decoded.copy()
    original = diagnosis.decode
    diagnosis.decode = monkeypatch_decode
    try:
        half = diagnosis.HalfSignals("pff", "1", "observed", 0, decoded, (), decoded)
        assert np.array_equal(st.predictor({}, None)(half), decoded)
        assert np.array_equal(st.predictor({}, st.post)(half), remove_brief_stoppages(decoded))
        assert np.array_equal(st.post(decoded), labels((H, 55), (A, 55)))
    finally:
        diagnosis.decode = original


def synthetic_halves(seed=0):
    rng = np.random.default_rng(seed)
    halves = []
    for game in ("1", "2"):
        for index in (0, 1):
            truth = labels((H, 300), (D, 100), (A, 300))
            signals = tuple(rng.uniform(.05, .95, len(truth)).astype(np.float32) for _ in range(4))
            halves.append(diagnosis.HalfSignals("pff", game, "observed", index, truth, signals, truth))
    return halves


def test_evaluate_without_the_post_step_is_d15s_evaluation():
    halves = synthetic_halves()
    recipe = {"cutoff": .53, "dead_cost": 2.0, "owner_cost": 8.0, "discount": .8}
    assert st.evaluate(halves, recipe) == aligned_trial.evaluate(halves, recipe)
    assert st.evaluate(halves, recipe, lambda codes: codes) == aligned_trial.evaluate(halves, recipe)


def test_group_stoppages_count_predicted_and_true_sub_second_stoppages():
    truth = labels((H, 100), (D, 50), (A, 100))
    prediction = labels((H, 40), (D, 10), (H, 50), (D, 50), (A, 100))
    scored = {("pff", "1", "observed"): (truth, prediction, None), ("dfl", "WPY", "masked"): (truth, truth, None)}
    counts = st.group_stoppages(scored)
    assert counts["pff"] == {"predicted": {"all": 1, "interior": 1}, "true": {"all": 0, "interior": 0}}
    assert counts["dfl_masked"]["predicted"] == {"all": 0, "interior": 0}


def test_half_decoder_post_processes_each_built_half():
    decoded = labels((H, 50), (D, 10), (A, 50))
    original = diagnosis.half_signals
    diagnosis.half_signals = lambda model, positions, ball, teams, pitch: ((), decoded.copy())
    try:
        assert np.array_equal(st.half_decoder("model", None)(None, None, None, None), decoded)
        assert np.array_equal(st.half_decoder("model", st.post)(None, None, None, None), labels((H, 55), (A, 55)))
    finally:
        diagnosis.half_signals = original
