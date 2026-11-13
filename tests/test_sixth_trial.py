"""D41 plumbing (dfl_submission/sixth_trial.py)."""

from dfl_submission import hyper_trial as ht, second_trial as st, sixth_trial as s


def test_candidates_and_gates():
    assert s.SETTING == {"lr_scale": 2., "batch": 32, "dropout": .3} and s.OWNER_COSTS == (8, 10, 12)
    assert s.candidate_models("D36") == (st.SETTING, st.MODELS)
    assert s.candidate_models("D41") == (s.SETTING, s.MODELS)
    assert (s.DEVELOPMENT_GAIN, s.CONFIRMATION_GAIN) == (.005, .01)


def test_development_gate_is_half_of_c3s():
    change = {"primary": .006, "conventions": {"aba1.75": 0., "aba4": 0., "aba4.5": 0.},
              "guards": {"dead_carried": 0., "team_swap": 0.}, "pff": 0.}
    assert s.eligible(change, True, 0.) and not s.eligible(dict(change, primary=.0049), True, 0.)
    assert not s.eligible(change, False, 0.) and not s.eligible(change, True, -.011)


def test_names():
    assert ht.name(s.SETTING) == "lr_scale2_batch32_dropout0.3"
