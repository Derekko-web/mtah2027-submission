"""D8 gate, selection and pipeline-check logic on synthetic scores. No PFF data is committed."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from dfl_submission import decoder_trial, retrain_trial, vendor_trial


ROOT_FOR_TESTS = Path(__file__).resolve().parents[1]


def scores(values, macro=0.8):
    """One match's scores per convention; conventions not in `values` score 0.6."""
    return {name: {"spell_f1": values.get(name, 0.6), "macro_f1": macro, "spells": 10}
            for name in decoder_trial.CONVENTIONS}


def guard_trial(standard, carried, swap):
    return {"conventions": {decoder_trial.PRIMARY: {"standard": {"spell_f1": standard, "macro_f1": 0.9},
                                                    "dead_carried": carried, "team_swap": swap}}}


def test_mixing_at_weight_zero_keeps_the_current_models():
    current = {"a": (np.array([0.2, 0.7], dtype=np.float32), None), "b": [np.array([0.1], dtype=np.float32)]}
    retrained = {"a": (np.array([0.9, 0.1], dtype=np.float32), None), "b": [np.array([0.5], dtype=np.float32)]}
    mixed = retrain_trial.mix(current, retrained, 0.0)
    np.testing.assert_array_equal(mixed["a"][0], current["a"][0])
    assert mixed["a"][1] is None
    np.testing.assert_array_equal(mixed["b"][0], current["b"][0])
    np.testing.assert_allclose(retrain_trial.mix(current, retrained, 0.25)["a"][0], [0.375, 0.55], rtol=1e-6)


def test_d8_weights_include_d3():
    assert vendor_trial.WEIGHTS == (0.0, 0.25, 0.5, 0.75, 1.0) and vendor_trial.D3_WEIGHT in vendor_trial.WEIGHTS
    assert vendor_trial.OTHER_CONVENTIONS == ("aba1.75", "aba4", "aba4.5")


def test_pff_gates():
    base = {"1": scores({"aba3": 0.60}), "2": scores({"aba3": 0.60})}
    better = {"1": scores({"aba3": 0.65}), "2": scores({"aba3": 0.63})}
    changes = vendor_trial.pff_changes(base, better)
    assert abs(changes["primary"] - 0.04) < 1e-12 and abs(changes["worst_match"] - 0.03) < 1e-12
    assert vendor_trial.pff_passed(changes)
    small = {"1": scores({"aba3": 0.61}), "2": scores({"aba3": 0.62})}  # +0.015
    assert not vendor_trial.pff_passed(vendor_trial.pff_changes(base, small))
    convention = {game: scores({"aba3": 0.66, "aba4.5": 0.585}) for game in base}  # N = 4.5 s drops 0.015
    assert not vendor_trial.pff_passed(vendor_trial.pff_changes(base, convention))
    one_match = {"1": scores({"aba3": 0.75}), "2": scores({"aba3": 0.565})}  # mean +0.0575, one match -0.035
    assert not vendor_trial.pff_passed(vendor_trial.pff_changes(base, one_match))
    raw_only = {game: scores({"aba3": 0.65, "raw": 0.3}) for game in base}  # raw labels are reported, not gated
    assert vendor_trial.pff_passed(vendor_trial.pff_changes(base, raw_only))


def test_guard_gates():
    changes = vendor_trial.guard_changes(guard_trial(0.68, 0.56, 0.58), guard_trial(0.67, 0.60, 0.575))
    assert set(changes) == {"standard", "dead_carried", "team_swap"}
    assert abs(changes["standard"] + 0.01) < 1e-12 and abs(changes["dead_carried"] - 0.04) < 1e-12
    assert vendor_trial.guard_passed(changes)
    limits = {"dead_carried": -0.01, "team_swap": -0.01, "standard": -0.02}
    assert vendor_trial.guard_passed(limits)
    for part, value in (("dead_carried", -0.011), ("team_swap", -0.02), ("standard", -0.021)):
        assert not vendor_trial.guard_passed(dict(limits, **{part: value}))


def test_selection_takes_the_best_eligible_weight():
    def entry(weight, eligible, spell, macro):
        return {"weight": weight, "eligible": eligible,
                "pff": {"means": {decoder_trial.PRIMARY: {"spell_f1": spell, "macro_f1": macro}}}}
    entries = [entry(0.0, False, 0.70, 0.8), entry(0.25, True, 0.66, 0.8), entry(0.75, True, 0.66, 0.85),
               entry(1.0, True, 0.65, 0.9)]
    assert vendor_trial.select(entries) == 0.75
    assert vendor_trial.select([dict(item, eligible=False) for item in entries]) is None


def test_reproduction_compares_the_d3_weight_with_the_d7_and_d3_records():
    pff_matches = {"3812": scores({"aba3": 0.61})}
    trial = {"recipe": {"cutoff": 0.53}, "conventions": {"aba3": {"dead_carried": 0.5}}}
    with TemporaryDirectory() as folder:
        d7, d3 = Path(folder) / "d7.json", Path(folder) / "d3.json"
        d7.write_text(json.dumps({"matches": pff_matches, "means": {}}))
        d3.write_text(json.dumps({"selected": trial, "reference": {}}))
        assert vendor_trial.reproduction(pff_matches, trial, d7, d3) == {"d7": True, "d3": True}
        changed = {"3812": scores({"aba3": 0.62})}
        assert vendor_trial.reproduction(changed, trial, d7, d3) == {"d7": False, "d3": True}
        assert vendor_trial.reproduction(pff_matches, dict(trial, recipe={"cutoff": 0.49}), d7, d3)["d3"] is False


def test_d8_commands_require_external_only():
    with patch.dict(os.environ, {"MTAH_DATA_DIR": str(ROOT_FOR_TESTS)}):
        for command in (lambda: vendor_trial.tune(Path("missing"), Path("missing"), Path("missing")),
                        lambda: vendor_trial.confirm(Path("missing"))):
            try:
                command()
            except SystemExit:
                continue
            raise AssertionError("D8 commands must refuse while the bundle path exists")


def test_confirmation_needs_a_development_winner_and_runs_once():
    with TemporaryDirectory() as folder, patch.dict(os.environ, {"MTAH_DATA_DIR": "/nonexistent/mtah-test"}):
        development, output = Path(folder) / "d8_development.json", Path(folder) / "d8_confirmation.json"
        development.write_text(json.dumps({"selected": None}))
        try:
            vendor_trial.confirm(Path("missing"), output, development)
        except ValueError:
            pass
        else:
            raise AssertionError("confirmation must not run without an eligible development weight")
        development.write_text(json.dumps({"selected": {"weight": 0.75}}))
        output.write_text("{}")
        try:
            vendor_trial.confirm(Path("missing"), output, development)
        except FileExistsError:
            pass
        else:
            raise AssertionError("confirmation is scored once")
