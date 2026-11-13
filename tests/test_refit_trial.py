"""D45: the train_dev split and refit plumbing (external_benchmark, dfl_submission/refit_trial.py)."""

from dfl_submission import external_benchmark as eb, refit_trial as rt, skillcorner
from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN


def test_train_dev_split_adds_only_the_development_matches():
    assert eb.DFL_SPLITS["train_dev"] == (*TRAIN, DEVELOPMENT)
    assert eb.SKILL_SPLITS["train_dev"] == (*skillcorner.TRAIN, *skillcorner.DEVELOPMENT)
    assert not set(eb.DFL_SPLITS["train_dev"]) & set(HOLDOUT)
    assert not set(eb.SKILL_SPLITS["train_dev"]) & set(skillcorner.HOLDOUT)
    assert "train_dev" in eb.TRAINING_VIEWS


def test_existing_splits_are_unchanged():
    assert eb.DFL_SPLITS["train"] == TRAIN and eb.DFL_SPLITS["all"] == (*TRAIN, DEVELOPMENT, *HOLDOUT)
    assert eb.TRAINING_VIEWS[:2] == ("train", "all")


def test_refit_constants():
    assert rt.SPLIT == "train_dev" and rt.COST == 10 and rt.network_folders()["current"] == rt.MODELS / "current"
