"""D23's `all` split on synthetic inputs: every labelled match once, with the training views."""

import numpy as np

from dfl_submission import augment, external_benchmark as benchmark, fusion_trial, skillcorner
from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN

DFL_ALL = (*TRAIN, DEVELOPMENT, *HOLDOUT)
SKILL_ALL = (*skillcorner.TRAIN, *skillcorner.DEVELOPMENT, *skillcorner.HOLDOUT)


def synthetic_half(seed, length=600, players=6):
    rng = np.random.default_rng(seed)
    labels = np.array(["H"] * 200 + ["D"] * 100 + ["A"] * 300)
    positions = rng.uniform(-30, 30, size=(length, players, 2)).astype(np.float32)
    ball = rng.uniform(-30, 30, size=(length, 2)).astype(np.float32)
    return {"labels": labels, "positions": positions, "ball": ball, "teams": np.repeat([0, 1], players // 2),
            "pitch": np.array([105., 68.]), "timestamps": np.arange(length, dtype=np.float64) / 25}


def patched(function):
    def wrapper(tmp_path):
        original = fusion_trial.external_halves, benchmark.skillcorner_paths
        paths = {}
        for k, match in enumerate(SKILL_ALL):
            for i in range(2):
                item = synthetic_half(1000 + 10 * k + i)
                path = tmp_path / f"{match}_{i + 1}.npz"
                np.savez(path, labels=item["labels"], positions=item["positions"], ball=item["ball"],
                         teams=item["teams"], pitch=item["pitch"])
                paths[match, i] = path
        lists = {"train": skillcorner.TRAIN, "development": skillcorner.DEVELOPMENT, "holdout": skillcorner.HOLDOUT,
                 "all": SKILL_ALL}

        def external_halves(dfl, match, raw=False):
            index = DFL_ALL.index(match)
            return {name: synthetic_half(100 * index + i) for i, name in enumerate(("firstHalf", "secondHalf"))}

        def skillcorner_paths(skill, split):
            for match in lists[split]:
                for i in range(2):
                    yield match, i, paths[match, i]

        fusion_trial.external_halves, benchmark.skillcorner_paths = external_halves, skillcorner_paths
        try:
            function()
        finally:
            fusion_trial.external_halves, benchmark.skillcorner_paths = original
    wrapper.__name__ = function.__name__
    return wrapper


def same_half(a, b):
    return (a.key == b.key and np.array_equal(a.labels, b.labels) and np.array_equal(a.positions, b.positions, equal_nan=True)
            and np.array_equal(a.ball, b.ball, equal_nan=True))


@patched
def check_benchmark_all_split():
    halves = list(benchmark.halves("dfl", "skill", "all", raw=True))
    keys = [half.key for half in halves]
    assert len(keys) == len(set(keys)) == len(DFL_ALL) * 2 * 3 + len(SKILL_ALL) * 2
    assert {(h.source, h.match) for h in halves} == {("dfl", m) for m in DFL_ALL} | {("skillcorner", str(m)) for m in SKILL_ALL}
    assert {h.view for h in halves if h.source == "dfl"} == {"clean", "masked", "stress"}
    train = {half.key: half for half in benchmark.halves("dfl", "skill", "train", raw=True)}
    development = {half.key: half for half in benchmark.halves("dfl", "skill", "development", raw=True)}
    for half in halves:
        if half.key in train:
            assert same_half(half, train[half.key])
        elif half.key in development:
            assert same_half(half, development[half.key])
    assert sum(half.key in train for half in halves) == len(train)


def test_benchmark_all_split(tmp_path):
    check_benchmark_all_split(tmp_path)


@patched
def check_augmented_all_split():
    halves = list(augment.training_halves("dfl", "skill", raw=True, split="all"))
    keys = [half.key for half in halves]
    assert len(keys) == len(set(keys)) == len(DFL_ALL) * 2 * 3 + len(SKILL_ALL) * 2
    train = {half.key: half for half in augment.training_halves("dfl", "skill", raw=True)}
    assert all(same_half(half, train[half.key]) for half in halves if half.key in train)
    assert sum(half.key in train for half in halves) == len(train) == len(TRAIN) * 2 * 3 + len(skillcorner.TRAIN) * 2
    new = [half for half in halves if half.source == "skillcorner" and int(half.match) not in skillcorner.TRAIN]
    assert len(new) == 16


def test_augmented_all_split(tmp_path):
    check_augmented_all_split(tmp_path)
