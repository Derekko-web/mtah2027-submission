"""External match splits and reference probabilities for the next submission trial."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from dfl_submission import fusion_trial, robust, skillcorner
from dfl_submission.robust_features import stress
from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN, views, write_json
from dfl_transfer_data import SECTIONS, digest
from evaluation import possession_metrics as pm


@dataclass
class Half:
    source: str
    match: str
    view: str
    index: int
    labels: np.ndarray
    positions: np.ndarray
    ball: np.ndarray
    teams: np.ndarray
    pitch: np.ndarray

    @property
    def key(self):
        return f"{self.source}_{self.match}_{self.view}_{self.index}"


def halves(dfl, skill, split, *, raw=False):
    dfl_matches = {"train": TRAIN, "development": (DEVELOPMENT,), "holdout": HOLDOUT}[split]
    skill_matches = {"train": skillcorner.TRAIN, "development": skillcorner.DEVELOPMENT,
                     "holdout": skillcorner.HOLDOUT}[split]
    for match in dfl_matches:
        index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(match)
        source = fusion_trial.external_halves(dfl, match, raw=raw)
        for i, name in enumerate(SECTIONS):
            item = source[name]
            labels = pm.label_codes(item["labels"])
            for view, positions, ball in views(item, index, i):
                yield Half("dfl", match, view, i, labels, positions, ball, item["teams"], item["pitch"])
            if split == "train":
                positions, ball = stress(item["positions"], item["ball"], 370000 + 100 * index + i)
                yield Half("dfl", match, "stress", i, labels, positions, ball, item["teams"], item["pitch"])
    manifest = json.loads((skill / "manifest.json").read_text())
    if manifest["revision"] != skillcorner.REVISION:
        raise ValueError("SkillCorner revision differs")
    for match in skill_matches:
        for i in range(2):
            name = f"{match}_{i + 1}.npz"
            path = skill / name
            if skillcorner.identity(path) != tuple(manifest["parsed"][name]):
                raise ValueError(f"SkillCorner input identity differs: {name}")
            with np.load(path, allow_pickle=False) as item:
                labels = pm.label_codes(item["labels"])
                yield Half("skillcorner", str(match), "observed", i, labels, item["positions"],
                           item["ball"], item["teams"], item["pitch"])


def reference(folder):
    package = folder / "dfl_submission"
    recipe = json.loads((package / "fusion_recipe/recipe.json").read_text())
    models = (robust.verified_models(package / "robust_models", recipe["dfl_models"]),
              robust.verified_models(package / "skillcorner_models", recipe["skillcorner_models"]))
    return recipe, models


def reference_probabilities(models, recipe, positions, ball, teams, pitch):
    dfl_owner, dfl_dead, skill_owner, skill_dead = fusion_trial.predict(models, positions, ball, teams, pitch)
    owner, dead = recipe["owner_weight"], recipe["dead_weight"]
    return (1 - owner) * dfl_owner + owner * skill_owner, (1 - dead) * dfl_dead + dead * skill_dead


def cache_reference(dfl, skill, split, fallback, output, *, raw=False):
    output.mkdir(parents=True, exist_ok=True)
    recipe, models = reference(fallback)
    identity = {"split": split, "reference": recipe, "skill_manifest": digest(skill / "manifest.json")}
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Existing reference cache belongs to a different run")
    else:
        write_json(manifest, identity)
    for half in halves(dfl, skill, split, raw=raw):
        destination = output / f"{half.key}.npz"
        if destination.exists():
            continue
        home, dead = reference_probabilities(models, recipe["recipe"], half.positions, half.ball, half.teams, half.pitch)
        partial = destination.with_suffix(".partial.npz")
        np.savez_compressed(partial, labels=half.labels, home=home, dead=dead)
        partial.replace(destination)
        print(f"Reference {half.key}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl", required=True, type=Path)
    parser.add_argument("--dfl-raw", action="store_true", help="Read pinned DFL XML instead of parsed caches")
    parser.add_argument("--skill", required=True, type=Path)
    parser.add_argument("--fallback", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("development", "holdout"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    cache_reference(args.dfl, args.skill, args.split, args.fallback, args.out, raw=args.dfl_raw)
