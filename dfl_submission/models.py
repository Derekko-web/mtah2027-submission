"""Shared XY features and CPU XGBoost inference for one 25 Hz half."""

from pathlib import Path
import sys

import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from dfl_dead_features import build as dead_features
from dfl_transfer_features import geometry as ownership_features
from dfl_transfer_teacher import probability as ownership_probability

MODEL_FILES = ("ownership.ubj", "dead.ubj")


def predict(ownership, dead, positions, ball, teams):
    blocks = [(0, len(ball))]
    features = ownership_features(positions, ball, teams, blocks)
    home_probability = ownership_probability(ownership, features)
    home_distance = features["h__distance1"].fillna(np.inf).to_numpy()
    away_distance = features["a__distance1"].fillna(np.inf).to_numpy()
    raw = np.where(np.isinf(home_distance) & np.isinf(away_distance), 2,
                   np.where(home_distance <= away_distance, 0, 1)).astype(np.int8)
    del features
    stoppage_probability = dead.predict_proba(dead_features(positions, ball, teams, blocks))[:, 1]
    if not np.isfinite(stoppage_probability).all():
        raise ValueError("Invalid live/dead probabilities")
    return home_probability, stoppage_probability, raw


def load_models(folder):
    models = []
    for name in MODEL_FILES:
        model = XGBClassifier(n_jobs=2)
        model.load_model(folder / name)
        models.append(model)
    return tuple(models)
