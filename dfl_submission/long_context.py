"""External-only possession heads with longer tracking context and pooled match weighting."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from dfl_submission import external_benchmark as benchmark, models, robust
from dfl_submission.robust_features import invariant
from dfl_submission.skillcorner_trial import inputs as local_inputs
from dfl_submission.train import SOURCE_MANIFEST, write_json
from dfl_transfer_data import digest
from dfl_transfer_features import swap
from evaluation import possession_metrics as pm


STRIDE = 5
FILE_NAMES = models.MODEL_FILES
TREE_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                   max_depth=5, n_estimators=420, learning_rate=.045, subsample=.82,
                   colsample_bytree=.85, min_child_weight=30, reg_lambda=12,
                   n_jobs=3, random_state=42)


def interval_stats(values, width, future):
    indices = np.arange(len(values))
    if future:
        left, right = indices, np.minimum(len(values), indices + width + 1)
    else:
        left, right = np.maximum(0, indices - width), indices + 1
    valid = np.isfinite(values)
    observed = np.r_[0, np.cumsum(valid)]
    total = np.r_[0.0, np.cumsum(np.where(valid, values, 0), dtype=np.float64)]
    count = observed[right] - observed[left]
    mean = np.divide(total[right] - total[left], count, out=np.full(len(values), np.nan), where=count > 0)
    return mean, count / (right - left)


def features(positions, ball, teams, pitch):
    """Sample at 5 Hz after computing local contact evidence on the full input grid."""
    base, _ = local_inputs(positions, ball, teams, pitch)
    frames = np.arange(0, len(ball), STRIDE)
    local = base.iloc[frames].reset_index(drop=True)
    fields = {}
    chosen = ("speed_12_0", "pitch_margin", "ball_support_12", "h__closest", "a__closest",
              "h__contact_strength_18", "a__contact_strength_18", "h__player_motion_25_0",
              "a__player_motion_25_0")
    for column in chosen:
        values = local[column].to_numpy()
        for seconds in (3, 9, 18):
            for future, direction in ((False, "past"), (True, "future")):
                mean, coverage = interval_stats(values, seconds * 5, future)
                fields[f"{column}_{direction}_{seconds}"] = mean
                if column in ("speed_12_0", "h__closest", "a__closest"):
                    fields[f"{column}_{direction}_{seconds}_coverage"] = coverage

    seen = np.isfinite(ball).all(axis=1)
    timeline = np.arange(len(ball))
    previous = np.maximum.accumulate(np.where(seen, timeline, -1))[frames]
    following = np.minimum.accumulate(np.where(seen, timeline, len(ball))[::-1])[::-1][frames]
    for indices, direction in ((previous, "previous"), (following, "following")):
        age = np.abs(frames - indices) / 25
        valid = (indices >= 0) & (indices < len(ball)) & (age <= 18)
        fields[f"ball_age_{direction}"] = np.where(valid, age, np.nan)
        clipped = np.clip(indices, 0, len(ball) - 1)
        for column in ("pitch_margin", "h__closest", "a__closest", "h__contact_strength_6", "a__contact_strength_6"):
            fields[f"{column}_{direction}_ball"] = np.where(valid, base[column].to_numpy()[clipped], np.nan)

    for team, prefix in enumerate(("h__", "a__")):
        xy = positions[frames][:, teams == team]
        valid = np.isfinite(xy).all(axis=2)
        count = valid.sum(axis=1, keepdims=True)
        center = np.divide(np.where(valid[..., None], xy, 0).sum(axis=1), count,
                           out=np.full((len(frames), 2), np.nan), where=count > 0)
        squared = (xy - center[:, None]) ** 2
        spread = np.sqrt(np.divide(np.where(valid[..., None], squared, 0).sum(axis=1), count,
                                  out=np.full_like(center, np.nan), where=count > 0))
        fields[prefix + "shape_depth"] = spread[:, 0]
        fields[prefix + "shape_width"] = spread[:, 1]
        fields[prefix + "center_ball_distance"] = np.linalg.norm(center - ball[frames], axis=1)
        for axis, name in enumerate(("long", "wide")):
            speed = np.r_[np.full(5, np.nan), np.abs(center[5:, axis] - center[:-5, axis])]
            speed[speed > 15] = np.nan
            for future, direction in ((False, "past"), (True, "future")):
                mean, _ = interval_stats(speed, 15, future)
                fields[f"{prefix}shape_motion_{name}_{direction}"] = mean
    owner = local.join(pd.DataFrame(fields, dtype=np.float32))
    return frames, owner, invariant(owner)


def source_weights(y, groups):
    weight = np.empty(len(y), dtype=np.float32)
    sources = {key.split(":", 1)[0] for key in groups}
    for source in sorted(sources):
        keys = np.unique([key for key in groups if key.startswith(f"{source}:")])
        for key in keys:
            for target in (0, 1):
                members = (groups == key) & (y == target)
                if not members.any():
                    raise ValueError(f"External match lacks both labels: {key}")
                weight[members] = len(y) / (len(sources) * len(keys) * 2 * members.sum())
    return weight


def fit_head(rows, *, ownership):
    features, targets, match_keys = zip(*rows)
    inputs = pd.concat(features, ignore_index=True)
    y = np.concatenate(targets).astype(int)
    weight = source_weights(y, np.concatenate(match_keys))
    if ownership:
        inputs = pd.concat([inputs, swap(inputs)], ignore_index=True)
        y, weight = np.r_[y, 1 - y], np.tile(weight, 2)
    model = XGBClassifier(**TREE_PARAMS)
    model.fit(inputs, y, sample_weight=weight)
    return model, len(y)


def train(dfl, skill, output, *, raw=False, augment=False):
    output.mkdir(parents=True, exist_ok=False)
    owner_rows, dead_rows = [], []
    if augment:
        from dfl_submission.augment import training_halves
        halves = training_halves(dfl, skill, raw=raw)
    else:
        halves = benchmark.halves(dfl, skill, "train", raw=raw)
    for half in halves:
        frames, owner, dead = features(half.positions, half.ball, half.teams, half.pitch)
        labels = half.labels[frames]
        live, known = labels < pm.D, labels != pm.X
        key = f"{half.source}:{half.match}"
        owner_rows.append((owner.loc[live].copy(), labels[live] == pm.H, np.full(live.sum(), key)))
        dead_rows.append((dead.loc[known].copy(), labels[known] == pm.D, np.full(known.sum(), key)))
        print(f"Prepared {half.key}", flush=True)
    hashes, counts = {}, {}
    for name, rows, ownership in zip(FILE_NAMES, (owner_rows, dead_rows), (True, False)):
        model, counts[name] = fit_head(rows, ownership=ownership)
        path = output / name
        model.save_model(path)
        hashes[name] = digest(path)
        rows.clear()
        print(f"Fitted {name}, {counts[name]:,} rows", flush=True)
    write_json(output / "fit.json", {"models": hashes, "rows": counts, "sample_stride": STRIDE,
               "dfl_manifest_sha256": digest(SOURCE_MANIFEST), "skill_manifest_sha256": digest(skill / "manifest.json")} | ({"augmented": True} if augment else {}))


def load(folder):
    fitted = json.loads((folder / "fit.json").read_text())
    return robust.verified_models(folder, fitted["models"]), fitted


def predict(checkpoints, positions, ball, teams, pitch):
    frames, owner, dead = features(positions, ball, teams, pitch)
    home = models.ownership_probability(checkpoints[0], owner)
    stoppage = checkpoints[1].predict_proba(dead)[:, 1]
    timeline = np.arange(len(ball))
    return np.interp(timeline, frames, home).astype(np.float32), np.interp(timeline, frames, stoppage).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl", required=True, type=Path)
    parser.add_argument("--dfl-raw", action="store_true")
    parser.add_argument("--skill", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--augment", action="store_true", help="Train on DECISIONS.md D2 augmented halves")
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    require_external_only()
    train(args.dfl, args.skill, args.out, raw=args.dfl_raw, augment=args.augment)
