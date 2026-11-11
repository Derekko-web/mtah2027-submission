"""Shared, origin-independent geometry for the external robustness trial."""

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def mean(values, radius):
    present = np.isfinite(values)
    size = 2 * radius + 1
    total = uniform_filter1d(np.where(present, values, 0.0), size, mode="constant")
    mass = uniform_filter1d(present.astype(float), size, mode="constant")
    return np.divide(total, mass, out=np.full(len(values), np.nan), where=mass > 1e-8)


def displacement(points, step, future=False):
    speed = np.full(len(points), np.nan)
    delta = np.linalg.norm(points[step:] - points[:-step], axis=1) * (25 / step)
    if future:
        speed[:-step] = delta
    else:
        speed[step:] = delta
    speed[speed > 50] = np.nan
    return speed


def extras(positions, ball, teams, pitch):
    """Use pitch geometry and contact continuity, never provider status or ownership."""
    ball = np.asarray(ball, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    nearest_margin = (pitch / 2 - np.abs(ball)).min(axis=1)
    goal_margin = pitch[0] / 2 - np.abs(ball[:, 0])
    touch_margin = pitch[1] / 2 - np.abs(ball[:, 1])
    features = {"pitch_margin": nearest_margin, "goal_margin": goal_margin,
                "touch_margin": touch_margin}
    speeds = {}
    for step in (3, 12, 30):
        for future in (False, True):
            name = f"speed_{step}_{int(future)}"
            speeds[name] = displacement(ball, step, future)
    features.update(speeds)
    features["speed_change"] = speeds["speed_12_1"] - speeds["speed_12_0"]
    for radius in (12, 50):
        features[f"margin_mean_{radius}"] = mean(nearest_margin, radius)
        features[f"outside_share_{radius}"] = mean(
            np.where(np.isfinite(nearest_margin), nearest_margin < 0, np.nan), radius)
        features[f"speed_mean_{radius}"] = mean(speeds["speed_12_0"], radius)

    for team, prefix in enumerate(("h__", "a__")):
        tracks = positions[:, teams == team]
        nearest_distance = np.full(len(ball), np.nan)
        relative_speed = np.full(len(ball), np.nan)
        nearby_heading = np.full(len(ball), np.nan)
        for start in range(0, len(ball), 4096):
            stop = min(len(ball), start + 4096)
            frames = np.arange(start, stop)
            distances = np.linalg.norm(tracks[start:stop] - ball[start:stop, None], axis=2)
            distances = np.where(np.isfinite(distances), distances, np.inf)
            slots = distances.argmin(axis=1)
            nearest = distances[np.arange(len(frames)), slots]
            nearest_distance[start:stop] = np.where(np.isfinite(nearest), nearest, np.nan)
            previous = np.maximum(0, frames - 5)
            player_delta = tracks[frames, slots] - tracks[previous, slots]
            ball_delta = ball[frames] - ball[previous]
            valid = (frames >= 5) & (nearest < 6) & (np.linalg.norm(player_delta, axis=1) < 3)
            relative_speed[start:stop] = np.where(valid, np.linalg.norm(player_delta - ball_delta, axis=1) * 5, np.nan)
            distance_past = np.linalg.norm(tracks[previous, slots] - ball[previous], axis=1)
            nearby_heading[start:stop] = np.where(valid, distance_past - nearest, np.nan)
        features[prefix + "contact_relative_speed"] = relative_speed
        features[prefix + "contact_closing"] = nearby_heading
        contact = np.where(np.isfinite(nearest_distance), np.exp(-nearest_distance / 2), np.nan)
        for radius in (6, 18, 50, 100):
            features[prefix + f"contact_strength_{radius}"] = mean(contact, radius)
            features[prefix + f"contact_distance_{radius}"] = mean(nearest_distance, radius)
    return pd.DataFrame(features, dtype=np.float32)


def invariant(features):
    """Turn the team-paired additions into symmetric live/dead features."""
    values = {}
    for name in features:
        if name.startswith("h__"):
            home, away = features[name], features["a__" + name[3:]]
            values[name[3:] + "_min"] = np.fmin(home, away)
            values[name[3:] + "_max"] = np.fmax(home, away)
        elif not name.startswith("a__"):
            values[name] = features[name]
    return pd.DataFrame(values)


def stress(positions, ball, seed):
    """Simulate longer occlusions, camera outages, and bad ball observations on DFL."""
    rng = np.random.default_rng(seed)
    positions = np.array(positions, dtype=np.float32, copy=True)
    ball = np.array(ball, dtype=np.float32, copy=True)
    frames = np.arange(len(ball))
    # Each alternating period lasts a random number of frames, independent of the labels.
    durations = rng.geometric(.025, len(ball)) + 1
    spans = np.searchsorted(np.cumsum(durations), frames, side="right")
    ball[spans % 2 == 0] = np.nan
    glitches = rng.random(len(ball)) < .035
    ball[glitches] += rng.normal(0, 5, (int(glitches.sum()), 2)).astype(np.float32)
    ball += rng.normal(0, .35, ball.shape).astype(np.float32)
    positions += rng.normal(0, .30, positions.shape).astype(np.float32)
    chunks = frames // 40
    drop = rng.random((int(chunks[-1]) + 1, positions.shape[1])) < .28
    outages = rng.random(len(drop)) < .07
    positions[drop[chunks] | outages[chunks, None]] = np.nan
    ball[outages[chunks]] = np.nan
    return positions, ball
