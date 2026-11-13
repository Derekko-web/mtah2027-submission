"""Pressure and support around the ball (DECISIONS.md D39).

The long-context table describes each team's nearest player to the ball only. These columns describe
the players around it: computed at 25 Hz from the ball and player positions, sampled at the
long-context frames. Every column is an `h__`/`a__` pair, so swapping the teams swaps the pair. Every
threshold is hand-set and was not tuned. All columns are missing where the ball is unseen at the frame
(and, for the motion columns, five frames earlier).
"""

import numpy as np
import pandas as pd


FPS = 25
LAG = 5                  # frames over which speeds and approach are measured (0.2 s)
RADII = (5.0, 10.0)      # m: player counts around the ball
APPROACHING = 3          # the team's nearest players whose approach is averaged
MAX_SPEED = 12.0         # m/s: faster player displacements are tracking jumps
MOVING = .5              # m/s: below this a direction is undefined
PREFIXES = ("h__", "a__")
CHUNK = 4096


def team_columns(positions, ball, frames):
    """The six columns of one team's tracks `positions` (T, P, 2) at `frames`."""
    rows = {name: np.full(len(frames), np.nan, dtype=np.float64) for name in
            ("second_closest", "count_5", "count_10", "approach_3", "nearest_speed", "nearest_alignment")}
    for start in range(0, len(frames), CHUNK):
        index = frames[start:start + CHUNK]
        part = slice(start, start + len(index))
        now = positions[index]
        seen = np.isfinite(ball[index]).all(axis=1)
        distance = np.linalg.norm(now - ball[index][:, None], axis=2)
        valid = np.isfinite(distance)
        ranked = np.sort(np.where(valid, distance, np.inf), axis=1)
        if ranked.shape[1] >= 2:
            second = ranked[:, 1]
            rows["second_closest"][part] = np.where(seen & np.isfinite(second), second, np.nan)
        for radius, name in zip(RADII, ("count_5", "count_10")):
            rows[name][part] = np.where(seen, (valid & (distance <= radius)).sum(axis=1), np.nan)
        earlier = np.maximum(index - LAG, 0)
        before = positions[earlier]
        ball_before = ball[earlier]
        has_past = (index >= LAG) & seen & np.isfinite(ball_before).all(axis=1)
        distance_before = np.linalg.norm(before - ball_before[:, None], axis=2)
        step = np.linalg.norm(now - before, axis=2) * (FPS / LAG)
        approach = (distance_before - distance) * (FPS / LAG)
        approach = np.where(np.isfinite(approach) & (step <= MAX_SPEED), approach, np.nan)
        order = np.argsort(np.where(valid, distance, np.inf), axis=1)[:, :APPROACHING]
        chosen = np.take_along_axis(approach, order, axis=1)
        chosen_valid = np.take_along_axis(valid, order, axis=1) & np.isfinite(chosen)
        count = chosen_valid.sum(axis=1)
        mean = np.divide(np.where(chosen_valid, chosen, 0).sum(axis=1), count, out=np.full(len(index), np.nan),
                         where=count > 0)
        rows["approach_3"][part] = np.where(has_past, mean, np.nan)
        nearest = order[:, 0]
        rows_index = np.arange(len(index))
        nearest_valid = valid[rows_index, nearest]
        velocity = (now[rows_index, nearest] - before[rows_index, nearest]) * (FPS / LAG)
        speed = np.linalg.norm(velocity, axis=1)
        speed_ok = has_past & nearest_valid & np.isfinite(speed) & (speed <= MAX_SPEED)
        rows["nearest_speed"][part] = np.where(speed_ok, speed, np.nan)
        ball_velocity = (ball[index] - ball_before) * (FPS / LAG)
        ball_speed = np.linalg.norm(ball_velocity, axis=1)
        aligned = speed_ok & (speed >= MOVING) & np.isfinite(ball_speed) & (ball_speed >= MOVING)
        cosine = np.einsum("ij,ij->i", np.nan_to_num(velocity), np.nan_to_num(ball_velocity)) / \
            np.maximum(speed * ball_speed, 1e-9)
        rows["nearest_alignment"][part] = np.where(aligned, cosine, np.nan)
    return rows


def features(positions, ball, teams, frames):
    """A DataFrame of the 12 pressure columns at `frames` (rows in frame order)."""
    positions = np.asarray(positions, dtype=np.float64)
    ball = np.asarray(ball, dtype=np.float64)
    frames = np.asarray(frames)
    fields = {}
    for team, prefix in enumerate(PREFIXES):
        for name, values in team_columns(positions[:, np.asarray(teams) == team], ball, frames).items():
            fields[prefix + name] = values
    return pd.DataFrame(fields, dtype=np.float32)
