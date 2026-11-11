"""Last-touch and possession-chain features (DECISIONS.md D16).

Computed at 25 Hz from the ball and player positions, then sampled at the long-context frames.
Every column is either team-free or comes as an `h__`/`a__` pair, so swapping the teams swaps
the pair. Every threshold below is hand-set and was not tuned. All columns are missing where the
ball is unseen at the sampled frame.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd


FPS = 25
TURN_LAGS = (5, 10)     # frames on each side of the frame whose ball turn is measured
TOUCH_TURN = 4.0        # m/s: the smallest change of ball velocity that counts as a touch
TOUCH_RADIUS = 2.5      # m: a touch needs a player this close; also the evidence weight's scale
SUPPRESS = 5            # frames: one touch per local maximum of the 5-frame turn
CAP = 20.0              # s: times since and until a touch are capped here
WINDOWS = (3, 9)        # s: touch counts in the past and coming windows
PREFIXES = ("h__", "a__")


def ball_turn(ball, lag):
    """Size of the change of ball velocity at each frame, in m/s, from `lag` frames before to `lag` after."""
    turn = np.full(len(ball), np.nan)
    if len(ball) > 2 * lag:
        before = (ball[lag:-lag] - ball[:-2 * lag]) * (FPS / lag)
        after = (ball[2 * lag:] - ball[lag:-lag]) * (FPS / lag)
        turn[lag:-lag] = np.linalg.norm(after - before, axis=1)
    return turn


def nearest(positions, ball, teams, team):
    """Distance from the ball to the team's nearest visible player; missing if none is visible."""
    distance = np.linalg.norm(positions[:, teams == team] - ball[:, None], axis=2)
    distance = np.where(np.isfinite(distance), distance, np.inf).min(axis=1, initial=np.inf)
    return np.where(np.isfinite(distance), distance, np.nan)


def touches(positions, ball, teams, distances=None):
    """Frames and teams (0 home, 1 away) of touches.

    A touch is a local maximum of the 5-frame ball turn of at least TOUCH_TURN, with a player within
    TOUCH_RADIUS of the ball; the team with the nearer player touched it.
    """
    home, away = distances if distances is not None else (nearest(positions, ball, teams, 0),
                                                          nearest(positions, ball, teams, 1))
    turn = ball_turn(ball, TURN_LAGS[0])
    value = np.where(np.isfinite(turn), turn, -np.inf)
    windows = sliding_window_view(np.r_[np.full(SUPPRESS, -np.inf), value, np.full(SUPPRESS, -np.inf)], 2 * SUPPRESS + 1)
    peak = (value >= TOUCH_TURN) & (value > windows[:, :SUPPRESS].max(axis=1)) & (value >= windows[:, SUPPRESS + 1:].max(axis=1))
    home = np.where(np.isfinite(home), home, np.inf)
    away = np.where(np.isfinite(away), away, np.inf)
    frames = np.flatnonzero(peak & (np.minimum(home, away) <= TOUCH_RADIUS))
    return frames, np.where(home[frames] <= away[frames], 0, 1)


def chain_features(frames, teams, timeline):
    """Per team at each `timeline` frame: time since and until its touches, touch counts, and run lengths.

    A run is a sequence of consecutive touches by one team. `touch_run_past` is how long the current
    run has lasted when the latest touch is the team's; `touch_run_future` is how long the run that
    starts with the next touch lasts from now, when that touch is the team's. Both are 0 otherwise.
    """
    fields = {}
    for team, prefix in enumerate(PREFIXES):
        own = frames[teams == team]
        previous = np.searchsorted(own, timeline, side="right") - 1
        following = previous + 1
        fields[prefix + "touch_since"] = np.where(
            previous >= 0, (timeline - own[np.clip(previous, 0, None)]) / FPS if len(own) else CAP, CAP)
        fields[prefix + "touch_until"] = np.where(
            following < len(own), (own[np.clip(following, None, len(own) - 1)] - timeline) / FPS if len(own) else CAP, CAP)
        for seconds in WINDOWS:
            width = seconds * FPS
            now = np.searchsorted(own, timeline, side="right")
            fields[f"{prefix}touches_past_{seconds}"] = now - np.searchsorted(own, timeline - width, side="right")
            fields[f"{prefix}touches_future_{seconds}"] = np.searchsorted(own, timeline + width, side="right") - now
    starts_run = np.r_[True, teams[1:] != teams[:-1]] if len(teams) else np.zeros(0, bool)
    run = np.cumsum(starts_run) - 1
    first = frames[starts_run]
    last = frames[np.r_[starts_run[1:], True]] if len(teams) else frames
    latest = np.searchsorted(frames, timeline, side="right") - 1
    upcoming = latest + 1
    safe_latest, safe_upcoming = np.clip(latest, 0, None), np.clip(upcoming, None, len(frames) - 1)
    for team, prefix in enumerate(PREFIXES):
        if not len(frames):
            fields[prefix + "touch_run_past"] = fields[prefix + "touch_run_future"] = np.zeros(len(timeline))
            continue
        current = (latest >= 0) & (teams[safe_latest] == team)
        coming = (upcoming < len(frames)) & (teams[safe_upcoming] == team)
        fields[prefix + "touch_run_past"] = np.where(current, (timeline - first[run[safe_latest]]) / FPS, 0)
        fields[prefix + "touch_run_future"] = np.where(coming, (last[run[safe_upcoming]] - timeline) / FPS, 0)
    return {name: np.minimum(values, CAP) if "touch_" in name and "touches_" not in name else values
            for name, values in fields.items()}


def features(positions, ball, teams, frames):
    """The D16 columns at the sampled `frames`, as float32, missing where the ball is unseen."""
    positions, ball, teams = np.asarray(positions), np.asarray(ball), np.asarray(teams)
    distances = nearest(positions, ball, teams, 0), nearest(positions, ball, teams, 1)
    turns = {lag: ball_turn(ball, lag) for lag in TURN_LAGS}
    fields = {f"ball_turn_{lag}": turns[lag][frames] for lag in TURN_LAGS}
    for lag in TURN_LAGS:
        for prefix, distance in zip(PREFIXES, distances):
            fields[f"{prefix}touch_evidence_{lag}"] = (turns[lag] * np.exp(-(distance / TOUCH_RADIUS) ** 2))[frames]
    fields.update(chain_features(*touches(positions, ball, teams, distances), frames))
    table = pd.DataFrame(fields).astype(np.float32)
    table.loc[~np.isfinite(ball[frames]).all(axis=1)] = np.nan
    return table
