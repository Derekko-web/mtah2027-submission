"""D55: the height-aware DFL ball reader and the five situation views of DFL WPY (DECISIONS.md D55).

The existing reader (`dfl_transfer_data.parse_match`) keeps the ball's X, Y, BallStatus, BallPossession and
T and drops Z. `parse_ball` reads the same ball frames again with Z; its X and Y go through exactly the
same steps (float32, NaN outside the pitch plus 5 m), so they can be checked against the existing reader
frame for frame (`reader_check`).

The views are built like the masked development view (`corrupt`'s coordinate noise and 2-second player
dropout, from new fixed seeds), with the hide process replaced:
- (a) air: the ball is hidden on every frame where its height is above 1.5 m;
- (b) crowd: hidden on every frame where at least three players are within 3 m of it;
- (c) change: hidden from 1 s before to 2 s after each ownership change of the N = 3 targets;
- (d) air or crowd: the union of (a) and (b);
- (e) duplicates: the masked development view itself (its own seed, so the same noise, dropout and hidden
  ball) with a duplicate track: in runs of 2-5 s covering about a tenth of the half, the player nearest
  the ball gets a second track of the same team within 1 m of the first.

For each of (a)-(d), the matched random view hides the same hidden runs (the same lengths) at positions
drawn at random in the same half, at least one frame apart so that no two runs merge. It has the same
noise and dropout as its view; only where the ball is hidden differs.

The situations are read from the clean tracking (true positions, true ball and its height). Every frame
is treated alike, dead or live, as the masked development view treats it. The constants are D55's, set
by judgement and not tuned.
"""

import xml.etree.ElementTree as ET

import numpy as np

from dfl_submission import hidden_views as hv
from dfl_transfer_data import SECTIONS, match_metadata, number
from evaluation import possession_metrics as pm


FPS = 25
AIR_HEIGHT = 1.5                       # m; hidden when the height is above it
CROWD_PLAYERS, CROWD_RADIUS = 3, 3.0   # at least three players within 3 m of the ball
CHANGE_BEFORE, CHANGE_AFTER = FPS, 2 * FPS   # frames [t - 25, t + 50) around a change at frame t
DUPLICATE_RUN = (2 * FPS, 5 * FPS)     # 2-5 s, both ends included
DUPLICATE_SHARE = .1                   # about a tenth of the half
DUPLICATE_RADIUS = 1.0                 # m; the second track lies within 1 m of the first
SITUATIONS = ("air", "crowd", "change", "air_or_crowd")
VIEWS = (*SITUATIONS, "duplicates")
RANDOM = "random"                      # suffix of a matched random view's name
SEEDS = {"air": 930000, "crowd": 940000, "change": 950000, "air_or_crowd": 960000, "duplicates": 970000}
PLACEMENT = 5000                       # a matched random view's placement seed is its view's seed + 5000
MASKED_SEED = 270000                   # the masked development view's generator (`train.views`)
OVER_3S, OVER_10S = 3 * FPS, 10 * FPS  # "runs over 3 s / 10 s"
LIVE = (pm.H, pm.A)


# -- the height-aware reader ---------------------------------------------------------------------

def parse_ball(info_path, position_path, expected):
    """{section: {"frames", "ball", "z", "status"}} of the ball frames, as `parse_match` reads them plus Z.

    `ball` is float32 with NaN outside the pitch plus 5 m (as `parse_match`); `z` is float64 and NaN only on
    frames the file does not hold.
    """
    meta = match_metadata(info_path, expected)
    balls = {section: {} for section in SECTIONS}
    unknown, keys = set(), set()
    for event, node in ET.iterparse(position_path, events=("end",)):
        if node.tag.rsplit("}", 1)[-1] != "FrameSet":
            continue
        if node.get("TeamId", "").lower() == "ball":
            section = node.get("GameSection")
            if section not in SECTIONS:
                unknown.add(section)
            else:
                for frame in node:
                    if frame.tag.rsplit("}", 1)[-1] != "Frame":
                        continue
                    keys.update(frame.attrib)
                    index = int(frame.get("N"))
                    if index in balls[section]:
                        raise ValueError("Duplicate ball frame key")
                    balls[section][index] = (number(frame.get("X")), number(frame.get("Y")), number(frame.get("Z")),
                                             number(frame.get("BallStatus")))
        node.clear()
    if unknown:
        raise ValueError("Unexpected ball game section")
    if not {"N", "X", "Y", "Z", "BallStatus"} <= keys:
        raise ValueError("The ball frames carry no height (Z)")
    result = {}
    for section in SECTIONS:
        if not balls[section]:
            raise ValueError("Missing ball half")
        first, last = min(balls[section]), max(balls[section])
        data = np.full((last - first + 1, 4), np.nan)
        for frame, value in balls[section].items():
            data[frame - first] = value
        ball = data[:, :2].astype(np.float32)
        good = np.isfinite(ball).all(axis=1) & (np.abs(ball) <= meta["pitch"] / 2 + 5).all(axis=1)
        ball[~good] = np.nan
        result[section] = {"frames": np.arange(first, last + 1), "ball": ball, "z": data[:, 2], "status": data[:, 3]}
    return result


def reader_check(height, existing):
    """Per half: the new reader's frames, ball x and y and status equal the existing reader's, and Z's facts."""
    result = {}
    for section in SECTIONS:
        new, old = height[section], existing[section]
        z, status = new["z"], new["status"]
        recorded = np.isfinite(status)
        live, dead = status == 1, status == 0
        result[section] = {
            "frames": int(len(z)),
            "frames_equal": bool(np.array_equal(new["frames"], old["frames"])),
            "ball_xy_equal": bool(new["ball"].dtype == np.asarray(old["ball"]).dtype
                                  and np.array_equal(new["ball"], old["ball"], equal_nan=True)),
            "status_equal": bool(np.array_equal(status, old["status"], equal_nan=True)),
            "ball_records": int(recorded.sum()),
            "numeric_z_on_every_record": bool(np.isfinite(z[recorded]).all()),
            "dead_frames": int(dead.sum()), "dead_frames_z_zero": int((dead & (z == 0)).sum()),
            "live_frames": int(live.sum()),
            "live_share_above_air_height": float((live & (z > AIR_HEIGHT)).sum() / live.sum()) if live.any() else None}
    return result


def reader_passed(check):
    return bool(all(part["frames_equal"] and part["ball_xy_equal"] and part["status_equal"]
                    and part["numeric_z_on_every_record"] for part in check.values()))


# -- situations (from the clean tracking) --------------------------------------------------------

def air_mask(z):
    return np.nan_to_num(np.asarray(z, dtype=np.float64), nan=-np.inf) > AIR_HEIGHT


def crowd_mask(positions, ball):
    """Frames where at least three players (either team) are within 3 m of the ball."""
    positions = np.asarray(positions, dtype=np.float64)
    ball = np.asarray(ball, dtype=np.float64)
    distance = np.linalg.norm(positions - ball[:, None, :], axis=2)
    close = np.where(np.isfinite(distance), distance <= CROWD_RADIUS, False)
    return close.sum(axis=1) >= CROWD_PLAYERS


def ownership_changes(target):
    """Frames t where the target goes straight from one team to the other (inside live play)."""
    target = np.asarray(target)
    previous, current = target[:-1], target[1:]
    switch = np.isin(previous, LIVE) & np.isin(current, LIVE) & (previous != current)
    return np.flatnonzero(switch) + 1


def change_mask(target):
    target = np.asarray(target)
    hidden = np.zeros(len(target), dtype=bool)
    for frame in ownership_changes(target):
        hidden[max(0, frame - CHANGE_BEFORE):min(len(target), frame + CHANGE_AFTER)] = True
    return hidden


def runs_of(flags):
    """(starts, lengths) of the maximal runs of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flags, dtype=np.int8), 0]))
    return edges[::2], edges[1::2] - edges[::2]


def place_runs(lengths, n, rng):
    """(flags, starts, lengths), in time order: the runs in random order at random positions in `n` frames, at
    least one frame apart (so no two merge)."""
    lengths = np.asarray(lengths, dtype=np.int64)
    flags = np.zeros(n, dtype=bool)
    if not len(lengths):
        return flags, np.zeros(0, dtype=np.int64), lengths
    free = n - int(lengths.sum()) - (len(lengths) - 1)
    if free < 0:
        raise ValueError("The runs do not fit in the half")
    placed = lengths[rng.permutation(len(lengths))]
    cuts = np.sort(rng.integers(0, free + 1, size=len(lengths)))
    starts = cuts + np.r_[0, np.cumsum(placed)[:-1]] + np.arange(len(lengths))
    for start, length in zip(starts, placed):
        flags[start:start + length] = True
    return flags, starts, placed


def matched_random(hidden, rng):
    """The same hidden runs as `hidden`, placed at random in the same half."""
    _, lengths = runs_of(hidden)
    flags, _, _ = place_runs(lengths, len(hidden), rng)
    return flags


# -- views ---------------------------------------------------------------------------------------

def view_seed(view, match_index, half_index):
    return SEEDS[view] + 100 * match_index + half_index


def situation_masks(positions, ball, z, target):
    """The hidden frames of (a)-(d) for one half."""
    air, crowd = air_mask(z), crowd_mask(positions, ball)
    return {"air": air, "crowd": crowd, "change": change_mask(target), "air_or_crowd": air | crowd}


def situation_pair(positions, ball, hidden, seed):
    """((positions, ball) of the view, (positions, ball) of its matched random view, random hidden flags)."""
    rng = np.random.default_rng(seed)
    noisy_positions, noisy_ball = hv.noise_and_dropout(positions, ball, rng)
    view_ball = noisy_ball.copy()
    view_ball[hidden] = np.nan
    placed = matched_random(hidden, np.random.default_rng(seed + PLACEMENT))
    random_ball = noisy_ball.copy()
    random_ball[placed] = np.nan
    return (noisy_positions, view_ball), (noisy_positions.copy(), random_ball), placed


def duplicate_runs(n, rng):
    """(starts, lengths) in time order: runs of 2-5 s until they cover a tenth of the half, placed at random."""
    target, lengths = int(round(DUPLICATE_SHARE * n)), []
    while sum(lengths) < target:
        lengths.append(int(rng.integers(DUPLICATE_RUN[0], DUPLICATE_RUN[1] + 1)))
    _, starts, placed = place_runs(lengths, n, rng)
    return list(zip(starts.tolist(), placed.tolist()))


def add_duplicates(view_positions, clean_positions, clean_ball, teams, seed):
    """(positions, teams, runs): the view with one extra track per team, live only in that team's duplicate runs.

    In each run the player nearest the ball (clean tracking, at the first frame of the run where the ball and a
    player are seen) gets a second track: its own view track moved by a fixed offset of up to 1 m (uniform
    distance and direction), so the two tracks lie within 1 m of each other wherever the first is seen.
    """
    rng = np.random.default_rng(seed)
    view_positions = np.asarray(view_positions, dtype=np.float32)
    clean_positions = np.asarray(clean_positions, dtype=np.float32)
    clean_ball = np.asarray(clean_ball, dtype=np.float32)
    teams = np.asarray(teams)
    n = len(view_positions)
    extra = np.full((n, 2, 2), np.nan, dtype=np.float32)
    runs = []
    for start, length in duplicate_runs(n, rng):
        stop = start + length
        distance = np.linalg.norm(clean_positions[start:stop] - clean_ball[start:stop, None, :], axis=2)
        seen = np.isfinite(distance).any(axis=1)
        if not seen.any():
            runs.append({"start": int(start), "length": int(length), "player": None})
            continue
        frame = int(np.flatnonzero(seen)[0])
        player = int(np.nanargmin(distance[frame]))
        team = int(teams[player])
        angle, radius = rng.uniform(0, 2 * np.pi), rng.uniform(0, DUPLICATE_RADIUS)
        offset = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)
        extra[start:stop, team] = view_positions[start:stop, player] + offset
        runs.append({"start": int(start), "length": int(length), "player": player, "team": team,
                     "offset_m": float(np.hypot(*offset))})
    positions = np.concatenate([view_positions, extra], axis=1)
    return positions, np.r_[teams, np.array([0, 1], dtype=teams.dtype)], runs


# -- statistics ----------------------------------------------------------------------------------

def run_statistics(balls, labels=None):
    """Visibility, hidden runs (share of hidden frames in runs over 3 s and over 10 s, mean run) pooled over halves;
    with `labels`, also the visibility on live frames of the labels."""
    frames = sum(len(ball) for ball in balls)
    runs = np.concatenate([hv.hidden_runs(ball) for ball in balls]) if balls else np.zeros(0, dtype=np.int64)
    hidden = int(runs.sum())
    result = {"halves": len(balls), "frames": int(frames), "hidden_frames": hidden,
              "visibility": 1 - hidden / frames if frames else None, "hidden_runs": int(len(runs)),
              "mean_hidden_run_s": float(runs.mean() / FPS) if len(runs) else None,
              "hidden_share_in_runs_over_3s": float(runs[runs > OVER_3S].sum() / hidden) if hidden else None,
              "hidden_share_in_runs_over_10s": float(runs[runs > OVER_10S].sum() / hidden) if hidden else None}
    if labels is not None:
        live = np.concatenate([np.isin(np.asarray(item), LIVE) for item in labels])
        seen = np.concatenate([np.isfinite(np.asarray(ball, dtype=np.float64)).all(axis=1) for ball in balls])
        result |= {"live_frames": int(live.sum()), "live_hidden_frames": int((live & ~seen).sum()),
                   "live_visibility": float((live & seen).sum() / live.sum()) if live.any() else None}
    return result
