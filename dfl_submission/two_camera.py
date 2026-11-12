"""D31's synthetic two-camera harness (DECISIONS.md D31), on external tracking only.

`simulate` turns one external half's clean tracking (the truth, centred metres) into a broadcast
recording: one live offset, live segments broken by cuts, normal-speed and slow replays (sometimes
wrapped in wipes), a camera window following the ball, partial and noisy players, the broadcast
ball's own dropouts and a heavy-tailed position error, and an unknown orientation. Every parameter
is fixed by the rule; none is tuned. `frame_counts` measures an alignment against the truth.
"""

import numpy as np

from dfl_submission.broadcast_ball import orient


OFFSET_RANGE = 3000                 # live offset d0 uniform on [-3000, 3000] frames
LIVE_MIN, LIVE_MEAN = 125, 900      # live segment: 125 frames plus an exponential with mean 900
BREAKS = (("cut", .5), ("replay", .3), ("slow", .2))
CUT_FRAMES = (50, 500)
REPLAY_FRAMES = (75, 200)           # broadcast frames of a replay
REPLAY_LAG = (125, 375)             # a replay starts this many frames before the live time
WIPE_PROBABILITY, WIPE_FRAMES = .5, 25
WINDOW = 50                         # metres of pitch length the camera sees
SMOOTHING = 25                      # frames: time constant of the camera following the ball
DROP = (.1, 25)                     # a seen player is dropped for a block of 25 frames with probability 0.1
BIAS, NOISE = .5, .3                # metres: per-segment bias and per-frame noise, per axis
BALL_RUNS = (100, 25)               # mean seen and hidden run of the broadcast ball, frames
BALL_ERROR = (1.5, 5.0)             # median and 90th percentile of the ball's radial error, metres
KINDS = {"live": 0, "cut": 1, "replay": 2, "slow": 3, "wipe": 4, "off": 5}
Z90 = 1.2815515655446004            # standard normal 90th percentile


def timeline(n, rng):
    """(kind, shown tactical frame or NaN, segment id) for each of n broadcast frames, and the live offset."""
    offset = int(rng.integers(-OFFSET_RANGE, OFFSET_RANGE + 1))
    kind = np.empty(n, dtype=np.int8)
    shows = np.full(n, np.nan)
    segment = np.full(n, -1, dtype=np.int64)
    b, number = 0, 0
    names, weights = zip(*BREAKS)
    while b < n:
        live = LIVE_MIN + int(rng.exponential(LIVE_MEAN))
        stop = min(n, b + live)
        kind[b:stop], shows[b:stop], segment[b:stop] = KINDS["live"], np.arange(b, stop) + offset, number
        b, number = stop, number + 1
        if b >= n:
            break
        name = names[int(rng.choice(len(names), p=weights))]
        if name == "cut":
            stop = min(n, b + int(rng.integers(CUT_FRAMES[0], CUT_FRAMES[1] + 1)))
            kind[b:stop] = KINDS["cut"]
            b = stop
            continue
        length = int(rng.integers(REPLAY_FRAMES[0], REPLAY_FRAMES[1] + 1))
        lag = int(rng.integers(REPLAY_LAG[0], REPLAY_LAG[1] + 1))
        wiped = rng.random() < WIPE_PROBABILITY
        if wiped:
            stop = min(n, b + WIPE_FRAMES)
            kind[b:stop] = KINDS["wipe"]
            b = stop
        stop = min(n, b + length)
        first = b + offset - lag
        speed = 1.0 if name == "replay" else .5
        kind[b:stop], shows[b:stop] = KINDS[name], first + speed * np.arange(stop - b)
        segment[b:stop] = number
        b, number = stop, number + 1
        if wiped:
            stop = min(n, b + WIPE_FRAMES)
            kind[b:stop] = KINDS["wipe"]
            b = stop
    return kind, shows, segment, offset


def camera_centre(ball):
    """The camera's x on the tactical timeline: the ball's x smoothed exponentially, held while unseen."""
    centre = np.empty(len(ball))
    seen = np.isfinite(ball).all(axis=1)
    current = float(ball[np.argmax(seen), 0]) if seen.any() else 0.0
    for frame in range(len(ball)):
        if seen[frame]:
            current += (float(ball[frame, 0]) - current) / SMOOTHING
        centre[frame] = current
    return centre


def ball_switch(n, rng):
    """A two-state seen/hidden switch with mean runs BALL_RUNS, started from its stationary share."""
    seen_run, hidden_run = BALL_RUNS
    state = rng.random() < seen_run / (seen_run + hidden_run)
    draws = rng.random(n)
    result = np.empty(n, dtype=bool)
    for frame in range(n):
        result[frame] = state
        state = draws[frame] >= 1 / seen_run if state else draws[frame] < 1 / hidden_run
    return result


def simulate(positions, ball, teams, seed):
    """A broadcast recording of one half's clean tracking; see the module docstring."""
    rng = np.random.default_rng(seed)
    positions, ball = np.asarray(positions, dtype=np.float64), np.asarray(ball, dtype=np.float64)
    n, players = positions.shape[:2]
    kind, shows, segment, offset = timeline(n, rng)
    orientation = int(rng.integers(4))
    inside = np.isfinite(shows) & (shows >= 0) & (np.floor(shows + .5) < n)
    kind[np.isfinite(shows) & ~inside] = KINDS["off"]
    frame = np.where(inside, np.floor(np.where(inside, shows, 0) + .5), 0).astype(np.int64)
    valid = inside.copy()
    centre = camera_centre(ball)[frame]
    true = positions[frame]
    seen = valid[:, None] & np.isfinite(true).all(axis=2) & (np.abs(true[:, :, 0] - centre[:, None]) <= WINDOW / 2)
    blocks = np.arange(n) // DROP[1]
    dropped = rng.random((blocks[-1] + 1, players)) < DROP[0]
    seen &= ~dropped[blocks]
    bias = rng.normal(0, BIAS, (int(segment.max()) + 2, 2))[segment]
    shown = true + bias[:, None, :] + rng.normal(0, NOISE, true.shape)
    shown[~seen] = np.nan
    ball_true = ball[frame]
    ball_seen = (valid & np.isfinite(ball_true).all(axis=1) & (np.abs(ball_true[:, 0] - centre) <= WINDOW / 2)
                 & ball_switch(n, rng))
    median, p90 = BALL_ERROR
    radius = np.exp(np.log(median) + (np.log(p90) - np.log(median)) / Z90 * rng.standard_normal(n))
    angle = rng.uniform(0, 2 * np.pi, n)
    broadcast_ball = ball_true + radius[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    broadcast_ball[~ball_seen] = np.nan
    return {"positions": orient(shown, orientation).astype(np.float32),
            "ball": orient(broadcast_ball, orientation).astype(np.float32),
            "teams": np.asarray(teams), "valid": valid, "shows": np.where(inside, shows, np.nan), "kind": kind,
            "offset": offset, "orientation": orientation}


def frame_counts(results, shows, orientation):
    """Broadcast frames in accepted stretches, and how many of them are wrong (D31's constant setting).

    A frame is wrong if its mapped tactical frame is more than 2 frames from the frame it shows
    (rounded half up) or its stretch has the wrong orientation.
    """
    accepted = wrong = 0
    for result in results:
        if not result["accepted"]:
            continue
        frames = np.arange(result["start"], result["stop"])
        accepted += len(frames)
        if result["orientation"] != orientation:
            wrong += len(frames)
            continue
        true = np.floor(np.asarray(shows)[frames] + .5)
        wrong += int((~np.isfinite(true) | (np.abs(frames + result["offset"] - true) > 2)).sum())
    return {"accepted": int(accepted), "wrong": int(wrong)}
