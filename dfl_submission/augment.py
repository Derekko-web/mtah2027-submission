"""Training augmentation and guard views that remove external dead-ball shortcuts (DECISIONS.md D2).

External providers mark dead ball in ways the competition tracking does not: DFL freezes the ball
and SkillCorner rarely detects it. Training halves get simulated dead-ball handling, a wide range
of ball visibility and cross-team identity swaps. Guard views use different settings, so the
evaluation does not test the model on its own simulation.
"""

import numpy as np

from dfl_submission import external_benchmark as benchmark, fusion_trial, skillcorner
from dfl_submission.robust_features import stress
from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN
from dfl_transfer_data import SECTIONS
from evaluation import possession_metrics as pm


FPS = 25
KEEP, REMOVE = .2, .2
TRAVEL_SHARE, REST_SHARE, REST_CAP = .3, .2, FPS
OFFSET, NOISE = .5, .2
VISIBILITY = (.3, .9)
HIDDEN_RUN = 50
TRAIN_SWAP_EVERY, TRAIN_SWAP_SECONDS = 12 * FPS, (2, 10)
GUARD_SWAP_EVERY, GUARD_SWAP_FRAMES = 10 * FPS, 10 * FPS


def dead_runs(labels):
    dead = np.r_[False, labels == pm.D, False]
    edges = np.flatnonzero(dead[1:] != dead[:-1])
    return zip(edges[::2], edges[1::2])


def observed(ball, index, step):
    while 0 <= index < len(ball):
        if np.isfinite(ball[index]).all():
            return ball[index]
        index += step
    return None


def handle_dead_ball(labels, positions, ball, rng):
    """Replace the provider's dead-ball artifact with simulated handling, run by run."""
    result = np.array(ball, dtype=np.float32, copy=True)
    for start, stop in dead_runs(labels):
        mode = rng.random()
        if mode < KEEP:
            continue
        if mode < KEEP + REMOVE:
            result[start:stop] = np.nan
            continue
        stopped = observed(ball, start, 1 if np.isfinite(ball[start]).all() else -1)
        restart = observed(ball, stop, 1)
        reference = min(stop, len(ball) - 1)
        if stopped is None or restart is None:
            continue
        distance = np.linalg.norm(positions[reference] - restart, axis=1)
        taker = int(np.nanargmin(distance)) if np.isfinite(distance).any() else None
        length = stop - start
        travel = max(1, int(TRAVEL_SHARE * length))
        rest = min(REST_CAP, max(1, int(REST_SHARE * length)))
        path = np.empty((length, 2), dtype=np.float32)
        offset = OFFSET * rng.standard_normal(2).astype(np.float32)
        carrier = (positions[start:stop, taker] + offset if taker is not None
                   else np.full((length, 2), np.nan, dtype=np.float32))
        arrival = carrier[min(travel, length - 1)]
        if not np.isfinite(arrival).all():
            arrival = restart
        fraction = (np.arange(travel) + 1)[:, None] / travel
        path[:travel] = stopped + fraction * (arrival - stopped)
        path[travel:] = carrier[travel:]
        # Without a visible taker, move the ball in a straight line towards the restart spot.
        straight = stopped + (np.arange(length)[:, None] / max(1, length - rest)).clip(0, 1) * (restart - stopped)
        unseen = ~np.isfinite(path).all(axis=1)
        path[unseen] = straight[unseen]
        path[length - rest:] = restart
        path += rng.normal(0, NOISE, path.shape).astype(np.float32)
        result[start:stop] = path
    return result


def wide_masking(positions, ball, rng):
    """Degraded view with per-half ball visibility drawn from a wide range (D2 change 2)."""
    positions = np.array(positions, dtype=np.float32, copy=True)
    ball = np.array(ball, dtype=np.float32, copy=True)
    n = len(ball)
    positions += rng.normal(0, .2, positions.shape).astype(np.float32)
    ball += rng.normal(0, .2, ball.shape).astype(np.float32)
    drops = rng.random(((n + 49) // 50, positions.shape[1])) < .2
    positions[np.repeat(drops, 50, axis=0)[:n]] = np.nan
    visibility = rng.uniform(*VISIBILITY)
    reveal = 1 / HIDDEN_RUN
    hide = reveal * (1 - visibility) / visibility
    visible = bool(rng.random() < visibility)
    switches = rng.random(n)
    for i in range(n):
        if switches[i] < (hide if visible else reveal):
            visible = not visible
        if not visible:
            ball[i] = np.nan
    return positions, ball


def swap_teams(positions, teams, rng, *, every, frames):
    """Exchange a home and an away track for a while, as a cross-team identity switch."""
    positions = np.array(positions, dtype=np.float32, copy=True)
    home, away = np.flatnonzero(np.asarray(teams) == 0), np.flatnonzero(np.asarray(teams) == 1)
    start = int(rng.exponential(every))
    while start < len(positions):
        length = frames(rng)
        stop = min(len(positions), start + length)
        seen = np.isfinite(positions[start]).all(axis=1)
        a, b = home[seen[home]], away[seen[away]]
        if len(a) and len(b):
            i, j = rng.choice(a), rng.choice(b)
            positions[start:stop, [i, j]] = positions[start:stop, [j, i]]
        start += max(1, int(rng.exponential(every)))
    return positions


def training_frames(rng):
    return int(rng.uniform(*TRAIN_SWAP_SECONDS) * FPS)


def training_halves(dfl, skill, *, raw=False):
    """Augmented external training halves, in the same form as external_benchmark.halves."""
    for match in TRAIN:
        index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(match)
        source = fusion_trial.external_halves(dfl, match, raw=raw)
        for i, name in enumerate(SECTIONS):
            item = source[name]
            labels = pm.label_codes(item["labels"])
            rng = np.random.default_rng(520000 + 100 * index + i)
            positions = np.asarray(item["positions"], dtype=np.float32)
            ball = handle_dead_ball(labels, positions, item["ball"], rng)
            yield benchmark.Half("dfl", match, "clean", i, labels, positions, ball, item["teams"], item["pitch"])
            masked_positions, masked_ball = wide_masking(positions, ball, rng)
            masked_positions = swap_teams(masked_positions, item["teams"], rng, every=TRAIN_SWAP_EVERY,
                                          frames=training_frames)
            yield benchmark.Half("dfl", match, "masked", i, labels, masked_positions, masked_ball, item["teams"],
                                 item["pitch"])
            stress_positions, stress_ball = stress(positions, ball, 370000 + 100 * index + i)
            stress_positions = swap_teams(stress_positions, item["teams"], rng, every=TRAIN_SWAP_EVERY,
                                          frames=training_frames)
            yield benchmark.Half("dfl", match, "stress", i, labels, stress_positions, stress_ball, item["teams"],
                                 item["pitch"])
    for half in benchmark.halves(dfl, skill, "train", raw=raw):
        if half.source != "skillcorner":
            continue
        rng = np.random.default_rng(530000 + 10 * skillcorner.TRAIN.index(int(half.match)) + half.index)
        ball = handle_dead_ball(half.labels, np.asarray(half.positions, dtype=np.float32), half.ball, rng)
        yield benchmark.Half(half.source, half.match, half.view, half.index, half.labels, half.positions, ball,
                             half.teams, half.pitch)


def nearest_player(positions, ball):
    distance = np.linalg.norm(positions - ball[:, None, :], axis=2)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    return positions[np.arange(len(ball)), np.argmin(distance, axis=1)]


def guard_halves(dfl, skill, split, *, raw=False):
    """Standard external views plus the two D2 guard views built from the DFL masked view."""
    clean = {}
    for half in benchmark.halves(dfl, skill, split, raw=raw):
        yield half
        if half.source != "dfl":
            continue
        if half.view == "clean":
            clean[half.match, half.index] = half
            continue
        reference = clean[half.match, half.index]
        index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(half.match)
        rng = np.random.default_rng(610000 + 100 * index + half.index)
        carried = np.array(half.ball, dtype=np.float32, copy=True)
        visible = np.isfinite(carried).all(axis=1) & (half.labels == pm.D)
        moved = nearest_player(np.asarray(reference.positions, dtype=np.float32),
                               np.asarray(reference.ball, dtype=np.float32))
        carried[visible] = moved[visible] + rng.normal(0, NOISE, (int(visible.sum()), 2)).astype(np.float32)
        yield benchmark.Half("dfl", half.match, "dead_carried", half.index, half.labels, half.positions, carried,
                             half.teams, half.pitch)
        swapped = swap_teams(half.positions, half.teams, rng, every=GUARD_SWAP_EVERY,
                             frames=lambda _: GUARD_SWAP_FRAMES)
        yield benchmark.Half("dfl", half.match, "team_swap", half.index, half.labels, swapped, half.ball,
                             half.teams, half.pitch)

