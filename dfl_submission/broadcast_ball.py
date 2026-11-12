"""D31: map broadcast stretches onto the tactical timeline and merge the broadcast ball (DECISIONS.md D31).

All positions are in metres, centred on the pitch, so an orientation is a sign flip of x (bit 1)
and/or y (bit 2). A *stretch* is a maximal run of usable broadcast frames (valid homography, at
least 4 players per team) of at least 5 s. Each stretch gets one frame offset (tactical frame =
broadcast frame + offset) and one orientation: a coarse search on a 25-frame grid in all four
orientations, then a fine search within ±25 frames. A stretch is accepted only if its fine cost is
low, clearly lower than any distant or differently oriented alternative, and not tied. The merge
fills tactical frames whose ball is missing with the broadcast ball of accepted stretches; it never
overwrites a seen tactical ball and never touches the players.
"""

import numpy as np


MIN_STRETCH = 125      # frames (5 s)
COARSE_STEP = 25       # frames between coarse samples and between tactical grid frames
COARSE_SAMPLES = 120
FINE_RADIUS = 25
FINE_STEP = 5
FINE_SAMPLES = 600
AWAY = 50              # frames: the margin's competitors lie further than this from the best offset
TIE = 1e-3             # metres: another fine offset this close to the minimum is a tie
CAP = 10.0             # metres: one player's error is capped here, as in formation_cost
MIN_PLAYERS = 4
TRIM = .25             # D31a: the share of each team's worst player errors left out of the cost


def orient(points, orientation):
    """Points with x negated if bit 1 of `orientation` is set and y negated if bit 2 is set."""
    result = np.array(points, copy=True)
    if orientation & 1:
        result[..., 0] = -result[..., 0]
    if orientation & 2:
        result[..., 1] = -result[..., 1]
    return result


def usable(positions, teams, valid):
    """Frames with a valid homography and at least 4 seen players in each team."""
    finite = np.isfinite(positions).all(axis=2)
    result = np.array(valid, dtype=bool, copy=True)
    for team in (0, 1):
        result &= finite[:, np.asarray(teams) == team].sum(axis=1) >= MIN_PLAYERS
    return result


def stretches(mask, minimum=MIN_STRETCH):
    """[(start, stop)] of the runs of True at least `minimum` frames long."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= minimum]


def pack(positions, teams):
    """Each team's seen players moved to its first slots, frame by frame, and empty slots dropped.

    A cost depends only on the set of seen players of each team in a frame, so packing changes no
    cost; it only spares the search the empty slots of substitutes and lost tracks.
    """
    positions, teams = np.asarray(positions, dtype=np.float64), np.asarray(teams)
    blocks, labels = [], []
    for team in (0, 1):
        own = positions[:, teams == team]
        seen = np.isfinite(own).all(axis=2)
        width = int(seen.sum(axis=1).max()) if own.shape[1] else 0
        order = np.argsort(~seen, axis=1, kind="stable")[:, :width]
        packed = np.take_along_axis(own, order[:, :, None], axis=1)
        packed[~np.take_along_axis(seen, order, axis=1)] = np.nan
        blocks.append(packed)
        labels.append(np.full(width, team))
    return np.concatenate(blocks, axis=1), np.concatenate(labels)


def frame_costs(frame, frame_teams, tactical, tactical_teams, trim=0.0, shift=None):
    """Same-team formation error of one broadcast frame against every tactical frame given.

    D31a: `trim` leaves out that share of each team's largest player errors (rounded down, so the
    kept share is rounded up), and `shift` (2, or one row per tactical frame) is subtracted from
    the broadcast players before they are matched.
    """
    frame, tactical = np.asarray(frame, dtype=np.float64), np.asarray(tactical, dtype=np.float64)
    frame_teams, tactical_teams = np.asarray(frame_teams), np.asarray(tactical_teams)
    errors = []
    for team in (0, 1):
        players = frame[frame_teams == team]
        players = players[np.isfinite(players).all(axis=1)]
        if len(players) < MIN_PLAYERS:
            return np.full(len(tactical), np.inf)
        candidates = tactical[:, tactical_teams == team]
        shown = players[None] if shift is None else players[None] - np.asarray(shift, dtype=np.float64).reshape(-1, 1, 2)
        distance = np.sqrt(((candidates[:, None, :, :] - shown[:, :, None, :]) ** 2).sum(axis=3))
        distance = np.where(np.isfinite(distance), distance, np.inf)
        nearest = (np.minimum(distance.min(axis=2), CAP) if candidates.shape[1]
                   else np.full((len(tactical), len(players)), CAP))
        if trim:
            nearest = np.sort(nearest, axis=1)[:, :int(np.ceil((1 - trim) * nearest.shape[1]))]
        error = (nearest ** 2).mean(axis=1)
        error[np.isfinite(candidates).all(axis=2).sum(axis=1) < MIN_PLAYERS] = np.inf
        errors.append(error)
    return np.sqrt((errors[0] + errors[1]) / 2)


def displacements(frame, frame_teams, tactical, tactical_teams, trim=0.0):
    """(n, players, 2): each kept broadcast player minus its nearest tactical teammate, NaN where not kept (D31a)."""
    frame, tactical = np.asarray(frame, dtype=np.float64), np.asarray(tactical, dtype=np.float64)
    frame_teams, tactical_teams = np.asarray(frame_teams), np.asarray(tactical_teams)
    parts = []
    for team in (0, 1):
        players = frame[frame_teams == team]
        players = players[np.isfinite(players).all(axis=1)]
        candidates = tactical[:, tactical_teams == team]
        if len(players) == 0 or candidates.shape[1] == 0:
            continue
        difference = players[None, :, None, :] - candidates[:, None, :, :]
        distance = np.sqrt((difference ** 2).sum(axis=3))
        distance = np.where(np.isfinite(distance), distance, np.inf)
        index = distance.argmin(axis=2)
        nearest = np.take_along_axis(distance, index[:, :, None], axis=2)[:, :, 0]
        shift = np.take_along_axis(difference, index[:, :, None, None], axis=2)[:, :, 0, :]
        kept = nearest <= CAP
        if trim:
            rank = np.argsort(np.argsort(nearest, axis=1, kind="stable"), axis=1, kind="stable")
            kept &= rank < int(np.ceil((1 - trim) * len(players)))
        shift[~kept] = np.nan
        parts.append(shift)
    return np.concatenate(parts, axis=1) if parts else np.full((len(tactical), 0, 2), np.nan)


def thinned(indices, limit):
    """At most `limit` of `indices`, evenly spread and in order."""
    if len(indices) <= limit:
        return indices
    return indices[np.unique(np.linspace(0, len(indices) - 1, limit).round().astype(int))]


def mean_finite(rows):
    """Column means over finite entries; infinite where fewer than half of a column is finite."""
    finite = np.isfinite(rows)
    count = finite.sum(axis=0)
    mean = np.where(finite, rows, 0.).sum(axis=0) / np.maximum(count, 1)
    return np.where(count * 2 >= len(rows), mean, np.inf)


def decide(result, constants):
    return bool(np.isfinite(result["cost"]) and result["cost"] <= constants["max_cost"]
                and result["margin"] >= constants["min_margin"] and not result["tie"])


def align_stretch(tactical, tactical_teams, broadcast, broadcast_teams, start, stop, constants=None, *, trim=0.0,
                  translate=False):
    """The offset, orientation, shift, cost, margin and decision for broadcast frames [start, stop).

    D31a: `trim` is used throughout; with `translate`, each fine candidate first estimates a shift
    (the median displacement of kept players from their nearest tactical teammates) and is costed
    after removing it.
    """
    result = {"start": int(start), "stop": int(stop), "offset": None, "orientation": None, "shift": [0.0, 0.0],
              "cost": float("inf"), "margin": float("-inf"), "tie": False, "accepted": False}
    n = len(tactical)
    grid = np.arange(0, n, COARSE_STEP)
    js = thinned(np.arange(0, (stop - start + COARSE_STEP - 1) // COARSE_STEP), COARSE_SAMPLES)
    grid_frames = tactical[grid]
    span = int(js.max())
    ks = np.arange(-span, len(grid))
    coarse = np.full((4, len(ks)), np.inf)
    for orientation in range(4):
        costs = np.stack([frame_costs(orient(broadcast[start + COARSE_STEP * j], orientation), broadcast_teams,
                                      grid_frames, tactical_teams, trim) for j in js])
        padded = np.concatenate([np.full((len(js), span), np.inf), costs, np.full((len(js), span), np.inf)], axis=1)
        coarse[orientation] = mean_finite(padded[np.arange(len(js))[:, None], ks[None, :] + js[:, None] + span])
    if not np.isfinite(coarse).any():
        return result
    best_orientation, best_k = np.unravel_index(int(np.argmin(coarse)), coarse.shape)
    centre = int(ks[best_k]) * COARSE_STEP - start
    candidates = centre + np.arange(-FINE_RADIUS, FINE_RADIUS + 1)
    frames = thinned(np.arange(start, stop, FINE_STEP), FINE_SAMPLES)
    shifts = np.zeros((len(candidates), 2))
    if translate:
        found = [[] for _ in candidates]
        for frame in frames:
            targets = frame + candidates
            inside = np.flatnonzero((targets >= 0) & (targets < n))
            if len(inside):
                moved = displacements(orient(broadcast[frame], best_orientation), broadcast_teams,
                                      tactical[targets[inside]], tactical_teams, trim)
                for row, column in enumerate(inside):
                    found[column].append(moved[row][np.isfinite(moved[row]).all(axis=1)])
        for column, parts in enumerate(found):
            pooled = np.concatenate(parts) if parts else np.empty((0, 2))
            if len(pooled):
                shifts[column] = np.median(pooled, axis=0)
    fine = np.full((len(frames), len(candidates)), np.inf)
    for row, frame in enumerate(frames):
        targets = frame + candidates
        inside = (targets >= 0) & (targets < n)
        if inside.any():
            fine[row, inside] = frame_costs(orient(broadcast[frame], best_orientation), broadcast_teams,
                                            tactical[targets[inside]], tactical_teams, trim,
                                            shift=shifts[inside] if translate else None)
    scores = mean_finite(fine)
    best = int(np.argmin(scores))
    if not np.isfinite(scores[best]):
        return result
    offset = int(candidates[best])
    others = np.delete(scores, best)
    away = np.abs(ks * COARSE_STEP - (start + offset)) > AWAY
    competitors = np.r_[np.delete(coarse, best_orientation, axis=0).ravel(), coarse[best_orientation][away]]
    runner = competitors.min() if len(competitors) else np.inf
    result.update(offset=offset, orientation=int(best_orientation), cost=float(scores[best]),
                  shift=[float(value) for value in shifts[best]],
                  margin=float(runner - scores[best]), tie=bool((others <= scores[best] + TIE).any()))
    if constants is not None:
        result["accepted"] = decide(result, constants)
    return result


def align(tactical, tactical_teams, broadcast, broadcast_teams, valid, constants=None, progress=None, *, trim=0.0,
          translate=False):
    """align_stretch on every stretch of usable broadcast frames, in order."""
    tactical, tactical_teams = pack(tactical, tactical_teams)
    results = []
    found = stretches(usable(broadcast, broadcast_teams, valid))
    for index, (start, stop) in enumerate(found):
        results.append(align_stretch(tactical, tactical_teams, broadcast, broadcast_teams, start, stop, constants,
                                     trim=trim, translate=translate))
        if progress is not None:
            progress(index + 1, len(found), results[-1])
    return results


def merge(tactical_ball, broadcast_ball, results, missing=None):
    """(merged ball, filled mask): missing tactical balls filled from accepted stretches, earliest first."""
    merged = np.array(tactical_ball, copy=True)
    n = len(merged)
    missing = ~np.isfinite(merged).all(axis=1) if missing is None else np.asarray(missing, dtype=bool)
    filled = np.zeros(n, dtype=bool)
    seen = np.isfinite(broadcast_ball).all(axis=1)
    for result in sorted((r for r in results if r["accepted"]), key=lambda r: r["start"]):
        frames = np.arange(result["start"], result["stop"])
        targets = frames + result["offset"]
        keep = (targets >= 0) & (targets < n) & seen[frames]
        frames, targets = frames[keep], targets[keep]
        take = missing[targets] & ~filled[targets]
        shown = orient(np.asarray(broadcast_ball)[frames[take]], result["orientation"])
        merged[targets[take]] = shown - np.asarray(result.get("shift", (0.0, 0.0)), dtype=np.float64)
        filled[targets[take]] = True
    return merged, filled
