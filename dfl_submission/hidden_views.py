"""D49: long hidden-ball runs (DECISIONS.md D49): the long hide process, the training views, the guard views.

Every DFL training view hides the ball with `augment.wide_masking`'s two-state process, whose hidden
runs last 2 s on average. D49 adds one training view per training half whose hidden runs are, with
probability 1/2, geometric with mean 50 frames and otherwise uniform on 125-750 frames (5-30 s); a
visible run is geometric with mean 243.75 * v / (1 - v) frames, so the expected visibility is v.

- DFL training halves: the set's own base tracking (the provider's for the original set; the
  dead-ball-handled ball for the augmented set) with `wide_masking`'s coordinate noise and player
  dropout and the long hide process in place of its two-state process, v uniform on (0.3, 0.9);
  for the augmented set optionally followed by its training identity swaps.
- SkillCorner training halves: the long hide process applied to the detected ball, v uniform on
  (0.5, 0.9), nothing else changed.
- Long-hidden guard views (development only): built like the masked development view (`corrupt`'s
  coordinate noise and player dropout), with hidden runs uniform on 200-500 frames and visible runs
  uniform on 300-750 frames, starting visible, from new seeds.

The constants are set in the rule by judgement and are not tuned.
"""

import numpy as np

from dfl_submission import augment
from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN


FPS = 25
VIEW = "long_hidden"
DFL_VISIBILITY = augment.VISIBILITY        # (0.3, 0.9), as in wide_masking
SKILL_VISIBILITY = (.5, .9)
SHORT_RUN = 50                             # geometric hidden runs, mean 2 s
LONG_RUN = (125, 750)                      # uniform hidden runs, 5-30 s, both ends included
LONG_SHARE = .5
MEAN_HIDDEN = (1 - LONG_SHARE) * SHORT_RUN + LONG_SHARE * (LONG_RUN[0] + LONG_RUN[1]) / 2   # 243.75 frames
NOISE, DROP, DROP_BLOCK = .2, .2, 50       # wide_masking's and corrupt's coordinate noise and player dropout
GUARD_HIDDEN = (200, 500)                  # 8-20 s, both ends included
GUARD_VISIBLE = (300, 750)                 # 12-30 s, both ends included
LONG = 10 * FPS                            # "runs over 10 s", for the view statistics
SEEDS = {"current_dfl": 700000, "augmented_dfl": 710000, "current_skill": 740000, "augmented_skill": 750000,
         "guard_dfl": 770000, "guard_metrica": 780000}
DFL_ORDER = (*TRAIN, DEVELOPMENT, *HOLDOUT)


# -- hide processes --------------------------------------------------------------------------------

def hidden_length(rng):
    if rng.random() < LONG_SHARE:
        return int(rng.geometric(1 / SHORT_RUN))
    return int(rng.integers(LONG_RUN[0], LONG_RUN[1] + 1))


def long_hide(n, visibility, rng):
    """Visible flags of `n` frames under D49's long hide process with expected visibility `visibility`."""
    if not 0 < visibility < 1:
        raise ValueError("visibility must lie strictly between 0 and 1")
    mean_visible = MEAN_HIDDEN * visibility / (1 - visibility)
    visible = np.empty(n, dtype=bool)
    state = bool(rng.random() < visibility)
    start = 0
    while start < n:
        length = int(rng.geometric(1 / mean_visible)) if state else hidden_length(rng)
        visible[start:start + length] = state
        start += length
        state = not state
    return visible


def guard_hide(n, rng):
    """Visible flags of the long-hidden guard views: uniform visible and hidden runs, starting visible."""
    visible = np.empty(n, dtype=bool)
    state, start = True, 0
    while start < n:
        low, high = GUARD_VISIBLE if state else GUARD_HIDDEN
        length = int(rng.integers(low, high + 1))
        visible[start:start + length] = state
        start += length
        state = not state
    return visible


def noise_and_dropout(positions, ball, rng):
    """`wide_masking`'s (and `corrupt`'s) coordinate noise and 2-second player dropout, same draw order."""
    positions = np.array(positions, dtype=np.float32, copy=True)
    ball = np.array(ball, dtype=np.float32, copy=True)
    n = len(ball)
    positions += rng.normal(0, NOISE, positions.shape).astype(np.float32)
    ball += rng.normal(0, NOISE, ball.shape).astype(np.float32)
    drops = rng.random(((n + DROP_BLOCK - 1) // DROP_BLOCK, positions.shape[1])) < DROP
    positions[np.repeat(drops, DROP_BLOCK, axis=0)[:n]] = np.nan
    return positions, ball


# -- views -----------------------------------------------------------------------------------------

def dfl_view(positions, ball, rng):
    """A DFL training half under `wide_masking` with the long hide process; returns positions, ball, v."""
    positions, ball = noise_and_dropout(positions, ball, rng)
    visibility = float(rng.uniform(*DFL_VISIBILITY))
    ball[~long_hide(len(ball), visibility, rng)] = np.nan
    return positions, ball, visibility


def skill_view(ball, rng):
    """The long hide process on a SkillCorner half's detected ball; returns ball, v."""
    ball = np.array(ball, dtype=np.float32, copy=True)
    visibility = float(rng.uniform(*SKILL_VISIBILITY))
    ball[~long_hide(len(ball), visibility, rng)] = np.nan
    return ball, visibility


def guard_view(positions, ball, seed):
    """A long-hidden guard view: built like the masked development view, with the guard hide process."""
    rng = np.random.default_rng(seed)
    positions, ball = noise_and_dropout(positions, ball, rng)
    ball[~guard_hide(len(ball), rng)] = np.nan
    return positions, ball


def dfl_seed(model_set, match, index):
    return SEEDS[f"{model_set}_dfl"] + 100 * DFL_ORDER.index(match) + index


def skill_seed(model_set, order, index):
    return SEEDS[f"{model_set}_skill"] + 10 * order + index


def guard_dfl_seed(match, index):
    return SEEDS["guard_dfl"] + 100 * DFL_ORDER.index(match) + index


def guard_metrica_seed(game, index):
    return SEEDS["guard_metrica"] + 10 * game + index


def training_view(model_set, half, *, swaps=True, provider_ball=None, skill_order=None):
    """The D49 view of one training half (a `external_benchmark.Half`), or None for halves that get none.

    DFL: from the set's clean half. SkillCorner: from the set's half, with `provider_ball` (the
    provider's detected ball) replacing the half's ball when given. `swaps` applies the augmented set's
    training identity swaps after the masking (augmented set only).
    """
    from dataclasses import replace
    if half.source == "dfl":
        if half.view != "clean":
            return None
        rng = np.random.default_rng(dfl_seed(model_set, half.match, half.index))
        positions, ball, visibility = dfl_view(half.positions, half.ball, rng)
        if model_set == "augmented" and swaps:
            positions = augment.swap_teams(positions, half.teams, rng, every=augment.TRAIN_SWAP_EVERY,
                                           frames=augment.training_frames)
        return replace(half, view=VIEW, positions=positions, ball=ball), visibility
    if half.source == "skillcorner":
        if skill_order is None:
            raise ValueError("A SkillCorner half needs its order among the SkillCorner matches")
        rng = np.random.default_rng(skill_seed(model_set, skill_order, half.index))
        ball, visibility = skill_view(half.ball if provider_ball is None else provider_ball, rng)
        return replace(half, view=VIEW, ball=ball), visibility
    raise ValueError(f"Unknown training source {half.source!r}")


# -- statistics ------------------------------------------------------------------------------------

def hidden_runs(ball):
    """Lengths of the maximal runs of frames without a ball position."""
    hidden = ~np.isfinite(np.asarray(ball, dtype=np.float64)).all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[0, hidden.astype(np.int8), 0]))
    return edges[1::2] - edges[::2]


def view_statistics(balls):
    """Visibility and the share of hidden frames lying in runs over 10 s, pooled over halves."""
    frames = sum(len(ball) for ball in balls)
    runs = np.concatenate([hidden_runs(ball) for ball in balls]) if balls else np.zeros(0, dtype=np.int64)
    hidden = int(runs.sum())
    return {"halves": len(balls), "frames": int(frames), "visibility": 1 - hidden / frames if frames else None,
            "hidden_frames": hidden, "hidden_share_in_runs_over_10s": float(runs[runs > LONG].sum() / hidden) if hidden else None,
            "hidden_runs": int(len(runs)), "mean_hidden_run_s": float(runs.mean() / FPS) if len(runs) else None}
