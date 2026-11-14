"""D62: Smart Tagging's phase teams as weak ownership evidence where the ball is long hidden (DECISIONS.md D62).

Pure functions on one half's 25 Hz frames; nothing here reads a file.

- `simulated_phases`: phases simulated from a half's N = 3 target under a profile (c, a, b, q). Every true
  possession (a maximal run of H or of A inside a graded run of the target) lasting at least 5 s (125 frames) is a
  candidate. For the n candidates, in time order, four vectors are drawn from the half's generator in this order:
  kept (uniform [0, 1) < c), delay (uniform [0, a) s), end offset (uniform [-1, b) s) and wrong team (uniform
  [0, 1) < q); every vector is drawn whole, so a candidate's draws do not depend on whether earlier ones were kept.
  A kept phase starts at the true start plus the delay and ends at the true end plus the offset, each rounded to
  the nearest frame (`np.rint`), clipped to the half's frames [0, n), and is dropped if shorter than 2 s
  (50 frames); with the wrong-team draw it names the other team.
- `tag_state`: +1 (home) where at least one interval covering the frame names the home team and none names the
  away team, -1 for the reverse, 0 otherwise. Intervals are half-open frame spans (start, end, team) with team +1 or
  -1; how they were made does not matter, so the same function serves real tags.
- `far_mask`: frames whose nearest ball sighting, earlier or later in the half, is more than 3 s (75 frames) away
  (D59's HT mask, `tree_share_trial.far_from_sighting`, imported); every frame if the half has no sighting.
- `carried_state`: the Carry profile's replacement on masked frames: the team that truly had the ball (N = 3
  target) at the last sighting before the hidden run, 0 if the target there is D or X, or if the half has no
  sighting before the frame.
- `shift_home`: on masked frames with a nonzero state, home' = expit(logit(clip(home, .001, .999)) + w * state), in
  float64 and stored back in the array's own dtype; every other frame keeps its value bit for bit. With w = 0 the
  array is returned unchanged (a copy).
- `agreement`: (frames whose tag team equals the decode's team, frames counted) over frames where the ball is seen,
  the decode is H or A and the state is not 0.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

from dfl_submission.tree_share_trial import GAP, far_from_sighting
from evaluation import possession_metrics as pm


FPS = 25
MIN_POSSESSION = 5 * FPS          # "each true possession lasting at least 5 s"
MIN_PHASE = 2 * FPS               # "dropped if shorter than 2 s"
HOME, AWAY, NONE = 1, -1, 0
CLIP = (.001, .999)
SEED_BASE = 620000                # 620000 + 1000 x half index + 10 x profile index + draw index
DRAWS = 3
WEIGHTS = (.5, 1.0, 2.0)


@dataclass(frozen=True)
class Profile:
    c: float                      # chance that a true possession of at least 5 s becomes a phase
    a: float                      # start delay, uniform 0 to a seconds
    b: float                      # end offset, uniform -1 to b seconds
    q: float                      # chance that the phase names the wrong team
    carry: bool = False           # on masked frames, the team at the last sighting replaces the state


PROFILES = {"Good": Profile(.8, 1., 1., .05), "Moderate": Profile(.6, 2., 3., .15),
            "Poor": Profile(.6, 3., 5., .30), "Carry": Profile(.6, 2., 3., .15, carry=True)}
PROFILE_ORDER = ("Good", "Moderate", "Poor", "Carry")


def seed(half_index, profile_index, draw):
    """D62's seed formula (zero-based indices)."""
    if not (0 <= profile_index < 10 and 0 <= draw < 10):
        raise ValueError("A profile index and a draw index are single digits in D62's seed formula")
    return SEED_BASE + 1000 * half_index + 10 * profile_index + draw


def possessions(target):
    """[(start, stop, code)] of the target's maximal H or A runs inside its graded runs."""
    target = np.asarray(target)
    return pm.possession_spells(target, pm.valid_runs(target))


def simulated_phases(target, profile, rng):
    """[(start, end, team)] simulated from the N = 3 target under `profile` with the generator `rng`."""
    target = np.asarray(target)
    length = len(target)
    candidates = [(start, stop, code) for start, stop, code in possessions(target) if stop - start >= MIN_POSSESSION]
    count = len(candidates)
    kept = rng.random(count) < profile.c
    delay = rng.uniform(0., profile.a, count)
    offset = rng.uniform(-1., profile.b, count)
    wrong = rng.random(count) < profile.q
    phases = []
    for i, (start, stop, code) in enumerate(candidates):
        if not kept[i]:
            continue
        first = min(max(start + int(np.rint(delay[i] * FPS)), 0), length)
        last = min(max(stop + int(np.rint(offset[i] * FPS)), 0), length)
        if last - first < MIN_PHASE:
            continue
        team = HOME if code == pm.H else AWAY
        phases.append((first, last, -team if wrong[i] else team))
    return phases


def tag_state(length, intervals):
    """int8 state per frame: +1 home, -1 away, 0 none or conflicting, from half-open (start, end, team) intervals."""
    home = np.zeros(length + 1, dtype=np.int64)
    away = np.zeros(length + 1, dtype=np.int64)
    for start, end, team in intervals:
        if team not in (HOME, AWAY):
            raise ValueError(f"A tag names team {team!r}; expected +1 (home) or -1 (away)")
        first, last = min(max(int(start), 0), length), min(max(int(end), 0), length)
        if last <= first:
            continue
        counts = home if team == HOME else away
        counts[first] += 1
        counts[last] -= 1
    home_cover, away_cover = np.cumsum(home)[:length] > 0, np.cumsum(away)[:length] > 0
    state = np.zeros(length, dtype=np.int8)
    state[home_cover & ~away_cover] = HOME
    state[away_cover & ~home_cover] = AWAY
    return state


def far_mask(seen):
    """D59's HT frames: the nearest sighting, earlier or later in the half, is more than 3 s (75 frames) away."""
    return far_from_sighting(np.asarray(seen, dtype=bool), GAP)


def carried_state(target, seen):
    """Per frame: the team (+1/-1) the target names at the last sighting strictly before the frame; 0 if that frame
    is D or X, or if there is none."""
    target, seen = np.asarray(target), np.asarray(seen, dtype=bool)
    index = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, index, -1))
    before = np.r_[-1, last[:-1]] if len(seen) else last
    codes = np.where(before >= 0, target[np.maximum(before, 0)], pm.X)
    return np.where(codes == pm.H, HOME, np.where(codes == pm.A, AWAY, NONE)).astype(np.int8)


def with_carry(state, target, seen, mask):
    """The Carry profile: on masked frames the state is replaced by `carried_state`; elsewhere it is unchanged."""
    return np.where(np.asarray(mask, dtype=bool), carried_state(target, seen), state).astype(np.int8)


def half_state(target, seen, profile, rng):
    """(state, mask, phases) of one half under `profile`."""
    target = np.asarray(target)
    phases = simulated_phases(target, profile, rng)
    state = tag_state(len(target), phases)
    mask = far_mask(seen)
    if profile.carry:
        state = with_carry(state, target, seen, mask)
    return state, mask, phases


def acting(state, mask):
    """Frames on which the rule acts: masked and a nonzero state."""
    return np.asarray(mask, dtype=bool) & (np.asarray(state) != NONE)


def shift_home(home, state, mask, weight):
    """The home-control input with the log-odds moved by weight x state on masked frames with a nonzero state."""
    home = np.asarray(home)
    result = home.copy()
    if weight == 0:
        return result
    where = acting(state, mask)
    values = np.clip(home[where].astype(np.float64), *CLIP)
    result[where] = expit(logit(values) + weight * np.asarray(state)[where].astype(np.float64))
    return result


def shifted_signals(signals, state, mask, weight):
    """The four decoder inputs with only the first (home) shifted."""
    home, *rest = signals
    return (shift_home(home, state, mask, weight), *rest)


def agreement(state, decoded, seen):
    """(agreeing frames, counted frames): ball seen, decode H or A, state not 0; agree when the teams match."""
    state, decoded, seen = np.asarray(state), np.asarray(decoded), np.asarray(seen, dtype=bool)
    counted = seen & np.isin(decoded, (pm.H, pm.A)) & (state != NONE)
    agree = counted & (((state == HOME) & (decoded == pm.H)) | ((state == AWAY) & (decoded == pm.A)))
    return int(agree.sum()), int(counted.sum())
