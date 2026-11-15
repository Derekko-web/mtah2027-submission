"""D63: a lower ownership cost next to stoppages, as a second pass of the standing decoder (DECISIONS.md D63).

Pure functions on one half's decoder inputs; nothing here reads a file.

- `standing_terms` builds the Viterbi inputs (emissions, per-step stoppage cost, per-step ownership cost) exactly as
  `joint_decoder.decode` builds them at its default step (STRIDE = 5 frames, the 5 Hz grid) with no restart option:
  the same expressions in the same order. `pass_one` runs `joint_decoder.path` on them as `decode` does
  (`restart_cost=None`, `min_dead=1`). That `pass_one` equals `joint_decoder.decode` label for label is tested on
  random inputs (`tests/test_edge_cost.py`) and checked on every cached half before anything is scored
  (`edge_trial.develop`).
- `windows`, from pass 1's step labels: t is the first dead step of a pass-1 stoppage and r the first live step after
  one. Steps t - 15 ... t - 1 are *before a stop*; steps r + 1 ... r + 15 are *after a restart* (15 steps are 3 s).
  A half that starts live has a restart at its first step (r = 0); a half that ends live has a stop just after its
  last step (t = the number of steps). Windows are clipped to the half's steps.
- `factors`: 1 everywhere, rho_pre on the steps before a stop, rho_post on the steps after a restart, the smaller of
  the two where both apply.
- `two_pass`: pass 2 is `joint_decoder.path` on the same emissions and stoppage costs with the ownership cost of each
  step multiplied by its factor (in the cost array's own dtype); its path is the output, repeated to frames as
  `decode` repeats it. With rho_pre = rho_post = 1 every cost is unchanged, so pass 2 is pass 1 exactly.

`joint_decoder.path` charges the ownership cost of step i on a change between steps i - 1 and i, so the factor of a
step applies to a change of team into that step.
"""

import numpy as np
from scipy.special import logit

from dfl_submission import joint_decoder
from dfl_submission.adaptive_decoder import STRIDE
from dfl_submission.decoder import centered_mean
from evaluation import possession_metrics as pm


WINDOW = 15                       # steps on the 5 Hz grid: 3 s, D1's N
FACTORS = (1.0, .75, .5, .25)     # D63's grid for rho_pre and rho_post
DEAD = 2                          # the decoder's stoppage label (pm.D)
assert DEAD == pm.D and (pm.H, pm.A) == (0, 1)


def standing_terms(home, dead, dead_edge, owner_edge, *, cutoff, dead_cost, owner_cost, discount):
    """(emissions, dead_steps, owner_steps) exactly as `joint_decoder.decode` builds them at its default step."""
    step = STRIDE
    starts = np.arange(0, len(home), step)
    scale = step / STRIDE

    def evidence(values):
        smoothed = centered_mean(values, 8, 0, len(values))
        return np.add.reduceat(logit(np.clip(smoothed, .001, .999)), starts) / STRIDE

    ownership = evidence(home)
    stoppage = evidence(dead) - logit(cutoff) * scale
    dead_steps, owner_steps = (base * (1 - discount * np.maximum.reduceat(edge, starts))
                               for base, edge in ((dead_cost, dead_edge), (owner_cost, owner_edge)))
    emissions = np.column_stack((np.minimum(ownership, 0), np.minimum(-ownership, 0), stoppage))
    return emissions, dead_steps, owner_steps


def viterbi(emissions, dead_steps, owner_steps):
    """`joint_decoder.path` as `joint_decoder.decode` calls it at its default step (no restart, min_dead 1)."""
    return joint_decoder.path(emissions, dead_steps, owner_steps, None, min_dead=1)


def to_frames(labels, length):
    """Step labels repeated to frames, as `joint_decoder.decode` returns them."""
    return np.repeat(labels, STRIDE)[:length]


def recipe_terms(signals, recipe):
    home, dead, dead_edge, owner_edge = signals
    return standing_terms(home, dead, dead_edge, owner_edge, cutoff=recipe["cutoff"], dead_cost=recipe["dead_cost"],
                          owner_cost=recipe["owner_cost"], discount=recipe["discount"])


def pass_one(signals, recipe):
    """Pass 1's step labels: the standing decode on the 5 Hz grid."""
    return viterbi(*recipe_terms(signals, recipe))


def stops_and_restarts(labels):
    """(stops, restarts): first dead steps of stoppages (plus the step after the last if the half ends live), and
    first live steps after a stoppage (plus step 0 if the half starts live)."""
    dead = np.asarray(labels) == DEAD
    count = len(dead)
    if not count:
        return [], []
    entering = np.flatnonzero(dead & np.r_[True, ~dead[:-1]])
    leaving = np.flatnonzero(~dead & np.r_[True, dead[:-1]])
    stops = [int(t) for t in entering] + ([count] if not dead[-1] else [])
    return stops, [int(r) for r in leaving]


def windows(labels, width=WINDOW):
    """(before, after): boolean per step, before a stop (t - width ... t - 1) and after a restart (r + 1 ... r + width)."""
    count = len(labels)
    before, after = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
    stops, restarts = stops_and_restarts(labels)
    for t in stops:
        before[max(t - width, 0):t] = True
    for r in restarts:
        after[r + 1:min(r + 1 + width, count)] = True
    return before, after


def factors(labels, rho_pre, rho_post, width=WINDOW):
    """Per step: rho_pre before a stop, rho_post after a restart, the smaller where both apply, 1 elsewhere."""
    before, after = windows(labels, width)
    result = np.ones(len(labels), dtype=np.float64)
    result = np.where(before, np.minimum(result, rho_pre), result)
    return np.where(after, np.minimum(result, rho_post), result)


def two_pass_steps(signals, recipe, rho_pre, rho_post):
    """(pass 1, pass 2, factors) on the 5 Hz grid."""
    emissions, dead_steps, owner_steps = recipe_terms(signals, recipe)
    first = viterbi(emissions, dead_steps, owner_steps)
    scale = factors(first, rho_pre, rho_post)
    second = viterbi(emissions, dead_steps, owner_steps * scale.astype(owner_steps.dtype))
    return first, second, scale


def two_pass(signals, recipe, rho_pre, rho_post):
    """D63's output on frames: pass 2's path repeated to the half's frames."""
    _, second, _ = two_pass_steps(signals, recipe, rho_pre, rho_post)
    return to_frames(second, len(signals[0]))
