"""Decode home, away and stoppage jointly with distinct switch costs."""

import numpy as np
from scipy.special import logit

from dfl_submission.adaptive_decoder import STRIDE
from dfl_submission.decoder import centered_mean


def path(evidence, dead_cost, owner_cost, restart_cost=None, min_dead=1):
    if restart_cost is None:
        restart_cost = dead_cost
    if min_dead > 1:
        return min_dead_path(evidence, dead_cost, owner_cost, restart_cost, min_dead)
    back = np.empty((len(evidence), 3), dtype=np.uint8)
    h, a, d = evidence[0]
    for i, (observed_h, observed_a, observed_d) in enumerate(evidence[1:], 1):
        oc, dc, rc = owner_cost[i], dead_cost[i], restart_cost[i]
        h_from_a, h_from_d = a - oc, d - rc
        a_from_h, a_from_d = h - oc, d - rc
        d_from_h, d_from_a = h - dc, a - dc
        best_h = max(h, h_from_a, h_from_d)
        best_a = max(a_from_h, a, a_from_d)
        best_d = max(d_from_h, d_from_a, d)
        back[i] = (0 if best_h == h else 1 if best_h == h_from_a else 2,
                   0 if best_a == a_from_h else 1 if best_a == a else 2,
                   0 if best_d == d_from_h else 1 if best_d == d_from_a else 2)
        h, a, d = best_h + observed_h, best_a + observed_a, best_d + observed_d
    labels = np.empty(len(evidence), dtype=np.int8)
    labels[-1] = np.argmax((h, a, d))
    for i in range(len(labels) - 1, 0, -1):
        labels[i - 1] = back[i, labels[i]]
    return labels


def min_dead_path(evidence, dead_cost, owner_cost, restart_cost, length):
    """The joint path when a stoppage entered inside the half lasts at least `length` steps.

    States are H, A and stoppage stages 1..length; only the last stage can restart or stay. A
    stoppage already under way at the first step, or still under way at the last, may be shorter.
    Without this, a one-step stoppage between two teams would be a cheaper ownership change.
    """
    last = length + 1
    states = length + 2
    back = np.zeros((len(evidence), states), dtype=np.uint8)
    h, a, d = evidence[0]
    score = [h, a] + [-np.inf] * (length - 1) + [d]
    for i, (observed_h, observed_a, observed_d) in enumerate(evidence[1:], 1):
        oc, dc, rc = owner_cost[i], dead_cost[i], restart_cost[i]
        h, a, settled = score[0], score[1], score[last]
        h_from_a, h_from_d = a - oc, settled - rc
        a_from_h, a_from_d = h - oc, settled - rc
        d_from_h, d_from_a = h - dc, a - dc
        best_h = max(h, h_from_a, h_from_d)
        best_a = max(a_from_h, a, a_from_d)
        best_entry = max(d_from_h, d_from_a)
        row = back[i]
        row[0] = 0 if best_h == h else 1 if best_h == h_from_a else last
        row[1] = 0 if best_a == a_from_h else 1 if best_a == a else last
        row[2] = 0 if best_entry == d_from_h else 1
        new = [best_h + observed_h, best_a + observed_a, best_entry + observed_d]
        for stage in range(3, last):
            row[stage] = stage - 1
            new.append(score[stage - 1] + observed_d)
        if length > 1:
            stay = score[last] >= score[last - 1]
            row[last] = last if stay else last - 1
            new.append(max(score[last], score[last - 1]) + observed_d)
        score = new
    states_path = np.empty(len(evidence), dtype=np.int64)
    states_path[-1] = int(np.argmax(score))
    for i in range(len(evidence) - 1, 0, -1):
        states_path[i - 1] = back[i, states_path[i]]
    return np.minimum(states_path, 2).astype(np.int8)


def decode(home, dead, dead_edge, owner_edge, *, cutoff, dead_cost, owner_cost, discount, restart=None, step=STRIDE):
    """Viterbi path on a grid of `step` frames.

    Evidence is always divided by STRIDE and per-step offsets are scaled by step / STRIDE, so the
    evidence stays the same per second at any step and the costs keep their meaning; a step of 1
    changes only where a boundary can fall (DECISIONS.md D12). At the default step the scale is
    exactly 1 and stoppages need no minimum length, so the arithmetic is unchanged.
    """
    if STRIDE % step:
        raise ValueError(f"The decoder step must divide {STRIDE}")
    starts = np.arange(0, len(home), step)
    scale = step / STRIDE

    def evidence(values):
        smoothed = centered_mean(values, 8, 0, len(values))
        return np.add.reduceat(logit(np.clip(smoothed, .001, .999)), starts) / STRIDE

    ownership = evidence(home)
    stoppage = evidence(dead) - logit(cutoff) * scale
    dead_steps, owner_steps = (base * (1 - discount * np.maximum.reduceat(edge, starts))
                               for base, edge in ((dead_cost, dead_edge), (owner_cost, owner_edge)))
    restart_steps = None
    if restart is not None:
        stoppage_bias, restart_discount = restart
        stoppage = stoppage + stoppage_bias * scale
        restart_steps = dead_steps * (1 - restart_discount)
    emissions = np.column_stack((np.minimum(ownership, 0), np.minimum(-ownership, 0), stoppage))
    # A stoppage decoded inside the half lasts at least STRIDE frames at every step, as at the default.
    labels = path(emissions, dead_steps, owner_steps, restart_steps, min_dead=STRIDE // step)
    return np.repeat(labels, step)[:len(home)]
