"""Decode home, away and stoppage jointly with distinct switch costs."""

import numpy as np
from scipy.special import logit

from dfl_submission.adaptive_decoder import STRIDE
from dfl_submission.decoder import centered_mean


def path(evidence, dead_cost, owner_cost):
    back = np.empty((len(evidence), 3), dtype=np.uint8)
    h, a, d = evidence[0]
    for i, (observed_h, observed_a, observed_d) in enumerate(evidence[1:], 1):
        oc, dc = owner_cost[i], dead_cost[i]
        h_from_a, h_from_d = a - oc, d - dc
        a_from_h, a_from_d = h - oc, d - dc
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


def decode(home, dead, dead_edge, owner_edge, *, cutoff, dead_cost, owner_cost, discount):
    starts = np.arange(0, len(home), STRIDE)

    def evidence(values):
        smoothed = centered_mean(values, 8, 0, len(values))
        return np.add.reduceat(logit(np.clip(smoothed, .001, .999)), starts) / STRIDE

    ownership = evidence(home)
    stoppage = evidence(dead) - logit(cutoff)
    emissions = np.column_stack((np.minimum(ownership, 0), np.minimum(-ownership, 0), stoppage))
    costs = (base * (1 - discount * np.maximum.reduceat(edge, starts))
             for base, edge in ((dead_cost, dead_edge), (owner_cost, owner_edge)))
    return np.repeat(path(emissions, *costs), STRIDE)[:len(home)]
