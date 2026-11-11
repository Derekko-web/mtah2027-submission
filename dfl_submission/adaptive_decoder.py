"""Offline state decoding with a switch cost conditioned on boundary evidence."""

import numpy as np
from scipy.special import logit

from dfl_submission.decoder import centered_mean
from evaluation import possession_metrics as pm


STRIDE = 5


def binary_path(evidence, cost):
    """Maximize state-one evidence minus the transition cost at each step."""
    delta = np.empty(len(evidence), dtype=np.float64)
    previous = 0.0
    for i, (value, penalty) in enumerate(zip(evidence, cost)):
        previous = value + min(penalty, max(-penalty, previous))
        delta[i] = previous
    result = np.empty(len(delta), dtype=bool)
    state = bool(delta[-1] >= 0)
    result[-1] = state
    for i in range(len(delta) - 2, -1, -1):
        if state:
            state = bool(delta[i] >= -cost[i + 1])
        else:
            state = bool(delta[i] > cost[i + 1])
        result[i] = state
    return result


def sequence(probability, hazard, threshold, transition_cost, discount):
    values = centered_mean(probability, 8, 0, len(probability))
    evidence = logit(np.clip(values, .001, .999)) - logit(threshold)
    starts = np.arange(0, len(values), STRIDE)
    block = np.add.reduceat(evidence, starts) / STRIDE
    strength = np.maximum.reduceat(hazard, starts)
    cost = transition_cost * (1 - discount * strength)
    return np.repeat(binary_path(block, cost), STRIDE)[:len(values)]


def decode(home, dead, dead_edge, owner_edge, *, cutoff, dead_cost, owner_cost, discount):
    stoppage = sequence(dead, dead_edge, cutoff, dead_cost, discount)
    result = np.full(len(dead), pm.D, dtype=np.int8)
    edges = np.r_[0, np.flatnonzero(stoppage[1:] != stoppage[:-1]) + 1, len(dead)]
    for start, stop in zip(edges[:-1], edges[1:]):
        if not stoppage[start]:
            result[start:stop] = ~sequence(home[start:stop], owner_edge[start:stop], .5, owner_cost, discount)
    return result
