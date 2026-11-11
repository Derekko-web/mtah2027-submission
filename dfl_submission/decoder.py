"""Decode one half from live/dead and conditional home probabilities."""

import numpy as np


def centered_mean(values, radius, start, stop):
    frames = np.arange(len(values))
    left = np.maximum(start, frames - radius)
    right = np.minimum(stop, frames + radius + 1)
    cumulative = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    return (cumulative[right] - cumulative[left]) / (right - left)


def decode(home, dead, *, dead_cutoff, dead_radius, owner_radius):
    stoppage = centered_mean(dead, dead_radius, 0, len(dead)) >= dead_cutoff
    boundaries = np.r_[0, np.flatnonzero(stoppage[1:] != stoppage[:-1]) + 1, len(dead)]
    lengths = np.diff(boundaries)
    # A restart starts a new spell. Never smooth ownership across a decoded stoppage.
    home_mean = centered_mean(home, owner_radius, np.repeat(boundaries[:-1], lengths),
                              np.repeat(boundaries[1:], lengths))
    return np.where(stoppage, 2, np.where(home_mean >= 0.5, 0, 1)).astype(np.int8)
