"""Soft set-piece restart evidence on the joint decoder's frame grid."""

import numpy as np

from dfl_submission.adaptive_decoder import STRIDE


FPS = 25


def decoder_adjustment(length, anchors, *, method, seconds, strength):
    """Turn half-local restart frames into stoppage bias and dead-to-live discounts."""
    if len(anchors) == 0:
        return None
    steps = np.arange(0, length, STRIDE)
    before = np.zeros(len(steps))
    after = np.zeros(len(steps))
    radius = seconds * FPS
    for frame in anchors:
        left, right = np.searchsorted(steps, (frame - radius, frame + radius))
        span = slice(left, right)
        offset = (steps[span] - frame) / radius
        if method == "evidence":
            weight = 1 - np.abs(2 * np.abs(offset) - 1)
            np.maximum(before[span], np.where(offset < 0, weight, 0), out=before[span])
            np.maximum(after[span], np.where(offset > 0, weight, 0), out=after[span])
        else:
            np.maximum(after[span], 1 - np.abs(offset), out=after[span])
    if method == "evidence":
        return strength * (before - after), np.zeros(len(steps))
    return before, strength * after
