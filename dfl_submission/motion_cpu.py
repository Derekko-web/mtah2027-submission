"""CPU inference for the player-set temporal model."""

import json

import numpy as np
from scipy.special import erf, expit

from dfl_submission import temporal_cpu
from experiments.dfl_transfer_data import digest


EMBED = 24
STEPS = (350, 700, 1050)


def gelu(values):
    return values * np.float32(.5) * (1 + erf(values * np.float32(2 ** -.5)))


def residual(players, weights):
    visible = players[..., -1:] > 0
    encoded = gelu(players @ weights["embed1.weight"].T + weights["embed1.bias"])
    encoded = gelu(encoded @ weights["embed2.weight"].T + weights["embed2.bias"])
    count = visible.sum(axis=-2).astype(np.float32)
    mean = np.where(visible, encoded, 0).sum(axis=-2) / np.maximum(count, 1)
    maximum = np.where(visible, encoded, np.float32(-1000)).max(axis=-2)
    pooled = np.concatenate((mean, np.where(count > 0, maximum, 0)), axis=-1)
    return pooled.reshape(*pooled.shape[:2], -1) @ weights["motion.weight"].T + weights["motion.bias"]


def forward(values, players, weights):
    side = {name.removeprefix("base."): array for name, array in weights.items() if name.startswith("base.")}
    return temporal_cpu.forward(values, side, residual=residual(players, weights))


def load(folder, filename):
    fit = json.loads((folder / "fit.json").read_text())
    norm, path = folder / "normalization.npz", folder / filename
    if digest(norm) != fit["normalization"] or digest(path) != fit["models"][filename]:
        raise ValueError("Player-motion checkpoint identity differs")
    with np.load(path, allow_pickle=False) as source:
        weights = {name: source[name] for name in source.files}
    with np.load(norm, allow_pickle=False) as source:
        location, scale, permutation = (source[name] for name in ("location", "scale", "permutation"))
    return weights, fit["columns"], location, scale, permutation


def predict(checkpoint, context, players):
    weights, expected, location, scale, permutation = checkpoint
    frames, values, columns, length = context
    if columns != expected:
        raise ValueError("Player-motion feature schema differs")
    variants = np.stack([temporal_cpu.normalize(values, location, scale),
                         temporal_cpu.normalize(values[:, permutation], location, scale)])
    both_sides = np.stack([players, players[:, ::-1]]).astype(np.float32)
    output = expit(forward(variants, both_sides, weights))
    home = (output[0, :, 1] + 1 - output[1, :, 1]) / 2
    dead, dead_edge, owner_edge = (output[:, :, k].mean(axis=0) for k in (0, 2, 3))
    timeline = np.arange(length)
    return tuple(np.interp(timeline, frames, p).astype(np.float32) for p in (home, dead, dead_edge, owner_edge))
