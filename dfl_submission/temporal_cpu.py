"""NumPy inference for the exported temporal possession network."""

import json

import numpy as np
from scipy.special import erf, expit

from experiments.dfl_transfer_data import digest


DILATIONS = (1, 2, 4, 8, 16, 32)
WIDTH = 48
STEPS = (600, 1200, 1800)


def normalize(values, location, scale):
    valid = np.isfinite(values)
    scaled = np.clip((values - location) / scale, -8, 8)
    return np.concatenate([np.where(valid, scaled, 0), valid.astype(np.float32)], axis=-1)


def norm(values, scale, bias):
    centered = values - values.mean(axis=-1, keepdims=True)
    variance = np.mean(centered ** 2, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5)) * scale + bias


def forward(values, weights, *, residual=None):
    hidden = values @ weights["project.weight"][:, :, 0].T + weights["project.bias"]
    if residual is not None:
        hidden += residual
    for index, dilation in enumerate(DILATIONS):
        prefix = f"temporal.{index}."
        layer = norm(hidden, weights[prefix + "norm.weight"], weights[prefix + "norm.bias"])
        kernels = weights[prefix + "temporal.weight"][:, 0]
        convolved = layer * kernels[:, 1] + weights[prefix + "temporal.bias"]
        convolved[:, dilation:] += layer[:, :-dilation] * kernels[:, 0]
        convolved[:, :-dilation] += layer[:, dilation:] * kernels[:, 2]
        gelu = convolved * np.float32(.5) * (1 + erf(convolved * np.float32(2 ** -.5)))
        hidden += gelu @ weights[prefix + "mix.weight"][:, :, 0].T + weights[prefix + "mix.bias"]
    hidden = norm(hidden, weights["norm.weight"], weights["norm.bias"])
    return hidden @ weights["outputs.weight"].T + weights["outputs.bias"]


def load(folder, filename):
    fit = json.loads((folder / "fit.json").read_text())
    path, norm_path = folder / filename, folder / "normalization.npz"
    if digest(path) != fit["models"][filename] or digest(norm_path) != fit["normalization"]:
        raise ValueError("Temporal checkpoint identity differs")
    with np.load(path, allow_pickle=False) as data:
        weights = {key: data[key] for key in data.files}
    with np.load(norm_path, allow_pickle=False) as data:
        location, scale, permutation = (data[name] for name in ("location", "scale", "permutation"))
    return weights, fit["columns"], location, scale, permutation


def predict(checkpoint, features):
    weights, expected, location, scale, permutation = checkpoint
    frames, values, columns, length = features
    if columns != expected:
        raise ValueError("Temporal inference feature columns differ")
    variants = np.stack([normalize(values, location, scale), normalize(values[:, permutation], location, scale)])
    p = expit(forward(variants, weights))
    home = (p[0, :, 1] + 1 - p[1, :, 1]) / 2
    dead = (p[0, :, 0] + p[1, :, 0]) / 2
    dead_edge, owner_edge = ((p[0, :, index] + p[1, :, index]) / 2 for index in (2, 3))
    timeline = np.arange(length)
    return tuple(np.interp(timeline, frames, values).astype(np.float32)
                 for values in (home, dead, dead_edge, owner_edge))
