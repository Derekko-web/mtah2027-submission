"""Train a dilated temporal possession network on external matches only."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dfl_submission import external_benchmark as benchmark, long_context
from dfl_submission.temporal_cpu import DILATIONS, STEPS, WIDTH, forward, normalize
from dfl_submission.train import SOURCE_MANIFEST, write_json
from dfl_transfer_data import digest
from evaluation import possession_metrics as pm


HALO = sum(DILATIONS)
CORE = 128
BATCH = 16


class TemporalBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.temporal = nn.Conv1d(WIDTH, WIDTH, 3, dilation=dilation, padding=dilation, groups=WIDTH)
        self.mix = nn.Conv1d(WIDTH, WIDTH, 1)
        self.dropout = nn.Dropout(.08)

    def forward(self, values):
        normalized = self.norm(values.transpose(1, 2)).transpose(1, 2)
        return values + self.dropout(self.mix(F.gelu(self.temporal(normalized))))


class Network(nn.Module):
    def __init__(self, columns):
        super().__init__()
        self.project = nn.Conv1d(columns * 2, WIDTH, 1)
        self.temporal = nn.Sequential(*(TemporalBlock(dilation) for dilation in DILATIONS))
        self.norm = nn.LayerNorm(WIDTH)
        self.outputs = nn.Linear(WIDTH, 4)

    def forward(self, values):
        hidden = self.temporal(self.project(values.transpose(1, 2))).transpose(1, 2)
        return self.outputs(self.norm(hidden))


def swap_indices(columns):
    lookup = {name: i for i, name in enumerate(columns)}
    return np.array([lookup["a__" + name[3:] if name.startswith("h__") else
                            "h__" + name[3:] if name.startswith("a__") else name] for name in columns])


def targets(labels):
    live = labels < pm.D
    known = labels != pm.X
    dead_edge = np.r_[False, ((labels[1:] == pm.D) != (labels[:-1] == pm.D)) & known[:-1] & known[1:]]
    owner_edge = np.r_[False, (labels[1:] != labels[:-1]) & live[:-1] & live[1:]]
    for edge in (dead_edge, owner_edge):
        edge[:] = edge | np.r_[edge[1:], False] | np.r_[False, edge[:-1]]
    return np.column_stack([labels == pm.D, labels == pm.H, dead_edge, owner_edge, known, live]).astype(np.float32)


def convention_labels(labels, convention):
    """Training labels under a controlled-possession convention (DECISIONS.md D13); None keeps them."""
    if convention is None:
        return labels
    from dfl_submission.conventions import collapse_brief_changes
    return collapse_brief_changes(labels, convention)


def prepare(dfl, skill, output, *, raw=False, augment=False, convention=None):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    columns = None
    if augment:
        from dfl_submission.augment import training_halves
        halves = training_halves(dfl, skill, raw=raw)
    else:
        halves = benchmark.halves(dfl, skill, "train", raw=raw)
    for half in halves:
        frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        if columns is None:
            columns = table.columns.tolist()
        elif columns != table.columns.tolist():
            raise ValueError("External feature schema differs")
        name = f"{half.key}.npz"
        labels = convention_labels(half.labels, convention)
        np.savez_compressed(output / name, values=table.to_numpy(dtype=np.float32), targets=targets(labels[frames]))
        records.append({"file": name, "source": half.source, "match": half.match, "sha256": digest(output / name)})
        print(f"Prepared temporal sequence {half.key}", flush=True)
    write_json(output / "manifest.json", {"columns": columns, "records": records, "dfl_manifest": digest(SOURCE_MANIFEST),
                                           "skill_manifest": digest(skill / "manifest.json")} | ({"augmented": True} if augment else {})
               | ({"convention": convention} if convention is not None else {}))


def training_data(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    groups = defaultdict(list)
    sample = []
    for record in manifest["records"]:
        path = folder / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Prepared sequence hash differs: {path}")
        with np.load(path, allow_pickle=False) as data:
            values, labels = data["values"], data["targets"]
        groups[record["source"]].append((values, labels))
        sample.append(values[::11])
    permutation = swap_indices(manifest["columns"])
    sample = np.concatenate(sample)
    observed = np.concatenate([sample, sample[:, permutation]])
    count = np.maximum(1, np.isfinite(observed).sum(axis=0))
    finite = np.where(np.isfinite(observed), observed, 0)
    location = finite.sum(axis=0, dtype=np.float64) / count
    variance = (finite ** 2).sum(axis=0, dtype=np.float64) / count - location ** 2
    scale = np.sqrt(np.maximum(variance, 1e-4))
    return groups, manifest, permutation, location.astype(np.float32), scale.astype(np.float32)


def batch(groups, rng, permutation, location, scale):
    xs, ys = [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(BATCH // len(groups)):
            values, labels = sequences[rng.integers(len(sequences))]
            start = rng.integers(0, len(values) - CORE - 2 * HALO + 1)
            window = values[start:start + CORE + 2 * HALO]
            output = labels[start + HALO:start + HALO + CORE]
            if rng.integers(2):
                window = window[:, permutation]
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            xs.append(normalize(window, location, scale))
            ys.append(output)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def loss(logits, labels, boundary_boost=.5):
    positive = logits.new_tensor([1., 1., 6., 6.])
    errors = F.binary_cross_entropy_with_logits(logits, labels[..., :4], pos_weight=positive, reduction="none")
    known, live = labels[..., 4], labels[..., 5]
    weights = torch.stack((known * (1 + boundary_boost * labels[..., 2]), live * (1 + boundary_boost * labels[..., 3]),
                           .2 * known, .2 * live), dim=-1)
    return (errors * weights).sum() / weights.sum().clamp_min(1)


def train(source, output, device, *, seed=42, boundary_boost=.5):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, manifest, permutation, location, scale = training_data(source)
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = Network(len(location)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.01)
    hashes = {}
    for step in range(1, STEPS[-1] + 1):
        inputs, labels = batch(groups, rng, permutation, location, scale)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.param_groups[0]["lr"] = .0002 + .0013 * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS[-1]))
        optimizer.zero_grad(set_to_none=True)
        value = loss(net(inputs)[:, HALO:HALO + CORE], labels, boundary_boost)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"Temporal training step {step} of {STEPS[-1]}, loss {value.item():.4f}", flush=True)
        if step in STEPS:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "training_device": device, "torch": torch.__version__, "seed": seed,
               "boundary_boost": boundary_boost})


def export(folder, output):
    output.mkdir(parents=True, exist_ok=False)
    fit = json.loads((folder / "fit.json").read_text())
    norm_path = folder / "normalization.npz"
    if digest(norm_path) != fit["normalization"]:
        raise ValueError("Temporal normalization identity differs")
    copyfile(norm_path, output / norm_path.name)
    torch.set_num_threads(1)
    columns = len(fit["columns"])
    net = Network(columns).eval()
    rng = np.random.default_rng(13)
    values = rng.normal(size=(2, 180, columns * 2)).astype(np.float32)
    hashes, differences = {}, {}
    for filename, expected in fit["models"].items():
        path = folder / filename
        if digest(path) != expected:
            raise ValueError(f"Temporal checkpoint identity differs: {filename}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        weights = {name: value.numpy() for name, value in state.items()}
        with torch.inference_mode():
            reference = net(torch.from_numpy(values)).numpy()
        result = forward(values, weights)
        np.testing.assert_allclose(result, reference, atol=2e-5, rtol=1e-5)
        destination = output / Path(filename).with_suffix(".npz").name
        np.savez(destination, **weights)
        hashes[destination.name] = digest(destination)
        differences[destination.name] = float(np.max(np.abs(result - reference)))
    write_json(output / "fit.json", dict(fit, models=hashes, original_checkpoints=fit["models"],
                                        export_max_logit_difference=differences))
    print(f"CPU export verified: {differences}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    preparation = actions.add_parser("prepare")
    preparation.add_argument("--dfl", required=True, type=Path)
    preparation.add_argument("--dfl-raw", action="store_true")
    preparation.add_argument("--skill", required=True, type=Path)
    preparation.add_argument("--out", required=True, type=Path)
    preparation.add_argument("--augment", action="store_true", help="Prepare DECISIONS.md D2 augmented halves")
    preparation.add_argument("--convention", type=float, help="Collapse brief changes under this many seconds (D13)")
    training = actions.add_parser("train")
    training.add_argument("--source", required=True, type=Path)
    training.add_argument("--out", required=True, type=Path)
    training.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--boundary-boost", type=float, default=.5)
    conversion = actions.add_parser("export")
    conversion.add_argument("--source", required=True, type=Path)
    conversion.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    if args.action != "export":
        require_external_only()
    if args.action == "prepare":
        prepare(args.dfl, args.skill, args.out, raw=args.dfl_raw, augment=args.augment, convention=args.convention)
    elif args.action == "train":
        train(args.source, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost)
    else:
        export(args.source, args.out)
