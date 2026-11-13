"""D34: the temporal network trained with other optimisation hyperparameters (DECISIONS.md D34).

The network, data, loss, schedule shape and export are `temporal_network`'s. Four training-only values
can change; inference (`temporal_cpu`) is untouched:
- `lr_scale` multiplies the whole cosine schedule (peak 0.0015, floor 0.0002 in D15);
- `batch` is the number of windows per step (16 in D15), split equally between the sources;
- `weight_decay` is AdamW's (0.01 in D15);
- `dropout` is each temporal block's (0.08 in D15).
D37 adds D33's spell weight (`alpha`) and boundary weight (`edge`), from `objective_network`.
At their defaults the trainer is `temporal_network.train`, step for step and bit for bit.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from dfl_submission import objective_network as on, temporal_network as tn
from dfl_submission.temporal_cpu import STEPS, WIDTH, normalize
from dfl_submission.train import write_json
from dfl_transfer_data import digest


DEFAULTS = {"lr_scale": 1., "batch": tn.BATCH, "weight_decay": .01, "dropout": .08}


def batch(groups, rng, permutation, location, scale, size=tn.BATCH):
    """`temporal_network.batch` with `size` windows; at 16 it draws exactly the same windows."""
    xs, ys = [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(size // len(groups)):
            values, labels = sequences[rng.integers(len(sequences))]
            start = rng.integers(0, len(values) - tn.CORE - 2 * tn.HALO + 1)
            window = values[start:start + tn.CORE + 2 * tn.HALO]
            output = labels[start + tn.HALO:start + tn.HALO + tn.CORE]
            if rng.integers(2):
                window = window[:, permutation]
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            xs.append(normalize(window, location, scale))
            ys.append(output)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def learning_rate(step, steps, lr_scale=1.):
    rate = tn.learning_rate(step, steps)
    return rate if lr_scale == 1 else rate * lr_scale


def options(**values):
    """Fit-record keys for non-default hyperparameters; empty at the defaults."""
    return {key: value for key, value in values.items() if value != DEFAULTS[key]}


def train(source, output, device, *, seed=42, boundary_boost=.5, width=WIDTH, steps=STEPS[-1],
          lr_scale=1., batch_size=tn.BATCH, weight_decay=.01, dropout=.08, alpha=0., edge=on.EDGE):
    output.mkdir(parents=True, exist_ok=False)
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, manifest, permutation, location, scale = tn.training_data(source)
    spell = alpha != 0 or edge != on.EDGE
    if spell:
        groups = on.weighted_groups(groups, alpha)
    if batch_size % len(groups):
        raise ValueError(f"A batch of {batch_size} does not split equally between {len(groups)} sources")
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = tn.Network(len(location), width).to(device)
    if dropout != DEFAULTS["dropout"]:
        for block in net.temporal:
            block.dropout.p = dropout
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes = {}
    for step in range(1, steps + 1):
        inputs, labels = batch(groups, rng, permutation, location, scale, batch_size)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.param_groups[0]["lr"] = learning_rate(step, steps, lr_scale)
        optimizer.zero_grad(set_to_none=True)
        logits = net(inputs)[:, tn.HALO:tn.HALO + tn.CORE]
        value = on.loss(logits, labels, boundary_boost, alpha=alpha, edge=edge) if spell else \
            tn.loss(logits, labels, boundary_boost)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"Temporal training step {step} of {steps}, loss {value.item():.4f}", flush=True)
        if step in saved:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "training_device": device, "torch": torch.__version__, "seed": seed,
               "boundary_boost": boundary_boost} | tn.options(width, steps)
               | options(lr_scale=lr_scale, batch=batch_size, weight_decay=weight_decay, dropout=dropout)
               | ({"alpha": alpha, "edge": edge} if spell else {}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-boost", type=float, default=.5)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--steps", type=int, default=STEPS[-1])
    parser.add_argument("--lr-scale", type=float, default=1.)
    parser.add_argument("--batch", type=int, default=tn.BATCH)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--dropout", type=float, default=.08)
    parser.add_argument("--alpha", type=float, default=0.)
    parser.add_argument("--edge", type=float, default=on.EDGE)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    require_external_only()
    train(args.source, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost, width=args.width,
          steps=args.steps, lr_scale=args.lr_scale, batch_size=args.batch, weight_decay=args.weight_decay,
          dropout=args.dropout, alpha=args.alpha, edge=args.edge)
