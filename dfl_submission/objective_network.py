"""D33: the temporal network trained with spell-balanced ownership weights (DECISIONS.md D33).

The network, data, batches, schedule and export are `temporal_network`'s. Two loss weights change:
- `alpha` multiplies each live frame's ownership and ownership-boundary terms by a weight that grows as
  its possession (under the N = 3 convention) gets shorter: (10 / max(L, 2)) ** alpha for L < 10 s,
  otherwise 1, renormalised to mean 1 over each source's live frames;
- `edge` is the weight of both boundary heads (0.2 in `temporal_network.loss`).
At alpha = 0 and edge = 0.2 the trainer is `temporal_network.train`, step for step and bit for bit.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dfl_submission import temporal_network as tn
from dfl_submission.conventions import collapse_brief_changes
from dfl_submission.temporal_cpu import STEPS, WIDTH
from dfl_submission.train import write_json
from dfl_transfer_data import digest
from evaluation import possession_metrics as pm


FPS = 5                      # prepared sequences are sampled every fifth 25 Hz frame
CONVENTION = 3.0             # seconds: the N = 3 targets' A-B-A merge
LONG = 10.0                  # seconds: possessions at least this long keep weight 1
SHORTEST = 2.0               # seconds: shorter possessions are weighted as if they lasted this long
EDGE = .2                    # temporal_network.loss's boundary-head weight


def sequence_labels(targets):
    """H/A/D/X codes rebuilt from a prepared sequence's stored targets (dead, home, ..., known, live)."""
    dead, home, known, live = targets[:, 0] > .5, targets[:, 1] > .5, targets[:, 4] > .5, targets[:, 5] > .5
    if not np.array_equal(live, known & ~dead):
        raise ValueError("Stored targets are inconsistent: live must be known and not dead")
    codes = np.full(len(targets), pm.X, dtype=np.int8)
    codes[known & dead] = pm.D
    codes[live & home] = pm.H
    codes[live & ~home] = pm.A
    return codes


def possession_lengths(codes, fps=FPS, convention=CONVENTION):
    """Seconds of each frame's run of one team after the A-B-A merge; NaN off live frames."""
    merged = collapse_brief_changes(codes, convention, fps=fps)
    lengths = np.full(len(codes), np.nan, dtype=np.float64)
    edges = np.r_[0, np.flatnonzero(merged[1:] != merged[:-1]) + 1, len(merged)]
    for start, stop in zip(edges[:-1], edges[1:]):
        if merged[start] in (pm.H, pm.A):
            lengths[start:stop] = (stop - start) / fps
    return lengths


def raw_weights(lengths, alpha):
    """(LONG / max(L, SHORTEST)) ** alpha below LONG seconds, 1 at or above it; 1 off live frames."""
    weights = np.ones(len(lengths), dtype=np.float64)
    if alpha == 0:
        return weights
    live = np.isfinite(lengths)
    short = live & (lengths < LONG)
    weights[short] = (LONG / np.maximum(lengths[short], SHORTEST)) ** alpha
    return weights


def spell_weights(groups, alpha):
    """{source: [weight per frame, per sequence]}, each source's live-frame mean scaled to 1."""
    result = {}
    for source, sequences in groups.items():
        raw, live = [], []
        for _, labels in sequences:
            lengths = possession_lengths(sequence_labels(labels))
            raw.append(raw_weights(lengths, alpha))
            live.append(np.isfinite(lengths))
        mean = np.concatenate([r[l] for r, l in zip(raw, live)]).mean()
        result[source] = [np.where(l, r / mean, 1).astype(np.float32) for r, l in zip(raw, live)]
    return result


def weighted_groups(groups, alpha):
    """`groups` with each sequence's spell weights appended as a seventh target column."""
    weights = spell_weights(groups, alpha)
    return {source: [(values, np.column_stack([labels, weight]).astype(np.float32))
                     for (values, labels), weight in zip(sequences, weights[source])]
            for source, sequences in groups.items()}


def loss(logits, labels, boundary_boost=.5, *, alpha=0., edge=EDGE):
    """`temporal_network.loss` with the spell weight (column 6) on the ownership terms and `edge` on both
    boundary heads. At alpha = 0 and edge = 0.2 the arithmetic is temporal_network's, operation for operation."""
    positive = logits.new_tensor([1., 1., 6., 6.])
    errors = F.binary_cross_entropy_with_logits(logits, labels[..., :4], pos_weight=positive, reduction="none")
    known, live = labels[..., 4], labels[..., 5]
    owner = live * labels[..., 6] if alpha else live
    weights = torch.stack((known * (1 + boundary_boost * labels[..., 2]), owner * (1 + boundary_boost * labels[..., 3]),
                           edge * known, edge * owner), dim=-1)
    return (errors * weights).sum() / weights.sum().clamp_min(1)


def train(source, output, device, *, seed=42, boundary_boost=.5, width=WIDTH, steps=STEPS[-1], alpha=0., edge=EDGE):
    """`temporal_network.train` with D33's loss weights; the fit record adds `alpha` and `edge`."""
    output.mkdir(parents=True, exist_ok=False)
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, manifest, permutation, location, scale = tn.training_data(source)
    groups = weighted_groups(groups, alpha)
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = tn.Network(len(location), width).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.01)
    hashes = {}
    for step in range(1, steps + 1):
        inputs, labels = tn.batch(groups, rng, permutation, location, scale)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.param_groups[0]["lr"] = tn.learning_rate(step, steps)
        optimizer.zero_grad(set_to_none=True)
        value = loss(net(inputs)[:, tn.HALO:tn.HALO + tn.CORE], labels, boundary_boost, alpha=alpha, edge=edge)
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
               "boundary_boost": boundary_boost, "alpha": alpha, "edge": edge} | tn.options(width, steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-boost", type=float, default=.5)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--steps", type=int, default=STEPS[-1])
    parser.add_argument("--alpha", type=float, default=0.)
    parser.add_argument("--edge", type=float, default=EDGE)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    require_external_only()
    train(args.source, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost, width=args.width,
          steps=args.steps, alpha=args.alpha, edge=args.edge)
