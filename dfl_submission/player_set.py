"""D42: a player-set network (DECISIONS.md D42).

`tokens` describes, at the long-context frames, each team's visible players nearest the ball (or, when
the ball is unseen, the mean position of all visible players). `PlayerSetNetwork` is D15's temporal
network with a small per-player encoder, pooled per team (masked mean and max) and added to the input
projection. `train` and `export` follow `temporal_network` and `hyper_network`; `forward` and `predict`
are the NumPy inference, which reuses `temporal_cpu.forward` for the temporal part.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import erf, expit

from dfl_submission.temporal_cpu import DILATIONS, forward as temporal_forward, normalize


PER_TEAM = 11                 # players per team kept, nearest the reference point
TOKEN = 8                     # values per player
HIDDEN = 32                   # per-player encoder width
LAG = 5                       # frames for the velocity
MAX_SPEED = 12.0              # m/s; faster displacements are tracking jumps
STRIDE = 5                    # long-context frames are every fifth 25 Hz frame
HALO = sum(DILATIONS)
CORE = 128


# -- tokens ----------------------------------------------------------------------------------------

def tokens(positions, ball, teams, pitch, frames):
    """(len(frames), 2, PER_TEAM, TOKEN) float32, NaN where no player fills a slot."""
    positions = np.asarray(positions, dtype=np.float64)
    ball = np.asarray(ball, dtype=np.float64)
    teams = np.asarray(teams)
    frames = np.asarray(frames)
    half_pitch = np.asarray(pitch, dtype=np.float64) / 2
    result = np.full((len(frames), 2, PER_TEAM, TOKEN), np.nan, dtype=np.float32)
    now = positions[frames]
    before = positions[np.maximum(frames - LAG, 0)]
    seen = np.isfinite(ball[frames]).all(axis=1)
    visible = np.isfinite(now).all(axis=2)
    count = visible.sum(axis=1)
    centroid = np.divide(np.where(visible[..., None], now, 0).sum(axis=1), count[:, None],
                         out=np.full((len(frames), 2), np.nan), where=count[:, None] > 0)
    reference = np.where(seen[:, None], ball[frames], centroid)
    velocity = (now - before) * (25 / LAG)
    speed = np.linalg.norm(velocity, axis=2)
    moving = np.isfinite(speed) & (speed <= MAX_SPEED) & (frames[:, None] >= LAG)
    velocity = np.where(moving[..., None], velocity, 0.)
    offset = now - reference[:, None]
    distance = np.linalg.norm(offset, axis=2)
    for team in (0, 1):
        members = np.flatnonzero(teams == team)
        if not len(members):
            continue
        team_distance = np.where(visible[:, members] & np.isfinite(distance[:, members]), distance[:, members], np.inf)
        order = np.argsort(team_distance, axis=1, kind="stable")[:, :PER_TEAM]
        chosen = members[order]
        rows = np.arange(len(frames))[:, None]
        filled = np.isfinite(np.take_along_axis(team_distance, order, axis=1))
        values = np.stack([offset[rows, chosen, 0] / 10, offset[rows, chosen, 1] / 10, distance[rows, chosen] / 10,
                           velocity[rows, chosen, 0] / 5, velocity[rows, chosen, 1] / 5,
                           now[rows, chosen, 0] / half_pitch[0], now[rows, chosen, 1] / half_pitch[1],
                           np.broadcast_to(seen[:, None], chosen.shape).astype(np.float64)], axis=-1)
        values[~filled] = np.nan
        width = order.shape[1]
        result[:, team, :width] = values.astype(np.float32)
    return result


def token_inputs(window):
    """(values with NaN set to 0, mask) of a token array."""
    mask = np.isfinite(window).all(axis=-1)
    return np.where(mask[..., None], np.nan_to_num(window), 0).astype(np.float32), mask.astype(np.float32)


# -- NumPy inference -------------------------------------------------------------------------------

def gelu(values):
    return values * np.float32(.5) * (1 + erf(values * np.float32(2 ** -.5)))


def pooled(token_values, mask, weights):
    """(..., 4 * HIDDEN) pooled encodings: home mean, home max, away mean, away max."""
    hidden = gelu(token_values @ weights["encoder.0.weight"].T + weights["encoder.0.bias"])
    hidden = gelu(hidden @ weights["encoder.2.weight"].T + weights["encoder.2.bias"])
    present = mask[..., None]
    count = present.sum(axis=-2)
    mean = (hidden * present).sum(axis=-2) / np.maximum(count, 1)
    largest = np.where(present > 0, hidden, -np.inf).max(axis=-2)
    largest = np.where(count > 0, largest, 0)
    parts = [mean[..., 0, :], largest[..., 0, :], mean[..., 1, :], largest[..., 1, :]]
    return np.concatenate(parts, axis=-1).astype(np.float32)


def forward(values, token_values, mask, weights):
    """Logits of the player-set network: (batch, time, 4)."""
    pool = pooled(token_values, mask, weights)
    residual = pool @ weights["pool_project.weight"].T + weights["pool_project.bias"]
    return temporal_forward(values, weights, residual=residual)


def load(folder, filename):
    from experiments.dfl_transfer_data import digest
    fit = json.loads((folder / "fit.json").read_text())
    path, norm_path = folder / filename, folder / "normalization.npz"
    if digest(path) != fit["models"][filename] or digest(norm_path) != fit["normalization"]:
        raise ValueError("Player-set checkpoint identity differs")
    with np.load(path, allow_pickle=False) as data:
        weights = {key: data[key] for key in data.files}
    with np.load(norm_path, allow_pickle=False) as data:
        location, scale, permutation = (data[name] for name in ("location", "scale", "permutation"))
    return weights, fit["columns"], location, scale, permutation


def predict(checkpoint, frames, values, columns, player_tokens, length):
    """Swap-averaged home, dead, dead edge and owner edge at every frame, like `temporal_cpu.predict`."""
    weights, expected, location, scale, permutation = checkpoint
    if columns != expected:
        raise ValueError("Player-set inference feature columns differ")
    token_values, mask = token_inputs(player_tokens)
    variants = np.stack([normalize(values, location, scale), normalize(values[:, permutation], location, scale)])
    token_variants = np.stack([token_values, token_values[:, ::-1]])
    mask_variants = np.stack([mask, mask[:, ::-1]])
    p = expit(forward(variants, token_variants, mask_variants, weights))
    home = (p[0, :, 1] + 1 - p[1, :, 1]) / 2
    dead = (p[0, :, 0] + p[1, :, 0]) / 2
    dead_edge, owner_edge = ((p[0, :, index] + p[1, :, index]) / 2 for index in (2, 3))
    timeline = np.arange(length)
    return tuple(np.interp(timeline, frames, item).astype(np.float32) for item in (home, dead, dead_edge, owner_edge))


# -- PyTorch training ------------------------------------------------------------------------------

def torch_network(columns, width, dropout):
    import torch
    from torch import nn
    from dfl_submission import temporal_network as tn

    class PlayerSetNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Conv1d(columns * 2, width, 1)
            self.encoder = nn.Sequential(nn.Linear(TOKEN, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU())
            self.pool_project = nn.Linear(4 * HIDDEN, width)
            self.temporal = nn.Sequential(*(tn.TemporalBlock(d, width) for d in DILATIONS))
            for block in self.temporal:
                block.dropout.p = dropout
            self.norm = nn.LayerNorm(width)
            self.outputs = nn.Linear(width, 4)

        def forward(self, values, token_values, mask):
            hidden = self.encoder(token_values)
            present = mask.unsqueeze(-1)
            count = present.sum(dim=-2)
            mean = (hidden * present).sum(dim=-2) / count.clamp_min(1)
            largest = torch.where(present > 0, hidden, torch.full_like(hidden, -1e9)).amax(dim=-2)
            largest = torch.where(count > 0, largest, torch.zeros_like(largest))
            pool = torch.cat([mean[..., 0, :], largest[..., 0, :], mean[..., 1, :], largest[..., 1, :]], dim=-1)
            projected = self.project(values.transpose(1, 2)) + self.pool_project(pool).transpose(1, 2)
            return self.outputs(self.norm(self.temporal(projected).transpose(1, 2)))

    return PlayerSetNetwork()


def training_data(source, token_folder):
    """`temporal_network.training_data` plus each record's tokens, hash-checked."""
    from dfl_submission import temporal_network as tn
    from experiments.dfl_transfer_data import digest
    groups, manifest, permutation, location, scale = tn.training_data(source)
    token_manifest = json.loads((token_folder / "manifest.json").read_text())
    if token_manifest["base_manifest"] != digest(source / "manifest.json"):
        raise ValueError(f"{token_folder} holds tokens for a different prepared folder")
    by_source = {}
    for record in manifest["records"]:
        entry = token_manifest["records"][record["file"]]
        path = token_folder / record["file"]
        if digest(path) != entry:
            raise ValueError(f"Token file hash differs: {path}")
        with np.load(path, allow_pickle=False) as item:
            by_source.setdefault(record["source"], []).append(item["tokens"])
    joined = {}
    for key, sequences in groups.items():
        if len(sequences) != len(by_source[key]):
            raise ValueError(f"Token records do not match the prepared records of {key}")
        joined[key] = [(values, labels, tok) for (values, labels), tok in zip(sequences, by_source[key], strict=True)]
        for values, _, tok in joined[key]:
            if len(tok) != len(values):
                raise ValueError("Token frames differ from the prepared frames")
    return joined, manifest, permutation, location, scale


def batch(groups, rng, permutation, location, scale, size):
    import torch
    xs, ts, ms, ys = [], [], [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(size // len(groups)):
            values, labels, tok = sequences[rng.integers(len(sequences))]
            start = rng.integers(0, len(values) - CORE - 2 * HALO + 1)
            window = values[start:start + CORE + 2 * HALO]
            token_window = tok[start:start + CORE + 2 * HALO].astype(np.float32)
            output = labels[start + HALO:start + HALO + CORE]
            if rng.integers(2):
                window = window[:, permutation]
                token_window = token_window[:, ::-1]
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            token_values, mask = token_inputs(token_window)
            xs.append(normalize(window, location, scale))
            ts.append(token_values)
            ms.append(mask)
            ys.append(output)
    return tuple(torch.from_numpy(np.ascontiguousarray(np.stack(item))) for item in (xs, ts, ms, ys))


def train(source, token_folder, output, device, *, seed, boundary_boost, width=96, steps=5400, lr_scale=2.,
          batch_size=32, weight_decay=.01, dropout=.2):
    import torch
    from dfl_submission import hyper_network as hn, temporal_network as tn
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    output.mkdir(parents=True, exist_ok=False)
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, manifest, permutation, location, scale = training_data(source, token_folder)
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = torch_network(len(location), width, dropout).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes = {}
    for step in range(1, steps + 1):
        values, token_values, mask, labels = (item.to(device) for item in
                                              batch(groups, rng, permutation, location, scale, batch_size))
        optimizer.param_groups[0]["lr"] = hn.learning_rate(step, steps, lr_scale)
        optimizer.zero_grad(set_to_none=True)
        value = tn.loss(net(values, token_values, mask)[:, HALO:HALO + CORE], labels, boundary_boost)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"Player-set training step {step} of {steps}, loss {value.item():.4f}", flush=True)
        if step in saved:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "token_manifest": digest(token_folder / "manifest.json"), "training_device": device,
               "torch": torch.__version__, "seed": seed, "boundary_boost": boundary_boost, "width": width,
               "steps": steps, "lr_scale": lr_scale, "batch": batch_size, "weight_decay": weight_decay,
               "dropout": dropout, "player_set": {"per_team": PER_TEAM, "token": TOKEN, "hidden": HIDDEN}})


def export(folder, output):
    """NumPy export of every snapshot, checked against PyTorch within D15's tolerance."""
    import torch
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    from shutil import copyfile
    output.mkdir(parents=True, exist_ok=False)
    fit = json.loads((folder / "fit.json").read_text())
    norm_path = folder / "normalization.npz"
    if digest(norm_path) != fit["normalization"]:
        raise ValueError("Player-set normalization identity differs")
    copyfile(norm_path, output / norm_path.name)
    torch.set_num_threads(1)
    columns = len(fit["columns"])
    net = torch_network(columns, fit["width"], fit["dropout"]).eval()
    rng = np.random.default_rng(13)
    values = rng.normal(size=(2, 180, columns * 2)).astype(np.float32)
    token_values = rng.normal(size=(2, 180, 2, PER_TEAM, TOKEN)).astype(np.float32)
    mask = (rng.uniform(size=(2, 180, 2, PER_TEAM)) > .3).astype(np.float32)
    mask[0, :5, 1] = 0                                   # a team with no visible player
    hashes, differences = {}, {}
    for filename, expected in fit["models"].items():
        path = folder / filename
        if digest(path) != expected:
            raise ValueError(f"Player-set checkpoint identity differs: {filename}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        weights = {name: tensor.numpy() for name, tensor in state.items()}
        with torch.inference_mode():
            reference = net(torch.from_numpy(values), torch.from_numpy(token_values), torch.from_numpy(mask)).numpy()
        result = forward(values, token_values, mask, weights)
        np.testing.assert_allclose(result, reference, atol=2e-5, rtol=1e-5)
        destination = output / Path(filename).with_suffix(".npz").name
        np.savez(destination, **weights)
        hashes[destination.name] = digest(destination)
        differences[destination.name] = float(np.max(np.abs(result - reference)))
    write_json(output / "fit.json", dict(fit, models=hashes, original_checkpoints=fit["models"],
                                        export_max_logit_difference=differences))
    print(f"CPU export verified: {differences}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="action", required=True)
    training = actions.add_parser("train")
    training.add_argument("--source", required=True, type=Path)
    training.add_argument("--tokens", required=True, type=Path)
    training.add_argument("--out", required=True, type=Path)
    training.add_argument("--device", choices=("cpu", "mps"), default="mps")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--boundary-boost", type=float, required=True)
    conversion = actions.add_parser("export")
    conversion.add_argument("--source", required=True, type=Path)
    conversion.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "train":
        from dfl_submission.external_only import require_external_only
        require_external_only()
        train(args.source, args.tokens, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost)
    else:
        export(args.source, args.out)
