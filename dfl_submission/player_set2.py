"""D44: configurable player-set tokens, network and trainer (DECISIONS.md D44).

`player_set` (on the submission's build path) stays byte-identical. This module adds three options:
`per_team` (players per team), `hidden` (encoder width) and `relative` (two more values per player: its
velocity relative to the ball's). Its inference is `player_set.forward` and `player_set.predict`, which
take any token width and encoder size from their inputs and weights.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_submission import player_set as ps


DEFAULT = {"per_team": ps.PER_TEAM, "hidden": ps.HIDDEN, "relative": False}


def token_width(config):
    return ps.TOKEN + (2 if config["relative"] else 0)


def tokens(positions, ball, teams, pitch, frames, config=DEFAULT):
    """`player_set.tokens` with `per_team` players per team and, if `relative`, the relative velocity appended."""
    per_team, relative = config["per_team"], config["relative"]
    positions = np.asarray(positions, dtype=np.float64)
    ball = np.asarray(ball, dtype=np.float64)
    teams = np.asarray(teams)
    frames = np.asarray(frames)
    half_pitch = np.asarray(pitch, dtype=np.float64) / 2
    width = token_width(config)
    result = np.full((len(frames), 2, per_team, width), np.nan, dtype=np.float32)
    now = positions[frames]
    earlier = np.maximum(frames - ps.LAG, 0)
    before = positions[earlier]
    seen = np.isfinite(ball[frames]).all(axis=1)
    visible = np.isfinite(now).all(axis=2)
    count = visible.sum(axis=1)
    centroid = np.divide(np.where(visible[..., None], now, 0).sum(axis=1), count[:, None],
                         out=np.full((len(frames), 2), np.nan), where=count[:, None] > 0)
    reference = np.where(seen[:, None], ball[frames], centroid)
    velocity = (now - before) * (25 / ps.LAG)
    speed = np.linalg.norm(velocity, axis=2)
    moving = np.isfinite(speed) & (speed <= ps.MAX_SPEED) & (frames[:, None] >= ps.LAG)
    velocity = np.where(moving[..., None], velocity, 0.)
    ball_velocity = (ball[frames] - ball[earlier]) * (25 / ps.LAG)
    ball_moving = seen & np.isfinite(ball[earlier]).all(axis=1) & (frames >= ps.LAG)
    relative_velocity = np.where((moving & ball_moving[:, None])[..., None], velocity - np.nan_to_num(ball_velocity)[:, None],
                                 0.)
    offset = now - reference[:, None]
    distance = np.linalg.norm(offset, axis=2)
    for team in (0, 1):
        members = np.flatnonzero(teams == team)
        if not len(members):
            continue
        team_distance = np.where(visible[:, members] & np.isfinite(distance[:, members]), distance[:, members], np.inf)
        order = np.argsort(team_distance, axis=1, kind="stable")[:, :per_team]
        chosen = members[order]
        rows = np.arange(len(frames))[:, None]
        filled = np.isfinite(np.take_along_axis(team_distance, order, axis=1))
        columns = [offset[rows, chosen, 0] / 10, offset[rows, chosen, 1] / 10, distance[rows, chosen] / 10,
                   velocity[rows, chosen, 0] / 5, velocity[rows, chosen, 1] / 5,
                   now[rows, chosen, 0] / half_pitch[0], now[rows, chosen, 1] / half_pitch[1],
                   np.broadcast_to(seen[:, None], chosen.shape).astype(np.float64)]
        if relative:
            columns += [relative_velocity[rows, chosen, 0] / 5, relative_velocity[rows, chosen, 1] / 5]
        values = np.stack(columns, axis=-1)
        values[~filled] = np.nan
        result[:, team, :order.shape[1]] = values.astype(np.float32)
    return result


def torch_network(columns, width, dropout, config=DEFAULT):
    import torch
    from torch import nn
    from dfl_submission import temporal_network as tn
    hidden, token = config["hidden"], token_width(config)

    class PlayerSetNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Conv1d(columns * 2, width, 1)
            self.encoder = nn.Sequential(nn.Linear(token, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
            self.pool_project = nn.Linear(4 * hidden, width)
            self.temporal = nn.Sequential(*(tn.TemporalBlock(d, width) for d in ps.DILATIONS))
            for block in self.temporal:
                block.dropout.p = dropout
            self.norm = nn.LayerNorm(width)
            self.outputs = nn.Linear(width, 4)

        def forward(self, values, token_values, mask):
            encoded = self.encoder(token_values)
            present = mask.unsqueeze(-1)
            total = present.sum(dim=-2)
            mean = (encoded * present).sum(dim=-2) / total.clamp_min(1)
            largest = torch.where(present > 0, encoded, torch.full_like(encoded, -1e9)).amax(dim=-2)
            largest = torch.where(total > 0, largest, torch.zeros_like(largest))
            pool = torch.cat([mean[..., 0, :], largest[..., 0, :], mean[..., 1, :], largest[..., 1, :]], dim=-1)
            projected = self.project(values.transpose(1, 2)) + self.pool_project(pool).transpose(1, 2)
            return self.outputs(self.norm(self.temporal(projected).transpose(1, 2)))

    return PlayerSetNetwork()


def training_data(source, token_folder):
    """`player_set.training_data` that also accepts a fold folder of the tokens' prepared folder."""
    from dfl_submission import temporal_network as tn
    from experiments.dfl_transfer_data import digest
    groups, manifest, permutation, location, scale = tn.training_data(source)
    token_manifest = json.loads((token_folder / "manifest.json").read_text())
    if token_manifest["base_manifest"] not in (digest(source / "manifest.json"), manifest.get("base_manifest")):
        raise ValueError(f"{token_folder} holds tokens for a different prepared folder")
    by_source = {}
    for record in manifest["records"]:
        path = token_folder / record["file"]
        if digest(path) != token_manifest["records"][record["file"]]:
            raise ValueError(f"Token file hash differs: {path}")
        with np.load(path, allow_pickle=False) as item:
            by_source.setdefault(record["source"], []).append(item["tokens"])
    joined = {}
    for key, sequences in groups.items():
        joined[key] = [(values, labels, tok) for (values, labels), tok in zip(sequences, by_source[key], strict=True)]
        if any(len(tok) != len(values) for values, _, tok in joined[key]):
            raise ValueError("Token frames differ from the prepared frames")
    return joined, manifest, permutation, location, scale


def train(source, token_folder, output, device, *, seed, boundary_boost, config=DEFAULT, width=96, steps=5400,
          lr_scale=2., batch_size=32, weight_decay=.01, dropout=.2):
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
    net = torch_network(len(location), width, dropout, config).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes = {}
    for step in range(1, steps + 1):
        values, token_values, mask, labels = (item.to(device) for item in
                                              ps.batch(groups, rng, permutation, location, scale, batch_size))
        optimizer.param_groups[0]["lr"] = hn.learning_rate(step, steps, lr_scale)
        optimizer.zero_grad(set_to_none=True)
        value = tn.loss(net(values, token_values, mask)[:, ps.HALO:ps.HALO + ps.CORE], labels, boundary_boost)
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
               "dropout": dropout, "player_set": dict(config, token=token_width(config))})


def final_weights(folder):
    """The last raw snapshot of a trained run, as NumPy arrays, with its normalisation and config."""
    import torch
    from experiments.dfl_transfer_data import digest
    fit = json.loads((folder / "fit.json").read_text())
    filename = f"temporal_{fit['steps']}.pt"
    if digest(folder / filename) != fit["models"][filename]:
        raise ValueError(f"Checkpoint identity differs: {folder / filename}")
    state = torch.load(folder / filename, map_location="cpu", weights_only=True)
    with np.load(folder / "normalization.npz", allow_pickle=False) as item:
        norm = tuple(item[key] for key in ("location", "scale", "permutation"))
    return {key: value.numpy() for key, value in state.items()}, norm, fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--boundary-boost", type=float, required=True)
    parser.add_argument("--per-team", type=int, default=ps.PER_TEAM)
    parser.add_argument("--hidden", type=int, default=ps.HIDDEN)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--lr-scale", type=float, default=2.)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=.2)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    require_external_only()
    train(args.source, args.tokens, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost,
          config={"per_team": args.per_team, "hidden": args.hidden, "relative": args.relative},
          lr_scale=args.lr_scale, batch_size=args.batch, dropout=args.dropout)
