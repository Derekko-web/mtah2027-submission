"""D51: an auxiliary ball-location target on D49's long-hidden views (DECISIONS.md D51).

On the frames of D49's new views where the view hides a ball the source tracking has, the network is
also asked where the ball is: its x and y offset from the token reference point (the mean position of
the visible players when the ball is hidden, `player_set.tokens`), divided by 10 m. The two extra
outputs come from a separate linear head on the player-set network's last hidden layer, trained with a
Huber loss (delta 1) added to D42's loss with weight 0.5. Other frames and views carry no auxiliary loss.

The original network, its loss, its optimiser and its gradient clipping are D42's (`player_set.train`):
the head is created after the network under a forked CPU random state, has its own AdamW with the same
settings and schedule, and is clipped on its own, so with the auxiliary weight at 0 the original
network trains exactly as D42's. Snapshots hold the original network's weights under D42's names
(`temporal_<step>.pt`), so `player_set export` and the NumPy inference are D42's; the head's weights
are saved beside them (`ball_<step>.pt`).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_submission import augment, hidden_views as hv, player_set
from dfl_submission.player_set import CORE, HALO


AUX_WEIGHT = .5
DELTA = 1.
SCALE = 10.            # metres, the token offset scale


# -- targets ---------------------------------------------------------------------------------------

def view_with_ball(model_set, half, *, swaps=True, provider_ball=None, skill_order=None):
    """`hidden_views.training_view` re-derived with the same seeds, also returning the ball before hiding.

    Returns (positions, view ball, ball before hiding, visible flags of the hide process), or None for
    halves that get no D49 view. The random draws are those of `training_view`, in the same order.
    """
    if half.source == "dfl":
        if half.view != "clean":
            return None
        rng = np.random.default_rng(hv.dfl_seed(model_set, half.match, half.index))
        positions, ball = hv.noise_and_dropout(half.positions, half.ball, rng)
        visibility = float(rng.uniform(*hv.DFL_VISIBILITY))
        visible = hv.long_hide(len(ball), visibility, rng)
        view_ball = ball.copy()
        view_ball[~visible] = np.nan
        if model_set == "augmented" and swaps:
            positions = augment.swap_teams(positions, half.teams, rng, every=augment.TRAIN_SWAP_EVERY,
                                           frames=augment.training_frames)
        return positions, view_ball, ball, visible
    if half.source == "skillcorner":
        rng = np.random.default_rng(hv.skill_seed(model_set, skill_order, half.index))
        ball = np.array(half.ball if provider_ball is None else provider_ball, dtype=np.float32, copy=True)
        visibility = float(rng.uniform(*hv.SKILL_VISIBILITY))
        visible = hv.long_hide(len(ball), visibility, rng)
        view_ball = ball.copy()
        view_ball[~visible] = np.nan
        return half.positions, view_ball, ball, visible
    raise ValueError(f"Unknown training source {half.source!r}")


def targets(positions, view_ball, source_ball, frames):
    """(len(frames), 2) float32 auxiliary targets, NaN where the frame carries no auxiliary loss.

    A frame carries a target where the view hides the ball, the source tracking has it and at least one
    player is visible; the target is the source ball's offset from the visible players' mean position
    (the token reference point when the ball is hidden), divided by 10 m.
    """
    positions = np.asarray(positions, dtype=np.float64)
    frames = np.asarray(frames)
    now = positions[frames]
    visible = np.isfinite(now).all(axis=2)
    count = visible.sum(axis=1)
    centroid = np.divide(np.where(visible[..., None], now, 0).sum(axis=1), count[:, None],
                         out=np.full((len(frames), 2), np.nan), where=count[:, None] > 0)
    hidden = ~np.isfinite(np.asarray(view_ball, dtype=np.float64)[frames]).all(axis=1)
    source = np.asarray(source_ball, dtype=np.float64)[frames]
    has = np.isfinite(source).all(axis=1)
    offset = (source - centroid) / SCALE
    offset[~(hidden & has & (count > 0))] = np.nan
    return offset.astype(np.float32)


# -- training data ---------------------------------------------------------------------------------

def training_data(source, token_folder, aux_folder):
    """`player_set.training_data` plus each record's auxiliary targets (None for records without)."""
    from experiments.dfl_transfer_data import digest
    groups, manifest, permutation, location, scale = player_set.training_data(source, token_folder)
    aux_manifest = json.loads((aux_folder / "manifest.json").read_text())
    if aux_manifest["base_manifest"] != digest(source / "manifest.json"):
        raise ValueError(f"{aux_folder} holds auxiliary targets for a different prepared folder")
    by_source = {}
    for record in manifest["records"]:
        name = record["file"]
        target = None
        if name in aux_manifest["records"]:
            path = aux_folder / name
            if digest(path) != aux_manifest["records"][name]:
                raise ValueError(f"Auxiliary target hash differs: {path}")
            with np.load(path, allow_pickle=False) as item:
                target = item["offset"]
        by_source.setdefault(record["source"], []).append(target)
    joined = {}
    for key, sequences in groups.items():
        joined[key] = []
        for (values, labels, tok), target in zip(sequences, by_source[key], strict=True):
            if target is not None and len(target) != len(values):
                raise ValueError("Auxiliary target frames differ from the prepared frames")
            joined[key].append((values, labels, tok, target))
    return joined, manifest, permutation, location, scale


def batch(groups, rng, permutation, location, scale, size):
    """`player_set.batch` with the same random draws, plus the windows' auxiliary targets (NaN where none)."""
    import torch
    xs, ts, ms, ys, bs = [], [], [], [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(size // len(groups)):
            values, labels, tok, target = sequences[rng.integers(len(sequences))]
            start = rng.integers(0, len(values) - CORE - 2 * HALO + 1)
            window = values[start:start + CORE + 2 * HALO]
            token_window = tok[start:start + CORE + 2 * HALO].astype(np.float32)
            output = labels[start + HALO:start + HALO + CORE]
            ball = (np.full((CORE, 2), np.nan, dtype=np.float32) if target is None
                    else target[start + HALO:start + HALO + CORE])
            if rng.integers(2):
                window = window[:, permutation]
                token_window = token_window[:, ::-1]
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            token_values, mask = player_set.token_inputs(token_window)
            xs.append(player_set.normalize(window, location, scale))
            ts.append(token_values)
            ms.append(mask)
            ys.append(output)
            bs.append(ball)
    return tuple(torch.from_numpy(np.ascontiguousarray(np.stack(item))) for item in (xs, ts, ms, ys, bs))


# -- the network -----------------------------------------------------------------------------------

def hidden_layer(net, values, token_values, mask):
    """The player-set network's last hidden layer, computed exactly as its `forward` does before `outputs`."""
    import torch
    hidden = net.encoder(token_values)
    present = mask.unsqueeze(-1)
    count = present.sum(dim=-2)
    mean = (hidden * present).sum(dim=-2) / count.clamp_min(1)
    largest = torch.where(present > 0, hidden, torch.full_like(hidden, -1e9)).amax(dim=-2)
    largest = torch.where(count > 0, largest, torch.zeros_like(largest))
    pool = torch.cat([mean[..., 0, :], largest[..., 0, :], mean[..., 1, :], largest[..., 1, :]], dim=-1)
    projected = net.project(values.transpose(1, 2)) + net.pool_project(pool).transpose(1, 2)
    return net.norm(net.temporal(projected).transpose(1, 2))


def ball_head(width):
    """The auxiliary head, initialised from a forked CPU random state so D42's random stream is untouched."""
    import torch
    with torch.random.fork_rng(devices=[]):
        return torch.nn.Linear(width, 2)


def aux_loss(predicted, target):
    """Huber loss (delta 1) over the frames and coordinates that carry a target; 0 when none does."""
    import torch
    from torch.nn import functional as F
    present = torch.isfinite(target).all(dim=-1)
    if not bool(present.any()):
        return predicted.sum() * 0
    return F.huber_loss(predicted[present], target[present], delta=DELTA, reduction="mean")


def train(source, token_folder, aux_folder, output, device, *, seed, boundary_boost, aux_weight=AUX_WEIGHT,
          width=96, steps=5400, lr_scale=2., batch_size=32, weight_decay=.01, dropout=.2):
    """`player_set.train` (D42's values) with the auxiliary head and loss."""
    import torch
    from dfl_submission import hyper_network as hn, temporal_network as tn
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    output.mkdir(parents=True, exist_ok=False)
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, manifest, permutation, location, scale = training_data(source, token_folder, aux_folder)
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = player_set.torch_network(len(location), width, dropout).to(device)
    head = ball_head(width).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes, head_hashes, aux_values = {}, {}, []
    for step in range(1, steps + 1):
        values, token_values, mask, labels, ball = (item.to(device) for item in
                                                    batch(groups, rng, permutation, location, scale, batch_size))
        rate = hn.learning_rate(step, steps, lr_scale)
        optimizer.param_groups[0]["lr"] = head_optimizer.param_groups[0]["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        head_optimizer.zero_grad(set_to_none=True)
        hidden = hidden_layer(net, values, token_values, mask)
        main = tn.loss(net.outputs(hidden)[:, HALO:HALO + CORE], labels, boundary_boost)
        extra = aux_loss(head(hidden)[:, HALO:HALO + CORE], ball)
        value = main + aux_weight * extra
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
        optimizer.step()
        head_optimizer.step()
        aux_values.append(float(extra.item()))
        if step % 100 == 0:
            print(f"Ball-target training step {step} of {steps}, loss {main.item():.4f}, auxiliary "
                  f"{np.mean(aux_values[-100:]):.4f}", flush=True)
        if step in saved:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
            head_name = f"ball_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in head.state_dict().items()}, output / head_name)
            head_hashes[head_name] = digest(output / head_name)
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "token_manifest": digest(token_folder / "manifest.json"), "training_device": device,
               "torch": torch.__version__, "seed": seed, "boundary_boost": boundary_boost, "width": width,
               "steps": steps, "lr_scale": lr_scale, "batch": batch_size, "weight_decay": weight_decay,
               "dropout": dropout, "player_set": {"per_team": player_set.PER_TEAM, "token": player_set.TOKEN,
                                                  "hidden": player_set.HIDDEN},
               "ball": {"protocol": "DECISIONS.md D51", "aux_weight": aux_weight, "delta": DELTA, "scale_m": SCALE,
                        "aux_manifest": digest(aux_folder / "manifest.json"), "heads": head_hashes,
                        "final_aux_loss_mean_last_100": float(np.mean(aux_values[-100:]))}})


# -- the auxiliary error on development views ------------------------------------------------------

def predicted_offsets(folder, step, values, player_tokens):
    """The head's offsets (len(values), 2) for one half, identity team order, on the CPU."""
    import torch
    fit = json.loads((folder / "fit.json").read_text())
    with np.load(folder / "normalization.npz", allow_pickle=False) as data:
        location, scale = data["location"], data["scale"]
    net = player_set.torch_network(len(fit["columns"]), fit["width"], fit["dropout"]).eval()
    net.load_state_dict(torch.load(folder / f"temporal_{step}.pt", map_location="cpu", weights_only=True))
    head = torch.nn.Linear(fit["width"], 2).eval()
    head.load_state_dict(torch.load(folder / f"ball_{step}.pt", map_location="cpu", weights_only=True))
    token_values, mask = player_set.token_inputs(player_tokens)
    inputs = player_set.normalize(values, location, scale)[None]
    with torch.inference_mode():
        hidden = hidden_layer(net, torch.from_numpy(inputs), torch.from_numpy(token_values[None]),
                              torch.from_numpy(mask[None]))
        return head(hidden)[0].numpy()


def distances(positions, view_ball, true_ball, frames, offsets):
    """Metres between the predicted and the true ball on frames where the view hides a ball the truth has,
    and the same for the visible players' mean position (the zero-offset baseline)."""
    target = targets(positions, view_ball, true_ball, frames).astype(np.float64)
    present = np.isfinite(target).all(axis=1)
    error = np.linalg.norm((offsets[present] - target[present]) * SCALE, axis=1)
    baseline = np.linalg.norm(target[present] * SCALE, axis=1)
    return error, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="action", required=True)
    training = actions.add_parser("train")
    training.add_argument("--source", required=True, type=Path)
    training.add_argument("--tokens", required=True, type=Path)
    training.add_argument("--aux", required=True, type=Path)
    training.add_argument("--out", required=True, type=Path)
    training.add_argument("--device", choices=("cpu", "mps"), default="mps")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--boundary-boost", type=float, required=True)
    training.add_argument("--aux-weight", type=float, default=AUX_WEIGHT)
    training.add_argument("--steps", type=int, default=5400)
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    require_external_only()
    train(args.source, args.tokens, args.aux, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost,
          aux_weight=args.aux_weight, steps=args.steps)


if __name__ == "__main__":
    main()
