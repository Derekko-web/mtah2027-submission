"""D50: player tokens in the attacking frame (DECISIONS.md D50).

The attacking sign of a half is read from the players' positions alone, without labels or fitted
values: over the half's long-context frames (every fifth 25 Hz frame) with at least seven visible
players in each team, D is the mean of (mean x of the away players - mean x of the home players). The
home team attacks towards +x when D > 0 and towards -x otherwise; the sign is +1 or -1 accordingly.
The pipeline check compares it with the sign of the same difference at the first such frame.

Canonical tokens multiply the token's x offset, x velocity and x position (values 0, 3 and 5 of
`player_set.tokens`) by the half's sign, so the home team always attacks towards +x. When the teams
are swapped (the trainer's team-order augmentation and the swap-averaged inference), those three
values are negated as well, so the first team still attacks towards +x. A column flip (the columns
D47's parity check finds to change sign under the x mirror) is applied the same way; with no such
column the 162 columns are unchanged.

`train` is `player_set.train` with that one change (the same network, loss, schedule, snapshots and
fit record, so `player_set export` exports and checks it); `predict` mirrors `player_set.predict`.
With every sign +1 and the swap negation off, both are `player_set`'s exactly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from dfl_submission import player_set as ps
from dfl_submission.temporal_cpu import normalize


X_VALUES = (0, 3, 5)                                   # x offset, x velocity, x position
X_SIGNS = np.ones(ps.TOKEN, np.float32)
X_SIGNS[list(X_VALUES)] = -1
MIN_PLAYERS = 7                                        # visible players per team for a frame to count


# -- the attacking sign ----------------------------------------------------------------------------

def long_context_frames(length):
    return np.arange(0, length, ps.STRIDE)


def side(difference):
    """+1 when the home team attacks towards +x (away players at larger x), -1 otherwise."""
    return 1 if difference > 0 else -1


def summary(difference, counts, frames):
    """The D50 sign of one half from per-frame (away - home) mean-x differences and visible counts.

    `counts` is (frames, 2): the number of visible players of each team counted in the means.
    """
    difference = np.asarray(difference, dtype=np.float64)
    counts = np.asarray(counts)
    frames = np.asarray(frames)
    qualifying = (counts >= MIN_PLAYERS).all(axis=1)
    if not qualifying.any():
        return {"frames": 0, "d": None, "sign": None, "first_frame": None, "first_difference": None,
                "first_sign": None, "agrees": False}
    d = float(difference[qualifying].mean())
    first = int(np.flatnonzero(qualifying)[0])
    return {"frames": int(qualifying.sum()), "d": d, "sign": side(d), "first_frame": int(frames[first]),
            "first_difference": float(difference[first]), "first_sign": side(difference[first]),
            "agrees": side(d) == side(difference[first])}


def attacking(positions, teams, frames=None):
    """D50's sign of one half from its tracking: every visible player of each team counts."""
    positions = np.asarray(positions, dtype=np.float64)
    teams = np.asarray(teams)
    frames = long_context_frames(len(positions)) if frames is None else np.asarray(frames)
    now = positions[frames]
    visible = np.isfinite(now).all(axis=2)
    counts, means = [], []
    for team in (0, 1):
        members = np.flatnonzero(teams == team)
        seen = visible[:, members]
        count = seen.sum(axis=1)
        total = np.where(seen, now[:, members, 0], 0.).sum(axis=1)
        counts.append(count)
        means.append(np.divide(total, count, out=np.full(len(frames), np.nan), where=count > 0))
    counts = np.stack(counts, axis=1)
    result = summary(means[1] - means[0], counts, frames)
    result["max_visible"] = [int(counts[:, 0].max(initial=0)), int(counts[:, 1].max(initial=0))]
    return result


def attacking_from_tokens(player_tokens, frames=None):
    """The same sign from a stored token array: value 5 is each kept player's x position / half-length.

    The tokens keep each team's visible players nearest the reference point, at most `PER_TEAM`, so
    this equals `attacking` (up to the positive half-length scale) wherever no team has more than
    `PER_TEAM` visible players.
    """
    values = np.asarray(player_tokens, dtype=np.float64)
    frames = np.arange(len(values)) if frames is None else np.asarray(frames)
    present = np.isfinite(values).all(axis=-1)
    counts = present.sum(axis=-1)
    total = np.where(present, values[..., 5], 0.).sum(axis=-1)
    means = np.divide(total, counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    return summary(means[:, 1] - means[:, 0], counts, frames)


# -- canonical tokens --------------------------------------------------------------------------------

def canonical(player_tokens, sign):
    """Tokens with the x offset, x velocity and x position multiplied by `sign` (exact in any float dtype)."""
    if sign not in (1, -1):
        raise ValueError(f"An attacking sign is +1 or -1, not {sign}")
    if sign == 1:
        return player_tokens
    factors = np.ones(ps.TOKEN, dtype=np.asarray(player_tokens).dtype)
    factors[list(X_VALUES)] = -1
    return player_tokens * factors


def column_factors(odd, sign):
    """(columns,) float32: `sign` on the x-odd columns, 1 elsewhere; None when there is no x-odd column."""
    if odd is None or not np.any(odd):
        return None
    return np.where(np.asarray(odd, dtype=bool), np.float32(sign), np.float32(1)).astype(np.float32)


def swapped(token_window, negate):
    """The team-swapped token window: team order reversed and, if `negate`, the x values negated."""
    result = token_window[:, ::-1]
    return result * X_SIGNS.astype(result.dtype) if negate else result


# -- NumPy inference -------------------------------------------------------------------------------

def predict(checkpoint, frames, values, columns, player_tokens, length, *, sign, negate=True, odd=None):
    """`player_set.predict` on canonical tokens (and columns), swap-averaged with the x negation.

    `sign` is the half's attacking sign; `odd` marks the x-odd columns (None: no column changes).
    With sign +1, `negate` False and no x-odd column, this is `player_set.predict` exactly.
    """
    weights, expected, location, scale, permutation = checkpoint
    if columns != expected:
        raise ValueError("Player-set inference feature columns differ")
    token_values, mask = ps.token_inputs(canonical(np.asarray(player_tokens), sign))
    factors, flip = column_factors(odd, sign), column_factors(odd, -1)
    values = values if factors is None else values * factors
    swapped_values = values[:, permutation] if flip is None or not negate else values[:, permutation] * flip
    variants = np.stack([normalize(values, location, scale), normalize(swapped_values, location, scale)])
    token_variants = np.stack([token_values, swapped(token_values, negate)])
    mask_variants = np.stack([mask, mask[:, ::-1]])
    p = expit(ps.forward(variants, token_variants, mask_variants, weights))
    home = (p[0, :, 1] + 1 - p[1, :, 1]) / 2
    dead = (p[0, :, 0] + p[1, :, 0]) / 2
    dead_edge, owner_edge = ((p[0, :, index] + p[1, :, index]) / 2 for index in (2, 3))
    timeline = np.arange(length)
    return tuple(np.interp(timeline, frames, item).astype(np.float32) for item in (home, dead, dead_edge, owner_edge))


# -- PyTorch training ------------------------------------------------------------------------------

def record_files(manifest):
    """{source: [record file, ...]} in the order `temporal_network.training_data` groups the records."""
    files = {}
    for record in manifest["records"]:
        files.setdefault(record["source"], []).append(record["file"])
    return files


def training_data(source, token_folder, signs, odd=None):
    """`player_set.training_data` with each record's tokens (and x-odd columns) in its attacking frame.

    `signs` is {record file: +1 or -1} for every record.
    """
    groups, manifest, permutation, location, scale = ps.training_data(source, token_folder)
    files = record_files(manifest)
    if set(signs) != {name for names in files.values() for name in names}:
        raise ValueError("The attacking signs do not cover exactly the prepared records")
    joined = {}
    for key, sequences in groups.items():
        joined[key] = []
        for (values, labels, tok), name in zip(sequences, files[key], strict=True):
            sign = int(signs[name])
            factors = column_factors(odd, sign)
            joined[key].append((values if factors is None else values * factors, labels, canonical(tok, sign)))
    return joined, manifest, permutation, location, scale


def batch(groups, rng, permutation, location, scale, size, *, negate, odd=None):
    """`player_set.batch`, with the team swap negating the x values when `negate`."""
    import torch
    flip = column_factors(odd, -1)
    xs, ts, ms, ys = [], [], [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(size // len(groups)):
            values, labels, tok = sequences[rng.integers(len(sequences))]
            start = rng.integers(0, len(values) - ps.CORE - 2 * ps.HALO + 1)
            window = values[start:start + ps.CORE + 2 * ps.HALO]
            token_window = tok[start:start + ps.CORE + 2 * ps.HALO].astype(np.float32)
            output = labels[start + ps.HALO:start + ps.HALO + ps.CORE]
            if rng.integers(2):
                window = window[:, permutation]
                if negate and flip is not None:
                    window = window * flip
                token_window = swapped(token_window, negate)
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            token_values, mask = ps.token_inputs(token_window)
            xs.append(normalize(window, location, scale))
            ts.append(token_values)
            ms.append(mask)
            ys.append(output)
    return tuple(torch.from_numpy(np.ascontiguousarray(np.stack(item))) for item in (xs, ts, ms, ys))


def read_signs(path, source, token_folder):
    """{record file: sign} from a D50 sign file, checked against the prepared and token folders."""
    from experiments.dfl_transfer_data import digest
    stored = json.loads(Path(path).read_text())
    if stored["base_manifest"] != digest(source / "manifest.json"):
        raise ValueError(f"{path} holds signs for a different prepared folder")
    if stored["token_manifest"] != digest(token_folder / "manifest.json"):
        raise ValueError(f"{path} holds signs for a different token folder")
    return {name: int(entry["sign"]) for name, entry in stored["records"].items()}


def train(source, token_folder, output, device, *, seed, boundary_boost, signs, negate, odd=None, width=96,
          steps=5400, lr_scale=2., batch_size=32, weight_decay=.01, dropout=.2, provenance=None):
    """`player_set.train` on canonical tokens with the swap negation; everything else is D42's."""
    import torch
    from dfl_submission import hyper_network as hn, temporal_network as tn
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    output.mkdir(parents=True, exist_ok=False)
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, manifest, permutation, location, scale = training_data(source, token_folder, signs, odd)
    if odd is not None and len(odd) != len(manifest["columns"]):
        raise ValueError("The x-odd column flags do not match the prepared columns")
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = ps.torch_network(len(location), width, dropout).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes = {}
    for step in range(1, steps + 1):
        values, token_values, mask, labels = (item.to(device) for item in batch(
            groups, rng, permutation, location, scale, batch_size, negate=negate, odd=odd))
        optimizer.param_groups[0]["lr"] = hn.learning_rate(step, steps, lr_scale)
        optimizer.zero_grad(set_to_none=True)
        value = tn.loss(net(values, token_values, mask)[:, ps.HALO:ps.HALO + ps.CORE], labels, boundary_boost)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"Direction-token training step {step} of {steps}, loss {value.item():.4f}", flush=True)
        if step in saved:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
    flipped = [] if odd is None else [manifest["columns"][i] for i in np.flatnonzero(odd)]
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "token_manifest": digest(token_folder / "manifest.json"), "training_device": device,
               "torch": torch.__version__, "seed": seed, "boundary_boost": boundary_boost, "width": width,
               "steps": steps, "lr_scale": lr_scale, "batch": batch_size, "weight_decay": weight_decay,
               "dropout": dropout, "player_set": {"per_team": ps.PER_TEAM, "token": ps.TOKEN, "hidden": ps.HIDDEN},
               "d50": {"negate_on_swap": negate, "negative_records": sorted(k for k, v in signs.items() if v < 0),
                       "records": len(signs), "x_odd_columns": flipped, **(provenance or {})}})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    actions = parser.add_subparsers(dest="action", required=True)
    training = actions.add_parser("train")
    training.add_argument("--source", required=True, type=Path)
    training.add_argument("--tokens", required=True, type=Path)
    training.add_argument("--out", required=True, type=Path)
    training.add_argument("--device", choices=("cpu", "mps"), default="mps")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--boundary-boost", type=float, required=True)
    training.add_argument("--signs", type=Path, help="D50 sign file of the prepared records")
    training.add_argument("--odd", type=Path, help="JSON list of x-odd column flags (D47's parity)")
    training.add_argument("--check", action="store_true",
                          help="every sign +1 and no swap negation: must reproduce player_set.train bit for bit")
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    require_external_only()
    manifest = json.loads((args.source / "manifest.json").read_text())
    if args.check:
        if args.signs is not None or args.odd is not None:
            raise ValueError("The check run takes no signs and no column flips")
        signs = {record["file"]: 1 for record in manifest["records"]}
        negate, odd, provenance = False, None, {"check": True}
    else:
        if args.signs is None:
            raise ValueError("A candidate run needs the sign file")
        signs = read_signs(args.signs, args.source, args.tokens)
        odd = None if args.odd is None else np.asarray(json.loads(args.odd.read_text()), dtype=bool)
        from experiments.dfl_transfer_data import digest
        negate = True
        provenance = {"check": False, "signs_file": digest(args.signs),
                      "odd_file": None if args.odd is None else digest(args.odd)}
    train(args.source, args.tokens, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost,
          signs=signs, negate=negate, odd=odd, provenance=provenance)


if __name__ == "__main__":
    main()
