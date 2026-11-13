"""D47: the player-set trainer with mirror and token-noise augmentation, and mirror-averaged NumPy inference.

`player_set` (on the submission's build path) stays byte-identical; this module reuses its network,
token layout, export and plain inference. Three training switches are added (DECISIONS.md D47):
- `seed_offset`: the run's seed is D42's seed plus this offset (S: +1000, +2000);
- `mirror`: each window is mirrored in x with probability 1/2 and in y with probability 1/2,
  independently; the token's x offset, x velocity and x position change sign (likewise y), and the
  162 columns take the sign map measured by `screen_trial parity`;
- `noise`: each present token is masked with probability 0.10 per frame; Gaussian noise of 0.3 m is
  added to each player's position (offset and pitch position together, distance recomputed) and of
  0.5 m/s to each velocity component. Tokens are stored scaled (`player_set.tokens`), so the noise is
  scaled the same way: offsets and distance by 1/10 m, velocity by 1/5 m/s, pitch position by the
  half-pitch of the record.
The augmentation draws come from a generator of their own, so the window and team-swap draws are
D42's whatever the switches; with every switch off, `train` reproduces `player_set.train` bit for bit.
`predict` averages the network over the four mirrors of the pitch times the two team orders.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from dfl_submission import player_set as ps
from dfl_submission.temporal_cpu import normalize


MIRRORS = ("identity", "x", "y", "xy")
TOKEN_SIGNS = {
    "identity": np.ones(ps.TOKEN, np.float32),
    "x": np.array([-1, 1, 1, -1, 1, -1, 1, 1], np.float32),      # x offset, x velocity, x position
    "y": np.array([1, -1, 1, 1, -1, 1, -1, 1], np.float32),      # y offset, y velocity, y position
}
TOKEN_SIGNS["xy"] = TOKEN_SIGNS["x"] * TOKEN_SIGNS["y"]
MASK_PROBABILITY = .10
POSITION_SD = .3            # metres
VELOCITY_SD = .5            # metres per second
OFFSET_SCALE, VELOCITY_SCALE = 10., 5.     # `player_set.tokens` divides offsets/distance by 10 m, velocity by 5 m/s
AUGMENT_STREAM = 4747       # second entropy word of the augmentation generator
DEFAULT = {"seed_offset": 0, "mirror": False, "noise": False}


def switches_off(config):
    return not config["mirror"] and not config["noise"]


# -- mirrors ---------------------------------------------------------------------------------------

def mirror_tracking(positions, ball, mirror):
    """Tracking reflected through the pitch centre line(s): x -> -x for `x`, y -> -y for `y`, both for `xy`.

    The coordinates are pitch-centred (`robust_features.extras`), so a sign change is the reflection;
    the arrays keep their dtype, so the extraction runs exactly as on the stored halves.
    """
    positions = np.array(positions, copy=True)
    ball = np.array(ball, copy=True)
    for axis, name in enumerate("xy"):
        if name in mirror and mirror != "identity":
            positions[..., axis] *= -1
            ball[..., axis] *= -1
    return positions, ball


def column_signs(sign_map, mirror, columns):
    """(len(columns),) float32 signs of the 162 columns under `mirror`, from a parity record's sign map."""
    if mirror == "identity":
        return np.ones(len(columns), np.float32)
    if sign_map is None:
        raise ValueError("Mirroring needs the column sign map of the parity check")
    if list(sign_map["columns"]) != list(columns):
        raise ValueError("The parity sign map was measured on different columns")
    return np.asarray(sign_map["signs"][mirror], dtype=np.float32)


def mirror_window(values, token_window, mirror, signs):
    """A raw feature window and a token window under `mirror` (NaN stays NaN)."""
    return values * signs, token_window * TOKEN_SIGNS[mirror]


# -- token noise -----------------------------------------------------------------------------------

def noisy_tokens(token_window, rng, half_pitch):
    """D47's token noise on a (time, 2, players, 8) token window; absent tokens stay absent."""
    window = np.asarray(token_window, dtype=np.float32)
    present = np.isfinite(window).all(axis=-1)
    shape = present.shape
    keep = rng.random(shape) >= MASK_PROBABILITY
    dx, dy = rng.normal(0., POSITION_SD, shape), rng.normal(0., POSITION_SD, shape)
    vx, vy = rng.normal(0., VELOCITY_SD, shape), rng.normal(0., VELOCITY_SD, shape)
    result = window.astype(np.float64)
    result[..., 0] += dx / OFFSET_SCALE
    result[..., 1] += dy / OFFSET_SCALE
    result[..., 2] = np.hypot(result[..., 0], result[..., 1])
    result[..., 3] += vx / VELOCITY_SCALE
    result[..., 4] += vy / VELOCITY_SCALE
    result[..., 5] += dx / half_pitch[0]
    result[..., 6] += dy / half_pitch[1]
    result[~(present & keep)] = np.nan
    return result.astype(np.float32)


# -- NumPy inference -------------------------------------------------------------------------------

def predict(checkpoint, frames, values, columns, player_tokens, length, *, mirrored=False, sign_map=None):
    """`player_set.predict`, or, if `mirrored`, its mean over the four mirrors of the pitch times the two team orders.

    For each mirror the two team orders are combined exactly as `player_set.predict` combines them
    (home: (p + 1 - p_swapped) / 2, the others: the mean); the four mirrors are then averaged.
    """
    if not mirrored:
        return ps.predict(checkpoint, frames, values, columns, player_tokens, length)
    weights, expected, location, scale, permutation = checkpoint
    if columns != expected:
        raise ValueError("Player-set inference feature columns differ")
    token_values, mask = ps.token_inputs(player_tokens)
    variants, token_variants, mask_variants = [], [], []
    for mirror in MIRRORS:
        signed = values * column_signs(sign_map, mirror, columns)
        signed_tokens = token_values * TOKEN_SIGNS[mirror]
        variants += [normalize(signed, location, scale), normalize(signed[:, permutation], location, scale)]
        token_variants += [signed_tokens, signed_tokens[:, ::-1]]
        mask_variants += [mask, mask[:, ::-1]]
    p = expit(ps.forward(np.stack(variants), np.stack(token_variants), np.stack(mask_variants), weights))
    home = np.mean([(p[2 * m, :, 1] + 1 - p[2 * m + 1, :, 1]) / 2 for m in range(len(MIRRORS))], axis=0)
    others = [np.mean([(p[2 * m, :, index] + p[2 * m + 1, :, index]) / 2 for m in range(len(MIRRORS))], axis=0)
              for index in (0, 2, 3)]
    timeline = np.arange(length)
    return tuple(np.interp(timeline, frames, item).astype(np.float32) for item in (home, *others))


# -- PyTorch training ------------------------------------------------------------------------------

def record_files(manifest):
    """{source: [record file, ...]} in the order `temporal_network.training_data` groups the records."""
    files = {}
    for record in manifest["records"]:
        files.setdefault(record["source"], []).append(record["file"])
    return files


def batch(groups, rng, augment_rng, permutation, location, scale, size, config, *, signs=None, pitches=None):
    """`player_set.batch`, with D47's mirror and noise applied after the team swap, drawn from `augment_rng`.

    `signs` is {mirror: (columns,) float32} and `pitches` {source: [(half-length, half-width), ...]}
    aligned with `groups`. With both switches off `augment_rng` is never used and the batch is
    `player_set.batch`'s exactly.
    """
    import torch
    xs, ts, ms, ys = [], [], [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(size // len(groups)):
            index = rng.integers(len(sequences))
            values, labels, tok = sequences[index]
            start = rng.integers(0, len(values) - ps.CORE - 2 * ps.HALO + 1)
            window = values[start:start + ps.CORE + 2 * ps.HALO]
            token_window = tok[start:start + ps.CORE + 2 * ps.HALO].astype(np.float32)
            output = labels[start + ps.HALO:start + ps.HALO + ps.CORE]
            if rng.integers(2):
                window = window[:, permutation]
                token_window = token_window[:, ::-1]
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            if config["mirror"]:
                flip_x, flip_y = augment_rng.integers(2), augment_rng.integers(2)
                mirror = {(0, 0): "identity", (1, 0): "x", (0, 1): "y", (1, 1): "xy"}[int(flip_x), int(flip_y)]
                window, token_window = mirror_window(window, token_window, mirror, signs[mirror])
            if config["noise"]:
                token_window = noisy_tokens(token_window, augment_rng, pitches[key][index])
            token_values, mask = ps.token_inputs(token_window)
            xs.append(normalize(window, location, scale))
            ts.append(token_values)
            ms.append(mask)
            ys.append(output)
    return tuple(torch.from_numpy(np.ascontiguousarray(np.stack(item))) for item in (xs, ts, ms, ys))


def load_pitches(path, manifest):
    """{source: [(half-length, half-width), ...]} aligned with the grouped records, from a parity pitch file."""
    stored = json.loads(Path(path).read_text())
    return {source: [tuple(np.asarray(stored[name], dtype=np.float64) / 2) for name in names]
            for source, names in record_files(manifest).items()}


def train(source, token_folder, output, device, *, seed, boundary_boost, config=DEFAULT, sign_map=None,
          pitch_file=None, width=96, steps=5400, lr_scale=2., batch_size=32, weight_decay=.01, dropout=.2):
    """`player_set.train` with D47's switches; `seed` is D42's seed, to which `seed_offset` is added."""
    import torch
    from dfl_submission import hyper_network as hn, temporal_network as tn
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    output.mkdir(parents=True, exist_ok=False)
    run_seed = seed + config["seed_offset"]
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    augment_rng = np.random.default_rng([run_seed, AUGMENT_STREAM])
    groups, manifest, permutation, location, scale = ps.training_data(source, token_folder)
    signs = ({mirror: column_signs(sign_map, mirror, manifest["columns"]) for mirror in MIRRORS}
             if config["mirror"] else None)
    pitches = load_pitches(pitch_file, manifest) if config["noise"] else None
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = ps.torch_network(len(location), width, dropout).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes = {}
    for step in range(1, steps + 1):
        values, token_values, mask, labels = (item.to(device) for item in batch(
            groups, rng, augment_rng, permutation, location, scale, batch_size, config, signs=signs, pitches=pitches))
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
    augmentation = dict(config, mask_probability=MASK_PROBABILITY, position_sd=POSITION_SD, velocity_sd=VELOCITY_SD,
                        sign_map=None if signs is None else {m: s.tolist() for m, s in signs.items()},
                        pitch_file=None if pitch_file is None else digest(Path(pitch_file)))
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "token_manifest": digest(token_folder / "manifest.json"), "training_device": device,
               "torch": torch.__version__, "seed": run_seed, "boundary_boost": boundary_boost, "width": width,
               "steps": steps, "lr_scale": lr_scale, "batch": batch_size, "weight_decay": weight_decay,
               "dropout": dropout, "player_set": {"per_team": ps.PER_TEAM, "token": ps.TOKEN, "hidden": ps.HIDDEN},
               "d47": augmentation})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--boundary-boost", type=float, required=True)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--sign-map", type=Path, help="parity record holding the column sign map")
    parser.add_argument("--pitches", type=Path, help="pitch sizes of the prepared records")
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    require_external_only()
    config = {"seed_offset": args.seed_offset, "mirror": args.mirror, "noise": args.noise}
    sign_map = None
    if args.mirror:
        record = json.loads(args.sign_map.read_text())
        if not record["passed"]:
            raise ValueError("The mirror parity check did not pass; nothing is trained with mirrors")
        sign_map = record["sign_map"]
    train(args.source, args.tokens, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost,
          config=config, sign_map=sign_map, pitch_file=args.pitches)


if __name__ == "__main__":
    main()
