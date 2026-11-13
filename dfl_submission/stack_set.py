"""D48: one player-set trainer and NumPy inference path holding every lever of the stack (DECISIONS.md D48).

Switches (`DEFAULT` is D42's recipe; the data folder is chosen by the caller: D42's or D49's sequences
and tokens):
- `direction` (C, D50): each record's tokens in its attacking frame (x offset, x velocity and x
  position times the half's sign); the team swap negates them as well (`direction_tokens`);
- `seed_offset` (D47's S1000 / S2000): the run's seed is D42's seed plus the offset;
- `mirror` (D47's F): None, "xy" (D47's mirrors in x and y) or "y" (y only). An x mirror would undo the
  attacking frame, so "xy" with `direction` is refused; "y" is the form that is compatible with C;
- `noise` (D47's N): D47's token noise (`player_set3.noisy_tokens`);
- `aux` (D51's LB): D51's auxiliary ball head and loss (`hidden_ball`), with its targets read from an
  auxiliary-target folder. With `direction` the target's x is put in the attacking frame and negated
  on the team swap like the token x; with a mirror it changes sign with the mirrored axis.

The window, team-swap and record draws come from D42's generator in D42's order; the mirror and noise
draws come from D47's second generator (`player_set3.AUGMENT_STREAM`); D51's head is created under a
forked random state. So with every switch off `train` is `player_set.train`; with only `direction` on it
is `direction_tokens.train`; with only `seed_offset`, `mirror` or `noise` it is `player_set3.train`; with
only `aux` it is `hidden_ball.train`. `predict` reduces likewise to `player_set.predict`,
`direction_tokens.predict` and `player_set3.predict`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from dfl_submission import direction_tokens as dt, player_set as ps, player_set3 as p3
from dfl_submission.temporal_cpu import normalize


DEFAULT = {"direction": False, "seed_offset": 0, "mirror": None, "noise": False, "aux": False}
MIRROR_CHOICES = (None, "xy", "y")
AXIS_SIGNS = {"identity": np.ones(2, np.float32), "x": np.array([-1, 1], np.float32), "y": np.array([1, -1], np.float32),
              "xy": np.array([-1, -1], np.float32)}


def config_of(**changes):
    config = dict(DEFAULT, **changes)
    validate(config)
    return config


def validate(config):
    if set(config) != set(DEFAULT):
        raise ValueError(f"Unknown or missing switches: {sorted(set(config) ^ set(DEFAULT))}")
    if config["mirror"] not in MIRROR_CHOICES:
        raise ValueError(f"mirror is one of {MIRROR_CHOICES}")
    if config["direction"] and config["mirror"] == "xy":
        raise ValueError("An x mirror undoes the attacking frame; with `direction` only the y mirror is allowed")
    if config["seed_offset"] not in (0, 1000, 2000):
        raise ValueError("D48's seed offsets are 0, 1000 and 2000")


def mirrors_of(config):
    """The pitch mirrors a network is trained with and averaged over at inference."""
    return {None: ("identity",), "xy": p3.MIRRORS, "y": ("identity", "y")}[config["mirror"]]


# -- the attacking sign with D50's fallback ----------------------------------------------------------

def other_half(name):
    """The record name of the other half of the same match and view (`..._0.npz` <-> `..._1.npz`)."""
    stem, _, index = Path(name).stem.rpartition("_")
    if index not in ("0", "1"):
        raise ValueError(f"{name} does not end in a half index")
    return f"{stem}_{1 - int(index)}.npz"


def with_fallback(entries, pair):
    """{key: sign} from {key: D50 entry}, a half without a sign taking D50's amendment's fallback.

    `pair(key)` names the other half of the same match. Returns (signs, {key: source of a fallback}).
    """
    signs, fallbacks = {}, {}
    for key, entry in entries.items():
        if entry["sign"] is not None:
            signs[key] = int(entry["sign"])
            continue
        other = entries.get(pair(key))
        other_sign = None if other is None else other["sign"]
        signs[key] = 1 if other_sign is None else -int(other_sign)
        fallbacks[key] = {"other_half": pair(key), "other_half_sign": other_sign}
    return signs, fallbacks


def read_signs(path, source, token_folder):
    """{record file: sign} from a sign file (D50's or D48's), checked against the folders, with the fallback."""
    from experiments.dfl_transfer_data import digest
    stored = json.loads(Path(path).read_text())
    if stored["base_manifest"] != digest(source / "manifest.json"):
        raise ValueError(f"{path} holds signs for a different prepared folder")
    if stored["token_manifest"] != digest(token_folder / "manifest.json"):
        raise ValueError(f"{path} holds signs for a different token folder")
    signs, _ = with_fallback(stored["records"], other_half)
    return signs


# -- training data ---------------------------------------------------------------------------------

def training_data(source, token_folder, *, signs=None, aux_folder=None):
    """`player_set.training_data` (or D51's, with auxiliary targets) with every record as a 4-tuple.

    With `signs` ({record file: sign}) each record's tokens, and the x of its auxiliary target, are put
    in its attacking frame.
    """
    if aux_folder is None:
        groups, manifest, permutation, location, scale = ps.training_data(source, token_folder)
        groups = {key: [(values, labels, tok, None) for values, labels, tok in items] for key, items in groups.items()}
    else:
        from dfl_submission import hidden_ball as hb
        groups, manifest, permutation, location, scale = hb.training_data(source, token_folder, aux_folder)
    if signs is not None:
        files = dt.record_files(manifest)
        if set(signs) != {name for names in files.values() for name in names}:
            raise ValueError("The attacking signs do not cover exactly the prepared records")
        for key, items in groups.items():
            canonical = []
            for (values, labels, tok, target), name in zip(items, files[key], strict=True):
                sign = int(signs[name])
                if target is not None and sign < 0:
                    target = target * np.array([-1, 1], dtype=target.dtype)
                canonical.append((values, labels, dt.canonical(tok, sign), target))
            groups[key] = canonical
    return groups, manifest, permutation, location, scale


def batch(groups, rng, augment_rng, permutation, location, scale, size, config, *, column_signs=None, pitches=None):
    """One batch with every switch; returns (values, tokens, mask, labels) plus the ball targets with `aux`.

    The draws from `rng` are D42's; mirror and noise draw from `augment_rng` after the team swap, as D47's.
    """
    import torch
    negate = config["direction"]
    xs, ts, ms, ys, bs = [], [], [], [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(size // len(groups)):
            index = rng.integers(len(sequences))
            values, labels, tok, target = sequences[index]
            start = rng.integers(0, len(values) - ps.CORE - 2 * ps.HALO + 1)
            window = values[start:start + ps.CORE + 2 * ps.HALO]
            token_window = tok[start:start + ps.CORE + 2 * ps.HALO].astype(np.float32)
            output = labels[start + ps.HALO:start + ps.HALO + ps.CORE]
            ball = None
            if config["aux"]:
                ball = (np.full((ps.CORE, 2), np.nan, dtype=np.float32) if target is None
                        else target[start + ps.HALO:start + ps.HALO + ps.CORE])
            if rng.integers(2):
                window = window[:, permutation]
                token_window = dt.swapped(token_window, negate)
                if ball is not None and negate:
                    ball = ball * AXIS_SIGNS["x"]
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            if config["mirror"] is not None:
                flip_x, flip_y = augment_rng.integers(2), augment_rng.integers(2)
                if config["mirror"] == "y":
                    flip_x = 0
                mirror = {(0, 0): "identity", (1, 0): "x", (0, 1): "y", (1, 1): "xy"}[int(flip_x), int(flip_y)]
                window, token_window = p3.mirror_window(window, token_window, mirror, column_signs[mirror])
                if ball is not None:
                    ball = ball * AXIS_SIGNS[mirror]
            if config["noise"]:
                token_window = p3.noisy_tokens(token_window, augment_rng, pitches[key][index])
            token_values, mask = ps.token_inputs(token_window)
            xs.append(normalize(window, location, scale))
            ts.append(token_values)
            ms.append(mask)
            ys.append(output)
            if ball is not None:
                bs.append(ball)
    items = (xs, ts, ms, ys, bs) if config["aux"] else (xs, ts, ms, ys)
    return tuple(torch.from_numpy(np.ascontiguousarray(np.stack(item))) for item in items)


def load_pitches(path, manifest):
    """{source: [(half-length, half-width), ...]} aligned with the grouped records (D47's format)."""
    return p3.load_pitches(path, manifest)


# -- training ----------------------------------------------------------------------------------------

def train(source, token_folder, output, device, *, seed, boundary_boost, config=DEFAULT, signs=None, sign_map=None,
          pitch_file=None, aux_folder=None, aux_weight=None, width=96, steps=5400, lr_scale=2., batch_size=32,
          weight_decay=.01, dropout=.2, provenance=None):
    """D42's training with the stack's switches; `seed` is D42's seed, to which `seed_offset` is added."""
    import torch
    from dfl_submission import hyper_network as hn, temporal_network as tn
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    validate(config)
    if config["direction"] != (signs is not None):
        raise ValueError("`direction` needs the records' signs, and signs need `direction`")
    if config["aux"] != (aux_folder is not None):
        raise ValueError("`aux` needs the auxiliary-target folder, and the folder needs `aux`")
    if config["noise"] and pitch_file is None:
        raise ValueError("`noise` needs the records' pitch sizes")
    output.mkdir(parents=True, exist_ok=False)
    run_seed = seed + config["seed_offset"]
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    augment_rng = np.random.default_rng([run_seed, p3.AUGMENT_STREAM])
    groups, manifest, permutation, location, scale = training_data(source, token_folder, signs=signs,
                                                                   aux_folder=aux_folder)
    column_signs = ({mirror: p3.column_signs(sign_map, mirror, manifest["columns"]) for mirror in p3.MIRRORS}
                    if config["mirror"] is not None else None)
    pitches = load_pitches(pitch_file, manifest) if config["noise"] else None
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = ps.torch_network(len(location), width, dropout).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    head = head_optimizer = None
    if config["aux"]:
        from dfl_submission import hidden_ball as hb
        aux_weight = hb.AUX_WEIGHT if aux_weight is None else aux_weight
        head = hb.ball_head(width).to(device)
        head_optimizer = torch.optim.AdamW(head.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes, head_hashes, aux_values = {}, {}, []
    for step in range(1, steps + 1):
        tensors = [item.to(device) for item in batch(groups, rng, augment_rng, permutation, location, scale, batch_size,
                                                     config, column_signs=column_signs, pitches=pitches)]
        rate = hn.learning_rate(step, steps, lr_scale)
        optimizer.param_groups[0]["lr"] = rate
        optimizer.zero_grad(set_to_none=True)
        if head is None:
            values, token_values, mask, labels = tensors
            value = tn.loss(net(values, token_values, mask)[:, ps.HALO:ps.HALO + ps.CORE], labels, boundary_boost)
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            optimizer.step()
        else:
            from dfl_submission import hidden_ball as hb
            values, token_values, mask, labels, ball = tensors
            head_optimizer.param_groups[0]["lr"] = rate
            head_optimizer.zero_grad(set_to_none=True)
            hidden = hb.hidden_layer(net, values, token_values, mask)
            main = tn.loss(net.outputs(hidden)[:, ps.HALO:ps.HALO + ps.CORE], labels, boundary_boost)
            extra = hb.aux_loss(head(hidden)[:, ps.HALO:ps.HALO + ps.CORE], ball)
            value = main + aux_weight * extra
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            head_optimizer.step()
            aux_values.append(float(extra.item()))
        if step % 100 == 0:
            print(f"Stack training step {step} of {steps}, loss {value.item():.4f}", flush=True)
        if step in saved:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
            if head is not None:
                head_name = f"ball_{step}.pt"
                torch.save({key: tensor.detach().cpu() for key, tensor in head.state_dict().items()}, output / head_name)
                head_hashes[head_name] = digest(output / head_name)
    stack = dict(config, negative_records=sorted(k for k, v in (signs or {}).items() if v < 0),
                 records=len(manifest["records"]), mirror_sign_map=None if column_signs is None else
                 {m: s.tolist() for m, s in column_signs.items()},
                 pitch_file=None if pitch_file is None else digest(Path(pitch_file)), **(provenance or {}))
    if head is not None:
        stack["aux"] = {"weight": aux_weight, "aux_manifest": digest(aux_folder / "manifest.json"), "heads": head_hashes,
                        "final_aux_loss_mean_last_100": float(np.mean(aux_values[-100:]))}
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "token_manifest": digest(token_folder / "manifest.json"), "training_device": device,
               "torch": torch.__version__, "seed": run_seed, "boundary_boost": boundary_boost, "width": width,
               "steps": steps, "lr_scale": lr_scale, "batch": batch_size, "weight_decay": weight_decay,
               "dropout": dropout, "player_set": {"per_team": ps.PER_TEAM, "token": ps.TOKEN, "hidden": ps.HIDDEN},
               "stack": stack})


# -- NumPy inference -------------------------------------------------------------------------------

def predict(checkpoint, frames, values, columns, player_tokens, length, *, config=DEFAULT, sign=None, sign_map=None):
    """Swap-averaged (and mirror-averaged) home, dead, dead edge and owner edge at every frame.

    With `direction`, `sign` is the half's attacking sign. One mirror: `direction_tokens.predict` (which
    is `player_set.predict` without `direction`); D47's four mirrors without `direction`:
    `player_set3.predict`; otherwise the mean over the mirrors of the swap-averaged outputs.
    """
    validate(config)
    if config["direction"] != (sign is not None):
        raise ValueError("`direction` needs the half's sign, and a sign needs `direction`")
    mirrors = mirrors_of(config)
    if mirrors == ("identity",):
        if sign is None:
            return dt.predict(checkpoint, frames, values, columns, player_tokens, length, sign=1, negate=False)
        return dt.predict(checkpoint, frames, values, columns, player_tokens, length, sign=sign, negate=True)
    if sign is None and mirrors == p3.MIRRORS:
        return p3.predict(checkpoint, frames, values, columns, player_tokens, length, mirrored=True, sign_map=sign_map)
    weights, expected, location, scale, permutation = checkpoint
    if columns != expected:
        raise ValueError("Player-set inference feature columns differ")
    negate = sign is not None
    token_values, mask = ps.token_inputs(np.asarray(player_tokens) if sign is None else dt.canonical(np.asarray(player_tokens), sign))
    variants, token_variants, mask_variants = [], [], []
    for mirror in mirrors:
        signed = values * p3.column_signs(sign_map, mirror, columns)
        signed_tokens = token_values * p3.TOKEN_SIGNS[mirror]
        variants += [normalize(signed, location, scale), normalize(signed[:, permutation], location, scale)]
        token_variants += [signed_tokens, dt.swapped(signed_tokens, negate)]
        mask_variants += [mask, mask[:, ::-1]]
    p = expit(ps.forward(np.stack(variants), np.stack(token_variants), np.stack(mask_variants), weights))
    count = len(mirrors)
    home = np.mean([(p[2 * m, :, 1] + 1 - p[2 * m + 1, :, 1]) / 2 for m in range(count)], axis=0)
    others = [np.mean([(p[2 * m, :, index] + p[2 * m + 1, :, index]) / 2 for m in range(count)], axis=0)
              for index in (0, 2, 3)]
    timeline = np.arange(length)
    return tuple(np.interp(timeline, frames, item).astype(np.float32) for item in (home, *others))


# -- command line ------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("train",), help="`train`: the name makes the run visible to training counters")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--boundary-boost", type=float, required=True)
    parser.add_argument("--config", required=True, help="JSON object of the switches")
    parser.add_argument("--signs", type=Path, help="sign file of the prepared records (`direction`)")
    parser.add_argument("--sign-map", type=Path, help="D47's parity record (`mirror`)")
    parser.add_argument("--pitches", type=Path, help="pitch sizes of the prepared records (`noise`)")
    parser.add_argument("--aux", type=Path, help="auxiliary-target folder (`aux`)")
    args = parser.parse_args()
    from dfl_submission.external_only import require_external_only
    from experiments.dfl_transfer_data import digest
    require_external_only()
    config = json.loads(args.config)
    validate(config)
    signs = read_signs(args.signs, args.source, args.tokens) if config["direction"] else None
    sign_map = None
    if config["mirror"] is not None:
        record = json.loads(args.sign_map.read_text())
        if not record["passed"]:
            raise ValueError("The mirror parity check did not pass; nothing is trained with mirrors")
        sign_map = record["sign_map"]
    provenance = {"signs_file": None if args.signs is None else digest(args.signs)}
    train(args.source, args.tokens, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost,
          config=config, signs=signs, sign_map=sign_map, pitch_file=args.pitches, aux_folder=args.aux,
          provenance=provenance)


if __name__ == "__main__":
    main()
