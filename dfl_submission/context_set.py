"""D52: the player-set network with a configurable stack of temporal blocks, its trainer, export and NumPy inference.

Candidate X of D52 adds a seventh temporal block with dilation 64 (`SEVEN`), so the network sees 127
long-context steps (25.4 s) to each side instead of 63; its training windows are 128 steps plus 127 on
each side. Everything else is the stack's recipe (`stack_set`): the same switches, data, signs, mirror and
generators. `temporal_cpu`, `player_set` and `stack_set` stay byte-identical; with `SIX` blocks this
module's network, batch, trainer and inference are theirs exactly (tests check it).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import erf, expit

from dfl_submission import direction_tokens as dt, player_set as ps, player_set3 as p3, stack_set as ss
from dfl_submission.temporal_cpu import DILATIONS as SIX, norm, normalize


SEVEN = (*SIX, 64)


def halo(dilations):
    return sum(dilations)


def dilations_of(weights):
    """The dilations of a checkpoint, from how many temporal blocks its weights hold (1, 2, 4, ... doubling)."""
    blocks = len({key.split(".")[1] for key in weights if key.startswith("temporal.") and key.endswith(".mix.weight")})
    dilations = tuple(2 ** index for index in range(blocks))
    if dilations not in (SIX, SEVEN):
        raise ValueError(f"Unexpected temporal blocks: {blocks}")
    return dilations


# -- NumPy inference -------------------------------------------------------------------------------

def temporal_forward(values, weights, dilations, *, residual=None):
    """`temporal_cpu.forward` with the dilations as an argument (the same operations in the same order)."""
    hidden = values @ weights["project.weight"][:, :, 0].T + weights["project.bias"]
    if residual is not None:
        hidden += residual
    for index, dilation in enumerate(dilations):
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


def forward(values, token_values, mask, weights, dilations=None):
    """`player_set.forward` for any number of temporal blocks (taken from the weights unless given)."""
    dilations = dilations_of(weights) if dilations is None else tuple(dilations)
    pool = ps.pooled(token_values, mask, weights)
    residual = pool @ weights["pool_project.weight"].T + weights["pool_project.bias"]
    return temporal_forward(values, weights, dilations, residual=residual)


def predict(checkpoint, frames, values, columns, player_tokens, length, *, config=ss.DEFAULT, sign=None, sign_map=None):
    """`stack_set.predict` for a checkpoint with any number of temporal blocks.

    The variants are built exactly as `stack_set.predict` builds them in its general branch (one
    forward pass over every mirror times both team orders); with one mirror that branch's result is the
    swap average of `direction_tokens.predict` / `player_set.predict`.
    """
    ss.validate(config)
    if config["direction"] != (sign is not None):
        raise ValueError("`direction` needs the half's sign, and a sign needs `direction`")
    weights, expected, location, scale, permutation = checkpoint
    if columns != expected:
        raise ValueError("Player-set inference feature columns differ")
    mirrors = ss.mirrors_of(config)
    negate = sign is not None
    tokens = np.asarray(player_tokens)
    token_values, mask = ps.token_inputs(tokens if sign is None else dt.canonical(tokens, sign))
    variants, token_variants, mask_variants = [], [], []
    for mirror in mirrors:
        signed = values * p3.column_signs(sign_map, mirror, columns)
        signed_tokens = token_values * p3.TOKEN_SIGNS[mirror]
        variants += [normalize(signed, location, scale), normalize(signed[:, permutation], location, scale)]
        token_variants += [signed_tokens, dt.swapped(signed_tokens, negate)]
        mask_variants += [mask, mask[:, ::-1]]
    p = expit(forward(np.stack(variants), np.stack(token_variants), np.stack(mask_variants), weights))
    count = len(mirrors)
    home = np.mean([(p[2 * m, :, 1] + 1 - p[2 * m + 1, :, 1]) / 2 for m in range(count)], axis=0)
    others = [np.mean([(p[2 * m, :, index] + p[2 * m + 1, :, index]) / 2 for m in range(count)], axis=0)
              for index in (0, 2, 3)]
    timeline = np.arange(length)
    return tuple(np.interp(timeline, frames, item).astype(np.float32) for item in (home, *others))


# -- PyTorch ---------------------------------------------------------------------------------------

def torch_network(columns, width, dropout, dilations=SEVEN):
    """`player_set.torch_network` with one temporal block per dilation (modules created in the same order)."""
    import torch
    from torch import nn
    from dfl_submission import temporal_network as tn

    class ContextSetNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Conv1d(columns * 2, width, 1)
            self.encoder = nn.Sequential(nn.Linear(ps.TOKEN, ps.HIDDEN), nn.GELU(), nn.Linear(ps.HIDDEN, ps.HIDDEN), nn.GELU())
            self.pool_project = nn.Linear(4 * ps.HIDDEN, width)
            self.temporal = nn.Sequential(*(tn.TemporalBlock(d, width) for d in dilations))
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

    return ContextSetNetwork()


def batch(groups, rng, augment_rng, permutation, location, scale, size, config, *, column_signs=None, pitches=None,
          dilations=SEVEN):
    """`stack_set.batch` (without `aux`) with windows of `player_set.CORE` steps plus the network's halo on each side."""
    import torch
    if config["aux"]:
        raise ValueError("D52 does not train the auxiliary head")
    side = halo(dilations)
    negate = config["direction"]
    xs, ts, ms, ys = [], [], [], []
    for key in sorted(groups):
        sequences = groups[key]
        for _ in range(size // len(groups)):
            index = rng.integers(len(sequences))
            values, labels, tok, _ = sequences[index]
            start = rng.integers(0, len(values) - ps.CORE - 2 * side + 1)
            window = values[start:start + ps.CORE + 2 * side]
            token_window = tok[start:start + ps.CORE + 2 * side].astype(np.float32)
            output = labels[start + side:start + side + ps.CORE]
            if rng.integers(2):
                window = window[:, permutation]
                token_window = dt.swapped(token_window, negate)
                output = output.copy()
                output[:, 1] = 1 - output[:, 1]
            if config["mirror"] is not None:
                flip_x, flip_y = augment_rng.integers(2), augment_rng.integers(2)
                if config["mirror"] == "y":
                    flip_x = 0
                mirror = {(0, 0): "identity", (1, 0): "x", (0, 1): "y", (1, 1): "xy"}[int(flip_x), int(flip_y)]
                window, token_window = p3.mirror_window(window, token_window, mirror, column_signs[mirror])
            if config["noise"]:
                token_window = p3.noisy_tokens(token_window, augment_rng, pitches[key][index])
            token_values, mask = ps.token_inputs(token_window)
            xs.append(normalize(window, location, scale))
            ts.append(token_values)
            ms.append(mask)
            ys.append(output)
    return tuple(torch.from_numpy(np.ascontiguousarray(np.stack(item))) for item in (xs, ts, ms, ys))


def train(source, token_folder, output, device, *, seed, boundary_boost, config=ss.DEFAULT, signs=None, sign_map=None,
          pitch_file=None, dilations=SEVEN, width=96, steps=5400, lr_scale=2., batch_size=32, weight_decay=.01, dropout=.2,
          provenance=None):
    """`stack_set.train` (without `aux`) for a network with one temporal block per dilation."""
    import torch
    from dfl_submission import hyper_network as hn, temporal_network as tn
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    ss.validate(config)
    if config["aux"]:
        raise ValueError("D52 does not train the auxiliary head")
    if config["direction"] != (signs is not None):
        raise ValueError("`direction` needs the records' signs, and signs need `direction`")
    if config["noise"] and pitch_file is None:
        raise ValueError("`noise` needs the records' pitch sizes")
    dilations = tuple(dilations)
    side = halo(dilations)
    output.mkdir(parents=True, exist_ok=False)
    run_seed = seed + config["seed_offset"]
    saved = tn.snapshots(steps)
    torch.set_num_threads(3)
    torch.manual_seed(run_seed)
    rng = np.random.default_rng(run_seed)
    augment_rng = np.random.default_rng([run_seed, p3.AUGMENT_STREAM])
    groups, manifest, permutation, location, scale = ss.training_data(source, token_folder, signs=signs)
    column_signs = ({mirror: p3.column_signs(sign_map, mirror, manifest["columns"]) for mirror in p3.MIRRORS}
                    if config["mirror"] is not None else None)
    pitches = ss.load_pitches(pitch_file, manifest) if config["noise"] else None
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = torch_network(len(location), width, dropout, dilations).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=weight_decay)
    hashes = {}
    for step in range(1, steps + 1):
        values, token_values, mask, labels = (item.to(device) for item in batch(
            groups, rng, augment_rng, permutation, location, scale, batch_size, config, column_signs=column_signs,
            pitches=pitches, dilations=dilations))
        optimizer.param_groups[0]["lr"] = hn.learning_rate(step, steps, lr_scale)
        optimizer.zero_grad(set_to_none=True)
        value = tn.loss(net(values, token_values, mask)[:, side:side + ps.CORE], labels, boundary_boost)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"Context-set training step {step} of {steps}, loss {value.item():.4f}", flush=True)
        if step in saved:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
    stack = dict(config, negative_records=sorted(k for k, v in (signs or {}).items() if v < 0),
                 records=len(manifest["records"]), mirror_sign_map=None if column_signs is None else
                 {m: s.tolist() for m, s in column_signs.items()},
                 pitch_file=None if pitch_file is None else digest(Path(pitch_file)), **(provenance or {}))
    write_json(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
               "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
               "token_manifest": digest(token_folder / "manifest.json"), "training_device": device,
               "torch": torch.__version__, "seed": run_seed, "boundary_boost": boundary_boost, "width": width,
               "steps": steps, "lr_scale": lr_scale, "batch": batch_size, "weight_decay": weight_decay,
               "dropout": dropout, "player_set": {"per_team": ps.PER_TEAM, "token": ps.TOKEN, "hidden": ps.HIDDEN},
               "stack": stack, "dilations": list(dilations)})


def export(folder, output):
    """`player_set.export` for any number of temporal blocks: every snapshot, checked within D15's tolerance."""
    import torch
    from shutil import copyfile
    from dfl_submission.train import write_json
    from experiments.dfl_transfer_data import digest
    output.mkdir(parents=True, exist_ok=False)
    fit = json.loads((folder / "fit.json").read_text())
    dilations = tuple(fit.get("dilations", SIX))
    norm_path = folder / "normalization.npz"
    if digest(norm_path) != fit["normalization"]:
        raise ValueError("Context-set normalization identity differs")
    copyfile(norm_path, output / norm_path.name)
    torch.set_num_threads(1)
    columns = len(fit["columns"])
    net = torch_network(columns, fit["width"], fit["dropout"], dilations).eval()
    rng = np.random.default_rng(13)
    length = 2 * halo(dilations) + 60
    values = rng.normal(size=(2, length, columns * 2)).astype(np.float32)
    token_values = rng.normal(size=(2, length, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float32)
    mask = (rng.uniform(size=(2, length, 2, ps.PER_TEAM)) > .3).astype(np.float32)
    mask[0, :5, 1] = 0
    hashes, differences = {}, {}
    for filename, expected in fit["models"].items():
        path = folder / filename
        if digest(path) != expected:
            raise ValueError(f"Context-set checkpoint identity differs: {filename}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        weights = {name: tensor.numpy() for name, tensor in state.items()}
        with torch.inference_mode():
            reference = net(torch.from_numpy(values), torch.from_numpy(token_values), torch.from_numpy(mask)).numpy()
        result = forward(values, token_values, mask, weights, dilations)
        np.testing.assert_allclose(result, reference, atol=2e-5, rtol=1e-5)
        destination = output / Path(filename).with_suffix(".npz").name
        np.savez(destination, **weights)
        hashes[destination.name] = digest(destination)
        differences[destination.name] = float(np.max(np.abs(result - reference)))
    write_json(output / "fit.json", dict(fit, models=hashes, original_checkpoints=fit["models"],
                                        export_max_logit_difference=differences))
    print(f"CPU export verified: {differences}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("export",))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.source, args.out)


if __name__ == "__main__":
    main()
