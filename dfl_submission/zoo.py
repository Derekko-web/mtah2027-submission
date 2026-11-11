"""D20: model search beyond D15 (DECISIONS.md D20).

Four families replace one part of D15's structure each: the temporal network at larger widths and
lengths (`tcn`), a bidirectional GRU (`gru`), a transformer encoder (`transformer`) and tuned XGBoost
heads (`trees`). Each family is tuned by match-grouped cross-validation on the external training
matches only (`cv`), on frame-level log-loss; only its selected setting is trained in full
(`train`) and scored once on development (`member`). `develop` picks an equal-weight ensemble by
greedy forward selection on the development primary and re-selects the ownership cost; `confirm`
scores it once on the reused DFL/SkillCorner holdout and PFF confirmation set C; `build` writes the
competition file. Network inference in the build is NumPy only.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.special import erf, expit

from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from dfl_submission.temporal_cpu import DILATIONS, normalize


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "zoo_recipe"
MODELS = PACKAGE / "zoo_models"
PROTOCOL = "DECISIONS.md D20"
FOLDS = 4
HALO = sum(DILATIONS)            # 63 steps of context on each side, as the temporal network sees
CORE = 128
WINDOW = CORE + 2 * HALO         # 254
CV_NETWORK = (42, .5)            # the representative network of each fold: seed and boundary weight
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
SETS = ("current", "augmented")
NETWORK_FAMILIES = ("tcn", "gru", "transformer")
FAMILIES = (*NETWORK_FAMILIES, "trees")
POOL = ("d15", *FAMILIES[:3], "trees")
GRIDS = {
    "tcn": tuple({"width": w, "steps": s} for w in (96, 192) for s in (5400, 10800, 19800)),
    "gru": tuple({"width": h, "steps": s} for h in (64, 128) for s in (5400, 10800)),
    "transformer": tuple({"width": d, "steps": s} for d in (64, 128) for s in (5400, 10800)),
    "trees": tuple({"max_depth": d, "n_estimators": n} for d in (3, 5, 7) for n in (420, 840, 1260)),
}
ANCHORS = {"tcn": {"width": 96, "steps": 5400}, "trees": {"max_depth": 5, "n_estimators": 420}}
TREE_LEARNING_RATE = .045
DROPOUT = .08
TRANSFORMER_LAYERS, TRANSFORMER_HEADS = 3, 4
WARMUP_SHARE = 18                # the transformer's learning rate rises over the first 1/18 of the steps
EPS = 1e-6
CPU_THREADS = 6               # GRU training runs on the CPU (D20 amendment)
DEVICES = {"tcn": "mps", "gru": "cpu", "transformer": "mps"}
DEAD, HOME, DEAD_EDGE, OWNER_EDGE, KNOWN, LIVE = range(6)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D20 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, record)


def name(setting):
    return "_".join(f"{key}{value}" for key, value in sorted(setting.items()))


# -- folds -----------------------------------------------------------------------------------------

def fold_of(manifest):
    """{(source, match): fold}. Fold k holds DFL training match k and SkillCorner matches k, k + 4, k + 8."""
    matches = sorted({(record["source"], record["match"]) for record in manifest["records"]})
    dfl = sorted(match for source, match in matches if source == "dfl")
    skill = sorted((match for source, match in matches if source == "skillcorner"), key=int)
    if len(dfl) != FOLDS or len(skill) != 3 * FOLDS or len(matches) != len(dfl) + len(skill):
        raise ValueError(f"Expected 4 DFL and 12 SkillCorner training matches, got {len(dfl)} and {len(skill)}")
    result = {("dfl", match): k for k, match in enumerate(dfl)}
    result |= {("skillcorner", match): k % FOLDS for k, match in enumerate(skill)}
    return result


def fold_folder(source, fold, output):
    """A prepared-sequence folder of every training half outside `fold` (hard links, same digests)."""
    manifest = json.loads((source / "manifest.json").read_text())
    folds = fold_of(manifest)
    if (output / "manifest.json").exists():
        stored = json.loads((output / "manifest.json").read_text())
        if stored["base_manifest"] != digest(source / "manifest.json") or stored["fold"] != fold:
            raise ValueError(f"{output} holds a different fold")
        return output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    records = [record for record in manifest["records"] if folds[record["source"], record["match"]] != fold]
    for record in records:
        target = output / record["file"]
        try:
            os.link(source / record["file"], target)
        except OSError:
            shutil.copyfile(source / record["file"], target)
        if digest(target) != record["sha256"]:
            raise ValueError(f"Prepared sequence hash differs: {record['file']}")
    write_json(output / "manifest.json", dict(manifest, records=records, fold=fold,
                                              base_manifest=digest(source / "manifest.json")))
    return output


def held_out(source, fold):
    """(record, values, targets) of the training halves in `fold`, hash-checked."""
    manifest = json.loads((source / "manifest.json").read_text())
    folds = fold_of(manifest)
    for record in manifest["records"]:
        if folds[record["source"], record["match"]] != fold:
            continue
        path = source / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError(f"Prepared sequence hash differs: {path}")
        with np.load(path, allow_pickle=False) as item:
            yield record, item["values"], item["targets"]


# -- network families in NumPy (the build's inference path) ----------------------------------------

def layer_norm(values, scale, bias):
    centered = values - values.mean(axis=-1, keepdims=True)
    variance = np.mean(centered ** 2, axis=-1, keepdims=True)
    return centered / np.sqrt(variance + np.float32(1e-5)) * scale + bias


def gelu(values):
    return values * np.float32(.5) * (1 + erf(values * np.float32(2 ** -.5)))


def sinusoid(length, width):
    position = np.arange(length, dtype=np.float64)[:, None]
    rate = np.exp(-math.log(10000.0) * np.arange(0, width, 2, dtype=np.float64) / width)
    table = np.zeros((length, width), dtype=np.float64)
    table[:, 0::2] = np.sin(position * rate)
    table[:, 1::2] = np.cos(position * rate)
    return table.astype(np.float32)


def windows(values):
    """(blocks, WINDOW, C): each block of CORE steps with HALO steps of context, zero-padded at the ends."""
    length = len(values)
    blocks = max(1, -(-length // CORE))
    padded = np.zeros((blocks * CORE + 2 * HALO, values.shape[-1]), dtype=values.dtype)
    padded[HALO:HALO + length] = values
    return np.stack([padded[k * CORE:k * CORE + WINDOW] for k in range(blocks)])


def unwindow(outputs, length):
    return outputs[:, HALO:HALO + CORE].reshape(-1, outputs.shape[-1])[:length]


def gru_direction(inputs, weights, prefix):
    w_ih, w_hh = weights[f"gru.weight_ih_{prefix}"], weights[f"gru.weight_hh_{prefix}"]
    b_ih, b_hh = weights[f"gru.bias_ih_{prefix}"], weights[f"gru.bias_hh_{prefix}"]
    units = w_hh.shape[1]
    projected = inputs @ w_ih.T + b_ih
    hidden = np.zeros((inputs.shape[0], units), dtype=np.float32)
    steps = range(inputs.shape[1] - 1, -1, -1) if prefix.endswith("_reverse") else range(inputs.shape[1])
    result = np.empty((inputs.shape[0], inputs.shape[1], units), dtype=np.float32)
    for t in steps:
        gates = hidden @ w_hh.T + b_hh
        x = projected[:, t]
        reset = expit(x[:, :units] + gates[:, :units])
        update = expit(x[:, units:2 * units] + gates[:, units:2 * units])
        candidate = np.tanh(x[:, 2 * units:] + reset * gates[:, 2 * units:])
        hidden = (1 - update) * candidate + update * hidden
        result[:, t] = hidden
    return result


def gru_forward(values, weights):
    """Logits of the bidirectional GRU for (batch, steps, 2C) inputs, as `GRUNetwork` computes them."""
    hidden = values @ weights["project.weight"].T + weights["project.bias"]
    layer = 0
    while f"gru.weight_ih_l{layer}" in weights:
        hidden = np.concatenate([gru_direction(hidden, weights, f"l{layer}"),
                                 gru_direction(hidden, weights, f"l{layer}_reverse")], axis=-1)
        layer += 1
    hidden = layer_norm(hidden, weights["norm.weight"], weights["norm.bias"])
    return hidden @ weights["outputs.weight"].T + weights["outputs.bias"]


def softmax(values):
    shifted = np.exp(values - values.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def transformer_forward(values, weights):
    """Logits of the transformer encoder for (batch, steps, 2C) inputs, as `TransformerNetwork` computes them."""
    width = weights["project.weight"].shape[0]
    heads = TRANSFORMER_HEADS
    size = width // heads
    hidden = values @ weights["project.weight"].T + weights["project.bias"] + sinusoid(values.shape[1], width)
    layer = 0
    while f"encoder.layers.{layer}.norm1.weight" in weights:
        prefix = f"encoder.layers.{layer}."
        normed = layer_norm(hidden, weights[prefix + "norm1.weight"], weights[prefix + "norm1.bias"])
        q, k, v = np.split(normed @ weights[prefix + "self_attn.in_proj_weight"].T
                           + weights[prefix + "self_attn.in_proj_bias"], 3, axis=-1)
        shape = (*normed.shape[:2], heads, size)
        q, k, v = (part.reshape(shape).transpose(0, 2, 1, 3) for part in (q, k, v))
        attention = softmax(q @ k.transpose(0, 1, 3, 2) / np.float32(math.sqrt(size)))
        mixed = (attention @ v).transpose(0, 2, 1, 3).reshape(*normed.shape[:2], width)
        hidden = hidden + mixed @ weights[prefix + "self_attn.out_proj.weight"].T + weights[prefix + "self_attn.out_proj.bias"]
        normed = layer_norm(hidden, weights[prefix + "norm2.weight"], weights[prefix + "norm2.bias"])
        inner = gelu(normed @ weights[prefix + "linear1.weight"].T + weights[prefix + "linear1.bias"])
        hidden = hidden + inner @ weights[prefix + "linear2.weight"].T + weights[prefix + "linear2.bias"]
        layer += 1
    hidden = layer_norm(hidden, weights["norm.weight"], weights["norm.bias"])
    return hidden @ weights["outputs.weight"].T + weights["outputs.bias"]


FORWARD = {"gru": gru_forward, "transformer": transformer_forward}


def windowed_logits(family, normalized, weights):
    """Logits for a stack of whole-half input variants (V, T, 2C), predicted block by block."""
    length = normalized.shape[1]
    blocks = [windows(variant) for variant in normalized]
    outputs = FORWARD[family](np.concatenate(blocks), weights)
    count = len(blocks[0])
    return np.stack([unwindow(outputs[i * count:(i + 1) * count], length) for i in range(len(blocks))])


class Checkpoint:
    """An exported GRU or transformer network with the interface of a `temporal_cpu` checkpoint."""

    def __init__(self, family, weights, columns, location, scale, permutation):
        self.family, self.weights, self.columns = family, weights, columns
        self.location, self.scale, self.permutation = location, scale, permutation

    def __getitem__(self, index):
        return (self.weights, self.columns, self.location, self.scale, self.permutation)[index]

    def predict(self, features):
        frames, values, columns, length = features
        if columns != self.columns:
            raise ValueError("Network inference feature columns differ")
        variants = np.stack([normalize(values, self.location, self.scale),
                             normalize(values[:, self.permutation], self.location, self.scale)])
        p = expit(windowed_logits(self.family, variants, self.weights))
        home = (p[0, :, 1] + 1 - p[1, :, 1]) / 2
        dead = (p[0, :, 0] + p[1, :, 0]) / 2
        dead_edge, owner_edge = ((p[0, :, index] + p[1, :, index]) / 2 for index in (2, 3))
        timeline = np.arange(length)
        return tuple(np.interp(timeline, frames, values).astype(np.float32)
                     for values in (home, dead, dead_edge, owner_edge))


def load_checkpoint(folder, filename):
    """A verified exported network: a `temporal_cpu` checkpoint tuple, or a `Checkpoint` for the new families."""
    from dfl_submission import temporal_cpu
    fit = json.loads((folder / "fit.json").read_text())
    family = fit.get("family", "tcn")
    if family == "tcn":
        return temporal_cpu.load(folder, filename)
    weights, columns, location, scale, permutation = temporal_cpu.load(folder, filename)
    return Checkpoint(family, weights, columns, location, scale, permutation)


def predict(checkpoint, features):
    from dfl_submission import temporal_cpu
    return checkpoint.predict(features) if isinstance(checkpoint, Checkpoint) else temporal_cpu.predict(checkpoint, features)


# -- network families in PyTorch (training only) ---------------------------------------------------

def torch_modules():
    """The PyTorch modules, built on first use so that the build never imports torch."""
    import torch
    from torch import nn

    class GRUNetwork(nn.Module):
        def __init__(self, columns, width):
            super().__init__()
            self.project = nn.Linear(columns * 2, width)
            self.gru = nn.GRU(width, width, num_layers=2, batch_first=True, bidirectional=True, dropout=DROPOUT)
            self.norm = nn.LayerNorm(2 * width)
            self.outputs = nn.Linear(2 * width, 4)

        def forward(self, values):
            hidden, _ = self.gru(self.project(values))
            return self.outputs(self.norm(hidden))

    class TransformerNetwork(nn.Module):
        def __init__(self, columns, width):
            super().__init__()
            self.project = nn.Linear(columns * 2, width)
            self.register_buffer("position", torch.from_numpy(sinusoid(WINDOW, width)), persistent=False)
            layer = nn.TransformerEncoderLayer(width, TRANSFORMER_HEADS, 2 * width, dropout=DROPOUT, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, TRANSFORMER_LAYERS, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.outputs = nn.Linear(width, 4)

        def forward(self, values):
            hidden = self.project(values) + self.position[:values.shape[1]]
            return self.outputs(self.norm(self.encoder(hidden)))

    return {"gru": GRUNetwork, "transformer": TransformerNetwork}


def learning_rate(family, step, steps):
    from dfl_submission import temporal_network
    rate = temporal_network.learning_rate(step, steps)
    if family == "transformer":
        warmup = max(1, steps // WARMUP_SHARE)
        rate *= min(1.0, step / warmup)
    return rate


def train_network(family, source, output, device, *, seed, boundary_boost, width, steps):
    """`temporal_network.train` with a GRU or transformer in place of the temporal network."""
    import torch
    from dfl_submission import temporal_network
    from dfl_submission.train import write_json as write_fit
    output.mkdir(parents=True, exist_ok=False)
    saved = temporal_network.snapshots(steps)
    torch.set_num_threads(CPU_THREADS if device == "cpu" else 3)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    groups, manifest, permutation, location, scale = temporal_network.training_data(source)
    np.savez(output / "normalization.npz", location=location, scale=scale, permutation=permutation)
    net = torch_modules()[family](len(location), width).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.01)
    hashes = {}
    for step in range(1, steps + 1):
        inputs, labels = temporal_network.batch(groups, rng, permutation, location, scale)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.param_groups[0]["lr"] = learning_rate(family, step, steps)
        optimizer.zero_grad(set_to_none=True)
        value = temporal_network.loss(net(inputs)[:, HALO:HALO + CORE], labels, boundary_boost)
        value.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"{family} training step {step} of {steps}, loss {value.item():.4f}", flush=True)
        if step in saved:
            filename = f"temporal_{step}.pt"
            torch.save({key: tensor.detach().cpu() for key, tensor in net.state_dict().items()}, output / filename)
            hashes[filename] = digest(output / filename)
    write_fit(output / "fit.json", {"models": hashes, "normalization": digest(output / "normalization.npz"),
              "columns": manifest["columns"], "train_manifest": digest(source / "manifest.json"),
              "training_device": device, "torch": torch.__version__, "seed": seed, "boundary_boost": boundary_boost,
              "family": family, "width": width, "steps": steps, "protocol": PROTOCOL,
              "threads": torch.get_num_threads()})


def export_network(folder, output):
    """NumPy export of a trained GRU or transformer, checked against PyTorch (D15's export tolerance)."""
    import torch
    from dfl_submission.train import write_json as write_fit
    output.mkdir(parents=True, exist_ok=False)
    fit = json.loads((folder / "fit.json").read_text())
    norm_path = folder / "normalization.npz"
    if digest(norm_path) != fit["normalization"]:
        raise ValueError("Network normalization identity differs")
    shutil.copyfile(norm_path, output / norm_path.name)
    torch.set_num_threads(1)
    columns = len(fit["columns"])
    net = torch_modules()[fit["family"]](columns, fit["width"]).eval()
    rng = np.random.default_rng(13)
    values = rng.normal(size=(2, 400, columns * 2)).astype(np.float32)
    hashes, differences = {}, {}
    for filename, expected in fit["models"].items():
        path = folder / filename
        if digest(path) != expected:
            raise ValueError(f"Network checkpoint identity differs: {filename}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        weights = {key: value.numpy() for key, value in state.items()}
        with torch.inference_mode():
            reference = np.stack([unwindow(net(torch.from_numpy(windows(v))).numpy(), len(v)) for v in values])
        result = windowed_logits(fit["family"], values, weights)
        np.testing.assert_allclose(result, reference, atol=2e-5, rtol=1e-5)
        destination = output / Path(filename).with_suffix(".npz").name
        np.savez(destination, **weights)
        hashes[destination.name] = digest(destination)
        differences[destination.name] = float(np.max(np.abs(result - reference)))
    write_fit(output / "fit.json", dict(fit, models=hashes, original_checkpoints=fit["models"],
                                       export_max_logit_difference=differences))
    print(f"CPU export verified: {differences}", flush=True)


def fit_network(family, source, output, *, seed, boundary_boost, setting, device="mps"):
    """Train one network of `family` at `setting` in a subprocess (torch never loads in the caller)."""
    module = "dfl_submission.temporal_network" if family == "tcn" else "dfl_submission.zoo"
    command = [sys.executable, "-m", module, "train", "--source", str(source), "--out", str(output),
               "--device", device, "--seed", str(seed), "--boundary-boost", str(boundary_boost),
               "--width", str(setting["width"]), "--steps", str(setting["steps"])]
    if family != "tcn":
        command[4:4] = ["--family", family]
    subprocess.run(command, check=True, cwd=ROOT)


def export(family, raw, output):
    module = "dfl_submission.temporal_network" if family == "tcn" else "dfl_submission.zoo"
    subprocess.run([sys.executable, "-m", module, "export", "--source", str(raw), "--out", str(output)],
                   check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- cross-validation of the network families ------------------------------------------------------

def final_weights(folder):
    """The last snapshot of a trained (not exported) run, as NumPy arrays, with its normalisation."""
    import torch
    fit = json.loads((folder / "fit.json").read_text())
    filename = f"temporal_{fit.get('steps', 1800)}.pt"
    if digest(folder / filename) != fit["models"][filename]:
        raise ValueError(f"Checkpoint identity differs: {folder / filename}")
    state = torch.load(folder / filename, map_location="cpu", weights_only=True)
    with np.load(folder / "normalization.npz", allow_pickle=False) as item:
        norm = tuple(item[key] for key in ("location", "scale", "permutation"))
    return fit.get("family", "tcn"), {key: value.numpy() for key, value in state.items()}, norm


def network_probabilities(family, weights, norm, values):
    """Swap-averaged probabilities at the prepared frames: home, dead, dead edge, owner edge."""
    from dfl_submission import temporal_cpu
    location, scale, permutation = norm
    variants = np.stack([normalize(values, location, scale), normalize(values[:, permutation], location, scale)])
    logits = temporal_cpu.forward(variants, weights) if family == "tcn" else windowed_logits(family, variants, weights)
    p = expit(logits)
    return ((p[0, :, 1] + 1 - p[1, :, 1]) / 2, (p[0, :, 0] + p[1, :, 0]) / 2,
            (p[0, :, 2] + p[1, :, 2]) / 2, (p[0, :, 3] + p[1, :, 3]) / 2)


def log_loss(p, y, mask=None, weight=None):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    values = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if mask is not None:
        values, weight = values[mask], None if weight is None else weight[mask]
    if weight is None:
        return float(values.mean()) if len(values) else float("nan")
    return float(np.sum(values * weight) / np.sum(weight))


def half_losses(probabilities, targets):
    home, dead, dead_edge, owner_edge = probabilities
    live, known = targets[:, LIVE].astype(bool), targets[:, KNOWN].astype(bool)
    return {"ownership": log_loss(home, targets[:, HOME], live), "stoppage": log_loss(dead, targets[:, DEAD], known),
            "dead_edge": log_loss(dead_edge, targets[:, DEAD_EDGE], known),
            "owner_edge": log_loss(owner_edge, targets[:, OWNER_EDGE], live)}


def summarise_matches(per_match, keys=("ownership", "stoppage")):
    """Match scores averaged within each source, sources weighed equally; the CV score is their sum."""
    sources = defaultdict(list)
    for (source, _), losses in per_match.items():
        sources[source].append(losses)
    summary = {}
    for key in {key for losses in per_match.values() for key in losses}:
        summary[key] = float(np.mean([np.mean([losses[key] for losses in values]) for values in sources.values()]))
    summary["score"] = float(sum(summary[key] for key in keys))
    return summary


def network_cv_losses(family, run, source, fold):
    family_of_run, weights, norm = final_weights(run)
    if family_of_run != family:
        raise ValueError(f"{run} holds a {family_of_run} run, not {family}")
    halves = defaultdict(list)
    for record, values, targets in held_out(source, fold):
        halves[record["source"], record["match"]].append(half_losses(network_probabilities(family, weights, norm, values), targets))
    return {key: {loss: float(np.mean([half[loss] for half in values])) for loss in values[0]} for key, values in halves.items()}


def run_complete(folder):
    return (folder / "fit.json").exists()


def network_cv(family, source, scratch, *, device=None, recipe=RECIPE):
    """Every setting of a network family, one representative network per fold; written once when complete."""
    require_external_only()
    output = recipe / f"cv_{family}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D20 cross-validation is recorded once")
    check_original_source(source)
    device = DEVICES[family] if device is None else device
    seed, boost = CV_NETWORK
    entries = []
    for setting in GRIDS[family]:
        per_match, runs = {}, {}
        for fold in range(FOLDS):
            folder = fold_folder(source, fold, scratch / "folds" / str(fold))
            run = scratch / family / name(setting) / f"fold{fold}"
            if not run_complete(run):
                if run.exists():
                    shutil.rmtree(run)
                fit_network(family, folder, run, seed=seed, boundary_boost=boost, setting=setting, device=device)
            losses = network_cv_losses(family, run, source, fold)
            per_match |= {f"{key[0]}:{key[1]}": value for key, value in losses.items()}
            runs[str(fold)] = {"fit_sha256": digest(run / "fit.json"), "train_manifest": digest(folder / "manifest.json")}
        summary = summarise_matches({tuple(key.split(":", 1)): value for key, value in per_match.items()})
        entries.append({"setting": setting, "summary": summary, "matches": per_match, "runs": runs})
        print(f"D20 CV {family} {name(setting)}: score {summary['score']:.5f} (ownership {summary['ownership']:.5f}, "
              f"stoppage {summary['stoppage']:.5f})", flush=True)
    selected = select_cv(entries)
    write_new(output, {"protocol": PROTOCOL, "family": family, "grid": list(GRIDS[family]), "entries": entries,
                       "selected": selected, "anchor": ANCHORS.get(family), "cv_network": {"seed": seed, "boundary_boost": boost},
                       "source_manifest": digest(source / "manifest.json"), "folds": fold_table(source)})
    print(f"D20 CV {family} selected {name(selected)}", flush=True)
    return selected


def select_cv(entries, key="score"):
    """The lowest cross-validated loss, ties to grid order."""
    return min(enumerate(entries), key=lambda item: (item[1]["summary"][key], item[0]))[1]["setting"]


def fold_table(source):
    folds = fold_of(json.loads((source / "manifest.json").read_text()))
    return {str(k): sorted(f"{s}:{m}" for (s, m), f in folds.items() if f == k) for k in range(FOLDS)}


def check_original_source(source):
    expected = json.loads((PACKAGE / "temporal_models" / "fit.json").read_text())["train_manifest"]
    if digest(source / "manifest.json") != expected:
        raise ValueError(f"{source} is not the original set's prepared-sequence folder")


# -- the tree family -------------------------------------------------------------------------------

TREE_HEADS = {"ownership": "ownership.ubj", "stoppage": "dead.ubj"}
CV_TREE_JOBS = 6                 # threads for cross-validation fits only; full refits keep D3's n_jobs


def tree_params(setting, **extra):
    from dfl_submission import long_context
    return dict(long_context.TREE_PARAMS, max_depth=setting["max_depth"], n_estimators=setting["n_estimators"],
                learning_rate=TREE_LEARNING_RATE) | extra


def fit_head(rows, *, ownership, params):
    """`long_context.fit_head` with explicit XGBoost parameters; identical to it at D3's parameters."""
    import pandas as pd
    from xgboost import XGBClassifier
    from dfl_submission import long_context
    from dfl_transfer_features import swap
    features, targets, match_keys = zip(*rows)
    inputs = pd.concat(features, ignore_index=True)
    y = np.concatenate(targets).astype(int)
    weight = long_context.source_weights(y, np.concatenate(match_keys))
    if ownership:
        inputs = pd.concat([inputs, swap(inputs)], ignore_index=True)
        y, weight = np.r_[y, 1 - y], np.tile(weight, 2)
    model = XGBClassifier(**params)
    model.fit(inputs, y, sample_weight=weight)
    return model, len(y)


def tree_rows(folder):
    from dfl_submission import aligned_trial, touch_trial
    return {"ownership": aligned_trial.ownership_rows(folder), "stoppage": touch_trial.stoppage_rows(folder)}


def head_probability(model, inputs, ownership, trees):
    """The head's probability from its first `trees` trees; the ownership head swap-averaged as at inference."""
    from dfl_transfer_features import swap
    p = model.predict_proba(inputs, iteration_range=(0, trees))[:, 1]
    if ownership:
        p = (p + 1 - model.predict_proba(swap(inputs), iteration_range=(0, trees))[:, 1]) / 2
    return p


def tree_fold_losses(head_rows, ownership, fold_by_key, fold, scratch, depths, counts, tag):
    """Held-out weighted log-loss of each depth and tree count for one head and fold."""
    import pandas as pd
    from xgboost import XGBClassifier
    from dfl_submission import long_context
    train = [row for row in head_rows if fold_by_key[row[2][0]] != fold]
    test = [row for row in head_rows if fold_by_key[row[2][0]] == fold]
    inputs = pd.concat([row[0] for row in test], ignore_index=True)
    y = np.concatenate([row[1] for row in test]).astype(int)
    weight = long_context.source_weights(y, np.concatenate([row[2] for row in test]))
    losses = {}
    for depth in depths:
        path = scratch / f"{tag}_depth{depth}_fold{fold}.ubj"
        model = XGBClassifier()
        if path.exists():
            model.load_model(path)
        else:
            model, _ = fit_head(train, ownership=ownership,
                                params=tree_params({"max_depth": depth, "n_estimators": max(counts)}, n_jobs=CV_TREE_JOBS))
            scratch.mkdir(parents=True, exist_ok=True)
            model.save_model(path)
        for count in counts:
            losses[depth, count] = log_loss(head_probability(model, inputs, ownership, count), y, weight=weight)
        print(f"D20 CV trees {tag} fold {fold} depth {depth}: "
              + ", ".join(f"{count} trees {losses[depth, count]:.5f}" for count in counts), flush=True)
    return losses


def tree_cv(source, scratch, *, recipe=RECIPE):
    """Each tree head's depth and tree count by grouped CV on the original set's prepared rows; written once."""
    require_external_only()
    output = recipe / "cv_trees.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D20 cross-validation is recorded once")
    check_original_source(source)
    folds = fold_of(json.loads((source / "manifest.json").read_text()))
    fold_by_key = {f"{s}:{m}": k for (s, m), k in folds.items()}
    depths = sorted({setting["max_depth"] for setting in GRIDS["trees"]})
    counts = sorted({setting["n_estimators"] for setting in GRIDS["trees"]})
    record = {"protocol": PROTOCOL, "family": "trees", "grid": list(GRIDS["trees"]), "anchor": ANCHORS["trees"],
              "learning_rate": TREE_LEARNING_RATE, "cv_n_jobs": CV_TREE_JOBS, "source_manifest": digest(source / "manifest.json"),
              "folds": fold_table(source), "heads": {}, "selected": {}}
    for head, head_rows in tree_rows(source).items():
        per_fold = [tree_fold_losses(head_rows, head == "ownership", fold_by_key, fold, scratch / "trees", depths, counts, head)
                    for fold in range(FOLDS)]
        entries = []
        for setting in GRIDS["trees"]:
            values = [losses[setting["max_depth"], setting["n_estimators"]] for losses in per_fold]
            entries.append({"setting": setting, "summary": {"score": float(np.mean(values))}, "folds": values})
            print(f"D20 CV trees {head} {name(setting)}: {np.mean(values):.5f}", flush=True)
        record["heads"][head] = entries
        record["selected"][head] = select_cv(entries)
    write_new(output, record)
    print(f"D20 CV trees selected {record['selected']}", flush=True)
    return record["selected"]


def tree_selection(recipe=RECIPE):
    selected = json.loads((recipe / "cv_trees.json").read_text())["selected"]
    return selected, all(selected[head] == ANCHORS["trees"] for head in TREE_HEADS)


def train_trees(sources, *, recipe=RECIPE, models=MODELS):
    """Both heads of both sets refit at their selected settings (D3's n_jobs), as `long_context.train` fits them."""
    require_external_only()
    from dfl_submission import scale_trial
    selected, anchor = tree_selection(recipe)
    if anchor:
        print("Both heads selected D3's setting; the tree family adds nothing", flush=True)
        return
    for model_set in SETS:
        source = sources[model_set]
        published = json.loads((scale_trial.TREES[model_set] / "fit.json").read_text())
        expected = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
        if digest(source / "manifest.json") != expected:
            raise ValueError(f"{source} is not the {model_set} set's prepared-sequence folder")
        output = models / "trees" / model_set
        if (output / "fit.json").exists():
            continue
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        hashes, counts, same = {}, {}, {}
        for head, rows in tree_rows(source).items():
            filename = TREE_HEADS[head]
            model, counts[filename] = fit_head(rows, ownership=head == "ownership", params=tree_params(selected[head]))
            model.save_model(output / filename)
            hashes[filename] = digest(output / filename)
            same[filename] = hashes[filename] == published["models"][filename]
            print(f"Refit {model_set} {filename} at {name(selected[head])}: {counts[filename]:,} rows", flush=True)
        write_json(output / "fit.json", dict(published, models=hashes, rows=counts, protocol=PROTOCOL,
                                             settings=selected, learning_rate=TREE_LEARNING_RATE,
                                             prepared_manifest_sha256=digest(source / "manifest.json"),
                                             equals_published=same))


# -- full training of a network family -------------------------------------------------------------

def train_family(family, sources, scratch, *, recipe=RECIPE, models=MODELS, device=None):
    """The four networks of a family's selected setting: both sets, D15's seeds and boundary weights."""
    require_external_only()
    from dfl_submission import scale_trial
    device = DEVICES[family] if device is None else device
    selected = json.loads((recipe / f"cv_{family}.json").read_text())["selected"]
    if selected == ANCHORS.get(family):
        print(f"{family} selected D15's own setting; the family adds nothing", flush=True)
        return
    for model_set in SETS:
        source = sources[model_set]
        expected = json.loads((scale_trial.PUBLISHED[model_set] / "temporal_models" / "fit.json").read_text())["train_manifest"]
        if digest(source / "manifest.json") != expected:
            raise ValueError(f"{source} is not the {model_set} set's prepared-sequence folder")
        for slot, (seed, boost) in NETWORKS.items():
            output = models / family / model_set / slot
            if (output / "fit.json").exists():
                continue
            raw = scratch / family / model_set / slot
            if raw.exists() and not run_complete(raw):
                shutil.rmtree(raw)
            if not run_complete(raw):
                fit_network(family, source, raw, seed=seed, boundary_boost=boost, setting=selected, device=device)
            if output.exists():
                shutil.rmtree(output)
            export(family, raw, output)


# -- members: D15's structure with one family's models ---------------------------------------------

def d15_setting():
    from dfl_submission import scale_trial
    frozen = json.loads((scale_trial.RECIPE / "recipe.json").read_text())
    return (frozen["setting"]["width"], frozen["setting"]["steps"]), frozen


def member_folders(member, models=MODELS):
    """Per model set, the member's tree folder and network folder."""
    from dfl_submission import scale_trial
    d15_networks = scale_trial.network_folders(d15_setting()[0])
    trees = dict(scale_trial.TREES)
    if member in NETWORK_FAMILIES:
        return trees, {key: models / member / key for key in SETS}
    if member == "trees":
        return {key: models / "trees" / key for key in SETS}, d15_networks
    if member == "d15":
        return trees, d15_networks
    raise ValueError(f"Unknown D20 member {member!r}")


def member_identity(member, models=MODELS):
    trees, networks = member_folders(member, models)
    return {"member": member, "trees": {key: digest(trees[key] / "fit.json") for key in SETS},
            "networks": {key: {slot: digest(networks[key] / slot / "fit.json") for slot in NETWORKS} for key in SETS}}


def package(trees, networks, now):
    """`scale_trial.package` for networks of any family."""
    from dfl_submission import long_context, scale_trial
    reference, checkpoints, variants, _, _ = now
    inference = scale_trial.INFERENCE_NAME
    if set(reference[1]) != {inference} or set(checkpoints) != {inference}:
        raise ValueError("D3 is expected to use only the last snapshot of each network")
    forest, _ = long_context.load(trees)
    original = {inference: load_checkpoint(networks / "temporal_models", scale_trial.final_snapshot(networks / "temporal_models"))}
    seeds = {inference: load_checkpoint(networks / "seed_models", scale_trial.final_snapshot(networks / "seed_models"))}
    return (forest, original, reference[2], reference[3]), seeds, variants


def teacher(member, models=MODELS):
    from dfl_submission import competition_adaptation, retrain_trial, scale_trial
    reference, _, frozen = competition_adaptation.reference()
    now = retrain_trial.current(ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe")
    trees, networks = member_folders(member, models)
    packages = {role: package(trees[key], networks[key], now) for role, key in scale_trial.ROLES.items()}
    return reference, packages, frozen


def forecast_features(reference, checkpoints, frames, owner, dead, length):
    """`temporal_trial.forecast_features` with network inference dispatched by family."""
    from dfl_submission import models, temporal_trial
    baseline = None
    if reference is not None:
        home_p = models.ownership_probability(reference[0], owner)
        dead_p = reference[1].predict_proba(dead)[:, 1]
        timeline = np.arange(length)
        baseline = tuple(np.interp(timeline, frames, p).astype(np.float32) for p in (home_p, dead_p))
    features = frames, owner.to_numpy(dtype=np.float32), owner.columns.tolist(), length
    neural = {key: predict(model, features) for key, model in checkpoints.items()}
    if all(key in neural for key in temporal_trial.VARIANTS["late_mean"]):
        first, second = (neural[key] for key in temporal_trial.VARIANTS["late_mean"])
        neural["late_mean"] = tuple((a + b) / 2 for a, b in zip(first, second))
    return baseline, neural


def forecasts(reference, checkpoints, variants, positions, ball, teams, pitch):
    """`ensemble_trial.forecasts` with network inference dispatched by family; identical for D15."""
    from dfl_submission import ensemble_trial, long_context
    trees, original, frozen, _ = reference
    recipes = frozen["recipe"]
    phases = ensemble_trial.PHASES if "phase_reference" in variants or "seed_phase" in variants else ensemble_trial.PHASES[:1]
    late = ensemble_trial.LATE
    phase_reference, phase_seed = [], []
    start_reference, seeds = None, {}
    for offset in phases:
        frames, owner, dead = long_context.features(positions[offset:], ball[offset:], teams, pitch)
        length = len(ball) - offset
        context = frames, owner.to_numpy(dtype=np.float32), owner.columns.tolist(), length
        if offset == 0 or "phase_reference" in variants:
            predicted = forecast_features(trees, original, frames, owner, dead, length)
            if offset == 0:
                start_reference = predicted
            if "phase_reference" in variants:
                phase_reference.append(tuple(np.pad(p, (offset, 0), mode="edge")
                                             for p in ensemble_trial.blend_reference(predicted, recipes)))
        if offset == 0:
            seeds = {key: predict(checkpoint, context) for key, checkpoint in checkpoints.items()}
        if "seed_phase" in variants:
            shifted = seeds[late] if offset == 0 else predict(checkpoints[late], context)
            phase_seed.append(tuple(np.pad(p, (offset, 0), mode="edge") for p in shifted))
    candidates = {}
    for key in variants:
        if key == "phase_reference":
            candidates[key] = ensemble_trial.mean(phase_reference)
        elif key == "seed_phase":
            candidates[key] = ensemble_trial.mean(phase_seed)
        else:
            parts = [seeds[part] for part in ensemble_trial.VARIANTS[key]]
            candidates[key] = parts[0] if len(parts) == 1 else ensemble_trial.mean(parts)
    return start_reference, candidates


def half_signals(model, positions, ball, teams, pitch):
    """`diagnosis.half_signals` with the family-dispatched forecasts."""
    from dfl_submission import diagnosis, motion_trial, retrain_trial
    reference, packages, frozen = model
    previous = reference[2]["recipe"]
    outputs = tuple(forecasts(*packages[key], positions, ball, teams, pitch) for key in ("current", "candidate"))
    mixed = retrain_trial.mix(*outputs, frozen["weight"])
    labels = motion_trial.decode(mixed, previous, frozen["recipe"])
    signals = tuple(motion_trial.blend_signals(mixed, previous, frozen["recipe"]))
    if not np.array_equal(diagnosis.decode(signals, frozen["recipe"]), labels):
        raise ValueError("Decoding the cached inputs does not reproduce the labels")
    return signals, labels


def member_halves(member, dfl, skill, parsed, games, split, *, raw, cache, models=MODELS):
    from dfl_submission import diagnosis, vendor_trial
    key = {"zoo": member_identity(member, models), "teacher_sha256": diagnosis.teacher_identity(), "split": split,
           "dfl_raw": raw, "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}
    model = teacher(member, models)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw, signals_of=half_signals)
        yield from diagnosis.pff_halves(parsed, games, model, signals_of=half_signals)

    return diagnosis.collect(cache, key, produce)


def metrica_signals(member, cache, models=MODELS):
    """The member's decoder inputs on the three Metrica sample games (D19), cached by identity."""
    from dfl_submission import diagnosis, metrica_sample as ms
    identity = {"zoo": member_identity(member, models), "teacher_sha256": diagnosis.teacher_identity(),
                "inputs": ms.inputs(ms.PARSED)}
    manifest = cache / "manifest.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["identity"] != identity:
            raise ValueError(f"{cache} holds Metrica inputs for a different member")
        result = {}
        for game in ms.GAMES:
            halves = []
            for index in (1, 2):
                path = cache / f"{game}_{index}.npz"
                if digest(path) != stored["files"][path.name]:
                    raise ValueError(f"Cached Metrica inputs differ from the manifest: {path.name}")
                with np.load(path, allow_pickle=False) as item:
                    halves.append(tuple(item[key] for key in ("home", "dead", "dead_edge", "owner_edge")))
            result[game] = tuple(halves)
        return result
    model = teacher(member, models)
    cache.mkdir(parents=True, exist_ok=False)
    result, files = {}, {}
    for game in ms.GAMES:
        halves = []
        for index, half in enumerate(ms.load(ms.PARSED, game), start=1):
            signals = half_signals(model, *ms.tracking(half))[0]
            path = cache / f"{game}_{index}.npz"
            np.savez_compressed(path, **dict(zip(("home", "dead", "dead_edge", "owner_edge"), signals)))
            files[path.name] = digest(path)
            halves.append(signals)
        result[game] = tuple(halves)
        print(f"Metrica game {game} forecast for {member}", flush=True)
    write_json(manifest, {"identity": identity, "files": files})
    return result


# -- ensembles -------------------------------------------------------------------------------------

def mean_signals(parts):
    """The equal-weight mean of members' four decoder inputs, summed in member order; one member unchanged."""
    if len(parts) == 1:
        return tuple(parts[0])
    total = [np.array(signal, dtype=np.float32, copy=True) for signal in parts[0]]
    for part in parts[1:]:
        for index, signal in enumerate(part):
            total[index] += signal
    count = np.float32(len(parts))
    return tuple((value / count).astype(np.float32) for value in total)


def ensemble(member_halves_list):
    """Halves whose decoder inputs are the mean of the members' inputs for the same half."""
    from dfl_submission import diagnosis
    result = []
    for parts in zip(*member_halves_list, strict=True):
        first = parts[0]
        if any(part.key != first.key or part.index != first.index or not np.array_equal(part.labels, first.labels)
               for part in parts):
            raise ValueError("Members' halves are not aligned")
        result.append(diagnosis.HalfSignals(first.source, first.match, first.view, first.index, first.labels,
                                            mean_signals([part.signals for part in parts]), first.d3))
    return result


def ensemble_metrica(members, inputs):
    from dfl_submission import metrica_sample as ms
    return {game: tuple(mean_signals([inputs[member][game][index] for member in members]) for index in (0, 1))
            for game in ms.GAMES}


def primary(trial, metric="spell_f1"):
    from dfl_submission import refine_trial
    return refine_trial.standard(trial, refine_trial.PRIMARY, metric)


def greedy(pool, evaluate, limit=4):
    """Greedy forward selection on the primary; ties to macroF1, then pool order. Returns members and steps."""
    def best(options):
        return max(options, key=lambda item: (primary(item[1]), primary(item[1], "macro_f1"), -pool.index(item[0][-1])))

    steps = []
    options = [((member,), evaluate((member,))) for member in pool]
    steps.append([{"members": list(members), "primary": primary(trial), "macro_f1": primary(trial, "macro_f1")}
                  for members, trial in options])
    chosen, trial = best(options)
    while len(chosen) < min(limit, len(pool)):
        options = [((*chosen, member), evaluate((*chosen, member))) for member in pool if member not in chosen]
        steps.append([{"members": list(members), "primary": primary(value), "macro_f1": primary(value, "macro_f1")}
                      for members, value in options])
        candidate, value = best(options)
        if primary(value) <= primary(trial):
            break
        chosen, trial = candidate, value
    return list(chosen), trial, steps


# -- development, confirmation and build -----------------------------------------------------------

def member_record(family, recipe=RECIPE):
    path = recipe / f"member_{family}.json"
    return json.loads(path.read_text()) if path.exists() else None


def anchored(family, recipe=RECIPE):
    if family == "trees":
        return tree_selection(recipe)[1]
    return json.loads((recipe / f"cv_{family}.json").read_text())["selected"] == ANCHORS.get(family)


def d15_development(dfl, skill, parsed, *, raw, cache):
    """D15's cached development inputs, checked against its development record (PFF development included)."""
    from dfl_submission import aligned_trial, scale_trial, vendor_trial
    setting, frozen = d15_setting()
    halves = scale_trial.setting_halves(dfl, skill, parsed, vendor_trial.development(), "development", setting,
                                        raw=raw, cache=cache)
    record = json.loads((scale_trial.RECIPE / "development.json").read_text())
    expected = next(entry["trial"] for entry in record["entries"] if entry["setting"] == record["selected"])
    result = aligned_trial.evaluate(halves, frozen["recipe"])
    if vendor_trial.canonical(result) != expected:
        raise ValueError("D15 does not reproduce its development record; nothing was written")
    return halves, frozen["recipe"], result


def same_halves(left, right):
    return len(left) == len(right) and all(
        a.key == b.key and a.index == b.index and np.array_equal(a.labels, b.labels)
        and all(np.array_equal(x, y) for x, y in zip(a.signals, b.signals)) for a, b in zip(left, right))


def score_member(family, dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe=RECIPE, models=MODELS):
    """A family's member at D15's recipe on development, once, after the D15 checks."""
    require_external_only()
    import time
    from dfl_submission import aligned_trial, refine_trial, vendor_trial
    output = recipe / f"member_{family}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; each D20 member is scored once")
    if anchored(family, recipe):
        write_new(output, {"protocol": PROTOCOL, "family": family, "anchor": True,
                           "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
        print(f"{family} selected its anchor; its member is D15 and is not scored", flush=True)
        return
    reference, d15_recipe, base = d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    games = vendor_trial.development()
    zoo_path = member_halves("d15", dfl, skill, parsed, games, "development", raw=raw, cache=cache_root / "d15", models=models)
    if not same_halves(zoo_path, reference):
        raise ValueError("D15 through the D20 forecasts differs from D15's cached inputs; nothing was written")
    halves = member_halves(family, dfl, skill, parsed, games, "development", raw=raw, cache=cache_root / family, models=models)
    trial = aligned_trial.evaluate(halves, d15_recipe)
    change = refine_trial.changes(base, trial)
    write_new(output, {"protocol": PROTOCOL, "family": family, "anchor": False,
                       "setting": json.loads((recipe / f"cv_{family}.json").read_text())["selected"],
                       "identity": member_identity(family, models), "reproduction": {"d15_development": True,
                       "d15_through_zoo_forecasts": True}, "reference": base, "trial": trial, "changes": change,
                       "teacher_sha256": diagnosis_identity(), "inputs": vendor_trial.inputs(parsed, games),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D20 member {family}: N = 3 standard {primary(trial):.5f} ({change['primary']:+.5f}), "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items()) + f", PFF {change['pff']:+.4f}", flush=True)


def diagnosis_identity():
    from dfl_submission import diagnosis
    return diagnosis.teacher_identity()


OWNER_COSTS = (6, 8, 10, 12)
GAIN = .01


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, metrica_root, recipe=RECIPE, models=MODELS):
    """Greedy ensemble on the development primary, then the ownership cost, with every gate; written once."""
    require_external_only()
    import time
    from dfl_submission import aligned_trial, metrica_sample as ms, refine_trial, vendor_trial
    output = recipe / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; the D20 ensemble is selected once")
    reference, d15_recipe, base = d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    games = vendor_trial.development()
    records = {family: member_record(family, recipe) for family in FAMILIES}
    pool = ["d15"] + [family for family in FAMILIES if records[family] is not None and not records[family]["anchor"]]
    halves = {"d15": reference}
    for family in pool[1:]:
        if records[family]["identity"] != member_identity(family, models):
            raise ValueError(f"The {family} models differ from the ones its member record scored")
        halves[family] = member_halves(family, dfl, skill, parsed, games, "development", raw=raw,
                                       cache=cache_root / family, models=models)
    evaluated = {}

    def evaluate(members, owner_cost=None):
        key = (members, owner_cost)
        if key not in evaluated:
            setting = d15_recipe if owner_cost is None else dict(d15_recipe, owner_cost=owner_cost)
            evaluated[key] = aligned_trial.evaluate(ensemble([halves[m] for m in members]), setting)
        return evaluated[key]

    chosen, trial, steps = greedy(pool, evaluate)
    entries, selected = [], None
    if chosen != ["d15"]:
        inputs = {member: metrica_signals(member, metrica_root / member, models) for member in chosen}
        candidate = ensemble_metrica(chosen, inputs)
        for cost in OWNER_COSTS:
            result = evaluate(tuple(chosen), cost)
            change = refine_trial.changes(base, result)
            veto = ms.veto(candidate, reference="D15", recipe=dict(d15_recipe, owner_cost=cost))
            allowed = refine_trial.passed(change, gain=GAIN) and veto["changes"]["passed"]
            entries.append({"owner_cost": cost, "eligible": allowed, "changes": change, "trial": result,
                            "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]}})
            print(f"D20 ensemble {'+'.join(chosen)} owner_cost {cost}: N = 3 standard {primary(result):.5f} "
                  f"({change['primary']:+.5f}), " + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items())
                  + ", " + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
                  + f", PFF {change['pff']:+.4f}, Metrica {veto['changes']['primary']:+.4f} -> "
                  + ("eligible" if allowed else "not eligible"), flush=True)
        eligible = [(i, entry) for i, entry in enumerate(entries) if entry["eligible"]]
        if eligible:
            selected = max(eligible, key=lambda item: (primary(item[1]["trial"]), primary(item[1]["trial"], "macro_f1"),
                                                       -item[0]))[1]["owner_cost"]
    record = {"protocol": PROTOCOL, "reference": base, "reproduction": {"d15_development": True}, "pool": pool,
              "members": {member: member_identity(member, models) for member in pool}, "greedy": steps,
              "chosen": chosen, "chosen_trial": trial, "entries": entries,
              "selected": None if selected is None else {"members": chosen, "owner_cost": selected},
              "teacher_sha256": diagnosis_identity(), "inputs": vendor_trial.inputs(parsed, games),
              "selected_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, record)
    if selected is not None:
        _, frozen = d15_setting()
        write_new(recipe / "recipe.json", {"protocol": PROTOCOL, "members": chosen,
                                           "identities": {member: member_identity(member, models) for member in chosen},
                                           "recipe": dict(d15_recipe, owner_cost=selected), "teacher_sha256": diagnosis_identity(),
                                           "weight": frozen["weight"], "pitch": frozen["pitch"]})
    print(f"D20 greedy selection {'+'.join(chosen)}; development selected "
          + (f"owner cost {selected}" if selected is not None else "nothing; D15 stays"), flush=True)
    return record["selected"]


def frozen_recipe(recipe=RECIPE, models=MODELS):
    frozen = json.loads((recipe / "recipe.json").read_text())
    if frozen["teacher_sha256"] != diagnosis_identity() or any(
            frozen["identities"][member] != member_identity(member, models) for member in frozen["members"]):
        raise ValueError("The D3 recipe or the D20 models differ from the ones D20 development selected")
    return frozen


def confirm(dfl, skill, parsed, *, raw, d15_cache, cache_root, metrica_root, recipe=RECIPE, models=MODELS):
    """The selected ensemble once on the reused DFL/SkillCorner holdout and PFF confirmation set C."""
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, refine_trial, scale_trial, vendor_trial
    if json.loads((recipe / "development.json").read_text())["selected"] is None:
        raise ValueError("Nothing was eligible on development; D15 stays and nothing is confirmed")
    output = recipe / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D20 confirmation is scored once")
    frozen = frozen_recipe(recipe, models)
    setting, d15 = d15_setting()
    games = vendor_trial.split_games("confirm_c")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games, "holdout", setting, raw=raw, cache=d15_cache)
    base = aligned_trial.evaluate(reference, d15["recipe"])
    expected = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]["external"]
    if vendor_trial.canonical(base["external"]) != expected:
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    halves = {"d15": reference}
    for member in frozen["members"]:
        if member != "d15":
            halves[member] = member_halves(member, dfl, skill, parsed, games, "holdout", raw=raw,
                                           cache=cache_root / member, models=models)
    challenger = aligned_trial.evaluate(ensemble([halves[member] for member in frozen["members"]]), frozen["recipe"])
    change = refine_trial.changes(base, challenger)
    inputs = {member: metrica_signals(member, metrica_root / member, models) for member in frozen["members"]}
    veto = ms.veto(ensemble_metrica(frozen["members"], inputs), reference="D15", recipe=frozen["recipe"])
    result = refine_trial.passed(change, gain=GAIN) and veto["changes"]["passed"]
    write_new(output, {"protocol": PROTOCOL, "members": frozen["members"], "recipe": frozen["recipe"],
                       "reproduction": {"d15_holdout": True}, "reference": base, "selected": challenger,
                       "changes": change, "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                       "passed": result, "identities": frozen["identities"], "inputs": vendor_trial.inputs(parsed, games)})
    print(f"D20 confirmation {'+'.join(frozen['members'])} owner_cost {frozen['recipe']['owner_cost']} (PFF = set C): "
          f"N = 3 standard {primary(challenger):.5f} ({change['primary']:+.5f}), "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f", PFF set C {change['pff']:+.4f}, Metrica {veto['changes']['primary']:+.4f} -> "
          + ("passed" if result else "failed"), flush=True)
    return result


def build(output, recipe=RECIPE, models=MODELS):
    """The competition file with the confirmed D20 ensemble."""
    from dfl_submission import diagnosis
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((recipe / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("The D20 ensemble did not pass confirmation; build D15 instead")
    frozen = frozen_recipe(recipe, models)
    teachers = [teacher(member, models) for member in frozen["members"]]

    def decode_half(positions, ball, teams, pitch):
        signals = mean_signals([half_signals(model, positions, ball, teams, pitch)[0] for model in teachers])
        return diagnosis.decode(signals, frozen["recipe"])

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen["pitch"]))


# -- cloud runs of the GRU and transformer families (D20 amendment) --------------------------------

CLOUD_FAMILIES = ("gru", "transformer")
CODE_PATHS = ("LICENSE", "LICENSE-SKILLCORNER", "dfl_submission", "evaluation", "experiments", "games.py", "starter")
INPUT_FOLDERS = {"current": "train_data", "augmented": "sequences_augmented"}
CLOUD_DEADLINE = "2026-11-13T08:00:00-05:00"
TRAINING_CODE = (  # definitions in this module that a cloud run's training and cross-validation execute
    "FOLDS", "HALO", "CORE", "WINDOW", "CV_NETWORK", "NETWORKS", "SETS", "GRIDS", "ANCHORS", "DROPOUT",
    "TRANSFORMER_LAYERS", "TRANSFORMER_HEADS", "WARMUP_SHARE", "EPS", "CPU_THREADS", "DEAD", "name", "fold_of",
    "fold_folder", "held_out", "layer_norm", "gelu", "sinusoid", "windows", "unwindow", "gru_direction", "gru_forward",
    "softmax", "transformer_forward", "FORWARD", "windowed_logits", "torch_modules", "learning_rate", "train_network",
    "export_network", "fit_network", "export", "final_weights", "network_probabilities", "log_loss", "half_losses",
    "summarise_matches", "network_cv_losses", "run_complete", "network_cv", "select_cv", "fold_table",
    "check_original_source", "train_family")
WHOLE_FILES = ("dfl_submission/temporal_network.py", "dfl_submission/temporal_cpu.py", "dfl_submission/train.py")


def tree_hash(files):
    """SHA-256 over sorted (path, SHA-256 of content) pairs: the identity of a code tree."""
    import hashlib
    lines = "".join(f"{path}\0{files[path]}\n" for path in sorted(files))
    return hashlib.sha256(lines.encode()).hexdigest()


def folder_files(root):
    """{relative path: SHA-256} of every file below `root`, without byte-code caches."""
    return {path.relative_to(root).as_posix(): digest(path) for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}


def git(*args, binary=False):
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode()


def commit_files(commit, paths=CODE_PATHS):
    """{path: SHA-256} of the tracked files under `paths` at `commit`, as `git archive` writes them."""
    import hashlib
    names = [name for name in git("ls-tree", "-r", "-z", "--name-only", commit, "--", *paths).split("\0") if name]
    return {name: hashlib.sha256(git("cat-file", "blob", f"{commit}:{name}", binary=True)).hexdigest() for name in names}


def definitions(text):
    """Source text of each top-level function, class and assignment, by name."""
    import ast
    result = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            result[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for item in target.elts if isinstance(target, ast.Tuple) else (target,):
                    if isinstance(item, ast.Name):
                        result[item.id] = ast.get_source_segment(text, node)
    return result


def training_code_matches(commit):
    """Whether the training code at `commit` is the training code here (named definitions and whole files)."""
    old = definitions(git("show", f"{commit}:dfl_submission/zoo.py"))
    new = definitions((PACKAGE / "zoo.py").read_text())
    same = {name: old.get(name) == new.get(name) and new.get(name) is not None for name in TRAINING_CODE}
    same |= {path: git("show", f"{commit}:{path}") == (ROOT / path).read_text() for path in WHOLE_FILES}
    return same


def input_identity(folder):
    manifest = folder / "manifest.json"
    return {"manifest_sha256": digest(manifest),
            "files": {record["file"]: record["sha256"] for record in json.loads(manifest.read_text())["records"]}}


def published_manifests():
    """The training manifest each model set's published networks were trained on."""
    from dfl_submission import scale_trial
    return {key: json.loads((scale_trial.PUBLISHED[key] / "temporal_models" / "fit.json").read_text())["train_manifest"]
            for key in SETS}


def verify_inputs(sources, manifests=None):
    """Check both prepared folders against the published training manifests and every file's hash."""
    manifests = published_manifests() if manifests is None else manifests
    identities = {}
    for model_set, folder in sources.items():
        expected = manifests[model_set]
        identity = input_identity(folder)
        if identity["manifest_sha256"] != expected:
            raise ValueError(f"{folder} is not the {model_set} set's prepared-sequence folder")
        present = {path.name for path in folder.iterdir()}
        if present != set(identity["files"]) | {"manifest.json"}:
            raise ValueError(f"{folder} holds files outside its manifest: {sorted(present - set(identity['files']))}")
        for filename, expected_hash in identity["files"].items():
            if digest(folder / filename) != expected_hash:
                raise ValueError(f"Prepared sequence hash differs: {folder / filename}")
        identities[INPUT_FOLDERS[model_set]] = identity
    return identities


def environment():
    import platform
    import torch
    record = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        record["gpu"] = torch.cuda.get_device_name(0)
    return record


def package_cloud(family, work, code, sources, output, *, recipe=RECIPE, manifests=None, environment_of=None):
    """Zip a cloud run's outputs with its record: code, environment, input hashes, seeds and output hashes."""
    import time
    import zipfile
    cv_record = recipe / f"cv_{family}.json"
    cv = json.loads(cv_record.read_text())
    files = {f"zoo_recipe/cv_{family}.json": cv_record}
    for entry in cv["entries"]:
        for fold in entry["runs"]:
            files[f"cv_runs/{name(entry['setting'])}/fold{fold}/fit.json"] = work / "cv" / family / name(entry["setting"]) / f"fold{fold}" / "fit.json"
    for model_set in SETS:
        for slot in NETWORKS:
            run = work / "networks" / family / model_set / slot
            for path in sorted(run.iterdir()):
                files[f"runs/{model_set}/{slot}/{path.name}"] = path
    for path in sorted((work / "logs").glob("*.log")) if (work / "logs").exists() else ():
        files[f"logs/{path.name}"] = path
    record = {"protocol": PROTOCOL, "family": family, "code": json.loads(code.read_text()),
              "environment": (environment if environment_of is None else environment_of)(),
              "inputs": verify_inputs(sources, manifests), "seeds": {"cv": {"seed": CV_NETWORK[0], "boundary_boost": CV_NETWORK[1]},
                                                         "full": {slot: list(value) for slot, value in NETWORKS.items()}},
              "selected": cv["selected"], "outputs": {key: digest(path) for key, path in files.items()},
              "created_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, path in files.items():
            archive.write(path, key)
        archive.writestr("cloud_record.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Packaged {len(files)} files into {output}", flush=True)
    return record


def safe_extract(archive, folder):
    import zipfile
    with zipfile.ZipFile(archive) as item:
        for member in item.namelist():
            target = (folder / member).resolve()
            if not str(target).startswith(str(folder.resolve()) + os.sep):
                raise ValueError(f"Unsafe path in the cloud archive: {member}")
        item.extractall(folder)


def check_cloud(family, folder, sources, scratch, manifests=None):
    """Every check the D20 amendment requires of a cloud run before its outputs are used."""
    manifests = published_manifests() if manifests is None else manifests
    record = json.loads((folder / "cloud_record.json").read_text())
    present = {key for key in folder_files(folder) if key != "cloud_record.json"}
    checks = {"family": record["family"] == family and record["protocol"] == PROTOCOL,
              "outputs": present == set(record["outputs"])
              and all(digest(folder / key) == value for key, value in record["outputs"].items())}
    checks["inputs"] = record["inputs"] == {INPUT_FOLDERS[key]: input_identity(path) for key, path in sources.items()}
    commit = record["code"]["commit"]
    try:
        checks["code_tree"] = tree_hash(commit_files(commit)) == record["code"]["tree_sha256"]
        checks["training_code"] = training_code_matches(commit)
    except subprocess.CalledProcessError:
        checks["code_tree"], checks["training_code"] = False, {"commit_missing": False}
    cv = json.loads((folder / "zoo_recipe" / f"cv_{family}.json").read_text())
    checks["cv"] = (cv["family"] == family and cv["grid"] == list(GRIDS[family]) and cv["folds"] == fold_table(sources["current"])
                    and cv["source_manifest"] == digest(sources["current"] / "manifest.json")
                    and cv["cv_network"] == {"seed": CV_NETWORK[0], "boundary_boost": CV_NETWORK[1]}
                    and cv["selected"] == select_cv(cv["entries"]) == record["selected"]
                    and [entry["setting"] for entry in cv["entries"]] == list(GRIDS[family]))
    folds = {}
    for entry in cv["entries"]:
        for fold, run in entry["runs"].items():
            fit_path = folder / "cv_runs" / name(entry["setting"]) / f"fold{fold}" / "fit.json"
            fit = json.loads(fit_path.read_text())
            fold_manifest = digest(fold_folder(sources["current"], int(fold), scratch / "folds" / fold) / "manifest.json")
            folds[f"{name(entry['setting'])}/fold{fold}"] = (
                digest(fit_path) == run["fit_sha256"] and fit["train_manifest"] == run["train_manifest"] == fold_manifest
                and fit["family"] == family and fit["width"] == entry["setting"]["width"]
                and fit["steps"] == entry["setting"]["steps"] and (fit["seed"], fit["boundary_boost"]) == CV_NETWORK)
    checks["cv_runs"] = folds
    runs = {}
    for model_set in SETS:
        expected = manifests[model_set]
        for slot, seed_boost in NETWORKS.items():
            run = folder / "runs" / model_set / slot
            fit = json.loads((run / "fit.json").read_text())
            runs[f"{model_set}/{slot}"] = (
                fit["family"] == family and fit["width"] == cv["selected"]["width"] and fit["steps"] == cv["selected"]["steps"]
                and (fit["seed"], fit["boundary_boost"]) == seed_boost and fit["train_manifest"] == expected
                and digest(run / "normalization.npz") == fit["normalization"]
                and all(digest(run / key) == value for key, value in fit["models"].items()))
    checks["runs"] = runs
    return record, cv, checks


def passed_checks(checks):
    return all(all(value.values()) if isinstance(value, dict) else value for value in checks.values())


def import_cloud(family, archive, folder, sources, scratch, *, recipe=RECIPE, models=MODELS, now=None, manifests=None,
                 exporter=None):
    """Check a cloud run against the D20 amendment, then adopt its CV record and export its networks locally."""
    require_external_only()
    import datetime
    import time
    output = recipe / f"cloud_{family}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; a cloud run is imported once")
    if (recipe / f"cv_{family}.json").exists():
        raise FileExistsError(f"cv_{family}.json exists (a local run); the cloud run is not used")
    now = datetime.datetime.now(datetime.timezone.utc) if now is None else now
    if now > datetime.datetime.fromisoformat(CLOUD_DEADLINE):
        raise ValueError(f"Past {CLOUD_DEADLINE}: the {family} family runs locally instead (D20 amendment)")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    safe_extract(archive, folder)
    record, cv, checks = check_cloud(family, folder, sources, scratch, manifests)
    exporter = export if exporter is None else exporter
    passed = passed_checks(checks)
    exported = {}
    error = None
    if passed:
        try:
            for model_set in SETS:
                for slot in NETWORKS:
                    destination = models / family / model_set / slot
                    if destination.exists():
                        shutil.rmtree(destination)
                    exporter(family, folder / "runs" / model_set / slot, destination)
                    exported[f"{model_set}/{slot}"] = digest(destination / "fit.json")
        except subprocess.CalledProcessError as failure:
            passed, error = False, f"local NumPy export check failed: {failure}"
        else:
            shutil.copyfile(folder / "zoo_recipe" / f"cv_{family}.json", recipe / f"cv_{family}.json")
    write_new(output, {"protocol": PROTOCOL, "family": family, "archive_sha256": digest(archive), "checks": checks,
                       "passed": passed, "error": error, "cloud_record": record, "exported": exported,
                       "imported_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"Cloud {family} run {'passed every check and was exported' if passed else 'FAILED a check; the family is left out'}: "
          f"{checks}", flush=True)
    return passed


# -- command line ----------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="one GRU or transformer network (internal)")
    training.add_argument("--family", choices=("gru", "transformer"), required=True)
    training.add_argument("--source", type=Path, required=True)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--boundary-boost", type=float, required=True)
    training.add_argument("--width", type=int, required=True)
    training.add_argument("--steps", type=int, required=True)
    conversion = commands.add_parser("export", help="NumPy export of one GRU or transformer network (internal)")
    conversion.add_argument("--source", type=Path, required=True)
    conversion.add_argument("--out", type=Path, required=True)
    cv = commands.add_parser("cv", help="cross-validate one family on the training matches")
    cv.add_argument("--family", choices=FAMILIES, required=True)
    cv.add_argument("--source", type=Path, required=True, help="the original set's prepared sequences")
    cv.add_argument("--scratch", type=Path, required=True)
    full = commands.add_parser("fit", help="train a family's selected setting on every training match")
    full.add_argument("--family", choices=FAMILIES, required=True)
    full.add_argument("--current", type=Path, required=True, help="the original set's prepared sequences")
    full.add_argument("--augmented", type=Path, required=True, help="the D2 set's prepared sequences")
    full.add_argument("--scratch", type=Path, required=True)
    for command in (cv, full):
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), help="default: the family's device in D20")
    tree = commands.add_parser("code-tree", help="hash an extracted code tree (cloud notebooks)")
    tree.add_argument("--root", type=Path, required=True)
    tree.add_argument("--commit", required=True)
    tree.add_argument("--out", type=Path, required=True)
    packing = commands.add_parser("package-cloud", help="zip a cloud run's outputs with its record (cloud notebooks)")
    packing.add_argument("--family", choices=CLOUD_FAMILIES, required=True)
    packing.add_argument("--work", type=Path, required=True)
    packing.add_argument("--code", type=Path, required=True, help="the code-tree record")
    packing.add_argument("--current", type=Path, required=True)
    packing.add_argument("--augmented", type=Path, required=True)
    packing.add_argument("--out", type=Path, required=True)
    importing = commands.add_parser("import-cloud", help="check a cloud run and export its networks locally")
    importing.add_argument("--family", choices=CLOUD_FAMILIES, required=True)
    importing.add_argument("--archive", type=Path, required=True)
    importing.add_argument("--current", type=Path, required=True)
    importing.add_argument("--augmented", type=Path, required=True)
    importing.add_argument("--scratch", type=Path, required=True, help="where the archive is extracted and folds rebuilt")
    for name_ in ("member", "develop", "confirm"):
        command = commands.add_parser(name_)
        if name_ == "member":
            command.add_argument("--family", choices=FAMILIES, required=True)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--d15-cache", type=Path, required=True, help="D15's decoder inputs for this split")
        command.add_argument("--cache", type=Path, required=True, help="one folder of decoder inputs per member below it")
        if name_ != "member":
            command.add_argument("--metrica-cache", type=Path, required=True, help="one folder per member below it")
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        export_network(args.source, args.out)
        return
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "train":
        train_network(args.family, args.source, args.out, args.device, seed=args.seed,
                      boundary_boost=args.boundary_boost, width=args.width, steps=args.steps)
    elif args.command == "cv":
        if args.family == "trees":
            tree_cv(args.source, args.scratch)
        else:
            network_cv(args.family, args.source, args.scratch, device=args.device)
    elif args.command == "fit":
        sources = {"current": args.current, "augmented": args.augmented}
        if args.family == "trees":
            train_trees(sources)
        else:
            train_family(args.family, sources, args.scratch, device=args.device)
    elif args.command == "code-tree":
        files = folder_files(args.root)
        write_json(args.out, {"commit": args.commit, "tree_sha256": tree_hash(files), "files": len(files)})
    elif args.command == "package-cloud":
        package_cloud(args.family, args.work, args.code, {"current": args.current, "augmented": args.augmented}, args.out)
    elif args.command == "import-cloud":
        import_cloud(args.family, args.archive, args.scratch / args.family,
                     {"current": args.current, "augmented": args.augmented}, args.scratch)
    elif args.command == "member":
        score_member(args.family, args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache,
                     cache_root=args.cache)
    else:
        action = develop if args.command == "develop" else confirm
        action(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache,
               metrica_root=args.metrica_cache)

if __name__ == "__main__":
    main()
