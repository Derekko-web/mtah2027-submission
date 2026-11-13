"""D47: mirror symmetry, token noise and seed averaging for the player-set networks, screened on development.

Commands (DECISIONS.md D47):
- `check-trainer` retrains D42's original-set seed-42 network with every `player_set3` switch off and
  compares its snapshots with D42's bit for bit;
- `parity` re-extracts the 162 columns (and the player tokens) of every prepared training half of one
  set from mirrored tracking and classifies each column per mirror; `parity-record` joins the two sets
  into the record and the sign map;
- `networks` trains and exports one set's two networks of a candidate into `data/d47/networks/<candidate>/`;
- `reference` checks that D43, decoded from `data/d42/cache`, reproduces its development, Metrica and
  D24 records, and records D43's eight group scores at ownership cost 10;
- `score` forecasts several candidates in one pass (the features, tokens and trees of each half are
  computed once; D43's own forecasts are recomputed on the way and must equal `data/d42/cache` bit for
  bit), decodes every candidate at its costs and writes one record per candidate;
- `summary` prints the table of every record.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "screen_recipe"
SCRATCH = ROOT / "data" / "d47"
D43_CACHE = ROOT / "data" / "d42" / "cache"
PROTOCOL = "DECISIONS.md D47"
TOLERANCE = 1e-4
MIRRORED = ("x", "y", "xy")
REFERENCE_COST = 10
LOSS = .01
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica", "d24")
SETS = ("current", "augmented")
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
TRAINED = {
    "S1000": {"seed_offset": 1000, "mirror": False, "noise": False},
    "S2000": {"seed_offset": 2000, "mirror": False, "noise": False},
    "F": {"seed_offset": 0, "mirror": True, "noise": False},
    "N": {"seed_offset": 0, "mirror": False, "noise": True},
    "FN": {"seed_offset": 0, "mirror": True, "noise": True},
}
CANDIDATES = {
    "S1000": {"slots": ("S1000",), "mirrored": False, "costs": (8, 10, 12)},
    "S2000": {"slots": ("S2000",), "mirrored": False, "costs": (8, 10, 12)},
    "M3": {"slots": ("D43", "S1000", "S2000"), "mirrored": False, "costs": (6, 8, 10, 12)},
    "F0": {"slots": ("D43",), "mirrored": True, "costs": (8, 10, 12)},
    "F": {"slots": ("F",), "mirrored": True, "costs": (8, 10, 12)},
    "N": {"slots": ("N",), "mirrored": False, "costs": (8, 10, 12)},
    "FN": {"slots": ("FN",), "mirrored": True, "costs": (8, 10, 12)},
}
LEVERS = ("M3", "F0", "F", "N", "FN")
MIRROR_FAMILY = ("F0", "F", "FN")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D47 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


# -- the trainer check -----------------------------------------------------------------------------

def published_raw_hashes():
    """D42's original-set seed-42 snapshots, by the hashes its export recorded."""
    from dfl_submission import player_trial
    fit = read(player_trial.network_folders()["current"] / "temporal_models" / "fit.json")
    return fit, fit["original_checkpoints"]


def check_trainer(source, tokens, scratch, *, output=None):
    """Train with every switch off (subprocess, MPS) and compare every snapshot's hash with D42's."""
    require_external_only()
    output = RECIPE / "trainer_check.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the trainer check runs once")
    fit, expected = published_raw_hashes()
    if digest(source / "manifest.json") != fit["train_manifest"] or digest(tokens / "manifest.json") != fit["token_manifest"]:
        raise ValueError("The check must use the prepared sequences and tokens D42's network trained on")
    run = scratch / "current" / "temporal_models"
    if not (run / "fit.json").exists():
        if run.exists():
            raise FileExistsError(f"{run} holds an unfinished run; remove it first")
        subprocess.run([sys.executable, "-m", "dfl_submission.player_set3", "--source", str(source), "--tokens",
                        str(tokens), "--out", str(run), "--seed", "42", "--boundary-boost", "0.5"], check=True, cwd=ROOT)
    trained = read(run / "fit.json")
    for key in ("seed", "boundary_boost", "width", "steps", "lr_scale", "batch", "weight_decay", "dropout",
                "train_manifest", "token_manifest", "training_device", "torch"):
        if trained[key] != fit[key]:
            raise ValueError(f"The check run differs from D42's in {key}: {trained[key]} against {fit[key]}")
    actual = trained["models"]
    passed = actual == expected
    write_new(output, {"protocol": PROTOCOL, "check": "switches-off retrain of D42's original-set seed-42 network",
                       "expected": expected, "actual": actual, "passed": passed, "switches": trained["d47"],
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D47 trainer check: {'bit for bit' if passed else 'DIFFERS'} "
          f"({actual.get('temporal_5400.pt', '')[:12]} against {expected['temporal_5400.pt'][:12]})", flush=True)
    return passed


# -- the mirror parity check -----------------------------------------------------------------------

def column_comparison(mirrored, stored, tolerance=TOLERANCE):
    """Per column: (unchanged, pure sign change, max |m - s|, max |m + s|) over one half, NaN patterns included."""
    mirrored = np.asarray(mirrored, dtype=np.float64)
    stored = np.asarray(stored, dtype=np.float64)
    same_missing = (np.isnan(mirrored) == np.isnan(stored)).all(axis=0)
    both = np.isfinite(mirrored) & np.isfinite(stored)
    same = np.where(both, np.abs(mirrored - stored), 0).max(axis=0, initial=0)
    flipped = np.where(both, np.abs(mirrored + stored), 0).max(axis=0, initial=0)
    return same_missing & (same <= tolerance), same_missing & (flipped <= tolerance), same, flipped


def token_difference(mirrored, expected):
    """Largest difference between tokens from mirrored tracking and sign-flipped tokens (inf if NaN patterns differ)."""
    if not np.array_equal(np.isnan(mirrored), np.isnan(expected)):
        return float("inf")
    both = np.isfinite(mirrored)
    return float(np.where(both, np.abs(mirrored.astype(np.float64) - expected), 0).max(initial=0))


def parity(model_set, source, dfl, skill, output, pitch_output, *, raw):
    """One set's half of the parity check, written to scratch; `parity_record` joins the sets."""
    require_external_only()
    from dfl_submission import aligned_trial, long_context, player_set as ps, player_set3 as p3, touch_trial
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for path in (output, pitch_output):
        if path.exists():
            raise FileExistsError(f"{path} exists")
    manifest = read(source / "manifest.json")
    columns = manifest["columns"]
    count = len(columns)
    unchanged = {m: np.ones(count, bool) for m in MIRRORED}
    flipped = {m: np.ones(count, bool) for m in MIRRORED}
    worst_same = {m: np.zeros(count) for m in MIRRORED}
    worst_flip = {m: np.zeros(count) for m in MIRRORED}
    tokens_worst = {m: 0. for m in MIRRORED}
    pitches, halves_checked = {}, 0
    halves = touch_trial.training_halves(model_set, dfl, skill, raw=raw)
    for (record, path), half in zip(aligned_trial.verified_records(source), halves, strict=True):
        if record["file"] != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match the prepared record {record['file']}")
        with np.load(path, allow_pickle=False) as item:
            stored = item["values"]
        frames = np.arange(0, len(half.ball), long_context.STRIDE)
        if len(frames) != len(stored):
            raise ValueError(f"Prepared frames differ: {record['file']}")
        base_tokens = ps.tokens(half.positions, half.ball, half.teams, half.pitch, frames)
        pitches[record["file"]] = [float(v) for v in np.asarray(half.pitch)]
        for mirror in MIRRORED:
            positions, ball = p3.mirror_tracking(half.positions, half.ball, mirror)
            mirrored_frames, table, _ = long_context.features(positions, ball, half.teams, half.pitch)
            if table.columns.tolist() != columns or not np.array_equal(mirrored_frames, frames):
                raise ValueError(f"Mirrored extraction changed the columns or frames: {record['file']}")
            same, sign, d_same, d_flip = column_comparison(table.to_numpy(dtype=np.float32), stored)
            unchanged[mirror] &= same
            flipped[mirror] &= sign
            worst_same[mirror] = np.maximum(worst_same[mirror], d_same)
            worst_flip[mirror] = np.maximum(worst_flip[mirror], d_flip)
            mirrored_tokens = ps.tokens(positions, ball, half.teams, half.pitch, frames)
            tokens_worst[mirror] = max(tokens_worst[mirror],
                                       token_difference(mirrored_tokens, base_tokens * p3.TOKEN_SIGNS[mirror]))
        halves_checked += 1
        print(f"Parity {model_set} {record['file']}", flush=True)
    pitch_output.parent.mkdir(parents=True, exist_ok=True)
    pitch_output.write_text(json.dumps(pitches, indent=2, sort_keys=True) + "\n")
    output.write_text(json.dumps({
        "protocol": PROTOCOL, "set": model_set, "columns": columns, "halves": halves_checked,
        "sequences_manifest": digest(source / "manifest.json"), "tolerance": TOLERANCE,
        "unchanged": {m: unchanged[m].tolist() for m in MIRRORED}, "sign_change": {m: flipped[m].tolist() for m in MIRRORED},
        "max_unchanged_difference": {m: worst_same[m].tolist() for m in MIRRORED},
        "max_sign_difference": {m: worst_flip[m].tolist() for m in MIRRORED},
        "token_max_difference": tokens_worst, "pitches_sha256": digest(pitch_output)}, indent=2, sort_keys=True) + "\n")
    return output


def classify(parts):
    """{mirror: [+1 / -1 / 0 per column]} from per-set parity parts: 0 marks a column that is neither."""
    columns = parts[0]["columns"]
    if any(part["columns"] != columns for part in parts):
        raise ValueError("The sets were checked on different columns")
    signs = {}
    for mirror in MIRRORED:
        same = np.logical_and.reduce([np.asarray(part["unchanged"][mirror]) for part in parts])
        flip = np.logical_and.reduce([np.asarray(part["sign_change"][mirror]) for part in parts])
        signs[mirror] = np.where(same, 1, np.where(flip, -1, 0)).tolist()
    return columns, signs


def worst_differences(part, signs):
    """{mirror: largest difference over the columns as classified (|m - s| if unchanged, |m + s| if a sign change)}."""
    result = {}
    for mirror in MIRRORED:
        values = [part["max_unchanged_difference"][mirror][i] if s == 1 else part["max_sign_difference"][mirror][i]
                  for i, s in enumerate(signs[mirror]) if s != 0]
        result[mirror] = max(values) if values else None
    return result


def parity_record(paths, output=None):
    require_external_only()
    output = RECIPE / "parity.json" if output is None else output
    parts = [read(path) for path in paths]
    if sorted(part["set"] for part in parts) != sorted(SETS):
        raise ValueError("The parity record needs one part per model set")
    columns, signs = classify(parts)
    neither = {m: [columns[i] for i, s in enumerate(signs[m]) if s == 0] for m in MIRRORED}
    flipped = {m: [columns[i] for i, s in enumerate(signs[m]) if s == -1] for m in MIRRORED}
    tokens_exact = {part["set"]: part["token_max_difference"] for part in parts}
    tokens_passed = all(value <= 1e-6 for item in tokens_exact.values() for value in item.values())
    passed = not any(neither.values())
    record = {"protocol": PROTOCOL, "passed": passed, "tolerance": TOLERANCE,
              "halves": {part["set"]: part["halves"] for part in parts},
              "sequences_manifests": {part["set"]: part["sequences_manifest"] for part in parts},
              "pitches_sha256": {part["set"]: part["pitches_sha256"] for part in parts},
              "neither": neither, "sign_change_columns": flipped,
              "max_difference": {part["set"]: worst_differences(part, signs) for part in parts},
              "tokens": {"max_difference": tokens_exact, "passed": tokens_passed},
              "sign_map": {"columns": columns, "signs": signs},
              "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, record)
    print(f"D47 parity: {'passed' if passed else 'FAILED'}; sign changes "
          + ", ".join(f"{m} {len(flipped[m])}" for m in MIRRORED) + "; neither "
          + ", ".join(f"{m} {neither[m]}" for m in MIRRORED) + f"; tokens {'exact' if tokens_passed else 'DIFFER'} "
          + json.dumps(tokens_exact), flush=True)
    return passed


def sign_map():
    record = read(RECIPE / "parity.json")
    if not record["passed"] or not record["tokens"]["passed"]:
        raise ValueError("The mirror parity check did not pass; F0, F and FN stop")
    return record["sign_map"]


# -- training ----------------------------------------------------------------------------------------

def network_root(candidate, scratch=SCRATCH / "networks"):
    return scratch / candidate


def export_folder(candidate, model_set, folder, scratch=SCRATCH / "networks"):
    return network_root(candidate, scratch) / "export" / model_set / folder


def train_networks(candidate, model_set, source, tokens, *, scratch=SCRATCH / "networks"):
    require_external_only()
    from dfl_submission import touch_trial
    config = TRAINED[candidate]
    if read(RECIPE / "trainer_check.json")["passed"] is not True:
        raise ValueError("The D47 trainer check did not pass; nothing is trained")
    if config["mirror"]:
        sign_map()
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for folder, (seed, boost) in NETWORKS.items():
        output = export_folder(candidate, model_set, folder, scratch)
        if (output / "fit.json").exists():
            continue
        raw = network_root(candidate, scratch) / "raw" / model_set / folder
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            command = [sys.executable, "-m", "dfl_submission.player_set3", "--source", str(source), "--tokens",
                       str(tokens), "--out", str(raw), "--seed", str(seed), "--boundary-boost", str(boost),
                       "--seed-offset", str(config["seed_offset"])]
            if config["mirror"]:
                command += ["--mirror", "--sign-map", str(RECIPE / "parity.json")]
            if config["noise"]:
                command += ["--noise", "--pitches", str(SCRATCH / f"pitches_{model_set}.json")]
            raw.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True, cwd=ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- candidates ----------------------------------------------------------------------------------------

def slot_folders(name, scratch=SCRATCH / "networks"):
    """{set: {folder: network folder}} of D43's networks or a trained D47 model."""
    from dfl_submission import player_trial
    if name == "D43":
        return {key: {folder: player_trial.network_folders()[key] / folder for folder in NETWORKS} for key in SETS}
    return {key: {folder: export_folder(name, key, folder, scratch) for folder in NETWORKS} for key in SETS}


def identity(candidate, scratch=SCRATCH / "networks"):
    from dfl_submission import scale_trial
    spec = CANDIDATES[candidate]
    result = {"protocol": PROTOCOL, "candidate": candidate, "mirrored": spec["mirrored"],
              "slots": {name: {key: {folder: digest(path / "fit.json") for folder, path in folders.items()}
                               for key, folders in slot_folders(name, scratch).items()} for name in spec["slots"]},
              "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in SETS},
              "player_set_sha256": digest(PACKAGE / "player_set.py"),
              "player_set3_sha256": digest(PACKAGE / "player_set3.py")}
    if spec["mirrored"]:
        result["sign_map_sha256"] = hashlib.sha256(json.dumps(sign_map(), sort_keys=True).encode()).hexdigest()
    return result


def variant_name(candidate, scratch=SCRATCH / "networks"):
    tag = hashlib.sha256(json.dumps(identity(candidate, scratch), sort_keys=True).encode()).hexdigest()[:16]
    return f"D47 {candidate} {tag}"


class Network:
    """One network as a forecast slot uses it: plain or mirror-averaged inference."""

    def __init__(self, folder, mirrored, signs):
        from dfl_submission import player_trial
        self.checkpoint = player_trial.PlayerCheckpoint(folder)
        self.mirrored, self.signs = mirrored, signs
        self.key = (str(folder), mirrored)

    def predict(self, frames, owner, player_tokens, length):
        from dfl_submission import player_set3 as p3
        columns = list(self.checkpoint.columns)
        values = owner[columns].to_numpy(dtype=np.float32)
        return p3.predict(self.checkpoint.checkpoint, frames, values, columns, player_tokens, length,
                          mirrored=self.mirrored, sign_map=self.signs)


def tracking_key(positions, ball, teams, pitch):
    hasher = hashlib.blake2b(digest_size=20)
    for array in (positions, ball, teams, pitch):
        array = np.ascontiguousarray(array)
        hasher.update(str((array.shape, array.dtype.str)).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


class Forecaster:
    """Decoder inputs of several candidates per half, with the features, tokens and trees computed once.

    The first candidate asked for a half triggers the forecasts of every pending candidate (and of D43's
    own networks, logged for the bit-for-bit check against `data/d42/cache`); the others are served from
    memory, matched by the half's tracking.
    """

    def __init__(self, candidates, scratch=SCRATCH / "networks"):
        from dfl_submission import long_context, player_trial, scale_trial
        reference, _, frozen = player_trial.teacher()
        self.previous, self.frozen = reference[2]["recipe"], frozen
        if self.previous["variant"] != "temporal_1800.npz" or frozen["recipe"]["variant"] != "seed_phase":
            raise ValueError("D47 expects D15's variants")
        self.trees = {role: long_context.load(scale_trial.TREES[key])[0] for role, key in scale_trial.ROLES.items()}
        self.roles = scale_trial.ROLES
        self.candidates = list(candidates)
        signs = sign_map() if any(CANDIDATES[c]["mirrored"] for c in self.candidates) else None
        loaded = {}

        def network(name, key, folder, mirrored):
            path = slot_folders(name, scratch)[key][folder]
            return loaded.setdefault((str(path), mirrored), Network(path, mirrored, signs if mirrored else None))

        self.slots = {}
        for candidate in ["D43", *self.candidates]:
            spec = CANDIDATES.get(candidate, {"slots": ("D43",), "mirrored": False})
            self.slots[candidate] = {role: {folder: [network(name, key, folder, spec["mirrored"]) for name in spec["slots"]]
                                            for folder in NETWORKS} for role, key in self.roles.items()}
        self.memo, self.d43_log = {}, []

    def forecast(self, positions, ball, teams, pitch):
        from dfl_submission import diagnosis, ensemble_trial, long_context, motion_trial, player_set as ps
        from dfl_submission import retrain_trial, temporal_trial
        phases, baselines = [], {}
        for offset in ensemble_trial.PHASES:
            shifted_positions, shifted_ball = positions[offset:], ball[offset:]
            frames, owner, dead = long_context.features(shifted_positions, shifted_ball, teams, pitch)
            length = len(ball) - offset
            player_tokens = ps.tokens(shifted_positions, shifted_ball, teams, pitch, frames)
            if offset == 0:
                for role in self.roles:
                    baselines[role], _ = temporal_trial.forecast_features(self.trees[role], {}, frames, owner, dead, length)
            phases.append((offset, frames, owner, player_tokens, length))
        outputs = {}

        def run(net, index):
            key = (net.key, index)
            if key not in outputs:
                _, frames, owner, player_tokens, length = phases[index]
                outputs[key] = net.predict(frames, owner, player_tokens, length)
            return outputs[key]

        def slot(nets, index):
            return retrain_trial.mean([run(net, index) for net in nets])

        result = {}
        for candidate, roles in self.slots.items():
            per_role = []
            for role in ("current", "candidate"):
                original, late = roles[role]["temporal_models"], roles[role]["seed_models"]
                start_reference = (baselines[role], {ensemble_trial.LATE: slot(original, 0)})
                phase_seed = [tuple(np.pad(p, (offset, 0), mode="edge") for p in slot(late, index))
                              for index, (offset, *_) in enumerate(phases)]
                per_role.append((start_reference, {"seed_phase": ensemble_trial.mean(phase_seed)}))
            mixed = retrain_trial.mix(*per_role, self.frozen["weight"])
            labels = motion_trial.decode(mixed, self.previous, self.frozen["recipe"])
            signals = tuple(motion_trial.blend_signals(mixed, self.previous, self.frozen["recipe"]))
            if not np.array_equal(diagnosis.decode(signals, self.frozen["recipe"]), labels):
                raise ValueError("Decoding the forecast signals does not reproduce the labels")
            result[candidate] = (signals, labels)
        return result

    def signals_of(self, candidate):
        def forecast(_, positions, ball, teams, pitch):
            key = tracking_key(positions, ball, teams, pitch)
            waiting = self.memo.get(key, {}).get(candidate)
            if not waiting:
                computed = self.forecast(positions, ball, teams, pitch)
                self.d43_log.append(computed.pop("D43"))
                for name, value in computed.items():
                    self.memo.setdefault(key, {}).setdefault(name, []).append(value)
                waiting = self.memo[key][candidate]
            value = waiting.pop(0)
            if not waiting:
                del self.memo[key][candidate]
                if not self.memo[key]:
                    del self.memo[key]
            return value
        return forecast

    def check_d43(self, halves):
        """D43's recomputed forecasts against its cached decoder inputs, half by half, bit for bit."""
        log, self.d43_log = self.d43_log, []
        if not log:
            return 0
        if len(log) != len(halves):
            raise ValueError(f"D43 was recomputed on {len(log)} halves, its cache holds {len(halves)}")
        for (signals, labels), half in zip(log, halves, strict=True):
            if not (all(np.array_equal(a, b) for a, b in zip(signals, half.signals, strict=True))
                    and np.array_equal(labels, half.d3)):
                raise ValueError(f"D43 recomputed differs from data/d42/cache on {half.key} half {half.index}")
        return len(log)


# -- D43's reference ---------------------------------------------------------------------------------

def d43_halves(dfl, skill, parsed, *, raw):
    """D43's development, Metrica and D24 decoder inputs from `data/d42/cache` (identity-checked)."""
    from dfl_submission import metrica_sample as ms, metrica_views as mv, player_trial, vendor_trial
    games = vendor_trial.development()
    development = player_trial.candidate_halves(dfl, skill, parsed, games, "development", raw=raw,
                                                cache=D43_CACHE / "development")
    model = player_trial.teacher()
    _, metrica = ms.reference_halves("D15", cache=D43_CACHE / "metrica", model=model,
                                     signals_of=player_trial.half_signals, variant="D42 player set")
    views = mv.view_halves(cache=D43_CACHE / "d24", signals_of=lambda _, *t: player_trial.half_signals(model, *t),
                           variant="D42 player set")
    return development, metrica, views


def group_scores(trial, metrica_means, d24_mean):
    """D47's eight development groups (N = 3 spellF1) and their unweighted mean, the composite."""
    matches = trial["external"]["conventions"]["aba3"]["matches"]
    groups = {f"dfl_{view}": matches[f"dfl/WPY/{view}"]["spell_f1"] for view in ("clean", "masked", "dead_carried", "team_swap")}
    skill = [value["spell_f1"] for key, value in matches.items() if key.startswith("skillcorner/")]
    if len(skill) != 4 or len(trial["pff"]["matches"]) != 8:
        raise ValueError("Development must hold four SkillCorner and eight PFF matches")
    groups["skillcorner"] = float(np.mean(skill))
    groups["pff"] = trial["pff"]["means"]["aba3"]["spell_f1"]
    groups["metrica"] = metrica_means["aba3"]["spell_f1"]
    groups["d24"] = d24_mean
    groups = {name: float(groups[name]) for name in GROUPS}
    return groups, float(np.mean([groups[name] for name in GROUPS]))


def evaluate(development, metrica, views, truths, cost):
    """Every D47 quantity of one set of decoder inputs at one ownership cost."""
    from dfl_submission import aligned_trial, diagnosis, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import metrica_views as mv, objective_trial, refine_trial
    recipe = ht.recipe_at(cost)
    trial = aligned_trial.evaluate(development, recipe)
    metrica_result = ms.veto(objective_trial.decoded(metrica, recipe), reference="D15")
    view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, recipe))
    d24 = mv.d24_mean(view_scores)
    groups, composite = group_scores(trial, metrica_result["candidate"]["means"], d24)
    return {"owner_cost": cost, "recipe": recipe, "groups": groups, "composite": composite,
            "primary": refine_trial.standard(trial, refine_trial.PRIMARY), "trial": trial,
            "metrica": {"means": metrica_result["candidate"]["means"], "matches": metrica_result["candidate"]["matches"]},
            "d24": {"mean": d24, "views": view_scores}}


def reference(dfl, skill, parsed, *, raw, output=None):
    """D43 from `data/d42/cache` must reproduce its development, Metrica and D24 records (cost 10)."""
    require_external_only()
    from dfl_submission import metrica_views as mv, player_trial, vendor_trial
    output = RECIPE / "reference.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D43's reference is recorded once")
    record = read(player_trial.RECIPE / "development.json")
    if record["models"] != player_trial.identity():
        raise ValueError("D43's networks differ from the ones D42's development scored")
    entry = next(e for e in record["entries"] if e["point"]["owner_cost"] == REFERENCE_COST)
    development, metrica, views = d43_halves(dfl, skill, parsed, raw=raw)
    _, _, truths = mv.reference_scores()
    scored = evaluate(development, metrica, views, truths, REFERENCE_COST)
    canonical = vendor_trial.canonical
    reproduction = {"development": canonical(scored["trial"]) == entry["trial"],
                    "metrica": canonical(scored["metrica"]["means"]) == entry["metrica"]["means"],
                    "d24": canonical(scored["d24"]["views"]) == entry["d24"]["views"]
                    and scored["d24"]["mean"] == entry["d24"]["candidate"]}
    passed = all(reproduction.values())
    write_new(output, {"protocol": PROTOCOL, "reference": "D43", "owner_cost": REFERENCE_COST,
                       "reproduction": reproduction, "passed": passed, "groups": scored["groups"],
                       "composite": scored["composite"], "primary": scored["primary"],
                       "models": player_trial.identity(), "halves": {"development": len(development),
                                                                    "metrica": len(metrica), "d24": len(views)},
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D47 reference: D43 {'reproduces' if passed else 'DOES NOT reproduce'} its records {reproduction}; "
          f"composite {scored['composite']:.5f}, primary {scored['primary']:.5f}", flush=True)
    return passed


# -- scoring -------------------------------------------------------------------------------------------

def no_harm(entries, base):
    """D47's screening rule: at the best cost by composite (ties to the lower cost), composite >= D43's and no group down > 0.01."""
    best = max(entries, key=lambda e: (e["composite"], -e["owner_cost"]))
    changes = {name: best["groups"][name] - base["groups"][name] for name in GROUPS}
    worst = min(changes, key=changes.get)
    passed = bool(best["composite"] >= base["composite"] and all(v >= -LOSS for v in changes.values()))
    return {"owner_cost": best["owner_cost"], "composite": best["composite"],
            "composite_change": best["composite"] - base["composite"], "group_changes": changes,
            "worst_group": worst, "worst_change": changes[worst], "passed": passed}


def candidate_record(candidate, entries, base, d43_trial, ident, checks, exports):
    from dfl_submission import refine_trial
    for entry in entries:
        entry["changes"] = refine_trial.changes(d43_trial, entry["trial"])
        entry["composite_change"] = entry["composite"] - base["composite"]
        entry["primary_change"] = entry["primary"] - base["primary"]
        entry["group_changes"] = {name: entry["groups"][name] - base["groups"][name] for name in GROUPS}
    screened = no_harm(entries, base)
    return {"protocol": PROTOCOL, "candidate": candidate, "lever": candidate in LEVERS,
            "costs": [entry["owner_cost"] for entry in entries], "entries": entries,
            "screen": screened if candidate in LEVERS else dict(screened, passed=None),
            "reference": {"name": "D43", "owner_cost": REFERENCE_COST, "groups": base["groups"],
                          "composite": base["composite"], "primary": base["primary"]},
            "identity": ident, "member_path_check": checks, "exports": exports,
            "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")}


def export_differences(candidate, scratch=SCRATCH / "networks"):
    result = {}
    for name in CANDIDATES[candidate]["slots"]:
        for key, folders in slot_folders(name, scratch).items():
            for folder, path in folders.items():
                fit = read(path / "fit.json")
                result[f"{name}/{key}/{folder}"] = fit["export_max_logit_difference"]
    return result


def score(candidates, dfl, skill, parsed, *, raw, cache_root=SCRATCH / "cache", scratch=SCRATCH / "networks"):
    require_external_only()
    from dfl_submission import diagnosis, metrica_sample as ms, metrica_views as mv, vendor_trial
    if read(RECIPE / "reference.json")["passed"] is not True:
        raise ValueError("D43 did not reproduce its records; nothing is scored")
    unknown = [c for c in candidates if c not in CANDIDATES]
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    for candidate in candidates:
        if (RECIPE / f"{candidate}.json").exists():
            raise FileExistsError(f"{RECIPE / candidate}.json exists; D47 records are written once")
    games = vendor_trial.development()
    d43_dev, d43_metrica, d43_views = d43_halves(dfl, skill, parsed, raw=raw)
    _, _, truths = mv.reference_scores()
    base = evaluate(d43_dev, d43_metrica, d43_views, truths, REFERENCE_COST)
    stored = read(RECIPE / "reference.json")
    if base["groups"] != stored["groups"] or base["composite"] != stored["composite"]:
        raise ValueError("D43's reference values differ from its reference record")
    idents = {c: identity(c, scratch) for c in candidates}
    variants = {c: variant_name(c, scratch) for c in candidates}
    checks = {}
    collected = {c: {} for c in candidates}

    def pending(group):
        return [c for c in candidates if not (cache_root / c / group / "manifest.json").exists()]

    # development
    waiting = pending("development")
    forecaster = Forecaster(waiting, scratch) if waiting else None
    for candidate in candidates:
        key = dict(idents[candidate], teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=raw,
                   skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games))
        forecast = forecaster.signals_of(candidate) if forecaster else None

        def produce(forecast=forecast):
            yield from diagnosis.external_halves(dfl, skill, "development", None, raw=raw, signals_of=forecast)
            yield from diagnosis.pff_halves(parsed, games, None, signals_of=forecast)

        collected[candidate]["development"] = diagnosis.collect(cache_root / candidate / "development", key, produce)
    checks["development"] = forecaster.check_d43(d43_dev) if forecaster else 0

    # Metrica sample games
    waiting = pending("metrica")
    forecaster = Forecaster(waiting, scratch) if waiting else None
    for candidate in candidates:
        forecast = forecaster.signals_of(candidate) if forecaster else (lambda *_: None)
        _, collected[candidate]["metrica"] = ms.reference_halves(
            "D15", cache=cache_root / candidate / "metrica", model="D47", signals_of=forecast, variant=variants[candidate])
    checks["metrica"] = forecaster.check_d43(d43_metrica) if forecaster else 0

    # D24 views
    waiting = pending("d24")
    forecaster = Forecaster(waiting, scratch) if waiting else None
    for candidate in candidates:
        forecast = forecaster.signals_of(candidate) if forecaster else (lambda *_: None)
        collected[candidate]["d24"] = mv.view_halves(cache=cache_root / candidate / "d24", signals_of=forecast,
                                                     variant=variants[candidate])
    checks["d24"] = forecaster.check_d43(d43_views) if forecaster else 0

    for candidate in candidates:
        halves = collected[candidate]
        entries = [evaluate(halves["development"], halves["metrica"], halves["d24"], truths, cost)
                   for cost in CANDIDATES[candidate]["costs"]]
        record = candidate_record(candidate, entries, base, base["trial"], idents[candidate], checks,
                                  export_differences(candidate, scratch))
        write_new(RECIPE / f"{candidate}.json", record)
        for entry in entries:
            print(f"D47 {candidate} cost {entry['owner_cost']}: composite {entry['composite']:.5f} "
                  f"({entry['composite_change']:+.5f}), primary {entry['primary']:.5f} ({entry['primary_change']:+.5f}), "
                  + ", ".join(f"{k} {v:+.4f}" for k, v in entry["group_changes"].items()), flush=True)
        s = record["screen"]
        print(f"D47 {candidate} screen: best cost {s['owner_cost']}, composite {s['composite_change']:+.5f}, worst "
              f"{s['worst_group']} {s['worst_change']:+.4f} -> "
              f"{'n/a' if s['passed'] is None else 'passes' if s['passed'] else 'fails'}", flush=True)


# -- summary ---------------------------------------------------------------------------------------------

SHORT = {"dfl_clean": "clean", "dfl_masked": "masked", "dfl_dead_carried": "dead carried", "dfl_team_swap": "swap",
         "skillcorner": "SkillCorner", "pff": "PFF", "metrica": "Metrica", "d24": "D24"}


def summary(folder=RECIPE):
    reference_record = read(folder / "reference.json")
    rows = [f"| Candidate | Cost | Composite | Δ | Primary | Δ | " + " | ".join(SHORT[g] for g in GROUPS) + " |",
            "|---|---:|---:|---:|---:|---:|" + "---:|" * len(GROUPS)]
    rows.append(f"| **D43** | {REFERENCE_COST} | {reference_record['composite']:.5f} | — | {reference_record['primary']:.5f} | — | "
                + " | ".join(f"{reference_record['groups'][g]:.4f}" for g in GROUPS) + " |")
    records = {}
    for name in CANDIDATES:
        path = folder / f"{name}.json"
        if not path.exists():
            continue
        records[name] = read(path)
        for entry in records[name]["entries"]:
            rows.append(f"| {name} | {entry['owner_cost']} | {entry['composite']:.5f} | {entry['composite_change']:+.4f} | "
                        f"{entry['primary']:.5f} | {entry['primary_change']:+.4f} | "
                        + " | ".join(f"{entry['groups'][g]:.4f}" for g in GROUPS) + " |")
    lines = ["\n".join(rows), ""]
    models = [("D43", reference_record)] + [(n, next(e for e in records[n]["entries"] if e["owner_cost"] == REFERENCE_COST))
                                            for n in ("S1000", "S2000") if n in records]
    if len(models) == 3:
        for metric in ("composite", "primary"):
            values = np.array([item[metric] for _, item in models])
            lines.append(f"Seed spread at cost {REFERENCE_COST}, {metric}: " + ", ".join(f"{n} {item[metric]:.5f}" for n, item in models)
                         + f"; mean {values.mean():.5f}, SD {values.std(ddof=1):.5f}, range {values.max() - values.min():.5f}")
    for name in LEVERS:
        if name in records:
            s = records[name]["screen"]
            lines.append(f"{name}: best cost {s['owner_cost']}, composite {s['composite_change']:+.5f}, worst group "
                         f"{s['worst_group']} {s['worst_change']:+.4f} -> {'passes' if s['passed'] else 'fails'} the no-harm rule")
    family = [n for n in MIRROR_FAMILY if n in records and records[n]["screen"]["passed"]]
    if family:
        chosen = max(family, key=lambda n: records[n]["screen"]["composite"])
        lines.append(f"Of F0, F and FN at most one is carried: {chosen} (highest composite among those passing)")
    text = "\n".join(lines)
    print(text)
    return text


# -- command line ------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("check-trainer")
    trainer.add_argument("--source", type=Path, required=True)
    trainer.add_argument("--tokens", type=Path, required=True)
    trainer.add_argument("--scratch", type=Path, default=SCRATCH / "check")
    mirror = commands.add_parser("parity")
    mirror.add_argument("--set", choices=SETS, required=True)
    mirror.add_argument("--source", type=Path, required=True)
    mirror.add_argument("--dfl", type=Path, required=True)
    mirror.add_argument("--dfl-raw", action="store_true")
    mirror.add_argument("--skill", type=Path, required=True)
    joined = commands.add_parser("parity-record")
    joined.add_argument("parts", type=Path, nargs="+")
    networks = commands.add_parser("networks")
    networks.add_argument("--candidate", choices=tuple(TRAINED), required=True)
    networks.add_argument("--set", choices=SETS, required=True)
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--tokens", type=Path, required=True)
    for name in ("reference", "score"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        if name == "score":
            command.add_argument("--candidates", nargs="+", required=True)
    commands.add_parser("summary")
    args = parser.parse_args()
    require_external_only()
    if args.command == "check-trainer":
        check_trainer(args.source, args.tokens, args.scratch)
    elif args.command == "parity":
        parity(args.set, args.source, args.dfl, args.skill, SCRATCH / f"parity_{args.set}.json",
               SCRATCH / f"pitches_{args.set}.json", raw=args.dfl_raw)
    elif args.command == "parity-record":
        parity_record(args.parts)
    elif args.command == "networks":
        train_networks(args.candidate, args.set, args.source, args.tokens)
    elif args.command == "reference":
        reference(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "score":
        score(args.candidates, args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    else:
        summary()


if __name__ == "__main__":
    main()
