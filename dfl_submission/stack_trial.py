"""D48: the stack (long-hidden views, attacking-frame tokens, three seeds), screened on development and confirmed once.

Commands (DECISIONS.md D48):
- `check --name direction|seed|hidden|mirror` trains one pipeline check with `stack_set` and compares every
  snapshot with the network it must reproduce bit for bit: D50's original-set seed-42 network (only
  `direction`, D42's data), D47's S1000 original-set seed-42 network (only seed offset 1000), D49's
  original-set seed-42 network (only D49's data), D47's F original-set seed-42 network (only the two-axis
  mirror, D42's data);
- `signs --set` re-reads one set's training halves, rebuilds D49's long-hidden views, checks every
  record's tokens against D49's stored tokens and writes every record's attacking sign and pitch size;
  `scoring-signs` writes the attacking sign of every half development scores (DFL WPY views,
  SkillCorner, PFF, Metrica, D24, D49's guard views), keyed by the hash of its tracking;
  `sign-record` joins them into the pipeline-check record (D50's amendment gives a half without a
  sign the other half's sign negated, or +1);
- `reference` checks that D43 reproduces its development, Metrica, D24 and guard-view records and
  records D43 at every cost;
- `levers` records, once, the lever list the orchestrator sends; `networks` trains and exports the
  stack's networks (three seed offsets x two sets x two networks), one at a time, only after D49's
  last training (`data/d49/TRAINING_DONE`) and while fewer than two trainings run machine-wide;
- `develop` forecasts the stack and each seed set alone (D43 recomputed on the way, bit for bit
  against its caches), and writes the development record with D48's selection score and gate;
- `confirm` (not to be run before the orchestrator says so) and `build` (the competition file behind
  its own command; `player_trial build` is unchanged).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from dfl_submission import direction_tokens as dt, stack_set as ss
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stack_recipe"
MODELS = PACKAGE / "stack_models"
SCRATCH = ROOT / "data" / "d48"
D42_CACHE = ROOT / "data" / "d42" / "cache"
D43_HOLDOUT_CACHE = ROOT / "data" / "d42" / "confirm" / "holdout"      # where D43's confirmation cached its inputs
D49_DONE = ROOT / "data" / "d49" / "TRAINING_DONE"
PROTOCOL = "DECISIONS.md D48"
SETS = ("current", "augmented")
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
OFFSETS = (0, 1000, 2000)
COSTS = (8, 10, 12, 14)
REFERENCE_COST = 10
SELECTION_GAIN, PFF_GAIN, LOSS = .015, .015, .01
CONFIRM_GAIN = .01
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica", "d24")
HOLDOUT_GUARD_SEED = 790000          # new fixed seeds of the holdout long-hidden views (D49's guard view (a))
TRAINING_LIMIT = 2
TRAINER = re.compile(r"-m dfl_submission\.(\w+\b.*\s--seed\s|[a-z_0-9]+ train\b|player_set3 )")
IN_PROCESS = re.compile(r"-m dfl_submission\.[a-z_0-9]+ networks\b")
CHECKS = {
    "direction": {"config": {"direction": True}, "data": "d42",
                  "expected": ROOT / "data" / "d50" / "networks" / "raw" / "current" / "temporal_models",
                  "signs": ROOT / "data" / "d50" / "signs_current.json"},
    "seed": {"config": {"seed_offset": 1000}, "data": "d42",
             "expected": ROOT / "data" / "d47" / "networks" / "S1000" / "raw" / "current" / "temporal_models"},
    "hidden": {"config": {}, "data": "d49", "expected": ROOT / "data" / "d49" / "networks" / "current" / "temporal_models"},
    "mirror": {"config": {"mirror": "xy"}, "data": "d42",
               "expected": ROOT / "data" / "d47" / "networks" / "F" / "raw" / "current" / "temporal_models"},
}
PARITY = PACKAGE / "screen_recipe" / "parity.json"
PIPELINE_CHECKS = ("check_direction.json", "check_seed.json", "check_hidden.json", "check_mirror.json")
MODEL_NAMES = ("stack", *(f"seed_{offset}" for offset in OFFSETS))


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D48 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def canonical_json(value):
    return json.loads(json.dumps(value))


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def data_folders(data, model_set, external):
    """(prepared sequences, tokens) of one set: D42's or D49's. `external` is the external data root."""
    if data == "d42":
        source = Path(external) / "external_temporal" / "train_data" if model_set == "current" else \
            ROOT / "data" / "d13" / "sequences_augmented"
        return source, ROOT / "data" / "d42" / f"tokens_{model_set}"
    if data == "d49":
        return ROOT / "data" / "d49" / f"sequences_{model_set}", ROOT / "data" / "d49" / f"tokens_{model_set}"
    raise ValueError(f"Unknown data folder {data}")


# -- the GPU share ---------------------------------------------------------------------------------

def trainings(rows, own=None):
    """Training jobs among (pid, ppid, command) rows: trainer processes, plus `networks` runs with no trainer child."""
    python = [(pid, ppid, command) for pid, ppid, command in rows if "ython" in command.split(" ", 1)[0]]
    trainers = {pid for pid, _, command in python if TRAINER.search(command)}
    parents = {ppid for pid, ppid, _ in python if pid in trainers}
    others = {pid for pid, _, command in python if IN_PROCESS.search(command) and pid not in parents and pid != own}
    return len(trainers) + len(others)


def running_trainings():
    output = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return trainings(rows, own=os.getpid())


def wait_for_gpu(what, *, limit=TRAINING_LIMIT, poll=30, done=D49_DONE):
    """The orchestrator's order: D49's last training has finished, and fewer than `limit` trainings run."""
    waited = False
    while not Path(done).exists() or (count := running_trainings()) >= limit:
        if not waited:
            reason = "D49 not done" if not Path(done).exists() else f"{count} trainings running"
            print(f"{time.strftime('%H:%M:%S')} waiting before {what}: {reason}", flush=True)
            waited = True
        time.sleep(poll)
    if waited:
        print(f"{time.strftime('%H:%M:%S')} starting {what}", flush=True)


def wait_for_scoring(what, *, limit=TRAINING_LIMIT, poll=30):
    """No scoring while two trainings run."""
    waited = False
    while running_trainings() >= limit:
        if not waited:
            print(f"{time.strftime('%H:%M:%S')} waiting before {what}: trainings running", flush=True)
            waited = True
        time.sleep(poll)


# -- pipeline checks -----------------------------------------------------------------------------------

FIT_KEYS = ("boundary_boost", "width", "steps", "lr_scale", "batch", "weight_decay", "dropout", "train_manifest",
            "token_manifest", "training_device", "torch")


def check(name, external, *, scratch=SCRATCH / "check", output=None):
    """One pipeline check: a `stack_set` run with one switch must hash as the network it reproduces."""
    require_external_only()
    spec = CHECKS[name]
    output = RECIPE / f"check_{name}.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; each D48 check runs once")
    source, tokens = data_folders(spec["data"], "current", external)
    expected_fit = read(spec["expected"] / "fit.json")
    if digest(source / "manifest.json") != expected_fit["train_manifest"] or \
            digest(tokens / "manifest.json") != expected_fit["token_manifest"]:
        raise ValueError(f"The {name} check must use the data its reference network trained on")
    config = ss.config_of(**spec["config"])
    run = scratch / name / "current" / "temporal_models"
    if not (run / "fit.json").exists():
        if run.exists():
            raise FileExistsError(f"{run} holds an unfinished run; remove it first")
        run.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "dfl_submission.stack_set", "train", "--source", str(source), "--tokens",
                   str(tokens), "--out", str(run), "--seed", "42", "--boundary-boost", "0.5", "--config", json.dumps(config)]
        if config["direction"]:
            command += ["--signs", str(spec["signs"])]
        if config["mirror"] is not None:
            command += ["--sign-map", str(PARITY)]
        wait_for_gpu(f"the {name} check")
        subprocess.run(command, check=True, cwd=ROOT)
    trained = read(run / "fit.json")
    differing = [key for key in (*FIT_KEYS, "seed") if trained[key] != expected_fit[key]]
    if differing:
        raise ValueError(f"The {name} check differs from its reference in {differing}")
    passed = trained["models"] == expected_fit["models"]
    write_new(output, {"protocol": PROTOCOL, "check": name, "config": config, "reference": str(spec["expected"].relative_to(ROOT)),
                       "expected": expected_fit["models"], "actual": trained["models"], "passed": passed,
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 {name} check: {'bit for bit' if passed else 'DIFFERS'} "
          f"({trained['models']['temporal_5400.pt'][:12]} against {expected_fit['models']['temporal_5400.pt'][:12]})",
          flush=True)
    return passed


# -- attacking signs -----------------------------------------------------------------------------------

def tracking_key(positions, ball, teams, pitch):
    from dfl_submission import direction_trial
    return direction_trial.tracking_key(positions, ball, teams, pitch)


def half_entry(positions, teams, frames=None):
    from dfl_submission import direction_trial
    return direction_trial.half_entry(positions, teams, frames)


def training_signs(model_set, external, dfl, skill, *, raw, output=None, pitch_output=None):
    """Every record of D49's folder for one set: tokens checked, sign (tracking, every visible player), pitch."""
    require_external_only()
    from dfl_submission import aligned_trial, external_benchmark as benchmark, hidden_trial as h49
    from dfl_submission import hidden_views as hv, player_set as ps, touch_trial
    output = SCRATCH / f"signs_{model_set}.json" if output is None else output
    pitch_output = SCRATCH / f"pitches_{model_set}.json" if pitch_output is None else pitch_output
    for path in (output, pitch_output):
        if path.exists():
            raise FileExistsError(f"{path} exists")
    stored_source, _ = data_folders("d42", model_set, external)
    source, tokens = data_folders("d49", model_set, external)
    prepared = read(h49.RECIPE / f"prepare_{model_set}.json")
    if (not prepared["passed"] or prepared["sequences_manifest"] != digest(source / "manifest.json")
            or prepared["tokens_manifest"] != digest(tokens / "manifest.json")):
        raise ValueError(f"D49's {model_set} folders are not the ones its preparation check passed")
    token_manifest = read(tokens / "manifest.json")
    choices = prepared["choices"]
    paths = h49.provider_paths(skill) if choices["skill_ball"] == "provider" else {}
    order = benchmark.SKILL_SPLITS["all"]
    records, pitches = {}, {}

    def enter(name, source_name, positions, ball, teams, pitch):
        frames = dt.long_context_frames(len(ball))
        token_path = tokens / name
        if digest(token_path) != token_manifest["records"][name]:
            raise ValueError(f"Token file hash differs: {token_path}")
        with np.load(token_path, allow_pickle=False) as item:
            stored = item["tokens"]
        recomputed = ps.tokens(positions, ball, teams, pitch, frames).astype(np.float16)
        entry = half_entry(positions, teams, frames)
        from_tokens = dt.attacking_from_tokens(stored, frames)
        entry |= {"source": source_name, "tokens_equal": bool(np.array_equal(recomputed, stored, equal_nan=True)),
                  "token": {key: from_tokens[key] for key in ("frames", "d", "sign", "first_sign", "agrees")}}
        records[name] = entry
        pitches[name] = [float(v) for v in np.asarray(pitch)]
        print(f"Signs {model_set} {name}: D {entry['d']}, sign {entry['sign']}, first {entry['first_sign']}, "
              f"tokens {'equal' if entry['tokens_equal'] else 'DIFFER'}", flush=True)

    halves = touch_trial.training_halves(model_set, dfl, skill, raw=raw)
    for (record, _), half in zip(aligned_trial.verified_records(stored_source), halves, strict=True):
        if record["file"] != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match the stored record {record['file']}")
        enter(record["file"], record["source"], half.positions, half.ball, half.teams, half.pitch)
        provider = None
        if half.source == "skillcorner" and model_set == "augmented" and choices["skill_ball"] == "provider":
            with np.load(paths[half.match, half.index], allow_pickle=False) as item:
                provider = item["ball"]
        made = hv.training_view(model_set, half, swaps=bool(choices["swaps"]), provider_ball=provider,
                                skill_order=order.index(int(half.match)) if half.source == "skillcorner" else None)
        if made is not None:
            view, _ = made
            enter(f"{view.key}.npz", view.source, view.positions, view.ball, view.teams, view.pitch)
    manifest = read(source / "manifest.json")
    if sorted(records) != sorted(record["file"] for record in manifest["records"]):
        raise ValueError("The rebuilt records are not exactly D49's records")
    output.parent.mkdir(parents=True, exist_ok=True)
    pitch_output.write_text(json.dumps(pitches, indent=2, sort_keys=True) + "\n")
    output.write_text(json.dumps({"protocol": PROTOCOL, "set": model_set, "base_manifest": digest(source / "manifest.json"),
                                  "token_manifest": digest(tokens / "manifest.json"), "records": records,
                                  "pitches_sha256": digest(pitch_output),
                                  "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py")},
                                 indent=2, sort_keys=True) + "\n")


def development_tracking(dfl, skill, parsed, *, raw):
    """(group, key, positions, ball, teams, pitch) of every half development scores, as the forecasts receive it."""
    from dfl_submission import augment, hidden_trial as h49, hidden_views as hv, metrica_sample as ms
    from dfl_submission import metrica_views as mv, pff, vendor_trial
    for half in augment.guard_halves(dfl, skill, "development", raw=raw):
        group = half.source if half.source != "dfl" else f"dfl_{half.view}"
        yield group, (half.source, half.match, half.view, half.index), half.positions, half.ball, half.teams, half.pitch
    for game in vendor_trial.development():
        for index, half in enumerate(pff.load(parsed, game)):
            yield ("pff", ("pff", str(game), "observed", index), *pff.tracking(half))
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            yield ("metrica", ("metrica", str(game), "observed", index), *ms.tracking(half))
            for view, (positions, ball) in mv.build_views(half, game, index).items():
                yield f"d24_{view}", ("metrica", str(game), view, index), positions, ball, half["teams"], half["pitch"]
    for (source, match, index), (_, positions, ball, teams, pitch) in sorted(h49.guard_tracking(dfl, raw=raw).items()):
        yield f"guard_{source}", (source, match, hv.VIEW, index), positions, ball, teams, pitch


def sign_table(items):
    """{tracking key: entry} with D50's sign and its amendment's fallback, from (group, key, tracking) items."""
    entries, groups, keys = {}, {}, {}
    for group, key, positions, ball, teams, pitch in items:
        entry = half_entry(positions, teams)
        entries[key], groups[key] = entry, group
        keys[key] = tracking_key(positions, ball, teams, pitch)
    signs, fallbacks = ss.with_fallback(entries, lambda key: (*key[:3], 1 - key[3]))
    table = {}
    for key, entry in entries.items():
        tracked = keys[key]
        row = dict(entry, half="/".join(map(str, key)), group=groups[key], sign_used=signs[key],
                   fallback=fallbacks.get(key))
        if tracked in table and table[tracked]["sign_used"] != row["sign_used"]:
            raise ValueError(f"Two halves with the same tracking take different signs: {table[tracked]['half']}, {row['half']}")
        table.setdefault(tracked, row)
    return table


def scoring_signs(dfl, skill, parsed, *, raw, output=None):
    require_external_only()
    output = SCRATCH / "sign_table_development.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists")
    table = sign_table(development_tracking(dfl, skill, parsed, raw=raw))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": PROTOCOL, "table": table,
                                  "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py")},
                                 indent=2, sort_keys=True) + "\n")
    print(f"D48 scoring signs: {len(table)} distinct halves, fallbacks "
          f"{[row['half'] for row in table.values() if row['fallback']]}", flush=True)


def summarise(entries, training):
    """The pipeline check over one group of halves: first-frame agreement (and, for training, the tokens)."""
    disagree = sorted(key for key, entry in entries.items() if entry["sign"] is not None and not entry["agrees"])
    unsigned = sorted(key for key, entry in entries.items() if entry["sign"] is None)
    magnitudes = [abs(entry["d"]) for entry in entries.values() if entry["d"] is not None]
    result = {"halves": len(entries), "first_frame_disagrees": disagree, "fallback_halves": unsigned,
              "smallest_abs_d_m": min(magnitudes) if magnitudes else None,
              "over_per_team": sorted(key for key, entry in entries.items() if max(entry["max_visible_all_frames"]) > 11)}
    passed = not disagree
    if training:
        unequal = sorted(key for key, entry in entries.items() if not entry["tokens_equal"])
        differs = sorted(key for key, entry in entries.items() if entry["token"]["sign"] != entry["sign"])
        result |= {"tokens_unequal": unequal, "token_sign_differs": differs}
        passed = passed and not unequal and not differs
    result["passed"] = bool(passed)
    return result


def sign_record(output=None):
    require_external_only()
    output = RECIPE / "signs.json" if output is None else output
    parts = {f"training_{model_set}": summarise(read(SCRATCH / f"signs_{model_set}.json")["records"], True)
             for model_set in SETS}
    table = read(SCRATCH / "sign_table_development.json")["table"]
    by_group = {}
    for row in table.values():
        by_group.setdefault(row["group"], {})[row["half"]] = row
    parts |= {group: summarise(rows, False) for group, rows in sorted(by_group.items())}
    passed = all(part["passed"] for part in parts.values())
    record = {"protocol": PROTOCOL, "check": "whole-half sign equals first-frame sign; D50's amendment for halves without one",
              "passed": passed, "parts": parts,
              "fallbacks": {name: part["fallback_halves"] for name, part in parts.items() if part["fallback_halves"]},
              "files": {name: digest(SCRATCH / f"signs_{name}.json") for name in SETS} | {
                  "development": digest(SCRATCH / "sign_table_development.json")},
              "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, record)
    print(f"D48 sign check: {'passed' if passed else 'FAILED'}; fallbacks {record['fallbacks']}", flush=True)
    for name, part in parts.items():
        print(f"  {name}: {part['halves']} halves, smallest |D| {part['smallest_abs_d_m']}, disagree "
              f"{part['first_frame_disagrees']}, fallback {part['fallback_halves']}", flush=True)
    return passed


def passed_record(name):
    path = RECIPE / name
    if not path.exists() or read(path)["passed"] is not True:
        raise ValueError(f"{path} is missing or did not pass")
    return read(path)


# -- D43's reference -------------------------------------------------------------------------------

def refuse():
    raise ValueError("A cache D48 only reads is missing")
    yield  # pragma: no cover


def d43_guards(raw):
    from dfl_submission import diagnosis, hidden_trial as h49, player_trial
    return diagnosis.collect(h49.DATA / "cache" / "guards_d43", h49.guard_identity(player_trial.identity(), raw=raw), refuse)


def reference(dfl, skill, parsed, *, raw, output=None):
    """D43 reproduces its development, Metrica, D24 and guard-view records; D43 at every cost, written once."""
    require_external_only()
    from dfl_submission import hidden_trial as h49, hyper_trial as ht, vendor_trial
    output = RECIPE / "reference.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D48's reference is written once")
    games = vendor_trial.development()
    development, metrica, d24, truths, checks = h49.d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=D42_CACHE)
    guards = d43_guards(raw)
    visibility = h49.pff_visibility(parsed, games)
    items = {cost: h49.entry(development, metrica, d24, guards, ht.recipe_at(cost), truths, visibility) for cost in COSTS}
    stored = read(h49.RECIPE / "reference.json")["reference"]
    reproduced = canonical_json(h49.compact(items[REFERENCE_COST])) == stored
    passed = bool(all(checks.values()) and reproduced)
    write_new(output, {"protocol": PROTOCOL, "passed": passed, "d42_development_row": checks,
                       "d49_reference_reproduced": reproduced, "entries": {str(c): h49.compact(i) for c, i in items.items()},
                       "selection": {str(c): selection(i) for c, i in items.items()},
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 reference: D43 {'reproduces' if passed else 'DOES NOT reproduce'} its records; selection score at "
          f"cost 10 {selection(items[REFERENCE_COST]):.5f}", flush=True)
    return passed


def selection(item):
    """D48's selection score: the mean of D47's composite and the long-hidden guard mean."""
    return (item["composite"] + item["guards"]["mean"]) / 2


# -- the lever list and the stack's networks -------------------------------------------------------

def record_levers(*, mirror=None, noise=False, aux=None, note="", output=None):
    """The orchestrator's final lever list, recorded once before any stack network is trained.

    L (D49's data) and C (`direction`) and the three seeds are always in; `mirror` ("y"), `noise` and
    `aux` (D51's target folder, replacing L's plain training by LB) only if the orchestrator says so.
    """
    require_external_only()
    output = RECIPE / "levers.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the lever list is recorded once")
    config = ss.config_of(direction=True, mirror=mirror, noise=noise, aux=aux is not None)
    write_new(output, {"protocol": PROTOCOL, "data": "d49", "config": config, "offsets": list(OFFSETS),
                       "aux_folder": None if aux is None else str(Path(aux).resolve().relative_to(ROOT)),
                       "note": note, "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 levers: {config}", flush=True)


def levers():
    record = read(RECIPE / "levers.json")
    ss.validate(record["config"])
    return record


def raw_folder(offset, model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / "raw" / str(offset) / model_set / folder


def export_folder(offset, model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / "export" / str(offset) / model_set / folder


def train_networks(external, *, sets=SETS, scratch=SCRATCH / "networks"):
    """The stack's networks of `sets` (offset, set, network), one at a time, each exported with the PyTorch check.

    Two processes with disjoint `sets` may run at once when the orchestrator allows two trainings.
    """
    require_external_only()
    for name in (*PIPELINE_CHECKS, "signs.json"):
        passed_record(name)
    chosen = levers()
    config = chosen["config"]
    signs_record = read(RECIPE / "signs.json")
    for offset in OFFSETS:
        for model_set in sets:
            source, tokens = data_folders(chosen["data"], model_set, external)
            signs_file = SCRATCH / f"signs_{model_set}.json"
            if digest(signs_file) != signs_record["files"][model_set]:
                raise ValueError(f"{signs_file} is not the file the sign check passed on")
            for folder, (seed, boost) in NETWORKS.items():
                output = export_folder(offset, model_set, folder, scratch)
                if (output / "fit.json").exists():
                    continue
                raw = raw_folder(offset, model_set, folder, scratch)
                if not (raw / "fit.json").exists():
                    if raw.exists():
                        raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    run_config = dict(config, seed_offset=offset)
                    command = [sys.executable, "-m", "dfl_submission.stack_set", "train", "--source", str(source),
                               "--tokens", str(tokens), "--out", str(raw), "--seed", str(seed), "--boundary-boost",
                               str(boost), "--config", json.dumps(run_config), "--signs", str(signs_file)]
                    if config["mirror"] is not None:
                        command += ["--sign-map", str(PARITY)]
                    if config["noise"]:
                        command += ["--pitches", str(SCRATCH / f"pitches_{model_set}.json")]
                    if config["aux"]:
                        command += ["--aux", str(ROOT / chosen["aux_folder"] / f"aux_{model_set}")]
                    wait_for_gpu(f"offset {offset} {model_set}/{folder}")
                    subprocess.run(command, check=True, cwd=ROOT)
                output.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                                "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- forecasts -----------------------------------------------------------------------------------------

class Network:
    """One exported player-set network with its switches (plain inference for D43)."""

    def __init__(self, folder, config, sign_map=None):
        from dfl_submission import player_trial
        self.checkpoint = player_trial.PlayerCheckpoint(folder)
        self.config, self.sign_map = config, sign_map
        self.key = str(folder)

    def predict(self, frames, owner, player_tokens, length, sign):
        columns = list(self.checkpoint.columns)
        values = owner[columns].to_numpy(dtype=np.float32)
        return ss.predict(self.checkpoint.checkpoint, frames, values, columns, player_tokens, length,
                          config=self.config, sign=sign if self.config["direction"] else None, sign_map=self.sign_map)


def stack_models(config, models_root, *, export=export_folder, sign_map=None, names=MODEL_NAMES):
    """{name: {role: {folder: [networks in seed order]}}} of the stack (three seeds per slot) and each seed set alone."""
    from dfl_submission import scale_trial
    loaded = {}

    def network(offset, key, folder):
        path = export(offset, key, folder, models_root)
        return loaded.setdefault(str(path), Network(path, config, sign_map))

    result = {}
    for name in names:
        offsets = OFFSETS if name == "stack" else (int(name.split("_")[1]),)
        result[name] = {role: {folder: [network(offset, key, folder) for offset in offsets] for folder in NETWORKS}
                        for role, key in scale_trial.ROLES.items()}
    return result


def d43_model():
    from dfl_submission import player_trial, scale_trial
    folders = player_trial.network_folders()
    return {role: {folder: [Network(folders[key] / folder, ss.DEFAULT)] for folder in NETWORKS}
            for role, key in scale_trial.ROLES.items()}


class Forecaster:
    """Decoder inputs of several models per half: features, tokens and trees computed once, network outputs memoised.

    Each half's attacking sign is looked up in a sign table by the hash of its tracking (every scored half
    must be there). If `with_d43`, D43's networks run too (plain inference) and are logged for the
    bit-for-bit check against its caches.
    """

    def __init__(self, models, table, *, owner_cost=None, with_d43=True):
        from dfl_submission import competition_adaptation, hyper_trial as ht, long_context, scale_trial
        reference_model, _, frozen = competition_adaptation.reference()
        self.previous = reference_model[2]["recipe"]
        self.frozen = frozen if owner_cost is None else dict(frozen, recipe=ht.recipe_at(owner_cost))
        if self.previous["variant"] != "temporal_1800.npz" or self.frozen["recipe"]["variant"] != "seed_phase":
            raise ValueError("D48 expects D15's variants")
        self.trees = {role: long_context.load(scale_trial.TREES[key])[0] for role, key in scale_trial.ROLES.items()}
        self.roles = scale_trial.ROLES
        self.models = dict(models)
        self.names = list(models)
        if with_d43:
            self.models["D43"] = d43_model()
        self.table = table
        self.memo, self.d43_log, self.signs = {}, [], []

    def sign_of(self, positions, ball, teams, pitch):
        row = self.table.get(tracking_key(positions, ball, teams, pitch))
        if row is None:
            raise ValueError("A scored half is not in the sign table; D48 stops")
        self.signs.append(row)
        return int(row["sign_used"])

    def forecast(self, positions, ball, teams, pitch):
        from dfl_submission import diagnosis, ensemble_trial, long_context, motion_trial, player_set as ps
        from dfl_submission import retrain_trial, temporal_trial
        sign = self.sign_of(positions, ball, teams, pitch)
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
                outputs[key] = net.predict(frames, owner, player_tokens, length, sign)
            return outputs[key]

        def slot(nets, index):
            return run(nets[0], index) if len(nets) == 1 else retrain_trial.mean([run(net, index) for net in nets])

        result = {}
        for name, roles in self.models.items():
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
            result[name] = (signals, labels)
        return result

    def signals_of(self, name):
        """A `signals_of(model, positions, ball, teams, pitch)` for one model, the others memoised by tracking."""
        def forecast(_, positions, ball, teams, pitch):
            key = tracking_key(positions, ball, teams, pitch)
            waiting = self.memo.get(key, {}).get(name)
            if not waiting:
                computed = self.forecast(positions, ball, teams, pitch)
                if "D43" in computed:
                    self.d43_log.append(computed.pop("D43"))
                for other, value in computed.items():
                    self.memo.setdefault(key, {}).setdefault(other, []).append(value)
                waiting = self.memo[key][name]
            value = waiting.pop(0)
            if not waiting:
                del self.memo[key][name]
                if not self.memo[key]:
                    del self.memo[key]
            return value
        return forecast

    def check_d43(self, halves):
        """D43 recomputed here against its cached decoder inputs, half by half, bit for bit; returns the count."""
        log, self.d43_log = self.d43_log, []
        if not log:
            return 0
        if len(log) != len(halves):
            raise ValueError(f"D43 was recomputed on {len(log)} halves, its cache holds {len(halves)}")
        for (signals, labels), half in zip(log, halves, strict=True):
            if not (all(np.array_equal(a, b) for a, b in zip(signals, half.signals, strict=True))
                    and np.array_equal(labels, half.d3)):
                raise ValueError(f"D43 recomputed differs from its cache on {half.key} half {half.index}")
        return len(log)


def table_of(path):
    return read(path)["table"]


def model_identity(name, config, scratch=SCRATCH / "networks", *, export=export_folder, levers_name="levers.json",
                   extra=None):
    from dfl_submission import scale_trial
    offsets = OFFSETS if name == "stack" else (int(name.split("_")[1]),)
    result = {"protocol": PROTOCOL, "model": name, "config": config,
              "networks": {str(o): {key: {folder: digest(export(o, key, folder, scratch) / "fit.json")
                                          for folder in NETWORKS} for key in SETS} for o in offsets},
              "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in SETS},
              "stack_set_sha256": digest(PACKAGE / "stack_set.py"), "player_set_sha256": digest(PACKAGE / "player_set.py"),
              "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py"),
              "signs_sha256": digest(RECIPE / "signs.json"), "levers_sha256": digest(RECIPE / levers_name)}
    return result if extra is None else dict(result, **extra)


# -- development -----------------------------------------------------------------------------------

def gate(entries, base):
    """D48's development gate: the cost with the highest selection score (ties to the lower cost) against D43 at cost 10."""
    from dfl_submission import refine_trial
    chosen = max(entries, key=lambda item: (selection(item), -item["owner_cost"]))
    groups = {name: chosen["groups"][name] - base["groups"][name] for name in GROUPS}
    conventions = refine_trial.changes(base["trial"], chosen["trial"])["conventions"]
    changes = {"selection": selection(chosen) - selection(base), "pff": groups["pff"],
               "composite": chosen["composite"] - base["composite"], "groups": groups, "conventions": conventions,
               "guard_mean": chosen["guards"]["mean"] - base["guards"]["mean"]}
    conditions = {"selection": bool(changes["selection"] >= SELECTION_GAIN), "pff": bool(changes["pff"] >= PFF_GAIN),
                  "composite": bool(changes["composite"] >= 0),
                  "groups": bool(all(value >= -LOSS for value in groups.values())),
                  "conventions": bool(all(value >= -LOSS for value in conventions.values()))}
    return {"selected_cost": chosen["owner_cost"], "selection_score": selection(chosen), "changes": changes,
            "conditions": conditions, "passed": bool(all(conditions.values()))}


def develop(dfl, skill, parsed, *, raw, cache_root=SCRATCH / "cache", scratch=SCRATCH / "networks", output=None,
            variant=None):
    """The stack and each seed set alone on development at every cost, with D48's gate; written once.

    `variant` (None for the 5,400-step stack) is {"levers": lever file, "export": export-folder function,
    "required": extra records that must have passed, "extra": extra identity keys}: the 8,100-step stack.
    """
    require_external_only()
    from dfl_submission import diagnosis, hidden_trial as h49, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import metrica_views as mv, vendor_trial
    output = RECIPE / "development.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D48 development is scored once")
    export = export_folder if variant is None else variant["export"]
    levers_name = "levers.json" if variant is None else variant["levers"]
    for name in (*PIPELINE_CHECKS, "signs.json", "reference.json", *(() if variant is None else variant["required"])):
        passed_record(name)
    chosen = levers() if variant is None else read(RECIPE / levers_name)
    config = chosen["config"]
    ss.validate(config)

    def model_identity(name, config, scratch):
        return globals()["model_identity"](name, config, scratch, export=export, levers_name=levers_name,
                                           extra=None if variant is None else variant["extra"])

    for offset in OFFSETS:
        for key in SETS:
            for folder in NETWORKS:
                if not (export(offset, key, folder, scratch) / "fit.json").exists():
                    raise ValueError(f"Stack network {offset}/{key}/{folder} is not exported yet")
    wait_for_scoring("development scoring")
    games = vendor_trial.development()
    d43_dev, d43_metrica, d43_d24, truths, _ = h49.d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=D42_CACHE)
    d43_guard = d43_guards(raw)
    sign_map = read(PARITY)["sign_map"] if config["mirror"] is not None else None
    forecaster = Forecaster(stack_models(config, scratch, export=export, sign_map=sign_map),
                            table_of(SCRATCH / "sign_table_development.json"))
    tracking = h49.guard_tracking(dfl, raw=raw)
    collected, checks = {name: {} for name in MODEL_NAMES}, {}

    def path_check(part, halves):
        stored = cache_root / f"{part}_d43_path_check.json"
        count = forecaster.check_d43(halves)
        manifests = {name: digest(cache_root / name / part / "manifest.json") for name in MODEL_NAMES}
        if count:
            stored.write_text(json.dumps({"halves": count, "caches": manifests}) + "\n")
        elif not stored.exists() or read(stored)["caches"] != manifests:
            raise ValueError(f"The {part} caches have no D43 path check; remove them and forecast again")
        checks[part] = read(stored)["halves"]

    for name in MODEL_NAMES:
        identity = model_identity(name, config, scratch)
        key = dict(identity, teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=raw,
                   skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games))
        forecast = forecaster.signals_of(name)

        def produce(forecast=forecast):
            yield from diagnosis.external_halves(dfl, skill, "development", None, raw=raw, signals_of=forecast)
            yield from diagnosis.pff_halves(parsed, games, None, signals_of=forecast)

        collected[name]["development"] = diagnosis.collect(cache_root / name / "development", key, produce)
    path_check("development", d43_dev)
    for name in MODEL_NAMES:
        tag = f"D48 {name} {json_digest(model_identity(name, config, scratch))[:16]}"
        _, collected[name]["metrica"] = ms.reference_halves("D15", cache=cache_root / name / "metrica", model="D48",
                                                            signals_of=forecaster.signals_of(name), variant=tag)
    path_check("metrica", d43_metrica)
    for name in MODEL_NAMES:
        tag = f"D48 {name} {json_digest(model_identity(name, config, scratch))[:16]}"
        collected[name]["d24"] = mv.view_halves(cache=cache_root / name / "d24", signals_of=forecaster.signals_of(name),
                                                variant=tag)
    path_check("d24", d43_d24)
    for name in MODEL_NAMES:
        forecast = forecaster.signals_of(name)
        guard_key = dict(model_identity(name, config, scratch), guard=h49.guard_identity({"model": name}, raw=raw))

        def produce_guards(forecast=forecast):
            from dfl_submission import hidden_views as hv
            for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(tracking.items()):
                signals, decoded = forecast(None, positions, ball, teams, pitch)
                yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)

        collected[name]["guards"] = diagnosis.collect(cache_root / name / "guards", guard_key, produce_guards)
    path_check("guards", d43_guard)
    print(f"D48 D43 recomputed bit for bit on {checks} halves", flush=True)

    visibility = h49.pff_visibility(parsed, games)
    base = read(RECIPE / "reference.json")
    base_entry = h49.entry(d43_dev, d43_metrica, d43_d24, d43_guard, ht.recipe_at(REFERENCE_COST), truths, visibility)
    if canonical_json(h49.compact(base_entry)) != base["entries"][str(REFERENCE_COST)]:
        raise ValueError("D43 differs from D48's reference record")
    entries = {}
    for name in MODEL_NAMES:
        halves = collected[name]
        entries[name] = [h49.entry(halves["development"], halves["metrica"], halves["d24"], halves["guards"],
                                   ht.recipe_at(cost), truths, visibility) for cost in COSTS]
        for item in entries[name]:
            print(f"D48 {name} cost {item['owner_cost']}: selection {selection(item):.5f} "
                  f"({selection(item) - selection(base_entry):+.4f}), composite {item['composite']:.5f}, guards "
                  f"{item['guards']['mean']:.4f}, PFF {item['groups']['pff']:.4f}", flush=True)
    decision = gate(entries["stack"], base_entry)
    spread = {name: gate(entries[name], base_entry) for name in MODEL_NAMES if name != "stack"}
    extra_record = {} if variant is None else {"variant": variant["name"], "amendment_choice": variant["choice"](decision)}
    write_new(output, {**extra_record, "protocol": PROTOCOL, "levers": chosen, "costs": list(COSTS),
                       "reference": {"name": "D43", "owner_cost": REFERENCE_COST, "selection": selection(base_entry),
                                     **h49.compact(base_entry)},
                       "entries": {name: [dict(h49.compact(item), selection=selection(item)) for item in items]
                                   for name, items in entries.items()},
                       "decision": decision, "seed_sets_alone": spread, "d43_path_bit_for_bit": checks,
                       "signs_used": {"halves": len(forecaster.signs),
                                      "fallbacks": sorted({row["half"] for row in forecaster.signs if row["fallback"]})},
                       "identity": {name: model_identity(name, config, scratch) for name in MODEL_NAMES},
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                                  "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": raw},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 development: cost {decision['selected_cost']}, selection {decision['changes']['selection']:+.4f}, PFF "
          f"{decision['changes']['pff']:+.4f}, composite {decision['changes']['composite']:+.4f} -> "
          f"{'the confirmation is spent' if decision['passed'] else 'gate not passed'} {decision['conditions']}", flush=True)
    return decision


# -- D52's T: the stack retrained at 8,100 steps (D48 amendment, 2026-11-13 03:54 EST) ------------------

T_STEPS = 8100
T_SCRATCH = SCRATCH / "t"
D52_T = ROOT / "data" / "d52" / "networks" / "T"
T_LEVERS = "levers_t.json"
T_OFFSETS = (1000, 2000)          # trained here; offset 0 is D52's T seed set, reused by digest


def train_steps(argv=None):
    """`stack_set`'s training call with a `steps` setting (the call D52's `length_trial train` makes at six blocks).

    `stack_set.py` stays byte-identical (its digest is part of the 5,400-step stack's identity).
    """
    from experiments.dfl_transfer_data import digest as file_digest
    parser = argparse.ArgumentParser(prog="stack_trial train-steps")
    for name in ("--source", "--tokens", "--out"):
        parser.add_argument(name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--boundary-boost", type=float, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--signs", type=Path)
    parser.add_argument("--sign-map", type=Path)
    parser.add_argument("--pitches", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)
    require_external_only()
    config = json.loads(args.config)
    ss.validate(config)
    if config["aux"]:
        raise ValueError("The 8,100-step stack trains no auxiliary head")
    signs = ss.read_signs(args.signs, args.source, args.tokens) if config["direction"] else None
    sign_map = None
    if config["mirror"] is not None:
        record = read(args.sign_map)
        if not record["passed"]:
            raise ValueError("The mirror parity check did not pass; nothing is trained with mirrors")
        sign_map = record["sign_map"]
    provenance = {"signs_file": None if args.signs is None else file_digest(args.signs)}
    ss.train(args.source, args.tokens, args.out, args.device, seed=args.seed, boundary_boost=args.boundary_boost,
             config=config, signs=signs, sign_map=sign_map, pitch_file=args.pitches, aux_folder=None,
             provenance=provenance, steps=args.steps)


def steps_command(source, tokens, out, seed, boost, config, model_set):
    command = [sys.executable, "-m", "dfl_submission.stack_trial", "train-steps", "--source", str(source), "--tokens",
               str(tokens), "--out", str(out), "--seed", str(seed), "--boundary-boost", str(boost), "--config",
               json.dumps(config), "--signs", str(SCRATCH / f"signs_{model_set}.json"), "--steps", str(T_STEPS)]
    if config["mirror"] is not None:
        command += ["--sign-map", str(PARITY)]
    if config["noise"]:
        command += ["--pitches", str(SCRATCH / f"pitches_{model_set}.json")]
    return command


def check_t(external, *, output=None):
    """The stack's trainer at 8,100 steps, offset 0, must reproduce D52's T original-set seed-42 network bit for bit."""
    require_external_only()
    output = RECIPE / "check_t.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the 8,100-step check runs once")
    base = levers()
    config = dict(base["config"], seed_offset=0)
    source, tokens = data_folders(base["data"], "current", external)
    expected_fit = read(D52_T / "raw" / "current" / "temporal_models" / "fit.json")
    if expected_fit["steps"] != T_STEPS or digest(source / "manifest.json") != expected_fit["train_manifest"] or \
            digest(tokens / "manifest.json") != expected_fit["token_manifest"]:
        raise ValueError("D52's T network was not trained at 8,100 steps on these folders")
    run = T_SCRATCH / "check" / "current" / "temporal_models"
    if not (run / "fit.json").exists():
        if run.exists():
            raise FileExistsError(f"{run} holds an unfinished run; remove it first")
        run.parent.mkdir(parents=True, exist_ok=True)
        wait_for_gpu("the 8,100-step check")
        subprocess.run(steps_command(source, tokens, run, 42, .5, config, "current"), check=True, cwd=ROOT)
    trained = read(run / "fit.json")
    differing = [key for key in (*FIT_KEYS, "seed") if trained[key] != expected_fit[key]]
    if differing:
        raise ValueError(f"The 8,100-step check differs from D52's T network in {differing}")
    passed = trained["models"] == expected_fit["models"]
    write_new(output, {"protocol": PROTOCOL, "check": "8,100 steps, offset 0: D52's T original-set seed-42 network",
                       "config": config, "steps": T_STEPS, "reference": str((D52_T / "raw" / "current" / "temporal_models")
                                                                            .relative_to(ROOT)),
                       "expected": expected_fit["models"], "actual": trained["models"], "passed": passed,
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 8,100-step check: {'bit for bit' if passed else 'DIFFERS'} "
          f"({trained['models']['temporal_8100.pt'][:12]} against {expected_fit['models']['temporal_8100.pt'][:12]})",
          flush=True)
    return passed


def t_export(offset, model_set, folder, scratch=T_SCRATCH / "networks"):
    """Offset 0: D52's T networks (reused by digest); offsets 1,000 and 2,000: trained here."""
    if offset == 0:
        return D52_T / "export" / model_set / folder
    return scratch / "export" / str(offset) / model_set / folder


def t_raw(offset, model_set, folder, scratch=T_SCRATCH / "networks"):
    if offset == 0:
        return D52_T / "raw" / model_set / folder
    return scratch / "raw" / str(offset) / model_set / folder


def record_levers_t(external, *, output=None):
    """The 8,100-step stack's lever record, once: D48's levers and offsets, steps 8,100, offset 0 = D52's T by digest."""
    require_external_only()
    output = RECIPE / T_LEVERS if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the 8,100-step lever record is written once")
    passed_record("check_t.json")
    base = levers()
    d52 = read(RECIPE / "d52_T.json")["identity"]
    reuse = {}
    for model_set in SETS:
        source, tokens = data_folders(base["data"], model_set, external)
        for folder, (seed, boost) in NETWORKS.items():
            raw_fit = read(t_raw(0, model_set, folder) / "fit.json")
            export_fit = t_export(0, model_set, folder) / "fit.json"
            stack = raw_fit["stack"]
            if (raw_fit["steps"] != T_STEPS or raw_fit["seed"] != seed or raw_fit["boundary_boost"] != boost
                    or {k: stack[k] for k in ss.DEFAULT} != dict(base["config"], seed_offset=0)
                    or raw_fit["train_manifest"] != digest(source / "manifest.json")
                    or raw_fit["token_manifest"] != digest(tokens / "manifest.json")
                    or digest(export_fit) != d52["networks"][model_set][folder]):
                raise ValueError(f"D52's T network {model_set}/{folder} is not the stack's recipe at 8,100 steps")
            reuse.setdefault(model_set, {})[folder] = {"export": str(export_fit.parent.relative_to(ROOT)),
                                                       "export_fit_sha256": digest(export_fit),
                                                       "raw_fit_sha256": digest(t_raw(0, model_set, folder) / "fit.json")}
    write_new(output, {"protocol": PROTOCOL, "variant": "t", "data": base["data"], "config": base["config"],
                       "offsets": list(OFFSETS), "trained_offsets": list(T_OFFSETS), "steps": T_STEPS,
                       "reuse": {"0": reuse}, "base_levers_sha256": digest(RECIPE / "levers.json"),
                       "check_t_sha256": digest(RECIPE / "check_t.json"), "d52_T_sha256": digest(RECIPE / "d52_T.json"),
                       "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 8,100-step levers recorded; offset 0 reuses D52's T networks", flush=True)


def train_networks_t(external, *, sets=SETS, scratch=T_SCRATCH / "networks"):
    """The 8,100-step stack's networks for offsets 1,000 and 2,000 (D48's levers), one at a time per process."""
    require_external_only()
    for name in (*PIPELINE_CHECKS, "signs.json"):
        passed_record(name)
    base = levers()
    signs_record = read(RECIPE / "signs.json")
    for offset in T_OFFSETS:
        for model_set in sets:
            source, tokens = data_folders(base["data"], model_set, external)
            if digest(SCRATCH / f"signs_{model_set}.json") != signs_record["files"][model_set]:
                raise ValueError(f"The {model_set} sign file is not the one the sign check passed on")
            for folder, (seed, boost) in NETWORKS.items():
                output = t_export(offset, model_set, folder, scratch)
                if (output / "fit.json").exists():
                    continue
                raw = t_raw(offset, model_set, folder, scratch)
                if not (raw / "fit.json").exists():
                    if raw.exists():
                        raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    wait_for_gpu(f"8,100 steps, offset {offset} {model_set}/{folder}")
                    subprocess.run(steps_command(source, tokens, raw, seed, boost, dict(base["config"], seed_offset=offset),
                                                 model_set), check=True, cwd=ROOT)
                output.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                                "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


def t_choice(decision):
    """The amendment's choice: the 8,100-step stack if it passes the gate and its selection score at its selected cost
    is at least the 5,400-step stack's; otherwise the 5,400-step stack at cost 14 (which passed)."""
    base = read(RECIPE / "development.json")["decision"]
    if base["passed"] is not True or base["selected_cost"] != 14:
        raise ValueError("The 5,400-step stack's development decision is not the one the amendment names")
    at_least = decision["selection_score"] >= base["selection_score"]
    chosen = bool(decision["passed"] and at_least)
    return {"stack": "8100" if chosen else "5400", "owner_cost": decision["selected_cost"] if chosen else 14,
            "selection_8100": decision["selection_score"], "selection_5400": base["selection_score"],
            "selection_at_least_5400": bool(at_least), "selection_at_least_0.7265": bool(decision["selection_score"] >= .7265),
            "gate_8100_passed": decision["passed"]}


def t_variant():
    reuse = read(RECIPE / T_LEVERS)["reuse"]["0"]
    for model_set in SETS:
        for folder in NETWORKS:
            if digest(t_export(0, model_set, folder) / "fit.json") != reuse[model_set][folder]["export_fit_sha256"]:
                raise ValueError(f"D52's T network {model_set}/{folder} differs from the one recorded for reuse")
    return {"name": "t", "levers": T_LEVERS, "export": t_export, "required": ("check_t.json",),
            "extra": {"steps": T_STEPS, "variant": "t"}, "choice": t_choice}


def develop_t(dfl, skill, parsed, *, raw):
    """The 8,100-step stack on development exactly as the 5,400-step stack was; `development_t.json`, written once."""
    return develop(dfl, skill, parsed, raw=raw, cache_root=T_SCRATCH / "cache", scratch=T_SCRATCH / "networks",
                   output=RECIPE / "development_t.json", variant=t_variant())


def summary_t(folder=RECIPE):
    """D43, the 5,400-step stack at cost 14, the 8,100-step stack and its seed sets; its gate and the amendment's choice."""
    reference_record = read(folder / "reference.json")
    first = read(folder / "development.json")
    development = read(folder / "development_t.json")
    base = development["reference"]
    base_conventions = conventions_of(base)
    rows = ["| Model | Cost | Composite | Guard mean (DFL / Metrica) | Selection | PFF | Primary | "
            + " | ".join(SHORT[g] for g in GROUPS) + " | N 1.75 / 4 / 4.5 vs D43@10 |",
            "|---|---:|---:|---|---:|---:|---:|" + "---:|" * len(GROUPS) + "---|"]

    def row(name, item):
        conventions = conventions_of(item)
        changes = " / ".join(f"{conventions[n] - base_conventions[n]:+.4f}" for n in ("aba1.75", "aba4", "aba4.5"))
        guards = item["guards"]
        return (f"| {name} | {item['owner_cost']} | {item['composite']:.4f} | {guards['mean']:.4f} "
                f"({guards['a_dfl']:.3f} / {guards['b_metrica']:.3f}) | {selection(item):.4f} | {item['groups']['pff']:.4f} | "
                f"{conventions['aba3']:.4f} | " + " | ".join(f"{item['groups'][g]:.3f}" for g in GROUPS) + f" | {changes} |")

    for cost in COSTS:
        rows.append(row("D43", reference_record["entries"][str(cost)]))
    rows.append(row("stack 5,400", next(item for item in first["entries"]["stack"] if item["owner_cost"] == 14)))
    for name in MODEL_NAMES[1:] + MODEL_NAMES[:1]:
        for item in development["entries"][name]:
            rows.append(row(f"{name} 8,100", item))
    decision, choice = development["decision"], development["amendment_choice"]
    changes = decision["changes"]
    lines = ["\n".join(rows), "",
             f"8,100-step stack: selected cost {decision['selected_cost']} (selection {decision['selection_score']:.5f} "
             f"against D43's {base['selection']:.5f}):",
             f"- selection {changes['selection']:+.5f} (needs +{SELECTION_GAIN}): {decision['conditions']['selection']}",
             f"- PFF {changes['pff']:+.5f} (needs +{PFF_GAIN}): {decision['conditions']['pff']}",
             f"- composite {changes['composite']:+.5f} (must not fall): {decision['conditions']['composite']}",
             f"- worst group {min(changes['groups'], key=changes['groups'].get)} {min(changes['groups'].values()):+.4f} "
             f"(each >= -{LOSS}): {decision['conditions']['groups']}",
             "- conventions " + ", ".join(f"{k} {v:+.4f}" for k, v in changes["conventions"].items())
             + f" (each >= -{LOSS}): {decision['conditions']['conventions']}",
             f"Gate {'passed' if decision['passed'] else 'NOT passed'}; selection {choice['selection_8100']:.5f} against "
             f"the 5,400-step stack's {choice['selection_5400']:.5f} -> confirmation goes to the {choice['stack']}-step "
             f"stack at cost {choice['owner_cost']}"]
    text = "\n".join(lines)
    print(text)
    return text


# -- confirmation (not run before the orchestrator says so) ------------------------------------------

def confirmation_stack():
    """The stack D48's amendment (03:54 EST) sends to confirmation, with its cost, folders and lever record."""
    first = read(RECIPE / "development.json")["decision"]
    default = {"stack": "5400", "cost": first["selected_cost"], "export": export_folder, "scratch": SCRATCH / "networks",
               "levers": "levers.json", "extra": None, "passed": first["passed"]}
    path = RECIPE / "development_t.json"
    if path.exists():
        choice = read(path)["amendment_choice"]
        if choice["stack"] == "8100":
            return {"stack": "8100", "cost": choice["owner_cost"], "export": t_export, "scratch": T_SCRATCH / "networks",
                    "levers": T_LEVERS, "extra": t_variant()["extra"], "passed": choice["gate_8100_passed"]}
        default["cost"] = choice["owner_cost"]
    if not default["passed"]:
        raise ValueError("The stack's development gate did not pass; the confirmation is not spent")
    return default


def holdout_seed(match, index):
    from dfl_submission import hidden_views as hv
    return HOLDOUT_GUARD_SEED + 100 * hv.DFL_ORDER.index(match) + index


def holdout_long_hidden(dfl, *, raw):
    """{(source, match, index): (labels, positions, ball, teams, pitch)}: D49's guard view (a) on the DFL holdout, new seeds."""
    from dfl_submission import external_benchmark as benchmark, hidden_views as hv
    result = {}
    for half in benchmark.halves(dfl, None, "holdout", raw=raw):
        if half.source == "dfl" and half.view == "clean":
            positions, ball = hv.guard_view(half.positions, half.ball, holdout_seed(half.match, half.index))
            result["dfl", half.match, half.index] = (np.asarray(half.labels), positions, ball, half.teams, half.pitch)
    return result


def long_hidden_scores(halves, recipe):
    """N = 3 spellF1 per DFL holdout match on its long-hidden view, and their mean."""
    from dfl_submission import diagnosis
    scored = diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe))
    matches = {key[1]: float(diagnosis.score(diagnosis.target(truth, 3.0), prediction)["spell_f1"])
               for key, (truth, prediction, _) in scored.items()}
    return {"matches": matches, "mean": float(np.mean(list(matches.values())))}


CONFIRM_CACHE = SCRATCH / "confirm"


def confirm_keys(target, config, parsed, games_u, skill, raw):
    """The cache identities of the confirmation's decoder inputs (D43 and the stack)."""
    from dfl_submission import diagnosis, player_trial, vendor_trial
    stack_identity = model_identity("stack", config, target["scratch"], export=target["export"],
                                    levers_name=target["levers"], extra=target["extra"])
    common = {"teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games_u),
              "long_hidden_seed": HOLDOUT_GUARD_SEED, "hidden_views_sha256": digest(PACKAGE / "hidden_views.py"),
              "dfl_raw": raw}
    d43 = dict(common, protocol=PROTOCOL, models=player_trial.identity())
    stack = dict(stack_identity, **common, skill_manifest_sha256=digest(skill / "manifest.json"))
    return d43, stack, stack_identity


def code_identity():
    names = ("stack_trial.py", "stack_set.py", "direction_tokens.py", "direction_trial.py", "player_set.py",
             "player_set3.py", "hidden_views.py", "hidden_trial.py", "temporal_cpu.py", "player_trial.py")
    return {name: digest(PACKAGE / name) for name in names}


def confirm_prepare(dfl, skill, parsed, pff_raw, *, raw, cache_root=CONFIRM_CACHE):
    """Everything before the confirmation's scores: D43 reproduces its holdout record, the untouched PFF set is
    parsed (only now), the holdout long-hidden views are built, every half gets its attacking sign, and the decoder
    inputs of D43 and the stack are cached. Nothing is scored here; `prepared.json` records the checks."""
    require_external_only()
    from dfl_submission import aligned_trial, augment, diagnosis, hidden_views as hv, hyper_trial as ht, player_trial
    from dfl_submission import pff, vendor_trial
    if (RECIPE / "confirmation.json").exists():
        raise FileExistsError("D48's confirmation is written; nothing is prepared again")
    prepared_path = cache_root / "prepared.json"
    target = confirmation_stack()
    if target["stack"] != "8100" or target["cost"] != 14:
        raise ValueError(f"The amendment's choice is {target['stack']} at cost {target['cost']}, not the 8,100-step stack at 14")
    chosen = read(RECIPE / target["levers"])
    config = chosen["config"]
    ss.validate(config)
    # 1. D43 reproduces its holdout record (reused holdout and PFF set C) from its confirmation cache.
    games_c = vendor_trial.split_games("confirm_c")
    stored = read(player_trial.RECIPE / "confirmation.json")
    d43_reused = player_trial.candidate_halves(dfl, skill, parsed, games_c, "holdout", raw=raw, cache=D43_HOLDOUT_CACHE)
    reproduced = canonical_json(aligned_trial.evaluate(d43_reused, ht.recipe_at(REFERENCE_COST))) == stored["trial"]
    print(f"D48 confirmation: D43 {'reproduces' if reproduced else 'DOES NOT reproduce'} its holdout record", flush=True)
    if not reproduced:
        raise ValueError("D43 does not reproduce its holdout record; nothing of the stack is scored")
    # 2. The untouched PFF set, parsed only now (as `player_trial confirm`'s set C was: `pff.prepare` with its games).
    games_u = sorted(vendor_trial.split_games("untouched"))
    manifest = read(parsed / "manifest.json")["parsed"]
    names = [f"{game}_{index}.npz" for game in games_u for index in (1, 2)]
    before = {"in_manifest": [name for name in names if name in manifest],
              "files": [name for name in names if (parsed / name).exists()]}
    if prepared_path.exists():
        parse_note = read(prepared_path)["untouched_parse"]
    else:
        if before["in_manifest"] or before["files"]:
            raise ValueError(f"The untouched PFF set was parsed before: {before}")
        pff.prepare(pff_raw, parsed, ids=games_u)
        parse_note = {"games": games_u, "parsed_before": False, "checked_before_parsing": before,
                      "parsed_at": time.strftime("%Y-%m-%d %H:%M %Z"), "inputs": vendor_trial.inputs(parsed, games_u)}
    if vendor_trial.inputs(parsed, games_u) != parse_note["inputs"]:
        raise ValueError("The parsed untouched PFF files differ from the ones parsed for this step")
    # 3. The holdout long-hidden views (new fixed seeds) and their hidden-run statistics.
    long_hidden = holdout_long_hidden(dfl, raw=raw)
    if sorted(long_hidden) != [("dfl", match, index) for match in ("WQQ", "WR9") for index in (0, 1)]:
        raise ValueError(f"Unexpected holdout long-hidden halves {sorted(long_hidden)}")
    views = {"seeds": {f"{match}/{index}": holdout_seed(match, index) for _, match, index in sorted(long_hidden)},
             "statistics": hv.view_statistics([value[2] for value in long_hidden.values()]),
             "per_match": {match: hv.view_statistics([value[2] for key, value in long_hidden.items() if key[1] == match])
                           for match in ("WQQ", "WR9")}}

    # 4. Attacking signs of every half the stack forecasts (D50's rule, its amendment's fallback; stop on disagreement).
    def items():
        for half in augment.guard_halves(dfl, skill, "holdout", raw=raw):
            yield "holdout", (half.source, half.match, half.view, half.index), half.positions, half.ball, half.teams, half.pitch
        for game in games_u:
            for index, half in enumerate(pff.load(parsed, game)):
                yield ("pff_untouched", ("pff", str(game), "observed", index), *pff.tracking(half))
        for (source, match, index), (_, positions, ball, teams, pitch) in sorted(long_hidden.items()):
            yield "holdout_long_hidden", (source, match, hv.VIEW, index), positions, ball, teams, pitch

    table = sign_table(items())
    by_group = {}
    for row in table.values():
        by_group.setdefault(row["group"], {})[row["half"]] = row
    signs = {group: summarise(rows, False) for group, rows in sorted(by_group.items())}
    disagree = {group: part["first_frame_disagrees"] for group, part in signs.items() if part["first_frame_disagrees"]}
    if disagree:
        raise ValueError(f"D50's sign check fails on confirmation halves {disagree}; nothing of the stack is scored")
    # 5. Decoder inputs, cached: D43 (untouched set, long-hidden views) and the stack (holdout, untouched, long-hidden).
    d43_key, stack_key, stack_identity = confirm_keys(target, config, parsed, games_u, skill, raw)
    model = player_trial.teacher()

    def guard_halves(signals_of):
        for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(long_hidden.items()):
            signals, decoded = signals_of(None, positions, ball, teams, pitch)
            yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)

    diagnosis.collect(cache_root / "d43_untouched", d43_key,
                      lambda: diagnosis.pff_halves(parsed, games_u, model, signals_of=player_trial.half_signals))
    diagnosis.collect(cache_root / "d43_long_hidden", d43_key,
                      lambda: guard_halves(lambda _, *t: player_trial.half_signals(model, *t)))
    sign_map = read(PARITY)["sign_map"] if config["mirror"] is not None else None
    forecaster = Forecaster(stack_models(config, target["scratch"], export=target["export"], sign_map=sign_map,
                                         names=("stack",)), table, with_d43=False)
    forecast = forecaster.signals_of("stack")

    def produce():
        yield from diagnosis.external_halves(dfl, skill, "holdout", None, raw=raw, signals_of=forecast)
        yield from diagnosis.pff_halves(parsed, games_u, None, signals_of=forecast)

    diagnosis.collect(cache_root / "stack", stack_key, produce)
    diagnosis.collect(cache_root / "stack_long_hidden", stack_key, lambda: guard_halves(forecast))
    fallbacks = sorted(row["half"] for row in table.values() if row["fallback"])
    record = {"protocol": PROTOCOL, "target": {key: target[key] for key in ("stack", "cost", "levers")},
              "d43_holdout_reproduced": True, "untouched_parse": parse_note, "long_hidden_views": views,
              "signs": {"groups": signs, "fallbacks": fallbacks},
              "caches": {name: digest(cache_root / name / "manifest.json")
                         for name in ("d43_untouched", "d43_long_hidden", "stack", "stack_long_hidden")},
              "keys": {"d43": d43_key, "stack": stack_key}, "prepared_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"D48 confirmation prepared: untouched games {games_u}; long-hidden views {views['statistics']}; sign fallbacks "
          f"{fallbacks}; no score computed", flush=True)
    return record


def confirm(dfl, skill, parsed, *, raw, cache_root=CONFIRM_CACHE, output=None):
    """D48's confirmation, once, from the prepared caches only: D43 at cost 10 against the stack at its cost.

    D43's numbers are computed first; the stack's scores are computed once and written once.
    """
    require_external_only()
    from dfl_submission import aligned_trial, diagnosis, hyper_trial as ht, player_trial, refine_trial, vendor_trial
    output = RECIPE / "confirmation.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D48's confirmation is scored once")
    prepared = read(cache_root / "prepared.json")
    target = confirmation_stack()
    if {key: target[key] for key in ("stack", "cost", "levers")} != prepared["target"]:
        raise ValueError("The confirmation target differs from the prepared one")
    cost = target["cost"]
    chosen = read(RECIPE / target["levers"])
    config = chosen["config"]
    games_u = prepared["untouched_parse"]["games"]
    games_c = vendor_trial.split_games("confirm_c")
    d43_key, stack_key, stack_identity = confirm_keys(target, config, parsed, games_u, skill, raw)
    if canonical_json({"d43": d43_key, "stack": stack_key}) != prepared["keys"]:
        raise ValueError("The confirmation's cache identities differ from the prepared ones")
    for name, value in prepared["caches"].items():
        if digest(cache_root / name / "manifest.json") != value:
            raise ValueError(f"The {name} cache differs from the prepared one")
    d43_reused = player_trial.candidate_halves(dfl, skill, parsed, games_c, "holdout", raw=raw, cache=D43_HOLDOUT_CACHE)
    d43_external = [half for half in d43_reused if half.source != "pff"]
    d43_untouched = diagnosis.collect(cache_root / "d43_untouched", d43_key, refuse)
    d43_long = diagnosis.collect(cache_root / "d43_long_hidden", d43_key, refuse)
    stack_main = diagnosis.collect(cache_root / "stack", stack_key, refuse)
    stack_long = diagnosis.collect(cache_root / "stack_long_hidden", stack_key, refuse)
    raw_fits = {str(offset): {key: {folder: digest(t_raw(offset, key, folder) / "fit.json") if target["stack"] == "8100"
                                    else digest(raw_folder(offset, key, folder) / "fit.json")
                                    for folder in NETWORKS} for key in SETS} for offset in OFFSETS}
    code = code_identity()

    def raw_standard(item):
        return item["external"]["conventions"]["raw"]["standard"]["spell_f1"]

    # D43 first (this also exercises every scoring function before any stack score exists).
    base = aligned_trial.evaluate(d43_external + d43_untouched, ht.recipe_at(REFERENCE_COST))
    long_base = long_hidden_scores(d43_long, ht.recipe_at(REFERENCE_COST))
    dry = refine_trial.changes(base, base)
    dry["raw_standard"] = raw_standard(base) - raw_standard(base)
    canonical_json({"base": base, "long": long_base, "dry": dry, "raw_fits": raw_fits, "code": code})
    # The stack, once.
    trial = aligned_trial.evaluate(stack_main, ht.recipe_at(cost))
    long_trial = long_hidden_scores(stack_long, ht.recipe_at(cost))
    change = refine_trial.changes(base, trial)
    change["long_hidden"] = long_trial["mean"] - long_base["mean"]
    change["raw_standard"] = raw_standard(trial) - raw_standard(base)
    conditions = {"pff_untouched": bool(change["pff"] >= CONFIRM_GAIN),
                  "long_hidden": bool(change["long_hidden"] >= CONFIRM_GAIN),
                  "primary": bool(change["primary"] >= -LOSS),
                  "conventions": bool(all(value >= -LOSS for value in change["conventions"].values())),
                  "guards": bool(all(value >= -LOSS for value in change["guards"].values()))}
    passed = bool(all(conditions.values()))
    record = canonical_json({
        "protocol": PROTOCOL, "stack": target["stack"], "owner_cost": cost, "reference_cost": REFERENCE_COST,
        "levers": chosen, "levers_sha256": digest(RECIPE / target["levers"]),
        "reproduction": {"d43_holdout": prepared["d43_holdout_reproduced"]},
        "reference": base, "trial": trial, "changes": change,
        "long_hidden": {"d43": long_base, "stack": long_trial, "views": prepared["long_hidden_views"]},
        "conditions": conditions, "passed": passed, "untouched_parse": prepared["untouched_parse"],
        "signs": prepared["signs"], "identity": stack_identity, "raw_fit_sha256": raw_fits, "code": code,
        "d43_models": player_trial.identity(), "prepared_sha256": digest(cache_root / "prepared.json"),
        "inputs": {"pff_untouched": vendor_trial.inputs(parsed, games_u),
                   "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": raw},
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    write_new(output, record)
    print(f"D48 confirmation at cost {cost}: untouched PFF {change['pff']:+.4f}, holdout long-hidden "
          f"{change['long_hidden']:+.4f}, primary {change['primary']:+.4f}, conventions "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items())
          + ", guards " + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f" -> {'PASSED' if passed else 'FAILED'} {conditions}", flush=True)
    return passed


# -- the competition build (behind its own command) ------------------------------------------------

def stack_model_folder(offset, model_set, folder, models_root=MODELS):
    return models_root / str(offset) / model_set / folder


BUILD_MODULES = ("stack_set.py", "player_set.py", "direction_tokens.py", "direction_trial.py", "player_set3.py",
                 "temporal_cpu.py", "player_trial.py")      # the modules the build runs besides this one
BUILD_IDENTITY = "build_identity.json"


class BuildRefused(ValueError):
    """The build refuses; the message names the failed check only."""


def confirmed_build(confirmation, *, models_root=MODELS, recipe=RECIPE, package=PACKAGE):
    """The confirmed switches and cost, after the build's identity checks (D48's adoption amendment, point 1-2).

    Refuses unless the confirmation passed; the lever record is the confirmed one (`confirmation["levers"]`, whose
    file digest is the confirmed one); all twelve networks in `models_root`, the trees and the code the build runs
    equal the confirmed identity; and this module equals the one whose forecasts reproduced the confirmed decoder
    inputs (`build_identity.json`, tied to this `confirmation.json`).
    """
    from dfl_submission import scale_trial
    if confirmation.get("passed") is not True:
        raise BuildRefused("D48 did not pass confirmation; build D43 instead (`player_trial build`)")
    chosen = confirmation["levers"]
    config = dict(chosen["config"])
    ss.validate(config)
    identity = confirmation["identity"]
    lever_file = recipe / (T_LEVERS if confirmation["stack"] == "8100" else "levers.json")
    if (identity["config"] != config or digest(lever_file) != confirmation["levers_sha256"]
            or digest(lever_file) != identity["levers_sha256"] or read(lever_file) != chosen):
        raise BuildRefused("Identity check failed: the lever record differs from the confirmed one")
    if sorted(identity["networks"]) != sorted(str(offset) for offset in OFFSETS):
        raise BuildRefused("Identity check failed: the confirmed identity does not hold three seed sets")
    for offset in OFFSETS:
        for model_set in SETS:
            for folder in NETWORKS:
                path = stack_model_folder(offset, model_set, folder, models_root) / "fit.json"
                if not path.exists() or digest(path) != identity["networks"][str(offset)][model_set][folder]:
                    raise BuildRefused("Identity check failed: a network differs from the confirmed one")
    for key in SETS:
        if digest(scale_trial.TREES[key] / "fit.json") != identity["trees"][key]:
            raise BuildRefused("Identity check failed: the trees differ from the confirmed ones")
    shas = {"stack_set.py": identity["stack_set_sha256"], "player_set.py": identity["player_set_sha256"],
            "direction_tokens.py": identity["direction_tokens_sha256"]}
    for name in BUILD_MODULES:
        if digest(package / name) != confirmation["code"][name] or (name in shas and digest(package / name) != shas[name]):
            raise BuildRefused("Identity check failed: the code differs from the confirmed code")
    locked = recipe / BUILD_IDENTITY
    if not locked.exists():
        raise BuildRefused("Identity check failed: no build identity record")
    lock = read(locked)
    if (lock.get("passed") is not True or lock["confirmation_sha256"] != digest(recipe / "confirmation.json")
            or digest(package / "stack_trial.py") != lock["stack_trial_sha256"]):
        raise BuildRefused("Identity check failed: the build driver differs from the reproduced one")
    return config, confirmation["owner_cost"]


def kickoff_frames(length, start):
    """The long-context frames of a half at or after frame `start` (all of them for `start` 0)."""
    frames = dt.long_context_frames(length)
    return frames[frames >= start]


def kickoff_starts(halves, tags, blocks):
    """Each graded half's tagged kick-off as a frame within the half, paired as D27 pairs a tag with its half.

    `halves` is [(half, span, tracking)] as `build.tactical_tracking_halves` yields it, `tags` the `Kick Off (Start)`
    instances of `kickoff.read_tags` and `blocks` `kickoff.half_blocks()`. A half owns the one tag that ends at or
    after its first frame and starts at most `kickoff.LIMIT` after it (`kickoff.apply`); a tag starting before the
    half's first frame gives frame 0. Refuses, naming the check only, unless every half owns exactly one tag.
    """
    from dfl_submission import kickoff
    starts = []
    for half, span, _ in halves:
        block = [(first, last) for number, first, last in blocks if number == half]
        if len(block) != 1 or block[0][0] != span.start or block[0][1] + 1 != span.stop:
            raise BuildRefused("Sign check failed: a graded half does not match its half block")
        first, last = block[0]
        owned = [start for start, end in tags if end >= first and start <= first + kickoff.LIMIT]
        if len(owned) != 1:
            raise BuildRefused("Sign check failed: a graded half has no single kick-off tag")
        start = max(0, owned[0] - first)
        if start > last - first:
            raise BuildRefused("Sign check failed: a half's kick-off tag lies after the half")
        starts.append(int(start))
    return starts


def build_signs(halves, starts=None):
    """Each graded half's attacking sign by D50's rule, failing closed (D48's adoption amendment, point 3).

    Refuses if a half has no sign of its own, if a half's sign disagrees with its first qualifying frame, or if the
    two halves have the same sign. The messages name the check only; nothing from the tracking is printed. With
    `starts` (one frame per half, the tagged kick-off), D and the first qualifying frame are read from the
    long-context frames at or after it; without, from the whole half (frame 0 gives the same result bit for bit).
    """
    if len(halves) != 2 or (starts is not None and len(starts) != 2):
        raise BuildRefused("Sign check failed: the build expects two graded halves")
    entries = [dt.attacking(tracking[0], tracking[2]) if starts is None
               else dt.attacking(tracking[0], tracking[2], kickoff_frames(len(tracking[0]), start))
               for tracking, start in zip(halves, starts or (None, None), strict=True)]
    if any(entry["sign"] is None for entry in entries):
        raise BuildRefused("Sign check failed: a graded half has no sign of its own")
    if not all(entry["agrees"] for entry in entries):
        raise BuildRefused("Sign check failed: a half's sign disagrees with its first qualifying frame")
    if entries[0]["sign"] == entries[1]["sign"]:
        raise BuildRefused("Sign check failed: the two halves have the same sign")
    return [int(entry["sign"]) for entry in entries]


def build_table(halves, starts=None):
    """The forecaster's sign table for the graded halves; no fallback can apply (a half without a sign is refused)."""
    signs = build_signs(halves, starts)
    return {tracking_key(*tracking): {"sign_used": sign, "half": str(index), "fallback": None}
            for index, (sign, tracking) in enumerate(zip(signs, halves, strict=True))}


def build(output, *, models_root=MODELS):
    """The competition file with the confirmed stack at its cost, then D27's rule. `player_trial build` is unchanged.

    The identity checks run before the competition tracking is read. Each graded half's attacking sign is computed
    from that half's tracking by D50's rule (a feature of the input, not a fitted value), read from the half's tagged
    `Kick Off (Start)` onward, the tag paired with its half as D27 pairs it (D48's second adoption amendment), failing
    closed; every frame of the half, the frames before the kick-off included, takes the sign.
    """
    # `long_context` (which the forecasts use) imports `models`, which puts `experiments/` (the bundle reader) on the
    # module path; `player_trial build` imports it through its teacher before reading the tracking, and so does this.
    from dfl_submission import kickoff, long_context, objective_trial  # noqa: F401 (long_context: the module path)
    from dfl_submission.build import tactical_labels, tactical_tracking_halves, write_csv
    config, cost = confirmed_build(read(RECIPE / "confirmation.json"), models_root=models_root)
    _, frozen_d15 = objective_trial.d15()
    sign_map = read(PARITY)["sign_map"] if config["mirror"] is not None else None
    models = stack_models(config, models_root, export=stack_model_folder, sign_map=sign_map, names=("stack",))
    halves = list(tactical_tracking_halves(centered=True, pitch=frozen_d15["pitch"]))
    starts = kickoff_starts(halves, kickoff.read_tags(kickoff.tag_path()), kickoff.half_blocks())
    table = build_table([tracking for _, _, tracking in halves], starts)
    forecaster = Forecaster(models, table, owner_cost=cost, with_d43=False)

    def decoded():
        for half, span, tracking in halves:
            print(f"Predicting tactical half {half}", flush=True)
            yield span, forecaster.forecast(*tracking)["stack"][1]

    labels = tactical_labels(decoded())
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


# -- adoption: the confirmed networks, and the build driver's reproduction ------------------------------

def confirmed_source(offset, model_set, folder):
    """Where the confirmed stack's export folder lives (8,100-step: D52's T for offset 0, `data/d48/t` otherwise)."""
    stack = read(RECIPE / "confirmation.json")["stack"]
    return t_export(offset, model_set, folder) if stack == "8100" else export_folder(offset, model_set, folder)


def folder_digests(folder):
    return {path.name: digest(path) for path in sorted(Path(folder).iterdir()) if path.is_file()}


def adopt_models(*, models_root=MODELS, output=None):
    """Copy the twelve confirmed export folders byte for byte into `stack_models/` and verify them, written once."""
    import shutil
    require_external_only()
    output = RECIPE / "adoption_models.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the confirmed networks are copied once")
    confirmation = read(RECIPE / "confirmation.json")
    if confirmation["passed"] is not True:
        raise ValueError("D48 did not pass confirmation; nothing is adopted")
    identity = confirmation["identity"]["networks"]
    copied = {}
    for offset in OFFSETS:
        for model_set in SETS:
            for folder in NETWORKS:
                source = confirmed_source(offset, model_set, folder)
                if digest(source / "fit.json") != identity[str(offset)][model_set][folder]:
                    raise ValueError(f"The source {source} differs from the confirmed network")
                destination = stack_model_folder(offset, model_set, folder, models_root)
                if destination.exists():
                    raise FileExistsError(f"{destination} exists")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
                if folder_digests(source) != folder_digests(destination):
                    raise ValueError(f"The copy of {source} differs from its source")
                if digest(destination / "fit.json") != identity[str(offset)][model_set][folder]:
                    raise ValueError(f"The copy {destination} differs from the confirmed network")
                copied[f"{offset}/{model_set}/{folder}"] = {"source": str(source.relative_to(ROOT)),
                                                            "files": folder_digests(destination)}
    write_new(output, {"protocol": PROTOCOL, "stack": confirmation["stack"], "copied": copied, "passed": True,
                       "confirmation_sha256": digest(RECIPE / "confirmation.json"),
                       "copied_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 adoption: {len(copied)} confirmed networks copied byte for byte into {models_root.relative_to(ROOT)} and "
          f"verified against the confirmed identity", flush=True)


def stored_halves(folder):
    """The halves of a cache folder under its own stored identity (for re-scoring cached inputs only)."""
    from dfl_submission import diagnosis
    return diagnosis.collect(Path(folder), read(Path(folder) / "manifest.json")["identity"], refuse)


def reproduce_build(dfl, skill, parsed, *, raw, output=None):
    """The edited build driver reproduces the confirmed run, then its digest is recorded (`build_identity.json`), once.

    (1) development_t.json's entries and confirmation.json's scores are recomputed from their cached decoder inputs;
    (2) the eight development guard halves are forecast afresh through the build's path (`Forecaster`, the networks
    in `stack_models/`) and must equal the confirmed run's cached decoder inputs bit for bit.
    """
    require_external_only()
    from dfl_submission import aligned_trial, diagnosis, hidden_trial as h49, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import player_trial, refine_trial, vendor_trial
    output = RECIPE / BUILD_IDENTITY if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the build identity is recorded once")
    confirmation = read(RECIPE / "confirmation.json")
    if confirmation["stack"] != "8100" or read(RECIPE / "adoption_models.json")["passed"] is not True:
        raise ValueError("The 8,100-step stack is not the adopted one")
    # (1a) development_t.json from its caches
    development = read(RECIPE / "development_t.json")
    games = vendor_trial.development()
    d43_dev, d43_metrica, d43_d24, truths, _ = h49.d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=D42_CACHE)
    visibility = h49.pff_visibility(parsed, games)
    base = h49.entry(d43_dev, d43_metrica, d43_d24, d43_guards(raw), ht.recipe_at(REFERENCE_COST), truths, visibility)
    development_same = canonical_json(dict(h49.compact(base), selection=selection(base))) == canonical_json(
        {key: value for key, value in development["reference"].items() if key not in ("name", "owner_cost")} |
        {"owner_cost": REFERENCE_COST})
    for name in MODEL_NAMES:
        halves = {part: stored_halves(T_SCRATCH / "cache" / name / part) for part in ("development", "metrica", "d24", "guards")}
        entries = [h49.entry(halves["development"], halves["metrica"], halves["d24"], halves["guards"], ht.recipe_at(cost),
                             truths, visibility) for cost in COSTS]
        recomputed = canonical_json([dict(h49.compact(item), selection=selection(item)) for item in entries])
        development_same = development_same and recomputed == development["entries"][name]
    # (1b) confirmation.json from its caches
    prepared = read(CONFIRM_CACHE / "prepared.json")
    games_u, games_c = prepared["untouched_parse"]["games"], vendor_trial.split_games("confirm_c")
    d43_reused = player_trial.candidate_halves(dfl, skill, parsed, games_c, "holdout", raw=raw, cache=D43_HOLDOUT_CACHE)
    keys = prepared["keys"]
    d43_untouched = diagnosis.collect(CONFIRM_CACHE / "d43_untouched", keys["d43"], refuse)
    d43_long = diagnosis.collect(CONFIRM_CACHE / "d43_long_hidden", keys["d43"], refuse)
    stack_main = diagnosis.collect(CONFIRM_CACHE / "stack", keys["stack"], refuse)
    stack_long = diagnosis.collect(CONFIRM_CACHE / "stack_long_hidden", keys["stack"], refuse)
    cost = confirmation["owner_cost"]
    reference = aligned_trial.evaluate([half for half in d43_reused if half.source != "pff"] + d43_untouched,
                                       ht.recipe_at(REFERENCE_COST))
    trial = aligned_trial.evaluate(stack_main, ht.recipe_at(cost))
    long_hidden = {"d43": long_hidden_scores(d43_long, ht.recipe_at(REFERENCE_COST)),
                   "stack": long_hidden_scores(stack_long, ht.recipe_at(cost))}
    change = refine_trial.changes(reference, trial)
    confirmation_same = (canonical_json(reference) == confirmation["reference"]
                         and canonical_json(trial) == confirmation["trial"]
                         and all(canonical_json(long_hidden[k]) == {key: confirmation["long_hidden"][k][key]
                                                                     for key in ("matches", "mean")} for k in long_hidden)
                         and all(canonical_json(change[k]) == confirmation["changes"][k] for k in change))
    print(f"D48 build reproduction: development_t {'same' if development_same else 'DIFFERS'}, confirmation "
          f"{'same' if confirmation_same else 'DIFFERS'}", flush=True)
    # (2) fresh forecasts through the build's path with the adopted networks
    config = confirmation["levers"]["config"]
    sign_map = read(PARITY)["sign_map"] if config["mirror"] is not None else None
    forecaster = Forecaster(stack_models(config, MODELS, export=stack_model_folder, sign_map=sign_map, names=("stack",)),
                            table_of(SCRATCH / "sign_table_development.json"), with_d43=False)
    cached = stored_halves(T_SCRATCH / "cache" / "stack" / "guards")
    tracking = h49.guard_tracking(dfl, raw=raw)
    forecasts_same = len(cached) == len(tracking)
    for half, ((source, match, index), (labels, positions, ball, teams, pitch)) in zip(cached, sorted(tracking.items()),
                                                                                       strict=True):
        signals, decoded = forecaster.forecast(positions, ball, teams, pitch)["stack"]
        same = ((half.source, half.match, half.index) == (source, match, index)
                and all(np.array_equal(a, b) for a, b in zip(signals, half.signals, strict=True))
                and np.array_equal(decoded, half.d3))
        forecasts_same = forecasts_same and same
        print(f"Reforecast guard {source} {match} half {index + 1}: {'bit for bit' if same else 'DIFFERS'}", flush=True)
    passed = bool(development_same and confirmation_same and forecasts_same)
    write_new(output, {"protocol": PROTOCOL, "passed": passed, "stack_trial_sha256": digest(PACKAGE / "stack_trial.py"),
                       "confirmation_sha256": digest(RECIPE / "confirmation.json"),
                       "adoption_models_sha256": digest(RECIPE / "adoption_models.json"),
                       "checks": {"development_t_recomputed": development_same,
                                  "confirmation_recomputed": confirmation_same,
                                  "guard_halves_reforecast_bit_for_bit": forecasts_same, "guard_halves": len(cached)},
                       "code": {name: digest(PACKAGE / name) for name in BUILD_MODULES},
                       "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D48 build identity: {'recorded (passed)' if passed else 'FAILED'}", flush=True)
    return passed


# -- summary ---------------------------------------------------------------------------------------------

SHORT = {"dfl_clean": "clean", "dfl_masked": "masked", "dfl_dead_carried": "dead", "dfl_team_swap": "swap",
         "skillcorner": "SC", "pff": "PFF", "metrica": "Metrica", "d24": "D24"}


def conventions_of(item):
    return {name: item["trial"]["external"]["conventions"][name]["standard"]["spell_f1"]
            for name in ("aba3", "aba1.75", "aba4", "aba4.5")}


def summary(folder=RECIPE):
    """The development table: D43, each seed set alone and the stack at every cost, then the gate."""
    reference_record = read(folder / "reference.json")
    development = read(folder / "development.json")
    base = development["reference"]
    base_conventions = conventions_of(base)
    rows = ["| Model | Cost | Composite | Guard mean (DFL / Metrica) | Selection | PFF | Primary | "
            + " | ".join(SHORT[g] for g in GROUPS) + " | N 1.75 / 4 / 4.5 vs D43@10 |",
            "|---|---:|---:|---|---:|---:|---:|" + "---:|" * len(GROUPS) + "---|"]

    def row(name, item):
        conventions = conventions_of(item)
        changes = " / ".join(f"{conventions[n] - base_conventions[n]:+.4f}" for n in ("aba1.75", "aba4", "aba4.5"))
        guards = item["guards"]
        return (f"| {name} | {item['owner_cost']} | {item['composite']:.4f} | {guards['mean']:.4f} "
                f"({guards['a_dfl']:.3f} / {guards['b_metrica']:.3f}) | {selection(item):.4f} | {item['groups']['pff']:.4f} | "
                f"{conventions['aba3']:.4f} | " + " | ".join(f"{item['groups'][g]:.3f}" for g in GROUPS) + f" | {changes} |")

    for cost in COSTS:
        rows.append(row("D43", reference_record["entries"][str(cost)]))
    for name in MODEL_NAMES[1:] + MODEL_NAMES[:1]:
        for item in development["entries"][name]:
            rows.append(row(name, item))
    decision = development["decision"]
    changes = decision["changes"]
    lines = ["\n".join(rows), "",
             f"Selected cost {decision['selected_cost']} (selection {decision['selection_score']:.5f} against D43's "
             f"{base['selection']:.5f}):",
             f"- selection {changes['selection']:+.5f} (needs +{SELECTION_GAIN}): {decision['conditions']['selection']}",
             f"- PFF {changes['pff']:+.5f} (needs +{PFF_GAIN}): {decision['conditions']['pff']}",
             f"- composite {changes['composite']:+.5f} (must not fall): {decision['conditions']['composite']}",
             f"- worst group {min(changes['groups'], key=changes['groups'].get)} {min(changes['groups'].values()):+.4f} "
             f"(each >= -{LOSS}): {decision['conditions']['groups']}",
             "- conventions " + ", ".join(f"{k} {v:+.4f}" for k, v in changes["conventions"].items())
             + f" (each >= -{LOSS}): {decision['conditions']['conventions']}",
             f"Gate {'PASSED: the confirmation may be spent' if decision['passed'] else 'NOT passed'}"]
    text = "\n".join(lines)
    print(text)
    return text


# -- command line ------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check")
    checking.add_argument("--name", choices=tuple(CHECKS), required=True)
    checking.add_argument("--external", type=Path, required=True, help="the external data root")
    signing = commands.add_parser("signs")
    signing.add_argument("--set", choices=SETS, required=True)
    signing.add_argument("--external", type=Path, required=True)
    scoring = commands.add_parser("scoring-signs")
    commands.add_parser("sign-record")
    ref = commands.add_parser("reference")
    dev = commands.add_parser("develop")
    prep = commands.add_parser("confirm-prepare")
    prep.add_argument("--pff-raw", type=Path, required=True)
    conf = commands.add_parser("confirm")
    for command in (signing, scoring, ref, dev, prep, conf):
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
    for command in (scoring, ref, dev, prep, conf):
        command.add_argument("--parsed", type=Path, required=True)
    lever = commands.add_parser("levers")
    lever.add_argument("--mirror", choices=("y",))
    lever.add_argument("--noise", action="store_true")
    lever.add_argument("--aux", type=Path, help="D51's folder holding aux_current/aux_augmented, if LB is in")
    lever.add_argument("--note", default="")
    networks = commands.add_parser("networks")
    networks.add_argument("--external", type=Path, required=True)
    networks.add_argument("--set", choices=SETS, action="append", help="train only these sets (repeatable)")
    commands.add_parser("summary")
    commands.add_parser("summary-t")
    commands.add_parser("adopt-models")
    repro = commands.add_parser("reproduce-build")
    for name, kind in (("--dfl", Path), ("--skill", Path), ("--parsed", Path)):
        repro.add_argument(name, type=kind, required=True)
    repro.add_argument("--dfl-raw", action="store_true")
    repro.add_argument("--out", type=Path, help="where the build identity lock is written (default: the recipe folder)")
    commands.add_parser("train-steps", add_help=False)
    for name in ("check-t", "levers-t", "networks-t"):
        command = commands.add_parser(name)
        command.add_argument("--external", type=Path, required=True)
        if name == "networks-t":
            command.add_argument("--set", choices=SETS, action="append")
    develop_t_command = commands.add_parser("develop-t")
    develop_t_command.add_argument("--dfl", type=Path, required=True)
    develop_t_command.add_argument("--dfl-raw", action="store_true")
    develop_t_command.add_argument("--skill", type=Path, required=True)
    develop_t_command.add_argument("--parsed", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    if len(sys.argv) > 1 and sys.argv[1] == "train-steps":
        train_steps(sys.argv[2:])
        return
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    if args.command == "summary":
        summary()
        return
    if args.command == "summary-t":
        summary_t()
        return
    require_external_only()
    if args.command == "check":
        check(args.name, args.external)
    elif args.command == "signs":
        training_signs(args.set, args.external, args.dfl, args.skill, raw=args.dfl_raw)
    elif args.command == "scoring-signs":
        scoring_signs(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "sign-record":
        sign_record()
    elif args.command == "reference":
        reference(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "levers":
        record_levers(mirror=args.mirror, noise=args.noise, aux=args.aux, note=args.note)
    elif args.command == "networks":
        train_networks(args.external, sets=tuple(args.set) if args.set else SETS)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "check-t":
        check_t(args.external)
    elif args.command == "levers-t":
        record_levers_t(args.external)
    elif args.command == "networks-t":
        train_networks_t(args.external, sets=tuple(args.set) if args.set else SETS)
    elif args.command == "develop-t":
        develop_t(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "adopt-models":
        adopt_models()
    elif args.command == "reproduce-build":
        reproduce_build(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, output=args.out)
    elif args.command == "confirm-prepare":
        confirm_prepare(args.dfl, args.skill, args.parsed, args.pff_raw, raw=args.dfl_raw)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)


if __name__ == "__main__":
    main()
