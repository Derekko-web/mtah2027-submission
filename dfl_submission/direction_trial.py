"""D50: player tokens in the attacking frame, screened on development (DECISIONS.md D50).

Commands:
- `signs` re-reads every prepared training half of one model set, checks that its player tokens
  recomputed from the tracking equal D42's stored tokens, and writes each record's attacking sign
  (from every visible player) with the first-frame sign, the sign read from the stored tokens alone,
  and the largest number of visible players per team; `dev-signs` does the same for every
  development half (DFL WPY views, SkillCorner, PFF), the Metrica sample halves and the D24 views;
  `sign-record` joins the three into the pipeline-check record; `sign-gap` records, by a rule the
  orchestrator chose, a sign for each development half the rule leaves without one (no frame with
  seven visible players in each team), keyed by the hash of the tracking its forecast receives;
- `check-trainer` trains D42's original-set seed-42 network with every sign +1 and no swap negation
  and compares its snapshots with D42's bit for bit;
- `columns` freezes which of the 162 columns are multiplied by the sign (the x mirror's sign changes in
  D47's parity record, if that record exists then);
- `networks` trains (one at a time, only while fewer than two trainings run on the machine) and
  exports one set's two candidate networks;
- `develop` checks that D43 reproduces its development, Metrica and D24 records from `data/d42/cache`,
  forecasts candidate C (and D43 again through the same path, which must equal the cache bit for bit),
  and writes one record per ownership cost;
- `guards` scores C on D49's long-hidden guard views (reported beside D50's rule), with D43 recomputed
  against D49's cache bit for bit;
- `summary` prints the table of the records.
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

from dfl_submission import direction_tokens as dt
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "direction_recipe"
SCRATCH = ROOT / "data" / "d50"
D43_CACHE = ROOT / "data" / "d42" / "cache"
PROTOCOL = "DECISIONS.md D50"
SETS = ("current", "augmented")
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
COSTS = (8, 10, 12)
REFERENCE_COST = 10
LOSS = .01
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica", "d24")
AGREEMENT_GROUPS = ("dfl_masked", "skillcorner", "pff", "d24_masked")
TRAINING_LIMIT = 2
TRAINER = re.compile(r"-m dfl_submission\.([a-z_0-9]+ train\b|player_set3 )")
IN_PROCESS = re.compile(r"-m dfl_submission\.[a-z_0-9]+ networks\b")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D50 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def canonical_json(value):
    return json.loads(json.dumps(value))


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# -- the GPU and memory share ----------------------------------------------------------------------

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


def wait_for_share(limit=TRAINING_LIMIT, poll=30, what="training"):
    """Block until fewer than `limit` trainings run machine-wide."""
    waited = False
    while (count := running_trainings()) >= limit:
        if not waited:
            print(f"{time.strftime('%H:%M:%S')} waiting before {what}: {count} trainings running", flush=True)
            waited = True
        time.sleep(poll)
    if waited:
        print(f"{time.strftime('%H:%M:%S')} starting {what}", flush=True)


# -- the attacking-sign pipeline check -----------------------------------------------------------

def visible_all_frames(positions, teams):
    """Largest number of visible players per team over every 25 Hz frame of the half."""
    visible = np.isfinite(np.asarray(positions, dtype=np.float64)).all(axis=2)
    teams = np.asarray(teams)
    return [int(visible[:, teams == team].sum(axis=1).max(initial=0)) for team in (0, 1)]


def half_entry(positions, teams, frames=None):
    entry = dt.attacking(positions, teams, frames)
    entry["max_visible_all_frames"] = visible_all_frames(positions, teams)
    return entry


def training_signs(model_set, source, tokens, dfl, skill, output, *, raw):
    """One set's training halves: tokens recomputed and compared, signs from tracking and from the stored tokens."""
    require_external_only()
    from dfl_submission import aligned_trial, player_set as ps, touch_trial
    if output.exists():
        raise FileExistsError(f"{output} exists")
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    token_manifest = read(tokens / "manifest.json")
    if token_manifest["base_manifest"] != digest(source / "manifest.json"):
        raise ValueError(f"{tokens} holds tokens for a different prepared folder")
    records = {}
    halves = touch_trial.training_halves(model_set, dfl, skill, raw=raw)
    for (record, path), half in zip(aligned_trial.verified_records(source), halves, strict=True):
        name = record["file"]
        if name != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match the prepared record {name}")
        with np.load(path, allow_pickle=False) as item:
            stored_frames = len(item["values"])
        frames = dt.long_context_frames(len(half.ball))
        if len(frames) != stored_frames:
            raise ValueError(f"Long-context frames differ from the prepared frames: {name}")
        token_path = tokens / name
        if digest(token_path) != token_manifest["records"][name]:
            raise ValueError(f"Token file hash differs: {token_path}")
        with np.load(token_path, allow_pickle=False) as item:
            stored = item["tokens"]
        recomputed = ps.tokens(half.positions, half.ball, half.teams, half.pitch, frames).astype(np.float16)
        entry = half_entry(half.positions, half.teams, frames)
        from_tokens = dt.attacking_from_tokens(stored, frames)
        entry |= {"source": record["source"], "tokens_equal": bool(np.array_equal(recomputed, stored, equal_nan=True)),
                  "token": {key: from_tokens[key] for key in ("frames", "d", "sign", "first_sign", "agrees")}}
        records[name] = entry
        print(f"Signs {model_set} {name}: D {entry['d']:+.3f} m, sign {entry['sign']:+d}, first {entry['first_sign']:+d}, "
              f"tokens {'equal' if entry['tokens_equal'] else 'DIFFER'}, token sign {entry['token']['sign']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": PROTOCOL, "set": model_set, "base_manifest": digest(source / "manifest.json"),
                                  "token_manifest": digest(tokens / "manifest.json"), "records": records,
                                  "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py")},
                                 indent=2, sort_keys=True) + "\n")


def development_signs(dfl, skill, parsed, output, *, raw):
    """Every development half, Metrica sample half and D24 view half read in D50's scoring."""
    require_external_only()
    from dfl_submission import augment, metrica_sample as ms, metrica_views as mv, pff, vendor_trial
    if output.exists():
        raise FileExistsError(f"{output} exists")
    halves = {}
    for half in augment.guard_halves(dfl, skill, "development", raw=raw):
        halves[f"{half.source}/{half.match}/{half.view}/{half.index}"] = dict(half_entry(half.positions, half.teams),
                                                                             group=half.source if half.source != "dfl" else f"dfl_{half.view}")
    games = vendor_trial.development()
    for game in games:
        for index, half in enumerate(pff.load(parsed, game)):
            positions, _, teams, _ = pff.tracking(half)
            halves[f"pff/{game}/observed/{index}"] = dict(half_entry(positions, teams), group="pff")
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            halves[f"metrica/{game}/observed/{index}"] = dict(half_entry(half["positions"], half["teams"]), group="metrica")
            for view, (positions, _) in mv.build_views(half, game, index).items():
                halves[f"metrica/{game}/{view}/{index}"] = dict(half_entry(positions, half["teams"]), group=f"d24_{view}")
    for key, entry in halves.items():
        print(f"Signs {key}: D {entry['d'] if entry['d'] is None else round(entry['d'], 3)}, sign {entry['sign']}, "
              f"first {entry['first_sign']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"protocol": PROTOCOL, "halves": halves, "pff": vendor_trial.inputs(parsed, games),
                                  "metrica": ms.inputs(ms.PARSED), "skill_manifest_sha256": digest(skill / "manifest.json"),
                                  "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py")},
                                 indent=2, sort_keys=True) + "\n")


def part_summary(entries, training):
    """Pass/fail and the numbers the report needs for one group of halves."""
    entries = dict(entries)
    missing = sorted(key for key, entry in entries.items() if entry["sign"] is None)
    disagree = sorted(key for key, entry in entries.items() if entry["sign"] is not None and not entry["agrees"])
    over = {key: entry["max_visible"] for key, entry in entries.items() if max(entry["max_visible"]) > dt.ps.PER_TEAM}
    over_all = {key: entry["max_visible_all_frames"] for key, entry in entries.items()
                if max(entry["max_visible_all_frames"]) > dt.ps.PER_TEAM}
    magnitudes = {key: abs(entry["d"]) for key, entry in entries.items() if entry["d"] is not None}
    smallest = min(magnitudes, key=magnitudes.get) if magnitudes else None
    result = {"halves": len(entries), "no_qualifying_frame": missing, "first_frame_disagrees": disagree,
              "smallest_abs_d_m": None if smallest is None else magnitudes[smallest], "smallest_abs_d_half": smallest,
              "negative_signs": sum(1 for entry in entries.values() if entry["sign"] == -1),
              "over_per_team_long_context": over, "over_per_team_any_frame": over_all,
              "min_qualifying_frames": min((entry["frames"] for entry in entries.values()), default=0)}
    passed = not missing and not disagree
    if training:
        unequal = sorted(key for key, entry in entries.items() if not entry["tokens_equal"])
        token_differs = sorted(key for key, entry in entries.items() if entry["token"]["sign"] != entry["sign"])
        token_first = sorted(key for key, entry in entries.items() if not entry["token"]["agrees"])
        result |= {"tokens_unequal": unequal, "token_sign_differs": token_differs, "token_first_frame_disagrees": token_first,
                   "smallest_abs_token_d": min((abs(entry["token"]["d"]) for entry in entries.values()
                                                if entry["token"]["d"] is not None), default=None)}
        passed = passed and not unequal and not token_differs
    result["passed"] = bool(passed)
    return result


def sign_record(output=None):
    require_external_only()
    output = RECIPE / "signs.json" if output is None else output
    parts = {model_set: read(SCRATCH / f"signs_{model_set}.json") for model_set in SETS}
    development = read(SCRATCH / "signs_development.json")
    summaries = {f"training_{model_set}": part_summary(part["records"], True) for model_set, part in parts.items()}
    by_group = {}
    for key, entry in development["halves"].items():
        by_group.setdefault(entry["group"], {})[key] = entry
    summaries |= {group: part_summary(entries, False) for group, entries in sorted(by_group.items())}
    passed = all(item["passed"] for item in summaries.values())
    smallest = min((item["smallest_abs_d_m"], name) for name, item in summaries.items() if item["smallest_abs_d_m"] is not None)
    record = {"protocol": PROTOCOL, "check": "whole-half sign equals first-frame sign (training and development halves)",
              "passed": passed, "parts": summaries, "smallest_abs_d_m": {"value": smallest[0], "part": smallest[1]},
              "halves": sum(item["halves"] for item in summaries.values()),
              "files": {name: digest(SCRATCH / f"signs_{name}.json") for name in (*SETS, "development")},
              "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, record)
    print(f"D50 sign check: {'passed' if passed else 'FAILED'}; {record['halves']} halves; smallest |D| "
          f"{smallest[0]:.3f} m ({smallest[1]})", flush=True)
    for name, item in summaries.items():
        print(f"  {name}: {item['halves']} halves, negative {item['negative_signs']}, smallest |D| {item['smallest_abs_d_m']:.3f} m, "
              f"disagree {item['first_frame_disagrees']}, none {item['no_qualifying_frame']}, >11 visible at token frames "
              f"{len(item['over_per_team_long_context'])}, any frame {len(item['over_per_team_any_frame'])}"
              + (f", tokens unequal {item['tokens_unequal']}, token sign differs {item['token_sign_differs']}"
                 if name.startswith("training") else ""), flush=True)
    return passed


def passed_record(name):
    path = RECIPE / name
    if not path.exists() or read(path)["passed"] is not True:
        raise ValueError(f"{path} is missing or did not pass; nothing is trained or scored")
    return read(path)


def training_signs_passed():
    """The sign check's gate for training: every training half passed, and no half anywhere disagrees.

    A development half with no qualifying frame (no sign under D50's rule) does not touch training; it
    is settled before scoring.
    """
    record = read(RECIPE / "signs.json")
    parts = record["parts"]
    disagree = {name: part["first_frame_disagrees"] for name, part in parts.items() if part["first_frame_disagrees"]}
    if disagree or not all(parts[f"training_{model_set}"]["passed"] for model_set in SETS):
        raise ValueError(f"The training sign check did not pass ({disagree}); nothing is trained")
    return record


def tracking_key(positions, ball, teams, pitch):
    """A content hash of the tracking a forecast receives, to recognise a half without its name."""
    hasher = hashlib.blake2b(digest_size=20)
    for array in (positions, ball, teams, pitch):
        array = np.ascontiguousarray(array)
        hasher.update(str((array.shape, array.dtype.str)).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


GAP_RULES = ("other_half", "plus", "relaxed")


def fallback_sign(other_sign):
    """D50's amendment: a half with no qualifying frame takes the opposite of the other half's sign (teams
    change ends at half time), or +1 (pitch coordinates) when the other half has no sign either."""
    return 1 if other_sign is None else -int(other_sign)
RELAXED_PLAYERS = 1


def relaxed_sign(positions, teams):
    """D (as `direction_tokens.attacking`) over frames with at least `RELAXED_PLAYERS` visible per team."""
    original = dt.MIN_PLAYERS
    try:
        dt.MIN_PLAYERS = RELAXED_PLAYERS
        return dt.attacking(positions, teams)
    finally:
        dt.MIN_PLAYERS = original


def sign_gap(rule, parsed, output=None):
    """Signs for the development halves that D50's rule leaves without one, by a rule the orchestrator chose.

    Written once, keyed by the tracking hash the forecast will see. Only PFF halves can be covered here
    (the only kind found without a sign); any other sign-less half stops.
    """
    require_external_only()
    from dfl_submission import pff
    output = RECIPE / "sign_gap.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the sign-gap decision is recorded once")
    if rule not in GAP_RULES:
        raise ValueError(f"Unknown sign-gap rule {rule}")
    record = read(RECIPE / "signs.json")
    development = read(SCRATCH / "signs_development.json")["halves"]
    missing = [key for part in record["parts"].values() for key in part["no_qualifying_frame"]]
    if any(not key.startswith("pff/") for key in missing):
        raise ValueError(f"Sign-less halves other than PFF: {missing}")
    halves = {}
    for key in missing:
        _, game, view, index = key.split("/")
        loaded = pff.load(parsed, int(game))
        positions, ball, teams, pitch = pff.tracking(loaded[int(index)])
        if rule == "other_half":                  # D50's amendment: the other half's sign negated, else +1
            other = development[f"pff/{game}/{view}/{1 - int(index)}"]
            sign = fallback_sign(other["sign"])
            basis = {"other_half_sign": other["sign"], "other_half_d": other["d"]}
        elif rule == "plus":
            sign, basis = 1, {}
        else:
            relaxed = relaxed_sign(positions, teams)
            if relaxed["sign"] is None:
                raise ValueError(f"{key} has no frame with {RELAXED_PLAYERS} visible player per team either")
            sign, basis = relaxed["sign"], {"relaxed": relaxed, "relaxed_players": RELAXED_PLAYERS}
        halves[key] = {"sign": int(sign), "tracking_key": tracking_key(positions, ball, teams, pitch), **basis}
    write_new(output, {"protocol": PROTOCOL, "rule": rule, "halves": halves,
                       "signs_sha256": digest(RECIPE / "signs.json"), "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D50 sign gap ({rule}): " + ", ".join(f"{k} sign {v['sign']:+d}" for k, v in halves.items()), flush=True)
    return halves


def scoring_signs():
    """The sign check's gate for scoring: no disagreement anywhere, training passed, and every sign-less
    development half covered by the recorded sign-gap decision. Returns {tracking key: sign} of those halves."""
    record = training_signs_passed()
    missing = sorted(key for part in record["parts"].values() for key in part["no_qualifying_frame"])
    if not missing:
        if record["passed"] is not True:
            raise ValueError("The sign check did not pass; nothing is scored")
        return {}
    path = RECIPE / "sign_gap.json"
    if not path.exists():
        raise ValueError(f"Sign-less development halves {missing} have no recorded decision; nothing is scored")
    gap = read(path)
    if gap["signs_sha256"] != digest(RECIPE / "signs.json") or sorted(gap["halves"]) != missing:
        raise ValueError("The sign-gap decision does not cover exactly the sign-less halves of this sign check")
    return {entry["tracking_key"]: entry["sign"] for entry in gap["halves"].values()}


# -- the trainer check -----------------------------------------------------------------------------

def check_trainer(source, tokens, scratch=SCRATCH / "check", *, output=None):
    """Every sign +1 and no swap negation (subprocess, MPS): every snapshot must hash as D42's."""
    require_external_only()
    from dfl_submission import player_trial
    output = RECIPE / "trainer_check.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the trainer check runs once")
    fit = read(player_trial.network_folders()["current"] / "temporal_models" / "fit.json")
    expected = fit["original_checkpoints"]
    if digest(source / "manifest.json") != fit["train_manifest"] or digest(tokens / "manifest.json") != fit["token_manifest"]:
        raise ValueError("The check must use the prepared sequences and tokens D42's network trained on")
    run = scratch / "current" / "temporal_models"
    if not (run / "fit.json").exists():
        if run.exists():
            raise FileExistsError(f"{run} holds an unfinished run; remove it first")
        run.parent.mkdir(parents=True, exist_ok=True)
        wait_for_share(what="the trainer check")
        subprocess.run([sys.executable, "-m", "dfl_submission.direction_tokens", "train", "--source", str(source),
                        "--tokens", str(tokens), "--out", str(run), "--seed", "42", "--boundary-boost", "0.5",
                        "--check"], check=True, cwd=ROOT)
    trained = read(run / "fit.json")
    for key in ("seed", "boundary_boost", "width", "steps", "lr_scale", "batch", "weight_decay", "dropout",
                "train_manifest", "token_manifest", "training_device", "torch"):
        if trained[key] != fit[key]:
            raise ValueError(f"The check run differs from D42's in {key}: {trained[key]} against {fit[key]}")
    actual = trained["models"]
    passed = actual == expected
    write_new(output, {"protocol": PROTOCOL, "check": "every sign +1, no swap negation: retrain of D42's original-set "
                       "seed-42 network", "expected": expected, "actual": actual, "passed": passed,
                       "d50": trained["d50"], "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D50 trainer check: {'bit for bit' if passed else 'DIFFERS'} "
          f"({actual.get('temporal_5400.pt', '')[:12]} against {expected['temporal_5400.pt'][:12]})", flush=True)
    return passed


# -- the x-odd columns ------------------------------------------------------------------------------

PARITY = PACKAGE / "screen_recipe" / "parity.json"


def x_odd(parity, columns):
    """Flags of the columns that change sign under the x mirror in a D47 parity record (sign map value -1)."""
    sign_map = parity["sign_map"]
    if list(sign_map["columns"]) != list(columns):
        raise ValueError("The parity record was measured on different columns")
    return [value == -1 for value in sign_map["signs"]["x"]]


def freeze_columns(source, output=None, parity_path=PARITY):
    """Record, once and before any candidate is trained, which columns take the attacking sign."""
    require_external_only()
    output = RECIPE / "columns.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; the column decision is recorded once")
    from dfl_submission import temporal_network as tn
    manifest = read(source / "manifest.json")
    columns = manifest["columns"]
    if parity_path.exists():
        parity = read(parity_path)
        flags = x_odd(parity, columns)
        neither = parity.get("neither", {}).get("x", [])
        status = "parity record read"
        parity_digest = digest(parity_path)
    else:
        flags, neither, status, parity_digest = [False] * len(columns), [], "no parity record yet", None
    permutation = np.asarray(tn.swap_indices(columns))
    if any(flags[i] != flags[j] for i, j in enumerate(permutation)):
        raise ValueError("The x-odd columns are not closed under the team swap")
    odd = [columns[i] for i, flag in enumerate(flags) if flag]
    write_new(output, {"protocol": PROTOCOL, "status": status, "parity_sha256": parity_digest, "flags": flags,
                       "x_odd_columns": odd, "x_neither_columns": neither, "columns_sha256": json_digest(columns),
                       "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D50 columns: {status}; {len(odd)} x-odd columns {odd}; x neither {neither}", flush=True)
    return flags


def column_flags(columns=None):
    record = read(RECIPE / "columns.json")
    if columns is not None and record["columns_sha256"] != json_digest(list(columns)):
        raise ValueError("The column decision was recorded on different columns")
    return np.asarray(record["flags"], dtype=bool) if any(record["flags"]) else None


# -- training ----------------------------------------------------------------------------------------

def raw_folder(model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / "raw" / model_set / folder


def export_folder(model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / "export" / model_set / folder


def train_networks(model_set, source, tokens, *, scratch=SCRATCH / "networks"):
    """Candidate C's two networks of one set, trained one at a time, then exported with the PyTorch check."""
    require_external_only()
    from dfl_submission import touch_trial
    passed_record("trainer_check.json")
    training_signs_passed()
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    signs_file = SCRATCH / f"signs_{model_set}.json"
    if digest(signs_file) != read(RECIPE / "signs.json")["files"][model_set]:
        raise ValueError(f"{signs_file} is not the file the sign check passed on")
    flags = column_flags(read(source / "manifest.json")["columns"])
    odd_file = None
    if flags is not None:
        odd_file = SCRATCH / "x_odd_columns.json"
        if not odd_file.exists():
            odd_file.write_text(json.dumps(flags.tolist()) + "\n")
        elif read(odd_file) != flags.tolist():
            raise ValueError(f"{odd_file} differs from the column decision")
    for folder, (seed, boost) in NETWORKS.items():
        output = export_folder(model_set, folder, scratch)
        if (output / "fit.json").exists():
            continue
        raw = raw_folder(model_set, folder, scratch)
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            raw.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-m", "dfl_submission.direction_tokens", "train", "--source", str(source),
                       "--tokens", str(tokens), "--signs", str(signs_file), "--out", str(raw), "--seed", str(seed),
                       "--boundary-boost", str(boost)]
            if odd_file is not None:
                command += ["--odd", str(odd_file)]
            wait_for_share(what=f"{model_set}/{folder}")
            subprocess.run(command, check=True, cwd=ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- forecasts -----------------------------------------------------------------------------------------

class Checkpoint:
    """A player-set network: candidate C (canonical tokens, swap negation) or D43's plain inference."""

    def __init__(self, folder, *, plain, odd=None):
        from dfl_submission import player_set as ps
        fit = read(folder / "fit.json")
        self.filename = f"temporal_{fit['steps']}.npz"
        self.checkpoint = ps.load(folder, self.filename)
        self.columns = list(self.checkpoint[1])
        self.plain, self.odd = plain, None if plain else odd

    def predict(self, frames, owner, player_tokens, length, sign):
        values = owner[self.columns].to_numpy(dtype=np.float32)
        if self.plain:
            return dt.predict(self.checkpoint, frames, values, self.columns, player_tokens, length, sign=1, negate=False)
        return dt.predict(self.checkpoint, frames, values, self.columns, player_tokens, length, sign=sign, negate=True,
                          odd=self.odd)


def candidate_identity(scratch=SCRATCH / "networks"):
    from dfl_submission import scale_trial
    return {"protocol": PROTOCOL, "candidate": "C",
            "networks": {key: {folder: digest(export_folder(key, folder, scratch) / "fit.json") for folder in NETWORKS}
                         for key in SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in SETS},
            "player_set_sha256": digest(PACKAGE / "player_set.py"),
            "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py"),
            "columns_sha256": digest(RECIPE / "columns.json"), "signs_sha256": digest(RECIPE / "signs.json")}


class Forecaster:
    """Decoder inputs of candidate C per half, with D43 recomputed through the same path on the way.

    The features, tokens and trees of each half and phase are computed once and shared by both models.
    D43's networks run with plain inference (sign +1, no swap negation: `player_set.predict` exactly);
    its results are logged for the bit-for-bit check against `data/d42/cache`.
    """

    def __init__(self, scratch=SCRATCH / "networks", gaps=None):
        from dfl_submission import competition_adaptation, long_context, player_trial, scale_trial
        self.gaps = {} if gaps is None else dict(gaps)
        reference, _, frozen = competition_adaptation.reference()
        self.previous, self.frozen = reference[2]["recipe"], frozen
        if self.previous["variant"] != "temporal_1800.npz" or frozen["recipe"]["variant"] != "seed_phase":
            raise ValueError("D50 expects D15's variants")
        self.trees = {role: long_context.load(scale_trial.TREES[key])[0] for role, key in scale_trial.ROLES.items()}
        self.roles = scale_trial.ROLES
        d43 = player_trial.network_folders()
        odd = column_flags()
        self.models = {
            "D43": {role: {folder: Checkpoint(d43[key] / folder, plain=True) for folder in NETWORKS}
                    for role, key in self.roles.items()},
            "C": {role: {folder: Checkpoint(export_folder(key, folder, scratch), plain=False, odd=odd) for folder in NETWORKS}
                  for role, key in self.roles.items()}}
        self.d43_log, self.signs = [], []

    def forecast(self, positions, ball, teams, pitch):
        from dfl_submission import diagnosis, ensemble_trial, long_context, motion_trial, player_set as ps
        from dfl_submission import retrain_trial, temporal_trial
        info = dt.attacking(positions, teams)
        if info["sign"] is None:
            key = tracking_key(positions, ball, teams, pitch)
            if key not in self.gaps:
                raise ValueError(f"A scored half has no attacking sign and no recorded decision ({info}); D50 stops")
            info = dict(info, sign=self.gaps[key], gap=True)
        elif not info["agrees"]:
            raise ValueError(f"D50's sign check fails on a scored half ({info}); D50 stops")
        self.signs.append(info)
        phases, baselines = [], {}
        for offset in ensemble_trial.PHASES:
            shifted_positions, shifted_ball = positions[offset:], ball[offset:]
            frames, owner, dead = long_context.features(shifted_positions, shifted_ball, teams, pitch)
            length = len(ball) - offset
            player_tokens = ps.tokens(shifted_positions, shifted_ball, teams, pitch, frames)
            if offset == 0:
                if not np.array_equal(frames, dt.long_context_frames(len(ball))):
                    raise ValueError("Long-context frames differ from the sign's frames")
                for role in self.roles:
                    baselines[role], _ = temporal_trial.forecast_features(self.trees[role], {}, frames, owner, dead, length)
            phases.append((offset, frames, owner, player_tokens, length))
        result = {}
        for name, roles in self.models.items():
            per_role = []
            for role in ("current", "candidate"):
                original, late = roles[role]["temporal_models"], roles[role]["seed_models"]
                _, frames, owner, player_tokens, length = phases[0]
                start_reference = (baselines[role], {ensemble_trial.LATE: original.predict(
                    frames, owner, player_tokens, length, info["sign"])})
                phase_seed = [tuple(np.pad(p, (offset, 0), mode="edge")
                                    for p in late.predict(frames, owner, player_tokens, length, info["sign"]))
                              for offset, frames, owner, player_tokens, length in phases]
                per_role.append((start_reference, {"seed_phase": ensemble_trial.mean(phase_seed)}))
            mixed = retrain_trial.mix(*per_role, self.frozen["weight"])
            labels = motion_trial.decode(mixed, self.previous, self.frozen["recipe"])
            signals = tuple(motion_trial.blend_signals(mixed, self.previous, self.frozen["recipe"]))
            if not np.array_equal(diagnosis.decode(signals, self.frozen["recipe"]), labels):
                raise ValueError("Decoding the forecast signals does not reproduce the labels")
            result[name] = (signals, labels)
        self.d43_log.append(result["D43"])
        return result["C"]

    def signals_of(self, _, positions, ball, teams, pitch):
        return self.forecast(positions, ball, teams, pitch)

    def check_d43(self, halves):
        """D43 recomputed through this path against its cached decoder inputs, half by half, bit for bit."""
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


# -- scoring ---------------------------------------------------------------------------------------------

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
    """The composite, D33's primary, the eight groups and the full trial of one set of decoder inputs at one cost."""
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


def reproduce_d43(scored):
    """D43's entries at costs 8 and 10 against D42's development record (both costs were scored there)."""
    from dfl_submission import player_trial, vendor_trial
    record = read(player_trial.RECIPE / "development.json")
    if record["models"] != player_trial.identity():
        raise ValueError("D43's networks differ from the ones D42's development scored")
    result = {}
    for entry in record["entries"]:
        cost = entry["point"]["owner_cost"]
        mine = scored[cost]
        result[str(cost)] = {"development": vendor_trial.canonical(mine["trial"]) == entry["trial"],
                             "metrica": vendor_trial.canonical(mine["metrica"]["means"]) == entry["metrica"]["means"],
                             "d24": vendor_trial.canonical(mine["d24"]["views"]) == entry["d24"]["views"]
                             and mine["d24"]["mean"] == entry["d24"]["candidate"]}
    if str(REFERENCE_COST) not in result:
        raise ValueError("D42's development record has no cost-10 entry")
    return result


class Visibility:
    """D46's ball visibility for every scored half, rebuilt from the external sources (`diagnosis_d43`)."""

    def __init__(self, dfl, skill, parsed, games, *, raw, halves):
        from dfl_submission import diagnosis_d43 as d46, metrica_views as mv
        self.d46 = d46
        rebuilt = d46.development_visibility(dfl, skill, parsed, games, raw=raw)
        metrica_rebuilt, rows = d46.metrica_visibility()
        if canonical_json(rows) != read(mv.RECORD)["visibility"]:
            raise ValueError("The rebuilt D24 views do not reproduce the D24 record's visibility counts")
        development, metrica, d24 = halves
        joined = development + metrica + d46.d24_grouped(d24)
        self.labels = {d46.half_key(half): half.labels for half in joined}
        visibility, self.frames = d46.attach(joined, rebuilt | d46.renamed(metrica_rebuilt))
        since = {key: d46.frames_since_seen(value) for key, value in visibility.items()}
        self.visible = d46.joined(joined, visibility, False)
        self.elapsed = d46.joined(joined, since, np.inf)

    def agreement(self, halves, recipe):
        """D46's owner agreement with the N = 3 target on frames both call live, seen / unseen, per group."""
        from dfl_submission import diagnosis
        development, metrica, d24 = halves
        joined = development + metrica + self.d46.d24_grouped(d24)
        keys = [self.d46.half_key(half) for half in joined]
        if sorted(keys) != sorted(self.labels) or not all(
                np.array_equal(half.labels, self.labels[key]) for key, half in zip(keys, joined, strict=True)):
            raise ValueError("The scored halves differ from the halves whose visibility was rebuilt")
        scored = diagnosis.matches(joined, lambda half: diagnosis.decode(half.signals, recipe))
        groups, _ = self.d46.measure_visibility(scored, self.visible, self.elapsed)
        return {group: {"seen": value["agreement"]["seen"], "unseen": value["agreement"]["unseen"],
                        "target_live_unseen_share": value["target_live_unseen_share"]}
                for group, value in groups.items()}


def d46_agreement_check(agreement):
    """D43's cost-10 agreement as computed here against D46's record (same functions, same halves)."""
    from dfl_submission import diagnosis_d43 as d46
    record = read(d46.OUTPUT)["visibility"]["groups"]
    return {group: canonical_json(agreement[group]["seen"]) == canonical_json(record[group]["agreement"]["seen"])
            and canonical_json(agreement[group]["unseen"]) == canonical_json(record[group]["agreement"]["unseen"])
            for group in record}


def carry(entries, base):
    """D50's carry rule: at C's best cost by composite (ties to the lower cost), composite >= D43's and no group down > 0.01."""
    best = max(entries, key=lambda e: (e["composite"], -e["owner_cost"]))
    changes = {name: best["groups"][name] - base["groups"][name] for name in GROUPS}
    worst = min(changes, key=changes.get)
    passed = bool(best["composite"] >= base["composite"] and all(v >= -LOSS for v in changes.values()))
    return {"owner_cost": best["owner_cost"], "composite": best["composite"],
            "composite_change": best["composite"] - base["composite"], "group_changes": changes,
            "worst_group": worst, "worst_change": changes[worst], "passed": passed}


def brief(entry):
    keys = ("owner_cost", "groups", "composite", "primary", "agreement")
    return {key: entry[key] for key in keys if key in entry}


def develop(dfl, skill, parsed, *, raw, cache_root=SCRATCH / "cache", scratch=SCRATCH / "networks"):
    """D43's reproduction, then candidate C on development at costs 8, 10 and 12; one record per cost."""
    require_external_only()
    from dfl_submission import diagnosis, metrica_sample as ms, metrica_views as mv, refine_trial, vendor_trial
    gaps = scoring_signs()
    signs = read(RECIPE / "signs.json")
    trainer = passed_record("trainer_check.json")
    outputs = {cost: RECIPE / f"cost_{cost}.json" for cost in COSTS}
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"{path} exists; D50 records are written once")
    for key in SETS:
        for folder in NETWORKS:
            if not (export_folder(key, folder, scratch) / "fit.json").exists():
                raise ValueError(f"Candidate C's {key}/{folder} network is not exported yet")
    wait_for_share(what="development scoring")
    games = vendor_trial.development()
    _, _, truths = mv.reference_scores()

    d43 = d43_halves(dfl, skill, parsed, raw=raw)
    d43_scored = {cost: evaluate(*d43, truths, cost) for cost in COSTS}
    reproduction = reproduce_d43(d43_scored)
    print(f"D50 reproduction of D43: {reproduction}", flush=True)
    if not all(all(item.values()) for item in reproduction.values()):
        raise ValueError(f"D43 does not reproduce its records {reproduction}; nothing is scored")
    screen_reference = PACKAGE / "screen_recipe" / "reference.json"
    composite_check = None
    if screen_reference.exists():
        stored = read(screen_reference)
        base = d43_scored[REFERENCE_COST]
        composite_check = bool(stored["groups"] == base["groups"] and stored["composite"] == base["composite"]
                               and stored["primary"] == base["primary"])
        print(f"D50 composite against D47's reference record: {'same' if composite_check else 'DIFFERS'}", flush=True)
        if not composite_check:
            raise ValueError("D43's composite here differs from D47's reference record; nothing is scored")

    identity = dict(candidate_identity(scratch), sign_gap_sha256=digest(RECIPE / "sign_gap.json") if gaps else None)
    variant = f"D50 C {json_digest(identity)[:16]}"
    forecaster = Forecaster(scratch, gaps)
    key = dict(identity, teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=raw,
               skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games))

    def produce():
        yield from diagnosis.external_halves(dfl, skill, "development", None, raw=raw, signals_of=forecaster.signals_of)
        yield from diagnosis.pff_halves(parsed, games, None, signals_of=forecaster.signals_of)

    checks, scored_signs = {}, {}

    def path_check(part, halves):
        """D43's bit-for-bit path check of one part, kept beside its cache so a re-run from cache keeps it."""
        stored = cache_root / f"{part}_d43_path_check.json"
        count = forecaster.check_d43(halves)
        seen_signs, forecaster.signs = forecaster.signs, []
        if count:
            stored.write_text(json.dumps({"halves": count, "cache": digest(cache_root / part / "manifest.json"),
                                          "signs": {"halves": len(seen_signs),
                                                    "negative": sum(1 for s in seen_signs if s["sign"] < 0),
                                                    "gap_halves": sum(1 for s in seen_signs if s.get("gap")),
                                                    "smallest_abs_d_m": min(abs(s["d"]) for s in seen_signs
                                                                            if s["d"] is not None)},
                                          "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")}) + "\n")
        elif not stored.exists() or read(stored)["cache"] != digest(cache_root / part / "manifest.json"):
            raise ValueError(f"The {part} cache has no D43 path check; remove the cache and forecast again")
        checks[part] = read(stored)["halves"]
        scored_signs[part] = read(stored)["signs"]

    development = diagnosis.collect(cache_root / "development", key, produce)
    path_check("development", d43[0])
    _, metrica = ms.reference_halves("D15", cache=cache_root / "metrica", model="D50", signals_of=forecaster.signals_of,
                                     variant=variant)
    path_check("metrica", d43[1])
    views = mv.view_halves(cache=cache_root / "d24", signals_of=forecaster.signals_of, variant=variant)
    path_check("d24", d43[2])
    print(f"D50 D43 recomputed through the D50 path, bit for bit on {checks} halves", flush=True)
    candidate = (development, metrica, views)

    visibility = Visibility(dfl, skill, parsed, games, raw=raw, halves=d43)
    c_scored = {}
    for cost in COSTS:
        c_scored[cost] = evaluate(*candidate, truths, cost)
        c_scored[cost]["agreement"] = visibility.agreement(candidate, c_scored[cost]["recipe"])
        d43_scored[cost]["agreement"] = visibility.agreement(d43, d43_scored[cost]["recipe"])
    agreement_check = d46_agreement_check(d43_scored[REFERENCE_COST]["agreement"])
    print(f"D50 D43 agreement against D46's record: {agreement_check}", flush=True)
    if not all(agreement_check.values()):
        raise ValueError("D43's owner agreement here differs from D46's record; nothing is written")

    base = d43_scored[REFERENCE_COST]
    entries = []
    for cost in COSTS:
        entry = c_scored[cost]
        entry["changes"] = refine_trial.changes(base["trial"], entry["trial"])
        entry["composite_change"] = entry["composite"] - base["composite"]
        entry["primary_change"] = entry["primary"] - base["primary"]
        entry["group_changes"] = {name: entry["groups"][name] - base["groups"][name] for name in GROUPS}
        entries.append(entry)
    rule = carry(entries, base)
    exports = {f"{key}/{folder}": read(export_folder(key, folder, scratch) / "fit.json")["export_max_logit_difference"]
               for key in SETS for folder in NETWORKS}
    d49 = sorted(str(p.relative_to(ROOT)) for p in (PACKAGE / "hidden_recipe").glob("*.json")) \
        if (PACKAGE / "hidden_recipe").exists() else []
    for cost, entry in zip(COSTS, entries, strict=True):
        write_new(outputs[cost], {
            "protocol": PROTOCOL, "candidate": "C", "owner_cost": cost, "entry": entry,
            "d43_same_cost": brief(d43_scored[cost]),
            "reference": {"name": "D43", **brief(base)}, "carry": rule,
            "checks": {"signs": {"passed": signs["passed"], "halves": signs["halves"],
                                 "smallest_abs_d_m": signs["smallest_abs_d_m"], "sha256": digest(RECIPE / "signs.json")},
                       "scored_halves_signs": scored_signs,
                       "trainer": {"passed": trainer["passed"], "sha256": digest(RECIPE / "trainer_check.json")},
                       "columns": read(RECIPE / "columns.json"), "d43_reproduction": reproduction,
                       "sign_gap": read(RECIPE / "sign_gap.json") if gaps else None,
                       "d43_composite_against_d47_reference": composite_check, "d43_path_bit_for_bit": checks,
                       "d43_agreement_against_d46": agreement_check, "rebuilt_frames": len(visibility.frames)},
            "d49_long_hidden_guard_views": {"records_present": d49, "scored": False},
            "identity": identity, "exports": exports,
            "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                       "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": raw},
            "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    summary()
    return rule


def refuse():
    raise ValueError("The D49 guard cache is missing; nothing is forecast for D43 here")
    yield  # pragma: no cover


def guards(dfl, *, raw, cache_root=SCRATCH / "cache", scratch=SCRATCH / "networks", output=None):
    """C on D49's long-hidden guard views (reported beside D50's rule, not part of it), written once.

    D43 is recomputed through the D50 path on the way and must equal D49's cached guard inputs bit for
    bit, and D43's cost-10 guard scores must equal D49's reference record.
    """
    require_external_only()
    from dfl_submission import hidden_trial as h49, hidden_views as hv, hyper_trial as ht, player_trial
    output = RECIPE / "guards.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D50 records are written once")
    gaps = scoring_signs()
    reference = read(h49.RECIPE / "reference.json")
    wait_for_share(what="guard-view scoring")
    d43_identity = h49.guard_identity(player_trial.identity(), raw=raw)
    if canonical_json(d43_identity) != reference["guard_identity"]:
        raise ValueError("D49's guard identity differs from its reference record")
    d43 = diagnosis_collect(h49.DATA / "cache" / "guards_d43", d43_identity, refuse)
    tracking = h49.guard_tracking(dfl, raw=raw)
    identity = dict(candidate_identity(scratch), sign_gap_sha256=digest(RECIPE / "sign_gap.json") if gaps else None,
                    guard=d43_identity)
    forecaster = Forecaster(scratch, gaps)

    def produce():
        from dfl_submission import diagnosis
        for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(tracking.items()):
            signals, decoded = forecaster.forecast(positions, ball, teams, pitch)
            yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)
            print(f"Forecast guard {source} {match} half {index + 1}", flush=True)

    halves = diagnosis_collect(cache_root / "guards", identity, produce)
    stored = cache_root / "guards_d43_path_check.json"
    count = forecaster.check_d43(d43)
    if count:
        stored.write_text(json.dumps({"halves": count, "cache": digest(cache_root / "guards" / "manifest.json"),
                                      "signs": [{k: s.get(k) for k in ("sign", "d", "agrees", "gap")} for s in forecaster.signs]})
                          + "\n")
    elif not stored.exists() or read(stored)["cache"] != digest(cache_root / "guards" / "manifest.json"):
        raise ValueError("The guard cache has no D43 path check; remove it and forecast again")
    path_check = read(stored)
    d43_scores = {cost: h49.guard_scores(d43, ht.recipe_at(cost)) for cost in COSTS}
    reproduced = canonical_json(d43_scores[REFERENCE_COST]) == reference["reference"]["guards"]
    print(f"D50 guards: D43 path bit for bit on {path_check['halves']} halves; D43 reproduces D49's guard reference: "
          f"{reproduced}", flush=True)
    if not reproduced:
        raise ValueError("D43 does not reproduce D49's guard reference; nothing is written")
    c_scores = {cost: h49.guard_scores(halves, ht.recipe_at(cost)) for cost in COSTS}
    base = d43_scores[REFERENCE_COST]
    entries = {str(cost): {"c": c_scores[cost], "d43": d43_scores[cost],
                           "change_against_d43_cost10": {key: c_scores[cost][key] - base[key]
                                                         for key in ("a_dfl", "b_metrica", "mean")}}
               for cost in COSTS}
    write_new(output, {"protocol": PROTOCOL, "note": "D49's long-hidden guard views, reported beside D50's carry rule",
                       "entries": entries, "reference": {"name": "D43", "owner_cost": REFERENCE_COST, "guards": base},
                       "checks": {"d43_path_bit_for_bit": path_check["halves"], "d43_reproduces_d49_reference": reproduced,
                                  "signs": path_check["signs"]},
                       "identity": identity, "d49_reference_sha256": digest(h49.RECIPE / "reference.json"),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    for cost in COSTS:
        item = entries[str(cost)]
        print(f"D50 guards cost {cost}: D43 {item['d43']['mean']:.4f} (DFL {item['d43']['a_dfl']:.4f}, Metrica "
              f"{item['d43']['b_metrica']:.4f}); C {item['c']['mean']:.4f} (DFL {item['c']['a_dfl']:.4f}, Metrica "
              f"{item['c']['b_metrica']:.4f}); C - D43@10 {item['change_against_d43_cost10']['mean']:+.4f}", flush=True)
    return entries


def diagnosis_collect(cache, identity, produce):
    from dfl_submission import diagnosis
    return diagnosis.collect(cache, identity, produce)


# -- summary ---------------------------------------------------------------------------------------------

SHORT = {"dfl_clean": "clean", "dfl_masked": "masked", "dfl_dead_carried": "dead carried", "dfl_team_swap": "swap",
         "skillcorner": "SkillCorner", "pff": "PFF", "metrica": "Metrica", "d24": "D24"}


def agreement_text(entry):
    parts = []
    for group in AGREEMENT_GROUPS:
        value = entry["agreement"][group]
        parts.append(f"{value['unseen']['agreement']:.3f}/{value['seen']['agreement']:.3f}")
    return " | ".join(parts)


def summary(folder=RECIPE):
    rows = ["| Model | Cost | Composite | Δ | Primary | Δ | " + " | ".join(SHORT[g] for g in GROUPS)
            + " | " + " | ".join(f"{g} unseen/seen" for g in AGREEMENT_GROUPS) + " |",
            "|---|---:|---:|---:|---:|---:|" + "---:|" * len(GROUPS) + "---|" * len(AGREEMENT_GROUPS)]
    records = {cost: read(folder / f"cost_{cost}.json") for cost in COSTS if (folder / f"cost_{cost}.json").exists()}
    if not records:
        print("No D50 record yet")
        return ""
    base = next(iter(records.values()))["reference"]
    for cost, record in records.items():
        d43 = record["d43_same_cost"]
        rows.append(f"| D43 | {cost} | {d43['composite']:.5f} | {d43['composite'] - base['composite']:+.4f} | "
                    f"{d43['primary']:.5f} | {d43['primary'] - base['primary']:+.4f} | "
                    + " | ".join(f"{d43['groups'][g]:.4f}" for g in GROUPS) + " | " + agreement_text(d43) + " |")
    for cost, record in records.items():
        entry = record["entry"]
        rows.append(f"| C | {cost} | {entry['composite']:.5f} | {entry['composite_change']:+.4f} | {entry['primary']:.5f} | "
                    f"{entry['primary_change']:+.4f} | " + " | ".join(f"{entry['groups'][g]:.4f}" for g in GROUPS)
                    + " | " + agreement_text(entry) + " |")
    rule = next(iter(records.values()))["carry"]
    lines = ["\n".join(rows), "",
             f"Carry rule: best cost {rule['owner_cost']}, composite {rule['composite_change']:+.5f}, worst group "
             f"{rule['worst_group']} {rule['worst_change']:+.4f} -> {'carried to D48' if rule['passed'] else 'not carried'}"]
    text = "\n".join(lines)
    print(text)
    return text


# -- command line ------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    signs = commands.add_parser("signs")
    signs.add_argument("--set", choices=SETS, required=True)
    signs.add_argument("--source", type=Path, required=True)
    signs.add_argument("--tokens", type=Path, required=True)
    for command in (signs, dev := commands.add_parser("dev-signs"), develop_command := commands.add_parser("develop")):
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
    dev.add_argument("--parsed", type=Path, required=True)
    develop_command.add_argument("--parsed", type=Path, required=True)
    commands.add_parser("sign-record")
    gap = commands.add_parser("sign-gap")
    gap.add_argument("--rule", choices=GAP_RULES, required=True)
    gap.add_argument("--parsed", type=Path, required=True)
    trainer = commands.add_parser("check-trainer")
    trainer.add_argument("--source", type=Path, required=True)
    trainer.add_argument("--tokens", type=Path, required=True)
    columns = commands.add_parser("columns")
    columns.add_argument("--source", type=Path, required=True)
    networks = commands.add_parser("networks")
    networks.add_argument("--set", choices=SETS, required=True)
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--tokens", type=Path, required=True)
    guard = commands.add_parser("guards")
    guard.add_argument("--dfl", type=Path, required=True)
    guard.add_argument("--dfl-raw", action="store_true")
    commands.add_parser("summary")
    args = parser.parse_args()
    require_external_only()
    if args.command == "signs":
        training_signs(args.set, args.source, args.tokens, args.dfl, args.skill, SCRATCH / f"signs_{args.set}.json",
                       raw=args.dfl_raw)
    elif args.command == "dev-signs":
        development_signs(args.dfl, args.skill, args.parsed, SCRATCH / "signs_development.json", raw=args.dfl_raw)
    elif args.command == "sign-record":
        sign_record()
    elif args.command == "sign-gap":
        sign_gap(args.rule, args.parsed)
    elif args.command == "check-trainer":
        check_trainer(args.source, args.tokens)
    elif args.command == "columns":
        freeze_columns(args.source)
    elif args.command == "networks":
        train_networks(args.set, args.source, args.tokens)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "guards":
        guards(args.dfl, raw=args.dfl_raw)
    else:
        summary()


if __name__ == "__main__":
    main()
