"""D57: SkillCorner's ball height, the SZ training views and the SkillCorner air view (DECISIONS.md D57).

The reader path. `skillcorner.parse_match` keeps the ball's x and y and drops `ball_data.z`. `parse_height` reads
the same tracking file again and keeps z on the frames where the existing reader keeps x and y (a detected ball
with x and y), NaN elsewhere; frames whose x or y lies outside the pitch plus 5 m lose x, y and z, as in the
existing reader. x, y and z then go through `skillcorner.at_25hz`, the existing reader's resampling, so z sits on
the same 25 Hz grid. `reader` checks, on the twelve training and four development matches (never the holdout),
that this ball's x and y equal the existing reader's and the stored parsed halves' on every frame.

SZ's views. D56's S, with one change in the SkillCorner situation views: the ball is also hidden on detected
frames whose height is above 1.5 m (`situation_views.air_mask` on the resampled height). The crowd rule, the
delay and the random part are D56's (`situation_train`), drawn from D56's seed for the half in D56's order, so the
visibility and the random part are D56's own draws, and each run of the situation flags that is one of D56's runs
keeps D56's delay; a run that the air rule creates or changes takes its delay (0-25 frames, uniform) from a new
fixed seed. The DFL views, D49's records and every other D56 record are D56's, reproduced byte for byte. SZ's
SkillCorner views keep D56's record names and order, so the trainer sees the same records in the same order and
only those 24 files differ.

The SkillCorner air view (reported only): the four development matches as stored, with the ball hidden on
detected frames above 1.5 m; nothing else changes.

Commands (each with `MTAH_DATA_DIR` set to a nonexistent path):
- `reader --skill DIR`: the height reader on the sixteen matches, checked frame for frame; heights saved to
  `data/d57/heights/`; `stack_recipe/d57_reader.json` written once.
- `prepare --set current|augmented --dfl DIR --skill DIR`: every record of D56's folders re-extracted and checked
  byte for byte, SZ's 24 SkillCorner views made, the folders written to `data/d57/` and
  `stack_recipe/d57_prepare_<set>.json` once.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np

from dfl_submission import situation_train as s56, situation_views as sv, skillcorner
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stack_recipe"
SCRATCH = ROOT / "data" / "d57"
D56 = ROOT / "data" / "d56"
HEIGHTS = SCRATCH / "heights"
READER = RECIPE / "d57_reader.json"
PROTOCOL = "DECISIONS.md D57"
MATCHES = (*skillcorner.TRAIN, *skillcorner.DEVELOPMENT)     # never the holdout
AIR_SEEDS = {"current": 1130000, "augmented": 1140000}      # delays of runs the air rule makes or changes
AIR_VIEW = "air"
LIVE = (pm.H, pm.A)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


# -- the reader path ----------------------------------------------------------------------------------

def raw_paths(skill, match):
    """(match.json, phases_of_play.csv, tracking_extrapolated.jsonl) of a match, each checked against the manifest."""
    if int(match) not in MATCHES:
        raise ValueError(f"D57 reads the training and development matches only, not {match}")
    manifest = read(Path(skill) / "manifest.json")
    if manifest["revision"] != skillcorner.REVISION:
        raise ValueError("SkillCorner revision differs")
    paths = []
    for suffix in ("match.json", "phases_of_play.csv", "tracking_extrapolated.jsonl"):
        name = f"{match}_{suffix}"
        path = Path(skill) / "raw" / name
        if skillcorner.identity(path) != tuple(manifest["files"][name]):
            raise ValueError(f"SkillCorner raw file differs from the manifest: {name}")
        paths.append(path)
    return paths


def parse_height(paths):
    """[{"ball", "z", "facts"} per half]: the ball's x and y as `parse_match` makes them, and z beside them.

    Native z is float32 like x and y, NaN wherever the native ball is NaN (undetected, missing x or y, outside the
    pitch plus 5 m) or the record has no z; both go through `skillcorner.at_25hz` (float32 out).
    """
    metadata = json.loads(Path(paths[0]).read_text())
    pitch = np.array([metadata["pitch_length"], metadata["pitch_width"]])
    periods = {period["period"]: period for period in metadata["match_periods"]}
    halves = []
    for number in (1, 2):
        period = periods[number]
        start, stop = period["start_frame"], period["end_frame"]
        n = stop - start
        if not 24000 <= n <= 42000:
            raise ValueError(f"Unexpected native half length: {n}")
        halves.append((start, stop, np.full((n, 2), np.nan, np.float32), np.full(n, np.nan, np.float32),
                       {"detected_without_z": 0, "undetected_with_z": 0}))
    with Path(paths[2]).open() as handle:
        for line in handle:
            row = json.loads(line)
            period = row["period"]
            if period not in (1, 2):
                continue
            start, stop, ball, height, facts = halves[period - 1]
            frame = row["frame"]
            if not start <= frame < stop:
                continue
            i = frame - start
            observed = row["ball_data"]
            if observed and observed["is_detected"] and observed["x"] is not None and observed["y"] is not None:
                ball[i] = observed["x"], observed["y"]
                z = observed.get("z")
                if z is None:
                    facts["detected_without_z"] += 1
                    height[i] = np.nan
                else:
                    height[i] = z
            elif observed and observed.get("z") is not None:
                facts["undetected_with_z"] += 1
    result = []
    for _, _, ball, height, facts in halves:
        ball[np.any(np.abs(ball) > pitch / 2 + 5, axis=1)] = np.nan
        detected = np.isfinite(ball).all(axis=1)
        height[~detected] = np.nan
        native = {"native_frames": int(len(ball)), "native_detected": int(detected.sum()),
                  "native_detected_with_z": int(np.isfinite(height).sum()),
                  "native_detected_above_air_height": int((np.nan_to_num(height, nan=-np.inf) > sv.AIR_HEIGHT).sum())}
        ball25, _ = skillcorner.at_25hz(ball)
        z25, _ = skillcorner.at_25hz(height[:, None])
        result.append({"ball": ball25, "z": z25[:, 0], "facts": facts | native})
    return result


def reader_check(new, existing, stored):
    """Per half: the new ball's x and y equal the existing reader's and the stored parsed half's, and z's facts."""
    result = {}
    for index, (half, old, kept) in enumerate(zip(new, existing, stored, strict=True)):
        ball, z = half["ball"], half["z"]
        seen = np.isfinite(ball).all(axis=1)
        labels = np.asarray(kept["labels"])
        live = np.isin(pm.label_codes(labels), LIVE)
        air = sv.air_mask(z)
        result[str(index)] = {
            "frames": int(len(ball)),
            "ball_xy_equal_existing_reader": bool(ball.dtype == old["ball"].dtype and ball.shape == old["ball"].shape
                                                  and np.array_equal(ball, old["ball"], equal_nan=True)),
            "ball_xy_equal_stored_half": bool(ball.dtype == kept["ball"].dtype and ball.shape == kept["ball"].shape
                                              and np.array_equal(ball, kept["ball"], equal_nan=True)),
            "z_on_the_same_grid": bool(z.shape == (len(ball),)),
            "z_only_where_the_ball_is_seen": bool(not np.isfinite(z[~seen]).any()),
            "seen_frames": int(seen.sum()), "seen_frames_without_z": int((seen & ~np.isfinite(z)).sum()),
            "seen_share_above_air_height": float(air.sum() / seen.sum()) if seen.any() else None,
            "live_share_above_air_height": float((air & live).sum() / live.sum()) if live.any() else None,
            "frames_above_air_height": int(air.sum()),
            "median_z_seen_m": float(np.median(z[seen & np.isfinite(z)])) if (seen & np.isfinite(z)).any() else None,
            "facts": half["facts"]}
    return result


def reader_passed(check):
    return bool(all(part["ball_xy_equal_existing_reader"] and part["ball_xy_equal_stored_half"]
                    and part["z_on_the_same_grid"] and part["z_only_where_the_ball_is_seen"] for part in check.values()))


def stored_halves(skill, match):
    """The stored parsed halves of a match (the ones training reads), identity-checked against the manifest."""
    manifest = read(Path(skill) / "manifest.json")
    result = []
    for i in (1, 2):
        name = f"{match}_{i}.npz"
        path = Path(skill) / name
        if skillcorner.identity(path) != tuple(manifest["parsed"][name]):
            raise ValueError(f"SkillCorner input identity differs: {name}")
        with np.load(path, allow_pickle=False) as item:
            result.append({"ball": item["ball"], "labels": item["labels"]})
    return result


def read_match(skill, match):
    """(check, z per half) of one match: the new reader against the existing reader and the stored halves."""
    paths = raw_paths(skill, match)
    existing = skillcorner.parse_match(paths)
    new = parse_height(paths)
    check = reader_check(new, existing, stored_halves(skill, match))
    return check, [half["z"] for half in new]


def reader(skill, *, output=READER, heights=HEIGHTS, workers=2):
    """The height reader on the sixteen matches, checked frame for frame; heights saved; written once."""
    require_external_only()
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D57's reader check is written once")
    heights.mkdir(parents=True, exist_ok=True)
    checks, files = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(MATCHES, pool.map(read_match, [skill] * len(MATCHES), MATCHES)))
    for match in MATCHES:
        check, z = results[match]
        checks[str(match)] = check
        path = heights / f"{match}.npz"
        if path.exists():
            raise FileExistsError(f"{path} exists")
        np.savez_compressed(path, z_0=z[0], z_1=z[1])
        files[str(match)] = digest(path)
        print(f"D57 reader {match}: ball x and y {'equal' if reader_passed(check) else 'DIFFER'}; "
              + "; ".join(f"half {int(i) + 1}: {p['frames']} frames, seen {p['seen_frames']}, seen without z "
                          f"{p['seen_frames_without_z']}, seen above 1.5 m {p['seen_share_above_air_height']:.3f}"
                          for i, p in check.items()), flush=True)
    passed = bool(all(reader_passed(check) for check in checks.values()))
    split = {"train": [str(m) for m in skillcorner.TRAIN], "development": [str(m) for m in skillcorner.DEVELOPMENT]}
    write_new(output, {"protocol": PROTOCOL, "passed": passed, "checks": checks, "heights": files, "splits": split,
                       "summary": reader_summary(checks, split),
                       "skill_manifest_sha256": digest(Path(skill) / "manifest.json"),
                       "code": {name: digest(PACKAGE / name) for name in ("skill_height.py", "skillcorner.py",
                                                                          "situation_views.py")},
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    if not passed:
        raise SystemExit("D57 stops: the height reader differs from the existing reader on a match")
    return passed


def reader_summary(checks, split):
    """Pooled per split: seen frames above 1.5 m, live frames above 1.5 m, seen frames without z."""
    result = {}
    for name, matches in split.items():
        parts = [part for match in matches for part in checks[match].values()]
        seen = sum(part["seen_frames"] for part in parts)
        result[name] = {"halves": len(parts), "seen_frames": seen,
                        "seen_share_above_air_height": sum(p["frames_above_air_height"] for p in parts) / seen,
                        "seen_frames_without_z": sum(p["seen_frames_without_z"] for p in parts),
                        "per_match_seen_share_above_air_height": {
                            match: sum(p["frames_above_air_height"] for p in checks[match].values())
                            / sum(p["seen_frames"] for p in checks[match].values()) for match in matches}}
    return result


def heights_of(record_path=READER, heights=HEIGHTS):
    """{(match, half index): z at 25 Hz} from the passed reader check, after checking the files' digests."""
    record = read(record_path)
    if record["passed"] is not True:
        raise SystemExit("D57 stops: the reader check did not pass")
    result = {}
    for match, value in record["heights"].items():
        path = Path(heights) / f"{match}.npz"
        if digest(path) != value:
            raise ValueError(f"{path} differs from the checked heights")
        with np.load(path, allow_pickle=False) as item:
            for i in (0, 1):
                result[match, i] = item[f"z_{i}"]
    return result


# -- SZ's hide --------------------------------------------------------------------------------------

def height_hide(air, crowd, seed, air_seed):
    """(hidden, parts, visibility, facts): D56's SkillCorner hide with the air rule added.

    D56's draws come first, from `seed` in D56's order (visibility, a delay per crowd run, the two-state random
    part), and are checked against `situation_train.situation_hide`. Each run of air | crowd that is one of D56's
    crowd runs keeps its D56 delay; every other run draws a delay from `air_seed`, in time order.
    """
    air, crowd = np.asarray(air, dtype=bool), np.asarray(crowd, dtype=bool)
    rng = np.random.default_rng(seed)
    visibility = float(rng.uniform(*s56.VISIBILITY))
    crowd_starts, crowd_lengths = sv.runs_of(crowd)
    crowd_delays = rng.integers(s56.DELAY[0], s56.DELAY[1] + 1, size=len(crowd_starts))
    random = s56.two_state(len(crowd), visibility, rng)
    d56_flags = crowd.copy()
    for start, length, delay in zip(crowd_starts, crowd_lengths, crowd_delays):
        d56_flags[start + length:start + length + delay] = True
    d56_hidden, d56_parts, d56_visibility = s56.situation_hide(np.zeros(len(crowd), dtype=bool), crowd,
                                                               np.random.default_rng(seed))
    if (d56_visibility != visibility or not np.array_equal(d56_hidden, d56_flags | random)
            or not np.array_equal(d56_parts["random"], random)):
        raise ValueError("SZ's draws do not reproduce D56's hide")
    known = {(int(s), int(l)): int(d) for s, l, d in zip(crowd_starts, crowd_lengths, crowd_delays)}
    flags = air | crowd
    starts, lengths = sv.runs_of(flags)
    new = [(int(s), int(l)) not in known for s, l in zip(starts, lengths)]
    drawn = iter(np.random.default_rng(air_seed).integers(s56.DELAY[0], s56.DELAY[1] + 1, size=int(sum(new))))
    extended = flags.copy()
    for start, length, fresh in zip(starts, lengths, new):
        delay = int(next(drawn)) if fresh else known[int(start), int(length)]
        extended[start + length:start + length + delay] = True
    parts = {"air": air, "crowd": crowd, "delay": extended & ~flags, "random": random}
    facts = {"runs": int(len(starts)), "runs_with_d56_delay": int(len(starts) - sum(new)), "runs_with_new_delay": int(sum(new)),
             "d56_hidden": d56_hidden}
    return extended | random, parts, visibility, facts


def air_seed(model_set, order, index):
    return AIR_SEEDS[model_set] + 10 * order + index


def skill_view(positions, ball, z, model_set, order, index):
    """(view ball, visibility, parts, facts) of one SkillCorner training half: D56's view plus the air rule."""
    crowd = sv.crowd_mask(positions, ball)
    air = sv.air_mask(z)
    hidden, parts, visibility, facts = height_hide(air, crowd, s56.skill_seed(model_set, order, index),
                                                   air_seed(model_set, order, index))
    view_ball = np.array(ball, dtype=np.float32, copy=True)
    view_ball[hidden] = np.nan
    return view_ball, visibility, parts, facts


def training_view(model_set, half, *, dfl_height=None, skill_height=None, swaps=True, skill_order=None):
    """(view half, visibility, parts, facts or None) of one training half; DFL halves get D56's view unchanged."""
    if half.source == "dfl":
        made = s56.training_view(model_set, half, height=dfl_height, swaps=swaps, skill_order=skill_order)
        return None if made is None else (*made, None)
    if half.source == "skillcorner":
        if skill_order is None:
            raise ValueError("A SkillCorner half needs its order among the SkillCorner matches")
        if skill_height is None or len(skill_height) != len(half.ball):
            raise ValueError(f"A SkillCorner training half needs its heights on the same frames: {half.key}")
        ball, visibility, parts, facts = skill_view(np.asarray(half.positions, dtype=np.float32), half.ball, skill_height,
                                                    model_set, skill_order, half.index)
        return replace(half, view=s56.VIEW, ball=ball), visibility, parts, facts
    raise ValueError(f"Unknown training source {half.source!r}")


def added_by_height(items):
    """Pooled: frames SZ hides that D56's view leaves seen, and the seen frames above 1.5 m (before the hide)."""
    frames = sum(len(ball) for ball, _, _, _, _ in items)
    added = sum(int((~np.isfinite(ball).all(axis=1) & ~d56).sum()) for ball, d56, _, _, _ in items)
    seen = [np.isfinite(np.asarray(original, dtype=np.float64)).all(axis=1) for _, _, _, original, _ in items]
    above = sum(int((s & parts["air"]).sum()) for s, (_, _, parts, _, _) in zip(seen, items))
    live_seen = [s & np.isin(np.asarray(labels), LIVE) for s, (_, _, _, _, labels) in zip(seen, items)]
    return {"frames_hidden_beyond_d56_view": added, "share_hidden_beyond_d56_view": added / frames if frames else None,
            "seen_frames_before_the_hide": int(sum(s.sum() for s in seen)),
            "seen_share_above_air_height": above / sum(int(s.sum()) for s in seen) if seen else None,
            "live_seen_share_above_air_height": sum(int((l & p["air"]).sum()) for l, (_, _, p, _, _) in zip(live_seen, items))
            / sum(int(l.sum()) for l in live_seen) if live_seen else None}


# -- prepare --------------------------------------------------------------------------------------

def d56_folders(model_set):
    return D56 / f"sequences_{model_set}", D56 / f"tokens_{model_set}", D56 / f"signs_{model_set}.json"


def d56_prepared(model_set, recipe=RECIPE):
    """D56's passed preparation record of the set, with its folders and sign file unchanged."""
    record = read(recipe / f"d56_prepare_{model_set}.json")
    sequences, tokens, signs = d56_folders(model_set)
    if (record["passed"] is not True or record["sequences_manifest"] != digest(sequences / "manifest.json")
            or record["tokens_manifest"] != digest(tokens / "manifest.json") or record["signs_file"] != digest(signs)):
        raise ValueError(f"D56's {model_set} folders are not the ones its preparation check passed")
    return record


def prepare(model_set, dfl, skill, *, scratch=SCRATCH, recipe=RECIPE):
    """D56's records re-extracted byte for byte, SZ's 24 SkillCorner views made; folders and record written once."""
    require_external_only()
    from dfl_submission import direction_tokens as dt, external_benchmark as benchmark, hidden_trial as h49
    from dfl_submission import hidden_views as hv, touch_trial
    if model_set not in s56.SETS:
        raise ValueError(f"Unknown set {model_set!r}")
    output = recipe / f"d57_prepare_{model_set}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D57 preparation is recorded once")
    d56 = d56_prepared(model_set, recipe)
    d49 = read(h49.RECIPE / f"prepare_{model_set}.json")
    d56_sequences, d56_tokens, d56_signs_path = d56_folders(model_set)
    manifest, token_manifest = read(d56_sequences / "manifest.json"), read(d56_tokens / "manifest.json")
    d56_signs = read(d56_signs_path)
    records = manifest["records"]
    stored, views49 = records[:d49["stored_records"]], records[d49["stored_records"]:d49["stored_records"] + d49["new_records"]]
    situation = records[d49["stored_records"] + d49["new_records"]:]
    if len(situation) != d56["new_records"] or len(records) != d56["d49_records"] + d56["new_records"]:
        raise ValueError("D56's manifest does not hold D49's records and then its 32 views")
    by_name = {record["file"]: record for record in records}
    dfl_heights, skill_heights = s56.heights_of(), heights_of()
    sequences, tokens = scratch / f"sequences_{model_set}", scratch / f"tokens_{model_set}"
    sequences.mkdir(parents=True, exist_ok=False)
    tokens.mkdir(parents=True, exist_ok=False)
    swaps = bool(d56["choices"]["swaps"])
    order = benchmark.SKILL_SPLITS["all"]
    compared, replaced, new_entries, items_sz, items_d56, visibilities, facts_all = {}, {}, {}, [], [], {}, {}

    def check_record(name, half):
        record = by_name[name]
        if digest(d56_sequences / name) != record["sha256"] or digest(d56_tokens / name) != token_manifest["records"][name]:
            raise ValueError(f"A D56 file differs from its manifest: {name}")
        frames, columns, values, targets, player_tokens = s56.extract(half)
        if columns != manifest["columns"]:
            raise ValueError("External feature schema differs")
        compared[name] = s56.compare(d56_sequences / name, d56_tokens / name, values, targets, player_tokens)

    halves = touch_trial.training_halves(model_set, dfl, skill, raw=True)
    for record, half in zip(stored, halves, strict=True):
        name = record["file"]
        if name != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match D56's record {name}")
        check_record(name, half)
        skill_order = order.index(int(half.match)) if half.source == "skillcorner" else None
        made49 = hv.training_view(model_set, half, swaps=swaps, provider_ball=None, skill_order=skill_order)
        if made49 is not None:
            check_record(f"{made49[0].key}.npz", made49[0])
        dfl_height = dfl_heights.get((half.match, half.index)) if half.source == "dfl" else None
        made56 = s56.training_view(model_set, half, height=dfl_height, swaps=swaps, skill_order=skill_order)
        if made56 is None:
            continue
        view56, visibility56, parts56 = made56
        view_name = f"{view56.key}.npz"
        check_record(view_name, view56)
        if half.source != "skillcorner":
            continue
        made = training_view(model_set, half, skill_height=skill_heights[half.match, half.index], swaps=swaps,
                             skill_order=skill_order)
        view, visibility, parts, facts = made
        if f"{view.key}.npz" != view_name or visibility != visibility56 or not np.array_equal(parts["random"], parts56["random"]):
            raise ValueError(f"SZ's view of {half.key} does not share D56's name, visibility or random part")
        d56_hidden = ~np.isfinite(view56.ball).all(axis=1)
        if not np.array_equal(d56_hidden, facts.pop("d56_hidden") | ~np.isfinite(half.ball).all(axis=1)):
            raise ValueError(f"D56's hidden frames of {half.key} are not D56's view")
        frames, columns, values, targets, player_tokens = s56.extract(view)
        if columns != manifest["columns"]:
            raise ValueError("External feature schema differs")
        np.savez_compressed(sequences / view_name, values=values, targets=targets)
        np.savez_compressed(tokens / view_name, tokens=player_tokens)
        replaced[view_name] = {"sha256": digest(sequences / view_name), "tokens": digest(tokens / view_name)}
        entry, _ = s56.sign_entry(view.positions, view.ball, view.teams, view.source)
        from_tokens = dt.attacking_from_tokens(player_tokens, frames)
        new_entries[view_name] = entry | {"token": {key: from_tokens[key] for key in ("frames", "d", "sign", "first_sign", "agrees")}}
        items_sz.append((view.ball, d56_hidden, parts, half.ball, np.asarray(view.labels)))
        items_d56.append((view56.ball, np.asarray(view56.labels), parts56, half.ball))
        visibilities[view_name] = visibility
        facts_all[view_name] = facts
        print(f"D57 {model_set} {half.key}: D56's records equal "
              f"{all(all(compared[n].values()) for n in compared)}; SZ view {view_name} (visible "
              f"{np.isfinite(view.ball).all(axis=1).mean():.3f} against D56's {np.isfinite(view56.ball).all(axis=1).mean():.3f}, "
              f"above 1.5 m {parts['air'].mean():.3f}, sign {entry['sign']})", flush=True)
    for record in records:
        if record["file"] in replaced:
            continue
        shutil.copyfile(d56_sequences / record["file"], sequences / record["file"])
        shutil.copyfile(d56_tokens / record["file"], tokens / record["file"])
        if digest(sequences / record["file"]) != record["sha256"] or \
                digest(tokens / record["file"]) != token_manifest["records"][record["file"]]:
            raise ValueError(f"A copied record differs from D56's: {record['file']}")
    skill_situation = sorted(r["file"] for r in situation if r["source"] == "skillcorner")
    new_records = [dict(record, sha256=replaced[record["file"]]["sha256"]) if record["file"] in replaced else dict(record)
                   for record in records]
    write_json(sequences / "manifest.json", manifest | {"records": new_records, "d57": {
        "base_manifest": digest(d56_sequences / "manifest.json"), "replaced": sorted(replaced),
        "skill_height_sha256": digest(PACKAGE / "skill_height.py"), "reader_sha256": digest(READER)}})
    write_json(tokens / "manifest.json", {"protocol": PROTOCOL, "base_manifest": digest(sequences / "manifest.json"),
                                          "records": token_manifest["records"] | {n: v["tokens"] for n, v in replaced.items()},
                                          "player_set_sha256": digest(PACKAGE / "player_set.py")})
    signs_path = scratch / f"signs_{model_set}.json"
    if signs_path.exists():
        raise FileExistsError(f"{signs_path} exists")
    entries = dict(d56_signs["records"]) | new_entries
    write_json(signs_path, {"protocol": PROTOCOL, "set": model_set, "base_manifest": digest(sequences / "manifest.json"),
                            "token_manifest": digest(tokens / "manifest.json"), "records": entries,
                            "d56_signs_sha256": digest(d56_signs_path),
                            "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py")})
    keys = ("agrees", "d", "first_difference", "first_frame", "first_sign", "frames", "max_visible",
            "max_visible_all_frames", "sign", "source")
    sign_part = {"new_records": len(new_entries),
                 "sign_differs_from_d56": sorted(n for n, e in new_entries.items()
                                                 if canonical({k: e[k] for k in keys}) != canonical({k: d56_signs["records"][n][k] for k in keys})),
                 "first_frame_disagrees": sorted(n for n, e in new_entries.items() if e["sign"] is not None and not e["agrees"]),
                 "unsigned": sorted(n for n, e in new_entries.items() if e["sign"] is None),
                 "token_sign_differs": sorted(n for n, e in new_entries.items() if e["token"]["sign"] != e["sign"])}
    reproduced = bool(len(compared) == len(records) and all(all(v.values()) for v in compared.values()))
    passed = bool(reproduced and sorted(replaced) == skill_situation and len(replaced) == 24
                  and not sign_part["sign_differs_from_d56"] and not sign_part["first_frame_disagrees"]
                  and not sign_part["token_sign_differs"] and not sign_part["unsigned"])
    runs = {key: sum(f[key] for f in facts_all.values()) for key in ("runs", "runs_with_d56_delay", "runs_with_new_delay")}
    write_new(output, {"protocol": PROTOCOL, "set": model_set, "passed": passed, "d56_records": len(records),
                       "reproduced_byte_for_byte": reproduced,
                       "differing": {name: item for name, item in compared.items() if not all(item.values())},
                       "replaced": sorted(replaced), "unchanged_records": len(records) - len(replaced),
                       "choices": {"swaps": swaps, "record_names": "D56's (the 24 SkillCorner situation views keep their names)",
                                   "delay": "D56's for each run of air | crowd that is one of D56's runs, otherwise drawn from the air seed"},
                       "air_seeds": AIR_SEEDS, "d56_seeds": s56.SEEDS,
                       "constants": {"air_height_m": sv.AIR_HEIGHT, "delay_frames": list(s56.DELAY),
                                     "visibility": list(s56.VISIBILITY)},
                       "visibility_drawn": visibilities, "runs": runs,
                       "views": {"sz_skillcorner": s56.view_statistics([(b, l, p, o) for b, _, p, o, l in items_sz])
                                 | added_by_height(items_sz),
                                 "d56_skillcorner": s56.view_statistics(items_d56)},
                       "signs": sign_part, "reader_sha256": digest(READER),
                       "d56_prepare_sha256": digest(recipe / f"d56_prepare_{model_set}.json"),
                       "d56_sequences_manifest": digest(d56_sequences / "manifest.json"),
                       "d56_tokens_manifest": digest(d56_tokens / "manifest.json"),
                       "sequences_manifest": digest(sequences / "manifest.json"),
                       "tokens_manifest": digest(tokens / "manifest.json"), "signs_file": digest(signs_path),
                       "code": {name: digest(PACKAGE / name) for name in ("skill_height.py", "situation_train.py",
                                                                          "situation_views.py", "skillcorner.py",
                                                                          "hidden_views.py", "player_set.py",
                                                                          "long_context.py", "direction_tokens.py")},
                       "prepared_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D57 prepare {model_set}: {len(compared)} D56 records reproduced byte for byte: {reproduced}; "
          f"{len(replaced)} SkillCorner views replaced; signs as D56's: {not sign_part['sign_differs_from_d56']}; "
          f"passed {passed}", flush=True)
    if not passed:
        raise SystemExit("D57 stops: the preparation check failed")
    return passed


# -- the SkillCorner air view (reported only) ---------------------------------------------------------

def air_view(half, z):
    """(view half, hidden by height): a stored development half with the ball hidden where z is above 1.5 m."""
    if len(z) != len(half.ball):
        raise ValueError(f"The heights of {half.key} are not on its frames")
    air = sv.air_mask(z)
    ball = np.array(half.ball, dtype=np.float32, copy=True)
    ball[air] = np.nan
    return replace(half, view=AIR_VIEW, ball=ball), air


def air_view_halves(skill, heights):
    """[(observed half, air view half, hidden by height)] of the four development matches, in the scored order."""
    from dfl_submission import external_benchmark as benchmark
    result = []
    for half in benchmark.skillcorner_halves(skill, "development"):
        view, air = air_view(half, heights[half.match, half.index])
        result.append((half, view, air))
    return result


def air_view_statistics(items):
    """Visibility before and after, the share hidden by height, on all and on live frames."""
    observed = sv.run_statistics([half.ball for half, _, _ in items], [half.labels for half, _, _ in items])
    view = sv.run_statistics([v.ball for _, v, _ in items], [v.labels for _, v, _ in items])
    frames = sum(len(air) for _, _, air in items)
    live = [np.isin(np.asarray(half.labels), LIVE) for half, _, _ in items]
    return {"observed": observed, "view": view, "share_hidden_by_height": sum(int(a.sum()) for _, _, a in items) / frames,
            "live_share_hidden_by_height": sum(int((a & l).sum()) for (_, _, a), l in zip(items, live))
            / sum(int(l.sum()) for l in live)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("reader")
    one.add_argument("--skill", type=Path, required=True)
    two = sub.add_parser("prepare")
    two.add_argument("--set", choices=s56.SETS, required=True)
    two.add_argument("--dfl", type=Path, required=True)
    two.add_argument("--skill", type=Path, required=True)
    args = parser.parse_args()
    require_external_only()
    if args.action == "reader":
        reader(args.skill)
    else:
        prepare(args.set, args.dfl, args.skill)


if __name__ == "__main__":
    main()
