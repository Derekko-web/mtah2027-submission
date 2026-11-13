"""D56: training views that hide the ball in the air and in a crowd (DECISIONS.md D56) - the hide, the views, the prepare step.

Situation hide (training): on the frames of a half, the ball is hidden while its height is above 1.5 m (DFL only) or
at least three players are within 3 m of it (D55's rules, `situation_views`), each such hidden run extended by a delay
drawn uniformly from 0 to 25 frames (0-1 s); on top, `augment.wide_masking`'s two-state random process hides the ball
with a visibility drawn per half from 0.6 to 0.95. The situations are read from the half's own clean tracking (the
set's ball: the provider's for the original set, the handled dead ball for the augmented set) and, for DFL, the
provider's height on the same frames.

Views, one per training half (32 per set), from new fixed seeds:
- DFL training halves: the set's clean half (`touch_trial.training_halves`) with `wide_masking`'s coordinate noise and
  player dropout (`hidden_views.noise_and_dropout`, the same draw order) and the situation hide; for the augmented set
  followed by its training identity swaps (as D49 with its amendment);
- SkillCorner training halves: the set's own half (the original set's provider tracking, the augmented set's handled
  dead ball) with the crowd rule, the delay and the random part; nothing else changes (as D49's SkillCorner views).

Commands (each with `MTAH_DATA_DIR` set to a nonexistent path):
- `reader --dfl DIR`: the height-aware reader on the four DFL training matches; its ball x and y (and status) must
  equal the existing reader's on every frame of each. Saves the heights to `data/d56/heights/` and writes
  `data/d56/reader.json` once.
- `prepare --set current|augmented --dfl DIR --skill DIR`: re-extracts all of D49's records of the set (D42's stored
  halves and D49's views) and checks their sequences and tokens byte for byte against `data/d49/`, copies them, adds
  the set's 32 situation views, computes every record's attacking sign as the stack does (D49's records checked against
  D48's sign file), and writes `data/d56/sequences_<set>`, `tokens_<set>`, `signs_<set>.json` and
  `stack_recipe/d56_prepare_<set>.json` once.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np

from dfl_submission import augment, hidden_views as hv, situation_views as sv
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stack_recipe"
SCRATCH = ROOT / "data" / "d56"
HEIGHTS = SCRATCH / "heights"
READER = SCRATCH / "reader.json"
PROTOCOL = "DECISIONS.md D56"
FPS = 25
DELAY = (0, FPS)                       # frames, both ends included: 0-1 s
VISIBILITY = (.6, .95)                 # the random part's visibility, drawn per half
HIDDEN_RUN = augment.HIDDEN_RUN        # 50 frames: the two-state process's mean hidden run
VIEW = "situation"
SEEDS = {"current_dfl": 1010000, "augmented_dfl": 1020000, "current_skill": 1030000, "augmented_skill": 1040000}
SETS = ("current", "augmented")
DFL_TRAIN = ("WMX", "WN1", "WOH", "WOY")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# -- the hide ------------------------------------------------------------------------------------

def two_state(n, visibility, rng):
    """Hidden flags of `augment.wide_masking`'s two-state process at `visibility` (its draws, in its order)."""
    reveal = 1 / HIDDEN_RUN
    hide = reveal * (1 - visibility) / visibility
    visible = bool(rng.random() < visibility)
    switches = rng.random(n)
    hidden = np.empty(n, dtype=bool)
    for i in range(n):
        if switches[i] < (hide if visible else reveal):
            visible = not visible
        hidden[i] = not visible
    return hidden


def extend_runs(flags, rng):
    """(flags with each run extended at its end by a delay uniform on 0-25 frames, the added frames)."""
    flags = np.asarray(flags, dtype=bool)
    starts, lengths = sv.runs_of(flags)
    delays = rng.integers(DELAY[0], DELAY[1] + 1, size=len(starts))
    extended = flags.copy()
    for start, length, delay in zip(starts, lengths, delays):
        extended[start + length:start + length + delay] = True
    return extended, extended & ~flags


def situation_hide(air, crowd, rng):
    """(hidden flags, parts, visibility): air or crowd, each run delayed, or the two-state random part."""
    air, crowd = np.asarray(air, dtype=bool), np.asarray(crowd, dtype=bool)
    visibility = float(rng.uniform(*VISIBILITY))
    situation, delay = extend_runs(air | crowd, rng)
    random = two_state(len(air), visibility, rng)
    parts = {"air": air, "crowd": crowd, "delay": delay, "random": random}
    return situation | random, parts, visibility


# -- the views -----------------------------------------------------------------------------------

def dfl_view(positions, ball, z, rng):
    """A DFL training half with `wide_masking`'s noise and dropout and the situation hide (situations from the clean half)."""
    air, crowd = sv.air_mask(z), sv.crowd_mask(positions, ball)
    noisy_positions, noisy_ball = hv.noise_and_dropout(positions, ball, rng)
    hidden, parts, visibility = situation_hide(air, crowd, rng)
    noisy_ball[hidden] = np.nan
    return noisy_positions, noisy_ball, visibility, parts


def skill_view(positions, ball, rng):
    """A SkillCorner training half's ball with the crowd rule, the delay and the random part (no height)."""
    crowd = sv.crowd_mask(positions, ball)
    hidden, parts, visibility = situation_hide(np.zeros(len(ball), dtype=bool), crowd, rng)
    view_ball = np.array(ball, dtype=np.float32, copy=True)
    view_ball[hidden] = np.nan
    return view_ball, visibility, parts


def dfl_seed(model_set, match, index):
    return SEEDS[f"{model_set}_dfl"] + 100 * hv.DFL_ORDER.index(match) + index


def skill_seed(model_set, order, index):
    return SEEDS[f"{model_set}_skill"] + 10 * order + index


def training_view(model_set, half, *, height=None, swaps=True, skill_order=None):
    """(view half, visibility, parts) of one training half (an `external_benchmark.Half`), or None if it gets none."""
    if half.source == "dfl":
        if half.view != "clean":
            return None
        if height is None or len(height) != len(half.ball):
            raise ValueError(f"A DFL training half needs its heights on the same frames: {half.key}")
        rng = np.random.default_rng(dfl_seed(model_set, half.match, half.index))
        positions, ball, visibility, parts = dfl_view(np.asarray(half.positions, dtype=np.float32),
                                                      np.asarray(half.ball, dtype=np.float32), height, rng)
        if model_set == "augmented" and swaps:
            positions = augment.swap_teams(positions, half.teams, rng, every=augment.TRAIN_SWAP_EVERY,
                                           frames=augment.training_frames)
        return replace(half, view=VIEW, positions=positions, ball=ball), visibility, parts
    if half.source == "skillcorner":
        if skill_order is None:
            raise ValueError("A SkillCorner half needs its order among the SkillCorner matches")
        rng = np.random.default_rng(skill_seed(model_set, skill_order, half.index))
        ball, visibility, parts = skill_view(np.asarray(half.positions, dtype=np.float32), half.ball, rng)
        return replace(half, view=VIEW, ball=ball), visibility, parts
    raise ValueError(f"Unknown training source {half.source!r}")


# -- statistics ----------------------------------------------------------------------------------

def view_statistics(items):
    """Pooled over (view ball, labels, parts, original ball): visibility, live visibility, runs, and each part's share."""
    balls = [ball for ball, _, _, _ in items]
    result = sv.run_statistics(balls, [labels for _, labels, _, _ in items])
    frames = result["frames"]
    if not frames:
        return result
    total = {name: sum(int(np.asarray(parts[name]).sum()) for _, _, parts, _ in items) for name in ("air", "crowd", "delay", "random")}
    situation = sum(int((parts["air"] | parts["crowd"] | parts["delay"]).sum()) for _, _, parts, _ in items)
    random_only = sum(int((parts["random"] & ~(parts["air"] | parts["crowd"] | parts["delay"])).sum())
                      for _, _, parts, _ in items)
    missing = sum(int((~np.isfinite(np.asarray(original, dtype=np.float64)).all(axis=1)).sum())
                  for _, _, _, original in items)
    result |= {f"share_hidden_by_{name if name != 'air' else 'height'}": total[name] / frames for name in total}
    result |= {"share_hidden_by_situation_with_delay": situation / frames, "share_hidden_by_random_only": random_only / frames,
               "share_ball_missing_before_the_hide": missing / frames}
    return result


# -- the reader check on the DFL training matches ---------------------------------------------------

def reader(dfl, *, output=READER, heights=HEIGHTS):
    """The height-aware reader on the four DFL training matches, checked frame for frame; heights saved; written once."""
    require_external_only()
    from dfl_submission import situation_trial as st55
    from dfl_transfer_data import SECTIONS, parse_match
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D56's reader check is written once")
    heights.mkdir(parents=True, exist_ok=True)
    checks, files = {}, {}
    for match in DFL_TRAIN:
        info, positions, name = st55.match_files(dfl, match)
        existing, _ = parse_match(info, positions, name)
        height = sv.parse_ball(info, positions, name)
        checks[match] = sv.reader_check(height, existing)
        path = heights / f"{match}.npz"
        if path.exists():
            raise FileExistsError(f"{path} exists")
        np.savez_compressed(path, **{f"z_{i}": height[section]["z"] for i, section in enumerate(SECTIONS)},
                            **{f"frames_{i}": height[section]["frames"] for i, section in enumerate(SECTIONS)})
        files[match] = digest(path)
        print(f"D56 reader {match}: ball x and y {'equal' if sv.reader_passed(checks[match]) else 'DIFFER'}; "
              + "; ".join(f"{s}: {p['frames']} frames, Z numeric {p['numeric_z_on_every_record']}, dead Z = 0 "
                          f"{p['dead_frames_z_zero']}/{p['dead_frames']}, live above 1.5 m {p['live_share_above_air_height']:.3f}"
                          for s, p in checks[match].items()), flush=True)
    passed = bool(all(sv.reader_passed(part) for part in checks.values()))
    write_new(output, {"protocol": PROTOCOL, "passed": passed, "checks": checks, "heights": files,
                       "situation_views_sha256": digest(PACKAGE / "situation_views.py"),
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    if not passed:
        raise SystemExit("D56 stops: the height-aware reader differs from the existing reader on a training match")
    return passed


def heights_of(record_path=READER, heights=HEIGHTS):
    """{(match, half index): heights} from the passed reader check, after checking the files' digests."""
    record = read(record_path)
    if record["passed"] is not True:
        raise SystemExit("D56 stops: the reader check did not pass")
    result = {}
    for match, value in record["heights"].items():
        path = Path(heights) / f"{match}.npz"
        if digest(path) != value:
            raise ValueError(f"{path} differs from the checked heights")
        with np.load(path, allow_pickle=False) as item:
            for i in (0, 1):
                result[match, i] = item[f"z_{i}"]
    return result


# -- prepare -------------------------------------------------------------------------------------

def sign_entry(positions, ball, teams, source):
    """D50's attacking-sign entry of a record, as `stack_trial.training_signs` makes it (no stored tokens to compare)."""
    from dfl_submission import direction_tokens as dt, stack_trial as st
    frames = dt.long_context_frames(len(ball))
    return dict(st.half_entry(positions, teams, frames), source=source), frames


def extract(half):
    """(frames, values, targets, tokens) of a half, as D49's prepare extracts them."""
    from dfl_submission import long_context, player_set, temporal_network as tn
    frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
    values, targets = table.to_numpy(dtype=np.float32), tn.targets(half.labels[frames])
    tokens = player_set.tokens(half.positions, half.ball, half.teams, half.pitch, frames).astype(np.float16)
    return frames, table.columns.tolist(), values, targets, tokens


def same_bytes(stored, fresh):
    stored, fresh = np.asarray(stored), np.asarray(fresh)
    return stored.dtype == fresh.dtype and stored.shape == fresh.shape and stored.tobytes() == fresh.tobytes()


def compare(sequence_path, token_path, values, targets, tokens):
    with np.load(sequence_path, allow_pickle=False) as item, np.load(token_path, allow_pickle=False) as stored:
        return {"values": same_bytes(item["values"], values), "targets": same_bytes(item["targets"], targets),
                "tokens": same_bytes(stored["tokens"], tokens)}


ENTRY_KEYS = ("agrees", "d", "first_difference", "first_frame", "first_sign", "frames", "max_visible",
              "max_visible_all_frames", "sign", "source")


def prepare(model_set, dfl, skill, *, scratch=SCRATCH, recipe=RECIPE):
    """One set's D49 records reproduced byte for byte and copied, plus its 32 situation views and every record's sign."""
    require_external_only()
    from dfl_submission import direction_tokens as dt, external_benchmark as benchmark, hidden_trial as h49
    from dfl_submission import player_set, stack_trial as st, touch_trial
    if model_set not in SETS:
        raise ValueError(f"Unknown set {model_set!r}")
    output = recipe / f"d56_prepare_{model_set}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D56 preparation is recorded once")
    d49_sequences, d49_tokens = st.data_folders("d49", model_set, None)
    d49 = read(h49.RECIPE / f"prepare_{model_set}.json")
    if (d49["passed"] is not True or d49["sequences_manifest"] != digest(d49_sequences / "manifest.json")
            or d49["tokens_manifest"] != digest(d49_tokens / "manifest.json")):
        raise ValueError(f"D49's {model_set} folders are not the ones its preparation check passed")
    d48_signs_path = st.SCRATCH / f"signs_{model_set}.json"
    if digest(d48_signs_path) != read(recipe / "signs.json")["files"][model_set]:
        raise ValueError(f"{d48_signs_path} is not the file D48's sign check passed on")
    d48_signs = read(d48_signs_path)
    if d48_signs["base_manifest"] != digest(d49_sequences / "manifest.json"):
        raise ValueError("D48's sign file is not for D49's folders")
    heights = heights_of()
    manifest, token_manifest = read(d49_sequences / "manifest.json"), read(d49_tokens / "manifest.json")
    stored, views49 = manifest["records"][:d49["stored_records"]], manifest["records"][d49["stored_records"]:]
    names49 = {record["file"] for record in views49}
    if len(views49) != d49["new_records"]:
        raise ValueError("D49's manifest does not hold its stored records and then its views")
    sequences, tokens = scratch / f"sequences_{model_set}", scratch / f"tokens_{model_set}"
    sequences.mkdir(parents=True, exist_ok=False)
    tokens.mkdir(parents=True, exist_ok=False)
    swaps = bool(d49["choices"]["swaps"])
    order = benchmark.SKILL_SPLITS["all"]
    compared, entries, signs_checked, new_records, new_tokens, items, visibilities = {}, {}, {}, [], {}, {}, {}

    def check_record(name, half):
        frames, columns, values, targets, player_tokens = extract(half)
        if columns != manifest["columns"]:
            raise ValueError("External feature schema differs")
        if digest(d49_sequences / name) != next(r["sha256"] for r in manifest["records"] if r["file"] == name) or \
                digest(d49_tokens / name) != token_manifest["records"][name]:
            raise ValueError(f"A D49 file differs from its manifest: {name}")
        compared[name] = compare(d49_sequences / name, d49_tokens / name, values, targets, player_tokens)
        entry, _ = sign_entry(half.positions, half.ball, half.teams, half.source)
        stored_entry = d48_signs["records"][name]
        signs_checked[name] = all(canonical(entry[key]) == canonical(stored_entry[key]) for key in ENTRY_KEYS)
        entries[name] = stored_entry

    halves = touch_trial.training_halves(model_set, dfl, skill, raw=True)
    for record, half in zip(stored, halves, strict=True):
        name = record["file"]
        if name != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match D49's record {name}")
        check_record(name, half)
        skill_order = order.index(int(half.match)) if half.source == "skillcorner" else None
        made = hv.training_view(model_set, half, swaps=swaps, provider_ball=None, skill_order=skill_order)
        if made is not None:
            view49, _ = made
            name49 = f"{view49.key}.npz"
            if name49 not in names49:
                raise ValueError(f"D49's view {name49} is not in its manifest")
            check_record(name49, view49)
        situation = training_view(model_set, half, height=heights.get((half.match, half.index)) if half.source == "dfl"
                                  else None, swaps=swaps, skill_order=skill_order)
        flag = "equal" if all(compared[name].values()) and signs_checked[name] else "DIFFERS"
        if situation is None:
            print(f"Reproduced {model_set} {name}: {flag}", flush=True)
            continue
        view, visibility, parts = situation
        frames, columns, values, targets, player_tokens = extract(view)
        if columns != manifest["columns"]:
            raise ValueError("External feature schema differs")
        view_name = f"{view.key}.npz"
        np.savez_compressed(sequences / view_name, values=values, targets=targets)
        np.savez_compressed(tokens / view_name, tokens=player_tokens)
        new_records.append({"file": view_name, "source": view.source, "match": view.match,
                            "sha256": digest(sequences / view_name)})
        new_tokens[view_name] = digest(tokens / view_name)
        entry, _ = sign_entry(view.positions, view.ball, view.teams, view.source)
        from_tokens = dt.attacking_from_tokens(player_tokens, frames)
        entries[view_name] = entry | {"token": {key: from_tokens[key] for key in ("frames", "d", "sign", "first_sign", "agrees")}}
        items.setdefault(view.source, []).append((view.ball, np.asarray(view.labels), parts, half.ball))
        visibilities[view_name] = visibility
        print(f"Reproduced {model_set} {name}: {flag}; added {view_name} (v {visibility:.3f}, visible "
              f"{np.isfinite(view.ball).all(axis=1).mean():.3f}, sign {entry['sign']})", flush=True)
    for record in manifest["records"]:
        shutil.copyfile(d49_sequences / record["file"], sequences / record["file"])
        shutil.copyfile(d49_tokens / record["file"], tokens / record["file"])
        if digest(sequences / record["file"]) != record["sha256"] or \
                digest(tokens / record["file"]) != token_manifest["records"][record["file"]]:
            raise ValueError(f"A copied record differs from D49's: {record['file']}")
    if {record["file"] for record in manifest["records"]} & {record["file"] for record in new_records}:
        raise ValueError("A D56 record would replace a D49 record")
    write_json(sequences / "manifest.json", {key: value for key, value in manifest.items() if key != "records"} | {
        "records": [dict(record) for record in manifest["records"]] + new_records,
        "d56": {"base_manifest": digest(d49_sequences / "manifest.json"), "view": VIEW,
                "situation_train_sha256": digest(PACKAGE / "situation_train.py"),
                "situation_views_sha256": digest(PACKAGE / "situation_views.py"), "swaps": swaps}})
    write_json(tokens / "manifest.json", {"protocol": PROTOCOL, "base_manifest": digest(sequences / "manifest.json"),
                                          "records": token_manifest["records"] | new_tokens,
                                          "player_set_sha256": digest(PACKAGE / "player_set.py")})
    signs_path = scratch / f"signs_{model_set}.json"
    if signs_path.exists():
        raise FileExistsError(f"{signs_path} exists")
    write_json(signs_path, {"protocol": PROTOCOL, "set": model_set, "base_manifest": digest(sequences / "manifest.json"),
                            "token_manifest": digest(tokens / "manifest.json"), "records": entries,
                            "d48_signs_sha256": digest(d48_signs_path),
                            "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py")})
    all_names = [record["file"] for record in manifest["records"]] + [record["file"] for record in new_records]
    if sorted(entries) != sorted(all_names) or sorted(compared) != sorted(r["file"] for r in manifest["records"]):
        raise ValueError("The signs or the comparisons do not cover exactly the records")
    new_entries = {name: entries[name] for name in new_tokens}
    sign_part = {"new_records": len(new_entries),
                 "first_frame_disagrees": sorted(n for n, e in new_entries.items() if e["sign"] is not None and not e["agrees"]),
                 "unsigned": sorted(n for n, e in new_entries.items() if e["sign"] is None),
                 "token_sign_differs": sorted(n for n, e in new_entries.items() if e["token"]["sign"] != e["sign"]),
                 "smallest_abs_d_m": min(abs(e["d"]) for e in new_entries.values() if e["d"] is not None),
                 "d49_records_same_as_d48": all(signs_checked.values())}
    reproduced = bool(len(compared) == len(manifest["records"]) and all(all(v.values()) for v in compared.values()))
    passed = bool(reproduced and all(signs_checked.values()) and len(new_records) == 32
                  and not sign_part["first_frame_disagrees"] and not sign_part["token_sign_differs"])
    write_new(output, {"protocol": PROTOCOL, "set": model_set, "passed": passed, "d49_records": len(manifest["records"]),
                       "reproduced_byte_for_byte": reproduced,
                       "differing": {name: item for name, item in compared.items() if not all(item.values())},
                       "signs_differing_from_d48": sorted(name for name, ok in signs_checked.items() if not ok),
                       "new_records": len(new_records), "choices": {"swaps": swaps, "skill_ball": "the set's own ball"},
                       "seeds": SEEDS, "constants": {"delay_frames": list(DELAY), "visibility": list(VISIBILITY),
                                                     "air_height_m": sv.AIR_HEIGHT, "crowd_players": sv.CROWD_PLAYERS,
                                                     "crowd_radius_m": sv.CROWD_RADIUS, "hidden_run": HIDDEN_RUN},
                       "visibility_drawn": visibilities,
                       "views": {source: view_statistics(value) for source, value in sorted(items.items())},
                       "signs": sign_part, "reader_sha256": digest(READER),
                       "d49_sequences_manifest": digest(d49_sequences / "manifest.json"),
                       "d49_tokens_manifest": digest(d49_tokens / "manifest.json"),
                       "sequences_manifest": digest(sequences / "manifest.json"),
                       "tokens_manifest": digest(tokens / "manifest.json"), "signs_file": digest(signs_path),
                       "code": {name: digest(PACKAGE / name) for name in ("situation_train.py", "situation_views.py",
                                                                          "hidden_views.py", "augment.py", "player_set.py",
                                                                          "long_context.py", "direction_tokens.py")},
                       "prepared_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D56 prepare {model_set}: {len(compared)} D49 records reproduced byte for byte: {reproduced}; signs as D48's: "
          f"{all(signs_checked.values())}; {len(new_records)} situation views added; passed {passed}", flush=True)
    if not passed:
        raise SystemExit("D56 stops: the preparation check failed")
    return passed


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("reader")
    one.add_argument("--dfl", type=Path, required=True)
    two = sub.add_parser("prepare")
    two.add_argument("--set", choices=SETS, required=True)
    two.add_argument("--dfl", type=Path, required=True)
    two.add_argument("--skill", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "reader":
        reader(args.dfl)
    else:
        prepare(args.set, args.dfl, args.skill)


if __name__ == "__main__":
    main()
