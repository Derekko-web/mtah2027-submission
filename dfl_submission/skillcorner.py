"""Fetch and parse pinned SkillCorner Open Data, with observed tracking only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

import numpy as np


REVISION = "4340d274572876239c154c90bc507a9b3250a656"
RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{REVISION}"
MEDIA = f"https://media.githubusercontent.com/media/SkillCorner/opendata/{REVISION}"
TRAIN = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691,
         1996435, 1996436, 2006229, 2006363)
DEVELOPMENT = (2007448, 2007721, 2010085, 2011166)
HOLDOUT = (2013725, 2015213, 2016236, 2017461)


def fetch(url, path, expected=None):
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with urlopen(url, timeout=90) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target)
            if expected and identity(partial) != expected:
                raise ValueError(f"Download does not match the pinned LFS object: {url}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    if expected and identity(path) != expected:
        raise ValueError(f"Local file does not match the pinned LFS object: {path}")
    return path


def identity(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return path.stat().st_size, digest


def fetch_match(folder, match):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("match.json", "phases_of_play.csv", "tracking_extrapolated.jsonl"):
        name = f"{match}_{suffix}"
        relative = f"data/matches/{match}/{name}"
        if suffix.endswith("jsonl"):
            with urlopen(f"{RAW}/{relative}", timeout=60) as handle:
                version, oid, size = handle.read().decode().splitlines()
            if version != "version https://git-lfs.github.com/spec/v1" or not oid.startswith("oid sha256:"):
                raise ValueError(f"Unexpected LFS pointer: {relative}")
            paths.append(fetch(f"{MEDIA}/{relative}", folder / name, (int(size.split()[1]), oid[11:])))
        else:
            paths.append(fetch(f"{RAW}/{relative}", folder / name))
    return paths


def at_25hz(xy, labels):
    """Resample 10 Hz observations without filling a missing native observation."""
    numerator = np.arange(((len(labels) - 1) * 5) // 2 + 1) * 2
    left, fraction = divmod(numerator, 5)
    right = np.minimum(left + 1, len(labels) - 1)
    weight = (fraction / 5).reshape((-1,) + (1,) * (xy.ndim - 1))
    interpolated = np.where(weight == 0, xy[left], xy[left] * (1 - weight) + xy[right] * weight)
    return interpolated.astype(np.float32), labels[left]


def parse_match(paths):
    metadata = json.loads(paths[0].read_text())
    sides = {metadata[key]["id"]: i for i, key in enumerate(("home_team", "away_team"))}
    players = sorted((p["id"], sides[p["team_id"]]) for p in metadata["players"])
    slots = {player_id: i for i, (player_id, _) in enumerate(players)}
    teams = np.array([team for _, team in players], dtype=np.int8)
    pitch = np.array([metadata["pitch_length"], metadata["pitch_width"]])
    periods = {period["period"]: period for period in metadata["match_periods"]}
    halves = []
    for number in (1, 2):
        period = periods[number]
        start, stop = period["start_frame"], period["end_frame"]
        n = stop - start
        if not 24000 <= n <= 42000:
            raise ValueError(f"Unexpected native half length: {n}")
        halves.append((start, stop, np.full((n, len(slots), 2), np.nan, np.float32),
                       np.full((n, 2), np.nan, np.float32), np.full(n, "X", dtype="<U1")))
    with paths[1].open(newline="") as handle:
        for phase in csv.DictReader(handle):
            period = int(phase["period"])
            if period not in (1, 2):
                continue
            start, stop, _, _, labels = halves[period - 1]
            lo = max(0, int(phase["frame_start"]) - start)
            hi = min(stop - start, int(phase["frame_end"]) - start)
            if hi <= lo:
                continue
            target = "HA"[sides[int(phase["team_in_possession_id"])]]
            if np.any((labels[lo:hi] != "X") & (labels[lo:hi] != target)):
                raise ValueError("Conflicting in-play phases")
            labels[lo:hi] = target
    with paths[2].open() as handle:
        for line in handle:
            row = json.loads(line)
            period = row["period"]
            if period not in (1, 2):
                continue
            start, stop, positions, ball, labels = halves[period - 1]
            frame = row["frame"]
            if not start <= frame < stop:
                continue
            i = frame - start
            observed = row["ball_data"]
            if observed and observed["is_detected"] and observed["x"] is not None and observed["y"] is not None:
                ball[i] = observed["x"], observed["y"]
            for player in row["player_data"]:
                if player["is_detected"] and player["x"] is not None and player["y"] is not None:
                    positions[i, slots[player["player_id"]]] = player["x"], player["y"]
            # Tracking possession only supplies fallback target coverage; it never enters a feature.
            if labels[i] == "X":
                group = (row["possession"] or {}).get("group")
                labels[i] = {"home team": "H", "away team": "A", None: "D"}[group]
    result = []
    for _, _, positions, ball, labels in halves:
        positions[np.any(np.abs(positions) > pitch / 2 + 5, axis=2)] = np.nan
        ball[np.any(np.abs(ball) > pitch / 2 + 5, axis=1)] = np.nan
        positions, targets = at_25hz(positions, labels)
        ball, _ = at_25hz(ball, labels)
        result.append(dict(positions=positions, ball=ball, labels=targets, teams=teams, pitch=pitch))
    return result


def prepare(folder, match):
    paths = fetch_match(folder / "raw", match)
    print(f"Fetched SkillCorner {match}", flush=True)
    halves = parse_match(paths)
    for half_index, half in enumerate(halves, 1):
        path = folder / f"{match}_{half_index}.npz"
        np.savez_compressed(path, **half)
        counts = dict(zip(*np.unique(half["labels"], return_counts=True)))
        print(f"Parsed {match} half {half_index}: {counts}, observed ball {np.isfinite(half['ball'][:, 0]).mean():.1%}", flush=True)
    return {path.name: identity(path) for path in paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matches", type=int, nargs="+", default=[*TRAIN, *DEVELOPMENT, *HOLDOUT])
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda match: prepare(args.out, match), args.matches))
    manifest = dict(revision=REVISION, files={k: v for match in results for k, v in match.items()},
                    parsed={f"{match}_{half}.npz": identity(args.out / f"{match}_{half}.npz")
                            for match in args.matches for half in (1, 2)})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
