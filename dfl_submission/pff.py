"""Read PFF FC's 2022 World Cup broadcast tracking for the unseen-vendor trials (DECISIONS.md D7).

Only periods 1 and 2 are read. Positions are the unsmoothed `homePlayers`, `awayPlayers` and
`balls` points that the vendor marks `VISIBLE`; smoothed and estimated points are ignored. Native
29.97 fps frames are resampled by time to the 25 Hz grid, without filling a missing native point,
and points more than 5 m outside the metadata pitch are discarded, as for SkillCorner.

Labels come from the vendor's game events, carried forward from each event's `start_frame`:
kick-offs and on-the-ball events start live play, with the team from `home_ball`; `OUT` starts a
dead ball; `END` and missing video end the labelled span. Frames before the kick-off, after the
end, during missing video, or without a vendor tracking frame are `X`. The mapping is fixed in D7.
"""

import argparse
from array import array
import bz2
import json
from pathlib import Path

import numpy as np

from dfl_submission.external_only import require_external_only
from dfl_submission.skillcorner import identity
from dfl_transfer_data import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "dfl_submission" / "pff_sources.json"
RELEASE = "PFF FC 2022 FIFA World Cup broadcast tracking, metadata and rosters"
FOLDERS = {"metadata": "Metadata", "roster": "Rosters", "tracking": "Tracking Data"}
TRACKING_SUFFIXES = (".jsonl.bz2", ".jsonl")
PERIODS = (1, 2)
TARGET_FPS = 25
OUTLIER_METRES = 5
VISIBLE = "VISIBLE"
LIVE_EVENTS = frozenset({"FIRSTKICKOFF", "SECONDKICKOFF", "THIRDKICKOFF", "FOURTHKICKOFF", "OTB"})
DEAD_EVENTS = frozenset({"OUT"})
END_EVENTS = frozenset({"END"})
MISSING_EVENTS = frozenset({"VID"})
SPLIT_SIZE = 64
SPLIT_ORDER = ("development", "confirm_a", "confirm_b", "confirm_c", "train", "train", "train", "untouched")


def sample_points(length, fps):
    """Native left index and fraction for each 25 Hz frame of a half with `length` native frames."""
    count = int(np.floor((length - 1) * TARGET_FPS / fps + 1e-9)) + 1
    position = np.arange(count) * fps / TARGET_FPS
    left = np.minimum(np.floor(position + 1e-9).astype(int), length - 1)
    fraction = position - left
    return left, np.where(fraction < 1e-9, 0.0, fraction)


def resample(xy, left, fraction):
    """Linear interpolation in time that never fills a missing native observation."""
    right = np.minimum(left + 1, len(xy) - 1)
    weight = fraction.reshape((-1,) + (1,) * (xy.ndim - 1))
    interpolated = np.where(weight == 0, xy[left], xy[left] * (1 - weight) + xy[right] * weight)
    return interpolated.astype(np.float32)


def event_key(event):
    return event["start_frame"], event.get("sequence") or 0


def native_labels(events, first, present):
    """Label codes per native frame of one period from its game events."""
    length = len(present)
    labels = np.full(length, pm.X, dtype=np.int8)
    phase, owner = "before", None
    for event in sorted(events, key=event_key):
        kind = event.get("game_event_type")
        if kind in LIVE_EVENTS:
            phase = "live"
            owner = {True: pm.H, False: pm.A}.get(event.get("home_ball"), owner)
        elif kind in DEAD_EVENTS:
            phase = "dead"
        elif kind in END_EVENTS:
            phase = "ended"
        elif kind in MISSING_EVENTS:
            phase = "missing"
        else:
            continue
        code = pm.D if phase == "dead" else owner if phase == "live" and owner is not None else pm.X
        labels[min(max(event["start_frame"] - first, 0), length):] = code
    labels[~present] = pm.X
    return labels


class _Period:
    def __init__(self):
        self.frames = array("q")
        self.player = (array("q"), array("q"), array("d"), array("d"))
        self.ball = (array("q"), array("d"), array("d"))
        self.events = {}

    def add(self, row, slots):
        number = int(row["frameNum"])
        self.frames.append(number)
        for side, field in enumerate(("homePlayers", "awayPlayers")):
            for point in row.get(field) or ():
                if point.get("visibility") != VISIBLE or point.get("x") is None or point.get("y") is None:
                    continue
                try:
                    slot = slots[side, int(point["jerseyNum"])]
                except (KeyError, TypeError, ValueError):
                    raise ValueError(f"PFF {'home away'.split()[side]} jersey {point.get('jerseyNum')!r} "
                                     f"at frame {number} is not on the roster") from None
                for values, value in zip(self.player, (number, slot, point["x"], point["y"])):
                    values.append(value)
        balls = [ball for ball in row.get("balls") or ()
                 if ball.get("visibility") == VISIBLE and ball.get("x") is not None and ball.get("y") is not None]
        if len(balls) == 1:
            for values, value in zip(self.ball, (number, balls[0]["x"], balls[0]["y"])):
                values.append(value)
        event = row.get("game_event")
        if event:
            start = event.get("start_frame")
            event = dict(event, start_frame=number if start is None else int(start))
            self.events[(*event_key(event), event.get("game_event_type"), event.get("home_ball"))] = event

    def finish(self, slots, teams, pitch, fps):
        frames = np.frombuffer(self.frames, dtype=np.int64)
        if len(frames) < 2:
            raise ValueError("A PFF period has fewer than two tracking frames")
        if len(np.unique(frames)) != len(frames):
            raise ValueError("Repeated PFF frame numbers within a period")
        first = int(frames.min())
        length = int(frames.max()) - first + 1
        present = np.zeros(length, dtype=bool)
        present[frames - first] = True
        positions = np.full((length, slots, 2), np.nan, dtype=np.float32)
        frame, slot, x, y = (np.frombuffer(values, dtype=values.typecode == "q" and np.int64 or np.float64)
                             for values in self.player)
        counts = np.zeros((length, slots), dtype=np.int32)
        np.add.at(counts, (frame - first, slot), 1)
        positions[frame - first, slot] = np.stack([x, y], axis=1)
        positions[counts > 1] = np.nan  # two visible points for one player are ambiguous
        ball = np.full((length, 2), np.nan, dtype=np.float32)
        frame, x, y = (np.frombuffer(values, dtype=values.typecode == "q" and np.int64 or np.float64)
                       for values in self.ball)
        ball[frame - first] = np.stack([x, y], axis=1)
        for points in (positions, ball):
            points[~(np.abs(points) <= pitch / 2 + OUTLIER_METRES).all(axis=-1)] = np.nan
        labels = native_labels(self.events.values(), first, present)
        left, fraction = sample_points(length, fps)
        return dict(positions=resample(positions, left, fraction), ball=resample(ball, left, fraction),
                    labels=labels[left], teams=teams, pitch=pitch)


def parse(metadata, roster, rows):
    """Two halves of 25 Hz positions, ball, labels, team of each slot and pitch, from streamed rows."""
    sides = {str(metadata["homeTeam"]["id"]): 0, str(metadata["awayTeam"]["id"]): 1}
    try:
        keys = sorted({(sides[str(entry["team"]["id"])], int(entry["shirtNumber"])) for entry in roster})
    except KeyError as error:
        raise ValueError(f"PFF roster entry for an unknown team: {error}") from None
    slots = {key: index for index, key in enumerate(keys)}
    teams = np.array([side for side, _ in keys], dtype=np.int8)
    size = metadata["stadium"]["pitches"][0]
    pitch = np.array([float(size["length"]), float(size["width"])])
    periods = {period: _Period() for period in PERIODS}
    for row in rows:
        if row.get("period") in periods:
            periods[row["period"]].add(row, slots)
    return [periods[period].finish(len(slots), teams, pitch, float(metadata["fps"])) for period in PERIODS]


def files(raw, game):
    tracking = [raw / FOLDERS["tracking"] / f"{game}{suffix}" for suffix in TRACKING_SUFFIXES]
    tracking = [path for path in tracking if path.exists()]
    if len(tracking) != 1:
        raise ValueError(f"Expected one PFF tracking file for game {game}")
    return raw / FOLDERS["metadata"] / f"{game}.json", raw / FOLDERS["roster"] / f"{game}.json", tracking[0]


def games(raw):
    """Game ids that have metadata, a roster and tracking."""
    ids = []
    for path in sorted((raw / FOLDERS["metadata"]).glob("*.json")):
        if path.stem.isdigit() and (raw / FOLDERS["roster"] / path.name).exists() and any(
                (raw / FOLDERS["tracking"] / f"{path.stem}{suffix}").exists() for suffix in TRACKING_SUFFIXES):
            ids.append(int(path.stem))
    return tuple(sorted(ids))


def split(ids):
    """D7 split: sort the 64 game ids and deal them in turn to the eight slots of SPLIT_ORDER."""
    ordered = sorted(int(game) for game in ids)
    if len(ordered) != SPLIT_SIZE or len(set(ordered)) != SPLIT_SIZE:
        raise ValueError(f"The D7 split is defined for {SPLIT_SIZE} distinct games, got {len(ordered)}")
    result = {name: [] for name in dict.fromkeys(SPLIT_ORDER)}
    for rank, game in enumerate(ordered):
        result[SPLIT_ORDER[rank % len(SPLIT_ORDER)]].append(game)
    return {name: tuple(values) for name, values in result.items()}


def relative(raw, path):
    return path.relative_to(raw).as_posix()


def pin(raw, sources=SOURCES):
    """Record the size and SHA-256 of every metadata, roster and tracking file."""
    pinned = {relative(raw, path): list(identity(path)) for game in games(raw) for path in files(raw, game)}
    sources.write_text(json.dumps({"release": RELEASE, "files": pinned}, indent=2, sort_keys=True) + "\n")


def rows(path):
    with (bz2.open(path, "rt") if path.suffix == ".bz2" else path.open()) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_match(raw, game, sources=SOURCES):
    pinned = json.loads(sources.read_text())["files"]
    paths = files(raw, game)
    for path in paths:
        if list(identity(path)) != pinned.get(relative(raw, path)):
            raise ValueError(f"PFF file differs from {sources.name}: {relative(raw, path)}")
    metadata, roster = json.loads(paths[0].read_text()), json.loads(paths[1].read_text())
    if isinstance(metadata, list):
        metadata, = metadata
    if int(metadata["id"]) != int(game):
        raise ValueError(f"PFF metadata id differs from game {game}")
    return parse(metadata, roster, rows(paths[2]))


def prepare(raw, out, sources=SOURCES, ids=None):
    """Parse pinned games into verified 25 Hz halves under `out`."""
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {"sources_sha256": digest(sources), "parsed": {}}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["sources_sha256"] == manifest["sources_sha256"]:
            manifest = previous
    for game in ids or sorted({int(Path(name).name.split(".")[0]) for name in json.loads(sources.read_text())["files"]}):
        for index, half in enumerate(read_match(raw, game, sources), 1):
            path = out / f"{game}_{index}.npz"
            np.savez_compressed(path, **half)
            manifest["parsed"][path.name] = list(identity(path))
            counts = {pm.LABELS[code]: int(count) for code, count in zip(*np.unique(half["labels"], return_counts=True))}
            print(f"Parsed PFF {game} half {index}: {counts}, visible ball "
                  f"{np.isfinite(half['ball'][:, 0]).mean():.1%}", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load(out, game):
    """Both halves of a prepared game, after checking them against the manifest."""
    parsed = json.loads((out / "manifest.json").read_text())["parsed"]
    halves = []
    for index in (1, 2):
        path = out / f"{game}_{index}.npz"
        if list(identity(path)) != parsed.get(path.name):
            raise ValueError(f"Prepared PFF half differs from the manifest: {path.name}")
        with np.load(path, allow_pickle=False) as item:
            halves.append({name: item[name] for name in item.files})
    return halves


def tracking(half):
    return half["positions"], half["ball"], half["teams"], half["pitch"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pinning = commands.add_parser("pin", help="write the source hashes")
    pinning.add_argument("--raw", type=Path, required=True)
    preparing = commands.add_parser("prepare", help="parse pinned games into 25 Hz halves")
    preparing.add_argument("--raw", type=Path, required=True)
    preparing.add_argument("--out", type=Path, required=True)
    preparing.add_argument("--games", type=int, nargs="+")
    args = parser.parse_args()
    require_external_only()
    if args.command == "pin":
        pin(args.raw)
    else:
        prepare(args.raw, args.out, ids=args.games)


if __name__ == "__main__":
    main()
