"""Metrica Sports' public sample matches as a benchmark and veto for candidates (DECISIONS.md D19).

Data: github.com/metrica-sports/sample-data at a pinned commit, games 1-3, with sizes and SHA-256
in `metrica_sources.json`. Games 1 and 2 come as CSV tracking (one file per team) and CSV events;
game 3 as FIFA EPTS tracking (a metadata XML and a raw text file) and JSON events. Coordinates are
normalised to 0-1 with (0, 0) at the top left of a 105 x 68 m pitch (the repository README; game 3's
metadata gives the same FieldSize). They are read exactly as the build reads the competition's
Metrica EPTS export: scaled by the pitch and centred, with no axis flipped
(`build.tactical_tracking_halves(centered=True)`). Points more than 5 m outside the pitch are
dropped, as for the other sources. Players are anonymised; home is the first team.

Targets, fixed in D19 before anything was scored:
- each period is labelled from its kick-off to its last visible ball; frames outside are X;
- inside that span a frame is dead (D) where the corrected tracking has no ball, except that a
  missing-ball run shorter than one second with the ball seen on both sides is a tracking dropout
  inside play and stays live;
- on live frames the team comes from the events. Set pieces, recoveries, passes, carries, shots,
  balls lost, balls out and faults received mark their team at their start frame; a ball lost
  also hands possession to the opponent at its end frame. Within each live segment a frame takes
  the latest mark, and frames before the segment's first mark take that first mark's team;
- `control` applies D1's control rule to those raw targets: a team's spell counts only if it is
  the first of its live segment, contains a set piece, a completed pass or a carry of at least
  0.5 s by that team, or the team holds the ball for at least 0.5 s before its first release
  (pass, shot, ball lost or ball out); otherwise possession stays with the last team that had it.
  Game 3 logs carries as short as one frame and games 1-2 log none, hence the carry length.

Commands: `fetch` downloads the pinned files, `prepare` parses them into 25 Hz halves, `sanity`
runs the reader gate with the organisers' smooth50 baseline, `measure` scores a reference
submission (D3, or D15 since its adoption) once and caches its decoder inputs, and `veto` scores a
candidate against the standing reference (also callable from Python, see `veto`).
"""

import argparse
from collections import namedtuple
import json
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np

from dfl_submission.conventions import live_segments, runs
from dfl_submission.external_only import require_external_only
from dfl_submission.skillcorner import identity
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
SOURCES = PACKAGE / "metrica_sources.json"
RECIPE = PACKAGE / "metrica_recipe"
RAW = ROOT / "data" / "metrica" / "raw"
PARSED = ROOT / "data" / "metrica" / "parsed"
CACHE = ROOT / "data" / "metrica_cache"
PROTOCOL = "DECISIONS.md D19"

REPOSITORY = "https://github.com/metrica-sports/sample-data"
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"
RAW_URL = f"https://raw.githubusercontent.com/metrica-sports/sample-data/{COMMIT}"
GAMES = (1, 2, 3)
GAME_FILES = {
    1: ("data/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "data/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
        "data/Sample_Game_1/Sample_Game_1_RawEventsData.csv"),
    2: ("data/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "data/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
        "data/Sample_Game_2/Sample_Game_2_RawEventsData.csv"),
    3: ("data/Sample_Game_3/Sample_Game_3_metadata.xml",
        "data/Sample_Game_3/Sample_Game_3_tracking.txt",
        "data/Sample_Game_3/Sample_Game_3_events.json"),
}
DOCUMENTS = ("README.md", "documentation/events-definitions.pdf")
TERMS = (
    "No licence file and no GitHub licence at the pinned commit, which is still the repository head "
    "(checked 2026-11-11). The README says the data is offered to potential clients and 'to the broader "
    "football analytics community' to play around with, asks users to 'be responsible with the use of "
    "this data' and to acknowledge the source in anything public. Metrica Sports, the owner, also "
    "organises the competition. Use: local evaluation only (D1's convention choice and the D19 benchmark "
    "and veto); nothing is trained on it, no file is redistributed, the repository holds code, hashes "
    "and averaged scores, and the technical document credits Metrica Sports."
)

FPS = 25
PITCH = (105.0, 68.0)
OUTLIER_METRES = 5
DROPOUT_FRAMES = FPS          # a missing-ball run shorter than one second inside play is a dropout
HOLD_SECONDS = 0.5
RESTART_WINDOW = 2 * FPS      # sanity: frames before a set piece checked for a missing ball

TEAM_EVENTS = frozenset({"SET PIECE", "RECOVERY", "PASS", "CARRY", "SHOT", "BALL LOST", "BALL OUT",
                         "FAULT RECEIVED"})
HANDOVER_EVENTS = frozenset({"BALL LOST"})
CONTROL_EVENTS = frozenset({"SET PIECE", "PASS"})
CARRY_EVENT = "CARRY"         # counts as control only when it lasts HOLD_SECONDS (game 3 logs one-frame carries)
RELEASE_EVENTS = frozenset({"PASS", "SHOT", "BALL LOST", "BALL OUT"})
FLIGHT_EVENTS = frozenset({"PASS", "CARRY"})
RESTART_EVENT = "SET PIECE"
KICK_OFF = "KICK OFF"

TARGETS = ("raw", "aba1.75", "aba3", "aba4", "aba4.5", "control")
PRIMARY = "aba3"
REFERENCES = ("D3", "D15")
STANDING = "D15"              # the submission since D15's confirmation (D19 amendment)
VETO_LOSS = .01
SANITY_RANGE = (0.15, 0.50)
FLIGHT_MISSING_MAX = .01
RESTART_MISSING_MIN = .90

Event = namedtuple("Event", "index team kind subtypes period start end")


def digest(path):
    return identity(path)[1]


# -- sources ---------------------------------------------------------------------------------------

def pinned_paths():
    return tuple(path for game in GAMES for path in GAME_FILES[game]) + DOCUMENTS


def pin(raw, sources=SOURCES):
    """Record the size and SHA-256 of every pinned file, with the commit and the terms we found."""
    files = {path: list(identity(raw / path)) for path in pinned_paths()}
    sources.write_text(json.dumps({"repository": REPOSITORY, "commit": COMMIT, "terms": TERMS, "files": files},
                                  indent=2, sort_keys=True) + "\n")


def verify(raw, paths, sources=SOURCES):
    pinned = json.loads(sources.read_text())
    if pinned["commit"] != COMMIT:
        raise ValueError(f"{sources.name} pins a different commit")
    for path in paths:
        if list(identity(raw / path)) != pinned["files"].get(path):
            raise ValueError(f"Metrica sample file differs from {sources.name}: {path}")


def fetch(raw, sources=SOURCES):
    """Download every pinned file at the pinned commit into `raw`, then check it."""
    for path in json.loads(sources.read_text())["files"]:
        target = raw / path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            with urlopen(f"{RAW_URL}/{path}", timeout=300) as source:
                partial.write_bytes(source.read())
            partial.replace(target)
    verify(raw, json.loads(sources.read_text())["files"], sources)


# -- tracking --------------------------------------------------------------------------------------

def to_metres(xy, pitch):
    """Normalised Metrica coordinates to centred metres, as the build reads the competition export."""
    pitch = np.asarray(pitch, dtype=np.float64)
    metres = (np.asarray(xy, dtype=np.float64) * pitch - pitch / 2).astype(np.float32)
    metres[~(np.abs(metres) <= pitch / 2 + OUTLIER_METRES).all(axis=-1)] = np.nan
    return metres


def _csv_side(path):
    import pandas as pd
    table = pd.read_csv(path, skiprows=2)
    columns = list(table.columns)
    if columns[:3] != ["Period", "Frame", "Time [s]"]:
        raise ValueError(f"Unexpected Metrica CSV header in {Path(path).name}")

    def pair(name):
        follower = columns[columns.index(name) + 1]
        if not follower.startswith("Unnamed"):
            raise ValueError(f"Column {name} is not followed by its y column in {Path(path).name}")
        return np.stack([table[name].to_numpy(np.float64), table[follower].to_numpy(np.float64)], axis=-1)

    players = [name for name in columns if name.startswith("Player")]
    xy = np.stack([pair(name) for name in players], axis=1)
    return table["Period"].to_numpy(np.int64), table["Frame"].to_numpy(np.int64), xy, pair("Ball")


def csv_tracking(home_path, away_path):
    """Frames, periods, normalised positions, ball and team of each slot from games 1-2's CSV pair."""
    periods, frames, home, ball = _csv_side(home_path)
    away_periods, away_frames, away, away_ball = _csv_side(away_path)
    if not (np.array_equal(periods, away_periods) and np.array_equal(frames, away_frames)):
        raise ValueError("Home and away tracking files cover different frames")
    if not np.array_equal(ball, away_ball, equal_nan=True):
        raise ValueError("Home and away tracking files disagree on the ball")
    teams = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    return frames, periods, np.concatenate([home, away], axis=1), ball, teams


def epts_metadata(path):
    """Periods, pitch, home and away team ids, player slots and the per-range player order."""
    root = ET.parse(path).getroot()
    config = root.find(".//GlobalConfig")
    if int(config.findtext("FrameRate")) != FPS:
        raise ValueError("Metrica EPTS metadata is not at 25 Hz")
    parameters = {item.findtext("Name"): item.findtext("Value") for item in config.iter("ProviderParameter")}
    periods = {1: (int(parameters["first_half_start"]), int(parameters["first_half_end"])),
               2: (int(parameters["second_half_start"]), int(parameters["second_half_end"]))}
    size = root.find(".//Sessions/Session/FieldSize")
    pitch = (float(size.findtext("Width")), float(size.findtext("Height")))
    team_ids = [team.get("id") for team in root.find(".//Metadata/Teams")]
    score = root.find(".//Score")
    if len(team_ids) != 2 or (score is not None and score.get("idLocalTeam") != team_ids[0]):
        raise ValueError("Expected two teams with the local (home) team first")
    players = [(player.get("id"), team_ids.index(player.get("teamId"))) for player in root.find(".//Metadata/Players")]
    units = {channel.findtext("Unit") for channel in root.iter("Channel")}
    if units != {"normalized"}:
        raise ValueError(f"Unexpected EPTS position units {units}")
    channels = {channel.get("id"): (channel.get("playerId"), channel.get("channelId"))
                for channel in root.iter("PlayerChannel")}
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        registers = list(spec)
        if (len(registers) != 3 or registers[0].get("name") != "frameCount" or spec.get("separator") != ":"
                or [ref.get("channelId") for ref in registers[2].iter("BallChannelRef")] != ["x", "y"]):
            raise ValueError("Unexpected EPTS data format specification")
        order = []
        for group in registers[1]:
            (player, x), (other, y) = (channels[ref.get("playerChannelId")] for ref in group.iter("PlayerChannelRef"))
            if player != other or (x, y) != ("x", "y"):
                raise ValueError("Unexpected EPTS player channel group")
            order.append(player)
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    return {"periods": periods, "pitch": pitch, "team_ids": team_ids, "players": players, "layouts": layouts}


def epts_tracking(metadata, lines):
    """Frames, periods, normalised positions, ball and team of each slot from game 3's EPTS pair."""
    slots = {player: index for index, (player, _) in enumerate(metadata["players"])}
    teams = np.array([team for _, team in metadata["players"]], dtype=np.int8)
    layouts = [(start, end, np.array([slots[player] for player in order])) for start, end, order in metadata["layouts"]]
    frames, positions, ball = [], [], []
    for line in lines:
        if not line.strip():
            continue
        count, players, located = line.rstrip("\n").split(":")
        frame = int(count)
        (columns,) = [order for start, end, order in layouts if start <= frame <= end]
        values = np.array(players.replace(";", ",").split(","), dtype=np.float64).reshape(-1, 2)
        if len(values) != len(columns):
            raise ValueError(f"EPTS frame {frame} has {len(values)} players, its layout {len(columns)}")
        row = np.full((len(slots), 2), np.nan)
        row[columns] = values
        frames.append(frame)
        positions.append(row)
        ball.append(np.array(located.split(",")[:2], dtype=np.float64))
    frames = np.array(frames, dtype=np.int64)
    periods = np.zeros(len(frames), dtype=np.int64)
    for period, (start, end) in metadata["periods"].items():
        periods[(frames >= start) & (frames <= end)] = period
    return frames, periods, np.stack(positions), np.stack(ball), teams


# -- events ----------------------------------------------------------------------------------------

def csv_events(path):
    import pandas as pd
    events = []
    for index, row in enumerate(pd.read_csv(path).to_dict("records")):
        team = {"Home": 0, "Away": 1}.get(row["Team"])
        if team is None:
            raise ValueError(f"Metrica event {index} has an unknown team {row['Team']!r}")
        subtypes = tuple(str(row["Subtype"]).split("-")) if isinstance(row["Subtype"], str) else ()
        events.append(Event(index, team, row["Type"], subtypes, int(row["Period"]), int(row["Start Frame"]),
                            int(row["End Frame"])))
    return events


def json_events(path, team_ids):
    events = []
    for index, item in enumerate(json.loads(Path(path).read_text())["data"]):
        team_id = (item.get("team") or {}).get("id")
        if team_id not in team_ids:
            raise ValueError(f"Metrica event {index} has an unknown team {team_id!r}")
        subtypes = item.get("subtypes") or ()
        subtypes = tuple(entry["name"] for entry in ([subtypes] if isinstance(subtypes, dict) else subtypes))
        start = int(item["start"]["frame"])
        end = item["end"]["frame"] if item.get("end") and item["end"].get("frame") is not None else start
        events.append(Event(index, team_ids.index(team_id), item["type"]["name"], subtypes, int(item["period"]),
                            start, int(end)))
    return events


def marks(events, first, length, handover=HANDOVER_EVENTS):
    """(frame index, team) possession marks of one period, in time order, ties in file order.

    `handover` lists the events that also mark the opponent at their end frame (D19: a ball lost).
    D26's second marking rule passes an empty set, so the opponent waits for its own next event.
    """
    found = []
    for event in events:
        if event.kind in TEAM_EVENTS and 0 <= event.start - first < length:
            found.append((event.start - first, event.index, 0, event.team))
        if event.kind in handover and event.end >= event.start and 0 <= event.end - first < length:
            found.append((event.end - first, event.index, 1, 1 - event.team))
    return [(frame, team) for frame, _, _, team in sorted(found)]


# -- targets ---------------------------------------------------------------------------------------

def kick_off(events, first, length):
    starts = [event.start - first for event in events
              if event.kind == RESTART_EVENT and event.subtypes[:1] == (KICK_OFF,) and 0 <= event.start - first < length]
    if not starts:
        raise ValueError("A Metrica period has no kick-off")
    return min(starts)


def live_mask(visible):
    """Ball seen, with missing runs shorter than DROPOUT_FRAMES between seen frames treated as seen."""
    live = np.asarray(visible, dtype=bool).copy()
    filled = []
    for start, stop, seen in runs(live.astype(np.int8)):
        if not seen and start > 0 and stop < len(live) and stop - start < DROPOUT_FRAMES:
            live[start:stop] = True
            filled.append(stop - start)
    return live, filled


def raw_labels(visible, events, first, handover=HANDOVER_EVENTS):
    """The raw D19 targets of one period and the counts its reader check records."""
    length = len(visible)
    start = kick_off(events, first, length)
    seen = np.flatnonzero(visible)
    stop = int(seen[-1]) + 1 if len(seen) else start
    live, filled = live_mask(visible)
    labels = np.full(length, pm.X, dtype=np.int8)
    labels[start:stop] = np.where(live[start:stop], pm.X, pm.D)
    found = marks(events, first, length, handover)
    frames = np.array([frame for frame, _ in found], dtype=np.int64)
    teams = np.array([team for _, team in found], dtype=np.int8)
    latest = np.full(length, -1, dtype=np.int64)
    if len(frames):
        position = np.searchsorted(frames, np.arange(length), side="right") - 1
        latest = np.where(position >= 0, teams[np.maximum(position, 0)], -1)
    empty = 0
    for low, high, value in runs(labels == pm.X):
        low, high = max(low, start), min(high, stop)
        if not value or low >= high:
            continue
        inside = np.flatnonzero((frames >= low) & (frames < high))
        team = latest[low:high].copy()
        if len(inside):
            team[:frames[inside[0]] - low] = latest[frames[inside[0]]]
        else:
            empty += 1
        labels[low:high] = np.where(team == 0, pm.H, np.where(team == 1, pm.A, pm.X))
    counts = {"kick_off": start, "first_seen": int(seen[0]) if len(seen) else None, "last_seen": stop - 1,
              "dropouts_filled": len(filled), "dropout_frames": int(sum(filled)), "segments_without_marks": empty}
    return labels, counts


def controlled(team, low, high, events, first):
    """D1's control rule for a raw spell of `team` over frame indices [low, high)."""
    own = [event for event in events if event.team == team and low <= event.start - first < high]
    if any(event.kind in CONTROL_EVENTS or (event.kind == CARRY_EVENT and event.end - event.start >= HOLD_SECONDS * FPS)
           for event in own):
        return True
    release = min((event.start - first for event in own if event.kind in RELEASE_EVENTS), default=high)
    return release - low >= HOLD_SECONDS * FPS


def control_labels(raw, events, first):
    """Raw targets with every uncontrolled spell given to the last team in control."""
    result = raw.copy()
    for low, high in live_segments(raw):
        current = None
        for start, stop, label in runs(raw[low:high]):
            team = int(label == pm.A)
            if current is None or controlled(team, low + start, low + stop, events, first):
                current = label
            result[low + start:low + stop] = current
    return result


def reader_checks(visible, events, first):
    """Reader-level evidence for the dead-ball convention: missing ball in flight and before restarts."""
    length = len(visible)
    start = kick_off(events, first, length)
    flight = np.zeros(length, dtype=bool)
    for event in events:
        if event.kind in FLIGHT_EVENTS and event.end >= event.start:
            flight[max(event.start - first, 0):max(min(event.end - first + 1, length), 0)] = True
    restarts = [event.start - first for event in events
                if event.kind == RESTART_EVENT and 0 <= event.start - first < length and event.start - first != start]
    before = [np.mean(~visible[max(frame - RESTART_WINDOW, 0):frame]) > .5 for frame in restarts
              if frame > 0]
    return {"flight_frames": int(flight.sum()), "flight_missing": int((flight & ~visible).sum()),
            "restarts": len(before), "restarts_after_missing": int(sum(before))}


def period_halves(frames, periods, positions, ball, teams, pitch, events, handover=HANDOVER_EVENTS):
    """Two 25 Hz halves: centred positions, ball, teams, pitch, raw and control targets, checks."""
    halves = []
    for period in (1, 2):
        rows = np.flatnonzero(periods == period)
        if len(rows) < 2 or np.any(np.diff(frames[rows]) != 1):
            raise ValueError(f"Metrica period {period} is missing or has non-contiguous frames")
        first = int(frames[rows[0]])
        own = [event for event in events if event.period == period]
        located = to_metres(ball[rows], pitch)
        visible = np.isfinite(located).all(axis=1)
        labels, counts = raw_labels(visible, own, first, handover)
        halves.append({"positions": to_metres(positions[rows], pitch), "ball": located, "teams": teams,
                       "pitch": np.asarray(pitch, dtype=np.float64), "labels": labels,
                       "control": control_labels(labels, own, first),
                       "checks": dict(counts, **reader_checks(visible, own, first))})
    return halves


def read_game(raw, game, sources=SOURCES, handover=HANDOVER_EVENTS):
    verify(raw, GAME_FILES[game], sources)
    paths = [raw / path for path in GAME_FILES[game]]
    if game == 3:
        metadata = epts_metadata(paths[0])
        with paths[1].open() as lines:
            tracking = epts_tracking(metadata, lines)
        return period_halves(*tracking, metadata["pitch"], json_events(paths[2], metadata["team_ids"]), handover)
    return period_halves(*csv_tracking(paths[0], paths[1]), PITCH, csv_events(paths[2]), handover)


# -- prepared halves -------------------------------------------------------------------------------

ARRAYS = ("positions", "ball", "teams", "pitch", "labels", "control")


def prepare(raw, out, sources=SOURCES):
    require_external_only()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"sources_sha256": digest(sources), "reader_sha256": digest(Path(__file__)), "parsed": {}, "checks": {}}
    for game in GAMES:
        for index, half in enumerate(read_game(raw, game, sources), 1):
            path = out / f"{game}_{index}.npz"
            np.savez_compressed(path, **{name: half[name] for name in ARRAYS})
            manifest["parsed"][path.name] = list(identity(path))
            manifest["checks"][path.name] = half["checks"]
            shares = {pm.LABELS[code]: round(float(np.mean(half["labels"] == code)), 4) for code in range(4)}
            print(f"Parsed Metrica game {game} half {index}: {len(half['labels'])} frames, raw shares {shares}, "
                  f"checks {half['checks']}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load(out, game):
    """Both halves of a prepared game, after checking them against the manifest."""
    parsed = json.loads((out / "manifest.json").read_text())["parsed"]
    halves = []
    for index in (1, 2):
        path = out / f"{game}_{index}.npz"
        if list(identity(path)) != parsed.get(path.name):
            raise ValueError(f"Prepared Metrica half differs from the manifest: {path.name}")
        with np.load(path, allow_pickle=False) as item:
            halves.append({name: item[name] for name in item.files})
    return halves


def tracking(half):
    return half["positions"], half["ball"], half["teams"], half["pitch"]


def inputs(parsed, sources=SOURCES):
    manifest = json.loads((parsed / "manifest.json").read_text())
    return {"sources_sha256": digest(sources), "reader_sha256": manifest["reader_sha256"],
            "parsed": manifest["parsed"]}


# -- scoring ---------------------------------------------------------------------------------------

def targets(halves):
    """Every D19 target over both halves of a game, with an X frame between them."""
    from dfl_submission.decoder_trial import CONVENTIONS, relabel
    raw = np.r_[halves[0]["labels"], pm.X, halves[1]["labels"]].astype(np.int8)
    result = {name: relabel(raw, seconds) for name, seconds in CONVENTIONS.items()}
    result["control"] = np.r_[halves[0]["control"], pm.X, halves[1]["control"]].astype(np.int8)
    return {name: result[name] for name in TARGETS}


def score(truth, predictions):
    """spellF1 and macroF1 of two halves of predicted label codes against every target."""
    predicted = np.r_[np.asarray(predictions[0]), pm.X, np.asarray(predictions[1])].astype(np.int8)
    result = {}
    for name, target in truth.items():
        spells = pm.spell_scores(target, predicted, pm.valid_runs(target))
        confusion = pm.scored_confusion(pm.confusion(target, predicted))
        result[name] = {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(confusion), "spells": spells["n_pred"]}
    return result


def means(matches):
    return {name: {metric: float(np.mean([match[name][metric] for match in matches.values()]))
                   for metric in ("spell_f1", "macro_f1")}
            for name in TARGETS}


def swapped(codes):
    codes = np.asarray(codes)
    return np.where(codes == pm.H, pm.A, np.where(codes == pm.A, pm.H, codes)).astype(np.int8)


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D19 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def convention_passed(checks):
    """D19's reader checks for one game, pooled over its halves."""
    flight = sum(item["flight_frames"] for item in checks)
    missing = sum(item["flight_missing"] for item in checks)
    restarts = sum(item["restarts"] for item in checks)
    after = sum(item["restarts_after_missing"] for item in checks)
    return {"flight_missing_share": missing / flight if flight else None,
            "restarts_after_missing_share": after / restarts if restarts else None,
            "passed": bool(flight and restarts and missing / flight <= FLIGHT_MISSING_MAX
                           and after / restarts >= RESTART_MISSING_MIN)}


def sanity_passed(baseline, swap, conventions):
    raw = means(baseline)["raw"]["spell_f1"]
    oriented = all(baseline[game]["raw"]["macro_f1"] > swap[game]["raw"]["macro_f1"] for game in baseline)
    return (SANITY_RANGE[0] <= raw <= SANITY_RANGE[1] and oriented
            and all(item["passed"] for item in conventions.values())), raw, oriented


def record_attempt(path, attempt):
    record = (json.loads(path.read_text()) if path.exists() else
              {"protocol": PROTOCOL, "range": SANITY_RANGE, "flight_missing_max": FLIGHT_MISSING_MAX,
               "restart_missing_min": RESTART_MISSING_MIN, "attempts": []})
    record["attempts"].append(attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def sanity(parsed=PARSED, output=RECIPE / "sanity.json"):
    """D19's reader gate: dead-ball convention checks and the smooth50 baseline, without our models."""
    require_external_only()
    from dfl_submission.vendor_trial import baseline_prediction
    manifest = json.loads((parsed / "manifest.json").read_text())
    baseline, swap, conventions, shares = {}, {}, {}, {}
    for game in GAMES:
        halves = load(parsed, game)
        truth = targets(halves)
        predictions = [baseline_prediction(half) for half in halves]
        baseline[str(game)] = score(truth, predictions)
        swap[str(game)] = score(truth, [swapped(codes) for codes in predictions])
        conventions[str(game)] = convention_passed([manifest["checks"][f"{game}_{i}.npz"] for i in (1, 2)])
        shares[str(game)] = {name: {pm.LABELS[code]: float(np.mean(target == code)) for code in range(4)}
                             for name, target in truth.items() if name in ("raw", "control")}
    passed, raw, oriented = sanity_passed(baseline, swap, conventions)
    record_attempt(output, {"passed": passed, "mean_raw_spell_f1": raw, "oriented": oriented,
                            "conventions": conventions, "baseline": baseline, "swapped": swap,
                            "label_shares": shares, "reader_checks": manifest["checks"], "inputs": inputs(parsed)})
    print(f"D19 reader gate: smooth50 mean raw spellF1 {raw:.4f}, orientation {'ok' if oriented else 'WRONG'}, "
          + ", ".join(f"game {game} flight missing {item['flight_missing_share']:.4f} restarts after missing "
                      f"{item['restarts_after_missing_share']:.3f}" for game, item in conventions.items())
          + f" -> {'passed' if passed else 'FAILED'}")
    return passed


def gate_passed(path, expected):
    latest = json.loads(path.read_text())["attempts"][-1]
    return latest["passed"] is True and latest["inputs"] == expected


def teacher_identity():
    from dfl_submission import competition_adaptation
    return digest(competition_adaptation.BLEND / "recipe.json")


# -- references: the submission a candidate is compared with ---------------------------------------

def record_path(reference):
    return RECIPE / f"{reference.lower()}_measurement.json"


def cache_path(reference):
    return CACHE if reference == "D3" else CACHE / reference.lower()


def d15_frozen():
    """D15's frozen recipe and network setting, after checking its confirmation and model identity."""
    from dfl_submission import scale_trial
    if json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D15 did not pass confirmation")
    return scale_trial.frozen_recipe(scale_trial.RECIPE, scale_trial.MODELS)


def reference_recipe(reference):
    """The decoder recipe of a reference submission."""
    if reference == "D3":
        from dfl_submission import competition_adaptation
        return json.loads((competition_adaptation.BLEND / "recipe.json").read_text())["recipe"]
    if reference == "D15":
        return d15_frozen()[0]["recipe"]
    raise ValueError(f"Unknown D19 reference {reference!r}; expected one of {REFERENCES}")


def reference_identity(reference, parsed):
    """What a reference's cached decoder inputs depend on."""
    record = {"protocol": PROTOCOL, "teacher_sha256": teacher_identity(), "inputs": inputs(parsed)}
    if reference == "D15":
        from dfl_submission import scale_trial
        frozen, _ = d15_frozen()
        record.update(reference="D15", scale_recipe_sha256=digest(scale_trial.RECIPE / "recipe.json"),
                      models=frozen["models"])
    elif reference != "D3":
        raise ValueError(f"Unknown D19 reference {reference!r}; expected one of {REFERENCES}")
    return record


def reference_model(reference):
    """The model `diagnosis.half_signals` runs for a reference, as its build constructs it."""
    from dfl_submission import competition_adaptation
    if reference == "D3":
        return competition_adaptation.reference()
    if reference == "D15":
        from dfl_submission import scale_trial
        frozen, setting = d15_frozen()
        original, packages, d3 = scale_trial.teacher(setting)
        return original, packages, dict(d3, recipe=frozen["recipe"])
    raise ValueError(f"Unknown D19 reference {reference!r}; expected one of {REFERENCES}")


def reference_halves(reference=STANDING, parsed=PARSED, cache=None, model=None):
    """A reference's decoder inputs and labels for every half, from its cache or computed and cached."""
    from dfl_submission import diagnosis
    cache = cache_path(reference) if cache is None else cache

    def produce():
        chosen = reference_model(reference) if model is None else model
        for game in GAMES:
            for index, half in enumerate(load(parsed, game)):
                signals, labels = diagnosis.half_signals(chosen, *tracking(half))
                yield diagnosis.HalfSignals("metrica", str(game), "observed", index, half["labels"], signals, labels)
            print(f"Forecast {reference} on Metrica game {game}", flush=True)

    halves = diagnosis.collect(cache, reference_identity(reference, parsed), produce)
    return {game: [half.d3 for half in halves if half.match == str(game)] for game in GAMES}, halves


def d3_halves(parsed=PARSED, cache=CACHE, teacher=None):
    """Frozen D3's decoder inputs and labels for every half."""
    return reference_halves("D3", parsed, cache, teacher)


def reference_scores(reference, parsed=PARSED, cache=None, record=None):
    """A reference's per-game scores from its cache, checked against its measurement record."""
    labels, halves = reference_halves(reference, parsed, cache)
    truths = {game: targets(load(parsed, game)) for game in GAMES}
    scores = {str(game): score(truths[game], labels[game]) for game in GAMES}
    record = record_path(reference) if record is None else record
    if json.loads(json.dumps(scores)) != json.loads(record.read_text())["matches"]:
        raise ValueError(f"{reference} from the cache does not reproduce its D19 measurement record")
    return scores, halves, truths


def measure(reference=STANDING, parsed=PARSED, cache=None, output=None, gate=RECIPE / "sanity.json"):
    """Score a reference submission once on the three games (D19 and its amendment)."""
    require_external_only()
    if not gate_passed(gate, inputs(parsed)):
        raise ValueError("The latest D19 reader attempt did not pass on these parsed inputs")
    output = record_path(reference) if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; {reference} is measured once")
    if reference != "D3":
        d3_scores = reference_scores("D3", parsed)[0]  # stops unless D3 reproduces its record
    labels, _ = reference_halves(reference, parsed, cache)
    matches = {str(game): score(targets(load(parsed, game)), labels[game]) for game in GAMES}
    summary = means(matches)
    record = {"protocol": PROTOCOL, "candidate": reference, "decision": None, "means": summary,
              "matches": matches, "teacher_sha256": teacher_identity(), "inputs": inputs(parsed)}
    if reference != "D3":
        record.update(identity=reference_identity(reference, parsed), recipe=reference_recipe(reference),
                      reproduction={"d3": True}, change_against_d3=changes(d3_scores, matches))
    write_new(output, record)
    for name, values in summary.items():
        print(f"{reference} on the Metrica sample games, {name}: spellF1 {values['spell_f1']:.5f}, "
              f"macroF1 {values['macro_f1']:.5f}")
    if reference != "D3":
        print(f"{reference} against D3 (reported only): mean N = 3 spellF1 {record['change_against_d3']['primary']:+.5f}")
    return summary


# -- veto ------------------------------------------------------------------------------------------

def as_codes(labels):
    labels = np.asarray(labels)
    return pm.label_codes(labels) if labels.dtype.kind in "UO" else labels.astype(np.int8)


def candidate_labels(candidate, halves, recipe):
    """Two halves of label codes per game from labels, decoder inputs or a function of cached inputs.

    `candidate` is one of:
    - {game: (labels_1, labels_2)}: label codes or letters per half;
    - {game: (signals_1, signals_2)}: the four decoder inputs (home, dead, dead_edge, owner_edge) as
      float arrays per half, decoded with `recipe`;
    - a function of one `diagnosis.HalfSignals` (the reference's cached inputs, a tracking-free view
      of a half) returning that half's labels. Tracking itself is available from `load` and `tracking`.
    """
    from dfl_submission import diagnosis
    if callable(candidate):
        grouped = {}
        for half in halves:
            grouped.setdefault(int(half.match), []).append(as_codes(candidate(half)))
        return grouped
    result = {}
    for game in GAMES:
        parts = candidate[game] if game in candidate else candidate[str(game)]
        result[game] = [diagnosis.decode(tuple(part), recipe) if is_signals(part) else as_codes(part)
                        for part in parts]
    return result


def is_signals(part):
    """Four float arrays (home, dead, dead_edge, owner_edge) rather than one half's labels."""
    return (isinstance(part, tuple) and len(part) == 4
            and all(isinstance(item, np.ndarray) and item.dtype.kind == "f" for item in part))


def changes(reference, challenger):
    """Mean spellF1 change against the reference per target, the worst game at the primary, and the veto."""
    before, after = means(reference), means(challenger)
    delta = {name: after[name]["spell_f1"] - before[name]["spell_f1"] for name in TARGETS}
    worst = min(challenger[game][PRIMARY]["spell_f1"] - reference[game][PRIMARY]["spell_f1"] for game in reference)
    return {"primary": delta[PRIMARY], "targets": delta, "worst_game": worst, "passed": delta[PRIMARY] >= -VETO_LOSS}


def veto(candidate, *, reference=STANDING, parsed=PARSED, cache=None, recipe=None, record=None):
    """Score a candidate on the three sample games against a reference submission (DECISIONS.md D19).

    `reference` is the submission standing when the candidate's rule was written: D15 by default, D3
    for decisions written before D15 was adopted. Returns the candidate's per-game scores and means,
    the reference's, and `changes`, whose `passed` is the veto: mean N = 3 spellF1 may drop at most
    0.01 below the reference. The reference is decoded from its cached inputs and must reproduce its
    measurement record exactly before the candidate is compared. Decoder inputs are decoded with
    `recipe`, by default the reference's.
    """
    require_external_only()
    scores, halves, truths = reference_scores(reference, parsed, cache, record)
    labels = candidate_labels(candidate, halves, reference_recipe(reference) if recipe is None else recipe)
    challenger = {str(game): score(truths[game], labels[game]) for game in GAMES}
    return {"protocol": PROTOCOL, "changes": changes(scores, challenger),
            "candidate": {"means": means(challenger), "matches": challenger},
            "reference": {"name": reference, "means": means(scores), "matches": scores}}


def load_labels(path):
    """Labels saved as an .npz with one array per half, named `<game>_<half>` (e.g. `3_2`)."""
    with np.load(path, allow_pickle=False) as item:
        return {game: (item[f"{game}_1"], item[f"{game}_2"]) for game in GAMES}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "pin", "prepare"):
        command = commands.add_parser(name)
        command.add_argument("--raw", type=Path, default=RAW)
        if name == "prepare":
            command.add_argument("--out", type=Path, default=PARSED)
    for name in ("sanity", "measure", "veto"):
        command = commands.add_parser(name)
        command.add_argument("--parsed", type=Path, default=PARSED)
        if name != "sanity":
            command.add_argument("--reference", choices=REFERENCES, default=STANDING)
        if name == "veto":
            command.add_argument("--labels", type=Path, required=True, help="npz with arrays named <game>_<half>")
    args = parser.parse_args()
    require_external_only()
    if args.command == "fetch":
        fetch(args.raw)
    elif args.command == "pin":
        pin(args.raw)
    elif args.command == "prepare":
        prepare(args.raw, args.out)
    elif args.command == "sanity":
        sanity(args.parsed)
    elif args.command == "measure":
        measure(args.reference, args.parsed)
    else:
        result = veto(load_labels(args.labels), reference=args.reference, parsed=args.parsed)
        delta = result["changes"]
        print(f"D19 veto: mean N = 3 spellF1 change {delta['primary']:+.5f} against {args.reference}, worst game "
              f"{delta['worst_game']:+.4f}, " + ", ".join(f"{name} {value:+.4f}" for name, value in delta["targets"].items())
              + f" -> {'passed' if delta['passed'] else 'VETOED'}")


if __name__ == "__main__":
    main()
