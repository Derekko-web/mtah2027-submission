"""Load the pinned 2019–20 SkillCorner release for the D6 proxy."""

from hashlib import sha1
import json

import numpy as np

from dfl_submission.skillcorner import at_25hz, fetch, sample_25hz
from evaluation import possession_metrics as pm


REVISION = "1b99d401ddc82e366c5c20d6860878668fd5858f"
RAW = f"https://raw.githubusercontent.com/SkillCorner/opendata/{REVISION}/data/matches"
BLOBS = {
    2068: ("27760525d6ced01a11b0ef72502f3c7b09817640", "0144001ad8ab81687517c1be9816ba9696dea9ae"),
    2269: ("86ea889735d1b942850e83e05311f58fcfbdcec5", "73be7a4c93e7b0f79115b496c933e69874fccd3a"),
    2417: ("9ec7c9ea82f9fc053a1afa24c2316a39370f9fa4", "d9230a6d14c281d83ea16f45c4377b04a593a821"),
    2440: ("4ccdba9d49b2d1be218d5bd08c9aa9552e3467b5", "5e840ded4a4a5fe3a5586e976c033407e3e7f7b0"),
    2841: ("f39e3776d6b682c097018f704294076b5b9d6841", "efb38c2ce5744b7f411893aa00ab51c4a4bd5923"),
    3442: ("2d773eac7d54df953b51ea606f9cca82a9fc7ccd", "18128315b2be1581bf93efbe529c5dcfebff69eb"),
    3518: ("ce7f165733840d9adc911a51cb20e7dc6d727714", "e2f6b889de7ddefe73dcdf0f6289f757eb735f13"),
    3749: ("21c727b486aaa1f6a6cc5b62301e395127875f7e", "91dc625afba303204ce9e6ea02fab66e3634a33c"),
    4039: ("8af6634222dea82b895c8bfb0258ceb90bfe5b0e", "ba8420673320c984c140a49285cae9dd8504258f"),
}
DEVELOPMENT = tuple(sorted(BLOBS)[:5])
HOLDOUT = tuple(sorted(BLOBS)[5:])


def load_blob(folder, match, name, expected):
    path = fetch(f"{RAW}/{match}/{name}", folder / f"{match}_{name}")
    data = path.read_bytes()
    digest = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if digest != expected:
        raise ValueError(f"SkillCorner 2020 blob differs from pinned revision: {path.name}")
    return json.loads(data)


def native_halves(rows):
    for period in (1, 2):
        half = [row for row in rows if row["period"] == period]
        frames = np.asarray([row["frame"] for row in half], dtype=int)
        if len(frames) < 2 or np.any(np.diff(frames) <= 0):
            raise ValueError(f"Invalid SkillCorner native frame order in period {period}")
        yield half, frames - frames[0]


def tracking(metadata, rows):
    sides = {metadata[key]["id"]: team for team, key in enumerate(("home_team", "away_team"))}
    roster = sorted((p["trackable_object"], sides[p["team_id"]]) for p in metadata["players"])
    slots = {identifier: index for index, (identifier, _) in enumerate(roster)}
    teams = np.array([team for _, team in roster], dtype=np.int8)
    pitch = np.array([metadata["pitch_length"], metadata["pitch_width"]])
    ball_id = metadata["ball"]["trackable_object"]
    for half, frames in native_halves(rows):
        length = frames[-1] + 1
        positions = np.full((length, len(slots), 2), np.nan, dtype=np.float32)
        ball = np.full((length, 2), np.nan, dtype=np.float32)
        for frame, row in zip(frames, half):
            for item in row["data"]:
                identifier = item.get("trackable_object")
                if identifier == ball_id:
                    ball[frame] = item["x"], item["y"]
                elif identifier in slots:
                    positions[frame, slots[identifier]] = item["x"], item["y"]
        for points in (positions, ball):
            points[~(np.isfinite(points) & (np.abs(points) <= pitch / 2 + 5)).all(axis=-1)] = np.nan
        yield at_25hz(positions)[0], at_25hz(ball)[0], teams, pitch


def read_tracking(folder, match):
    metadata, tracking_blob = BLOBS[match]
    return list(tracking(load_blob(folder, match, "match_data.json", metadata),
                         load_blob(folder, match, "structured_data.json", tracking_blob)))


def ownership(rows):
    sides = {"home team": pm.H, "away team": pm.A}
    for half, frames in native_halves(rows):
        labels = np.full(frames[-1] + 1, pm.X, dtype=np.int8)
        for frame, row in zip(frames, half):
            labels[frame] = sides.get((row["possession"] or {}).get("group"), pm.X)
        left, _ = sample_25hz(len(labels))
        yield labels[left]


def read_ownership(folder, match):
    return list(ownership(load_blob(folder, match, "structured_data.json", BLOBS[match][1])))
