"""Synthetic checks of the PFF reader. No PFF data is committed; the fixtures mimic its field names."""

import bz2
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dfl_submission import pff
from evaluation import possession_metrics as pm


FPS = 29.97
METADATA = {"id": "3812", "homeTeam": {"id": "10"}, "awayTeam": {"id": "20"}, "fps": FPS,
            "homeTeamStartLeft": True, "stadium": {"pitches": [{"length": 105.0, "width": 68.0}]}}
ROSTER = [{"team": {"id": "10"}, "shirtNumber": "9"}, {"team": {"id": "10"}, "shirtNumber": "1"},
          {"team": {"id": "20"}, "shirtNumber": "9"}]


def player(jersey, x, y, visibility="VISIBLE"):
    return {"jerseyNum": str(jersey), "visibility": visibility, "confidence": "HIGH", "x": x, "y": y}


def row(period, number, home=(), away=(), balls=(), event=None):
    return {"period": period, "frameNum": number, "homePlayers": list(home), "awayPlayers": list(away),
            "balls": list(balls), "game_event": event,
            # Smoothed fields must never be read.
            "homePlayersSmoothed": [player(9, 40.0, 30.0)], "awayPlayersSmoothed": [player(9, 40.0, 30.0)],
            "ballsSmoothed": {"visibility": "VISIBLE", "x": 40.0, "y": 30.0, "z": 0.0}}


def event(kind, number, home_ball=None, sequence=0):
    return {"game_event_type": kind, "start_frame": number, "home_ball": home_ball, "sequence": sequence}


def test_sample_points_resample_by_time():
    left, fraction = pff.sample_points(31, FPS)
    assert len(left) == 26  # 30 native steps of 1/29.97 s cover 1.001 s, so 25 Hz frames 0..25
    np.testing.assert_allclose(left + fraction, np.arange(26) * FPS / 25)
    native = np.arange(31, dtype=np.float32)[:, None].repeat(2, axis=1)
    native[3] = np.nan
    resampled = pff.resample(native, left, fraction)
    np.testing.assert_allclose(resampled[:, 0][[0, 1, 5]], [0, FPS / 25, 5 * FPS / 25], rtol=1e-6)
    # 25 Hz frame 2 sits between native frames 2 and 3; a missing native point is never filled.
    assert np.isnan(resampled[2]).all() and np.isnan(resampled[3]).all()


def test_tracking_reads_visible_unsmoothed_points_only():
    rows = [row(1, 100 + i, home=[player(9, 1.0, 2.0), player(1, 3.0, 4.0, "ESTIMATED")],
                away=[player(9, -5.0, -6.0)], balls=[{"visibility": "VISIBLE", "x": 7.0, "y": 8.0, "z": 0.1}])
            for i in range(31)]
    rows[0]["balls"] = [{"visibility": "ESTIMATED", "x": 7.0, "y": 8.0, "z": 0.1}]
    rows[1]["balls"] = [{"visibility": "VISIBLE", "x": 60.0, "y": 0.0, "z": 0.1}]  # beyond the 5 m limit
    rows[2]["balls"] = [{"visibility": "VISIBLE", "x": 1.0, "y": 0.0}, {"visibility": "VISIBLE", "x": 2.0, "y": 0.0}]
    rows += [row(2, 500 + i, home=[player(9, 1.0, 2.0)]) for i in range(31)]
    rows += [row(3, 900, home=[player(9, 1.0, 2.0)]), row(None, 901)]
    first, second = pff.parse(METADATA, ROSTER, rows)
    np.testing.assert_array_equal(first["teams"], [0, 0, 1])  # home 1, home 9, away 9
    np.testing.assert_array_equal(first["pitch"], [105.0, 68.0])
    assert first["positions"].shape == (26, 3, 2) and first["positions"].dtype == np.float32
    np.testing.assert_allclose(first["positions"][:, 1], np.tile([1.0, 2.0], (26, 1)))
    assert np.isnan(first["positions"][:, 0]).all()  # estimated, never visible
    np.testing.assert_allclose(first["positions"][:, 2], np.tile([-5.0, -6.0], (26, 1)))
    # Native frames 0 (estimated), 1 (outlier) and 2 (two visible balls) have no ball.
    assert np.isnan(first["ball"][:3]).all()
    np.testing.assert_allclose(first["ball"][3:], np.tile([7.0, 8.0], (23, 1)))
    assert second["positions"].shape == (26, 3, 2) and np.isnan(second["ball"]).all()


def test_labels_follow_provider_events():
    def at(offset, kind, home_ball=None):
        return event(kind, 1000 + offset, home_ball, offset)
    events = [at(10, "FIRSTKICKOFF", True), at(40, "OTB", False), at(50, "SUB", True), at(70, "OUT", True),
              at(100, "OTB", True), at(130, "VID"), at(150, "OTB", None), at(180, "END", False)]
    present = np.ones(200, dtype=bool)
    present[[20, 21]] = False
    labels = pff.native_labels(events, 1000, present)
    expected = np.full(200, pm.X, dtype=np.int8)
    expected[10:40], expected[40:70], expected[70:100], expected[100:130] = pm.H, pm.A, pm.D, pm.H
    expected[150:180] = pm.H  # an on-the-ball event without a team keeps the last team
    expected[[20, 21]] = pm.X  # no tracking frame from the vendor
    np.testing.assert_array_equal(labels, expected)


def test_parse_takes_events_from_tracking_rows():
    rows = [row(period, start + i, home=[player(9, 0.0, 0.0)]) for period, start in ((1, 0), (2, 5000))
            for i in range(61)]
    rows[0]["game_event"] = event("FIRSTKICKOFF", 0, False)
    rows[1]["game_event"] = event("FIRSTKICKOFF", 0, False)  # repeated on the next frame
    rows[30]["game_event"] = event("OUT", 30, False)
    rows[61 + 12]["game_event"] = event("SECONDKICKOFF", 5012, True)
    first, second = pff.parse(METADATA, ROSTER, rows)
    native = np.r_[np.full(30, pm.A), np.full(31, pm.D)]
    np.testing.assert_array_equal(first["labels"], native[pff.sample_points(61, FPS)[0]])
    assert second["labels"][:10].tolist() == [pm.X] * 10 and second["labels"][11:].tolist() == [pm.H] * 40


def test_unknown_jersey_is_an_error():
    rows = [row(period, i, home=[player(77, 0.0, 0.0)]) for period in (1, 2) for i in range(3)]
    try:
        pff.parse(METADATA, ROSTER, rows)
    except ValueError as error:
        assert "jersey" in str(error)
    else:
        raise AssertionError("an unrostered jersey must not be dropped silently")


def test_split_interleaves_sorted_game_ids():
    games = list(range(3000, 3064))[::-1]
    split = pff.split(games)
    ordered = sorted(games)
    assert split["development"] == tuple(ordered[0::8])
    assert split["confirm_a"] == tuple(ordered[1::8]) and split["confirm_c"] == tuple(ordered[3::8])
    assert split["untouched"] == tuple(ordered[7::8])
    assert len(split["train"]) == 24 and sorted(sum(split.values(), ())) == ordered
    try:
        pff.split(games[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("the D7 split is defined for exactly 64 games")


def write_match(raw, game, rows):
    for folder, name, payload in (("Metadata", f"{game}.json", json.dumps([dict(METADATA, id=str(game))])),
                                  ("Rosters", f"{game}.json", json.dumps(ROSTER))):
        (raw / folder).mkdir(parents=True, exist_ok=True)
        (raw / folder / name).write_text(payload)
    (raw / "Tracking Data").mkdir(exist_ok=True)
    with bz2.open(raw / "Tracking Data" / f"{game}.jsonl.bz2", "wt") as handle:
        handle.writelines(json.dumps(item) + "\n" for item in rows)


def test_prepare_pins_and_verifies_sources():
    rows = [row(period, i, home=[player(9, 0.0, 0.0)]) for period in (1, 2) for i in range(31)]
    with TemporaryDirectory() as folder:
        raw, out = Path(folder) / "raw", Path(folder) / "parsed"
        write_match(raw, 3812, rows)
        assert pff.games(raw) == (3812,)
        sources = Path(folder) / "sources.json"
        pff.pin(raw, sources)
        pff.prepare(raw, out, sources, (3812,))
        halves = pff.load(out, 3812)
        assert len(halves) == 2 and halves[0]["positions"].shape == (26, 3, 2)
        np.savez_compressed(out / "3812_1.npz", **dict(halves[0], positions=halves[0]["positions"] + 1))
        try:
            pff.load(out, 3812)
        except ValueError:
            pass
        else:
            raise AssertionError("a changed parsed file must be rejected")
        write_match(raw, 3812, rows[:-1])
        try:
            pff.prepare(raw, Path(folder) / "again", sources, (3812,))
        except ValueError:
            pass
        else:
            raise AssertionError("a changed raw file must be rejected")


def test_baseline_codes_pick_the_nearest_visible_player():
    from dfl_submission import vendor_trial
    from starter.baseline import AWAY, HOME, NOBODY
    positions = np.full((4, 3, 2), np.nan, dtype=np.float32)
    positions[:3, 0], positions[:3, 2] = (0.0, 0.0), (10.0, 0.0)
    ball = np.array([[1.0, 0.0], [9.0, 0.0], [np.nan, np.nan], [5.0, 5.0]], dtype=np.float32)
    codes = vendor_trial.baseline_codes(positions, ball, np.array([0, 0, 1], dtype=np.int8))
    np.testing.assert_array_equal(codes, [HOME, AWAY, NOBODY, NOBODY])


def test_scores_cover_every_convention_and_split_the_halves():
    from dfl_submission import decoder_trial, vendor_trial
    first = np.r_[np.full(200, pm.H), np.full(100, pm.D), np.full(200, pm.A)].astype(np.int8)
    second = np.r_[np.full(30, pm.X), np.full(300, pm.A)].astype(np.int8)
    perfect = vendor_trial.score_match([first, second], [first, np.where(second == pm.X, pm.H, second)])
    assert set(perfect) == set(decoder_trial.CONVENTIONS)
    assert all(score["spell_f1"] == 1.0 and score["macro_f1"] == 1.0 for score in perfect.values())
    swapped = vendor_trial.score_match([first, second], [np.where(first == pm.H, pm.A, first), second])
    assert swapped["raw"]["spell_f1"] < 1.0


def test_sanity_range_is_inclusive():
    from dfl_submission import vendor_trial
    assert [vendor_trial.sane(value) for value in (0.149, 0.15, 0.40, 0.401)] == [False, True, True, False]


def test_trial_commands_require_external_only():
    import os
    from unittest.mock import patch
    from dfl_submission import vendor_trial
    for command in (vendor_trial.sanity, vendor_trial.measure):
        with patch.dict(os.environ, {"MTAH_DATA_DIR": str(ROOT_FOR_TESTS)}):
            try:
                command(Path("missing"))
            except SystemExit:
                continue
            raise AssertionError(f"{command.__name__} must refuse while the bundle path exists")


ROOT_FOR_TESTS = Path(__file__).resolve().parents[1]


def test_sanity_attempts_accumulate_and_gate_the_measured_inputs():
    from dfl_submission import vendor_trial
    with TemporaryDirectory() as folder:
        gate = Path(folder) / "sanity.json"
        first = {"passed": False, "inputs": {"parsed": {"1_1.npz": [1, "a"]}}}
        second = {"passed": True, "inputs": {"parsed": {"1_1.npz": [1, "b"]}}}
        vendor_trial.record_attempt(gate, first)
        vendor_trial.record_attempt(gate, second)
        assert [item["passed"] for item in json.loads(gate.read_text())["attempts"]] == [False, True]
        assert vendor_trial.gate_passed(gate, second["inputs"])
        assert not vendor_trial.gate_passed(gate, first["inputs"])  # re-parsed data needs a new attempt
        vendor_trial.record_attempt(gate, first)
        assert not vendor_trial.gate_passed(gate, first["inputs"])


def test_frame_rate_may_sit_in_the_videos_dict():
    metadata = {key: value for key, value in METADATA.items() if key != "fps"}
    metadata["videos"] = {"fps": FPS}
    rows = [row(period, i, home=[player(9, 0.0, 0.0)]) for period in (1, 2) for i in range(31)]
    first, _ = pff.parse(metadata, ROSTER, rows)
    assert len(first["labels"]) == 26


def test_command_line_entry_points_import_on_their_own():
    import os
    import subprocess
    import sys
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    for module in ("dfl_submission.pff", "dfl_submission.vendor_trial"):
        result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT_FOR_TESTS, env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, (module, result.stderr[-500:])
