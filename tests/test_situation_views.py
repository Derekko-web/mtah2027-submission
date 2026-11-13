"""D55's height-aware DFL reader and situation views on synthetic fixtures."""

from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET

import numpy as np

from dfl_submission import hidden_views as hv, situation_views as sv
from dfl_transfer_data import parse_match
from evaluation import possession_metrics as pm


def xml_fixture(folder, *, z_missing=False):
    """A two-half DFL pair with heights: frame 3 of the first half is absent, X = 70 lies outside the pitch."""
    match = "DFL-MAT-SYNTH"
    root = ET.Element("PutDataRequest")
    info = ET.SubElement(root, "MatchInformation")
    ET.SubElement(info, "General", MatchId=match, HomeTeamId="home", GuestTeamId="away")
    ET.SubElement(info, "Environment", PitchX="105", PitchY="68")
    teams = ET.SubElement(info, "Teams")
    for team in ("home", "away"):
        players = ET.SubElement(ET.SubElement(teams, "Team", TeamId=team), "Players")
        for i in range(11):
            ET.SubElement(players, "Player", PersonId=team + str(i))
    info_path = folder / "info.xml"
    ET.ElementTree(root).write(info_path)
    root = ET.Element("PutDataRequest")
    positions = ET.SubElement(root, "Positions", MatchId=match)
    rows = {"firstHalf": (10, [(0.5, 1.25, 0.0, 0), (1.0, 2.0, 0.11, 1), (70.0, 0.0, 2.5, 1), None, (3.0, -1.0, 1.6, 1)]),
            "secondHalf": (100, [(-2.0, 0.5, 1.5, 1), (-2.5, 0.25, 4.0, 1), (0.0, 0.0, 0.0, 0)])}
    for half, (start, frames) in rows.items():
        ball = ET.SubElement(positions, "FrameSet", TeamId="BALL", GameSection=half, PersonId="ball")
        for j, values in enumerate(frames):
            if values is None:
                continue
            x, y, z, status = values
            n = start + j
            stamp = (datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=n * .04)).isoformat()
            attributes = dict(N=str(n), T=stamp, X=str(x), Y=str(y), BallStatus=str(status), BallPossession="1")
            if not z_missing:
                attributes["Z"] = str(z)
            ET.SubElement(ball, "Frame", **attributes)
        for team in ("home", "away"):
            for i in range(11):
                frame_set = ET.SubElement(positions, "FrameSet", TeamId=team, GameSection=half, PersonId=team + str(i))
                for j, values in enumerate(frames):
                    if values is None:
                        continue
                    n = start + j
                    stamp = (datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=n * .04)).isoformat()
                    ET.SubElement(frame_set, "Frame", N=str(n), T=stamp, X=str(i - 5), Y="5" if team == "home" else "-5")
    positions_path = folder / "positions.xml"
    ET.ElementTree(root).write(positions_path)
    return info_path, positions_path, match


def test_parse_ball_returns_height_on_the_existing_reader_s_frames(tmp_path):
    info, positions, match = xml_fixture(tmp_path)
    existing, _ = parse_match(info, positions, match)
    height = sv.parse_ball(info, positions, match)
    first = height["firstHalf"]
    assert np.array_equal(first["frames"], np.arange(10, 15))
    assert first["ball"].dtype == np.float32
    assert np.isnan(first["ball"][2]).all() and np.isnan(first["ball"][3]).all()   # outside the pitch; absent frame
    assert np.isclose(first["z"][2], 2.5) and np.isnan(first["z"][3])               # Z kept even outside the pitch
    assert np.allclose(height["secondHalf"]["z"], [1.5, 4.0, 0.0])
    check = sv.reader_check(height, existing)
    assert sv.reader_passed(check)
    part = check["firstHalf"]
    assert part["ball_records"] == 4 and part["numeric_z_on_every_record"]
    assert part["dead_frames"] == 1 and part["dead_frames_z_zero"] == 1
    assert part["live_frames"] == 3 and np.isclose(part["live_share_above_air_height"], 2 / 3)
    assert np.isclose(check["secondHalf"]["live_share_above_air_height"], .5)   # 1.5 itself is not above 1.5


def test_reader_check_fails_on_any_difference_and_parse_ball_needs_z(tmp_path):
    info, positions, match = xml_fixture(tmp_path)
    existing, _ = parse_match(info, positions, match)
    height = sv.parse_ball(info, positions, match)
    height["firstHalf"]["ball"] = height["firstHalf"]["ball"].copy()
    height["firstHalf"]["ball"][1, 0] += np.float32(1e-3)
    assert not sv.reader_passed(sv.reader_check(height, existing))
    other = tmp_path / "other"
    other.mkdir()
    info, positions, match = xml_fixture(other, z_missing=True)
    try:
        sv.parse_ball(info, positions, match)
    except ValueError as error:
        assert "height" in str(error)
    else:
        raise AssertionError("A file without Z was read")


def test_air_and_crowd_masks():
    assert sv.air_mask([0.0, 1.5, 1.51, np.nan, 3.0]).tolist() == [False, False, True, False, True]
    ball = np.array([[0, 0], [0, 0], [0, 0], [np.nan, np.nan]], dtype=np.float32)
    positions = np.full((4, 5, 2), 50.0, dtype=np.float32)
    positions[0, :3] = [[3.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]       # three within 3 m (one exactly 3 m)
    positions[1, :3] = [[3.1, 0.0], [0.0, 1.0], [-1.0, -1.0]]       # only two
    positions[2, :3] = [[1.0, 0.0], [0.0, 1.0], [np.nan, np.nan]]   # a hidden player does not count
    positions[3, :3] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]         # no ball, no crowd
    assert sv.crowd_mask(positions, ball).tolist() == [True, False, False, False]


def test_change_mask_hides_one_second_before_to_two_after_direct_changes_only():
    target = np.full(400, pm.H, dtype=np.int8)
    target[100:] = pm.A                    # a direct change at 100
    target[200:210] = pm.D                 # A -> D -> H: a restart, not a change
    target[210:] = pm.H
    target[390:] = pm.A                    # a change at 390, clipped at the end
    assert sv.ownership_changes(target).tolist() == [100, 390]
    hidden = sv.change_mask(target)
    assert np.flatnonzero(hidden).tolist() == list(range(75, 150)) + list(range(365, 400))


def test_place_runs_keeps_the_runs_apart_and_their_lengths():
    lengths = [50, 7, 120, 1, 33, 33]
    for seed in range(20):
        flags, starts, placed = sv.place_runs(lengths, 400, np.random.default_rng(seed))
        found_starts, found = sv.runs_of(flags)
        assert sorted(found.tolist()) == sorted(lengths)
        assert found_starts.tolist() == starts.tolist() and found.tolist() == placed.tolist()
    flags, _, _ = sv.place_runs(lengths, 400, np.random.default_rng(3))
    again, _, _ = sv.place_runs(lengths, 400, np.random.default_rng(3))
    assert np.array_equal(flags, again)
    tight, _, _ = sv.place_runs([3, 3], 7, np.random.default_rng(0))
    assert tight.tolist() == [True] * 3 + [False] + [True] * 3
    try:
        sv.place_runs([4, 3], 7, np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("Runs that cannot be kept apart were placed")


def tracking(n=3000, players=22, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-40, 40, (n, players, 2)).astype(np.float32)
    ball = rng.uniform(-40, 40, (n, 2)).astype(np.float32)
    teams = np.repeat(np.array([0, 1], dtype=np.int8), players // 2)
    return positions, ball, teams


def test_situation_pair_shares_noise_and_moves_only_the_hidden_runs():
    positions, ball, _ = tracking()
    hidden = np.zeros(len(ball), dtype=bool)
    hidden[100:180] = hidden[1000:1003] = hidden[2500:2900] = True
    (vp, vb), (rp, rb), placed = sv.situation_pair(positions, ball, hidden, 931234)
    reference_positions, reference_ball = hv.noise_and_dropout(positions, ball, np.random.default_rng(931234))
    assert np.array_equal(vp, reference_positions, equal_nan=True) and np.array_equal(rp, vp, equal_nan=True)
    assert np.array_equal(np.isnan(vb).any(axis=1), hidden)
    assert np.array_equal(vb[~hidden], reference_ball[~hidden]) and np.array_equal(rb[~placed], reference_ball[~placed])
    assert np.array_equal(np.isnan(rb).any(axis=1), placed)
    assert sorted(sv.runs_of(placed)[1].tolist()) == [3, 80, 400]
    assert not np.array_equal(placed, hidden)
    stats = sv.run_statistics([vb]), sv.run_statistics([rb])
    assert stats[0] == stats[1]


def test_duplicates_add_one_track_per_team_within_a_metre_of_the_nearest_player():
    positions, ball, teams = tracking(n=4000, seed=1)
    view = positions + np.float32(.1)
    view[200:260, 3] = np.nan                         # a dropped stretch: the copy is dropped with it
    dup_positions, dup_teams, runs = sv.add_duplicates(view, positions, ball, teams, 971234)
    assert dup_positions.shape == (4000, 24, 2) and dup_teams.tolist() == teams.tolist() + [0, 1]
    assert np.array_equal(dup_positions[:, :22], view, equal_nan=True)
    covered = sum(run["length"] for run in runs)
    assert 400 <= covered < 400 + 125 and all(50 <= run["length"] <= 125 for run in runs)
    starts = [run["start"] for run in runs]
    assert starts == sorted(starts)
    present = np.isfinite(dup_positions[:, 22:]).all(axis=2)
    inside = np.zeros(4000, dtype=bool)
    for run in runs:
        start, stop = run["start"], run["start"] + run["length"]
        inside[start:stop] = True
        nearest = int(np.argmin(np.linalg.norm(positions[start] - ball[start], axis=1)))
        assert run["player"] == nearest and run["team"] == int(teams[nearest]) and run["offset_m"] <= 1.0
        copy = dup_positions[start:stop, 22 + run["team"]]
        seen = np.isfinite(view[start:stop, nearest]).all(axis=1)
        assert np.array_equal(np.isfinite(copy).all(axis=1), seen)
        gap = np.linalg.norm(copy[seen] - view[start:stop, nearest][seen], axis=1)
        assert np.allclose(gap, run["offset_m"], atol=1e-4)
        assert not np.isfinite(dup_positions[start:stop, 23 - run["team"]]).any()
    assert not present[~inside].any()
    again = sv.add_duplicates(view, positions, ball, teams, 971234)
    assert np.array_equal(again[0], dup_positions, equal_nan=True) and again[2] == runs


def test_run_statistics_by_hand():
    ball = np.zeros((1000, 2), dtype=np.float32)
    ball[0:50] = ball[100:200] = ball[400:700] = np.nan
    labels = np.full(1000, pm.D, dtype=np.int8)
    labels[0:500] = pm.H
    stats = sv.run_statistics([ball], [labels])
    assert stats["hidden_frames"] == 450 and np.isclose(stats["visibility"], .55)
    assert stats["hidden_runs"] == 3 and np.isclose(stats["mean_hidden_run_s"], 6.0)
    assert np.isclose(stats["hidden_share_in_runs_over_3s"], 400 / 450)
    assert np.isclose(stats["hidden_share_in_runs_over_10s"], 300 / 450)
    assert stats["live_frames"] == 500 and stats["live_hidden_frames"] == 250 and np.isclose(stats["live_visibility"], .5)


def test_seeds_are_new_and_distinct():
    import re
    from pathlib import Path
    package = Path(sv.__file__).parent
    used = {270000, 370000, 520000, 530000, 610000, 790000, *hv.SEEDS.values()}
    for name in ("metrica_trial.py", "metrica_views.py", "broadcast_trial.py"):
        block = re.search(r"^SEEDS = \{([^}]*)\}", (package / name).read_text(), re.M).group(1)
        used |= {int(value) for value in re.findall(r":\s*(\d+)", block)}
    assert {640000, 870000, 920000} <= used
    mine = {seed + offset for seed in sv.SEEDS.values() for offset in (0, sv.PLACEMENT)}
    assert len(mine) == 2 * len(sv.SEEDS)
    for seed in mine:
        assert all(abs(seed - other) >= 1000 for other in used), seed
