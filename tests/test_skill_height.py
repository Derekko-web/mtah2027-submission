"""D57's SkillCorner height reader, SZ's hide and views, and the SkillCorner air view on synthetic fixtures."""

import csv
import json

import numpy as np

from dfl_submission import situation_train as s56, situation_views as sv, skill_height as sh, skillcorner
from dfl_submission.external_benchmark import Half
from dfl_submission.pff import digest


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


def synthetic_match(folder, *, start=(10, 24020), length=24000):
    """Three SkillCorner files: two halves of 24,000 native frames, two players, a ball with and without z."""
    metadata = {"home_team": {"id": 1}, "away_team": {"id": 2}, "pitch_length": 105, "pitch_width": 68,
                "players": [{"id": 11, "team_id": 1}, {"id": 22, "team_id": 2}],
                "match_periods": [{"period": 1, "start_frame": start[0], "end_frame": start[0] + length},
                                  {"period": 2, "start_frame": start[1], "end_frame": start[1] + length}]}
    paths = [folder / "m_match.json", folder / "m_phases_of_play.csv", folder / "m_tracking_extrapolated.jsonl"]
    paths[0].write_text(json.dumps(metadata))
    with paths[1].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["period", "frame_start", "frame_end", "team_in_possession_id"])
        writer.writeheader()
        writer.writerow({"period": 1, "frame_start": start[0] + 5, "frame_end": start[0] + 500, "team_in_possession_id": 1})
    rng = np.random.default_rng(0)
    with paths[2].open("w") as handle:
        for period in (1, 2):
            first = start[period - 1]
            for frame in range(first - 3, first + length + 3):
                i = frame - first
                if i % 97 == 50:
                    continue                                       # a frame the file does not hold
                ball = {"x": float(rng.uniform(-50, 50)), "y": float(rng.uniform(-30, 30)),
                        "z": float(abs(rng.normal(0, 1.5))), "is_detected": True}
                if i % 13 == 0:
                    ball["is_detected"] = False                    # undetected, an extrapolated z
                if i % 211 == 7:
                    ball["z"] = None                               # detected without a height
                if i % 307 == 9:
                    ball["x"] = 70.0                               # outside the pitch plus 5 m
                if i % 401 == 11:
                    ball = None
                players = [{"player_id": 11, "x": 1.0, "y": 2.0, "is_detected": True},
                           {"player_id": 22, "x": -1.0, "y": -2.0, "is_detected": i % 5 != 0}]
                row = {"period": period, "frame": frame, "ball_data": ball, "player_data": players,
                       "possession": {"group": "home team" if i % 2 else None}}
                handle.write(json.dumps(row) + "\n")
    return paths


def test_parse_height_keeps_the_existing_ball_and_adds_z(tmp_path):
    paths = synthetic_match(tmp_path)
    existing = skillcorner.parse_match(paths)
    new = sh.parse_height(paths)
    stored = [{"ball": half["ball"], "labels": half["labels"]} for half in existing]
    check = sh.reader_check(new, existing, stored)
    assert sh.reader_passed(check)
    for half, old in zip(new, existing):
        assert half["ball"].dtype == np.float32 and half["z"].dtype == np.float32
        assert half["z"].shape == (len(old["ball"]),)
        seen = np.isfinite(old["ball"]).all(axis=1)
        assert not np.isfinite(half["z"][~seen]).any()
        assert np.isfinite(half["z"][seen]).mean() > .95             # z is missing only next to a detected-without-z frame
    facts = check["0"]["facts"]
    assert facts["detected_without_z"] > 0 and facts["undetected_with_z"] > 0
    missing = facts["native_detected"] - facts["native_detected_with_z"]
    assert 0 < missing <= facts["detected_without_z"]
    # z is resampled exactly as x is: a 10 Hz z equal to x gives a 25 Hz z equal to x
    native = np.full(24000, np.nan, dtype=np.float32)
    native[::7] = np.arange(0, 24000, 7, dtype=np.float32) / 1000
    x_like, _ = skillcorner.at_25hz(np.c_[native, native])
    z_like, _ = skillcorner.at_25hz(native[:, None])
    assert np.array_equal(z_like[:, 0], x_like[:, 0], equal_nan=True)


def test_reader_check_finds_a_difference(tmp_path):
    paths = synthetic_match(tmp_path)
    existing = skillcorner.parse_match(paths)
    new = sh.parse_height(paths)
    changed = [dict(half, ball=half["ball"].copy()) for half in new]
    changed[1]["ball"][100, 0] += 1e-3
    stored = [{"ball": half["ball"], "labels": half["labels"]} for half in existing]
    check = sh.reader_check(changed, existing, stored)
    assert not sh.reader_passed(check) and check["0"]["ball_xy_equal_existing_reader"]
    assert not check["1"]["ball_xy_equal_existing_reader"] and not check["1"]["ball_xy_equal_stored_half"]
    leaked = [dict(half, z=half["z"].copy()) for half in new]
    unseen = np.flatnonzero(~np.isfinite(leaked[0]["ball"]).all(axis=1))[0]
    leaked[0]["z"][unseen] = 2.0
    assert not sh.reader_passed(sh.reader_check(leaked, existing, stored))


def test_raw_paths_refuse_the_holdout_and_a_changed_file(tmp_path):
    assert refused(lambda: sh.raw_paths(tmp_path, skillcorner.HOLDOUT[0]), ValueError)
    match = skillcorner.TRAIN[0]
    (tmp_path / "raw").mkdir()
    files = {}
    for suffix in ("match.json", "phases_of_play.csv", "tracking_extrapolated.jsonl"):
        path = tmp_path / "raw" / f"{match}_{suffix}"
        path.write_text(suffix)
        files[path.name] = list(skillcorner.identity(path))
    (tmp_path / "manifest.json").write_text(json.dumps({"revision": skillcorner.REVISION, "files": files, "parsed": {}}))
    assert [path.name for path in sh.raw_paths(tmp_path, match)] == list(files)
    (tmp_path / "raw" / f"{match}_match.json").write_text("changed")
    assert refused(lambda: sh.raw_paths(tmp_path, match), ValueError)


def flags_with(n, runs):
    flags = np.zeros(n, dtype=bool)
    for start, stop in runs:
        flags[start:stop] = True
    return flags


def test_height_hide_without_air_is_d56_s_hide():
    n = 6000
    crowd = flags_with(n, [(100, 120), (900, 905), (3000, 3040)])
    for seed in range(5):
        hidden, parts, visibility, facts = sh.height_hide(np.zeros(n, dtype=bool), crowd, seed, 777 + seed)
        expected, d56_parts, d56_visibility = s56.situation_hide(np.zeros(n, dtype=bool), crowd, np.random.default_rng(seed))
        assert np.array_equal(hidden, expected) and visibility == d56_visibility
        assert all(np.array_equal(parts[name], d56_parts[name]) for name in parts)
        assert facts["runs_with_new_delay"] == 0 and facts["runs_with_d56_delay"] == 3


def test_height_hide_adds_the_air_and_keeps_d56_s_draws():
    n = 6000
    crowd = flags_with(n, [(100, 120), (900, 905), (3000, 3040)])
    air = flags_with(n, [(2000, 2050), (3030, 3060), (4000, 4001)])
    hidden, parts, visibility, facts = sh.height_hide(air, crowd, 11, 99)
    d56, d56_parts, d56_visibility = s56.situation_hide(np.zeros(n, dtype=bool), crowd, np.random.default_rng(11))
    assert visibility == d56_visibility and np.array_equal(parts["random"], d56_parts["random"])
    assert hidden[air].all() and hidden[crowd].all()
    assert np.array_equal(hidden, air | crowd | parts["delay"] | parts["random"])
    # the two crowd runs the air leaves alone keep D56's delays exactly
    assert np.array_equal(parts["delay"][:1500], d56_parts["delay"][:1500])
    # the merged run (3000-3060) and the two new air runs take fresh delays
    assert facts["runs"] == 5 and facts["runs_with_d56_delay"] == 2 and facts["runs_with_new_delay"] == 3
    again = sh.height_hide(air, crowd, 11, 99)
    assert np.array_equal(again[0], hidden)
    other = sh.height_hide(air, crowd, 11, 100)
    assert np.array_equal(other[1]["random"], parts["random"])        # only the new delays depend on the air seed
    for start, length in zip(*sv.runs_of(parts["delay"])):
        assert 1 <= length <= 25 and (air | crowd)[start - 1]


def test_height_hide_refuses_if_its_draws_are_not_d56_s():
    crowd = flags_with(3000, [(10, 20)])
    original_hide = s56.situation_hide
    s56.situation_hide = lambda air, crowd, rng: original_hide(air, crowd, np.random.default_rng(12345))
    try:
        assert refused(lambda: sh.height_hide(np.zeros(3000, dtype=bool), crowd, 1, 2), ValueError)
    finally:
        s56.situation_hide = original_hide


def skill_half(n=4000, view="observed", match="1874553", index=0):
    rng = np.random.default_rng(3)
    positions = rng.uniform(-40, 40, (n, 22, 2)).astype(np.float32)
    ball = rng.uniform(-40, 40, (n, 2)).astype(np.float32)
    ball[500:700] = np.nan
    positions[1000:1100, :4] = ball[1000:1100, None, :] + .5          # a crowd
    labels = np.full(n, 0, dtype=np.int8)
    return Half("skillcorner", match, view, index, labels, positions, ball, np.r_[np.zeros(11), np.ones(11)].astype(np.int8),
                np.array([105.0, 68.0]))


def test_skill_view_hides_d56_s_frames_and_the_air():
    half = skill_half()
    z = np.full(len(half.ball), .2, dtype=np.float32)
    z[2000:2040] = 3.0
    z[600:620] = 5.0                                                   # on frames where the ball is unseen (no effect)
    view, visibility, parts, facts = sh.skill_view(half.positions, half.ball, z, "current", 0, 0)
    d56_ball, d56_visibility, d56_parts = s56.skill_view(half.positions, half.ball,
                                                         np.random.default_rng(s56.skill_seed("current", 0, 0)))
    assert visibility == d56_visibility and np.array_equal(parts["random"], d56_parts["random"])
    assert np.isnan(view[2000:2040]).all()
    hidden_d56 = np.isnan(d56_ball).any(axis=1)
    hidden_sz = np.isnan(view).any(axis=1)
    outside = np.ones(len(view), dtype=bool)
    outside[1990:2070] = False                                        # the air run and its delay
    assert np.array_equal(hidden_sz[outside], hidden_d56[outside])


def test_training_view_names_and_refusals():
    half = skill_half()
    z = np.zeros(len(half.ball), dtype=np.float32)
    view, visibility, parts, facts = sh.training_view("augmented", half, skill_height=z, skill_order=4)
    assert view.key == "skillcorner_1874553_situation_0" and view.view == s56.VIEW
    d56_view, d56_visibility, _ = s56.training_view("augmented", half, skill_order=4)
    assert np.array_equal(view.ball, d56_view.ball, equal_nan=True) and visibility == d56_visibility
    assert np.array_equal(view.positions, half.positions)
    assert refused(lambda: sh.training_view("current", half, skill_height=z[:-1], skill_order=0), ValueError)
    assert refused(lambda: sh.training_view("current", half, skill_height=z), ValueError)
    assert refused(lambda: sh.training_view("current", Half("pff", "1", "observed", 0, half.labels, half.positions,
                                                            half.ball, half.teams, half.pitch)), ValueError)


def test_training_view_gives_dfl_halves_d56_s_view():
    rng = np.random.default_rng(5)
    n = 3000
    positions = rng.uniform(-40, 40, (n, 22, 2)).astype(np.float32)
    ball = rng.uniform(-40, 40, (n, 2)).astype(np.float32)
    labels = np.full(n, 0, dtype=np.int8)
    teams = np.r_[np.zeros(11), np.ones(11)].astype(np.int8)
    half = Half("dfl", "WMX", "clean", 1, labels, positions, ball, teams, np.array([105.0, 68.0]))
    z = rng.uniform(0, 3, n)
    view, visibility, parts, facts = sh.training_view("current", half, dfl_height=z, swaps=True)
    expected, expected_visibility, _ = s56.training_view("current", half, height=z, swaps=True)
    assert facts is None and visibility == expected_visibility
    assert np.array_equal(view.ball, expected.ball, equal_nan=True) and np.array_equal(view.positions, expected.positions,
                                                                                       equal_nan=True)
    assert sh.training_view("current", Half("dfl", "WMX", "masked", 1, labels, positions, ball, teams,
                                            np.array([105.0, 68.0])), dfl_height=z) is None


def test_added_by_height_counts_frames_beyond_d56():
    half = skill_half()
    z = np.full(len(half.ball), .2, dtype=np.float32)
    z[2000:2040] = 3.0
    view, visibility, parts, facts = sh.skill_view(half.positions, half.ball, z, "current", 0, 0)
    d56_ball, _, _ = s56.skill_view(half.positions, half.ball, np.random.default_rng(s56.skill_seed("current", 0, 0)))
    d56_hidden = np.isnan(d56_ball).any(axis=1)
    result = sh.added_by_height([(view, d56_hidden, parts, half.ball, half.labels)])
    assert result["frames_hidden_beyond_d56_view"] == int((np.isnan(view).any(axis=1) & ~d56_hidden).sum()) > 0
    seen = np.isfinite(half.ball).all(axis=1)
    assert result["seen_share_above_air_height"] == (seen & parts["air"]).sum() / seen.sum()


def test_air_view_hides_only_frames_above_the_air_height():
    half = skill_half()
    z = np.full(len(half.ball), np.nan, dtype=np.float32)
    seen = np.isfinite(half.ball).all(axis=1)
    z[seen] = .3
    z[3000:3010] = 1.6
    z[3010:3020] = 1.5                                                 # not above 1.5 m
    view, air = sh.air_view(half, z)
    assert view.view == sh.AIR_VIEW and np.array_equal(view.positions, half.positions)
    assert np.isnan(view.ball[3000:3010]).all() and not np.isnan(view.ball[3010:3020]).any()
    assert np.array_equal(np.isnan(view.ball).any(axis=1), np.isnan(half.ball).any(axis=1) | air)
    assert int(air.sum()) == 10
    assert refused(lambda: sh.air_view(half, z[:-1]), ValueError)
    stats = sh.air_view_statistics([(half, view, air)])
    assert stats["share_hidden_by_height"] == 10 / len(z)
    assert stats["view"]["hidden_frames"] == stats["observed"]["hidden_frames"] + 10


def test_heights_of_checks_digests(tmp_path):
    folder = tmp_path / "heights"
    folder.mkdir()
    path = folder / "1874553.npz"
    np.savez_compressed(path, z_0=np.arange(5, dtype=np.float32), z_1=np.arange(3, dtype=np.float32))
    record = tmp_path / "reader.json"
    record.write_text(json.dumps({"passed": True, "heights": {"1874553": digest(path)}}))
    heights = sh.heights_of(record, folder)
    assert np.array_equal(heights["1874553", 1], np.arange(3, dtype=np.float32))
    record.write_text(json.dumps({"passed": False, "heights": {"1874553": digest(path)}}))
    assert refused(lambda: sh.heights_of(record, folder), SystemExit)
    record.write_text(json.dumps({"passed": True, "heights": {"1874553": "0" * 64}}))
    assert refused(lambda: sh.heights_of(record, folder), ValueError)


def test_write_new_once_and_seeds(tmp_path):
    sh.write_new(tmp_path / "r.json", {"a": 1})
    assert refused(lambda: sh.write_new(tmp_path / "r.json", {"a": 2}), FileExistsError)
    seeds = {sh.air_seed(model_set, order, index) for model_set in ("current", "augmented") for order in range(12)
             for index in (0, 1)}
    assert len(seeds) == 48
    d56 = {s56.skill_seed(model_set, order, index) for model_set in ("current", "augmented") for order in range(12)
           for index in (0, 1)}
    assert not seeds & d56
    assert sh.MATCHES == (*skillcorner.TRAIN, *skillcorner.DEVELOPMENT) and not set(sh.MATCHES) & set(skillcorner.HOLDOUT)
