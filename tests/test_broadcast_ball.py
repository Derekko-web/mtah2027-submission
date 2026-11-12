"""D31 on synthetic data: broadcast stretches, the offset search and the ball merge."""

import numpy as np

from dfl_submission import broadcast_ball as bb


TEAMS = np.array([0] * 11 + [1] * 11)
LOOSE = {"max_cost": 3.0, "min_margin": .5}


def match(frames=6000, seed=0):
    """A synthetic tactical half: 22 players on smooth, non-repeating paths (metres, centred)."""
    rng = np.random.default_rng(seed)
    time = np.arange(frames)[:, None, None]
    positions = rng.uniform([-40, -25], [40, 25], (22, 2))[None].repeat(frames, axis=0)
    for _ in range(3):
        amplitude = rng.uniform(2, 10, (22, 2))
        period = rng.uniform(200, 2000, (22, 2))
        phase = rng.uniform(0, 2 * np.pi, (22, 2))
        positions = positions + amplitude * np.sin(2 * np.pi * time / period + phase)
    return np.clip(positions, [-52, -34], [52, 34]).astype(np.float32)


def broadcast_view(tactical, frames, orientation, *, noise=.2, seed=1, window=35.):
    """Broadcast players for tactical `frames`: oriented, noisy, only those within `window` m of the centre in x."""
    rng = np.random.default_rng(seed)
    shown = tactical[frames] + rng.normal(0, noise, tactical[frames].shape).astype(np.float32)
    centre = np.median(shown[:, :, 0], axis=1, keepdims=True)
    shown[np.abs(shown[:, :, 0] - centre) > window] = np.nan
    return bb.orient(shown, orientation)


def test_orient_flips_x_and_y_and_is_its_own_inverse():
    points = np.array([[1., 2.], [-3., 4.]])
    assert np.array_equal(bb.orient(points, 1), [[-1., 2.], [3., 4.]])
    assert np.array_equal(bb.orient(points, 2), [[1., -2.], [-3., -4.]])
    assert np.array_equal(bb.orient(points, 3), -points)
    for orientation in range(4):
        assert np.array_equal(bb.orient(bb.orient(points, orientation), orientation), points)


def test_usable_needs_valid_homography_and_four_players_per_team():
    positions = np.zeros((3, 22, 2), dtype=np.float32)
    positions[1, 3:11] = np.nan       # team 0 keeps 3 players
    valid = np.array([True, True, False])
    assert list(bb.usable(positions, TEAMS, valid)) == [True, False, False]


def test_stretches_are_runs_of_usable_frames_of_at_least_the_minimum():
    usable = np.zeros(1000, dtype=bool)
    usable[10:200] = True
    usable[300:400] = True
    usable[500:625] = True
    assert bb.stretches(usable) == [(10, 200), (500, 625)]
    assert bb.MIN_STRETCH == 125


def test_frame_costs_are_zero_on_the_same_frame_and_infinite_without_four_players():
    tactical = match(100)
    costs = bb.frame_costs(tactical[40], TEAMS, tactical, TEAMS)
    assert costs[40] == 0 and (costs[np.arange(100) != 40] > 0).all()
    few = tactical[40].copy()
    few[3:11] = np.nan
    assert np.isinf(bb.frame_costs(few, TEAMS, tactical, TEAMS)).all()


def test_frame_costs_cap_each_player_at_ten_metres():
    tactical = np.zeros((1, 22, 2), dtype=np.float32)
    far = np.full((22, 2), 100., dtype=np.float32)
    assert bb.frame_costs(far, TEAMS, tactical, TEAMS)[0] == 10.


def test_align_recovers_a_known_offset_and_orientation():
    tactical = match()
    start, stop, offset = 1000, 1600, 1234
    broadcast = np.full((3000, 22, 2), np.nan, dtype=np.float32)
    broadcast[start:stop] = broadcast_view(tactical, np.arange(start, stop) + offset, 3)
    result = bb.align_stretch(tactical, TEAMS, broadcast, TEAMS, start, stop, LOOSE)
    assert result["offset"] == offset and result["orientation"] == 3
    assert result["accepted"] and not result["tie"] and result["margin"] > 1


def test_align_rejects_a_slow_replay():
    tactical = match(seed=3)
    start, stop = 1000, 1400
    shown = 2500 + (np.arange(stop - start) // 2)
    broadcast = np.full((3000, 22, 2), np.nan, dtype=np.float32)
    broadcast[start:stop] = broadcast_view(tactical, shown, 0)
    assert not bb.align_stretch(tactical, TEAMS, broadcast, TEAMS, start, stop, {"max_cost": 1.5, "min_margin": .5})["accepted"]


def test_align_rejects_ties():
    tactical = np.repeat(match(1)[:1], 3000, axis=0)   # nothing moves: every offset costs the same
    broadcast = np.full((3000, 22, 2), np.nan, dtype=np.float32)
    broadcast[1000:1200] = tactical[1100:1300]
    result = bb.align_stretch(tactical, TEAMS, broadcast, TEAMS, 1000, 1200, LOOSE)
    assert result["tie"] and not result["accepted"]


def test_align_runs_every_long_enough_stretch():
    tactical = match(seed=4)
    broadcast = np.full((3000, 22, 2), np.nan, dtype=np.float32)
    broadcast[100:400] = broadcast_view(tactical, np.arange(100, 400) + 2000, 0)
    broadcast[600:650] = broadcast_view(tactical, np.arange(600, 650), 0)       # too short
    broadcast[900:1300] = broadcast_view(tactical, np.arange(900, 1300) + 3100, 1)
    valid = np.ones(3000, dtype=bool)
    results = bb.align(tactical, TEAMS, broadcast, TEAMS, valid, LOOSE)
    assert [(r["start"], r["stop"], r["offset"], r["orientation"]) for r in results] == [
        (100, 400, 2000, 0), (900, 1300, 3100, 1)]


def stretch(start, stop, offset, orientation=0, accepted=True):
    return {"start": start, "stop": stop, "offset": offset, "orientation": orientation, "accepted": accepted}


def test_merge_fills_only_missing_tactical_frames_in_orientation():
    tactical = np.full((20, 2), np.nan)
    tactical[5] = [9., 9.]
    broadcast = np.arange(40, dtype=float).reshape(20, 2)
    merged, filled = bb.merge(tactical, broadcast, [stretch(2, 8, 2, orientation=1)])
    assert np.array_equal(merged[5], [9., 9.]) and not filled[5]
    assert np.array_equal(merged[4], [-broadcast[2, 0], broadcast[2, 1]]) and filled[4]
    assert np.flatnonzero(filled).tolist() == [4, 6, 7, 8, 9]
    assert np.isnan(tactical[4]).all()


def test_merge_skips_unseen_broadcast_balls_rejected_stretches_and_frames_off_the_timeline():
    tactical = np.full((10, 2), np.nan)
    broadcast = np.ones((10, 2))
    broadcast[3] = np.nan
    merged, filled = bb.merge(tactical, broadcast, [stretch(2, 5, 6), stretch(0, 2, 0, accepted=False)])
    assert np.flatnonzero(filled).tolist() == [8]     # b=2 -> 8; b=3 unseen; b=4 -> 10 is off the timeline


def test_merge_earlier_stretch_wins_on_a_shared_frame():
    tactical = np.full((10, 2), np.nan)
    broadcast = np.arange(20, dtype=float).reshape(10, 2)
    merged, _ = bb.merge(tactical, broadcast, [stretch(6, 8, -3), stretch(2, 4, 1)])
    assert np.array_equal(merged[3], broadcast[2])    # b=2 (stretch starting at 2) beats b=6
    assert np.array_equal(merged[4], broadcast[3])


def test_merge_uses_a_given_missing_mask():
    tactical = np.zeros((6, 2))
    broadcast = np.ones((6, 2))
    missing = np.array([False, True, False, True, False, False])
    merged, filled = bb.merge(tactical, broadcast, [stretch(0, 6, 0)], missing=missing)
    assert np.flatnonzero(filled).tolist() == [1, 3] and merged[1, 0] == 1 and merged[0, 0] == 0


def test_packing_tactical_slots_changes_no_cost():
    tactical = match(400, seed=5).astype(np.float64)
    teams = np.r_[TEAMS, [0, 1, 0, -1]]
    extra = np.full((400, 4, 2), np.nan)
    extra[100:200, 0] = 3.0                   # a substitute seen for a while
    tactical = np.concatenate([tactical, extra], axis=1)
    tactical[50:80, 2] = np.nan               # a player unseen for a while
    packed, packed_teams = bb.pack(tactical, teams)
    assert packed.shape[1] < tactical.shape[1]
    frame = tactical[150, :22] + .4
    np.testing.assert_array_equal(bb.frame_costs(frame, TEAMS, tactical, teams),
                                  bb.frame_costs(frame, TEAMS, packed, packed_teams))


# -- D31a: trimmed cost and a translation per stretch ----------------------------------------------

def test_trimmed_cost_drops_the_worst_quarter_of_each_teams_players():
    tactical = match(10, seed=7)
    frame = tactical[4].copy()
    frame[[0, 1, 11, 12]] += 30.0                      # two players per team without a counterpart nearby
    assert bb.frame_costs(frame, TEAMS, tactical[4:5], TEAMS)[0] > 1
    assert bb.frame_costs(frame, TEAMS, tactical[4:5], TEAMS, trim=.25)[0] == 0
    assert bb.TRIM == .25


def test_a_shift_moves_the_broadcast_players_before_matching():
    tactical = match(10, seed=8)
    frame = tactical[2] + np.array([1.0, -.5], dtype=np.float32)
    assert bb.frame_costs(frame, TEAMS, tactical[2:3], TEAMS)[0] > .5
    assert bb.frame_costs(frame, TEAMS, tactical[2:3], TEAMS, shift=np.array([[1.0, -.5]]))[0] < 1e-6


def test_translated_search_recovers_offset_and_bias_despite_missing_tracks():
    tactical = match(seed=9)
    masked = tactical.copy()
    rng = np.random.default_rng(2)
    for player in range(22):                           # drop tracks in 50-frame blocks, as the masked view does
        for block in range(0, len(masked), 50):
            if rng.random() < .2:
                masked[block:block + 50, player] = np.nan
    start, stop, offset, bias = 1000, 1500, 777, np.array([1.5, -1.0], dtype=np.float32)
    broadcast = np.full((3000, 22, 2), np.nan, dtype=np.float32)
    frames = np.arange(start, stop) + offset
    broadcast[start:stop] = bb.orient(broadcast_view(tactical, frames, 0, noise=.3) + bias, 1)
    result = bb.align_stretch(masked, TEAMS, broadcast, TEAMS, start, stop, LOOSE, trim=.25, translate=True)
    untranslated = bb.align_stretch(masked, TEAMS, broadcast, TEAMS, start, stop, LOOSE, trim=.25)
    assert result["offset"] == offset and result["orientation"] == 1 and result["accepted"]
    assert np.allclose(result["shift"], bias, atol=.2) and result["cost"] < untranslated["cost"]


def test_untranslated_search_reports_no_shift():
    tactical = match()
    broadcast = np.full((3000, 22, 2), np.nan, dtype=np.float32)
    broadcast[1000:1300] = broadcast_view(tactical, np.arange(1000, 1300) + 50, 0)
    assert bb.align_stretch(tactical, TEAMS, broadcast, TEAMS, 1000, 1300, LOOSE)["shift"] == [0.0, 0.0]


def test_merge_moves_the_broadcast_ball_by_the_stretch_shift():
    tactical = np.full((10, 2), np.nan)
    broadcast = np.arange(20, dtype=float).reshape(10, 2)
    result = dict(stretch(0, 10, 0, orientation=2), shift=[1.0, 2.0])
    merged, _ = bb.merge(tactical, broadcast, [result])
    assert np.allclose(merged[3], [broadcast[3, 0] - 1.0, -broadcast[3, 1] - 2.0])
