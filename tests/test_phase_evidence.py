"""D62's simulated phases, tag state, mask, Carry replacement, shift and agreement on synthetic arrays."""

import numpy as np
from scipy.special import expit, logit

from dfl_submission import phase_evidence as pe, tree_share_trial as tt
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X
FPS = pe.FPS


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


class Fixed:
    """A generator stand-in: `random` gives the queued vectors in turn, `uniform` the queued values in turn."""

    def __init__(self, randoms, uniforms):
        self.randoms, self.uniforms = list(randoms), list(uniforms)

    def random(self, count):
        return np.asarray(self.randoms.pop(0), dtype=np.float64)[:count]

    def uniform(self, low, high, count):
        values = np.asarray(self.uniforms.pop(0), dtype=np.float64)[:count]
        assert np.all(values >= low) and np.all(values < high)
        return values


def alternating(count, seconds=10, gap=1):
    """count possessions of `seconds`, alternating H and A, each followed by `gap` s of D."""
    parts = []
    for i in range(count):
        parts += [(H if i % 2 == 0 else A, seconds * FPS), (D, gap * FPS)]
    return labels(*parts)


def test_profiles_and_the_seed_formula_are_the_rule_s():
    assert pe.PROFILE_ORDER == ("Good", "Moderate", "Poor", "Carry")
    assert pe.PROFILES["Good"] == pe.Profile(.8, 1, 1, .05)
    assert pe.PROFILES["Moderate"] == pe.Profile(.6, 2, 3, .15)
    assert pe.PROFILES["Poor"] == pe.Profile(.6, 3, 5, .30)
    assert pe.PROFILES["Carry"] == pe.Profile(.6, 2, 3, .15, carry=True)
    assert pe.WEIGHTS == (.5, 1., 2.) and pe.DRAWS == 3
    assert pe.seed(0, 0, 0) == 620000 and pe.seed(2, 3, 1) == 622031 and pe.seed(67, 1, 2) == 687012
    assert refused(lambda: pe.seed(0, 10, 0), ValueError) and refused(lambda: pe.seed(0, 0, 10), ValueError)


def test_simulation_is_reproducible_from_its_seed():
    target = alternating(60)
    for name in pe.PROFILE_ORDER:
        first = pe.simulated_phases(target, pe.PROFILES[name], np.random.default_rng(pe.seed(3, 1, 2)))
        again = pe.simulated_phases(target, pe.PROFILES[name], np.random.default_rng(pe.seed(3, 1, 2)))
        other = pe.simulated_phases(target, pe.PROFILES[name], np.random.default_rng(pe.seed(3, 1, 1)))
        assert first == again and first != other and first


def test_only_possessions_of_at_least_5_s_are_candidates():
    target = labels((H, 5 * FPS - 1), (D, 50), (A, 5 * FPS), (D, 50), (H, 300), (X, 10), (A, 200))
    spells = pe.possessions(target)
    assert [(a, b, k) for a, b, k in spells] == [(0, 124, H), (174, 299, A), (349, 649, H), (659, 859, A)]
    rng = Fixed([[0, 0, 0], [.99, .99, .99]], [[0, 0, 0], [0, 0, 0]])     # all kept, no delay, no offset, right team
    phases = pe.simulated_phases(target, pe.Profile(.5, 1, 1, .5), rng)
    assert phases == [(174, 299, pe.AWAY), (349, 649, pe.HOME), (659, 859, pe.AWAY)]


def test_delays_and_offsets_stay_within_the_profile_s_limits():
    target = alternating(400, seconds=12, gap=8)
    truth = [(a, b, k) for a, b, k in pe.possessions(target)]
    for name in ("Good", "Moderate", "Poor"):
        profile = pe.PROFILES[name]
        everything = pe.Profile(1., profile.a, profile.b, 0.)
        phases = pe.simulated_phases(target, everything, np.random.default_rng(pe.seed(0, 0, 0)))
        assert len(phases) == len(truth)
        delays = np.array([p[0] - t[0] for p, t in zip(phases, truth)])
        offsets = np.array([p[1] - t[1] for p, t in zip(phases, truth)])
        assert delays.min() >= 0 and delays.max() <= profile.a * FPS
        assert offsets.min() >= -FPS and offsets.max() <= profile.b * FPS
        assert delays.max() > .8 * profile.a * FPS and offsets.min() < -.8 * FPS and offsets.max() > .8 * profile.b * FPS
        assert all(p[2] == (pe.HOME if t[2] == H else pe.AWAY) for p, t in zip(phases, truth))


def test_rounding_to_the_nearest_frame():
    target = labels((D, 100), (H, 300), (D, 100))
    rng = Fixed([[0], [.99]], [[.5 / FPS + .01 / FPS], [-.4 / FPS]])     # +0.51 frame, -0.4 frame
    assert pe.simulated_phases(target, pe.Profile(1, 1, 1, .1), rng) == [(101, 400, pe.HOME)]


def test_the_keep_and_wrong_team_rates():
    target = alternating(4000, seconds=6, gap=1)
    truth = pe.possessions(target)
    profile = pe.PROFILES["Poor"]
    phases = pe.simulated_phases(target, profile, np.random.default_rng(pe.seed(5, 2, 0)))
    starts = np.array([a for a, _, _ in truth])
    codes = np.array([k for _, _, k in truth])
    owner = codes[np.searchsorted(starts, [start for start, _, _ in phases], side="right") - 1]
    wrong = [team != (pe.HOME if code == H else pe.AWAY) for (_, _, team), code in zip(phases, owner)]
    # 6 s possessions: a phase keeps at least 150 - 75 - 25 = 50 frames under Poor, so none is dropped
    assert abs(len(phases) / len(truth) - profile.c) < .03
    assert abs(np.mean(wrong) - profile.q) < .03


def test_phases_are_clipped_to_the_half_and_dropped_under_2_s():
    target = labels((H, 200), (D, 10), (A, 130), (D, 10), (H, 300))
    rng = Fixed([[0, 0, 0], [.9, .9, .9]], [[0, 3.5, 0], [-1, 0, 4.9]])
    phases = pe.simulated_phases(target, pe.Profile(1, 4, 5, .1), rng)
    # first: end 200 - 25; second: start 210 + 87.5 -> 298 (rint), end 340, 42 frames < 50: dropped;
    # third: end 650 + 122.5 -> clipped to the half's 650 frames
    assert phases == [(0, 175, pe.HOME), (350, 650, pe.HOME)]
    exactly = labels((D, 10), (H, 150), (D, 10))
    rng = Fixed([[0], [.9]], [[3.], [-1.]])                  # starts 75 frames late, ends 25 early: exactly 50 frames
    assert pe.simulated_phases(exactly, pe.Profile(1, 5, 1, .1), rng) == [(85, 135, pe.HOME)]
    rng = Fixed([[0], [.9]], [[3.04], [-1.]])                # 49 frames: dropped
    assert pe.simulated_phases(exactly, pe.Profile(1, 5, 1, .1), rng) == []


def test_tag_state_from_intervals():
    state = pe.tag_state(20, [(2, 6, 1), (4, 8, 1), (10, 14, 1), (12, 16, -1), (17, 30, -1), (-5, 1, -1), (9, 9, 1)])
    assert state.dtype == np.int8
    assert state.tolist() == [-1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, -1, -1, 0, -1, -1, -1]
    assert pe.tag_state(5, []).tolist() == [0] * 5
    assert refused(lambda: pe.tag_state(5, [(0, 2, "Barcelona")]), ValueError)
    assert refused(lambda: pe.tag_state(5, [(0, 2, 0)]), ValueError)


def test_the_mask_is_d59_s_ht_definition():
    seen = np.zeros(300, dtype=bool)
    seen[[0, 200]] = True
    mask = pe.far_mask(seen)
    assert np.flatnonzero(mask).tolist() == list(range(76, 125)) + list(range(276, 300))
    assert np.array_equal(mask, tt.far_from_sighting(seen))
    assert pe.far_mask(np.zeros(10, dtype=bool)).all()
    assert not pe.far_mask(np.ones(10, dtype=bool)).any()
    rng = np.random.default_rng(0)
    for _ in range(20):
        seen = rng.random(2000) < .02
        assert np.array_equal(pe.far_mask(seen), tt.far_from_sighting(seen, 75))


def test_the_shift_touches_only_masked_tagged_frames_and_is_exact():
    rng = np.random.default_rng(1)
    home = rng.random(500).astype(np.float32)
    home[10], home[11] = 0., 1.
    state = rng.choice(np.array([-1, 0, 1], dtype=np.int8), 500)
    state[10], state[11] = 1, -1
    mask = rng.random(500) < .5
    mask[10] = mask[11] = True
    for weight in pe.WEIGHTS:
        shifted = pe.shift_home(home, state, mask, weight)
        assert shifted.dtype == np.float32 and shifted is not home
        where = mask & (state != 0)
        assert np.array_equal(shifted[~where], home[~where])
        expected = expit(logit(np.clip(home[where].astype(np.float64), .001, .999)) + weight * state[where])
        assert np.array_equal(shifted[where], expected.astype(np.float32))
        assert np.all(np.sign(logit(np.clip(shifted[where].astype(np.float64), 1e-9, 1 - 1e-9))
                              - logit(np.clip(home[where].astype(np.float64), .001, .999))) == state[where])
    assert np.isclose(pe.shift_home(home, state, mask, 1.)[10], expit(logit(.001) + 1), rtol=1e-6)
    assert np.isclose(pe.shift_home(home, state, mask, 1.)[11], expit(logit(.999) - 1), rtol=1e-6)
    signals = (home, np.ones(500, dtype=np.float32), np.zeros(500), np.zeros(500))
    shifted = pe.shifted_signals(signals, state, mask, 2.)
    assert all(a is b for a, b in zip(shifted[1:], signals[1:]))


def test_w0_and_an_all_zero_state_are_the_identity():
    rng = np.random.default_rng(2)
    home = rng.random(300).astype(np.float32)
    home[0] = 1.                                  # clipping would change this frame if it were touched
    state = rng.choice(np.array([-1, 1], dtype=np.int8), 300)
    mask = np.ones(300, dtype=bool)
    same = pe.shift_home(home, state, mask, 0.)
    assert same.dtype == home.dtype and np.array_equal(same, home) and same is not home
    for weight in pe.WEIGHTS:
        assert np.array_equal(pe.shift_home(home, np.zeros(300, dtype=np.int8), mask, weight), home)
        assert np.array_equal(pe.shift_home(home, state, np.zeros(300, dtype=bool), weight), home)


def test_the_carried_team_is_the_target_s_at_the_last_sighting():
    target = labels((H, 10), (A, 10), (D, 10), (A, 10))
    seen = np.zeros(40, dtype=bool)
    seen[[3, 15, 25]] = True
    carried = pe.carried_state(target, seen)
    assert carried[:4].tolist() == [0, 0, 0, 0]              # no sighting before (frame 3 itself counts from frame 4)
    assert np.all(carried[4:16] == pe.HOME)                  # last sighting at 3 (H)
    assert np.all(carried[16:26] == pe.AWAY)                 # at 15 (A)
    assert np.all(carried[26:] == 0)                         # at 25 (D)
    x_target = target.copy()
    x_target[25] = X
    assert np.all(pe.carried_state(x_target, seen)[26:] == 0)
    assert pe.carried_state(np.zeros(0, dtype=np.int8), np.zeros(0, dtype=bool)).tolist() == []


def test_carry_replaces_the_state_only_on_masked_frames():
    target = alternating(30, seconds=10, gap=2)
    seen = np.ones(len(target), dtype=bool)
    seen[400:900] = False                                    # a 20 s hidden run, masked from 475 to 824
    seen[2000:2100] = False                                  # 4 s: no frame is more than 75 frames from a sighting
    moderate, carry = pe.PROFILES["Moderate"], pe.PROFILES["Carry"]
    plain, mask, phases = pe.half_state(target, seen, moderate, np.random.default_rng(7))
    carried, mask2, phases2 = pe.half_state(target, seen, carry, np.random.default_rng(7))
    assert phases == phases2 and np.array_equal(mask, mask2)
    assert np.flatnonzero(mask).tolist() == list(range(475, 825))
    assert np.array_equal(carried[~mask], plain[~mask])
    team = pe.HOME if target[399] == H else (pe.AWAY if target[399] == A else 0)
    assert np.all(carried[mask] == team)
    dead_at_sighting = target.copy()
    dead_at_sighting[399] = D
    carried, _, _ = pe.half_state(dead_at_sighting, seen, carry, np.random.default_rng(7))
    assert np.all(carried[mask] == 0)


def test_agreement_counts_seen_live_tagged_frames():
    state = np.array([1, 1, -1, -1, 0, 1, -1, 1], dtype=np.int8)
    decoded = np.array([H, A, A, H, H, D, A, H], dtype=np.int8)
    seen = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    assert pe.agreement(state, decoded, seen) == (3, 5)        # frames 0, 2, 6 agree of 0, 1, 2, 3, 6
    assert pe.agreement(np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8), np.ones(3, dtype=bool)) == (0, 0)
    acting = pe.acting(state, seen)
    assert acting.tolist() == [True, True, True, True, False, True, True, False]
