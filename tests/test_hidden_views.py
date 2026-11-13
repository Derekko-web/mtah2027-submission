"""D49's long hide process, training views, guard views and trial helpers on synthetic fixtures."""

import numpy as np

from dfl_submission import augment, hidden_trial as ht49, hidden_views as hv
from dfl_submission.external_benchmark import Half
from dfl_transfer_features import corrupt
from evaluation import possession_metrics as pm


def runs(flags, value):
    padded = np.r_[not value, flags, not value]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    lengths = edges[1::2] - edges[::2]
    starts = edges[::2]
    return [int(length) for length, start in zip(lengths, starts) if flags[start] == value]


def test_long_hide_has_the_rule_s_run_lengths_and_expected_visibility():
    rng = np.random.default_rng(1)
    visible = hv.long_hide(3_000_000, .6, rng)
    assert abs(visible.mean() - .6) < .02
    hidden = np.array(runs(visible, False)[:-1])
    shown = np.array(runs(visible, True)[:-1])
    assert abs(hidden.mean() - hv.MEAN_HIDDEN) / hv.MEAN_HIDDEN < .05
    assert abs(shown.mean() - hv.MEAN_HIDDEN * .6 / .4) / (hv.MEAN_HIDDEN * 1.5) < .06
    # Half the hidden runs are uniform on 125-750 frames; a geometric run with mean 50 reaches 125 with
    # probability 0.98 ** 124, about 0.08.
    long = np.mean((hidden >= 125) & (hidden <= 750))
    assert abs(long - (.5 + .5 * .98 ** 124)) < .03
    assert np.mean(hidden > 750) < .001
    assert hidden.min() >= 1


def test_long_hide_visibility_follows_v_across_the_range():
    for visibility in (.3, .5, .9):
        rng = np.random.default_rng(int(visibility * 100))
        shares = [hv.long_hide(80_000, visibility, rng).mean() for _ in range(60)]
        assert abs(np.mean(shares) - visibility) < .03, (visibility, np.mean(shares))
    try:
        hv.long_hide(10, 1.0, np.random.default_rng(0))
    except ValueError:
        pass
    else:
        raise AssertionError("A visibility of 1 was accepted")


def test_guard_hide_alternates_uniform_runs_starting_visible():
    visible = hv.guard_hide(200_000, np.random.default_rng(3))
    assert visible[0]
    shown, hidden = runs(visible, True), runs(visible, False)
    complete_shown = shown[:-1] if visible[-1] else shown
    complete_hidden = hidden if visible[-1] else hidden[:-1]
    assert min(complete_shown) >= 300 and max(complete_shown) <= 750
    assert min(complete_hidden) >= 200 and max(complete_hidden) <= 500
    assert abs(visible.mean() - 525 / 875) < .03


def tracking(n=6000, players=22, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-50, 50, (n, players, 2)).astype(np.float32)
    ball = rng.uniform(-50, 50, (n, 2)).astype(np.float32)
    return positions, ball


def test_noise_and_dropout_is_corrupt_s_and_wide_masking_s_first_part():
    positions, ball = tracking()
    noisy_positions, noisy_ball = hv.noise_and_dropout(positions, ball, np.random.default_rng(270007))
    masked_positions, masked_ball = corrupt(positions, ball, 270007)
    assert np.array_equal(noisy_positions, masked_positions, equal_nan=True)
    seen = np.isfinite(masked_ball).all(axis=1)
    assert np.array_equal(noisy_ball[seen], masked_ball[seen])
    wide_positions, _ = augment.wide_masking(positions, ball, np.random.default_rng(5))
    again, _ = hv.noise_and_dropout(positions, ball, np.random.default_rng(5))
    assert np.array_equal(wide_positions, again, equal_nan=True)


def test_guard_view_is_seeded_and_leaves_its_input_alone():
    positions, ball = tracking()
    before = positions.copy(), ball.copy()
    first = hv.guard_view(positions, ball, hv.guard_dfl_seed("WPY", 0))
    second = hv.guard_view(positions, ball, hv.guard_dfl_seed("WPY", 0))
    other = hv.guard_view(positions, ball, hv.guard_dfl_seed("WPY", 1))
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(first, second))
    assert not np.array_equal(first[1], other[1], equal_nan=True)
    assert np.array_equal(positions, before[0]) and np.array_equal(ball, before[1])
    assert np.isfinite(first[1][:300]).all()


def half(source, view, match="WMX", index=0, n=6000):
    positions, ball = tracking(n)
    labels = np.where(np.arange(n) % 1000 < 900, pm.H, pm.D).astype(np.int8)
    teams = np.array([0] * 11 + [1] * 11)
    return Half(source, match, view, index, labels, positions, ball, teams, np.array([105.0, 68.0]))


def test_training_view_builds_one_view_per_clean_dfl_half_and_leaves_the_half_untouched():
    from dfl_submission.train import TRAIN
    match = TRAIN[0]
    clean = half("dfl", "clean", match)
    before = clean.positions.copy(), clean.ball.copy()
    for view in ("masked", "stress"):
        assert hv.training_view("current", half("dfl", view, match)) is None
    made, visibility = hv.training_view("current", clean)
    again, _ = hv.training_view("current", clean)
    assert made.view == hv.VIEW and made.key == f"dfl_{match}_{hv.VIEW}_0"
    assert hv.DFL_VISIBILITY[0] < visibility < hv.DFL_VISIBILITY[1]
    assert np.array_equal(made.ball, again.ball, equal_nan=True)
    assert np.array_equal(made.labels, clean.labels) and made.teams is clean.teams
    assert np.array_equal(clean.positions, before[0]) and np.array_equal(clean.ball, before[1])
    augmented, _ = hv.training_view("augmented", clean, swaps=True)
    unswapped, _ = hv.training_view("augmented", clean, swaps=False)
    assert not np.array_equal(augmented.ball, made.ball, equal_nan=True)  # another seed
    assert np.array_equal(augmented.ball, unswapped.ball, equal_nan=True)  # swaps move players only
    assert not np.array_equal(augmented.positions, unswapped.positions, equal_nan=True)


def test_skillcorner_view_only_hides_more_of_the_detected_ball():
    observed = half("skillcorner", "observed", "2017", 1)
    observed.ball[100:400] = np.nan
    made, visibility = hv.training_view("current", observed, skill_order=3)
    assert hv.SKILL_VISIBILITY[0] < visibility < hv.SKILL_VISIBILITY[1]
    assert made.positions is observed.positions and made.view == hv.VIEW
    detected = np.isfinite(observed.ball).all(axis=1)
    kept = np.isfinite(made.ball).all(axis=1)
    assert not (kept & ~detected).any() and (detected & ~kept).any()
    assert np.array_equal(made.ball[kept], observed.ball[kept])
    provider = np.zeros_like(observed.ball)
    from_provider, _ = hv.training_view("augmented", observed, provider_ball=provider, skill_order=3)
    assert np.all(from_provider.ball[np.isfinite(from_provider.ball).all(axis=1)] == 0)
    try:
        hv.training_view("current", observed)
    except ValueError:
        pass
    else:
        raise AssertionError("A SkillCorner half without its order was accepted")


def test_seeds_are_new_and_do_not_overlap():
    existing = (270000, 370000, 520000, 530000, 610000, 640000, 650000, 660000)
    spans = {name: (base, base + 10_000) for name, base in hv.SEEDS.items()}
    for name, (low, high) in spans.items():
        assert all(not low <= seed < high for seed in existing), name
    ordered = sorted(spans.values())
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))
    assert hv.dfl_seed("current", "WPY", 1) != hv.dfl_seed("augmented", "WPY", 1)


def test_view_statistics_measure_visibility_and_long_runs():
    ball = np.zeros((1000, 2), dtype=np.float32)
    ball[100:400] = np.nan       # 300 frames: over 10 s
    ball[500:510] = np.nan
    stats = hv.view_statistics([ball])
    assert stats["visibility"] == 1 - 310 / 1000
    assert stats["hidden_share_in_runs_over_10s"] == 300 / 310
    assert stats["hidden_runs"] == 2


# -- the trial's helpers ---------------------------------------------------------------------------

def test_training_processes_counts_seeded_trainers_only():
    listing = "\n".join([
        "/x/Python -m dfl_submission.player_set3 --source a --tokens b --out c --seed 42 --boundary-boost 0.5",
        "/x/Python -m dfl_submission.player_set train --source a --tokens b --out c --seed 97 --boundary-boost 1.4",
        "/x/Python -m dfl_submission.player_set export --source c --out d",
        "/x/Python -m dfl_submission.screen_trial networks --candidate S1",
        "grep dfl_submission --seed",
    ])
    assert ht49.training_processes(listing) == 2


def test_wait_for_slot_waits_until_fewer_than_two_run():
    counts = iter([3, 2, 1])
    ht49.wait_for_slot(pause=0, count=lambda: next(counts))
    try:
        next(counts)
    except StopIteration:
        pass
    else:
        raise AssertionError("wait_for_slot stopped too early")


def test_same_bytes_is_exact():
    values = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    assert ht49.same_bytes(values, values.copy())
    assert not ht49.same_bytes(values, values.astype(np.float64))
    other = values.copy()
    other[2] = np.nextafter(np.float32(2.0), np.float32(3.0))
    assert not ht49.same_bytes(values, other)


def test_merged_manifest_keeps_the_stored_records_first_and_unchanged():
    stored = {"columns": ["a"], "augmented": True, "records": [{"file": "dfl_X_clean_0.npz", "sha256": "1"}]}
    merged = ht49.merged_manifest(stored, [{"file": "dfl_X_long_hidden_0.npz", "sha256": "2"}], {"view": hv.VIEW})
    assert merged["records"][0] == stored["records"][0] and merged["records"][1]["file"] == "dfl_X_long_hidden_0.npz"
    assert merged["columns"] == ["a"] and merged["augmented"] is True and merged["d49"] == {"view": hv.VIEW}
    try:
        ht49.merged_manifest(stored, [{"file": "dfl_X_clean_0.npz", "sha256": "3"}], {})
    except ValueError:
        pass
    else:
        raise AssertionError("A new record replaced a stored one")


def test_prepare_needs_explicit_choices_for_the_augmented_set(tmp_path):
    try:
        ht49.prepare("augmented", tmp_path, tmp_path, tmp_path, tmp_path, tmp_path / "s", tmp_path / "t", raw=True)
    except ValueError as error:
        assert "explicit" in str(error)
    else:
        raise AssertionError("The augmented set was prepared without its choices")
    assert not (tmp_path / "s").exists()


def entry(cost, guard, pff, composite, groups=None):
    groups = dict.fromkeys(ht49.GROUPS, .7) | {"pff": pff} | (groups or {})
    return {"owner_cost": cost, "guards": {"mean": guard}, "groups": groups, "composite": composite}


def test_carry_rule_selects_by_guard_and_pff_then_checks_the_conditions():
    reference = entry(10, .60, .66, .72)
    entries = [entry(8, .61, .66, .72), entry(10, .62, .665, .719), entry(12, .60, .66, .73)]
    decision = ht49.carry(reference, entries)
    assert decision["selected_cost"] == 10
    assert decision["conditions"] == {"guard_or_pff": True, "composite": True, "groups": True}
    assert decision["carried"] is True
    worse = [entry(8, .63, .66, .71)]
    assert ht49.carry(reference, worse)["conditions"]["composite"] is False
    harmed = [entry(8, .63, .66, .72, {"metrica": .68})]
    assert ht49.carry(reference, harmed)["conditions"]["groups"] is False
    flat = [entry(8, .61, .66, .72)]
    assert ht49.carry(reference, flat)["carried"] is False
