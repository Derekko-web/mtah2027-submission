"""D56's situation hide, training views and preparation helpers on synthetic fixtures."""

import json

import numpy as np

from dfl_submission import augment, hidden_views as hv, situation_train as s56, situation_views as sv
from dfl_submission.external_benchmark import Half
from evaluation import possession_metrics as pm


def refused(call, kind=Exception):
    try:
        call()
    except kind:
        return True
    return False


def test_two_state_is_wide_masking_s_random_process():
    n, players = 3000, 6
    positions = np.zeros((n, players, 2), dtype=np.float32)
    ball = np.zeros((n, 2), dtype=np.float32)
    for seed in range(5):
        _, masked = augment.wide_masking(positions, ball, np.random.default_rng(seed))
        rng = np.random.default_rng(seed)
        rng.normal(0, .2, positions.shape)
        rng.normal(0, .2, ball.shape)
        rng.random(((n + 49) // 50, players))
        visibility = rng.uniform(*augment.VISIBILITY)
        assert np.array_equal(s56.two_state(n, visibility, rng), np.isnan(masked).any(axis=1))


def test_two_state_visibility_and_extend_runs():
    shares = [1 - s56.two_state(50_000, .8, np.random.default_rng(seed)).mean() for seed in range(20)]
    assert abs(np.mean(shares) - .8) < .02
    flags = np.zeros(200, dtype=bool)
    flags[10:20] = flags[50:51] = flags[190:200] = True
    extended, added = s56.extend_runs(flags, np.random.default_rng(1))
    starts, lengths = sv.runs_of(flags)
    for start, length in zip(starts, lengths):
        stop = start + length
        assert extended[start:stop].all()
    assert not (added & flags).any() and np.array_equal(extended, flags | added)
    for start, length in zip(*sv.runs_of(added)):
        assert 1 <= length <= 25 and flags[start - 1]       # each added stretch follows a run directly
    delays = [s56.extend_runs(np.r_[True, np.zeros(40, dtype=bool)], np.random.default_rng(seed))[1].sum()
              for seed in range(400)]
    assert min(delays) == 0 and max(delays) == 25 and abs(np.mean(delays) - 12.5) < 1.0


def test_situation_hide_combines_air_crowd_delay_and_random():
    n = 5000
    air, crowd = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    air[1000:1100] = True
    crowd[3000:3010] = True
    hidden, parts, visibility = s56.situation_hide(air, crowd, np.random.default_rng(4))
    assert .6 <= visibility <= .95
    assert hidden[air].all() and hidden[crowd].all() and hidden[parts["delay"]].all() and hidden[parts["random"]].all()
    assert np.array_equal(hidden, air | crowd | parts["delay"] | parts["random"])
    rng = np.random.default_rng(4)
    v = rng.uniform(.6, .95)
    _, delay = s56.extend_runs(air | crowd, rng)
    assert v == visibility and np.array_equal(delay, parts["delay"])
    assert np.array_equal(s56.two_state(n, v, rng), parts["random"])


def tracking(n=3000, players=22, seed=0):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-30, 30, (n, players, 2)).astype(np.float32)
    ball = rng.uniform(-30, 30, (n, 2)).astype(np.float32)
    positions[::9, :3] = ball[::9, None, :] + np.float32(.4)
    teams = np.repeat(np.array([0, 1], dtype=np.int8), players // 2)
    labels = np.repeat(np.array([pm.H, pm.A, pm.D], dtype=np.int8), n // 3)
    return positions, ball, teams, labels


def test_dfl_view_keeps_noise_and_dropout_and_hides_the_situations():
    positions, ball, teams, labels = tracking()
    z = np.zeros(len(ball))
    z[500:700] = 4.0
    rng = np.random.default_rng(10)
    view_positions, view_ball, visibility, parts = s56.dfl_view(positions, ball, z, rng)
    reference_positions, reference_ball = hv.noise_and_dropout(positions, ball, np.random.default_rng(10))
    assert np.array_equal(view_positions, reference_positions, equal_nan=True)
    hidden = np.isnan(view_ball).any(axis=1)
    assert np.array_equal(hidden, parts["air"] | parts["crowd"] | parts["delay"] | parts["random"])
    assert np.array_equal(view_ball[~hidden], reference_ball[~hidden])
    assert parts["air"][500:700].all() and parts["air"].sum() == 200
    assert np.array_equal(parts["crowd"], sv.crowd_mask(positions, ball)) and parts["crowd"][::9].all()


def test_training_view_per_source_and_set():
    positions, ball, teams, labels = tracking()
    pitch = np.array([105.0, 68.0])
    clean = Half("dfl", "WMX", "clean", 1, labels, positions, ball, teams, pitch)
    masked = Half("dfl", "WMX", "masked", 1, labels, positions, ball, teams, pitch)
    z = np.zeros(len(ball))
    assert s56.training_view("current", masked, height=z) is None
    assert refused(lambda: s56.training_view("current", clean, height=z[:-1]), ValueError)
    current, v1, _ = s56.training_view("current", clean, height=z)
    assert current.view == "situation" and current.key == "dfl_WMX_situation_1"
    seed = s56.SEEDS["current_dfl"] + 100 * hv.DFL_ORDER.index("WMX") + 1
    expected_positions, _, _, _ = s56.dfl_view(positions, ball, z, np.random.default_rng(seed))
    assert np.array_equal(current.positions, expected_positions, equal_nan=True)
    swapped, _, _ = s56.training_view("augmented", clean, height=z, swaps=True)
    plain, _, _ = s56.training_view("augmented", clean, height=z, swaps=False)
    assert np.array_equal(swapped.ball, plain.ball, equal_nan=True)
    assert not np.array_equal(swapped.positions, plain.positions, equal_nan=True)
    skill = Half("skillcorner", "1925299", "observed", 0, labels, positions, ball, teams, pitch)
    assert refused(lambda: s56.training_view("current", skill), ValueError)
    view, visibility, parts = s56.training_view("augmented", skill, skill_order=3)
    assert view.positions is positions and not parts["air"].any()
    assert np.array_equal(np.isnan(view.ball).any(axis=1), parts["crowd"] | parts["delay"] | parts["random"])
    other, _, _ = s56.training_view("current", skill, skill_order=3)
    assert not np.array_equal(other.ball, view.ball, equal_nan=True)     # the sets draw from different seeds


def test_view_statistics_by_hand():
    n = 1000
    ball = np.zeros((n, 2), dtype=np.float32)
    original = ball.copy()
    original[900:950] = np.nan
    parts = {name: np.zeros(n, dtype=bool) for name in ("air", "crowd", "delay", "random")}
    parts["air"][100:200] = True
    parts["crowd"][150:160] = True
    parts["delay"][200:210] = True
    parts["random"][195:205] = parts["random"][600:700] = True
    hidden = parts["air"] | parts["crowd"] | parts["delay"] | parts["random"]
    ball[hidden] = np.nan
    ball[900:950] = np.nan
    labels = np.full(n, pm.H, dtype=np.int8)
    labels[500:] = pm.D
    stats = s56.view_statistics([(ball, labels, parts, original)])
    assert stats["hidden_frames"] == 260 and np.isclose(stats["visibility"], .74)
    assert np.isclose(stats["share_hidden_by_height"], .1) and np.isclose(stats["share_hidden_by_crowd"], .01)
    assert np.isclose(stats["share_hidden_by_delay"], .01) and np.isclose(stats["share_hidden_by_random"], .11)
    assert np.isclose(stats["share_hidden_by_situation_with_delay"], .11)
    assert np.isclose(stats["share_hidden_by_random_only"], .1) and np.isclose(stats["share_ball_missing_before_the_hide"], .05)
    assert stats["live_hidden_frames"] == 110 and np.isclose(stats["live_visibility"], .78)


def test_seeds_are_new():
    import re
    from pathlib import Path
    package = Path(sv.__file__).parent
    used = {270000, 370000, 520000, 530000, 610000, 790000, *hv.SEEDS.values(), *sv.SEEDS.values(),
            *(seed + sv.PLACEMENT for seed in sv.SEEDS.values())}
    for name in ("metrica_trial.py", "metrica_views.py", "broadcast_trial.py"):
        block = re.search(r"^SEEDS = \{([^}]*)\}", (package / name).read_text(), re.M).group(1)
        used |= {int(value) for value in re.findall(r":\s*(\d+)", block)}
    for seed in s56.SEEDS.values():
        assert all(abs(seed - other) >= 10000 for other in used), seed
    assert len(set(s56.SEEDS.values())) == 4


def test_heights_of_checks_digests_and_writes_are_once(tmp_path):
    folder = tmp_path / "heights"
    folder.mkdir()
    np.savez_compressed(folder / "WMX.npz", z_0=np.zeros(3), z_1=np.ones(4), frames_0=np.arange(3), frames_1=np.arange(4))
    from dfl_submission.pff import digest
    record = tmp_path / "reader.json"
    record.write_text(json.dumps({"passed": True, "heights": {"WMX": digest(folder / "WMX.npz")}}))
    heights = s56.heights_of(record, folder)
    assert heights["WMX", 1].tolist() == [1, 1, 1, 1]
    record.write_text(json.dumps({"passed": True, "heights": {"WMX": "0" * 64}}))
    assert refused(lambda: s56.heights_of(record, folder), ValueError)
    record.write_text(json.dumps({"passed": False, "heights": {}}))
    assert refused(lambda: s56.heights_of(record, folder), SystemExit)
    exists = tmp_path / "exists.json"
    exists.write_text("{}")
    assert refused(lambda: s56.reader(None, output=exists), FileExistsError)
    (tmp_path / "d56_prepare_current.json").write_text("{}")
    assert refused(lambda: s56.prepare("current", None, None, scratch=tmp_path, recipe=tmp_path), FileExistsError)
    assert refused(lambda: s56.prepare("other", None, None, scratch=tmp_path, recipe=tmp_path), ValueError)
