"""D18 PathCRF adaptor on synthetic tracks (no match data, no PathCRF clone)."""

import json
import os
from pathlib import Path

import numpy as np

from dfl_submission import diagnosis, pathcrf_trial as trial
from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X
ROOT = Path(__file__).resolve().parents[1]


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def test_pins_match_the_decision_record():
    record = (ROOT / "DECISIONS.md").read_text()
    section = record[record.index("## D18"):]
    assert trial.COMMIT in section
    assert trial.PINNED[trial.CHECKPOINT][:8] in section
    assert trial.PINNED["LICENSE"][:8] in section
    assert trial.CHECKPOINT in section and "weights_only=True" in section
    assert trial.SANITY_MIN == .80 and "at least 0.80" in section


def test_interpolate_interior_fills_only_between_observations():
    positions = np.full((8, 2, 2), np.nan)
    positions[2, 0] = (0, 0)
    positions[6, 0] = (4, 8)
    positions[3, 1] = (1, 1)
    filled = trial.interpolate_interior(positions)
    assert np.isnan(filled[:2, 0]).all() and np.isnan(filled[7, 0]).all()
    assert np.allclose(filled[2:7, 0, 0], [0, 1, 2, 3, 4]) and np.allclose(filled[2:7, 0, 1], [0, 2, 4, 6, 8])
    assert np.isfinite(filled[3, 1]).all() and np.isnan(np.delete(filled[:, 1], 3, axis=0)).all()


def test_to_pathcrf_maps_centred_metres_to_the_corner_frame():
    positions = np.array([[[-52, -34], [0, 0], [52, 34]]], dtype=float)
    mapped = trial.to_pathcrf(positions, np.array([104, 68]))
    assert np.allclose(mapped[0], [[0, 0], [52.5, 34], [105, 68]])


def test_running_features_on_a_straight_run():
    frames = np.arange(100)
    xy = np.c_[frames * 0.2, np.full(100, 10.0)]          # 5 m/s along x
    features = trial.running_features(xy)
    assert np.allclose(features[:, :2], xy)
    assert np.allclose(features[:, 2], 5.0) and np.allclose(features[:, 3], 0.0)
    assert np.allclose(features[:, 4], 5.0)
    assert np.allclose(features[:, 5], 0.0, atol=1e-6)


def test_running_features_replaces_frames_that_imply_more_than_twelve_metres_per_second():
    xy = np.c_[np.arange(40) * 0.1, np.zeros(40)]
    xy[20] = (30.0, 0.0)
    features = trial.running_features(xy)
    assert abs(features[20, 0] - 2.0) < 1e-9
    assert features[:, 4].max() < 3.0


def test_running_features_uses_plain_differences_on_short_tracks():
    xy = np.c_[np.arange(5) * 0.1, np.zeros(5)]
    features = trial.running_features(xy)
    assert np.allclose(features[:, 2], 2.5) and features[0, 5] == 0 and features[-1, 5] == 0


def team_scene(frames=60, home_keeper_x=-48.0, away_keeper_x=48.0, home=11, away=11):
    """Home slots first, then away; the keepers are the first slot of each team."""
    teams = np.r_[np.zeros(home, int), np.ones(away, int)]
    positions = np.zeros((frames, home + away, 2))
    positions[:, :home, 0] = np.linspace(-20, 0, home)
    positions[:, home:, 0] = np.linspace(0, 20, away)
    positions[:, 0, 0] = home_keeper_x
    positions[:, home, 0] = away_keeper_x
    positions[:, :, 1] = np.linspace(-30, 30, home + away)
    return positions, teams


def test_goalkeepers_pick_the_deepest_slot_per_team():
    positions, teams = team_scene()
    features = trial.half_features(positions, np.array([105, 68]))
    assert trial.goalkeepers(features, teams) == {0: 0, 1: 11}


def test_segment_input_puts_the_team_defending_the_left_goal_first_and_pads():
    positions, teams = team_scene(frames=23, home_keeper_x=48.0, away_keeper_x=-48.0, away=10)
    features = trial.half_features(positions, np.array([105, 68]))
    keepers = trial.goalkeepers(features, teams)
    nodes, steps, node_team = trial.segment_input(features, teams, keepers, 0, 23)
    assert nodes.shape == (6, 26, 8) and steps.tolist() == [0, 5, 10, 15, 20, 22]
    assert node_team[:10].tolist() == [1] * 10 and node_team[10] == -1        # away first, padded to 11
    assert node_team[11:22].tolist() == [0] * 11 and (node_team[22:] == -1).all()
    assert (nodes[:, :10, 0] == -1).all() and (nodes[:, 10] == 0).all() and (nodes[:, 11:22, 0] == 1).all()
    assert nodes[0, :, 1].sum() == 2                                           # one keeper flag per team
    assert np.allclose(nodes[0, 22:, 2:4], trial.OUTSIDE) and (nodes[:, 22:, :2] == 0).all()


def test_segment_input_keeps_eleven_per_team_and_holds_missing_edges():
    pitch = np.array([105, 68])
    positions, teams = team_scene(frames=30, home=12)
    positions[:10, 11] = np.nan              # a twelfth home slot, seen on the fewest frames
    positions[:5, 3] = np.nan                # a player first seen at frame 5
    positions[5:, 3, 0] += np.arange(25) * 0.1
    features = trial.half_features(positions, pitch)
    keepers = trial.goalkeepers(features, teams)
    nodes, steps, node_team = trial.segment_input(features, teams, keepers, 0, 30)
    assert (node_team == 0).sum() == 11 and (node_team == 1).sum() == 11
    dropped = trial.to_pathcrf(positions[10:11, 11:12], pitch)[0, 0]
    assert not any(np.allclose(xy, dropped) for xy in nodes[2, node_team == 0, 2:4])
    first_seen = trial.to_pathcrf(positions[5:6, 3:4], pitch)[0, 0]
    rows = [k for k in np.flatnonzero(node_team == 0) if np.allclose(nodes[0, k, 2:4], first_seen)]
    assert len(rows) == 1 and (nodes[0, rows[0], 4:] == 0).all() and nodes[1, rows[0], 4] > 0
    assert np.isfinite(nodes).all()


def test_windows_cover_every_step_and_blend_to_the_same_value():
    for length in (1, 30, 50, 51, 74, 75, 76, 130, 401):
        spans = trial.windows(length)
        assert spans[0][0] == 0 and spans[-1][1] == length
        assert all(stop - start == min(length, trial.WINDOW) for start, stop in spans)
        total = np.zeros(length)
        for i, (start, stop) in enumerate(spans):
            total[start:stop] += trial.window_weights(spans, i)
        assert (total > 0).all()


class FakeCRF:
    def decode(self, emissions, embeds):
        import torch
        assert embeds.shape[:3] == emissions.shape
        return emissions.argmax(dim=-1)


class FakeModel:
    """Edge (0 -> 0) scores highest before step 20, edge (11 -> 11) after, whatever the window."""

    def __init__(self):
        import torch
        n = 26
        self.valid_orig_edge_ids = torch.tensor([s * n + d for s in range(22) for d in range(n)]
                                                + [o * n + o for o in range(22, n)])
        self.comp_src = self.valid_orig_edge_ids // n
        self.comp_dst = self.valid_orig_edge_ids % n
        self.crf = FakeCRF()
        self.calls = []

    def __call__(self, x):
        import torch
        steps = x.shape[1]
        self.calls.append(steps)
        logits = torch.zeros(1, steps, 26 * 26)
        late = x[0, :, 0, 2] > 0.5
        logits[0, ~late, 0] = 1.0
        logits[0, late, 11 * 26 + 11] = 1.0
        return None, logits, torch.zeros(1, steps, 26 * 26, 3)


def test_segment_edges_blends_windows_and_decodes_every_step():
    nodes = np.zeros((130, 26, 8), dtype=np.float32)
    nodes[20:, 0, 2] = 1.0
    model = FakeModel()
    sender, receiver = trial.segment_edges(model, nodes)
    assert len(sender) == 130 and model.calls == [50] * len(trial.windows(130))
    assert (sender[:20] == 0).all() and (sender[20:] == 11).all() and (sender == receiver).all()


def test_step_teams_follow_holder_passer_and_the_split_rule():
    node_team = np.r_[np.zeros(11, int), np.ones(11, int), -np.ones(4, int)]
    node_team[10] = -1                                   # a pad in the first block
    sender = np.array([0, 0, 1, 1, 2, 2, 2, 2, 12, 12, 22, 10])
    receiver = np.array([0, 1, 1, 22, 13, 13, 13, 13, 12, 3, 22, 10])
    teams = trial.step_teams(sender, receiver, node_team)
    assert teams[:4].tolist() == [0, 0, 0, 0]            # hold, pass to a teammate, hold, out of play
    assert teams[4:8].tolist() == [0, 0, 1, 1]           # in flight to an opponent: passer, then receiver
    assert teams[8] == 1 and teams[9] == 1               # away holds, then a one-step cross pass stays away
    assert teams[10] == -1 and teams[11] == -1           # ball out of play; a pad never owns the ball


def test_frame_teams_hold_each_step_until_the_next():
    steps = trial.downsample(12)
    assert steps.tolist() == [0, 5, 10, 11]
    held = trial.frame_teams(np.array([0, 1, -1, 0]), steps, 12)
    assert held.tolist() == [0] * 5 + [1] * 5 + [-1, 0]


def test_hybrid_takes_pathcrf_teams_only_where_d3_is_live():
    d3 = np.array([H, H, D, A, A, X], dtype=np.int8)
    pathcrf = np.array([A, D, H, H, A, H], dtype=np.int8)
    assert trial.hybrid(d3, pathcrf).tolist() == [A, H, D, H, A, X]


def test_team_counts_fill_predicted_dead_frames_inside_true_live_segments():
    truth = labels((H, 10), (D, 5), (A, 10))
    prediction = labels((H, 6), (D, 4), (D, 5), (H, 3), (A, 7))
    counts = trial.team_counts(truth, prediction, trial.team_home(prediction), np.ones(len(truth), bool))
    assert counts == (17, 20)
    hidden = np.zeros(len(truth), bool)
    hidden[15:20] = True
    assert trial.team_counts(truth, prediction, trial.team_home(prediction), hidden) == (2, 5)


def test_trigger_needs_both_sets():
    def accuracy(masked, skill):
        return {"groups": {"aba3": {"dfl_masked": {"ball_hidden": {"frames": 9, "d3": .8, "pathcrf_native": masked}},
                                    "skillcorner": {"ball_hidden": {"frames": 9, "d3": .7, "pathcrf_native": skill}}}}}
    assert trial.trigger(accuracy(.81, .71))["b2_triggered"]
    assert not trial.trigger(accuracy(.81, .70))["b2_triggered"]
    assert not trial.trigger(accuracy(.80, .9))["b2_triggered"]


def test_native_source_maps_dead_carried_to_masked():
    assert trial.native_source(("dfl", "WPY", "dead_carried", 1)) == ("dfl", "WPY", "masked", 1)
    assert trial.native_source(("skillcorner", "1", "observed", 0)) == ("skillcorner", "1", "observed", 0)


def test_cached_halves_never_open_pff_entries(tmp_path):
    half = diagnosis.HalfSignals("dfl", "WPY", "clean", 0, labels((H, 3)), tuple(np.zeros(3) for _ in range(4)),
                                 labels((H, 3)))
    entry = diagnosis.save_half(tmp_path, half)
    missing = {"source": "pff", "match": "1", "view": "observed", "index": 0, "file": "absent.npz", "sha256": "0"}
    (tmp_path / "manifest.json").write_text(json.dumps({"identity": {}, "halves": [entry, missing]}))
    loaded = trial.cached_halves(tmp_path)
    assert list(loaded) == [("dfl", "WPY", "clean", 0)]


def test_clone_identity_rejects_another_commit(tmp_path):
    (tmp_path / ".git" / "refs" / "heads").mkdir(parents=True)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git" / "refs" / "heads" / "main").write_text("0" * 40 + "\n")
    try:
        trial.clone_identity(tmp_path)
    except ValueError as error:
        assert "pinned" in str(error)
    else:
        raise AssertionError("an unpinned commit was accepted")


def test_write_new_writes_once(tmp_path):
    path = tmp_path / "record.json"
    trial.write_new(path, {"a": 1})
    try:
        trial.write_new(path, {"a": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("the record was overwritten")
    assert json.loads(path.read_text()) == {"a": 1}


def test_noise_only_reproduces_the_masked_views_noise_and_dropouts_only_adds_none():
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    from dfl_transfer_features import corrupt
    rng = np.random.default_rng(3)
    clean = rng.uniform(-40, 40, (120, 6, 2)).astype(np.float32)
    clean[:30, 5] = np.nan
    masked, _ = corrupt(clean, rng.uniform(-40, 40, (120, 2)), 270401)
    noisy = trial.noise_only(clean, masked, 270401)
    assert np.isfinite(noisy[30:]).all() and np.isnan(noisy[:30, 5]).all()
    dropped = trial.dropouts_only(clean, masked)
    seen = np.isfinite(dropped)
    assert np.array_equal(seen, np.isfinite(masked)) and np.array_equal(dropped[seen], clean[seen])
    try:
        trial.noise_only(clean, masked, 270402)
    except ValueError:
        pass
    else:
        raise AssertionError("noise from another seed was accepted")


def test_measure_refuses_while_the_competition_bundle_is_reachable(tmp_path):
    previous = os.environ.get("MTAH_DATA_DIR")
    os.environ["MTAH_DATA_DIR"] = str(tmp_path)
    try:
        trial.measure(tmp_path, tmp_path, tmp_path, tmp_path, output=tmp_path / "out.json")
    except SystemExit:
        pass
    else:
        raise AssertionError("measure ran with the bundle reachable")
    finally:
        if previous is None:
            os.environ.pop("MTAH_DATA_DIR", None)
        else:
            os.environ["MTAH_DATA_DIR"] = previous
    assert not (tmp_path / "out.json").exists()
