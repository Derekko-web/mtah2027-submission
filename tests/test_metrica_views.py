"""Synthetic checks of D24's degraded Metrica views and their measurements (DECISIONS.md D24)."""

import numpy as np

from dfl_submission import augment, diagnosis, metrica_views as mv
from dfl_transfer_features import corrupt
from evaluation import possession_metrics as pm


H, A, D, X = pm.H, pm.A, pm.D, pm.X


def labels(*parts):
    return np.concatenate([np.full(length, code, dtype=np.int8) for code, length in parts])


def synthetic_half(seed=0):
    """A half with a ball missing exactly on its dead runs, as in Metrica's corrected tracking."""
    rng = np.random.default_rng(seed)
    truth = labels((X, 5), (H, 300), (D, 120), (A, 300), (D, 80), (H, 200), (X, 5))
    n = len(truth)
    positions = rng.uniform(-40, 40, (n, 6, 2)).astype(np.float32)
    ball = rng.uniform(-30, 30, (n, 2)).astype(np.float32)
    ball[truth == D] = np.nan
    ball[truth == X] = np.nan
    return {"positions": positions, "ball": ball, "teams": np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
            "pitch": np.array([105.0, 68.0]), "labels": truth}


def test_masked_view_is_the_dfl_degradation_with_its_seed():
    half = synthetic_half()
    views = mv.build_views(half, 2, 1)
    positions, ball = corrupt(half["positions"], half["ball"], 640000 + 10 * 2 + 1)
    np.testing.assert_array_equal(views["masked"][0], positions)
    np.testing.assert_array_equal(views["masked"][1], ball)


def test_handled_view_differs_from_the_masked_view_only_in_the_dead_ball():
    half = synthetic_half()
    views = mv.build_views(half, 1, 0)
    masked, handled = views["masked"], views["handled"]
    np.testing.assert_array_equal(masked[0], handled[0])
    live = half["labels"] != D
    np.testing.assert_array_equal(masked[1][live], handled[1][live])
    dead = half["labels"] == D
    assert not np.isfinite(masked[1][dead]).any()
    expected = augment.handle_dead_ball(half["labels"], half["positions"], half["ball"],
                                        np.random.default_rng(650000 + 10 * 1 + 0))
    _, again = corrupt(half["positions"], expected, 640000 + 10 * 1 + 0)
    np.testing.assert_array_equal(handled[1], again)


def test_team_swap_view_swaps_players_of_the_handled_view():
    half = synthetic_half()
    views = mv.build_views(half, 3, 1)
    handled, swapped = views["handled"], views["team_swap"]
    np.testing.assert_array_equal(handled[1], swapped[1])
    expected = augment.swap_teams(handled[0], half["teams"], np.random.default_rng(660000 + 10 * 3 + 1),
                                  every=augment.GUARD_SWAP_EVERY, frames=lambda _: augment.GUARD_SWAP_FRAMES)
    np.testing.assert_array_equal(swapped[0], expected)
    assert not np.array_equal(swapped[0], handled[0], equal_nan=True)


def test_view_checks_need_hidden_live_ball_and_more_dead_ball_when_handled():
    half = synthetic_half()
    views = mv.build_views(half, 1, 0)
    rows = [mv.visibility(half["labels"], views)]
    check = mv.view_check(rows)
    live = np.isin(half["labels"], (H, A))
    assert check["masked_live_hidden"] == float(np.mean(~np.isfinite(views["masked"][1][live]).all(axis=1)))
    assert check["passed"] is (0.30 <= check["masked_live_hidden"] <= 0.50
                               and check["handled_dead_seen"] > check["masked_dead_seen"])
    bad = dict(rows[0], handled_dead_seen=0, masked_dead_seen=0)
    assert mv.view_check([bad])["passed"] is False


def test_brief_stoppages_count_all_and_interior_runs():
    runs = pm.valid_runs(labels((H, 30), (D, 10), (H, 30)))
    predicted = labels((D, 5), (H, 25), (D, 10), (A, 30))
    counts = mv.brief_stoppages(predicted, runs)
    assert counts == {"all": 2, "interior": 1}
    predicted = labels((H, 30), (D, 30), (A, 10))
    assert mv.brief_stoppages(predicted, runs) == {"all": 0, "interior": 0}


def test_view_scores_and_d24_mean():
    truth = labels((H, 100), (D, 50), (A, 100))
    halves = []
    for view in mv.VIEWS:
        for game in (1, 2, 3):
            for index in (0, 1):
                halves.append(diagnosis.HalfSignals("metrica", str(game), view, index, truth, (), truth.copy()))
    truths = {game: {"aba3": np.r_[truth, X, truth].astype(np.int8)} for game in (1, 2, 3)}
    scores = mv.view_scores(halves, truths, lambda half: half.d3)
    assert set(scores) == set(mv.VIEWS)
    assert scores["masked"]["1"]["aba3"]["spell_f1"] == 1.0
    assert mv.d24_mean(scores) == 1.0
    worse = mv.view_scores(halves, truths, lambda half: np.where(half.d3 == D, H, half.d3).astype(np.int8))
    assert mv.d24_mean(worse) < 1.0


def test_measure_is_written_once(tmp_path):
    path = tmp_path / "record.json"
    mv.write_new(path, {"a": 1})
    try:
        mv.write_new(path, {"a": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("D24 records are written once")


def test_check_record_reproduces_or_refuses(tmp_path):
    truth = labels((H, 100), (D, 50), (A, 100))
    halves = [diagnosis.HalfSignals("metrica", str(game), view, index, truth, (), truth.copy())
              for view in mv.VIEWS for game in (1, 2, 3) for index in (0, 1)]
    truths = {game: {"aba3": np.r_[truth, X, truth].astype(np.int8)} for game in (1, 2, 3)}
    scores = mv.view_scores(halves, truths, lambda half: half.d3)
    record = tmp_path / "d24.json"
    record.write_text(__import__("json").dumps({"views_report": {view: {"matches": scores[view]} for view in mv.VIEWS}}))
    assert mv.check_record(halves, truths, record) == scores
    halves[0].d3[:10] = A
    try:
        mv.check_record(halves, truths, record)
    except ValueError as error:
        assert "D24" in str(error)
    else:
        raise AssertionError("Changed labels must not reproduce the D24 record")
