"""D28 on synthetic data: the ball fill, its hook into the forecasts, the gates and the build."""

import json
import os
from pathlib import Path

import numpy as np

from dfl_submission import diagnosis, gap_trial as gt, kickoff, metrica_sample as ms, metrica_views as mv
from evaluation import possession_metrics as pm


NAN = np.nan


def ball_with_gap(before, length, after, start=(0.0, 0.0), stop=None):
    """Seen ball for `before` frames, missing for `length`, seen for `after`."""
    n = before + length + after
    ball = np.tile(np.array(start, dtype=np.float32), (n, 1))
    if stop is not None:
        ball[before + length:] = np.array(stop, dtype=np.float32)
    ball[before:before + length] = NAN
    return ball


def test_gap_of_k_frames_is_filled_on_the_straight_line():
    ball = ball_with_gap(10, 6, 10, start=(0.0, 0.0), stop=(7.0, 14.0))
    filled = gt.fill(ball, 6)
    expected = np.array([[i, 2 * i] for i in range(1, 7)], dtype=np.float32)
    assert np.allclose(filled[10:16], expected)
    assert np.array_equal(filled[:10], ball[:10]) and np.array_equal(filled[16:], ball[16:])
    assert filled.dtype == ball.dtype


def test_gap_of_k_plus_one_frames_is_not_filled():
    ball = ball_with_gap(10, 7, 10, stop=(7.0, 14.0))
    assert np.isnan(gt.fill(ball, 6)[10:17]).all()
    assert np.isfinite(gt.fill(ball, 7)[10:17]).all()


def test_each_gap_is_judged_on_its_own_length():
    ball = np.zeros((60, 2), dtype=np.float32)
    ball[5:8] = NAN      # 3 frames
    ball[20:26] = NAN    # 6 frames
    ball[40:48] = NAN    # 8 frames
    filled = gt.fill(ball, 6)
    assert np.isfinite(filled[5:8]).all() and np.isfinite(filled[20:26]).all()
    assert np.isnan(filled[40:48]).all()


def test_gaps_at_the_half_edges_are_untouched():
    ball = np.zeros((40, 2), dtype=np.float32)
    ball[:3] = NAN
    ball[-3:] = NAN
    filled = gt.fill(ball, 25)
    assert np.isnan(filled[:3]).all() and np.isnan(filled[-3:]).all()
    assert np.isfinite(filled[3:-3]).all()


def test_an_all_missing_ball_is_untouched():
    ball = np.full((30, 2), NAN, dtype=np.float32)
    assert np.isnan(gt.fill(ball, 25)).all()


def test_a_frame_with_one_missing_coordinate_counts_as_missing():
    ball = np.zeros((20, 2), dtype=np.float32)
    ball[10] = (NAN, 5.0)
    filled = gt.fill(ball, 6)
    assert np.array_equal(filled[10], np.zeros(2, dtype=np.float32))
    assert not gt.seen(ball)[10]


def test_the_input_array_is_not_modified():
    ball = ball_with_gap(10, 6, 10, stop=(7.0, 14.0))
    kept = ball.copy()
    gt.fill(ball, 25)
    assert np.array_equal(ball, kept, equal_nan=True)


def test_zero_limit_leaves_every_ball_unchanged():
    rng = np.random.default_rng(0)
    ball = rng.uniform(-50, 50, (500, 2)).astype(np.float32)
    ball[rng.random(500) < .4] = NAN
    assert np.array_equal(gt.fill(ball, 0), ball, equal_nan=True)


def fake_half_signals(teacher, positions, ball, teams, pitch):
    """A deterministic stand-in for D15: its 'signals' are the ball it was given."""
    signals = (np.nan_to_num(ball[:, 0], nan=-1.0), np.nan_to_num(ball[:, 1], nan=-1.0), np.zeros(len(ball)),
               np.zeros(len(ball)))
    return signals, np.isfinite(ball).all(axis=1).astype(np.int8)


def test_a_ball_without_gaps_gives_identical_decoder_inputs():
    original = diagnosis.half_signals
    diagnosis.half_signals = fake_half_signals
    try:
        rng = np.random.default_rng(1)
        ball = rng.uniform(-50, 50, (300, 2)).astype(np.float32)
        ball[:4] = NAN                       # an edge gap is never filled either
        direct = fake_half_signals(None, None, ball, None, None)
        for limit in gt.GRID:
            hooked = gt.signals_with(limit)(None, None, ball, None, None)
            assert all(np.array_equal(a, b) for a, b in zip(hooked[0], direct[0]))
            assert np.array_equal(hooked[1], direct[1])
        # With an interior gap the hook passes the filled ball on.
        ball[100:105] = NAN
        hooked = gt.signals_with(6)(None, None, ball, None, None)
        assert np.array_equal(hooked[1], np.r_[np.zeros(4), np.ones(296)].astype(np.int8))
    finally:
        diagnosis.half_signals = original


def test_the_grid_is_six_twelve_and_twenty_five_frames():
    assert gt.GRID == (6, 12, 25)


def test_coverage_counts_missing_and_filled_frames():
    ball = np.zeros((100, 2), dtype=np.float32)
    ball[:2] = NAN          # edge: never filled
    ball[10:16] = NAN       # 6
    ball[30:42] = NAN       # 12
    ball[60:80] = NAN       # 20
    live = np.zeros(100, dtype=bool)
    live[10:50] = True
    counts = gt.counts(ball, live, (6, 12, 25))
    assert counts["missing"] == 40 and counts["live_missing"] == 18
    assert counts["filled"] == {"6": 6, "12": 18, "25": 38}
    assert counts["live_filled"] == {"6": 6, "12": 18, "25": 18}
    total = gt.summarise([counts, counts])
    assert total["missing"] == 80 and total["share"]["12"] == 18 / 40 and total["live_share"]["25"] == 1.0


def change(primary=.005, conventions=0.0, guards=(0.0, 0.0), pff=0.0):
    return {"primary": primary, "conventions": {"aba1.75": conventions, "aba4": 0.0, "aba4.5": 0.0},
            "guards": {"dead_carried": guards[0], "team_swap": guards[1]}, "pff": pff}


def test_development_gates():
    assert gt.eligible(change(), True, 0.0)
    assert gt.eligible(change(primary=.02, conventions=-.01, guards=(-.01, -.01), pff=-.01), True, 0.0)
    assert not gt.eligible(change(primary=.0049), True, 0.0)
    assert not gt.eligible(change(conventions=-.0101), True, 0.0)
    assert not gt.eligible(change(guards=(-.0101, 0.0)), True, 0.0)
    assert not gt.eligible(change(guards=(0.0, -.0101)), True, 0.0)
    assert not gt.eligible(change(pff=-.0101), True, 0.0)
    assert not gt.eligible(change(), False, 0.0)
    assert not gt.eligible(change(), True, -1e-9)


def test_confirmation_gates_leave_out_metrica_and_d24():
    assert gt.eligible(change())
    assert not gt.eligible(change(pff=-.011))


def test_selection_takes_the_highest_primary_and_the_smaller_limit_on_a_tie():
    entries = [{"limit": 6, "eligible": True, "changes": change(primary=.010)},
               {"limit": 12, "eligible": True, "changes": change(primary=.012)},
               {"limit": 25, "eligible": False, "changes": change(primary=.020)}]
    assert gt.select(entries) == 12
    entries[0]["changes"] = change(primary=.012)
    assert gt.select(entries) == 6
    assert gt.select([dict(entry, eligible=False) for entry in entries]) is None


def test_records_are_written_once(tmp_path):
    path = tmp_path / "record.json"
    gt.write_new(path, {"a": 1})
    try:
        gt.write_new(path, {"a": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("D28 records are written once")


def test_cache_identity_names_the_limit_and_the_fill_code():
    assert gt.variant(6) != gt.variant(12)
    assert gt.variant(6)["limit"] == 6 and gt.variant(6)["protocol"] == gt.PROTOCOL
    assert len(gt.variant(6)["fill_sha256"]) == 64


def test_build_fills_then_runs_d15_then_d27(tmp_path):
    confirmation = tmp_path / "confirmation.json"
    confirmation.write_text(json.dumps({"passed": True, "limit": 12}))
    labels = np.full(31000, "X", dtype="<U1")
    labels[12000:30000] = "H"
    blocks = [(1, 1000, 9999), (2, 12000, 29999)]
    tags = tmp_path / "tags.xml"
    tags.write_text("<file><ALL_INSTANCES><instance><ID>1</ID><start>500.0</start><end>505.0</end>"
                    "<code>SET PIECES</code><label><group>Type</group><text>Kick Off (Start)</text></label>"
                    "</instance></ALL_INSTANCES></file>", encoding="utf-8")
    seen_prepare, written = [], {}

    def d15_labels(prepare=None):
        seen_prepare.append(prepare)
        return labels.copy()

    originals = kickoff.d15_labels, kickoff.tag_path, kickoff.half_blocks, gt.write_csv
    kickoff.d15_labels, kickoff.tag_path, kickoff.half_blocks = d15_labels, (lambda: tags), (lambda: blocks)
    gt.write_csv = lambda output, values: written.update(output=output, labels=values)
    try:
        gt.build(tmp_path / "out.csv", recipe_folder=tmp_path)
    finally:
        kickoff.d15_labels, kickoff.tag_path, kickoff.half_blocks, gt.write_csv = originals
    prepare = seen_prepare[0]
    ball = ball_with_gap(5, 12, 5, stop=(13.0, 0.0))
    assert np.array_equal(prepare(ball), gt.fill(ball, 12))
    assert np.isnan(prepare(ball_with_gap(5, 13, 5))[5:18]).all()
    assert (written["labels"][12000:12500] == "D").all() and (written["labels"][12500:30000] == "H").all()


def test_build_refuses_without_a_passed_confirmation(tmp_path):
    (tmp_path / "confirmation.json").write_text(json.dumps({"passed": False, "limit": 12}))
    try:
        gt.build(tmp_path / "out.csv", recipe_folder=tmp_path)
    except ValueError as error:
        assert "confirmation" in str(error)
    else:
        raise AssertionError("D28 builds only after a passed confirmation")


# -- the Metrica hooks -------------------------------------------------------------------------------

def metrica_half(seed):
    rng = np.random.default_rng(seed)
    labels = np.r_[np.full(5, pm.X), np.full(200, pm.H), np.full(60, pm.D), np.full(200, pm.A),
                   np.full(5, pm.X)].astype(np.int8)
    n = len(labels)
    ball = rng.uniform(-30, 30, (n, 2)).astype(np.float32)
    ball[100:104] = NAN
    ball[labels == pm.D] = NAN
    control = np.where(labels == pm.D, pm.D, labels).astype(np.int8)
    return {"positions": rng.uniform(-40, 40, (n, 6, 2)).astype(np.float32), "ball": ball,
            "teams": np.array([0, 0, 0, 1, 1, 1], dtype=np.int8), "pitch": np.array([105.0, 68.0]),
            "labels": labels, "control": control}


def parsed_folder(folder):
    parsed = Path(folder) / "parsed"
    parsed.mkdir()
    entries = {}
    for game in ms.GAMES:
        for index in (1, 2):
            half = metrica_half(10 * game + index)
            path = parsed / f"{game}_{index}.npz"
            np.savez_compressed(path, **{name: half[name] for name in ms.ARRAYS})
            entries[path.name] = list(ms.identity(path))
    (parsed / "manifest.json").write_text(json.dumps({"reader_sha256": "test", "parsed": entries}))
    return parsed


def test_metrica_reference_halves_hook_runs_on_every_half_and_names_its_variant(tmp_path):
    parsed = parsed_folder(tmp_path)
    original = ms.reference_identity
    ms.reference_identity = lambda reference, parsed_: {"reference": reference}
    calls = []

    def hook(model, positions, ball, teams, pitch):
        calls.append(model)
        return fake_half_signals(model, positions, gt.fill(ball, 6), teams, pitch)

    try:
        _, halves = ms.reference_halves("D15", parsed, tmp_path / "hooked", model="model", signals_of=hook,
                                        variant={"limit": 6})
        assert calls == ["model"] * 6 and len(halves) == 6
        assert all(half.d3[100:104].all() for half in halves)       # the 4-frame gap was filled
        stored = json.loads((tmp_path / "hooked" / "manifest.json").read_text())["identity"]
        assert stored == {"reference": "D15", "variant": {"limit": 6}}
        for bad in ({"variant": None, "cache": tmp_path / "other"}, {"variant": {"limit": 6}, "cache": None}):
            try:
                ms.reference_halves("D15", parsed, bad["cache"], model="model", signals_of=hook, variant=bad["variant"])
            except ValueError:
                pass
            else:
                raise AssertionError("A hooked run needs a variant and its own cache")
    finally:
        ms.reference_identity = original


def test_metrica_view_halves_hook_runs_on_every_view_and_names_its_variant(tmp_path):
    parsed = parsed_folder(tmp_path)
    originals = ms.reference_model, mv.identity
    ms.reference_model = lambda reference: "model"
    mv.identity = lambda parsed_, reference="D15": {"reference": reference}
    calls = []

    def hook(model, positions, ball, teams, pitch):
        calls.append(model)
        return fake_half_signals(model, positions, ball, teams, pitch)

    try:
        halves = mv.view_halves(parsed, tmp_path / "views", signals_of=hook, variant={"limit": 6})
        assert len(calls) == 18 and len(halves) == 18
        assert {half.view for half in halves} == set(mv.VIEWS)
        stored = json.loads((tmp_path / "views" / "manifest.json").read_text())["identity"]
        assert stored == {"reference": "D15", "variant": {"limit": 6}}
        for bad in ({"variant": None, "cache": tmp_path / "other"}, {"variant": {"limit": 6}, "cache": mv.CACHE}):
            try:
                mv.view_halves(parsed, bad["cache"], signals_of=hook, variant=bad["variant"])
            except ValueError:
                pass
            else:
                raise AssertionError("A hooked run needs a variant and its own cache")
    finally:
        ms.reference_model, mv.identity = originals


def test_metrica_hooks_default_to_d15s_own_forecast(tmp_path):
    parsed = parsed_folder(tmp_path)
    originals = ms.reference_identity, diagnosis.half_signals
    ms.reference_identity = lambda reference, parsed_: {"reference": reference}
    diagnosis.half_signals = fake_half_signals
    try:
        _, halves = ms.reference_halves("D15", parsed, tmp_path / "plain", model="model")
        stored = json.loads((tmp_path / "plain" / "manifest.json").read_text())["identity"]
        assert stored == {"reference": "D15"}
        assert all(not half.d3[100:104].any() for half in halves)   # nothing filled by default
    finally:
        ms.reference_identity, diagnosis.half_signals = originals


def test_every_command_but_build_needs_the_bundle_out_of_reach():
    previous = os.environ.get("MTAH_DATA_DIR")
    os.environ["MTAH_DATA_DIR"] = str(Path(__file__).resolve().parent)    # an existing folder
    try:
        for command in (lambda: gt.coverage(None, None, None, raw=True),
                        lambda: gt.develop(None, None, None, raw=True, d15_cache=None),
                        lambda: gt.confirm(None, None, None, raw=True, d15_holdout_cache=None)):
            try:
                command()
            except SystemExit:
                pass
            else:
                raise AssertionError("D28's coverage, develop and confirm must refuse while the bundle is reachable")
    finally:
        if previous is None:
            os.environ.pop("MTAH_DATA_DIR")
        else:
            os.environ["MTAH_DATA_DIR"] = previous
