"""C2 re-declared for D48 (dfl_submission/output_check_d48.py), on synthetic inputs and the stored range files.

No competition file and no external match is read: the cached halves, the records and the predictions are
synthetic, and the D15 / D43 range files are rewritten from their own stored statistics.
"""

import inspect
import json
import sys

import numpy as np

from dfl_submission import diagnosis, hyper_trial as ht, output_check as oc, output_check_d48 as od, stack_trial as st
from evaluation import possession_metrics as pm


class Patch:
    """A minimal monkeypatch (the test runner provides no pytest fixtures besides tmp_path)."""

    def __init__(self):
        self.saved = []

    def setattr(self, owner, name, value):
        self.saved.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    def undo(self):
        for owner, name, value in reversed(self.saved):
            setattr(owner, name, value)


def stored(path):
    return json.loads(path.read_text())


def test_check_reads_the_d48_ranges_by_default_and_the_older_files_keep_their_commands():
    assert oc.RANGES.name == "output_check_ranges_d48.json"
    assert oc.D43_RANGES.name == "output_check_ranges_d43.json" and oc.D15_RANGES.name == "output_check_ranges.json"
    assert inspect.signature(oc.check).parameters["ranges"].default == oc.RANGES
    assert inspect.signature(oc.write_ranges).parameters["output"].default == oc.D15_RANGES
    assert inspect.signature(oc.write_d43_ranges).parameters["output"].default == oc.D43_RANGES
    assert inspect.signature(od.write_d48_ranges).parameters["output"].default == oc.RANGES
    assert oc.D15_RANGES.exists() and oc.D43_RANGES.exists()


def test_the_commands_write_and_read_the_right_files(tmp_path):
    calls = []
    patch = Patch()
    try:
        patch.setattr(oc, "write_ranges", lambda *a, **k: calls.append(("ranges", a, k)))
        patch.setattr(oc, "write_d43_ranges", lambda *a, **k: calls.append(("d43", a, k)))
        patch.setattr(od, "write_d48_ranges", lambda *a, **k: calls.append(("d48", a, k)))
        patch.setattr(oc, "check", lambda csv, ranges=oc.RANGES: calls.append(("check", csv, ranges)) or True)
        common = ["--dfl", "x", "--skill", "s", "--parsed", "p"]
        for argv in (["ranges", *common, "--d15-cache", "c"], ["ranges-d43", *common, "--dfl-raw"], ["ranges-d48", *common],
                     ["ranges-d48", *common, "--dfl-raw"], ["check", "f.csv"], ["check", "f.csv", "--ranges", "r.json"]):
            patch.setattr(sys, "argv", ["output_check", *argv])
            try:
                oc.main()
            except SystemExit as stop:
                assert stop.code == 0
    finally:
        patch.undo()
    names = [call[0] for call in calls]
    assert names == ["ranges", "d43", "d48", "d48", "check", "check"]
    assert calls[0][2]["output"] == oc.D15_RANGES and calls[1][2] == {"raw": True, "output": oc.D43_RANGES}
    assert calls[2][2] == {"raw": False, "output": oc.RANGES} and calls[3][2] == {"raw": True, "output": oc.RANGES}
    assert [str(a) for a in calls[2][1]] == ["x", "s", "p"]
    assert [str(value) for value in calls[4][1:]] == ["f.csv", str(oc.RANGES)]
    assert [str(value) for value in calls[5][1:]] == ["f.csv", "r.json"]


def test_every_stored_range_follows_from_its_stored_statistics_over_the_same_28_views():
    files = [path for path in (oc.D15_RANGES, oc.D43_RANGES, oc.RANGES) if path.exists()]
    views = None
    for path in files:
        record = stored(path)
        assert len(record["external"]) == 28 and record["factors"] == [oc.LOW, oc.HIGH]
        assert oc.ranges_from(list(record["external"].values())) == record["ranges"]
        views = sorted(record["external"]) if views is None else views
        assert sorted(record["external"]) == views
    assert od.expected_views() == views


def test_ranges_and_ranges_d43_rewrite_their_files_byte_for_byte(tmp_path):
    """Given the statistics their files hold, `ranges` and `ranges-d43` write exactly those files again."""
    for path, writer, source in ((oc.D15_RANGES, oc.write_ranges, "external_predictions"),
                                 (oc.D43_RANGES, oc.write_d43_ranges, "d43_predictions")):
        values = stored(path)["external"]
        patch = Patch()
        try:
            patch.setattr(oc, source, lambda *a, **k: {name: name for name in values})
            patch.setattr(oc, "statistics", lambda codes: values[codes])
            output = tmp_path / path.name
            options = {"d15_cache": tmp_path} if writer is oc.write_ranges else {}
            writer(tmp_path, tmp_path, tmp_path, raw=True, output=output, **options)
        finally:
            patch.undo()
        assert output.read_bytes() == path.read_bytes()
        try:
            writer(tmp_path, tmp_path, tmp_path, raw=True, output=output, **options)
            raise AssertionError("a ranges file was written twice")
        except FileExistsError:
            pass


def records(tmp_path, **changes):
    identity = {"model": "stack", "steps": 8100, "networks": {"0": {"current": {"temporal_models": "abc"}}},
                "config": {"direction": True, "mirror": "y"}}
    confirmation = {"passed": True, "stack": "8100", "owner_cost": 14, "identity": identity}
    development = {"amendment_choice": {"stack": "8100", "owner_cost": 14}, "decision": {"selected_cost": 14},
                   "identity": {"stack": json.loads(json.dumps(identity))}, "costs": [8, 10, 12, 14],
                   "entries": {"stack": [{"owner_cost": cost, "composite": cost / 20} for cost in (8, 10, 12, 14)]}}
    for name, (record, key, value) in changes.items():
        target = confirmation if record == "confirmation" else development
        *path, last = key
        for step in path:
            target = target[step]
        target[last] = value
    first, second = tmp_path / "confirmation.json", tmp_path / "development_t.json"
    first.write_text(json.dumps(confirmation))
    second.write_text(json.dumps(development))
    return first, second, identity


def test_confirmed_needs_the_8100_step_stack_at_cost_14_in_both_records(tmp_path):
    first, second, identity = records(tmp_path)
    found, development = od.confirmed(first, second)
    assert found == identity and od.stored_row(development) == {"owner_cost": 14, "composite": .7}
    for change in ({"a": ("confirmation", ("owner_cost",), 12)}, {"a": ("confirmation", ("stack",), "5400")},
                   {"a": ("confirmation", ("passed",), False)}, {"a": ("development", ("decision", "selected_cost"), 12)},
                   {"a": ("development", ("amendment_choice", "stack"), "5400")},
                   {"a": ("development", ("identity", "stack", "networks"), {})},
                   {"a": ("confirmation", ("identity", "steps"), 5400), "b": ("development", ("identity", "stack", "steps"), 5400)}):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        try:
            od.confirmed(*records(folder, **change)[:2])
            raise AssertionError(f"accepted {change}")
        except ValueError:
            pass
    _, development = od.confirmed(first, second)
    development["entries"]["stack"][3]["owner_cost"] = 12
    try:
        od.stored_row(development)
        raise AssertionError("a row at another cost was taken")
    except ValueError:
        pass


def test_json_digest_is_stack_trials():
    value = {"b": [1, 2.5, None], "a": {"z": True, "y": "text"}}
    assert od.json_digest(value) == st.json_digest(value)


def half(source, match, view, index, length, seed):
    rng = np.random.default_rng(seed)
    signals = (rng.uniform(.05, .95, length), rng.uniform(.05, .95, length), rng.uniform(0, .3, length),
               rng.uniform(0, .3, length))
    labels = rng.integers(0, 3, length).astype(np.int8)
    return diagnosis.HalfSignals(source, match, view, index, labels, signals, labels)


def test_view_predictions_decode_both_halves_at_the_cost_with_an_x_between():
    recipe = ht.recipe_at(14)
    halves = {"development": [half("dfl", "WPY", "clean", 0, 400, 1), half("dfl", "WPY", "clean", 1, 300, 2)],
              "metrica": [half("metrica", "1", "observed", 0, 250, 3), half("metrica", "1", "observed", 1, 260, 4)],
              "d24": [half("metrica", "1", "masked", 0, 250, 5), half("metrica", "1", "masked", 1, 260, 6)],
              "guards": [half("dfl", "WPY", "long_hidden", 0, 100, 7)]}
    found = od.view_predictions(halves, recipe)
    assert sorted(found) == ["dfl/WPY/clean", "metrica/1/masked", "metrica/1/observed"]   # the guards are not C2's
    parts = halves["development"]
    expected = np.r_[diagnosis.decode(parts[0].signals, recipe), pm.X, diagnosis.decode(parts[1].signals, recipe)]
    assert np.array_equal(found["dfl/WPY/clean"], expected.astype(np.int8))
    halves["d24"] = halves["metrica"]
    try:
        od.view_predictions(halves, recipe)
        raise AssertionError("a view counted twice")
    except ValueError:
        pass


def test_cached_halves_load_only_under_their_development_identity(tmp_path):
    from dfl_submission import hidden_trial as h49, metrica_sample as ms, metrica_views as mv, vendor_trial
    identity = {"model": "stack", "steps": 8100, "networks": {"0": "abc"}}
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "manifest.json").write_text("{}")
    cache = tmp_path / "cache"
    seen = {}
    patch = Patch()
    try:
        patch.setattr(vendor_trial, "development", lambda: [1, 2])
        patch.setattr(vendor_trial, "inputs", lambda parsed, games: {"games": games})
        patch.setattr(diagnosis, "teacher_identity", lambda: "teacher")
        patch.setattr(h49, "guard_identity", lambda models, raw: {"models": models, "raw": raw})

        def metrica(reference, cache=None, model=None, signals_of=None, variant=None):
            seen["metrica"] = (reference, cache, model, variant)
            try:
                signals_of(None)
                raise AssertionError("a forecast was allowed")
            except ValueError:
                pass
            return None, ["metrica halves"]

        def views(cache=None, signals_of=None, variant=None):
            seen["d24"] = (cache, variant)
            return ["d24 halves"]

        patch.setattr(ms, "reference_halves", metrica)
        patch.setattr(mv, "view_halves", views)
        development_key = dict(identity, teacher_sha256="teacher", split="development", dfl_raw=True,
                               skill_manifest_sha256=od.digest(skill / "manifest.json"), pff={"games": [1, 2]})
        guard_key = dict(identity, guard={"models": {"model": "stack"}, "raw": True})
        made = {"development": [half("dfl", "WPY", "clean", 0, 50, 1)], "guards": [half("dfl", "WPY", "long_hidden", 0, 40, 2)]}
        try:
            od.cached_halves(tmp_path, skill, tmp_path, identity, raw=True, cache=cache)
            raise AssertionError("loaded without caches")
        except ValueError:
            pass
        diagnosis.collect(cache / "development", development_key, lambda: made["development"])
        diagnosis.collect(cache / "guards", guard_key, lambda: made["guards"])
        for part in ("metrica", "d24"):
            (cache / part).mkdir()
            (cache / part / "manifest.json").write_text("{}")
        loaded = od.cached_halves(tmp_path, skill, tmp_path, identity, raw=True, cache=cache)
        assert loaded["metrica"] == ["metrica halves"] and loaded["d24"] == ["d24 halves"]
        for part in ("development", "guards"):
            (got,), (want,) = loaded[part], made[part]
            assert got.key == want.key and all(np.array_equal(a, b) for a, b in zip(got.signals, want.signals, strict=True))
        tag = f"D48 stack {st.json_digest(identity)[:16]}"
        assert seen["metrica"] == ("D15", cache / "metrica", "D48", tag) and seen["d24"] == (cache / "d24", tag)
        for other in (dict(identity, steps=5400), dict(identity, networks={"0": "def"})):
            try:
                od.cached_halves(tmp_path, skill, tmp_path, other, raw=True, cache=cache)
                raise AssertionError("loaded under another identity")
            except ValueError:
                pass
        try:
            od.cached_halves(tmp_path, skill, tmp_path, identity, raw=False, cache=cache)
            raise AssertionError("loaded with other inputs")
        except ValueError:
            pass
    finally:
        patch.undo()


def test_write_d48_ranges_needs_an_exact_reproduction_and_writes_once(tmp_path):
    first, second, identity = records(tmp_path)
    cache = tmp_path / "cache"
    for part in od.PARTS:
        (cache / part).mkdir(parents=True)
        (cache / part / "manifest.json").write_text(json.dumps({"part": part}))
    rng = np.random.default_rng(9)
    predictions = {}
    for name in od.expected_views():
        codes = rng.choice([pm.H, pm.A, pm.D], size=6000, p=[.35, .35, .3]).astype(np.int8)
        codes = np.repeat(codes[::40], 40)[:6000]
        codes[3000] = pm.X
        predictions[name] = codes
    row = {"owner_cost": 14, "composite": .73, "guards": {"mean": .72}, "selection": .725}
    outputs = {}
    patch = Patch()
    try:
        patch.setattr(od, "cached_halves", lambda *a, **k: {part: [] for part in od.PARTS})
        patch.setattr(od, "stored_row", lambda development: row)
        patch.setattr(od, "view_predictions", lambda halves, recipe: outputs.setdefault("recipe", recipe) and predictions)
        output = tmp_path / "ranges.json"
        patch.setattr(od, "recomputed_row", lambda halves, parsed: dict(row, composite=.7300001))
        try:
            od.write_d48_ranges(tmp_path, tmp_path, tmp_path, raw=True, output=output, cache=cache, confirmation=first,
                                development=second)
            raise AssertionError("written without an exact reproduction")
        except ValueError:
            assert not output.exists() and "recipe" not in outputs
        patch.setattr(od, "recomputed_row", lambda halves, parsed: json.loads(json.dumps(row)))
        ranges = od.write_d48_ranges(tmp_path, tmp_path, tmp_path, raw=True, output=output, cache=cache,
                                     confirmation=first, development=second)
        assert outputs["recipe"] == ht.recipe_at(14)
        try:
            od.write_d48_ranges(tmp_path, tmp_path, tmp_path, raw=True, output=output, cache=cache, confirmation=first,
                                development=second)
            raise AssertionError("written twice")
        except FileExistsError:
            pass
        missing = dict(predictions)
        missing.pop(sorted(missing)[0])
        patch.setattr(od, "view_predictions", lambda halves, recipe: missing)
        try:
            od.write_d48_ranges(tmp_path, tmp_path, tmp_path, raw=True, output=tmp_path / "other.json", cache=cache,
                                confirmation=first, development=second)
            raise AssertionError("written from 27 views")
        except ValueError:
            assert not (tmp_path / "other.json").exists()
    finally:
        patch.undo()
    record = stored(output)
    values = {name: oc.statistics(codes) for name, codes in predictions.items()}
    assert record["external"] == json.loads(json.dumps(values)) and record["ranges"] == ranges == oc.ranges_from(list(values.values()))
    assert record["protocol"] == "DECISIONS.md C2 (amendment for D48)" and record["factors"] == [.5, 1.5]
    assert record["owner_cost"] == 14 and record["source"]["model_identity_sha256"] == st.json_digest(identity)
    assert record["source"]["confirmation_sha256"] == od.digest(first) and record["source"]["development_sha256"] == od.digest(second)
    assert record["source"]["caches"] == {part: od.digest(cache / part / "manifest.json") for part in od.PARTS}
    path = tmp_path / "r.json"
    path.write_text(output.read_text())
    labels = predictions[sorted(predictions)[0]]
    csv = tmp_path / "file.csv"
    csv.write_text("frame,label\n" + "".join(f"{i},{pm.LABELS[c]}\n" for i, c in enumerate(labels)))
    assert oc.check(csv, path) is True                         # an external view's own labels pass its ranges
