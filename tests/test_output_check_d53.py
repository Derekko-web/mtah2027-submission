"""C2 for the D53 fallback (dfl_submission/output_check_d53.py), on synthetic inputs and the stored records.

No competition file and no external match is read: the cached halves and the predictions are synthetic; the
real D48 / D53 records are only read as JSON, and the D53 ranges file (once written) only from its own values.
"""

import inspect
import json
import sys

import numpy as np

from dfl_submission import hyper_trial as ht, output_check as oc, output_check_d48 as od, output_check_d53 as of
from dfl_submission import stack_trial as st
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


def refuses(call, error=ValueError):
    try:
        call()
    except error:
        return True
    return False


def test_the_d53_file_is_separate_and_check_keeps_the_d48_default():
    from dfl_submission import fallback_trial
    assert of.RANGES.name == "output_check_ranges_d53.json" and of.RANGES.parent == oc.PACKAGE
    assert oc.RANGES.name == "output_check_ranges_d48.json"                    # check's default is unchanged
    assert inspect.signature(oc.check).parameters["ranges"].default == oc.RANGES
    assert inspect.signature(of.write_d53_ranges).parameters["output"].default == of.RANGES
    assert of.CONFIG == fallback_trial.CONFIG and of.STEPS == fallback_trial.STEPS == 8100
    assert of.CACHE == fallback_trial.SCRATCH / "cache" / "stack" and of.PARTS == od.PARTS
    assert (of.CONFIRMATION.name, of.DEVELOPMENT.name, of.BUILD_IDENTITY.name) == (
        "confirmation_d53.json", "development_d53.json", "build_identity_d53.json")
    assert of.RANGES not in (oc.D15_RANGES, oc.D43_RANGES, oc.RANGES)


def test_ranges_d53_writes_the_d53_file_and_check_reads_d48_unless_told(tmp_path):
    calls = []
    patch = Patch()
    try:
        patch.setattr(of, "write_d53_ranges", lambda *a, **k: calls.append(("d53", a, k)))
        patch.setattr(od, "write_d48_ranges", lambda *a, **k: calls.append(("d48", a, k)))
        patch.setattr(oc, "check", lambda csv, ranges=oc.RANGES: calls.append(("check", csv, ranges)) or True)
        common = ["--dfl", "x", "--skill", "s", "--parsed", "p"]
        for argv in (["ranges-d53", *common, "--dfl-raw"], ["ranges-d53", *common], ["check", "f.csv"],
                     ["check", "f.csv", "--ranges", str(of.RANGES)]):
            patch.setattr(sys, "argv", ["output_check", *argv])
            try:
                oc.main()
            except SystemExit as stop:
                assert stop.code == 0
    finally:
        patch.undo()
    assert [call[0] for call in calls] == ["d53", "d53", "check", "check"]
    assert calls[0][2] == {"raw": True, "output": of.RANGES} and calls[1][2] == {"raw": False, "output": of.RANGES}
    assert [str(a) for a in calls[0][1]] == ["x", "s", "p"]
    assert [str(value) for value in calls[2][1:]] == ["f.csv", str(oc.RANGES)]
    assert [str(value) for value in calls[3][1:]] == ["f.csv", str(of.RANGES)]


def records(folder, **changes):
    identity = {"model": "stack", "steps": 8100, "variant": "d53", "config": dict(of.CONFIG),
                "networks": {"0": {"current": {"temporal_models": "abc"}}}}
    confirmation = {"passed": True, "protocol": "DECISIONS.md D53", "owner_cost": 14, "identity": identity}
    development = {"variant": "d53", "amendment_choice": {"gate_passed": True, "selected_cost": 14},
                   "decision": {"selected_cost": 14, "passed": True},
                   "identity": {"stack": json.loads(json.dumps(identity))}, "costs": [8, 10, 12, 14],
                   "entries": {"stack": [{"owner_cost": cost, "composite": cost / 20} for cost in (8, 10, 12, 14)]}}
    lock = {"passed": True, "protocol": "DECISIONS.md D53"}
    targets = {"confirmation": confirmation, "development": development, "lock": lock}
    for record, key, value in changes.values():
        target = targets[record]
        *path, last = key
        for step in path:
            target = target[step]
        target[last] = value
    first, second, third = folder / "confirmation_d53.json", folder / "development_d53.json", folder / "lock.json"
    first.write_text(json.dumps(confirmation))
    lock.setdefault("confirmation_sha256", of.digest(first))
    second.write_text(json.dumps(development))
    third.write_text(json.dumps(lock))
    return first, second, third, identity


def test_confirmed_needs_the_fallback_at_cost_14_in_its_records_and_its_build_lock(tmp_path):
    first, second, third, identity = records(tmp_path)
    found, development = of.confirmed(first, second, third)
    assert found == identity and of.stored_row(development) == {"owner_cost": 14, "composite": .7}
    for change in (("confirmation", ("owner_cost",), 12), ("confirmation", ("passed",), False),
                   ("confirmation", ("protocol",), "DECISIONS.md D48"), ("development", ("decision", "selected_cost"), 12),
                   ("development", ("decision", "passed"), False), ("development", ("amendment_choice", "gate_passed"), False),
                   ("development", ("amendment_choice", "selected_cost"), 12), ("development", ("variant",), "t"),
                   ("development", ("identity", "stack", "networks"), {}), ("lock", ("passed",), False),
                   ("lock", ("confirmation_sha256",), "0" * 64), ("lock", ("protocol",), "DECISIONS.md D48")):
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        assert refuses(lambda: of.confirmed(*records(folder, a=change)[:3])), change
    for key, value in (("steps", 5400), ("variant", "t"), ("model", "seed_0"),
                       ("config", dict(of.CONFIG, direction=True, mirror="y"))):     # the stack with C is not the fallback
        folder = tmp_path / str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        both = {"a": ("confirmation", ("identity", key), value), "b": ("development", ("identity", "stack", key), value)}
        assert refuses(lambda: of.confirmed(*records(folder, **both)[:3])), key


def test_the_real_records_name_the_fallback_and_not_the_stack_with_c():
    identity, development = of.confirmed()
    assert identity["config"]["direction"] is False and identity["variant"] == "d53"
    assert of.stored_row(development)["owner_cost"] == 14
    stack_identity, _ = od.confirmed()
    assert stack_identity != identity
    assert refuses(lambda: od.confirmed(of.CONFIRMATION, of.DEVELOPMENT))               # D48's writer refuses D53
    assert refuses(lambda: of.confirmed(od.CONFIRMATION, od.DEVELOPMENT, of.RECIPE / "build_identity.json"))
    assert refuses(lambda: of.confirmed(of.CONFIRMATION, of.DEVELOPMENT, of.RECIPE / "build_identity.json"))


def synthetic_predictions(seed):
    rng = np.random.default_rng(seed)
    predictions = {}
    for name in of.expected_views():
        codes = rng.choice([pm.H, pm.A, pm.D], size=6000, p=[.35, .35, .3]).astype(np.int8)
        codes = np.repeat(codes[::40], 40)[:6000]
        codes[3000] = pm.X
        predictions[name] = codes
    return predictions


def test_write_d53_ranges_needs_an_exact_reproduction_and_writes_once(tmp_path):
    first, second, third, identity = records(tmp_path)
    cache = tmp_path / "cache"
    for part in of.PARTS:
        (cache / part).mkdir(parents=True)
        (cache / part / "manifest.json").write_text(json.dumps({"part": part}))
    predictions = synthetic_predictions(11)
    row = {"owner_cost": 14, "composite": .72, "guards": {"mean": .71}, "selection": .715}
    seen = {}
    patch = Patch()
    try:
        def halves(dfl, skill, parsed, found, *, raw, cache):
            seen["halves"] = (found, raw, cache)
            return {part: [] for part in of.PARTS}

        patch.setattr(of, "cached_halves", halves)
        patch.setattr(of, "view_predictions", lambda halves, recipe: seen.setdefault("recipe", recipe) and predictions)
        output = tmp_path / "ranges.json"
        options = dict(raw=True, cache=cache, confirmation=first, development=second, build_identity=third)
        patch.setattr(of, "recomputed_row", lambda halves, parsed, cost: dict(row, composite=.7200001))
        patch.setattr(of, "stored_row", lambda development, cost: row)
        assert refuses(lambda: of.write_d53_ranges(tmp_path, tmp_path, tmp_path, output=output, **options))
        assert not output.exists() and "recipe" not in seen
        assert seen["halves"] == (identity, True, cache)
        patch.setattr(of, "recomputed_row", lambda halves, parsed, cost: json.loads(json.dumps(row)) if cost == 14 else {})
        ranges = of.write_d53_ranges(tmp_path, tmp_path, tmp_path, output=output, **options)
        assert seen["recipe"] == ht.recipe_at(14)
        assert refuses(lambda: of.write_d53_ranges(tmp_path, tmp_path, tmp_path, output=output, **options), FileExistsError)
        missing = dict(predictions)
        missing.pop(sorted(missing)[-1])
        patch.setattr(of, "view_predictions", lambda halves, recipe: missing)
        other = tmp_path / "other.json"
        assert refuses(lambda: of.write_d53_ranges(tmp_path, tmp_path, tmp_path, output=other, **options))
        assert not other.exists()
    finally:
        patch.undo()
    record = json.loads(output.read_text())
    values = {name: oc.statistics(codes) for name, codes in predictions.items()}
    assert record["external"] == json.loads(json.dumps(values)) and record["ranges"] == ranges == oc.ranges_from(list(values.values()))
    assert record["protocol"] == "DECISIONS.md C2 (amendment for D53)" and record["factors"] == [.5, 1.5]
    assert record["owner_cost"] == 14 and record["source"]["model_identity_sha256"] == st.json_digest(identity)
    source = record["source"]
    assert (source["confirmation_sha256"], source["development_sha256"], source["build_identity_sha256"]) == (
        of.digest(first), of.digest(second), of.digest(third))
    assert source["caches"] == {part: of.digest(cache / part / "manifest.json") for part in of.PARTS}
    labels = predictions[sorted(predictions)[-1]]
    csv = tmp_path / "file.csv"
    csv.write_text("frame,label\n" + "".join(f"{i},{pm.LABELS[c]}\n" for i, c in enumerate(labels)))
    assert oc.check(csv, output) is True                       # an external view's own labels pass its ranges


def test_the_stored_d53_ranges_follow_from_their_statistics_and_name_the_fallbacks_records():
    if not of.RANGES.exists():
        return
    record = json.loads(of.RANGES.read_text())
    assert sorted(record["external"]) == of.expected_views() and len(record["external"]) == 28
    assert record["factors"] == [oc.LOW, oc.HIGH] and oc.ranges_from(list(record["external"].values())) == record["ranges"]
    assert record["protocol"] == "DECISIONS.md C2 (amendment for D53)" and record["owner_cost"] == 14
    source = record["source"]
    assert source["confirmation_sha256"] == of.digest(of.CONFIRMATION)
    assert source["development_sha256"] == of.digest(of.DEVELOPMENT)
    assert source["build_identity_sha256"] == of.digest(of.BUILD_IDENTITY)
    identity, development = of.confirmed()
    assert source["model_identity_sha256"] == st.json_digest(identity)
    stored = of.stored_row(development)
    assert source["reproduced_row"] == {"owner_cost": 14, "selection": stored["selection"],
                                        "composite": stored["composite"], "guard_mean": stored["guards"]["mean"]}
    assert record["external"] != json.loads(oc.RANGES.read_text())["external"]    # not a copy of D48's file
