"""C2 re-declared for D48 (DECISIONS.md C2, "Amendment for D48, 2026-11-13"): ranges from the confirmed stack.

The predictions are the confirmed stack's (`stack_recipe/confirmation.json`: 8,100 steps, ownership cost
14) on C2's 28 external match views, decoded from the cached decoder inputs its development record
(`stack_recipe/development_t.json`) was scored from (`data/d48/t/cache/stack`). The caches are only read:
each must carry the identity development gave it (the confirmed model identity, the same inputs), and
nothing is forecast again. Before anything is written, that record's cost-14 row is recomputed from the
same caches and must match exactly. The statistics, the views and the factors are C2's
(`output_check.statistics`, `output_check.ranges_from`). Run through `python -m dfl_submission.output_check
ranges-d48`, with `MTAH_DATA_DIR` set to a nonexistent path; the file is written once.
"""

import hashlib
import json
from pathlib import Path

from dfl_submission import output_check as oc
from dfl_submission.pff import digest


ROOT = oc.PACKAGE.parent
RECIPE = oc.PACKAGE / "stack_recipe"
CONFIRMATION = RECIPE / "confirmation.json"
DEVELOPMENT = RECIPE / "development_t.json"
CACHE = ROOT / "data" / "d48" / "t" / "cache" / "stack"
MODEL = "stack"
COST = 14                      # the amendment's ownership cost; checked against both records
STEPS = 8100
CANDIDATE = "D48 stack (L + C + three seeds + y mirror, 8,100 steps), cost 14"
PARTS = ("development", "metrica", "d24", "guards")


def json_digest(value):
    """`stack_trial.json_digest` (the digest in the variant tag of the stack's Metrica and D24 caches)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def roundtrip(value):
    return json.loads(json.dumps(value))


def refuse():
    raise ValueError("A cache of the stack's decoder inputs is missing; C2's D48 ranges are decoded from the caches "
                     "development was scored from, never forecast again")


def refuse_forecast(*_):
    refuse()


def confirmed(confirmation=CONFIRMATION, development=DEVELOPMENT):
    """(model identity, development record) of the confirmed stack; stops unless both records name it at cost 14."""
    confirmation, development = json.loads(Path(confirmation).read_text()), json.loads(Path(development).read_text())
    if confirmation.get("passed") is not True or confirmation.get("stack") != str(STEPS):
        raise ValueError("D48's confirmation record does not show the 8,100-step stack passing")
    choice = development.get("amendment_choice", {})
    if (confirmation.get("owner_cost") != COST or development["decision"]["selected_cost"] != COST
            or choice.get("stack") != str(STEPS) or choice.get("owner_cost") != COST):
        raise ValueError(f"The records do not name ownership cost {COST} for the 8,100-step stack")
    identity = confirmation["identity"]
    if development["identity"][MODEL] != identity or identity.get("steps") != STEPS or identity.get("model") != MODEL:
        raise ValueError("The development record's stack is not the confirmed stack")
    return identity, development


def cached_halves(dfl, skill, parsed, identity, *, raw, cache=CACHE):
    """{part: halves} of the stack from its development caches, each loaded only under its development identity."""
    from dfl_submission import diagnosis, hidden_trial as h49, metrica_sample as ms, metrica_views as mv, vendor_trial
    games = vendor_trial.development()
    key = dict(identity, teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=raw,
               skill_manifest_sha256=digest(Path(skill) / "manifest.json"), pff=vendor_trial.inputs(parsed, games))
    tag = f"D48 {MODEL} {json_digest(identity)[:16]}"
    for part in PARTS:
        if not (Path(cache) / part / "manifest.json").exists():
            refuse()
    development = diagnosis.collect(Path(cache) / "development", key, refuse)
    _, metrica = ms.reference_halves("D15", cache=Path(cache) / "metrica", model="D48", signals_of=refuse_forecast,
                                     variant=tag)
    d24 = mv.view_halves(cache=Path(cache) / "d24", signals_of=refuse_forecast, variant=tag)
    guards = diagnosis.collect(Path(cache) / "guards", dict(identity, guard=h49.guard_identity({"model": MODEL}, raw=raw)),
                               refuse)
    return {"development": development, "metrica": metrica, "d24": d24, "guards": guards}


def stored_row(development, cost=COST):
    """The development record's row of the stack at `cost`."""
    row = development["entries"][MODEL][list(development["costs"]).index(cost)]
    if row["owner_cost"] != cost:
        raise ValueError(f"The development record's stack row is not at cost {cost}")
    return row


def recomputed_row(halves, parsed, cost=COST):
    """The stack's development row at `cost`, recomputed from its cached halves exactly as `stack_trial.develop` wrote it."""
    from dfl_submission import hidden_trial as h49, hyper_trial as ht, metrica_sample as ms, vendor_trial
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    visibility = h49.pff_visibility(parsed, vendor_trial.development())
    item = h49.entry(halves["development"], halves["metrica"], halves["d24"], halves["guards"], ht.recipe_at(cost),
                     truths, visibility)
    return roundtrip(dict(h49.compact(item), selection=(item["composite"] + item["guards"]["mean"]) / 2))


def view_predictions(halves, recipe):
    """{view name: labels over both halves} on C2's views: development (DFL WPY views, SkillCorner, PFF), Metrica, D24."""
    from dfl_submission import diagnosis, stoppage_trial
    predict = stoppage_trial.predictor(recipe)
    found = {}
    for part in ("development", "metrica", "d24"):
        for key, (_, prediction, _) in diagnosis.matches(halves[part], predict).items():
            name = "/".join(key)
            if name in found:
                raise ValueError(f"View {name} appears twice")
            found[name] = prediction
    return found


def expected_views(path=oc.D43_RANGES):
    """C2's 28 view names, as the D43 ranges file (and D15's) lists them."""
    return sorted(json.loads(Path(path).read_text())["external"])


def write_d48_ranges(dfl, skill, parsed, *, raw, output=oc.RANGES, cache=CACHE, confirmation=CONFIRMATION,
                     development=DEVELOPMENT):
    from dfl_submission import hyper_trial as ht
    from dfl_submission.external_only import require_external_only
    require_external_only()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; C2's D48 ranges are written once")
    identity, record = confirmed(confirmation, development)
    halves = cached_halves(dfl, skill, parsed, identity, raw=raw, cache=cache)
    stored, recomputed = stored_row(record), recomputed_row(halves, parsed)
    if recomputed != stored:
        differing = sorted(key for key in set(stored) | set(recomputed) if stored.get(key) != recomputed.get(key))
        raise ValueError(f"The cached inputs do not reproduce development_t.json's cost-{COST} row ({differing}); "
                         "nothing was written")
    print(f"C2 (D48): the stack's cached inputs reproduce development_t.json's cost-{COST} row exactly (selection "
          f"{recomputed['selection']:.5f}, composite {recomputed['composite']:.5f}, guard mean "
          f"{recomputed['guards']['mean']:.4f})", flush=True)
    predictions = view_predictions(halves, ht.recipe_at(COST))
    if sorted(predictions) != expected_views():
        raise ValueError(f"C2 names 28 external match views; found {len(predictions)} that differ from them")
    values = {name: oc.statistics(codes) for name, codes in sorted(predictions.items())}
    ranges = oc.ranges_from(list(values.values()))
    output.write_text(json.dumps({
        "protocol": oc.PROTOCOL + " (amendment for D48)", "candidate": CANDIDATE, "factors": [oc.LOW, oc.HIGH],
        "ranges": ranges, "external": values, "owner_cost": COST,
        "source": {"confirmation_sha256": digest(Path(confirmation)), "development_sha256": digest(Path(development)),
                   "model_identity_sha256": json_digest(identity),
                   "caches": {part: digest(Path(cache) / part / "manifest.json") for part in PARTS},
                   "reproduced_row": {"owner_cost": COST, "selection": recomputed["selection"],
                                      "composite": recomputed["composite"], "guard_mean": recomputed["guards"]["mean"]}},
    }, indent=2, sort_keys=True) + "\n")
    for name in oc.STATISTICS:
        print(f"C2 (D48) {name}: external {min(v[name] for v in values.values()):.4f} to "
              f"{max(v[name] for v in values.values()):.4f}; range {ranges[name][0]:.4f} to {ranges[name][1]:.4f}")
    return ranges
