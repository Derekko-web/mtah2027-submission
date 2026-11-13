"""C2 for the D53 fallback (DECISIONS.md C2, "Amendment for D53, 2026-11-13"): ranges from the fallback.

The predictions are the D53 fallback's (`stack_recipe/confirmation_d53.json`: the stack without the attacking
frame, mirrors in x and y, three seeds, 8,100 steps, ownership cost 14) on C2's 28 external match views, decoded
from the cached decoder inputs its development record (`stack_recipe/development_d53.json`) was scored from
(`data/d53/cache/stack`). As for D48 (`output_check_d48`), the caches are only read, each under the identity
development gave it, nothing is forecast again, and that record's cost-14 row is recomputed from the same caches
and must match exactly before anything is written. The confirmation record must also be the one the fallback's
build lock (`stack_recipe/build_identity_d53.json`) names. The statistics, the views and the factors are C2's.
Run through `python -m dfl_submission.output_check ranges-d53`, with `MTAH_DATA_DIR` set to a nonexistent path;
the file is written once and is not `check`'s default (`check --ranges` names it if the fallback ships).
"""

import json
from pathlib import Path

from dfl_submission import output_check as oc, output_check_d48 as od
from dfl_submission.pff import digest


RECIPE = od.RECIPE
CONFIRMATION = RECIPE / "confirmation_d53.json"
DEVELOPMENT = RECIPE / "development_d53.json"
BUILD_IDENTITY = RECIPE / "build_identity_d53.json"
CACHE = od.ROOT / "data" / "d53" / "cache" / "stack"
RANGES = oc.PACKAGE / "output_check_ranges_d53.json"
PROTOCOL = "DECISIONS.md D53"
MODEL = od.MODEL
COST = 14                      # the amendment's ownership cost; checked against both records
STEPS = 8100
VARIANT = "d53"
CONFIG = {"aux": False, "direction": False, "mirror": "xy", "noise": False, "seed_offset": 0}   # fallback_trial.CONFIG
CANDIDATE = "D53 fallback (the stack without C: L + mirrors in x and y + three seeds, 8,100 steps), cost 14"
PARTS = od.PARTS

# The cache reading, row recomputation and decoding are D48's (the fallback was scored through the same
# `stack_trial.develop`, so its caches carry the same keys and variant tags).
cached_halves = od.cached_halves
stored_row = od.stored_row
recomputed_row = od.recomputed_row
view_predictions = od.view_predictions
expected_views = od.expected_views
json_digest = od.json_digest


def confirmed(confirmation=CONFIRMATION, development=DEVELOPMENT, build_identity=BUILD_IDENTITY):
    """(model identity, development record) of the fallback; stops unless the records name it at cost 14."""
    path = Path(confirmation)
    confirmation, development = json.loads(path.read_text()), json.loads(Path(development).read_text())
    lock = json.loads(Path(build_identity).read_text())
    if confirmation.get("passed") is not True or confirmation.get("protocol") != PROTOCOL:
        raise ValueError("D53's confirmation record does not show the fallback passing")
    choice = development.get("amendment_choice", {})
    decision = development["decision"]
    if (confirmation.get("owner_cost") != COST or decision["selected_cost"] != COST or decision.get("passed") is not True
            or choice.get("gate_passed") is not True or choice.get("selected_cost") != COST):
        raise ValueError(f"The records do not name ownership cost {COST} for the fallback")
    identity = confirmation["identity"]
    if (development.get("variant") != VARIANT or development["identity"][MODEL] != identity
            or identity.get("steps") != STEPS or identity.get("model") != MODEL or identity.get("variant") != VARIANT
            or identity.get("config") != CONFIG):
        raise ValueError("The development record's stack is not the confirmed fallback")
    if lock.get("passed") is not True or lock.get("protocol") != PROTOCOL or lock.get("confirmation_sha256") != digest(path):
        raise ValueError("The fallback's build lock does not name this confirmation record")
    return identity, development


def write_d53_ranges(dfl, skill, parsed, *, raw, output=RANGES, cache=CACHE, confirmation=CONFIRMATION,
                     development=DEVELOPMENT, build_identity=BUILD_IDENTITY):
    from dfl_submission import hyper_trial as ht
    from dfl_submission.external_only import require_external_only
    require_external_only()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; C2's D53 ranges are written once")
    identity, record = confirmed(confirmation, development, build_identity)
    halves = cached_halves(dfl, skill, parsed, identity, raw=raw, cache=cache)
    stored, recomputed = stored_row(record, COST), recomputed_row(halves, parsed, COST)
    if recomputed != stored:
        differing = sorted(key for key in set(stored) | set(recomputed) if stored.get(key) != recomputed.get(key))
        raise ValueError(f"The cached inputs do not reproduce development_d53.json's cost-{COST} row ({differing}); "
                         "nothing was written")
    print(f"C2 (D53): the fallback's cached inputs reproduce development_d53.json's cost-{COST} row exactly (selection "
          f"{recomputed['selection']:.5f}, composite {recomputed['composite']:.5f}, guard mean "
          f"{recomputed['guards']['mean']:.4f})", flush=True)
    predictions = view_predictions(halves, ht.recipe_at(COST))
    if sorted(predictions) != expected_views():
        raise ValueError(f"C2 names 28 external match views; found {len(predictions)} that differ from them")
    values = {name: oc.statistics(codes) for name, codes in sorted(predictions.items())}
    ranges = oc.ranges_from(list(values.values()))
    output.write_text(json.dumps({
        "protocol": oc.PROTOCOL + " (amendment for D53)", "candidate": CANDIDATE, "factors": [oc.LOW, oc.HIGH],
        "ranges": ranges, "external": values, "owner_cost": COST,
        "source": {"confirmation_sha256": digest(Path(confirmation)), "development_sha256": digest(Path(development)),
                   "build_identity_sha256": digest(Path(build_identity)),
                   "model_identity_sha256": json_digest(identity),
                   "caches": {part: digest(Path(cache) / part / "manifest.json") for part in PARTS},
                   "reproduced_row": {"owner_cost": COST, "selection": recomputed["selection"],
                                      "composite": recomputed["composite"], "guard_mean": recomputed["guards"]["mean"]}},
    }, indent=2, sort_keys=True) + "\n")
    for name in oc.STATISTICS:
        print(f"C2 (D53) {name}: external {min(v[name] for v in values.values()):.4f} to "
              f"{max(v[name] for v in values.values()):.4f}; range {ranges[name][0]:.4f} to {ranges[name][1]:.4f}")
    return ranges
