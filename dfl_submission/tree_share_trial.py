"""D59: the trees' share where the ball is unseen (H), trees refitted on D49's views (R), and both (RH) (DECISIONS.md D59).

Commands (each with `MTAH_DATA_DIR` set to a nonexistent path):
- `refit --external DIR`: pipeline check 2, then R. For each set, both heads refitted exactly as D3 fitted them
  (`touch_trial.fit_trees`: `long_context.fit_head` with `TREE_PARAMS`, the same rows, weights and seeds, as D13 and
  D16 verified) on the sequences D3's trees were fitted on, which must reproduce D3's trees byte for byte; then on
  D49's prepared sequences (D42's 48 records per set and D49's 32 long-hidden views). Trees to `data/d59/trees/`;
  `stack_recipe/d59_refit.json` written once.
- `members --dfl DIR --skill DIR --parsed DIR`: one pass of the adopted stack over every scored half, keeping its
  members (`tree_share.MemberForecaster`); cached in `data/d59/members/`.
- `score --dfl DIR --skill DIR --parsed DIR`: pipeline check 1, then the reference and every candidate at each cost;
  D59's rule; `stack_recipe/d59_scores.json` and `d59_decision.json` written once.
- `score-amended --dfl DIR --skill DIR --parsed DIR`: D59's amendment of 18:18 EST adds HT(w) and RHT(w): the trees'
  share of the first home blend is w on frames whose nearest ball sighting, earlier or later in the half, is more
  than 3 s (75 frames) away on the same 25 Hz grid as H's visibility, and 0.5 elsewhere; a half with no sighting is
  farther than 3 s everywhere. The amendment reached this session after `d59_scores.json` and `d59_decision.json`
  were written (19:12 EST), so those records cover the five original candidates; this command scores all nine from
  the same cached members (the five checked against `d59_scores.json`), applies the unchanged rule to all nine, and
  writes `stack_recipe/d59_scores_amended.json` and `d59_decision_amended.json` once.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import stack_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = st.RECIPE
SCRATCH = ROOT / "data" / "d59"
TREES = SCRATCH / "trees"
PROTOCOL = "DECISIONS.md D59"
GAP = 75                               # frames at 25 Hz: 3 s (D59's amendment of 18:18 EST)
TIMED = {"HT(0.25)": ("d3", .25), "HT(0)": ("d3", 0.0), "RHT(0.25)": ("r", .25), "RHT(0)": ("r", 0.0)}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


# -- R: the refit ------------------------------------------------------------------------------------

def original_folder(model_set, external):
    """The prepared sequences D3's trees were fitted on (D42's stored records), checked against the published manifest."""
    from dfl_submission import touch_trial
    source, _ = st.data_folders("d42", model_set, external)
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared folder D3's {model_set} trees were fitted on")
    return source


def d49_folder(model_set):
    """D49's prepared sequences, as its passed preparation record names them; their first records are D42's."""
    from dfl_submission import hidden_trial as h49
    source, _ = st.data_folders("d49", model_set, None)
    record = read(h49.RECIPE / f"prepare_{model_set}.json")
    if record["passed"] is not True or record["sequences_manifest"] != digest(source / "manifest.json"):
        raise ValueError(f"{source} is not the folder D49's preparation check passed")
    return source


def refit(external, *, scratch=TREES, output=None):
    """Pipeline check 2 on both sets, then both sets' trees refitted on D49's sequences; written once."""
    require_external_only()
    from dfl_submission import scale_trial, touch_trial
    output = RECIPE / "d59_refit.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D59's refit is recorded once")
    checks, refits, sources = {}, {}, {}
    for model_set in st.SETS:
        published = read(scale_trial.TREES[model_set] / "fit.json")["models"]
        original = original_folder(model_set, external)
        fitted = touch_trial.fit_trees(original, model_set, scratch / "check" / model_set)
        checks[model_set] = {name: fitted[name] == published[name] for name in published}
        print(f"D59 check 2, {model_set}: the refit reproduces D3's trees byte for byte: {checks[model_set]}", flush=True)
        if not all(checks[model_set].values()):
            break
        source = d49_folder(model_set)
        refits[model_set] = touch_trial.fit_trees(source, model_set, scratch / "refit" / model_set)
        sources[model_set] = {"original_manifest": digest(original / "manifest.json"),
                              "d49_manifest": digest(source / "manifest.json"),
                              "d49_records": len(read(source / "manifest.json")["records"]),
                              "original_records": len(read(original / "manifest.json")["records"]),
                              "rows": read(scratch / "refit" / model_set / "fit.json")["rows"],
                              "original_rows": read(scratch / "check" / model_set / "fit.json")["rows"],
                              "refit_fit_sha256": digest(scratch / "refit" / model_set / "fit.json")}
        print(f"D59 R, {model_set}: refitted on D49's sequences: {refits[model_set]}", flush=True)
    passed = bool(len(checks) == len(st.SETS) and all(all(value.values()) for value in checks.values()))
    write_new(output, {"protocol": PROTOCOL, "passed": passed, "check_reproduces_d3_trees": checks, "refits": refits,
                       "sources": sources, "published": {key: read(scale_trial.TREES[key] / "fit.json")["models"]
                                                         for key in st.SETS},
                       "code": {name: digest(PACKAGE / name) for name in ("touch_trial.py", "long_context.py",
                                                                          "aligned_trial.py", "tree_share_trial.py")},
                       "refitted_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    if not passed:
        raise SystemExit("D59 stops: the refit does not reproduce D3's trees")
    return passed


def refit_folders(record_path=None, scratch=TREES):
    """{set: folder} of R's trees, after checking the passed refit record and the trees' fit records."""
    record = read(RECIPE / "d59_refit.json" if record_path is None else record_path)
    if record["passed"] is not True:
        raise SystemExit("D59 stops: the refit record did not pass")
    folders = {key: scratch / "refit" / key for key in st.SETS}
    for key, folder in folders.items():
        if digest(folder / "fit.json") != record["sources"][key]["refit_fit_sha256"] or \
                read(folder / "fit.json")["models"] != record["refits"][key]:
            raise ValueError(f"R's {key} trees differ from the refit record")
    return folders


# -- the amendment: HT(w) and RHT(w) -------------------------------------------------------------------

def far_from_sighting(visible, limit=GAP):
    """Frames whose nearest seen frame (earlier or later, same half) is more than `limit` frames away; all frames if
    the half has no seen frame."""
    visible = np.asarray(visible, dtype=bool)
    frames = np.arange(len(visible))
    sightings = np.flatnonzero(visible)
    if not len(sightings):
        return np.ones(len(visible), dtype=bool)
    position = np.searchsorted(sightings, frames)
    after = np.where(position < len(sightings), sightings[np.minimum(position, len(sightings) - 1)] - frames, np.inf)
    before = np.where(position > 0, frames - sightings[np.maximum(position - 1, 0)], np.inf)
    return np.minimum(before, after) > limit


def timed_signals(members, tree_set, share, previous, recipe):
    """HT(w) / RHT(w): `tree_share.blend_h` with the share w on frames far from any sighting, 0.5 elsewhere."""
    from dfl_submission import tree_share as ts
    near = ~far_from_sighting(members["seen"])
    return tuple(ts.blend_h(ts.structure(members, tree_set), previous, recipe, near, share))


def timed_scored(parts, tree_set, share, previous, recipe):
    from dfl_submission import diagnosis
    result = {}
    for part, halves in parts.items():
        result[part] = []
        for half in halves:
            blended = timed_signals(half.members, tree_set, share, previous, recipe)
            result[part].append(diagnosis.HalfSignals(half.source, half.match, half.view, half.index, half.labels, blended,
                                                      diagnosis.decode(blended, recipe)))
    return result


def far_share(parts):
    """Per part: the share of frames that are unseen, and that are farther than 3 s from a sighting."""
    result = {}
    for part, halves in parts.items():
        frames = sum(len(half.members["seen"]) for half in halves)
        unseen = sum(int((~half.members["seen"]).sum()) for half in halves)
        far = sum(int(far_from_sighting(half.members["seen"]).sum()) for half in halves)
        result[part] = {"frames": frames, "unseen_share": unseen / frames, "far_share": far / frames}
    return result


def score_amended(dfl, skill, parsed, *, output=None):
    """All nine candidates of the amended grid from the cached members, the unchanged rule; written once."""
    require_external_only()
    from dfl_submission import competition_adaptation, situation_fit as sf, tree_share as ts
    output = RECIPE / "d59_scores_amended.json" if output is None else output
    decision_path = output.parent / "d59_decision_amended.json"
    for path in (output, decision_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; D59 records are written once")
    first = read(RECIPE / "d59_scores.json")
    if first["passed_checks"] is not True:
        raise SystemExit("D59 stops: the first scoring's pipeline checks did not pass")
    parts, _ = ts._member_pass(dfl, skill, parsed, cache=ts.MEMBERS)
    reference_model, _, frozen = competition_adaptation.reference()
    previous, recipe = reference_model[2]["recipe"], frozen["recipe"]
    bits = ts.check_against_stack(parts, previous, recipe)
    games, truths, visibility = sf.development_inputs(dfl, skill, parsed)
    tracking = {key: value for key, value in sf.situation_tracking(dfl).items() if key[0] in sf.VIEWS.values()}
    entries, items = {}, {}
    variants = {"reference": ts.REFERENCE} | ts.CANDIDATES
    for name, (tree_set, share) in variants.items():
        pairs = ts.entries_for(ts.scored(parts, tree_set, share, previous, recipe, recipe), ts.COSTS, truths, visibility,
                               tracking)
        entries[name], items[name] = [full for full, _ in pairs], [item for _, item in pairs]
    same_as_first = {name: canonical(items[name]) == canonical(first["entries"][name]) for name in variants}
    for name, (tree_set, share) in TIMED.items():
        pairs = ts.entries_for(timed_scored(parts, tree_set, share, previous, recipe), ts.COSTS, truths, visibility, tracking)
        entries[name], items[name] = [full for full, _ in pairs], [item for _, item in pairs]
        for item in items[name]:
            print(f"D59 {name} cost {item['owner_cost']}: selection {item['selection']:.5f}, composite {item['composite']:.5f}, "
                  f"guards {item['guards']['mean']:.4f}, PFF {item['groups']['pff']:.4f}, (d) {item['situation_d']:.4f}",
                  flush=True)
    at = [item["owner_cost"] for item in items["reference"]].index(ts.REFERENCE_COST)
    reference, reference_full = items["reference"][at], entries["reference"][at]
    passed = bool(all(same_as_first.values()) and all(part["bit_for_bit"] for part in bits.values()))
    order = ["H(0.25)", "H(0)", "HT(0.25)", "HT(0)", "R", "RH(0.25)", "RH(0)", "RHT(0.25)", "RHT(0)"]
    results, per_cost = {}, {}
    for name in order:
        pairs = list(zip(entries[name], items[name]))
        per_cost[name] = {str(item["owner_cost"]): ts.judge(item, full, reference, reference_full) for full, item in pairs}
        full, item = ts.best(pairs)
        results[name] = dict(ts.judge(item, full, reference, reference_full), selection=item["selection"])
    taken = ts.decision(results) if passed else None
    note = ("D59's amendment of 18:18 EST (HT, RHT) reached the coding session after d59_scores.json and d59_decision.json "
            "were written at 19:12 EST; those records hold the five original candidates. This record scores all nine from "
            "the same cached members, the five equal to the first records, and applies the unchanged rule to all nine.")
    write_new(output, canonical({
        "protocol": PROTOCOL, "note": note, "passed_checks": passed, "members_bit_for_bit": bits,
        "five_equal_to_d59_scores": same_as_first, "costs": list(ts.COSTS), "gap_frames": GAP,
        "reference": {"name": "the adopted stack", "owner_cost": ts.REFERENCE_COST, **reference},
        "variants": {name: {"trees": tree_set, "share": share, "frames": "unseen" if name in ts.CANDIDATES else
                            (None if name == "reference" else "more than 3 s from a sighting")}
                     for name, (tree_set, share) in (variants | TIMED).items()},
        "entries": {name: items[name] for name in ["reference", *order]},
        "frame_shares": far_share(parts), "d59_scores_sha256": digest(RECIPE / "d59_scores.json"),
        "member_caches": first["member_caches"], "tree_share_trial_sha256": digest(PACKAGE / "tree_share_trial.py"),
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")}))
    write_new(decision_path, canonical({
        "protocol": PROTOCOL, "note": note, "passed_checks": passed, "results": results, "per_cost": per_cost,
        "taken": taken, "first_decision_taken": read(RECIPE / "d59_decision.json")["taken"],
        "rule": read(RECIPE / "d59_decision.json")["rule"], "scores_sha256": digest(output),
        "decided_at": time.strftime("%Y-%m-%d %H:%M %Z")}))
    print(ts.summary({name: items[name] for name in ["reference", *order]}), flush=True)
    for name, result in results.items():
        changes = result["changes"]
        print(f"D59 {name}: cost {result['owner_cost']}, selection {changes['selection']:+.4f} (+{ts.SELECTION_GAIN}), PFF "
              f"{changes['pff']:+.4f} (+{ts.PFF_GAIN}), composite {changes['composite']:+.4f} (>= 0), worst group "
              f"{min(changes['groups'].values()):+.4f}, worst convention {min(changes['conventions'].values()):+.4f} "
              f"(>= -{ts.LOSS}) -> {'holds' if result['holds'] else 'does not hold'}", flush=True)
    print(f"D59 amended: checks passed {passed}; decision: {taken if taken else 'none holds; the adopted stack stands'}",
          flush=True)
    return taken


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("refit")
    one.add_argument("--external", type=Path, required=True)
    for action in ("members", "score", "score-amended"):
        part = sub.add_parser(action)
        for flag in ("--dfl", "--skill", "--parsed"):
            part.add_argument(flag, type=Path, required=True)
    args = parser.parse_args()
    require_external_only()
    if args.action == "refit":
        refit(args.external)
    elif args.action == "score-amended":
        score_amended(args.dfl, args.skill, args.parsed)
    else:
        from dfl_submission import tree_share
        (tree_share.member_pass if args.action == "members" else tree_share.score)(args.dfl, args.skill, args.parsed)


if __name__ == "__main__":
    main()
