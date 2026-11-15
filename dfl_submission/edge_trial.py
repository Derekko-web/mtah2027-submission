"""D63 development: a lower ownership cost next to stoppages (DECISIONS.md D63).

`develop` (with `MTAH_DATA_DIR` set to a nonexistent path) decodes the adopted stack (D48: the 8,100-step stack of
`stack_recipe/development_t.json`, three seeds per slot, ownership cost 14) from the decoder inputs its development
record was scored from (`data/d48/t/cache/stack`: `development`, `metrica`, `d24`, `guards`), as D58 and D62 did.
Nothing is forecast and nothing is trained.

How a candidate is scored. The standing scorer (`diagnosis_stack.recomputed_row`, D58's reproduction of
`stack_trial.develop`'s row, plus D33's conventions through `refine_trial.standard`) decodes every half through
`diagnosis.decode`. While a candidate is scored, `decoding_with` puts D63's two-pass decode
(`edge_cost.two_pass` at the candidate's factors) in place of `diagnosis.decode` and makes `joint_decoder.decode`
raise, so a decode that bypassed the substitution would stop the run instead of scoring the standing decode silently;
both are restored afterwards. No file of an existing module changes. Every number of the row (D47's eight groups,
the composite, D49's guard views, D33's conventions, PFF, Metrica, the D24 views) therefore comes from the same code
as the standing row; D19's control-rule targets are among the targets `metrica_sample.score` already scores for the
Metrica games and the D24 views, so condition (f) reads them from the same row.

Then, in this order, stopping without a record if a check fails:
1. modules: no tracked file under `dfl_submission/`, `evaluation/`, `tests/` or `release/` differs from HEAD (git);
2. restated pass 1: `edge_cost.pass_one` equals `joint_decoder.decode` label for label on every cached half, and
   pass 2 at rho_pre = rho_post = 1 equals pass 1 on every cached half;
3. pipeline check: with both factors at 1 the scorer, through the substitution, reproduces the cost-14 row of
   `development_t.json` exactly (every key);
4. the reported-item code at both factors 1 reproduces D58's length recall (`diagnosis_recipe/d58.json`) for every
   one of D58's twelve groups;
5. the fifteen candidates (rho_pre, rho_post in {1, 0.75, 0.5, 0.25}, beside the standing decode), each scored once;
6. D63's reading, mechanically: conditions (a)-(f), the selection, the gate.

Records, each written once: `stack_recipe/d63_development.json` and `stack_recipe/d63_decision.json`.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from dfl_submission import edge_cost as ec
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stack_recipe"
CACHE = ROOT / "data" / "d48" / "t" / "cache"
D58_RECORD = PACKAGE / "diagnosis_recipe" / "d58.json"
PROTOCOL = "DECISIONS.md D63"
COST = 14
TARGET_SECONDS = 3.0
PARTS = ("development", "metrica", "d24", "guards")
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica", "d24")
CONVENTIONS = ("aba1.75", "aba3", "aba4", "aba4.5")
BOUNDED = ("aba1.75", "aba4", "aba4.5")             # "N = 1.75, 4 and 4.5"
REPORT_GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica",
                 "d24_handled", "d24_masked", "d24_team_swap", "guard_dfl", "guard_metrica")   # D58's twelve
D24_VIEWS = ("handled", "masked", "team_swap")
SHORT_BINS = ("0-1.75s", "1.75-3s")
GROUP_LOSS, GUARD_LOSS, CONVENTION_LOSS, PFF_LOSS, CONTROL_LOSS = .01, .005, .01, .005, .01
SELECTION_GAIN, METRICA_D24_GAIN = .008, .008
MATCH_IOU = .5
GUARDED = ("dfl_submission", "evaluation", "tests", "release")
CODE = ("edge_cost.py", "edge_trial.py", "joint_decoder.py", "diagnosis.py", "diagnosis_stack.py", "diagnosis_d43.py",
        "hidden_trial.py", "aligned_trial.py", "objective_trial.py", "metrica_sample.py", "metrica_views.py",
        "refine_trial.py", "decoder_trial.py", "decoder.py", "adaptive_decoder.py")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def key_of(rho_pre, rho_post):
    return f"{float(rho_pre):g}/{float(rho_post):g}"


def candidates():
    """D63's fifteen candidates beside the standing decode (1, 1), rho_pre first."""
    return [(p, q) for p in ec.FACTORS for q in ec.FACTORS if (p, q) != (1.0, 1.0)]


# -- the substitution ---------------------------------------------------------------------------------

@contextmanager
def decoding_with(rho_pre, rho_post, recipe):
    """`diagnosis.decode` replaced by D63's two-pass decode at these factors; `joint_decoder.decode` raises."""
    from dfl_submission import diagnosis, joint_decoder
    original, standing = diagnosis.decode, joint_decoder.decode
    calls = {"count": 0}

    def decode(signals, given, **options):
        if options or given != recipe:
            raise ValueError(f"D63 decodes only the standing recipe with no options (got {given}, {options})")
        calls["count"] += 1
        return ec.two_pass(signals, given, rho_pre, rho_post)

    def bypass(*args, **kwargs):
        raise RuntimeError("D63: joint_decoder.decode was called while a candidate is scored")

    diagnosis.decode, joint_decoder.decode = decode, bypass
    try:
        yield calls
    finally:
        diagnosis.decode, joint_decoder.decode = original, standing


def row_at(parts, recipe, truths, pff_seen, rho_pre, rho_post):
    """(row, conventions, decode calls): the standing scorer's row with every half decoded by D63 at these factors."""
    from dfl_submission import diagnosis_stack as ds, refine_trial
    with decoding_with(rho_pre, rho_post, recipe) as calls:
        row = ds.recomputed_row(parts["development"], parts["metrica"], parts["d24"], parts["guards"], recipe, truths,
                                pff_seen)
    return row, {name: refine_trial.standard(row["trial"], name) for name in CONVENTIONS}, calls["count"]


# -- the numbers of a row -----------------------------------------------------------------------------

def target_means(row, name):
    """(Metrica mean over the three games, D24 mean over the three views of the mean over their games) at a target."""
    metrica = float(np.mean([match[name]["spell_f1"] for match in row["metrica_matches"].values()]))
    d24 = float(np.mean([np.mean([match[name]["spell_f1"] for match in row["d24_views"][view].values()])
                         for view in D24_VIEWS]))
    return metrica, d24


def raw_groups(row):
    """D47's eight groups against the raw targets, and the standard combined spellF1 at raw."""
    conventions = row["trial"]["external"]["conventions"]["raw"]
    matches = conventions["matches"]
    metrica, d24 = target_means(row, "raw")
    return {"dfl_clean": matches["dfl/WPY/clean"]["spell_f1"], "dfl_masked": matches["dfl/WPY/masked"]["spell_f1"],
            "dfl_dead_carried": conventions["dead_carried"], "dfl_team_swap": conventions["team_swap"],
            "skillcorner": float(np.mean([value["spell_f1"] for key, value in matches.items()
                                          if key.startswith("skillcorner/")])),
            "pff": row["trial"]["pff_means"]["raw"]["spell_f1"], "metrica": metrica, "d24": d24,
            "standard_combined": conventions["standard"]["spell_f1"]}


def summary_of(row, conventions):
    """The numbers D63's conditions, selection and gate read, from one scored row."""
    control_metrica, control_d24 = target_means(row, "control")
    return {"composite": row["composite"], "primary": row["primary"], "selection": row["selection"],
            "groups": dict(row["groups"]), "guard_mean": row["guards"]["mean"], "guard_dfl": row["guards"]["a_dfl"],
            "guard_metrica": row["guards"]["b_metrica"], "conventions": dict(conventions),
            "control": {"metrica": control_metrica, "d24": control_d24}, "raw": raw_groups(row)}


def change(summary, reference):
    """summary - reference for every number (same structure)."""
    if isinstance(summary, dict):
        return {key: change(summary[key], reference[key]) for key in summary}
    return float(summary - reference)


# -- D63's reading ------------------------------------------------------------------------------------

def conditions(changes):
    """D63's conditions (a)-(f) on one candidate's changes against the standing decode."""
    held = {"a_composite": bool(changes["composite"] >= 0),
            "b_groups": bool(all(changes["groups"][name] >= -GROUP_LOSS for name in GROUPS)),
            "c_guard_mean": bool(changes["guard_mean"] >= -GUARD_LOSS),
            "d_conventions": bool(all(changes["conventions"][name] >= -CONVENTION_LOSS for name in BOUNDED)),
            "e_pff": bool(changes["groups"]["pff"] >= -PFF_LOSS),
            "f_control": bool(changes["control"]["metrica"] >= -CONTROL_LOSS
                              and changes["control"]["d24"] >= -CONTROL_LOSS)}
    return {"conditions": held, "holds": bool(all(held.values())),
            "failed": [name for name, ok in held.items() if not ok],
            "largest_group_fall": min(changes["groups"][name] for name in GROUPS),
            "largest_group_fall_name": min(GROUPS, key=lambda name: (changes["groups"][name], name))}


def selected(items):
    """Among candidates meeting (a)-(f), the highest D48 selection score; ties to the factors nearer 1, rho_pre first.

    `items` maps a key to {"rho_pre", "rho_post", "summary", "reading"}; None if no candidate meets (a)-(f)."""
    eligible = [key for key, item in items.items() if item["reading"]["holds"]]
    if not eligible:
        return None
    return max(eligible, key=lambda key: (items[key]["summary"]["selection"], items[key]["rho_pre"],
                                          items[key]["rho_post"]))


def gate(changes):
    """D63's gate on the selected candidate's changes."""
    numbers = {"selection": changes["selection"],
               "metrica_d24_mean": (changes["groups"]["metrica"] + changes["groups"]["d24"]) / 2,
               "metrica": changes["groups"]["metrica"], "d24": changes["groups"]["d24"]}
    held = {"selection": bool(numbers["selection"] >= SELECTION_GAIN),
            "metrica_d24_mean": bool(numbers["metrica_d24_mean"] >= METRICA_D24_GAIN)}
    return {"numbers": numbers, "conditions": held, "passed": bool(all(held.values()))}


def reading(items):
    """The whole reading: which candidates meet (a)-(f), the selection and the gate."""
    key = selected(items)
    decision = {"eligible": [k for k, item in items.items() if item["reading"]["holds"]], "selected": key,
                "gate": None, "passed": False}
    if key is None:
        decision["outcome"] = "no candidate meets (a)-(f): nothing goes to the confirmation and set E is not parsed"
        return decision
    decision["selected_factors"] = {"rho_pre": items[key]["rho_pre"], "rho_post": items[key]["rho_post"]}
    decision["gate"] = gate(items[key]["changes"])
    decision["passed"] = decision["gate"]["passed"]
    decision["outcome"] = (f"{key} is selected and passes the gate: the confirmation may follow" if decision["passed"]
                           else f"{key} is selected and fails the gate: nothing goes to the confirmation and set E "
                                f"is not parsed")
    return decision


# -- reported, not gated --------------------------------------------------------------------------------

def report_halves(parts):
    """The scored halves under D58's group names (`diagnosis.measure_all` groups a match by source and view)."""
    from dfl_submission import diagnosis_d43 as d46, diagnosis_stack as ds
    return parts["development"] + parts["metrica"] + d46.d24_grouped(parts["d24"]) + ds.guard_grouped(parts["guards"])


def report_group(key):
    source, _, view = key
    return source if source != "dfl" else f"dfl_{view}"


def half_id(half):
    return (half.source, half.match, half.view, int(half.index))


def decoded_halves(halves, recipe, rho_pre, rho_post):
    """{half id: (pass 1 on frames, pass 2 on frames, window counts)}."""
    result = {}
    for half in halves:
        first, second, scale = ec.two_pass_steps(half.signals, recipe, rho_pre, rho_post)
        length = len(half.signals[0])
        live = first != ec.DEAD
        result[half_id(half)] = (ec.to_frames(first, length), ec.to_frames(second, length),
                                 {"steps": int(len(first)), "live_steps": int(live.sum()),
                                  "window_steps": int((scale < 1).sum()), "live_window_steps": int((live & (scale < 1)).sum()),
                                  "changed_steps": int((first != second).sum())})
    return result


def match_items(truth, standing, candidate):
    """One match's reported items against its N = 3 target (D58's spell lengths) and against pass 1."""
    from dfl_submission import diagnosis
    target = diagnosis.target(truth, TARGET_SECONDS)
    runs = pm.valid_runs(target)
    true_lengths, true_matched, _ = diagnosis.spell_lengths(target, candidate)
    true_spells = pm.possession_spells(target, runs)
    first, second = pm.possession_spells(standing, runs), pm.possession_spells(candidate, runs)
    _, kept = diagnosis.matched_flags(pm.spell_overlaps(first, second), len(first), len(second), MATCH_IOU)
    lost, _ = diagnosis.matched_flags(pm.spell_overlaps(first, second), len(first), len(second), MATCH_IOU)
    _, hits = diagnosis.matched_flags(pm.spell_overlaps(true_spells, second), len(true_spells), len(second), MATCH_IOU)
    added = ~kept
    return {"true_lengths": true_lengths, "true_matched": true_matched,
            "predicted_standing": len(first), "predicted_candidate": len(second), "added": int(added.sum()),
            "added_matching_truth": int((added & hits).sum()), "removed": int((~lost).sum()),
            "added_lengths_frames": [int(stop - start) for (start, stop, _), new in zip(second, added) if new]}


def reported(parts, recipe, rho_pre, rho_post):
    """Per D58 group: N = 3 length recall, predicted possessions added against pass 1 and their match to the truth."""
    from dfl_submission import diagnosis
    halves = report_halves(parts)
    decoded = decoded_halves(halves, recipe, rho_pre, rho_post)
    standing = diagnosis.matches(halves, lambda half: decoded[half_id(half)][0])
    candidate = diagnosis.matches(halves, lambda half: decoded[half_id(half)][1])
    records = {}
    for key, (truth, prediction, _) in candidate.items():
        records.setdefault(report_group(key), []).append(match_items(truth, standing[key][1], prediction))
    windows = {}
    for half in halves:
        windows.setdefault(report_group(half.key), []).append(decoded[half_id(half)][2])
    result = {}
    for group in REPORT_GROUPS:
        items = records[group]
        added = sum(item["added"] for item in items)
        matched = sum(item["added_matching_truth"] for item in items)
        counts = {name: int(sum(item[name] for item in windows[group])) for name in windows[group][0]}
        result[group] = {
            "matches": len(items),
            "length_recall": diagnosis.length_recall(items),
            "predicted_standing": int(sum(item["predicted_standing"] for item in items)),
            "predicted_candidate": int(sum(item["predicted_candidate"] for item in items)),
            "added": int(added), "added_matching_truth_iou0.5": int(matched),
            "added_matching_share": matched / added if added else None,
            "removed": int(sum(item["removed"] for item in items)),
            "added_under_3s": int(sum(length < 3 * pm.FPS for item in items for length in item["added_lengths_frames"])),
            "windows": dict(counts, live_window_share=counts["live_window_steps"] / counts["live_steps"]
                            if counts["live_steps"] else None)}
    return result


def short_view(items):
    """The two short bins of each group's length recall, for the printed table."""
    return {group: {name: items[group]["length_recall"].get(name) for name in SHORT_BINS} for group in items}


def d58_check(items):
    """Whether the reported-item code at factors 1 reproduces D58's length recall for every one of its twelve groups."""
    stored = read(D58_RECORD)["groups"]["aba3"]
    result = {group: canonical(items[group]["length_recall"]) == canonical(stored[group]["length_recall"])
              for group in REPORT_GROUPS}
    return dict(result, passed=bool(all(result.values())))


# -- checks ---------------------------------------------------------------------------------------------

def modules_unchanged(root=ROOT):
    """No tracked file under the guarded folders differs from HEAD (git), with the lists recorded."""
    def lines(*args):
        output = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout
        return [line for line in output.splitlines() if line]
    tracked = lines("diff", "--name-only", "HEAD")
    untracked = lines("ls-files", "--others", "--exclude-standard")
    touched = [path for path in tracked if path.split("/")[0] in GUARDED]
    return {"head": lines("rev-parse", "HEAD")[0], "tracked_changes": tracked, "guarded_changes": touched,
            "untracked": untracked, "passed": not touched}


def pass_checks(halves, recipe):
    """Restated pass 1 equals `joint_decoder.decode` on every half; pass 2 at factors 1 equals pass 1."""
    from dfl_submission import joint_decoder
    one, two = 0, 0
    for half in halves:
        home, dead, dead_edge, owner_edge = half.signals
        expected = joint_decoder.decode(home, dead, dead_edge, owner_edge, cutoff=recipe["cutoff"],
                                        dead_cost=recipe["dead_cost"], owner_cost=recipe["owner_cost"],
                                        discount=recipe["discount"])
        first, second, scale = ec.two_pass_steps(half.signals, recipe, 1.0, 1.0)
        one += bool(np.array_equal(ec.to_frames(first, len(home)), expected))
        two += bool(np.array_equal(first, second) and np.all(scale == 1)
                    and np.array_equal(ec.two_pass(half.signals, recipe, 1.0, 1.0), expected))
    return {"halves": len(halves), "pass_one_equals_joint_decoder": one, "pass_two_at_one_equals_pass_one": two,
            "passed": bool(one == len(halves) and two == len(halves))}


# -- the run --------------------------------------------------------------------------------------------

def inputs(dfl, skill, parsed, *, raw, cache_root):
    """The stack's identity, its cached halves at cost 14, the Metrica truths and PFF visibility (D58's loading)."""
    from dfl_submission import diagnosis_stack as ds, hidden_trial as h49, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import stack_trial as st, vendor_trial
    games = vendor_trial.development()
    if not vendor_trial.gate_passed(vendor_trial.RECIPE / "sanity.json", vendor_trial.inputs(parsed, games)):
        raise SystemExit("D63 stops: the latest PFF reader sanity attempt did not pass on these parsed inputs")
    if not ms.gate_passed(ms.RECIPE / "sanity.json", ms.inputs(ms.PARSED)):
        raise SystemExit("D63 stops: the latest D19 reader attempt did not pass on these parsed inputs")
    record = read(st.RECIPE / "development_t.json")
    choice = record["amendment_choice"]
    if not (record["decision"]["selected_cost"] == COST and choice["stack"] == "8100" and choice["owner_cost"] == COST):
        raise SystemExit("D63 stops: the development record does not name the 8,100-step stack at cost 14")
    identity = ds.stack_identity(record)
    if not ds.adopted_networks(identity, read(st.RECIPE / "adoption_models.json")):
        raise SystemExit("D63 stops: the networks development scored are not the ones the build ships")
    recipe = ht.recipe_at(COST)
    development, metrica, d24, guards = ds.stack_halves(dfl, skill, parsed, games, identity, raw=raw,
                                                        cache_root=cache_root)
    parts = {"development": development, "metrica": metrica, "d24": d24, "guards": guards}
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    pff_seen = h49.pff_visibility(parsed, games)
    return games, record, identity, recipe, parts, truths, pff_seen


def develop(dfl, skill, parsed, *, raw=True, cache_root=CACHE, output=None):
    """D63's development, written once (DECISIONS.md D63)."""
    require_external_only()
    from dfl_submission import diagnosis, diagnosis_stack as ds, stack_trial as st
    output = RECIPE / "d63_development.json" if output is None else Path(output)
    decision_path = output.parent / "d63_decision.json"
    for path in (output, decision_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; D63's development records are written once")
    st.wait_for_scoring("D63 development")
    started = time.time()

    modules = modules_unchanged()
    print(f"D63 check, modules unchanged: {modules['passed']} (tracked changes {modules['tracked_changes']})", flush=True)
    if not modules["passed"]:
        raise SystemExit(f"D63 stops: modules written before D63 differ from HEAD {modules['guarded_changes']}")
    games, record, identity, recipe, parts, truths, pff_seen = inputs(dfl, skill, parsed, raw=raw,
                                                                      cache_root=cache_root)
    halves = [half for part in PARTS for half in parts[part]]
    print(f"D63: stack halves loaded ({len(halves)}); {time.time() - started:.0f} s", flush=True)

    passes = pass_checks(halves, recipe)
    print(f"D63 check, passes: {passes}", flush=True)
    if not passes["passed"]:
        raise SystemExit(f"D63 stops: the restated pass 1 or pass 2 at factors 1 differs {passes}; no record was written")

    row, conventions, calls = row_at(parts, recipe, truths, pff_seen, 1.0, 1.0)
    check1 = ds.compare_row(row, ds.stored_row(record, COST))
    print("D63 check, both factors at 1 against the cost-14 row: " + ", ".join(
        f"{name} {'ok' if ok else 'DIFFERS'}" for name, ok in check1.items()) + f" ({calls} decodes)", flush=True)
    if not all(check1.values()) or calls == 0:
        raise SystemExit(f"D63 stops: pipeline check failed {check1}; no record was written")
    reference = summary_of(row, conventions)
    reference_reported = reported(parts, recipe, 1.0, 1.0)
    d58 = d58_check(reference_reported)
    print(f"D63 check, reported-item code against D58's length recall: {d58}", flush=True)
    if not d58["passed"]:
        raise SystemExit(f"D63 stops: the reported-item code does not reproduce D58's length recall {d58}")
    if any(item["added"] or item["removed"] for item in reference_reported.values()):
        raise SystemExit("D63 stops: at factors 1 the reported-item code finds possessions added or removed")
    print(f"D63: checks passed; {time.time() - started:.0f} s", flush=True)

    items = {}
    for rho_pre, rho_post in candidates():
        key = key_of(rho_pre, rho_post)
        scored_row, scored_conventions, scored_calls = row_at(parts, recipe, truths, pff_seen, rho_pre, rho_post)
        if scored_calls != calls:
            raise SystemExit(f"D63 stops: {key} decoded {scored_calls} times, the standing row {calls}")
        summary = summary_of(scored_row, scored_conventions)
        changes = change(summary, reference)
        items[key] = {"rho_pre": rho_pre, "rho_post": rho_post, "row": scored_row, "summary": summary,
                      "changes": changes, "reading": conditions(changes),
                      "reported": reported(parts, recipe, rho_pre, rho_post)}
        print(f"D63 {key}: selection {changes['selection']:+.4f}, composite {changes['composite']:+.4f}, guard mean "
              f"{changes['guard_mean']:+.4f}, PFF {changes['groups']['pff']:+.4f}, Metrica "
              f"{changes['groups']['metrica']:+.4f}, D24 {changes['groups']['d24']:+.4f}, control "
              f"{changes['control']['metrica']:+.4f}/{changes['control']['d24']:+.4f}, holds "
              f"{items[key]['reading']['holds']} {items[key]['reading']['failed']}; {time.time() - started:.0f} s",
              flush=True)

    decision = reading(items)
    sides = {"rho_pre_alone": [key_of(p, 1.0) for p in ec.FACTORS if p != 1.0],
             "rho_post_alone": [key_of(1.0, q) for q in ec.FACTORS if q != 1.0]}
    development_record = canonical({
        "protocol": PROTOCOL, "passed_checks": True, "owner_cost": COST, "recipe": recipe,
        "checks": {"modules": modules, "passes": passes, "row_at_factors_1": check1, "decode_calls_per_row": calls,
                   "reported_code_against_d58": d58},
        "reference": {"name": "the standing decode (pass 1; both factors 1)", "row": row, "summary": reference,
                      "reported": reference_reported},
        "grid": {"factors": list(ec.FACTORS), "window_steps": ec.WINDOW, "step_frames": 5,
                 "candidates": [key_of(p, q) for p, q in candidates()], "key": "rho_pre/rho_post",
                 "definition": ec.__doc__},
        "candidates": items, "sides": sides,
        "reported_definitions": {
            "length_recall": "D58's: per true possession of the N = 3 target, matched at each IoU by the metric's greedy "
                             "one-to-one match (diagnosis.spell_lengths, diagnosis.length_recall); bins in seconds",
            "added": "a predicted possession of the candidate that the metric's greedy one-to-one match at IoU 0.5 does "
                     "not pair with a possession of the same team in pass 1 (the standing decode)",
            "added_matching_truth_iou0.5": "an added possession that the greedy one-to-one match at IoU 0.5 pairs "
                                           "with a true possession of the N = 3 target",
            "removed": "a pass-1 possession that the same match does not pair with a candidate possession",
            "windows": "steps of the 5 Hz grid: live in pass 1, with a factor below 1, changed by pass 2",
            "raw": "D47's eight groups scored against the raw targets by the same row (summary.raw)"},
        "identity": identity, "development_t_sha256": digest(RECIPE / "development_t.json"),
        "d58_sha256": digest(D58_RECORD),
        "caches": {part: digest(cache_root / "stack" / part / "manifest.json") for part in PARTS},
        "teacher_sha256": diagnosis.teacher_identity(),
        "code": {name: digest(PACKAGE / name) for name in CODE},
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    write_new(output, development_record)
    write_new(decision_path, canonical(dict(
        decision, protocol=PROTOCOL, development_sha256=digest(output),
        candidates={key: {"rho_pre": item["rho_pre"], "rho_post": item["rho_post"],
                          "selection_change": item["changes"]["selection"], "conditions": item["reading"]["conditions"],
                          "holds": item["reading"]["holds"]} for key, item in items.items()},
        rule={"against": "the standing decode (cutoff 0.53, stoppage cost 2, ownership cost 14, discount 0.8)",
              "conditions": {"a_composite": 0.0, "b_groups": -GROUP_LOSS, "c_guard_mean": -GUARD_LOSS,
                             "d_conventions": {name: -CONVENTION_LOSS for name in BOUNDED}, "e_pff": -PFF_LOSS,
                             "f_control": {"metrica": -CONTROL_LOSS, "d24": -CONTROL_LOSS}, "groups": list(GROUPS)},
              "selection": "among the candidates meeting (a)-(f), the highest D48 selection score (mean of the "
                           "composite and the long-hidden guard mean); ties to the factors nearer 1, rho_pre first",
              "gate": {"selection": SELECTION_GAIN, "metrica_d24_mean": METRICA_D24_GAIN}},
        decided_at=time.strftime("%Y-%m-%d %H:%M %Z"))))
    print(table(items), flush=True)
    print(f"D63 eligible {decision['eligible']}; selected {decision['selected']}; gate {decision['gate']}; "
          f"{decision['outcome']}", flush=True)
    return decision


def table(items):
    lines = ["| rho_pre/rho_post | Selection | Composite | Guard mean | PFF | Metrica | D24 | Control Metrica | "
             "Control D24 | Conditions |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key, item in items.items():
        c = item["changes"]
        failed = item["reading"]["failed"]
        lines.append(f"| {key} | {c['selection']:+.4f} | {c['composite']:+.4f} | {c['guard_mean']:+.4f} | "
                     f"{c['groups']['pff']:+.4f} | {c['groups']['metrica']:+.4f} | {c['groups']['d24']:+.4f} | "
                     f"{c['control']['metrica']:+.4f} | {c['control']['d24']:+.4f} | "
                     f"{'all' if not failed else 'fails ' + ', '.join(failed)} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("develop")
    one.add_argument("--dfl", type=Path, required=True)
    one.add_argument("--dfl-raw", action="store_true")
    one.add_argument("--skill", type=Path, required=True)
    one.add_argument("--parsed", type=Path, required=True)
    one.add_argument("--cache", type=Path, default=CACHE, help="the 8,100-step stack's cache root")
    args = parser.parse_args()
    require_external_only()
    develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache_root=args.cache)


if __name__ == "__main__":
    main()
