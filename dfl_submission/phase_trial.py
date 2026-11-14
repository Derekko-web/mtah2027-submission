"""D62 development: Smart Tagging's phase teams as ownership evidence where the ball is long hidden (DECISIONS.md D62).

`develop` (with `MTAH_DATA_DIR` set to a nonexistent path) decodes the adopted stack (D48: the 8,100-step stack of
`stack_recipe/development_t.json`, three seeds per slot, ownership cost 14) from the decoder inputs its development
record was scored from (`data/d48/t/cache/stack`: `development`, `metrica`, `d24`, `guards`), as D58 did. Nothing is
forecast. The ball's visibility per half is rebuilt from the external sources as D58 rebuilt it (each rebuilt half
must have the cached half's frame count and labels; the D24 views must reproduce the D24 record's visibility counts
and the guard views D49's guard statistics) and must equal, on every half, the `seen` array D59's member caches hold
(the arrays D59's HT mask was computed from).

Then, in this order, stopping without a record if a check fails:
1. pipeline check 1: with no tags the scorer reproduces the cost-14 row of `development_t.json` exactly;
2. pipeline check 2: a tag state of 0 everywhere (at each weight of the grid), and w = 0 (under every profile and
   draw), each leave the home input bit for bit as it is and decode to the standing labels on every scored half;
3. tags simulated from each half's N = 3 target under the four profiles, three draws each (seed
   620000 + 1000 x half index + 10 x profile index + draw index; the half index is the position in the scored
   order: development, Metrica, D24, guards, each in its cache manifest's order); each weight of the grid scored on
   D47's eight groups, D49's two long-hidden guard views and D33's conventions, against the stack with no tags;
   scores averaged over the draws;
4. D62's readings, mechanically: bounded loss (Poor and Carry), selection (the largest safe weight), the gate
   (Moderate), and the agreement shares with θ.

Records, each written once: `stack_recipe/d62_development.json` (every score, per group, per profile, per draw and
averaged; the half order; the acting and far shares; the agreement counts; code digests) and
`stack_recipe/d62_decision.json` (the reading).
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import phase_evidence as pe
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stack_recipe"
CACHE = ROOT / "data" / "d48" / "t" / "cache"
D59_MEMBERS = ROOT / "data" / "d59" / "members"
PROTOCOL = "DECISIONS.md D62"
COST = 14
TARGET_SECONDS = 3.0                       # the N = 3 target
PARTS = ("development", "metrica", "d24", "guards")
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica", "d24")
CONVENTIONS = ("aba1.75", "aba3", "aba4", "aba4.5")
BOUNDED = ("aba1.75", "aba4", "aba4.5")    # "N = 1.75, 4 and 4.5"
SAFETY_PROFILES = ("Poor", "Carry")
GATE_PROFILE = "Moderate"
THETA_PROFILES = ("Moderate", "Poor")
COMPOSITE_LOSS, PFF_LOSS, GUARD_LOSS, GROUP_LOSS, CONVENTION_LOSS = .003, .005, .005, .01, .01
PFF_GAIN, GUARD_GAIN = .010, .010
CODE = ("phase_evidence.py", "phase_trial.py", "tree_share_trial.py", "diagnosis_stack.py", "diagnosis_d43.py",
        "hidden_trial.py", "diagnosis.py", "joint_decoder.py", "conventions.py")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def weight_key(weight):
    return str(float(weight))


# -- the halves, their groups and visibility -------------------------------------------------------

def group_of(part, half):
    """The scoring group of a half: D47's groups for development and Metrica, the D24 view, the guard source."""
    if part == "development":
        return half.source if half.source != "dfl" else f"dfl_{half.view}"
    if part == "metrica":
        return "metrica"
    if part == "d24":
        return f"d24_{half.view}"
    if part == "guards":
        return f"guard_{half.source}"
    raise ValueError(f"Unknown part {part}")


def ordered(parts):
    """[(part, half)] in D62's scored order: development, Metrica, D24, guards, each in its cache's order."""
    return [(part, half) for part in PARTS for half in parts[part]]


def half_order(parts):
    return [{"index": i, "part": part, "source": half.source, "match": half.match, "view": half.view,
             "half": int(half.index), "frames": int(len(half.labels)), "group": group_of(part, half)}
            for i, (part, half) in enumerate(ordered(parts))]


def guard_visibility(tracking):
    from dfl_submission import diagnosis_d43 as d46, hidden_views as hv
    return {(source, match, hv.VIEW, index): (np.asarray(labels), d46.seen(ball))
            for (source, match, index), (labels, _, ball, _, _) in tracking.items()}


def rebuilt_visibility(dfl, skill, parsed, games, parts, *, raw):
    """{part: [seen per half]} rebuilt as D58 rebuilt it, with D58's checks; stops on any difference."""
    from dfl_submission import diagnosis_d43 as d46, hidden_trial as h49, metrica_views as mv
    development = d46.development_visibility(dfl, skill, parsed, games, raw=raw)
    metrica, rows = d46.metrica_visibility()
    if canonical(rows) != canonical(read(mv.RECORD)["visibility"]):
        raise SystemExit("D62 stops: the rebuilt D24 views do not reproduce the D24 record's visibility counts")
    tracking = h49.guard_tracking(dfl, raw=raw)
    if canonical(h49.guard_statistics(tracking)) != canonical(read(h49.RECIPE / "reference.json")["guard_views"]):
        raise SystemExit("D62 stops: the rebuilt guard views do not reproduce D49's guard statistics")
    sources = {"development": development, "metrica": metrica, "d24": metrica, "guards": guard_visibility(tracking)}
    result, frames = {}, {}
    for part in PARTS:
        attached, checks = d46.attach(parts[part], sources[part])
        result[part] = [attached[d46.half_key(half)] for half in parts[part]]
        frames[part] = sum(checks.values())
    return result, frames


def d59_seen_check(parts, visible, members=D59_MEMBERS):
    """The rebuilt visibility equals, half by half, the `seen` arrays of D59's member caches (D59's HT input)."""
    stored = read(RECIPE / "d59_scores.json")
    if stored["passed_checks"] is not True:
        raise SystemExit("D62 stops: D59's member caches did not pass their checks")
    result = {}
    for part in PARTS:
        manifest = members / part / "manifest.json"
        if digest(manifest) != stored["member_caches"][part]:
            raise SystemExit(f"D62 stops: D59's {part} member manifest differs from the one d59_scores.json names")
        entries = read(manifest)["halves"]
        keys = [(e["source"], e["match"], e["view"], e["index"]) for e in entries]
        if keys != [(h.source, h.match, h.view, h.index) for h in parts[part]]:
            raise SystemExit(f"D62 stops: D59's {part} members are not the stack's halves in order")
        equal = True
        for entry, half, seen in zip(entries, parts[part], visible[part], strict=True):
            path = members / part / entry["file"]
            if digest(path) != entry["sha256"]:
                raise SystemExit(f"D62 stops: {path.name} differs from D59's manifest")
            with np.load(path, allow_pickle=False) as item:
                equal = equal and np.array_equal(item["seen"], seen) and np.array_equal(item["labels"], half.labels)
        result[part] = {"halves": len(entries), "seen_and_labels_equal": bool(equal)}
    return result


# -- scoring ------------------------------------------------------------------------------------------

def shifted_parts(parts, states, masks, weight):
    """{part: [HalfSignals]} with each half's home input shifted by its state on its mask (None: no tags)."""
    if states is None:
        return parts
    result, i = {}, 0
    for part in PARTS:
        result[part] = []
        for half in parts[part]:
            result[part].append(replace(half, signals=pe.shifted_signals(half.signals, states[i], masks[i], weight)))
            i += 1
    return result


def row_of(parts, recipe, truths, pff_seen):
    """The row exactly as `stack_trial.develop` stored it (D58's `recomputed_row`), plus D33's conventions."""
    from dfl_submission import diagnosis_stack as ds, refine_trial
    row = ds.recomputed_row(parts["development"], parts["metrica"], parts["d24"], parts["guards"], recipe, truths,
                            pff_seen)
    return row, {name: refine_trial.standard(row["trial"], name) for name in CONVENTIONS}


def summary_of(row, conventions):
    """The numbers D62's readings use, from one scored row."""
    return {"composite": row["composite"], "primary": row["primary"], "selection": row["selection"],
            "groups": dict(row["groups"]), "guard_mean": row["guards"]["mean"], "guard_dfl": row["guards"]["a_dfl"],
            "guard_metrica": row["guards"]["b_metrica"], "conventions": dict(conventions),
            "pff_agreement_unseen": row["pff_visibility"]["agreement_unseen"]["agreement"],
            "pff_agreement_seen": row["pff_visibility"]["agreement_seen"]["agreement"]}


def average(summaries):
    """The mean over draws of every number in the summaries."""
    def mean(values):
        first = values[0]
        if isinstance(first, dict):
            return {key: mean([value[key] for value in values]) for key in first}
        return float(np.mean(values))
    return mean(list(summaries))


def change(summary, reference):
    """summary - reference for every number (same structure)."""
    if isinstance(summary, dict):
        return {key: change(summary[key], reference[key]) for key in summary}
    return float(summary - reference)


# -- D62's readings ---------------------------------------------------------------------------------

def bounded(changes):
    """D62's bounded-loss conditions for one profile's averaged changes."""
    conditions = {"composite": bool(changes["composite"] >= -COMPOSITE_LOSS),
                  "pff": bool(changes["groups"]["pff"] >= -PFF_LOSS),
                  "guard_mean": bool(changes["guard_mean"] >= -GUARD_LOSS),
                  "groups": bool(all(changes["groups"][name] >= -GROUP_LOSS for name in GROUPS)),
                  "conventions": bool(all(changes["conventions"][name] >= -CONVENTION_LOSS for name in BOUNDED))}
    numbers = {"composite": changes["composite"], "pff": changes["groups"]["pff"], "guard_mean": changes["guard_mean"],
               "largest_group_fall": min(changes["groups"][name] for name in GROUPS),
               "largest_group_fall_name": min(GROUPS, key=lambda name: (changes["groups"][name], name)),
               "conventions": {name: changes["conventions"][name] for name in BOUNDED}}
    return {"conditions": conditions, "numbers": numbers, "holds": bool(all(conditions.values()))}


def why_not(safety):
    """The failed conditions of one weight, named with their profile."""
    return [f"{profile}: {name}" for profile in SAFETY_PROFILES
            for name, ok in safety[profile]["conditions"].items() if not ok]


def safety_of(averaged_changes):
    """{weight: {Poor, Carry: bounded(...)}, safe, why_not} from {weight: {profile: averaged changes}}."""
    result = {}
    for weight, profiles in averaged_changes.items():
        safety = {profile: bounded(profiles[profile]) for profile in SAFETY_PROFILES}
        result[weight] = dict(safety, safe=bool(all(safety[profile]["holds"] for profile in SAFETY_PROFILES)),
                              why_not=why_not(safety))
    return result


def selected(safety):
    """The largest safe weight (not chosen by its gain); None if no weight is safe."""
    safe = [float(weight) for weight, item in safety.items() if item["safe"]]
    return weight_key(max(safe)) if safe else None


def gate(changes):
    """D62's gate on the selected weight's averaged changes under Moderate."""
    numbers = {"pff": changes["groups"]["pff"], "guard_mean": changes["guard_mean"], "composite": changes["composite"]}
    conditions = {"pff": bool(numbers["pff"] >= PFF_GAIN), "guard_mean": bool(numbers["guard_mean"] >= GUARD_GAIN),
                  "composite": bool(numbers["composite"] >= 0)}
    return {"numbers": numbers, "conditions": conditions, "passed": bool(all(conditions.values()))}


def theta(shares):
    """θ: the midpoint of the agreement share under Moderate and under Poor."""
    return (shares["Moderate"] + shares["Poor"]) / 2


def reading(averaged_changes, shares):
    """The whole reading: which weights are safe and why not, the selection, the gate, θ."""
    safety = safety_of(averaged_changes)
    weight = selected(safety)
    decision = {"safety": safety, "safe_weights": [w for w, item in safety.items() if item["safe"]],
                "selected_weight": weight, "gate": None, "passed": False,
                "good_reported": {w: averaged_changes[w].get("Good") for w in averaged_changes},
                "agreement_shares": dict(shares), "theta": theta(shares)}
    if weight is None:
        decision["outcome"] = "no weight is safe: nothing ships and set E is not parsed"
        return decision
    decision["gate"] = gate(averaged_changes[weight][GATE_PROFILE])
    decision["passed"] = decision["gate"]["passed"]
    decision["outcome"] = (f"w = {weight} is selected and passes the gate: the confirmation on set E may follow"
                           if decision["passed"] else
                           f"w = {weight} is selected and fails the gate: nothing ships and set E is not parsed")
    return decision


# -- shares -------------------------------------------------------------------------------------------

def pooled(counts):
    """{key: [numerator, denominator]} summed, with the share."""
    numerator = int(sum(value[0] for value in counts))
    denominator = int(sum(value[1] for value in counts))
    return {"numerator": numerator, "denominator": denominator,
            "share": numerator / denominator if denominator else None}


def frame_shares(target, seen, mask):
    """Counts of one half's frames: all, live (target H or A), unseen, far (masked), far and live."""
    live = np.isin(np.asarray(target), (pm.H, pm.A))
    return {"frames": int(len(target)), "live": int(live.sum()), "unseen": int((~np.asarray(seen, dtype=bool)).sum()),
            "far": int(np.asarray(mask).sum()), "far_live": int((np.asarray(mask) & live).sum())}


def state_counts(target, state, mask, phases):
    """Counts of one half under one draw: live frames tagged, acted on; phases simulated and naming the wrong team."""
    target, state = np.asarray(target), np.asarray(state)
    live = np.isin(target, (pm.H, pm.A))
    acts = pe.acting(state, mask)
    true_team = np.where(target == pm.H, pe.HOME, np.where(target == pm.A, pe.AWAY, pe.NONE))
    tagged_live = live & (state != pe.NONE)
    return {"live": int(live.sum()), "tagged_live": int(tagged_live.sum()),
            "tagged_live_right": int((tagged_live & (state == true_team)).sum()),
            "acting": int(acts.sum()), "acting_live": int((acts & live).sum()),
            "acting_live_right": int((acts & live & (state == true_team)).sum()), "phases": len(phases)}


def summed(items):
    keys = items[0].keys()
    return {key: int(sum(item[key] for item in items)) for key in keys}


def share_table(counts, order):
    """Per group and over all halves: the acting share of live frames and the other shares, from summed counts."""
    def ratio(a, b):
        return a / b if b else None
    groups = {}
    for i, item in enumerate(order):
        groups.setdefault(item["group"], []).append(counts[i])
    result = {}
    for name, items in {**groups, "all": list(counts)}.items():
        total = summed(items)
        result[name] = dict(total, acting_share_of_live=ratio(total["acting_live"], total["live"]),
                            tagged_share_of_live=ratio(total["tagged_live"], total["live"]),
                            tagged_right_share=ratio(total["tagged_live_right"], total["tagged_live"]),
                            acting_right_share=ratio(total["acting_live_right"], total["acting_live"]))
    return result


def far_table(frames, order):
    def ratio(a, b):
        return a / b if b else None
    groups = {}
    for i, item in enumerate(order):
        groups.setdefault(item["group"], []).append(frames[i])
    result = {}
    for name, items in {**groups, "all": list(frames)}.items():
        total = summed(items)
        result[name] = dict(total, far_share=ratio(total["far"], total["frames"]),
                            far_share_of_live=ratio(total["far_live"], total["live"]),
                            unseen_share=ratio(total["unseen"], total["frames"]))
    return result


# -- the run -------------------------------------------------------------------------------------------

def inputs(dfl, skill, parsed, *, raw, cache_root):
    """The stack's identity, its cached halves at cost 14, the Metrica truths and PFF visibility (D58's loading)."""
    from dfl_submission import diagnosis_stack as ds, hidden_trial as h49, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import stack_trial as st, vendor_trial
    games = vendor_trial.development()
    if not vendor_trial.gate_passed(vendor_trial.RECIPE / "sanity.json", vendor_trial.inputs(parsed, games)):
        raise SystemExit("D62 stops: the latest PFF reader sanity attempt did not pass on these parsed inputs")
    if not ms.gate_passed(ms.RECIPE / "sanity.json", ms.inputs(ms.PARSED)):
        raise SystemExit("D62 stops: the latest D19 reader attempt did not pass on these parsed inputs")
    record = read(st.RECIPE / "development_t.json")
    choice = record["amendment_choice"]
    if not (record["decision"]["selected_cost"] == COST and choice["stack"] == "8100" and choice["owner_cost"] == COST):
        raise SystemExit("D62 stops: the development record does not name the 8,100-step stack at cost 14")
    identity = ds.stack_identity(record)
    if not ds.adopted_networks(identity, read(st.RECIPE / "adoption_models.json")):
        raise SystemExit("D62 stops: the networks development scored are not the ones the build ships")
    recipe = ht.recipe_at(COST)
    development, metrica, d24, guards = ds.stack_halves(dfl, skill, parsed, games, identity, raw=raw,
                                                        cache_root=cache_root)
    parts = {"development": development, "metrica": metrica, "d24": d24, "guards": guards}
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    pff_seen = h49.pff_visibility(parsed, games)
    return games, record, identity, recipe, parts, truths, pff_seen


def check_two(halves, targets, seen, masks, standing, recipe):
    """Pipeline check 2 on every scored half: zero state at each weight, and w = 0 under every profile and draw."""
    from dfl_submission import diagnosis
    zero_bits, zero_labels, w0_bits, w0_labels = True, True, True, True
    for i, (_, half) in enumerate(halves):
        home = half.signals[0]
        zeros = np.zeros(len(home), dtype=np.int8)
        for weight in pe.WEIGHTS:
            shifted = pe.shifted_signals(half.signals, zeros, masks[i], weight)
            zero_bits &= bool(shifted[0].dtype == home.dtype and np.array_equal(shifted[0], home)
                              and all(a is b for a, b in zip(shifted[1:], half.signals[1:])))
            zero_labels &= bool(np.array_equal(diagnosis.decode(shifted, recipe), standing[i]))
        for p, name in enumerate(pe.PROFILE_ORDER):
            for draw in range(pe.DRAWS):
                state, mask, _ = pe.half_state(targets[i], seen[i], pe.PROFILES[name],
                                               np.random.default_rng(pe.seed(i, p, draw)))
                shifted = pe.shifted_signals(half.signals, state, mask, 0.)
                w0_bits &= bool(shifted[0].dtype == home.dtype and np.array_equal(shifted[0], home))
                w0_labels &= bool(np.array_equal(diagnosis.decode(shifted, recipe), standing[i]))
    return {"zero_state_home_bit_for_bit_at_each_weight": zero_bits,
            "zero_state_standing_labels_at_each_weight": zero_labels,
            "w0_home_bit_for_bit_every_profile_and_draw": w0_bits,
            "w0_standing_labels_every_profile_and_draw": w0_labels,
            "passed": bool(zero_bits and zero_labels and w0_bits and w0_labels)}


def develop(dfl, skill, parsed, *, raw=True, cache_root=CACHE, output=None):
    """D62's development, written once (DECISIONS.md D62)."""
    require_external_only()
    from dfl_submission import diagnosis, diagnosis_stack as ds, stack_trial as st
    output = RECIPE / "d62_development.json" if output is None else Path(output)
    decision_path = output.parent / "d62_decision.json"
    for path in (output, decision_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; D62's development records are written once")
    st.wait_for_scoring("D62 development")
    started = time.time()
    games, record, identity, recipe, parts, truths, pff_seen = inputs(dfl, skill, parsed, raw=raw,
                                                                      cache_root=cache_root)
    print(f"D62: stack halves loaded ({sum(len(v) for v in parts.values())}); {time.time() - started:.0f} s", flush=True)

    # pipeline check 1
    row, conventions = row_of(parts, recipe, truths, pff_seen)
    check1 = ds.compare_row(row, ds.stored_row(record, COST))
    print("D62 check 1, no tags against the cost-14 row: " + ", ".join(
        f"{name} {'ok' if ok else 'DIFFERS'}" for name, ok in check1.items()), flush=True)
    if not all(check1.values()):
        raise SystemExit(f"D62 stops: pipeline check 1 failed {check1}; no record was written")
    reference = summary_of(row, conventions)

    # visibility, the masks and the standing labels
    visible, rebuilt_frames = rebuilt_visibility(dfl, skill, parsed, games, parts, raw=raw)
    d59 = d59_seen_check(parts, visible)
    if not all(item["seen_and_labels_equal"] for item in d59.values()):
        raise SystemExit(f"D62 stops: the rebuilt visibility differs from D59's member caches {d59}")
    halves = ordered(parts)
    order = half_order(parts)
    seen = [s for part in PARTS for s in visible[part]]
    targets = [diagnosis.target(np.asarray(half.labels), TARGET_SECONDS) for _, half in halves]
    masks = [pe.far_mask(s) for s in seen]
    standing = [diagnosis.decode(half.signals, recipe) for _, half in halves]
    print(f"D62: visibility rebuilt and equal to D59's members; {time.time() - started:.0f} s", flush=True)

    # pipeline check 2
    check2 = check_two(halves, targets, seen, masks, standing, recipe)
    print(f"D62 check 2: {check2}", flush=True)
    if not check2["passed"]:
        raise SystemExit(f"D62 stops: pipeline check 2 failed {check2}; no record was written")

    # the grid
    frames = [frame_shares(t, s, m) for t, s, m in zip(targets, seen, masks)]
    entries = {weight_key(w): {name: [] for name in pe.PROFILE_ORDER} for w in pe.WEIGHTS}
    draws = {weight_key(w): {name: [] for name in pe.PROFILE_ORDER} for w in pe.WEIGHTS}
    counts = {name: [] for name in pe.PROFILE_ORDER}
    agreements = {name: [] for name in pe.PROFILE_ORDER}
    for p, name in enumerate(pe.PROFILE_ORDER):
        profile = pe.PROFILES[name]
        for draw in range(pe.DRAWS):
            states, draw_counts, draw_agreement = [], [], []
            for i in range(len(halves)):
                state, mask, phases = pe.half_state(targets[i], seen[i], profile,
                                                    np.random.default_rng(pe.seed(i, p, draw)))
                if not np.array_equal(mask, masks[i]):
                    raise SystemExit("D62 stops: a half's mask changed between the checks and the grid")
                states.append(state)
                draw_counts.append(state_counts(targets[i], state, mask, phases))
                draw_agreement.append(list(pe.agreement(state, standing[i], seen[i])))
            counts[name].append(draw_counts)
            agreements[name].append(draw_agreement)
            for weight in pe.WEIGHTS:
                scored_row, scored_conventions = row_of(shifted_parts(parts, states, masks, weight), recipe, truths,
                                                        pff_seen)
                key = weight_key(weight)
                entries[key][name].append(scored_row)
                draws[key][name].append(summary_of(scored_row, scored_conventions))
                item = draws[key][name][-1]
                print(f"D62 {name} draw {draw} w {weight}: composite {item['composite'] - reference['composite']:+.4f}, "
                      f"PFF {item['groups']['pff'] - reference['groups']['pff']:+.4f}, guard mean "
                      f"{item['guard_mean'] - reference['guard_mean']:+.4f}; {time.time() - started:.0f} s", flush=True)

    averaged = {w: {name: average(draws[w][name]) for name in pe.PROFILE_ORDER} for w in draws}
    averaged_changes = {w: {name: change(averaged[w][name], reference) for name in pe.PROFILE_ORDER} for w in draws}
    per_draw_changes = {w: {name: [change(item, reference) for item in draws[w][name]] for name in pe.PROFILE_ORDER}
                        for w in draws}
    agreement_pooled = {name: pooled([pair for draw in agreements[name] for pair in draw]) for name in pe.PROFILE_ORDER}
    shares = {name: agreement_pooled[name]["share"] for name in pe.PROFILE_ORDER}
    agreement_groups = {}
    for name in pe.PROFILE_ORDER:
        grouped = {}
        for draw in agreements[name]:
            for i, pair in enumerate(draw):
                grouped.setdefault(order[i]["group"], []).append(pair)
        agreement_groups[name] = {group: pooled(pairs) for group, pairs in grouped.items()}
    acting = {name: share_table([summed([counts[name][d][i] for d in range(pe.DRAWS)]) for i in range(len(halves))],
                                order) for name in pe.PROFILE_ORDER}
    decision = reading(averaged_changes, shares)

    development_record = canonical({
        "protocol": PROTOCOL, "passed_checks": True, "owner_cost": COST,
        "check_1": check1, "check_2": check2, "d59_seen_check": d59, "rebuilt_frames": rebuilt_frames,
        "reference": {"name": "the adopted stack, no tags", "row": row, "summary": reference},
        "grid": {"weights": [weight_key(w) for w in pe.WEIGHTS], "profiles": {
            name: {"c": pe.PROFILES[name].c, "a_seconds": pe.PROFILES[name].a, "b_seconds": pe.PROFILES[name].b,
                   "q": pe.PROFILES[name].q, "carry": pe.PROFILES[name].carry} for name in pe.PROFILE_ORDER},
                 "profile_order": list(pe.PROFILE_ORDER), "draws": pe.DRAWS,
                 "seed": "620000 + 1000 x half index + 10 x profile index + draw index (zero-based; NumPy default_rng)",
                 "mask": "nearest ball sighting, earlier or later in the half, more than 75 frames (3 s) away "
                         "(tree_share_trial.far_from_sighting); every frame if the half has no sighting",
                 "shift": "home' = expit(logit(clip(home, .001, .999)) + w x state) on masked frames with a nonzero "
                          "state, stored in the cached input's dtype (float32); every other frame and input unchanged",
                 "simulation": pe.__doc__},
        "half_order": order,
        "rows": entries, "draws": draws, "averaged": averaged, "averaged_changes": averaged_changes,
        "per_draw_changes": per_draw_changes,
        "agreement": {"counts": agreements, "pooled": agreement_pooled, "per_group": agreement_groups,
                      "definition": "tag team equal to the standing decode's team, over frames where the ball is seen, "
                                    "the standing decode (cost 14, no tags) is H or A and the tag state is not 0; "
                                    "pooled over every scored half and the three draws"},
        "acting": {"per_profile": acting, "counts": counts,
                   "definition": "acting: masked and tag state not 0; live: N = 3 target H or A; right: the state names "
                                 "the target's team"},
        "frames": far_table(frames, order), "frames_per_half": frames,
        "identity": identity, "development_t_sha256": digest(RECIPE / "development_t.json"),
        "caches": {part: digest(cache_root / "stack" / part / "manifest.json") for part in PARTS},
        "d59_member_manifests": {part: digest(D59_MEMBERS / part / "manifest.json") for part in PARTS},
        "teacher_sha256": diagnosis.teacher_identity(),
        "code": {name: digest(PACKAGE / name) for name in CODE},
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    write_new(output, development_record)
    write_new(decision_path, canonical(dict(
        decision, protocol=PROTOCOL, development_sha256=digest(output),
        rule={"against": "the adopted stack at cost 14 with no tags; scores averaged over three draws",
              "bounded_loss": {"profiles": list(SAFETY_PROFILES), "composite": -COMPOSITE_LOSS, "pff": -PFF_LOSS,
                               "guard_mean": -GUARD_LOSS, "group": -GROUP_LOSS, "groups": list(GROUPS),
                               "conventions": {name: -CONVENTION_LOSS for name in BOUNDED}},
              "selection": "the largest safe weight, not chosen by its gain",
              "gate": {"profile": GATE_PROFILE, "pff": PFF_GAIN, "guard_mean": GUARD_GAIN, "composite": 0.0},
              "theta": "the midpoint of the agreement share under Moderate and under Poor"},
        agreement_note="the agreement share does not depend on the weight: the standing decode and the simulated tags "
                       "are both computed without it, so it is the same for the selected weight and every other",
        decided_at=time.strftime("%Y-%m-%d %H:%M %Z"))))
    print(table(averaged_changes), flush=True)
    print(f"D62 safe weights: {decision['safe_weights']}; selected {decision['selected_weight']}; gate "
          f"{decision['gate']}; agreement {shares}; theta {decision['theta']:.4f}; {decision['outcome']}", flush=True)
    return decision


def table(averaged_changes):
    lines = ["| w | Profile | Composite | PFF | Guard mean | Largest group fall | N = 1.75 | N = 4 | N = 4.5 |",
             "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for weight, profiles in averaged_changes.items():
        for name in pe.PROFILE_ORDER:
            item = profiles[name]
            worst = min(GROUPS, key=lambda group: (item["groups"][group], group))
            lines.append(f"| {weight} | {name} | {item['composite']:+.4f} | {item['groups']['pff']:+.4f} | "
                         f"{item['guard_mean']:+.4f} | {worst} {item['groups'][worst]:+.4f} | "
                         + " | ".join(f"{item['conventions'][c]:+.4f}" for c in BOUNDED) + " |")
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
