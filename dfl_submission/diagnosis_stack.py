"""D58: where the adopted stack loses spellF1 now, and what kind of stoppage error costs most (DECISIONS.md D58).

`measure` decodes the adopted stack (D48: the 8,100-step stack of `stack_recipe/development_t.json`,
three seeds per slot, ownership cost 14) from the decoder inputs its development record was scored
from (`data/d48/t/cache/stack`: `development`, `metrica`, `d24`, `guards`). Nothing is forecast
here: a missing cache, or one whose identity differs from the record's, stops the run. It first
checks that the decode reproduces the record's cost-14 row exactly (D47's eight groups, D49's two
long-hidden guard views, PFF visibility, the D24 views and the Metrica games); otherwise it stops
without a record.

It then writes one record, `diagnosis_recipe/d58.json`, against the N = 3 targets:
- D21's four measurements, unchanged (`diagnosis.measure_all`), for twelve groups: DFL WPY clean,
  masked, dead ball carried and team swap; SkillCorner; PFF; Metrica; the three D24 views; and
  D49's two long-hidden guard views (DFL WPY, the three Metrica games);
- D58's stoppage measurements. A stoppage is a maximal run of D inside a graded half. A true
  stoppage is *found* when one predicted stoppage covers at least half of it; a predicted stoppage
  is *real* when at least half of it lies inside one true stoppage, and *invented* otherwise. Both
  are counted by length (under 3 s, 3-10 s, 10-30 s, 30 s and over), with, for the invented ones,
  the share whose ball is hidden on at least 80% of their frames;
- the two one-sided true-stoppage oracles (only adding missed stoppages; only removing invented
  ones), as mean spellF1 per group, at the level of whole stoppages (D58's amendment), with the rest
  of the full oracle's gain reported as "edges"; the frame-level split (all live-to-dead changes
  against all dead-to-live changes) is reported as supplementary;
- stop and restart offsets (predicted minus true frame, median and quartiles) of the found stoppages;
- the reading of D58's rule on the groups with a realistic dead ball, averaged over four families
  (D58's second amendment: dead ball carried; PFF; the D24 views as one; the guard views as one);
  the plain mean over the seven views is supplementary.

A frame's ball is seen when that view's tracking has a ball position there. The caches hold no
tracking, so the views are rebuilt from the external sources with the functions that made them,
through `diagnosis_d43` (unchanged) and `hidden_trial.guard_tracking`; each rebuilt half must have
the cached half's frame count and labels, the rebuilt D24 views must reproduce the D24 record's
visibility counts and the rebuilt guard views D49's guard statistics.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from dfl_submission import diagnosis, diagnosis_d43 as d46
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = diagnosis.RECIPE / "d58.json"
CACHE = ROOT / "data" / "d48" / "t" / "cache"
PROTOCOL = "DECISIONS.md D58"
CANDIDATE = "the adopted stack (D48, 8,100 steps, three seeds per slot)"
MODEL = "stack"
COST = 14
TARGET = d46.TARGET                  # N = 3
GUARD_GROUPS = ("guard_dfl", "guard_metrica")
GROUPS = (*d46.GROUPS, *GUARD_GROUPS)
VENDORS = d46.VENDORS | {group: (group,) for group in GUARD_GROUPS}
REALISTIC = ("dfl_dead_carried", "pff", *d46.D24_GROUPS, *GUARD_GROUPS)
FAMILIES = {"dfl_dead_carried": ("dfl_dead_carried",), "pff": ("pff",), "d24": d46.D24_GROUPS,
            "guards": GUARD_GROUPS}
FPS = diagnosis.FPS
LENGTH_BINS = (("under_3s", 0, 3 * FPS), ("3-10s", 3 * FPS, 10 * FPS), ("10-30s", 10 * FPS, 30 * FPS),
               ("over_30s", 30 * FPS, np.inf))          # frames, [low, high)
HIDDEN = (4, 5)                      # "ball hidden on at least 80% of the run's frames", as an exact ratio
GAIN_LINE = .04                      # "the true-stoppage oracle is worth at least 0.04 on average"
SIDE_SHARE = (2, 3)                  # "one side carries at least two thirds of that gain"
SIDES = ("add", "remove")            # adding missed stoppages; removing invented ones
READINGS = ("stoppage", "frame")
READING = "stoppage"                 # D58's amendment (17:36 EST): whole stoppages; the frame split is supplementary
AVERAGE = "families"                 # D58's second amendment (17:37 EST): four families; the seven views supplementary
TOLERANCE = 1e-12                    # D21's group means against the row's own (summation order only)


def canonical(value):
    return json.loads(json.dumps(value))


def read(path):
    return json.loads(Path(path).read_text())


# -- stoppages and the one-sided oracles -----------------------------------------------------------

def stoppage_runs(labels, runs):
    """[(start, stop)] of the maximal D runs inside the graded runs."""
    return [(start, stop) for start, stop, code in pm.segments(labels, runs) if code == pm.D]


def best_cover(item, others):
    """(index, overlap) of the run in `others` overlapping `item` most, the earliest on ties; (None, 0) if none."""
    if not others:
        return None, 0
    starts = np.array([start for start, _ in others], dtype=np.int64)
    stops = np.array([stop for _, stop in others], dtype=np.int64)
    overlap = np.maximum(0, np.minimum(stops, item[1]) - np.maximum(starts, item[0]))
    index = int(np.argmax(overlap))
    return (index, int(overlap[index])) if overlap[index] > 0 else (None, 0)


def at_least_half(part, whole):
    return 2 * part >= whole


def mostly_hidden(hidden, length):
    return HIDDEN[1] * hidden >= HIDDEN[0] * length


def stoppage_items(truth, prediction, visible):
    """The true and predicted stoppages of one match with D58's flags (truth is the target, halves joined).

    Each true stoppage: length, found (one predicted stoppage covers at least half of it), found by
    the union of predicted stoppages (a side count), and the stop and restart offsets against the
    predicted stoppage covering it most. Each predicted stoppage: length, real (at least half inside
    one true stoppage), real by the union of true stoppages (a side count), hidden-ball frames.
    """
    runs = pm.valid_runs(truth)
    true_runs, predicted_runs = stoppage_runs(truth, runs), stoppage_runs(prediction, runs)
    true_dead, predicted_dead = truth == pm.D, prediction == pm.D
    hidden = ~np.asarray(visible, dtype=bool)
    true_items, predicted_items = [], []
    for start, stop in true_runs:
        length = stop - start
        index, overlap = best_cover((start, stop), predicted_runs)
        found = index is not None and at_least_half(overlap, length)
        item = {"start": start, "stop": stop, "length": length, "found": bool(found),
                "found_union": bool(at_least_half(int(predicted_dead[start:stop].sum()), length)),
                "stop_offset": None, "restart_offset": None}
        if found:
            item["stop_offset"] = predicted_runs[index][0] - start
            item["restart_offset"] = predicted_runs[index][1] - stop
        true_items.append(item)
    for start, stop in predicted_runs:
        length = stop - start
        index, overlap = best_cover((start, stop), true_runs)
        hidden_frames = int(hidden[start:stop].sum())
        predicted_items.append({
            "start": start, "stop": stop, "length": length,
            "real": bool(index is not None and at_least_half(overlap, length)),
            "real_union": bool(at_least_half(int(true_dead[start:stop].sum()), length)),
            "hidden_frames": hidden_frames, "mostly_hidden": bool(mostly_hidden(hidden_frames, length))})
    return true_items, predicted_items


def span_mask(length, items):
    mask = np.zeros(length, dtype=bool)
    for item in items:
        mask[item["start"]:item["stop"]] = True
    return mask


def one_sided_oracles(truth, prediction, home, true_items, predicted_items):
    """D11's true-stoppage oracle, and its output taken only on one side of the errors, in both readings.

    stoppage reading: only on the frames of missed true stoppages (`add`), or only on the frames of
    invented predicted stoppages (`remove`); corrections of found or real stoppages are in neither.
    frame reading: only where the oracle turns live into D (`add`), or only where it turns D into
    live (`remove`); together they are the full oracle.
    """
    full = diagnosis.oracle_stoppage(truth, prediction, home)
    missed = span_mask(len(truth), [item for item in true_items if not item["found"]])
    invented = span_mask(len(truth), [item for item in predicted_items if not item["real"]])
    return {"full": full,
            "stoppage": {"add": np.where(missed, full, prediction), "remove": np.where(invented, full, prediction)},
            "frame": {"add": np.where(truth == pm.D, full, prediction),
                      "remove": np.where(truth != pm.D, full, prediction)}}


def stoppage_match(truth, prediction, home, visible):
    """D58's per-match items and oracle scores."""
    true_items, predicted_items = stoppage_items(truth, prediction, visible)
    oracles = one_sided_oracles(truth, prediction, home, true_items, predicted_items)
    scores = {"decoded": diagnosis.score(truth, prediction)["spell_f1"],
              "full": diagnosis.score(truth, oracles["full"])["spell_f1"]}
    for reading in READINGS:
        for side in SIDES:
            scores[f"{reading}_{side}"] = diagnosis.score(truth, oracles[reading][side])["spell_f1"]
    changed = oracles["full"] != prediction
    return {"true": true_items, "predicted": predicted_items, "oracle": scores,
            "full_changed": {"frames": int(changed.sum()), "to_dead": int((changed & (oracles["full"] == pm.D)).sum())}}


# -- summaries -------------------------------------------------------------------------------------

def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def length_bin(length):
    return next(name for name, low, high in LENGTH_BINS if low <= length < high)


def quartiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "median": float(np.median(values)), "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75))}


def by_length(true_items, predicted_items):
    """Found and real shares per length bin, and the hidden-ball share of the invented stoppages, plus all lengths."""
    result = {}
    for name in (*(item[0] for item in LENGTH_BINS), "all"):
        chosen_true = [item for item in true_items if name == "all" or length_bin(item["length"]) == name]
        chosen_predicted = [item for item in predicted_items if name == "all" or length_bin(item["length"]) == name]
        found = sum(item["found"] for item in chosen_true)
        real = sum(item["real"] for item in chosen_predicted)
        invented = [item for item in chosen_predicted if not item["real"]]
        hidden = sum(item["mostly_hidden"] for item in invented)
        result[name] = {"true": len(chosen_true), "found": int(found), "found_share": ratio(found, len(chosen_true)),
                        "predicted": len(chosen_predicted), "real": int(real),
                        "real_share": ratio(real, len(chosen_predicted)), "invented": len(invented),
                        "invented_mostly_hidden": int(hidden), "invented_hidden_share": ratio(hidden, len(invented)),
                        "true_frames": int(sum(item["length"] for item in chosen_true)),
                        "missed_frames": int(sum(item["length"] for item in chosen_true if not item["found"])),
                        "predicted_frames": int(sum(item["length"] for item in chosen_predicted)),
                        "invented_frames": int(sum(item["length"] for item in invented))}
    return result


def summarise_stoppages(records):
    true_items = [item for record in records for item in record["true"]]
    predicted_items = [item for record in records for item in record["predicted"]]
    found = [item for item in true_items if item["found"]]
    names = ("decoded", "full", *(f"{reading}_{side}" for reading in READINGS for side in SIDES))
    means = {name: float(np.mean([record["oracle"][name] for record in records])) for name in names}
    gains = {name: means[name] - means["decoded"] for name in names if name != "decoded"}
    return {"matches": len(records),
            "counts": {"true": len(true_items), "predicted": len(predicted_items)},
            "by_length": by_length(true_items, predicted_items),
            "union_disagreements": {"found": sum(item["found"] != item["found_union"] for item in true_items),
                                    "real": sum(item["real"] != item["real_union"] for item in predicted_items)},
            "offsets_frames": {"stop": quartiles([item["stop_offset"] for item in found]),
                               "restart": quartiles([item["restart_offset"] for item in found])},
            "oracle": {"spell_f1_mean": means, "gain": gains,
                       "edges": {reading: gains["full"] - sum(gains[f"{reading}_{side}"] for side in SIDES)
                                 for reading in READINGS},
                       "share_of_full_gain": {name: ratio(gains[name], gains["full"])
                                              for name in gains if name != "full"}},
            "full_changed_frames": {"frames": int(sum(record["full_changed"]["frames"] for record in records)),
                                    "to_dead": int(sum(record["full_changed"]["to_dead"] for record in records))}}


def group_keys(scored):
    result = {}
    for key in scored:
        source, _, view = key
        result.setdefault(source if source != "dfl" else f"dfl_{view}", []).append(key)
    return result


def measure_stoppages(scored, visible, groups=GROUPS):
    """D58's points 2-4 against the N = 3 targets, per group, and the standard combined aggregate."""
    keys = group_keys(scored)
    records = {key: stoppage_match(diagnosis.target(truth, diagnosis.TARGETS[TARGET]), prediction, home, visible[key])
               for key, (truth, prediction, home) in scored.items()}
    result = {group: summarise_stoppages([records[key] for key in sorted(keys[group])]) for group in groups}
    names = ("decoded", "full", *(f"{reading}_{side}" for reading in READINGS for side in SIDES))
    per_match = {key: {name: record["oracle"][name] for name in names}
                 for key, record in records.items() if key[0] in ("dfl", "skillcorner")}
    combined = {name: diagnosis.combined(per_match, name) for name in names} if per_match else None
    return result, combined


def gains_over(groups, members, reading):
    """The mean gains of the full oracle and of both sides over `members`, and each side's share of the full gain."""
    full = float(np.mean([groups[group]["oracle"]["gain"]["full"] for group in members]))
    sides = {side: float(np.mean([groups[group]["oracle"]["gain"][f"{reading}_{side}"] for group in members]))
             for side in SIDES}
    return {"full": full, **sides, "shares": {side: ratio(value, full) for side, value in sides.items()}}


def rule_reading(groups, *, reading, average, realistic=REALISTIC, families=FAMILIES):
    """D58's rule on the realistic-dead-ball groups, in the named reading; the other readings as supplements."""
    def one(reading_name, average_name):
        if average_name == "groups":
            values = gains_over(groups, realistic, reading_name)
        else:
            per_family = {name: gains_over(groups, members, reading_name) for name, members in families.items()}
            full = float(np.mean([item["full"] for item in per_family.values()]))
            sides = {side: float(np.mean([item[side] for item in per_family.values()])) for side in SIDES}
            values = {"full": full, **sides, "shares": {side: ratio(value, full) for side, value in sides.items()}}
        worth = bool(values["full"] >= GAIN_LINE)
        carried = [side for side in SIDES if values["shares"][side] is not None
                   and SIDE_SHARE[1] * values[side] >= SIDE_SHARE[0] * values["full"] and values["full"] > 0]
        edges = values["full"] - sum(values[side] for side in SIDES)
        return dict(values, edges=edges, edges_share=ratio(edges, values["full"]), worth_at_least=GAIN_LINE,
                    worth=worth, sides_at_two_thirds=carried, rule_for=(carried if worth and carried else None))

    if reading not in READINGS or average not in ("groups", "families"):
        raise ValueError("D58's reading and average must be named before anything is read")
    result = {"reading": reading, "average": average, "groups": list(realistic),
              "families": {name: list(members) for name, members in families.items()},
              "per_group": {group: {"full": groups[group]["oracle"]["gain"]["full"],
                                    **{side: groups[group]["oracle"]["gain"][f"{reading}_{side}"] for side in SIDES},
                                    "edges": groups[group]["oracle"]["edges"][reading]}
                            for group in realistic},
              "result": one(reading, average)}
    result["supplementary"] = {
        "note": "not what D58 is read on (D58's amendments): the other one-sided split and the other average",
        **{f"{name}/{mean}": one(name, mean) for name in READINGS for mean in ("groups", "families")
           if (name, mean) != (reading, average)}}
    outcome = result["result"]
    if outcome["rule_for"]:
        sides = " and ".join({"add": "missed", "remove": "invented"}[side] for side in outcome["rule_for"])
        result["outcome"] = f"a stoppage rule is written for the {sides} side"
    else:
        result["outcome"] = "round 13 ends with the adopted stack"
    return result


# -- the adopted stack's caches and the reproduction check -----------------------------------------

def guard_grouped(halves):
    """The guard halves with each source as its own group (`guard_dfl`, `guard_metrica`)."""
    return [replace(half, source=f"guard_{half.source}") for half in halves]


def stack_identity(record):
    """The adopted stack's model identity, recomputed as `stack_trial.develop_t` made it, checked against the record."""
    from dfl_submission import stack_trial as st
    variant = st.t_variant()                      # stops unless offset 0 is D52's T by digest
    config = st.read(st.RECIPE / variant["levers"])["config"]
    identity = st.model_identity(MODEL, config, st.T_SCRATCH / "networks", export=variant["export"],
                                 levers_name=variant["levers"], extra=variant["extra"])
    if canonical(identity) != record["identity"][MODEL]:
        raise ValueError("The stack's networks or code differ from the ones its development record names")
    return identity


def adopted_networks(identity, adoption):
    """Whether every network the record scored is, by its fit.json digest, the network the build ships."""
    copied = adoption["copied"]
    return all(copied[f"{offset}/{model_set}/{folder}"]["files"]["fit.json"] == value
               for offset, sets in identity["networks"].items() for model_set, folders in sets.items()
               for folder, value in folders.items())


def stack_halves(dfl, skill, parsed, games, identity, *, raw, cache_root=CACHE):
    """The stack's cached development, Metrica, D24 and guard halves (the identities of `stack_trial.develop`)."""
    from dfl_submission import hidden_trial as h49, metrica_sample as ms, metrica_views as mv, stack_trial as st
    from dfl_submission import vendor_trial
    folder = cache_root / MODEL
    tag = f"D48 {MODEL} {st.json_digest(identity)[:16]}"
    development = d46.cached(folder / "development", dict(
        identity, teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=raw,
        skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games)))
    metrica = d46.cached(folder / "metrica", dict(ms.reference_identity("D15", ms.PARSED), variant=tag))
    d24 = d46.cached(folder / "d24", dict(mv.identity(ms.PARSED, "D15"), variant=tag))
    guards = d46.cached(folder / "guards", dict(identity, guard=h49.guard_identity({"model": MODEL}, raw=raw)))
    return development, metrica, d24, guards


def stored_row(record, cost=COST):
    return next(item for item in record["entries"][MODEL] if item["owner_cost"] == cost)


def recomputed_row(development, metrica, d24, guards, recipe, truths, visibility):
    """The cost-14 row exactly as `stack_trial.develop` stored it."""
    from dfl_submission import hidden_trial as h49, stack_trial as st
    item = h49.entry(development, metrica, d24, guards, recipe, truths, visibility)
    return dict(h49.compact(item), selection=st.selection(item))


def compare_row(recomputed, stored):
    """Per-part exact comparison of the recomputed row with the stored one."""
    recomputed, stored = canonical(recomputed), canonical(stored)
    result = {name: recomputed.get(name) == stored.get(name) for name in sorted(stored)}
    result["whole_row"] = recomputed == stored
    return result


def row_group_means(row):
    """The row's own N = 3 spellF1 for each of D58's twelve groups."""
    from dfl_submission import metrica_sample as ms
    result = {name: row["groups"][name] for name in
              ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica")}
    for view in d46.D24_VIEWS:
        result[f"d24_{view}"] = float(np.mean([match[ms.PRIMARY]["spell_f1"] for match in row["d24_views"][view].values()]))
    result["guard_dfl"], result["guard_metrica"] = row["guards"]["a_dfl"], row["guards"]["b_metrica"]
    return result


def guard_visibility(tracking):
    from dfl_submission import hidden_views as hv
    return {(f"guard_{source}", match, hv.VIEW, index): (np.asarray(labels), d46.seen(ball))
            for (source, match, index), (labels, _, ball, _, _) in tracking.items()}


# -- the run ---------------------------------------------------------------------------------------

def measure(dfl, skill, parsed, *, raw, cache_root=CACHE, output=OUTPUT, reading=None, average=None):
    """D58, written once (DECISIONS.md D58)."""
    require_external_only()
    from dfl_submission import hidden_trial as h49, hyper_trial as ht, metrica_sample as ms, metrica_views as mv
    from dfl_submission import stack_trial as st, vendor_trial
    reading = READING if reading is None else reading
    average = AVERAGE if average is None else average
    if reading not in READINGS or average not in ("groups", "families"):
        raise ValueError("D58's one-sided reading and its average must be named before anything is decoded")
    if output.exists():
        raise FileExistsError(f"{output} exists; D58 is written once")
    games = vendor_trial.development()
    if not vendor_trial.gate_passed(vendor_trial.RECIPE / "sanity.json", vendor_trial.inputs(parsed, games)):
        raise ValueError("The latest PFF reader sanity attempt did not pass on these parsed inputs")
    if not ms.gate_passed(ms.RECIPE / "sanity.json", ms.inputs(ms.PARSED)):
        raise ValueError("The latest D19 reader attempt did not pass on these parsed inputs")
    record = read(st.RECIPE / "development_t.json")
    choice = record["amendment_choice"]
    if not (record["decision"]["selected_cost"] == COST and choice["stack"] == "8100" and choice["owner_cost"] == COST):
        raise ValueError("The development record does not name the 8,100-step stack at cost 14")
    identity = stack_identity(record)
    shipped = adopted_networks(identity, read(st.RECIPE / "adoption_models.json"))
    if not shipped:
        raise ValueError("The networks development scored are not the ones the build ships")
    recipe = ht.recipe_at(COST)
    development, metrica, d24, guards = stack_halves(dfl, skill, parsed, games, identity, raw=raw,
                                                     cache_root=cache_root)
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    pff_seen = h49.pff_visibility(parsed, games)
    row = recomputed_row(development, metrica, d24, guards, recipe, truths, pff_seen)
    stored = stored_row(record)
    checks = compare_row(row, stored)
    print("D58 reproduction of the cost-14 row: " + ", ".join(f"{name} {'ok' if ok else 'DIFFERS'}"
                                                             for name, ok in checks.items()), flush=True)
    if not all(checks.values()):
        raise ValueError(f"D58 pipeline check failed {checks}; no record was written")

    rebuilt = d46.development_visibility(dfl, skill, parsed, games, raw=raw)
    metrica_rebuilt, rows = d46.metrica_visibility()
    if canonical(rows) != read(mv.RECORD)["visibility"]:
        raise ValueError("The rebuilt D24 views do not reproduce the D24 record's visibility counts; nothing was written")
    tracking = h49.guard_tracking(dfl, raw=raw)
    if canonical(h49.guard_statistics(tracking)) != read(h49.RECIPE / "reference.json")["guard_views"]:
        raise ValueError("The rebuilt guard views do not reproduce D49's guard statistics; nothing was written")
    halves = development + metrica + d46.d24_grouped(d24) + guard_grouped(guards)
    visibility, frames = d46.attach(halves, rebuilt | d46.renamed(metrica_rebuilt) | guard_visibility(tracking))
    visible = d46.joined(halves, visibility, False)

    scored = diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe))
    report = diagnosis.measure_all(scored, groups=GROUPS, vendors=VENDORS, name="stack", with_shifts=False)
    expected = row_group_means(stored)
    differences = {group: abs(report["groups"][TARGET][group]["spell_f1_mean"] - expected[group]) for group in GROUPS}
    if max(differences.values()) > TOLERANCE:
        raise ValueError(f"D21's group means disagree with the reproduced row {differences}; nothing was written")
    groups, combined = measure_stoppages(scored, visible)
    for group in GROUPS:
        entry, mine = report["groups"][TARGET][group], groups[group]["oracle"]["spell_f1_mean"]
        if mine["decoded"] != entry["spell_f1_mean"] or mine["full"] != entry["oracle"]["stoppage"]:
            raise ValueError(f"D58's oracle scores disagree with D21's measurements for {group}")
    verdict = rule_reading(groups, reading=reading, average=average)
    d46.write_new(output, {
        "protocol": PROTOCOL, "candidate": CANDIDATE, "decision": None, "recipe": recipe,
        "reproduction": {"row": checks, "networks_shipped": shipped, "d24_visibility_counts": True,
                         "guard_statistics": True, "rebuilt_frames": frames,
                         "d21_group_means_max_difference": max(differences.values())},
        "tolerance_frames": diagnosis.TOLERANCE, **report,
        "stoppages": {"target": TARGET, "groups": groups, "combined": combined, "reading": verdict,
                      "length_bins_frames": {name: [low, None if not np.isfinite(high) else high]
                                             for name, low, high in LENGTH_BINS},
                      "hidden_share": HIDDEN[0] / HIDDEN[1],
                      "definitions": {
                          "stoppage": "a maximal run of D inside a graded half",
                          "found": "one predicted stoppage covers at least half of the true stoppage",
                          "real": "at least half of the predicted stoppage lies inside one true stoppage",
                          "invented": "a predicted stoppage that is not real",
                          "mostly_hidden": "the ball is hidden on at least 80% of the stoppage's frames",
                          "offsets": "predicted minus true start (stop) and end (restart) frame of the predicted "
                                     "stoppage covering a found true stoppage most",
                          "union": "side count only: found or real by the union of the other side's stoppages",
                          "oracle_stoppage_reading": "the full oracle's output only on the frames of missed true "
                                                     "stoppages (add), or of invented predicted stoppages (remove)",
                          "oracle_frame_reading": "supplementary only: the full oracle's output only on true-D frames "
                                                  "(add), or only on true-live frames (remove); together the full oracle",
                          "edges": "the full oracle's gain minus both sides' gains (corrections to the edges of "
                                   "found and real stoppages, and the non-additivity of spellF1)",
                          "average": "D58 is read on the mean over four families: dead ball carried; PFF; the three "
                                     "D24 views averaged; the two guard views averaged"}},
        "models": identity, "teacher_sha256": diagnosis.teacher_identity(),
        "caches": {name: digest(cache_root / MODEL / name / "manifest.json")
                   for name in ("development", "metrica", "d24", "guards")},
        "development_record_sha256": digest(st.RECIPE / "development_t.json"),
        "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                   "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": raw}})
    summary(report, groups, verdict)
    return report, groups, verdict


def summary(report, groups, verdict):
    reading = verdict["reading"]
    print(f"D58 per group, N = 3 (one-sided reading: {reading}):")
    for group in GROUPS:
        entry, extra = report["groups"][TARGET][group], groups[group]
        oracle = extra["oracle"]["spell_f1_mean"]
        print(f"  {group}: stack {entry['spell_f1_mean']:.4f}, true stoppages {oracle['full']:.4f}, add only "
              f"{oracle[f'{reading}_add']:.4f}, remove only {oracle[f'{reading}_remove']:.4f}, true ownership "
              f"{entry['oracle']['ownership']:.4f}, stoppages {extra['counts']['true']} true / "
              f"{extra['counts']['predicted']} predicted")
    result = verdict["result"]
    shares = ", ".join(f"{side} {'n/a' if value is None else f'{value:.3f}'}" for side, value in result["shares"].items())
    print(f"D58 realistic-dead-ball groups ({verdict['average']}): true-stoppage gain {result['full']:+.4f}, add "
          f"{result['add']:+.4f}, remove {result['remove']:+.4f}, edges {result['edges']:+.4f}; shares {shares} -> "
          f"{verdict['outcome']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure",))
    parser.add_argument("--parsed", type=Path, required=True, help="folder written by `dfl_submission.pff prepare`")
    parser.add_argument("--dfl", type=Path, required=True)
    parser.add_argument("--dfl-raw", action="store_true")
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE, help="the 8,100-step stack's cache root")
    args = parser.parse_args()
    require_external_only()
    measure(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache_root=args.cache)


if __name__ == "__main__":
    main()
