"""D46: where frozen D43 loses spellF1, and how much of it lies where the ball is hidden (DECISIONS.md D46).

`measure` decodes frozen D43 (D42's player-set networks, D3's trees, ownership cost 10) afresh from
the decoder inputs D42's development scoring cached (`data/d42/cache`: `development`, `metrica`,
`d24`) and first checks that it reproduces the cost-10 row of `player_recipe/development.json`
exactly, PFF, Metrica and D24 included. Nothing is forecast here: a missing cache, or one whose
identity differs, stops the run.

It then writes one record, `diagnosis_recipe/d46.json`:
- D21's four measurements, unchanged (`diagnosis.measure_all`), for ten groups: DFL WPY clean,
  masked, dead ball carried and team swap; SkillCorner; PFF; Metrica; the three D24 views;
- D46's ball-visibility measurements against the N = 3 targets: the share of target-live frames with
  the ball unseen; the decoded owner's agreement with the target on frames both call live, by seen /
  unseen and by time since the ball was last seen; recall at IoU 0.5 by the share of a true spell's
  frames with the ball seen; and D11's true-ownership oracle applied only on unseen-ball frames, and
  only on seen-ball frames.

A frame's ball is seen when that view's tracking has a ball position there. The caches hold no
tracking, so the same views are rebuilt from the external sources with the functions that made
them (`augment.guard_halves`, `pff.load`, `metrica_sample.load`, `metrica_views.build_views`); each
rebuilt half must have the cached half's frame count and labels, and the rebuilt D24 views must
reproduce the visibility counts stored in the D24 record.

`measure-hidden` (DECISIONS.md D46a) uses the same matches, views, decoding and checks and writes
`diagnosis_recipe/d46a.json` once: how long the hidden-ball runs that cover unseen target-live
frames are, the owner agreement on unseen frames by run length, and, for SkillCorner and PFF,
whether at least 16 players are visible on those frames.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
OUTPUT = diagnosis.RECIPE / "d46.json"
CACHE = ROOT / "data" / "d42" / "cache"
PROTOCOL = "DECISIONS.md D46"
CANDIDATE = "D43"
COST = 10
VARIANT = "D42 player set"          # the variant name D42's Metrica and D24 caches were made under
TARGET = "aba3"                     # D46's measurements are against the N = 3 targets
D24_VIEWS = ("masked", "handled", "team_swap")
D24_GROUPS = tuple(f"d24_{view}" for view in D24_VIEWS)
GROUPS = (*diagnosis.D21_GROUPS, *D24_GROUPS)
VENDORS = diagnosis.D21_VENDORS | {group: (group,) for group in D24_GROUPS}
DEGRADED = ("dfl_masked", "skillcorner", "pff", *D24_GROUPS)
SINCE = (("0", 0, 0), ("up_to_1s", 1, 25), ("1-3s", 26, 75), ("over_3s", 76, np.inf))  # frames since last seen
SEEN_SHARE = (("under_20%", 0.0, .2), ("20-80%", .2, .8), ("80%+", .8, np.inf))
HEADLINE_IOU = diagnosis.HEADLINE_IOU
LIVE = (pm.H, pm.A)


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D46 is written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value))


# -- the cached decoder inputs ---------------------------------------------------------------------

def refuse():
    raise ValueError("D46 forecasts nothing: the D42 cache is missing")
    yield  # pragma: no cover


def cached(folder, identity):
    """The halves of one D42 cache folder, after checking its manifest identity; never computed here."""
    if not (Path(folder) / "manifest.json").exists():
        raise ValueError(f"{folder} holds no D42 cache; D46 forecasts nothing")
    return diagnosis.collect(Path(folder), identity, refuse)


def development_identity(dfl, skill, parsed, games, *, raw, models):
    """`player_trial.candidate_halves`' cache identity for the development split."""
    from dfl_submission import player_trial, vendor_trial
    return dict(player_trial.identity(models), teacher_sha256=diagnosis.teacher_identity(), split="development",
                dfl_raw=raw, skill_manifest_sha256=digest(skill / "manifest.json"),
                pff=vendor_trial.inputs(parsed, games))


def cached_halves(dfl, skill, parsed, games, *, raw, cache_root=CACHE, models=None):
    """D43's cached development, Metrica and D24 halves (the identities of `player_trial.develop`)."""
    from dfl_submission import metrica_sample as ms, metrica_views as mv, player_trial
    models = player_trial.MODELS if models is None else models
    development = cached(cache_root / "development",
                         development_identity(dfl, skill, parsed, games, raw=raw, models=models))
    metrica = cached(cache_root / "metrica", dict(ms.reference_identity("D15", ms.PARSED), variant=VARIANT))
    d24 = cached(cache_root / "d24", dict(mv.identity(ms.PARSED, "D15"), variant=VARIANT))
    return development, metrica, d24


# -- the reproduction check ------------------------------------------------------------------------

def stored_row(record, cost=COST):
    return next(entry for entry in record["entries"] if entry["point"]["owner_cost"] == cost)


def row(development, metrica, d24, record, recipe, truths):
    """The cost-10 development row recomputed exactly as `player_trial.develop` made it.

    D15's development record and D24 mean are the row's own stored references; the Metrica veto
    decodes D15 from its cache, as D42 did.
    """
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    stored = stored_row(record)
    trial = aligned_trial.evaluate(development, recipe)
    change = refine_trial.changes(record["reference"], trial)
    veto = ms.veto(objective_trial.decoded(metrica, recipe), reference="D15")
    view_scores = mv.view_scores(d24, truths, lambda half: diagnosis.decode(half.signals, recipe))
    d24_entry = {"reference": stored["d24"]["reference"], "candidate": mv.d24_mean(view_scores)}
    d24_entry["change"] = d24_entry["candidate"] - d24_entry["reference"]
    allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24_entry["change"])
    return {"point": {"owner_cost": recipe["owner_cost"]}, "eligible": allowed, "changes": change, "trial": trial,
            "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
            "d24": dict(d24_entry, views=view_scores)}


def compare_row(recomputed, stored):
    """Per-part exact comparison of a recomputed row with the stored one."""
    recomputed, stored = canonical(recomputed), canonical(stored)
    parts = {"trial_external": ("trial", "external"), "trial_pff": ("trial", "pff"), "changes": ("changes",),
             "metrica": ("metrica",), "d24": ("d24",), "eligible": ("eligible",), "point": ("point",)}
    result = {}
    for name, path in parts.items():
        left, right = recomputed, stored
        for step in path:
            left, right = left.get(step) if isinstance(left, dict) else None, right.get(step)
        result[name] = left == right
    result["whole_row"] = recomputed == stored
    return result


# -- ball visibility -------------------------------------------------------------------------------

def seen(ball):
    return np.isfinite(np.asarray(ball, dtype=np.float64)).all(axis=1)


def frames_since_seen(visible):
    """Frames since the ball was last seen: 0 where seen, inf before the half's first seen frame."""
    index = np.arange(len(visible))
    last = np.maximum.accumulate(np.where(visible, index, -1))
    return np.where(last >= 0, index - last, np.inf).astype(np.float64)


def half_key(half):
    return half.source, half.match, half.view, half.index


def development_visibility(dfl, skill, parsed, games, *, raw):
    """{(source, match, view, index): (labels, seen)} of the rebuilt development views."""
    from dfl_submission import augment, pff
    result = {}
    for half in augment.guard_halves(dfl, skill, "development", raw=raw):
        result[half.source, half.match, half.view, half.index] = (np.asarray(half.labels), seen(half.ball))
    for game in games:
        for index, half in enumerate(pff.load(parsed, game)):
            result["pff", str(game), "observed", index] = (np.asarray(half["labels"]), seen(half["ball"]))
    return result


def metrica_visibility(parsed=None):
    """The rebuilt Metrica views, and the D24 visibility counts recomputed from them."""
    from dfl_submission import metrica_sample as ms, metrica_views as mv
    parsed = ms.PARSED if parsed is None else parsed
    result, rows = {}, {}
    for game in ms.GAMES:
        rows[str(game)] = []
        for index, half in enumerate(ms.load(parsed, game)):
            labels = np.asarray(half["labels"])
            result["metrica", str(game), "observed", index] = (labels, seen(half["ball"]))
            views = mv.build_views(half, game, index)
            for view, (_, ball) in views.items():
                result["metrica", str(game), view, index] = (labels, seen(ball))
            rows[str(game)].append(mv.visibility(half["labels"], views))
    return result, rows


def attach(halves, visibility):
    """{key: seen} for every cached half; stops unless the rebuilt half has its frame count and labels."""
    result, checks = {}, {}
    for half in halves:
        key = half_key(half)
        if key not in visibility:
            raise ValueError(f"No rebuilt view for the cached half {key}")
        labels, visible = visibility[key]
        same = len(labels) == len(half.labels) == len(visible) and np.array_equal(labels, half.labels)
        if not same:
            raise ValueError(f"The rebuilt view of {key} differs from the cached half; nothing was written")
        result[key] = visible
        checks["/".join(map(str, key))] = int(len(visible))
    return result, checks


def joined(halves, values, filler):
    """{match key: values of half 0, filler, values of half 1}, aligned with `diagnosis.matches`."""
    grouped = {}
    for half in halves:
        grouped.setdefault(half.key, []).append(half)
    result = {}
    for key, parts in grouped.items():
        if [part.index for part in parts] != [0, 1]:
            raise ValueError(f"Expected two halves in order for {key}")
        result[key] = np.r_[values[half_key(parts[0])], filler, values[half_key(parts[1])]]
    return result


def d24_grouped(halves):
    """The D24 halves with each view as its own source, so `diagnosis.measure_all` groups them apart."""
    return [replace(half, source=f"d24_{half.view}") for half in halves]


def renamed(visibility):
    return {(f"d24_{key[2]}", *key[1:]) if key[0] == "metrica" and key[2] in D24_VIEWS else key: value
            for key, value in visibility.items()}


# -- D46's measurements per match ------------------------------------------------------------------

def split_oracles(truth, prediction, visible):
    """D11's true-ownership oracle on every frame, on unseen-ball frames only, and on seen-ball frames only."""
    full = diagnosis.oracle_ownership(truth, prediction)
    return {"full": full, "unseen": np.where(visible, prediction, full), "seen": np.where(visible, full, prediction)}


def since_bin(since):
    labels = np.full(len(since), "", dtype=object)
    for name, low, high in SINCE:
        labels[(since >= low) & (since <= high)] = name
    return labels


def true_spells(truth, prediction):
    """[(start, stop, matched at IoU 0.5)] of the true spells, with the metric's greedy matching."""
    runs = pm.valid_runs(truth)
    spells, predicted = pm.possession_spells(truth, runs), pm.possession_spells(prediction, runs)
    matched, _ = diagnosis.matched_flags(pm.spell_overlaps(spells, predicted), len(spells), len(predicted),
                                         HEADLINE_IOU)
    return [(start, stop, bool(hit)) for (start, stop, _), hit in zip(spells, matched)]


def visibility_match(truth, prediction, visible, since):
    """D46's per-match counts, spells and oracle scores (truth is the target, both halves joined)."""
    live = np.isin(truth, LIVE)
    both = live & np.isin(prediction, LIVE)
    agree = both & (prediction == truth)
    bins = since_bin(since)
    agreement = {"seen": {"frames": int((both & visible).sum()), "agree": int((agree & visible).sum())},
                 "unseen": {"frames": int((both & ~visible).sum()), "agree": int((agree & ~visible).sum())}}
    by_since = {name: {"frames": int((both & (bins == name)).sum()), "agree": int((agree & (bins == name)).sum())}
                for name, _, _ in SINCE}
    oracles = split_oracles(truth, prediction, visible)
    changed = oracles["full"] != prediction
    scores = {"decoded": diagnosis.score(truth, prediction)["spell_f1"]}
    scores |= {name: diagnosis.score(truth, labels)["spell_f1"] for name, labels in oracles.items()}
    spells = [(float(visible[start:stop].mean()), stop - start, hit) for start, stop, hit in true_spells(truth, prediction)]
    return {"target_live": int(live.sum()), "target_live_unseen": int((live & ~visible).sum()),
            "agreement": agreement, "by_since": by_since, "oracle": scores,
            "oracle_changed": {"frames": int(changed.sum()), "unseen": int((changed & ~visible).sum())},
            "spells": spells}


# -- summaries -------------------------------------------------------------------------------------

def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def pooled_agreement(items):
    frames, agree = sum(item["frames"] for item in items), sum(item["agree"] for item in items)
    return {"frames": int(frames), "agreement": ratio(agree, frames)}


def seen_share_recall(spells):
    """Recall at IoU 0.5 by the share of a true spell's frames with the ball seen, and by that share × length."""
    shares = np.array([share for share, _, _ in spells], dtype=np.float64)
    lengths = np.array([length for _, length, _ in spells], dtype=np.float64) / diagnosis.FPS
    hits = np.array([hit for _, _, hit in spells], dtype=bool)
    result, crossed = {}, {}
    for name, low, high in SEEN_SHARE:
        inside = (shares >= low) & (shares < high)
        result[name] = {"count": int(inside.sum()), "recall_iou0.5": ratio(float(hits[inside].sum()), int(inside.sum()))}
        crossed[name] = {}
        for short, long in zip(diagnosis.LENGTH_BINS[:-1], diagnosis.LENGTH_BINS[1:]):
            label = f"{short:g}-{long:g}s" if np.isfinite(long) else f"{short:g}s+"
            cell = inside & (lengths >= short) & (lengths < long)
            crossed[name][label] = {"count": int(cell.sum()),
                                    "recall_iou0.5": ratio(float(hits[cell].sum()), int(cell.sum()))}
    return result, crossed


def summarise_visibility(records):
    live = sum(record["target_live"] for record in records)
    unseen = sum(record["target_live_unseen"] for record in records)
    spells = [spell for record in records for spell in record["spells"]]
    by_share, crossed = seen_share_recall(spells)
    oracle = {name: float(np.mean([record["oracle"][name] for record in records]))
              for name in ("decoded", "full", "unseen", "seen")}
    gains = {name: oracle[name] - oracle["decoded"] for name in ("full", "unseen", "seen")}
    changed = sum(record["oracle_changed"]["frames"] for record in records)
    return {"matches": len(records), "target_live_frames": int(live), "target_live_unseen_share": ratio(unseen, live),
            "agreement": {side: pooled_agreement([record["agreement"][side] for record in records])
                          for side in ("seen", "unseen")},
            "agreement_by_time_since_seen": {name: pooled_agreement([record["by_since"][name] for record in records])
                                             for name, _, _ in SINCE},
            "recall_by_seen_share": by_share, "recall_by_seen_share_and_length": crossed,
            "split_oracle": {"spell_f1_mean": oracle, "gain": gains,
                             "unseen_share_of_full_gain": ratio(gains["unseen"], gains["full"]),
                             "seen_share_of_full_gain": ratio(gains["seen"], gains["full"])},
            "full_oracle_changed_frames": {"frames": int(changed), "unseen_share": ratio(
                sum(record["oracle_changed"]["unseen"] for record in records), changed)}}


def degraded_shares(groups, degraded=DEGRADED):
    """The unseen-only oracle's gain as a share of the full ownership oracle's gain on the degraded groups."""
    per_group = {group: groups[group]["split_oracle"]["unseen_share_of_full_gain"] for group in degraded}
    gains = {group: groups[group]["split_oracle"]["gain"] for group in degraded}
    d24 = [group for group in degraded if group in D24_GROUPS]
    families = [group for group in degraded if group not in D24_GROUPS]
    family_gains = [gains[group] for group in families]
    if d24:
        family_gains.append({name: float(np.mean([gains[group][name] for group in d24])) for name in ("full", "unseen")})
    return {"per_group": per_group,
            "d24_mean": ratio(sum(gains[group]["unseen"] for group in d24), sum(gains[group]["full"] for group in d24))
            if d24 else None,
            "pooled_groups": ratio(sum(item["unseen"] for item in gains.values()),
                                   sum(item["full"] for item in gains.values())),
            "pooled_families": ratio(sum(item["unseen"] for item in family_gains),
                                     sum(item["full"] for item in family_gains)),
            "groups": list(degraded), "at_least_half": {group: (share is not None and share >= .5)
                                                        for group, share in per_group.items()}}


def split_combined(records):
    """The standard combined aggregate of the decoded labels and the three oracles (DFL and SkillCorner only)."""
    per_match = {key: {name: record["oracle"][name] for name in ("decoded", "full", "unseen", "seen")}
                 for key, record in records.items() if key[0] in ("dfl", "skillcorner")}
    if not per_match:
        return None
    return {name: diagnosis.combined(per_match, name) for name in ("decoded", "full", "unseen", "seen")}


def measure_visibility(scored, visible, since, groups=GROUPS):
    """D46's points 2-4 against the N = 3 targets, per group."""
    by_group = {}
    for key in scored:
        source, _, view = key
        by_group.setdefault(source if source != "dfl" else f"dfl_{view}", []).append(key)
    records = {key: visibility_match(diagnosis.target(truth, diagnosis.TARGETS[TARGET]), prediction, visible[key],
                                     since[key])
               for key, (truth, prediction, _) in scored.items()}
    result = {group: summarise_visibility([records[key] for key in sorted(by_group[group])]) for group in groups}
    return result, split_combined(records)


# -- the run ---------------------------------------------------------------------------------------

def measure(dfl, skill, parsed, *, raw, cache_root=CACHE, output=OUTPUT):
    """D46, written once (DECISIONS.md D46)."""
    require_external_only()
    from dfl_submission import hyper_trial as ht, metrica_sample as ms, metrica_views as mv, player_trial, vendor_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D46 is written once")
    games = vendor_trial.development()
    if not vendor_trial.gate_passed(vendor_trial.RECIPE / "sanity.json", vendor_trial.inputs(parsed, games)):
        raise ValueError("The latest PFF reader sanity attempt did not pass on these parsed inputs")
    if not ms.gate_passed(ms.RECIPE / "sanity.json", ms.inputs(ms.PARSED)):
        raise ValueError("The latest D19 reader attempt did not pass on these parsed inputs")
    record = json.loads((player_trial.RECIPE / "development.json").read_text())
    if record["models"] != player_trial.identity():
        raise ValueError("The D42 networks differ from the ones development scored; nothing was written")
    recipe = ht.recipe_at(COST)
    development, metrica, d24 = cached_halves(dfl, skill, parsed, games, raw=raw, cache_root=cache_root)
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    checks = compare_row(row(development, metrica, d24, record, recipe, truths), stored_row(record))
    print("D46 reproduction of the cost-10 row: " + ", ".join(f"{name} {'ok' if ok else 'DIFFERS'}"
                                                             for name, ok in checks.items()), flush=True)
    if not all(checks.values()):
        raise ValueError(f"D46 pipeline check failed {checks}; no record was written")

    rebuilt = development_visibility(dfl, skill, parsed, games, raw=raw)
    metrica_rebuilt, rows = metrica_visibility()
    stored_rows = json.loads(mv.RECORD.read_text())["visibility"]
    if canonical(rows) != stored_rows:
        raise ValueError("The rebuilt D24 views do not reproduce the D24 record's visibility counts; nothing was written")
    halves = development + metrica + d24_grouped(d24)
    visibility, frames = attach(halves, rebuilt | renamed(metrica_rebuilt))
    since = {key: frames_since_seen(value) for key, value in visibility.items()}
    visible = joined(halves, visibility, False)
    elapsed = joined(halves, since, np.inf)

    scored = diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe))
    report = diagnosis.measure_all(scored, groups=GROUPS, vendors=VENDORS, name="d43", with_shifts=False)
    groups, combined = measure_visibility(scored, visible, elapsed)
    for group in GROUPS:
        expected = report["groups"][TARGET][group]
        mine = groups[group]["split_oracle"]["spell_f1_mean"]
        if mine["decoded"] != expected["spell_f1_mean"] or mine["full"] != expected["oracle"]["ownership"]:
            raise ValueError(f"D46's split-oracle scores disagree with D21's measurements for {group}")
    shares = degraded_shares(groups)
    write_new(output, {
        "protocol": PROTOCOL, "candidate": CANDIDATE, "decision": None, "recipe": recipe,
        "reproduction": {"row": checks, "d24_visibility_counts": True, "rebuilt_frames": frames},
        "tolerance_frames": diagnosis.TOLERANCE, **report,
        "visibility": {"target": TARGET, "groups": groups, "combined": combined, "degraded_shares": shares,
                       "time_since_seen_frames": {name: [low, None if not np.isfinite(high) else high]
                                                  for name, low, high in SINCE},
                       "seen_share_bins": {name: [low, None if not np.isfinite(high) else high]
                                           for name, low, high in SEEN_SHARE}},
        "models": player_trial.identity(), "teacher_sha256": diagnosis.teacher_identity(),
        "caches": {name: digest(cache_root / name / "manifest.json") for name in ("development", "metrica", "d24")},
        "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                   "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": raw}})
    summary(report, groups, combined, shares)
    return report, groups, shares


def summary(report, groups, combined, shares):
    values = report["combined"][TARGET]
    print(f"D46 N = 3 standard combined: D43 {values['d43']['combined']:.4f}, true stoppages "
          f"{values['oracle_stoppage']['combined']:.4f}, true ownership {values['oracle_ownership']['combined']:.4f}, "
          f"unseen-only {combined['unseen']['combined']:.4f}, seen-only {combined['seen']['combined']:.4f}")
    for group in GROUPS:
        entry, extra = report["groups"][TARGET][group], groups[group]
        oracle = extra["split_oracle"]["spell_f1_mean"]
        share = extra["split_oracle"]["unseen_share_of_full_gain"]
        print(f"  {group}: spellF1 {entry['spell_f1_mean']:.4f}, stoppages {entry['oracle']['stoppage']:.4f}, "
              f"ownership {entry['oracle']['ownership']:.4f}, unseen-only {oracle['unseen']:.4f}, seen-only "
              f"{oracle['seen']:.4f}, live unseen {extra['target_live_unseen_share']:.3f}, unseen gain share "
              f"{'n/a' if share is None else f'{share:.3f}'}")
    print("D46 unseen-only share of the ownership gain, degraded groups: " + ", ".join(
        f"{group} {'n/a' if value is None else f'{value:.3f}'}" for group, value in shares["per_group"].items())
        + f"; pooled {shares['pooled_groups']:.3f}")


# -- D46a: hidden-run lengths and visible players --------------------------------------------------

D46A_OUTPUT = diagnosis.RECIPE / "d46a.json"
D46A_PROTOCOL = "DECISIONS.md D46a"
RUN_BINS = (("up_to_1s", 1, 25), ("1-3s", 26, 75), ("3-10s", 76, 250), ("over_10s", 251, np.inf))  # run length, frames
PLAYERS = 16                         # "at least 16 visible players (both teams)"
PLAYER_GROUPS = ("skillcorner", "pff")


def hidden_run_lengths(visible):
    """Per frame, the length of the maximal hidden run it lies in (0 where the ball is seen), within one half."""
    hidden = ~np.asarray(visible, dtype=bool)
    result = np.zeros(len(hidden), dtype=np.int64)
    edges = np.flatnonzero(np.diff(np.r_[0, hidden.astype(np.int8), 0]))
    for start, stop in zip(edges[::2], edges[1::2]):
        result[start:stop] = stop - start
    return result


def visible_players(positions):
    return np.isfinite(np.asarray(positions, dtype=np.float64)).all(axis=-1).sum(axis=-1).astype(np.int64)


def run_bin(lengths):
    labels = np.full(len(lengths), "", dtype=object)
    for name, low, high in RUN_BINS:
        labels[(lengths >= low) & (lengths <= high)] = name
    return labels


def hidden_match(truth, prediction, visible, lengths, players=None):
    """D46a's per-match counts (truth is the target, both halves joined)."""
    live = np.isin(truth, LIVE)
    unseen_live = live & ~visible
    both = unseen_live & np.isin(prediction, LIVE)
    agree = both & (prediction == truth)
    bins = run_bin(lengths)
    record = {"unseen_live_lengths": lengths[unseen_live].tolist(),
              "by_run": {name: {"unseen_live": int((unseen_live & (bins == name)).sum()),
                                "frames": int((both & (bins == name)).sum()),
                                "agree": int((agree & (bins == name)).sum())} for name, _, _ in RUN_BINS}}
    if players is not None:
        many = players >= PLAYERS
        record["players"] = {
            "unseen_live": int(unseen_live.sum()), "unseen_live_many": int((unseen_live & many).sum()),
            "agreement": {"many": {"frames": int((both & many).sum()), "agree": int((agree & many).sum())},
                          "few": {"frames": int((both & ~many).sum()), "agree": int((agree & ~many).sum())}},
            "spells": [(float(visible[start:stop].mean()), float(players[start:stop].mean()), hit)
                       for start, stop, hit in true_spells(truth, prediction)]}
    return record


def summarise_hidden(records):
    lengths = np.array([value for record in records for value in record["unseen_live_lengths"]], dtype=np.float64)
    total = len(lengths)
    result = {"unseen_live_frames": int(total),
              "share_by_run": {name: ratio(sum(record["by_run"][name]["unseen_live"] for record in records), total)
                               for name, _, _ in RUN_BINS},
              "run_length_s": ({"median": float(np.median(lengths)) / diagnosis.FPS,
                                "p90": float(np.percentile(lengths, 90)) / diagnosis.FPS} if total else None),
              "agreement_by_run": {name: pooled_agreement([record["by_run"][name] for record in records])
                                   for name, _, _ in RUN_BINS}}
    if all("players" in record for record in records):
        unseen = sum(record["players"]["unseen_live"] for record in records)
        spells = [spell for record in records for spell in record["players"]["spells"] if spell[0] < SEEN_SHARE[0][2]]
        recall = {}
        for name, keep in (("many", lambda count: count >= PLAYERS), ("few", lambda count: count < PLAYERS)):
            chosen = [hit for _, count, hit in spells if keep(count)]
            recall[name] = {"count": len(chosen), "recall_iou0.5": ratio(float(sum(chosen)), len(chosen))}
        result["players"] = {
            "unseen_live_share_many": ratio(sum(record["players"]["unseen_live_many"] for record in records), unseen),
            "agreement_unseen": {side: pooled_agreement([record["players"]["agreement"][side] for record in records])
                                 for side in ("many", "few")},
            "under_20%_seen_spell_recall": recall}
    return result


def development_players(dfl, skill, parsed, games, *, raw, sources=PLAYER_GROUPS):
    """{(source, match, view, index): (labels, visible players)} of the rebuilt SkillCorner and PFF halves."""
    from dfl_submission import augment, pff
    result = {}
    if "skillcorner" in sources:
        for half in augment.guard_halves(dfl, skill, "development", raw=raw):
            if half.source == "skillcorner":
                result[half.source, half.match, half.view, half.index] = (np.asarray(half.labels),
                                                                          visible_players(half.positions))
    if "pff" in sources:
        for game in games:
            for index, half in enumerate(pff.load(parsed, game)):
                result["pff", str(game), "observed", index] = (np.asarray(half["labels"]),
                                                               visible_players(half["positions"]))
    return result


def measure_hidden(dfl, skill, parsed, *, raw, cache_root=CACHE, output=D46A_OUTPUT):
    """D46a, written once: D46's matches and views, D43 decoded as in D46 after the same checks."""
    require_external_only()
    from dfl_submission import hyper_trial as ht, metrica_sample as ms, metrica_views as mv, player_trial, vendor_trial
    if output.exists():
        raise FileExistsError(f"{output} exists; D46a is written once")
    games = vendor_trial.development()
    record = json.loads((player_trial.RECIPE / "development.json").read_text())
    if record["models"] != player_trial.identity():
        raise ValueError("The D42 networks differ from the ones development scored; nothing was written")
    recipe = ht.recipe_at(COST)
    development, metrica, d24 = cached_halves(dfl, skill, parsed, games, raw=raw, cache_root=cache_root)
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    checks = compare_row(row(development, metrica, d24, record, recipe, truths), stored_row(record))
    if not all(checks.values()):
        raise ValueError(f"D46a pipeline check failed {checks}; no record was written")
    rebuilt = development_visibility(dfl, skill, parsed, games, raw=raw)
    metrica_rebuilt, rows = metrica_visibility()
    if canonical(rows) != json.loads(mv.RECORD.read_text())["visibility"]:
        raise ValueError("The rebuilt D24 views do not reproduce the D24 record's visibility counts; nothing was written")
    halves = development + metrica + d24_grouped(d24)
    visibility, frames = attach(halves, rebuilt | renamed(metrica_rebuilt))
    player_halves = [half for half in halves if half.source in PLAYER_GROUPS]
    players, _ = attach(player_halves, development_players(dfl, skill, parsed, games, raw=raw))
    lengths = {key: hidden_run_lengths(value) for key, value in visibility.items()}
    visible = joined(halves, visibility, False)
    run = joined(halves, lengths, 0)
    counts = joined(player_halves, players, 0)
    scored = diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe))
    by_group = {}
    for key in scored:
        by_group.setdefault(key[0] if key[0] != "dfl" else f"dfl_{key[2]}", []).append(key)
    records = {key: hidden_match(diagnosis.target(truth, diagnosis.TARGETS[TARGET]), prediction, visible[key], run[key],
                                 counts.get(key))
               for key, (truth, prediction, _) in scored.items()}
    groups = {group: summarise_hidden([records[key] for key in sorted(by_group[group])]) for group in GROUPS}
    write_new(output, {"protocol": D46A_PROTOCOL, "candidate": CANDIDATE, "decision": None, "recipe": recipe,
                       "target": TARGET, "groups": groups,
                       "run_bins_frames": {name: [low, None if not np.isfinite(high) else high]
                                           for name, low, high in RUN_BINS},
                       "players_threshold": PLAYERS, "player_groups": list(PLAYER_GROUPS),
                       "reproduction": {"row": checks, "d24_visibility_counts": True, "rebuilt_frames": frames},
                       "models": player_trial.identity(),
                       "caches": {name: digest(cache_root / name / "manifest.json")
                                  for name in ("development", "metrica", "d24")},
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                                  "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": raw}})
    for group in GROUPS:
        entry = groups[group]
        shares = ", ".join(f"{name} {'n/a' if value is None else f'{value:.3f}'}"
                           for name, value in entry["share_by_run"].items())
        lengths_s = entry["run_length_s"]
        print(f"D46a {group}: unseen live frames {entry['unseen_live_frames']}, shares {shares}"
              + ("" if lengths_s is None else f", median {lengths_s['median']:.2f} s, p90 {lengths_s['p90']:.2f} s"))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("measure", "measure-hidden"))
    parser.add_argument("--parsed", type=Path, required=True, help="folder written by `dfl_submission.pff prepare`")
    parser.add_argument("--dfl", type=Path, required=True)
    parser.add_argument("--dfl-raw", action="store_true")
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE, help="D42's cache root (development, metrica, d24)")
    args = parser.parse_args()
    require_external_only()
    if args.command == "measure-hidden":
        measure_hidden(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache_root=args.cache)
        return
    measure(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache_root=args.cache)


if __name__ == "__main__":
    main()
