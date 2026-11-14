"""D61: D57's networks at three seeds (candidate Q) to the fresh PFF games, confirmed once (DECISIONS.md D61).

Q is D60's twelve networks (D57's four SZ networks as offset 0, D60's eight for offsets 1,000 and 2,000) with D3's
trees, the standing mixes, D43's cutoff, stoppage cost and discount, at ownership cost 14: only the networks differ
from the adopted stack. The confirmation follows D48's (`stack_trial.confirm_prepare` / `confirm`) with PFF set D in
place of the untouched set and the adopted stack in place of D43.

Commands (each with `MTAH_DATA_DIR` set to a nonexistent path):
- `reproduce --dfl DIR --skill DIR`: before any score of Q, the adopted stack (its confirmed twelve networks in
  `stack_models/`) is forecast again on the holdout and the two long-hidden holdout views (D48's seeds); every half
  must equal D48's cached decoder inputs bit for bit, and the holdout numbers (primary, conventions, dead ball carried,
  team swap, raw) and the long-hidden scores at cost 14 must equal `confirmation.json`'s. `data/d61/reproduction.json`.
- `prepare --dfl DIR --skill DIR --parsed DIR --pff-raw DIR`: set D's sixteen games, checked against D7's first and
  second `train` slots and against the parsed folder, parsed now; every half's attacking sign (D50's rule, stopping on a
  first-frame disagreement as D48 did); Q's decoder inputs on the holdout, the long-hidden views and set D, and the
  stack's on set D, cached. Nothing is scored. `data/d61/prepared.json`.
- `confirm --dfl DIR --skill DIR --parsed DIR`: from the caches only, the adopted stack first, then Q once; D61's rule;
  `stack_recipe/confirmation_d61.json` written once.
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
SCRATCH = ROOT / "data" / "d61"
PROTOCOL = "DECISIONS.md D61"
COST = 14
GAIN, LOSS = .01, .01
SET_D = (3816, 3824, 3832, 3840, 3848, 3856, 10506, 10514, 3817, 3825, 3833, 3841, 3849, 3857, 10507, 10515)
SET_E = (3818, 3826, 3834, 3842, 3850, 3858, 10508, 10516)
D48_CACHE = st.CONFIRM_CACHE


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def train_slots(ids):
    """{1, 2, 3: games} of D7's three `train` slots (`pff.split` deals sorted ids in turn to SPLIT_ORDER's slots)."""
    from dfl_submission import pff
    ordered = sorted(int(game) for game in ids)
    positions = [index for index, name in enumerate(pff.SPLIT_ORDER) if name == "train"]
    return {slot + 1: tuple(game for rank, game in enumerate(ordered) if rank % len(pff.SPLIT_ORDER) == position)
            for slot, position in enumerate(positions)}


def pinned_ids(sources=None):
    from dfl_submission import pff
    pinned = read(pff.SOURCES if sources is None else sources)["files"]
    return {int(Path(name).name.split(".")[0]) for name in pinned}


def check_set_d(ids):
    """Set D is D7's first and second `train` slots and set E its third, as the amendment of 16:23 EST names them."""
    from dfl_submission import pff
    slots = train_slots(ids)
    merged = pff.split(ids)["train"]
    return {"set_d_is_slots_1_and_2": sorted(SET_D) == sorted(slots[1] + slots[2]),
            "set_e_is_slot_3": sorted(SET_E) == sorted(slots[3]),
            "slots_are_pff_split_train": sorted(slots[1] + slots[2] + slots[3]) == sorted(merged),
            "slots": {str(key): list(value) for key, value in slots.items()}}


def rule(change):
    """D61's conditions on the changes (Q minus the adopted stack, both at cost 14)."""
    return {"set_d": bool(change["pff"] >= GAIN), "primary": bool(change["primary"] >= -LOSS),
            "long_hidden": bool(change["long_hidden"] >= -LOSS),
            "conventions": bool(all(value >= -LOSS for value in change["conventions"].values())),
            "guards": bool(all(value >= -LOSS for value in change["guards"].values()))}


# -- the models ----------------------------------------------------------------------------------------

def stack_models():
    """(config, models, identity) of the adopted stack, checked against its confirmed identity (`tree_share.adopted`)."""
    from dfl_submission import tree_share as ts
    config, _, models, identity = ts.adopted()
    return config, models, identity


def q_models():
    """(models, identity) of Q: D60's twelve networks (offset 0 checked against D57's record) with D3's trees."""
    from dfl_submission import combined_trial as ct, height_fit as hf, scale_trial
    networks = ct.network_identity()
    raw = {str(offset): {key: {folder: digest((hf.raw_folder(key, folder) if offset == 0 else
                                               ct.raw_folder(offset, key, folder)) / "fit.json")
                               for folder in st.NETWORKS} for key in st.SETS} for offset in st.OFFSETS}
    check = read(RECIPE / "d60_check.json")
    if check["passed"] is not True:
        raise SystemExit("D61 stops: D60's trainer check did not pass")
    identity = {"networks": networks, "raw_fit_sha256": raw, "config": ct.config(0),
                "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in st.SETS},
                "d57_prepare": {key: digest(RECIPE / f"d57_prepare_{key}.json") for key in st.SETS},
                "d60_check_sha256": digest(RECIPE / "d60_check.json"),
                "d60_development_sha256": digest(RECIPE / "d60_development.json")}
    return ct.p_models(), identity


def code_identity():
    names = ("confirm_d61.py", "combined_trial.py", "tree_share.py", "height_fit.py", "skill_height.py")
    return st.code_identity() | {name: digest(PACKAGE / name) for name in names}


def guard_halves(long_hidden, signals_of):
    from dfl_submission import diagnosis, hidden_views as hv
    for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(long_hidden.items()):
        signals, decoded = signals_of(None, positions, ball, teams, pitch)
        yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)


def same_halves(fresh, stored):
    """Per half key: decoder inputs and labels bit for bit."""
    by_key = {(h.source, h.match, h.view, h.index): h for h in stored}
    result = {}
    for half in fresh:
        key = (half.source, half.match, half.view, half.index)
        other = by_key.get(key)
        result["/".join(map(str, key))] = bool(other is not None and np.array_equal(half.labels, other.labels)
                                               and np.array_equal(half.d3, other.d3)
                                               and all(np.array_equal(a, b) for a, b in zip(half.signals, other.signals,
                                                                                             strict=True)))
    return result


# -- 1. the adopted stack reproduces its holdout numbers --------------------------------------------------

def reproduce(dfl, skill, *, scratch=SCRATCH, output=None):
    """The adopted stack forecast again on the holdout and its long-hidden views, against D48's confirmation."""
    require_external_only()
    from dfl_submission import aligned_trial, augment, diagnosis, hidden_views as hv, hyper_trial as ht
    output = scratch / "reproduction.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; the reproduction runs once")
    confirmation = read(RECIPE / "confirmation.json")
    config, models, identity = stack_models()
    long_hidden = st.holdout_long_hidden(dfl, raw=True)

    def items():
        for half in augment.guard_halves(dfl, skill, "holdout", raw=True):
            yield "holdout", (half.source, half.match, half.view, half.index), half.positions, half.ball, half.teams, half.pitch
        for (source, match, index), (_, positions, ball, teams, pitch) in sorted(long_hidden.items()):
            yield "holdout_long_hidden", (source, match, hv.VIEW, index), positions, ball, teams, pitch

    table = st.sign_table(items())
    forecaster = st.Forecaster(models, table, with_d43=False)
    forecast = forecaster.signals_of("stack")
    key = {"protocol": PROTOCOL, "model": "the adopted stack", "identity": identity,
           "teacher_sha256": diagnosis.teacher_identity(), "skill_manifest_sha256": digest(skill / "manifest.json"),
           "long_hidden_seed": st.HOLDOUT_GUARD_SEED, "hidden_views_sha256": digest(PACKAGE / "hidden_views.py"),
           "dfl_raw": True}
    holdout = diagnosis.collect(scratch / "stack_holdout", key,
                                lambda: diagnosis.external_halves(dfl, skill, "holdout", None, raw=True, signals_of=forecast))
    long = diagnosis.collect(scratch / "stack_long_hidden", key, lambda: guard_halves(long_hidden, forecast))
    stored_main = [half for half in diagnosis.collect(D48_CACHE / "stack", read(D48_CACHE / "prepared.json")["keys"]["stack"],
                                                      st.refuse) if half.source != "pff"]
    stored_long = diagnosis.collect(D48_CACHE / "stack_long_hidden", read(D48_CACHE / "prepared.json")["keys"]["stack"],
                                    st.refuse)
    bits = {"holdout": same_halves(holdout, stored_main), "long_hidden": same_halves(long, stored_long)}
    trial = aligned_trial.evaluate(holdout, ht.recipe_at(COST))
    long_scores = st.long_hidden_scores(long, ht.recipe_at(COST))
    numbers = {"holdout_external_equal": canonical(trial["external"]) == canonical(confirmation["trial"]["external"]),
               "long_hidden_equal": canonical(long_scores) == canonical(confirmation["long_hidden"]["stack"]),
               "same_number_of_halves": len(holdout) == len(stored_main) and len(long) == len(stored_long)}
    passed = bool(all(numbers.values()) and all(all(part.values()) for part in bits.values()))
    record = {"protocol": PROTOCOL, "passed": passed, "numbers": numbers, "bit_for_bit": bits,
              "primary": trial["external"]["conventions"]["aba3"]["standard"]["spell_f1"],
              "long_hidden": long_scores, "identity": identity,
              "caches": {name: digest(scratch / name / "manifest.json") for name in ("stack_holdout", "stack_long_hidden")},
              "signs": {"fallbacks": sorted(row["half"] for row in table.values() if row["fallback"])},
              "confirmation_sha256": digest(RECIPE / "confirmation.json"), "key": key,
              "reproduced_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, canonical(record))
    print(f"D61 reproduction: the adopted stack {'reproduces' if passed else 'DOES NOT reproduce'} D48's holdout numbers "
          f"{numbers}; halves bit for bit: holdout {sum(bits['holdout'].values())}/{len(bits['holdout'])}, long-hidden "
          f"{sum(bits['long_hidden'].values())}/{len(bits['long_hidden'])}", flush=True)
    if not passed:
        raise SystemExit("D61 stops: the adopted stack does not reproduce its holdout numbers; nothing of Q is scored")
    return record


# -- 2. set D and the decoder inputs ------------------------------------------------------------------------

FIRST_ATTEMPT = {
    "when": "2026-11-13, 22:22-22:24 EST (after the reproduction, from `prepare`'s first run)",
    "checked_before_parsing": "none of the sixteen in the parsed manifest or folder (the check raises otherwise)",
    "parsed": [3816, 3817, 3824, 3825, 3832, 3833],
    "refused": {"3840": "ValueError: Repeated PFF frame 180900 with different tracking or events"},
    "log": "data/d61/prepare_first_attempt.log",
}
MIN_GAMES = 12


def parse_set_d(parsed, pff_raw, slots, *, prepare_game=None, first_attempt=FIRST_ATTEMPT, output=None):
    """Set D parsed for this step only, one game at a time in sorted order (D61's amendment of 22:25 EST): a game the
    standing reader refuses is left out with its one-line reason; fewer than twelve read games stop D61 unscored.

    The first attempt parsed six games and stopped at 3840; those six are kept (checked in the manifest), and every
    game not yet parsed, 3840 included, is checked unparsed and then parsed now."""
    from dfl_submission import pff, vendor_trial
    prepare_game = prepare_game or (lambda game: pff.prepare(pff_raw, parsed, ids=[game]))
    manifest = read(parsed / "manifest.json")["parsed"]
    kept = []
    for game in first_attempt["parsed"]:
        names = [f"{game}_{index}.npz" for index in (1, 2)]
        if not all(name in manifest and (parsed / name).exists() for name in names):
            raise ValueError(f"Game {game} of the first attempt is not in the parsed manifest")
        kept.append(game)
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    read_now, refused, checked = [], {}, {}
    for game in sorted(SET_D):
        if game in kept:
            continue
        names = [f"{game}_{index}.npz" for index in (1, 2)]
        manifest = read(parsed / "manifest.json")["parsed"]
        before = {"in_manifest": [n for n in names if n in manifest], "files": [n for n in names if (parsed / n).exists()]}
        if before["in_manifest"] or before["files"]:
            raise ValueError(f"Set D game {game} was parsed before: {before}")
        checked[str(game)] = before
        try:
            prepare_game(game)
        except ValueError as error:
            refused[str(game)] = f"{type(error).__name__}: {error}"
            stray = [n for n in names if (parsed / n).exists()]
            if stray:
                refused[str(game)] += f" (half files left without a manifest entry: {stray})"
            print(f"D61: the standing PFF reader refuses set D game {game}: {refused[str(game)]}; it is left out", flush=True)
            continue
        read_now.append(game)
    games = sorted(kept + read_now)
    note = {"set_d": sorted(SET_D), "games": games, "left_out": refused, "never_parsed_before": True,
            "first_attempt": first_attempt, "checked_before_parsing_now": checked, "parse_started": started,
            "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "slots": slots, "set_e_not_parsed": list(SET_E),
            "amendment": "DECISIONS.md D61, amendment of 22:25 EST", "minimum_games": MIN_GAMES,
            "inputs": vendor_trial.inputs(parsed, games) if games else None}
    output = SCRATCH / "set_d_parse.json" if output is None else output
    write_new(output, canonical(note))
    if len(games) < MIN_GAMES:
        raise SystemExit(f"D61 stops unscored: only {len(games)} of the sixteen set D games can be read ({refused})")
    return note


SIGN_FIELDS = ("group", "d", "frames", "sign", "sign_used", "first_frame", "first_sign", "agrees", "fallback")


def sign_report(table):
    """D61's second amendment (22:32 EST): per half the whole-half D, the sign used, the first qualifying frame's sign
    and any fallback; the first-frame agreement reported, not enforced; and, enforced by the caller, whether the two
    halves of every match (source, match, view) take opposite signs."""
    rows = {row["half"]: row for row in table.values()}
    per_half = {half: {key: row.get(key) for key in SIGN_FIELDS} for half, row in sorted(rows.items())}
    matches = {}
    for half, row in rows.items():
        stem, _, index = half.rpartition("/")
        matches.setdefault(stem, {})[index] = int(row["sign_used"])
    opposite = {stem: bool(set(value) == {"0", "1"} and value["0"] == -value["1"]) for stem, value in sorted(matches.items())}
    by_group = {}
    for row in rows.values():
        by_group.setdefault(row["group"], {})[row["half"]] = row
    return {"per_half": per_half, "opposite_halves": opposite,
            "first_frame_disagrees": sorted(half for half, row in rows.items() if row["sign"] is not None and not row["agrees"]),
            "fallbacks": sorted(half for half, row in rows.items() if row["fallback"]),
            "groups": {group: st.summarise(part, False) for group, part in sorted(by_group.items())},
            "rule": "DECISIONS.md D61, second amendment of 22:32 EST: first-frame agreement reported, opposite halves enforced"}


def prepare(dfl, skill, parsed, pff_raw, *, scratch=SCRATCH):
    """Set D parsed now; the signs; Q's and the stack's decoder inputs cached. Nothing is scored."""
    require_external_only()
    from dfl_submission import augment, diagnosis, hidden_views as hv, pff, vendor_trial
    prepared_path = scratch / "prepared.json"
    if (RECIPE / "confirmation_d61.json").exists():
        raise FileExistsError("D61's confirmation is written; nothing is prepared again")
    reproduction = read(scratch / "reproduction.json")
    if reproduction["passed"] is not True:
        raise SystemExit("D61 stops: the reproduction did not pass")
    ids = pinned_ids()
    slots = check_set_d(ids)
    if not (slots["set_d_is_slots_1_and_2"] and slots["set_e_is_slot_3"] and slots["slots_are_pff_split_train"]):
        raise ValueError(f"Set D is not D7's first and second train slots: {slots}")
    if prepared_path.exists():
        parse_note = read(prepared_path)["set_d_parse"]
    elif (scratch / "set_d_parse.json").exists():
        parse_note = read(scratch / "set_d_parse.json")
        if len(parse_note["games"]) < MIN_GAMES:
            raise SystemExit("D61 stops unscored: fewer than twelve set D games can be read")
    else:
        parse_note = parse_set_d(parsed, pff_raw, slots, output=scratch / "set_d_parse.json")
    games = parse_note["games"]
    if vendor_trial.inputs(parsed, games) != parse_note["inputs"]:
        raise ValueError("The parsed set D files differ from the ones parsed for this step")
    if any((parsed / f"{g}_{i}.npz").exists() for g in SET_E for i in (1, 2)):
        raise ValueError("A game of set E is parsed")
    long_hidden = st.holdout_long_hidden(dfl, raw=True)

    def items():
        for half in augment.guard_halves(dfl, skill, "holdout", raw=True):
            yield "holdout", (half.source, half.match, half.view, half.index), half.positions, half.ball, half.teams, half.pitch
        for game in games:
            for index, half in enumerate(pff.load(parsed, game)):
                yield "pff_set_d", ("pff", str(game), "observed", index), *pff.tracking(half)
        for (source, match, index), (_, positions, ball, teams, pitch) in sorted(long_hidden.items()):
            yield "holdout_long_hidden", (source, match, hv.VIEW, index), positions, ball, teams, pitch

    table = st.sign_table(items())
    signs = sign_report(table)
    if not all(signs["opposite_halves"].values()):
        failing = sorted(match for match, ok in signs["opposite_halves"].items() if not ok)
        raise ValueError(f"D61 stops: the two halves of {failing} do not take opposite signs; nothing of Q is scored")
    config, stack, stack_identity = stack_models()
    q, q_identity = q_models()
    common = {"protocol": PROTOCOL, "teacher_sha256": diagnosis.teacher_identity(), "dfl_raw": True,
              "skill_manifest_sha256": digest(skill / "manifest.json"), "long_hidden_seed": st.HOLDOUT_GUARD_SEED,
              "hidden_views_sha256": digest(PACKAGE / "hidden_views.py"), "pff": vendor_trial.inputs(parsed, games)}
    keys = {"stack": dict(common, model="the adopted stack", identity=stack_identity),
            "q": dict(common, model="Q", identity=q_identity)}
    alone = st.Forecaster({"Q": q["stack"]}, table, with_d43=False)
    q_forecast = alone.signals_of("Q")
    diagnosis.collect(scratch / "q_holdout", keys["q"],
                      lambda: diagnosis.external_halves(dfl, skill, "holdout", None, raw=True, signals_of=q_forecast))
    diagnosis.collect(scratch / "q_long_hidden", keys["q"], lambda: guard_halves(long_hidden, q_forecast))
    both = st.Forecaster({"stack": stack["stack"], "Q": q["stack"]}, table, with_d43=False)
    diagnosis.collect(scratch / "stack_set_d", keys["stack"],
                      lambda: diagnosis.pff_halves(parsed, games, None, signals_of=both.signals_of("stack")))
    diagnosis.collect(scratch / "q_set_d", keys["q"],
                      lambda: diagnosis.pff_halves(parsed, games, None, signals_of=both.signals_of("Q")))
    record = {"protocol": PROTOCOL, "reproduction_sha256": digest(scratch / "reproduction.json"), "set_d_parse": parse_note,
              "signs": signs,
              "caches": {name: digest(scratch / name / "manifest.json")
                         for name in ("stack_holdout", "stack_long_hidden", "q_holdout", "q_long_hidden", "stack_set_d",
                                      "q_set_d")},
              "keys": keys, "prepared_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    prepared_path.write_text(json.dumps(canonical(record), indent=2, sort_keys=True) + "\n")
    print(f"D61 prepared: set D {games} parsed at {parse_note['parsed_at']} (never parsed before; left out "
          f"{parse_note['left_out']}); signs: every match's halves opposite, first-frame disagreements (reported) "
          f"{signs['first_frame_disagrees']}, fallbacks {signs['fallbacks']}; no score computed", flush=True)
    return record


# -- 3. the confirmation, once --------------------------------------------------------------------------

def confirm(dfl, skill, parsed, *, scratch=SCRATCH, output=None):
    """The adopted stack first, then Q once, from the prepared caches only; D61's rule; written once."""
    require_external_only()
    from dfl_submission import aligned_trial, diagnosis, hyper_trial as ht, refine_trial, vendor_trial
    output = RECIPE / "confirmation_d61.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D61's confirmation is scored once")
    prepared = read(scratch / "prepared.json")
    reproduction = read(scratch / "reproduction.json")
    for name, value in prepared["caches"].items():
        if digest(scratch / name / "manifest.json") != value:
            raise ValueError(f"The {name} cache differs from the prepared one")
    games = prepared["set_d_parse"]["games"]
    if vendor_trial.inputs(parsed, games) != prepared["set_d_parse"]["inputs"]:
        raise ValueError("The parsed set D files differ from the ones parsed for this step")
    _, _, stack_identity = stack_models()
    _, q_identity = q_models()
    if canonical(stack_identity) != prepared["keys"]["stack"]["identity"] or canonical(q_identity) != prepared["keys"]["q"]["identity"]:
        raise ValueError("The models differ from the prepared ones")
    stack_holdout = diagnosis.collect(scratch / "stack_holdout", reproduction["key"], st.refuse)
    stack_long = diagnosis.collect(scratch / "stack_long_hidden", reproduction["key"], st.refuse)
    stack_d = diagnosis.collect(scratch / "stack_set_d", prepared["keys"]["stack"], st.refuse)
    q_holdout = diagnosis.collect(scratch / "q_holdout", prepared["keys"]["q"], st.refuse)
    q_long = diagnosis.collect(scratch / "q_long_hidden", prepared["keys"]["q"], st.refuse)
    q_d = diagnosis.collect(scratch / "q_set_d", prepared["keys"]["q"], st.refuse)
    recipe = ht.recipe_at(COST)

    def raw_standard(item):
        return item["external"]["conventions"]["raw"]["standard"]["spell_f1"]

    # The adopted stack first; its holdout part must still be D48's.
    base = aligned_trial.evaluate(stack_holdout + stack_d, recipe)
    long_base = st.long_hidden_scores(stack_long, recipe)
    if canonical(base["external"]) != canonical(read(RECIPE / "confirmation.json")["trial"]["external"]):
        raise ValueError("The adopted stack's holdout numbers differ from D48's confirmation")
    dry = refine_trial.changes(base, base)
    canonical({"base": base, "long": long_base, "dry": dry})
    # Q, once.
    trial = aligned_trial.evaluate(q_holdout + q_d, recipe)
    long_trial = st.long_hidden_scores(q_long, recipe)
    change = refine_trial.changes(base, trial)
    change["long_hidden"] = long_trial["mean"] - long_base["mean"]
    change["raw_standard"] = raw_standard(trial) - raw_standard(base)
    change["set_d_means"] = {name: trial["pff"]["means"][name]["spell_f1"] - base["pff"]["means"][name]["spell_f1"]
                             for name in base["pff"]["means"]}
    per_game = {game: {"stack": base["pff"]["matches"][game]["aba3"]["spell_f1"], "Q": trial["pff"]["matches"][game]["aba3"]["spell_f1"],
                       "change": trial["pff"]["matches"][game]["aba3"]["spell_f1"] - base["pff"]["matches"][game]["aba3"]["spell_f1"]}
                for game in sorted(base["pff"]["matches"], key=int)}
    conditions = rule(change)
    passed = bool(all(conditions.values()))
    confirmation = read(RECIPE / "confirmation.json")
    record = canonical({
        "protocol": PROTOCOL, "candidate": "Q: D60's twelve networks (D57's views, three seeds), D3's trees, the standing mix",
        "owner_cost": COST, "recipe": recipe, "levers": confirmation["levers"], "levers_sha256": confirmation["levers_sha256"],
        "reproduction": {key: reproduction[key] for key in ("passed", "numbers")} | {
            "halves_bit_for_bit": {part: f"{sum(v.values())}/{len(v)}" for part, v in reproduction["bit_for_bit"].items()}},
        "reference": base, "trial": trial, "changes": change, "per_game_set_d": per_game,
        "long_hidden": {"stack": long_base, "Q": long_trial},
        "conditions": conditions, "passed": passed,
        "rule": {"set_d_gain": GAIN, "losses": LOSS, "against": "the adopted stack at cost 14"},
        "set_d_parse": prepared["set_d_parse"], "signs": prepared["signs"],
        "identity": {"Q": q_identity, "adopted_stack": stack_identity}, "code": code_identity(),
        "prepared_sha256": digest(scratch / "prepared.json"), "reproduction_sha256": digest(scratch / "reproduction.json"),
        "inputs": {"pff_set_d": vendor_trial.inputs(parsed, games), "skill_manifest_sha256": digest(skill / "manifest.json"),
                   "dfl_raw": True},
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    write_new(output, record)
    print(f"D61 confirmation at cost {COST}: set D {change['pff']:+.4f} (+{GAIN}), holdout primary {change['primary']:+.4f}, "
          f"holdout long-hidden {change['long_hidden']:+.4f}, conventions "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items())
          + ", guards " + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f" -> {'PASSED' if passed else 'FAILED'} {conditions}; set D games {len(games)}, left out "
          f"{prepared['set_d_parse']['left_out']}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("reproduce")
    for flag in ("--dfl", "--skill"):
        one.add_argument(flag, type=Path, required=True)
    two = sub.add_parser("prepare")
    for flag in ("--dfl", "--skill", "--parsed", "--pff-raw"):
        two.add_argument(flag, type=Path, required=True)
    three = sub.add_parser("confirm")
    for flag in ("--dfl", "--skill", "--parsed"):
        three.add_argument(flag, type=Path, required=True)
    args = parser.parse_args()
    require_external_only()
    if args.action == "reproduce":
        reproduce(args.dfl, args.skill)
    elif args.action == "prepare":
        prepare(args.dfl, args.skill, args.parsed, args.pff_raw)
    else:
        confirm(args.dfl, args.skill, args.parsed)


if __name__ == "__main__":
    main()
