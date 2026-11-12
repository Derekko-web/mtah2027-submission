"""D26: ownership-change timing in Metrica's convention (DECISIONS.md D26).

On the three Metrica sample games D15's ownership changes land a few frames after Metrica's marks.
`develop` first checks that the lag holds under two marking rules (D19's, and a second one where the
opponent of a lost ball waits for its own next event) on the clean and masked views of every game,
derives one shift from those 12 medians, then scores it: leave one game out on the clean view, the
D24 mean, and guards on the DFL/SkillCorner/PFF development matches. `confirm` scores the guards
once on the reused holdout and the PFF untouched set, and `build` writes the competition file.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.conventions import runs
from dfl_submission.external_only import require_external_only
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "timing_recipe"
PROTOCOL = "DECISIONS.md D26"
MIN_SHIFT = 2                       # frames: every median must reach this before anything is shifted
LOGO_GAIN = .003
GUARD_LOSS = .01
SECOND_RULE = frozenset()           # no handover mark: the opponent waits for its own next event
RULES = ("d19", "second")
VIEWS = ("clean", "masked")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D26 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


# -- the shift and its rule ------------------------------------------------------------------------

def shift_changes(labels, frames):
    """Move every change directly between the two teams `frames` earlier, never past the previous boundary."""
    labels = np.asarray(labels)
    result = labels.copy()
    if frames <= 0:
        return result
    previous = 0
    spells = runs(labels)
    for i, (start, _, label) in enumerate(spells):
        begin = start
        before = spells[i - 1][2] if i else None
        if i and label in pm.POSSESSION_LABELS and before in pm.POSSESSION_LABELS and label != before:
            begin = max(start - frames, previous + 1)
            result[begin:start] = label
        previous = begin
    return result


def robust_shift(medians):
    """D26 step 2: the smallest median, rounded half away from zero, if every median is at least +2; else 0."""
    if any(value is None or value < MIN_SHIFT for value in medians):
        return 0
    return diagnosis.round_half_away(min(medians))


def ownership_median(truth, prediction):
    """Median signed offset (predicted minus true frame) of matched changes between teams; None if none."""
    found = diagnosis.offsets(truth, prediction)["ownership"]["offsets"]
    return float(np.median(found)) if found else None


def leave_one_game_out(medians, change):
    """Each game's shift from the other games' medians, then `change(game, shift)` on that game."""
    shifts, changes = {}, {}
    for game in sorted(medians):
        shift = robust_shift([value for other, values in medians.items() if other != game for value in values])
        shifts[str(game)], changes[str(game)] = shift, change(game, shift)
    mean = float(np.mean(list(changes.values())))
    return {"shifts": shifts, "changes": changes, "mean": mean,
            "passed": bool(all(value >= 0 for value in changes.values()) and mean >= LOGO_GAIN)}


def guards_passed(guards):
    return all(value >= -GUARD_LOSS for value in guards.values())


def post(frames):
    return lambda labels: shift_changes(labels, frames)


def guard_changes(change):
    """Standard combined N = 3, the two DFL guard views and PFF, from `refine_trial.changes`."""
    return {"primary": change["primary"], **change["guards"], "pff": change["pff"]}


# -- Metrica targets under both marking rules ------------------------------------------------------

def second_rule_targets(raw, parsed):
    """D19's targets under the second marking rule per game, after the D26 pipeline checks."""
    from dfl_submission import metrica_sample as ms
    result = {}
    for game in ms.GAMES:
        stored = ms.load(parsed, game)
        d19 = ms.read_game(raw, game)
        second = ms.read_game(raw, game, handover=SECOND_RULE)
        for kept, fresh, other in zip(stored, d19, second):
            if not np.array_equal(kept["labels"], fresh["labels"]):
                raise ValueError(f"Re-reading Metrica game {game} does not reproduce its parsed D19 targets")
            if not (np.array_equal(kept["labels"] == pm.D, other["labels"] == pm.D)
                    and np.array_equal(kept["labels"] == pm.X, other["labels"] == pm.X)):
                raise ValueError(f"The second marking rule changes dead or X frames in game {game}")
        result[game] = ms.targets(second)
    return result


def joined(parts):
    return np.r_[parts[0], pm.X, parts[1]].astype(np.int8)


def by_game(halves, predict):
    """{game: [labels of half 0, labels of half 1]}."""
    result = {}
    for half in sorted(halves, key=lambda item: (int(item.match), item.index)):
        result.setdefault(int(half.match), []).append(np.asarray(predict(half)))
    return result


# -- development, confirmation and build -----------------------------------------------------------

def develop(dfl, skill, parsed, *, raw, d15_cache, metrica_raw=None, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import metrica_sample as ms, metrica_views as mv, refine_trial, scale_trial, stoppage_trial
    from dfl_submission import vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D26 development is scored once")
    metrica_raw = ms.RAW if metrica_raw is None else metrica_raw
    halves, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)  # stops unless reproduced
    _, d19_halves, d19_truths = ms.reference_scores("D15")             # stops unless D15 reproduces its D19 record
    d24_reference, d24_halves, truths = mv.reference_scores()         # and its D24 record
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    targets = {"d19": d19_truths, "second": second_rule_targets(metrica_raw, ms.PARSED)}
    plain = stoppage_trial.predictor(recipe)
    predictions = {"clean": by_game(d19_halves, plain),
                   "masked": by_game([half for half in d24_halves if half.view == "masked"], plain)}
    medians = {game: {f"{rule}/{view}": ownership_median(targets[rule][game][ms.PRIMARY], joined(predictions[view][game]))
                      for rule in RULES for view in VIEWS} for game in ms.GAMES}
    shift = robust_shift([value for game in medians.values() for value in game.values()])
    record = {"protocol": PROTOCOL, "recipe": recipe, "medians": {str(game): values for game, values in medians.items()},
              "shift": shift, "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True,
                                               "second_rule_structure": True},
              "models": scale_trial.frozen_recipe(scale_trial.RECIPE, scale_trial.MODELS)[0]["models"],
              "teacher_sha256": diagnosis.teacher_identity(),
              "inputs": {"pff": vendor_trial.inputs(parsed, vendor_trial.development()), "metrica": ms.inputs(ms.PARSED)}}
    print("D26 ownership medians (frames): " + "; ".join(
        f"game {game} " + ", ".join(f"{key} {value}" for key, value in values.items()) for game, values in medians.items()),
        flush=True)
    if shift == 0:
        write_new(output, dict(record, stopped="not every median reached +2 frames; no shift", passed=False,
                               scored_at=time.strftime("%Y-%m-%d %H:%M %Z")))
        print("D26 stops at step 2: the lag is not robust, so nothing is shifted or scored", flush=True)
        return False

    def clean_change(game, frames):
        parts = predictions["clean"][game]
        before = ms.score(d19_truths[game], parts)[ms.PRIMARY]["spell_f1"]
        after = ms.score(d19_truths[game], [shift_changes(part, frames) for part in parts])[ms.PRIMARY]["spell_f1"]
        return after - before

    logo = leave_one_game_out({game: list(values.values()) for game, values in medians.items()}, clean_change)
    shifted = stoppage_trial.predictor(recipe, post(shift))
    candidate_views = mv.view_scores(d24_halves, truths, shifted)
    d24 = {"reference": mv.d24_mean(d24_reference), "candidate": mv.d24_mean(candidate_views)}
    d24["change"] = d24["candidate"] - d24["reference"]
    trial = stoppage_trial.evaluate(halves, recipe, post(shift))
    change = refine_trial.changes(base, trial)
    guards = guard_changes(change)
    passed = logo["passed"] and d24["change"] >= 0 and guards_passed(guards)
    metrica = {rule: {"reference": ms.means({str(g): ms.score(targets[rule][g], predictions["clean"][g]) for g in ms.GAMES}),
                      "candidate": ms.means({str(g): ms.score(targets[rule][g], [shift_changes(part, shift) for part in
                                                                                  predictions["clean"][g]])
                                             for g in ms.GAMES})}
               for rule in RULES}
    write_new(output, dict(record, logo=logo, d24=dict(d24, views=candidate_views), reference=base, trial=trial,
                           changes=change, guards=guards, metrica_reported=metrica, passed=passed,
                           scored_at=time.strftime("%Y-%m-%d %H:%M %Z")))
    print(f"D26 development, shift {shift} frames: leave one game out {logo['changes']} (mean {logo['mean']:+.5f}), "
          f"D24 {d24['change']:+.5f}, guards " + ", ".join(f"{k} {v:+.4f}" for k, v in guards.items())
          + f" -> {'passed' if passed else 'failed'}", flush=True)
    return passed


def untouched_halves(parsed, setting, cache):
    """D15's decoder inputs on the PFF untouched set, computed once and cached by identity."""
    from dfl_submission import scale_trial, vendor_trial
    games = vendor_trial.split_games("untouched")
    identity = {"protocol": PROTOCOL, "scale_models": scale_trial.identity(setting),
                "teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games)}
    model = scale_trial.teacher(setting)
    return diagnosis.collect(cache, identity, lambda: diagnosis.pff_halves(parsed, games, model)), games


def confirm(dfl, skill, parsed, *, raw, d15_cache, untouched_cache, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, refine_trial, scale_trial, stoppage_trial, vendor_trial, zoo
    development = json.loads((recipe_folder / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("D26 failed development; D15 stays and nothing is confirmed")
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D26 confirmation is scored once")
    setting, frozen = zoo.d15_setting()
    recipe, shift = frozen["recipe"], development["shift"]
    if development["recipe"] != recipe:
        raise ValueError("D15's recipe differs from the one D26 development used")
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting, raw=raw, cache=d15_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full["external"]) != stored["external"]
            or vendor_trial.canonical(full["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    external = [half for half in reference if half.source != "pff"]
    untouched, games_u = untouched_halves(parsed, setting, untouched_cache)
    base = stoppage_trial.evaluate(external + untouched, recipe)
    trial = stoppage_trial.evaluate(external + untouched, recipe, post(shift))
    change = refine_trial.changes(base, trial)
    guards = guard_changes(change)
    passed = guards_passed(guards)
    write_new(output, {"protocol": PROTOCOL, "shift": shift, "recipe": recipe, "reproduction": {"d15_holdout": True},
                       "reference": base, "trial": trial, "changes": change, "guards": guards, "passed": passed,
                       "models": scale_trial.identity(setting),
                       "inputs": {"untouched": vendor_trial.inputs(parsed, games_u)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D26 confirmation, shift {shift} frames (PFF = untouched set): "
          + ", ".join(f"{k} {v:+.4f}" for k, v in guards.items()) + f" -> {'passed' if passed else 'failed'}", flush=True)
    return passed


def build(output, recipe_folder=RECIPE):
    """The competition file: D15, then D26's shift on each half."""
    from dfl_submission import scale_trial, stoppage_trial
    from dfl_submission.build import tactical_halves, write_csv
    confirmation = json.loads((recipe_folder / "confirmation.json").read_text())
    if confirmation["passed"] is not True:
        raise ValueError("D26 did not pass confirmation; build D15 instead")
    frozen, setting = scale_trial.frozen_recipe(scale_trial.RECIPE, scale_trial.MODELS)
    reference, packages, d3 = scale_trial.teacher(setting)
    model = (reference, packages, dict(d3, recipe=frozen["recipe"]))
    decode_half = stoppage_trial.half_decoder(model, post(confirmation["shift"]))
    write_csv(output, tactical_halves(decode_half, centered=True, pitch=frozen["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("develop", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--d15-cache", type=Path, required=True)
        if name == "confirm":
            command.add_argument("--untouched-cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache,
                untouched_cache=args.untouched_cache)


if __name__ == "__main__":
    main()
