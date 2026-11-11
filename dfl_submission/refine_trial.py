"""D12: decoder output refinements on frozen D3 (DECISIONS.md D12).

Every setting decodes the same D3 inputs; only the decoder step, an optional collapse of brief
interior spells in the decoded labels and optional boundary shifts from D11 change. `tune` scores
the grid on the development matches (DFL WPY and four SkillCorner matches select; the PFF
development matches can only veto), after checking that the D3 setting reproduces the D3 and D7
records. `confirm` scores the selected setting once on the DFL/SkillCorner holdout and PFF
confirmation set A. `build` writes the competition file with a confirmed setting.
"""

import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.conventions import collapse_brief_changes
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "refine_recipe"
D11 = PACKAGE / "diagnosis_recipe" / "d11.json"
PROTOCOL = "DECISIONS.md D12"
STEPS = (5, 1)
COLLAPSE = (None, 1.75, 3.0)
PRIMARY = "aba3"
OTHER_CONVENTIONS = ("aba1.75", "aba4", "aba4.5")
GAIN, LOSS = .005, .01
REFERENCE = {"step": 5, "collapse": None, "shift": None}


# -- the refinements -------------------------------------------------------------------------------

def shift_boundaries(labels, shifts):
    """Move each transition by its type's shift in frames (negative is earlier).

    A segment never loses more than (length - 1) // 2 frames at either end, so no segment
    disappears and the moves of neighbouring transitions never overlap.
    """
    labels = np.asarray(labels)
    if not shifts or not len(labels):
        return labels.copy()
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    result = labels.copy()
    for k in range(1, len(edges) - 1):
        left_start, frame, right_stop = edges[k - 1], edges[k], edges[k + 1]
        before, after = labels[frame - 1], labels[frame]
        if pm.X in (before, after):
            continue
        amount = shifts.get(diagnosis.transition_type(before, after))
        if not amount:
            continue
        if amount < 0:
            moved = min(-amount, (frame - left_start - 1) // 2)
            result[frame - moved:frame] = after
        else:
            moved = min(amount, (right_stop - frame - 1) // 2)
            result[frame:frame + moved] = before
    return result


def refine(signals, recipe, setting):
    """Decode one half at the setting's step, then collapse brief interior spells, then shift boundaries."""
    labels = diagnosis.decode(signals, recipe, step=setting["step"])
    if setting["collapse"] is not None:
        labels = collapse_brief_changes(labels, setting["collapse"])
    if setting["shift"]:
        labels = shift_boundaries(labels, setting["shift"])
    return labels


def d11_shifts(path=D11):
    record = json.loads(path.read_text())
    shifts = {kind: entry["shift"] for kind, entry in record["shifts"].items() if entry["shift"] is not None}
    return shifts or None, digest(path)


def grid(shifts):
    options = (None,) if shifts is None else (None, shifts)
    return [{"step": step, "collapse": collapse, "shift": shift}
            for step, collapse, shift in product(STEPS, COLLAPSE, options)]


# -- scoring ---------------------------------------------------------------------------------------

def scored_setting(halves, recipe, setting):
    """Every setting, the reference included, is decoded afresh from the cached inputs."""
    return diagnosis.matches(halves, lambda half: refine(half.signals, recipe, setting))


def evaluate(halves, recipe, setting):
    from dfl_submission import vendor_trial
    external, vendor = diagnosis.split_sources(scored_setting(halves, recipe, setting))
    result = {"setting": setting}
    if external:
        result["external"] = diagnosis.external_record(external, recipe)
    if vendor:
        matches = diagnosis.pff_record(vendor)
        result["pff"] = {"means": vendor_trial.means(matches), "matches": matches}
    return result


def standard(trial, name, metric="spell_f1"):
    return trial["external"]["conventions"][name]["standard"][metric]


def changes(base, trial):
    result = {"primary": standard(trial, PRIMARY) - standard(base, PRIMARY),
              "conventions": {name: standard(trial, name) - standard(base, name) for name in OTHER_CONVENTIONS},
              "guards": {view: trial["external"]["conventions"][PRIMARY][view]
                         - base["external"]["conventions"][PRIMARY][view] for view in ("dead_carried", "team_swap")},
              "pff": trial["pff"]["means"][PRIMARY]["spell_f1"] - base["pff"]["means"][PRIMARY]["spell_f1"]}
    return result


def passed(change, gain=GAIN):
    return (change["primary"] >= gain and all(value >= -LOSS for value in change["conventions"].values())
            and all(value >= -LOSS for value in change["guards"].values()) and change["pff"] >= -LOSS)


def select(entries):
    """The eligible setting with the highest primary, then primary macroF1, then grid order; None if none."""
    eligible = [(i, entry) for i, entry in enumerate(entries) if entry["eligible"]]
    if not eligible:
        return None
    return max(eligible, key=lambda item: (standard(item[1]["trial"], PRIMARY),
                                           standard(item[1]["trial"], PRIMARY, "macro_f1"), -item[0]))[1]["setting"]


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D12 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def teacher():
    from dfl_submission import competition_adaptation
    return competition_adaptation.reference()


def tune(dfl, skill, parsed, *, raw, cache=None, trial=RECIPE):
    require_external_only()
    from dfl_submission import vendor_trial
    output = trial / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D12 development is scored once")
    shifts, d11_sha = d11_shifts()
    settings = grid(shifts)
    model = teacher()
    recipe = model[2]["recipe"]
    halves = diagnosis.development_halves(dfl, skill, parsed, model, raw=raw, cache=cache)
    checks = diagnosis.reproduces(scored_setting(halves, recipe, REFERENCE), recipe)
    if not all(checks.values()):
        raise ValueError(f"D12 pipeline check failed {checks}; no record was written")
    base = evaluate(halves, recipe, REFERENCE)
    entries = []
    for setting in settings:
        result = base if setting == REFERENCE else evaluate(halves, recipe, setting)
        change = changes(base, result)
        eligible = setting != REFERENCE and passed(change)
        entries.append({"setting": setting, "eligible": eligible, "changes": change, "trial": result})
        print(f"D12 {setting}: N = 3 standard {standard(result, PRIMARY):.5f} ({change['primary']:+.5f}), "
              + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
              + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
              + f", PFF {change['pff']:+.4f} -> {'eligible' if eligible else 'not eligible'}", flush=True)
    selected = select(entries)
    write_new(output, {"protocol": PROTOCOL, "grid": settings, "reference": REFERENCE, "reproduction": checks,
                       "entries": entries, "selected": selected, "d11_sha256": d11_sha,
                       "teacher_sha256": diagnosis.teacher_identity(),
                       "inputs": vendor_trial.inputs(parsed, vendor_trial.development())})
    if selected is not None:
        write_new(trial / "recipe.json", {"protocol": PROTOCOL, "setting": selected, "d11_sha256": d11_sha,
                                          "teacher_sha256": diagnosis.teacher_identity()})
    print(f"D12 development selected {selected if selected is not None else 'no setting; D3 stays'}", flush=True)
    return selected


def holdout_halves(dfl, skill, parsed, model, *, raw, cache=None):
    """DFL/SkillCorner holdout halves with the D2 guard views, then PFF confirmation set A."""
    from dfl_submission import vendor_trial
    games = vendor_trial.split_games("confirm_a")
    identity = {"teacher_sha256": diagnosis.teacher_identity(), "split": "holdout+confirm_a", "dfl_raw": raw,
                "skill_manifest_sha256": digest(skill / "manifest.json"), "pff": vendor_trial.inputs(parsed, games)}

    def produce():
        yield from diagnosis.external_halves(dfl, skill, "holdout", model, raw=raw)
        yield from diagnosis.pff_halves(parsed, games, model)

    return diagnosis.collect(cache, identity, produce), games


def confirm(dfl, skill, parsed, *, raw, cache=None, trial=RECIPE):
    require_external_only()
    from dfl_submission import vendor_trial
    development = json.loads((trial / "development.json").read_text())
    if development["selected"] is None:
        raise ValueError("No setting was eligible on development; D3 stays and nothing is confirmed")
    output = trial / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D12 confirmation is scored once")
    frozen = json.loads((trial / "recipe.json").read_text())
    if frozen["teacher_sha256"] != diagnosis.teacher_identity():
        raise ValueError("The D3 recipe differs from the one D12 development used")
    model = teacher()
    recipe = model[2]["recipe"]
    halves, games = holdout_halves(dfl, skill, parsed, model, raw=raw, cache=cache)
    base = evaluate(halves, recipe, REFERENCE)
    d3_holdout = json.loads((PACKAGE / "blend_recipe" / "external_validation.json").read_text())["selected"]
    reproduced = vendor_trial.canonical(base["external"]) == d3_holdout
    if not reproduced:
        raise ValueError("The D3 setting does not reproduce the D3 holdout record; no record was written")
    challenger = evaluate(halves, recipe, frozen["setting"])
    change = changes(base, challenger)
    result = passed(change)
    write_new(output, {"protocol": PROTOCOL, "setting": frozen["setting"], "reproduction": {"d3_holdout": reproduced},
                       "reference": base, "selected": challenger, "changes": change, "passed": result,
                       "teacher_sha256": frozen["teacher_sha256"], "inputs": vendor_trial.inputs(parsed, games)})
    print(f"D12 confirmation {frozen['setting']}: N = 3 {change['primary']:+.5f}, "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
          + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f", PFF set A {change['pff']:+.4f} -> {'passed' if result else 'failed'}", flush=True)
    return result


def build(output, trial=RECIPE):
    """The competition file with the confirmed setting; the models and blend are D3's."""
    from dfl_submission import motion_trial, retrain_trial
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((trial / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("The D12 setting did not pass confirmation; build D3 instead")
    frozen = json.loads((trial / "recipe.json").read_text())
    blend = PACKAGE / "blend_recipe"
    if digest(blend / "recipe.json") != frozen["teacher_sha256"]:
        raise ValueError("The D3 recipe differs from the one D12 was confirmed with")
    now, models_by_name, blended = retrain_trial.frozen_candidate(
        ROOT, PACKAGE / "seed_models", PACKAGE / "ensemble_recipe", PACKAGE / "decoder_recipe",
        PACKAGE / "retrain_models", blend)
    previous = models_by_name["current"][0][2]["recipe"]

    def decode_half(positions, ball, teams, pitch):
        _, mixed = retrain_trial.forecast_pair(models_by_name, blended["weight"], positions, ball, teams, pitch)
        signals = motion_trial.blend_signals(mixed, previous, blended["recipe"])
        return refine(signals, blended["recipe"], frozen["setting"])

    write_csv(output, tactical_halves(decode_half, centered=True, pitch=blended["pitch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("tune", "confirm", "build"))
    parser.add_argument("--parsed", type=Path, help="folder written by `dfl_submission.pff prepare`")
    parser.add_argument("--dfl", type=Path)
    parser.add_argument("--dfl-raw", action="store_true")
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--cache", type=Path, help="folder for the decoder inputs of the scored halves")
    parser.add_argument("--out", type=Path, help="CSV path (build only)")
    args = parser.parse_args()
    if args.command == "build":
        if args.out is None:
            parser.error("build needs --out")
        build(args.out)
        return
    if None in (args.parsed, args.dfl, args.skill):
        parser.error(f"{args.command} needs --parsed, --dfl and --skill")
    action = tune if args.command == "tune" else confirm
    action(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache=args.cache)


if __name__ == "__main__":
    main()
