"""D25: no stoppage shorter than one second (DECISIONS.md D25).

The candidate is D15 followed by `conventions.remove_brief_stoppages` on each decoded half: a
stoppage shorter than 25 frames with a team on both sides is given to those teams. `develop`
scores it once on D15's cached development inputs (DFL WPY in four views, four SkillCorner
matches, eight PFF matches), the Metrica sample games (D19) and the D24 views, after D15
reproduces each of their records. `confirm` scores it once on the reused DFL/SkillCorner holdout and
PFF confirmation set C. `build` writes the competition file. Nothing is tuned: the rule asks only
that nothing gets worse.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import diagnosis
from dfl_submission.conventions import remove_brief_stoppages
from dfl_submission.external_only import require_external_only
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stoppage_recipe"
PROTOCOL = "DECISIONS.md D25"
BRIEF_FRAMES = 25                      # one second: no restart is taken faster
PRIMARY_LOSS, CONVENTION_LOSS = .002, .005


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D25 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def post(labels):
    """D25's one post-processing step on one decoded half."""
    return remove_brief_stoppages(labels, BRIEF_FRAMES)


def predictor(recipe, post_step=None):
    """A half's labels: decoded from its cached inputs with `recipe`, then `post_step` if given."""
    def predict(half):
        labels = diagnosis.decode(half.signals, recipe)
        return labels if post_step is None else post_step(labels)
    return predict


def evaluate(halves, recipe, post_step=None):
    """`aligned_trial.evaluate` with an optional post-processing step after decoding."""
    from dfl_submission import vendor_trial
    external, vendor = diagnosis.split_sources(diagnosis.matches(halves, predictor(recipe, post_step)))
    result = {"recipe": recipe}
    if external:
        result["external"] = diagnosis.external_record(external, recipe)
    if vendor:
        matches = diagnosis.pff_record(vendor)
        result["pff"] = {"means": vendor_trial.means(matches), "matches": matches}
    return result


def passed(change, *, metrica=None, d24=None):
    """D25's no-harm conditions; development also passes the Metrica and D24 changes."""
    result = (change["primary"] >= -PRIMARY_LOSS
              and all(value >= -CONVENTION_LOSS for value in change["conventions"].values())
              and float(np.mean(list(change["guards"].values()))) >= 0
              and change["pff"] >= 0)
    if metrica is not None:
        result = result and metrica >= 0
    if d24 is not None:
        result = result and d24 >= 0
    return bool(result)


def group_name(key):
    source, _, view = key
    return f"{source}_{view}" if source in ("dfl", "metrica") else source


def group_stoppages(scored):
    """Predicted and true sub-second stoppages per group, from {key: (truth, prediction, home)}."""
    from dfl_submission.metrica_views import brief_stoppages
    result = {}
    for key, (truth, prediction, _) in scored.items():
        runs = pm.valid_runs(truth)
        entry = result.setdefault(group_name(key), {side: {"all": 0, "interior": 0} for side in ("predicted", "true")})
        for side, labels in (("predicted", prediction), ("true", truth)):
            for kind, count in brief_stoppages(labels, runs).items():
                entry[side][kind] += count
    return result


def stoppages(halves, recipe):
    return {"before": group_stoppages(diagnosis.matches(halves, predictor(recipe))),
            "after": group_stoppages(diagnosis.matches(halves, predictor(recipe, post)))}


def half_decoder(model, post_step=None):
    """The build's per-half decoder: D15's labels, then `post_step` if given."""
    def decode_half(positions, ball, teams, pitch):
        labels = diagnosis.half_signals(model, positions, ball, teams, pitch)[1]
        return labels if post_step is None else post_step(labels)
    return decode_half


def same_labels(halves, recipe):
    """Whether decoding the cached inputs with `recipe` gives the cached labels for every half."""
    return all(np.array_equal(diagnosis.decode(half.signals, recipe), half.d3) for half in halves)


def summary_line(prefix, trial, change, extra, verdict):
    from dfl_submission import refine_trial
    return (f"{prefix}: N = 3 standard {refine_trial.standard(trial, 'aba3'):.5f} ({change['primary']:+.5f}), "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items()) + ", "
            + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items()) + f", PFF {change['pff']:+.4f}"
            + extra + f" -> {verdict}")


# -- development, confirmation and build -----------------------------------------------------------

def develop(dfl, skill, parsed, *, raw, d15_cache, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import metrica_sample as ms, metrica_views as mv, refine_trial, scale_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D25 development is scored once")
    halves, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)  # stops unless reproduced
    _, d19_halves, _ = ms.reference_scores("D15")                     # stops unless D15 reproduces its D19 record
    d24_reference, d24_halves, truths = mv.reference_scores()         # and its D24 record
    if not (same_labels(d19_halves, recipe) and same_labels(d24_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    trial = evaluate(halves, recipe, post)
    change = refine_trial.changes(base, trial)
    veto = ms.veto(predictor(recipe, post), reference="D15")
    candidate_views = mv.view_scores(d24_halves, truths, predictor(recipe, post))
    d24 = {"reference": mv.d24_mean(d24_reference), "candidate": mv.d24_mean(candidate_views)}
    d24["change"] = d24["candidate"] - d24["reference"]
    result = passed(change, metrica=veto["changes"]["primary"], d24=d24["change"])
    counts = {"development": stoppages(halves, recipe), "metrica": stoppages(d19_halves, recipe),
              "d24": stoppages(d24_halves, recipe)}
    frozen, setting = scale_trial.frozen_recipe(scale_trial.RECIPE, scale_trial.MODELS)
    write_new(output, {"protocol": PROTOCOL, "brief_frames": BRIEF_FRAMES, "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True}, "trial": trial,
                       "changes": change, "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                       "d24": dict(d24, views=candidate_views), "stoppages": counts, "passed": result,
                       "models": frozen["models"], "teacher_sha256": diagnosis.teacher_identity(),
                       "inputs": {"pff": vendor_trial.inputs(parsed, vendor_trial.development()),
                                  "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(summary_line("D25 development", trial, change,
                       f", Metrica {veto['changes']['primary']:+.4f}, D24 {d24['change']:+.4f}",
                       "passed" if result else "failed"), flush=True)
    return result


def set_c_halves(parsed, setting, cache):
    """D15's decoder inputs on PFF confirmation set C, computed once and cached by identity."""
    from dfl_submission import scale_trial, vendor_trial
    games = vendor_trial.split_games("confirm_c")
    identity = {"protocol": PROTOCOL, "scale_models": scale_trial.identity(setting),
                "teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games)}
    model = scale_trial.teacher(setting)
    return diagnosis.collect(cache, identity, lambda: diagnosis.pff_halves(parsed, games, model)), games


def confirm(dfl, skill, parsed, *, raw, d15_cache, set_c_cache, recipe_folder=RECIPE):
    require_external_only()
    from dfl_submission import aligned_trial, refine_trial, scale_trial, vendor_trial, zoo
    development = json.loads((recipe_folder / "development.json").read_text())
    if development["passed"] is not True:
        raise ValueError("D25 failed development; D15 stays and nothing is confirmed")
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D25 confirmation is scored once")
    setting, frozen = zoo.d15_setting()
    recipe = frozen["recipe"]
    if development["recipe"] != recipe:
        raise ValueError("D15's recipe differs from the one D25 development used")
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting, raw=raw, cache=d15_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full["external"]) != stored["external"]
            or vendor_trial.canonical(full["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    external = [half for half in reference if half.source != "pff"]
    set_a = [half for half in reference if half.source == "pff"]
    set_c, games_c = set_c_halves(parsed, setting, set_c_cache)
    base = evaluate(external + set_c, recipe)
    trial = evaluate(external + set_c, recipe, post)
    change = refine_trial.changes(base, trial)
    result = passed(change)
    set_a_change = (evaluate(set_a, recipe, post)["pff"]["means"]["aba3"]["spell_f1"]
                    - evaluate(set_a, recipe)["pff"]["means"]["aba3"]["spell_f1"])
    write_new(output, {"protocol": PROTOCOL, "brief_frames": BRIEF_FRAMES, "recipe": recipe,
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": result, "reported": {"pff_set_a_change": set_a_change},
                       "stoppages": {"holdout_and_set_c": stoppages(external + set_c, recipe),
                                     "set_a": stoppages(set_a, recipe)},
                       "models": scale_trial.identity(setting),
                       "inputs": {"set_c": vendor_trial.inputs(parsed, games_c),
                                  "set_a": vendor_trial.inputs(parsed, games_a)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(summary_line("D25 confirmation (PFF = set C)", trial, change, f", PFF set A (reported) {set_a_change:+.4f}",
                       "passed" if result else "failed"), flush=True)
    return result


def build(output, recipe_folder=RECIPE):
    """The competition file: D15, then D25's step on each half."""
    from dfl_submission import scale_trial
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D25 did not pass confirmation; build D15 instead")
    frozen, setting = scale_trial.frozen_recipe(scale_trial.RECIPE, scale_trial.MODELS)
    reference, packages, d3 = scale_trial.teacher(setting)
    model = (reference, packages, dict(d3, recipe=frozen["recipe"]))
    write_csv(output, tactical_halves(half_decoder(model, post), centered=True, pitch=frozen["pitch"]))


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
            command.add_argument("--set-c-cache", type=Path, required=True)
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
                set_c_cache=args.set_c_cache)


if __name__ == "__main__":
    main()
