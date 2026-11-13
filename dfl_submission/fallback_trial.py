"""D53: a fallback, the stack without the attacking frame (DECISIONS.md D53).

The candidate is D48's stack with `direction` off: D49's long-hidden sequences and tokens in pitch coordinates,
D47's mirror lever as screened (`mirror` "xy": windows mirrored in x and in y with probability 1/2 each; inference
over the four mirrors times the two team orders), offsets 0 / 1,000 / 2,000, 8,100 steps. It is trained, scored and
confirmed through `stack_set` and `stack_trial` (unchanged; `stack_trial.py` stays the file the build lock names):

- `levers` records the candidate once (`stack_recipe/levers_d53.json`); no new trainer check (D48's `check_mirror`,
  the two-axis mirror on D42's data, and `check_hidden`, D49's data, cover both switches; 8,100 steps is
  `stack_trial train-steps`, whose call D52 checked);
- `networks` trains and exports the twelve networks (two processes at most, one per set);
- `develop` is `stack_trial.develop` with this candidate (`development_d53.json`), D48's selection score and gate;
- `confirm-prepare` and `confirm`: the second use of D48's confirmation sets (the holdout; WQQ and WR9 long-hidden
  views with D48's seeds; the PFF games parsed for D48, from the same files), D43 first reproducing its holdout
  record and D48's confirmation record's D43 numbers; D48's pass rule; `confirmation_d53.json`, written once;
- `adopt-models`, `reproduce-build` and `build`: the fallback's competition build (preparation), `stack_trial build`'s
  path with direction off and its identity checks against `confirmation_d53.json`, the networks copied to
  `fallback_models/`, the driver locked by `build_identity_d53.json`.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dfl_submission import stack_set as ss, stack_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = st.ROOT
RECIPE = st.RECIPE
SCRATCH = ROOT / "data" / "d53"
PROTOCOL = "DECISIONS.md D53"
LEVERS = "levers_d53.json"
STEPS = 8100
CONFIG = ss.config_of(mirror="xy")                       # direction off: tokens in pitch coordinates
TRAINER_CHECKS = ("check_mirror.json", "check_hidden.json")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D53 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


read = st.read


# -- the candidate ---------------------------------------------------------------------------------

def record_levers(output=None):
    require_external_only()
    output = RECIPE / LEVERS if output is None else output
    for name in TRAINER_CHECKS:
        st.passed_record(name)
    write_new(output, {"protocol": PROTOCOL, "data": "d49", "config": CONFIG, "offsets": list(st.OFFSETS), "steps": STEPS,
                       "trainer_checks": {name: digest(RECIPE / name) for name in TRAINER_CHECKS},
                       "note": "No new trainer check: check_mirror (the two-axis mirror, D42's data, 5,400 steps) and "
                               "check_hidden (D49's data) cover both switches; no bit-for-bit reference exists at "
                               "8,100 steps for this combination.",
                       "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D53 levers: {CONFIG}, {STEPS} steps", flush=True)


def candidate():
    record = read(RECIPE / LEVERS)
    ss.validate(record["config"])
    if record["config"] != CONFIG or record["steps"] != STEPS:
        raise ValueError("The D53 lever record is not D53's candidate")
    return record


def export_folder(offset, model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / "export" / str(offset) / model_set / folder


def raw_folder(offset, model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / "raw" / str(offset) / model_set / folder


def train_command(source, tokens, out, seed, boost, offset):
    """`stack_trial train-steps` with D53's switches (no signs: the attacking frame is off)."""
    config = dict(CONFIG, seed_offset=offset)
    return [sys.executable, "-m", "dfl_submission.stack_trial", "train-steps", "--source", str(source), "--tokens",
            str(tokens), "--out", str(out), "--seed", str(seed), "--boundary-boost", str(boost), "--config",
            json.dumps(config), "--sign-map", str(st.PARITY), "--steps", str(STEPS)]


def train_networks(external, *, sets=st.SETS, scratch=SCRATCH / "networks"):
    """The twelve networks of `sets` (offset, set, network), one at a time per process, each exported and checked."""
    require_external_only()
    for name in TRAINER_CHECKS:
        st.passed_record(name)
    chosen = candidate()
    for offset in st.OFFSETS:
        for model_set in sets:
            source, tokens = st.data_folders(chosen["data"], model_set, external)
            for folder, (seed, boost) in st.NETWORKS.items():
                output = export_folder(offset, model_set, folder, scratch)
                if (output / "fit.json").exists():
                    continue
                raw = raw_folder(offset, model_set, folder, scratch)
                if not (raw / "fit.json").exists():
                    if raw.exists():
                        raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                    raw.parent.mkdir(parents=True, exist_ok=True)
                    st.wait_for_gpu(f"D53 offset {offset} {model_set}/{folder}")
                    subprocess.run(train_command(source, tokens, raw, seed, boost, offset), check=True, cwd=ROOT)
                output.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                                "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- development -----------------------------------------------------------------------------------

def gate_summary(decision):
    return {"gate_passed": decision["passed"], "selected_cost": decision["selected_cost"],
            "selection_score": decision["selection_score"]}


def variant():
    return {"name": "d53", "levers": LEVERS, "export": export_folder, "required": (),
            "extra": {"steps": STEPS, "variant": "d53"}, "choice": gate_summary}


def develop(dfl, skill, parsed, *, raw):
    """D53 on development exactly as D48's stack was (`stack_trial.develop`); `development_d53.json`, written once."""
    candidate()
    return st.develop(dfl, skill, parsed, raw=raw, cache_root=SCRATCH / "cache", scratch=SCRATCH / "networks",
                      output=RECIPE / "development_d53.json", variant=variant())


def summary(folder=RECIPE):
    """D43, the confirmed stack (8,100 steps, cost 14) and D53 at every cost; D53's gate."""
    reference_record = read(folder / "reference.json")
    confirmed = read(folder / "development_t.json")
    development = read(folder / "development_d53.json")
    base = development["reference"]
    base_conventions = st.conventions_of(base)
    rows = ["| Model | Cost | Composite | Guard mean (DFL / Metrica) | Selection | PFF | Primary | "
            + " | ".join(st.SHORT[g] for g in st.GROUPS) + " | N 1.75 / 4 / 4.5 vs D43@10 |",
            "|---|---:|---:|---|---:|---:|---:|" + "---:|" * len(st.GROUPS) + "---|"]

    def row(name, item):
        conventions = st.conventions_of(item)
        changes = " / ".join(f"{conventions[n] - base_conventions[n]:+.4f}" for n in ("aba1.75", "aba4", "aba4.5"))
        guards = item["guards"]
        return (f"| {name} | {item['owner_cost']} | {item['composite']:.4f} | {guards['mean']:.4f} "
                f"({guards['a_dfl']:.3f} / {guards['b_metrica']:.3f}) | {st.selection(item):.4f} | "
                f"{item['groups']['pff']:.4f} | {conventions['aba3']:.4f} | "
                + " | ".join(f"{item['groups'][g]:.3f}" for g in st.GROUPS) + f" | {changes} |")

    for cost in st.COSTS:
        rows.append(row("D43", reference_record["entries"][str(cost)]))
    rows.append(row("stack with C (confirmed)", next(item for item in confirmed["entries"]["stack"] if item["owner_cost"] == 14)))
    for name in st.MODEL_NAMES[1:] + st.MODEL_NAMES[:1]:
        for item in development["entries"][name]:
            rows.append(row(f"D53 {name}", item))
    decision = development["decision"]
    changes = decision["changes"]
    lines = ["\n".join(rows), "",
             f"D53: selected cost {decision['selected_cost']} (selection {decision['selection_score']:.5f} against D43's "
             f"{base['selection']:.5f}):",
             f"- selection {changes['selection']:+.5f} (needs +{st.SELECTION_GAIN}): {decision['conditions']['selection']}",
             f"- PFF {changes['pff']:+.5f} (needs +{st.PFF_GAIN}): {decision['conditions']['pff']}",
             f"- composite {changes['composite']:+.5f} (must not fall): {decision['conditions']['composite']}",
             f"- worst group {min(changes['groups'], key=changes['groups'].get)} {min(changes['groups'].values()):+.4f} "
             f"(each >= -{st.LOSS}): {decision['conditions']['groups']}",
             "- conventions " + ", ".join(f"{k} {v:+.4f}" for k, v in changes["conventions"].items())
             + f" (each >= -{st.LOSS}): {decision['conditions']['conventions']}",
             f"Gate {'PASSED' if decision['passed'] else 'NOT passed'}"]
    text = "\n".join(lines)
    print(text)
    return text


# -- confirmation (a second use of D48's sets) -------------------------------------------------------

class Unsigned(dict):
    """A sign table for a model without the attacking frame: every half is accepted, its sign is never used."""

    def get(self, key, default=None):
        return {"sign_used": 1, "half": "unsigned", "fallback": None}


CONFIRM_CACHE = SCRATCH / "confirm"


def stack_identity(scratch=SCRATCH / "networks"):
    return st.model_identity("stack", CONFIG, scratch, export=export_folder, levers_name=LEVERS,
                             extra=variant()["extra"])


def confirm_key(parsed, games_u, skill, raw, scratch=SCRATCH / "networks"):
    from dfl_submission import diagnosis, vendor_trial
    common = {"teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games_u),
              "long_hidden_seed": st.HOLDOUT_GUARD_SEED, "hidden_views_sha256": digest(st.PACKAGE / "hidden_views.py"),
              "dfl_raw": raw}
    return dict(stack_identity(scratch), **common, skill_manifest_sha256=digest(skill / "manifest.json"))


def d48_sets(dfl, skill, parsed, *, raw):
    """D48's confirmation sets as built for it, after D43 reproduces its holdout record and D48's D43 numbers."""
    from dfl_submission import aligned_trial, diagnosis, hidden_views as hv, hyper_trial as ht, player_trial, vendor_trial
    prepared = read(st.CONFIRM_CACHE / "prepared.json")
    d48 = read(RECIPE / "confirmation.json")
    games_c = vendor_trial.split_games("confirm_c")
    d43_reused = player_trial.candidate_halves(dfl, skill, parsed, games_c, "holdout", raw=raw, cache=st.D43_HOLDOUT_CACHE)
    holdout_record = st.canonical_json(aligned_trial.evaluate(d43_reused, ht.recipe_at(st.REFERENCE_COST))) == \
        read(player_trial.RECIPE / "confirmation.json")["trial"]
    games_u = prepared["untouched_parse"]["games"]
    same_files = vendor_trial.inputs(parsed, games_u) == prepared["untouched_parse"]["inputs"]
    d43_untouched = diagnosis.collect(st.CONFIRM_CACHE / "d43_untouched", prepared["keys"]["d43"], st.refuse)
    d43_long = diagnosis.collect(st.CONFIRM_CACHE / "d43_long_hidden", prepared["keys"]["d43"], st.refuse)
    d43_external = [half for half in d43_reused if half.source != "pff"]
    base = aligned_trial.evaluate(d43_external + d43_untouched, ht.recipe_at(st.REFERENCE_COST))
    long_base = st.long_hidden_scores(d43_long, ht.recipe_at(st.REFERENCE_COST))
    d48_numbers = (st.canonical_json(base) == d48["reference"]
                   and st.canonical_json(long_base) == {key: d48["long_hidden"]["d43"][key] for key in ("matches", "mean")})
    long_hidden = st.holdout_long_hidden(dfl, raw=raw)
    views = hv.view_statistics([value[2] for value in long_hidden.values()])
    same_views = (st.canonical_json(views) == prepared["long_hidden_views"]["statistics"]
                  and {f"{match}/{index}": st.holdout_seed(match, index) for _, match, index in sorted(long_hidden)}
                  == prepared["long_hidden_views"]["seeds"])
    checks = {"d43_holdout_record": bool(holdout_record), "d43_d48_confirmation_numbers": bool(d48_numbers),
              "pff_same_parsed_files": bool(same_files), "long_hidden_same_views": bool(same_views)}
    return {"checks": checks, "games_u": games_u, "long_hidden": long_hidden, "d43_external": d43_external,
            "d43_untouched": d43_untouched, "d43_long": d43_long, "base": base, "long_base": long_base, "d48": d48}


def confirm_prepare(dfl, skill, parsed, *, raw, cache_root=CONFIRM_CACHE):
    """Before D53's confirmation scores: the checks on D43 and on D48's sets, then D53's decoder inputs, cached."""
    require_external_only()
    from dfl_submission import diagnosis, hidden_views as hv
    if (RECIPE / "confirmation_d53.json").exists():
        raise FileExistsError("D53's confirmation is written; nothing is prepared again")
    development = read(RECIPE / "development_d53.json")
    if development["decision"]["passed"] is not True:
        raise ValueError("D53's development gate did not pass; D53 stops")
    sets = d48_sets(dfl, skill, parsed, raw=raw)
    print(f"D53 confirmation checks: {sets['checks']}", flush=True)
    if not all(sets["checks"].values()):
        raise ValueError(f"A check before D53's confirmation failed {sets['checks']}; nothing of D53 is scored")
    sign_map = read(st.PARITY)["sign_map"]
    forecaster = st.Forecaster(st.stack_models(CONFIG, SCRATCH / "networks", export=export_folder, sign_map=sign_map,
                                               names=("stack",)), Unsigned(), with_d43=False)
    forecast = forecaster.signals_of("stack")
    key = confirm_key(parsed, sets["games_u"], skill, raw)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, "holdout", None, raw=raw, signals_of=forecast)
        yield from diagnosis.pff_halves(parsed, sets["games_u"], None, signals_of=forecast)

    def guard_halves():
        for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(sets["long_hidden"].items()):
            signals, decoded = forecast(None, positions, ball, teams, pitch)
            yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)

    diagnosis.collect(cache_root / "stack", key, produce)
    diagnosis.collect(cache_root / "stack_long_hidden", key, guard_halves)
    record = {"protocol": PROTOCOL, "checks": sets["checks"], "owner_cost": development["decision"]["selected_cost"],
              "key": key, "caches": {name: digest(cache_root / name / "manifest.json") for name in ("stack", "stack_long_hidden")},
              "prepared_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    (cache_root / "prepared.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print("D53 confirmation prepared; no score computed", flush=True)
    return record


def confirm(dfl, skill, parsed, *, raw, cache_root=CONFIRM_CACHE, output=None):
    """D53's confirmation, once, from the prepared caches: D43 at cost 10 against D53 at its selected cost."""
    require_external_only()
    from dfl_submission import aligned_trial, diagnosis, hyper_trial as ht, refine_trial, vendor_trial
    output = RECIPE / "confirmation_d53.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D53's confirmation is scored once")
    prepared = read(cache_root / "prepared.json")
    development = read(RECIPE / "development_d53.json")
    cost = development["decision"]["selected_cost"]
    if prepared["owner_cost"] != cost or not all(prepared["checks"].values()):
        raise ValueError("D53's prepared confirmation does not match its development record")
    sets = d48_sets(dfl, skill, parsed, raw=raw)
    if not all(sets["checks"].values()):
        raise ValueError(f"A check before D53's confirmation failed {sets['checks']}")
    key = confirm_key(parsed, sets["games_u"], skill, raw)
    if st.canonical_json(key) != prepared["key"]:
        raise ValueError("D53's confirmation cache identity differs from the prepared one")
    for name, value in prepared["caches"].items():
        if digest(cache_root / name / "manifest.json") != value:
            raise ValueError(f"The {name} cache differs from the prepared one")
    stack_main = diagnosis.collect(cache_root / "stack", key, st.refuse)
    stack_long = diagnosis.collect(cache_root / "stack_long_hidden", key, st.refuse)
    base, long_base, d48 = sets["base"], sets["long_base"], sets["d48"]

    def raw_standard(item):
        return item["external"]["conventions"]["raw"]["standard"]["spell_f1"]

    raw_standard(base)
    identity = stack_identity()
    raw_fits = {str(offset): {key_set: {folder: digest(raw_folder(offset, key_set, folder) / "fit.json")
                                        for folder in st.NETWORKS} for key_set in st.SETS} for offset in st.OFFSETS}
    st.canonical_json({"identity": identity, "raw_fits": raw_fits})
    # D53, once.
    trial = aligned_trial.evaluate(stack_main, ht.recipe_at(cost))
    long_trial = st.long_hidden_scores(stack_long, ht.recipe_at(cost))
    change = refine_trial.changes(base, trial)
    change["long_hidden"] = long_trial["mean"] - long_base["mean"]
    change["raw_standard"] = raw_standard(trial) - raw_standard(base)
    conditions = {"pff": bool(change["pff"] >= st.CONFIRM_GAIN), "long_hidden": bool(change["long_hidden"] >= st.CONFIRM_GAIN),
                  "primary": bool(change["primary"] >= -st.LOSS),
                  "conventions": bool(all(value >= -st.LOSS for value in change["conventions"].values())),
                  "guards": bool(all(value >= -st.LOSS for value in change["guards"].values()))}
    passed = bool(all(conditions.values()))
    record = st.canonical_json({
        "protocol": PROTOCOL, "second_use_of": "DECISIONS.md D48 confirmation sets", "owner_cost": cost,
        "reference_cost": st.REFERENCE_COST, "levers": candidate(), "levers_sha256": digest(RECIPE / LEVERS),
        "checks": sets["checks"], "reference": base, "trial": trial, "changes": change,
        "long_hidden": {"d43": long_base, "d53": long_trial, "stack_with_c": d48["long_hidden"]["stack"]},
        "stack_with_c": {"owner_cost": d48["owner_cost"], "changes": d48["changes"], "trial": d48["trial"]},
        "conditions": conditions, "passed": passed, "identity": identity, "raw_fit_sha256": raw_fits,
        "code": st.code_identity() | {"fallback_trial.py": digest(st.PACKAGE / "fallback_trial.py")},
        "prepared_sha256": digest(cache_root / "prepared.json"),
        "inputs": {"pff": vendor_trial.inputs(parsed, sets["games_u"]), "skill_manifest_sha256": digest(skill / "manifest.json"),
                   "dfl_raw": raw},
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    write_new(output, record)
    print(f"D53 confirmation at cost {cost}: PFF {change['pff']:+.4f}, holdout long-hidden {change['long_hidden']:+.4f}, "
          f"primary {change['primary']:+.4f}, conventions " + ", ".join(f"{k} {v:+.4f}" for k, v in change["conventions"].items())
          + ", guards " + ", ".join(f"{k} {v:+.4f}" for k, v in change["guards"].items())
          + f" -> {'PASSED' if passed else 'FAILED'} {conditions}", flush=True)
    return passed


# -- the fallback's competition build (preparation; behind its own command) --------------------------

FALLBACK_MODELS = st.PACKAGE / "fallback_models"
BUILD_IDENTITY = "build_identity_d53.json"
BUILD_MODULES = (*st.BUILD_MODULES, "stack_trial.py")    # the modules the fallback build runs besides this one


def folder_digests(folder):
    return {path.name: digest(path) for path in sorted(Path(folder).iterdir()) if path.is_file()}


def adopt_models(*, models_root=FALLBACK_MODELS, output=None):
    """Copy D53's twelve confirmed export folders byte for byte into `fallback_models/` and verify them, written once."""
    import shutil
    require_external_only()
    output = RECIPE / "adoption_models_d53.json" if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D53's networks are copied once")
    confirmation = read(RECIPE / "confirmation_d53.json")
    if confirmation["passed"] is not True:
        raise ValueError("D53 did not pass its confirmation; nothing is copied")
    identity = confirmation["identity"]["networks"]
    copied = {}
    for offset in st.OFFSETS:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                source = export_folder(offset, model_set, folder)
                if digest(source / "fit.json") != identity[str(offset)][model_set][folder]:
                    raise ValueError(f"The source {source} differs from D53's confirmed network")
                destination = st.stack_model_folder(offset, model_set, folder, models_root)
                if destination.exists():
                    raise FileExistsError(f"{destination} exists")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
                if folder_digests(source) != folder_digests(destination):
                    raise ValueError(f"The copy of {source} differs from its source")
                if digest(destination / "fit.json") != identity[str(offset)][model_set][folder]:
                    raise ValueError(f"The copy {destination} differs from D53's confirmed network")
                copied[f"{offset}/{model_set}/{folder}"] = {"source": str(source.relative_to(ROOT)),
                                                            "files": folder_digests(destination)}
    write_new(output, {"protocol": PROTOCOL, "copied": copied, "passed": True,
                       "confirmation_sha256": digest(RECIPE / "confirmation_d53.json"),
                       "copied_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D53: {len(copied)} confirmed networks copied byte for byte into {models_root.relative_to(ROOT)} and verified",
          flush=True)


def confirmed_build(confirmation, *, models_root=FALLBACK_MODELS, recipe=RECIPE, package=st.PACKAGE):
    """D53's switches and cost after the build's identity checks against `confirmation_d53.json`; refuses otherwise.

    The same checks as `stack_trial.confirmed_build`: the confirmed lever record (direction off: this path computes no
    attacking sign), all twelve networks in `models_root`, the trees, the code the build runs, and this driver as
    locked by `build_identity_d53.json` (written after it reproduced D53's records and decoder inputs).
    """
    from dfl_submission import scale_trial
    refuse = st.BuildRefused
    if confirmation.get("passed") is not True:
        raise refuse("D53 did not pass its confirmation; this build does not run")
    chosen = confirmation["levers"]
    config = dict(chosen["config"])
    ss.validate(config)
    if config["direction"]:
        raise refuse("Identity check failed: this build path takes no attacking frame")
    identity = confirmation["identity"]
    lever_file = recipe / LEVERS
    if (identity["config"] != config or digest(lever_file) != confirmation["levers_sha256"]
            or digest(lever_file) != identity["levers_sha256"] or read(lever_file) != chosen):
        raise refuse("Identity check failed: the lever record differs from the confirmed one")
    if sorted(identity["networks"]) != sorted(str(offset) for offset in st.OFFSETS):
        raise refuse("Identity check failed: the confirmed identity does not hold three seed sets")
    for offset in st.OFFSETS:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                path = st.stack_model_folder(offset, model_set, folder, models_root) / "fit.json"
                if not path.exists() or digest(path) != identity["networks"][str(offset)][model_set][folder]:
                    raise refuse("Identity check failed: a network differs from the confirmed one")
    for key in st.SETS:
        if digest(scale_trial.TREES[key] / "fit.json") != identity["trees"][key]:
            raise refuse("Identity check failed: the trees differ from the confirmed ones")
    shas = {"stack_set.py": identity["stack_set_sha256"], "player_set.py": identity["player_set_sha256"],
            "direction_tokens.py": identity["direction_tokens_sha256"]}
    for name in BUILD_MODULES:
        if digest(package / name) != confirmation["code"][name] or (name in shas and digest(package / name) != shas[name]):
            raise refuse("Identity check failed: the code differs from the confirmed code")
    locked = recipe / BUILD_IDENTITY
    if not locked.exists():
        raise refuse("Identity check failed: no build identity record")
    lock = read(locked)
    if (lock.get("passed") is not True or lock["confirmation_sha256"] != digest(recipe / "confirmation_d53.json")
            or digest(package / "fallback_trial.py") != lock["fallback_trial_sha256"]):
        raise refuse("Identity check failed: the build driver differs from the reproduced one")
    return config, confirmation["owner_cost"]


def build(output, *, models_root=FALLBACK_MODELS):
    """The competition file with D53 at its confirmed cost, then D27's rule; no attacking sign is computed.

    The path is `stack_trial build`'s with direction off: the identity checks run before the competition tracking is
    read, the networks are loaded first (`long_context` puts the bundle reader on the module path, as `player_trial
    build`'s teacher does), and every half is read and decoded as `player_trial build` reads and decodes it.
    """
    from dfl_submission import kickoff, long_context, objective_trial  # noqa: F401 (long_context: the module path)
    from dfl_submission.build import tactical_labels, tactical_tracking_halves, write_csv
    config, cost = confirmed_build(read(RECIPE / "confirmation_d53.json"), models_root=models_root)
    _, frozen_d15 = objective_trial.d15()
    sign_map = read(st.PARITY)["sign_map"] if config["mirror"] is not None else None
    models = st.stack_models(config, models_root, export=st.stack_model_folder, sign_map=sign_map, names=("stack",))
    halves = list(tactical_tracking_halves(centered=True, pitch=frozen_d15["pitch"]))
    forecaster = st.Forecaster(models, Unsigned(), owner_cost=cost, with_d43=False)

    def decoded():
        for half, span, tracking in halves:
            print(f"Predicting tactical half {half}", flush=True)
            yield span, forecaster.forecast(*tracking)["stack"][1]

    labels = tactical_labels(decoded())
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def reproduce_build(dfl, skill, parsed, *, raw, output=None):
    """This driver reproduces D53's records and decoder inputs; then its digest is locked (`build_identity_d53.json`).

    (1) development_d53.json's entries and confirmation_d53.json's scores are recomputed from their cached decoder
    inputs; (2) the eight development guard halves are forecast afresh through the build's path with the networks in
    `fallback_models/` and must equal D53's cached decoder inputs bit for bit.
    """
    require_external_only()
    from dfl_submission import aligned_trial, diagnosis, hidden_trial as h49, hyper_trial as ht, refine_trial, vendor_trial
    output = RECIPE / BUILD_IDENTITY if output is None else output
    if output.exists():
        raise FileExistsError(f"{output} exists; D53's build identity is recorded once")
    confirmation = read(RECIPE / "confirmation_d53.json")
    if read(RECIPE / "adoption_models_d53.json")["passed"] is not True:
        raise ValueError("D53's networks are not copied")
    # (1a) development_d53.json
    development = read(RECIPE / "development_d53.json")
    games = vendor_trial.development()
    d43_dev, d43_metrica, d43_d24, truths, _ = h49.d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=st.D42_CACHE)
    visibility = h49.pff_visibility(parsed, games)
    base = h49.entry(d43_dev, d43_metrica, d43_d24, st.d43_guards(raw), ht.recipe_at(st.REFERENCE_COST), truths, visibility)
    development_same = st.canonical_json(dict(h49.compact(base), selection=st.selection(base))) == st.canonical_json(
        {key: value for key, value in development["reference"].items() if key not in ("name", "owner_cost")} |
        {"owner_cost": st.REFERENCE_COST})
    for name in st.MODEL_NAMES:
        halves = {part: st.stored_halves(SCRATCH / "cache" / name / part) for part in ("development", "metrica", "d24", "guards")}
        entries = [h49.entry(halves["development"], halves["metrica"], halves["d24"], halves["guards"], ht.recipe_at(cost),
                             truths, visibility) for cost in st.COSTS]
        recomputed = st.canonical_json([dict(h49.compact(item), selection=st.selection(item)) for item in entries])
        development_same = development_same and recomputed == development["entries"][name]
    # (1b) confirmation_d53.json
    sets = d48_sets(dfl, skill, parsed, raw=raw)
    prepared = read(CONFIRM_CACHE / "prepared.json")
    stack_main = diagnosis.collect(CONFIRM_CACHE / "stack", prepared["key"], st.refuse)
    stack_long = diagnosis.collect(CONFIRM_CACHE / "stack_long_hidden", prepared["key"], st.refuse)
    cost = confirmation["owner_cost"]
    trial = aligned_trial.evaluate(stack_main, ht.recipe_at(cost))
    long_trial = st.long_hidden_scores(stack_long, ht.recipe_at(cost))
    change = refine_trial.changes(sets["base"], trial)
    confirmation_same = (all(sets["checks"].values()) and st.canonical_json(sets["base"]) == confirmation["reference"]
                         and st.canonical_json(trial) == confirmation["trial"]
                         and st.canonical_json(long_trial) == {key: confirmation["long_hidden"]["d53"][key]
                                                               for key in ("matches", "mean")}
                         and all(st.canonical_json(change[k]) == confirmation["changes"][k] for k in change))
    print(f"D53 build reproduction: development_d53 {'same' if development_same else 'DIFFERS'}, confirmation_d53 "
          f"{'same' if confirmation_same else 'DIFFERS'}", flush=True)
    # (2) fresh forecasts through the build's path with the copied networks
    config = confirmation["levers"]["config"]
    sign_map = read(st.PARITY)["sign_map"] if config["mirror"] is not None else None
    forecaster = st.Forecaster(st.stack_models(config, FALLBACK_MODELS, export=st.stack_model_folder, sign_map=sign_map,
                                               names=("stack",)), Unsigned(), with_d43=False)
    cached = st.stored_halves(SCRATCH / "cache" / "stack" / "guards")
    tracking = h49.guard_tracking(dfl, raw=raw)
    forecasts_same = len(cached) == len(tracking)
    for half, ((source, match, index), (labels, positions, ball, teams, pitch)) in zip(cached, sorted(tracking.items()),
                                                                                       strict=True):
        signals, decoded = forecaster.forecast(positions, ball, teams, pitch)["stack"]
        same = ((half.source, half.match, half.index) == (source, match, index)
                and all(np.array_equal(a, b) for a, b in zip(signals, half.signals, strict=True))
                and np.array_equal(decoded, half.d3))
        forecasts_same = forecasts_same and same
        print(f"Reforecast guard {source} {match} half {index + 1}: {'bit for bit' if same else 'DIFFERS'}", flush=True)
    passed = bool(development_same and confirmation_same and forecasts_same)
    write_new(output, {"protocol": PROTOCOL, "passed": passed, "fallback_trial_sha256": digest(st.PACKAGE / "fallback_trial.py"),
                       "confirmation_sha256": digest(RECIPE / "confirmation_d53.json"),
                       "adoption_models_sha256": digest(RECIPE / "adoption_models_d53.json"),
                       "checks": {"development_d53_recomputed": development_same,
                                  "confirmation_d53_recomputed": confirmation_same,
                                  "guard_halves_reforecast_bit_for_bit": forecasts_same, "guard_halves": len(cached)},
                       "code": {name: digest(st.PACKAGE / name) for name in BUILD_MODULES},
                       "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D53 build identity: {'recorded (passed)' if passed else 'FAILED'}", flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("levers")
    networks = commands.add_parser("networks")
    networks.add_argument("--external", type=Path, required=True)
    networks.add_argument("--set", choices=st.SETS, action="append")
    for name in ("develop", "confirm-prepare", "confirm"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
    commands.add_parser("summary")
    commands.add_parser("adopt-models")
    repro = commands.add_parser("reproduce-build")
    for name in ("--dfl", "--skill", "--parsed"):
        repro.add_argument(name, type=Path, required=True)
    repro.add_argument("--dfl-raw", action="store_true")
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "levers":
        record_levers()
    elif args.command == "networks":
        train_networks(args.external, sets=tuple(args.set) if args.set else st.SETS)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "confirm-prepare":
        confirm_prepare(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "confirm":
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "adopt-models":
        adopt_models()
    elif args.command == "reproduce-build":
        reproduce_build(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    else:
        summary()


if __name__ == "__main__":
    main()
