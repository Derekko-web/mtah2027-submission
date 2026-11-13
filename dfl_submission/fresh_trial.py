"""D54: D43, the stack with C and the D53 fallback, each as confirmed, on the four 2019–20 SkillCorner confirmation
matches (3442, 3518, 3749, 4039). Measurement only; it selects nothing (DECISIONS.md D54).

Inputs, targets and scoring are D6's (`skillcorner_2020`, `competition_adaptation.score_match`): roster players and
the ball, the metadata pitch, 25 Hz interpolation and the five-metre outlier limit; provider home and away
possession, null or unrecognised ownership excluded, ownership gaps splitting the scoring runs; spellF1 per match over
its two halves, then the mean of the four matches.

Commands (every one runs with `MTAH_DATA_DIR` set to a nonexistent path):
- `reproduce --legacy-raw DIR`: frozen D3 through D6's reader, forecast, decoder (multiplier 0) and scoring on the
  release's five DEVELOPMENT matches must reproduce D6's multiplier-0 row and target coverage
  (`competition_adaptation_recipe/development.json`) exactly. Writes `data/d54/reproduction.json` once. If it does
  not reproduce, it stops; none of the four confirmation matches is read.
- `prepare --legacy-raw DIR`: only after a passed reproduction. Checks the three models against their confirmed
  identities, then reads the four matches' tracking (no ownership), computes each half's D50 sign with its
  amendment's fallback (agreement with the first qualifying frame is reported, not enforced), forecasts D43
  (`player_trial`, as D48's confirmation forecast it), the stack and the fallback (`stack_trial.Forecaster`, the
  networks copied at adoption), checks D43 through the forecaster against `player_trial` bit for bit, and caches
  the decoder inputs in `data/d54/forecasts/`. Nothing is scored.
- `score --legacy-raw DIR`: once. Reads the provider ownership, decodes the cached inputs (D43 at cost 10, the stack
  and the fallback at 14), scores them and writes `stack_recipe/fresh_2019.json` once.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import skillcorner_2020 as sc
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stack_recipe"
RECORD = RECIPE / "fresh_2019.json"
SCRATCH = ROOT / "data" / "d54"
D6_RECORD = PACKAGE / "competition_adaptation_recipe" / "development.json"
PROTOCOL = "DECISIONS.md D54"
MATCHES = (3442, 3518, 3749, 4039)
DEVELOPMENT = (2068, 2269, 2417, 2440, 2841)
MODELS = ("D43", "stack", "fallback")
CANDIDATES = ("stack", "fallback")
COSTS = {"D43": 10, "stack": 14, "fallback": 14}
CONFIRMATIONS = {"stack": "confirmation.json", "fallback": "confirmation_d53.json"}
ADOPTIONS = {"stack": "adoption_models.json", "fallback": "adoption_models_d53.json"}
# The one driver module of each candidate that changed after its confirmation, and the reproduction lock that covers it
# (written after the changed file reproduced the confirmed records and re-forecast halves bit for bit).
LOCKS = {"stack": ("stack_trial.py", ROOT / "data" / "d48" / "build" / "build_identity_kickoff.json", "stack_trial_sha256"),
         "fallback": ("fallback_trial.py", RECIPE / "build_identity_d53.json", "fallback_trial_sha256")}
CONVENTIONS = ("raw", "aba1.75", "aba3", "aba4", "aba4.5")
PRIMARY = "aba3"
CLEAR = .01
GROUP = "skillcorner_2020"
SIGNALS = 4
MODULES = ("fresh_trial.py", "skillcorner_2020.py", "skillcorner.py", "competition_adaptation.py", "stack_trial.py",
           "fallback_trial.py", "stack_set.py", "player_set.py", "player_set3.py", "direction_tokens.py",
           "direction_trial.py", "temporal_cpu.py", "player_trial.py", "diagnosis.py", "joint_decoder.py")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    """Write a JSON record that must not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def code_identity(package=PACKAGE):
    return {name: digest(package / name) for name in MODULES}


def label_coverage(labels):
    """The share of a match's 25 Hz timeline (both halves) with an H/A target, as D6 computes it."""
    from evaluation import possession_metrics as pm
    return float(np.mean(np.concatenate(labels) < pm.D))


def means(per_match):
    """D6's mean over matches of each convention's spellF1 and macroF1."""
    return {name: {metric: float(np.mean([match[name][metric] for match in per_match.values()]))
                   for metric in ("spell_f1", "macro_f1")} for name in CONVENTIONS}


def compare_scores(computed, recorded):
    """The entries of two {matches, means} score blocks that differ (exact comparison after a JSON round trip)."""
    computed, recorded = canonical(computed), canonical(recorded)
    differences = [f"match {match}" for match in sorted(set(computed["matches"]) | set(recorded["matches"]))
                   if computed["matches"].get(match) != recorded["matches"].get(match)]
    if computed["means"] != recorded["means"]:
        differences.append("means")
    return differences


# -- 1. the reproduction check (development matches only) -------------------------------------------

def reproduce(raw, *, output=None, record=D6_RECORD):
    """Frozen D3 on the release's five development matches, through D6's own path, against D6's multiplier-0 row."""
    require_external_only()
    output = SCRATCH / "reproduction.json" if output is None else Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; the reproduction check runs once")
    from dfl_submission import competition_adaptation as ca
    if tuple(sc.DEVELOPMENT) != DEVELOPMENT or tuple(sc.HOLDOUT) != MATCHES:
        raise ValueError("The release's split is not D6's")
    stored = read(record)
    identity = {"teacher": stored["identity"]["teacher_sha256"] == digest(ca.BLEND / "recipe.json"),
                "revision": stored["identity"]["skillcorner_revision"] == sc.REVISION}
    teacher = ca.reference()
    recipe = teacher[2]["recipe"]
    per_match, coverage = {}, {}
    for match in DEVELOPMENT:
        tracking = sc.read_tracking(raw, match)
        predictions = {"0.0": [ca.decode(ca.forecast(half, teacher), None, 0.0, recipe) for half in tracking]}
        labels = sc.read_ownership(raw, match)
        per_match[str(match)] = ca.score_match(labels, predictions)["0.0"]
        coverage[str(match)] = label_coverage(labels)
        print(f"D54 reproduction: development match {match} scored", flush=True)
    computed = canonical({"matches": per_match, "means": means(per_match)})
    differences = compare_scores(computed, stored["scores"]["0.0"])
    coverage_equal = {match: value == stored["targets"][match]["label_coverage"] for match, value in coverage.items()}
    passed = bool(all(identity.values()) and not differences and all(coverage_equal.values()))
    write_new(output, {"protocol": PROTOCOL, "matches": list(DEVELOPMENT), "identity": identity,
                       "scores": computed, "differences": differences, "coverage": coverage,
                       "coverage_equal": coverage_equal, "passed": passed, "d6_record_sha256": digest(record),
                       "code": code_identity(), "run_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    primary, raw_mean = computed["means"][PRIMARY]["spell_f1"], computed["means"]["raw"]["spell_f1"]
    print(f"D54 reproduction: N = 3 spellF1 {primary:.5f}, raw {raw_mean:.5f}; "
          f"{'REPRODUCES' if passed else 'DOES NOT REPRODUCE'} D6's multiplier-0 row "
          f"(differences {differences}, coverage equal {all(coverage_equal.values())}, identity {identity})", flush=True)
    if not passed:
        raise SystemExit("D54 stops: D3 does not reproduce D6's development row; no confirmation match is read")
    return passed


def reproduced(scratch=SCRATCH):
    path = Path(scratch) / "reproduction.json"
    if not path.exists() or read(path).get("passed") is not True:
        raise SystemExit("D54 stops: no passed reproduction check; no confirmation match is read")
    return digest(path)


# -- 2. the confirmed models --------------------------------------------------------------------------

def network_checks(models_root, adoption, confirmation, *, offsets, sets, networks, folder_of):
    """Every file of every network equals the adoption copy record, and every fit.json the confirmed identity."""
    identity = confirmation["identity"]["networks"]
    failures = []
    for offset in offsets:
        for model_set in sets:
            for folder in networks:
                path = folder_of(offset, model_set, folder, models_root)
                files = {item.name: digest(item) for item in sorted(Path(path).iterdir()) if item.is_file()} \
                    if Path(path).is_dir() else {}
                if files != adoption["copied"][f"{offset}/{model_set}/{folder}"]["files"]:
                    failures.append(f"{offset}/{model_set}/{folder}: files differ from the adoption copy")
                elif digest(Path(path) / "fit.json") != identity[str(offset)][model_set][folder]:
                    failures.append(f"{offset}/{model_set}/{folder}: fit.json differs from the confirmed identity")
    return failures


def code_check(confirmation, confirmation_path, lock, package=PACKAGE):
    """(passed, modules that differ): every module the confirmation recorded is unchanged, except the candidate's own
    driver, which must equal the passed reproduction lock written for this confirmation."""
    module, lock_path, field = lock
    differ = sorted(name for name, value in confirmation["code"].items() if digest(package / name) != value)
    locked = read(lock_path) if Path(lock_path).exists() else {}
    covered = (locked.get("passed") is True and locked.get("confirmation_sha256") == digest(confirmation_path)
               and locked.get(field) == digest(package / module))
    return bool(set(differ) <= {module} and (module not in differ or covered)), differ


def confirmed_models(recipe=RECIPE, package=PACKAGE, locks=LOCKS):
    """{name: (config, cost, models_root)} of the stack and the fallback, plus the identity checks of all three models.

    Refuses unless both candidates passed their confirmation, their networks are the adopted copies of the confirmed
    ones, the trees and the modules equal the confirmed identity, the costs are the confirmed ones, and D43's
    networks and trees are the ones D48's confirmation used.
    """
    from dfl_submission import fallback_trial as ft, player_trial, scale_trial, stack_set as ss, stack_trial as st
    roots = {"stack": st.MODELS, "fallback": ft.FALLBACK_MODELS}
    result, checks, failures = {}, {}, []
    for name in CANDIDATES:
        confirmation = read(recipe / CONFIRMATIONS[name])
        adoption = read(recipe / ADOPTIONS[name])
        config = dict(confirmation["levers"]["config"])
        ss.validate(config)
        identity = confirmation["identity"]
        part = {"passed": confirmation["passed"] is True,
                "cost": confirmation["owner_cost"] == COSTS[name],
                "config": identity["config"] == config and config["direction"] is (name == "stack"),
                "adoption_of_confirmation": adoption["confirmation_sha256"] == digest(recipe / CONFIRMATIONS[name]),
                "trees": all(digest(scale_trial.TREES[key] / "fit.json") == identity["trees"][key] for key in st.SETS),
                "modules": all(digest(package / module) == identity[key] for module, key in (
                    ("stack_set.py", "stack_set_sha256"), ("player_set.py", "player_set_sha256"),
                    ("direction_tokens.py", "direction_tokens_sha256")))}
        part["code"], part["code_changed_since_confirmation"] = code_check(
            confirmation, recipe / CONFIRMATIONS[name], locks[name], package)
        networks = network_checks(roots[name], adoption, confirmation, offsets=st.OFFSETS, sets=st.SETS,
                                  networks=st.NETWORKS, folder_of=st.stack_model_folder)
        part["networks"] = not networks
        failures += [f"{name}: {key}" for key, value in part.items() if value is False]
        failures += [f"{name}: {item}" for item in networks]
        checks[name] = part
        result[name] = (config, confirmation["owner_cost"], roots[name])
    d43 = read(recipe / CONFIRMATIONS["stack"])["d43_models"]
    checks["D43"] = {"identity": canonical(player_trial.identity()) == d43}
    if not checks["D43"]["identity"]:
        failures.append("D43: networks or trees differ from the ones D48's confirmation used")
    if failures:
        raise SystemExit(f"D54 stops: a model differs from its confirmed identity: {failures}")
    return result, checks


# -- 3. signs, forecasts and caches (no score) ---------------------------------------------------------

def half_key(match, index):
    return (GROUP, str(match), "observed", index)


def sign_report(table):
    """Per half: D50's whole-half sign, its first-qualifying-frame sign and agreement, and any fallback (not enforced)."""
    halves = {row["half"]: {key: row[key] for key in ("sign", "first_sign", "agrees", "d", "frames", "first_frame",
                                                      "max_visible", "max_visible_all_frames", "sign_used", "fallback")}
              for row in table.values()}
    signed = [half for half, row in halves.items() if row["sign"] is not None]
    by_match = {}
    for half, row in halves.items():
        by_match.setdefault(half.split("/")[1], {})[half] = row["sign_used"]
    return {"halves": dict(sorted(halves.items())),
            "fallbacks": sorted(half for half, row in halves.items() if row["fallback"] is not None),
            "fallback_count": sum(row["fallback"] is not None for row in halves.values()),
            "first_frame_disagrees": sorted(half for half in signed if not halves[half]["agrees"]),
            "signed_count": len(signed),
            "same_sign_matches": sorted(match for match, rows in by_match.items()
                                        if len(rows) == 2 and len(set(rows.values())) == 1)}


def cache_name(match, index):
    return f"{match}_{index}.npz"


def save_forecast(path, key, results, d43_equal):
    """One half's decoder inputs and labels of every model, published only when complete."""
    arrays = {"key": np.array(json.dumps(key, sort_keys=True)), "d43_forecaster_equal": np.array(bool(d43_equal))}
    for name in MODELS:
        signals, labels = results[name]
        if len(signals) != SIGNALS:
            raise ValueError(f"{name}: expected {SIGNALS} decoder signals")
        for index, signal in enumerate(signals):
            arrays[f"{name}__signal_{index}"] = np.asarray(signal)
        arrays[f"{name}__labels"] = np.asarray(labels)
    partial = path.with_name(path.stem + ".part.npz")
    np.savez_compressed(partial, **arrays)
    partial.replace(path)


def load_forecast(path, key, *, with_check=False):
    with np.load(path, allow_pickle=False) as item:
        if json.loads(str(item["key"])) != canonical(key):
            raise ValueError(f"{path} was cached under a different identity")
        result = {name: (tuple(item[f"{name}__signal_{index}"] for index in range(SIGNALS)), item[f"{name}__labels"])
                  for name in MODELS}
        return (result, bool(item["d43_forecaster_equal"])) if with_check else result


def cache_identity(models, checks):
    from dfl_submission import stack_trial as st
    return canonical({"protocol": PROTOCOL, "revision": sc.REVISION,
                      "blobs": {str(match): list(sc.BLOBS[match]) for match in MATCHES},
                      "models": {name: {"config": config, "cost": cost, "root": str(Path(root).relative_to(ROOT)),
                                        "confirmation_sha256": digest(RECIPE / CONFIRMATIONS[name]),
                                        "adoption_sha256": digest(RECIPE / ADOPTIONS[name])}
                                 for name, (config, cost, root) in models.items()},
                      "d43": read(RECIPE / CONFIRMATIONS["stack"])["d43_models"], "checks": checks,
                      "sign_map_sha256": digest(st.PARITY), "code": code_identity()})


def prepare(raw, *, scratch=SCRATCH, record=RECORD):
    """Everything before D54's scores; the provider ownership is not read here."""
    require_external_only()
    scratch = Path(scratch)
    if Path(record).exists():
        raise FileExistsError(f"{record} exists; D54 is scored once")
    reproduction_sha256 = reproduced(scratch)
    prepared_path = scratch / "prepared.json"
    if prepared_path.exists():
        raise FileExistsError(f"{prepared_path} exists; D54 is prepared once")
    from dfl_submission import player_trial, stack_trial as st
    models, checks = confirmed_models()
    key = cache_identity(models, checks)
    sign_map = read(st.PARITY)["sign_map"]
    if any(config["mirror"] is None for config, _, _ in models.values()):
        raise ValueError("Both candidates were confirmed with a pitch mirror")
    # The four confirmation matches are read from here on (tracking only).
    halves = {(match, index): half for match in MATCHES for index, half in enumerate(sc.read_tracking(raw, match))}
    if sorted(halves) != [(match, index) for match in MATCHES for index in (0, 1)]:
        raise ValueError(f"Expected two halves per match, got {sorted(halves)}")
    table = st.sign_table((GROUP, half_key(match, index), *half) for (match, index), half in halves.items())
    signs = sign_report(table)
    print(f"D54 signs: {signs['signed_count']} of 8 halves signed, fallbacks {signs['fallbacks']}, first-frame "
          f"disagreements {signs['first_frame_disagrees']} (reported, not enforced)", flush=True)
    stacked = {name: st.stack_models(config, root, export=st.stack_model_folder, sign_map=sign_map,
                                     names=("stack",))["stack"] for name, (config, _, root) in models.items()}
    forecaster = st.Forecaster(stacked, table, with_d43=True)
    teacher = player_trial.teacher()
    folder = scratch / "forecasts"
    folder.mkdir(parents=True, exist_ok=True)
    d43_equal = {}
    for (match, index), half in halves.items():
        path = folder / cache_name(match, index)
        if path.exists():
            _, d43_equal[f"{match}/{index}"] = load_forecast(path, key, with_check=True)
            print(f"D54: {path.name} already cached under this identity", flush=True)
            continue
        started = time.time()
        computed = forecaster.forecast(*half)
        d43 = player_trial.half_signals(teacher, *half)
        through = computed.pop("D43")
        d43_equal[f"{match}/{index}"] = bool(
            all(np.array_equal(a, b) for a, b in zip(d43[0], through[0], strict=True))
            and np.array_equal(d43[1], through[1]))
        if not d43_equal[f"{match}/{index}"]:
            raise SystemExit(f"D54 stops: D43 through the forecaster differs from player_trial on {match}/{index}")
        save_forecast(path, key, dict(computed, D43=d43), d43_equal[f"{match}/{index}"])
        print(f"D54: forecast {match}/{index} ({time.time() - started:.0f} s)", flush=True)
    manifest = {path.name: digest(path) for path in sorted(folder.glob("*.npz")) if not path.name.endswith(".part.npz")}
    if sorted(manifest) != sorted(cache_name(match, index) for match, index in halves):
        raise ValueError("The forecast cache does not hold exactly the eight halves")
    write_new(prepared_path, {"protocol": PROTOCOL, "reproduction_sha256": reproduction_sha256, "identity": key,
                              "signs": signs, "d43_forecaster_equal": d43_equal, "forecasts": manifest,
                              "ownership_read": False, "prepared_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print("D54 prepared: models checked, eight halves forecast and cached; nothing scored", flush=True)


# -- 4. the scores, once ------------------------------------------------------------------------------

def reading(scores):
    """D54's reading of each candidate against D43 (supported: N = 3 mean >= D43's; clearly: >= D43's + 0.01)."""
    base = scores["D43"]
    result = {}
    for name in CANDIDATES:
        trial = scores[name]
        primary, reference = trial["means"][PRIMARY]["spell_f1"], base["means"][PRIMARY]["spell_f1"]
        result[name] = {
            "primary": primary, "d43_primary": reference, "primary_change": primary - reference,
            "supported": bool(primary >= reference), "clearly_supported": bool(primary >= reference + CLEAR),
            "match_changes": {match: trial["matches"][match][PRIMARY]["spell_f1"] - base["matches"][match][PRIMARY]["spell_f1"]
                              for match in base["matches"]},
            "convention_changes": {convention: trial["means"][convention]["spell_f1"] - base["means"][convention]["spell_f1"]
                                   for convention in CONVENTIONS},
            "primary_macro_change": trial["means"][PRIMARY]["macro_f1"] - base["means"][PRIMARY]["macro_f1"]}
    return result


def score(raw, *, scratch=SCRATCH, output=None):
    """D54's scores, once, from the prepared caches."""
    require_external_only()
    scratch = Path(scratch)
    output = RECORD if output is None else Path(output)
    if output.exists():
        raise FileExistsError(f"{output} exists; D54 is scored once")
    prepared_path = scratch / "prepared.json"
    if not prepared_path.exists():
        raise SystemExit("D54 is not prepared")
    if (scratch / "score_started").exists():
        raise SystemExit("D54's scoring has started before; it is not run again")
    from dfl_submission import competition_adaptation as ca, diagnosis, hyper_trial as ht
    prepared = read(prepared_path)
    models, checks = confirmed_models()
    key = cache_identity(models, checks)
    if key != prepared["identity"]:
        raise SystemExit("D54 stops: the models or code differ from the prepared ones")
    folder = scratch / "forecasts"
    for name, value in prepared["forecasts"].items():
        if digest(folder / name) != value:
            raise SystemExit(f"D54 stops: the cached forecast {name} differs from the prepared one")
    if tuple(sorted(CONVENTIONS)) != tuple(sorted(read_conventions())):
        raise ValueError("D6's scoring conventions differ from D54's list")
    with (scratch / "score_started").open("x") as marker:
        marker.write(digest(prepared_path) + "\n")
    recipes = {name: ht.recipe_at(COSTS[name]) for name in MODELS}
    per_model = {name: {} for name in MODELS}
    coverage = {}
    for match in MATCHES:
        cached = [load_forecast(folder / cache_name(match, index), key) for index in (0, 1)]
        labels = sc.read_ownership(raw, match)
        if [len(part) for part in labels] != [len(half["D43"][1]) for half in cached]:
            raise ValueError(f"Match {match}: target and forecast lengths differ")
        coverage[str(match)] = label_coverage(labels)
        predictions = {name: [diagnosis.decode(half[name][0], recipes[name]) for half in cached] for name in MODELS}
        for name, result in ca.score_match(labels, predictions).items():
            per_model[name][str(match)] = result
        print(f"D54: scored match {match}", flush=True)
    scores = canonical({name: {"matches": matches, "means": means(matches)} for name, matches in per_model.items()})
    read_out = canonical(reading(scores))
    record = canonical({
        "protocol": PROTOCOL, "matches": list(MATCHES), "costs": COSTS, "primary": PRIMARY, "clear_margin": CLEAR,
        "reproduction": {key_: read(scratch / "reproduction.json")[key_] for key_ in ("passed", "scores", "coverage")}
        | {"sha256": prepared["reproduction_sha256"], "d6_row": "competition_adaptation_recipe/development.json 0.0"},
        "models": prepared["identity"]["models"], "model_checks": checks, "coverage": coverage, "scores": scores,
        "reading": read_out, "signs": prepared["signs"], "d43_forecaster_equal": prepared["d43_forecaster_equal"],
        "prepared_sha256": digest(prepared_path), "inputs": {"revision": sc.REVISION,
                                                            "blobs": prepared["identity"]["blobs"]},
        "code": code_identity(), "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    write_new(output, record)
    print(summary(record), flush=True)
    return record


def read_conventions():
    from dfl_submission import decoder_trial
    return decoder_trial.CONVENTIONS


def summary(record):
    lines = ["| Match | coverage | " + " | ".join(MODELS) + " |", "|---|---:|" + "---:|" * len(MODELS)]
    for match in record["matches"]:
        values = [record["scores"][name]["matches"][str(match)][PRIMARY]["spell_f1"] for name in MODELS]
        lines.append(f"| {match} | {record['coverage'][str(match)]:.3f} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    for convention in (PRIMARY, "aba1.75", "aba4", "aba4.5", "raw"):
        values = [record["scores"][name]["means"][convention]["spell_f1"] for name in MODELS]
        lines.append(f"| mean {convention} | | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    for name in CANDIDATES:
        item = record["reading"][name]
        lines.append(f"{name}: N = 3 change {item['primary_change']:+.4f}; supported {item['supported']}, "
                     f"clearly supported {item['clearly_supported']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("reproduce", "prepare", "score"):
        command = commands.add_parser(action)
        command.add_argument("--legacy-raw", required=True, type=Path, help="SkillCorner 2020 JSON cache")
    args = parser.parse_args()
    {"reproduce": reproduce, "prepare": prepare, "score": score}[args.action](args.legacy_raw)


if __name__ == "__main__":
    main()
