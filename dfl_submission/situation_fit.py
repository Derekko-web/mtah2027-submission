"""D56: the candidates S (8,100 steps) and SL (10,800 steps) on the stack's first seed set, against the base (DECISIONS.md D56).

Each candidate is the adopted stack's recipe (D49's views, attacking-frame tokens, the y mirror; `stack_set`'s trainer
through `length_trial train`, as D52's T) on seed offset 0, trained on `situation_train`'s folders (D49's 80 records
plus 32 situation views per set). The base is the adopted stack's first seed set alone: D52's T networks.

Commands (each with `MTAH_DATA_DIR` set to a nonexistent path):
- `base --dfl DIR --skill DIR --parsed DIR`: T's development, Metrica, D24 and guard numbers recomputed from D52's caches
  must reproduce `stack_recipe/d52_T.json` exactly; T re-forecasts the eight guard halves bit for bit against its cache;
  D55's (d) and (a) views of WPY are rebuilt (checked through D43's D55 forecasts, bit for bit) and T is forecast and
  scored on them. Writes `stack_recipe/d56_base.json` once. Runs only while fewer than two trainings run.
- `queue`: trains and exports the eight networks (S, then SL; original set first), at most two trainings machine-wide.
- `score --dfl DIR --skill DIR --parsed DIR`: S and SL forecast on D47's eight groups, D49's two long-hidden guard
  views and D55's (d) and (a) views, at ownership costs 8, 10, 12 and 14; the round-13 selection score (mean of the
  composite, the guard mean and the (d) view) and D56's rule. Writes `d56_S.json`, `d56_SL.json`, `d56_decision.json`
  once. Waits while two trainings run.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dfl_submission import stack_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = st.RECIPE
SCRATCH = ROOT / "data" / "d56"
NETWORKS = SCRATCH / "networks"
CACHE = SCRATCH / "cache"
PROTOCOL = "DECISIONS.md D56"
CANDIDATES = {"S": {"steps": 8100}, "SL": {"steps": 10800}}
BASE = "T"                             # D52's T networks: the adopted stack's first seed set
OFFSET = 0
COSTS = (8, 10, 12, 14)
SELECTION_MARGIN, GROUP_LOSS = .008, .01
GROUPS = st.GROUPS
VIEWS = {"situation_d": "air_or_crowd", "situation_a": "air"}   # D55's (d) view (scored) and (a) (for information)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def config():
    """T's switches: the stack's lever record, seed offset 0."""
    return dict(st.levers()["config"], seed_offset=OFFSET)


def selection(item):
    """The round-13 selection score: the mean of D47's composite, the long-hidden guard mean and the (d) view."""
    return (item["composite"] + item["guards"]["mean"] + item["situation_d"]) / 3


# -- training ------------------------------------------------------------------------------------

def folders(model_set):
    return SCRATCH / f"sequences_{model_set}", SCRATCH / f"tokens_{model_set}", SCRATCH / f"signs_{model_set}.json"


def prepared(model_set, recipe=RECIPE):
    """The set's passed D56 preparation record, with its folders and sign file unchanged."""
    record = read(recipe / f"d56_prepare_{model_set}.json")
    sequences, tokens, signs = folders(model_set)
    if (record["passed"] is not True or record["sequences_manifest"] != digest(sequences / "manifest.json")
            or record["tokens_manifest"] != digest(tokens / "manifest.json") or record["signs_file"] != digest(signs)):
        raise ValueError(f"D56's {model_set} folders are not the ones its preparation check passed")
    return record


def raw_folder(candidate, model_set, folder, scratch=NETWORKS):
    return scratch / candidate / "raw" / model_set / folder


def export_folder(candidate, model_set, folder, scratch=NETWORKS):
    return scratch / candidate / "export" / model_set / folder


def command_for(candidate, model_set, folder, *, scratch=NETWORKS):
    """`length_trial train` (the stack's trainer with `--steps`), on D56's folders, seed offset 0."""
    prepared(model_set)
    chosen = config()
    seed, boost = st.NETWORKS[folder]
    sequences, tokens, signs = folders(model_set)
    command = [sys.executable, "-m", "dfl_submission.length_trial", "train", "--source", str(sequences), "--tokens",
               str(tokens), "--out", str(raw_folder(candidate, model_set, folder, scratch)), "--seed", str(seed),
               "--boundary-boost", str(boost), "--config", json.dumps(chosen), "--signs", str(signs)]
    if chosen["mirror"] is not None:
        command += ["--sign-map", str(st.PARITY)]
    if chosen["noise"] or chosen["aux"]:
        raise ValueError("D56 trains with the stack's switches: no token noise, no auxiliary head")
    return command + ["--steps", str(CANDIDATES[candidate]["steps"]), "--blocks", "6"]


def export_command(candidate, model_set, folder, scratch=NETWORKS):
    return [sys.executable, "-m", "dfl_submission.player_set", "export", "--source",
            str(raw_folder(candidate, model_set, folder, scratch)), "--out", str(export_folder(candidate, model_set, folder, scratch))]


def jobs(scratch=NETWORKS):
    return [(c, s, f) for c in CANDIDATES for s in st.SETS for f in st.NETWORKS
            if not (export_folder(c, s, f, scratch) / "fit.json").exists()]


def queue(*, scratch=NETWORKS, poll=30, limit=st.TRAINING_LIMIT):
    """Train and export the eight networks, at most `limit` trainings machine-wide; one queue at a time (a lock)."""
    require_external_only()
    for model_set in st.SETS:
        prepared(model_set)
    scratch.mkdir(parents=True, exist_ok=True)
    lock = open(scratch / "queue.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, running, said = jobs(scratch), {}, False
    while pending or running:
        for job, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[job]
            if code != 0:
                raise RuntimeError(f"Training {job} failed with exit code {code}")
            subprocess.run(export_command(*job, scratch), check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
            print(f"{time.strftime('%H:%M:%S')} {'/'.join(job)} trained and exported", flush=True)
        started = False
        if pending:
            job = pending[0]
            raw = raw_folder(*job, scratch)
            if (raw / "fit.json").exists():
                pending.pop(0)
                subprocess.run(export_command(*job, scratch), check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
                print(f"{time.strftime('%H:%M:%S')} {'/'.join(job)} exported (trained earlier)", flush=True)
                continue
            machine = st.running_trainings()
            if machine < limit:
                if raw.exists():
                    raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                raw.parent.mkdir(parents=True, exist_ok=True)
                log = open(SCRATCH / f"train_{'_'.join(job)}.log", "w")
                running[job] = subprocess.Popen(command_for(*job, scratch=scratch), cwd=ROOT, stdout=log,
                                                stderr=subprocess.STDOUT)
                pending.pop(0)
                started, said = True, False
                print(f"{time.strftime('%H:%M:%S')} started {'/'.join(job)} (machine-wide before: {machine})", flush=True)
            elif not said:
                print(f"{time.strftime('%H:%M:%S')} waiting: {machine} trainings machine-wide", flush=True)
                said = True
        time.sleep(15 if started else poll)
    print(f"{time.strftime('%H:%M:%S')} D56 networks all exported", flush=True)


# -- models and identities -------------------------------------------------------------------------

def base_folder(model_set, folder):
    from dfl_submission import length_trial
    return length_trial.export_folder(BASE, model_set, folder)


def models_of(folder_of, chosen, sign_map):
    from dfl_submission import scale_trial
    return {role: {folder: [st.Network(folder_of(key, folder), chosen, sign_map)] for folder in st.NETWORKS}
            for role, key in scale_trial.ROLES.items()}


def identity(name, chosen, folder_of):
    from dfl_submission import scale_trial
    result = {"protocol": PROTOCOL, "model": name, "config": chosen,
              "networks": {key: {folder: digest(folder_of(key, folder) / "fit.json") for folder in st.NETWORKS}
                           for key in st.SETS},
              "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in st.SETS},
              "stack_set_sha256": digest(PACKAGE / "stack_set.py"), "player_set_sha256": digest(PACKAGE / "player_set.py"),
              "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py"),
              "sign_map_sha256": digest(st.PARITY)}
    if name in CANDIDATES:
        result |= {"steps": CANDIDATES[name]["steps"],
                   "prepare": {key: digest(RECIPE / f"d56_prepare_{key}.json") for key in st.SETS}}
    return result


# -- D55's views and their scores --------------------------------------------------------------------

def situation_tracking(dfl):
    """{(view name, index): (labels, positions, ball, teams, pitch)} of D55's (d) and (a) views, rebuilt."""
    from dfl_submission import situation_trial as st55
    existing, height, reader = st55.read_match(dfl)
    if canonical(reader) != read(st55.CHECK)["reader"]:
        raise ValueError("The WPY reader check differs from D55's")
    views, _ = st55.build_views(existing, height)
    return {(name, i): views[name][i] for name in VIEWS.values() for i in (0, 1)}


def situation_table(tracking):
    return st.sign_table((name, ("dfl", "WPY", name, i), *value[1:]) for (name, i), value in tracking.items())


def check_against_d55(tracking):
    """The rebuilt views are D55's: D43 forecast on them equals D55's cached D43 decoder inputs, bit for bit."""
    from dfl_submission import diagnosis, player_trial, situation_trial as st55
    stored = read(st55.CACHE / "d43" / "manifest.json")
    cached = {(half.view, half.index): half for half in diagnosis.collect(st55.CACHE / "d43", stored["identity"], st.refuse)}
    teacher = player_trial.teacher()
    result = {}
    for (name, i), (_, positions, ball, teams, pitch) in sorted(tracking.items()):
        signals, labels = player_trial.half_signals(teacher, positions, ball, teams, pitch)
        half = cached[name, i]
        result[f"{name}/{i}"] = bool(all(np.array_equal(a, b) for a, b in zip(signals, half.signals, strict=True))
                                     and np.array_equal(labels, half.d3))
    return result


def situation_halves(tracking, forecast):
    from dfl_submission import diagnosis
    for (name, i), (labels, positions, ball, teams, pitch) in sorted(tracking.items()):
        signals, decoded = forecast(None, positions, ball, teams, pitch)
        yield diagnosis.HalfSignals("dfl", "WPY", name, i, np.asarray(labels), signals, decoded)
        print(f"D56 forecast {name} half {i + 1}", flush=True)


def situation_scores(halves, tracking, cost):
    """N = 3 spellF1 of the (d) and (a) views at `cost` (D55's scoring: both halves joined)."""
    from dfl_submission import diagnosis, hyper_trial as ht, situation_trial as st55
    from evaluation import possession_metrics as pm
    recipe = ht.recipe_at(cost)
    result = {}
    for key, name in VIEWS.items():
        parts = sorted((half for half in halves if half.view == name), key=lambda half: half.index)
        if [half.index for half in parts] != [0, 1]:
            raise ValueError(f"Expected two halves of the {name} view")
        _, target = st55.truth_of(tracking[name, 0][0], tracking[name, 1][0])
        prediction = np.r_[diagnosis.decode(parts[0].signals, recipe), pm.X,
                           diagnosis.decode(parts[1].signals, recipe)].astype(np.int8)
        result[key] = float(diagnosis.score(target, prediction)["spell_f1"])
    return result


def with_situation(entry, halves, tracking):
    item = dict(entry, **situation_scores(halves, tracking, entry["owner_cost"]))
    return dict(item, selection_d56=selection(item), selection_d48=st.selection(item))


# -- the base -----------------------------------------------------------------------------------------

def development_inputs(dfl, skill, parsed):
    from dfl_submission import hidden_trial as h49, metrica_sample as ms, vendor_trial
    games = vendor_trial.development()
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    return games, truths, h49.pff_visibility(parsed, games)


def base(dfl, skill, parsed, *, output=None, cache=CACHE):
    """T's d52 numbers reproduced from its caches, its guard forecasts bit for bit, and T on D55's views; written once."""
    require_external_only()
    from dfl_submission import diagnosis, hidden_trial as h49, hyper_trial as ht, length_trial
    output = RECIPE / "d56_base.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D56's base is written once")
    st.wait_for_scoring("D56 base scoring")
    d52 = read(RECIPE / "d52_T.json")
    chosen = config()
    base_identity = identity(BASE, chosen, base_folder)
    confirmation = read(RECIPE / "confirmation.json")
    checks = {"networks_are_d52_T": base_identity["networks"] == d52["identity"]["networks"],
              "networks_are_adopted_seed_set_0": base_identity["networks"] == confirmation["identity"]["networks"]["0"],
              "config_is_d52_T": chosen == d52["identity"]["config"]}
    if not all(checks.values()):
        raise SystemExit(f"D56 stops: the base is not D52's T / the adopted first seed set: {checks}")
    games, truths, visibility = development_inputs(dfl, skill, parsed)
    folder = length_trial.SCRATCH / "cache" / BASE
    halves = {part: st.stored_halves(folder / part) for part in ("development", "metrica", "d24", "guards")}
    entries = [h49.entry(halves["development"], halves["metrica"], halves["d24"], halves["guards"], ht.recipe_at(c),
                         truths, visibility) for c in COSTS]
    recomputed = canonical([dict(h49.compact(item), selection=st.selection(item)) for item in entries])
    checks["d52_T_reproduced"] = recomputed == d52["entries"]
    sign_map = read(st.PARITY)["sign_map"] if chosen["mirror"] is not None else None
    tracking = situation_tracking(dfl)
    checks["d55_views_rebuilt"] = check_against_d55(tracking)
    table = st.table_of(st.SCRATCH / "sign_table_development.json") | situation_table(tracking)
    forecaster = st.Forecaster({BASE: models_of(base_folder, chosen, sign_map)}, table, with_d43=False)
    guards = h49.guard_tracking(dfl, raw=True)
    bits = {}
    for half, ((source, match, index), (_, positions, ball, teams, pitch)) in zip(halves["guards"], sorted(guards.items()),
                                                                                  strict=True):
        signals, decoded = forecaster.forecast(positions, ball, teams, pitch)[BASE]
        bits[f"{source}/{match}/{index}"] = bool((half.source, half.match, half.index) == (source, match, index)
                                                 and all(np.array_equal(a, b) for a, b in zip(signals, half.signals))
                                                 and np.array_equal(decoded, half.d3))
        print(f"D56 base: guard {source} {match} half {index + 1} re-forecast {'bit for bit' if bits[f'{source}/{match}/{index}'] else 'DIFFERS'}",
              flush=True)
    checks["guards_bit_for_bit"] = bits
    passed = bool(checks["d52_T_reproduced"] and all(checks["d55_views_rebuilt"].values()) and all(bits.values()))
    if not passed:
        raise SystemExit(f"D56 stops: the base's checks failed: {checks}")
    key = dict(base_identity, situation=situation_identity())
    forecast = forecaster.signals_of(BASE)
    situation = diagnosis.collect(cache / BASE / "situation", key, lambda: situation_halves(tracking, forecast))
    items = [with_situation(dict(h49.compact(item)), situation, tracking) for item in entries]
    write_new(output, {"protocol": PROTOCOL, "model": "base: the adopted stack's first seed set alone (D52's T)",
                       "checks": checks, "passed": passed, "costs": list(COSTS), "entries": items,
                       "identity": base_identity, "situation_cache": digest(cache / BASE / "situation" / "manifest.json"),
                       "d52_T_sha256": digest(RECIPE / "d52_T.json"), "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    for item in items:
        print(f"D56 base cost {item['owner_cost']}: selection {item['selection_d56']:.5f}, composite {item['composite']:.5f}, "
              f"guards {item['guards']['mean']:.4f}, (d) {item['situation_d']:.4f}, (a) {item['situation_a']:.4f}, PFF "
              f"{item['groups']['pff']:.4f}", flush=True)
    return items


def situation_identity():
    from dfl_submission import situation_trial as st55
    return {"d55_check_sha256": digest(st55.CHECK), "situation_views_sha256": digest(PACKAGE / "situation_views.py"),
            "situation_trial_sha256": digest(PACKAGE / "situation_trial.py"), "views": sorted(VIEWS.values())}


# -- the candidates -------------------------------------------------------------------------------------

def best(entries):
    """The entry with the highest round-13 selection score, ties to the lower cost."""
    return max(entries, key=lambda item: (item["selection_d56"], -item["owner_cost"]))


def holds(candidate_entries, base_entries):
    """D56's rule for one candidate (at its best cost) against the base (at its best cost)."""
    chosen, reference = best(candidate_entries), best(base_entries)
    groups = {name: chosen["groups"][name] - reference["groups"][name] for name in GROUPS}
    changes = {"selection": chosen["selection_d56"] - reference["selection_d56"], "pff": groups["pff"], "groups": groups,
               "composite": chosen["composite"] - reference["composite"],
               "guard_mean": chosen["guards"]["mean"] - reference["guards"]["mean"],
               "situation_d": chosen["situation_d"] - reference["situation_d"],
               "situation_a": chosen["situation_a"] - reference["situation_a"],
               "primary": chosen["primary"] - reference["primary"]}
    conditions = {"selection": bool(changes["selection"] >= SELECTION_MARGIN), "pff": bool(changes["pff"] >= 0),
                  "groups": bool(all(value >= -GROUP_LOSS for value in groups.values()))}
    same = next(item for item in base_entries if item["owner_cost"] == chosen["owner_cost"])
    same_groups = {name: chosen["groups"][name] - same["groups"][name] for name in GROUPS}
    return {"cost": chosen["owner_cost"], "base_cost": reference["owner_cost"], "selection": chosen["selection_d56"],
            "base_selection": reference["selection_d56"], "changes": changes, "conditions": conditions,
            "holds": bool(all(conditions.values())),
            "against_base_at_same_cost": {"note": "reported only", "selection": chosen["selection_d56"] - same["selection_d56"],
                                          "pff": same_groups["pff"], "worst_group": min(same_groups.values()),
                                          "groups": same_groups}}


def decision(results):
    passing = [name for name, result in results.items() if result["holds"]]
    return max(passing, key=lambda name: results[name]["selection"]) if passing else None


def score(dfl, skill, parsed, *, cache=CACHE, scratch=NETWORKS):
    """S and SL on development, the guard views and D55's views at every cost; D56's rule; records written once."""
    require_external_only()
    from dfl_submission import diagnosis, hidden_trial as h49, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import metrica_views as mv, vendor_trial
    names = list(CANDIDATES)
    for name in (*names, "decision"):
        if (RECIPE / f"d56_{name}.json").exists():
            raise FileExistsError(f"d56_{name}.json exists; D56 records are written once")
    base_record = read(RECIPE / "d56_base.json")
    if base_record["passed"] is not True:
        raise SystemExit("D56 stops: the base record did not pass its checks")
    for model_set in st.SETS:
        prepared(model_set)
        for name in names:
            for folder in st.NETWORKS:
                if not (export_folder(name, model_set, folder, scratch) / "fit.json").exists():
                    raise ValueError(f"D56 network {name}/{model_set}/{folder} is not exported yet")
    st.wait_for_scoring("D56 scoring")
    chosen = config()
    sign_map = read(st.PARITY)["sign_map"] if chosen["mirror"] is not None else None
    games, truths, visibility = development_inputs(dfl, skill, parsed)
    tracking = situation_tracking(dfl)
    table = st.table_of(st.SCRATCH / "sign_table_development.json") | situation_table(tracking)

    def folder_of(name):
        return lambda key, folder: export_folder(name, key, folder, scratch)

    forecaster = st.Forecaster({name: models_of(folder_of(name), chosen, sign_map) for name in names}, table, with_d43=False)
    identities = {name: identity(name, chosen, folder_of(name)) for name in names}
    tags = {name: f"D56 {name} {st.json_digest(identities[name])[:16]}" for name in names}
    guards = h49.guard_tracking(dfl, raw=True)
    collected = {name: {} for name in names}
    for name in names:
        key = dict(identities[name], teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=True,
                   skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games))
        forecast = forecaster.signals_of(name)

        def produce(forecast=forecast):
            yield from diagnosis.external_halves(dfl, skill, "development", None, raw=True, signals_of=forecast)
            yield from diagnosis.pff_halves(parsed, games, None, signals_of=forecast)

        collected[name]["development"] = diagnosis.collect(cache / name / "development", key, produce)
    for name in names:
        _, collected[name]["metrica"] = ms.reference_halves("D15", cache=cache / name / "metrica", model="D56",
                                                            signals_of=forecaster.signals_of(name), variant=tags[name])
    for name in names:
        collected[name]["d24"] = mv.view_halves(cache=cache / name / "d24", signals_of=forecaster.signals_of(name),
                                                variant=tags[name])
    for name in names:
        forecast = forecaster.signals_of(name)
        guard_key = dict(identities[name], guard=h49.guard_identity({"model": name}, raw=True))

        def produce_guards(forecast=forecast):
            from dfl_submission import hidden_views as hv
            for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(guards.items()):
                signals, decoded = forecast(None, positions, ball, teams, pitch)
                yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)

        collected[name]["guards"] = diagnosis.collect(cache / name / "guards", guard_key, produce_guards)
    for name in names:
        forecast = forecaster.signals_of(name)
        key = dict(identities[name], situation=situation_identity())
        collected[name]["situation"] = diagnosis.collect(cache / name / "situation", key,
                                                         lambda forecast=forecast: situation_halves(tracking, forecast))
    base_entries = base_record["entries"]
    results, records = {}, {}
    for name in names:
        halves = collected[name]
        items = [with_situation(dict(h49.compact(h49.entry(halves["development"], halves["metrica"], halves["d24"],
                                                           halves["guards"], ht.recipe_at(c), truths, visibility))),
                                halves["situation"], tracking) for c in COSTS]
        exports = {f"{key}/{folder}": read(export_folder(name, key, folder, scratch) / "fit.json")["export_max_logit_difference"]
                   for key in st.SETS for folder in st.NETWORKS}
        results[name] = holds(items, base_entries)
        records[name] = {"protocol": PROTOCOL, "candidate": name, **CANDIDATES[name], "costs": list(COSTS),
                         "entries": items, "rule": results[name], "identity": identities[name], "exports": exports,
                         "caches": {part: digest(cache / name / part / "manifest.json")
                                    for part in ("development", "metrica", "d24", "guards", "situation")},
                         "signs_used": {"halves": len(forecaster.signs),
                                        "fallbacks": sorted({row["half"] for row in forecaster.signs if row["fallback"]})},
                         "base_sha256": digest(RECIPE / "d56_base.json"),
                         "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                                    "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": True},
                         "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    for name in names:
        write_new(RECIPE / f"d56_{name}.json", canonical(records[name]))
    taken = decision(results)
    write_new(RECIPE / "d56_decision.json", canonical({
        "protocol": PROTOCOL, "results": results, "taken": taken,
        "rule": {"selection_margin": SELECTION_MARGIN, "pff": "at least the base's at its best cost",
                 "group_loss": GROUP_LOSS, "groups": list(GROUPS)},
        "records": {name: digest(RECIPE / f"d56_{name}.json") for name in (*names, "base")},
        "decided_at": time.strftime("%Y-%m-%d %H:%M %Z")}))
    print(summary(), flush=True)
    for name, result in results.items():
        changes = result["changes"]
        print(f"D56 {name}: cost {result['cost']} (base {result['base_cost']}), selection {changes['selection']:+.4f} "
              f"(+{SELECTION_MARGIN} needed), PFF {changes['pff']:+.4f} (>= 0), worst group {min(changes['groups'].values()):+.4f} "
              f"(>= -{GROUP_LOSS}) -> {'holds' if result['holds'] else 'does not hold'}", flush=True)
    print(f"D56 decision: {taken if taken else 'neither holds; the adopted stack stands'}", flush=True)
    return taken


# -- summary ---------------------------------------------------------------------------------------

SHORT = {"dfl_clean": "clean", "dfl_masked": "masked", "dfl_dead_carried": "dead carried", "dfl_team_swap": "swap",
         "skillcorner": "SkillCorner", "pff": "PFF", "metrica": "Metrica", "d24": "D24"}


def rows(name, entries):
    for item in entries:
        yield (f"| {name} | {item['owner_cost']} | {item['selection_d56']:.5f} | {item['composite']:.5f} | "
               f"{item['guards']['mean']:.4f} | {item['situation_d']:.4f} | {item['groups']['pff']:.4f} | "
               f"{item['primary']:.5f} | " + " | ".join(f"{item['groups'][g]:.4f}" for g in GROUPS)
               + f" | {item['situation_a']:.4f} |")


def summary():
    lines = ["| Model | Cost | Selection | Composite | Guard mean | (d) view | PFF | Primary | "
             + " | ".join(SHORT[g] for g in GROUPS) + " | (a) view |", "|---|---:|" + "---:|" * (7 + len(GROUPS))]
    for name, path in (("base", RECIPE / "d56_base.json"), *((n, RECIPE / f"d56_{n}.json") for n in CANDIDATES)):
        if path.exists():
            lines += list(rows(name, read(path)["entries"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("base", "score"):
        one = sub.add_parser(action)
        one.add_argument("--dfl", type=Path, required=True)
        one.add_argument("--skill", type=Path, required=True)
        one.add_argument("--parsed", type=Path, required=True)
    sub.add_parser("queue")
    sub.add_parser("summary")
    args = parser.parse_args()
    require_external_only()
    if args.action == "base":
        base(args.dfl, args.skill, args.parsed)
    elif args.action == "score":
        score(args.dfl, args.skill, args.parsed)
    elif args.action == "queue":
        queue()
    else:
        print(summary())


if __name__ == "__main__":
    main()
