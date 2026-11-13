"""D57: the candidate SZ on the stack's first seed set, the SkillCorner air view, and D57's rule (DECISIONS.md D57).

SZ is D56's S (the adopted stack's recipe through `length_trial train`, seed offset 0, 8,100 steps, six blocks) trained
on `skill_height`'s folders: D56's 112 records per set, the 24 SkillCorner situation views made with the air rule
too. It is scored exactly as D56 scores S and SL, with `situation_fit`'s code: D47's eight groups, D49's two
long-hidden guard views and D55's (d) and (a) views at ownership costs 8, 10, 12 and 14, the round-13 selection
score, and D56's three conditions against D56's base record.

Commands (each with `MTAH_DATA_DIR` set to a nonexistent path):
- `queue`: trains and exports SZ's four networks (original set first). At most two trainings machine-wide, and at
  most one of its own until D56's decision record exists, so that D56's scoring is never held up by SZ.
- `score --dfl DIR --skill DIR --parsed DIR`: SZ on development, the guard views and D55's views; writes
  `stack_recipe/d57_SZ.json` once. Waits while two trainings run.
- `air --skill DIR`: the SkillCorner air view of the four development matches (`skill_height.air_view`), reported
  only: the base, S, SL (those whose networks are exported) and SZ at every cost, beside each model's score on the
  observed halves from its own development cache; first checks that this scoring reproduces the SkillCorner group
  of every model record it can read. Writes `stack_recipe/d57_air_view.json` once.
- `decide`: of S, SL (D56's records) and SZ that hold, the one with the highest selection score; writes
  `stack_recipe/d57_decision.json` once, only when D56's records exist.
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

from dfl_submission import situation_fit as sf, stack_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = st.RECIPE
SCRATCH = ROOT / "data" / "d57"
NETWORKS = SCRATCH / "networks"
CACHE = SCRATCH / "cache"
PROTOCOL = "DECISIONS.md D57"
NAME = "SZ"
STEPS = 8100
BLOCKS = 6
COSTS = sf.COSTS
GROUPS = sf.GROUPS
OTHERS = ("S", "SL")                  # D56's candidates
AIR_TARGET = 3.0                      # N = 3, the SkillCorner group's convention


def read(path):
    return sf.read(path)


def write_new(path, record):
    sf.write_new(path, record)


canonical = sf.canonical


# -- training ------------------------------------------------------------------------------------

def folders(model_set, scratch=SCRATCH):
    return scratch / f"sequences_{model_set}", scratch / f"tokens_{model_set}", scratch / f"signs_{model_set}.json"


def prepared(model_set, recipe=RECIPE, scratch=SCRATCH):
    """The set's passed D57 preparation record, with its folders and sign file unchanged."""
    record = read(recipe / f"d57_prepare_{model_set}.json")
    sequences, tokens, signs = folders(model_set, scratch)
    if (record["passed"] is not True or record["sequences_manifest"] != digest(sequences / "manifest.json")
            or record["tokens_manifest"] != digest(tokens / "manifest.json") or record["signs_file"] != digest(signs)):
        raise ValueError(f"D57's {model_set} folders are not the ones its preparation check passed")
    return record


def raw_folder(model_set, folder, scratch=NETWORKS):
    return scratch / NAME / "raw" / model_set / folder


def export_folder(model_set, folder, scratch=NETWORKS):
    return scratch / NAME / "export" / model_set / folder


def command_for(model_set, folder, *, scratch=NETWORKS, data=SCRATCH):
    """S's command (`situation_fit.command_for`) with D57's folders in place of D56's."""
    chosen = sf.config()
    seed, boost = st.NETWORKS[folder]
    sequences, tokens, signs = folders(model_set, data)
    command = [sys.executable, "-m", "dfl_submission.length_trial", "train", "--source", str(sequences), "--tokens",
               str(tokens), "--out", str(raw_folder(model_set, folder, scratch)), "--seed", str(seed),
               "--boundary-boost", str(boost), "--config", json.dumps(chosen), "--signs", str(signs)]
    if chosen["mirror"] is not None:
        command += ["--sign-map", str(st.PARITY)]
    if chosen["noise"] or chosen["aux"]:
        raise ValueError("D57 trains with the stack's switches: no token noise, no auxiliary head")
    return command + ["--steps", str(STEPS), "--blocks", str(BLOCKS)]


def export_command(model_set, folder, scratch=NETWORKS):
    return [sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw_folder(model_set, folder, scratch)),
            "--out", str(export_folder(model_set, folder, scratch))]


def jobs(scratch=NETWORKS):
    return [(s, f) for s in st.SETS for f in st.NETWORKS if not (export_folder(s, f, scratch) / "fit.json").exists()]


def own_limit(recipe=RECIPE):
    """One SZ training at a time until D56's scoring has written its decision, then two."""
    return 2 if (recipe / "d56_decision.json").exists() else 1


def queue(*, scratch=NETWORKS, poll=30, limit=st.TRAINING_LIMIT, running_trainings=st.running_trainings,
          own=own_limit, sleep=time.sleep):
    """Train and export the four networks: fewer than `limit` machine-wide at each start, at most `own()` of its own."""
    require_external_only()
    for model_set in st.SETS:
        prepared(model_set)
    scratch.mkdir(parents=True, exist_ok=True)
    lock = open(scratch / "queue.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, running, said = jobs(scratch), {}, None
    while pending or running:
        for job, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[job]
            if code != 0:
                raise RuntimeError(f"Training {job} failed with exit code {code}")
            subprocess.run(export_command(*job, scratch), check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
            print(f"{time.strftime('%H:%M:%S')} {NAME}/{'/'.join(job)} trained and exported", flush=True)
        started = False
        if pending:
            job = pending[0]
            raw = raw_folder(*job, scratch)
            if (raw / "fit.json").exists():
                pending.pop(0)
                subprocess.run(export_command(*job, scratch), check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
                print(f"{time.strftime('%H:%M:%S')} {NAME}/{'/'.join(job)} exported (trained earlier)", flush=True)
                continue
            machine, mine = running_trainings(), len(running)
            if machine < limit and mine < own():
                if raw.exists():
                    raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                raw.parent.mkdir(parents=True, exist_ok=True)
                log = open(scratch.parent / f"train_{NAME}_{'_'.join(job)}.log", "w")
                running[job] = subprocess.Popen(command_for(*job, scratch=scratch), cwd=ROOT, stdout=log,
                                                stderr=subprocess.STDOUT)
                pending.pop(0)
                started, said = True, None
                print(f"{time.strftime('%H:%M:%S')} started {NAME}/{'/'.join(job)} (machine-wide before: {machine}, "
                      f"own before: {mine})", flush=True)
            elif said != (machine, mine):
                print(f"{time.strftime('%H:%M:%S')} waiting: {machine} trainings machine-wide, {mine} of SZ's "
                      f"(own limit {own()})", flush=True)
                said = (machine, mine)
        sleep(15 if started else poll)
    print(f"{time.strftime('%H:%M:%S')} D57 networks all exported", flush=True)


# -- identity -----------------------------------------------------------------------------------------

def folder_of(scratch=NETWORKS):
    return lambda key, folder: export_folder(key, folder, scratch)


def identity(chosen, scratch=NETWORKS, recipe=RECIPE):
    result = sf.identity(NAME, chosen, folder_of(scratch))
    return result | {"protocol": PROTOCOL, "steps": STEPS, "blocks": BLOCKS,
                     "prepare": {key: digest(recipe / f"d57_prepare_{key}.json") for key in st.SETS},
                     "skill_height_sha256": digest(PACKAGE / "skill_height.py")}


# -- scoring SZ ---------------------------------------------------------------------------------------

def score(dfl, skill, parsed, *, cache=CACHE, scratch=NETWORKS):
    """SZ at every cost on D56's groups and views, and D56's rule against D56's base; written once."""
    require_external_only()
    from dfl_submission import diagnosis, hidden_trial as h49, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import metrica_views as mv, vendor_trial
    output = RECIPE / f"d57_{NAME}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D57 records are written once")
    base_record = read(RECIPE / "d56_base.json")
    if base_record["passed"] is not True:
        raise SystemExit("D57 stops: D56's base record did not pass its checks")
    for model_set in st.SETS:
        prepared(model_set)
        for folder in st.NETWORKS:
            if not (export_folder(model_set, folder, scratch) / "fit.json").exists():
                raise ValueError(f"D57 network {NAME}/{model_set}/{folder} is not exported yet")
    st.wait_for_scoring("D57 scoring")
    chosen = sf.config()
    sign_map = read(st.PARITY)["sign_map"] if chosen["mirror"] is not None else None
    games, truths, visibility = sf.development_inputs(dfl, skill, parsed)
    tracking = sf.situation_tracking(dfl)
    table = st.table_of(st.SCRATCH / "sign_table_development.json") | sf.situation_table(tracking)
    forecaster = st.Forecaster({NAME: sf.models_of(folder_of(scratch), chosen, sign_map)}, table, with_d43=False)
    model_identity = identity(chosen, scratch)
    tag = f"D57 {NAME} {st.json_digest(model_identity)[:16]}"
    forecast = forecaster.signals_of(NAME)
    halves = {}
    key = dict(model_identity, teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=True,
               skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games))

    def produce():
        yield from diagnosis.external_halves(dfl, skill, "development", None, raw=True, signals_of=forecast)
        yield from diagnosis.pff_halves(parsed, games, None, signals_of=forecast)

    halves["development"] = diagnosis.collect(cache / NAME / "development", key, produce)
    _, halves["metrica"] = ms.reference_halves("D15", cache=cache / NAME / "metrica", model="D57", signals_of=forecast,
                                               variant=tag)
    halves["d24"] = mv.view_halves(cache=cache / NAME / "d24", signals_of=forecast, variant=tag)
    guards = h49.guard_tracking(dfl, raw=True)
    guard_key = dict(model_identity, guard=h49.guard_identity({"model": NAME}, raw=True))

    def produce_guards():
        from dfl_submission import hidden_views as hv
        for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(guards.items()):
            signals, decoded = forecast(None, positions, ball, teams, pitch)
            yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)

    halves["guards"] = diagnosis.collect(cache / NAME / "guards", guard_key, produce_guards)
    situation_key = dict(model_identity, situation=sf.situation_identity())
    halves["situation"] = diagnosis.collect(cache / NAME / "situation", situation_key,
                                            lambda: sf.situation_halves(tracking, forecast))
    items = [sf.with_situation(dict(h49.compact(h49.entry(halves["development"], halves["metrica"], halves["d24"],
                                                          halves["guards"], ht.recipe_at(c), truths, visibility))),
                               halves["situation"], tracking) for c in COSTS]
    rule = sf.holds(items, base_record["entries"])
    exports = {f"{key}/{folder}": read(export_folder(key, folder, scratch) / "fit.json")["export_max_logit_difference"]
               for key in st.SETS for folder in st.NETWORKS}
    record = {"protocol": PROTOCOL, "candidate": NAME, "steps": STEPS, "costs": list(COSTS), "entries": items, "rule": rule,
              "identity": model_identity, "exports": exports,
              "caches": {part: digest(cache / NAME / part / "manifest.json")
                         for part in ("development", "metrica", "d24", "guards", "situation")},
              "signs_used": {"halves": len(forecaster.signs),
                             "fallbacks": sorted({row["half"] for row in forecaster.signs if row["fallback"]})},
              "base_sha256": digest(RECIPE / "d56_base.json"),
              "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                         "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": True},
              "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, canonical(record))
    print(summary(), flush=True)
    changes = rule["changes"]
    print(f"D57 {NAME}: cost {rule['cost']} (base {rule['base_cost']}), selection {changes['selection']:+.4f} "
          f"(+{sf.SELECTION_MARGIN} needed), PFF {changes['pff']:+.4f} (>= 0), worst group "
          f"{min(changes['groups'].values()):+.4f} (>= -{sf.GROUP_LOSS}) -> {'holds' if rule['holds'] else 'does not hold'}",
          flush=True)
    return rule


# -- the SkillCorner air view (reported only) ---------------------------------------------------------

def match_scores(halves, recipe, source="skillcorner"):
    """{match: N = 3 spellF1} of the source's matches decoded at `recipe`, as D47's SkillCorner group scores them
    (in the halves' order, so that their mean is computed as the group's is)."""
    from dfl_submission import diagnosis
    scored = diagnosis.matches([half for half in halves if half.source == source],
                               lambda half: diagnosis.decode(half.signals, recipe))
    return {key[1]: float(diagnosis.score(diagnosis.target(truth, AIR_TARGET), prediction)["spell_f1"])
            for key, (truth, prediction, _) in scored.items()}


def models_for_air(scratch=NETWORKS):
    """{name: (export folder function, development cache, record path)} of the base, S and SL if exported, and SZ."""
    from dfl_submission import length_trial
    result = {"base": (sf.base_folder, length_trial.SCRATCH / "cache" / sf.BASE / "development", RECIPE / "d56_base.json")}
    for name in OTHERS:
        folder = lambda key, network, name=name: sf.export_folder(name, key, network)
        if all((folder(key, network) / "fit.json").exists() for key in st.SETS for network in st.NETWORKS):
            result[name] = (folder, sf.CACHE / name / "development", RECIPE / f"d56_{name}.json")
    result[NAME] = (folder_of(scratch), CACHE / NAME / "development", RECIPE / f"d57_{NAME}.json")
    return result


def air(skill, *, cache=CACHE, scratch=NETWORKS, output=None):
    """The base, S, SL and SZ on the SkillCorner air view at every cost, beside the observed halves; written once."""
    require_external_only()
    from dfl_submission import diagnosis, hyper_trial as ht, skill_height as sh
    output = RECIPE / "d57_air_view.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D57 records are written once")
    st.wait_for_scoring("D57 air view scoring")
    chosen = sf.config()
    sign_map = read(st.PARITY)["sign_map"] if chosen["mirror"] is not None else None
    heights = sh.heights_of()
    items = sh.air_view_halves(skill, heights)
    models = models_for_air(scratch)
    table = st.sign_table((sh.AIR_VIEW, ("skillcorner", view.match, sh.AIR_VIEW, view.index), view.positions, view.ball,
                           view.teams, view.pitch) for _, view, _ in items)
    forecaster = st.Forecaster({name: sf.models_of(folder, chosen, sign_map) for name, (folder, _, _) in models.items()},
                               table, with_d43=False)
    view_identity = {"protocol": PROTOCOL, "view": sh.AIR_VIEW, "reader_sha256": digest(sh.READER),
                     "skill_height_sha256": digest(PACKAGE / "skill_height.py"),
                     "skill_manifest_sha256": digest(skill / "manifest.json")}
    collected, observed, checks, identities = {}, {}, {}, {}
    for name, (folder, development, record_path) in models.items():
        identities[name] = (identity(chosen, scratch) if name == NAME
                            else sf.identity(sf.BASE if name == "base" else name, chosen, folder))
        forecast = forecaster.signals_of(name)

        def produce(forecast=forecast, name=name):
            for half, view, _ in items:
                signals, decoded = forecast(None, view.positions, view.ball, view.teams, view.pitch)
                print(f"D57 air view: {name} forecast {view.match} half {view.index + 1}", flush=True)
                yield diagnosis.HalfSignals(view.source, view.match, view.view, view.index, np.asarray(view.labels),
                                            signals, decoded)

        collected[name] = diagnosis.collect(cache / name / "skill_air", dict(identities[name], air_view=view_identity), produce)
        if development.exists():
            stored = read(development / "manifest.json")
            halves = [diagnosis.load_half(development, entry) for entry in stored["halves"] if entry["source"] == "skillcorner"]
            observed[name] = {str(c): match_scores(halves, ht.recipe_at(c)) for c in COSTS}
            if record_path.exists():
                record = read(record_path)
                checks[name] = {str(item["owner_cost"]): bool(
                    float(np.mean(list(observed[name][str(item["owner_cost"])].values()))) == item["groups"]["skillcorner"])
                    for item in record["entries"]}
    if "base" not in checks or not all(checks["base"].values()):
        raise SystemExit(f"D57 stops: the air view's scoring does not reproduce the base's SkillCorner group: {checks}")
    if not all(all(value.values()) for value in checks.values()):
        raise SystemExit(f"D57 stops: the air view's scoring does not reproduce a record's SkillCorner group: {checks}")
    scores = {}
    for name in models:
        scores[name] = {}
        for c in COSTS:
            per_match = match_scores(collected[name], ht.recipe_at(c))
            row = {"air_view": per_match, "air_view_mean": float(np.mean(list(per_match.values())))}
            if name in observed:
                row |= {"observed": observed[name][str(c)],
                        "observed_mean": float(np.mean(list(observed[name][str(c)].values())))}
                row["air_minus_observed"] = row["air_view_mean"] - row["observed_mean"]
            scores[name][str(c)] = row
    record = {"protocol": PROTOCOL, "reported_only": True, "costs": list(COSTS), "models": sorted(models),
              "matches": sorted({view.match for _, view, _ in items}), "statistics": sh.air_view_statistics(items),
              "scores": scores, "checks_skillcorner_group_reproduced": checks,
              "observed_from_caches": sorted(observed), "identities": identities,
              "caches": {name: digest(cache / name / "skill_air" / "manifest.json") for name in models},
              "signs_used": {"halves": len(forecaster.signs),
                             "fallbacks": sorted({row["half"] for row in forecaster.signs if row["fallback"]}),
                             "signs": {row["half"]: row["sign_used"] for row in table.values()}},
              "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, canonical(record))
    print(air_summary(record), flush=True)
    return record


def air_summary(record):
    lines = ["| Model | " + " | ".join(f"cost {c}: air / observed" for c in record["costs"]) + " |",
             "|---|" + "---|" * len(record["costs"])]
    for name in record["models"]:
        cells = []
        for c in record["costs"]:
            row = record["scores"][name][str(c)]
            cells.append(f"{row['air_view_mean']:.4f} / {row['observed_mean']:.4f}" if "observed_mean" in row
                         else f"{row['air_view_mean']:.4f} / -")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# -- the decision -------------------------------------------------------------------------------------

def decide(*, output=None, recipe=None):
    """Of S, SL and SZ, those that hold under D56's three conditions, the highest selection score; written once."""
    recipe = RECIPE if recipe is None else Path(recipe)
    output = recipe / "d57_decision.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D57 records are written once")
    paths = {name: recipe / f"d56_{name}.json" for name in OTHERS} | {NAME: recipe / f"d57_{NAME}.json"}
    missing = [str(path.name) for path in [*paths.values(), recipe / "d56_decision.json"] if not path.exists()]
    if missing:
        raise SystemExit(f"D57's decision waits for {missing}")
    records = {name: read(path) for name, path in paths.items()}
    base = digest(recipe / "d56_base.json")
    if any(record["base_sha256"] != base for record in records.values()):
        raise ValueError("The candidates were not all judged against the same base record")
    base_entries = read(recipe / "d56_base.json")["entries"]
    for name, record in records.items():
        if canonical(sf.holds(record["entries"], base_entries)) != canonical(record["rule"]):
            raise ValueError(f"The rule for {name} does not recompute from its record")
    results = {name: record["rule"] for name, record in records.items()}
    taken = sf.decision(results)
    write_new(output, canonical({
        "protocol": PROTOCOL, "results": results, "taken": taken,
        "rule": {"selection_margin": sf.SELECTION_MARGIN, "pff": "at least the base's at its best cost",
                 "group_loss": sf.GROUP_LOSS, "groups": list(GROUPS),
                 "choice": "of the candidates of D56 and D57 that hold, the highest selection score"},
        "d56_decision_taken": read(recipe / "d56_decision.json")["taken"],
        "records": {name: digest(path) for name, path in paths.items()} | {"base": base,
                                                                            "d56_decision": digest(recipe / "d56_decision.json")},
        "decided_at": time.strftime("%Y-%m-%d %H:%M %Z")}))
    for name, result in results.items():
        changes = result["changes"]
        print(f"D57 {name}: cost {result['cost']}, selection {changes['selection']:+.4f}, PFF {changes['pff']:+.4f}, "
              f"worst group {min(changes['groups'].values()):+.4f} -> {'holds' if result['holds'] else 'does not hold'}",
              flush=True)
    print(f"D57 decision: {taken if taken else 'none holds; the adopted stack stands'}", flush=True)
    return taken


# -- summary ---------------------------------------------------------------------------------------

def summary():
    lines = ["| Model | Cost | Selection | Composite | Guard mean | (d) view | PFF | Primary | "
             + " | ".join(sf.SHORT[g] for g in GROUPS) + " | (a) view |", "|---|---:|" + "---:|" * (7 + len(GROUPS))]
    for name, path in (("base", RECIPE / "d56_base.json"), *((n, RECIPE / f"d56_{n}.json") for n in OTHERS),
                       (NAME, RECIPE / f"d57_{NAME}.json")):
        if path.exists():
            lines += list(sf.rows(name, read(path)["entries"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("score")
    for flag in ("--dfl", "--skill", "--parsed"):
        one.add_argument(flag, type=Path, required=True)
    two = sub.add_parser("air")
    two.add_argument("--skill", type=Path, required=True)
    for action in ("queue", "decide", "summary"):
        sub.add_parser(action)
    args = parser.parse_args()
    require_external_only()
    if args.action == "score":
        score(args.dfl, args.skill, args.parsed)
    elif args.action == "air":
        air(args.skill)
    elif args.action == "queue":
        queue()
    elif args.action == "decide":
        decide()
    else:
        print(summary())


if __name__ == "__main__":
    main()
