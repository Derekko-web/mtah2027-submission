"""D52: training length (T) and temporal context (X) for the stack's recipe, first seed set only.

Commands (DECISIONS.md D52):
- `train` trains one network: `stack_set.train` with `--steps` (T: 8,100), or `context_set.train` with seven
  temporal blocks (X); its arguments are `stack_set`'s command line plus `--steps` and `--blocks`, so at
  5,400 steps and six blocks the call is the stack's;
- `queue` trains and exports both candidates' eight networks (T first) as the orchestrator schedules them:
  a network starts only while fewer than two trainings run machine-wide, at most one of D52's at a time
  while D48's stack is still training, two once its twelve networks are exported;
- `score` forecasts T and X on D48's development groups, D49's guard views and D48's costs (D43
  recomputed on the way, bit for bit against its caches), writes one record per candidate, and applies
  D52's rule if D48's development record (the first seed set alone) exists;
- `decide` applies D52's rule later, once D48's record exists.
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

from dfl_submission import context_set as cs, stack_set as ss, stack_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = st.RECIPE
SCRATCH = ROOT / "data" / "d52"
PROTOCOL = "DECISIONS.md D52"
CANDIDATES = {"T": {"steps": 8100, "blocks": 6}, "X": {"steps": 5400, "blocks": 7}}
BLOCKS = {6: cs.SIX, 7: cs.SEVEN}
OFFSET = 0                         # the first seed set
BASE = "seed_0"                    # its name in D48's development record
COSTS = st.COSTS
SELECTION_MARGIN, PFF_LOSS, GROUP_LOSS = .008, .005, .01
GROUPS = st.GROUPS
STACK_NETWORKS = len(st.OFFSETS) * len(st.SETS) * len(st.NETWORKS)


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D52 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


# -- one network -------------------------------------------------------------------------------------

def parser_for_train(parser):
    """`stack_set`'s training arguments plus `--steps` and `--blocks`."""
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--boundary-boost", type=float, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--signs", type=Path)
    parser.add_argument("--sign-map", type=Path)
    parser.add_argument("--pitches", type=Path)
    parser.add_argument("--aux", type=Path)
    parser.add_argument("--steps", type=int, default=5400)
    parser.add_argument("--blocks", type=int, choices=tuple(BLOCKS), default=6)
    return parser


def train_one(args):
    """The call `stack_set.main` makes (with `steps`), or the same call to `context_set.train` with seven blocks."""
    config = json.loads(args.config)
    ss.validate(config)
    if args.aux is not None or config["aux"]:
        raise ValueError("D52 trains no auxiliary head")
    signs = ss.read_signs(args.signs, args.source, args.tokens) if config["direction"] else None
    sign_map = None
    if config["mirror"] is not None:
        record = json.loads(args.sign_map.read_text())
        if not record["passed"]:
            raise ValueError("The mirror parity check did not pass; nothing is trained with mirrors")
        sign_map = record["sign_map"]
    provenance = {"signs_file": None if args.signs is None else digest(args.signs)}
    common = dict(seed=args.seed, boundary_boost=args.boundary_boost, config=config, signs=signs, sign_map=sign_map,
                  pitch_file=args.pitches, provenance=provenance, steps=args.steps)
    if args.blocks == 6:
        ss.train(args.source, args.tokens, args.out, args.device, aux_folder=args.aux, **common)
    else:
        cs.train(args.source, args.tokens, args.out, args.device, dilations=BLOCKS[args.blocks], **common)


def raw_folder(candidate, model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / candidate / "raw" / model_set / folder


def export_folder(candidate, model_set, folder, scratch=SCRATCH / "networks"):
    return scratch / candidate / "export" / model_set / folder


def command_for(candidate, model_set, folder, external, *, scratch=SCRATCH / "networks"):
    """The training command of one network: `stack_trial.train_networks`' command for offset 0, through `train`."""
    chosen = st.levers()
    config = dict(chosen["config"], seed_offset=OFFSET)
    seed, boost = st.NETWORKS[folder]
    source, tokens = st.data_folders(chosen["data"], model_set, external)
    signs_file = st.SCRATCH / f"signs_{model_set}.json"
    if digest(signs_file) != read(RECIPE / "signs.json")["files"][model_set]:
        raise ValueError(f"{signs_file} is not the file D48's sign check passed on")
    spec = CANDIDATES[candidate]
    command = [sys.executable, "-m", "dfl_submission.length_trial", "train", "--source", str(source), "--tokens",
               str(tokens), "--out", str(raw_folder(candidate, model_set, folder, scratch)), "--seed", str(seed),
               "--boundary-boost", str(boost), "--config", json.dumps(config), "--signs", str(signs_file)]
    if config["mirror"] is not None:
        command += ["--sign-map", str(st.PARITY)]
    if config["noise"]:
        command += ["--pitches", str(st.SCRATCH / f"pitches_{model_set}.json")]
    if config["aux"]:
        raise ValueError("D52 trains no auxiliary head")
    return command + ["--steps", str(spec["steps"]), "--blocks", str(spec["blocks"])]


def export_command(candidate, model_set, folder, scratch=SCRATCH / "networks"):
    module = "dfl_submission.player_set" if CANDIDATES[candidate]["blocks"] == 6 else "dfl_submission.context_set"
    return [sys.executable, "-m", module, "export", "--source", str(raw_folder(candidate, model_set, folder, scratch)),
            "--out", str(export_folder(candidate, model_set, folder, scratch))]


def jobs(candidates=tuple(CANDIDATES), scratch=SCRATCH / "networks"):
    """(candidate, set, network) still to train or export, in order: T before X, current before augmented."""
    return [(c, s, f) for c in candidates for s in st.SETS for f in st.NETWORKS
            if not (export_folder(c, s, f, scratch) / "fit.json").exists()]


def stack_finished(scratch=st.SCRATCH / "networks"):
    return all((st.export_folder(o, s, f, scratch) / "fit.json").exists()
               for o in st.OFFSETS for s in st.SETS for f in st.NETWORKS)


def allowed(running_machine, running_mine, finished):
    """Whether one more D52 training may start: under two machine-wide, and one of D52's at a time while the stack trains."""
    return running_machine < st.TRAINING_LIMIT and running_mine < (2 if finished else 1)


def queue(external, *, scratch=SCRATCH / "networks", poll=30):
    """Train and export the eight networks, as the orchestrator schedules them; one queue at a time (a lock)."""
    require_external_only()
    for name in (*st.PIPELINE_CHECKS, "signs.json"):
        st.passed_record(name)
    scratch.mkdir(parents=True, exist_ok=True)
    lock = open(scratch / "queue.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = jobs(scratch=scratch)
    running = {}
    waiting_said = False
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
            machine, finished = st.running_trainings(), stack_finished()
            if allowed(machine, len(running), finished):
                if raw.exists():
                    raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                raw.parent.mkdir(parents=True, exist_ok=True)
                log = open(SCRATCH / f"train_{'_'.join(job)}.log", "w")
                running[job] = subprocess.Popen(command_for(*job, external, scratch=scratch), cwd=ROOT, stdout=log,
                                                stderr=subprocess.STDOUT)
                pending.pop(0)
                started, waiting_said = True, False
                print(f"{time.strftime('%H:%M:%S')} started {'/'.join(job)} (machine-wide before: {machine}, "
                      f"stack {'finished' if finished else 'training'})", flush=True)
            elif not waiting_said:
                print(f"{time.strftime('%H:%M:%S')} waiting: {machine} trainings machine-wide, {len(running)} of D52's, "
                      f"stack {'finished' if finished else 'training'}", flush=True)
                waiting_said = True
        time.sleep(5 if started else poll)
    print(f"{time.strftime('%H:%M:%S')} D52 networks all exported", flush=True)


# -- scoring ---------------------------------------------------------------------------------------

class ContextNetwork(st.Network):
    """`stack_trial.Network` whose inference is `context_set.predict` (any number of temporal blocks)."""

    def predict(self, frames, owner, player_tokens, length, sign):
        columns = list(self.checkpoint.columns)
        values = owner[columns].to_numpy(dtype=np.float32)
        return cs.predict(self.checkpoint.checkpoint, frames, values, columns, player_tokens, length,
                          config=self.config, sign=sign if self.config["direction"] else None, sign_map=self.sign_map)


def candidate_models(config, scratch=SCRATCH / "networks", sign_map=None):
    """{candidate: {role: {folder: [network]}}}: T with `stack_set`'s inference, X with `context_set`'s."""
    from dfl_submission import scale_trial
    result = {}
    for candidate, spec in CANDIDATES.items():
        kind = st.Network if spec["blocks"] == 6 else ContextNetwork
        result[candidate] = {role: {folder: [kind(export_folder(candidate, key, folder, scratch), config, sign_map)]
                                    for folder in st.NETWORKS} for role, key in scale_trial.ROLES.items()}
    return result


def model_identity(candidate, config, scratch=SCRATCH / "networks"):
    from dfl_submission import scale_trial
    return {"protocol": PROTOCOL, "candidate": candidate, "config": dict(config, seed_offset=OFFSET), **CANDIDATES[candidate],
            "networks": {key: {folder: digest(export_folder(candidate, key, folder, scratch) / "fit.json")
                               for folder in st.NETWORKS} for key in st.SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in st.SETS},
            "stack_set_sha256": digest(PACKAGE / "stack_set.py"), "context_set_sha256": digest(PACKAGE / "context_set.py"),
            "player_set_sha256": digest(PACKAGE / "player_set.py"),
            "direction_tokens_sha256": digest(PACKAGE / "direction_tokens.py"),
            "signs_sha256": digest(RECIPE / "signs.json"), "levers_sha256": digest(RECIPE / "levers.json")}


def best(entries):
    """The entry with the highest selection score, ties to the lower cost (D48's gate)."""
    return max(entries, key=lambda item: (st.selection(item), -item["owner_cost"]))


def holds(candidate_entries, base_entries):
    """D52's rule for one candidate against the base (both at their best cost by selection score)."""
    chosen, base = best(candidate_entries), best(base_entries)
    groups = {name: chosen["groups"][name] - base["groups"][name] for name in GROUPS}
    changes = {"selection": st.selection(chosen) - st.selection(base), "pff": groups["pff"], "groups": groups,
               "composite": chosen["composite"] - base["composite"],
               "guard_mean": chosen["guards"]["mean"] - base["guards"]["mean"]}
    conditions = {"selection": bool(changes["selection"] >= SELECTION_MARGIN), "pff": bool(changes["pff"] >= -PFF_LOSS),
                  "groups": bool(all(value >= -GROUP_LOSS for value in groups.values()))}
    same = next(item for item in base_entries if item["owner_cost"] == chosen["owner_cost"])
    same_groups = {name: chosen["groups"][name] - same["groups"][name] for name in GROUPS}
    reported = {"note": "reported only: the base at the candidate's cost instead of at its best cost",
                "pff": same_groups["pff"], "worst_group": min(same_groups.values()), "groups": same_groups}
    return {"cost": chosen["owner_cost"], "base_cost": base["owner_cost"], "selection": st.selection(chosen),
            "base_selection": st.selection(base), "changes": changes, "conditions": conditions,
            "holds": bool(all(conditions.values())), "against_base_at_same_cost": reported}


def decision(results):
    """{candidate: holds(...)} -> the candidate taken: the higher selection score among those that hold, else None."""
    passing = [name for name, result in results.items() if result["holds"]]
    return max(passing, key=lambda name: results[name]["selection"]) if passing else None


def base_entries():
    """The first seed set alone at every cost, from D48's development record (None until it exists)."""
    path = RECIPE / "development.json"
    if not path.exists():
        return None
    record = read(path)
    entries = record["entries"][BASE]
    if [item["owner_cost"] for item in entries] != list(COSTS):
        raise ValueError("D48's record does not hold the first seed set at D52's costs")
    return entries, record["identity"][BASE]


def decide(output=None):
    require_external_only()
    output = RECIPE / "d52_decision.json" if output is None else output
    found = base_entries()
    if found is None:
        print("D52: D48's development record is not written yet; the comparison waits for it", flush=True)
        return None
    entries, base_identity = found
    results = {name: holds(read(RECIPE / f"d52_{name}.json")["entries"], entries) for name in CANDIDATES}
    taken = decision(results)
    write_new(output, {"protocol": PROTOCOL, "base": {"name": "D48 first seed set alone", "identity": base_identity,
                                                      "entries": entries},
                       "results": results, "taken": taken, "decided_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    for name, result in results.items():
        print(f"D52 {name}: cost {result['cost']} (base {result['base_cost']}), selection {result['changes']['selection']:+.4f} "
              f"(+{SELECTION_MARGIN} needed), PFF {result['changes']['pff']:+.4f} (>= -{PFF_LOSS}), worst group "
              f"{min(result['changes']['groups'].values()):+.4f} (>= -{GROUP_LOSS}) -> "
              f"{'holds' if result['holds'] else 'does not hold'}", flush=True)
    print(f"D52 decision: {taken if taken else 'neither holds; D48 goes on as trained'}", flush=True)
    return taken


def score(dfl, skill, parsed, *, raw, cache_root=SCRATCH / "cache", scratch=SCRATCH / "networks"):
    """T and X on development at D48's costs: forecasts as `stack_trial.develop` makes them, one record per candidate."""
    require_external_only()
    from dfl_submission import diagnosis, hidden_trial as h49, hyper_trial as ht, metrica_sample as ms
    from dfl_submission import metrica_views as mv, vendor_trial
    names = list(CANDIDATES)
    for name in names:
        if (RECIPE / f"d52_{name}.json").exists():
            raise FileExistsError(f"{RECIPE / f'd52_{name}.json'} exists; D52 records are written once")
        for key in st.SETS:
            for folder in st.NETWORKS:
                if not (export_folder(name, key, folder, scratch) / "fit.json").exists():
                    raise ValueError(f"D52 network {name}/{key}/{folder} is not exported yet")
    for record in (*st.PIPELINE_CHECKS, "signs.json", "reference.json"):
        st.passed_record(record)
    config = dict(st.levers()["config"], seed_offset=OFFSET)
    st.wait_for_scoring("D52 scoring")
    games = vendor_trial.development()
    d43_dev, d43_metrica, d43_d24, truths, _ = h49.d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=st.D42_CACHE)
    d43_guard = st.d43_guards(raw)
    sign_map = read(st.PARITY)["sign_map"] if config["mirror"] is not None else None
    forecaster = st.Forecaster(candidate_models(config, scratch, sign_map), st.table_of(st.SCRATCH / "sign_table_development.json"))
    tracking = h49.guard_tracking(dfl, raw=raw)
    identities = {name: model_identity(name, config, scratch) for name in names}
    variants = {name: f"D52 {name} {st.json_digest(identities[name])[:16]}" for name in names}
    collected, checks = {name: {} for name in names}, {}

    def path_check(part, halves):
        stored = cache_root / f"{part}_d43_path_check.json"
        count = forecaster.check_d43(halves)
        manifests = {name: digest(cache_root / name / part / "manifest.json") for name in names}
        if count:
            stored.write_text(json.dumps({"halves": count, "caches": manifests}) + "\n")
        elif not stored.exists() or read(stored)["caches"] != manifests:
            raise ValueError(f"The {part} caches have no D43 path check; remove them and forecast again")
        checks[part] = read(stored)["halves"]

    for name in names:
        key = dict(identities[name], teacher_sha256=diagnosis.teacher_identity(), split="development", dfl_raw=raw,
                   skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games))
        forecast = forecaster.signals_of(name)

        def produce(forecast=forecast):
            yield from diagnosis.external_halves(dfl, skill, "development", None, raw=raw, signals_of=forecast)
            yield from diagnosis.pff_halves(parsed, games, None, signals_of=forecast)

        collected[name]["development"] = diagnosis.collect(cache_root / name / "development", key, produce)
    path_check("development", d43_dev)
    for name in names:
        _, collected[name]["metrica"] = ms.reference_halves("D15", cache=cache_root / name / "metrica", model="D52",
                                                            signals_of=forecaster.signals_of(name), variant=variants[name])
    path_check("metrica", d43_metrica)
    for name in names:
        collected[name]["d24"] = mv.view_halves(cache=cache_root / name / "d24", signals_of=forecaster.signals_of(name),
                                                variant=variants[name])
    path_check("d24", d43_d24)
    for name in names:
        forecast = forecaster.signals_of(name)
        guard_key = dict(identities[name], guard=h49.guard_identity({"model": name}, raw=raw))

        def produce_guards(forecast=forecast):
            from dfl_submission import hidden_views as hv
            for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(tracking.items()):
                signals, decoded = forecast(None, positions, ball, teams, pitch)
                yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)

        collected[name]["guards"] = diagnosis.collect(cache_root / name / "guards", guard_key, produce_guards)
    path_check("guards", d43_guard)
    print(f"D52 D43 recomputed bit for bit on {checks} halves", flush=True)

    visibility = h49.pff_visibility(parsed, games)
    reference = read(RECIPE / "reference.json")
    base_entry = h49.entry(d43_dev, d43_metrica, d43_d24, d43_guard, ht.recipe_at(st.REFERENCE_COST), truths, visibility)
    if st.canonical_json(h49.compact(base_entry)) != reference["entries"][str(st.REFERENCE_COST)]:
        raise ValueError("D43 differs from D48's reference record")
    for name in names:
        halves = collected[name]
        items = [h49.entry(halves["development"], halves["metrica"], halves["d24"], halves["guards"], ht.recipe_at(cost),
                           truths, visibility) for cost in COSTS]
        exports = {f"{key}/{folder}": read(export_folder(name, key, folder, scratch) / "fit.json")["export_max_logit_difference"]
                   for key in st.SETS for folder in st.NETWORKS}
        write_new(RECIPE / f"d52_{name}.json", {
            "protocol": PROTOCOL, "candidate": name, **CANDIDATES[name], "costs": list(COSTS),
            "entries": [dict(h49.compact(item), selection=st.selection(item)) for item in items],
            "d43": {"owner_cost": st.REFERENCE_COST, "selection": st.selection(base_entry), **h49.compact(base_entry)},
            "d43_path_bit_for_bit": checks, "identity": identities[name], "exports": exports,
            "signs_used": {"halves": len(forecaster.signs),
                           "fallbacks": sorted({row["half"] for row in forecaster.signs if row["fallback"]})},
            "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED),
                       "skill_manifest_sha256": digest(skill / "manifest.json"), "dfl_raw": raw},
            "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
        for item in items:
            print(f"D52 {name} cost {item['owner_cost']}: selection {st.selection(item):.5f}, composite {item['composite']:.5f}, "
                  f"guards {item['guards']['mean']:.4f} (a {item['guards']['a_dfl']:.4f}, b {item['guards']['b_metrica']:.4f}), "
                  f"PFF {item['groups']['pff']:.4f}, primary {item['primary']:.5f}", flush=True)
    return decide()


# -- summary ---------------------------------------------------------------------------------------

SHORT = {"dfl_clean": "clean", "dfl_masked": "masked", "dfl_dead_carried": "dead carried", "dfl_team_swap": "swap",
         "skillcorner": "SkillCorner", "pff": "PFF", "metrica": "Metrica", "d24": "D24"}


def rows(name, entries):
    for item in entries:
        yield (f"| {name} | {item['owner_cost']} | {st.selection(item):.5f} | {item['composite']:.5f} | "
               f"{item['guards']['mean']:.4f} | {item['guards']['a_dfl']:.4f} | {item['guards']['b_metrica']:.4f} | "
               f"{item['groups']['pff']:.4f} | {item['primary']:.5f} | "
               + " | ".join(f"{item['groups'][g]:.4f}" for g in GROUPS) + " |")


def summary():
    lines = ["| Model | Cost | Selection | Composite | Guard mean | Guard a (DFL) | Guard b (Metrica) | PFF | Primary | "
             + " | ".join(SHORT[g] for g in GROUPS) + " |", "|---|---:|" + "---:|" * (7 + len(GROUPS))]
    found = base_entries()
    if found is not None:
        lines += list(rows("base (D48 seed set 0)", found[0]))
    for name in CANDIDATES:
        path = RECIPE / f"d52_{name}.json"
        if path.exists():
            lines += list(rows(name, read(path)["entries"]))
    text = "\n".join(lines)
    print(text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    parser_for_train(commands.add_parser("train"))
    queued = commands.add_parser("queue")
    queued.add_argument("--external", type=Path, required=True, help="the external data root")
    scoring = commands.add_parser("score")
    scoring.add_argument("--dfl", type=Path, required=True)
    scoring.add_argument("--dfl-raw", action="store_true")
    scoring.add_argument("--skill", type=Path, required=True)
    scoring.add_argument("--parsed", type=Path, required=True)
    commands.add_parser("decide")
    commands.add_parser("summary")
    args = parser.parse_args()
    require_external_only()
    if args.command == "train":
        train_one(args)
    elif args.command == "queue":
        queue(args.external)
    elif args.command == "score":
        score(args.dfl, args.skill, args.parsed, raw=args.dfl_raw)
    elif args.command == "decide":
        decide()
    else:
        summary()


if __name__ == "__main__":
    main()
