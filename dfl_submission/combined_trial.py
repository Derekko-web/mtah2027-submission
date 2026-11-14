"""D60: the air rule on every training match (D57's views) with refitted trees left out where the ball is unseen (DECISIONS.md D60).

Candidate P: the adopted stack's recipe (attacking-frame tokens, the y mirror, 8,100 steps, three seeds per slot) on
D57's prepared sequences and tokens; D57's four SZ networks are its first seed set (offset 0) if the stack's trainer
reproduces the original-set seed-42 one bit for bit; eight networks are trained for offsets 1,000 and 2,000 with the
stack's trainer (`stack_trial train-steps`, as D48's 8,100-step offsets were). Both sets' trees are refitted on D57's
sequences exactly as D3 fitted them (`touch_trial.fit_trees`, the call D59's byte-for-byte check covers). Scored with
D59's member scorer: H(0) on the refitted trees (`tree_share`), the standard mix with D3's trees for information.

Commands (each with `MTAH_DATA_DIR` set to a nonexistent path):
- `queue`: the trainer check (offset 0, original set, seed 42, into `data/d60/networks/check/`) and the eight networks,
  fewer than two trainings machine-wide at each start; each exported after training. `stack_recipe/d60_check.json`
  is written once when the check's network exists.
- `refit`: both sets' trees on D57's sequences, after confirming D59's refit check stands for this code path;
  `stack_recipe/d60_refit.json` written once.
- `members --dfl DIR --skill DIR --parsed DIR`: P's members (D3's and the refitted trees, the slots) on every scored
  half, once, under `data/d60/members/`.
- `develop --dfl DIR --skill DIR --parsed DIR`: the adopted stack (D59's members, checked against `development_t.json`)
  and P at costs 12, 14 and 16; D60's development gate; `stack_recipe/d60_development.json` written once.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from dfl_submission import height_fit as hf, situation_fit as sf, stack_trial as st
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = st.RECIPE
SCRATCH = ROOT / "data" / "d60"
NETWORKS = SCRATCH / "networks"
TREES = SCRATCH / "trees"
MEMBERS = SCRATCH / "members"
PROTOCOL = "DECISIONS.md D60"
OFFSETS = st.OFFSETS                   # (0, 1000, 2000)
TRAINED = (1000, 2000)
STEPS = st.T_STEPS                     # 8,100
COSTS = (12, 14, 16)
REFERENCE_COST = 14
PFF_GAIN, COMPOSITE_LOSS, GUARD_LOSS, LOSS = .012, .005, .005, .01
CHECK = ("check", "current", "temporal_models")


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(record, indent=2, sort_keys=True) + "\n")


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True))


def config(offset):
    """The adopted stack's switches (its confirmed lever record) at a seed offset."""
    chosen = dict(read(RECIPE / "confirmation.json")["levers"]["config"], seed_offset=offset)
    if dict(chosen, seed_offset=0) != sf.config():
        raise ValueError("The confirmed switches differ from the stack's lever record")
    return chosen


# -- networks ------------------------------------------------------------------------------------------

def raw_folder(offset, model_set, folder, scratch=NETWORKS):
    return scratch / "raw" / str(offset) / model_set / folder


def export_folder(offset, model_set, folder, scratch=NETWORKS):
    """Offset 0: D57's SZ networks; offsets 1,000 and 2,000: trained here."""
    if offset == 0:
        return hf.export_folder(model_set, folder)
    return scratch / "export" / str(offset) / model_set / folder


def check_folder(scratch=NETWORKS):
    return scratch / CHECK[0] / CHECK[1] / CHECK[2]


def train_command(offset, model_set, folder, out):
    """`stack_trial train-steps` (the trainer of the stack's 8,100-step offsets) on D57's folders and signs."""
    hf.prepared(model_set)
    seed, boost = st.NETWORKS[folder]
    sequences, tokens, signs = hf.folders(model_set)
    chosen = config(offset)
    command = [sys.executable, "-m", "dfl_submission.stack_trial", "train-steps", "--source", str(sequences), "--tokens",
               str(tokens), "--out", str(out), "--seed", str(seed), "--boundary-boost", str(boost), "--config",
               json.dumps(chosen), "--signs", str(signs), "--steps", str(STEPS)]
    if chosen["mirror"] is not None:
        command += ["--sign-map", str(st.PARITY)]
    if chosen["noise"] or chosen["aux"]:
        raise ValueError("P trains with the stack's switches: no token noise, no auxiliary head")
    return command


def export_command(source, out):
    return [sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(source), "--out", str(out)]


def jobs(scratch=NETWORKS):
    """(name, raw folder, export folder or None, command) of every training not done yet: the check first."""
    result = []
    check = check_folder(scratch)
    if not (check / "fit.json").exists():
        result.append(("check", check, None, train_command(0, CHECK[1], CHECK[2], check)))
    for offset in TRAINED:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                out = export_folder(offset, model_set, folder, scratch)
                if not (out / "fit.json").exists():
                    raw = raw_folder(offset, model_set, folder, scratch)
                    result.append((f"{offset}/{model_set}/{folder}", raw, out, train_command(offset, model_set, folder, raw)))
    return result


def shown(path):
    path = Path(path)
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check_record(scratch=NETWORKS, output=None):
    """The trainer check: the retrained offset-0 original-set seed-42 network equals D57's SZ one bit for bit."""
    output = RECIPE / "d60_check.json" if output is None else output
    if Path(output).exists():
        return read(output)
    trained = read(check_folder(scratch) / "fit.json")
    expected = read(hf.raw_folder(CHECK[1], CHECK[2]) / "fit.json")
    differing = [key for key in (*st.FIT_KEYS, "seed") if trained[key] != expected[key]]
    passed = bool(not differing and trained["models"] == expected["models"])
    record = {"protocol": PROTOCOL, "check": "the stack's trainer, offset 0, D57's original set, seed 42, 8,100 steps",
              "reference": shown(hf.raw_folder(CHECK[1], CHECK[2])), "expected": expected["models"],
              "actual": trained["models"], "differing_fit_keys": differing, "passed": passed,
              "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, record)
    print(f"D60 trainer check: {'bit for bit' if passed else 'DIFFERS'}", flush=True)
    return record


def queue(*, scratch=NETWORKS, poll=30, limit=st.TRAINING_LIMIT, running_trainings=st.running_trainings,
          sleep=time.sleep):
    """The check and the eight networks, fewer than `limit` trainings machine-wide at each start; one queue at a time."""
    require_external_only()
    for model_set in st.SETS:
        hf.prepared(model_set)
    scratch.mkdir(parents=True, exist_ok=True)
    lock = open(scratch / "queue.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending, running, said = jobs(scratch), {}, None
    while pending or running:
        for name, (process, raw, out) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            del running[name]
            if code != 0:
                raise RuntimeError(f"Training {name} failed with exit code {code}")
            if out is not None:
                subprocess.run(export_command(raw, out), check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
            else:
                check_record(scratch)
            print(f"{time.strftime('%H:%M:%S')} {name} trained{' and exported' if out is not None else ''}", flush=True)
        started = False
        if pending:
            name, raw, out, command = pending[0]
            if (raw / "fit.json").exists():
                pending.pop(0)
                if out is not None:
                    subprocess.run(export_command(raw, out), check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
                else:
                    check_record(scratch)
                print(f"{time.strftime('%H:%M:%S')} {name} exported (trained earlier)", flush=True)
                continue
            machine = running_trainings()
            if machine < limit:
                if raw.exists():
                    raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
                raw.parent.mkdir(parents=True, exist_ok=True)
                log = open(scratch.parent / f"train_{name.replace('/', '_')}.log", "w")
                running[name] = (subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), raw, out)
                pending.pop(0)
                started, said = True, None
                print(f"{time.strftime('%H:%M:%S')} started {name} (machine-wide before: {machine})", flush=True)
            elif said != machine:
                print(f"{time.strftime('%H:%M:%S')} waiting: {machine} trainings machine-wide", flush=True)
                said = machine
        sleep(15 if started else poll)
    print(f"{time.strftime('%H:%M:%S')} D60 networks all trained", flush=True)


def network_identity(scratch=NETWORKS):
    """{offset: {set: {folder: fit digest}}} of P's twelve networks, offset 0 checked against D57's SZ record."""
    sz = read(RECIPE / "d57_SZ.json")["identity"]["networks"]
    result = {}
    for offset in OFFSETS:
        for model_set in st.SETS:
            for folder in st.NETWORKS:
                path = export_folder(offset, model_set, folder, scratch) / "fit.json"
                if not path.exists():
                    raise ValueError(f"P's network {offset}/{model_set}/{folder} is not exported yet")
                value = digest(path)
                if offset == 0 and value != sz[model_set][folder]:
                    raise ValueError(f"SZ's network {model_set}/{folder} differs from D57's record")
                fit = read(path)
                if offset and (fit["stack"]["seed_offset"] != offset or fit["steps"] != STEPS
                               or fit["train_manifest"] != digest(hf.folders(model_set)[0] / "manifest.json")):
                    raise ValueError(f"P's network {offset}/{model_set}/{folder} is not the recipe on D57's folders")
                result.setdefault(str(offset), {}).setdefault(model_set, {})[folder] = value
    return result


# -- trees -----------------------------------------------------------------------------------------------

def refit(*, scratch=TREES, output=None):
    """Both sets' trees on D57's sequences, as D3 fitted them, after D59's check is confirmed for this code path."""
    require_external_only()
    from dfl_submission import touch_trial
    output = RECIPE / "d60_refit.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D60's refit is recorded once")
    d59 = read(RECIPE / "d59_refit.json")
    path_same = {name: digest(PACKAGE / name) == d59["code"][name]
                 for name in ("touch_trial.py", "long_context.py", "aligned_trial.py")}
    if d59["passed"] is not True or not all(path_same.values()):
        raise SystemExit(f"D60 stops: D59's refit check does not cover this code path ({path_same}); re-run it first")
    refits, sources = {}, {}
    for model_set in st.SETS:
        hf.prepared(model_set)
        source = hf.folders(model_set)[0]
        refits[model_set] = touch_trial.fit_trees(source, model_set, scratch / model_set)
        sources[model_set] = {"manifest": digest(source / "manifest.json"),
                              "records": len(read(source / "manifest.json")["records"]),
                              "rows": read(scratch / model_set / "fit.json")["rows"],
                              "fit_sha256": digest(scratch / model_set / "fit.json")}
        print(f"D60 trees, {model_set}: refitted on D57's sequences: {refits[model_set]} ({sources[model_set]['rows']})",
              flush=True)
    write_new(output, {"protocol": PROTOCOL, "passed": True, "d59_check": d59["check_reproduces_d3_trees"],
                       "d59_refit_sha256": digest(RECIPE / "d59_refit.json"), "code_path_same_as_d59": path_same,
                       "refits": refits, "sources": sources, "refitted_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    return refits


def tree_folders(record_path=None, scratch=TREES):
    record = read(RECIPE / "d60_refit.json" if record_path is None else record_path)
    folders = {key: scratch / key for key in st.SETS}
    for key, folder in folders.items():
        if digest(folder / "fit.json") != record["sources"][key]["fit_sha256"] or \
                read(folder / "fit.json")["models"] != record["refits"][key]:
            raise ValueError(f"P's {key} trees differ from the refit record")
    return folders


# -- members -----------------------------------------------------------------------------------------------

def p_models(scratch=NETWORKS):
    chosen = config(0)
    sign_map = read(st.PARITY)["sign_map"] if chosen["mirror"] is not None else None
    export = lambda offset, key, folder, root: export_folder(offset, key, folder, root)
    return st.stack_models(chosen, scratch, export=export, sign_map=sign_map, names=("stack",))


def p_members(dfl, skill, parsed, *, cache=MEMBERS, scratch=NETWORKS):
    """P's members on every scored half (development with PFF, Metrica, D24, the guard views, D55's views), cached once."""
    from dfl_submission import diagnosis, hidden_trial as h49, tree_share as ts, vendor_trial
    networks = network_identity(scratch)
    trees = tree_folders()
    games = vendor_trial.development()
    situation = {key: value for key, value in sf.situation_tracking(dfl).items() if key[0] in sf.VIEWS.values()}
    table = st.table_of(st.SCRATCH / "sign_table_development.json") | sf.situation_table(situation)
    forecaster = ts.MemberForecaster(p_models(scratch), table, refit=trees)
    forecast = forecaster.member_fn()
    identity = {"protocol": PROTOCOL, "networks": networks, "config": config(0),
                "trees_d3": read(RECIPE / "confirmation.json")["identity"]["trees"],
                "trees_refit": {key: digest(folder / "fit.json") for key, folder in trees.items()},
                "refit_record_sha256": digest(RECIPE / "d60_refit.json"), "teacher_sha256": diagnosis.teacher_identity(),
                "sign_table_sha256": digest(st.SCRATCH / "sign_table_development.json"),
                "inputs": ts.inputs_identity(skill, parsed, games), "guard": h49.guard_identity({"model": "P"}, raw=True),
                "situation": sf.situation_identity(), "tree_share_sha256": digest(PACKAGE / "tree_share.py"),
                "code": {name: digest(PACKAGE / name) for name in ("stack_set.py", "player_set.py", "direction_tokens.py",
                                                                    "player_set3.py", "temporal_cpu.py")}}
    guards = h49.guard_tracking(dfl, raw=True)
    producers = {"development": lambda: ts.development_halves(dfl, skill, parsed, games, forecast),
                 "metrica": lambda: ts.metrica_halves(forecast), "d24": lambda: ts.d24_halves(forecast),
                 "guards": lambda: ts.guard_halves(guards, forecast),
                 "situation": lambda: ts.situation_halves(situation, forecast)}
    parts = {part: ts.collect(cache / part, dict(identity, part=part), producers[part]) for part in ts.PARTS}
    return parts, forecaster


def members(dfl, skill, parsed):
    require_external_only()
    st.wait_for_scoring("D60 member pass")
    parts, forecaster = p_members(dfl, skill, parsed)
    print(f"D60 members: {sum(len(h) for h in parts.values())} halves; signs used {len(forecaster.signs)}, fallbacks "
          f"{sorted({row['half'] for row in forecaster.signs if row['fallback']})}", flush=True)
    return parts


# -- development ---------------------------------------------------------------------------------------------

def gate(items, fulls, reference, reference_full):
    """D60's gate: the cost with the highest PFF mean (ties to the lower), then the five conditions."""
    from dfl_submission import refine_trial
    index = max(range(len(items)), key=lambda i: (items[i]["groups"]["pff"], -items[i]["owner_cost"]))
    chosen, full = items[index], fulls[index]
    groups = {name: chosen["groups"][name] - reference["groups"][name] for name in st.GROUPS}
    named = {name: refine_trial.standard(full["trial"], name) - refine_trial.standard(reference_full["trial"], name)
             for name in refine_trial.OTHER_CONVENTIONS}
    changes = {"pff": groups["pff"], "composite": chosen["composite"] - reference["composite"],
               "guard_mean": chosen["guards"]["mean"] - reference["guards"]["mean"], "groups": groups, "conventions": named,
               "selection": chosen["selection"] - reference["selection"], "primary": chosen["primary"] - reference["primary"]}
    conditions = {"pff": bool(changes["pff"] >= PFF_GAIN), "composite": bool(changes["composite"] >= -COMPOSITE_LOSS),
                  "guard_mean": bool(changes["guard_mean"] >= -GUARD_LOSS),
                  "groups": bool(all(value >= -LOSS for value in groups.values())),
                  "conventions": bool(all(value >= -LOSS for value in named.values()))}
    return {"selected_cost": chosen["owner_cost"], "changes": changes, "conditions": conditions,
            "passed": bool(all(conditions.values()))}


def develop(dfl, skill, parsed, *, output=None):
    """The adopted stack at cost 14 (D59's members, checked against `development_t.json`) and P; D60's gate; once."""
    require_external_only()
    from dfl_submission import competition_adaptation, refine_trial, tree_share as ts
    output = RECIPE / "d60_development.json" if output is None else output
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D60's development is recorded once")
    check = read(RECIPE / "d60_check.json")
    if check["passed"] is not True:
        raise SystemExit("D60 stops: the trainer check did not pass, so SZ's networks are not P's first seed set")
    st.wait_for_scoring("D60 development scoring")
    reference_model, _, frozen = competition_adaptation.reference()
    previous, recipe = reference_model[2]["recipe"], frozen["recipe"]
    games, truths, visibility = sf.development_inputs(dfl, skill, parsed)
    tracking = {key: value for key, value in sf.situation_tracking(dfl).items() if key[0] in sf.VIEWS.values()}
    stack_parts, _ = ts._member_pass(dfl, skill, parsed, cache=ts.MEMBERS)
    bits = ts.check_against_stack(stack_parts, previous, recipe)
    pair = ts.entries_for(ts.scored(stack_parts, "d3", None, previous, recipe, recipe), (REFERENCE_COST,), truths,
                          visibility, tracking)[0]
    reference_full, reference = pair
    stored = next(item for item in read(RECIPE / "development_t.json")["entries"]["stack"] if item["owner_cost"] == REFERENCE_COST)
    mine = {key: value for key, value in reference.items() if key not in ("situation_d", "situation_a")}
    check1 = {"cost_14_row_equal": canonical(mine) == canonical(stored),
              "differing_keys": sorted(key for key in stored if canonical(stored[key]) != canonical(mine.get(key))),
              "members_bit_for_bit": bits}
    p_parts, forecaster = p_members(dfl, skill, parsed)
    variants = {"P_H0": ("r", 0.0), "P_standard_d3": ("d3", None)}
    items, fulls = {}, {}
    for name, (tree_set, share) in variants.items():
        pairs = ts.entries_for(ts.scored(p_parts, tree_set, share, previous, recipe, recipe), COSTS, truths, visibility,
                               tracking)
        fulls[name], items[name] = [full for full, _ in pairs], [item for _, item in pairs]
        for item in items[name]:
            print(f"D60 {name} cost {item['owner_cost']}: PFF {item['groups']['pff']:.4f}, composite {item['composite']:.5f}, "
                  f"guards {item['guards']['mean']:.4f}, selection {item['selection']:.5f}", flush=True)
    passed_checks = bool(check["passed"] and check1["cost_14_row_equal"] and all(p["bit_for_bit"] for p in bits.values()))
    decision = gate(items["P_H0"], fulls["P_H0"], reference, reference_full)
    if not passed_checks:
        decision = dict(decision, passed=False, note="pipeline checks failed")
    information = {str(item["owner_cost"]): gate([item], [full], reference, reference_full)
                   for item, full in zip(items["P_standard_d3"], fulls["P_standard_d3"])}
    conventions = {name: [{key: refine_trial.standard(full["trial"], key)
                           for key in (refine_trial.PRIMARY, *refine_trial.OTHER_CONVENTIONS)} for full in fulls[name]]
                   for name in variants}
    conventions["reference"] = {key: refine_trial.standard(reference_full["trial"], key)
                                for key in (refine_trial.PRIMARY, *refine_trial.OTHER_CONVENTIONS)}
    write_new(output, canonical({
        "protocol": PROTOCOL, "passed_checks": passed_checks, "trainer_check": check, "check_reference": check1,
        "reference": {"name": "the adopted stack", "owner_cost": REFERENCE_COST, **reference},
        "entries": items, "conventions": conventions, "gate": decision,
        "information_only": {"P_standard_d3": "outside the rule", "against_reference": information},
        "rule": {"cost": "the highest PFF development mean, ties to the lower", "pff_gain": PFF_GAIN,
                 "composite_loss": COMPOSITE_LOSS, "guard_loss": GUARD_LOSS, "group_loss": LOSS, "conventions_loss": LOSS},
        "networks": network_identity(), "refit_sha256": digest(RECIPE / "d60_refit.json"),
        "member_caches": {part: digest(MEMBERS / part / "manifest.json") for part in ts.PARTS},
        "signs_used": {"halves": len(forecaster.signs),
                       "fallbacks": sorted({row["half"] for row in forecaster.signs if row["fallback"]})},
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")}))
    print(ts.summary({"reference": [reference], **items}), flush=True)
    changes = decision["changes"]
    print(f"D60 gate: cost {decision['selected_cost']}, PFF {changes['pff']:+.4f} (+{PFF_GAIN}), composite "
          f"{changes['composite']:+.4f} (>= -{COMPOSITE_LOSS}), guard mean {changes['guard_mean']:+.4f} (>= -{GUARD_LOSS}), "
          f"worst group {min(changes['groups'].values()):+.4f}, worst convention {min(changes['conventions'].values()):+.4f} "
          f"(>= -{LOSS}) -> {'passes: set D would be spent' if decision['passed'] else 'fails'}", flush=True)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("queue", "refit"):
        sub.add_parser(action)
    for action in ("members", "develop"):
        part = sub.add_parser(action)
        for flag in ("--dfl", "--skill", "--parsed"):
            part.add_argument(flag, type=Path, required=True)
    args = parser.parse_args()
    require_external_only()
    if args.action == "queue":
        queue()
    elif args.action == "refit":
        refit()
    elif args.action == "members":
        members(args.dfl, args.skill, args.parsed)
    else:
        develop(args.dfl, args.skill, args.parsed)


if __name__ == "__main__":
    main()
