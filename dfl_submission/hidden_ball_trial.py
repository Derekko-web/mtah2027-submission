"""D51: candidate LB, D49's candidate L with an auxiliary ball-location target (DECISIONS.md D51).

`targets` re-derives each D49 view of one model set from its seeds, checks that it is the stored view
(the view's ball equals `hidden_views.training_view`'s, and its prepared sequence and tokens equal the
stored ones byte for byte), and writes the auxiliary targets. `check-trainer` trains 300 steps with
`player_set.train` and with `hidden_ball.train` at auxiliary weight 0 on L's original-set sequences and
compares the snapshots' hashes. `networks` trains and exports one set's two LB networks, one at a time,
only while fewer than two trainings run on the machine. `aux-error` measures the head's error in metres
on development views. `score` scores LB at ownership costs 8, 10 and 12 against L and D43 and applies
D51's rule, once. `hidden_trial.py` is reused unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dfl_submission import hidden_ball as hb, hidden_trial as h49, hidden_views as hv
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "hidden_recipe"
DATA = ROOT / "data" / "d51"
PROTOCOL = "DECISIONS.md D51"
VARIANT = "D51 ball target"
COSTS = (8, 10, 12)
GAIN, COMPOSITE_LOSS, GROUP_LOSS = .005, .003, .01
CHECK_STEPS = 300
ERROR_STEP = 5400      # the snapshot the inference uses (`player_trial.PlayerCheckpoint`)


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D51 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def prepared(model_set, sequences, tokens, record_folder=RECIPE):
    """D49's preparation record of one set, after checking it passed for these folders."""
    record = json.loads((record_folder / f"prepare_{model_set}.json").read_text())
    if (not record["passed"] or record["sequences_manifest"] != digest(sequences / "manifest.json")
            or record["tokens_manifest"] != digest(tokens / "manifest.json")):
        raise ValueError(f"These folders are not D49's checked {model_set} preparation")
    return record


# -- targets ---------------------------------------------------------------------------------------

def target_summary(offset):
    present = np.isfinite(offset).all(axis=1)
    distance = np.linalg.norm(offset[present].astype(np.float64) * hb.SCALE, axis=1)
    return {"frames": int(len(offset)), "target_frames": int(present.sum()),
            "mean_distance_from_reference_m": float(distance.mean()) if len(distance) else None}


def make_targets(model_set, dfl, skill, sequences, tokens, output, *, raw, record_folder=RECIPE):
    """Re-derive one set's D49 views from their seeds, check them against the stored views, write the targets."""
    require_external_only()
    from dfl_submission import external_benchmark as benchmark, long_context, player_set
    from dfl_submission import temporal_network as tn, touch_trial
    record = prepared(model_set, sequences, tokens, record_folder)
    check_path = record_folder / f"ball_targets_{model_set}.json"
    if check_path.exists():
        raise FileExistsError(f"{check_path} exists; D51's targets are checked once")
    manifest = json.loads((sequences / "manifest.json").read_text())
    token_manifest = json.loads((tokens / "manifest.json").read_text())
    stored = {item["file"]: item for item in manifest["records"] if f"_{hv.VIEW}_" in item["file"]}
    swaps, skill_ball = record["choices"]["swaps"], record["choices"]["skill_ball"]
    paths = h49.provider_paths(skill)
    order = benchmark.SKILL_SPLITS["all"]
    output.mkdir(parents=True, exist_ok=False)
    checks, summaries, records = {}, {}, {}
    for half in touch_trial.training_halves(model_set, dfl, skill, raw=raw):
        skill_order = order.index(int(half.match)) if half.source == "skillcorner" else None
        provider = None
        if half.source == "skillcorner" and model_set == "augmented" and skill_ball == "provider":
            with np.load(paths[half.match, half.index], allow_pickle=False) as item:
                provider = item["ball"]
        derived = hb.view_with_ball(model_set, half, swaps=bool(swaps), provider_ball=provider, skill_order=skill_order)
        if derived is None:
            continue
        positions, view_ball, source_ball, _ = derived
        view, _ = hv.training_view(model_set, half, swaps=bool(swaps), provider_ball=provider, skill_order=skill_order)
        name = f"{view.key}.npz"
        if name not in stored:
            raise ValueError(f"{name} is not a stored D49 view")
        frames, table, _ = long_context.features(view.positions, view.ball, view.teams, view.pitch)
        with np.load(sequences / name, allow_pickle=False) as item, np.load(tokens / name, allow_pickle=False) as tok:
            checks[name] = {
                "positions": h49.same_bytes(np.asarray(view.positions, dtype=np.float32), np.asarray(positions, dtype=np.float32)),
                "ball": h49.same_bytes(np.asarray(view.ball, dtype=np.float32), view_ball),
                "hidden_ball_is_source_ball": bool(np.array_equal(source_ball[np.isfinite(view_ball).all(axis=1)],
                                                                  view_ball[np.isfinite(view_ball).all(axis=1)])),
                "values": h49.same_bytes(item["values"], table.to_numpy(dtype=np.float32)),
                "targets": h49.same_bytes(item["targets"], tn.targets(view.labels[frames])),
                "tokens": h49.same_bytes(tok["tokens"], player_set.tokens(view.positions, view.ball, view.teams,
                                                                          view.pitch, frames).astype(np.float16)),
                "stored_hashes": bool(digest(sequences / name) == stored[name]["sha256"]
                                      and digest(tokens / name) == token_manifest["records"][name])}
        offset = hb.targets(positions, view_ball, source_ball, frames)
        np.savez_compressed(output / name, offset=offset)
        records[name] = digest(output / name)
        summaries[name] = target_summary(offset)
        print(f"Targets {model_set} {name}: {'checked' if all(checks[name].values()) else 'DIFFERS'}, "
              f"{summaries[name]['target_frames']} target frames", flush=True)
    write_json(output / "manifest.json", {"protocol": PROTOCOL, "base_manifest": digest(sequences / "manifest.json"),
                                          "records": records, "hidden_ball_sha256": digest(PACKAGE / "hidden_ball.py")})
    passed = set(checks) == set(stored) and all(all(item.values()) for item in checks.values())
    frames = sum(item["frames"] for item in summaries.values())
    present = sum(item["target_frames"] for item in summaries.values())
    write_new(check_path, {"protocol": PROTOCOL, "set": model_set, "passed": passed, "views": len(checks),
                           "stored_views": len(stored), "checks": checks, "differing": sorted(
                               name for name, item in checks.items() if not all(item.values())),
                           "targets": summaries, "target_frame_share": present / frames if frames else None,
                           "aux_manifest": digest(output / "manifest.json"),
                           "sequences_manifest": digest(sequences / "manifest.json"),
                           "hidden_ball_sha256": digest(PACKAGE / "hidden_ball.py"),
                           "written_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D51 targets {model_set}: {len(checks)} views re-derived and checked: {passed}; "
          f"{present} target frames of {frames}", flush=True)
    if not passed:
        raise ValueError("D51 pipeline check failed: the re-derived views differ from the stored ones")


# -- the trainer check -----------------------------------------------------------------------------

def other_trainings(own=None):
    """`hidden_trial.training_processes` without this process, which carries `--seed` so others count it."""
    own = os.getpid() if own is None else own
    listing = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True).stdout
    return h49.training_processes("\n".join(line for line in listing.splitlines()
                                             if line.split(None, 1)[0] != str(own)))


def check_trainer(sequences, tokens, aux, scratch, *, seed=42, boost=.5, steps=CHECK_STEPS, device="mps",
                  record_folder=RECIPE):
    """D42's trainer and D51's at auxiliary weight 0, `steps` steps each on L's sequences: equal snapshots?"""
    require_external_only()
    from dfl_submission import player_set
    output = record_folder / "ball_trainer_check.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D51's trainer check is recorded once")
    h49.wait_for_slot(count=other_trainings)
    player_set.train(sequences, tokens, scratch / "d42", device, seed=seed, boundary_boost=boost, steps=steps)
    hb.train(sequences, tokens, aux, scratch / "lb0", device, seed=seed, boundary_boost=boost, aux_weight=0., steps=steps)
    d42 = json.loads((scratch / "d42" / "fit.json").read_text())
    lb0 = json.loads((scratch / "lb0" / "fit.json").read_text())
    passed = d42["models"] == lb0["models"] and d42["normalization"] == lb0["normalization"]
    write_new(output, {"protocol": PROTOCOL, "passed": bool(passed), "steps": steps, "device": device, "seed": seed,
                       "boundary_boost": boost, "d42_snapshots": d42["models"], "lb_weight0_snapshots": lb0["models"],
                       "sequences_manifest": digest(sequences / "manifest.json"),
                       "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D51 trainer check ({steps} steps, {device}): weight-0 snapshots equal D42's: {passed}", flush=True)
    if not passed:
        raise ValueError("D51 pipeline check failed: at weight 0 the trainer does not reproduce D42's")


# -- networks --------------------------------------------------------------------------------------

def train_networks(model_set, sequences, tokens, aux, scratch, models, *, record_folder=RECIPE):
    require_external_only()
    from dfl_submission import hyper_trial as ht
    prepared(model_set, sequences, tokens, record_folder)
    targets_check = json.loads((record_folder / f"ball_targets_{model_set}.json").read_text())
    trainer = json.loads((record_folder / "ball_trainer_check.json").read_text())
    if not (targets_check["passed"] and targets_check["aux_manifest"] == digest(aux / "manifest.json") and trainer["passed"]):
        raise ValueError("The D51 pipeline checks did not pass for these targets")
    for folder, (seed, boost) in ht.NETWORKS.items():
        output = models / model_set / folder
        if (output / "fit.json").exists():
            continue
        raw = scratch / model_set / folder
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            h49.wait_for_slot()
            print(f"{time.strftime('%H:%M:%S')} training {model_set} {folder}", flush=True)
            subprocess.run([sys.executable, "-m", "dfl_submission.hidden_ball", "train", "--source", str(sequences),
                            "--tokens", str(tokens), "--aux", str(aux), "--out", str(raw), "--seed", str(seed),
                            "--boundary-boost", str(boost)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
        print(f"{time.strftime('%H:%M:%S')} exported {model_set} {folder}", flush=True)


# -- the auxiliary error on development views ------------------------------------------------------

def error_views(dfl, *, raw):
    """{name: [(positions, view ball, true ball, teams, pitch)]}: DFL WPY masked (truth: clean), the DFL
    long-hidden guard view (truth: clean) and the D24 masked Metrica views (truth: the sample tracking)."""
    from dfl_submission import external_benchmark as benchmark, metrica_sample as ms, metrica_views as mv
    views = {"dfl_masked": [], "dfl_guard": [], "d24_masked": []}
    clean = {}
    for half in benchmark.halves(dfl, None, "development", raw=raw):
        if half.view == "clean":
            clean[half.index] = half
            positions, ball = hv.guard_view(half.positions, half.ball, hv.guard_dfl_seed(half.match, half.index))
            views["dfl_guard"].append((positions, ball, half.ball, half.teams, half.pitch))
        elif half.view == "masked":
            views["dfl_masked"].append((half.positions, half.ball, clean[half.index].ball, half.teams, half.pitch))
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            positions, ball = mv.build_views(half, game, index)["masked"]
            views["d24_masked"].append((positions, ball, half["ball"], half["teams"], half["pitch"]))
    return views


def aux_error(dfl, scratch, *, raw, record_folder=RECIPE):
    """Mean distance (m) between the head's ball and the true ball on hidden frames of development views."""
    require_external_only()
    from dfl_submission import hyper_trial as ht, long_context, player_set
    output = record_folder / "ball_aux_error.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D51's auxiliary error is measured once")
    prepared_views = {}
    for name, items in error_views(dfl, raw=raw).items():
        prepared_views[name] = []
        for positions, ball, truth, teams, pitch in items:
            frames, table, _ = long_context.features(positions, ball, teams, pitch)
            prepared_views[name].append((positions, ball, truth, frames, table.to_numpy(dtype=np.float32),
                                         player_set.tokens(positions, ball, teams, pitch, frames)))
    result = {}
    for model_set in ht.SETS:
        for folder in ht.NETWORKS:
            raw_folder = scratch / model_set / folder
            fit = json.loads((raw_folder / "fit.json").read_text())
            entry = {"training_aux_loss_last_100": fit["ball"]["final_aux_loss_mean_last_100"]}
            for name, items in prepared_views.items():
                errors, baselines = [], []
                for positions, ball, truth, frames, values, player_tokens in items:
                    offsets = hb.predicted_offsets(raw_folder, ERROR_STEP, values, player_tokens)
                    error, baseline = hb.distances(positions, ball, truth, frames, offsets)
                    errors.append(error)
                    baselines.append(baseline)
                error, baseline = np.concatenate(errors), np.concatenate(baselines)
                entry[name] = {"frames": int(len(error)), "mean_error_m": float(error.mean()),
                               "median_error_m": float(np.median(error)),
                               "mean_distance_from_reference_m": float(baseline.mean())}
            result[f"{model_set}/{folder}"] = entry
            print(f"D51 auxiliary error {model_set} {folder}: " + ", ".join(
                f"{name} {entry[name]['mean_error_m']:.2f} m (reference {entry[name]['mean_distance_from_reference_m']:.2f} m)"
                for name in prepared_views) + f"; training auxiliary loss {entry['training_aux_loss_last_100']:.4f}", flush=True)
    write_new(output, {"protocol": PROTOCOL, "step": ERROR_STEP, "networks": result,
                       "views": {name: len(items) for name, items in prepared_views.items()},
                       "measured_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    return result


# -- scoring ---------------------------------------------------------------------------------------

def best(entries):
    return max(entries, key=lambda item: (item["guards"]["mean"] + item["groups"]["pff"]) / 2)


def replaces(l_entries, d43, entries):
    """D51's rule: LB's best mean of (guard mean, PFF) beats L's best by 0.005, composite within 0.003 of L's,
    and no group more than 0.01 below D43."""
    chosen, l_best = best(entries), best(l_entries)
    mean = lambda item: (item["guards"]["mean"] + item["groups"]["pff"]) / 2
    groups = {name: chosen["groups"][name] - d43["groups"][name] for name in h49.GROUPS}
    conditions = {"guard_pff_mean": bool(mean(chosen) - mean(l_best) >= GAIN),
                  "composite": bool(chosen["composite"] - l_best["composite"] >= -COMPOSITE_LOSS),
                  "groups": bool(all(value >= -GROUP_LOSS for value in groups.values()))}
    return {"selected_cost": chosen["owner_cost"], "l_cost": l_best["owner_cost"],
            "changes": {"guard_pff_mean_vs_l": mean(chosen) - mean(l_best),
                        "composite_vs_l": chosen["composite"] - l_best["composite"], "groups_vs_d43": groups,
                        "guard_mean_vs_l": chosen["guards"]["mean"] - l_best["guards"]["mean"],
                        "pff_vs_l": chosen["groups"]["pff"] - l_best["groups"]["pff"]},
            "conditions": conditions, "replaces_l": bool(all(conditions.values()))}


def score(dfl, skill, parsed, *, raw, d42_cache, models, cache_root=DATA / "cache", record_folder=RECIPE):
    """LB at ownership costs 8, 10 and 12 against L and D43, with D51's rule, written once."""
    require_external_only()
    from dfl_submission import hyper_trial as ht, metrica_sample as ms, metrica_views as mv, player_trial, vendor_trial
    output = record_folder / "ball_candidate.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D51's candidate is scored once")
    reference = json.loads((record_folder / "reference.json").read_text())["reference"]
    l_record = json.loads((record_folder / "candidate.json").read_text())
    h49.wait_for_slot()
    games = vendor_trial.development()
    _, _, _, truths, checks = h49.d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=d42_cache)
    models_identity = player_trial.identity(models)
    model = player_trial.teacher(models=models)
    variant = {"name": VARIANT, "models": models_identity}
    development = player_trial.candidate_halves(dfl, skill, parsed, games, "development", raw=raw,
                                                cache=cache_root / "development", models=models)
    _, metrica = ms.reference_halves("D15", cache=cache_root / "metrica", model=model,
                                     signals_of=player_trial.half_signals, variant=variant)
    d24 = mv.view_halves(cache=cache_root / "d24", signals_of=lambda _, *tracking: player_trial.half_signals(model, *tracking),
                         variant=variant)
    tracking = h49.guard_tracking(dfl, raw=raw)
    guards = h49.guard_halves(tracking, model, cache_root / "guards_lb", h49.guard_identity(models_identity, raw=raw))
    visibility = h49.pff_visibility(parsed, games)
    entries = []
    for cost in COSTS:
        item = h49.entry(development, metrica, d24, guards, ht.recipe_at(cost), truths, visibility)
        entries.append(item)
        print(h49.summary_line("D51 candidate LB", item), flush=True)
    decision = replaces(l_record["entries"], reference, entries)
    write_new(output, {"protocol": PROTOCOL, "candidate": "LB", "costs": list(COSTS), "d43_reproduction": checks,
                       "reference_record": digest(record_folder / "reference.json"),
                       "l_record": digest(record_folder / "candidate.json"),
                       "entries": [h49.compact(item) for item in entries], "decision": decision, "models": models_identity,
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D51 decision: LB cost {decision['selected_cost']} vs L cost {decision['l_cost']}: guard/PFF mean "
          f"{decision['changes']['guard_pff_mean_vs_l']:+.4f}, composite {decision['changes']['composite_vs_l']:+.4f}, "
          f"worst group vs D43 {min(decision['changes']['groups_vs_d43'].values()):+.4f} -> "
          f"{'LB replaces L' if decision['replaces_l'] else 'L stays'}", flush=True)
    return decision


def main():
    from dfl_submission import hyper_trial as ht
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    making = commands.add_parser("targets")
    making.add_argument("--set", choices=ht.SETS, required=True)
    making.add_argument("--dfl", type=Path, required=True)
    making.add_argument("--dfl-raw", action="store_true")
    making.add_argument("--skill", type=Path, required=True)
    making.add_argument("--sequences", type=Path, required=True)
    making.add_argument("--tokens", type=Path, required=True)
    making.add_argument("--out", type=Path, required=True)
    checking = commands.add_parser("check-trainer")
    checking.add_argument("--sequences", type=Path, required=True)
    checking.add_argument("--tokens", type=Path, required=True)
    checking.add_argument("--aux", type=Path, required=True)
    checking.add_argument("--scratch", type=Path, required=True)
    checking.add_argument("--seed", type=int, default=42, help="also marks this process as a training for the slot count")
    networks = commands.add_parser("networks")
    networks.add_argument("--set", choices=ht.SETS, required=True)
    networks.add_argument("--sequences", type=Path, required=True)
    networks.add_argument("--tokens", type=Path, required=True)
    networks.add_argument("--aux", type=Path, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    networks.add_argument("--models", type=Path, required=True)
    erring = commands.add_parser("aux-error")
    erring.add_argument("--dfl", type=Path, required=True)
    erring.add_argument("--dfl-raw", action="store_true")
    erring.add_argument("--scratch", type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--dfl", type=Path, required=True)
    scoring.add_argument("--dfl-raw", action="store_true")
    scoring.add_argument("--skill", type=Path, required=True)
    scoring.add_argument("--parsed", type=Path, required=True)
    scoring.add_argument("--d42-cache", type=Path, required=True)
    scoring.add_argument("--models", type=Path, required=True)
    scoring.add_argument("--cache", type=Path, default=DATA / "cache")
    args = parser.parse_args()
    require_external_only()
    if args.command == "targets":
        make_targets(args.set, args.dfl, args.skill, args.sequences, args.tokens, args.out, raw=args.dfl_raw)
    elif args.command == "check-trainer":
        check_trainer(args.sequences, args.tokens, args.aux, args.scratch, seed=args.seed)
    elif args.command == "networks":
        train_networks(args.set, args.sequences, args.tokens, args.aux, args.scratch, args.models)
    elif args.command == "aux-error":
        aux_error(args.dfl, args.scratch, raw=args.dfl_raw)
    else:
        score(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d42_cache=args.d42_cache, models=args.models,
              cache_root=args.cache)


if __name__ == "__main__":
    main()
