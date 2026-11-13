"""D49: long hidden-ball runs in training (DECISIONS.md D49).

`prepare` re-extracts every stored training half of one model set and checks that its prepared
sequence and player tokens are reproduced byte for byte. It then copies the stored files unchanged
and adds the set's D49 views (`hidden_views.training_view`), one per DFL training half and one per
SkillCorner training half. `networks` trains and exports the set's two player-set networks exactly
as D42 did (`player_set train`, then `player_set export`). It trains one network at a time, and
starts one only while fewer than two training processes run on the machine.

`reference` scores frozen D43 on the two long-hidden guard views (and on D47's eight development
groups, after D43 reproduces its D42 development row exactly), once, before any candidate is scored.
`candidate` scores candidate L at ownership costs 8, 10 and 12 and applies D49's carry rule, once.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from dfl_submission import diagnosis, hidden_views as hv
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest
from evaluation import possession_metrics as pm


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "hidden_recipe"
DATA = ROOT / "data" / "d49"
PROTOCOL = "DECISIONS.md D49"
VARIANT = "D49 long hidden"
COSTS = (8, 10, 12)
D43_COST = 10
TARGET = "aba3"
GUARD_GAIN, PFF_GAIN = .015, .005
COMPOSITE_LOSS, GROUP_LOSS = .003, .01
GROUPS = ("dfl_clean", "dfl_masked", "dfl_dead_carried", "dfl_team_swap", "skillcorner", "pff", "metrica", "d24")
MAX_TRAININGS = 2
TRAINING = re.compile(r"-m dfl_submission\.\w+\b.*\s--seed\s")
SWAP_CHOICES = {"yes": True, "no": False}
SKILL_BALLS = ("provider", "handled")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D49 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


# -- the shared machine ----------------------------------------------------------------------------

def training_processes(listing=None):
    """How many network trainings run on the machine: `-m dfl_submission.<module>` processes given a `--seed`."""
    if listing is None:
        listing = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True, check=True).stdout
    return sum(bool(TRAINING.search(line)) for line in listing.splitlines())


def wait_for_slot(limit=MAX_TRAININGS, pause=60, count=training_processes):
    """Block until fewer than `limit` trainings run (to start one, or to score)."""
    while (running := count()) >= limit:
        print(f"{time.strftime('%H:%M:%S')} {running} trainings running; waiting", flush=True)
        time.sleep(pause)


# -- preparation -----------------------------------------------------------------------------------

def same_bytes(stored, fresh):
    stored, fresh = np.asarray(stored), np.asarray(fresh)
    return stored.dtype == fresh.dtype and stored.shape == fresh.shape and stored.tobytes() == fresh.tobytes()


def merged_manifest(stored, new_records, note):
    """The stored manifest with its records first and unchanged, then the D49 records, and a D49 note."""
    if {record["file"] for record in stored["records"]} & {record["file"] for record in new_records}:
        raise ValueError("A D49 record would replace a stored record")
    return {key: value for key, value in stored.items() if key != "records"} | {
        "records": [dict(record) for record in stored["records"]] + list(new_records), "d49": note}


def provider_paths(skill):
    from dfl_submission import external_benchmark as benchmark
    return {(str(match), index): path for match, index, path in benchmark.skillcorner_paths(skill, "train")}


def prepare(model_set, dfl, skill, source, stored_tokens, sequences, tokens, *, raw, swaps=None, skill_ball=None,
            record_folder=RECIPE):
    """One set's stored training records, reproduced byte for byte and copied, plus its D49 views."""
    require_external_only()
    from dfl_submission import aligned_trial, external_benchmark as benchmark, long_context, player_set
    from dfl_submission import temporal_network as tn, touch_trial
    if model_set == "augmented" and (swaps is None or skill_ball not in SKILL_BALLS):
        raise ValueError("The augmented set needs explicit `swaps` and `skill_ball` choices")
    if model_set == "current":
        swaps, skill_ball = None, None
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    manifest = json.loads((source / "manifest.json").read_text())
    token_manifest = json.loads((stored_tokens / "manifest.json").read_text())
    if (token_manifest["base_manifest"] != digest(source / "manifest.json")
            or token_manifest["player_set_sha256"] != digest(PACKAGE / "player_set.py")):
        raise ValueError(f"{stored_tokens} holds tokens for a different prepared folder or token code")
    output = record_folder / f"prepare_{model_set}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D49 preparation is recorded once")
    sequences.mkdir(parents=True, exist_ok=False)
    tokens.mkdir(parents=True, exist_ok=False)
    paths = provider_paths(skill)
    order = benchmark.SKILL_SPLITS["all"]
    compared, new_records, new_tokens, balls, visibilities = {}, [], {}, {"dfl": [], "skillcorner": []}, {}
    halves = touch_trial.training_halves(model_set, dfl, skill, raw=raw)
    for (record, path), half in zip(aligned_trial.verified_records(source), halves, strict=True):
        name = record["file"]
        if name != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match the stored record {name}")
        frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        if table.columns.tolist() != manifest["columns"]:
            raise ValueError("External feature schema differs")
        values, targets = table.to_numpy(dtype=np.float32), tn.targets(half.labels[frames])
        player_tokens = player_set.tokens(half.positions, half.ball, half.teams, half.pitch, frames).astype(np.float16)
        token_path = stored_tokens / name
        if digest(token_path) != token_manifest["records"][name]:
            raise ValueError(f"Stored token file hash differs: {token_path}")
        with np.load(path, allow_pickle=False) as item, np.load(token_path, allow_pickle=False) as stored:
            compared[name] = {"values": same_bytes(item["values"], values), "targets": same_bytes(item["targets"], targets),
                              "tokens": same_bytes(stored["tokens"], player_tokens)}
        shutil.copyfile(path, sequences / name)
        shutil.copyfile(token_path, tokens / name)
        if digest(sequences / name) != record["sha256"] or digest(tokens / name) != token_manifest["records"][name]:
            raise ValueError(f"A copied record differs from the stored one: {name}")
        provider = None
        if half.source == "skillcorner" and model_set == "augmented" and skill_ball == "provider":
            with np.load(paths[half.match, half.index], allow_pickle=False) as item:
                provider = item["ball"]
        made = hv.training_view(model_set, half, swaps=bool(swaps), provider_ball=provider,
                                skill_order=order.index(int(half.match)) if half.source == "skillcorner" else None)
        print(f"Reproduced {model_set} {name}: {'equal' if all(compared[name].values()) else 'DIFFERS'}", flush=True)
        if made is None:
            continue
        view, visibility = made
        view_frames, view_table, _ = long_context.features(view.positions, view.ball, view.teams, view.pitch)
        if view_table.columns.tolist() != manifest["columns"]:
            raise ValueError("External feature schema differs")
        view_name = f"{view.key}.npz"
        np.savez_compressed(sequences / view_name, values=view_table.to_numpy(dtype=np.float32),
                            targets=tn.targets(view.labels[view_frames]))
        new_records.append({"file": view_name, "source": view.source, "match": view.match,
                            "sha256": digest(sequences / view_name)})
        np.savez_compressed(tokens / view_name, tokens=player_set.tokens(view.positions, view.ball, view.teams, view.pitch,
                                                                         view_frames).astype(np.float16))
        new_tokens[view_name] = digest(tokens / view_name)
        balls[view.source].append(np.asarray(view.ball))
        visibilities[view_name] = visibility
        print(f"Added {model_set} {view_name} (v {visibility:.3f})", flush=True)
    write_json(sequences / "manifest.json", merged_manifest(manifest, new_records, {
        "base_manifest": digest(source / "manifest.json"), "view": hv.VIEW,
        "hidden_views_sha256": digest(PACKAGE / "hidden_views.py"), "swaps": swaps, "skill_ball": skill_ball}))
    write_json(tokens / "manifest.json", {"protocol": PROTOCOL, "base_manifest": digest(sequences / "manifest.json"),
                                          "records": token_manifest["records"] | new_tokens,
                                          "player_set_sha256": digest(PACKAGE / "player_set.py")})
    passed = len(compared) == len(manifest["records"]) and all(all(item.values()) for item in compared.values())
    write_new(output, {"protocol": PROTOCOL, "set": model_set, "passed": passed, "stored_records": len(manifest["records"]),
                       "reproduced": len(compared), "differing": {name: item for name, item in compared.items()
                                                                  if not all(item.values())},
                       "new_records": len(new_records), "choices": {"swaps": swaps, "skill_ball": skill_ball},
                       "seeds": hv.SEEDS, "visibility_drawn": visibilities,
                       "views": {key: hv.view_statistics(value) for key, value in balls.items()},
                       "stored_manifest": digest(source / "manifest.json"),
                       "stored_token_manifest": digest(stored_tokens / "manifest.json"),
                       "sequences_manifest": digest(sequences / "manifest.json"),
                       "tokens_manifest": digest(tokens / "manifest.json"),
                       "hidden_views_sha256": digest(PACKAGE / "hidden_views.py"),
                       "prepared_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D49 prepare {model_set}: {len(compared)} stored records reproduced byte for byte: {passed}; "
          f"{len(new_records)} views added", flush=True)
    if not passed:
        raise ValueError("D49 pipeline check failed: the stored training records are not reproduced byte for byte")


# -- networks --------------------------------------------------------------------------------------

def network_folders(models):
    from dfl_submission import hyper_trial as ht
    return {key: models / key for key in ht.SETS}


def train_networks(model_set, sequences, tokens, scratch, models, *, record_folder=RECIPE, count=training_processes):
    """D42's training and export for one set, one network at a time, only while a training slot is free."""
    require_external_only()
    from dfl_submission import hyper_trial as ht
    check = json.loads((record_folder / f"prepare_{model_set}.json").read_text())
    if (not check["passed"] or check["sequences_manifest"] != digest(sequences / "manifest.json")
            or check["tokens_manifest"] != digest(tokens / "manifest.json")):
        raise ValueError("The D49 pipeline check did not pass for these sequences and tokens")
    for folder, (seed, boost) in ht.NETWORKS.items():
        output = network_folders(models)[model_set] / folder
        if (output / "fit.json").exists():
            continue
        raw = scratch / model_set / folder
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            wait_for_slot(count=count)
            print(f"{time.strftime('%H:%M:%S')} training {model_set} {folder}", flush=True)
            subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "train", "--source", str(sequences),
                            "--tokens", str(tokens), "--out", str(raw), "--seed", str(seed), "--boundary-boost",
                            str(boost)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))
        print(f"{time.strftime('%H:%M:%S')} exported {model_set} {folder}", flush=True)


# -- guard views -----------------------------------------------------------------------------------

def guard_tracking(dfl, *, raw):
    """{(source, match, index): (labels, positions, ball, teams, pitch)} of the two long-hidden guard views."""
    from dfl_submission import external_benchmark as benchmark, metrica_sample as ms
    result = {}
    for half in benchmark.halves(dfl, None, "development", raw=raw):
        if half.source == "dfl" and half.view == "clean":
            positions, ball = hv.guard_view(half.positions, half.ball, hv.guard_dfl_seed(half.match, half.index))
            result["dfl", half.match, half.index] = (np.asarray(half.labels), positions, ball, half.teams, half.pitch)
    for game in ms.GAMES:
        for index, half in enumerate(ms.load(ms.PARSED, game)):
            positions, ball = hv.guard_view(np.asarray(half["positions"], dtype=np.float32), half["ball"],
                                            hv.guard_metrica_seed(game, index))
            result["metrica", str(game), index] = (np.asarray(half["labels"]), positions, ball, half["teams"], half["pitch"])
    return result


def guard_statistics(tracking):
    return {name: hv.view_statistics([value[2] for key, value in tracking.items() if key[0] == name])
            for name in ("dfl", "metrica")}


def guard_identity(models_identity, *, raw):
    from dfl_submission import metrica_sample as ms
    return {"protocol": PROTOCOL, "models": models_identity, "hidden_views_sha256": digest(PACKAGE / "hidden_views.py"),
            "seeds": {key: hv.SEEDS[key] for key in ("guard_dfl", "guard_metrica")}, "dfl_raw": raw,
            "metrica": ms.inputs(ms.PARSED), "teacher_sha256": diagnosis.teacher_identity()}


def guard_halves(tracking, model, cache, identity):
    """The model's decoder inputs on every guard half, cached by identity."""
    from dfl_submission import player_trial

    def produce():
        for (source, match, index), (labels, positions, ball, teams, pitch) in sorted(tracking.items()):
            signals, decoded = player_trial.half_signals(model, positions, ball, teams, pitch)
            yield diagnosis.HalfSignals(source, match, hv.VIEW, index, labels, signals, decoded)
            print(f"Forecast guard {source} {match} half {index + 1}", flush=True)

    return diagnosis.collect(cache, identity, produce)


def guard_scores(halves, recipe):
    """N = 3 spellF1 of (a) DFL WPY and (b) the mean of the three Metrica games, and their mean."""
    from dfl_submission import metrica_sample as ms
    scored = diagnosis.matches([half for half in halves if half.source == "dfl"],
                               lambda half: diagnosis.decode(half.signals, recipe))
    dfl = {key[1]: diagnosis.score(diagnosis.target(truth, 3.0), prediction)["spell_f1"]
           for key, (truth, prediction, _) in scored.items()}
    games = {}
    for half in sorted((half for half in halves if half.source == "metrica"), key=lambda item: (int(item.match), item.index)):
        games.setdefault(int(half.match), []).append(half)
    metrica = {}
    for game, parts in games.items():
        if [part.index for part in parts] != [0, 1]:
            raise ValueError(f"Expected two halves in order for Metrica game {game}")
        truths = ms.targets(ms.load(ms.PARSED, game))
        if not np.array_equal(truths["raw"], np.r_[parts[0].labels, pm.X, parts[1].labels].astype(np.int8)):
            raise ValueError("The Metrica guard labels differ from D19's raw target")
        metrica[str(game)] = ms.score(truths, [diagnosis.decode(part.signals, recipe) for part in parts])[TARGET]["spell_f1"]
    a, b = float(np.mean(list(dfl.values()))), float(np.mean(list(metrica.values())))
    return {"dfl": dfl, "metrica": metrica, "a_dfl": a, "b_metrica": b, "mean": (a + b) / 2}


# -- development groups ----------------------------------------------------------------------------

def group_scores(development, metrica, d24, recipe, truths):
    """D47's eight N = 3 groups, D47's composite and D33's primary, decoded at `recipe`."""
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    trial = aligned_trial.evaluate(development, recipe)
    conventions = trial["external"]["conventions"][TARGET]
    matches = conventions["matches"]
    decoded = objective_trial.decoded(metrica, recipe)
    metrica_scores = {str(game): ms.score(truths[game], decoded[game]) for game in ms.GAMES}
    views = mv.view_scores(d24, truths, lambda half: diagnosis.decode(half.signals, recipe))
    groups = {"dfl_clean": matches["dfl/WPY/clean"]["spell_f1"], "dfl_masked": matches["dfl/WPY/masked"]["spell_f1"],
              "dfl_dead_carried": conventions["dead_carried"], "dfl_team_swap": conventions["team_swap"],
              "skillcorner": float(np.mean([value["spell_f1"] for key, value in matches.items()
                                            if key.startswith("skillcorner/")])),
              "pff": trial["pff"]["means"][TARGET]["spell_f1"],
              "metrica": ms.means(metrica_scores)[TARGET]["spell_f1"], "d24": mv.d24_mean(views)}
    return {"groups": groups, "composite": float(np.mean([groups[name] for name in GROUPS])),
            "primary": refine_trial.standard(trial, TARGET), "d24_views": views, "metrica_matches": metrica_scores,
            "trial": trial}


def pff_visibility(parsed, games):
    from dfl_submission import pff
    return {("pff", str(game), "observed", index): (np.asarray(half["labels"]), seen(half["ball"]))
            for game in games for index, half in enumerate(pff.load(parsed, game))}


def seen(ball):
    return np.isfinite(np.asarray(ball, dtype=np.float64)).all(axis=1)


def pff_statistics(development, recipe, visibility):
    """PFF's owner agreement on seen / unseen frames and recall at IoU 0.5 of under-20%-seen spells (D46)."""
    from dfl_submission import diagnosis_d43 as d46
    halves = [half for half in development if half.source == "pff"]
    attached, _ = d46.attach(halves, visibility)
    visible = d46.joined(halves, attached, False)
    since = d46.joined(halves, {key: d46.frames_since_seen(value) for key, value in attached.items()}, np.inf)
    scored = diagnosis.matches(halves, lambda half: diagnosis.decode(half.signals, recipe))
    records = [d46.visibility_match(diagnosis.target(truth, 3.0), prediction, visible[key], since[key])
               for key, (truth, prediction, _) in sorted(scored.items())]
    spells = [hit for record in records for share, _, hit in record["spells"] if share < .2]
    return {"agreement_seen": d46.pooled_agreement([record["agreement"]["seen"] for record in records]),
            "agreement_unseen": d46.pooled_agreement([record["agreement"]["unseen"] for record in records]),
            "under_20%_seen_spells": {"count": len(spells), "recall_iou0.5": d46.ratio(float(sum(spells)), len(spells))}}


def entry(development, metrica, d24, guards, recipe, truths, visibility):
    scores = group_scores(development, metrica, d24, recipe, truths)
    return {"owner_cost": recipe["owner_cost"], "composite": scores["composite"], "primary": scores["primary"],
            "groups": scores["groups"], "guards": guard_scores(guards, recipe),
            "pff_visibility": pff_statistics(development, recipe, visibility), "d24_views": scores["d24_views"],
            "metrica_matches": scores["metrica_matches"], "trial": scores["trial"]}


def summary_line(name, item):
    return (f"{name} cost {item['owner_cost']}: composite {item['composite']:.5f}, primary {item['primary']:.5f}, "
            + ", ".join(f"{group} {value:.4f}" for group, value in item["groups"].items())
            + f"; guards a {item['guards']['a_dfl']:.4f} b {item['guards']['b_metrica']:.4f} mean {item['guards']['mean']:.4f}; "
            f"PFF agreement seen {item['pff_visibility']['agreement_seen']['agreement']:.3f} unseen "
            f"{item['pff_visibility']['agreement_unseen']['agreement']:.3f}, under-20%-seen recall "
            f"{item['pff_visibility']['under_20%_seen_spells']['recall_iou0.5']:.3f}")


def carry(reference, entries):
    """D49's carry rule: the cost with the highest mean of (guard mean, PFF mean), then the three conditions."""
    chosen = max(entries, key=lambda item: (item["guards"]["mean"] + item["groups"]["pff"]) / 2)
    guard = chosen["guards"]["mean"] - reference["guards"]["mean"]
    pff_change = chosen["groups"]["pff"] - reference["groups"]["pff"]
    composite = chosen["composite"] - reference["composite"]
    groups = {name: chosen["groups"][name] - reference["groups"][name] for name in GROUPS}
    conditions = {"guard_or_pff": bool(guard >= GUARD_GAIN or pff_change >= PFF_GAIN),
                  "composite": bool(composite >= -COMPOSITE_LOSS),
                  "groups": bool(all(value >= -GROUP_LOSS for value in groups.values()))}
    return {"selected_cost": chosen["owner_cost"], "changes": {"guard_mean": guard, "pff": pff_change,
                                                                "composite": composite, "groups": groups},
            "conditions": conditions, "carried": bool(all(conditions.values()))}


def compact(item):
    return {key: value for key, value in item.items() if key != "trial"} | {
        "trial": {"external": item["trial"]["external"], "pff_means": item["trial"]["pff"]["means"]}}


def d43_inputs(dfl, skill, parsed, games, *, raw, d42_cache):
    """D43's cached development, Metrica and D24 inputs, after its D42 development row reproduces exactly."""
    from dfl_submission import diagnosis_d43 as d46, hyper_trial as ht, metrica_sample as ms, player_trial
    record = json.loads((player_trial.RECIPE / "development.json").read_text())
    if record["models"] != player_trial.identity():
        raise ValueError("The D42 networks differ from the ones development scored")
    development, metrica, d24 = d46.cached_halves(dfl, skill, parsed, games, raw=raw, cache_root=d42_cache)
    truths = {game: ms.targets(ms.load(ms.PARSED, game)) for game in ms.GAMES}
    checks = d46.compare_row(d46.row(development, metrica, d24, record, ht.recipe_at(D43_COST), truths),
                             d46.stored_row(record))
    if not all(checks.values()):
        raise ValueError(f"D43 does not reproduce its development row {checks}; nothing was written")
    return development, metrica, d24, truths, checks


def reference(dfl, skill, parsed, *, raw, d42_cache, cache_root=DATA / "cache", record_folder=RECIPE, wait=True):
    """D43 on the guard views and D47's eight groups at cost 10, written once before any candidate is scored.

    `wait=False` scores without waiting for a free training slot (CPU only; the orchestrator allowed it).
    """
    require_external_only()
    from dfl_submission import hyper_trial as ht, player_trial, vendor_trial
    output = record_folder / "reference.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D49's reference is written once")
    if wait:
        wait_for_slot()
    games = vendor_trial.development()
    development, metrica, d24, truths, checks = d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=d42_cache)
    tracking = guard_tracking(dfl, raw=raw)
    identity = guard_identity(player_trial.identity(), raw=raw)
    guards = guard_halves(tracking, player_trial.teacher(), cache_root / "guards_d43", identity)
    item = entry(development, metrica, d24, guards, ht.recipe_at(D43_COST), truths, pff_visibility(parsed, games))
    write_new(output, {"protocol": PROTOCOL, "candidate": "D43", "reproduction": checks, "recipe": ht.recipe_at(D43_COST),
                       "reference": compact(item), "guard_views": guard_statistics(tracking), "guard_identity": identity,
                       "models": player_trial.identity(), "inputs": {"pff": vendor_trial.inputs(parsed, games)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(summary_line("D49 reference D43", item), flush=True)
    return item


def candidate(dfl, skill, parsed, *, raw, d42_cache, models, cache_root=DATA / "cache", record_folder=RECIPE):
    """Candidate L at ownership costs 8, 10 and 12 against D43, with D49's carry rule, written once."""
    require_external_only()
    from dfl_submission import hyper_trial as ht, metrica_sample as ms, metrica_views as mv, player_trial, vendor_trial
    output = record_folder / "candidate.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D49's candidate is scored once")
    stored = json.loads((record_folder / "reference.json").read_text())
    for model_set in ("current", "augmented"):
        check = json.loads((record_folder / f"prepare_{model_set}.json").read_text())
        if not check["passed"]:
            raise ValueError(f"The {model_set} set's D49 preparation did not pass")
    wait_for_slot()
    games = vendor_trial.development()
    _, _, _, truths, checks = d43_inputs(dfl, skill, parsed, games, raw=raw, d42_cache=d42_cache)
    models_identity = player_trial.identity(models)
    model = player_trial.teacher(models=models)
    variant = {"name": VARIANT, "models": models_identity}
    development = player_trial.candidate_halves(dfl, skill, parsed, games, "development", raw=raw,
                                                cache=cache_root / "development", models=models)
    _, metrica = ms.reference_halves("D15", cache=cache_root / "metrica", model=model,
                                     signals_of=player_trial.half_signals, variant=variant)
    d24 = mv.view_halves(cache=cache_root / "d24", signals_of=lambda _, *tracking: player_trial.half_signals(model, *tracking),
                         variant=variant)
    tracking = guard_tracking(dfl, raw=raw)
    guards = guard_halves(tracking, model, cache_root / "guards_l", guard_identity(models_identity, raw=raw))
    visibility = pff_visibility(parsed, games)
    entries = []
    for cost in COSTS:
        item = entry(development, metrica, d24, guards, ht.recipe_at(cost), truths, visibility)
        entries.append(item)
        print(summary_line("D49 candidate L", item), flush=True)
    reference_item = stored["reference"]
    decision = carry(reference_item, entries)
    write_new(output, {"protocol": PROTOCOL, "candidate": "L", "costs": list(COSTS), "reference_record":
                       digest(record_folder / "reference.json"), "d43_reproduction": checks,
                       "entries": [compact(item) for item in entries], "decision": decision, "models": models_identity,
                       "prepare": {model_set: digest(record_folder / f"prepare_{model_set}.json")
                                   for model_set in ("current", "augmented")},
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D49 decision: cost {decision['selected_cost']}, guard mean {decision['changes']['guard_mean']:+.4f}, PFF "
          f"{decision['changes']['pff']:+.4f}, composite {decision['changes']['composite']:+.4f}, worst group "
          f"{min(decision['changes']['groups'].values()):+.4f} -> {'carried' if decision['carried'] else 'not carried'}",
          flush=True)
    return decision


def main():
    from dfl_submission import hyper_trial as ht
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--set", choices=ht.SETS, required=True)
    preparation.add_argument("--source", type=Path, required=True, help="the set's stored prepared sequences")
    preparation.add_argument("--stored-tokens", type=Path, required=True, help="D42's tokens for that folder")
    preparation.add_argument("--dfl", type=Path, required=True)
    preparation.add_argument("--dfl-raw", action="store_true")
    preparation.add_argument("--skill", type=Path, required=True)
    preparation.add_argument("--sequences", type=Path, required=True)
    preparation.add_argument("--tokens", type=Path, required=True)
    preparation.add_argument("--augmented-swaps", choices=tuple(SWAP_CHOICES))
    preparation.add_argument("--augmented-skill-ball", choices=SKILL_BALLS)
    networks = commands.add_parser("networks")
    networks.add_argument("--set", choices=ht.SETS, required=True)
    networks.add_argument("--sequences", type=Path, required=True)
    networks.add_argument("--tokens", type=Path, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    networks.add_argument("--models", type=Path, required=True)
    for name in ("reference", "candidate"):
        command = commands.add_parser(name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--d42-cache", type=Path, required=True)
        command.add_argument("--cache", type=Path, default=DATA / "cache")
        if name == "candidate":
            command.add_argument("--models", type=Path, required=True)
        else:
            command.add_argument("--no-slot-wait", action="store_true", help="score without waiting for a training slot")
    args = parser.parse_args()
    require_external_only()
    if args.command == "prepare":
        prepare(args.set, args.dfl, args.skill, args.source, args.stored_tokens, args.sequences, args.tokens,
                raw=args.dfl_raw, swaps=None if args.augmented_swaps is None else SWAP_CHOICES[args.augmented_swaps],
                skill_ball=args.augmented_skill_ball)
    elif args.command == "networks":
        train_networks(args.set, args.sequences, args.tokens, args.scratch, args.models)
    elif args.command == "reference":
        reference(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d42_cache=args.d42_cache, cache_root=args.cache,
                  wait=not args.no_slot_wait)
    else:
        candidate(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d42_cache=args.d42_cache, models=args.models,
                  cache_root=args.cache)


if __name__ == "__main__":
    main()
