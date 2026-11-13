"""D45: the player-set model refit on the training and development matches (DECISIONS.md D45).

`prepare` builds one set's `train_dev` prepared sequences and player tokens and checks that every half
of the 16 training matches equals the stored training sequence exactly. `networks` trains and exports
the set's two player-set networks as D42 did. `vetoes` compares the refit with D43 on the PFF
development matches, the Metrica games and the D24 views; `confirm` scores it once on the reused holdout
and the untouched PFF set against D43; `build` writes the competition file. The model code is
`player_set`'s and `player_trial`'s with another models folder.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "refit_recipe"
MODELS = PACKAGE / "refit_models"
PROTOCOL = "DECISIONS.md D45"
SPLIT = "train_dev"
COST = 10
LOSS, GAIN = .01, .01
NETWORKS = {"temporal_models": (42, .5), "seed_models": (97, 1.4)}
SETS = ("current", "augmented")


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D45 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def halves_of(model_set, dfl, skill, *, raw):
    from dfl_submission import augment, external_benchmark
    if model_set == "current":
        return external_benchmark.halves(dfl, skill, SPLIT, raw=raw)
    return augment.training_halves(dfl, skill, raw=raw, split=SPLIT)


def prepare(model_set, dfl, skill, stored, sequences, tokens, *, raw, record_folder=RECIPE):
    """`train_dev` sequences and tokens of one set, checked against the stored training sequences."""
    require_external_only()
    from dfl_submission import long_context, player_set, temporal_network as tn, touch_trial
    from dfl_submission.train import SOURCE_MANIFEST
    if digest(stored / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{stored} is not the {model_set} set's stored training sequences")
    stored_manifest = json.loads((stored / "manifest.json").read_text())
    stored_files = {record["file"]: record for record in stored_manifest["records"]}
    sequences.mkdir(parents=True, exist_ok=False)
    tokens.mkdir(parents=True, exist_ok=False)
    records, token_records, columns, compared = [], {}, None, {}
    for half in halves_of(model_set, dfl, skill, raw=raw):
        frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        if columns is None:
            columns = table.columns.tolist()
        elif columns != table.columns.tolist():
            raise ValueError("External feature schema differs")
        name = f"{half.key}.npz"
        values, targets = table.to_numpy(dtype=np.float32), tn.targets(half.labels[frames])
        if name in stored_files:
            with np.load(stored / name, allow_pickle=False) as item:
                compared[name] = bool(np.array_equal(item["values"], values, equal_nan=True)
                                      and np.array_equal(item["targets"], targets))
        np.savez_compressed(sequences / name, values=values, targets=targets)
        records.append({"file": name, "source": half.source, "match": half.match, "sha256": digest(sequences / name)})
        np.savez_compressed(tokens / name, tokens=player_set.tokens(half.positions, half.ball, half.teams, half.pitch,
                                                                     frames).astype(np.float16))
        token_records[name] = digest(tokens / name)
        print(f"Prepared {model_set} {name}" + (f" (stored: {'equal' if compared[name] else 'DIFFERS'})"
                                                  if name in compared else ""), flush=True)
    write_json(sequences / "manifest.json", {"columns": columns, "records": records, "dfl_manifest": digest(SOURCE_MANIFEST),
                                             "skill_manifest": digest(skill / "manifest.json"), "split": SPLIT}
               | ({"augmented": True} if model_set == "augmented" else {}))
    write_json(tokens / "manifest.json", {"protocol": PROTOCOL, "base_manifest": digest(sequences / "manifest.json"),
                                          "records": token_records, "player_set_sha256": digest(PACKAGE / "player_set.py")})
    passed = len(compared) == len(stored_files) and all(compared.values())
    write_new(record_folder / f"prepare_{model_set}.json", {"protocol": PROTOCOL, "set": model_set, "passed": passed,
              "training_halves_compared": len(compared), "stored_halves": len(stored_files),
              "differing": sorted(name for name, same in compared.items() if not same),
              "halves": len(records), "sequences_manifest": digest(sequences / "manifest.json"),
              "tokens_manifest": digest(tokens / "manifest.json")})
    print(f"D45 prepare {model_set}: {len(records)} halves; stored training halves reproduced: {passed}", flush=True)
    if not passed:
        raise ValueError("D45 pipeline check failed: the train_dev split does not reproduce the stored training halves")


def network_folders(models=MODELS):
    return {key: models / key for key in SETS}


def train_networks(model_set, sequences, tokens, scratch, *, models=MODELS, record_folder=RECIPE):
    check = json.loads((record_folder / f"prepare_{model_set}.json").read_text())
    if not check["passed"] or check["sequences_manifest"] != digest(sequences / "manifest.json"):
        raise ValueError("The D45 pipeline check did not pass for these sequences")
    for folder, (seed, boost) in NETWORKS.items():
        output = network_folders(models)[model_set] / folder
        if (output / "fit.json").exists():
            continue
        raw = scratch / model_set / folder
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "train", "--source", str(sequences),
                            "--tokens", str(tokens), "--out", str(raw), "--seed", str(seed), "--boundary-boost",
                            str(boost)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


def recipe():
    from dfl_submission import hyper_trial as ht
    return ht.recipe_at(COST)


def metrica_mean(halves, candidate_recipe):
    from dfl_submission import metrica_sample as ms, objective_trial
    return ms.veto(objective_trial.decoded(halves, candidate_recipe), reference="D15")["candidate"]["means"]["aba3"]["spell_f1"]


def view_mean(views, truths, candidate_recipe):
    from dfl_submission import diagnosis, metrica_views as mv
    return mv.d24_mean(mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate_recipe)))


def pff_mean(halves, candidate_recipe):
    from dfl_submission import aligned_trial
    return aligned_trial.evaluate(halves, candidate_recipe)["pff"]["means"]["aba3"]["spell_f1"]


def vetoes(dfl, skill, parsed, *, raw, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import diagnosis, metrica_sample as ms, metrica_views as mv, player_trial, vendor_trial
    output = recipe_folder / "vetoes.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D45's vetoes are scored once")
    games = vendor_trial.development()
    d43_cache = ROOT / "data" / "d42" / "cache"
    d43_dev = player_trial.candidate_halves(dfl, skill, parsed, games, "development", raw=raw,
                                            cache=d43_cache / "development")
    d43_pff = [half for half in d43_dev if half.source == "pff"]
    d43_model = player_trial.teacher()
    _, d43_metrica = ms.reference_halves("D15", cache=d43_cache / "metrica", model=d43_model,
                                         signals_of=player_trial.half_signals, variant="D42 player set")
    d43_views = mv.view_halves(cache=d43_cache / "d24", signals_of=lambda _, *t: player_trial.half_signals(d43_model, *t),
                               variant="D42 player set")
    _, _, truths = mv.reference_scores()
    model = player_trial.teacher(models=models)
    key = dict(player_trial.identity(models), protocol=PROTOCOL, teacher_sha256=diagnosis.teacher_identity(),
               pff=vendor_trial.inputs(parsed, games))
    pff_halves = diagnosis.collect(cache_root / "pff_development", key,
                                   lambda: diagnosis.pff_halves(parsed, games, model, signals_of=player_trial.half_signals))
    _, metrica = ms.reference_halves("D15", cache=cache_root / "metrica", model=model,
                                     signals_of=player_trial.half_signals, variant="D45 refit")
    views = mv.view_halves(cache=cache_root / "d24", signals_of=lambda _, *t: player_trial.half_signals(model, *t),
                           variant="D45 refit")
    candidate_recipe = recipe()
    changes = {"pff": pff_mean(pff_halves, candidate_recipe) - pff_mean(d43_pff, candidate_recipe),
               "metrica": metrica_mean(metrica, candidate_recipe) - metrica_mean(d43_metrica, candidate_recipe),
               "d24": view_mean(views, truths, candidate_recipe) - view_mean(d43_views, truths, candidate_recipe)}
    passed = all(value >= -LOSS for value in changes.values())
    write_new(output, {"protocol": PROTOCOL, "reference": "D43", "changes": changes, "passed": passed,
                       "models": player_trial.identity(models), "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print("D45 vetoes against D43: " + ", ".join(f"{k} {v:+.4f}" for k, v in changes.items())
          + f" -> {'passed' if passed else 'failed'}", flush=True)
    return passed


def confirm(dfl, skill, parsed, *, raw, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, diagnosis, objective_trial, player_trial, refine_trial, vendor_trial
    record = json.loads((recipe_folder / "vetoes.json").read_text())
    if record["passed"] is not True or record["models"] != player_trial.identity(models):
        raise ValueError("D45's vetoes did not pass for these networks; D43 stays")
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D45 confirmation is scored once")
    candidate_recipe = recipe()
    games_c = vendor_trial.split_games("confirm_c")
    holdout = player_trial.candidate_halves(dfl, skill, parsed, games_c, "holdout", raw=raw,
                                            cache=ROOT / "data" / "d42" / "confirm" / "holdout")
    stored = json.loads((PACKAGE / "player_recipe" / "confirmation.json").read_text())["trial"]
    if vendor_trial.canonical(aligned_trial.evaluate(holdout, candidate_recipe)["external"]) != stored["external"]:
        raise ValueError("D43 does not reproduce its holdout record; no record was written")
    games_u = vendor_trial.split_games("untouched")
    d43_model = player_trial.teacher()
    key = {"protocol": PROTOCOL, "d43": player_trial.identity(), "teacher_sha256": diagnosis.teacher_identity(),
           "pff": vendor_trial.inputs(parsed, games_u)}
    untouched = diagnosis.collect(cache_root / "d43_untouched", key,
                                  lambda: diagnosis.pff_halves(parsed, games_u, d43_model,
                                                               signals_of=player_trial.half_signals))
    base = aligned_trial.evaluate([half for half in holdout if half.source != "pff"] + untouched, candidate_recipe)
    candidate = player_trial.candidate_halves(dfl, skill, parsed, games_u, "holdout", raw=raw,
                                              cache=cache_root / "holdout", models=models)
    trial = aligned_trial.evaluate(candidate, candidate_recipe)
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "reference": "D43", "recipe": candidate_recipe, "base": base,
                       "trial": trial, "changes": change, "passed": passed, "models": player_trial.identity(models),
                       "inputs": {"untouched": vendor_trial.inputs(parsed, games_u)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line("D45 confirmation against D43 (PFF = untouched set)", trial, change, "",
                                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the refit player-set networks at cost 10 in D15's structure, then D27's rule."""
    from dfl_submission import kickoff, objective_trial, player_trial
    from dfl_submission.build import tactical_halves, write_csv
    record = json.loads((recipe_folder / "confirmation.json").read_text())
    if record["passed"] is not True or record["models"] != player_trial.identity(models):
        raise ValueError("D45 did not pass confirmation; build D43 instead (`player_trial build`)")
    _, frozen_d15 = objective_trial.d15()
    model = player_trial.teacher(COST, models)

    def decode_half(positions, ball, teams, pitch):
        return player_trial.half_signals(model, positions, ball, teams, pitch)[1]

    labels = tactical_halves(decode_half, centered=True, pitch=frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--set", choices=SETS, required=True)
    preparation.add_argument("--dfl", type=Path, required=True)
    preparation.add_argument("--dfl-raw", action="store_true")
    preparation.add_argument("--skill", type=Path, required=True)
    preparation.add_argument("--stored", type=Path, required=True)
    preparation.add_argument("--sequences", type=Path, required=True)
    preparation.add_argument("--tokens", type=Path, required=True)
    networks = commands.add_parser("networks")
    networks.add_argument("--set", choices=SETS, required=True)
    networks.add_argument("--sequences", type=Path, required=True)
    networks.add_argument("--tokens", type=Path, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    for command_name in ("vetoes", "confirm"):
        command = commands.add_parser(command_name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "prepare":
        prepare(args.set, args.dfl, args.skill, args.stored, args.sequences, args.tokens, raw=args.dfl_raw)
    elif args.command == "networks":
        train_networks(args.set, args.sequences, args.tokens, args.scratch)
    elif args.command == "vetoes":
        vetoes(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, cache_root=args.cache)


if __name__ == "__main__":
    main()
