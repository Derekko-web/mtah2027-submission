"""D42: the player-set network in D15's structure (DECISIONS.md D42).

`prepare` re-extracts every external training half of one model set, checks the 162 stored columns
and frames, and writes the half's player tokens. `networks` trains and exports the set's two
player-set networks. `develop` scores the candidate once on development against D15, `confirm` once on
the reused holdout and PFF confirmation set C, and `build` writes the competition file.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from dfl_submission import diagnosis, hyper_trial as ht
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "player_recipe"
MODELS = PACKAGE / "player_models"
PROTOCOL = "DECISIONS.md D42"
OWNER_COSTS = (8, 10)
GAIN = .01


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D42 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


# -- tokens for the training halves ----------------------------------------------------------------

def prepare(model_set, source, dfl, skill, output, *, raw):
    from dfl_submission import aligned_trial, long_context, player_set, touch_trial
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    halves = touch_trial.training_halves(model_set, dfl, skill, raw=raw)
    for (record, path), half in zip(aligned_trial.verified_records(source), halves, strict=True):
        if record["file"] != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match the prepared record {record['file']}")
        frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        with np.load(path, allow_pickle=False) as item:
            values = item["values"]
        if table.columns.tolist() != manifest["columns"] or len(frames) != len(values) or not np.array_equal(
                table.to_numpy(dtype=np.float32), values, equal_nan=True):
            raise ValueError(f"Re-extracted features differ from the stored prepared sequence: {record['file']}")
        player_tokens = player_set.tokens(half.positions, half.ball, half.teams, half.pitch, frames)
        np.savez_compressed(output / record["file"], tokens=player_tokens.astype(np.float16))
        records[record["file"]] = digest(output / record["file"])
        print(f"Tokens {record['file']}", flush=True)
    (output / "manifest.json").write_text(json.dumps(
        {"protocol": PROTOCOL, "base_manifest": digest(source / "manifest.json"), "records": records,
         "player_set_sha256": digest(PACKAGE / "player_set.py")}, indent=2, sort_keys=True) + "\n")


def network_folders(models=MODELS):
    return {key: models / key for key in ht.SETS}


def train_networks(source, tokens, model_set, scratch, *, models=MODELS):
    from dfl_submission import touch_trial
    if digest(source / "manifest.json") != touch_trial.published_manifest(model_set):
        raise ValueError(f"{source} is not the prepared-sequence folder the {model_set} set was trained on")
    for folder, (seed, boost) in ht.NETWORKS.items():
        output = network_folders(models)[model_set] / folder
        if (output / "fit.json").exists():
            continue
        raw = scratch / model_set / folder
        if not (raw / "fit.json").exists():
            if raw.exists():
                raise FileExistsError(f"{raw} holds an unfinished run; remove it first")
            subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "train", "--source", str(source),
                            "--tokens", str(tokens), "--out", str(raw), "--seed", str(seed), "--boundary-boost",
                            str(boost)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "dfl_submission.player_set", "export", "--source", str(raw),
                        "--out", str(output)], check=True, cwd=ROOT, env=dict(os.environ, OMP_NUM_THREADS="1"))


# -- forecasts ---------------------------------------------------------------------------------------

class PlayerCheckpoint:
    """A player-set network as the forecasts below use it."""

    def __init__(self, folder):
        from dfl_submission import player_set
        fit = json.loads((folder / "fit.json").read_text())
        self.filename = f"temporal_{fit['steps']}.npz"
        self.checkpoint = player_set.load(folder, self.filename)
        self.columns = self.checkpoint[1]

    def predict(self, frames, owner, player_tokens, length):
        from dfl_submission import player_set
        values = owner[self.columns].to_numpy(dtype=np.float32)
        return player_set.predict(self.checkpoint, frames, values, list(self.columns), player_tokens, length)


def models_of(folder):
    """(trees, seed-42 network, seed-97 network) of one model set."""
    from dfl_submission import long_context, scale_trial
    return folder


def forecasts(trees, original, late, positions, ball, teams, pitch):
    """`ensemble_trial.forecasts` for D15's variants (seed_phase) with player-set networks."""
    from dfl_submission import ensemble_trial, long_context, player_set, temporal_trial
    phase_seed = []
    start_reference = None
    for offset in ensemble_trial.PHASES:
        shifted_positions, shifted_ball = positions[offset:], ball[offset:]
        frames, owner, dead = long_context.features(shifted_positions, shifted_ball, teams, pitch)
        length = len(ball) - offset
        player_tokens = player_set.tokens(shifted_positions, shifted_ball, teams, pitch, frames)
        if offset == 0:
            baseline, _ = temporal_trial.forecast_features(trees, {}, frames, owner, dead, length)
            start_reference = (baseline, {ensemble_trial.LATE: original.predict(frames, owner, player_tokens, length)})
        phase_seed.append(tuple(np.pad(p, (offset, 0), mode="edge")
                                for p in late.predict(frames, owner, player_tokens, length)))
    return start_reference, {"seed_phase": ensemble_trial.mean(phase_seed)}


def teacher(owner_cost=None, models=MODELS):
    from dfl_submission import competition_adaptation, long_context, scale_trial
    reference, _, frozen = competition_adaptation.reference()
    packages = {}
    for role, key in scale_trial.ROLES.items():
        forest, _ = long_context.load(scale_trial.TREES[key])
        folder = network_folders(models)[key]
        packages[role] = (forest, PlayerCheckpoint(folder / "temporal_models"), PlayerCheckpoint(folder / "seed_models"))
    if owner_cost is not None:
        frozen = dict(frozen, recipe=ht.recipe_at(owner_cost))
    return reference, packages, frozen


def half_signals(model, positions, ball, teams, pitch):
    """`diagnosis.half_signals` with the player-set forecasts."""
    from dfl_submission import motion_trial, retrain_trial
    reference, packages, frozen = model
    previous = reference[2]["recipe"]
    if previous["variant"] != "temporal_1800.npz" or frozen["recipe"]["variant"] != "seed_phase":
        raise ValueError("D42 expects D15's variants")
    outputs = tuple(forecasts(*packages[name], positions, ball, teams, pitch) for name in ("current", "candidate"))
    mixed = retrain_trial.mix(*outputs, frozen["weight"])
    labels = motion_trial.decode(mixed, previous, frozen["recipe"])
    signals = tuple(motion_trial.blend_signals(mixed, previous, frozen["recipe"]))
    if not np.array_equal(diagnosis.decode(signals, frozen["recipe"]), labels):
        raise ValueError("Decoding the cached inputs does not reproduce the labels")
    return signals, labels


def identity(models=MODELS):
    from dfl_submission import scale_trial
    folders = network_folders(models)
    return {"protocol": PROTOCOL,
            "networks": {key: {folder: digest(folders[key] / folder / "fit.json") for folder in ht.NETWORKS}
                         for key in ht.SETS},
            "trees": {key: digest(scale_trial.TREES[key] / "fit.json") for key in ht.SETS},
            "player_set_sha256": digest(PACKAGE / "player_set.py")}


def candidate_halves(dfl, skill, parsed, games, split, *, raw, cache, models=MODELS):
    from dfl_submission import vendor_trial
    key = dict(identity(models), teacher_sha256=diagnosis.teacher_identity(), split=split, dfl_raw=raw,
               skill_manifest_sha256=digest(skill / "manifest.json"), pff=vendor_trial.inputs(parsed, games))
    model = teacher(models=models)

    def produce():
        yield from diagnosis.external_halves(dfl, skill, split, model, raw=raw, signals_of=half_signals)
        yield from diagnosis.pff_halves(parsed, games, model, signals_of=half_signals)

    return diagnosis.collect(cache, key, produce)


def develop(dfl, skill, parsed, *, raw, d15_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, metrica_sample as ms, metrica_views as mv, objective_trial, refine_trial
    from dfl_submission import stoppage_trial, vendor_trial, zoo
    output = recipe_folder / "development.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D42 development is scored once")
    _, recipe, base = zoo.d15_development(dfl, skill, parsed, raw=raw, cache=d15_cache)
    _, d19_halves, _ = ms.reference_scores("D15")
    d24_reference, d24_reference_halves, truths = mv.reference_scores()
    if not (stoppage_trial.same_labels(d19_halves, recipe) and stoppage_trial.same_labels(d24_reference_halves, recipe)):
        raise ValueError("D15's recipe does not reproduce its cached Metrica labels; nothing was written")
    d24_base = mv.d24_mean(d24_reference)
    games = vendor_trial.development()
    halves = candidate_halves(dfl, skill, parsed, games, "development", raw=raw, cache=cache_root / "development",
                              models=models)
    model = teacher(models=models)
    _, metrica = ms.reference_halves("D15", cache=cache_root / "metrica", model=model, signals_of=half_signals,
                                     variant="D42 player set")
    views = mv.view_halves(cache=cache_root / "d24", signals_of=lambda _, *tracking: half_signals(model, *tracking),
                           variant="D42 player set")
    entries = []
    for cost in OWNER_COSTS:
        candidate = ht.recipe_at(cost)
        trial = aligned_trial.evaluate(halves, candidate)
        change = refine_trial.changes(base, trial)
        veto = ms.veto(objective_trial.decoded(metrica, candidate), reference="D15")
        view_scores = mv.view_scores(views, truths, lambda half: diagnosis.decode(half.signals, candidate))
        d24 = {"reference": d24_base, "candidate": mv.d24_mean(view_scores)}
        d24["change"] = d24["candidate"] - d24["reference"]
        allowed = objective_trial.eligible(change, veto["changes"]["passed"], d24["change"])
        entries.append({"point": {"owner_cost": cost}, "eligible": allowed, "changes": change, "trial": trial,
                        "metrica": {"changes": veto["changes"], "means": veto["candidate"]["means"]},
                        "d24": dict(d24, views=view_scores)})
        print(objective_trial.summary_line(f"D42 development cost {cost}", trial, change,
                                           f", Metrica {veto['changes']['primary']:+.4f} (veto "
                                           f"{'passed' if veto['changes']['passed'] else 'failed'}), D24 {d24['change']:+.4f}",
                                           "eligible" if allowed else "not eligible"), flush=True)
    selected = objective_trial.select(entries)
    write_new(output, {"protocol": PROTOCOL, "costs": list(OWNER_COSTS), "recipe": recipe, "reference": base,
                       "reproduction": {"d15_development": True, "d15_d19": True, "d15_d24": True},
                       "entries": entries, "selected": selected, "models": identity(models),
                       "inputs": {"pff": vendor_trial.inputs(parsed, games), "metrica": ms.inputs(ms.PARSED)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D42 development selected {selected if selected else 'no point; D15 stays'}", flush=True)
    return selected


D43_RECORD = "development_d43.json"
D43_COST, D43_GAIN = 10, .005


def d43_select(recipe_folder=RECIPE, models=MODELS):
    """D43: D42's cost-10 development row under a +0.005 primary gate (every other D33 gate unchanged)."""
    require_external_only()
    from dfl_submission import refine_trial
    output = recipe_folder / D43_RECORD
    if output.exists():
        raise FileExistsError(f"{output} exists; D43's selection is recorded once")
    development = json.loads((recipe_folder / "development.json").read_text())
    if development["models"] != identity(models):
        raise ValueError("The D42 networks differ from the ones development scored")
    entry = next(e for e in development["entries"] if e["point"]["owner_cost"] == D43_COST)
    allowed = bool(refine_trial.passed(entry["changes"], gain=D43_GAIN) and entry["metrica"]["changes"]["passed"]
                   and entry["d24"]["change"] >= -.01)
    write_new(output, {"protocol": "DECISIONS.md D43", "from": "DECISIONS.md D42 development", "gain": D43_GAIN,
                       "entry": {key: entry[key] for key in ("point", "changes")}, "eligible": allowed,
                       "selected": entry["point"] if allowed else None, "models": development["models"],
                       "recorded_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D43 selection: cost {D43_COST}, primary {entry['changes']['primary']:+.5f} -> "
          f"{'confirmation' if allowed else 'not eligible'}", flush=True)
    return allowed


def frozen(recipe_folder=RECIPE, models=MODELS, record=D43_RECORD):
    development = json.loads((recipe_folder / record).read_text())
    point = development["selected"]
    if point is None:
        raise ValueError("No D42 point was eligible on development; D15 stays and nothing is confirmed")
    if development["models"] != identity(models):
        raise ValueError("The D42 networks differ from the ones development scored")
    return point


def confirm(dfl, skill, parsed, *, raw, d15_holdout_cache, cache_root, recipe_folder=RECIPE, models=MODELS):
    require_external_only()
    from dfl_submission import aligned_trial, objective_trial, refine_trial, scale_trial, vendor_trial
    point = frozen(recipe_folder, models)
    output = recipe_folder / "confirmation.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; D42 confirmation is scored once")
    setting_d15, frozen_d15 = objective_trial.d15()
    recipe = frozen_d15["recipe"]
    games_a = vendor_trial.split_games("confirm_a")
    reference = scale_trial.setting_halves(dfl, skill, parsed, games_a, "holdout", setting_d15, raw=raw,
                                           cache=d15_holdout_cache)
    stored = json.loads((scale_trial.RECIPE / "confirmation.json").read_text())["selected"]
    full_record = aligned_trial.evaluate(reference, recipe)
    if (vendor_trial.canonical(full_record["external"]) != stored["external"]
            or vendor_trial.canonical(full_record["pff"]) != stored["pff"]):
        raise ValueError("D15 does not reproduce its holdout record; no record was written")
    external = [half for half in reference if half.source != "pff"]
    games_c = vendor_trial.split_games("confirm_c")
    model_d15 = scale_trial.teacher(setting_d15)
    key = {"protocol": PROTOCOL, "scale_models": scale_trial.identity(setting_d15),
           "teacher_sha256": diagnosis.teacher_identity(), "pff": vendor_trial.inputs(parsed, games_c)}
    set_c = diagnosis.collect(cache_root / "d15_set_c", key, lambda: diagnosis.pff_halves(parsed, games_c, model_d15))
    base = aligned_trial.evaluate(external + set_c, recipe)
    candidate = candidate_halves(dfl, skill, parsed, games_c, "holdout", raw=raw, cache=cache_root / "holdout",
                                 models=models)
    trial = aligned_trial.evaluate(candidate, ht.recipe_at(point["owner_cost"]))
    change = refine_trial.changes(base, trial)
    passed = bool(refine_trial.passed(change, gain=GAIN))
    write_new(output, {"protocol": PROTOCOL, "point": point, "recipe": ht.recipe_at(point["owner_cost"]),
                       "reproduction": {"d15_holdout": True}, "reference": base, "trial": trial, "changes": change,
                       "passed": passed, "models": identity(models), "inputs": {"set_c": vendor_trial.inputs(parsed, games_c)},
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(objective_trial.summary_line(f"D42 confirmation cost {point['owner_cost']} (PFF = set C)", trial, change, "",
                                       "passed" if passed else "failed"), flush=True)
    return passed


def build(output, recipe_folder=RECIPE, models=MODELS):
    """The competition file: the confirmed player-set networks and cost in D15's structure, then D27's rule."""
    from dfl_submission import kickoff, objective_trial
    from dfl_submission.build import tactical_halves, write_csv
    if json.loads((recipe_folder / "confirmation.json").read_text())["passed"] is not True:
        raise ValueError("D42 did not pass confirmation; build D27 instead (`kickoff build`)")
    point = frozen(recipe_folder, models)
    _, frozen_d15 = objective_trial.d15()
    model = teacher(point["owner_cost"], models)

    def decode_half(positions, ball, teams, pitch):
        return half_signals(model, positions, ball, teams, pitch)[1]

    labels = tactical_halves(decode_half, centered=True, pitch=frozen_d15["pitch"])
    write_csv(output, kickoff.constrained(labels, kickoff.tag_path(), kickoff.half_blocks()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("d43-select")
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--set", choices=ht.SETS, required=True)
    preparation.add_argument("--source", type=Path, required=True)
    preparation.add_argument("--dfl", type=Path, required=True)
    preparation.add_argument("--dfl-raw", action="store_true")
    preparation.add_argument("--skill", type=Path, required=True)
    preparation.add_argument("--out", type=Path, required=True)
    networks = commands.add_parser("networks")
    networks.add_argument("--source", type=Path, required=True)
    networks.add_argument("--tokens", type=Path, required=True)
    networks.add_argument("--set", choices=ht.SETS, required=True)
    networks.add_argument("--scratch", type=Path, required=True)
    for command_name in ("develop", "confirm"):
        command = commands.add_parser(command_name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--parsed", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        if command_name == "develop":
            command.add_argument("--d15-cache", type=Path, required=True)
        else:
            command.add_argument("--d15-holdout-cache", type=Path, required=True)
    finished = commands.add_parser("build")
    finished.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.out)
        return
    require_external_only()
    if args.command == "d43-select":
        d43_select()
    elif args.command == "prepare":
        prepare(args.set, args.source, args.dfl, args.skill, args.out, raw=args.dfl_raw)
    elif args.command == "networks":
        train_networks(args.source, args.tokens, args.set, args.scratch)
    elif args.command == "develop":
        develop(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_cache=args.d15_cache, cache_root=args.cache)
    else:
        confirm(args.dfl, args.skill, args.parsed, raw=args.dfl_raw, d15_holdout_cache=args.d15_holdout_cache,
                cache_root=args.cache)


if __name__ == "__main__":
    main()
