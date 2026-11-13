"""D44: player-set variants by cross-validation (DECISIONS.md D44), Stage A.

`tokens` writes the original set's training tokens for a variant's token options, `stage-a` trains one
network per fold for the anchor and each variant, decodes each alone on its held-out training halves
(D22 Stage A's method, with `player_set`'s NumPy inference) and selects a point.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.special import expit

from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "player2_recipe"
PROTOCOL = "DECISIONS.md D44"
ANCHOR = {"per_team": 11, "hidden": 32, "relative": False, "lr_scale": 2., "batch": 32, "dropout": .2}
VARIANTS = (("V1", {"per_team": 16}), ("V2", {"hidden": 64}), ("V3", {"relative": True}),
            ("V4", {"lr_scale": 1., "batch": 16, "dropout": .08}))
COSTS = (8, 10, 12)
STAGE_A_GAIN = .005
D42_TOKENS = ROOT / "data" / "d42" / "tokens_current"


def write_new(path, record):
    if path.exists():
        raise FileExistsError(f"{path} exists; D44 records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def settings():
    return [("anchor", dict(ANCHOR))] + [(name, dict(ANCHOR, **change)) for name, change in VARIANTS]


def token_config(setting):
    return {key: setting[key] for key in ("per_team", "hidden", "relative")}


def token_folder(setting):
    """D42's tokens serve every setting with D42's token options; others get their own folder."""
    if (setting["per_team"], setting["relative"]) == (11, False):
        return D42_TOKENS
    return ROOT / "data" / "d44" / f"tokens_current_p{setting['per_team']}{'_rel' if setting['relative'] else ''}"


def prepare_tokens(setting, source, dfl, skill, *, raw):
    """The original set's training tokens with a setting's token options (same checks as D42's prepare)."""
    from dfl_submission import aligned_trial, long_context, player_set2, touch_trial
    output = token_folder(setting)
    if output == D42_TOKENS:
        return output
    if (output / "manifest.json").exists():
        return output
    if digest(source / "manifest.json") != touch_trial.published_manifest("current"):
        raise ValueError(f"{source} is not the original set's prepared-sequence folder")
    manifest = json.loads((source / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    halves = touch_trial.training_halves("current", dfl, skill, raw=raw)
    for (record, path), half in zip(aligned_trial.verified_records(source), halves, strict=True):
        if record["file"] != f"{half.key}.npz":
            raise ValueError(f"Training half {half.key} does not match the prepared record {record['file']}")
        frames, table, _ = long_context.features(half.positions, half.ball, half.teams, half.pitch)
        with np.load(path, allow_pickle=False) as item:
            values = item["values"]
        if table.columns.tolist() != manifest["columns"] or len(frames) != len(values) or not np.array_equal(
                table.to_numpy(dtype=np.float32), values, equal_nan=True):
            raise ValueError(f"Re-extracted features differ from the stored prepared sequence: {record['file']}")
        np.savez_compressed(output / record["file"], tokens=player_set2.tokens(
            half.positions, half.ball, half.teams, half.pitch, frames, token_config(setting)).astype(np.float16))
        records[record["file"]] = digest(output / record["file"])
        print(f"Tokens {output.name} {record['file']}", flush=True)
    (output / "manifest.json").write_text(json.dumps(
        {"protocol": PROTOCOL, "base_manifest": digest(source / "manifest.json"), "records": records,
         "config": token_config(setting), "player_set2_sha256": digest(PACKAGE / "player_set2.py")},
        indent=2, sort_keys=True) + "\n")
    return output


def train_command(source, tokens, output, setting, *, seed, boost):
    command = [sys.executable, "-m", "dfl_submission.player_set2", "--source", str(source), "--tokens", str(tokens),
               "--out", str(output), "--seed", str(seed), "--boundary-boost", str(boost),
               "--per-team", str(setting["per_team"]), "--hidden", str(setting["hidden"]),
               "--lr-scale", str(setting["lr_scale"]), "--batch", str(setting["batch"]), "--dropout", str(setting["dropout"])]
    return command + (["--relative"] if setting["relative"] else [])


def train_folds(name, setting, source, cv_root, scratch):
    from dfl_submission import zoo
    seed, boost = zoo.CV_NETWORK
    runs = []
    for fold in range(zoo.FOLDS):
        folder = zoo.fold_folder(source, fold, cv_root / "folds" / str(fold))
        run = scratch / name / f"fold{fold}"
        if not (run / "fit.json").exists():
            if run.exists():
                shutil.rmtree(run)
            subprocess.run(train_command(folder, token_folder(setting), run, setting, seed=seed, boost=boost),
                           check=True, cwd=ROOT)
        runs.append(run)
    return runs


def fold_halves(run, source, fold, labels, tokens):
    """A fold network's decoder inputs on its held-out training halves (`zoo.fold_halves` for player sets)."""
    from dfl_submission import diagnosis, player_set, zoo
    from dfl_submission.temporal_cpu import normalize
    from dfl_submission.temporal_network import targets as prepared_targets
    weights, (location, scale, permutation), _ = __import__("dfl_submission.player_set2", fromlist=["x"]).final_weights(run)
    token_manifest = json.loads((tokens / "manifest.json").read_text())
    halves = []
    for record, values, stored in zoo.held_out(source, fold):
        name, match, view, index = Path(record["file"]).stem.split("_")
        truth = np.asarray(labels[name, match, int(index)], dtype=np.int8)
        frames = np.arange(0, len(truth), zoo.FEATURE_STRIDE)
        if len(frames) != len(values) or not np.array_equal(prepared_targets(truth[frames]), stored):
            raise ValueError(f"Reloaded labels do not reproduce the prepared targets of {record['file']}")
        path = tokens / record["file"]
        if digest(path) != token_manifest["records"][record["file"]]:
            raise ValueError(f"Token file hash differs: {path}")
        with np.load(path, allow_pickle=False) as item:
            token_values, mask = player_set.token_inputs(item["tokens"].astype(np.float32))
        variants = np.stack([normalize(values, location, scale), normalize(values[:, permutation], location, scale)])
        p = expit(player_set.forward(variants, np.stack([token_values, token_values[:, ::-1]]),
                                     np.stack([mask, mask[:, ::-1]]), weights))
        probabilities = ((p[0, :, 1] + 1 - p[1, :, 1]) / 2, (p[0, :, 0] + p[1, :, 0]) / 2,
                         (p[0, :, 2] + p[1, :, 2]) / 2, (p[0, :, 3] + p[1, :, 3]) / 2)
        if view in zoo.SCORED_VIEWS:
            halves.append(diagnosis.HalfSignals(name, match, view, int(index), truth,
                                                zoo.fold_signals(probabilities, len(truth)), None))
    return halves


def primary(trial, metric="spell_f1"):
    return trial["standard"]["aba3"][metric]


def stage_a(dfl, skill, source, cv_root, scratch, *, raw, output=RECIPE / "stage_a.json"):
    require_external_only()
    from dfl_submission import aligned_trial, triple_trial, zoo
    if output.exists():
        raise FileExistsError(f"{output} exists; D44 Stage A is scored once")
    zoo.check_original_source(source)
    labels = aligned_trial.training_labels(dfl, skill, raw=raw)
    recipe = triple_trial.d15_recipe()
    entries = []
    for name, setting in settings():
        runs = train_folds(name, setting, source, cv_root, scratch)
        halves = []
        for fold, run in enumerate(runs):
            halves += fold_halves(run, source, fold, labels, token_folder(setting))
        trials = {cost: triple_trial.score_point(halves, dict(recipe, owner_cost=cost)) for cost in COSTS}
        entries.append({"name": name, "setting": setting, "trials": {str(c): t for c, t in trials.items()},
                        "runs": {str(i): digest(run / "fit.json") for i, run in enumerate(runs)}})
        print(f"D44 Stage A {name} {setting}: " + ", ".join(f"cost {c} {primary(t):.5f}" for c, t in trials.items()),
              flush=True)
    points = [(entry, int(cost), trial) for entry in entries for cost, trial in sorted(entry["trials"].items(),
                                                                                       key=lambda item: int(item[0]))]
    _, (entry, cost, trial) = max(enumerate(points), key=lambda item: (primary(item[1][2]), primary(item[1][2], "macro_f1"),
                                                                       -item[0]))
    anchor_best = max(primary(t) for t in entries[0]["trials"].values())
    selected = {"name": entry["name"], "setting": entry["setting"], "owner_cost": cost, "primary": primary(trial),
                "gain_over_anchor_best": primary(trial) - anchor_best}
    proceeds = entry["name"] != "anchor" and selected["gain_over_anchor_best"] >= STAGE_A_GAIN
    write_new(output, {"protocol": PROTOCOL, "stage": "A", "entries": entries, "anchor_best": anchor_best,
                       "selected": selected, "proceeds": proceeds, "recipe": recipe,
                       "source_manifest": digest(source / "manifest.json"),
                       "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    print(f"D44 Stage A selected {entry['name']} at cost {cost}: {selected['primary']:.5f} "
          f"({selected['gain_over_anchor_best']:+.5f} over the anchor's best) -> "
          f"{'Stage B' if proceeds else 'stop; D43 stays'}", flush=True)
    return proceeds



def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for command_name in ("tokens", "stage-a"):
        command = commands.add_parser(command_name)
        command.add_argument("--dfl", type=Path, required=True)
        command.add_argument("--dfl-raw", action="store_true")
        command.add_argument("--skill", type=Path, required=True)
        command.add_argument("--source", type=Path, required=True)
        if command_name == "stage-a":
            command.add_argument("--cv", type=Path, required=True)
            command.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    require_external_only()
    if args.command == "tokens":
        for _, setting in settings():
            prepare_tokens(setting, args.source, args.dfl, args.skill, raw=args.dfl_raw)
    else:
        stage_a(args.dfl, args.skill, args.source, args.cv, args.scratch, raw=args.dfl_raw)


if __name__ == "__main__":
    main()
