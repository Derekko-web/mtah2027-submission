"""D55: D43 (cost 10) and the adopted stack (cost 14) on the situation views of DFL WPY. Measurement only; it selects
nothing and changes nothing in the submission (DECISIONS.md D55).

Commands (each runs with `MTAH_DATA_DIR` set to a nonexistent path; `--dfl` is the DFL XML folder, read as
development reads it with `--dfl-raw`):
- `check --dfl DIR --skill DIR --parsed DIR`: (1) the height-aware reader's ball x and y (and status) equal the
  existing reader's on every WPY frame, with the facts about Z; (2) D43 is the model D48's confirmation used and
  the stack is the adopted one (the networks in `stack_models/` equal the adoption copy and the confirmed identity,
  the trees and modules the confirmed ones, `stack_trial.py` the one the passed build lock covers); (3) both
  re-forecast the four standing WPY views (clean, masked, dead carried, team swap) bit for bit against their
  development caches, and their N = 3 spellF1 there equals the development records (D43 at cost 10:
  `development_t.json`'s reference; the stack at cost 14: its cost-14 entry). `--skill` and `--parsed` serve only
  D43's cache identity (file digests). Writes `data/d55/check.json` once. No D55 view is built.
- `measure --dfl DIR`: only after a passed check, once. Builds the five views and the four matched random views,
  computes each half's attacking sign from the half (D50's development form, with its amendment's fallback; a
  first-frame disagreement stops it), forecasts D43 (`player_trial`) and the stack (`stack_trial.Forecaster`, the
  networks in `stack_models/`) on every view half (decoder inputs cached in `data/d55/cache/`), scores them and
  writes `stack_recipe/situation_views.json` once.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from dfl_submission import situation_views as sv
from dfl_submission.external_only import require_external_only
from dfl_submission.pff import digest


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dfl_submission"
RECIPE = PACKAGE / "stack_recipe"
RECORD = RECIPE / "situation_views.json"
SCRATCH = ROOT / "data" / "d55"
CHECK = SCRATCH / "check.json"
CACHE = SCRATCH / "cache"
D42_CACHE = ROOT / "data" / "d42" / "cache"
PROTOCOL = "DECISIONS.md D55"
MATCH = "WPY"
COSTS = {"D43": 10, "stack": 14}
MODELS = ("D43", "stack")
STANDING = ("clean", "masked", "dead_carried", "team_swap")
TARGET_SECONDS = 3.0
GAP_LIMIT = -.02            # the stack's gap on (d) at or below this: a real weakness, D56 trains
DUPLICATE_FALL = .02        # a fall of this much on (e) against the masked view calls for its own rule
LOCK = ("stack_trial.py", RECIPE / "build_identity.json", "stack_trial_sha256")
MODULES = ("situation_trial.py", "situation_views.py", "hidden_views.py", "stack_trial.py", "stack_set.py",
           "player_set.py", "player_set3.py", "direction_tokens.py", "direction_trial.py", "temporal_cpu.py",
           "player_trial.py", "diagnosis.py", "joint_decoder.py", "augment.py", "train.py")


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


# -- reading WPY ---------------------------------------------------------------------------------

def match_files(raw, match=MATCH):
    """(info, positions) of a DFL match, after the source manifest's size and SHA-256 checks (as `train.dfl_halves`)."""
    from dfl_submission import train
    manifest = read(train.SOURCE_MANIFEST)
    name, files = f"DFL-MAT-J03{match}", {}
    for record in manifest["files"]:
        if not record["name"].endswith(f"_{name}.xml"):
            continue
        path = Path(raw) / record["name"]
        if path.stat().st_size != record["size"] or digest(path) != record["sha256"]:
            raise ValueError(f"DFL file identity mismatch: {path}")
        files["info" if "matchinformation" in path.name else "positions"] = path
    return files["info"], files["positions"], name


def read_match(raw):
    """(existing reader's halves, height-aware reader's halves, reader check) of WPY."""
    from dfl_transfer_data import parse_match
    info, positions, name = match_files(raw)
    existing, _ = parse_match(info, positions, name)
    height = sv.parse_ball(info, positions, name)
    return existing, height, sv.reader_check(height, existing)


# -- the models ----------------------------------------------------------------------------------

def stack_checks(recipe=RECIPE, package=PACKAGE, lock=LOCK):
    """(config, cost, checks): the adopted stack, after its identity checks; refuses on any failure."""
    from dfl_submission import fresh_trial as ft, player_trial, scale_trial, stack_set as ss, stack_trial as st
    confirmation_path = recipe / "confirmation.json"
    confirmation, adoption = read(confirmation_path), read(recipe / "adoption_models.json")
    config = dict(confirmation["levers"]["config"])
    ss.validate(config)
    identity = confirmation["identity"]
    checks = {"passed": confirmation["passed"] is True, "cost": confirmation["owner_cost"] == COSTS["stack"],
              "stack_8100": confirmation["stack"] == "8100", "config": identity["config"] == config,
              "adoption": adoption["passed"] is True and adoption["confirmation_sha256"] == digest(confirmation_path),
              "trees": all(digest(scale_trial.TREES[key] / "fit.json") == identity["trees"][key] for key in st.SETS),
              "modules": all(digest(package / module) == identity[key] for module, key in (
                  ("stack_set.py", "stack_set_sha256"), ("player_set.py", "player_set_sha256"),
                  ("direction_tokens.py", "direction_tokens_sha256")))}
    checks["code"], changed = ft.code_check(confirmation, confirmation_path, lock, package)
    failures = ft.network_checks(st.MODELS, adoption, confirmation, offsets=st.OFFSETS, sets=st.SETS,
                                 networks=st.NETWORKS, folder_of=st.stack_model_folder)
    checks["networks"] = not failures
    checks["d43"] = canonical(player_trial.identity()) == confirmation["d43_models"]
    bad = [key for key, value in checks.items() if value is not True] + failures
    if bad:
        raise SystemExit(f"D55 stops: a model differs from its confirmed identity: {bad}")
    return config, confirmation["owner_cost"], dict(checks, code_changed_since_confirmation=changed)


def stack_forecaster(config, table):
    from dfl_submission import stack_trial as st
    sign_map = st.read(st.PARITY)["sign_map"] if config["mirror"] is not None else None
    models = st.stack_models(config, st.MODELS, export=st.stack_model_folder, sign_map=sign_map, names=("stack",))
    return st.Forecaster(models, table, with_d43=False)


# -- scoring -------------------------------------------------------------------------------------

def joined(first, second, filler):
    return np.r_[first, filler, second]


def recipe(model):
    from dfl_submission import hyper_trial as ht
    return ht.recipe_at(COSTS[model])


def match_truth(halves):
    """(raw truth, N = 3 target) of a match from its two halves (an X frame between them)."""
    return truth_of(halves[0].labels, halves[1].labels)


def truth_of(first, second):
    from dfl_submission import diagnosis
    from evaluation import possession_metrics as pm
    truth = joined(np.asarray(first), np.asarray(second), pm.X).astype(np.int8)
    return truth, diagnosis.target(truth, TARGET_SECONDS)


def half_targets(first, second):
    """The N = 3 target of each half, computed over the joined match (as scoring computes it) and split again."""
    _, target = truth_of(first, second)
    return target[:len(first)], target[len(first) + 1:]


def agreement(target, prediction, visible):
    """D46's owner agreement (target and prediction both live, prediction equal to target) on hidden and seen frames."""
    live = np.isin(target, sv.LIVE)
    both = live & np.isin(prediction, sv.LIVE)
    agree = both & (prediction == target)
    result = {}
    for name, frames in (("hidden", ~visible), ("seen", visible)):
        count, hits = int((both & frames).sum()), int((agree & frames).sum())
        result[name] = {"frames": count, "agree": hits, "agreement": hits / count if count else None,
                        "target_live_frames": int((live & frames).sum())}
    return result


def view_scores(halves, balls, model):
    """N = 3 spellF1 (and macro F1, predicted spells) of one view's match, and owner agreement on hidden frames."""
    from dfl_submission import diagnosis
    from evaluation import possession_metrics as pm
    if [half.index for half in halves] != [0, 1]:
        raise ValueError("Expected the two halves of the view in order")
    truth, target = match_truth(halves)
    prediction = joined(diagnosis.decode(halves[0].signals, recipe(model)),
                        diagnosis.decode(halves[1].signals, recipe(model)), pm.X).astype(np.int8)
    seen = [np.isfinite(np.asarray(ball, dtype=np.float64)).all(axis=1) for ball in balls]
    visible = joined(seen[0], seen[1], True)
    scores = diagnosis.score(target, prediction)
    return {"owner_cost": COSTS[model], "spell_f1": float(scores["spell_f1"]), "macro_f1": float(scores["macro_f1"]),
            "predicted_spells": int(scores["spells"]), "agreement": agreement(target, prediction, visible)}


# -- 1-3. the check ------------------------------------------------------------------------------

def same_half(fresh, cached):
    signals, labels = fresh
    return bool(all(np.array_equal(a, b) for a, b in zip(signals, cached.signals, strict=True))
                and np.array_equal(labels, cached.d3))


def check(dfl, skill, parsed, *, output=CHECK):
    """The pipeline checks before any D55 view is built; written once."""
    require_external_only()
    from dfl_submission import augment, diagnosis, diagnosis_d43 as d46, player_trial, stack_trial as st, vendor_trial
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D55's check is written once")
    if RECORD.exists():
        raise FileExistsError(f"{RECORD} exists; D55 is measured once")
    started = time.time()
    # (1) the reader
    existing, height, reader = read_match(dfl)
    reader_passed = sv.reader_passed(reader)
    print(f"D55 reader: ball x and y {'equal' if reader_passed else 'DIFFER'} on every WPY frame; "
          + "; ".join(f"{s}: {p['frames']} frames, Z numeric on every record {p['numeric_z_on_every_record']}, dead "
                      f"frames with Z = 0 {p['dead_frames_z_zero']}/{p['dead_frames']}, live share above 1.5 m "
                      f"{p['live_share_above_air_height']:.3f}" for s, p in reader.items()), flush=True)
    if not reader_passed:
        raise SystemExit("D55 stops: the height-aware reader's ball differs from the existing reader's")
    # (2) the models
    config, cost, model_checks = stack_checks()
    # (3) the standing WPY views, re-forecast and compared with the development caches and records
    games = vendor_trial.development()
    d43_cache = [half for half in d46.cached(D42_CACHE / "development", d46.development_identity(
        dfl, skill, parsed, games, raw=True, models=player_trial.MODELS)) if half.source == "dfl"]
    stack_folder = st.T_SCRATCH / "cache" / "stack" / "development"
    stored = read(stack_folder / "manifest.json")["identity"]
    confirmation = read(RECIPE / "confirmation.json")
    if stored["networks"] != confirmation["identity"]["networks"] or stored["config"] != config:
        raise SystemExit("D55 stops: the stack's development cache is not the confirmed stack's")
    stack_cache = [half for half in st.stored_halves(stack_folder) if half.source == "dfl"]
    teacher = player_trial.teacher()
    forecaster = stack_forecaster(config, st.table_of(st.SCRATCH / "sign_table_development.json"))
    fresh = {"D43": [], "stack": []}
    bits = {"D43": {}, "stack": {}}
    standing = list(augment.guard_halves(dfl, None, "development", raw=True))
    keys = [(half.match, half.view, half.index) for half in standing]
    for name, cached in (("D43", d43_cache), ("stack", stack_cache)):
        if [(half.match, half.view, half.index) for half in cached] != keys:
            raise SystemExit(f"D55 stops: the {name} development cache does not hold the standing WPY views in order")
    for half, d43_half, stack_half in zip(standing, d43_cache, stack_cache, strict=True):
        tick = time.time()
        computed = {"D43": player_trial.half_signals(teacher, half.positions, half.ball, half.teams, half.pitch),
                    "stack": forecaster.forecast(half.positions, half.ball, half.teams, half.pitch)["stack"]}
        for name, cached in (("D43", d43_half), ("stack", stack_half)):
            bits[name][f"{half.view}/{half.index}"] = same_half(computed[name], cached)
            fresh[name].append(diagnosis.HalfSignals(half.source, half.match, half.view, half.index,
                                                     np.asarray(half.labels), *computed[name]))
        print(f"D55 re-forecast WPY {half.view} half {half.index + 1} ({time.time() - tick:.0f} s): D43 "
              f"{'bit for bit' if bits['D43'][f'{half.view}/{half.index}'] else 'DIFFERS'}, stack "
              f"{'bit for bit' if bits['stack'][f'{half.view}/{half.index}'] else 'DIFFERS'}", flush=True)
    development = read(RECIPE / "development_t.json")
    records = {"D43": development["reference"]["groups"],
               "stack": next(item for item in development["entries"]["stack"]
                             if item["owner_cost"] == COSTS["stack"])["groups"]}
    if development["reference"]["owner_cost"] != COSTS["D43"]:
        raise ValueError("development_t.json's reference is not D43 at cost 10")
    balls = {half.view: {} for half in standing}
    for half in standing:
        balls[half.view][half.index] = half.ball
    scores, reproduced = {}, {}
    for name in MODELS:
        scores[name], reproduced[name] = {}, {}
        for view in STANDING:
            halves = [half for half in fresh[name] if half.view == view]
            scores[name][view] = view_scores(halves, [balls[view][0], balls[view][1]], name)
            recorded = records[name][f"dfl_{view}"]
            reproduced[name][view] = {"record": recorded, "recomputed": scores[name][view]["spell_f1"],
                                      "equal": scores[name][view]["spell_f1"] == recorded}
    masked = [half for half in standing if half.view == "masked"]
    passed = bool(reader_passed and all(all(part.values()) for part in bits.values())
                  and all(item["equal"] for part in reproduced.values() for item in part.values()))
    record = {"protocol": PROTOCOL, "passed": passed, "reader": reader, "models": model_checks,
              "stack_config": config, "owner_costs": COSTS,
              "standing": {"bit_for_bit": bits, "reproduced": reproduced, "scores": scores,
                           "masked_statistics": sv.run_statistics([half.ball for half in masked],
                                                                  [half.labels for half in masked])},
              "caches": {"d43_development": digest(D42_CACHE / "development" / "manifest.json"),
                         "stack_development": digest(stack_folder / "manifest.json"),
                         "stack_signs": digest(st.SCRATCH / "sign_table_development.json")},
              "code": code_identity(), "confirmation_sha256": digest(RECIPE / "confirmation.json"),
              "development_t_sha256": digest(RECIPE / "development_t.json"),
              "seconds": round(time.time() - started), "checked_at": time.strftime("%Y-%m-%d %H:%M %Z")}
    write_new(output, record)
    print(f"D55 check {'PASSED' if passed else 'FAILED'}: reproduced "
          + ", ".join(f"{name} {view} {item['recomputed']:.5f}{'' if item['equal'] else ' (record ' + str(item['record']) + ')'}"
                      for name, part in reproduced.items() for view, item in part.items()), flush=True)
    return passed


def checked(path=CHECK):
    """The passed check record, with the code and records it ran on unchanged."""
    if not Path(path).exists():
        raise SystemExit("D55 stops: run `check` first")
    record = read(path)
    if record["passed"] is not True:
        raise SystemExit("D55 stops: the check did not pass")
    if (record["code"] != code_identity() or record["confirmation_sha256"] != digest(RECIPE / "confirmation.json")
            or record["development_t_sha256"] != digest(RECIPE / "development_t.json")):
        raise SystemExit("D55 stops: the code or records changed since the check")
    return record


# -- 4. the views --------------------------------------------------------------------------------

def view_name(view, random=False):
    return f"{view}_{sv.RANDOM}" if random else view


def build_views(existing, height):
    """{name: {index: (labels, positions, ball, teams, pitch)}} of the nine views, and their construction notes."""
    from dfl_submission import train
    from dfl_transfer_data import SECTIONS
    from evaluation import possession_metrics as pm
    index = (*train.TRAIN, train.DEVELOPMENT, *train.HOLDOUT).index(MATCH)
    labels = [pm.label_codes(existing[section]["labels"]) for section in SECTIONS]
    targets = half_targets(labels[0], labels[1])
    views, notes = {}, {"changes": {}, "situation_frames": {}, "duplicates": {}, "seeds": {}}
    for i, section in enumerate(SECTIONS):
        item = existing[section]
        positions = np.asarray(item["positions"], dtype=np.float32)
        ball = np.asarray(item["ball"], dtype=np.float32)
        teams, pitch = item["teams"], item["pitch"]
        masks = sv.situation_masks(positions, ball, height[section]["z"], targets[i])
        notes["changes"][i] = int(len(sv.ownership_changes(targets[i])))
        notes["situation_frames"][i] = {name: int(mask.sum()) for name, mask in masks.items()}
        for view in sv.SITUATIONS:
            seed = sv.view_seed(view, index, i)
            (vp, vb), (rp, rb), placed = sv.situation_pair(positions, ball, masks[view], seed)
            if int(placed.sum()) != int(masks[view].sum()):
                raise ValueError("A matched random view hides a different number of frames")
            views.setdefault(view, {})[i] = (labels[i], vp, vb, teams, pitch)
            views.setdefault(view_name(view, True), {})[i] = (labels[i], rp, rb, teams, pitch)
            notes["seeds"][f"{view}/{i}"] = {"view": seed, "placement": seed + sv.PLACEMENT}
        masked_positions, masked_ball = dict((name, (p, b)) for name, p, b in train.views(item, index, i))["masked"]
        seed = sv.view_seed("duplicates", index, i)
        dup_positions, dup_teams, runs = sv.add_duplicates(masked_positions, positions, ball, teams, seed)
        views.setdefault("duplicates", {})[i] = (labels[i], dup_positions, masked_ball, dup_teams, pitch)
        covered = sum(run["length"] for run in runs)
        notes["duplicates"][i] = {"runs": len(runs), "frames": covered, "share": covered / len(masked_ball),
                                  "runs_without_player": sum(run["player"] is None for run in runs),
                                  "largest_offset_m": max(run.get("offset_m", 0) for run in runs)}
        notes["seeds"][f"duplicates/{i}"] = {"duplicates": seed, "masked_view": sv.MASKED_SEED + 100 * index + i}
    return views, notes


def view_items(views):
    for name, halves in views.items():
        for i, (_, positions, ball, teams, pitch) in sorted(halves.items()):
            yield name, ("dfl", MATCH, name, i), positions, ball, teams, pitch


def sign_summary(table):
    rows = {row["half"]: row for row in table.values()}
    by_view = {}
    for half, row in rows.items():
        by_view.setdefault(half.rsplit("/", 1)[0], []).append(row["sign_used"])
    return {"halves": len(rows),
            "same_sign_views": sorted(view for view, signs in by_view.items() if len(signs) == 2 and len(set(signs)) == 1),
            "first_frame_disagrees": sorted(half for half, row in rows.items()
                                            if row["sign"] is not None and not row["agrees"]),
            "fallbacks": sorted(half for half, row in rows.items() if row["fallback"] is not None),
            "signs": {half: row["sign_used"] for half, row in sorted(rows.items())},
            "smallest_abs_d_m": min(abs(row["d"]) for row in rows.values() if row["d"] is not None)}


def statistics(views):
    return {name: sv.run_statistics([halves[i][2] for i in (0, 1)], [halves[i][0] for i in (0, 1)])
            for name, halves in views.items()}


# -- 5. forecasts, scores, the record ------------------------------------------------------------

def forecast_identity(model, check_record, notes, config=None):
    from dfl_submission import player_trial, stack_trial as st
    common = {"protocol": PROTOCOL, "match": MATCH, "check_sha256": digest(CHECK), "code": code_identity(),
              "constants": constants(), "seeds": notes["seeds"]}
    if model == "D43":
        return canonical(dict(common, model="D43", models=player_trial.identity()))
    return canonical(dict(common, model="stack", config=config, confirmation_sha256=digest(RECIPE / "confirmation.json"),
                          adoption_sha256=digest(RECIPE / "adoption_models.json"), sign_map_sha256=digest(st.PARITY)))


def constants():
    return {"air_height_m": sv.AIR_HEIGHT, "crowd_players": sv.CROWD_PLAYERS, "crowd_radius_m": sv.CROWD_RADIUS,
            "change_frames": [-sv.CHANGE_BEFORE, sv.CHANGE_AFTER], "duplicate_run_frames": list(sv.DUPLICATE_RUN),
            "duplicate_share": sv.DUPLICATE_SHARE, "duplicate_radius_m": sv.DUPLICATE_RADIUS,
            "noise_m": 0.2, "dropout": [0.2, 50], "seeds": sv.SEEDS, "placement_offset": sv.PLACEMENT}


def forecast_views(views, model, produce_one, cache, identity):
    from dfl_submission import diagnosis

    def produce():
        for name, halves in views.items():
            for i, (labels, positions, ball, teams, pitch) in sorted(halves.items()):
                tick = time.time()
                signals, decoded = produce_one(positions, ball, teams, pitch)
                print(f"D55 forecast {model} {name} half {i + 1} ({time.time() - tick:.0f} s)", flush=True)
                yield diagnosis.HalfSignals("dfl", MATCH, name, i, np.asarray(labels), signals, decoded)

    return diagnosis.collect(cache, identity, produce)


def reading(results, check_record):
    """D55's reading: the stack's gap on (d) against −0.02; the fall on (e) against the masked view (both models)."""
    gaps = {model: {view: results[model][view]["spell_f1"] - results[model][view_name(view, True)]["spell_f1"]
                    for view in sv.SITUATIONS} for model in MODELS}
    masked = {model: check_record["standing"]["scores"][model]["masked"]["spell_f1"] for model in MODELS}
    duplicates = {model: results[model]["duplicates"]["spell_f1"] - masked[model] for model in MODELS}
    weakness = bool(gaps["stack"]["air_or_crowd"] <= GAP_LIMIT)
    own_rule = bool(duplicates["stack"] <= -DUPLICATE_FALL)
    return {"gaps": gaps, "duplicates_minus_masked": duplicates, "masked_view": masked,
            "stack_gap_air_or_crowd": gaps["stack"]["air_or_crowd"], "gap_limit": GAP_LIMIT,
            "real_weakness": weakness, "duplicate_fall_limit": DUPLICATE_FALL, "duplicates_call_for_own_rule": own_rule,
            "conclusion": ("hiding by situation is a real weakness: D56 trains with such views, confirmed on fresh PFF "
                           "games" if weakness else "no training follows from D55")
                          + ("; the fall on (e) calls for its own rule" if own_rule else "")}


def measure(dfl, *, output=RECORD, cache=CACHE):
    """D55's measurement, once."""
    require_external_only()
    from dfl_submission import player_trial, stack_trial as st
    if Path(output).exists():
        raise FileExistsError(f"{output} exists; D55 is measured once")
    check_record = checked()
    started = time.time()
    config, cost, model_checks = stack_checks()
    if canonical(model_checks) != check_record["models"]:
        raise SystemExit("D55 stops: the model checks differ from the passed check")
    existing, height, reader = read_match(dfl)
    if canonical(reader) != check_record["reader"]:
        raise SystemExit("D55 stops: the reader check differs from the passed one")
    views, notes = build_views(existing, height)
    table = st.sign_table(view_items(views))
    signs = sign_summary(table)
    print(f"D55 signs: {signs['halves']} halves, first-frame disagreements {signs['first_frame_disagrees']}, fallbacks "
          f"{signs['fallbacks']}", flush=True)
    if signs["first_frame_disagrees"]:
        raise SystemExit(f"D55 stops: D50's first-frame check fails on {signs['first_frame_disagrees']}")
    teacher = player_trial.teacher()
    forecaster = stack_forecaster(config, table)
    halves = {
        "D43": forecast_views(views, "D43", lambda *t: player_trial.half_signals(teacher, *t), cache / "d43",
                              forecast_identity("D43", check_record, notes)),
        "stack": forecast_views(views, "stack", lambda *t: forecaster.forecast(*t)["stack"], cache / "stack",
                                forecast_identity("stack", check_record, notes, config))}
    stats = statistics(views)
    results = {model: {name: view_scores([half for half in halves[model] if half.view == name],
                                         [views[name][0][2], views[name][1][2]], model) for name in views}
               for model in MODELS}
    read_out = reading(results, check_record)
    record = canonical({
        "protocol": PROTOCOL, "match": MATCH, "owner_costs": COSTS, "constants": constants(),
        "views": {"statistics": stats, "masked_statistics": check_record["standing"]["masked_statistics"],
                  "notes": notes},
        "scores": results, "standing_scores": check_record["standing"]["scores"], "reading": read_out,
        "signs": signs, "checks": {"reader": reader, "models": model_checks,
                                   "standing_bit_for_bit": check_record["standing"]["bit_for_bit"],
                                   "standing_reproduced": check_record["standing"]["reproduced"]},
        "caches": {model: digest(cache / model.lower() / "manifest.json") for model in MODELS},
        "check_sha256": digest(CHECK), "code": code_identity(), "stack_config": config,
        "confirmation_sha256": digest(RECIPE / "confirmation.json"), "seconds": round(time.time() - started),
        "scored_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    write_new(output, record)
    print(summary(record), flush=True)
    return record


def summary(record):
    lines = ["D55 situation views of WPY (N = 3 spellF1; D43 at cost 10, the stack at 14):"]
    stats, scores, gaps = record["views"]["statistics"], record["scores"], record["reading"]["gaps"]
    for view in (*sv.SITUATIONS, "duplicates"):
        item = stats[view]
        line = (f"  {view}: visibility {item['visibility']:.3f} (live {item['live_visibility']:.3f}), runs "
                f"{item['hidden_runs']}, mean {item['mean_hidden_run_s']:.2f} s, over 3 s "
                f"{item['hidden_share_in_runs_over_3s']:.3f}, over 10 s {item['hidden_share_in_runs_over_10s']:.3f}")
        for model in MODELS:
            line += f"; {model} {scores[model][view]['spell_f1']:.4f}"
            if view in sv.SITUATIONS:
                random = scores[model][view_name(view, True)]["spell_f1"]
                line += f" vs random {random:.4f} (gap {gaps[model][view]:+.4f})"
        lines.append(line)
    lines.append(f"  reading: {record['reading']['conclusion']}")
    return "\n".join(lines)


# -- command line ----------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("check", help="the pipeline checks, written once (no D55 view is built)")
    one.add_argument("--dfl", type=Path, required=True)
    one.add_argument("--skill", type=Path, required=True)
    one.add_argument("--parsed", type=Path, required=True)
    two = sub.add_parser("measure", help="the views, forecasts and scores, written once")
    two.add_argument("--dfl", type=Path, required=True)
    sub.add_parser("summary", help="print the written record")
    args = parser.parse_args()
    if args.action == "check":
        check(args.dfl, args.skill, args.parsed)
    elif args.action == "measure":
        measure(args.dfl)
    else:
        print(summary(read(RECORD)))


if __name__ == "__main__":
    main()
