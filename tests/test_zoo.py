"""D20 model search (DECISIONS.md D20) on synthetic data: folds, network families, CV and ensembles."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from dfl_submission import zoo
from dfl_submission.pff import digest

ROOT = Path(__file__).resolve().parents[1]
DFL = ("WOY", "WMX", "WOH", "WN1")
SKILL = tuple(str(1000000 + 7 * k) for k in range(12))[::-1]


def isolated(name):
    """Run a torch check in a fresh process; torch and XGBoost must not share a test process."""
    code = f"import sys; sys.path[:0] = ['.', 'experiments']; from tests.test_zoo import {name}; {name}()"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MTAH_DATA_DIR="/nonexistent/mtah"))
    assert result.returncode == 0, result.stderr[-3000:]


def targets(labels):
    """`temporal_network.targets`, restated so that torch never loads in the shared test process."""
    from evaluation import possession_metrics as pm
    live, known = labels < pm.D, labels != pm.X
    dead_edge = np.r_[False, ((labels[1:] == pm.D) != (labels[:-1] == pm.D)) & known[:-1] & known[1:]]
    owner_edge = np.r_[False, (labels[1:] != labels[:-1]) & live[:-1] & live[1:]]
    for edge in (dead_edge, owner_edge):
        edge[:] = edge | np.r_[edge[1:], False] | np.r_[False, edge[:-1]]
    return np.column_stack([labels == pm.D, labels == pm.H, dead_edge, owner_edge, known, live]).astype(np.float32)


def synthetic_folder(folder, length=330, seed=0):
    """A prepared-sequence folder with 4 DFL matches (2 views x 2 halves) and 12 SkillCorner matches."""
    from evaluation import possession_metrics as pm
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    columns = ["h__near", "a__near", "speed"]
    records = []
    matches = [("dfl", match, view) for match in DFL for view in ("clean", "masked")]
    matches += [("skillcorner", match, "observed") for match in SKILL]
    for source, match, view in matches:
        for half in (0, 1):
            labels = np.where(np.arange(length) % 90 < 45, pm.H, pm.A).astype(np.int8)
            labels[200:220] = pm.D
            values = rng.normal(size=(length, 3)).astype(np.float32)
            values[:, 0] += (labels == pm.H) * 1.5
            values[:, 1] += (labels == pm.A) * 1.5
            values[50:60, 2] = np.nan
            filename = f"{source}_{match}_{view}_{half}.npz"
            np.savez_compressed(folder / filename, values=values, targets=targets(labels))
            records.append({"file": filename, "source": source, "match": match, "sha256": digest(folder / filename)})
    (folder / "manifest.json").write_text(json.dumps({"columns": columns, "records": records}))
    return folder


def test_folds_hold_one_dfl_and_three_skillcorner_matches_each():
    with tempfile.TemporaryDirectory() as tmp:
        source = synthetic_folder(Path(tmp) / "prepared")
        folds = zoo.fold_of(json.loads((source / "manifest.json").read_text()))
        assert [folds["dfl", match] for match in sorted(DFL)] == [0, 1, 2, 3]
        ordered = sorted(SKILL, key=int)
        assert [folds["skillcorner", match] for match in ordered] == [0, 1, 2, 3] * 3
        for k in range(zoo.FOLDS):
            assert sum(1 for f in folds.values() if f == k) == 4
        table = zoo.fold_table(source)
        assert table["0"] == sorted(["dfl:WMX", f"skillcorner:{ordered[0]}", f"skillcorner:{ordered[4]}",
                                     f"skillcorner:{ordered[8]}"])


def test_fold_folder_keeps_the_other_folds_unchanged_and_is_reused():
    with tempfile.TemporaryDirectory() as tmp:
        source = synthetic_folder(Path(tmp) / "prepared")
        output = zoo.fold_folder(source, 2, Path(tmp) / "fold2")
        manifest = json.loads((output / "manifest.json").read_text())
        held = {record["file"] for record, _, _ in zoo.held_out(source, 2)}
        kept = {record["file"] for record in manifest["records"]}
        everything = {record["file"] for record in json.loads((source / "manifest.json").read_text())["records"]}
        assert kept | held == everything and not kept & held and len(held) == 4 + 6
        assert all(digest(output / record["file"]) == record["sha256"] for record in manifest["records"])
        assert manifest["base_manifest"] == digest(source / "manifest.json") and manifest["fold"] == 2
        before = digest(output / "manifest.json")
        assert zoo.fold_folder(source, 2, output) == output and digest(output / "manifest.json") == before
        try:
            zoo.fold_folder(source, 1, output)
        except ValueError:
            pass
        else:
            raise AssertionError("a folder of another fold must be refused")


def test_windows_cover_every_step_once_with_halo_context():
    values = np.arange(300, dtype=np.float32)[:, None] + 1
    blocks = zoo.windows(values)
    assert blocks.shape == (3, zoo.WINDOW, 1)
    assert np.array_equal(zoo.unwindow(blocks, 300), values)
    assert np.all(blocks[0, :zoo.HALO] == 0) and blocks[1, 0, 0] == zoo.CORE - zoo.HALO + 1
    assert np.all(blocks[-1, zoo.HALO + 300 - 2 * zoo.CORE:] == 0)
    short = zoo.windows(values[:5])
    assert short.shape == (1, zoo.WINDOW, 1) and np.array_equal(zoo.unwindow(short, 5), values[:5])


def test_log_loss_summary_weighs_sources_equally():
    assert abs(zoo.log_loss([.5, .5], [1, 0]) - np.log(2)) < 1e-12
    assert zoo.log_loss([1.0], [1]) < 1e-5 and zoo.log_loss([0.0], [1]) < 14
    assert abs(zoo.log_loss([.9, .1, .5], [1, 0, 1], mask=np.array([True, True, False])) + np.log(.9)) < 1e-12
    per_match = {("dfl", "A"): {"ownership": 1.0, "stoppage": .5}, ("skillcorner", "B"): {"ownership": .2, "stoppage": .1},
                 ("skillcorner", "C"): {"ownership": .4, "stoppage": .3}}
    summary = zoo.summarise_matches(per_match)
    assert abs(summary["ownership"] - (1.0 + .3) / 2) < 1e-12 and abs(summary["stoppage"] - (.5 + .2) / 2) < 1e-12
    assert abs(summary["score"] - (summary["ownership"] + summary["stoppage"])) < 1e-12
    entries = [{"setting": {"a": 1}, "summary": {"score": .3}}, {"setting": {"a": 2}, "summary": {"score": .2}},
               {"setting": {"a": 3}, "summary": {"score": .2}}]
    assert zoo.select_cv(entries) == {"a": 2}


def check_grids_and_schedules():
    from dfl_submission import temporal_network
    labels = np.array([0, 0, 1, 2, 2, 1, 3, 0], dtype=np.int8)
    assert np.array_equal(targets(labels), temporal_network.targets(labels))
    assert len(zoo.GRIDS["tcn"]) == 6 and zoo.ANCHORS["tcn"] in zoo.GRIDS["tcn"]
    assert len(zoo.GRIDS["gru"]) == len(zoo.GRIDS["transformer"]) == 4 and len(zoo.GRIDS["trees"]) == 9
    assert zoo.ANCHORS["trees"] in zoo.GRIDS["trees"]
    for grid in zoo.GRIDS.values():
        for setting in grid:
            if "steps" in setting:
                assert temporal_network.snapshots(setting["steps"])[-1] == setting["steps"]
    assert zoo.learning_rate("gru", 1, 5400) == temporal_network.learning_rate(1, 5400)
    assert zoo.learning_rate("transformer", 150, 5400) == temporal_network.learning_rate(150, 5400) * .5
    assert zoo.learning_rate("transformer", 300, 5400) == temporal_network.learning_rate(300, 5400)


def test_grids_and_schedules_are_the_rules():
    isolated("check_grids_and_schedules")


def check_numpy_matches_torch():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(3)
    modules = zoo.torch_modules()
    rng = np.random.default_rng(4)
    for family, width in (("gru", 8), ("transformer", 16)):
        net = modules[family](3, width).eval()
        weights = {key: value.numpy() for key, value in net.state_dict().items()}
        values = rng.normal(size=(2, zoo.WINDOW, 6)).astype(np.float32)
        with torch.inference_mode():
            reference = net(torch.from_numpy(values)).numpy()
        np.testing.assert_allclose(zoo.FORWARD[family](values, weights), reference, atol=2e-5, rtol=1e-5)
        sequence = rng.normal(size=(2, 333, 6)).astype(np.float32)
        with torch.inference_mode():
            blocked = np.stack([zoo.unwindow(net(torch.from_numpy(zoo.windows(v))).numpy(), 333) for v in sequence])
        np.testing.assert_allclose(zoo.windowed_logits(family, sequence, weights), blocked, atol=2e-5, rtol=1e-5)


def test_numpy_networks_match_torch():
    isolated("check_numpy_matches_torch")


def check_train_export_and_cv_losses():
    from dfl_submission import temporal_cpu
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = synthetic_folder(tmp / "prepared")
        folder = zoo.fold_folder(source, 1, tmp / "fold1")
        for family in ("gru", "transformer", "tcn"):
            setting = {"width": 8 if family != "transformer" else 16, "steps": 6}
            run = tmp / family
            zoo.fit_network(family, folder, run, seed=42, boundary_boost=.5, setting=setting, device="cpu")
            fit = json.loads((run / "fit.json").read_text())
            assert fit["train_manifest"] == digest(folder / "manifest.json") and fit["seed"] == 42
            losses = zoo.network_cv_losses(family, run, source, 1)
            assert set(losses) == {("dfl", "WN1")} | {("skillcorner", m) for m in sorted(SKILL, key=int)[1::4]}
            assert all(np.isfinite(list(value.values())).all() for value in losses.values())
            exported = tmp / f"{family}_cpu"
            zoo.export(family, run, exported)
            checkpoint = zoo.load_checkpoint(exported, "temporal_6.npz")
            assert isinstance(checkpoint, zoo.Checkpoint) == (family != "tcn")
            with np.load(source / "dfl_WN1_clean_0.npz") as item:
                values = item["values"]
            frames = np.arange(0, 5 * len(values), 5)
            features = frames, values, ["h__near", "a__near", "speed"], 5 * len(values)
            signals = zoo.predict(checkpoint, features)
            assert len(signals) == 4 and all(s.shape == (5 * len(values),) for s in signals)
            family_, weights, norm = zoo.final_weights(run)
            at_frames = zoo.network_probabilities(family_, weights, norm, values)
            assert np.allclose(signals[0][frames], at_frames[0], atol=1e-5)
            if family == "tcn":
                assert np.array_equal(signals[0], temporal_cpu.predict(checkpoint, features)[0])


def test_training_export_and_cv_losses_on_synthetic_folds():
    isolated("check_train_export_and_cv_losses")


def synthetic_rows(seed, keys=("dfl:A", "dfl:B", "skillcorner:1", "skillcorner:2"), length=200):
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for key in keys:
        frame = pd.DataFrame({"h__near": rng.normal(size=length), "a__near": rng.normal(size=length),
                              "speed": rng.normal(size=length)}, dtype=np.float32)
        target = (frame["h__near"] - frame["a__near"] + rng.normal(0, .5, length) > 0).to_numpy()
        rows.append((frame, target, np.full(length, key)))
    return rows


def test_tree_fit_equals_the_published_fit_at_d3s_parameters():
    from dfl_submission import long_context
    rows = synthetic_rows(1)
    with tempfile.TemporaryDirectory() as tmp:
        for ownership in (True, False):
            ours, _ = zoo.fit_head(rows, ownership=ownership, params=zoo.tree_params(zoo.ANCHORS["trees"]))
            theirs, _ = long_context.fit_head(rows, ownership=ownership)
            ours.save_model(Path(tmp) / "ours.ubj")
            theirs.save_model(Path(tmp) / "theirs.ubj")
            assert digest(Path(tmp) / "ours.ubj") == digest(Path(tmp) / "theirs.ubj")


def test_a_longer_tree_run_contains_the_shorter_one():
    import pandas as pd
    rows = synthetic_rows(2)
    inputs = pd.concat([row[0] for row in rows], ignore_index=True)
    long_run, _ = zoo.fit_head(rows, ownership=True, params=zoo.tree_params({"max_depth": 3, "n_estimators": 30}))
    short_run, _ = zoo.fit_head(rows, ownership=True, params=zoo.tree_params({"max_depth": 3, "n_estimators": 10}))
    assert np.array_equal(zoo.head_probability(long_run, inputs, True, 10), zoo.head_probability(short_run, inputs, True, 10))
    from dfl_transfer_features import swap
    p = zoo.head_probability(long_run, inputs, True, 30)
    assert np.allclose(p, 1 - zoo.head_probability(long_run, swap(inputs), True, 30), atol=1e-6)


def test_tree_fold_losses_score_held_out_matches_only():
    rows = synthetic_rows(3, keys=("dfl:A", "dfl:B", "skillcorner:1", "skillcorner:2", "skillcorner:3"))
    folds = {"dfl:A": 0, "dfl:B": 1, "skillcorner:1": 0, "skillcorner:2": 1, "skillcorner:3": 1}
    with tempfile.TemporaryDirectory() as tmp:
        losses = zoo.tree_fold_losses(rows, True, folds, 0, Path(tmp), (2,), (5, 10), "ownership")
        assert set(losses) == {(2, 5), (2, 10)} and all(0 < value < np.log(2) + .2 for value in losses.values())
        again = zoo.tree_fold_losses(rows, True, folds, 0, Path(tmp), (2,), (5, 10), "ownership")
        assert again == losses


def test_mean_signals_keep_one_member_and_average_in_order():
    a = tuple(np.full(4, v, dtype=np.float32) for v in (.2, .4, .6, .8))
    b = tuple(np.full(4, v, dtype=np.float32) for v in (.4, .2, .2, .0))
    assert all(x is y or np.array_equal(x, y) for x, y in zip(zoo.mean_signals([a]), a))
    mean = zoo.mean_signals([a, b])
    assert all(m.dtype == np.float32 for m in mean)
    assert np.allclose(mean[0], .3) and np.allclose(mean[3], .4)


def test_ensemble_refuses_misaligned_halves():
    from dfl_submission import diagnosis
    signals = tuple(np.zeros(3, dtype=np.float32) for _ in range(4))
    labels = np.zeros(3, dtype=np.int8)
    first = [diagnosis.HalfSignals("dfl", "WPY", "clean", 0, labels, signals, labels)]
    other = [diagnosis.HalfSignals("dfl", "WPY", "masked", 0, labels, signals, labels)]
    assert len(zoo.ensemble([first, first])) == 1
    try:
        zoo.ensemble([first, other])
    except ValueError:
        pass
    else:
        raise AssertionError("misaligned halves must be refused")


def fake_trial(value, macro=.5):
    return {"external": {"conventions": {"aba3": {"standard": {"spell_f1": value, "macro_f1": macro}}}}}


def test_greedy_forward_selection_adds_while_the_primary_rises():
    scores = {("d15",): .70, ("tcn",): .72, ("gru",): .69, ("trees",): .71,
              ("tcn", "d15"): .73, ("tcn", "gru"): .725, ("tcn", "trees"): .73,
              ("tcn", "d15", "gru"): .729, ("tcn", "d15", "trees"): .735,
              ("tcn", "d15", "trees", "gru"): .734}
    chosen, trial, steps = zoo.greedy(["d15", "tcn", "gru", "trees"], lambda members: fake_trial(scores[members]))
    assert chosen == ["tcn", "d15", "trees"] and zoo.primary(trial) == .735 and len(steps) == 4
    alone, _, _ = zoo.greedy(["d15", "tcn"], lambda members: fake_trial({("d15",): .7, ("tcn",): .6, ("d15", "tcn"): .7}[members]))
    assert alone == ["d15"]
    chosen, _, _ = zoo.greedy(["a", "b", "c", "d", "e"], lambda members: fake_trial(.7 + .01 * len(members)))
    assert chosen == ["a", "b", "c", "d"]


def synthetic_tracking(seed, length=1500):
    rng = np.random.default_rng(seed)
    start = rng.uniform([-45, -30], [45, 30], size=(22, 2))
    positions = (start + np.cumsum(rng.normal(0, .15, size=(length, 22, 2)), axis=0)).astype(np.float32)
    ball = (np.cumsum(rng.normal(0, .5, size=(length, 2)), axis=0) % 40 - 20).astype(np.float32)
    ball[400:430] = np.nan
    positions[700:760, 3] = np.nan
    return positions, ball, np.repeat([0, 1], 11), np.array([105., 68.])


def test_d20_forecasts_equal_d15s_for_d15():
    from dfl_submission import diagnosis, scale_trial
    tracking = synthetic_tracking(5)
    ours = zoo.half_signals(zoo.teacher("d15"), *tracking)
    theirs = diagnosis.half_signals(scale_trial.teacher(zoo.d15_setting()[0]), *tracking)
    assert np.array_equal(ours[1], theirs[1])
    assert all(np.array_equal(a, b) for a, b in zip(ours[0], theirs[0]))
    assert zoo.member_identity("d15")["networks"] == scale_trial.identity(zoo.d15_setting()[0])["networks"]


def fake_network(folder, family, columns, width, seed):
    """An exported network folder with random NumPy weights of the family's shapes (no torch)."""
    rng = np.random.default_rng(seed)
    shapes = {"project.weight": (width, 2 * columns), "project.bias": (width,)}
    if family == "gru":
        for layer, inputs in ((0, width), (1, 2 * width)):
            for suffix in ("", "_reverse"):
                shapes |= {f"gru.weight_ih_l{layer}{suffix}": (3 * width, inputs), f"gru.weight_hh_l{layer}{suffix}": (3 * width, width),
                           f"gru.bias_ih_l{layer}{suffix}": (3 * width,), f"gru.bias_hh_l{layer}{suffix}": (3 * width,)}
        out = 2 * width
    else:
        for layer in range(zoo.TRANSFORMER_LAYERS):
            prefix = f"encoder.layers.{layer}."
            shapes |= {prefix + "self_attn.in_proj_weight": (3 * width, width), prefix + "self_attn.in_proj_bias": (3 * width,),
                       prefix + "self_attn.out_proj.weight": (width, width), prefix + "self_attn.out_proj.bias": (width,),
                       prefix + "linear1.weight": (2 * width, width), prefix + "linear1.bias": (2 * width,),
                       prefix + "linear2.weight": (width, 2 * width), prefix + "linear2.bias": (width,),
                       prefix + "norm1.weight": (width,), prefix + "norm1.bias": (width,),
                       prefix + "norm2.weight": (width,), prefix + "norm2.bias": (width,)}
        out = width
    shapes |= {"norm.weight": (out,), "norm.bias": (out,), "outputs.weight": (4, out), "outputs.bias": (4,)}
    weights = {key: (rng.normal(0, .2, shape) + (1 if key.endswith("norm.weight") or "norm1.weight" in key
                                                  or "norm2.weight" in key else 0)).astype(np.float32)
               for key, shape in shapes.items()}
    folder.mkdir(parents=True)
    np.savez(folder / "temporal_5400.npz", **weights)
    published = ROOT / "dfl_submission" / "temporal_models"
    (folder / "normalization.npz").write_bytes((published / "normalization.npz").read_bytes())
    names = json.loads((published / "fit.json").read_text())["columns"]
    (folder / "fit.json").write_text(json.dumps({"models": {"temporal_5400.npz": digest(folder / "temporal_5400.npz")},
                                                 "normalization": digest(folder / "normalization.npz"), "columns": names,
                                                 "family": family, "width": width, "steps": 5400}))


def test_gru_and_transformer_members_run_through_d15s_structure():
    from dfl_submission import diagnosis
    columns = len(json.loads((ROOT / "dfl_submission" / "temporal_models" / "fit.json").read_text())["columns"])
    tracking = synthetic_tracking(6, length=1400)
    d15 = zoo.half_signals(zoo.teacher("d15"), *tracking)[0]
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp)
        for seed, family in enumerate(("gru", "transformer")):
            for model_set in zoo.SETS:
                for slot in zoo.NETWORKS:
                    fake_network(models / family / model_set / slot, family, columns, 8, seed)
            identity = zoo.member_identity(family, models)
            assert set(identity["networks"]) == set(zoo.SETS)
            model = zoo.teacher(family, models)
            assert isinstance(model[1]["current"][1][next(iter(model[1]["current"][1]))], zoo.Checkpoint)
            signals, labels = zoo.half_signals(model, *tracking)
            assert len(labels) == len(tracking[1]) and all(np.isfinite(s).all() for s in signals)
            assert all(s.min() >= 0 and s.max() <= 1 for s in signals)
            assert not np.array_equal(signals[0], d15[0])
            assert np.array_equal(diagnosis.decode(signals, model[2]["recipe"]), labels)
            mixed = zoo.mean_signals([d15, signals])
            assert np.allclose(mixed[0], (d15[0] + signals[0]) / 2, atol=1e-7)


def test_git_archive_of_the_code_paths_hashes_like_the_commit():
    import zipfile
    head = zoo.git("rev-parse", "HEAD").strip()
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "code.zip"
        subprocess.run(["git", "archive", "--format=zip", "-o", str(archive), head, "--", *zoo.CODE_PATHS], cwd=ROOT, check=True)
        with zipfile.ZipFile(archive) as item:
            item.extractall(Path(tmp) / "repo")
        extracted = zoo.folder_files(Path(tmp) / "repo")
        assert extracted == zoo.commit_files(head) and len(extracted) > 200
        assert zoo.tree_hash(extracted) == zoo.tree_hash(zoo.commit_files(head))
        assert "dfl_submission/temporal_network.py" in extracted and not any(k.startswith("data/") for k in extracted)


def test_definitions_compare_named_code_only():
    old = "import os\nA = 1\n\ndef f(x):\n    return x + A\n\nclass K:\n    pass\n"
    new = "import os, sys\nA = 1\n\ndef f(x):\n    return x + A\n\ndef g():\n    return 2\n\nclass K:\n    pass\n"
    first, second = zoo.definitions(old), zoo.definitions(new)
    assert first["f"] == second["f"] and first["A"] == second["A"] and first["K"] == second["K"] and "g" not in first
    assert zoo.definitions(new.replace("x + A", "x - A"))["f"] != second["f"]
    assert set(zoo.TRAINING_CODE) <= set(zoo.definitions((ROOT / "dfl_submission" / "zoo.py").read_text()))


def fake_cloud_run(tmp, family="gru"):
    """A cloud work folder as `network_cv` and `train_family` leave it, with a CV record, on synthetic inputs."""
    current = synthetic_folder(tmp / "train_data", seed=0)
    augmented = synthetic_folder(tmp / "sequences_augmented", seed=1)
    sources = {"current": current, "augmented": augmented}
    manifests = {key: digest(folder / "manifest.json") for key, folder in sources.items()}
    recipe, work = tmp / "cloud_recipe", tmp / "work"
    recipe.mkdir()
    entries = []
    for setting in zoo.GRIDS[family]:
        runs = {}
        for fold in range(zoo.FOLDS):
            folder = zoo.fold_folder(current, fold, work / "folds" / str(fold))
            run = work / "cv" / family / zoo.name(setting) / f"fold{fold}"
            run.mkdir(parents=True)
            (run / "fit.json").write_text(json.dumps({"family": family, "width": setting["width"], "steps": setting["steps"],
                                                      "seed": 42, "boundary_boost": .5, "models": {},
                                                      "train_manifest": digest(folder / "manifest.json")}))
            runs[str(fold)] = {"fit_sha256": digest(run / "fit.json"), "train_manifest": digest(folder / "manifest.json")}
        entries.append({"setting": setting, "summary": {"score": 1 - .01 * len(entries) % .03}, "matches": {}, "runs": runs})
    selected = zoo.select_cv(entries)
    zoo.write_json(recipe / f"cv_{family}.json", {
        "protocol": zoo.PROTOCOL, "family": family, "grid": list(zoo.GRIDS[family]), "entries": entries, "selected": selected,
        "anchor": None, "cv_network": {"seed": 42, "boundary_boost": .5}, "source_manifest": manifests["current"],
        "folds": zoo.fold_table(current)})
    for model_set in zoo.SETS:
        for slot, (seed, boost) in zoo.NETWORKS.items():
            run = work / "networks" / family / model_set / slot
            run.mkdir(parents=True)
            (run / f"temporal_{selected['steps']}.pt").write_bytes(f"weights {model_set} {slot}".encode())
            (run / "normalization.npz").write_bytes(b"normalization")
            (run / "fit.json").write_text(json.dumps({
                "family": family, "width": selected["width"], "steps": selected["steps"], "seed": seed, "boundary_boost": boost,
                "train_manifest": manifests[model_set], "normalization": digest(run / "normalization.npz"),
                "models": {f"temporal_{selected['steps']}.pt": digest(run / f"temporal_{selected['steps']}.pt")}}))
    head = zoo.git("rev-parse", "HEAD").strip()
    code = tmp / "code.json"
    zoo.write_json(code, {"commit": head, "tree_sha256": zoo.tree_hash(zoo.commit_files(head)), "files": 0})
    archive = tmp / "outputs.zip"
    zoo.package_cloud(family, work, code, sources, archive, recipe=recipe, manifests=manifests,
                      environment_of=lambda: {"python": "test"})
    return sources, manifests, recipe, archive


def test_cloud_runs_are_checked_before_use():
    import datetime
    import zipfile
    before = datetime.datetime(2026, 11, 12, 20, tzinfo=datetime.timezone.utc)
    exported = []

    def exporter(family, raw, destination):
        destination.mkdir(parents=True)
        (destination / "fit.json").write_text(json.dumps({"from": raw.name}))
        exported.append(raw)

    original = zoo.training_code_matches
    zoo.training_code_matches = lambda commit: {"stub": True}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sources, manifests, recipe, archive = fake_cloud_run(tmp)
            options = dict(manifests=manifests, exporter=exporter, now=before)
            local = tmp / "local"
            local.mkdir()
            assert zoo.import_cloud("gru", archive, tmp / "x1", sources, tmp / "scratch", recipe=local, models=tmp / "m1", **options)
            assert len(exported) == 4 and digest(local / "cv_gru.json") == digest(recipe / "cv_gru.json")
            assert json.loads((local / "cloud_gru.json").read_text())["passed"] is True
            try:
                zoo.import_cloud("gru", archive, tmp / "x1", sources, tmp / "scratch", recipe=local, models=tmp / "m1", **options)
            except FileExistsError:
                pass
            else:
                raise AssertionError("a cloud run is imported once")
            with zipfile.ZipFile(archive) as item:
                contents = {key: item.read(key) for key in item.namelist()}
            for attempt, (key, value) in enumerate((
                    ("runs/current/seed_models/temporal_5400.pt", b"tampered"),
                    ("zoo_recipe/cv_gru.json", contents["zoo_recipe/cv_gru.json"].replace(b'"seed": 42', b'"seed": 7')))):
                altered = dict(contents)
                if key not in altered:
                    key = next(k for k in altered if k.startswith("runs/current/seed_models/temporal_"))
                altered[key] = value
                bad, other = tmp / f"bad_{attempt}.zip", tmp / f"local_{attempt}"
                with zipfile.ZipFile(bad, "w") as item:
                    for name_, data in altered.items():
                        item.writestr(name_, data)
                other.mkdir()
                assert not zoo.import_cloud("gru", bad, tmp / "x2", sources, tmp / "scratch", recipe=other, models=tmp / "m2", **options)
                assert not (other / "cv_gru.json").exists() and len(exported) == 4
            late = tmp / "late"
            late.mkdir()
            try:
                zoo.import_cloud("gru", archive, tmp / "x3", sources, tmp / "scratch", recipe=late, models=tmp / "m3",
                                 manifests=manifests, exporter=exporter,
                                 now=datetime.datetime(2026, 11, 13, 13, 30, tzinfo=datetime.timezone.utc))
            except ValueError:
                pass
            else:
                raise AssertionError("past the deadline the family runs locally")
    finally:
        zoo.training_code_matches = original
