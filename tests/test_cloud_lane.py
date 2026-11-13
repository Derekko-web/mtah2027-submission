"""Cloud lane: staging refuses anything but external training data, and a job's round trip is checked end to end
(cloud/lane.py, cloud/lane_remote.py)."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cloud"))

import lane                                                                     # noqa: E402
import lane_remote as remote                                                    # noqa: E402
from dfl_submission import player_set as ps, stack_set as ss, temporal_network  # noqa: E402
from evaluation import possession_metrics as pm                                 # noqa: E402

COLUMNS = ["h__x", "a__x", "h__y", "a__y"]
SIGN_MAP = {"columns": COLUMNS, "signs": {"x": [1, 1, 1, 1], "y": [1, 1, 1, 1], "xy": [1, 1, 1, 1]}}
CONFIG = {"direction": True, "seed_offset": 0, "mirror": "y", "noise": False, "aux": False}
NETWORKS = {"temporal_models": (42, .5)}
OPTIONS = {"width": 8, "batch_size": 4}


def prepared(tmp_path, *, source_of=lambda index: "dfl" if index < 2 else "skillcorner", match="M"):
    """One set's tiny prepared folder, tokens, sign file, pitch file and parity record, as the lane stages them."""
    folder, tokens = tmp_path / "sequences", tmp_path / "tokens"
    folder.mkdir()
    tokens.mkdir()
    rng = np.random.default_rng(3)
    records, token_records, signs, pitches = [], {}, {}, {}
    for index in range(4):
        labels = np.where(np.arange(420) % 90 < 45, pm.H, pm.A).astype(np.int8)
        labels[200:230] = pm.D
        source = source_of(index)
        name = f"{source}_{match}{index // 2}_clean_{index % 2}.npz"
        np.savez_compressed(folder / name, values=rng.normal(size=(420, 4)).astype(np.float32),
                            targets=temporal_network.targets(labels))
        records.append({"file": name, "source": source, "match": f"{match}{index // 2}", "sha256": remote.sha256(folder / name)})
        values = rng.normal(size=(420, 2, ps.PER_TEAM, ps.TOKEN)).astype(np.float16)
        values[:, :, 9:] = np.nan
        np.savez_compressed(tokens / name, tokens=values)
        token_records[name] = remote.sha256(tokens / name)
        signs[name] = {"sign": -1 if index % 2 else 1}
        pitches[name] = [105., 68.]
    (folder / "manifest.json").write_text(json.dumps({"columns": COLUMNS, "records": records}))
    (tokens / "manifest.json").write_text(json.dumps({"base_manifest": remote.sha256(folder / "manifest.json"),
                                                      "records": token_records}))
    (tmp_path / "signs.json").write_text(json.dumps({"base_manifest": remote.sha256(folder / "manifest.json"),
                                                     "token_manifest": remote.sha256(tokens / "manifest.json"),
                                                     "records": signs}))
    (tmp_path / "pitches.json").write_text(json.dumps(pitches))
    (tmp_path / "parity.json").write_text(json.dumps({"passed": True, "sign_map": SIGN_MAP}))
    (tmp_path / "DECISIONS.md").write_text("# Decisions\n\n## D99 — A test rule (written first)\n")
    folders = {"sequences_current": folder, "tokens_current": tokens}
    files = {"signs_current.json": tmp_path / "signs.json", "pitches_current.json": tmp_path / "pitches.json",
             "parity.json": tmp_path / "parity.json"}
    return folders, files


def raises(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError) as error:
        return str(error)
    raise AssertionError("No refusal")


def job_of(tmp_path, **changes):
    options = dict(config=CONFIG, networks=NETWORKS, sets=("current",), offsets=(0, 1000), steps=6, options=OPTIONS,
                   decisions=tmp_path / "DECISIONS.md")
    return lane.spec("unit", "D99", tmp_path / "stage", **dict(options, **changes))


def test_stage_takes_only_external_training_records(tmp_path):
    folders, files = prepared(tmp_path, source_of=lambda index: "pff" if index == 3 else "dfl")
    assert "only DFL and SkillCorner" in raises(lane.stage, folders, files, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()


def test_stage_refuses_competition_file_names(tmp_path):
    folders, files = prepared(tmp_path, match="BAR-ATH")
    assert "must never be staged" in raises(lane.stage, folders, files, tmp_path / "stage")
    second = tmp_path / "second"
    second.mkdir()
    other, extra = prepared(second)
    assert "not a file the lane stages" in raises(lane.stage, other, dict(extra, **{"submission.csv": extra["parity.json"]}),
                                                  tmp_path / "stage2")


def test_stage_checks_hashes_and_pairs(tmp_path):
    folders, files = prepared(tmp_path)
    (folders["sequences_current"] / "stray.npz").write_bytes(b"x")
    assert "outside its manifest" in raises(lane.stage, folders, files, tmp_path / "a")
    (folders["sequences_current"] / "stray.npz").unlink()
    manifest = json.loads((folders["tokens_current"] / "manifest.json").read_text())
    (folders["tokens_current"] / "manifest.json").write_text(json.dumps(dict(manifest, base_manifest="0" * 64)))
    assert "does not belong" in raises(lane.stage, folders, files, tmp_path / "b")


def test_staging_is_repeatable_and_ships_no_data_or_tests(tmp_path):
    folders, files = prepared(tmp_path)
    first, code = lane.stage(folders, files, tmp_path / "one")
    second, _ = lane.stage(folders, files, tmp_path / "two")
    assert first == second
    for name in ("sequences_current.zip", "tokens_current.zip"):
        assert remote.sha256(tmp_path / "one" / "mtah-lane-data" / name) == remote.sha256(tmp_path / "two" / "mtah-lane-data" / name)
    assert remote.sha256(tmp_path / "one" / "mtah-lane-code" / "code.zip") == \
        remote.sha256(tmp_path / "two" / "mtah-lane-code" / "code.zip")
    shipped = set(code["files"])
    assert {"dfl_submission/stack_set.py", "dfl_submission/context_set.py", "cloud/lane_remote.py",
            "dfl_submission/requirements-training.txt", "experiments/dfl_transfer_data.py"} <= shipped
    allowed = (".py", ".txt", "LICENSE", "LICENSE-SKILLCORNER")
    assert all(name.endswith(allowed) for name in shipped)
    assert not any(Path(name).name.startswith("test_") or name.startswith(("data/", "tests/", "rules/")) for name in shipped)
    assert not (tmp_path / "one" / "mtah-lane-data" / "LOCAL.json").exists()        # local paths are never uploaded


def test_a_job_needs_a_written_rule_and_a_sound_shape(tmp_path):
    folders, files = prepared(tmp_path)
    lane.stage(folders, files, tmp_path / "stage")
    assert "no heading for D100" in raises(lane.spec, "unit", "D100", tmp_path / "stage", config=CONFIG, networks=NETWORKS,
                                           decisions=tmp_path / "DECISIONS.md")
    job = job_of(tmp_path)
    assert [run["name"] for run in job["runs"]] == ["0/current/temporal_models", "1000/current/temporal_models"]
    assert [run["config"]["seed_offset"] for run in job["runs"]] == [0, 1000]
    for change in ({"blocks": 5}, {"options": {"epochs": 3}}, {"name": "../escape"}, {"config": dict(CONFIG, aux=True)}):
        broken = dict(job, runs=[dict(job["runs"][0], **change)])
        assert raises(remote.check_job, broken)
    assert raises(remote.check_job, dict(job, runs=[job["runs"][0], job["runs"][0]]))


def test_kernel_and_launcher(tmp_path):
    folders, files = prepared(tmp_path)
    lane.stage(folders, files, tmp_path / "stage")
    job = job_of(tmp_path)
    identity = lane.kernel(job, tmp_path / "kernel", "someone", parallel=3)
    metadata = json.loads((tmp_path / "kernel" / "kernel-metadata.json").read_text())
    assert identity == metadata["id"] == "someone/mtah-lane-unit"
    assert metadata["is_private"] is True and metadata["enable_gpu"] is True and metadata["kernel_type"] == "script"
    assert metadata["dataset_sources"] == ["someone/mtah-lane-data", "someone/mtah-lane-code"]
    script = (tmp_path / "kernel" / "lane_kernel.py").read_text()
    scope = {}
    exec(compile(script.split("INPUT, BOOT")[0], "lane_kernel.py", "exec"), scope)   # the embedded job only
    assert scope["JOB"] == job
    compile(script, "lane_kernel.py", "exec")
    commands = lane.colab(job, tmp_path / "stage", tmp_path / "colab", "s1", parallel=2)
    compile((tmp_path / "colab" / "lane_launcher.py").read_text(), "lane_launcher.py", "exec")
    assert commands[0] == "colab new -s s1 --gpu T4" and commands[-1] == "colab stop -s s1"
    assert "cloud/lane.py colab-upload --session s1" in commands[1] and commands[1].startswith("MTAH_DATA_DIR=/nonexistent/mtah ")
    assert "cloud/lane.py colab-wait --session s1" in commands[2] and len(commands) == 4
    launcher = (tmp_path / "colab" / "lane_launcher.py").read_text()
    assert "sleep" not in launcher and '"\\n".join' in launcher and '"--parallel",\n' in launcher and '"2"' in launcher
    assert set(lane.colab_files(tmp_path / "stage", tmp_path / "colab")) == {
        "sequences_current.zip", "tokens_current.zip", "signs_current.json", "pitches_current.json", "parity.json",
        "INPUTS.json", "CODE.json", "code.zip", "job.json"}
    assert json.loads((tmp_path / "colab" / "job.json").read_text()) == job


FAKE_COLAB = '''#!{python}
"""A stand-in for the `colab` CLI: `upload` copies into a local folder, `exec` runs the piped code against it."""
import shutil, subprocess, sys
from pathlib import Path
ROOT = {root!r}
command = sys.argv[1]
if command == "upload":
    local, target = sys.argv[4], sys.argv[5]
    if Path(local).stat().st_size > {limit}:
        print("[colab] Upload failed: 400 Client Error")
        sys.exit(0)
    destination = Path(target.replace("/content/lane/in", ROOT))
    if destination.name == {drop!r} and not Path(ROOT, "dropped").exists():
        Path(ROOT, "dropped").write_text("once")           # lose one upload, once
        print("[colab] Upload failed: 500 Server Error")
        sys.exit(0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(local, destination)
    print("[colab] Uploaded")
elif command == "exec":
    code = sys.stdin.read().replace("/content/lane/in", ROOT)
    sys.exit(subprocess.run([sys.executable, "-c", code]).returncode)
'''


def test_colab_upload_sends_large_files_in_parts_and_checks_hashes(tmp_path):
    remote_root, local = tmp_path / "remote", tmp_path / "local"
    remote_root.mkdir()
    local.mkdir()
    rng = np.random.default_rng(5)
    files = {"big.zip": local / "big.zip", "small.json": local / "small.json", "exact.zip": local / "exact.zip"}
    files["big.zip"].write_bytes(rng.bytes(2500))
    files["small.json"].write_bytes(b"{}")
    files["exact.zip"].write_bytes(rng.bytes(1000))
    binary = tmp_path / "colab"
    binary.write_text(FAKE_COLAB.format(python=sys.executable, root=str(remote_root), limit=1000, drop="big.zip.part0001"))
    binary.chmod(0o755)
    sent = lane.colab_upload("s", files, tmp_path / "parts", binary=str(binary), chunk=1000)
    assert sent == ["big.zip", "small.json", "exact.zip"]
    for name, path in files.items():
        assert (remote_root / name).read_bytes() == path.read_bytes()
    assert not list((remote_root / "parts").iterdir()) and not list((tmp_path / "parts").iterdir())
    assert (remote_root / "dropped").exists()                                       # the lost part was sent again
    (remote_root / "dropped").unlink()
    assert lane.colab_upload("s", files, tmp_path / "parts", binary=str(binary), chunk=1000) == []   # nothing to resend
    (remote_root / "small.json").write_bytes(b"[]")                                 # a file that changed on the machine
    assert lane.colab_upload("s", files, tmp_path / "parts", binary=str(binary), chunk=1000) == ["small.json"]


FAKE_WAIT = '''#!{python}
"""A stand-in for the `colab` CLI that answers `exec` with one prepared state per call and `download` with a copy."""
import shutil, sys
from pathlib import Path
ROOT, STATES = Path({root!r}), {states!r}
if sys.argv[1] == "exec":
    count = int((ROOT / "calls").read_text()) if (ROOT / "calls").exists() else 0
    (ROOT / "calls").write_text(str(count + 1))
    print("log line", count)
    print(STATES[min(count, len(STATES) - 1)])
elif sys.argv[1] == "download":
    shutil.copyfile(ROOT / Path(sys.argv[4]).name, sys.argv[5])
'''


def test_colab_wait_polls_until_done_and_checks_the_download(tmp_path):
    machine, out = tmp_path / "machine", tmp_path / "job"
    machine.mkdir()
    out.mkdir()
    (out / "job.json").write_text(json.dumps({"name": "unit"}))
    (out / "lane_launcher.py").write_text("")
    (machine / "lane_outputs_unit.zip").write_bytes(b"the output zip")
    digest = remote.sha256(machine / "lane_outputs_unit.zip")

    def fake(states):
        (machine / "calls").unlink(missing_ok=True)
        binary = tmp_path / "colab"
        binary.write_text(FAKE_WAIT.format(python=sys.executable, root=str(machine), states=states))
        binary.chmod(0o755)
        return str(binary)

    waits = []
    done = f"LANE_STATE done lane_outputs_unit.zip {digest}"
    archive = lane.colab_wait("s", out, binary=fake(["LANE_STATE running", "LANE_STATE running", done]), sleep=waits.append)
    assert archive.read_bytes() == b"the output zip" and (machine / "calls").read_text() == "3" and waits == [60, 60]
    assert "failed on Colab" in raises(lane.colab_wait, "s", out, binary=fake(["LANE_STATE failed RuntimeError: no GPU"]))
    assert "does not hash" in raises(lane.colab_wait, "s", out, binary=fake([f"LANE_STATE done lane_outputs_unit.zip {'0' * 64}"]))
    assert "no job state" in raises(lane.colab_wait, "s", out, binary=fake(["something else"]))


def rewritten(archive, target, change):
    """A copy of the zip with one member's bytes changed."""
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(target, "w") as copy:
        for name in source.namelist():
            data = source.read(name)
            copy.writestr(name, change(data) if name.endswith(change.member) else data)
    return target


def test_round_trip_is_the_local_trainer_and_import_refuses_anything_else(tmp_path):
    folders, files = prepared(tmp_path)
    stage = tmp_path / "stage"
    lane.stage(folders, files, stage)
    job = job_of(tmp_path)
    job["runs"].append(dict(job["runs"][0], name="seven/current/temporal_models", blocks=7))
    remote.check_job(job)
    (tmp_path / "job.json").write_text(json.dumps(job))
    result = subprocess.run([sys.executable, str(ROOT / "cloud" / "lane_remote.py"), "run", "--inputs", str(stage), "--work",
                             str(tmp_path / "work"), "--out", str(tmp_path / "out"), "--job", str(tmp_path / "job.json"),
                             "--python", sys.executable, "--device", "cpu", "--parallel", "2", "--state",
                             str(tmp_path / "STATE")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    archive = tmp_path / "out" / "lane_outputs_unit.zip"
    assert (tmp_path / "STATE").read_text().split()[:2] == ["done", archive.name]
    assert not list((tmp_path / "work" / "repo").rglob("*.npz"))                    # the code tree carries no weights

    record = lane.import_archive(archive, job, stage, tmp_path / "imported")
    assert record["environment"]["device"] == "cpu"
    name = "1000/current/temporal_models"
    raw, exported = tmp_path / "imported" / "raw" / name, tmp_path / "imported" / "export" / name
    fit = json.loads((raw / "fit.json").read_text())
    assert not (raw / "lane_modules.json").exists() and fit["seed"] == 1042 and fit["stack"]["mirror"] == "y"
    assert set(json.loads((exported / "fit.json").read_text())["models"]) == {p.name for p in exported.glob("temporal_*.npz")}
    seven = json.loads((tmp_path / "imported" / "export" / "seven" / "current" / "temporal_models" / "fit.json").read_text())
    assert seven["dilations"] == [1, 2, 4, 8, 16, 32, 64] and "dilations" not in fit      # the seven-block trainer and export

    # The same call made here gives the same network, bit for bit.
    ss.train(folders["sequences_current"], folders["tokens_current"], tmp_path / "direct", "cpu", seed=42, boundary_boost=.5,
             config=dict(CONFIG, seed_offset=1000),
             signs=ss.read_signs(files["signs_current.json"], folders["sequences_current"], folders["tokens_current"]),
             sign_map=SIGN_MAP, steps=6, provenance={"signs_file": remote.sha256(files["signs_current.json"])}, **OPTIONS)
    direct = json.loads((tmp_path / "direct" / "fit.json").read_text())
    assert direct["models"] == fit["models"] and direct["normalization"] == fit["normalization"]
    assert {key: value for key, value in direct.items()} == {key: value for key, value in fit.items()}

    assert "Already imported" in raises(lane.import_archive, archive, job, stage, tmp_path / "imported")
    assert "another job" in raises(lane.verify, archive, dict(job, note="changed"), stage)

    def flip(data):
        return data[:-1] + bytes([data[-1] ^ 1])
    flip.member = "temporal_6.pt"
    assert "does not hash" in raises(lane.verify, rewritten(archive, tmp_path / "flipped.zip", flip), job, stage)

    def other_seed(data):
        return json.dumps(dict(json.loads(data), seed=7)).encode()
    other_seed.member = "0/current/temporal_models/fit.json"
    tampered = rewritten(archive, tmp_path / "seed.zip", other_seed)
    with zipfile.ZipFile(tampered) as item:                                         # make its record agree with the file
        members = {member: item.read(member) for member in item.namelist()}
    stored = json.loads(members["RECORD.json"])
    import hashlib
    stored["runs"]["0/current/temporal_models"]["files"]["fit.json"] = \
        hashlib.sha256(members["runs/0/current/temporal_models/fit.json"]).hexdigest()
    members["RECORD.json"] = json.dumps(stored).encode()
    with zipfile.ZipFile(tampered, "w") as item:
        for member, data in members.items():
            item.writestr(member, data)
    assert "differs from the job in ['seed']" in raises(lane.verify, tampered, job, stage)

    # Training code that changed here after the run: the import refuses.
    copy = tmp_path / "root"
    for relative, path in lane.code_files().items():
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy / relative)
    lane.verify(archive, job, stage, root=copy)
    with open(copy / "dfl_submission" / "stack_set.py", "a") as handle:
        handle.write("\n# edited after the cloud run\n")
    assert "dfl_submission/stack_set.py" in raises(lane.verify, archive, job, stage, root=copy)

    # A staged folder that changed locally after staging: the import refuses.
    (folders["tokens_current"] / "manifest.json").write_text("{}")
    assert "has changed since it was staged" in raises(lane.verify, archive, job, stage)
