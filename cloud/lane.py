"""Cloud lane, local side: stage the training inputs, write jobs and launchers, and import what comes back.

The lane trains player-set networks on Kaggle or Colab with `cloud/lane_remote.py`, so that trainings
there do not count against the Mac's two. It decides nothing: a job names the DECISIONS.md rule it runs
under, and `spec` refuses a rule that is not written there yet. Commands (see `cloud/LANE.md`):

- `stage` copies the prepared training folders (sequences and tokens of both sets, their sign and pitch
  files, the mirror parity record) and the repository's Python files into two upload folders, with the
  SHA-256 of every file (`INPUTS.json`, `CODE.json`). It refuses any record that is not DFL or
  SkillCorner and any file whose name points at the competition bundle, PFF or Metrica;
- `spec` writes a job: the stack recipe's networks (`stack_recipe/levers.json`) for the chosen sets and
  seed offsets, with the steps and temporal blocks given;
- `kernel` writes the Kaggle script kernel of a job; `colab` writes the Colab launcher and prints the
  commands that run it;
- `import` checks an output zip (the job, the input and code digests, every output hash, every run's
  settings, and that the training code each run loaded is the code here now), unpacks the raw networks
  and exports each one with the NumPy check, as the local chain does.

Every command refuses to run while the competition bundle is reachable (`MTAH_DATA_DIR`).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "cloud")]

import lane_remote as remote                                                    # noqa: E402
from dfl_submission.external_only import require_external_only                  # noqa: E402

SCRATCH = ROOT / "data" / "cloud_lane"
SETS = ("current", "augmented")
SOURCES = {"dfl", "skillcorner"}
FORBIDDEN = re.compile(r"tactical|broadcast|bar[-_]?ath|pff|metrica|possession_gt|gt_public|submission", re.I)
CODE_FOLDERS = ("dfl_submission", "experiments", "evaluation", "starter")
CODE_FILES = ("games.py", "cloud/lane_remote.py", "dfl_submission/requirements-training.txt", "LICENSE",
              "LICENSE-SKILLCORNER")
DATASETS = {"data": "mtah-lane-data", "code": "mtah-lane-code"}
DEFAULTS = {"width": 96, "lr_scale": 2., "batch": 32, "weight_decay": .01, "dropout": .2}
OPTION_FIT = {"width": "width", "lr_scale": "lr_scale", "batch_size": "batch", "weight_decay": "weight_decay",
              "dropout": "dropout"}
SIX = [1, 2, 4, 8, 16, 32]
ZIP_DATE = (1980, 1, 1, 0, 0, 0)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; lane records are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


# -- staging -------------------------------------------------------------------------------------------

def folder_entry(key, folder, partner=None):
    """{manifest digest, files} of one prepared folder, after checking that it is external training data.

    A sequences folder lists its records with their source; a token folder names the sequences folder it
    belongs to (`partner`: that folder's manifest digest).
    """
    folder = Path(folder)
    names = sorted(path.name for path in folder.iterdir())
    if any(not (folder / name).is_file() for name in names):
        raise ValueError(f"{folder} holds something that is not a file")
    manifest = read(folder / "manifest.json")
    records = manifest["records"]
    if isinstance(records, list):
        sources = {record["source"] for record in records}
        if not sources <= SOURCES:
            raise ValueError(f"{key}: records from {sorted(sources - SOURCES)}; only DFL and SkillCorner are staged")
        if any(not record["file"].startswith(record["source"] + "_") for record in records):
            raise ValueError(f"{key}: a record's file is not named after its source")
        hashes = {record["file"]: record["sha256"] for record in records}
    else:
        if partner is None or manifest["base_manifest"] != partner:
            raise ValueError(f"{key} does not belong to the staged sequences folder")
        hashes = dict(records)
    if set(names) != set(hashes) | {"manifest.json"}:
        raise ValueError(f"{key} holds files outside its manifest")
    bad = [name for name in names if FORBIDDEN.search(name)]
    if bad:
        raise ValueError(f"{key}: file names that must never be staged: {bad[:3]}")
    wrong = [name for name, expected in hashes.items() if remote.sha256(folder / name) != expected]
    if wrong:
        raise ValueError(f"{key}: hash mismatch in {wrong[:3]}")
    files = dict(hashes, **{"manifest.json": remote.sha256(folder / "manifest.json")})
    return {"manifest": files["manifest.json"], "files": files}


def stored_zip(archive, members):
    """A zip without compression or timestamps, so the same inputs always give the same archive."""
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as item:
        for name, path in sorted(members):
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.external_attr = 0o644 << 16
            item.writestr(info, Path(path).read_bytes())


def code_files(root=ROOT):
    """The repository's Python files (no tests, weights, recipes or data) and the files the remote side needs."""
    root = Path(root)
    paths = [path for folder in CODE_FOLDERS for path in sorted((root / folder).glob("*.py"))
             if not path.name.startswith("test_")]
    paths += [root / name for name in CODE_FILES]
    return {str(path.relative_to(root)): path for path in paths}


def git_state(root, names):
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(["git", "status", "--porcelain", "--", *names], cwd=root, capture_output=True, text=True,
                                 check=True).stdout.splitlines()
        return head, sorted(line[3:] for line in changed)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def stage(folders, files, out, *, root=ROOT, user=None, label=None):
    """Write `out/mtah-lane-data` and `out/mtah-lane-code`.

    `folders`: {"sequences_<set>": path, "tokens_<set>": path}; `files`: {staged name: path} (sign, pitch
    and parity files). Returns (inputs record, code record).
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; stage into a new folder")
    entries = {}
    for key in sorted(folders):
        kind, _, model_set = key.partition("_")
        if kind not in ("sequences", "tokens") or model_set not in SETS:
            raise ValueError(f"Unknown staged folder {key}")
        if kind == "sequences":
            entries[key] = folder_entry(key, folders[key])
    for key in sorted(folders):
        kind, _, model_set = key.partition("_")
        if kind == "tokens":
            entries[key] = folder_entry(key, folders[key], entries[f"sequences_{model_set}"]["manifest"])
    for name, path in files.items():
        if FORBIDDEN.search(name) or not re.fullmatch(r"(signs|pitches)_(current|augmented)\.json|parity\.json", name):
            raise ValueError(f"{name} is not a file the lane stages")
        if name.startswith("signs_"):
            model_set = name[len("signs_"):-len(".json")]
            stored = read(path)
            if stored["base_manifest"] != entries[f"sequences_{model_set}"]["manifest"] or \
                    stored["token_manifest"] != entries[f"tokens_{model_set}"]["manifest"]:
                raise ValueError(f"{name} holds signs for other folders than the staged ones")
    data_out, code_out = out / DATASETS["data"], out / DATASETS["code"]
    data_out.mkdir(parents=True)
    code_out.mkdir(parents=True)
    for key, entry in entries.items():
        stored_zip(data_out / f"{key}.zip", [(name, Path(folders[key]) / name) for name in entry["files"]])
    for name, path in files.items():
        (data_out / name).write_bytes(Path(path).read_bytes())
    inputs = {"lane": remote.LANE, "label": label, "folders": entries,
              "files": {name: remote.sha256(data_out / name) for name in sorted(files)}}
    write_new(data_out / "INPUTS.json", inputs)
    shipped = code_files(root)
    stored_zip(code_out / "code.zip", shipped.items())
    head, changed = git_state(root, sorted(shipped))
    code = {"lane": remote.LANE, "files": {name: remote.sha256(path) for name, path in sorted(shipped.items())},
            "head": head, "not_committed": changed}
    write_new(code_out / "CODE.json", code)
    (data_out / "README.md").write_text(
        "Prepared training sequences and player tokens for the cloud lane (private). Derived only from external data:\n"
        "DFL Bundesliga tracking and events (Bassek et al. 2025, CC BY 4.0) and SkillCorner open data (MIT).\n"
        "No competition file, public ground truth, PFF or Metrica file is here. Hashes: INPUTS.json.\n")
    (code_out / "README.md").write_text(
        f"Repository Python files for the cloud lane (private). HEAD {head}; hashes of every file: CODE.json.\n")
    owner = user or os.environ.get("KAGGLE_USERNAME") or "YOUR_KAGGLE_USERNAME"
    for kind, folder in (("data", data_out), ("code", code_out)):
        (folder / "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASETS[kind], "id": f"{owner}/{DATASETS[kind]}", "licenses": [{"name": "other"}]}) + "\n")
    write_new(out / "LOCAL.json", {"folders": {key: str(Path(path).resolve()) for key, path in folders.items()},
                                   "files": {name: str(Path(path).resolve()) for name, path in files.items()},
                                   "root": str(Path(root).resolve()), "staged_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    return inputs, code


def stack_sources(external):
    """The folders and files the stack recipe trains on (`stack_recipe/levers.json`), from `stack_trial`."""
    from dfl_submission import stack_trial as st
    chosen = st.levers()
    folders, files = {}, {"parity.json": st.PARITY}
    signs_record = st.read(st.RECIPE / "signs.json")
    for model_set in SETS:
        source, tokens = st.data_folders(chosen["data"], model_set, external)
        folders[f"sequences_{model_set}"], folders[f"tokens_{model_set}"] = source, tokens
        signs_file = st.SCRATCH / f"signs_{model_set}.json"
        if remote.sha256(signs_file) != signs_record["files"][model_set]:
            raise ValueError(f"{signs_file} is not the file D48's sign check passed on")
        files[f"signs_{model_set}.json"] = signs_file
        files[f"pitches_{model_set}.json"] = st.SCRATCH / f"pitches_{model_set}.json"
    return folders, files, chosen["data"]


# -- jobs ----------------------------------------------------------------------------------------------

def rule_written(protocol, decisions=ROOT / "DECISIONS.md"):
    """Whether DECISIONS.md has a heading for this rule (e.g. `D53`)."""
    pattern = re.compile(rf"^#+ .*\b{re.escape(protocol)}\b", re.M)
    return bool(pattern.search(Path(decisions).read_text()))


def spec(name, protocol, stage_folder, *, config, networks, sets=SETS, offsets=(0,), steps=5400, blocks=6, options=None,
         note="", decisions=ROOT / "DECISIONS.md"):
    """A job: one run per (seed offset, set, network), named `<offset>/<set>/<network folder>` as D48's folders."""
    if not rule_written(protocol, decisions):
        raise ValueError(f"DECISIONS.md has no heading for {protocol}; the rule is written before anything is trained")
    stage_folder = Path(stage_folder)
    runs = []
    for offset in offsets:
        for model_set in sets:
            for folder, (seed, boost) in networks.items():
                runs.append({"name": f"{offset}/{model_set}/{folder}", "source": f"sequences_{model_set}",
                             "tokens": f"tokens_{model_set}", "signs": f"signs_{model_set}.json",
                             "pitches": f"pitches_{model_set}.json", "seed": seed, "boundary_boost": boost,
                             "config": dict(config, seed_offset=offset), "steps": steps, "blocks": blocks,
                             "options": dict(options or {})})
    job = {"name": name, "protocol": protocol, "note": note,
           "inputs_digest": remote.sha256(stage_folder / DATASETS["data"] / "INPUTS.json"),
           "code_digest": remote.sha256(stage_folder / DATASETS["code"] / "CODE.json"), "runs": runs}
    remote.check_job(job)
    return job


KERNEL = '''# Kaggle script kernel of cloud-lane job {name} ({protocol}); written by cloud/lane.py, do not edit.
# Inputs: the private datasets mtah-lane-data and mtah-lane-code (external training data and this repository's code).
import json, subprocess, sys, zipfile
from pathlib import Path

JOB = json.loads({job!r})
INPUT, BOOT = Path("/kaggle/input"), Path("/tmp/lane_boot")
archives = sorted(INPUT.rglob("code.zip"))
if archives:
    with zipfile.ZipFile(archives[0]) as item:
        item.extractall(BOOT)
    runner = BOOT / "cloud" / "lane_remote.py"
else:
    found = sorted(path for path in INPUT.rglob("lane_remote.py") if path.parent.name == "cloud")
    assert found, "Attach the mtah-lane-code dataset"
    runner = found[0]
job_file = Path("/tmp/lane_job.json")
job_file.write_text(json.dumps(JOB))
result = subprocess.run([sys.executable, str(runner), "run", "--inputs", str(INPUT), "--work", "/tmp/lane", "--out",
                         "/kaggle/working", "--job", str(job_file), "--parallel", "{parallel}"])
if result.returncode:
    raise RuntimeError(f"The lane job failed with exit code {{result.returncode}}")
'''

LAUNCHER = '''# Colab launcher of cloud-lane job {name} ({protocol}); written by cloud/lane.py, do not edit.
# Run with `colab exec -s SESSION -f` once the staged files and job.json are in /content/lane/in. The first run starts
# the job detached from the kernel. Every run prints the end of the job's log and its state, and returns at once: the
# kernel stays free, and `cloud/lane.py colab-wait` asks again every minute. (`colab exec` gives up on a cell that
# prints nothing for 30 s, so nothing here waits.)
import subprocess, sys, zipfile
from pathlib import Path

IN, BOOT, OUT = Path("/content/lane/in"), Path("/content/lane/boot"), Path("/content/lane/out")
OUT.mkdir(parents=True, exist_ok=True)
state, log, pid = OUT / "STATE", OUT / "lane.log", OUT / "lane.pid"
if not pid.exists():
    with zipfile.ZipFile(IN / "code.zip") as item:
        item.extractall(BOOT)
    process = subprocess.Popen([sys.executable, str(BOOT / "cloud" / "lane_remote.py"), "run", "--inputs", str(IN), "--work",
                                "/content/lane/work", "--out", str(OUT), "--job", str(IN / "job.json"), "--parallel",
                                "{parallel}", "--state", str(state)], stdout=open(log, "w"), stderr=subprocess.STDOUT,
                               start_new_session=True)
    pid.write_text(str(process.pid))
lines = log.read_text().splitlines() if log.exists() else []
print("\\n".join(lines[-8:]))
if state.exists():
    print("LANE_STATE", state.read_text().strip())
elif Path("/proc/" + pid.read_text().strip()).exists():
    print("LANE_STATE running")
else:
    print("LANE_STATE failed: the job's process ended without writing its state")
'''


def kernel(job, out, user, *, parallel=0):
    """The Kaggle kernel folder of a job: the script and its metadata (private, GPU, Internet for pip)."""
    remote.check_job(job)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "lane_kernel.py").write_text(KERNEL.format(name=job["name"], protocol=job["protocol"], job=json.dumps(job),
                                                      parallel=parallel))
    slug = "mtah-lane-" + re.sub(r"[^a-z0-9]+", "-", job["name"].lower()).strip("-")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{slug}", "title": slug, "code_file": "lane_kernel.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASETS['data']}", f"{user}/{DATASETS['code']}"],
        "competition_sources": [], "kernel_sources": []}, indent=2) + "\n")
    return f"{user}/{slug}"


def colab_files(stage_folder, out):
    """{name under /content/lane/in: local path} of everything a Colab job needs: the staged files and the job."""
    stage_folder = Path(stage_folder)
    paths = [path for kind in ("data", "code") for path in sorted((stage_folder / DATASETS[kind]).iterdir())
             if path.suffix in (".zip", ".json") and path.name != "dataset-metadata.json"]
    return {path.name: path for path in [*paths, Path(out) / "job.json"]}


def colab(job, stage_folder, out, session, *, parallel=0):
    """The Colab launcher and job file of a job, and the commands that upload, run, fetch and stop."""
    remote.check_job(job)
    out, stage_folder = Path(out), Path(stage_folder)
    out.mkdir(parents=True, exist_ok=False)
    (out / "lane_launcher.py").write_text(LAUNCHER.format(name=job["name"], protocol=job["protocol"], parallel=parallel))
    (out / "job.json").write_text(json.dumps(job, indent=2, sort_keys=True) + "\n")
    commands = [f"colab new -s {session} --gpu T4",
                f"MTAH_DATA_DIR={remote.ABSENT} .venv/bin/python cloud/lane.py colab-upload --session {session} "
                f"--stage {stage_folder} --job-folder {out}",
                f"MTAH_DATA_DIR={remote.ABSENT} .venv/bin/python cloud/lane.py colab-wait --session {session} "
                f"--job-folder {out}",
                f"colab stop -s {session}"]
    (out / "COMMANDS.txt").write_text("\n".join(commands) + "\n")
    return commands


CHUNK = 24 * 1024 * 1024            # `colab upload` took 20 MB and refused 80 MB (2026-11-13)
REMOTE_HASHES = '''
import hashlib, json
from pathlib import Path
root = Path("/content/lane/in")
(root / "parts").mkdir(parents=True, exist_ok=True)
found = {}
for path in sorted(root.iterdir()):
    if path.is_file():
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        found[path.name] = digest.hexdigest()
print("LANE_JSON", json.dumps(found))
'''
REMOTE_JOIN = '''
from pathlib import Path
root = Path("/content/lane/in")
parts = sorted((root / "parts").glob({name!r} + ".part*"))
assert len(parts) == {count}, f"{{len(parts)}} of {count} parts arrived"
with open(root / {name!r}, "wb") as joined:
    for part in parts:
        joined.write(part.read_bytes())
for part in parts:
    part.unlink()
print("LANE_JOINED", {name!r})
'''


def colab_exec(session, code, binary="colab"):
    result = subprocess.run([binary, "exec", "-s", session], input=code, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"colab exec failed: {(result.stdout + result.stderr)[-400:]}")
    return result.stdout


def colab_hashes(session, binary="colab"):
    lines = [line for line in colab_exec(session, REMOTE_HASHES, binary).splitlines() if line.startswith("LANE_JSON ")]
    return json.loads(lines[-1][len("LANE_JSON "):])


def colab_send(session, local, target, binary="colab", attempts=3):
    for attempt in range(attempts):
        result = subprocess.run([binary, "upload", "-s", session, str(local), target], capture_output=True, text=True)
        if result.returncode == 0 and "Upload failed" not in result.stdout + result.stderr:
            return
    raise RuntimeError(f"colab upload of {Path(local).name} failed {attempts} times: "
                       f"{(result.stdout + result.stderr).strip()[:90]}")       # the full message carries a session token


def colab_wait(session, out, *, binary="colab", poll=60, sleep=time.sleep):
    """Start the job of the folder `colab` wrote (if it is not running), ask for its state every `poll` seconds, and
    download its output zip when it is done. Returns the zip's local path."""
    out = Path(out)
    name = read(out / "job.json")["name"]
    last = None
    while True:
        result = subprocess.run([binary, "exec", "-s", session, "-f", str(out / "lane_launcher.py")], capture_output=True,
                                text=True)
        lines = result.stdout.splitlines()
        states = [line[len("LANE_STATE "):] for line in lines if line.startswith("LANE_STATE ")]
        if result.returncode or not states:
            raise RuntimeError(f"colab exec gave no job state: {(result.stdout + result.stderr).strip()[-300:]}")
        shown = [line for line in lines if line.strip() and not line.startswith("LANE_STATE ")]
        if shown and shown[-1] != last:
            last = shown[-1]
            print(last, flush=True)
        if states[-1].startswith("done"):
            archive = out / f"lane_outputs_{name}.zip"
            subprocess.run([binary, "download", "-s", session, f"/content/lane/out/{archive.name}", str(archive)],
                           capture_output=True, text=True, check=True)
            if remote.sha256(archive) != states[-1].split()[2]:
                raise RuntimeError("The downloaded zip does not hash as the job reported")
            return archive
        if states[-1].startswith("failed"):
            raise RuntimeError(f"The job failed on Colab: {states[-1]}\n" + "\n".join(shown[-8:]))
        sleep(poll)


def colab_upload(session, files, scratch, *, binary="colab", chunk=CHUNK):
    """Put `files` ({name: local path}) in /content/lane/in, large ones in parts joined on the machine; check every hash.

    A file already there with the right hash is not sent again. Returns the names sent.
    """
    expected = {name: remote.sha256(path) for name, path in files.items()}
    present = colab_hashes(session, binary)
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    sent = []
    for name, path in files.items():
        if present.get(name) == expected[name]:
            continue
        size = Path(path).stat().st_size
        if size <= chunk:
            colab_send(session, path, f"/content/lane/in/{name}", binary)
        else:
            count = -(-size // chunk)
            with open(path, "rb") as handle:
                for index in range(count):
                    part = scratch / f"{name}.part{index:04d}"
                    part.write_bytes(handle.read(chunk))
                    colab_send(session, part, f"/content/lane/in/parts/{part.name}", binary)
                    part.unlink()
                    print(f"{time.strftime('%H:%M:%S')} {name}: part {index + 1} of {count}", flush=True)
            colab_exec(session, REMOTE_JOIN.format(name=name, count=count), binary)
        sent.append(name)
    present = colab_hashes(session, binary)
    wrong = [name for name in files if present.get(name) != expected[name]]
    if wrong:
        raise RuntimeError(f"Files that did not arrive intact: {wrong}")
    return sent


# -- import --------------------------------------------------------------------------------------------

def check_fit(fit, run, inputs, device):
    """A run's fit record against the job: every setting, and the data it names."""
    config = run["config"]
    expected = dict(DEFAULTS, **{OPTION_FIT[key]: value for key, value in run["options"].items()})
    checks = {"seed": run["seed"] + config["seed_offset"], "boundary_boost": run["boundary_boost"], "steps": run["steps"],
              "train_manifest": inputs["folders"][run["source"]]["manifest"],
              "token_manifest": inputs["folders"][run["tokens"]]["manifest"], "training_device": device, **expected}
    differing = [key for key, value in checks.items() if fit.get(key) != value]
    if {key: fit["stack"].get(key) for key in config} != config:
        differing.append("stack switches")
    signs = inputs["files"][run["signs"]] if config["direction"] else None
    if fit["stack"].get("signs_file") != signs:
        differing.append("signs file")
    if len(fit.get("dilations", SIX)) != run["blocks"]:
        differing.append("temporal blocks")
    if differing:
        raise ValueError(f"{run['name']}: the fit record differs from the job in {differing}")


def verify(archive, job, stage_folder, *, root=ROOT):
    """Check an output zip against the job, the staged inputs and the code here now. Returns its record."""
    stage_folder, root = Path(stage_folder), Path(root)
    remote.check_job(job)
    inputs_file = stage_folder / DATASETS["data"] / "INPUTS.json"
    code_file = stage_folder / DATASETS["code"] / "CODE.json"
    if remote.sha256(inputs_file) != job["inputs_digest"] or remote.sha256(code_file) != job["code_digest"]:
        raise ValueError("The job names other staged inputs or code than this stage folder holds")
    inputs, code, local = read(inputs_file), read(code_file), read(stage_folder / "LOCAL.json")
    for key, entry in inputs["folders"].items():
        if remote.sha256(Path(local["folders"][key]) / "manifest.json") != entry["manifest"]:
            raise ValueError(f"The local folder {key} has changed since it was staged")
    for name, expected in inputs["files"].items():
        if remote.sha256(local["files"][name]) != expected:
            raise ValueError(f"The local file {name} has changed since it was staged")
    with zipfile.ZipFile(archive) as item:
        record = json.loads(item.read("RECORD.json"))
        if record["lane"] != remote.LANE or record["job"] != job or record["job_digest"] != remote.json_digest(job):
            raise ValueError("The output zip is from another job")
        if set(record["runs"]) != {run["name"] for run in job["runs"]}:
            raise ValueError("The output zip does not hold exactly the job's runs")
        members = set(item.namelist())
        device = record["environment"]["device"]
        for run in job["runs"]:
            name = run["name"]
            files = record["runs"][name]["files"]
            for filename, expected in files.items():
                member = f"runs/{name}/{filename}"
                if member not in members:
                    raise ValueError(f"{member} is missing from the output zip")
                if hashlib.sha256(item.read(member)).hexdigest() != expected:
                    raise ValueError(f"{member} does not hash as its record says")
            fit = json.loads(item.read(f"runs/{name}/fit.json"))
            if set(files) != set(fit["models"]) | {"fit.json", "normalization.npz", "lane_modules.json"}:
                raise ValueError(f"{name}: unexpected files {sorted(files)}")
            if any(files[model] != expected for model, expected in fit["models"].items()) or \
                    files["normalization.npz"] != fit["normalization"]:
                raise ValueError(f"{name}: the fit record's hashes differ from the files")
            check_fit(fit, run, inputs, device)
            loaded = json.loads(item.read(f"runs/{name}/lane_modules.json"))
            drifted = [module for module in loaded
                       if module not in code["files"] or not (root / module).is_file()
                       or remote.sha256(root / module) != code["files"][module]]
            if drifted:
                raise ValueError(f"{name}: training code that differs here from the code that ran: {drifted}")
    return record


def import_archive(archive, job, stage_folder, out, *, root=ROOT, export=True):
    """Verify, unpack the raw networks to `out/raw/<run>` and export each to `out/export/<run>` with the NumPy check."""
    require_external_only()
    out, root = Path(out), Path(root)
    record = verify(archive, job, stage_folder, root=root)
    targets = {run["name"]: (out / "raw" / run["name"], out / "export" / run["name"]) for run in job["runs"]}
    existing = [str(path) for pair in targets.values() for path in pair if path.exists()]
    if existing:
        raise FileExistsError(f"Already imported: {existing[:3]}")
    with zipfile.ZipFile(archive) as item:
        for run in job["runs"]:
            raw, exported = targets[run["name"]]
            raw.mkdir(parents=True)
            for filename in record["runs"][run["name"]]["files"]:
                if filename != "lane_modules.json":
                    (raw / filename).write_bytes(item.read(f"runs/{run['name']}/{filename}"))
            if export:
                module = "dfl_submission.player_set" if run["blocks"] == 6 else "dfl_submission.context_set"
                exported.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([sys.executable, "-m", module, "export", "--source", str(raw), "--out", str(exported)],
                               check=True, cwd=root, env=dict(os.environ, OMP_NUM_THREADS="1"))
    write_new(out / f"IMPORT_{job['name']}.json", {
        "job": job, "archive_sha256": remote.sha256(archive), "environment": record["environment"],
        "seconds": {name: entry["seconds"] for name, entry in record["runs"].items()},
        "raw": {name: record["runs"][name]["files"] for name in targets},
        "exported": export, "imported_at": time.strftime("%Y-%m-%d %H:%M %Z")})
    return record


# -- command line --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage")
    staging.add_argument("--external", type=Path, required=True, help="the external data root")
    staging.add_argument("--out", type=Path, default=SCRATCH / "stage")
    staging.add_argument("--user", help="Kaggle user name for the dataset metadata")
    writing = commands.add_parser("spec")
    writing.add_argument("--name", required=True)
    writing.add_argument("--protocol", required=True, help="the DECISIONS.md rule the job runs under, e.g. D53")
    writing.add_argument("--stage", type=Path, default=SCRATCH / "stage")
    writing.add_argument("--sets", default=",".join(SETS))
    writing.add_argument("--offsets", default="0", help="seed offsets, comma-separated")
    writing.add_argument("--steps", type=int, default=5400)
    writing.add_argument("--blocks", type=int, choices=(6, 7), default=6)
    writing.add_argument("--options", default="{}", help="JSON of width, lr_scale, batch_size, weight_decay, dropout")
    writing.add_argument("--note", default="")
    writing.add_argument("--out", type=Path, required=True)
    pushing = commands.add_parser("kernel")
    launching = commands.add_parser("colab")
    launching.add_argument("--stage", type=Path, default=SCRATCH / "stage")
    launching.add_argument("--session", default="lane")
    for command in (pushing, launching):
        command.add_argument("--job", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--parallel", type=int, default=0, help="trainings at once (default: one per CPU)")
    pushing.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"), required="KAGGLE_USERNAME" not in os.environ)
    sending = commands.add_parser("colab-upload")
    sending.add_argument("--session", required=True)
    sending.add_argument("--stage", type=Path, default=SCRATCH / "stage")
    sending.add_argument("--job-folder", type=Path, required=True, help="the folder `colab` wrote")
    sending.add_argument("--binary", default="colab")
    waiting = commands.add_parser("colab-wait")
    waiting.add_argument("--session", required=True)
    waiting.add_argument("--job-folder", type=Path, required=True, help="the folder `colab` wrote")
    waiting.add_argument("--binary", default="colab")
    waiting.add_argument("--poll", type=int, default=60)
    importing = commands.add_parser("import")
    importing.add_argument("--archive", type=Path, required=True)
    importing.add_argument("--job", type=Path, required=True)
    importing.add_argument("--stage", type=Path, default=SCRATCH / "stage")
    importing.add_argument("--out", type=Path, required=True)
    importing.add_argument("--no-export", action="store_true")
    args = parser.parse_args()
    require_external_only()
    if args.command == "stage":
        folders, files, label = stack_sources(args.external)
        inputs, code = stage(folders, files, args.out, user=args.user, label=label)
        size = sum(path.stat().st_size for path in args.out.rglob("*") if path.is_file()) / 1e6
        print(f"Staged {len(inputs['folders'])} folders and {len(code['files'])} code files in {args.out} ({size:.0f} MB); "
              f"{len(code['not_committed'] or [])} code files are not committed")
    elif args.command == "spec":
        from dfl_submission import stack_trial as st
        job = spec(args.name, args.protocol, args.stage, config=st.levers()["config"], networks=st.NETWORKS,
                   sets=tuple(args.sets.split(",")), offsets=tuple(int(value) for value in args.offsets.split(",")),
                   steps=args.steps, blocks=args.blocks, options=json.loads(args.options), note=args.note)
        write_new(args.out, job)
        print(f"Wrote {args.out}: {len(job['runs'])} trainings under {job['protocol']}")
    elif args.command == "kernel":
        print(f"Wrote {args.out} for kernel {kernel(read(args.job), args.out, args.user, parallel=args.parallel)}")
    elif args.command == "colab":
        print("\n".join(colab(read(args.job), args.stage, args.out, args.session, parallel=args.parallel)))
    elif args.command == "colab-upload":
        files = colab_files(args.stage, args.job_folder)
        sent = colab_upload(args.session, files, args.job_folder / "parts", binary=args.binary)
        print(f"{len(files)} files are in /content/lane/in with the right hashes ({len(sent)} sent now)")
    elif args.command == "colab-wait":
        print(f"Downloaded {colab_wait(args.session, args.job_folder, binary=args.binary, poll=args.poll)}")
    else:
        record = import_archive(args.archive, read(args.job), args.stage, args.out, export=not args.no_export)
        print(f"Imported {len(record['runs'])} networks into {args.out} ({record['environment']['gpus']})")


if __name__ == "__main__":
    main()
