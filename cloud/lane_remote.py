"""Cloud lane, remote side: run a job's player-set trainings on a cloud GPU and zip the raw networks.

It runs on Kaggle and on Colab (and on the CPU in the tests), and uses the standard library only until
a worker starts. `run` does four things:
1. finds the staged inputs under `--inputs` (the zips `cloud/lane.py stage` wrote, or the folders Kaggle
   makes of them) and checks every file against `INPUTS.json` and `CODE.json`, and those two files
   against the digests the job names;
2. makes a Python 3.12 environment with the pinned training requirements (skipped with `--python`);
3. trains each of the job's networks in its own process (`worker`), `--parallel` at a time, with
   `MTAH_DATA_DIR` pointing nowhere. The worker makes the call `length_trial.train_one` makes on the Mac:
   `stack_set.train` for six temporal blocks, `context_set.train` for seven;
4. zips the raw networks, the logs and `RECORD.json` (the job, the digests, the environment and every
   output hash). `cloud/lane.py import` checks that zip before anything in it is used.

The inputs are prepared training sequences from DFL (CC BY 4.0) and SkillCorner (MIT) and this
repository's code. No competition file, public ground truth, PFF or Metrica file is staged, and none
exists on the cloud machine. CUDA does not reproduce the Mac's weights bit for bit, so a network
trained here is its own reference.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ABSENT = "/nonexistent/mtah"
LANE = 1
RUN_KEYS = {"name", "source", "tokens", "signs", "pitches", "seed", "boundary_boost", "config", "steps", "blocks",
            "options"}
JOB_KEYS = {"name", "protocol", "note", "inputs_digest", "code_digest", "runs"}
OPTION_KEYS = {"width", "lr_scale", "batch_size", "weight_decay", "dropout"}
NAME = re.compile(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def stamp():
    return time.strftime("%H:%M:%S")


def flat(name):
    return name.replace("/", "__")


def check_job(job):
    """The job's shape: named runs with the trainer's arguments and nothing else."""
    if set(job) != JOB_KEYS:
        raise ValueError(f"Job keys differ: {sorted(set(job) ^ JOB_KEYS)}")
    if not NAME.fullmatch(job["name"]) or "/" in job["name"]:
        raise ValueError(f"Unusable job name {job['name']!r}")
    names = [run["name"] for run in job["runs"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("A job needs runs with distinct names")
    for run in job["runs"]:
        if set(run) != RUN_KEYS:
            raise ValueError(f"Run keys differ in {run.get('name')}: {sorted(set(run) ^ RUN_KEYS)}")
        if not NAME.fullmatch(run["name"]) or ".." in run["name"].split("/"):
            raise ValueError(f"Unusable run name {run['name']!r}")
        if run["blocks"] not in (6, 7):
            raise ValueError("A network has six or seven temporal blocks")
        if not set(run["options"]) <= OPTION_KEYS:
            raise ValueError(f"Unknown training options: {sorted(set(run['options']) - OPTION_KEYS)}")
        if run["config"].get("aux"):
            raise ValueError("The lane trains no auxiliary head")


# -- inputs --------------------------------------------------------------------------------------------

def locate(roots, filename):
    found = sorted(path for root in roots for path in Path(root).rglob(filename) if path.is_file())
    if not found:
        raise FileNotFoundError(f"{filename} was not found under {[str(root) for root in roots]}")
    return found[0]


def fetch_inputs(roots, work, job):
    """`work/data` holding every staged folder and file (extracted, or linked where Kaggle unzipped it), checked."""
    inputs_file = locate(roots, "INPUTS.json")
    if sha256(inputs_file) != job["inputs_digest"]:
        raise ValueError("INPUTS.json is not the staged inputs this job names")
    inputs = json.loads(inputs_file.read_text())
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    for key, entry in inputs["folders"].items():
        target = data / key
        if not target.exists():
            archives = sorted(path for root in roots for path in Path(root).rglob(f"{key}.zip"))
            if archives:
                with zipfile.ZipFile(archives[0]) as item:
                    item.extractall(target)
            else:
                manifests = [path for root in roots for path in Path(root).rglob("manifest.json")
                             if sha256(path) == entry["manifest"]]
                if not manifests:
                    raise FileNotFoundError(f"The staged folder {key} was not found")
                target.symlink_to(manifests[0].parent.resolve(), target_is_directory=True)
        present = {path.name for path in target.iterdir()}
        if present != set(entry["files"]):
            raise ValueError(f"{key} holds files outside the staged list: {sorted(present ^ set(entry['files']))[:5]}")
        wrong = [name for name, expected in entry["files"].items() if sha256(target / name) != expected]
        if wrong:
            raise ValueError(f"Hash mismatch in {key}: {wrong[:5]}")
        print(f"{stamp()} {key}: {len(entry['files'])} files checked", flush=True)
    for name, expected in inputs["files"].items():
        target = data / name
        if not target.exists():
            shutil.copyfile(locate(roots, name), target)
        if sha256(target) != expected:
            raise ValueError(f"Hash mismatch in {name}")
    return data, inputs


def fetch_code(roots, work, job):
    """`work/repo` holding exactly the staged code files, checked."""
    code_file = locate(roots, "CODE.json")
    if sha256(code_file) != job["code_digest"]:
        raise ValueError("CODE.json is not the staged code this job names")
    code = json.loads(code_file.read_text())
    repo = work / "repo"
    if not repo.exists():
        archives = sorted(path for root in roots for path in Path(root).rglob("code.zip"))
        if archives:
            with zipfile.ZipFile(archives[0]) as item:
                item.extractall(repo)
        else:
            trees = sorted(path.parents[1] for root in roots for path in Path(root).rglob("lane_remote.py")
                           if path.parent.name == "cloud" and (path.parents[1] / "dfl_submission").is_dir())
            if not trees:
                raise FileNotFoundError("The staged code was not found")
            shutil.copytree(trees[0], repo)
    present = {str(path.relative_to(repo)) for path in repo.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    if present != set(code["files"]):
        raise ValueError(f"The code tree differs from the staged list: {sorted(present ^ set(code['files']))[:5]}")
    wrong = [name for name, expected in code["files"].items() if sha256(repo / name) != expected]
    if wrong:
        raise ValueError(f"Hash mismatch in the code: {wrong[:5]}")
    print(f"{stamp()} code: {len(code['files'])} files checked", flush=True)
    return repo, code


# -- environment ---------------------------------------------------------------------------------------

PROBE = """
import json, platform, sys
import torch
names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
print(json.dumps({"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
                  "cuda": torch.version.cuda, "gpus": names}))
"""


def quiet(command, log):
    with open(log, "w") as handle:
        result = subprocess.run([str(part) for part in command], stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        print("".join(Path(log).read_text().splitlines(keepends=True)[-20:]), flush=True)
        raise RuntimeError(f"{command[:4]} failed with exit code {result.returncode}")


def environment(work, repo, python, device, logs):
    """The interpreter that trains, and what it runs on. Without `python`: a fresh 3.12 venv with the pinned requirements."""
    if python is None:
        quiet([sys.executable, "-m", "pip", "install", "-q", "uv"], logs / "uv.log")
        venv = work / "venv"
        if not venv.exists():
            quiet([sys.executable, "-m", "uv", "venv", "--python", "3.12", venv], logs / "venv.log")
        python = venv / "bin" / "python"
        quiet([sys.executable, "-m", "uv", "pip", "install", "--python", python, "-r",
               repo / "dfl_submission" / "requirements-training.txt"], logs / "install.log")
    result = subprocess.run([str(python), "-c", PROBE], capture_output=True, text=True, check=True)
    found = json.loads(result.stdout.strip().splitlines()[-1])
    if device == "cuda" and not found["gpus"]:
        raise RuntimeError("No CUDA GPU on this machine; choose a GPU runtime")
    return Path(python), dict(found, device=device, cpus=os.cpu_count())


# -- trainings -----------------------------------------------------------------------------------------

def train_all(job, python, repo, data, runs_root, logs, device, parallel, gpus):
    """Every run not yet finished, `parallel` at a time, each in its own process. Returns {name: seconds}."""
    pending = [run for run in job["runs"] if not (runs_root / run["name"] / "fit.json").exists()]
    slots, active, seconds, failed = list(range(parallel)), [], {}, []
    environment_ = dict(os.environ, MTAH_DATA_DIR=ABSENT, PYTHONDONTWRITEBYTECODE="1")
    environment_.pop("PYTHONPATH", None)
    while pending or active:
        while pending and slots:
            run, slot = pending.pop(0), slots.pop(0)
            folder = runs_root / run["name"]
            if folder.exists():
                shutil.rmtree(folder)                       # an unfinished run of an earlier attempt
            folder.parent.mkdir(parents=True, exist_ok=True)
            env = dict(environment_)
            if device == "cuda" and gpus > 1:
                env["CUDA_VISIBLE_DEVICES"] = str(slot % gpus)
            log = open(logs / f"{flat(run['name'])}.log", "w")
            process = subprocess.Popen([str(python), str(Path(__file__).resolve()), "worker", "--repo", str(repo), "--data",
                                        str(data), "--out", str(folder), "--device", device, "--run", json.dumps(run)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env, cwd=repo)
            active.append((run["name"], slot, process, log, time.monotonic()))
            print(f"{stamp()} started {run['name']}", flush=True)
        time.sleep(1)
        for item in list(active):
            name, slot, process, log, started = item
            if process.poll() is None:
                continue
            log.close()
            active.remove(item)
            slots.append(slot)
            seconds[name] = round(time.monotonic() - started, 1)
            if process.returncode or not (runs_root / name / "fit.json").exists():
                failed.append(name)
                tail = (logs / f"{flat(name)}.log").read_text().splitlines()[-25:]
                print(f"{stamp()} FAILED {name}\n" + "\n".join(tail), flush=True)
            else:
                print(f"{stamp()} finished {name} in {seconds[name]:.0f} s", flush=True)
    if failed:
        raise RuntimeError(f"Trainings failed: {failed}")
    return seconds


def package(job, environment_, runs_root, logs, out, seconds):
    """The zip `cloud/lane.py import` reads: RECORD.json, every run's files and the logs."""
    record = {"lane": LANE, "job": job, "job_digest": json_digest(job), "environment": environment_, "runs": {}}
    members = []
    for run in job["runs"]:
        folder = runs_root / run["name"]
        files = sorted(path for path in folder.iterdir() if path.is_file())
        record["runs"][run["name"]] = {"files": {path.name: sha256(path) for path in files},
                                       "seconds": seconds.get(run["name"])}
        members += [(path, f"runs/{run['name']}/{path.name}") for path in files]
    members += [(path, f"logs/{path.name}") for path in sorted(logs.iterdir()) if path.is_file()]
    archive = out / f"lane_outputs_{job['name']}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as item:
        item.writestr("RECORD.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
        for path, name in members:
            item.write(path, name)
    return archive


def run_job(args):
    job = json.loads(args.job.read_text())
    check_job(job)
    work, out = args.work, args.out
    logs = work / "logs"
    for folder in (work, out, logs):
        folder.mkdir(parents=True, exist_ok=True)
    state = args.state
    try:
        print(f"{stamp()} job {job['name']} ({job['protocol']}): {len(job['runs'])} trainings", flush=True)
        data, _ = fetch_inputs(args.inputs, work, job)
        repo, _ = fetch_code(args.inputs, work, job)
        python, found = environment(work, repo, args.python, args.device, logs)
        print(f"{stamp()} environment: {json.dumps(found)}", flush=True)
        gpus = len(found["gpus"])
        parallel = args.parallel or max(1, min(len(job["runs"]), os.cpu_count() or 1))
        seconds = train_all(job, python, repo, data, work / "runs", logs, args.device, parallel, gpus)
        archive = package(job, dict(found, parallel=parallel), work / "runs", logs, out, seconds)
        line = f"done {archive.name} {sha256(archive)}"
    except BaseException as error:
        if state is not None:
            state.write_text(f"failed {type(error).__name__}: {error}\n")
        raise
    print(f"{stamp()} {line}", flush=True)
    if state is not None:
        state.write_text(line + "\n")


# -- one training --------------------------------------------------------------------------------------

def worker(args):
    """One network, by the call `length_trial.train_one` makes; then the list of repository modules it loaded."""
    repo = args.repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "experiments")]
    os.chdir(repo)
    from dfl_submission.external_only import require_external_only
    require_external_only()
    run = json.loads(args.run)
    from dfl_submission import stack_set as ss
    from experiments.dfl_transfer_data import digest
    config = run["config"]
    ss.validate(config)
    if config["aux"]:
        raise ValueError("The lane trains no auxiliary head")
    data = args.data
    source, tokens = data / run["source"], data / run["tokens"]
    signs_file = data / run["signs"] if config["direction"] else None
    signs = ss.read_signs(signs_file, source, tokens) if config["direction"] else None
    sign_map = None
    if config["mirror"] is not None:
        record = json.loads((data / "parity.json").read_text())
        if not record["passed"]:
            raise ValueError("The mirror parity check did not pass; nothing is trained with mirrors")
        sign_map = record["sign_map"]
    if args.device.startswith("cuda"):
        import torch
        torch.backends.cudnn.deterministic = True      # the same algorithms on every run of the same machine
        torch.backends.cudnn.benchmark = False
    common = dict(seed=run["seed"], boundary_boost=run["boundary_boost"], config=config, signs=signs, sign_map=sign_map,
                  pitch_file=data / run["pitches"] if config["noise"] else None,
                  provenance={"signs_file": None if signs_file is None else digest(signs_file)}, steps=run["steps"],
                  **run["options"])
    if run["blocks"] == 6:
        ss.train(source, tokens, args.out, args.device, **common)
    else:
        from dfl_submission import context_set as cs
        cs.train(source, tokens, args.out, args.device, dilations=cs.SEVEN, **common)
    loaded = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)          # some torch modules carry a bare file name: not ours
        if path and Path(path).is_absolute() and Path(path).resolve().is_relative_to(repo):
            loaded.add(str(Path(path).resolve().relative_to(repo)))
    (args.out / "lane_modules.json").write_text(json.dumps(sorted(loaded)) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run")
    running.add_argument("--inputs", type=Path, nargs="+", required=True, help="folders holding the staged inputs")
    running.add_argument("--work", type=Path, required=True, help="scratch folder (not saved)")
    running.add_argument("--out", type=Path, required=True, help="where the output zip goes")
    running.add_argument("--job", type=Path, required=True)
    running.add_argument("--python", type=Path, help="use this interpreter instead of making an environment")
    running.add_argument("--device", default="cuda")
    running.add_argument("--parallel", type=int, default=0, help="trainings at once (default: one per CPU)")
    running.add_argument("--state", type=Path, help="write `done ...` or `failed ...` here at the end")
    working = commands.add_parser("worker")
    working.add_argument("--repo", type=Path, required=True)
    working.add_argument("--data", type=Path, required=True)
    working.add_argument("--out", type=Path, required=True)
    working.add_argument("--device", required=True)
    working.add_argument("--run", required=True, help="the run as JSON")
    args = parser.parse_args()
    {"run": run_job, "worker": worker}[args.command](args)


if __name__ == "__main__":
    main()
