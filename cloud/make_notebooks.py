"""Write the D20 Kaggle notebooks (DECISIONS.md D20, cloud amendment): one per cloud family.

Each notebook finds the two private datasets, checks every input hash, installs the pinned training
requirements in a Python 3.12 environment, hashes the code tree, runs that family's four-fold
cross-validation and full training on the GPU with `dfl_submission.zoo`, and zips the outputs with
their record. Run `python cloud/make_notebooks.py` after changing this file.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFESTS = {"train_data": "d3891c24c478584b2ad587c5e3efcc112cb1bef67405046b8db8da953545c2fc",
             "sequences_augmented": "2828b0c31e527426e762251a207c505a75cd740a9c213f8a44ef4c25d57539ed"}

INTRO = """# D20 cloud run: {family}

This notebook cross-validates and trains the **{family}** family exactly as `DECISIONS.md` D20 and its cloud amendment prescribe. It runs the repository's own commands (`python -m dfl_submission.zoo cv`, then `fit`, then `package-cloud`), which refuse to run unless `MTAH_DATA_DIR` points to a directory that does not exist.

**Inputs:** attach two private datasets only.
- `mtah-d20-sequences` holds the prepared external training sequences, derived from DFL (CC BY 4.0) and SkillCorner (MIT) alone.
- `mtah-d20-code` holds the repository code at a committed revision.

No competition data, public ground truth, PFF or Metrica file is used, or exists here.

**Settings:** set the accelerator to *GPU T4 x2* (any CUDA GPU with compute capability 7.0 or above works) and turn on *Internet*, which pip needs. Use *Save Version → Save & Run All*.

**Output:** `d20_{family}_outputs.zip` in the Output tab. Download it and give it to the repo's `zoo import-cloud`, which checks every hash and setting before anything is used.
"""

SETUP = '''FAMILY = "{family}"
EXPECTED_MANIFESTS = {manifests}

import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
RUN = WORK / "d20"
LOGS = RUN / "logs"
LOGS.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args, log, cwd=None, env=None):
    """Run a command with its output in a log file; print the log's tail; stop on failure."""
    with open(log, "w") as handle:
        result = subprocess.run([str(a) for a in args], cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
    print("".join(Path(log).read_text().splitlines(keepends=True)[-12:]))
    if result.returncode:
        raise RuntimeError(f"{{args[:4]}} failed with exit code {{result.returncode}}; see {{log}}")
'''

INPUTS = '''# The two prepared folders, found by their manifest hashes (as uploaded folders or as zips), then every file checked.
def find_inputs(root):
    found = {}
    for manifest in root.rglob("manifest.json"):
        digest = sha256(manifest)
        for key, expected in EXPECTED_MANIFESTS.items():
            if digest == expected:
                found[key] = manifest.parent
    return found

inputs = find_inputs(INPUT)
if set(inputs) != set(EXPECTED_MANIFESTS):
    for archive in INPUT.rglob("*.zip"):
        if archive.stem in EXPECTED_MANIFESTS and archive.stem not in inputs:
            with zipfile.ZipFile(archive) as item:
                item.extractall(WORK / "inputs" / archive.stem)
    inputs |= {key: value for key, value in find_inputs(WORK / "inputs").items() if key not in inputs}
assert set(inputs) == set(EXPECTED_MANIFESTS), f"Prepared folders not found: {sorted(set(EXPECTED_MANIFESTS) - set(inputs))}"

for key, folder in inputs.items():
    records = {r["file"]: r["sha256"] for r in json.loads((folder / "manifest.json").read_text())["records"]}
    present = {path.name for path in folder.iterdir()}
    assert present == set(records) | {"manifest.json"}, f"{folder} holds files outside its manifest"
    wrong = [name for name, expected in records.items() if sha256(folder / name) != expected]
    assert not wrong, f"Hash mismatch in {folder}: {wrong}"
    print(f"{key}: {folder}, manifest {EXPECTED_MANIFESTS[key][:8]}..., {len(records)} files checked")
'''

CODE = '''# The repository code at its committed revision, copied to a writable folder.
commits = list(INPUT.rglob("CODE_COMMIT.txt"))
assert len(commits) == 1, "Attach the mtah-d20-code dataset (exactly once)"
COMMIT = commits[0].read_text().strip()
REPO = WORK / "repo"
if not REPO.exists():
    archives = list(INPUT.rglob("mtah-code.zip"))
    if archives:
        with zipfile.ZipFile(archives[0]) as item:
            item.extractall(REPO)
    else:
        trees = [path.parent.parent for path in INPUT.rglob("dfl_submission/temporal_network.py")]
        assert len(trees) == 1, "The code tree was not found in mtah-d20-code"
        shutil.copytree(trees[0], REPO)
print("Code commit", COMMIT)
'''

ENVIRONMENT = '''# A Python 3.12 environment with the pinned training requirements (PyTorch from PyPI, with CUDA).
run([sys.executable, "-m", "pip", "install", "-q", "uv"], LOGS / "uv.log")
VENV = WORK / "venv"
run([sys.executable, "-m", "uv", "venv", "--python", "3.12", VENV], LOGS / "venv.log")
PY = VENV / "bin" / "python"
run([sys.executable, "-m", "uv", "pip", "install", "--python", PY, "-r", REPO / "dfl_submission" / "requirements-training.txt"],
    LOGS / "install.log")
ENV = dict(os.environ, MTAH_DATA_DIR="/nonexistent/mtah", PYTHONDONTWRITEBYTECODE="1")
ENV.pop("PYTHONPATH", None)
run([PY, "-c", "import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); "
     "assert torch.cuda.is_available(), 'No CUDA GPU: set the accelerator to GPU T4 x2'; "
     "print(torch.cuda.get_device_name(0)); x = torch.randn(256, 256, device='cuda'); print(float((x @ x).abs().sum()) > 0)"],
    LOGS / "gpu.log", cwd=REPO, env=ENV)
run([PY, "-m", "dfl_submission.zoo", "code-tree", "--root", REPO, "--commit", COMMIT, "--out", RUN / "code_tree.json"],
    LOGS / "code_tree.log", cwd=REPO, env=ENV)
print(Path(RUN / "code_tree.json").read_text())
'''

CV = '''# Four-fold match-grouped cross-validation of every setting in the family's D20 grid (seed 42, boundary weight 0.5).
run([PY, "-m", "dfl_submission.zoo", "cv", "--family", FAMILY, "--source", inputs["train_data"], "--scratch", RUN / "cv",
     "--device", "cuda"], LOGS / f"cv_{FAMILY}.log", cwd=REPO, env=ENV)
print(json.loads((REPO / "dfl_submission" / "zoo_recipe" / f"cv_{FAMILY}.json").read_text())["selected"])
'''

FIT = '''# The selected setting's four networks: both model sets, seeds 42 and 97, boundary weights 0.5 and 1.4.
run([PY, "-m", "dfl_submission.zoo", "fit", "--family", FAMILY, "--current", inputs["train_data"],
     "--augmented", inputs["sequences_augmented"], "--scratch", RUN / "networks", "--device", "cuda"],
    LOGS / f"fit_{FAMILY}.log", cwd=REPO, env=ENV)
'''

PACKAGE = '''# The outputs and their record (code, environment, input hashes, seeds, output hashes) in one zip.
OUTPUT = WORK / f"d20_{FAMILY}_outputs.zip"
run([PY, "-m", "dfl_submission.zoo", "package-cloud", "--family", FAMILY, "--work", RUN, "--code", RUN / "code_tree.json",
     "--current", inputs["train_data"], "--augmented", inputs["sequences_augmented"], "--out", OUTPUT],
    RUN / "package.log", cwd=REPO, env=ENV)
print(OUTPUT, sha256(OUTPUT))
'''


def cell(kind, text):
    lines = text.strip("\n").splitlines(keepends=True)
    base = {"cell_type": kind, "metadata": {}, "source": lines}
    return base | ({"execution_count": None, "outputs": []} if kind == "code" else {})


def notebook(family):
    manifests = json.dumps(MANIFESTS, indent=4)
    cells = [cell("markdown", INTRO.format(family=family)), cell("code", SETUP.format(family=family, manifests=manifests)),
             cell("code", INPUTS), cell("code", CODE), cell("code", ENVIRONMENT), cell("code", CV), cell("code", FIT),
             cell("code", PACKAGE)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


if __name__ == "__main__":
    for family in ("gru", "transformer"):
        path = HERE / f"d20_{family}.ipynb"
        path.write_text(json.dumps(notebook(family), indent=1) + "\n")
        print(f"Wrote {path}")
