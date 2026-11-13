# Cloud lane: player-set trainings on Kaggle and Colab

The Mac trains two networks at a time. The lane trains more of them on Kaggle or Colab GPUs, so they do not count against that limit. It runs the repository's own trainers (`stack_set.train`, `context_set.train`) and decides nothing: every job names the `DECISIONS.md` rule it runs under, and `lane.py spec` refuses a rule that has no heading there yet.

**Status (2026-11-13):** both services have run the lane end to end, and both imports passed every check. The timing job `smoke` trained four networks (the D48 recipe at seed offset 0, 5,400 steps). They were never scored and sit in `data/cloud_lane/imported/`.

| Where | Machine | Trainings at once | Seconds per network | Networks per hour |
|---|---|---|---|---|
| Mac | M1 Pro, MPS | 2 | about 570 | about 12 |
| Kaggle, one session | 2 × Tesla T4, 4 CPUs | 4 | 599 to 605 | about 24 |
| Kaggle, one session | 2 × Tesla T4, 4 CPUs | 8 | 1,308 to 1,320 | about 22 |
| Colab, free account | 1 × Tesla T4, 2 CPUs | 2 | 616 to 637 | about 11 |

- **One training per CPU is the limit.** Eight at once in one Kaggle session (job `smoke8`) took more than twice as long each as four at once, so the default of one training per CPU stays. Two Kaggle sessions at once were not tried here (D20 ran two).
- **Quota:** Kaggle charges the session's wall time: 0.6 of the week's 30 GPU hours for both jobs (12 networks, 36 minutes).
- **Start-up:** checking the inputs took about 30 s and the environment 60 to 90 s on each service. A Colab session also needs the 1,150 MB uploaded first (11 minutes here); Kaggle keeps its datasets.
- **Reproducibility:** the same four networks came out bit for bit three times: on Kaggle four at once, on Kaggle eight at once, and on Colab (Tesla T4, PyTorch 2.15.0+cu132). They differ from the Mac's MPS networks, as expected.
- **Tests:** `tests/test_cloud_lane.py`, 9 tests on synthetic data, including a full round trip on the CPU.

## What goes up, and what never does

- **Uploaded** (private):
  - the prepared training sequences and player tokens of both model sets, with their sign files, pitch files and the mirror parity record. For the stack recipe these are D49's folders: 1,150 MB, 80 records per set, every record from DFL (CC BY 4.0) or SkillCorner (MIT);
  - the repository's Python files (`dfl_submission/*.py`, `experiments/*.py`, `evaluation/*.py`, `starter/*.py`, `games.py`, `cloud/lane_remote.py`), the pinned training requirements and the two licence files. No tests, weights, recipes or data.
- **Never uploaded:** the competition bundle or anything derived from it, the public ground truth, PFF files, Metrica files, any cache under `data/`, `rules/`, `AGENTS.md`.

`lane.py stage` enforces this. It refuses a record whose source is not `dfl` or `skillcorner`, a file outside a folder's manifest, a hash mismatch, and any file name that points at the competition bundle, PFF or Metrica. Every command refuses to run unless `MTAH_DATA_DIR` points to a path that does not exist.

## What CUDA changes

CUDA does not reproduce the Mac's weights bit for bit, so a network trained on the lane is not the network the Mac would train. The lane's own runs do repeat: the timing job gave identical weights on Kaggle and on Colab (the worker asks cuDNN for deterministic algorithms). That was seen once, on Tesla T4s with one PyTorch build; another GPU model or build may differ.

- Compare like with like: a candidate and the baseline it is measured against should come from the same hardware.
- For weights that ship, the cautious choice is to retrain the winner on the Mac, so the shipped networks still retrain exactly here (D20 kept its final trees local for the same reason). Shipping cloud weights instead is a decision for Derek and needs its own line in the technical document.

## One-time setup

Both CLIs are installed as isolated uv tools (`~/.local/bin/kaggle`, `~/.local/bin/colab`); the repo `.venv` is untouched. The sign-ins are Derek's:

```sh
kaggle auth login
```

```sh
colab sessions
```

The first opens Kaggle's browser sign-in. The second prints a Google link on first use: approve in the browser, then paste the authorization code the page shows back into the terminal; it then lists the account's sessions. Both were done on 2026-11-13. `kaggle quota` shows the weekly GPU hours (30, reset on Saturdays 00:00 UTC); `colab usage` shows the compute-unit balance (0 on the free account, which still got a T4).

## Steps

All commands run from the repository root with `MTAH_DATA_DIR=/nonexistent/mtah` and the repo `.venv`.

1. **Stage** (needs about 1.2 GB of free disk; the zips can be deleted after the upload, but keep `INPUTS.json`, `CODE.json` and `LOCAL.json`):
   ```sh
   .venv/bin/python cloud/lane.py stage --external /path/to/research/data --user KAGGLE_NAME
   ```
   This writes `data/cloud_lane/stage/mtah-lane-data/` and `.../mtah-lane-code/`. Stage again (into a new folder) whenever the training code or the recipe's data changes: a job is tied to one stage by two digests.

2. **Write the job.** The rule comes first: the orchestrator writes it in `DECISIONS.md`, then
   ```sh
   .venv/bin/python cloud/lane.py spec --name d53-seeds --protocol D53 --offsets 0,1000,2000 --out data/cloud_lane/d53-seeds.json
   ```
   A run is named `<seed offset>/<set>/<network folder>`, as D48's folders are. `--steps`, `--blocks 7`, `--sets` and `--options` change the recipe; the switches come from `stack_recipe/levers.json`. `stack_set.validate` allows the seed offsets 0, 1000 and 2000 only, so more seeds need a code change by the round's coders first.

3. **Kaggle.**
   ```sh
   kaggle datasets create -p data/cloud_lane/stage/mtah-lane-data
   kaggle datasets create -p data/cloud_lane/stage/mtah-lane-code
   .venv/bin/python cloud/lane.py kernel --job data/cloud_lane/d53-seeds.json --user KAGGLE_NAME --out data/cloud_lane/kaggle/d53-seeds
   kaggle kernels push -p data/cloud_lane/kaggle/d53-seeds --accelerator NvidiaTeslaT4
   kaggle kernels status KAGGLE_NAME/mtah-lane-d53-seeds
   kaggle kernels output KAGGLE_NAME/mtah-lane-d53-seeds -p data/cloud_lane/in/d53-seeds
   ```
   Datasets are private unless `--public` is passed; never pass it. A changed stage goes up with `kaggle datasets version -m "..." -p <folder>`. The kernel is private, with GPU and Internet on (pip needs Internet).

4. **Colab.**
   ```sh
   .venv/bin/python cloud/lane.py colab --job data/cloud_lane/d53-seeds.json --session d53 --out data/cloud_lane/colab/d53-seeds
   ```
   This prints (and writes to `COMMANDS.txt`) the commands to run in order: `colab new`, `lane.py colab-upload`, `lane.py colab-wait`, `colab stop`.
   - `colab upload` refuses large files (20 MB went through, 80 MB did not), so `colab-upload` sends anything over 24 MB in parts, joins the parts on the machine and compares every file's SHA-256 with the local one. A file already there with the right hash is not sent again.
   - `colab-wait` runs the launcher once a minute. The first run starts the job detached from the kernel; every run prints the end of the log and the job's state and returns at once. When the job is done it downloads the output zip and checks its hash. It can be stopped and started again. Nothing waits inside the kernel, because `colab exec` gives up on a cell that prints nothing for 30 s.
   - A free account got a T4 with 2 CPUs; the CLI's guide says GPUs are not guaranteed. `colab new` fails with a 400 when the account may not have one.
   - A Colab machine is new each time, so the 1,150 MB go up once per session. Always finish with `colab stop`.

5. **Import.**
   ```sh
   .venv/bin/python cloud/lane.py import --archive data/cloud_lane/in/d53-seeds/lane_outputs_d53-seeds.zip --job data/cloud_lane/d53-seeds.json --out data/d53/networks
   ```
   The import checks, before anything is used:
   - the zip is this job's, and the job names this stage's inputs and code;
   - the staged folders and files are unchanged here;
   - every output file hashes as its record says, and each fit record's hashes match its files;
   - each run's seed, boundary weight, steps, width, learning-rate scale, batch, weight decay, dropout, switches, sign file, temporal blocks and training manifests are the job's;
   - every repository module the run loaded is byte-identical here now.

   It then writes the raw networks to `<out>/raw/<run>` and exports each to `<out>/export/<run>` with the NumPy check (`player_set export`, or `context_set export` for seven blocks), and records the import in `<out>/IMPORT_<job>.json`. Scoring stays local and follows the job's rule.

## Notes

- **Parallel trainings.** The remote side starts one training per CPU unless `--parallel` is given, and spreads them over the GPUs it finds. Batch assembly is NumPy on the CPU, and the CPU count sets the speed (see the table above), so leave `--parallel` at its default.
- **Disk.** The staged zips (1.1 GB in `data/cloud_lane/stage/mtah-lane-data/`) are needed again for every Colab session; Kaggle keeps its own copy. Delete them if space is short and stage again later, but keep `INPUTS.json`, `CODE.json` and `LOCAL.json`: the import needs them.
- **Failures.** A run that stops is trained again from the start when the job is run again on the same machine; a finished run is kept. An output zip that fails the import is not used.
- **Local commands.** `cloud/lane.py` and `cloud/lane_remote.py` are not `-m dfl_submission.…` commands, so the Mac's training counters (`stack_trial.running_trainings`) do not see them.
