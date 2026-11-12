# D20 cloud runs: GRU and transformer on Kaggle

`DECISIONS.md` D20 searches four model families. Its cloud amendment moves two of them, the bidirectional GRU and the transformer encoder, to Kaggle GPUs. Its tree amendment moves the tree heads' cross-validation to a Kaggle CPU session; their final trees are still fit on the Mac, so they refit byte for byte. The temporal network runs on the Mac. The grids, folds, scoring and selection are D20's. Everything after training stays local: the NumPy export check, development scoring, the vetoes, the ensemble and the confirmation.

Derek uploads and runs everything here. Claude never runs the notebooks.

## What goes to Kaggle, and what never does

- **Uploaded** (two private datasets):
  - `mtah-d20-sequences`: the two prepared training-sequence folders, derived only from DFL (CC BY 4.0) and SkillCorner (MIT). They are `.../more-than-a-hack-2027/data/external_temporal/train_data` (manifest `d3891c24…`) and `data/d13/sequences_augmented` (manifest `2828b0c3…`).
  - `mtah-d20-code`: `git archive` of this repository's code paths at a committed revision.
- **Never uploaded:** the competition bundle (`data/tactical`, `data/broadcast`, any `BAR-ATH*` file) or anything derived from it, the public ground truth, PFF files, the Metrica games, and every cache under `data/` (several contain PFF or Metrica halves).

`make_bundles.sh` stages only the files listed above, and only after checking every input hash against the published training manifests.

## Steps, in order

1. **Commit Track A's files first.** That means `dfl_submission/zoo.py`, `tests/test_zoo.py`, `cloud/` and the D20 section of `DECISIONS.md`. The cloud run records the code commit, and the repo checks it before use. `make_bundles.sh` refuses to run if `zoo.py` is not committed, or if anything under the code paths has uncommitted changes.

2. **Stage the datasets.** Set your Kaggle user name, so the dataset metadata carries it:
   ```sh
   export KAGGLE_USERNAME=your-kaggle-name
   D20_TRAIN_DATA=/path/to/research/data/external_temporal/train_data ./cloud/make_bundles.sh
   ```
   This writes two folders:
   - `data/d20/kaggle/mtah-d20-sequences/` (about 455 MB): `train_data.zip`, `sequences_augmented.zip`, `README.md`, `dataset-metadata.json`;
   - `data/d20/kaggle/mtah-d20-code/`: `mtah-code.zip`, `CODE_COMMIT.txt`, `README.md`, `dataset-metadata.json`.

3. **Upload both as private datasets.** With the Kaggle CLI, datasets are private unless you pass `--public`, so do not pass it:
   ```sh
   kaggle datasets create -p data/d20/kaggle/mtah-d20-sequences
   kaggle datasets create -p data/d20/kaggle/mtah-d20-code
   ```
   Or use the web: go to kaggle.com, then *Datasets*, then *New Dataset*. Drop in the files of each folder (all except `dataset-metadata.json`) and keep the dataset *Private*. Kaggle may unzip the archives; the notebooks accept either form.

4. **Run the two notebooks.** They can run at the same time, as two Kaggle sessions. Do this for `cloud/d20_gru.ipynb`, then for `cloud/d20_transformer.ipynb`:
   1. Go to kaggle.com, then *Code*, then *New Notebook*. Choose *File → Import Notebook* and upload the `.ipynb`.
   2. Under *Add Input*, attach both datasets: `mtah-d20-sequences` and `mtah-d20-code`.
   3. In *Settings*, set the accelerator to **GPU T4 x2** and turn **Internet on**. Internet is needed for pip, and needs a phone-verified Kaggle account. Do not pick the P100: current PyTorch wheels do not support it.
   4. Choose *Save Version → Save & Run All (Commit)*. The notebook checks every input hash, installs the pinned training requirements (Python 3.12, PyTorch 2.15 with CUDA), cross-validates the family's four settings over four folds, trains the selected setting's four networks and packages everything. Expect roughly one to two hours per notebook.
   5. When the version finishes, open its *Output* tab and download `d20_gru_outputs.zip` (or `d20_transformer_outputs.zip`).

5. **Bring the zips back.** Put them in `data/d20/cloud_in/` and tell the Track A session. It imports each one:
   ```sh
   export MTAH_DATA_DIR=/nonexistent/mtah
   R=/path/to/research/data   # the folder that holds external_temporal/train_data
   .venv/bin/python -m dfl_submission.zoo import-cloud --family gru --archive data/d20/cloud_in/d20_gru_outputs.zip \
     --current $R/external_temporal/train_data --augmented data/d13/sequences_augmented --scratch data/d20/cloud
   ```
   The import checks:
   - every output hash;
   - the input hashes against the local manifests;
   - that the code tree is the committed revision, and that the training code in it is the code here;
   - every run's setting, seed, boundary weight and training manifest;
   - that the selection follows from the cross-validation record.

   It then exports each network locally with the NumPy check. Development scoring follows (`zoo member --family gru`).

## The tree heads' cross-validation (`cloud/d20_trees.ipynb`)

It runs the same way as the network notebooks, with three differences:
- **Code dataset:** it needs a new version of `mtah-d20-code`, built by `make_bundles.sh` at the commit that adds the tree notebook. The sequences dataset is unchanged and is not uploaded again.
- **Session:** no accelerator (a CPU session), with Internet on. It installs the build requirements (`dfl_submission/requirements.txt`, no PyTorch).
- **Output:** `d20_trees_outputs.zip`, holding the cross-validation record, its logs, and the hashes of the 24 cross-validation models (the models themselves are not shipped).

Import it with `zoo import-cloud --family trees …`. The local chain then fits the selected setting's four trees on the Mac and scores the member.

The GRU and transformer runs were made with the code at `c3cf4b3`. Their import still checks their own training code only (`zoo.TRAINING`), and a test keeps that code identical to `c3cf4b3`.

## Timing and failures

- **Deadline:** a family's outputs must pass the import by **2026-11-13 08:00 EST**. Otherwise that family runs locally instead, and later Kaggle outputs are not used (D20 amendment).
- **A notebook that stops before writing its zip** may be run again. A run whose outputs fail a check is not repeated; its family is left out.
- **Weights:** CUDA training is not bit-reproducible, so the imported weights are the reference.

To regenerate the notebooks after editing `make_notebooks.py`: `python cloud/make_notebooks.py`.
