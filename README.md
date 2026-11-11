# Team possession from temporal ensembles and joint decoding

This repository regenerates team Anfield's possession file, `Anfield.csv`, for the More Than A Hack 2027 Initial Challenge. The file labels each frame of the tactical video as `H` (home possession), `A` (away possession) or `D` (dead ball), and writes `X` outside the two official halves.

**All training, tuning and model selection use external matches about other games.** The competition bundle is read only by frozen inference. Nothing in this repository reads the public 5-minute ground truth, the provided video, Smart Tagging or the broadcast recording.

## How it works

1. **Inputs.** The build reads three files from the competition bundle: the tactical ATD tracking and its metadata, and the tactical homography's `valid` flag. Coordinates are converted to metres with the 105 × 68 m pitch reported by the ATD metadata. Invalid frames are masked, ball positions far outside the pitch are discarded, and each half is processed separately.
2. **Features.** 162 team-symmetric features are computed from relative geometry and motion in metres:
   - nearest-player distances and local player counts around the ball
   - sustained player-to-ball contact
   - ball speed and pitch-boundary distance
   - team shape
   - context windows of 3, 9 and 18 seconds
3. **Models.**
   - An extended-context XGBoost model estimates home ownership given live play, and the probability of a stoppage.
   - Two temporal convolutional networks estimate ownership, stoppage and possession-boundary probabilities. Each has 96 channels per block and is trained for 5,400 steps (D15, below). The second network is trained independently with seed 97 and a heavier weight near labelled boundaries. Inference uses NumPy exports of their weights, so the build does not need PyTorch.
   - The seed-97 network predicts on three 5 Hz grids, offset by 0, 2 and 4 frames. Its predictions are interpolated back to 25 Hz and averaged with the reference signals using equal weights.
   - The whole set of models exists twice: once trained on the external data as provided, and once retrained after removing the providers' dead-ball artifacts. Every output of the two sets is averaged with equal weight. See "Dead-ball artifacts and the model blend" below.
4. **Decoder.** One three-state Viterbi decoder picks the `H`/`A`/`D` sequence for each half.
   - The stoppage cutoff is 0.53.
   - A direct ownership change costs 8, and a transition into or out of a stoppage costs 2. Each cost is cut by up to 80% when the matching boundary signal is strong.
   - The high ownership cost reflects the challenge's definition: only *controlled* possession changes hands, so a brief touch or deflection should not split a possession. See "Decoder retune" below; D15 re-selected the cost for the larger networks.
   - In loose or contested play, the last team in possession keeps the ball unless the evidence supports a change.

## Training data

| Dataset | Source and licence | Who labelled it | Amount | Role |
| --- | --- | --- | --- | --- |
| DFL / Sportec integrated tracking and events (Bassek, Rein, Weber and Memmert) | [doi:10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | DFL data provider: ball status and ball-owning team in the position feed | 7 Bundesliga matches. WMX, WN1, WOH and WOY are used for fitting, WPY for development, and WQQ and WR9 for confirmation. | Fit, tune, validate |
| SkillCorner Open Data | [revision 4340d274](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656), MIT ([LICENSE-SKILLCORNER](LICENSE-SKILLCORNER)) | SkillCorner: phases of play and provider possession on detected broadcast tracking | 20 A-League matches: 12 for fitting, 4 for development and 4 for confirmation | Fit, tune, validate |

None of the training data comes from the provided game. Both halves of each match always stay in the same split. The DFL training views add synthetic degradation to the tracking: coordinate noise, missing players, runs of missing ball observations, false ball observations and camera outages. The retrained model set also varies the ball visibility per half, simulates dead-ball handling and adds cross-team identity swaps (`dfl_submission/augment.py`).

## External evaluation

The development search compared 238 ensemble recipes on one DFL match and four SkillCorner development matches. The settings were frozen before confirmation on the other two DFL matches and four SkillCorner matches.

| External confirmation (spellF1) | Previous reference | This ensemble |
| --- | ---: | ---: |
| DFL (1/3 clean, 2/3 degraded tracking) | 0.5672 | 0.5740 |
| SkillCorner (detected broadcast tracking) | 0.5964 | 0.6094 |
| Combined (equal weight) | 0.5818 | 0.5917 |

These confirmation matches also informed earlier comparisons between model families. Treat them as development evidence rather than an untouched test. The competition score is unknown. Full records:
- [ensemble development search](dfl_submission/ensemble_recipe/development.json)
- [per-match confirmation](dfl_submission/ensemble_recipe/external_validation.json)
- the earlier stages in `dfl_submission/temporal_recipe/` and `dfl_submission/context_recipe/`

### Decoder retune

The external labels change the owning team on every opponent touch. The challenge counts only controlled possession. So we scored the decoder against external targets re-expressed under that definition: within each live segment, an A–B–A sequence whose interior B lasts under N seconds is merged back into A (`dfl_submission/conventions.py`). Dead-ball labels are unchanged.

We take N = 3 s. External sources put the plausible value between 1.75 s and 4.5 s: Metrica sample events, raw and with a control rule, and SkillCorner phases of play. N = 3 s minimises the worst-case loss over that interval.

The grid, the selection rule and the guards were written down in [DECISIONS.md](DECISIONS.md) (D1) before scoring. Development selected an ownership cost of 6 and kept the stoppage cost at 2. Confirmation on the holdout:

| Holdout combined spellF1 | Costs 2 / 2 | Costs 6 / 2 | Change |
| --- | ---: | ---: | ---: |
| Targets with N = 3 s (primary) | 0.6782 | 0.6923 | +0.0141 |
| N = 1.75 s (guard) | 0.6360 | 0.6361 | +0.0001 |
| N = 4 s (guard) | 0.6759 | 0.7022 | +0.0262 |
| N = 4.5 s (guard) | 0.6638 | 0.6961 | +0.0324 |
| Raw provider labels (reported only) | 0.5917 | 0.5794 | −0.0123 |

All eight holdout match views improved on the primary target. Records: [development grid](dfl_submission/decoder_recipe/development.json) · [confirmation](dfl_submission/decoder_recipe/external_validation.json)

### Dead-ball artifacts and the model blend

Both external providers mark dead ball in a way the competition tracking will not. DFL freezes the ball on every dead frame. SkillCorner detects the ball on only about 18% of dead frames, against about 91% of live frames. A model trained on them can learn that a still or missing ball means a stoppage.

We retrained every model on the same matches after changing the training data only ([DECISIONS.md](DECISIONS.md) D2):
- dead-ball runs keep the ball, lose it, or have it carried by the restart taker to the restart spot;
- ball visibility in the degraded DFL view is drawn per half from 30–90%, instead of a fixed 60%;
- home and away tracks occasionally swap positions, as in a cross-team identity switch.

We added two guard views to the external DFL matches, with settings that differ from the training augmentation:
- **dead ball carried:** the visible dead-frame ball follows the player nearest to it;
- **team swap:** 10-second cross-team swaps.

The retrained models alone gained +0.090 with the dead ball carried, but lost 0.015 on the standard views, which still contain the artifacts. That missed the pre-registered limit of 0.01, so they were not adopted on their own.

An equal blend of the original and retrained models was then tested under a new rule ([DECISIONS.md](DECISIONS.md) D3). This was a second attempt on the same development matches, and the record says so. Against targets with N = 3 s:

| Holdout spellF1 | Current models | Blend | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Standard views, combined | 0.6923 | 0.6873 | −0.0050 | ≥ −0.01 |
| Dead ball carried (DFL) | 0.5364 | 0.5980 | +0.0616 | ≥ +0.03 |
| Team swap (DFL) | 0.5646 | 0.5736 | +0.0090 | ≥ −0.01 |

The blend keeps the stoppage cutoff at 0.53. Records: [retrained models](dfl_submission/retrain_recipe/development.json) · [blend development](dfl_submission/blend_recipe/development.json) · [blend confirmation](dfl_submission/blend_recipe/external_validation.json)

### Larger, longer-trained networks (D15)

Each network originally had 48 channels per block (about 32,000 parameters) and trained for 1,800 steps. Neither value had been searched. D15 retrained all four networks of both model sets at widths 48 and 96 and lengths of 1,800 and 5,400 steps, with the learning-rate schedule stretched to the new length. The ownership cost was re-selected from 4, 6 and 8. The trees, features, seeds and training data are unchanged. Every new setting cleared every development gate. Development selected width 96, 5,400 steps and ownership cost 8 (+0.041 on the primary). Confirmation, once, against D3:

| Holdout spellF1 | D3 | D15 | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Standard views, N = 3 s (primary) | 0.6873 | 0.7087 | +0.0214 | ≥ +0.01 |
| Dead ball carried (DFL) | 0.5980 | 0.6104 | +0.0124 | ≥ −0.01 |
| Team swap (DFL) | 0.5736 | 0.6322 | +0.0586 | ≥ −0.01 |
| PFF confirmation set A, N = 3 s (fresh vendor) | 0.6355 | 0.6542 | +0.0187 | ≥ −0.01 |
| Raw provider labels (reported only) | 0.5746 | 0.6179 | +0.0433 | — |

N = 1.75, 4 and 4.5 s rose by +0.039, +0.013 and +0.005. The holdout gain is about half the development gain, as expected after picking the best of nine eligible settings on reused matches. Records: [development](dfl_submission/scale_recipe/development.json) · [confirmation](dfl_submission/scale_recipe/confirmation.json) · [protocol](DECISIONS.md#d15--network-width-and-training-length-2026-11-11-written-before-any-candidate-network-is-trained-or-scored)

### Set-piece restart experiment

D4 tested ten decoder settings on DFL WPY using simulated Smart Tagging inputs from the DFL event feed. The simulations included missing tags, timing errors and spurious tags. On the masked and dead-ball-carried views, the best setting scored 0.61376 spellF1 against the N = 3 targets, compared with 0.61382 for D3. The required gain was +0.003, so the experiment ended at development. See the [protocol](DECISIONS.md#d4--smart-tagging-restart-anchors-2026-11-11-written-before-scoring) and [full grid results](dfl_submission/restart_recipe/development.json).

## Reproduce the submission

The build is verified to give the same file on Apple Silicon macOS (Homebrew Python 3.12.13) and on Linux (x86_64 and arm64, Python 3.12.14), with the pinned CPU dependencies. On macOS, XGBoost needs the system OpenMP runtime. Point `MTAH_DATA_DIR` at a directory containing `tactical/` with `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_homography.csv`; no other files are needed.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r dfl_submission/requirements.txt
export MTAH_DATA_DIR=/path/to/competition-bundle
.venv/bin/python -m dfl_submission.scale_trial build --out data/submission.csv
.venv/bin/python evaluation/validate_submission.py data/submission.csv
shasum -a 256 -c SUBMISSION_SHA256.txt
```

Inference takes under a minute. The build checks every model digest against the confirmed D15 recipe (`dfl_submission/scale_recipe/`) before predicting. Two builds in a fresh virtual environment reproduced the file byte for byte. Its SHA-256 is `14d18a238afce84b663cc430161240e00c93b62e3a65a0e6aa47577288b7af78`.

On Linux, the same build runs in the official Python image (`python:3.12.14-slim`, index digest `sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f`):

```sh
docker run --rm --platform linux/amd64 -v "$PWD":/repo:ro -v /path/to/competition-bundle:/bundle:ro \
  -v "$PWD/out":/out -e MTAH_DATA_DIR=/bundle -e PYTHONDONTWRITEBYTECODE=1 python:3.12.14-slim bash -c '
  cd /repo && python -m venv /venv && /venv/bin/pip install -q -r dfl_submission/requirements.txt &&
  /venv/bin/python -m dfl_submission.scale_trial build --out /out/submission.csv &&
  sha256sum /out/submission.csv'
```

On Nov 11 the D15 build gave `14d18a23…` and passed the validator under both `linux/arm64` and `linux/amd64` in this image (Python 3.12.14), matching the two macOS builds. The bundle held only `tactical/`. Later on Nov 11, a fresh clone at commit `9b68a6f` (after D18 and D19) was rebuilt in the same four ways: twice on macOS, each in a fresh venv with Homebrew Python 3.12.13, and once each in the image above (pinned by its digest) under `linux/arm64` and `linux/amd64`. All four gave `14d18a23…` and passed the validator. The bundle must hold real files, because Docker cannot follow a symlink that points outside the mounted folder.

For D3, the previous submission, the same command with `retrain_trial build` produced the same SHA-256 under `linux/amd64` and `linux/arm64` as on macOS on Nov 10. On Nov 11, a fresh clone at commit `c8cdcad` (after D7 and D8) was rebuilt in four ways, and all four gave `f0b03d2c…` and passed the validator: twice on macOS in a fresh venv with Homebrew Python 3.12.13, and once each in the image above under `linux/arm64` and `linux/amd64` (Python 3.12.14). For these runs the bundle held only `tactical/`.

Earlier versions remain available as fallbacks:
- D3, the equal blend with the original networks: `retrain_trial build --fallback . --model dfl_submission/seed_models --ensemble dfl_submission/ensemble_recipe --decoder dfl_submission/decoder_recipe --retrained dfl_submission/retrain_models --trial dfl_submission/blend_recipe`, SHA-256 `f0b03d2c8a6718ee73b52c786854c3520a6cd4bd22fdedb7475b871510cdbfa5`;
- original models with the retuned decoder: `decoder_trial build` with `--trial dfl_submission/decoder_recipe`, SHA-256 `1cc8a858354df9c8ffc024540793caa045d845d321e144b6223ba1421a56ecc1`;
- the original ensemble: `ensemble_trial build` with `--trial dfl_submission/ensemble_recipe`, SHA-256 `ed91c2362f0069c20c271e865bbc5f22bcb8331ffc5e27015130234abdb6fc2e`.

The submitted file is the build output under the team name. The scorer names each leaderboard row after the file. After the build above:

```sh
cp data/submission.csv Anfield.csv
.venv/bin/python evaluation/validate_submission.py Anfield.csv
shasum -a 256 Anfield.csv  # 14d18a238afce84b663cc430161240e00c93b62e3a65a0e6aa47577288b7af78
```

## Audit the training and selection

Every fitting, tuning and confirmation command refuses to run unless `MTAH_DATA_DIR` points to a directory that does not exist. This proves the competition bundle was unavailable while fitting.

Download the DFL XML files listed in [dfl_sources.json](dfl_submission/dfl_sources.json) into one directory. The SkillCorner preparation step fetches its inputs at the pinned revision and checks the tracking files against their hashes.

```sh
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -r dfl_submission/requirements-training.txt
export MTAH_DATA_DIR=/path/that/does/not/exist
export DFL_XML=/path/to/dfl-xml
.venv/bin/python -m dfl_submission.skillcorner --out data/skillcorner
.venv-train/bin/python -m dfl_submission.temporal_network prepare \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner --out data/train_sequences
.venv-train/bin/python -m dfl_submission.temporal_network train \
  --source data/train_sequences --out data/seed_refit --device mps --seed 97 --boundary-boost 1.4
OMP_NUM_THREADS=1 .venv-train/bin/python -m dfl_submission.temporal_network export \
  --source data/seed_refit --out data/seed_cpu
.venv/bin/python -m dfl_submission.ensemble_trial tune --fallback . \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner --model data/seed_cpu --trial data/refit_trial
.venv/bin/python -m dfl_submission.ensemble_trial confirm --fallback . \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner --model data/seed_cpu --trial data/refit_trial
```

The published weights are the reference for exact inference. Network training on Apple MPS reproduces them bit for bit. Training on CPU gives slightly different weights.

To retrain the second model set, add `--augment` to `long_context` and to `temporal_network prepare`, then train the two networks with seeds 42 (boundary weight 0.5) and 97 (boundary weight 1.4) and export them into `retrain_models/`. The comparison and blend use `retrain_trial tune` and `retrain_trial confirm`, with `--weight 0.5` for the blend.

To reproduce the D4 development grid, download the [DFL event XMLs](dfl_submission/dfl_event_sources.json) into a separate directory and keep `MTAH_DATA_DIR` set to a nonexistent path:

```sh
.venv/bin/python -m dfl_submission.restart_trial tune --dfl "$DFL_XML" \
  --events /path/to/dfl-event-xml --trial data/restart_replay
```

The D5 external proxy adapts the DFL-only reference using labelled DFL matches and the tracking fields of the SkillCorner training matches. The development command scores the variants on separate SkillCorner matches. Keep `MTAH_DATA_DIR` set to a nonexistent path.

```sh
.venv/bin/python -m dfl_submission.adaptation_trial fit \
  --dfl "$DFL_XML" --skill data/skillcorner --trial data/adaptation_d5
.venv/bin/python -m dfl_submission.adaptation_trial tune \
  --skill data/skillcorner --trial data/adaptation_d5
```

`confirm` requires a development winner and records one holdout attempt in the trial directory. [D5 protocol](DECISIONS.md#d5--label-free-transfer-from-dfl-to-skillcorner-2026-11-11-written-before-fitting-or-scoring)

D6 starts from the frozen DFL + SkillCorner D3 ensemble. It continues the original ownership tree once per target match, using confident D3 ownership predictions and replay from the original external training matches. The proxy uses the nine 2019–20 SkillCorner matches at [revision 1b99d401](https://github.com/SkillCorner/opendata/tree/1b99d401ddc82e366c5c20d6860878668fd5858f). Its provider labels are read for scoring after each match is predicted. Keep `MTAH_DATA_DIR` set to a nonexistent path.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m dfl_submission.competition_adaptation prepare-replay \
  --dfl "$DFL_XML" --skill data/skillcorner --replay data/d6_replay.npz
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m dfl_submission.competition_adaptation tune \
  --legacy-raw data/skillcorner_2020/raw --replay data/d6_replay.npz --trial data/d6_replay_trial
```

The legacy JSON files are downloaded on demand and checked against the pinned Git blobs. `confirm` uses the same arguments and requires a development winner. The `build` subcommand requires a passed confirmation and written organiser approval for label-free fitting on the challenge tracking. D6 changed nothing in the submission. [D6 protocol](DECISIONS.md#d6--label-free-ownership-adaptation-of-the-dfl--skillcorner-ensemble-2026-11-11-written-before-fitting-or-scoring)

The best D6 setting scored 0.62431 spellF1 against the N = 3 targets over five development matches, up from D3's 0.62374. The +0.00056 gain missed the +0.005 gate, so the four confirmation matches went unscored. [Development scores](dfl_submission/competition_adaptation_recipe/development.json)

D7 measures frozen D3 on PFF FC's 2022 World Cup broadcast tracking, from a vendor none of the models has seen. PFF distributes the data on request under its own terms, so it is not fetched automatically. Put the `Metadata/`, `Rosters/` and `Tracking Data/` folders in one directory. `pin` records their hashes in `dfl_submission/pff_sources.json`. `prepare` checks the files against those hashes and writes 25 Hz halves. `sanity` checks the reader with the organisers' smooth50 baseline. `measure` then scores D3 once on the eight development matches. Keep `MTAH_DATA_DIR` set to a nonexistent path.

```sh
.venv/bin/python -m dfl_submission.pff pin --raw data/pff/raw
.venv/bin/python -m dfl_submission.pff prepare --raw data/pff/raw --out data/pff/parsed
.venv/bin/python -m dfl_submission.vendor_trial sanity --parsed data/pff/parsed
.venv/bin/python -m dfl_submission.vendor_trial measure --parsed data/pff/parsed
```

`prepare --games 3812 3820 3828 3836 3844 3852 10502 10510` parses only the development matches, which is all D7 reads. The reader sanity gate passed on its first attempt, with a baseline mean raw spellF1 of 0.345. Frozen D3 then scored 0.62560 spellF1 against the N = 3 targets and 0.54874 against the raw provider labels. That is 0.038 below its SkillCorner holdout score and above the 0.624 trigger for retraining (D9). [Sanity record](dfl_submission/vendor_recipe/sanity.json) · [Measurement](dfl_submission/vendor_recipe/d7_measurement.json)

[D7 protocol](DECISIONS.md#d7--unseen-vendor-benchmark-on-pff-world-cup-tracking-2026-11-11-written-before-any-pff-data-is-read-or-scored)

D8 asked whether the PFF development matches support a different share of the retrained model set than D3's 0.5. The grid was 0, 0.25, 0.5, 0.75 and 1, with D3's decoder unchanged, and the DFL/SkillCorner development matches served as guards. It first checks that the 0.5 run reproduces D7 and D3 exactly. The best share, 0, gained +0.0081 N = 3 spellF1 on PFF against a required +0.02, and lost 0.068 on the carried-dead-ball guard. No share was eligible, so D3 stays and confirmation was not scored.

```sh
.venv/bin/python -m dfl_submission.vendor_trial tune --parsed data/pff/parsed \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner
```

[D8 development](dfl_submission/vendor_recipe/d8_development.json) · [D8 protocol](DECISIONS.md#d8--model-set-weight-on-unseen-pff-tracking-2026-11-11-written-before-any-weight-other-than-d3s-is-scored)

D11 measured where D3 loses spellF1 on the development matches. It first checks that it reproduces the D3 and D7 records exactly. Substituting the true ownership raises the standard combined N = 3 spellF1 from 0.682 to 0.887; substituting the true stoppages raises it only to 0.714. D3 predicts 20–25% fewer spells than the targets: precision is high and recall is low, mostly on spells shorter than 5 s. Boundary timing shows no bias that the vendors agree on. `--cache` keeps the decoder inputs of every development half, so later trials skip the model runs.

```sh
.venv/bin/python -m dfl_submission.diagnosis measure --parsed data/pff/parsed \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner --cache data/diagnosis_cache/development
```

D12 then tested a 1-frame decoder step and collapsing brief interior spells in the decoded labels, six settings in all. The best changed the primary by −0.001 against a required +0.005, so D3's 5-frame decoder stays and confirmation was not scored.

```sh
.venv/bin/python -m dfl_submission.refine_trial tune --parsed data/pff/parsed \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner --cache data/diagnosis_cache/development
```

D13 refit both model sets on controlled-possession targets: ownership and ownership-boundary targets computed after merging A–B–A spells shorter than 3 s. The refit reuses the original prepared training sequences and recomputes only their targets. Three checks passed first: the stored targets were reproduced from the reloaded labels; the ownership trees were refit byte for byte; and one original network was retrained bit for bit on MPS. Over ownership costs 2–8 and three stoppage cutoffs, the best setting changed the N = 3 primary by −0.00002 against a required +0.01, and every setting lost more than 0.01 at N = 1.75. D3 stays.

```sh
.venv/bin/python -m dfl_submission.aligned_trial retarget --source /path/to/train_data --out data/d13/sequences_original_aba3 \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner
.venv/bin/python -m dfl_submission.aligned_trial trees --source data/d13/sequences_original_aba3 --set current \
  --out dfl_submission/aligned_models/current/long_models
.venv/bin/python -m dfl_submission.aligned_trial networks --source data/d13/sequences_original_aba3 \
  --scratch data/d13/networks/current --out dfl_submission/aligned_models/current
# the same three steps for the augmented set (--set augmented), then:
.venv/bin/python -m dfl_submission.aligned_trial tune --parsed data/pff/parsed --dfl "$DFL_XML" --dfl-raw \
  --skill data/skillcorner --d3-cache data/diagnosis_cache/development --cache data/d13/cache/development
```

`trees --check` on an original prepared folder refits the published ownership tree and stops unless it matches byte for byte.

D14 added the five 2019–20 SkillCorner development matches to the training of both model sets as a third source, with ownership-only targets. It failed the no-loss development gates: −0.018 on the N = 3 primary, −0.049 with the dead ball carried, −0.028 on team swaps and −0.023 on PFF. D3 stays, and the four 2019–20 confirmation matches remain unscored.

```sh
.venv/bin/python -m dfl_submission.extra_trial prepare --legacy-raw data/skillcorner_2020/raw --extra data/d14/extra \
  --original /path/to/train_data data/d13/sequences_augmented --out data/d14/sequences
# aligned_trial trees and networks on data/d14/sequences/{current,augmented} into dfl_submission/extra_models/, then:
.venv/bin/python -m dfl_submission.extra_trial tune --parsed data/pff/parsed --dfl "$DFL_XML" --dfl-raw \
  --skill data/skillcorner --d3-cache data/diagnosis_cache/development --cache data/d14/cache/development
```

D15 retrained all four networks of both model sets at widths 48 and 96 and lengths of 1,800 and 5,400 steps, and re-selected the ownership cost from 4, 6 and 8. It first checked that the trainer with its new options at their defaults retrains a published network bit for bit. Width 96 with 5,400 steps and cost 8 passed development and its single confirmation, and is the submission (above).

```sh
# once per set and setting: --set current with /path/to/train_data, --set augmented with data/d13/sequences_augmented
.venv/bin/python -m dfl_submission.scale_trial networks --set current --source /path/to/train_data \
  --width 96 --steps 5400 --scratch data/d15/networks
.venv/bin/python -m dfl_submission.scale_trial tune --parsed data/pff/parsed --dfl "$DFL_XML" --dfl-raw \
  --skill data/skillcorner --d3-cache data/diagnosis_cache/development --cache data/d15/cache
.venv/bin/python -m dfl_submission.pff prepare --raw data/pff/raw --out data/pff/parsed \
  --games 3813 3821 3829 3837 3845 3853 10503 10511   # PFF confirmation set A, for D15 only
.venv/bin/python -m dfl_submission.scale_trial confirm --parsed data/pff/parsed --dfl "$DFL_XML" --dfl-raw \
  --skill data/skillcorner --d3-cache data/diagnosis_cache/holdout_confirm_a --cache data/d15/cache_holdout
```

D16 added 22 last-touch and possession-chain features (ball turns, touch evidence, time since and until each team's touches, touch counts and run lengths; `touch_features.py`). Three checks passed first: the 162 existing columns were re-extracted exactly for all 96 training halves, and both trees of each set refit byte for byte. The trees refit with the new features then failed the tree gate: their best change against D15 was −0.0004, short of the +0.003 needed before retraining the networks. D15 stays.

```sh
.venv/bin/python -m dfl_submission.touch_trial prepare --set current --source /path/to/train_data \
  --dfl "$DFL_XML" --dfl-raw --skill data/skillcorner --out data/d16/sequences_current
.venv/bin/python -m dfl_submission.touch_trial trees --set current --source /path/to/train_data --check \
  --out data/d16/check_trees_current
.venv/bin/python -m dfl_submission.touch_trial trees --set current --source data/d16/sequences_current \
  --out dfl_submission/touch_models/trees/current
# the same three steps for --set augmented with data/d13/sequences_augmented, then:
.venv/bin/python -m dfl_submission.touch_trial gate --parsed data/pff/parsed --dfl "$DFL_XML" --dfl-raw \
  --skill data/skillcorner --d3-cache data/diagnosis_cache/development \
  --reference-cache data/d15/cache/w96_s5400 --cache data/d16/cache_trees
```

[D15 development](dfl_submission/scale_recipe/development.json) · [D15 confirmation](dfl_submission/scale_recipe/confirmation.json) · [D16 tree gate](dfl_submission/touch_recipe/tree_gate.json)

D18 checked whether a published ball-free model infers team possession better than D3 where our ball is hidden. It ran one checkpoint of PathCRF (Kim et al., KDD 2026; [code](https://github.com/hyunsungkim-ds/pathcrf) at commit `1b7c6f85`, MPL-2.0, trial 140), which reads player trajectories only. None of its code is copied here: `pathcrf_trial.py` imports the model classes from a separate clone after checking its files against pinned hashes. PathCRF's checkpoints were trained on DFL WPY, our development match, so WPY favours it; the four SkillCorner development matches are unseen. On ball-hidden frames, against the N = 3 targets, D3 scored 0.870 against PathCRF's 0.595 on masked WPY, and 0.786 against 0.710 on SkillCorner. PathCRF lost on both sets, so the planned retrain (B2) did not run. A diagnostic traced the masked-view collapse to the 0.2 m per-frame position noise, not to the dropped tracks. Nothing from PathCRF enters the submission.

```sh
git clone https://github.com/hyunsungkim-ds/pathcrf /path/to/pathcrf
git -C /path/to/pathcrf checkout 1b7c6f85a6b78c92e4543fde72ca2b86a933f997
.venv-train/bin/python -m dfl_submission.pathcrf_trial measure --clone /path/to/pathcrf --dfl "$DFL_XML" \
  --skill data/skillcorner --cache data/diagnosis_cache/development
.venv-train/bin/python -m dfl_submission.pathcrf_trial diagnose --clone /path/to/pathcrf --dfl "$DFL_XML" \
  --cache data/diagnosis_cache/development
```

D19 measures the submission on Metrica Sports' three public sample matches ([sample-data](https://github.com/metrica-sports/sample-data) at commit `e706dd50`, pinned in `dfl_submission/metrica_sources.json`). They follow the conventions CHALLENGE.md describes for the ground truth: corrected tracking in which the ball is absent while dead, and events that carry the team. The repository has no licence file; its README offers the data to the analytics community and asks for credit. We use it for local evaluation only and redistribute nothing. A reader sanity gate (dead-ball convention, team orientation and the smooth50 baseline at 0.423 raw spellF1) passed on its first attempt. Against the N = 3 targets, frozen D3 scored 0.760 and D15 0.780, and every game improved. The tracking is corrected and the dead ball is exactly the missing ball, so these games flatter every model; the scores are not estimates of the competition score. Nothing is trained or chosen on them. From D19 on, a candidate is vetoed if its mean N = 3 spellF1 there falls more than 0.01 below the standing submission's, which puts the line at 0.77004.

```sh
.venv/bin/python -m dfl_submission.metrica_sample fetch      # pinned files into data/metrica/raw, checked against their hashes
.venv/bin/python -m dfl_submission.metrica_sample prepare
.venv/bin/python -m dfl_submission.metrica_sample sanity
.venv/bin/python -m dfl_submission.metrica_sample measure --reference D15
```

[D18 measurement](dfl_submission/pathcrf_recipe/d18_measurement.json) · [D18 diagnostic](dfl_submission/pathcrf_recipe/d18_diagnostic.json) · [D18 protocol](DECISIONS.md#d18--pathcrf-ball-free-possession-check-2026-11-11-written-before-any-pathcrf-output-is-computed-or-scored) · [D19 sanity](dfl_submission/metrica_recipe/sanity.json) · [D19 D3 measurement](dfl_submission/metrica_recipe/d3_measurement.json) · [D19 D15 measurement](dfl_submission/metrica_recipe/d15_measurement.json) · [D19 protocol](DECISIONS.md#d19--metrica-sample-match-benchmark-2026-11-11-written-before-any-prediction-or-score-on-these-games-is-computed)

[D11 record](dfl_submission/diagnosis_recipe/d11.json) · [D12 development](dfl_submission/refine_recipe/development.json) · [D13 development](dfl_submission/aligned_recipe/development.json) · [D14 development](dfl_submission/extra_recipe/development.json) · [D11 protocol](DECISIONS.md#d11--where-frozen-d3-loses-spellf1-2026-11-11-written-before-anything-is-scored) · [D12 protocol](DECISIONS.md#d12--decoder-output-refinements-2026-11-11-written-after-d11s-record-and-before-any-d12-setting-is-scored)

The earlier stages have their own `train`/`tune`/`confirm` commands, which are subject to the same guard:
- the extended-context XGBoost model (`long_context`)
- the first temporal network and context stages (`temporal_trial`, `context_trial`)
- the DFL and SkillCorner reference heads (`robust`, `skillcorner_trial`, `fusion_trial`)

## Tests

```sh
.venv-train/bin/python tests/run_all.py
```

Each test file runs in its own process, because PyTorch and XGBoost each bring an OpenMP runtime. The tests use synthetic fixtures and need neither the competition bundle nor the external data; the runner sets `MTAH_DATA_DIR` to a nonexistent path.

## Limitations

- **Domain shift.** The external datasets come from other competitions and other tracking systems. Their dead-ball periods carry provider artifacts: DFL freezes the ball, and SkillCorner rarely detects it. The blend halves the weight of models trained with those artifacts, but our dead-ball simulation is itself an approximation.
- **Possession definitions.** These differ between providers. DFL changes the owning team on any opponent touch. The challenge counts only controlled possession. Brief deflections may therefore appear as short possessions, and the decoder costs only partly absorb them.
- **Smoothing.** The decoder can merge very short real possessions or shift their boundaries.
- **Holdout reuse.** The external confirmation matches were reused across model-family decisions.

## Licences and attribution

- The challenge's evaluation code, `games.py`, `starter/` and `DATA.md` come from the organisers' repository under its [MIT licence](LICENSE).
- DFL data is used under CC BY 4.0 and credited to the DFL and to Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12:195 (2025). We parsed it, resampled it, synthetically degraded it and used it to fit the published weights.
- SkillCorner Open Data is used under the MIT licence; credit to SkillCorner.
- The Metrica Sports sample data ([metrica-sports/sample-data](https://github.com/metrica-sports/sample-data)) is credited to Metrica Sports. It is used for evaluation only (D1's convention choice and the D19 benchmark) and is not redistributed.
- PathCRF (Kim et al., KDD 2026) is run from its own repository under MPL-2.0 for the D18 comparison only. None of its code or weights is included here.
- The match data is not included and must not be committed (see `.gitignore`).
