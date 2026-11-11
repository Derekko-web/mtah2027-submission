# Team possession from temporal ensembles and joint decoding

This repository regenerates our possession file for the More Than A Hack 2027 Initial Challenge. The file labels each frame of the tactical video as `H` (home possession), `A` (away possession) or `D` (dead ball), and writes `X` outside the two official halves.

**All training, tuning and model selection use external matches about other games.** The competition bundle is read only by frozen inference. Nothing in this repository reads the public 5-minute ground truth, the provided video, Smart Tagging or the broadcast recording.

## How it works

1. **Inputs.** The build reads three files from the competition bundle: the tactical ATD tracking and its metadata, and the tactical homography's `valid` flag. Invalid frames are masked, ball positions far outside the pitch are discarded, and each half is processed separately.
2. **Features.** 162 team-symmetric features are computed from relative geometry and motion in metres:
   - nearest-player distances and local player counts around the ball
   - sustained player-to-ball contact
   - ball speed and pitch-boundary distance
   - team shape
   - context windows of 3, 9 and 18 seconds
3. **Models.**
   - An extended-context XGBoost model estimates home ownership given live play, and the probability of a stoppage.
   - Two temporal convolutional networks estimate ownership, stoppage and possession-boundary probabilities. The second network is trained independently with seed 97 and a heavier weight near labelled boundaries. Inference uses NumPy exports of their weights, so the build does not need PyTorch.
   - The seed-97 network predicts on three 5 Hz grids, offset by 0, 2 and 4 frames. Its predictions are interpolated back to 25 Hz and averaged with the reference signals using equal weights.
4. **Decoder.** One three-state Viterbi decoder picks the `H`/`A`/`D` sequence for each half.
   - The stoppage cutoff is 0.53.
   - A direct ownership change costs 6, and a transition into or out of a stoppage costs 2. Each cost is cut by up to 80% when the matching boundary signal is strong.
   - The higher ownership cost reflects the challenge's definition: only *controlled* possession changes hands, so a brief touch or deflection should not split a possession. See "Decoder retune" below.
   - In loose or contested play, the last team in possession keeps the ball unless the evidence supports a change.

## Training data

| Dataset | Source and licence | Who labelled it | Amount | Role |
| --- | --- | --- | --- | --- |
| DFL / Sportec integrated tracking and events (Bassek, Rein, Weber and Memmert) | [doi:10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | DFL data provider: ball status and ball-owning team in the position feed | 7 Bundesliga matches. WMX, WN1, WOH and WOY are used for fitting, WPY for development, and WQQ and WR9 for confirmation. | Fit, tune, validate |
| SkillCorner Open Data | [revision 4340d274](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656), MIT ([LICENSE-SKILLCORNER](LICENSE-SKILLCORNER)) | SkillCorner: phases of play and provider possession on detected broadcast tracking | 20 A-League matches: 12 for fitting, 4 for development and 4 for confirmation | Fit, tune, validate |

None of the training data comes from the provided game. Both halves of each match always stay in the same split. The DFL training views add synthetic degradation to the tracking: coordinate noise, missing players, runs of missing ball observations, false ball observations and camera outages.

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

## Reproduce the submission

The verified environment is Apple Silicon macOS with Python 3.12.14 and the pinned CPU dependencies. XGBoost needs the system OpenMP runtime on macOS. Point `MTAH_DATA_DIR` at a directory containing `tactical/` with `BAR-ATH_tactical_atd.xml`, `BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_homography.csv`; no other files are needed.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r dfl_submission/requirements.txt
export MTAH_DATA_DIR=/path/to/competition-bundle
.venv/bin/python -m dfl_submission.decoder_trial build --fallback . \
  --model dfl_submission/seed_models --ensemble dfl_submission/ensemble_recipe \
  --trial dfl_submission/decoder_recipe --out data/submission.csv
.venv/bin/python evaluation/validate_submission.py data/submission.csv
shasum -a 256 -c SUBMISSION_SHA256.txt
```

Inference takes under a minute. The build checks every checkpoint digest against the frozen recipe before predicting. A rebuild from a fresh copy and a fresh virtual environment reproduced the file byte for byte, including with a bundle containing only the three tactical files above. Its SHA-256 is `1cc8a858354df9c8ffc024540793caa045d845d321e144b6223ba1421a56ecc1`.

The ensemble with its original decoder costs is still available as a fallback: `ensemble_trial build` with `--trial dfl_submission/ensemble_recipe`, SHA-256 `ed91c2362f0069c20c271e865bbc5f22bcb8331ffc5e27015130234abdb6fc2e`.

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

The earlier stages have their own `train`/`tune`/`confirm` commands, which are subject to the same guard:
- the extended-context XGBoost model (`long_context`)
- the first temporal network and context stages (`temporal_trial`, `context_trial`)
- the DFL and SkillCorner reference heads (`robust`, `skillcorner_trial`, `fusion_trial`)

## Limitations

- **Domain shift.** The external datasets come from other competitions and other tracking systems. Their dead-ball periods carry provider artifacts: DFL freezes the ball, and SkillCorner rarely detects it. The model may therefore read a still or missing ball as a stoppage more readily than the competition tracking justifies.
- **Possession definitions.** These differ between providers. DFL changes the owning team on any opponent touch. The challenge counts only controlled possession. Brief deflections may therefore appear as short possessions, and the decoder costs only partly absorb them.
- **Smoothing.** The decoder can merge very short real possessions or shift their boundaries.
- **Holdout reuse.** The external confirmation matches were reused across model-family decisions.

## Licences and attribution

- The challenge's evaluation code, `games.py`, `starter/` and `DATA.md` come from the organisers' repository under its [MIT licence](LICENSE).
- DFL data is used under CC BY 4.0 and credited to the DFL and to Bassek et al., *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data 12:195 (2025). We parsed it, resampled it, synthetically degraded it and used it to fit the published weights.
- SkillCorner Open Data is used under the MIT licence; credit to SkillCorner.
- The match data is not included and must not be committed (see `.gitignore`).
