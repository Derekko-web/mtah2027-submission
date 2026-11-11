# Team possession from tracking: technical document

*More Than A Hack 2027, Initial Challenge.* Repository: https://github.com/Derekko-web/mtah2027-submission. Submission SHA-256: `f0b03d2c8a6718ee73b52c786854c3520a6cd4bd22fdedb7475b871510cdbfa5`.

> **Draft.** Before submitting: confirm section 7 with the whole team, add the team name, and rename this file `<team_name>.md` or export it as `<team_name>.pdf`.

## 1. Summary

We label every frame of the tactical video as `H`, `A` or `D`, and write `X` outside the two official halves. The system reads only the automatic tactical tracking. It has three parts:

- a 162-feature description of the players and the ball;
- two sets of three models that estimate who controls the ball, whether play is stopped, and where possessions change;
- a three-state Viterbi decoder that turns those estimates into possessions.

**Every model, threshold and decoder setting was trained, tuned or selected on external matches from other competitions.** The public 5-minute ground truth, the provided video and any human-produced data about Barcelona–Athletic were not used for training, tuning, calibration or selection.

## 2. Approach

### Why this design

The ranking metric scores possessions as objects, so the main risks are splitting one possession into several and missing a real change. We therefore:

1. estimate per-frame probabilities with models that see several seconds of context;
2. decode with explicit costs for changing state, so that an isolated flicker cannot create a possession on its own.

Both parts are fitted on external data where the truth is known. The public 5-minute window cannot be used for tuning. It is also far too small to choose between versions: the challenge's own baselines show it picking the wrong smoothing window.

### Inputs

- **Read:** the tactical ATD tracking and its metadata (team of each track), and the homography `valid` flag.
- **Coordinates:** converted to metres with the 105 × 68 m pitch reported by the ATD metadata.
- **Cleaning:** frames whose homography is invalid are masked, and ball positions more than 5 m outside the pitch are discarded.
- **Halves:** each half is processed separately.

### Features

The model sees 162 features, computed at 5 Hz and made symmetric between the teams. They cover:

- nearest-player distances and player counts around the ball, per team;
- sustained player-to-ball contact (strength and distance over 0.25 to 4 s);
- ball speed, speed change, distance to the touchlines and goal lines, and time spent outside the pitch;
- team shape and motion;
- summaries over context windows of 3, 9 and 18 s.

There are no player identities in the data, so every feature is collective.

### Models

| Model | What it estimates | Notes |
|---|---|---|
| Extended-context XGBoost (two heads) | home control given live play; probability of a stoppage | The ownership head trains and predicts in both team orders. Classes and matches are weighted equally. |
| Temporal convolutional network (seed 42) | ownership, stoppage and boundary probabilities | Dilated 1-D convolutions over 5 Hz sequences. |
| Second temporal network (seed 97) | same | Heavier loss weight near labelled boundaries. Predicts on three 5 Hz grids offset by 0, 2 and 4 frames, then averaged. |

The tree and network outputs are blended with equal weights. This ensemble was chosen on external development matches from 238 recipes, then confirmed on external holdout matches.

**Two model sets.** The ensemble exists twice:
1. trained on the external data as provided;
2. retrained after removing the providers' dead-ball artifacts (section 5).

Every output of the two sets is averaged with equal weight. Inference uses NumPy exports of the network weights and needs no PyTorch.

### Decoder

A three-state (`H`, `A`, `D`) Viterbi decoder runs over each half.

- **State:** a frame is `D` when the stoppage probability passes 0.53.
- **Switch costs:** a direct change of possession costs 6; entering or leaving a stoppage costs 2. Each cost is cut by up to 80% when the matching boundary signal is strong.
- **How these values were chosen:**
  - The stoppage cost, cutoff and discount came from the external development grid (section 4).
  - The ownership cost of 6 came from a later retune against controlled-possession targets.

## 3. Training data

| Dataset | Source and licence | Who labelled it | Amount | Use |
|---|---|---|---|---|
| DFL / Sportec integrated tracking and event data (Bassek, Rein, Weber & Memmert, *Scientific Data* 12:195, 2025) | [doi:10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0 | The data provider. Ball status and ball-owning team come from the official position feed. | 7 Bundesliga matches, about 1.0 M frames. Fitting: WMX, WN1, WOH, WOY. Development: WPY. Confirmation: WQQ, WR9. | Training, tuning, validation |
| SkillCorner Open Data | [revision 4340d274](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656), MIT | SkillCorner: phases of play and possession on its own detected broadcast tracking | 20 A-League matches: 12 fitting, 4 development, 4 confirmation | Training, tuning, validation |
| Metrica Sports sample data, games 1–3 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data); no licence file, credit requested | Metrica: event data (passes, recoveries, ball out, set pieces) on anonymised matches | 3 matches of event data | Only to choose the controlled-possession convention (section 6). No model was trained on it. |

**None of the training data comes from the provided game.** Both halves of a match always stay in the same split.

- The XGBoost heads were fitted on 763,244 ownership rows and 696,959 stoppage rows, sampled at 5 Hz. The ownership count includes the team-swapped copies.
- The DFL training views include:
  - the clean tracking;
  - a degraded view: coordinate noise, 20% of player tracks dropped in 2 s blocks, and runs of missing ball;
  - a stress view: longer ball outages, false ball observations, heavier player dropout and camera outages.
- SkillCorner data is used as detected, at 25 Hz after interpolation between supported observations.

## 4. External validation

The confirmation matches were reused across several decisions, so they are development evidence rather than an untouched test. Every decision after the first ensemble was governed by a rule written down before scoring ([DECISIONS.md](DECISIONS.md)), with fixed gates and a single holdout check.

### Ensemble against provider labels

Combined spellF1 weights DFL (one-third clean, two-thirds degraded tracking) and SkillCorner equally.

| Holdout | DFL | SkillCorner | Combined |
|---|---:|---:|---:|
| Previous reference (XGBoost and first network) | 0.5672 | 0.5964 | 0.5818 |
| Ensemble | 0.5740 | 0.6094 | 0.5917 |

### Against controlled-possession targets (N = 3 s, section 6)

These are combined holdout spellF1 scores.

| Stage | Standard views | Dead ball carried (DFL) | Raw provider labels |
|---|---:|---:|---:|
| Ensemble, switch costs 2 / 2 | 0.6782 | — | 0.5917 |
| Decoder retune, ownership cost 6 (D1) | 0.6923 | 0.5364 | 0.5794 |
| **Blend with the retrained models (D3), submitted** | **0.6873** | **0.5980** | 0.5746 |

The decoder retune improved all eight holdout match views on the primary target. The blend trades 0.005 on views that still contain the provider dead-ball artifacts for +0.062 when the dead ball moves (section 5).

## 5. Dead-ball artifacts and the retrained model set

Both providers mark dead ball in a way automatic tracking will not:
- DFL freezes the ball on every dead frame.
- SkillCorner detects the ball on about 18% of dead frames, against about 91% of live frames.

Models trained on them can learn that a still or missing ball means a stoppage. On external DFL matches where the dead-frame ball instead follows the nearest player, spellF1 against the N = 3 targets fell from about 0.68 to about 0.53.

We retrained every model on the same matches with three changes to the training data (D2):
1. **Dead-ball handling.** Each dead run keeps its ball (20%), loses it (20%), or has it handled (60%). Handled means it travels to the restart taker, follows the taker, then rests at the restart spot.
2. **Ball visibility.** The degraded view's ball visibility is drawn per half from 30–90%.
3. **Identity swaps.** Home and away tracks occasionally swap positions, as in a cross-team identity switch.

**Guard views.** Two guard views with deliberately different settings measure the effect:
- the dead-frame ball follows the player nearest to it;
- 10-second cross-team swaps.

**Result.**
- The retrained set on its own gained +0.090 with the dead ball carried, but lost 0.015 on the standard views, missing its 0.01 limit. It was not adopted alone.
- An equal blend with the original set, tested under a new rule (D3), passed on development and on the holdout:

| Blend vs current models | Holdout change |
|---|---:|
| Dead ball carried | +0.062 |
| Standard views | −0.005 |
| Team swap | +0.009 |

D3 was a second attempt on the same development matches, made after seeing the D2 result. We report it as weaker evidence for that reason.

## 6. Possession rules implemented

| Situation | What the system does |
|---|---|
| **What counts as possession** | Following CHALLENGE.md, only *controlled* possession changes hands. The external labels instead change the owning team on every opponent touch. We therefore tuned the decoder against external targets re-expressed under the challenge's definition: within a live segment, an A–B–A sequence whose interior B lasts under N seconds is merged into A. |
| **Choosing N** | External sources put N between 1.75 s and 4.5 s: Metrica sample events, raw (1.75 s) and with a pass/carry/hold control rule (4.25–4.5 s), and SkillCorner phases of play (4.0 s). We use N = 3 s, which minimises the worst-case loss over that interval. Without the Metrica data, SkillCorner alone gives 2.0–4.0 s, whose midpoint is also 3 s. N is used only to score external candidates; it is not a rule applied to the output. |
| **Loose balls and contested play** | The decoder's ownership cost (6) keeps the last team in possession unless the models give sustained evidence of a change. A brief touch, deflection or aerial duel therefore rarely creates a possession. |
| **Challenges** | A change is accepted when the other team's control probability stays high long enough to outweigh the switch cost. The cost is lower when the boundary network signals a genuine change. |
| **Stoppages** | `D` when the blended stoppage probability passes 0.53 under the decoder. The evidence includes ball speed, the ball leaving the pitch, distance to the lines, and player movement. |
| **Restarts** | Leaving `D` costs 2. The team after a restart follows the ownership evidence; there is no rule that hands the restart to a particular team. |
| **Halves** | Each half is decoded separately. Frames outside the published half spans are `X`. Inside the halves, every frame gets `H`, `A` or `D`. No rule or constant is tied to particular frames; the graded frames before the second-half kickoff are decided like any others. |
| **Abstention** | We do not write `X` inside the halves. The ranking metric forward-fills it, so it would only lower macroF1. |

## 7. Assets used and dropped, and development history

> **Team decision:** confirm the wording of this section before submission.

**Used:** tactical ATD tracking, its metadata and the homography `valid` flag.

**Dropped:**
- **Tactical and broadcast video.** Ball detectors and locally run vision models were explored during earlier research and dropped. The submitted pipeline does not read the video.
- **Broadcast ATD.** A camera-alignment prototype failed its external checks.
- **Smart Tagging.** There is no external data to tune rules that read it.
- **The public 5-minute ground truth.** The rules forbid training or tuning on it.

**Development history.** Before the organisers' clarification of 3 November, part of our research trained and tuned models on the public 5-minute window. That included a TabPFN-based pipeline, described in an earlier, now archived repository. We also ran a short hand-annotation pilot on the provided game on 24 October; it was rejected and never used. None of that work, its models, thresholds or outputs is used in this submission. The earlier research repository is available to the organisers on request. Some feature formulas were first written during that period; they were re-implemented and fitted on external data only. During development we used hosted coding assistants in a workspace that contained the match data. Match data is now kept out of any AI tool's reach, and competition builds are run locally.

Two items we are disclosing:
- The original model set's degraded training view keeps the ball visible 60% of the time, which matches DATA.md's published figure of 41% missing. The retrained set uses 30–90% instead.
- The organisers' `games.py` pitch size (105.3 × 68.0 m) was replaced at inference by the metadata's 105 × 68 m.

## 8. Limitations

- **Domain shift.** The training data come from other leagues and other tracking systems. The dead-ball simulation and the degraded views approximate automatic tracking but are not fitted to it.
- **No labelled validation in the target domain.** Every number in this document is external. The competition score is unknown.
- **Definition uncertainty.** If the ground truth counts every touch as a change of possession, our decoder settings cost about 0.01–0.02 on that convention, judging from the raw-label scores.
- **Short possessions.** Possessions of a second or two are hard to recover from noisy tracking. The decoder can merge them, or shift boundaries by a few frames.
- **Identity errors.** Only cross-team swaps were simulated. Fragmented tracks and duplicate tracks are handled only through dropout in the degraded views.
- **Holdout reuse.** The confirmation matches were reused across decisions, which makes the gains optimistic.

What we would do with more time:
- train on more and noisier external tracking;
- model the ball's trajectory when it is missing;
- validate on a fully untouched external competition.

## 9. Reproduction

Verified to give the same file on Apple Silicon macOS and in Linux containers (x86_64 and arm64), with Python 3.12.14. Point `MTAH_DATA_DIR` at a folder containing `tactical/BAR-ATH_tactical_atd.xml`, `tactical/BAR-ATH_tactical_atd.txt` and `tactical/BAR-ATH_tactical_homography.csv`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r dfl_submission/requirements.txt
export MTAH_DATA_DIR=/path/to/competition-bundle
.venv/bin/python -m dfl_submission.retrain_trial build --fallback . \
  --model dfl_submission/seed_models --ensemble dfl_submission/ensemble_recipe \
  --decoder dfl_submission/decoder_recipe --retrained dfl_submission/retrain_models \
  --trial dfl_submission/blend_recipe --out data/submission.csv
.venv/bin/python evaluation/validate_submission.py data/submission.csv
shasum -a 256 -c SUBMISSION_SHA256.txt
```

Inference takes about a minute. The build verifies every checkpoint digest before predicting. A rebuild from a fresh copy and a fresh environment reproduced the submitted file byte for byte, on macOS and in the official `python:3.12.14-slim` image under `linux/amd64` and `linux/arm64`. The README gives the Docker command.

The README documents:
- how to retrain every model from the public external data;
- how to re-run every tuning and confirmation step;
- the guard that refuses to fit while the competition data is reachable.

Every constant and its source are listed in [CONSTANTS.md](CONSTANTS.md).

## 10. Licences and credits

- The DFL data is credited to the DFL and to Bassek et al. (2025), under CC BY 4.0. It was parsed, resampled, synthetically degraded and used to fit our model weights.
- SkillCorner Open Data is used under the MIT licence.
- The Metrica Sports sample data is credited to Metrica Sports.
- The challenge's evaluation code, `games.py` and starter code come from the organisers' repository under its MIT licence.
