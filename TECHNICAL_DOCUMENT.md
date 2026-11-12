# Team possession from tracking: technical document

*More Than A Hack 2027, Initial Challenge.* Team: **Anfield**. Repository: https://github.com/Derekko-web/mtah2027-submission. Submission file: `Anfield.csv`, SHA-256 `14d18a238afce84b663cc430161240e00c93b62e3a65a0e6aa47577288b7af78`.

> **Draft.** Before submitting: confirm section 7 with the whole team, then upload this document as `Anfield.pdf` or `Anfield.md`.

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
| Temporal convolutional network (seed 42) | ownership, stoppage and boundary probabilities | Dilated 1-D convolutions over 5 Hz sequences: six blocks of 96 channels, trained for 5,400 steps (D15). |
| Second temporal network (seed 97) | same | Heavier loss weight near labelled boundaries. Predicts on three 5 Hz grids offset by 0, 2 and 4 frames, then averaged. |

The tree and network outputs are blended with equal weights. This ensemble was chosen on external development matches from 238 recipes, then confirmed on external holdout matches.

**Two model sets.** The ensemble exists twice:
1. trained on the external data as provided;
2. retrained after removing the providers' dead-ball artifacts (section 5).

Every output of the two sets is averaged with equal weight. Inference uses NumPy exports of the network weights and needs no PyTorch.

### Decoder

A three-state (`H`, `A`, `D`) Viterbi decoder runs over each half.

- **State:** a frame is `D` when the stoppage probability passes 0.53.
- **Switch costs:** a direct change of possession costs 8; entering or leaving a stoppage costs 2. Each cost is cut by up to 80% when the matching boundary signal is strong.
- **How these values were chosen:**
  - The stoppage cost, cutoff and discount came from the external development grid (section 4).
  - The ownership cost rose from 2 to 6 in a retune against controlled-possession targets (D1), then to 8 when it was re-selected for the larger networks (D15).

## 3. Training data

| Dataset | Source and licence | Who labelled it | Amount | Use |
|---|---|---|---|---|
| DFL / Sportec integrated tracking and event data (Bassek, Rein, Weber & Memmert, *Scientific Data* 12:195, 2025) | [doi:10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0 | The data provider. Ball status and ball-owning team come from the official position feed. | 7 Bundesliga matches, about 1.0 M frames. Fitting: WMX, WN1, WOH, WOY. Development: WPY. Confirmation: WQQ, WR9. | Training, tuning, validation |
| SkillCorner Open Data | [revision 4340d274](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656), MIT | SkillCorner: phases of play and possession on its own detected broadcast tracking | 20 A-League matches: 12 fitting, 4 development, 4 confirmation | Training, tuning, validation |
| PFF FC 2022 FIFA World Cup broadcast tracking | Shared on request by PFF FC (Gradient Sports). No dataset licence; see DECISIONS.md D7 for the terms we found. | PFF: possession sequences in its game events | 8 of 64 matches (development) | Measurement (D7), a weight selection that changed nothing (D8), and a veto in D11–D16. Confirmation set A (8 further matches) was parsed and scored once, as a veto in D15's confirmation. No model was trained on it, and no value in the submission comes from it. |
| SkillCorner Open Data, 2019–20 release | [revision 1b99d401](https://github.com/SkillCorner/opendata/tree/1b99d401ddc82e366c5c20d6860878668fd5858f), MIT | SkillCorner: home/away possession on its detected broadcast tracking; no dead-ball label | 9 matches: 5 development, 4 confirmation (never scored) | The D6 proxy, and D14 training models that were not adopted. No model in the submission was trained on it. |
| Metrica Sports sample data, games 1–3 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data); no licence file, credit requested | Metrica: event data (passes, recoveries, ball out, set pieces) on anonymised matches; corrected tracking | 3 matches: event data, and in D19 also the tracking | To choose the controlled-possession convention (section 6), and in D19 as a benchmark that can veto later candidates. No model was trained on it. |

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
| Blend with the retrained models (D3) | 0.6873 | 0.5980 | 0.5746 |
| **Larger, longer-trained networks, ownership cost 8 (D15), submitted** | **0.7087** | **0.6104** | 0.6179 |

The decoder retune improved all eight holdout match views on the primary target. The blend trades 0.005 on views that still contain the provider dead-ball artifacts for +0.062 when the dead ball moves (section 5).

**D15: larger, longer-trained networks.** Each network had 48 channels per block and trained for 1,800 steps, values that were never searched. D15 retrained all four networks of both model sets at widths 48 and 96 and lengths of 1,800 and 5,400 steps, and re-selected the ownership cost from 4, 6 and 8, under a rule written before any candidate was trained. Every new setting passed every development gate. Width 96, 5,400 steps and cost 8 gained +0.041 on development. On its single confirmation it gained +0.021 on the holdout, +0.012 with the dead ball carried, +0.059 on team swaps and +0.019 on PFF confirmation set A, eight PFF matches never scored before. Training length mattered more than width. The holdout gain is about half the development gain, as expected after picking the best of nine eligible settings on reused matches.

### Experiments that did not change the submission

Each followed a rule written before scoring, and each outcome is recorded in [DECISIONS.md](DECISIONS.md) whether it passed or failed. Scores are spellF1 against the N = 3 targets.

| Decision | What was tested | Result | Effect on the submission |
|---|---|---|---|
| D4 | Set-piece restart timestamps as decoder evidence. The tags were simulated from the DFL event feed with missing, late and spurious tags. | Best of ten settings: 0.61376 against D3's 0.61382 on development. A gain of +0.003 was required. | None. The build does not read Smart Tagging. |
| D5 | Label-free transfer: DFL-only heads adapted to unlabelled SkillCorner tracking, by speed alignment, matched corruption or self-training on confident pseudo-labels. | Self-training at 0.90 confidence passed. On the four confirmation matches it raised spellF1 from 0.493 to 0.576 over the DFL-only reference. | None. It shows that label-free adaptation transfers across tracking providers on external data. Applying it to the provided game awaits the organisers' answer (section 8). |
| D6 | The same idea applied to the submitted ensemble, adapted separately for each of five SkillCorner 2019–20 development matches. | Best setting +0.0006, against a required +0.005. Confirmation was not scored. | None. |
| D7 | Frozen D3, then the submission, measured on PFF FC's 2022 World Cup broadcast tracking, a vendor none of the models has seen. Eight development matches, after a baseline sanity gate. | spellF1 0.626 against N = 3 controlled-possession targets and 0.549 against raw provider labels. The organisers' smooth50 baseline scores 0.339 and 0.345. | None. A measurement only; nothing was trained or tuned on PFF data. |
| D8 | The share of the retrained model set (0, 0.25, 0.5, 0.75, 1), chosen on the PFF development matches, with DFL/SkillCorner guard views. | Best +0.008 (share 0), against a required +0.02. That share also lost 0.068 when the dead ball moves. Confirmation was not scored. | None. The equal blend stays. |
| D11 | Where D3, then the submission, lost spellF1 on the development matches: per-IoU detection, two oracle runs, boundary offsets by transition type, recall by spell length. | With true ownership substituted, standard combined spellF1 rises from 0.682 to 0.887; with true stoppages, only to 0.714. D3 predicts 20–25% fewer spells than the targets, with high precision and low recall; it recovers under half of the true spells shorter than 5 s. Boundary offsets have medians of −2 to +2 frames on DFL and SkillCorner, with no sign the vendors agree on. | None. A measurement only. It handed no boundary shift to D12. |
| D12 | Decoder output refinements: a 1-frame decoder step instead of 5, and collapsing brief interior spells (1.75 s or 3 s) in the decoded labels. | Best change −0.001 on development, against a required +0.005. D3 predicts almost no brief interior spells, so collapsing has little to remove. Confirmation was not scored. | None. The 5-frame decoder stays. |
| D13 | Both model sets refit on controlled-possession targets (brief A–B–A spells under 3 s merged before training), with ownership costs 2–8 and three stoppage cutoffs. Checks first confirmed that the refit reproduces the published trees and networks when given the original targets. | Best change −0.00002 at N = 3, against a required +0.01. Every setting lost more than 0.01 at N = 1.75; at D3's costs the aligned models gain +0.011 at N = 4 and 4.5. Confirmation was not scored. | None. Training on the challenge's convention moves the score between conventions; it does not recover missed changes. |
| D16 | 22 last-touch and possession-chain features (ball turns next to players, time since and until each team's touches, touch counts and run lengths), with both trees refit first; the networks would follow only if the trees gained +0.003 against D15. Checks first confirmed that the 162 existing features were re-extracted exactly and that the trees refit byte for byte. | Best tree-only change −0.0004 against D15. The new trees matched more ownership changes on clean DFL (104 against 93 of 184) but not on SkillCorner or PFF, and lost at N = 4 and 4.5 and on team swaps. | None. The networks were not retrained; confirmation was not scored. |
| D14 | Five 2019–20 SkillCorner matches added to training as a third source, with ownership-only targets. | Every development gate failed: −0.018 at N = 3, −0.049 with the dead ball carried, −0.028 on team swaps, −0.023 on PFF. Confirmation was not scored. | None. |
| D18 | A published ball-free model, PathCRF (Kim et al., KDD 2026), against D3 on frames where our ball is hidden. PathCRF infers possession from player trajectories alone. We ran one released checkpoint unchanged. Its checkpoints were trained on DFL WPY, our development match, which favours it there; the four SkillCorner development matches are unseen. | Team-possession accuracy on ball-hidden frames against the N = 3 targets: D3 0.870 against PathCRF 0.595 on masked WPY, and 0.786 against 0.710 on SkillCorner. A diagnostic traced the masked-view collapse to the 0.2 m per-frame position noise, not to the dropped tracks. PathCRF's teams with D3's stoppages scored 0.337 combined spellF1 against D3's 0.682, because PathCRF changes team at every touch. | None. PathCRF lost on both sets, so the planned retrain did not run. |
| D19 | Metrica Sports' three public sample matches as a benchmark. Their corrected tracking has no ball while play is dead, and their events carry the team: the conventions CHALLENGE.md describes for the ground truth. A reader sanity gate came first. | D3 0.760 and D15 0.780; every game improved. The smooth50 baseline scores 0.423 against the raw targets. The tracking is corrected and the dead ball is exactly the missing ball, so these scores flatter every model and do not estimate the competition score. | None. A measurement and a veto: from D19 on, a candidate whose mean N = 3 spellF1 falls more than 0.01 below the submission's (line 0.770) is rejected. Nothing is trained or chosen on these games. |
| D20 | A model search beyond D15, tuned by match-grouped cross-validation on the training matches only (frame log-loss): the temporal network at widths 96 and 192 and 5,400 to 19,800 steps, a bidirectional GRU, a transformer encoder and deeper XGBoost tree heads. Each family's best setting was scored once on development, then a greedy equal-weight ensemble. | The temporal network selected D15's own setting. Against D15's 0.723 on development: GRU −0.003, trees −0.003, transformer −0.031. The best pair, D15 + GRU, scored 0.722, so the selection kept D15 alone. | None. Frame log-loss favoured the shortest training in every network family, the opposite of D15's spellF1 gain from longer training. |
| D21 | Where the submission loses spellF1: D11's measurements repeated for D15, with the Metrica sample matches added. | With true ownership substituted, standard combined spellF1 rises from 0.723 to 0.885; with true stoppages, to 0.750 (D3: 0.682, 0.887 and 0.714). D15 predicts 6–22% fewer spells than the targets (D3: 20–25% fewer). On SkillCorner it recovers 46–52% of the true spells lasting 1.75–5 s. | None. A measurement only. |
| D22 | Finishing D15's search, then averaging seeds. Stage A decoded D20's 24 fold networks (widths 96 and 192, 5,400 to 19,800 steps) on the training matches they had not seen, at ownership costs 8, 10 and 12. Stage B trained two more seeds for each of the four network slots, and each slot became the mean of three networks. | Stage A: the best point (width 96, 10,800 steps, cost 12) beat D15's own setting by +0.002 on the held-out training matches, against a required +0.005, so D15's setting and cost stayed. Three seeds per slot scored 0.717 against D15's 0.723 on development (+0.005 was required). They gained with the dead ball carried (+0.013), on team swaps (+0.011) and on Metrica (+0.002). | None. Confirmation was not scored, and PFF confirmation set C is unused. |
| D23 | Refitting the submission on all 27 labelled external matches. It would run only if, on development, a network trained on 16 matches beat four trained on 12 by +0.005. | The 16-match network scored 0.7114 against the 12-match networks' mean of 0.7135; single networks ranged from 0.708 to 0.722. | None. The refit did not run. The untouched PFF matches and the four 2019–20 confirmation matches were not scored. |

### The one correctness check

The public 5-minute window is used once, on the final file, under a procedure fixed in advance (DECISIONS.md C1). The check confirms that the file is well formed, that frames and halves are aligned, and that home and away are not swapped. It cannot choose between versions. A failure may only lead to a pipeline fix, never to a change of model or setting.

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
| **Loose balls and contested play** | The decoder's ownership cost (8) keeps the last team in possession unless the models give sustained evidence of a change. A brief touch, deflection or aerial duel therefore rarely creates a possession. |
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
- **Smart Tagging.** No external data matches its format. D4 tested restart timestamps simulated from DFL events, and no setting improved on D3 (section 4).
- **The public 5-minute ground truth.** The rules forbid training or tuning on it.

**Development history.** Before the organisers' clarification of 3 November, part of our research trained and tuned models on the public 5-minute window. That included a TabPFN-based pipeline, described in an earlier, now archived repository. We also ran a short hand-annotation pilot on the provided game on 24 October; it was rejected and never used. None of that work, its models, thresholds or outputs is used in this submission. The earlier research repository is available to the organisers on request. Some feature formulas were first written during that period; they were re-implemented and fitted on external data only. During development we used hosted coding assistants in a workspace that contained the match data. Match data is now kept out of any AI tool's reach, and competition builds are run locally.

Two items we are disclosing:
- The original model set's degraded training view keeps the ball visible 60% of the time, which matches DATA.md's published figure of 41% missing. The retrained set uses 30–90% instead.
- The organisers' `games.py` pitch size (105.3 × 68.0 m) was replaced at inference by the metadata's 105 × 68 m.

## 8. Limitations

- **Domain shift.** The training data come from other leagues and other tracking systems. The dead-ball simulation and the degraded views approximate automatic tracking but are not fitted to it.
- **No labelled validation in the target domain.** Every number in this document is external. The competition score is unknown.
- **Definition uncertainty.** If the ground truth counts every touch as a change of possession, our decoder settings cost about 0.01–0.02 on that convention, judging from the raw-label scores.
- **Short possessions.** Possessions of a second or two are hard to recover from noisy tracking. On external development matches D3, the previous submission, recovered under half of the true possessions shorter than 5 s, and predicted 20–25% fewer possessions than the targets (D11). Neither a finer decoder (D12) nor training on the challenge's convention (D13) improved this. Longer training narrowed the gap (D15): on clean DFL the submission predicts 263 possessions against 281 true (D3: 219) and matches 93 of 184 ownership changes within one second (D3: 68).
- **Identity errors.** Only cross-team swaps were simulated. Fragmented tracks and duplicate tracks are handled only through dropout in the degraded views.
- **Holdout reuse.** The confirmation matches were reused across decisions, which makes the gains optimistic.
- **No adaptation to the provided game.** D5 showed on external data that label-free adaptation helps across tracking providers. RULES.md allows learning without human labels on the provided game, but the organisers' 3 November clarification requires external data for all training. We asked on 11 November which applies. Until we have a written answer, nothing is fitted on the provided tracking.

What we would do with more time:
- train on more and noisier external tracking;
- model the ball's trajectory when it is missing;
- validate on a fully untouched external competition.

## 9. Reproduction

Verified to give the same file on Apple Silicon macOS (Homebrew Python 3.12.13) and in Linux containers (x86_64 and arm64, Python 3.12.14). Point `MTAH_DATA_DIR` at a folder containing `tactical/BAR-ATH_tactical_atd.xml`, `tactical/BAR-ATH_tactical_atd.txt` and `tactical/BAR-ATH_tactical_homography.csv`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r dfl_submission/requirements.txt
export MTAH_DATA_DIR=/path/to/competition-bundle
.venv/bin/python -m dfl_submission.scale_trial build --out data/submission.csv
.venv/bin/python evaluation/validate_submission.py data/submission.csv
shasum -a 256 -c SUBMISSION_SHA256.txt
```

Inference takes under a minute. The build verifies every model digest before predicting. Rebuilds in a fresh environment reproduced the submitted file byte for byte, twice on macOS and in the official `python:3.12.14-slim` image under `linux/arm64` and `linux/amd64`. The README gives the Docker command.

The README documents:
- how to retrain every model from the public external data;
- how to re-run every tuning and confirmation step;
- the guard that refuses to fit while the competition data is reachable.

Every constant and its source are listed in [CONSTANTS.md](CONSTANTS.md).

## 10. Licences and credits

- The DFL data is credited to the DFL and to Bassek et al. (2025), under CC BY 4.0. It was parsed, resampled, synthetically degraded and used to fit our model weights.
- SkillCorner Open Data is used under the MIT licence.
- The Metrica Sports sample data is credited to Metrica Sports.
- PathCRF (Kim et al., KDD 2026) was run from its own repository under MPL-2.0 for the D18 comparison only. None of its code or weights is in our repository or in the submission.
- The challenge's evaluation code, `games.py` and starter code come from the organisers' repository under its MIT licence.
