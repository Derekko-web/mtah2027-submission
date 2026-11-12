# Constant provenance

This file lists every constant that affects the submitted file and records where each one came from.

- "External grid" means the value was chosen from a pre-defined grid. Each grid was selected on the external development matches: DFL WPY plus four SkillCorner matches. D12 and D15 also scored their grids on eight PFF development matches, as a veto only; no value was chosen on PFF. The search records are in the `development.json` files.
- "Hand-set" means a person chose the value when writing the code. It was never searched or tuned.
- No value below was fitted, tuned, calibrated or selected using the public 5-minute ground truth, the provided video, or any human-produced data about the provided game.

## What the build loads

The submission build loads two sets of the same three model groups, and averages their outputs with equal weight:

- the extended-context XGBoost heads (`long_models/`, `retrain_models/long_models/`)
- the first temporal networks (`scale_models/w96_s5400/current/temporal_models/`, `scale_models/w96_s5400/augmented/temporal_models/`; D15)
- the seed-97 network (`scale_models/w96_s5400/current/seed_models/`, `scale_models/w96_s5400/augmented/seed_models/`; D15)

The build still loads D3's frozen recipes and checks every digest they name (`competition_adaptation.reference`), then replaces each network with its D15 counterpart. The weights of D3's networks (`temporal_models/`, `seed_models/` and their `retrain_models/` counterparts) no longer enter the submission.

The DFL and SkillCorner reference heads in `models/`, `robust_models/`, `skillcorner_models/` and `fusion_recipe/` are never loaded at inference. They are read only when you re-run the context-stage tune or confirm audit, through `external_benchmark.reference`. The context stage itself selected `owner_mix = dead_mix = 1`, which gives the reference heads zero weight.

**Disclosed lineage of the reference-head settings.** The XGBoost settings in `experiments/dfl_transfer_teacher.PARAMS` match an earlier model of ours that was trained on the public 5-minute window, before the organisers' 3 November clarification. Only heads the build never loads use them: the DFL-only, SkillCorner and fusion reference heads (through `train.py`, `robust.py` and `skillcorner_trial.py`) and the D5 students, which were fitted with the robust-head settings. The submitted heads use `long_context.TREE_PARAMS`, which were hand-set on Nov 10 and never searched.

The context stage's island-suppression settings (`island_frames`, `island_confidence` in `context_trial.py`) are not on the joint-decoder path that produces the submission, so they are not listed below.

## Inference inputs

| Constant | Value | Where | Source |
|---|---|---|---|
| Half spans | frames 11,185–90,651 and 110,808–190,197 | `games.half_video_blocks` | Published by the organisers; DATA.md calls them boundaries that reveal no label. Frames outside them are written as `X`. |
| Pitch size | 105 × 68 m | `build.METADATA_PITCH` | Reported by the tactical ATD metadata (DATA.md). The organisers' `games.py` carries 105.3 × 68.0 m, measured from corrected tracking. The submission rescales to the metadata size; the external training data use each provider's own pitch size. |
| Ball validity margin | ±5 m outside the pitch | `build.tactical_tracking_halves` | Hand-set |
| Homography validity | mask frames with `valid = 0` | `build.tactical_tracking_halves` | Organiser-defined flag |
| Ball gaps | not filled; a missing ball stays missing | `build.tactical_tracking_halves` | DECISIONS.md D28 tested straight-line fills of gaps up to 6, 12 and 25 frames; none passed development, so nothing is filled. D31 and D31a tested filling from the broadcast recording; both failed, so the build's `merge_ball` hook is unused and the broadcast files are not read. |

## Models

| Constant | Value | Where | Source |
|---|---|---|---|
| 162 tracking features, with context windows of 3, 9 and 18 s | — | `long_context.features`, `robust_features`, `skillcorner_trial.inputs` | Hand-designed. Some feature formulas were first written during our earlier research, then re-implemented and fitted on external data only. |
| Long-context XGBoost | depth 5, 420 trees, learning rate 0.045, subsample 0.82, colsample 0.85, min child weight 30, λ 12, seed 42 | `long_context.fit_head` | Hand-set when the module was written (Nov 10). Never searched. D30 (refit with Metrica's sample games added) and D32 (per-match continuation on PFF) failed, so the trees are D3's. |
| Class and match balancing | equal weight per source, match and class | `long_context.fit_head` | Hand-set |
| Team-swap augmentation | train and predict in both team orders | `models.ownership_probability` | Symmetric by construction |
| Temporal network architecture and optimiser | dilations, dropout 0.08, AdamW at 0.0015 with cosine decay, weight decay 0.01 | `temporal_network.py` | Hand-set (Nov 10) |
| Network width and training length | 96 channels per block, 5,400 steps, with the cosine schedule stretched over the run | `scale_models/w96_s5400/*/fit.json`, `temporal_network.train` | External grid under DECISIONS.md D15: width 48 or 96 × 1,800 or 5,400 steps × ownership cost 4, 6 or 8; confirmed once. Replaces width 48 and 1,800 steps. D20 (frame log-loss) and D22 Stage A (spellF1 on held-out training matches) scored widths 96 and 192 × 5,400–19,800 steps; neither replaced it. |
| Network snapshot | the last snapshot of each network (step 5,400); the seed-97 network's predictions are phase-averaged | `scale_trial.final_snapshot`, `ensemble_recipe` | External grid (snapshot variant), then D15's stretched schedule |
| Boundary loss weight | 0.5 for the first networks, 1.4 for the seed-97 network | `temporal_network.loss`, `scale_models/w96_s5400/*/*/fit.json` (`boundary_boost`) | Hand-set; there is no search record. The ensemble that uses the seed-97 network was then accepted or rejected on external data. |
| Network seeds | 42 and 97, one network per slot | `scale_models/w96_s5400/*/*/fit.json` (`seed`) | Hand-set. DECISIONS.md D22 tested the mean of three seeds per slot (s, s + 1000, s + 2000), which failed development; one seed per slot stays. |
| 5 Hz phase offsets | 0, 2, 4 frames | `ensemble_trial.PHASES` | Hand-set (covers the 5-frame stride) |

## Training-data degradation (DFL views only)

| Constant | Value | Where | Source |
|---|---|---|---|
| "Masked" view | coordinate noise 0.2 m; 20% of player tracks dropped in 2 s blocks; ball hidden and revealed by a two-state switch (probability 1/75 to hide, 1/50 to reveal) | `experiments/dfl_transfer_features.corrupt` | Hand-set, independent of all labels. The long-run ball visibility is 60%, which matches the "ball missing on 41% of graded frames" figure published in DATA.md. Disclosed here. |
| "Stress" view | alternating ball outages of geometric(0.025) length; 3.5% displaced ball observations; noise 0.30–0.35 m; 28% player drops in 40-frame chunks; 7% camera outages | `robust_features.stress` | Hand-set, independent of all labels |

## Retrained model set (DECISIONS.md D2)

All values are hand-set and were not tuned. The architecture, features, hyperparameters, seeds and training matches are the same as for the original set.

| Constant | Value | Where | Source |
|---|---|---|---|
| Dead-ball handling | each dead run: keep (20%), remove the ball (20%) or handled (60%). Handled means the ball travels to the restart taker over the first 30% of the run, follows the taker with a 0.5 m offset, and rests at the restart spot for the final 20% (capped at 1 s), with 0.2 m noise. Without a visible taker, the ball moves in a straight line to the restart spot. | `augment.handle_dead_ball` | Hand-set |
| Ball visibility in the degraded DFL view | drawn per half from U(0.3, 0.9); mean hidden run 50 frames | `augment.wide_masking` | Hand-set; replaces the fixed 60% |
| Cross-team swaps in the degraded and stress views | 2–10 s, starting once per 12 s on average | `augment.swap_teams` | Hand-set |
| Guard views (evaluation only) | dead ball follows the player nearest to it, 0.2 m noise; 10 s swaps once per 10 s | `augment.guard_halves` | Hand-set, deliberately different from training |

## Decoding

| Constant | Value | Where | Source |
|---|---|---|---|
| Blend of the trees and the first network | owner and dead mix 0.5 | `temporal_recipe/recipe.json` | External grid (467 recipes) |
| Blend with the seed-97 network | owner and dead mix 0.5 | `ensemble_recipe/recipe.json` | External grid (238 recipes) |
| Decoder | joint three-state Viterbi | `joint_decoder.py`, `ensemble_recipe` | External grid: joint, separate or reference |
| Stoppage cutoff | 0.53 | `ensemble_recipe`, re-selected for the blend in `blend_recipe` | External grid (0.53, 0.57, 0.61); re-selected for the blend from (0.45, 0.49, 0.53, 0.57, 0.61) under DECISIONS.md D3 |
| Model blend | 0.5 original, 0.5 retrained, applied to every model output | `blend_recipe` | Fixed in DECISIONS.md D3 before scoring, not tuned |
| Ownership switch cost | 8 | `scale_recipe/recipe.json` | External grid (4, 6, 8) under DECISIONS.md D15, chosen with the D15 networks and confirmed once. D22 Stage A re-scored 8, 10 and 12 on held-out training matches; none beat 8 by the required +0.005. Before D15: 6, from the D1 grid (2, 3, 4, 6, 8, 12, 16) against controlled-possession targets with N = 3 s, replacing the original ensemble value of 2. |
| Stoppage switch cost | 2 | `ensemble_recipe`, kept by `decoder_recipe` | External grid (ensemble: 1–4; retune: 2, 3, 4) |
| Controlled-possession convention | N = 3 s (merge an A–B–A whose interior B lasts under 3 s) | `conventions.py`, DECISIONS.md D1 | Minimax over the externally derived interval of 1.75–4.5 s (Metrica sample events, SkillCorner phases). Used only to score external candidates, not at inference. |
| Boundary discount | 0.8 | `ensemble_recipe` | External grid (0.55, 0.8) |
| Decoder step | 5 frames, no output collapse or boundary shift | `joint_decoder.decode` default | External grid under DECISIONS.md D12: step 5 or 1 frame × collapse none, 1.75 s or 3 s. No setting beat step 5, so the default stays. D11 handed no boundary shift. D25 (removing stoppages under 25 frames) and D26 (moving ownership changes 2 frames earlier) failed their rules, so the decoded labels are written unchanged apart from D27's kick-off rule (below). |

## After decoding: the kick-off rule (DECISIONS.md D27)

| Constant | Value | Where | Source |
|---|---|---|---|
| Kick-off tags | Smart Tagging instances with code `SET PIECES` and `Type` `Kick Off (Start)`, compared case-insensitively after trimming | `kickoff.read_tags` | The code and value as DATA.md and the organisers' `starter/load_smart_tagging.py` give them. `Kick Off (After Goal)` does not match. |
| Tag time base | frame = round(seconds × 25) | `kickoff.FPS` | DATA.md, whose time base the organisers verified against both kick-offs |
| Kick-off search limit | a tag belongs to a half if it ends at or after the half's first frame and starts at most 5 minutes (7,500 frames) after it; exactly one tag is required | `kickoff.LIMIT`, `kickoff.apply` | Hand-set, no grid |
| Dead until | the start of the half's kick-off tag | `kickoff.apply` | The Laws of the Game: no possession before the kick-off. The organisers' loader says each tag is a window of about 5 s containing the real kick-off, so its start is never later than the kick. |
| Goals to kick-offs | not applied | `kickoff.apply_goals` (unused) | DECISIONS.md D29 (dead ball from each tagged goal to the kick-off after it) could ship only alongside a round 8 step that cleared its bar (C3); none did. |

## Notes

- **Pitch size.** The submission now uses the metadata's 105 × 68 m. Only the two earlier fallback builds still use the 105.3 × 68.0 m from `games.py`.
- **Ball-dropout rate.** The original model set was trained with 60% ball visibility in the masked view, which matches the "41% missing" figure DATA.md publishes. The retrained set draws visibility from 30–90% instead. Both sets contribute equally to the submission.
- **Holdout reuse.** The external confirmation matches (DFL WQQ and WR9, four SkillCorner matches) were used to confirm several successive stages. Treat them as development evidence.
