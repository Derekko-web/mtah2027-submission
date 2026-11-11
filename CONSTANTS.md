# Constant provenance

This file lists every constant that affects the submitted file and records where each one came from.

- "External grid" means the value was chosen from a pre-defined grid. Each grid was scored only on the external development matches: DFL WPY plus four SkillCorner matches. The search records are in the `development.json` files.
- "Hand-set" means a person chose the value when writing the code. It was never searched or tuned.
- No value below was fitted, tuned, calibrated or selected using the public 5-minute ground truth, the provided video, or any human-produced data about the provided game.

## What the build loads

The submission build loads two sets of the same three model groups, and averages their outputs with equal weight:

- the extended-context XGBoost heads (`long_models/`, `retrain_models/long_models/`)
- the first temporal networks (`temporal_models/`, `retrain_models/temporal_models/`)
- the seed-97 network (`seed_models/`, `retrain_models/seed_models/`)

The DFL and SkillCorner reference heads in `models/`, `robust_models/`, `skillcorner_models/` and `fusion_recipe/` are never loaded at inference. They are read only when you re-run the context-stage tune or confirm audit, through `external_benchmark.reference`. The context stage itself selected `owner_mix = dead_mix = 1`, which gives the reference heads zero weight.

**Disclosed lineage of the reference-head settings.** The XGBoost settings in `experiments/dfl_transfer_teacher.PARAMS` match an earlier model of ours that was trained on the public 5-minute window, before the organisers' 3 November clarification. Only heads the build never loads use them: the DFL-only, SkillCorner and fusion reference heads (through `train.py`, `robust.py` and `skillcorner_trial.py`) and the D5 students, which were fitted with the robust-head settings. The submitted heads use `long_context.TREE_PARAMS`, which were hand-set on Nov 10 and never searched.

The context stage's island-suppression settings (`island_frames`, `island_confidence` in `context_trial.py`) are not on the joint-decoder path that produces the submission, so they are not listed below.

## Inference inputs

| Constant | Value | Where | Source |
|---|---|---|---|
| Half spans | frames 11,185–90,651 and 110,808–190,197 | `games.half_video_blocks` | Published by the organisers; DATA.md calls them boundaries that reveal no label. Frames outside them are written as `X`. |
| Pitch size | 105 × 68 m | `build.METADATA_PITCH` | Reported by the tactical ATD metadata (DATA.md). The organisers' `games.py` carries 105.3 × 68.0 m, measured from corrected tracking. The submission rescales to the metadata size; the external training data use each provider's own pitch size. |
| Ball validity margin | ±5 m outside the pitch | `build.tactical_halves` | Hand-set |
| Homography validity | mask frames with `valid = 0` | `build.tactical_halves` | Organiser-defined flag |

## Models

| Constant | Value | Where | Source |
|---|---|---|---|
| 162 tracking features, with context windows of 3, 9 and 18 s | — | `long_context.features`, `robust_features`, `skillcorner_trial.inputs` | Hand-designed. Some feature formulas were first written during our earlier research, then re-implemented and fitted on external data only. |
| Long-context XGBoost | depth 5, 420 trees, learning rate 0.045, subsample 0.82, colsample 0.85, min child weight 30, λ 12, seed 42 | `long_context.fit_head` | Hand-set when the module was written (Nov 10). Never searched. |
| Class and match balancing | equal weight per source, match and class | `long_context.fit_head` | Hand-set |
| Team-swap augmentation | train and predict in both team orders | `models.ownership_probability` | Symmetric by construction |
| Temporal network architecture and optimiser | width, dilations, dropout 0.08, AdamW at 0.0015 with cosine decay, weight decay 0.01 | `temporal_network.py` | Hand-set (Nov 10) |
| Network snapshot | step 1800 for the first network; the seed-97 network's 1200/1800 snapshots, phase-averaged | `temporal_recipe`, `ensemble_recipe` | External grid |
| Boundary loss weight | 0.5 for the first networks, 1.4 for the seed-97 network | `temporal_network.loss`, `seed_models/fit.json` | Hand-set; there is no search record. The ensemble that uses the seed-97 network was then accepted or rejected on external data. |
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
| Ownership switch cost | 6 | `decoder_recipe` | External grid (2, 3, 4, 6, 8, 12, 16) scored against controlled-possession targets with N = 3 s. Rule pre-registered in DECISIONS.md D1, then confirmed once on the external holdout. Replaces the original ensemble value of 2. |
| Stoppage switch cost | 2 | `ensemble_recipe`, kept by `decoder_recipe` | External grid (ensemble: 1–4; retune: 2, 3, 4) |
| Controlled-possession convention | N = 3 s (merge an A–B–A whose interior B lasts under 3 s) | `conventions.py`, DECISIONS.md D1 | Minimax over the externally derived interval of 1.75–4.5 s (Metrica sample events, SkillCorner phases). Used only to score external candidates, not at inference. |
| Boundary discount | 0.8 | `ensemble_recipe` | External grid (0.55, 0.8) |
| Decoder step | 5 frames, no output collapse or boundary shift | `joint_decoder.decode` default | External grid under DECISIONS.md D12: step 5 or 1 frame × collapse none, 1.75 s or 3 s. No setting beat step 5, so the default stays. D11 handed no boundary shift. |

## Notes

- **Pitch size.** The submission now uses the metadata's 105 × 68 m. Only the two earlier fallback builds still use the 105.3 × 68.0 m from `games.py`.
- **Ball-dropout rate.** The original model set was trained with 60% ball visibility in the masked view, which matches the "41% missing" figure DATA.md publishes. The retrained set draws visibility from 30–90% instead. Both sets contribute equally to the submission.
- **Holdout reuse.** The external confirmation matches (DFL WQQ and WR9, four SkillCorner matches) were used to confirm several successive stages. Treat them as development evidence.
