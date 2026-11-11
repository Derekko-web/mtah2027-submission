# Constant provenance

This file lists every constant that affects the submitted file and records where each one came from.

- "External grid" means the value was chosen from a pre-defined grid. Each grid was scored only on the external development matches: DFL WPY plus four SkillCorner matches. The search records are in the `development.json` files.
- "Hand-set" means a person chose the value when writing the code. It was never searched or tuned.
- No value below was fitted, tuned, calibrated or selected using the public 5-minute ground truth, the provided video, or any human-produced data about the provided game.

## What the build loads

The submission build loads only three groups of model files:

- the extended-context XGBoost heads (`long_models/`)
- the first temporal networks (`temporal_models/`)
- the seed-97 network (`seed_models/`)

The DFL and SkillCorner reference heads in `models/`, `robust_models/`, `skillcorner_models/` and `fusion_recipe/` are never loaded at inference. They are read only when you re-run the context-stage tune or confirm audit, through `external_benchmark.reference`. The context stage itself selected `owner_mix = dead_mix = 1`, which gives the reference heads zero weight.

## Inference inputs

| Constant | Value | Where | Source |
|---|---|---|---|
| Half spans | frames 11,185–90,651 and 110,808–190,197 | `games.half_video_blocks` | Published by the organisers; DATA.md calls them boundaries that reveal no label. Frames outside them are written as `X`. |
| Pitch size | 105.3 × 68.0 m | `games.py` (organiser file) | Organiser-supplied. DATA.md says it was measured from the corrected tracking; the ATD metadata gives 105 × 68. See the note below. |
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

## Decoding

| Constant | Value | Where | Source |
|---|---|---|---|
| Blend of the trees and the first network | owner and dead mix 0.5 | `temporal_recipe/recipe.json` | External grid (467 recipes) |
| Blend with the seed-97 network | owner and dead mix 0.5 | `ensemble_recipe/recipe.json` | External grid (238 recipes) |
| Decoder | joint three-state Viterbi | `joint_decoder.py`, `ensemble_recipe` | External grid: joint, separate or reference |
| Stoppage cutoff | 0.53 | `ensemble_recipe` | External grid (0.53, 0.57, 0.61) |
| Ownership and stoppage switch costs | 2 and 2 | `ensemble_recipe` | External grid (owner 1–3, dead 1–4) |
| Boundary discount | 0.8 | `ensemble_recipe` | External grid (0.55, 0.8) |

## Notes

- **Pitch size.** The 105.3 × 68.0 m constant is part of the organisers' `games.py`. Switching inference to the metadata's 105 × 68 would remove any dependence on a figure derived from corrected tracking. It would change the output file, so it is planned for the next rebuild together with the other model changes.
- **Ball-dropout rate.** The 60% visibility in the masked view will be replaced by a wide, pre-declared range in the next retraining, unless the organisers confirm the published figure may be used.
- **Holdout reuse.** The external confirmation matches (DFL WQQ and WR9, four SkillCorner matches) were used to confirm several successive stages. Treat them as development evidence.
