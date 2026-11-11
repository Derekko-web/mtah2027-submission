# Provenance and open items

*Internal note. Review it before you push, and trim it or move it into the technical document.*

## Where the files came from (assembled Nov 10 2026)

This repository was assembled from two releases in our research workspace. Both were built only from external data:

- **Motion-ensemble release** (`data/external_motion/ensemble_release/`) is the submitted pipeline. It provides the build, models, recipes and SHA.
- **Fusion release** (`data/external_only/release/`) adds the DFL and SkillCorner reference heads (`models/`, `robust_models/`, `skillcorner_models/`, `fusion_recipe/`), their tests and the `validation/` records. These files are needed only to audit the earlier context stage. Their hashes match the ensemble recipe's `reference` entries.

Changes made on top of those releases:

- Added `dfl_submission/external_only.py`. Every train, tune, confirm and prepare entry point now calls it, and it refuses to run while the competition bundle is reachable. Inference code is unchanged.
- Added `tests/test_compliance.py`. It checks that the code never names the ground truth, that there are no graded-frame literals outside `games.py`, and that `evaluation/gt` is absent.
- Rewrote `README.md`.

Verification:
- A fresh copy with a fresh venv (Python 3.12.14, `dfl_submission/requirements.txt`) rebuilt `ed91c236…` byte for byte.
- So did a bundle containing only the three tactical files.

The research history lives in a separate repository, available to the organisers on request. Earlier, superseded work there trained on the public 5-minute window and is not used here.

## Open items before submission (from the Nov 10 audit)

1. **Lineage audit: done Nov 10, passed.** See CONSTANTS.md. Findings:
   - The XGBoost settings in `experiments/dfl_transfer_teacher.PARAMS` match an earlier model trained on the public window. Only the DFL-only and fusion reference heads use them, and the build never loads those heads.
   - The long-context hyperparameters were hand-set on Nov 10 and never searched.
   - Every decoder and blend constant came from an external development grid.
   - The context-stage island constants are not on the final joint-decoder path.
   - Still to disclose or fix: the 60% ball visibility, the `games.py` pitch size and the lineage of the feature formulas.
2. **Corruption schedule.** The original model set's degraded DFL views keep the ball visible about 60% of the time, which matches DATA.md's published 41% missing. The retrained set uses a 30–90% range instead, and both contribute equally. Disclose this, or follow the organisers' answer.
3. **Linux byte-identity: done Nov 10.** The committed repo (exported with `git archive`) rebuilt `f0b03d2c…` and the `1cc8a858…` fallback byte for byte in `python:3.12.14-slim` under `linux/amd64` (emulated) and `linux/arm64`.
4. **Technical document** covering approach, training data, possession rules, constant provenance, limitations, development history and reproduction.
5. **Decoder retune: done Nov 10.** The ownership switch cost rose from 2 to 6 under DECISIONS.md D1, and the holdout confirmation passed. The submission is now built with `decoder_trial` and its SHA is `1cc8a858…`. The previous ensemble file (`ed91c236…`) remains the fallback.
6. **Shortcut-free retrain: failed alone, adopted as a 50/50 blend on Nov 10.**
   - Alone (DECISIONS.md D2), it gained +0.090 when the dead ball moves but lost 0.015 on the standard views, against a limit of 0.01.
   - The equal blend with the original models (D3) passed on development and on the one-time holdout check: +0.062 with the dead ball carried, −0.005 on the standard views, +0.009 under team swaps.
   - The submission is now built with `retrain_trial` and the metadata pitch size; its SHA is `f0b03d2c…`.
   - Fallbacks: `1cc8a858…` (original models, retuned decoder) and `ed91c236…` (original ensemble).
