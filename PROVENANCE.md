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

1. **Lineage audit: done Nov 11, passed.** See CONSTANTS.md. Findings:
   - The XGBoost settings in `experiments/dfl_transfer_teacher.PARAMS` match an earlier model trained on the public window. Only the DFL-only and fusion reference heads use them, and the build never loads those heads.
   - The long-context hyperparameters were hand-set on Nov 10 and never searched.
   - Every decoder and blend constant came from an external development grid.
   - The context-stage island constants are not on the final joint-decoder path.
   - Still to disclose or fix: the 60% ball visibility, the `games.py` pitch size and the lineage of the feature formulas.
2. **Corruption schedule.** The DFL degraded views keep the ball visible about 60% of the time, which matches DATA.md's published 41% missing. Disclose this, or follow the organisers' answer.
3. **Linux/amd64 byte-identity.** So far the build is verified only on Apple Silicon.
4. **Technical document** covering approach, training data, possession rules, constant provenance, limitations, development history and reproduction.
5. **Planned model changes** (see NEXT_STEPS.md in the research repository). If any of them ships, update the recipe, the SHA and this note.
   - dead-ball handling augmentation
   - cross-team swap augmentation
   - decoder switch cost
