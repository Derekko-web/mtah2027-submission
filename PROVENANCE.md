# Provenance

## Where the files came from

This repository was assembled on Nov 10 2026 from two releases in our research workspace. Both were built only from external data.

- **Motion-ensemble release.** The submitted pipeline: build, models, recipes and SHA.
- **Fusion release.** The DFL and SkillCorner reference heads (`models/`, `robust_models/`, `skillcorner_models/`, `fusion_recipe/`), their tests and the `validation/` records. These files are needed only to audit the earlier context stage, and their hashes match the ensemble recipe's `reference` entries.

Every later change is recorded in [DECISIONS.md](DECISIONS.md) and in the git history.

Changes made on top of those releases:

- `dfl_submission/external_only.py`. Every train, tune, confirm and prepare entry point calls it, and it refuses to run while the competition bundle is reachable. Inference code was not changed.
- `tests/test_compliance.py`. It checks that the code never names the ground truth, that no graded-frame literals appear outside `games.py`, and that `evaluation/gt` is absent.
- A rewritten `README.md`.

## Verification

- A fresh copy with a fresh venv (Python 3.12.14, `dfl_submission/requirements.txt`) rebuilt the first submission (`ed91c236…`) byte for byte, as did a bundle containing only the three tactical files that build reads (ATD XML, ATD text and homography).
- The committed repository, exported with `git archive`, rebuilt the D3 submission of the time (`f0b03d2c…`) and the `1cc8a858…` fallback byte for byte in `python:3.12.14-slim` under `linux/amd64` (emulated) and `linux/arm64`.
- The submission now is `0ddbb342…`: D15 followed by D27's kick-off rule (`kickoff build`). It also reads a fourth tactical file, the Smart Tagging XML, for the kick-off tags only. `release/check.sh` rebuilds it from a fresh clone, twice on macOS and in `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`; the README records each run.

## Research history

The research history lives in a separate repository, available to the organisers on request. Earlier, superseded work there trained on the public 5-minute window. None of it is used here. The [technical document](TECHNICAL_DOCUMENT.md) (section 7) describes that period, and [CONSTANTS.md](CONSTANTS.md) discloses the one setting whose values match it.
