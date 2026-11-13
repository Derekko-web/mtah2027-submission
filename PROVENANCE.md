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
- The submission now is `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`: D48's stack (three player-set networks per slot, trained with long hidden-ball views and tokens in each team's direction of attack) followed by D27's kick-off rule (`stack_trial build`). Like the builds before it, it reads four tactical files, the Smart Tagging XML only for the kick-off tags, which now also mark where each half's direction of attack is read from. It was rebuilt on a copy of the working tree twice on macOS in fresh virtual environments and in `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`, all giving the same SHA-256. Rounds 12 and 13 were committed as `49cec0d`; `release/check.sh` on that commit passed on 13 November: a fresh clone rebuilt the file twice on macOS (fresh venvs, Homebrew Python 3.12.13) and in the pinned `python:3.12.14-slim` image under `linux/arm64` and `linux/amd64`, all four giving `f6d7b293…` with the validator's OK line, and 668 of 668 tests passed in the clone. The README records each run. The earlier files are still built by their own commands: D43 (`a56574fe…`, `player_trial build`) and D15 with D27, the file C1 checked (`0ddbb342…`, `kickoff build`).

## Research history

The research history lives in a separate repository, available to the organisers on request. Earlier, superseded work there trained on the public 5-minute window. None of it is used here. The [technical document](TECHNICAL_DOCUMENT.md) (section 7) describes that period, and [CONSTANTS.md](CONSTANTS.md) discloses the one setting whose values match it.
