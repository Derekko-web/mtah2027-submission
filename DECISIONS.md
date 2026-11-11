# Decision record

Each entry is written before the evaluation it governs runs. Dates and times are local (EST).

## D1 — Decoder switch costs and the possession convention (2026-11-10, written before tuning)

**Question.** Which ownership and stoppage switch costs should the joint decoder use?

**Why the targets need re-expressing.** Our external labels do not use the challenge's definition of a possession change:
- DFL changes the owning team on every opponent touch, including deflections and duels without control.
- CHALLENGE.md says those do not change possession: only *controlled* possession does, and a contested ball stays with the last team.

So we score candidate decoders against external targets re-expressed under that definition. Within each live segment, every A–B–A sequence whose interior B lasts less than N seconds is merged back into A, shortest first (`dfl_submission/conventions.py`). Dead-ball labels are unchanged.

**How N was chosen.** N was chosen from external sources only:

| Source | N that best matches it (DFL spell-duration distribution, KS distance) |
|---|---|
| Metrica sample events, raw possession changes | 1.75 s |
| Metrica sample events with the CHALLENGE.md control rule (completed pass or carry, or a hold of at least 0.5 s) | 4.25–4.5 s |
| SkillCorner phases of play | 4.0 s |

The plausible interval is therefore [1.75 s, 4.5 s]. We take **N = 3 s**, which minimises the worst-case spellF1 loss over that interval under perfect perception.

Raw provider labels (no merging) are reported but not used to decide, because CHALLENGE.md explicitly excludes uncontrolled touches.

Team members saw the public 5-minute window during earlier research. No statistic from it was used here: not its possession count, not its durations, nothing else.

**Grid.** Everything else in the frozen ensemble recipe stays fixed: models, blends, cutoff 0.53 and discount 0.8.
- ownership switch cost ∈ {2, 3, 4, 6, 8, 12, 16}
- stoppage switch cost ∈ {2, 3, 4}

**Selection.** Scored on the external development matches (DFL WPY, clean and masked; four SkillCorner matches), with combined spellF1 weighted as in the existing stages.
- **Primary:** combined spellF1 against the N = 3 targets. Ties are broken by macroF1 against the same targets.
- **Development guard:** against the N = 1.75 and N = 4.5 targets, the selected costs must each score at least the shipped recipe's combined spellF1 minus 0.01.

**Confirmation.** On the external holdout matches (DFL WQQ and WR9, four SkillCorner matches). The selected costs replace the shipped costs only if all of these hold:
- combined spellF1 against N = 3 improves by at least +0.005;
- against N = 1.75, 4 and 4.5, combined spellF1 drops by no more than 0.01;
- raw-label scores are reported, not gated.

If confirmation fails, the shipped recipe (costs 2 and 2) stays.

**Outcome (added after confirmation).** Development selected an ownership cost of 6 with a stoppage cost of 2. An ownership cost of 8 scored higher on the primary target but missed the N = 1.75 guard by 0.0002, so the rule excluded it. Holdout gains: +0.0141 at N = 3, +0.0001 at N = 1.75, +0.0262 at N = 4, +0.0324 at N = 4.5 (raw labels −0.0123). The change was adopted.

**Disclosure.** These holdout matches are not untouched. They confirmed earlier stages. During the Nov 10 review, an exploratory analysis also scored an ownership cost of 8 on them, before this record was written. The record fixes the grid and the rule; the holdout result is supporting evidence, not an independent test.

## D2 — Retraining without dead-ball shortcuts (2026-11-10, written before any training or scoring)

**Problem.** Every external training source marks dead ball in a way the competition tracking will not:
- DFL freezes the ball on every dead frame.
- SkillCorner detects the ball on 18% of dead frames, against 91% of live frames.

So the stoppage heads learned "still or missing ball means dead". In an external stress test, the dead-frame ball was moved with the player nearest to it. Spell F1 against the N = 3 targets then fell from about 0.68 to about 0.53.

Separately, the degraded DFL training view keeps the ball visible 60% of the time. That matches the "41% missing" figure DATA.md publishes for the competition tracking.

**Changes.** Training data only; the external evaluation views are unchanged. Every constant below is set by hand here and will not be tuned.

1. **Dead-ball handling.** Each dead run in the DFL and SkillCorner training halves gets one mode at random:
   - keep as is (20%)
   - remove the ball (20%)
   - handled (60%). Over the first 30% of the run the ball travels from where play stopped to the restart taker, defined as the player nearest the restart spot at the restart. It then follows the taker with a 0.5 m offset. For the final 20% of the run, capped at 1 s, it rests at the restart spot. Coordinate noise is 0.2 m.

   Runs with no observed ball on either side are left unchanged. The degraded views hide ball frames after this step, as before.

   *Amendment, 2026-11-10, before any training or scoring:* where the taker is not visible, the ball moves in a straight line from where play stopped to the restart spot. A sanity check of the augmented training data found that skipping such frames left the SkillCorner ball visible on only 26% of dead frames, against 92% of live frames, which would keep the shortcut. No model output or score was looked at.
2. **Ball dropout.** The degraded DFL training view draws its long-run ball visibility per half, uniformly from 0.3 to 0.9, instead of fixing it at 0.6. The mean hidden run stays at 50 frames. Player noise and dropout are unchanged.
3. **Cross-team identity swaps.** In the degraded and stress DFL training views, a home track and an away track that are both observed exchange positions for 2–10 s (uniform). New swaps start at a rate of one per 12 s on average.
4. **Pitch size at inference.** The build uses the ATD metadata's 105 × 68 m instead of the 105.3 × 68.0 m in `games.py`, which DATA.md says was measured from corrected tracking.

The architecture, features, hyperparameters, seeds (XGBoost 42; networks 42 with boundary weight 0.5 and 97 with 1.4), training matches and step counts stay unchanged.

**Guard views.** Both are added to the external development and holdout matches. They are built from the DFL masked view with fixed seeds. Their settings deliberately differ from the training augmentation, so the model is not tested on its own simulation.
- **Dead ball carried:** on visible dead frames, the ball follows the player nearest the true ball position, with 0.2 m noise. This is the stress test described above.
- **Team swap:** cross-team swaps of exactly 10 s, starting once per 10 s on average.

**Candidate.** The retrained models are decoded with the current recipe: ownership cost 6, stoppage cost 2, discount 0.8, the same blends and phase averaging. The only exception is the stoppage cutoff, which is re-selected for the candidate on development from {0.45, 0.49, 0.53, 0.57, 0.61}. The objective is the standard combined spellF1 against the N = 3 targets (DFL clean and masked, SkillCorner observed). Ties are broken by macroF1.

**Gates.** All three must hold on development before the holdout is scored, and again on the one-time holdout confirmation. Every comparison is against the current models with the current recipe, scored against the N = 3 targets.
- Dead ball carried: mean DFL spellF1 improves by at least +0.03.
- Standard combined spellF1: drops by no more than 0.01.
- Team swap: mean DFL spellF1 drops by no more than 0.01.

Other conventions and raw labels are reported, not gated.

If any gate fails, the current models stay and the pitch-size change is not shipped on its own. A failed or passed result is recorded here either way.

**Outcome (added after development scoring, 2026-11-10 23:40).** Failed at the development gate, so the holdout was not scored. The development stoppage cutoff for the candidate was 0.45. Against the N = 3 targets, compared with the current models:

| Gate | Change | Required |
|---|---:|---|
| Dead ball carried (DFL WPY) | +0.0895 (0.497 → 0.587) | ≥ +0.03 |
| Standard combined | −0.0152 (0.687 → 0.672) | ≥ −0.01 |
| Team swap (DFL WPY) | −0.0067 | ≥ −0.01 |

Most of the standard loss is on DFL WPY masked (−0.028) and clean (−0.014), with smaller losses on three of the four SkillCorner matches. The standard views still contain the provider artifacts the retrain removes: a frozen DFL dead ball and a missing SkillCorner dead ball. Some loss there was therefore expected, but the gate was set at 0.01 and the candidate missed it. As the rule requires, the current models remain in the submission, and the pitch-size change is not shipped on its own. The retrained models and the development record are kept in `dfl_submission/retrain_models/` and `dfl_submission/retrain_recipe/` for reference.

## D3 — Equal blend of current and retrained models (2026-11-10, written before scoring the blend)

**Why.** D2's retrained models were much better when the dead ball moves (+0.090) but lost 0.015 on the standard views, which still contain the provider artifacts. We do not know which of these conditions the competition tracking resembles. An equal blend hedges between them.

**This is a second attempt on the same development matches, made after seeing D2's development result. It is weaker evidence than a first attempt, and is disclosed as such.** Nothing below was chosen by looking at blend scores. The weight is fixed at 0.5 and will not be tuned.

**Candidate.** Every model output from the current and the retrained model sets is averaged with equal weight before decoding: ownership, stoppage and boundary signals of the XGBoost heads and of both networks. The rest is as in D2:
- the current decoder recipe (ownership cost 6, stoppage cost 2, discount 0.8, same blends and phase averaging);
- the stoppage cutoff re-selected on development from {0.45, 0.49, 0.53, 0.57, 0.61}, by standard combined spellF1 against the N = 3 targets, with ties broken by macroF1;
- the ATD metadata pitch of 105 × 68 m at inference.

**Gates.** The same three gates and thresholds as D2, against the current models with the current recipe, on development and then once on the holdout:
- dead ball carried improves by at least +0.03;
- standard combined drops by no more than 0.01;
- team swap drops by no more than 0.01.

If the blend fails, the current models stay and no further variant is tried on these matches.

**Outcome (added after confirmation, 2026-11-10).** Passed. Development selected a stoppage cutoff of 0.53. Against the N = 3 targets, compared with the current models:

| Gate | Development | Holdout | Required |
|---|---:|---:|---|
| Dead ball carried | +0.0683 | +0.0616 | ≥ +0.03 |
| Standard combined | −0.0052 | −0.0050 | ≥ −0.01 |
| Team swap | +0.0108 | +0.0090 | ≥ −0.01 |

On the holdout, the standard loss was spread over all four SkillCorner matches (−0.003 to −0.010) and DFL WQQ masked (−0.010). The carried-ball gain held on both DFL matches (+0.050 and +0.073). The blend was adopted, with the metadata pitch size; the submission SHA-256 is `f0b03d2c…`.
