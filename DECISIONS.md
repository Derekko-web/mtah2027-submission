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

## D4 — Smart Tagging restart anchors (2026-11-11, written before scoring)

**Candidate.** Keep the frozen D3 blend and D1 costs. Use only set-piece restart timestamps for throw-ins, corners, goal kicks, free kicks and kickoffs after a goal. Ignore the team and phase tags. In the external trial, read the DFL event `EventTime` on the corresponding tracking half's 25 Hz clock. Exclude opening kickoffs. For a competition build, use the midpoint of each supported Smart Tagging interval on the documented tactical-video clock, separately within each half. No information from the competition export is used to set the rule or select parameters.

**Simulated tags.** Use the same noisy tag draws for every tracking view and candidate. Each half gets three draws per profile. The profiles are fixed stress scenarios, with values chosen before scoring.

| Profile | Tag loss | Timing error | Spurious tags |
|---|---:|---|---:|
| Moderate | 50% | Gaussian with SD 1 s, clipped to ±2.5 s | 10% of retained in-half tags |
| Heavy | 75% | Gaussian with SD 2 s, clipped to ±5 s | 25% of retained in-half tags |

Drop each event independently, perturb the rest on the frame grid, and discard tags that move outside the half. Add the rounded number of spurious tags uniformly over the half. Use NumPy default RNG seeds `740000 + 1000 × DFL split-order match index + 100 × half index + 10 × profile index + draw index` with zero-based indices.

**Grid.** Test two ways to use a tag in the joint Viterbi decoder. All offsets are relative to the 5 Hz decoder step start. Overlapping tags use the maximum support separately on each side.

- **Stoppage evidence.** Raise the dead-ball log-odds before a restart and lower them after it. The triangular weights on each side are zero at the tag and the outside edge, and peak halfway between. Test a span of 2 or 4 s per side and a peak adjustment of 0.5, 1 or 2 log-odds units.
- **Restart transition.** Discount the existing cost of the dead-to-live transition by a triangular weight centred on the tag, with no adjustment to the live-to-dead or ownership costs. Test a radius of 2 or 4 s and a peak discount of 0.5 or 1.

**Selection and confirmation.** Use DFL WPY for development and DFL WQQ and WR9 for one confirmation of the selected setting. SkillCorner is excluded because there is no comparable restart event feed for those views. All comparisons use the same D3 blend with no anchors as the reference. Aggregate equally over matches and draws within each profile. For the primary score, average masked and dead-ball-carried tracking equally, then average the two profiles. Rank eligible candidates by primary spellF1 against N = 3 targets, then by primary macroF1, then by grid order above with smaller span and strength first.

Apply all gates on development and again on the one-time confirmation:

- The primary spellF1 must improve by at least 0.003. For each match separately the primary drop may be at most 0.01.
- Heavy-profile primary spellF1 may drop by at most 0.005. Within each profile, the standard DFL score (one-third clean, two-thirds masked) and team-swap spellF1 may each drop by at most 0.01.
- Primary spellF1 for N = 1.75, 4 and 4.5 may each drop by at most 0.01. Report raw provider scores.

If no development candidate passes or the selected one fails confirmation, keep D3. D4 is a new decoder experiment on matches already used in D1–D3. Reuse makes its confirmation weaker than a fresh test. The tag simulation is an approximation and the two profiles do not establish the actual tag error distribution.

**Outcome (added after development scoring, 2026-11-11).** D4 stopped at development. All ten settings scored below the D3 reference of 0.61382 on primary spellF1. The highest score was 0.61376 (−0.00006, against a required gain of +0.003), shared by the 2 s and 4 s restart-transition settings at a peak discount of 0.5. The best stoppage-evidence setting scored 0.61359. The development gate ended the experiment before holdout scoring. D3 is the submission. See the [complete development results](dfl_submission/restart_recipe/development.json).

## D5 — Label-free transfer from DFL to SkillCorner (2026-11-11, written before fitting or scoring)

**Scope.** Test whether adapting a DFL-only model to unlabelled SkillCorner tracking improves possession spells on separate SkillCorner matches. Fit from the four pinned DFL training matches and the tracking fields of the twelve pinned SkillCorner training matches. The fitting code never reads SkillCorner labels. The development and confirmation matches supply only evaluation data. All commands are external-only. This proxy grants no authority to fit anything on the challenge match.

**Reference.** Start with the verified DFL-only `robust_models` heads. All entries use those features, a 5 Hz feature sample interpolated to the 25 Hz grid, and the same joint decoder (cutoff 0.53, ownership cost 6, stoppage cost 2, zero boundary evidence). Fit replacement heads with the existing robust-head XGBoost parameters and home/away symmetry. There is no decoder search.

**Grid, in tie-break order.**

1. **Speed alignment.** Map SkillCorner's eight extra ball-speed features (`speed_{3,12,30}_{0,1}`, `speed_mean_{12,50}`) to the reference DFL scale. Compute the median and IQR of `log1p` of positive, finite values, at 5 Hz. Pool the original masked and stress views of the DFL training matches and pool the observed SkillCorner training halves separately. Apply the DFL-to-SkillCorner IQR ratio clipped to [0.5, 2] and align the medians in log space, then invert and floor at zero. Leave zero and missing values unchanged. Skip a feature if either domain has no valid IQR. The DFL heads are frozen.
2. **Matched corruption.** Refit with the original clean DFL view and two matched views replacing masked and stress. For each matched view, draw independent 60 s contiguous windows uniformly over donor halves, then uniformly over their valid starting frames, from the unlabelled SkillCorner training set. Replay the ball visibility and both visible player counts jointly. Cap counts at the number of DFL players available in each frame, keeping players by a random priority order fixed per view. Add Gaussian coordinate noise with standard deviation 0.2 m. Use NumPy default RNG with seed `810000 + 100 × DFL match index + 10 × half index + view index`, with zero-based indices and the existing training order. This matches missingness and co-visibility, with simplified player identities and window edges.
3. **Self-training at 0.90 confidence.** On observed SkillCorner training tracking at 5 Hz, use the frozen reference to pseudo-label live/dead when either binary probability reaches 0.90. For ownership also require a pseudo-live frame and the same confidence on one team. Fit a student once on the original DFL clean, masked and stress views plus those pseudo-labels. DFL contributes 75% of the total head weight, shared equally over matches and classes as in the reference. SkillCorner contributes 25%, equally over its two pseudo-classes and uniformly within each class. If a pseudo-class is absent, the setting is ineligible. The student replaces both heads.
4. **Self-training at 0.97 confidence.** Use the same rule at 0.97.

**Selection and confirmation.** Compare each setting with the DFL-only reference on the four established SkillCorner development matches. Compute each match's spellF1 over both halves, then average the matches equally. Use the N = 3 s controlled-possession target as primary. A setting is eligible if its primary mean gain is at least +0.010, no match loses more than 0.020 on primary, and mean spellF1 for each of N = 1.75, 4 and 4.5 s drops by no more than 0.010. Report raw provider scores. Rank eligible settings by primary spellF1, then by mean primary macroF1, then by the grid order above. Freeze the one winner and run it and the reference once on the four established SkillCorner confirmation matches with the same gates. A development failure ends the trial without scoring confirmation. The confirmation matches have been used in earlier stages, so the result is weaker than confirmation on fresh matches. A pass would demonstrate transfer against the DFL-only reference on this proxy. D3 remains the submission, and any application to the challenge tracking depends on written organiser permission.

**Outcome (added after confirmation, 2026-11-11).** Self-training at 0.90 confidence ranked first on development and passed the single holdout confirmation.

| Development setting | N = 3 spellF1 | Gain | Raw spellF1 | Passed |
|---|---:|---:|---:|---|
| DFL-only reference | 0.49679 | — | 0.45179 | — |
| Speed alignment | 0.49912 | +0.00233 | 0.45652 | No |
| Matched corruption | 0.50771 | +0.01092 | 0.45853 | Yes |
| Self-training at 0.90 | 0.58298 | +0.08619 | 0.52692 | Yes |
| Self-training at 0.97 | 0.57876 | +0.08197 | 0.52099 | Yes |

On confirmation, N = 3 spellF1 rose from **0.49322 to 0.57614 (+0.08292)**. All four matches improved, with gains from +0.06557 to +0.09762. The gains at N = 1.75, 4 and 4.5 were +0.08507, +0.08136 and +0.08221. Raw spellF1 rose from 0.44016 to 0.52152. The confirmation matches had been used in earlier stages. All variants shared the D1/D3 decoder settings. D3 is the active submission.

[Fitting record](dfl_submission/adaptation_recipe/fit.json) · [Development scores](dfl_submission/adaptation_recipe/development.json) · [Confirmation scores](dfl_submission/adaptation_recipe/external_validation.json)

## D6 — Label-free ownership adaptation of the DFL + SkillCorner ensemble (2026-11-11, written before fitting or scoring)

**Reference and split.** Use the verified D3 ensemble, whose original and augmented model sets were both fit on DFL and the 2024–25 SkillCorner release. Use the nine independent 2019–20 SkillCorner matches at revision `1b99d401ddc82e366c5c20d6860878668fd5858f` (MIT) as the target proxy. Sort by numeric match ID and use the first five (2068, 2269, 2417, 2440, 2841) for development and the last four (3442, 3518, 3749, 4039) for one confirmation. For each match fit one student from its two tracking halves and score on those same halves after fitting, matching the proposed single-match use. The labels stay outside the adaptation code and are read only after prediction. The confirmation matches are fresh to this trial, and the within-match result measures transductive adaptation.

**Inputs and targets.** Keep tracked observations associated with roster player IDs or the ball ID, use the metadata pitch, and apply the existing 25 Hz interpolation and five-metre outlier limit. Use provider home and away possession where supplied. Mark null or unrecognised ownership as X and exclude it from scoring. The legacy release supplies no usable dead-ball target, and ownership gaps split scoring runs. Report the share of the 25 Hz timeline with an H/A target for each match.

**Fit.** At 5 Hz take pseudo-ownership from frozen D3 where its stoppage probability is at most 0.10 and one team has ownership probability of at least 0.90. If either team has zero examples across both halves, use D3 for every setting on that match and record the fallback. Continue the original D3 long-context ownership XGBoost checkpoint for 40 rounds with its original training parameters. Mix pseudo-labels with replay from the original long-context training views on the pinned DFL and 2024–25 SkillCorner fitting matches. Use the replay ownership targets at frame offsets divisible by 25 in each half. Give replay 75% of the weight, shared equally by dataset, match within dataset and team within match, as in the long-context fit. Give pseudo-labels 25%, shared equally by team. Mirror every row with the team features and label swapped, as in the original fit.

**Grid.** Compute the continued tree's change in symmetric home-ownership probability from the original tree at 5 Hz, interpolate the change back to 25 Hz, and add it to D3's final home-ownership probability. Try multipliers 0.25, 0.5 and 1.0 in that order, clipping the resulting probability to [0, 1]. Use the same fitted student for all three. D3 is the reference with multiplier zero. Keep its stoppage and boundary signals and joint decoder fixed.

**Selection and confirmation.** Compute spellF1 per match over the two halves, then average the matches equally. Primary is N = 3 s. A setting qualifies if mean primary spellF1 improves by at least 0.005, no match loses more than 0.020 primary spellF1, and mean spellF1 at N = 1.75, 4 and 4.5 s each drops by at most 0.010. Report raw provider scores. Rank eligible settings by primary spellF1, then mean primary H/A macroF1, then grid order. If no setting qualifies, stop after development. Otherwise freeze the winner, evaluate it and D3 once on confirmation, and apply the same gates. Both phases run with the external-only guard. D3 remains the submission. Applying adaptation to the challenge requires written organiser permission and a passed D6 confirmation.

**Outcome (added after development scoring, 2026-11-11).** All five development matches supplied pseudo-labels for both teams, so each fit used its own student. Provider H/A targets covered 59.8%–72.0% of the evaluated timelines.

| Ownership adjustment multiplier | N = 3 spellF1 | Gain vs. D3 | Raw spellF1 | Passed |
|---|---:|---:|---:|---|
| 0 (D3) | 0.62374 | — | 0.54793 | — |
| 0.25 | 0.62431 | +0.00056 | 0.54852 | No |
| 0.5 | 0.62429 | +0.00055 | 0.54825 | No |
| 1.0 | 0.62272 | −0.00102 | 0.54698 | No |

Every multiplier satisfied the per-match and alternate-convention loss limits. None reached the required +0.005 mean primary gain. Development selected no setting and the four confirmation matches went unscored. All adaptation ran on external data. D3 remains the submission.

[Per-match scores, pseudo-label counts and input hashes](dfl_submission/competition_adaptation_recipe/development.json)

## C1 — The one correctness check on the public window (2026-11-11, written before any check is run)

**Purpose.** RULES.md allows the public 5-minute window only for checking that the pipeline is correct: frames aligned, file well-formed, labels meaning what we think. This record fixes the only time we will score against it. It cannot choose between versions, files, models or settings.

**When and who.** Derek runs it once, on the final frozen file, after the last rebuild matches `SUBMISSION_SHA256.txt` and before the final form edit. It runs in a fresh clone of the organisers' challenge repository, outside this repository. No AI tool sees the file, the window or the scores.

**Procedure.**

1. `python evaluation/validate_submission.py <team_name>.csv` in the organisers' clone.
2. Copy `<team_name>.csv` into the clone's `evaluation/results/`. Next to it, write `swap_check.csv`, the same file with every `H` and `A` exchanged:
   `awk -F, 'NR==1{print;next}{l=$2; if(l=="H")l="A"; else if(l=="A")l="H"; print $1","l}' <team_name>.csv > evaluation/results/swap_check.csv`
3. `python evaluation/run_evaluation.py`, which scores everything in `evaluation/results/` against the public window by default.

**Pass.** All three must hold:
- the validator accepts the file;
- the submission's public spellF1 is at least 0.25 and its macroF1 at least 0.50;
- `swap_check.csv` scores lower than the submission on both metrics.

These are floors that only a broken pipeline should miss, not targets. A team-orientation error fails the swap test, a frame offset or wrong half span fails the floors, and a malformed file fails the validator. The spellF1 floor sits below the organisers' smooth50 baseline on the same window (0.3256).

**Reporting.** Derek reports PASS, or FAIL with the name of the failed criterion (validator, floors or swap). No number, count or label from the check is written to this repository or shown to an AI tool.

**After a fail.** Only a pipeline fix is allowed: team orientation, frame offset or half spans. The defect must be found in the code or in the documented data format, never by trying variants against the window. Models, recipes, thresholds and decoder settings do not change. After a fix, the file is rebuilt twice, its SHA is updated, and the check runs once more under the same rule. If that second run also fails, nothing further is changed, the validated file that matches `SUBMISSION_SHA256.txt` is submitted, and both runs are disclosed in the technical document.

**Outcome (2026-11-12).** **PASS.** Derek ran it once on the frozen file, `Anfield.csv`, a copy of `data/submission.csv` with SHA-256 `0ddbb34216fdab55752a5b46458639f745cc4e88c074ec0787d771f3630fceff` (D27). It ran in a fresh clone of the organisers' repository, after the SHA was checked against `SUBMISSION_SHA256.txt` and after C2 passed. All three criteria held: the validator accepted the file, both floors were met, and `swap_check.csv` scored lower than the submission on both metrics. No pipeline fix is needed, and the check does not run again.

*Disclosure.* The steps were run from inside the AI coding session, so their printed output reached the AI assistant. That output was the validator's label and possession counts for the file, and `run_evaluation.py`'s full scoreboard and per-file reports against the public window. This breaks the reporting clause above. The assistant had wrongly said the validator prints only an OK line, and the evaluation was run in the session despite advice to use a separate terminal. The file had been frozen, rebuilt four ways and checked by C2 before any of this was shown. Nothing has been or will be changed, chosen or tuned because of it, and no number from that output is written to this repository.

*Amendment, 2026-11-12 (C3):* development continued after C1 under C3, and D43 changed the file. The submission is now the player-set model of D42 at ownership cost 10, followed by D27's kick-off rule (`player_trial build`, SHA-256 `a56574fefd83af81747206e3aedcaf625cdd170e7376d4e00e5f0abee7c431e3`). C1 is not run again (C3). What C1 checked carries over because the reading, team orientation, frame indexing and half spans are D15's code, unchanged: `tactical_halves` with the same pitch and centring, the same half spans and the same D27 rule; only the networks and the ownership cost changed, and with the old build command (`kickoff build`) the code still writes `0ddbb342…`. C1's PASS applies to `0ddbb342…`; the new file is checked by C2's re-declaration.

*Amendment, 2026-11-13 (C3):* development continued under C3 in round 12, and D48 changed the file again. The submission is now the stack of D48 (D42's player-set networks trained on D49's long-hidden views, with D50's attacking-frame tokens, the y mirror, three seeds per slot and 8,100 steps) at ownership cost 14, followed by D27's kick-off rule (`stack_trial build`, SHA-256 `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`). C1 is not run again (C3). What C1 checked carries over because the reading, centring, pitch, frame indexing, half spans, label assembly and the D27 call are the standing code, unchanged, and with the old build command (`player_trial build`) the code still writes `a56574fe…`. One thing is new on the build path and was not covered by C1: the attacking sign of each graded half, computed from that half's tracking from its tagged kick-off on. It is guarded by three fail-closed checks instead (each half has a sign, it agrees with the first qualifying frame from the kick-off on, the two halves are opposite), which refused once before the rule was aligned to the kick-off and passed afterwards (D48's adoption records). C1's PASS applies to `0ddbb342…`; the new file is checked by C2's re-declaration for D48.

## D7 — Unseen-vendor benchmark on PFF World Cup tracking (2026-11-11, written before any PFF data is read or scored)

**Question.** How does the frozen D3 submission score on tracking from a vendor that none of its models has seen? The competition match is tracked by Metrica, and every score we have so far is on DFL or SkillCorner, the two vendors the models were trained on. D7 is a measurement only. It changes nothing in the submission and serves as the reference for D8–D10.

**Data.** PFF FC's 2022 FIFA World Cup release: broadcast tracking, metadata and rosters for 64 matches. Before anything is parsed, the licence terms and the access date are saved to `rules/PFF_TERMS.md`. If the terms do not allow this use, D7–D10 stop and D3 ships unchanged. File sizes and SHA-256 hashes are pinned in `dfl_submission/pff_sources.json` (`python -m dfl_submission.pff pin`), and every later step checks them.

**Reader** (`dfl_submission/pff.py`, modelled on `skillcorner_2020.py`):
- periods 1 and 2 only;
- unsmoothed `homePlayers`, `awayPlayers` and `balls` points, kept only where the vendor marks them `VISIBLE`. Estimated points are dropped, so the input resembles detection-only tracking. A frame with two visible balls has no ball;
- native 29.97 fps resampled by time to the 25 Hz grid, without filling a missing native point;
- points more than 5 m outside the metadata pitch discarded;
- players identified by team and shirt number from the roster. An unrostered visible shirt number is an error, not a silent drop. Home and away come from the metadata team ids.

**Labels** (from the vendor's game events, carried forward from each event's `start_frame` within a period):
- `FIRSTKICKOFF`, `SECONDKICKOFF` and `OTB` (on-the-ball) events start live play, owned by the home team when `home_ball` is true and by the away team when it is false. A missing `home_ball` keeps the previous team;
- `OUT` starts a dead ball (`D`);
- `END` and missing video (`VID`) make the following frames `X`;
- other events (substitutions, ball staying in play, players on or off) change nothing;
- frames before the period's kick-off, or without a vendor tracking frame, are `X`.

The event names were taken from the open-source kloppy PFF reader and checked against its public PFF sample files. Once the data is downloaded, this mapping is checked against PFF's own data specification. If the specification disagrees, this record is amended before any score is computed, the sanity gate included. In particular, if fouls or offsides stop play without an `OUT` event, those stoppages remain live in the targets. The amendment then says so, and the D targets are a known underestimate.

**Split.** Sort the 64 numeric game ids and deal them in turn to eight slots: development, confirmation A, confirmation B, confirmation C, training, training, training, untouched (`pff.split`). That gives 8 development, three confirmation sets of 8, 24 training and 8 untouched matches. Dealing in turn rather than in blocks keeps each set spread across group and knockout games, because the ids follow the tournament order. Both halves of a match stay together.

**Reader sanity gate.** On the 8 development matches, the organisers' smooth50 baseline (nearest visible player to the visible ball, then a ±50-frame majority vote) must score a mean raw spellF1 between 0.15 and 0.40 inclusive. It scores 0.25–0.26 on DFL and SkillCorner. The gate checks alignment, team orientation and labels without using our models. If it fails, only the reader may be fixed, using reader-level evidence (label shares, visibility, event timing), and the gate is run again. Every attempt is recorded here. D3 is not scored until the gate passes.

**Measurement.** `python -m dfl_submission.vendor_trial measure` scores frozen D3 (D1 costs, D3 blend, cutoff 0.53) on the 8 development matches: spellF1 and macroF1 over both halves of each match, averaged equally over matches, against the raw labels and the N = 1.75, 3, 4 and 4.5 s targets. Nothing is selected. The record goes to `dfl_submission/vendor_recipe/d7_measurement.json`, which is written once. The confirmation sets, training and untouched matches are not scored in D7.

**What follows from it.** D9 is triggered only if D3's mean N = 3 spellF1 here is more than 0.04 below its 0.664 on the four SkillCorner holdout matches, that is, below 0.624. D8 uses this measurement as its reference.

Every D7 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** Run with the reader amendment and terms note below.
- **Sanity gate:** passed on the first attempt. The smooth50 baseline's mean raw spellF1 was 0.34476 (per match 0.293–0.413), inside 0.15–0.40. That is above its 0.25–0.26 on DFL and SkillCorner, which fits PFF's coarser targets: possession sequences, with no stoppage except `OUT`. [Record](dfl_submission/vendor_recipe/sanity.json)
- **Frozen D3 on the eight development matches:**

  | Target | spellF1 | macroF1 |
  |---|---|---|
  | Raw provider labels | 0.54874 | 0.84922 |
  | N = 1.75 s | 0.59325 | 0.85104 |
  | N = 3 s | 0.62560 | 0.85304 |
  | N = 4 s | 0.64030 | 0.85501 |
  | N = 4.5 s | 0.64128 | 0.85536 |

  N = 3 spellF1 ranged from 0.471 (game 10502) to 0.718 (game 10510). The baseline scored 0.33866 against the same N = 3 targets. [Record](dfl_submission/vendor_recipe/d7_measurement.json)
- **What follows:** D3's N = 3 spellF1 of 0.62560 is 0.038 below its 0.664 on the SkillCorner holdout, and above the 0.624 trigger, so D9 is not triggered. The margin is 0.0016. D8 uses 0.62560 as its reference. The submission is unchanged, and the confirmation, training and untouched PFF matches remain unparsed and unscored.

*Amendment, 2026-11-11, after reading PFF's Tracking Data Specification v2.2 and before any PFF tracking file was downloaded or scored:*
- The specification lists exactly these game event types: `FIRSTKICKOFF`, `SECONDKICKOFF`, `END`, `G`, `OFF`, `ON`, `OTB`, `OUT`, `SUB` and `VID`. It defines `home_ball` as whether the home team has control of the ball during the possession sequence. The mapping above is unchanged.
- The specification has no foul, offside or other stoppage event apart from `OUT` and `END`. Stoppages where the ball stays on the pitch therefore remain live in the D7 targets, owned by the last team. The `D` targets underestimate dead ball, and a predicted stoppage at a foul counts against every candidate scored on PFF, including the baseline.
- The specification places the frame rate inside a `videos` dict, while the public sample has it at the top level. The reader accepts either.

*Terms, 2026-11-11, before any PFF file was parsed:* the dataset ships with no licence of its own. The only applicable terms found are Gradient Sports' general website Terms of Use, which do not mention datasets but bar derivative works and "benchmarking" of site content. PFF releases the dataset publicly for analysis. Derek decided to run D7 before Gradient Sports replied to our question about this use, because D7 changes nothing in the submission and the repository publishes only code, hashes and averaged scores. Only the eight development games are parsed. Nothing trained on PFF enters the submission without written permission or a judgement disclosed in the technical document.

*Reader amendment, 2026-11-11, after the first `prepare` stopped and before any score was computed:* the reader refused game 3812 because some frame numbers repeat within a period. All eight development games do this, with 29–72 repeated frame numbers per game. The copies always sit on adjacent rows, one for each possession event on that frame. They differ only in `possession_event`, `possession_event_id` and `ballsSmoothed`, which the reader never uses, and never in the players, balls or game event. The reader now reads adjacent copies once when those fields agree. It still stops on copies that disagree, and on a frame number that repeats on non-adjacent rows.

## D8 — Model-set weight on unseen PFF tracking (2026-11-11, written before any weight other than D3's is scored)

**Question.** D3 averages the current and retrained model sets with equal weight, because no real tracking was available to show which dead-ball condition the competition match resembles. PFF's broadcast tracking comes from a vendor none of the models has seen, and its ball keeps moving during stoppages. Does it support a different share of the retrained set?

**What is new, and what is reused.** D3 fixed its weight at 0.5 and said it would not be tuned on the DFL and SkillCorner matches. D8 does not tune it there. It chooses the weight on a different vendor, and uses the DFL and SkillCorner development matches only as guards. Those matches have now been seen by D2, D3, D4 and D6, so the guards are weaker evidence than a first look; this is disclosed. The PFF development matches were scored once before, in D7, for D3 only.

**Known bias.** PFF's targets have no foul or offside stoppage (D7 amendment), so a candidate that misses such stoppages is penalised less on PFF than on DFL. The DFL dead-ball-carried guard bounds this.

**Grid.** The share of the retrained set, w ∈ {0, 0.25, 0.5 (D3), 0.75, 1}. Every model output is mixed exactly as the build does (`retrain_trial.mix`, before decoding). Everything else is D3's frozen recipe: stoppage cutoff 0.53, ownership cost 6, stoppage cost 2, discount 0.8, the same mixes and phase averaging. The pitch comes from the PFF metadata.

**Pipeline checks, before any result is read.** The w = 0.5 run must reproduce exactly the D7 PFF measurement and the D3 DFL/SkillCorner development record (`blend_recipe/development.json`). If either differs, D8 stops without writing a record, and only the pipeline may be fixed.

**Primary.** Mean spellF1 over the eight PFF development matches against the N = 3 targets. Ties are broken by mean macroF1 against the same targets.

**Eligible if,** compared with D3 (w = 0.5) on development:
- the primary rises by at least +0.02;
- mean spellF1 against the N = 1.75, 4 and 4.5 s targets each drops by no more than 0.01;
- no PFF development match's N = 3 spellF1 drops by more than 0.03;
- on the DFL/SkillCorner development matches (WPY and the four SkillCorner development matches), against the N = 3 targets, dead ball carried and team swap each drop by no more than 0.01, and standard combined spellF1 drops by no more than 0.02. The standard limit is looser than D2's 0.01 because those views still contain the provider artifacts.

Raw provider labels are reported, not gated.

**Selection.** The eligible weight with the highest primary, ties broken by macroF1. If no weight is eligible, D3 stays and nothing is confirmed.

**Confirmation.** Once, on PFF confirmation set A (eight matches, parsed only for this step), against D3, with the three PFF gates above: primary gain ≥ +0.02, the other conventions within 0.01, and no match more than 0.03 down. The DFL/SkillCorner guards are not re-scored. They are deterministic for the frozen weight, and the DFL/SkillCorner holdout is not used again. Pass → the blend weight changes. Fail → D3 stays. Either way, the outcome is recorded here.

**Terms condition.** A passed D8 would put a value chosen on PFF data into the submission. The weight therefore changes only after Gradient Sports confirms the use in writing, or after Derek records a decision to proceed and the technical document discloses it. Otherwise D3 ships unchanged.

Records: `dfl_submission/vendor_recipe/d8_development.json` and `d8_confirmation.json`, each written once. Every D8 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** No weight was eligible, so D3 stays and confirmation set A was neither parsed nor scored. Both pipeline checks passed: the w = 0.5 run reproduced the D7 measurement and the D3 development record exactly. Development, compared with D3:

| w | PFF N = 3 spellF1 | Change | Worst match | N = 1.75 / 4 / 4.5 | Standard | Dead ball carried | Team swap |
|---|---:|---:|---:|---|---:|---:|---:|
| 0 | 0.63370 | +0.0081 | −0.0095 | +0.006 / +0.004 / +0.003 | +0.0052 | **−0.0683** | **−0.0108** |
| 0.25 | 0.63270 | +0.0071 | −0.0066 | +0.006 / +0.006 / +0.005 | +0.0017 | **−0.0406** | **−0.0131** |
| 0.5 (D3) | 0.62560 | — | — | — | — | — | — |
| 0.75 | 0.61755 | −0.0081 | −0.0252 | −0.010 / −0.008 / −0.007 | −0.0058 | +0.0097 | +0.0029 |
| 1 | 0.61101 | −0.0146 | −0.0260 | **−0.016 / −0.016 / −0.014** | −0.0123 | +0.0166 | −0.0022 |

No weight reached the +0.02 primary gain. The highest PFF score, at w = 0, gained +0.0081, and would also have failed both DFL guards. PFF leans slightly towards the current models, as the known bias predicts: its targets keep foul stoppages live. The lean is less than half the required gain, and the current models lose 0.068 when the dead ball moves. Against raw PFF labels, spellF1 was 0.5545, 0.5539, 0.5487, 0.5400 and 0.5346 for w = 0 to 1. [Record](dfl_submission/vendor_recipe/d8_development.json)

## D11 — Where frozen D3 loses spellF1 (2026-11-11, written before anything is scored)

**Question.** D3 scores 0.74–0.76 spellF1 on clean DFL tracking, so about a quarter of the score is lost in segmentation and timing rather than tracking noise. Is the loss in stoppages, in ownership changes or in boundary timing? D11 is a measurement only. It selects nothing and changes nothing in the submission. It fixes the boundary shifts that D12 may test, and shows which of D12–D14 has the most headroom.

**Candidate and data.** Frozen D3 only: the 0.5 model-set blend, stoppage cutoff 0.53, ownership cost 6, stoppage cost 2, discount 0.8, the same mixes and phase averaging. Development matches only:
- DFL WPY in its four views: clean, masked, dead ball carried and team swap (the D2 guard views, from the raw position files);
- the four SkillCorner development matches;
- the eight PFF development matches, on the parsed inputs that passed the D7 sanity gate.

Holdout, confirmation, training and untouched matches are not read.

**Pipeline check, before any result is read.** The D11 run must reproduce exactly D3's DFL/SkillCorner development record (`blend_recipe/development.json`, selected trial) and the D7 PFF measurement (`vendor_recipe/d7_measurement.json`). If either differs, D11 stops without writing a record, and only the pipeline may be fixed.

**Targets.** The raw provider labels and the N = 3 s controlled-possession targets. Groups: DFL clean, DFL masked, DFL dead ball carried, DFL team swap, SkillCorner and PFF.

**What is measured.**
1. **Spell detection** at each IoU from 0.3 to 0.7: precision, recall and F1, with true and predicted spell counts summed per group.
2. **Two oracle runs**, scored with spellF1 like D3:
   - *True stoppages:* frames the target calls dead become `D`. On target-live frames that D3 calls `D`, the team is the nearest D3 team in time within that target-live segment, earlier first. A target-live segment that D3 calls dead throughout takes the team with the higher mean blended home-ownership probability.
   - *True ownership:* D3's `D` frames stay `D`. Every other frame takes the target's team; where the target is dead there, the target's most recent team in the same run, or else its next one.
3. **Signed boundary offset** (predicted frame minus true frame). Transitions come from `pm.transitions` and are matched one-to-one with `pm.match_transitions` at a tolerance of 25 frames with `typed="pair"`. They are pooled into three types: *ownership* (H→A, A→H), *stop* (H→D, A→D) and *restart* (D→H, D→A). For each type and group: matched count, median, 25th and 75th percentiles, missed and spurious counts. Per vendor, DFL pools the clean and masked views, SkillCorner its four matches and PFF its eight.
4. **Spell length:** recall at IoU 0.5 and mean recall over the five IoUs, by true spell length (under 1.75 s, 1.75–3 s, 3–5 s, 5–10 s, 10–20 s, 20 s or more). The number of predicted *interior* spells (an A–B–A inside a predicted live segment) under 1.75 s and under 3 s, and how many of them are matched at IoU 0.5. Stoppage lengths, predicted and true: count, 10th, 50th and 90th percentiles, and the number under 1 s and under 3 s.

**Boundary shifts handed to D12 (fixed here, before scoring).** For each transition type, take the N = 3 median offsets of DFL, SkillCorner and PFF. The shift is minus the mean of the DFL and SkillCorner medians, rounded to the nearest frame with halves away from zero. It is used only if all three medians are non-zero with the same sign and the rounded shift is at least 2 frames. Otherwise that type has no shift. PFF can only block a shift; it never sets its size, so no value chosen on PFF enters the submission.

**Record.** `dfl_submission/diagnosis_recipe/d11.json`, written once by `python -m dfl_submission.diagnosis measure`. Every D11 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** The pipeline check passed: decoded from the cached inputs, D3 reproduced the D3 development record and the D7 measurement exactly. Against the N = 3 targets:

| Group | D3 spellF1 | True stoppages | True ownership | True / predicted spells | Precision / recall at IoU 0.5 |
|---|---:|---:|---:|---:|---|
| DFL clean | 0.742 | 0.746 | 0.934 | 281 / 219 | 0.88 / 0.68 |
| DFL masked | 0.662 | 0.680 | 0.925 | 281 / 215 | 0.78 / 0.60 |
| DFL dead ball carried | 0.566 | 0.661 | 0.793 | 281 / 253 | 0.61 / 0.55 |
| DFL team swap | 0.579 | 0.598 | 0.923 | 281 / 241 | 0.63 / 0.54 |
| SkillCorner | 0.676 | 0.726 | 0.846 | 971 / 795 | 0.76 / 0.62 |
| PFF | 0.626 | 0.701 | 0.783 | 1783 / 1438 | 0.72 / 0.58 |
| Standard combined | 0.6821 | 0.7138 | 0.8869 | | |

- **Ownership is where the score is lost.** True ownership adds +0.205 to the standard combined score; true stoppages add +0.032. Only where the dead ball moves (DFL dead ball carried, PFF) are stoppages a large share.
- **D3 under-segments.** It predicts 20–25% fewer spells than the targets, with high precision and low recall. Recall at IoU 0.5 by true spell length on SkillCorner: 4% under 1.75 s, 40% at 1.75–3 s, 48% at 3–5 s, 72% at 5–10 s, 84–92% above 10 s. DFL clean is similar (5%, 40%, 53%, 80%, 94–98%).
- **There are almost no brief interior spells to collapse.** D3 predicts none under 1.75 s in any group, and 0, 2, 4 and 8 under 3 s on DFL clean, DFL masked, SkillCorner and PFF.
- **Boundary timing is not biased.** N = 3 median offsets (frames, predicted minus true) for ownership, stop and restart were −1, −2 and −0.5 on DFL, +2, 0 and 0 on SkillCorner, and −2, −8 and −2 on PFF. Spreads (interquartile ranges) are 8–25 frames. The vendors disagree in sign for every type, so D11 hands **no shift** to D12.

What follows: D12's output collapse and shifts have little to act on, and only the 1-frame step remains a live option there. The headroom is in ownership changes that D3 misses, which is what D13 addresses. [Record](dfl_submission/diagnosis_recipe/d11.json)

## D12 — Decoder output refinements (2026-11-11, written after D11's record and before any D12 setting is scored)

**Question.** Does decoding at the metric's resolution, or removing brief interior spells from the decoded labels, raise spellF1? Inference only: the models, the blend and the decoder costs are D3's.

**What was seen before this rule.** D11's record, which shows that D3 predicts almost no brief interior spells and hands no boundary shift. The grid below is the one the plan fixed before D11, minus the shift option that D11 removed. Nothing else is changed in response to D11.

**Grid (6 settings, in this order).** Decoder step 5 (D3) or 1 frame × output collapse none, 1.75 s or 3 s.
- *Step 1* keeps the evidence per second and the costs of step 5: each frame's evidence is a fifth of a 5-frame block's, and the stoppage cutoff offset is scaled the same way. The boundary discount at a frame uses that frame's boundary signal; at step 5 it uses the block's maximum. A stoppage decoded inside a half lasts at least 5 frames, as at step 5. Without that, a one-frame stoppage between two teams would cost 2 + 2 instead of the ownership cost of 6, silently lowering it. With constant boundary signals and block-constant evidence the two steps give identical labels (`tests/test_refine.py`).
- *Output collapse:* `conventions.collapse_brief_changes` on each decoded half, after decoding. 1.75 s is the lower end of D1's interval.

**Pipeline check, before any result is read.** The D3 setting, decoded afresh from D11's cached inputs, must reproduce the D3 development record and the D7 measurement exactly. Otherwise D12 stops without writing a record.

**Primary.** Standard combined spellF1 against the N = 3 targets on DFL WPY (1/3 clean, 2/3 masked) and the four SkillCorner development matches.

**Eligible if,** compared with D3 on development:
- the primary rises by at least +0.005;
- standard combined spellF1 at N = 1.75, 4 and 4.5 s each drops by no more than 0.01;
- DFL dead ball carried and team swap, at N = 3, each drop by no more than 0.01;
- *PFF veto:* mean N = 3 spellF1 over the eight PFF development matches drops by no more than 0.01.

**Selection.** The eligible setting with the highest primary, then the highest N = 3 standard macroF1, then grid order. If none is eligible, D3 stays and nothing is confirmed.

**Confirmation, once.** On the DFL/SkillCorner holdout (WQQ, WR9 and their guard views; the four SkillCorner holdout matches) and PFF confirmation set A, parsed only for this step, with the same gates against D3. Before the challenger is scored, the D3 setting must reproduce D3's holdout record (`blend_recipe/external_validation.json`). The DFL/SkillCorner holdout has confirmed D1–D3 before, so this is reused evidence; set A is fresh.

**If it passes,** the setting becomes the submission (`refine_trial build`) and the reference for D13, after the rebuild checks. PFF only vetoes; no value chosen on PFF enters the submission. **If it fails,** D3 stays.

Records: `dfl_submission/refine_recipe/development.json`, `recipe.json` and `confirmation.json`, each written once. Every D12 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** No setting was eligible, so D3 stays and confirmation set A was neither parsed nor scored. The pipeline check passed. Development, compared with D3:

| Step | Collapse | N = 3 standard | Change | N = 1.75 / 4 / 4.5 | Dead ball carried | Team swap | PFF |
|---|---|---:|---:|---|---:|---:|---:|
| 5 | none (D3) | 0.68209 | — | — | — | — | — |
| 5 | 1.75 s | 0.68209 | 0 | 0 / 0 / 0 | 0 | 0 | 0 |
| 5 | 3 s | 0.68061 | −0.0015 | −0.004 / +0.004 / +0.005 | −0.0019 | +0.0074 | +0.0014 |
| 1 | none | 0.68098 | −0.0011 | −0.003 / −0.000 / −0.000 | −0.0004 | +0.0015 | −0.0014 |
| 1 | 1.75 s | 0.68098 | −0.0011 | −0.003 / −0.000 / −0.000 | −0.0004 | +0.0099 | −0.0013 |
| 1 | 3 s | 0.68065 | −0.0014 | −0.007 / +0.006 / +0.006 | −0.0024 | +0.0152 | −0.0011 |

As D11 predicted, collapsing at 1.75 s changes nothing on the standard views, because D3 predicts no interior spell that short there. The 1-frame step moved 0.4% of frames in a test half and lowered the primary slightly. The decoder's resolution is not what limits the score. [Record](dfl_submission/refine_recipe/development.json)

## D13 — Train the models on controlled-possession targets (2026-11-11, written before any model is refit or scored)

**Question.** D1 moved only the decoder to controlled possession. The ownership and boundary heads still learn every-touch labels, so the boundary signal fires on brief touches and the ownership signal flips on them. D11 shows that ownership is where the score is lost. Do heads trained on controlled-possession targets do better?

**What was seen before this rule.** D11's and D12's records. The candidate and grid below are the ones the plan fixed before D11.

**Candidate.** Both model sets, the original and the D2-augmented one, are refit with ownership and ownership-boundary targets computed after `collapse_brief_changes(labels, 3)` on each 25 Hz training half. Stoppage targets are unchanged, because the collapse never touches a dead frame. Per set:
- the long-context ownership tree is refit; the stoppage tree is unchanged and reused, since its targets and inputs are identical;
- the two temporal networks (seed 42 with boundary weight 0.5, seed 97 with 1.4) are retrained on Apple MPS and exported, with the same features, hyperparameters, seeds and steps.

The 0.5 blend, the mixes, phase averaging, discount 0.8 and stoppage cost 2 stay.

**How the refit reuses the original inputs.** Feature extraction is unchanged, so the refit reads the original prepared training sequences (`external_temporal/train_data` for the original set, the D2 `retrain/sequences` for the augmented set) and recomputes only their targets from the 25 Hz labels. Three pipeline checks run before any candidate is scored:
1. for every prepared half, the stored targets must be reproduced exactly from the reloaded labels without the collapse;
2. refitting each set's ownership tree from the prepared inputs with the original targets must reproduce the published `ownership.ubj` byte for byte;
3. retraining the original set's seed-42 network on its original sequences is compared with the published checkpoint and the result reported. If it does not reproduce, the record says so: the candidate then differs from D3 by its targets and by run-to-run training variation.

If check 1 or 2 fails, D13 stops and only the pipeline may be fixed.

**Grid (15 settings).** Ownership cost ∈ {2, 3, 4, 6, 8} × stoppage cutoff ∈ {0.49, 0.53, 0.57}, in that order. The aligned models should need a lower ownership cost.

**Pipeline check on scoring.** D3, decoded from D11's cached inputs, must reproduce the D3 development record and the D7 measurement exactly.

**Primary.** Standard combined spellF1 against the N = 3 targets on DFL WPY and the four SkillCorner development matches.

**Eligible if,** compared with D3 on development:
- the primary rises by at least +0.01;
- standard combined spellF1 at N = 1.75, 4 and 4.5 s each drops by no more than 0.01. The N = 1.75 guard matters most here, because the models are trained towards the N = 3 convention they are scored on;
- DFL dead ball carried and team swap, at N = 3, each drop by no more than 0.01;
- *PFF veto:* mean N = 3 spellF1 on the PFF development matches drops by no more than 0.01.

**Selection.** The eligible setting with the highest primary, then N = 3 standard macroF1, then grid order. None eligible → D3 stays.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and PFF confirmation set B (parsed only for this step), with the same gates against D3. D3 must first reproduce its holdout record.

**If it passes,** the aligned models and the selected costs become the submission after the rebuild checks. **If it fails,** D3 stays. Either way the outcome is recorded here.

Code: `--convention` in `long_context` and `temporal_network prepare`, a `retarget` path for prepared sequences, and `dfl_submission/aligned_trial.py`. Weights go to `dfl_submission/aligned_models/`, records to `dfl_submission/aligned_recipe/`, each written once. Every D13 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** No setting was eligible, so D3 stays and confirmation set B was neither parsed nor scored.
- **Pipeline checks all passed.** (1) Every stored target in both prepared folders (48 halves each) was reproduced from the reloaded labels. (2) Refitting each set's ownership tree from the prepared inputs reproduced the published `ownership.ubj` byte for byte. (3) Retraining the original seed-42 network on MPS reproduced all three published checkpoints bit for bit, so the candidate differs from D3 only by its targets. On scoring, D3 reproduced the D3 development record and the D7 measurement exactly.
- **Development, compared with D3** (N = 3 standard 0.68209). Every setting failed the N = 1.75 guard, and none raised the primary:

| Ownership cost | Best cutoff | N = 3 standard | Change | N = 1.75 / 4 / 4.5 | Dead ball carried | Team swap | PFF |
|---|---|---:|---:|---|---:|---:|---:|
| 2 | 0.53 | 0.67722 | −0.0049 | −0.012 / −0.005 / −0.006 | −0.0111 | −0.0046 | −0.0112 |
| 3 | 0.53 | 0.68146 | −0.0006 | −0.012 / +0.007 / +0.005 | −0.0116 | +0.0048 | −0.0064 |
| 4 | 0.57 | 0.67706 | −0.0050 | −0.017 / +0.005 / +0.005 | −0.0209 | +0.0191 | −0.0035 |
| 6 | 0.53 | 0.68207 | −0.0000 | −0.015 / +0.011 / +0.011 | −0.0065 | +0.0211 | +0.0004 |
| 8 | 0.53 | 0.67779 | −0.0043 | −0.020 / +0.014 / +0.013 | −0.0080 | +0.0161 | −0.0022 |

  The other cutoffs scored lower on the primary for every cost; all 15 settings are in the record.

What it shows: the aligned models move the score from the short convention to the long ones. At D3's costs they gain +0.011 at N = 4 and 4.5 and +0.021 on team swaps, lose 0.015 at N = 1.75 and 0.020 against raw labels, and leave N = 3 unchanged. They predict slightly fewer spells (209 against 219 on DFL clean), so the under-segmentation D11 found gets a little worse. Lower ownership costs did not recover the missed changes. Training on controlled-possession targets does not address D3's recall problem. The aligned weights are kept in `dfl_submission/aligned_models/` for reference. [Record](dfl_submission/aligned_recipe/development.json)

## D14 — SkillCorner 2019–20 ownership labels in training (2026-11-11, written before any model is refit or scored)

**Question.** D11 shows that D3 misses ownership changes. Does more labelled ownership data help? The plan rates this the lowest-value step, because its gain is measured on SkillCorner itself, which says little about Metrica tracking.

**What was seen before this rule.** D11–D13's records. D6 scored D3 on the five 2019–20 development matches (N = 3 spellF1 0.62374). The four 2019–20 confirmation matches have never been scored.

**Candidate.** The five 2019–20 development matches (2068, 2269, 2417, 2440, 2841, revision `1b99d401`, MIT) join the training of both model sets as a third source, `skillcorner2020`. Their tracking is read as in D6 and turned into the same 162 features.
- **Targets: ownership only.** The release has no dead-ball label, so each frame's stoppage and stoppage-boundary losses are switched off (`known` = 0). The ownership and ownership-boundary targets come from the provider's home/away possession, raw and uncollapsed, like the other sources. Frames without a provider team are excluded from the ownership loss. The D2 augmentation does not apply, because there are no dead runs to simulate, so both sets get the same 2019–20 halves.
- **Trees:** each set's ownership tree is refit on its original rows plus the 2019–20 live rows. The existing balancing gives each source a third of the weight and equal weight to matches within a source. The stoppage trees are unchanged and reused.
- **Networks:** the four networks are retrained on each set's original prepared halves plus these ten, with the same seeds, boundary weights and steps. With three sources, each batch draws 5 windows per source instead of 8, a direct consequence of adding a source.
- The decoder recipe and the 0.5 blend are D3's. There is no grid.

**Development gates,** against D3:
- on DFL WPY and the four SkillCorner development matches, N = 3 standard combined spellF1, dead ball carried and team swap each drop by no more than 0.01;
- *PFF veto:* mean N = 3 spellF1 on the PFF development matches drops by no more than 0.01.

No gain is required on development, because the matches the candidate is meant to help are scored only at confirmation. Other conventions and raw labels are reported, not gated.

**Confirmation, once,** if development passes. The four never-scored 2019–20 matches (3442, 3518, 3749, 4039): mean N = 3 spellF1 per match over both halves must rise by at least +0.01 over D3, scored as in D6 (frames without a provider team are not scored). PFF confirmation set C, parsed only for this step, is a veto: its mean N = 3 spellF1 drops by no more than 0.01.

**If it passes,** the D14 models become the submission after the rebuild checks. **If it fails,** D3 stays.

Code: `dfl_submission/extra_trial.py`; weights in `dfl_submission/extra_models/`, records in `dfl_submission/extra_recipe/`, each written once. Every D14 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** Failed at development, so D3 stays. The four 2019–20 confirmation matches remain unscored, and PFF confirmation set C was neither parsed nor scored. D3 reproduced the D3 development record and the D7 measurement exactly. Compared with D3:

| Gate | Change | Required |
|---|---:|---|
| N = 3 standard combined | −0.0179 (0.6821 → 0.6642) | ≥ −0.01 |
| Dead ball carried | −0.0490 | ≥ −0.01 |
| Team swap | −0.0279 | ≥ −0.01 |
| PFF development (veto) | −0.0227 | ≥ −0.01 |

Reported only: N = 1.75, 4 and 4.5 changed by −0.030, −0.014 and −0.016, and raw-label standard spellF1 by −0.029. The loss is spread over every view: DFL WPY lost 0.010–0.049 across its four views, and three of the four SkillCorner development matches lost about 0.016. The largest drop is with the dead ball carried, so the shared network trunk lost stoppage skill, even though the 2019–20 halves carry no stoppage loss. Giving a third of the training weight to five matches without a dead-ball label, with fewer windows per batch from each of the original sources, made both heads worse. The D14 weights are kept in `dfl_submission/extra_models/` for reference. [Record](dfl_submission/extra_recipe/development.json)

## D15 — Network width and training length (2026-11-11, written before any candidate network is trained or scored)

**Question.** Each temporal network has about 32,000 parameters (width 48, six blocks) and trains for 1,800 steps. Neither value was ever searched (`CONSTANTS.md`), and in the original grid the later snapshot always won. D11 places the loss in missed ownership changes, and D12–D14 showed that the decoder, the targets and more data do not recover them. Do wider or longer-trained networks?

**What was seen before this rule.** D11–D14's records. The grid below is the one the third plan fixed before any of it was run. One pipeline check was run before this rule was written: the trainer with its new options at their defaults retrained the original seed-42 network on MPS, and it reproduced the published checkpoints bit for bit.

**Candidate.** Per network setting, all four networks are retrained: in each of the two model sets, seed 42 with boundary weight 0.5 and seed 97 with 1.4. The features, seeds, boundary weights, batches and prepared sequences are D3's. The original set trains on the original prepared sequences (manifest `d3891c24…`), the augmented set on the D2 sequences (manifest `2828b0c3…`).
- **Width** is the number of channels in every block. **Steps** is the training length. The cosine learning-rate schedule is stretched over the new length, and snapshots are kept at the stretched 1/3, 2/3 and 3/3 points. Only the last snapshot is used, as D3 uses its 1,800-step snapshot.
- The XGBoost trees are D3's and are not refit. The 0.5 blend, the mixes, phase averaging, discount 0.8, stoppage cost 2 and stoppage cutoff 0.53 stay.

**Grid (12 settings, in this order).** Network setting (width, steps) ∈ (48, 1,800), (48, 5,400), (96, 1,800), (96, 5,400) × ownership cost ∈ {4, 6, 8}. The first network setting is D3's own networks, and at cost 6 it is D3. Its rows at costs 4 and 8 are scored and reported but are not eligible, because that would be a decoder-only change, which D1 covers.

**Pipeline check on scoring.** D3, decoded from D11's cached inputs, must reproduce the D3 development record and the D7 measurement exactly. Otherwise D15 stops without writing a record.

**Primary.** Standard combined spellF1 against the N = 3 targets on DFL WPY and the four SkillCorner development matches.

**Eligible if,** compared with D3 on development:
- the primary rises by at least +0.01;
- standard combined spellF1 at N = 1.75, 4 and 4.5 s each drops by no more than 0.01;
- DFL dead ball carried and team swap, at N = 3, each drop by no more than 0.01;
- *PFF veto:* mean N = 3 spellF1 on the eight PFF development matches drops by no more than 0.01.

**Selection.** The eligible setting with the highest primary, then N = 3 standard macroF1, then grid order. None eligible → D3 stays. If no new network setting beats D3 by at least +0.003 at cost 6, the plan moves straight to D16.

**Confirmation, once,** on the DFL/SkillCorner holdout (WQQ, WR9 and their guard views, and the four SkillCorner holdout matches) and PFF confirmation set A, parsed only for this step, with the same gates against D3. D3 must first reproduce its holdout record (`blend_recipe/external_validation.json`). The DFL/SkillCorner holdout has confirmed D1–D3 before, so it is reused evidence; set A is fresh.

**If it passes,** the selected networks and cost become the submission after the rebuild checks, which include both Linux platforms because a wider network changes the arithmetic. They also become the reference for D16. PFF only vetoes; no value chosen on PFF enters the submission. **If it fails,** D3 stays.

Code: `--width` and `--steps` in `temporal_network train` (defaults byte-identical), and `dfl_submission/scale_trial.py`. Weights go to `dfl_submission/scale_models/`, records to `dfl_submission/scale_recipe/`, each written once. Every D15 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** Passed, and adopted: width 96, 5,400 steps, ownership cost 8 is the submission and the reference for D16.
- **Pipeline checks all passed.** The trainer at its defaults retrained the original seed-42 network bit for bit on MPS. D3 reproduced its development record and the D7 measurement before development, and its holdout record before confirmation.
- **Development, compared with D3** (N = 3 standard 0.68209). Every new network setting was eligible at every cost. Rows at cost 6, plus the selected row:

| Width | Steps | Cost | N = 3 standard | Change | N = 1.75 / 4 / 4.5 | Dead ball carried | Team swap | PFF |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 48 | 1,800 | 6 | 0.68209 | — (D3) | — | — | — | — |
| 48 | 5,400 | 6 | 0.71043 | +0.0283 | +0.029 / +0.012 / +0.003 | +0.0222 | +0.0560 | +0.0217 |
| 96 | 1,800 | 6 | 0.69235 | +0.0103 | +0.007 / +0.005 / +0.004 | +0.0103 | +0.0153 | +0.0051 |
| 96 | 5,400 | 6 | 0.72165 | +0.0396 | +0.049 / +0.018 / +0.004 | +0.0424 | +0.0610 | +0.0291 |
| **96** | **5,400** | **8** | **0.72280** | **+0.0407** | **+0.048 / +0.024 / +0.016** | **+0.0429** | **+0.0808** | **+0.0305** |

  D3's own networks at costs 4 and 8 (reported, not eligible) scored 0.67817 and 0.68612. All 12 rows are in the record. The best new setting beat D3 by +0.040 at cost 6, so the +0.003 early-stop line did not apply.
- **Confirmation, once, compared with D3:**

| Gate | Change | Required |
|---|---:|---|
| N = 3 standard combined, holdout (0.68734 → 0.70874) | +0.0214 | ≥ +0.01 |
| N = 1.75 / 4 / 4.5 | +0.039 / +0.013 / +0.005 | each ≥ −0.01 |
| Dead ball carried | +0.0124 | ≥ −0.01 |
| Team swap | +0.0586 | ≥ −0.01 |
| PFF confirmation set A (0.6355 → 0.6542, fresh) | +0.0187 | ≥ −0.01 |

What it shows: the networks were under-trained. Training length mattered more than width: tripling the steps added +0.028 at width 48, doubling the width added +0.010 at 1,800 steps, and both together added +0.040. The holdout gain is about half the development gain. That is expected after choosing the best of nine eligible settings, and the holdout has been reused since D1, so +0.021 is the better estimate. The fresh PFF set agrees in sign and size. [Development](dfl_submission/scale_recipe/development.json) · [Confirmation](dfl_submission/scale_recipe/confirmation.json)

The competition file is built by `scale_trial build`; its SHA-256 is `14d18a238afce84b663cc430161240e00c93b62e3a65a0e6aa47577288b7af78` (reproduced twice on macOS in a fresh virtual environment, and in `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`; each build passed the validator).

## D16 — Last-touch and possession-chain features (2026-11-11, written before D15 was scored and before any model with these features is fit or scored)

**Question.** The features hold nearest-player distances, averaged contact strength and scalar ball speed. Nothing encodes a change in the ball's direction next to a player, which team touched last, or how long a team's run of touches has lasted, which is the state possession is defined by. On clean DFL tracking D3 matches only 68 of 184 true ownership changes within one second (D11). Do explicit touch features recover them?

**What was seen before this rule.** D11–D14's records and the D15 rule; no D15 candidate had been scored. To time the extraction, the new columns were computed once on one external training half (DFL WMX, first half, clean view) and their summary statistics printed. Nothing in the definitions below was changed after that.

**New features** (`dfl_submission/touch_features.py`, 22 columns). They are computed at 25 Hz from the ball and player positions and sampled at the long-context frames, every 5th frame. Each is either team-free or an `h__`/`a__` pair, so swapping the teams swaps the pair. All are missing where the ball is unseen at the sampled frame.
- *Ball turn* over 5 and 10 frames: the size of the change of the ball's velocity, in m/s, between the 5 (10) frames before a frame and the 5 (10) frames after it.
- *Touch evidence* per team: each ball turn times exp(−(d / 2.5 m)²), where d is the distance from the ball to the team's nearest player.
- *Touches:* a local maximum of the 5-frame turn (over ±5 frames) of at least 4 m/s with a player within 2.5 m; the team with the nearer player touched the ball.
- Per team: time since its last touch and until its next, capped at 20 s; its touches in the past and coming 3 s and 9 s; how long its current run of consecutive touches has lasted, when the latest touch is its own; and how long the run starting with the next touch lasts, when that touch is its own (0 otherwise, capped at 20 s).
- Every threshold is hand-set and not tuned.

**Candidate.** Both model sets are refit with the 162 existing columns plus these 22, on the same training halves, targets, sample weights and hyperparameters:
- the long-context ownership tree, and the stoppage tree on the team-invariant form of the extended table;
- then, if the tree gate below passes, the two networks of each set, at the width and length standing after D15 (D3's 48 and 1,800 unless D15 is adopted), with D3's seeds and boundary weights.

The 0.5 blend, the mixes, phase averaging, discount 0.8 and stoppage cost 2 stay.

**Pipeline checks, before anything is scored.**
1. For every prepared training half of both sets, the 162 re-extracted columns must equal the stored prepared sequences exactly.
2. Refitting both trees of each set from the original prepared sequences must reproduce the published trees byte for byte.
3. D3, decoded from D11's cached inputs, must reproduce the D3 development record and the D7 measurement exactly. (If D15 is adopted, its model must first reproduce its own development record.)

If any check fails, D16 stops and only the pipeline may be fixed.

**Tree gate.** The new trees with the standing networks are scored on the grid below. If the best primary change over the grid, against the reference, is under +0.003, the networks are not retrained and the reference stays. The tree-only stage is a gate, not a candidate.

**Grid (12 settings, in this order).** Ownership cost ∈ {3, 4, 6, 8} × stoppage cutoff ∈ {0.49, 0.53, 0.57}.

**Primary.** Standard combined spellF1 against the N = 3 targets on DFL WPY and the four SkillCorner development matches.

**Eligible if,** compared on development with the submission standing after D15 (the reference; D3 unless D15 is adopted):
- the primary rises by at least +0.01;
- standard combined spellF1 at N = 1.75, 4 and 4.5 s each drops by no more than 0.01;
- DFL dead ball carried and team swap, at N = 3, each drop by no more than 0.01;
- the SkillCorner part of the primary (the mean over the four SkillCorner matches) does not drop;
- mean N = 3 spellF1 on the eight PFF development matches does not drop.

DFL labels change on every touch, so a model that follows touches more closely could gain on DFL while moving away from controlled possession. The SkillCorner and PFF labels are possession-based and catch that.

**Selection.** The eligible setting with the highest primary, then N = 3 standard macroF1, then grid order. None eligible → the reference stays.

**Reported, not gated.** D11's ownership-change matching (within one second) and recall by true spell length, on clean DFL, SkillCorner and PFF, for the reference, the tree-only stage at D3's costs and the full refit at its best primary setting.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and PFF confirmation set B, parsed only for this step, with the same conditions against the reference, which must first reproduce its holdout record.

**If it passes,** the D16 models and the selected costs become the submission after the rebuild checks. **If it fails,** the reference stays.

Code: `dfl_submission/touch_features.py` and `dfl_submission/touch_trial.py`. Weights go to `dfl_submission/touch_models/`, records to `dfl_submission/touch_recipe/`, each written once. Every D16 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** Failed at the tree gate. The networks were not retrained, and D15 stays the submission. Confirmation set B was neither parsed nor scored.
- **Pipeline checks all passed.** All 96 prepared training halves (48 per set) were re-extracted exactly. All four trees refit byte for byte from the original prepared sequences. D3 reproduced its development record and the D7 measurement, and the adopted D15 model reproduced its own development record.
- **Tree gate, compared with D15** (N = 3 standard 0.72280; ownership cost 8, cutoff 0.53). The best change was −0.00043, against the +0.003 needed to train the networks. Best cutoff per ownership cost:

| Ownership cost | Best cutoff | N = 3 standard | Change | SkillCorner part | N = 1.75 / 4 / 4.5 | Dead ball carried | Team swap | PFF |
|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 3 | 0.53 | 0.71152 | −0.0113 | −0.0104 | +0.006 / −0.020 / −0.025 | −0.0121 | −0.0289 | −0.0098 |
| 4 | 0.49 | 0.71343 | −0.0094 | −0.0044 | +0.006 / −0.017 / −0.022 | −0.0043 | −0.0208 | −0.0045 |
| 6 | 0.53 | 0.72238 | −0.0004 | +0.0027 | +0.008 / −0.008 / −0.013 | +0.0019 | −0.0177 | −0.0027 |
| 8 | 0.49 | 0.72217 | −0.0006 | +0.0044 | +0.005 / −0.001 / −0.003 | +0.0075 | −0.0081 | −0.0008 |

- **Reported: segmentation at D15's recipe, against the N = 3 targets.** On clean DFL, ownership changes matched within one second rose from 68 of 184 (D3, D11) to 93 (D15) and 104 (tree-only stage). Predicted spells went from 219 to 263 and 282, against 281 true. Recall at IoU 0.5 for 3–5 s spells went from 0.53 to 0.69 and 0.76. On SkillCorner and PFF the tree-only stage changed almost nothing: 243 → 244 of 554 and 319 → 315 of 1,024 ownership changes matched.

What it shows: the touch features help the trees follow touches on clean DFL, as the rule anticipated, but that does not carry over to the possession-based SkillCorner and PFF labels. Every setting lost at N = 4 and 4.5 and on team swaps. Most of the missed-change gap that D11 found was closed by D15's longer training, not by new inputs. The D16 trees are kept in `dfl_submission/touch_models/trees/` for reference. [Tree gate record](dfl_submission/touch_recipe/tree_gate.json)

## D19 — Metrica sample-match benchmark (2026-11-11, written before any prediction or score on these games is computed)

**Question.** Every labelled source so far comes from another vendor. The competition match is tracked by Metrica, and Metrica's three public sample matches follow the conventions CHALLENGE.md describes for its ground truth: corrected tracking in which the ball is absent while dead, and events that carry the team. How does frozen D3 score on them, and can they veto later candidates beside PFF? D19 is a measurement and a veto definition. It changes nothing in the submission.

**Data and licence.** [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) at commit `e706dd50` (2021-04-15), still the repository head on 2026-11-11: games 1 and 2 (CSV tracking per team and CSV events), game 3 (FIFA EPTS metadata and tracking, JSON events), the README and Metrica's event definitions. Sizes and SHA-256 hashes are pinned in `dfl_submission/metrica_sources.json`. These are the same commit and hashes as the files D1 used for its convention choice, plus game 3's tracking and the definitions, fetched now.
- *Licence finding:* the repository has no licence file, and GitHub reports none. The README offers the data to prospective clients and "to the broader football analytics community" to play around with. It asks users to be responsible with it and to acknowledge the source in anything public. Metrica Sports owns the data and also organises this competition.
- *Our use:* local evaluation only, with no training. Nothing is redistributed: the repository holds code, hashes and averaged scores, and the technical document credits Metrica Sports.
- *Judgement:* we read this use as within the README's terms, and it matches the D1 use already disclosed. The Nov 3 email allows "any dataset you have the rights to use". Derek can overrule this; if he does, D19 is withdrawn and nothing else changes.

The sample games are anonymised and are not the competition match.

**Reader** (`dfl_submission/metrica_sample.py`):
- periods 1 and 2 at their native 25 Hz, with every player of both teams (a substitute is missing while off the pitch). Home is the first team: the CSV home file, or EPTS's local team, which is listed first;
- coordinates are normalised, with (0, 0) at the top left of a 105 × 68 m pitch. The README states this for games 1–2, and game 3's metadata gives the same field size. They are converted exactly as the build converts the competition's Metrica EPTS export: multiplied by the pitch, then centred, with no axis flipped. Points more than 5 m outside the pitch are dropped;
- game 3 follows each of its eleven player layouts, which change with substitutions. The CSV home and away files must agree on every frame and on the ball.

**Targets.** Scored over both halves of a game, like every other source.
- *Span:* each period is labelled from its kick-off to its last visible ball. Frames outside are `X`.
- *Dead ball:* `D` where the tracking has no ball. A missing-ball run shorter than 1 s with the ball seen on both sides is a tracking dropout inside play and stays live.
- *Team (raw):* set pieces, recoveries, passes, carries, shots, balls lost, balls out and faults received mark their team at their start frame. A ball lost also marks the opponent at its end frame: Metrica defines it as losing possession to an opponent. Challenges and cards mark nothing. Within each live segment, a frame takes the latest mark, ties in file order. Frames before the segment's first mark take the team at that mark, since a restart's set piece can follow the ball's reappearance by a few frames.
- *Conventions:* N = 1.75, 3, 4 and 4.5 s via `decoder_trial.relabel` on the raw targets, as for every source.
- *Control:* D1's control rule, applied to the raw targets. A team's spell counts as controlled if any of these holds:
  - it is the first spell of its live segment;
  - it contains that team's set piece, completed pass (Metrica logs failed passes as balls lost) or a carry of at least 0.5 s;
  - the team holds the ball for at least 0.5 s from the spell's start to its first release (a pass, shot, ball lost or ball out).

  Otherwise possession stays with the last team in control. The carry length is an amendment to D1's wording ("completed pass or carry"), made from event durations before any score. Game 3 logs carries as short as one frame: 40% of its 1,395 carries last under 0.5 s. Counting every carry kept 88 of its 298 interior spells through a carry alone, while games 1–2 log no carries. With the length rule, all three games relate the same way to the collapses (control spell counts 242, 237 and 265, next to N = 4–4.5's 244/244, 245/233 and 280/270).

**What was seen before this rule.** Reader-level evidence only, with no prediction or score:
- Ball missing on frames inside pass and carry flights: 0, 1, 0, 0, 0 and 2 frames in the six halves, out of 13,491–42,984 flight frames each.
- Set pieces with the ball missing on more than half of the 2 s before them (excluding each period's opening kick-off): 73 of 75 in game 1, 76 of 78 in game 2 and 88 of 88 in game 3.
- The first visible ball falls exactly on the kick-off frame in all six halves. In games 1–2 the last visible ball falls on the half's last event.
- Missing-ball runs shorter than 1 s with play visible on both sides: 24–36 per half, 61–175 frames each. Every live segment contains at least one possession mark.
- Raw label shares: `D` 28–51% per half, `X` 0.2–3.0%.
- Spell counts per game: raw 384, 371 and 404; N = 3 272, 259 and 294.

**Reader sanity gate.** All three conditions must hold before D3 is scored:
1. *Dead-ball convention,* per game with both halves pooled. The ball is missing on at most 1% of frames inside pass and carry flights. At least 90% of set pieces other than each period's opening kick-off have the ball missing on more than half of the 2 s before them.
2. *Smooth50 baseline* (`vendor_trial.baseline_prediction`: the nearest visible player to the visible ball, then a ±50-frame majority vote). Its mean raw spellF1 over the three games lies between 0.15 and 0.50 inclusive. The upper bound is wider than D7's 0.40 because this tracking is clean and the ball disappears exactly when dead, so the baseline should do better here than on any source so far. A gross misalignment still falls below 0.15.
3. *Team orientation:* in every game, the baseline's raw macroF1 is higher than that of the same baseline with `H` and `A` swapped.

If the gate fails, only the reader may be fixed, using reader-level evidence, and the gate is run again. Every attempt is kept in `dfl_submission/metrica_recipe/sanity.json`.

**Measurement.** `python -m dfl_submission.metrica_sample measure` scores frozen D3 on all three games: the D3 blend, cutoff 0.53, ownership cost 6, stoppage cost 2 and discount 0.8. The teacher is `competition_adaptation.reference()`, and decoder inputs come from `diagnosis.half_signals`. It reports spellF1 and macroF1 over both halves of each game, averaged equally over games, against the raw targets, N = 1.75, 3, 4 and 4.5, and control. The record is written once to `dfl_submission/metrica_recipe/d3_measurement.json`. D3's decoder inputs are cached in `data/metrica_cache/` (gitignored). Nothing is selected.

**Known biases.**
- *Corrected tracking.* The tracking is corrected, not automatic: all players are present and unfragmented, and the ball is seen exactly when live. It flatters every model compared with the competition's automatic tracking, where the ball is missing on 41% of graded frames.
- *Missing-ball shortcut.* Because the dead ball is exactly the missing ball, these games reward the shortcut D2 removed. A candidate that relies on it less, like D2's retrained set, can lose here for that reason alone. The veto limits such a loss; it never rewards a gain.
- *Small sample.* Three matches, with 237–404 spells each depending on the target.

**Veto for later candidates.** From this record on, a candidate's mean N = 3 spellF1 over the three games may drop by at most 0.01 against D3. `metrica_sample.veto` takes the candidate's labels, its decoder inputs, or a function of D3's cached inputs. It first checks that D3, decoded from the cache, reproduces `d3_measurement.json` exactly. Other targets and the worst game are reported, not gated. As with PFF, the Metrica games only veto: no value is chosen on them, and nothing is trained on them. The veto binds decisions written after D19. Decisions written earlier keep their own gates and may report their Metrica change.

Every D19 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).**
- **Reader sanity gate:** passed on the first attempt.
  - *Dead-ball convention:* the ball was missing on 0.004%, 0% and 0.002% of flight frames in games 1–3. 97.3%, 97.4% and 100% of later set pieces followed a missing ball.
  - *Smooth50 baseline:* mean raw spellF1 0.4232 (games 0.448, 0.420 and 0.402). This is inside 0.15–0.50, and above its 0.345 on PFF and 0.25–0.26 on DFL and SkillCorner, as expected for clean tracking with an exact dead-ball signal.
  - *Team orientation:* with `H` and `A` swapped, the baseline's raw macroF1 fell from 0.83–0.87 to 0.37–0.40 in every game.

  [Record](dfl_submission/metrica_recipe/sanity.json)
- **Frozen D3 on the three games:**

  | Target | spellF1 | macroF1 | smooth50 spellF1 |
  |---|---:|---:|---:|
  | Raw event possession | 0.62511 | 0.91465 | 0.42318 |
  | N = 1.75 s | 0.71114 | 0.91784 | |
  | **N = 3 s (veto)** | **0.76049** | 0.92103 | 0.42719 |
  | N = 4 s | 0.76524 | 0.92233 | |
  | N = 4.5 s | 0.76177 | 0.92150 | |
  | Control rule | 0.83236 | 0.92994 | 0.44263 |

  N = 3 spellF1 per game was 0.7235, 0.7839 and 0.7741. [Record](dfl_submission/metrica_recipe/d3_measurement.json)
- **Reading.** This is D3's highest N = 3 score on any source. For comparison: DFL clean 0.742, SkillCorner development 0.676, PFF development 0.626. That fits the biases above: corrected tracking, and a dead ball that is exactly the missing ball. D3 scores higher against the control-rule targets (0.832) than against any collapse. This is consistent with D11's finding that D3 under-segments: its spells are closer to the longer controlled-possession convention. The score is not an estimate of the competition score.
- **Veto reference.** From here on, a candidate is vetoed if its mean N = 3 spellF1 on these games is below 0.75049. D3 passed through `veto` (as labels, as decoder inputs, and as a function of the cached inputs) reproduced the record with a change of exactly zero. The submission is unchanged.

*Amendment, 2026-11-11, after D15 was adopted and before D15 is scored on these games:* while D19 ran, D15 (width 96, 5,400 steps, ownership cost 8) replaced D3 as the submission. The veto now compares a candidate with the submission standing when the candidate's rule is written, which from now on is D15.
- *Measurement.* D15 is measured once, exactly as D3 was:
  - on the same parsed inputs that passed the gate;
  - with the model `scale_trial build` uses: D3's trees and blend with D15's networks and recipe;
  - with decoder inputs from `diagnosis.half_signals`, cached under `data/metrica_cache/d15/`;
  - against the same targets.

  The record is written once to `dfl_submission/metrica_recipe/d15_measurement.json`. Before D15 is scored, D3 decoded from its cache must reproduce `d3_measurement.json` exactly; otherwise nothing is written.
- *Veto.* `veto(candidate, reference=...)` takes `D15` by default and `D3` for decisions written while D3 was the submission. The veto line is the reference's mean N = 3 spellF1 minus 0.01. For a candidate given as decoder inputs, the default decoder recipe is the reference's.
- *What it does not do.* D15's change against D3 here is reported only. It does not reopen D15, whose rule was written before D19, and nothing is selected.

**Outcome of the amendment (2026-11-11).** D3 decoded from its cache reproduced `d3_measurement.json` exactly, and D15 was then measured once:

| Target | D15 spellF1 | D15 macroF1 | Change against D3 (reported) |
|---|---:|---:|---:|
| Raw event possession | 0.69292 | 0.92809 | +0.0678 |
| N = 1.75 s | 0.77470 | 0.93057 | +0.0636 |
| **N = 3 s (veto)** | **0.78004** | 0.92997 | **+0.0196** |
| N = 4 s | 0.76183 | 0.92809 | −0.0034 |
| N = 4.5 s | 0.75128 | 0.92647 | −0.0105 |
| Control rule | 0.83660 | 0.93533 | +0.0042 |

- **Per game:** N = 3 spellF1 was 0.7655, 0.7895 and 0.7852, so every game improved on D3 (the smallest gain was +0.0056).
- **Spells:** D15 predicts more of them than D3 (249, 236 and 273 against 214, 213 and 247). That is the recall gain D15 found on DFL, and it carries over: the shorter conventions gain most, while N = 4.5 drops slightly.
- **Veto reference:** from here on it is D15, so a candidate is vetoed if its mean N = 3 spellF1 on these games is below 0.77004. Passed back through `veto`, D15 reproduced its record with a change of exactly zero.

[Record](dfl_submission/metrica_recipe/d15_measurement.json)

## D18 — PathCRF ball-free possession check (2026-11-11, written before any PathCRF output is computed or scored)

**Question.** Does a published ball-free model infer team possession better than frozen D3 on frames where our ball is hidden? PathCRF (Kim et al., KDD 2026) picks one possession edge per time step from player trajectories alone, using a set-attention network and a CRF. Our ball is hidden on 41% of graded competition frames (DATA.md), and D3's features are built around the ball. D18 is round 4's Track B1. It is a measurement only and changes nothing in the submission. Its one consequence is whether B2 runs: retraining PathCRF on our four training matches with our degradations.

**Code and checkpoint, vetted before anything was run.**
- The repository is `github.com/hyunsungkim-ds/pathcrf` at commit `1b7c6f85a6b78c92e4543fde72ca2b86a933f997` (8 June 2026). It is cloned outside this repository and treated as untrusted. The code licence is MPL-2.0 (LICENSE SHA-256 `3f3d9e00…`), and the checkpoints ship in the same repository. No PathCRF code is copied here: `dfl_submission/pathcrf_trial.py` imports the model classes from the clone at run time.
- One checkpoint only, fixed here: trial 140, `saved/140/model/state_dict_best_acc.pt` (SHA-256 `55757d9d…`), loaded with `weights_only=True`. Trial 140 is the dynamic masked CRF, the paper's main model and the one its tutorial uses. The other nine checkpoints are not scored, so no choice is made between them.
- Inference follows the published pipeline: 25 Hz downsampled to 5 Hz, 10 s windows (50 steps) overlapping by half and blended linearly, then the CRF's Viterbi decode over each segment. It runs on CPU.

**Leakage finding.** The paper does not name its split; the code fixes it. `train.py` sorts the seven DFL files and, for these checkpoints, trains on the first five, validates on the sixth and tests on the seventh. Trial 140 uses leave-one-match-out fold 6, which gives the same split. So every checkpoint was trained on WMX, WN1, WOH, WOY **and WPY**, picked its best epoch on **WQQ**, and was tested on WR9. This agrees with the tutorial, which evaluates on the last file (WR9), and with Table 1's five training matches, one validation match and one test match.
- WPY is our only DFL development match, and it is in PathCRF's training set. Scores there are contaminated in PathCRF's favour.
- WQQ and WR9 are our holdout, and neither is scored in D18. WR9 is the only DFL match PathCRF never saw. But the plan scores the holdout once, at the final confirmation, and a B2 decision taken on it would weaken that confirmation. WQQ chose PathCRF's checkpoint.
- The four SkillCorner development matches are uncontaminated: PathCRF never saw SkillCorner. They are harsher than the competition in one respect. Broadcast tracking shows about 13 of 22 players per live frame, against a mean of 20.5 tracked objects per graded competition frame (DATA.md). Their ball is hidden on 5–9% of live frames.

So D18 uses two sets with opposite biases. WPY is contaminated and favours PathCRF; its masked view shows about 17.6 of 22 players per frame. SkillCorner development is unseen and harsher for PathCRF.

**Adaptor** (`dfl_submission/pathcrf_trial.py`). It has no learned or tuned values; every constant is set here.
- **Halves** come from `augment.guard_halves(dfl, skill, "development", raw=True)`: DFL WPY clean, masked, dead ball carried and team swap, plus the four SkillCorner development matches (observed). The dead-ball-carried view moves only the ball, so PathCRF's input there is the masked view's.
- **Coordinates:** centred metres, mapped linearly onto PathCRF's 105 × 68 m frame with the origin at a corner.
- **Missing players:** each slot's gaps between its first and last observation in the half are filled by linear interpolation in time, with nothing extrapolated beyond them. This is required: the model cannot take a missing player, because its training dropped every episode that had one. Team-swap identity switches are left as they are, since they are the test.
- **Features**, as in PathCRF's preprocessing, per slot over the half:
  - frames implying more than 12 m/s are dropped and re-interpolated;
  - velocity comes from a Savitzky–Golay filter (15 frames, order 2) on frame differences;
  - acceleration magnitude comes from a second filter (9 frames, order 2);
  - a slot observed on fewer than 17 frames uses plain differences.
- **Per segment:**
  - Each team's active slots are those with a position somewhere in the segment, at most 11, keeping the most observed. Frames outside a slot's observed span hold its nearest position, with zero velocity.
  - The goalkeeper is the team's slot whose mean x lies farthest from the centre line over the half, among slots observed on at least 10% of the half's frames. The team whose goalkeeper is further left in the segment comes first.
  - Four fixed outside nodes are added, then padding to 26 nodes, as in PathCRF.
- **Team from the decoded edge** at each 5 Hz step, held until the next step:
  - a player holding the ball, or passing to a teammate or out of play: that player's team;
  - a ball in flight to an opponent: the passer's team for the first half of that edge's run and the receiver's for the second half (PathCRF's own "split" rule);
  - the ball out of play: no team.

**Runs.**
- *Native:* PathCRF on each run of target-live frames. This is the setting in which it was trained and evaluated.
- *Realistic:* PathCRF on each run of D3-live frames, so no target information reaches the model.

**Targets and frames.** The N = 3 s controlled-possession targets are primary; the raw provider labels are also reported, as in D11. Frames are target-live frames: all of them, and those where the view's ball is hidden (not finite).

**What is measured,** per view and group (DFL clean, masked, dead ball carried, team swap; SkillCorner).
1. **Team-possession accuracy** on target-live frames, all and ball-hidden, pooled over frames. Both models get the true stoppages by D11's rule. A target-live frame with no predicted team takes the nearest predicted team in time within that live segment, the earlier one on ties. A live segment with no predicted team takes the team with the higher mean home probability. For D3 that is its blended home-ownership probability. For PathCRF it is 1 on home frames, 0 on away frames and 0.5 on frames with no team, with ties going home as in D11.
   - D3's predictions are its cached decoded labels.
   - PathCRF's come from the native run (1a) and from the realistic run (1b). The realistic run enters as a hybrid: D3's labels, with the team replaced by PathCRF's wherever both give one.
2. **spellF1 and macroF1 of the hybrid** against both targets, next to D3's: per match, per group, and the standard combined aggregate (DFL 1/3 clean + 2/3 masked, SkillCorner mean).
3. **SkillCorner ball-hidden accuracy by visible players per frame** (under 11, 11–14, 15 or more), reported only.

**Pipeline checks, run before any PathCRF output is read.**
- **D3 reproduction:** D3's numbers come from the D11 cache (`data/diagnosis_cache/development`), DFL and SkillCorner entries only. The PFF entries are never opened. Decoded from the cache, D3 must reproduce exactly D11's group spellF1 means and true-stoppage oracle means for the five DFL and SkillCorner groups, under both targets. The labels of every half built by `guard_halves` must equal the cached labels.
- **Adaptor sanity gate:** on WPY clean, PathCRF's native team-possession accuracy on target-live frames against the N = 3 targets must be at least 0.80. WPY is a training match for PathCRF, and the paper reports 92.4% on its test match. If the gate fails, only the adaptor may be fixed (coordinates, orientation, node order, features), using adaptor-level evidence, and the gate is run again. Every attempt is recorded here. Nothing else is computed until the gate passes.

**B2 trigger.** B2 runs only if PathCRF beats D3 on native team-possession accuracy against the N = 3 targets, on ball-hidden target-live frames pooled over the halves, in both of these sets:
- **T1, the WPY masked view.** This is the trigger as first set for the track. It is contaminated in PathCRF's favour, so failing it is decisive.
- **T2, the four SkillCorner development matches**, which are uncontaminated.

"Beats" means strictly higher. Everything else is reported, not gated. B2 would get its own decision record, written before anything is trained, and D19's Metrica veto would bind it.

**Record.** `dfl_submission/pathcrf_recipe/d18_measurement.json` is written once by `python -m dfl_submission.pathcrf_trial measure` and carries the PathCRF commit and file hashes. Every D18 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`. Inference runs on CPU, one job at a time.

**Outcome (2026-11-11).** Both pipeline checks passed. Decoded from the cache, D3 reproduced D11's group means exactly under both targets, and every half's labels matched the cache. The adaptor sanity gate passed on the first attempt: 0.888 on WPY clean (81,460 target-live frames). **B2 is not triggered: PathCRF loses to D3 on both sets.**

| Team-possession accuracy, N = 3 targets, target-live frames | Frames | D3 | PathCRF native | Hybrid |
|---|---:|---:|---:|---:|
| WPY clean, all | 81,460 | 0.910 | 0.888 | 0.888 |
| **WPY masked, ball hidden (T1)** | 32,011 | **0.870** | **0.595** | 0.583 |
| WPY masked, all | 81,460 | 0.884 | 0.597 | 0.585 |
| WPY dead ball carried, ball hidden | 32,011 | 0.856 | 0.595 | 0.585 |
| WPY team swap, ball hidden | 32,011 | 0.839 | 0.555 | 0.542 |
| **SkillCorner, ball hidden (T2)** | 23,200 | **0.786** | **0.710** | 0.736 |
| SkillCorner, all | 319,119 | 0.915 | 0.855 | 0.859 |

WPY clean has no ball-hidden frames. The raw labels agree: T1 0.858 for D3 against 0.608 for PathCRF, and T2 0.788 against 0.720.
- **Clean tracking:** on a match it was trained on, PathCRF comes close to D3 but does not beat it: 0.888 against 0.910, or 0.894 against 0.897 on raw labels.
- **Masked view:** PathCRF falls to about 0.60, near a coin toss, on ball-hidden and ball-visible frames alike. It never sees the ball, so the drop comes from the degraded player tracks. Team swaps take it to 0.56.
- **SkillCorner:** PathCRF holds up better (0.855 overall), even with about 9 interpolated players per frame, but still trails D3 by 0.060 overall and by 0.076 on ball-hidden frames. On ball-hidden frames, by visible players (reported only):
  - under 11 visible: 0.738 against D3's 0.832 (19,044 frames);
  - 11–14 visible: 0.656 against 0.618 (1,172 frames);
  - 15 or more: 0.552 against 0.562 (2,984 frames).
- **Hybrid spellF1 collapses:** standard combined 0.337 against D3's 0.682 at N = 3. PathCRF's team changes at every touch and at every pass in flight to an opponent. On clean WPY against the raw labels the hybrid beats D3 (0.623 against 0.555): it follows the touch-level changes that DFL records and that the controlled-possession convention removes.

The submission is unchanged. WQQ, WR9 and PFF were not read. [Record](dfl_submission/pathcrf_recipe/d18_measurement.json) (SHA-256 `7f8366ab…`).

*Post-hoc diagnostic, written after the record above and before the diagnostic is run.* The masked view adds two things to the clean tracks:
- 0.2 m of independent noise on every frame;
- 2 s dropouts on 20% of track blocks.

PathCRF's velocity and acceleration features are frame differences, so the per-frame noise may matter more than the dropouts. To separate them, PathCRF's native run is repeated on WPY with each degradation alone:
- *noise only:* the clean tracks plus exactly the masked view's noise, the first draw of its generator (`default_rng(270000 + 100 × 4 + half).normal(0, 0.2)`), checked equal to the masked view wherever the masked view has a position;
- *dropouts only:* the clean tracks with exactly the masked view's missing blocks, and no noise.

Accuracy is reported as above, with the ball-hidden frames of the masked view. The diagnostic is reported only. It cannot trigger B2, it changes nothing in the adaptor, and WPY remains a PathCRF training match. Record: `dfl_submission/pathcrf_recipe/d18_diagnostic.json`, written once by `python -m dfl_submission.pathcrf_trial diagnose`.

*Diagnostic result (2026-11-11).* The noise-only view reproduced the masked view's noise exactly.

| PathCRF native accuracy on WPY, N = 3 targets | All target-live frames | Ball hidden in the masked view |
|---|---:|---:|
| Clean | 0.888 | — |
| Dropouts only | 0.877 | 0.884 |
| Noise only | 0.546 | 0.558 |
| Masked (both) | 0.597 | 0.595 |
| *D3, masked view* | *0.884* | *0.870* |

The raw labels agree: 0.883 with dropouts only, 0.560 with noise only.
- **The per-frame noise causes the collapse.** PathCRF's features are frame differences. Under its 12 m/s outlier rule, independent 0.2 m noise floods them, and nothing in its training data was noisy.
- **Dropouts are cheap for it.** Interpolated 2 s gaps on 20% of track blocks cost 0.011.
- **Caveats:** WPY is a PathCRF training match, and D3 was not run on the dropouts-only view. So the 0.884 against 0.870 is not a like-for-like comparison and decides nothing.

[Record](dfl_submission/pathcrf_recipe/d18_diagnostic.json) (SHA-256 `db3f758d…`).

**What this means for B2 (not triggered).**
- **Against a retrain:** retraining with noise and our degradations might recover the noise loss. But even fully recovered, it would at best restore PathCRF's clean-track level, which trails D3 on a match PathCRF was trained on. On the uncontaminated SkillCorner matches it trails D3 by 0.060 overall and by 0.076 on ball-hidden frames. Its touch-level output also halves the hybrid's spellF1 under the controlled-possession convention.
- **The open question**, if anyone returns to this, is whether a noise-robust PathCRF beats D3 on ball-hidden frames of an unseen DFL match. Answering it needs a retrain that leaves out a match (WPY) and the holdout. That is B2's cost, and D18 gives no reason to pay it.

## D20 — Model search beyond D15 (2026-11-11, written before any D20 model is trained or scored)

**Question.** D15 showed that the temporal networks were under-trained: tripling the training length added +0.028 and doubling the width +0.010. Nothing beyond width 96 and 5,400 steps was tried, no other model family was ever compared on the same inputs, and the tree heads' hyperparameters were never searched (`CONSTANTS.md`). Does a searched model, or an equal-weight ensemble of searched models, beat D15? This is round 4's Track A.

**What was seen before this rule.** The records of D1–D19, including D18 and D19. D15's training logs were read to size the grid. No D20 model has been trained, and nothing has been scored.

**Reference.** D15, the submission: D3's trees and 0.5 model-set blend, D15's four networks (width 96, 5,400 steps), cutoff 0.53, ownership cost 8, stoppage cost 2, discount 0.8.

**Families.** Each family's models replace one part of D15's structure. Everything else stays D15's: the four network slots (in each model set, seed 42 with boundary weight 0.5 and seed 97 with 1.4), the mixes, phase averaging, the 0.5 blend and the recipe. Every network trains on the same prepared sequences as D15 (manifests `d3891c24…` and `2828b0c3…`), with the same windows (128 output steps with 63 steps of context on each side), batches, loss, AdamW settings, gradient clipping and swap augmentation. The cosine schedule is stretched over the run as in D15, and only the last snapshot is used.
1. **Temporal network** (D15's architecture). Width ∈ {96, 192} × steps ∈ {5,400, 10,800, 19,800}. 19,800 rather than 20,000, because the stretched snapshot points need a multiple of 3. (96, 5,400) is D15's own setting and anchors the search.
2. **Bidirectional GRU.** The same normalised inputs (162 columns and their validity flags), a linear projection to h units, a two-layer bidirectional GRU with h units per direction (dropout 0.08 between layers), a layer norm and a linear layer to the four outputs. h ∈ {64, 128} × steps ∈ {5,400, 10,800}.
3. **Transformer encoder.** A linear projection to d units plus fixed sinusoidal positions over the 254 window steps, three pre-norm encoder layers (4 heads, feed-forward 2d, GELU, dropout 0.08), a layer norm and a linear layer to the four outputs. d ∈ {64, 128} × steps ∈ {5,400, 10,800}. Its learning rate rises linearly over the first 1/18 of the steps, then follows the stretched schedule.
4. **XGBoost tree heads.** Both heads of both sets refit as `long_context.fit_head` fits them, from the prepared rows (the D16 check showed the published trees refit byte for byte this way), with max depth ∈ {3, 5, 7} × trees ∈ {420, 840, 1,260} at learning rate 0.045. The other parameters are D3's. Each head (ownership, stoppage) selects its own setting. (5, 420) is D3's own setting and anchors the search.

The GRU and the transformer see exactly the training windows at inference too: each block of 128 steps is predicted from a window with 63 steps of context on each side, zero-padded beyond the half like missing inputs. LightGBM and CatBoost are not searched: they would add build dependencies, and XGBoost is already in the build.

**Tuning, on the training matches only.** Match-grouped cross-validation over four folds. Fold k holds out DFL training match k in sorted order (WMX, WN1, WOH, WOY), with its clean, masked and stress views, and the SkillCorner training matches at sorted positions k, k + 4 and k + 8. To keep the cost affordable, each setting is cross-validated with one representative model per fold, trained on the other folds of the original set's prepared sequences:
- *Networks:* seed 42, boundary weight 0.5. The score is frame-level log-loss on the held-out prepared frames, with probabilities averaged over the input and its team swap as at inference: ownership on live frames plus stoppage on known frames. Each match's score is the mean over its halves. Matches are averaged within each source, and the two sources weigh equally. Each match is held out exactly once. Boundary heads are reported only.
- *Trees:* each head's log-loss on the held-out rows, weighted as the fit weighs them (`long_context.source_weights`: sources, matches and classes balanced). The ownership head is scored with its swap-averaged probability, as at inference.

The setting with the lowest cross-validated loss is selected, with ties going to grid order. Only that setting is then trained in full on every training match: the four networks for a network family, or all four trees. If a family selects its anchor, its member is D15 itself and it adds nothing. Nothing from the development, holdout, PFF or Metrica matches enters this step.

**Development, each family once.** A family's member is D15's structure with that family's selected models. It is scored once on the development matches at D15's recipe: DFL WPY in its four views, the four SkillCorner development matches and the eight PFF development matches.

**Ensemble.** The pool is D15 plus each family's member, in the order D15, temporal network, GRU, transformer, trees. Greedy forward selection on the development primary at D15's recipe:
- start from the pool member with the highest primary;
- add the member whose inclusion gives the highest primary, where an ensemble decodes the equal-weight mean of its members' four decoder inputs;
- stop when no addition raises the primary or at four members. Ties go to macroF1, then pool order.

If the greedy selection is D15 alone, D15 stays. Otherwise the ownership cost is re-selected for the selected ensemble from {6, 8, 10, 12}, with cutoff 0.53, stoppage cost 2 and discount 0.8 fixed. It selects once.

**Primary.** Standard combined spellF1 against the N = 3 targets on DFL WPY (1/3 clean, 2/3 masked) and the four SkillCorner development matches.

**Eligible if,** compared with D15 on development:
- the primary rises by at least +0.01;
- standard combined spellF1 at N = 1.75, 4 and 4.5 s each drops by no more than 0.01;
- DFL dead ball carried and team swap, at N = 3, each drop by no more than 0.01;
- *PFF veto:* mean N = 3 spellF1 on the eight PFF development matches drops by no more than 0.01;
- *Metrica veto (D19, reference D15):* mean N = 3 spellF1 on the three Metrica sample matches, from `metrica_sample.veto`, is at least 0.77004.

**Selection.** The eligible cost with the highest primary, then N = 3 standard macroF1, then grid order. None eligible → D15 stays.

**Pipeline checks, before anything is scored.** D15, decoded from its cached development inputs, must reproduce exactly its development record (the selected trial in `scale_recipe/development.json`, PFF development included). Before confirmation, D15 must reproduce its holdout record (`scale_recipe/confirmation.json`). A network family's NumPy export must match its PyTorch outputs to within 2e-5 absolute plus 1e-5 relative on the logits, the D15 export test. A family that fails this is left out of the pool and recorded.

**Confirmation, once,** on the reused DFL/SkillCorner holdout (WQQ, WR9 and their guard views; the four SkillCorner holdout matches) and PFF confirmation set C, parsed only for this step, with the same gates against D15. PFF and Metrica only veto. The holdout has confirmed D1, D3 and D15 before, so it is reused evidence; set C is fresh. Sets A and B are spent, and the untouched set stays in reserve.

**If it passes,** the selected ensemble and cost become the submission after the rebuild checks: twice on macOS in a fresh virtual environment, then in `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`. Every adopted model has a deterministic NumPy or XGBoost CPU path, and the build environment stays without PyTorch. Then C1 runs once. **If it fails,** D15 stays.

**Schedule.** Families run in pool order, one heavy job at a time on this machine (Apple MPS for networks). A family whose member has not been scored on development by 2026-11-14 12:00 EST is left out of the pool and recorded as not run. The ensemble is selected once, after the last family is scored or at that time, whichever comes first. No cloud run is planned, because the estimated local compute fits. A cloud run would need an amendment here before it runs.

Code: `dfl_submission/zoo.py`. Cross-validation records go to `dfl_submission/zoo_recipe/cv_<family>.json`, weights to `dfl_submission/zoo_models/`, and the development, recipe and confirmation records to `dfl_submission/zoo_recipe/`, each written once. Every D20 training, tuning and scoring command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

*Amendment, 2026-11-11, after the temporal-network cross-validation started and before any GRU or transformer is trained.* A timing run on random tensors (no data, no loss looked at) measured PyTorch's GRU on Apple MPS at 0.6–0.7 s per training step, against 0.10–0.17 s on the CPU. The transformer took 0.04 s per step on MPS. So the GRU networks train on the CPU with 6 threads, and everything else about them is unchanged. The transformer and the temporal network stay on MPS.

*Amendment, 2026-11-11, before any cloud run and before any GRU or transformer is trained.* Derek decided, through the coordinating session, that the GRU and transformer families are cross-validated and trained on Kaggle GPUs. They run there in parallel with the local temporal-network and tree searches, and this replaces the GRU-on-CPU amendment for the cloud runs.
- **Unchanged:** the grids, folds, cross-validation networks, scoring and selection, the four full-training slots, and the pool order. Everything after training stays local: the NumPy export check, development scoring, the vetoes, the ensemble and the confirmation.
- **Inputs:** two private Kaggle datasets. One holds only the two prepared training-sequence folders (manifests `d3891c24…` and `2828b0c3…`), derived from DFL (CC BY 4.0) and SkillCorner (MIT) alone. The other holds the repository code at a committed revision. Never the competition bundle or anything derived from it, the public ground truth, PFF, Metrica, or any cache.
- **Reproducibility:** CUDA training is not bit-reproducible, so the published weights are the reference, as for the existing networks.
- **Record:** each notebook run writes the code commit and a hash of the code tree, the Python, PyTorch and CUDA versions and the GPU, the hash of every input file, the seeds, and the SHA-256 of every output file. Before use, the repo checks all of these:
  - the input hashes against the local manifests;
  - every output hash;
  - that the code tree is the committed revision, and that the training code in it is the code here;
  - that each run's setting, seed, boundary weight and training manifest are the ones D20 prescribes, and that the selection follows from the cross-validation record.

  It then exports and checks each network locally, at D15's tolerance. A family whose outputs fail a check is left out. A notebook that stops before writing its outputs may be run again.
- **Fallback:** if a family's Kaggle outputs have not passed these checks by 2026-11-13 08:00 EST, that family runs locally instead, as first written (GRU on the CPU, transformer on MPS). Kaggle outputs that arrive later are not used. The 2026-11-14 12:00 EST cutoff is unchanged.

*Cloud run note, 2026-11-11 (a record of how the runs were made; no rule changes).* At Derek's request, the coordinating Claude session made the uploads and started the runs through the Kaggle CLI, not Derek through the web interface.
- **Bundle:** `cloud/make_bundles.sh` at commit `c3cf4b3`. SHA-256: `mtah-code.zip` `ba99275c…`, `train_data.zip` `1621b97b…`, `sequences_augmented.zip` `fe62038e…`. The coordinating session checked both bundles before upload: they hold code, models and records only, with no CSV, ground truth or cache.
- **Datasets:** the private datasets `mtah-d20-code` and `mtah-d20-sequences`. Kaggle unzipped the archives into folders, a layout the notebooks accept.
- **Runs:** the committed notebooks, unchanged, as private kernels `mtah-d20-gru` and `mtah-d20-transformer`, version 1. Each ran on a single NVIDIA T4 rather than the "T4 x2" that `cloud/README.md` suggests, which is enough because a notebook uses one CUDA GPU. GPU and Internet were enabled. Kaggle records the starts as 22:19:31 and 22:19:33 UTC (17:19 EST).

The outputs still go through every check in the cloud amendment (`zoo import-cloud`) before anything is used.

*Amendment, 2026-11-11, before any tree-family run.* Derek decided, through the coordinating session, that the tree family's cross-validation runs in a Kaggle CPU session. Its local job would have queued behind the temporal-network cross-validation, which already loads this 8-core machine heavily.
- **Unchanged:** the 3 × 3 grid, both heads, the folds, the loss, its weighting and the selection.
- **The cloud step:** the four-fold fits and their held-out losses only. The notebook installs the build requirements (`dfl_submission/requirements.txt`, no PyTorch) and reads the same private sequences dataset.
- **The final trees stay local.** The selected setting's four trees (both heads of both sets) are fit on this machine, so they refit byte for byte here, as D16's check showed for the published trees. XGBoost on Kaggle's x86 CPUs need not match this machine's arm64 bit for bit, and cross-validated losses are used only to select.
- **Checks:** the record and checks are those of the cloud amendment, applied to the tree family's own training code, so the GRU and transformer archives are still checked against theirs.
- **Fallback:** as in the cloud amendment. If the tree record has not passed its checks by 2026-11-13 08:00 EST, the tree cross-validation runs locally.

*Cloud run note, 2026-11-11 (tree family; no rule changes).* The coordinating session committed the tree changes as `f6010c6` and rebuilt the code bundle there (`mtah-code.zip` `d846082c…`), pushing it as a new version of the private code dataset. The sequences dataset was not uploaded again; the rebuilt zips hash as before. The committed `cloud/d20_trees.ipynb`, unchanged, runs as private kernel `mtah-d20-trees` version 1, with no accelerator and Internet on. Kaggle records the start as 22:54:34 UTC (17:54 EST). The GRU and transformer runs keep their `c3cf4b3` code, and each import checks only its own family's training code against the commit that family ran.

**Outcome (2026-11-11 22:10 EST).** No ensemble beat D15, so **D15 stays the submission**. The greedy selection kept D15 alone, so the ownership-cost grid, the Metrica veto and the confirmation did not run. The DFL/SkillCorner holdout was not scored, and PFF confirmation set C was neither parsed nor scored. The untouched PFF set stays in reserve.
- **Pipeline checks all passed.**
  - *D15:* decoded from its cached development inputs, it reproduced its development record exactly, PFF development included. D15 run through the D20 forecast path gave bit-identical decoder inputs on every development half.
  - *Cloud runs:* each of the three passed every import check: inputs, outputs, code tree, training code, cross-validation record and runs.
  - *Exports:* the GRU and transformer NumPy exports matched PyTorch locally, with largest logit differences of 5.7e-6 and 1.9e-5.
- **Cross-validation on the training matches** (frame log-loss, ownership plus stoppage; lower is better):

| Family | Settings (score) | Selected |
|---|---|---|
| Temporal network | width 96: 5,400 steps 0.29535, 10,800 0.30699, 19,800 0.34715; width 192: 0.29926, 0.35998, 0.50319 | width 96, 5,400 steps: D15's own setting, so its member is D15 |
| GRU | h 64: 5,400 0.29398, 10,800 0.34509; h 128: 0.31492, 0.41539 | h 64, 5,400 steps |
| Transformer | d 64: 5,400 0.33413, 10,800 0.33995; d 128: 0.33252, 0.36723 | d 128, 5,400 steps |
| Trees, ownership head | depth 3 / 5 / 7 at 420 trees: 0.26931, 0.25151 (D3), 0.24318; best 0.23913 | depth 7, 1,260 trees (the grid's edge) |
| Trees, stoppage head | depth 3 / 5 / 7 at 420 trees: 0.08936, 0.07871 (D3), 0.07517; best 0.07472 | depth 7, 840 trees |

- **Development, each member once, against D15** (N = 3 standard 0.72280):

| Member | N = 3 standard | Change | N = 1.75 / 4 / 4.5 | Dead ball carried | Team swap | PFF |
|---|---:|---:|---|---:|---:|---:|
| GRU | 0.71949 | −0.0033 | −0.000 / −0.012 / −0.015 | −0.0085 | −0.0134 | −0.0125 |
| Transformer | 0.69212 | −0.0307 | −0.040 / −0.020 / −0.015 | −0.0510 | −0.0567 | −0.0231 |
| Trees | 0.71982 | −0.0030 | −0.004 / −0.003 / −0.003 | +0.0057 | −0.0053 | +0.0009 |

- **Greedy selection.**
  - *Step 1:* D15 was the best single member (0.72280).
  - *Step 2:* the best pair was D15 + GRU at 0.72233, then D15 + trees at 0.72088 and D15 + transformer at 0.71209. None raised the primary, so the selection stopped at D15 alone.
- **Disclosures.**
  - The coordinating session ran the cloud steps through the Kaggle CLI at Derek's request (run notes above).
  - The first GRU import passed its checks and exported its networks, then stopped while copying the cross-validation record, because `zoo_recipe/` did not yet exist. It wrote no record. A one-line fix (`import_cloud` creates the folder) was made and the import run again. The cloud run was not repeated, and no output was looked at between the two attempts beyond the import's own log.
  - The tree cross-validation fits used 6 threads (`CV_TREE_JOBS`). The four final trees were fit on this machine with D3's settings otherwise.

What it shows:
- *Training length:* frame log-loss on held-out training matches favoured the shortest training in every network family, and for the temporal network it rose steadily with length. That is the opposite of D15's development result, where tripling the length raised spellF1 by +0.028. Longer training makes the networks more confident: log-loss penalises that, while the decoder gains from the sharper ownership changes. So the plan's tuning criterion and the ranking metric disagree on exactly the lever D15 found. Anyone who returns to training length should select on decoded spellF1 from held-out training matches, under a new rule.
- *Other families:* the recurrent and attention models add nothing over D15's convolutional networks on these inputs.
- *Tree heads:* the deeper heads lower the trees' own log-loss and help with the dead ball carried, but cost a little on the standard views.

Records: [cross-validation](dfl_submission/zoo_recipe/) (`cv_*.json`, `cloud_*.json`) · [members](dfl_submission/zoo_recipe/) (`member_*.json`) · [selection](dfl_submission/zoo_recipe/development.json). Weights are kept in `dfl_submission/zoo_models/` for reference.

## D21 — Where D15 loses spellF1 (2026-11-11, written before anything is scored)

**Question.** D11 found that D3 lost its score in ownership changes rather than stoppages, and that it under-segmented. D15 then gained +0.04 by training longer and wider. Is ownership still the main loss for D15, or are stoppages now? D21 is a measurement only, round 5's step 1. It selects nothing and changes nothing in the submission. If steps 2 and 3 of the fifth plan (D22, D23) fall short, it says what to try next.

**What was seen before this rule.** The records of D1–D20. Nothing has been decoded or scored for D21.

**Candidate.** Frozen D15, the submission: D3's trees and 0.5 blend with D15's networks, cutoff 0.53, ownership cost 8, stoppage cost 2, discount 0.8. Decoder inputs come from the caches its own records were scored from, decoded afresh at D15's recipe:
- development: DFL WPY in its four views, the four SkillCorner development matches and the eight PFF development matches (`data/d15/cache/w96_s5400`);
- the three Metrica sample matches (`data/metrica_cache/d15`), a seventh group, `metrica`.

Holdout, confirmation, training and untouched matches are not read. For the Metrica group only, frozen D3 is measured too from its cached inputs (`data/metrica_cache`), so the new group has a reference; for the other six groups D3's numbers are D11's record.

**Pipeline check, before any result is read.** D15 decoded from its development cache must reproduce exactly its development record (the selected trial in `scale_recipe/development.json`, PFF development included). D15 and D3 decoded from their Metrica caches must reproduce `metrica_recipe/d15_measurement.json` and `d3_measurement.json` exactly. If any check fails, D21 stops without writing a record, and only the pipeline may be fixed.

**Targets.** As in D11: the raw provider labels and the N = 3 s controlled-possession targets. For Metrica the raw target is D19's raw event-possession target.

**What is measured.** D11's four measurements, unchanged, per group and target:
1. spell detection at each IoU from 0.3 to 0.7 (precision, recall, F1, true and predicted spell counts);
2. the two oracle runs, true stoppages and true ownership, defined exactly as in D11, as mean spellF1 per group and as the standard combined aggregate;
3. signed boundary offsets by transition type (ownership, stop, restart), matched at 25 frames, per group and per vendor (DFL clean + masked, SkillCorner, PFF, Metrica);
4. recall by true spell length, brief interior spells and stoppage lengths.

D11's boundary-shift formula is not applied: D12 is closed, and nothing is handed on.

**Record.** `dfl_submission/diagnosis_recipe/d21.json`, written once by `python -m dfl_submission.diagnosis measure-d15`. Every D21 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** The pipeline checks passed: from their caches, D15 reproduced its development record (PFF development included), and D15 and D3 each reproduced their Metrica measurement exactly. Against the N = 3 targets:

| Group | D15 spellF1 | True stoppages | True ownership | True / predicted spells | Precision / recall at IoU 0.5 | D3 spellF1 |
|---|---:|---:|---:|---|---|---:|
| DFL clean | 0.778 | 0.778 | 0.933 | 281 / 263 | 0.82 / 0.77 | 0.742 |
| DFL masked | 0.723 | 0.731 | 0.926 | 281 / 231 | 0.81 / 0.67 | 0.662 |
| DFL dead ball carried | 0.609 | 0.708 | 0.792 | 281 / 287 | 0.61 / 0.63 | 0.566 |
| DFL team swap | 0.659 | 0.672 | 0.920 | 281 / 245 | 0.72 / 0.63 | 0.579 |
| SkillCorner | 0.704 | 0.753 | 0.841 | 971 / 792 | 0.79 / 0.65 | 0.676 |
| PFF | 0.656 | 0.735 | 0.790 | 1783 / 1394 | 0.76 / 0.60 | 0.626 |
| Metrica | 0.780 | 0.823 | 0.852 | 825 / 758 | 0.83 / 0.76 | 0.760 |
| Standard combined | 0.7228 | 0.7496 | 0.8849 | | | 0.6821 |

D3 on Metrica, for reference: true stoppages 0.814, true ownership 0.846, 674 predicted spells, precision / recall 0.86 / 0.70.

- **Ownership is still where most of the score is lost.** On the standard views true ownership adds +0.162 and true stoppages +0.027 (D3: +0.205 and +0.032). D15 closed about a fifth of the ownership gap. Stoppages are a larger share only where the dead ball is hard to see: DFL dead ball carried (+0.100), PFF (+0.079) and, less so, Metrica (+0.043).
- **D15 still under-segments, less than D3.** It predicts 6–22% fewer spells than the targets on every group except the dead ball carried view (D3: 20–25% fewer). Recall at IoU 0.5 by true spell length on SkillCorner: 4% under 1.75 s, 46% at 1.75–3 s, 52% at 3–5 s, 74% at 5–10 s, 89–93% above 10 s (D3: 4%, 40%, 48%, 72%, 84–92%). On DFL clean the 1.75–3 s bin rose from 40% to 87%; on PFF it is still 11%.
- **Brief interior spells are still rare.** D15 predicts at most 4 interior spells under 1.75 s in any group, and 3–28 under 3 s.
- **Boundary timing has no shared bias.** N = 3 median offsets (frames, predicted minus true) for ownership, stop and restart: DFL 0, −1, −0.5; SkillCorner 0, 0, 0; PFF −2, −5, −3; Metrica +4, +3, −1.
- **Sub-second stoppages on PFF.** D15 predicts 87 stoppages under 1 s on the PFF development matches, against 1 in the targets (SkillCorner 10 against 30, Metrica 7 against 0, DFL clean none). D21 did not examine their cause.

What follows: the headroom is still in ownership changes the models miss, mostly spells under 5 s. D22 and D23 act on the models, which is where that headroom is. If both fall short, the next candidates are more ownership-change recall in the models and, for PFF-like tracking only, a minimum stoppage length. Neither is part of this plan. [Record](dfl_submission/diagnosis_recipe/d21.json)

## D22 — Finish D15's search and average three seeds (2026-11-11, written before any D22 decoding, training or scoring)

**Question.** D15 stopped at the edge of its grid: ownership cost 8 was its largest cost, and nothing longer than 5,400 steps was scored on spellF1. D20 trained longer and wider networks but chose among them by frame log-loss, and its own record says log-loss and spellF1 disagree on training length. Separately, each of D15's four network slots holds one seed. Do a longer or wider setting, a higher ownership cost, or the mean of three seeds per slot raise spellF1? This is round 5's step 2.

**What was seen before this rule.** The records of D1–D21. D20's cross-validation record (`zoo_recipe/cv_tcn.json`) holds only frame log-losses; no fold network has been decoded or scored on spellF1.

**Reference.** D15, the submission (D3's trees and 0.5 blend, D15's four networks at width 96 and 5,400 steps, cutoff 0.53, ownership cost 8, stoppage cost 2, discount 0.8).

**Stage A — setting and cost, on the training matches only, with no new training.**
- *Networks.* The 24 fold networks of D20's temporal-network cross-validation, in `data/d20/cv/tcn`: for each of the six settings (width 96 or 192 × 5,400, 10,800 or 19,800 steps), fold k's network (original set, seed 42, boundary weight 0.5, last snapshot) trained on the other three folds. Each run's `fit.json` must match the hash in `cv_tcn.json`.
- *Decoder inputs.* For each held-out training half, the fold network's swap-averaged probabilities at the prepared 5 Hz frames, computed from the prepared sequence exactly as D20 computed its log-loss (`zoo.network_probabilities`), then interpolated linearly to every 25 Hz frame as the build's network inference does. These four signals (home, dead, dead edge, owner edge) go straight into the joint decoder: no trees, no second model set, no phase averaging.
- *Views scored.* DFL clean and masked, and SkillCorner. The DFL stress view is a training augmentation and is not scored.
- *Targets.* Each half's 25 Hz provider labels (`aligned_trial.training_labels`), with the N = 3 s collapse for the primary.
- *Grid (18 points).* The six settings × ownership cost ∈ {8, 10, 12}, with D15's cutoff, stoppage cost and discount.
- *Primary.* The standard combined N = 3 spellF1 over the 16 training matches, each scored once, by the fold network that did not train on it: per DFL match 1/3 clean + 2/3 masked, averaged over the four DFL matches; the mean over the 12 SkillCorner matches; then the mean of the two.
- *Pipeline checks, before any spellF1 is computed.* (1) For every fold network, the held-out log-losses recomputed from these probabilities must equal the per-match values in `cv_tcn.json`. (2) For every held-out half, the stored targets of its prepared sequence must be reproduced exactly from the reloaded 25 Hz labels at the prepared frames. If either fails, Stage A stops and only the pipeline may be fixed.
- *Selection.* The anchor is (width 96, 5,400 steps, cost 8), D15's own setting. The point with the highest primary is taken, ties to N = 3 macroF1, then grid order. It replaces the anchor only if its primary is at least +0.005 above the anchor's; otherwise the standing setting stays (96, 5,400, cost 8). Other conventions are reported, not gated.

**Stage B — three seeds per slot, at the Stage A setting.**
- Each of the four slots (in each model set, seed 42 with boundary weight 0.5 and seed 97 with 1.4) becomes the mean of three networks: its own seed s and two new seeds, s + 1000 and s + 2000 (1042 and 2042, 1097 and 2097), the same in both sets.
- If Stage A keeps (96, 5,400), the slot's own network is D15's published one and eight new networks are trained. If Stage A selects another width or length, all twelve networks are trained at it.
- Training is D15's: `temporal_network train --width --steps` on the same prepared sequences (manifests `d3891c24…` and `2828b0c3…`), on Apple MPS, last snapshot only, then the NumPy export with its own check (2e-5 absolute plus 1e-5 relative on the logits).
- *Averaging.* A slot's four 25 Hz outputs are the mean of its networks' outputs, summed in seed order in float32 and divided by the count; a slot of one network passes through unchanged. Everything downstream (variants, phase averaging, the trees, the 0.5 blend) is D15's.

**Development, once.** The three-seed candidate at the Stage A setting and cost, on DFL WPY in its four views, the four SkillCorner development matches and the eight PFF development matches. D15 decoded from its cached development inputs must first reproduce its development record exactly. The candidate is eligible if, compared with D15:
- the primary (standard combined N = 3 spellF1 on DFL WPY and the four SkillCorner development matches) rises by at least +0.005;
- standard combined spellF1 at N = 1.75, 4 and 4.5 s each drops by no more than 0.01;
- DFL dead ball carried and team swap, at N = 3, each drop by no more than 0.01;
- *PFF veto:* mean N = 3 spellF1 on the eight PFF development matches drops by no more than 0.01;
- *Metrica veto (D19, reference D15):* mean N = 3 spellF1 on the three Metrica sample matches is at least 0.77004.

There is one candidate, so nothing is selected at development.

**Confirmation, once,** if development passes: the reused DFL/SkillCorner holdout (WQQ, WR9 and their guard views; the four SkillCorner holdout matches) and PFF confirmation set C, parsed only for this step, with the same gates against D15 (primary at least +0.005, the conventions, the guards, and the set C veto). D15 must first reproduce its holdout record (`scale_recipe/confirmation.json`); it is then scored on set C as the reference. The Metrica veto is not repeated: it is the same three games. The holdout has confirmed D1, D3 and D15 before, so it is reused evidence; set C is fresh. The untouched set stays in reserve for D23.

**If it passes,** the three-seed model and its cost become the submission after the rebuild checks (twice on macOS in a fresh virtual environment, then `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`), and the standing recipe for D23. **If it fails,** D15 stays. PFF and Metrica only veto; no value is chosen on them.

Code: the fold decode path in `dfl_submission/zoo.py`; slot averaging in `retrain_trial` and `zoo`; `dfl_submission/triple_trial.py` (`stage-a`, `networks`, `develop`, `confirm`, `build`). Weights go to `dfl_submission/triple_models/`, records to `dfl_submission/triple_recipe/`, each written once. Every D22 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-11).**

*Stage A.* Every pipeline check passed: all 24 fold runs matched their recorded hashes, reproduced D20's held-out log-losses exactly, and every held-out half's prepared targets were reproduced from the reloaded labels. Standard combined N = 3 spellF1 over the 16 training matches, each scored by the fold network that did not train on it (each network decoded alone):

| Width | Steps | Cost 8 | Cost 10 | Cost 12 |
|---:|---:|---:|---:|---:|
| 96 | 5,400 | **0.71293** (anchor) | 0.71409 | 0.71363 |
| 96 | 10,800 | 0.70924 | 0.71261 | 0.71509 |
| 96 | 19,800 | 0.70136 | 0.70645 | 0.70786 |
| 192 | 5,400 | 0.70406 | 0.70700 | 0.71032 |
| 192 | 10,800 | 0.69796 | 0.70026 | 0.70282 |
| 192 | 19,800 | 0.68692 | 0.69158 | 0.69356 |

The best point, width 96 with 10,800 steps at cost 12, beat the anchor by +0.0022, under the +0.005 the rule required, so D15's setting and cost 8 stood for Stage B. Longer or wider networks scored higher at N = 1.75 and on raw labels and lower at N = 4 and 4.5. That is, they split possessions more finely, and a higher ownership cost recovered most but not all of the loss. D15's development gain from longer training (1,800 to 5,400 steps) does not extend past 5,400 steps on these matches. [Record](dfl_submission/triple_recipe/stage_a.json)

*Stage B.* The eight new networks (seeds 1042, 2042, 1097 and 2097 in each set) trained at width 96 and 5,400 steps on the D15 sequences, and each passed its NumPy export check.

*Development, once.* D15 reproduced its development record exactly first. The three-seed candidate, at cost 8, failed the primary gate, so **D15 stays**. Confirmation did not run: the holdout was not scored, and PFF confirmation set C was neither parsed nor scored.

| Gate | Change | Required |
|---|---:|---|
| N = 3 standard combined (0.72280 → 0.71692) | −0.0059 | ≥ +0.005 |
| N = 1.75 / 4 / 4.5 | −0.008 / −0.003 / −0.005 | each ≥ −0.01 |
| Dead ball carried | +0.0133 | ≥ −0.01 |
| Team swap | +0.0107 | ≥ −0.01 |
| PFF development (veto) | −0.0042 | ≥ −0.01 |
| Metrica (veto; line 0.77004) | 0.78220 (+0.0022) | ≥ 0.77004 |

What it shows: the mean of three seeds per slot lost on every convention of the standard views, while the guard views and the Metrica games gained. One likely reading, not measured here, is that the mean of three networks is less decisive at ownership changes than one network, and the decoder at cost 8 rewards decisive changes on the standard views. Stage A and D20 point the same way: training length and seeds move the score between conventions and views rather than raising it. The eight networks are kept in `dfl_submission/triple_models/` for reference. [Stage A](dfl_submission/triple_recipe/stage_a.json) · [Development](dfl_submission/triple_recipe/development.json)

## D23 — Final fit on all 27 labelled matches (2026-11-11, written before any D23 decoding, fitting or scoring)

**Question.** The models train on 16 of the 27 labelled external matches. The other 11 (DFL WPY, WQQ and WR9; eight SkillCorner matches) are held back for development and holdout. Once selection is over, refitting the chosen recipe on every labelled match is the standard last step. Does it help? This is round 5's step 3 and the last modelling step: after it, the development and holdout matches are training data, so nothing more can be selected on them.

**What was seen before this rule.** The records of D1–D21 and D22's Stage A record, which kept D15's setting and ownership cost (the best point gained +0.0022, under +0.005). D22's extra seeds were training when this was written; no D22 development or confirmation score had been read.

**Standing recipe.** D22's three-seed model if it is confirmed, otherwise D15. Both use D15's setting (width 96, 5,400 steps) and decoder recipe (cutoff 0.53, ownership cost 8, stoppage cost 2, discount 0.8). The evidence step below is therefore the same whichever stands.

**Evidence first, with no new training.** On development (DFL WPY clean and masked, the four SkillCorner development matches), each of these networks is decoded alone, its four outputs going straight into the joint decoder at the standing recipe, as in D22's Stage A:
- D20's four fold networks at (96, 5,400): original set, seed 42, boundary weight 0.5, each trained on 12 training matches;
- the standing recipe's matching network: the original set's seed-42 network of D15, trained on all 16.

The primary is standard combined N = 3 spellF1. D23 proceeds only if the 16-match network's primary is at least +0.005 above the mean of the four fold networks' primaries, which would show that more matches still help. Otherwise D23 stops: nothing is refit, and the untouched PFF set and the four 2019–20 matches stay unscored.

**Candidate.** The standing recipe, unchanged, refit on all 7 DFL matches (WMX, WN1, WOH, WOY, WPY, WQQ, WR9) and all 20 SkillCorner matches:
- *Prepared sequences:* `temporal_network prepare`'s loop over an `all` split, for both sets. Every match gets the training views: DFL clean, masked and stress, and SkillCorner as observed, for the original set; the D2 handling for the augmented set. Every match keeps the seeds its formulas give, so the training matches' halves are unchanged.
- *Trees:* both heads of both sets refit from the prepared rows (`touch_trial.fit_trees`, which D16 showed refits the published trees byte for byte), with D3's parameters. The existing weights balance sources, matches and classes.
- *Networks:* every network in the standing recipe's slots is retrained on the new sequences with its own seed and boundary weight, at width 96 and 5,400 steps, on Apple MPS, last snapshot only, then exported with the NumPy check.
- Everything else (variants, phase averaging, the 0.5 blend, the decoder recipe) is the standing recipe's.

**Pipeline check.** Restricted to the 16 training matches, the new prepared folders must reproduce the stored training sequences exactly, array by array (values and targets), for the original set (manifest `d3891c24…`) and the augmented set (`2828b0c3…`). If not, D23 stops and only the pipeline may be fixed.

**Adopt if,** on three sources that neither model was trained on, comparing the candidate with the standing model (both scored now, once each), no change is below −0.01 and the mean of the three changes is not negative:
- *the PFF untouched set:* mean N = 3 spellF1 over its eight games, parsed only for this step;
- *the three Metrica sample matches:* mean N = 3 spellF1 against D19's targets. A standing D15 must first reproduce its D19 measurement; a standing D22 must reproduce the Metrica means in its development record;
- *the four never-scored SkillCorner 2019–20 matches* (3442, 3518, 3749, 4039): ownership only, scored as in D6 (mean N = 3 spellF1 per match over both halves; frames without a provider team are not scored).

**Disclosure.** The technical document reports in-vendor scores (development and holdout) for the 16-match version, and the three independent checks for the submitted one. If D23 is adopted, the submitted models have trained on every development and holdout match, so no in-vendor score of them is reported.

**If adopted,** the refit model becomes the submission after the rebuild checks (twice on macOS, then `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`). **If not,** the standing recipe stays. This step spends the last reserved PFF set; Derek approved the plan that spends it. No value is chosen on PFF, Metrica or the 2019–20 matches.

Code: an `all` split in `external_benchmark.halves` and `augment.training_halves`; `dfl_submission/final_trial.py` (`evidence`, `prepare`, `fit`, `check`, `build`). `temporal_network.py` stays byte-identical to the code D20's cloud runs used, so `final_trial` restates its prepare loop. Weights go to `dfl_submission/final_models/`, records to `dfl_submission/final_recipe/`, each written once. Every D23 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-11).** D23 stopped at its evidence step, so nothing was refit and the standing recipe stays. The untouched PFF set and the four 2019–20 matches were neither parsed nor scored, so the untouched set is still in reserve. Each network decoded alone at the standing recipe on development (standard combined spellF1):

| Network | Trained on | N = 3 | N = 1.75 | N = 4 | N = 4.5 | Raw |
|---|---|---:|---:|---:|---:|---:|
| D20 fold 0 | 12 matches | 0.70795 | 0.6890 | 0.7042 | 0.6966 | 0.6115 |
| D20 fold 1 | 12 matches | 0.71106 | 0.6894 | 0.7012 | 0.6956 | 0.6161 |
| D20 fold 2 | 12 matches | 0.71280 | 0.6942 | 0.7135 | 0.7024 | 0.6133 |
| D20 fold 3 | 12 matches | 0.72199 | 0.7054 | 0.7121 | 0.7020 | 0.6255 |
| Fold mean | | 0.71345 | | | | |
| D15's network | 16 matches | 0.71139 | 0.6960 | 0.7010 | 0.6935 | 0.6191 |

The 16-match network scored 0.0021 below the fold mean, against a required +0.005. What it shows: a single network's score moves by up to 0.014 with its training matches (0.708 to 0.722). Against that spread, a third more training matches showed no gain, so the evidence does not show that more matches still help, and the refit was not run. [Record](dfl_submission/final_recipe/evidence.json)

## D24 — Metrica sample matches under competition-like tracking (2026-11-11, written before any D24 view is built or scored)

**Question.** D19 scores the three Metrica sample matches on corrected tracking, where every player is present and the ball is missing exactly when the ball is dead. The competition's automatic tracking has neither property: the ball is missing on 41% of graded frames (DATA.md), and it can be seen while dead. How does D15 score when the same games are degraded towards that? D24 is a measurement, round 6's step 1. It selects nothing and changes nothing in the submission. Its views become a gate for D25 and D26.

**What was seen before this rule.** The records of D1–D23. No degraded view of these games has been built, forecast or scored.

**Views.** Built from the parsed halves that passed D19's reader gate, per game g ∈ {1, 2, 3} and half i ∈ {0, 1}. Every constant is an existing one; nothing is tuned.
- *masked:* `dfl_transfer_features.corrupt` with seed 640000 + 10g + i, the degradation of the DFL masked view, unchanged: 0.2 m noise on players and ball, each player track dropped in 50-frame blocks with probability 0.2, and the ball hidden by a two-state switch (mean hidden run 50 frames, mean visible run 75 frames, so about 40% hidden), on top of the dead-ball absence.
- *handled:* D2's dead-ball handling first (`augment.handle_dead_ball` on D19's raw dead runs, with `numpy.random.default_rng(650000 + 10g + i)`), then the same `corrupt` call as the masked view with the same seed. Each dead run keeps its missing ball (20%), has it removed (20%, also missing here), or gets a ball carried from where play stopped to the restart taker and resting at the restart spot (60%). The masking is drawn identically to the masked view, so the two views differ only in the dead ball.
- *team swap:* the handled view with D2's guard swaps (`augment.swap_teams`, exactly 10 s each, one per 10 s on average) drawn with `default_rng(660000 + 10g + i)`.

**View checks, before any forecast.** Pooled per game over both halves: the masked view hides the ball on 30–50% of D19's live frames, and the handled view shows the ball on more dead frames than the masked view. If a check fails, only the view code may be fixed.

**Candidate.** Frozen D15, the submission, built as D19's amendment measured it (`metrica_sample.reference_model("D15")`, decoder inputs from `diagnosis.half_signals`, D15's recipe). Before any view is scored, D15 and D3 decoded from their D19 caches must reproduce `d15_measurement.json` and `d3_measurement.json` exactly; otherwise D24 stops without a record. The decoder inputs on the views are cached in `data/metrica_cache/d24/` (gitignored).

**Targets.** D19's targets, unchanged: raw event possession, N = 1.75, 3, 4 and 4.5 s, and the control rule. The tracking is degraded; the truth is not.

**Measured, per view (and the D19 clean view from its cache, for comparison).**
- spellF1 and macroF1 against every target, per game and as the mean over the three games;
- sub-second stoppages: predicted and true stoppages shorter than 25 frames, all and interior (a team on both sides);
- signed boundary offsets by transition type (ownership, stop, restart) against the raw and N = 3 targets, matched at 25 frames as in D11, pooled and per game;
- the organisers' smooth50 baseline (`vendor_trial.baseline_prediction`), raw and N = 3 spellF1, reported only.

**The gate quantity for later steps.** A candidate's *D24 mean* is the mean over the three views of its mean N = 3 spellF1 over the three games. D25 and D26 compare it with D15's.

**Known biases.** The handled view uses D2's training simulation, which the augmented model set was trained on, so it may flatter D15 there (D2's guard views were built differently for that reason; that construction needs the true ball position, which Metrica does not record while the ball is dead). The corrected player tracks are degraded synthetically, not by a real tracker. Three matches.

**Record.** `dfl_submission/metrica_recipe/d24_measurement.json`, written once by `python -m dfl_submission.metrica_views measure`. Every D24 command runs with `MTAH_DATA_DIR` set to a nonexistent path.

**Outcome (2026-11-11).** The view checks passed: the masked view hid the ball on 38–40% of live frames per game, and the handled view showed it on 34–40% of dead frames (masked: 0%). D15 and D3 reproduced their D19 records from their caches before any view was scored.

| View | Raw | N = 1.75 | **N = 3** | N = 4 | N = 4.5 | Control | N = 3 macroF1 | Smooth50 N = 3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean (D19) | 0.6929 | 0.7747 | **0.7800** | 0.7618 | 0.7513 | 0.8366 | 0.930 | 0.427 |
| Masked | 0.5940 | 0.6757 | **0.7047** | 0.7099 | 0.7041 | 0.7656 | 0.899 | 0.187 |
| Handled | 0.5973 | 0.6773 | **0.7059** | 0.7124 | 0.7071 | 0.7589 | 0.892 | 0.159 |
| Team swap | 0.5129 | 0.5781 | **0.6105** | 0.6150 | 0.6098 | 0.6517 | 0.859 | 0.140 |

- **D24 mean** (the gate quantity for D25 and D26): **0.67367**, against 0.78004 on the clean view. N = 3 per game, masked / handled / team swap: game 1 0.682 / 0.676 / 0.586, game 2 0.706 / 0.708 / 0.589, game 3 0.726 / 0.734 / 0.656.
- **Masking costs about 0.075, the dead ball almost nothing.** The handled view, where the missing ball no longer marks the stoppages, scored the same as the masked view (+0.001), so D15 does not depend on the missing-ball shortcut here. Team swaps cost a further 0.095, more than on DFL (D21: masked 0.723 to swap 0.659).
- **Sub-second stoppages,** predicted (interior) against true: clean 7 (7) against 0, masked 12 (9), handled 16 (13), team swap 16 (13), all against 0. Far fewer than PFF's 87 on eight matches.
- **Ownership changes still land late.** N = 3 median offsets for ownership, stop and restart (frames, predicted minus true): clean +4, +3, −1; masked +4, +2, −2; handled +4, +0.5, −1; team swap +4, +1, 0. Per game, the ownership median was +3, +6 and +3 on every degraded view (clean +3, +4, +3).
- **Smooth50** scored 0.14–0.19 at N = 3 on the views, against 0.43 clean: it loses most where the ball is hidden and the dead ball is visible.

Nothing was selected. [Record](dfl_submission/metrica_recipe/d24_measurement.json)

## D25 — No stoppage shorter than one second (2026-11-11, written before any D25 output is scored, and before D24's results were read)

**Question.** D21 found that D15 predicts 87 stoppages under 1 s on the eight PFF development matches, where the targets have 1; 7 on Metrica (targets 0); 5 and 3 on the DFL dead-ball-carried and team-swap views (targets 0); none on clean DFL. Each one cuts a possession in two. Does removing them help, or at least do no harm? This is round 6's step 2.

**What was seen before this rule.** The records of D1–D23 and the D24 rule. D24 was running when this rule was written; none of its scores had been read.

**Candidate.** D15, unchanged, followed by one post-processing step on each decoded half (`conventions.remove_brief_stoppages`):
- every predicted stoppage shorter than 25 frames with a team on both sides is removed;
- if the same team is on both sides, that team takes its frames; otherwise the earlier team takes the first ⌊n/2⌋ of its n frames and the later team the rest;
- a stoppage at a half's edge (next to the start or end of the half, or to an X frame) and every stoppage of 25 frames or more is untouched.

One hand-set constant, 25 frames (1 s), with no grid. Why 1 s: a restart cannot be taken faster, and the targets agree (0 stoppages under 1 s on DFL and Metrica, 1 of 748 on PFF). Metrica has 29 true stoppages between 1 s and 3 s, so the limit must not go higher. SkillCorner's targets have 30 stoppages under 1 s (detector-based dead ball); the gates below catch a loss there.

**Pipeline checks, before any candidate score is read.** D15 decoded from its cached inputs must reproduce its development record (PFF development included), its D19 measurement and its D24 measurement exactly. Otherwise D25 stops without a record.

**Adopt if nothing gets worse.** This is a constraint, not a tuned value, so the rule asks for no harm rather than a gain. On development (DFL WPY in its four views, the four SkillCorner development matches, the eight PFF development matches, the three Metrica games), compared with D15:
- standard combined N = 3 spellF1 drops by at most 0.002; at N = 1.75, 4 and 4.5 each drops by at most 0.005;
- the mean of the dead-ball-carried and team-swap changes (DFL, N = 3) is not negative;
- mean N = 3 spellF1 on the PFF development matches, on the Metrica games (D19) and the D24 mean are each not lower.

Reported, not gated: raw-label scores, and sub-second stoppage counts before and after, per group.

**Confirmation, once,** if development passes: the reused DFL/SkillCorner holdout (WQQ, WR9 and their guard views; the four SkillCorner holdout matches) with the same conditions, plus mean N = 3 spellF1 on PFF confirmation set C not lower. Set C is parsed only for this step, and D15 is forecast on it once. D15 must first reproduce its holdout record (`scale_recipe/confirmation.json`). The Metrica and D24 conditions are not repeated: they are the same games. PFF set A, which D15's holdout cache holds, is reported, not gated. The holdout has confirmed D1, D3 and D15 before, so it is reused evidence; set C is fresh.

**If it passes,** D15 with this step becomes the submission and the standing reference for D26, after the rebuild checks (twice on macOS in a fresh virtual environment, then `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`). **If it fails,** D15 stays. PFF and Metrica only veto; no value is chosen on them.

Code: `conventions.remove_brief_stoppages`; `dfl_submission/stoppage_trial.py` (`develop`, `confirm`, `build`) on D15's cached decoder inputs. Records go to `dfl_submission/stoppage_recipe/`, each written once. Every D25 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-11).** Failed at development, so **D15 stays**. Confirmation did not run: the holdout was not scored, and PFF confirmation set C was neither parsed nor scored. The pipeline checks passed first: D15 reproduced its development record (PFF development included), its D19 and its D24 measurements, and its recipe reproduced every cached Metrica label.

| Condition, against D15 | Change | Required |
|---|---:|---|
| Standard combined N = 3 (0.72280 → 0.72269) | −0.00011 | ≥ −0.002 |
| N = 1.75 / 4 / 4.5 | −0.0001 / −0.0001 / −0.0001 | each ≥ −0.005 |
| Mean of dead ball carried and team swap | 0 (both unchanged) | ≥ 0 |
| PFF development | +0.00066 | ≥ 0 |
| Metrica (D19) | 0 (unchanged) | ≥ 0 |
| D24 mean (0.67367 → 0.67359) | **−0.00008** | ≥ 0 |

- The D24 loss came from one game in one view: handled, game 3, −0.0007. Every other view and game was unchanged.
- **Every interior sub-second stoppage was removed:** PFF 87 → 0, SkillCorner 10 → 0 (its targets hold 27), DFL dead ball carried 5 → 0, team swap 3 → 0, masked 1 → 0, Metrica 7 → 0, D24 views 9, 13 and 13 → 0. Three sub-second stoppages at a half's edge stay in each D24 view, as the rule says.

What it shows: the sub-second stoppages cost D15 almost nothing. Removing all 87 on PFF gained 0.0007, and the standard views moved by 0.0001. The rule asked for no harm anywhere, one D24 game lost 0.0007, and so the step is not adopted. The code is kept (`conventions.remove_brief_stoppages`, `stoppage_trial.py`); its `confirm` and `build` paths have not run on real data. [Development](dfl_submission/stoppage_recipe/development.json)

## D26 — Ownership-change timing in Metrica's convention (2026-11-11, written before any D26 offset under the second marking rule, shift or score is computed)

**Question.** On the three Metrica sample games, D15's ownership changes land a median 4 frames after Metrica's marks (D21), while on DFL and SkillCorner the median is 0. The competition's ground truth comes from Metrica. Is the lag robust, and does moving D15's ownership changes earlier by that much help on Metrica's games without hurting elsewhere? This is round 6's step 3.

**What was seen before this rule.** The records of D1–D25. In particular D24 reported, per game, the median N = 3 ownership offset under D19's marking: +3, +4 and +3 frames on the clean view, and +3, +6 and +3 on the masked view. No offset under the second marking rule below has been computed, and no shift has been applied or scored.

**Standing reference.** D15 (D25 was not adopted).

**Step 1: is the lag our own rule?** D19 marks the opponent at the end frame of a "ball lost" event. The *second marking rule* drops that mark, so the opponent's possession starts at its own next team event; everything else in D19's targets (span, dead ball, the other marks, the first-mark rule) is unchanged. A pipeline check comes first: the second rule must leave every dead and X frame of D19's raw targets unchanged in all six halves.

For each game, each marking rule (D19's and the second) and each of two views (D19's clean view and D24's masked view), the median signed offset of D15's ownership changes against the rule's N = 3 targets is computed as in D11 (changes between teams only, matched at 25 frames). That is 12 medians.

**Step 2: shift only if robust.** If all 12 medians are at least +2 frames, the shift s is the smallest of them, rounded half away from zero. Otherwise D26 stops here: no shift, nothing else scored.

**The shift.** After decoding, each predicted change directly between the two teams moves s frames earlier: the new team starts s frames sooner. A change never moves to or before the previous boundary of any type (the earlier spell keeps at least one frame). Stops and restarts are not moved.

**Adopt if all of these hold**, compared with D15:
- *leave one game out, on D19's clean view:* for each game, the shift estimated by step 2 from the other two games alone (8 medians; no shift if any is under +2) must not lower that game's N = 3 spellF1, and the mean of the three changes must be at least +0.003;
- *D24:* with s, the D24 mean is not lower;
- *guards on development, with s:* standard combined N = 3 spellF1 (DFL WPY and the four SkillCorner matches), DFL dead ball carried and team swap, and mean N = 3 spellF1 on the eight PFF development matches each drop by at most 0.01. Those vendors mark changes at other moments, so a small loss there is expected.

Reported, not gated: N = 1.75, 4 and 4.5, raw labels, and every Metrica score against the second rule's targets.

**Confirmation, once,** if development passes: with s, the same guards on the reused DFL/SkillCorner holdout (WQQ, WR9 and their guard views; the four SkillCorner holdout matches) and on the PFF untouched set (mean N = 3 spellF1 drops by at most 0.01), which is parsed only for this step, with D15 forecast on it once. D15 must first reproduce its holdout record. The Metrica conditions are not repeated: they are the same games.

**Disclosure.** If adopted, this puts one value estimated on Metrica's sample data, s, into the submission. D19 said no value would be chosen on these games; this rule amends that for s alone. The precedent is D1, whose convention interval came from the same files. Derek approved the plan that includes this step and spends the untouched PFF set, the last one in reserve. The technical document will say so.

**If it passes,** D15 with the shift becomes the submission after the rebuild checks (twice on macOS in a fresh virtual environment, then `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`). **If it fails,** D15 stays.

Code: a `handover` option in `metrica_sample`'s marks (default unchanged), and `dfl_submission/timing_trial.py` (`develop`, `confirm`, `build`). Records go to `dfl_submission/timing_recipe/`, each written once. Every D26 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-11).** Failed at development, so **D15 stays**. Confirmation did not run: the holdout was not scored, and the PFF untouched set was neither parsed nor scored, so it is still in reserve. The pipeline checks passed first: D15 reproduced its development record (PFF development included), its D19 and D24 measurements; re-reading the Metrica files reproduced D19's parsed targets; and the second marking rule left every dead and X frame unchanged.

*Steps 1–2, the lag is robust but smaller under the second rule.* Median N = 3 ownership offsets (frames, predicted minus true):

| Game | D19 rule, clean | D19 rule, masked | Second rule, clean | Second rule, masked |
|---:|---:|---:|---:|---:|
| 1 | +3 | +3 | +2 | +2 |
| 2 | +4 | +6 | +2.5 | +4 |
| 3 | +3 | +3 | +2 | +2 |

All 12 medians reached +2, so the shift was s = 2 frames. About a third of the lag D21 saw comes from D19 marking the opponent at the end of a lost ball.

*Adoption conditions, with s = 2, compared with D15:*

| Condition | Change | Required |
|---|---:|---|
| Leave one game out, clean view (each game's shift from the other two was 2) | game 1 +0.0031, game 2 **−0.0008**, game 3 **−0.0014**; mean **+0.0003** | each ≥ 0, mean ≥ +0.003 |
| D24 mean (0.67367 → 0.67376) | +0.00009 | ≥ 0 |
| Standard combined N = 3 | −0.0020 | ≥ −0.01 |
| DFL dead ball carried / team swap | −0.0042 / −0.0015 | each ≥ −0.01 |
| PFF development | +0.0002 | ≥ −0.01 |

Reported: N = 1.75 / 4 / 4.5 on the standard views −0.0015 / −0.0018 / −0.0017, raw −0.0014. On the Metrica clean view with s = 2, against D19's targets: N = 3 +0.0003, raw +0.0007, N = 4.5 −0.0005; against the second rule's targets: N = 3 +0.0003, raw −0.0002.

What it shows: the lag is real under both marking rules, but moving D15's changes two frames earlier hardly changes spellF1. Spell IoU over possessions that last several seconds barely moves with a two-frame boundary, so the timing offset costs D15 little. Nothing estimated on the Metrica games enters the submission. [Development](dfl_submission/timing_recipe/development.json)

## C2 — A check of the final file without ground truth (2026-11-11, written before its ranges are computed)

**Purpose.** C1 checks alignment and orientation against the public window. C2 checks, with no ground truth at all, that the final file looks like D15's output on external matches. A broken pipeline (everything dead, flickering labels, a missing half) shows up as a statistic far outside anything D15 produces on external data. C2 selects nothing and is not used to choose between files.

**Statistics.** Computed the same way on a label sequence of the competition file and on D15's predictions for external matches. *Graded* frames are the sequence's own non-X frames, and everything is counted inside its non-X runs:
1. *possessions per graded minute:* maximal H or A runs, divided by graded frames / 1,500;
2. *dead share:* D frames divided by graded frames;
3. *sub-second stoppages per 90 graded minutes:* D runs shorter than 25 frames, divided by graded frames / 135,000.

**Ranges, from external predictions only.** D15's predictions, decoded at its recipe from the cached inputs its records were scored from, on 28 external match views: DFL WPY in four views, the four SkillCorner development matches, the eight PFF development matches, and the three Metrica games clean (D19) and in D24's three views. For each statistic the range is [½ × the smallest value, 1.5 × the largest]. They are written once to `dfl_submission/output_check_ranges.json` by `python -m dfl_submission.output_check ranges`, with `MTAH_DATA_DIR` set to a nonexistent path.

**Procedure.** Derek runs it once, after the final rebuild matches `SUBMISSION_SHA256.txt`: `python -m dfl_submission.output_check check data/submission.csv`. It prints PASS, or FAIL with the names of the statistics outside their ranges, and no number. No AI tool sees the file or the statistics.

**After a fail.** Only a pipeline investigation, as with C1: team orientation, frame offset, half spans, decoder inputs. Models, recipes and thresholds do not change.

**Ranges (computed 2026-11-11, before any check).** Over the 28 external views D15 produced 1.53–2.94 possessions per graded minute, a dead share of 0.30–0.58, and 0–20.2 sub-second stoppages per 90 minutes. The ranges are therefore 0.76–4.42, 0.15–0.87 and 0–30.3. [Ranges](dfl_submission/output_check_ranges.json)

*Amendment, 2026-11-12, before any check has run:* D27 replaced D15 as the submission. D27 changes nothing on an external match, because every external half starts at its kick-off. D15's external predictions are therefore also D27's, and the ranges stand unchanged. If D28 is adopted, its own amendment recomputes them.

**Outcome (2026-11-12).** **PASS.** Derek ran it on the frozen file, `data/submission.csv` with SHA-256 `0ddbb34216fdab55752a5b46458639f745cc4e88c074ec0787d771f3630fceff` (D27; the SHA was checked against `SUBMISSION_SHA256.txt` first). The check printed only `PASS`; no statistic was shown. D28 was not adopted, so the ranges were not recomputed. Later the same day, after C1, it was run a second time on the same file and again printed only `PASS`. The check is deterministic, so this adds no information and changed nothing.

*Amendment for D43, 2026-11-12, 18:48 EST, written before the new ranges are computed or any check runs on the new file:* D43 replaced the submission with the player-set model (`a56574fe…`). C2 is re-declared for it as C3 requires. The statistics, the 28 external match views and the factors (½ × the smallest value, 1.5 × the largest) are unchanged; the predictions are the player-set model's at ownership cost 10, decoded from the cached inputs its D42 development record was scored from (`data/d42/cache`). The ranges are written once to `dfl_submission/output_check_ranges_d43.json` (`python -m dfl_submission.output_check ranges-d43`), and `check` reads that file by default; the D15 ranges stay in `output_check_ranges.json` for the record. Derek runs `python -m dfl_submission.output_check check data/submission.csv` once, in a terminal outside the AI session, after checking the file's SHA-256 against `SUBMISSION_SHA256.txt`, and reports PASS or FAIL only.

*D43 ranges (computed 2026-11-12, 18:50 EST, before any check of the new file).* Over the 28 external views the player-set model produced 1.53–2.91 possessions per graded minute, a dead share of 0.30–0.58 and 0.92–23.9 sub-second stoppages per 90 minutes. The ranges are therefore 0.77–4.37, 0.15–0.87 and 0.46–35.8. Unlike D15's (0–30.3), the last range starts above zero, because every external view has at least one sub-second stoppage; a file with none would fail. [Ranges](dfl_submission/output_check_ranges_d43.json)

*Amendment for D48, 2026-11-13, 06:37 EST, written before the new ranges are computed, before any stack build has run and before any check of a new file:* D48's confirmation passed, so the stack (D49's long-hidden views, D50's attacking-frame tokens, three seeds, the y mirror, 8,100 steps, ownership cost 14) is to replace D43 in the submission. C2 is re-declared for it as C3 requires. The statistics, the 28 external match views and the factors (½ × the smallest value, 1.5 × the largest) are unchanged; the predictions are the confirmed stack's at ownership cost 14, decoded from the cached inputs its development record (`stack_recipe/development_t.json`) was scored from. The ranges are written once to `dfl_submission/output_check_ranges_d48.json`, and `check` reads that file by default from then on; the D15 and D43 ranges stay in their files for the record. Derek runs `python -m dfl_submission.output_check check data/submission.csv` once, in a terminal outside the AI session, after `shasum -a 256 -c SUBMISSION_SHA256.txt`, and reports PASS or FAIL only.

*D48 ranges (computed 2026-11-13, 06:43 EST, before any stack build ran).* Over the 28 external views the confirmed stack at ownership cost 14 produced 1.65–2.78 possessions per graded minute, a dead share of 0.30–0.55 and 0–37.7 sub-second stoppages per 90 minutes; decoding its caches first reproduced the cost-14 row of `development_t.json`. The ranges are therefore 0.83–4.16, 0.15–0.82 and 0–56.5. `check` reads `output_check_ranges_d48.json` by default from now on and takes `--ranges` for another file; `ranges` and `ranges-d43` still reproduce their files. **While the standing file is still D43's (`a56574fe…`), C2 for it must be run with `--ranges dfl_submission/output_check_ranges_d43.json`.** [Ranges](dfl_submission/output_check_ranges_d48.json)

**Outcome for D48 (2026-11-13, 17:43 EST). PASS.** Run once by the orchestrator under Derek's instruction of 17:42 EST (the amended procedure is in the note "Derek hands over the remaining steps"): `shasum -a 256 -c SUBMISSION_SHA256.txt` printed `data/submission.csv: OK` for `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`, and `python -m dfl_submission.output_check check data/submission.csv`, read with only its verdict line kept, printed `PASS`. No statistic was shown.

*Amendment for D53, 2026-11-13, 12:48 EST, written before these ranges are computed and before any check of any new file; it applies only if Derek chooses to ship the D53 fallback:* the statistics, the 28 external match views and the factors are unchanged; the predictions are the fallback's at ownership cost 14, decoded from the cached inputs `stack_recipe/development_d53.json` was scored from. The ranges are written once to `dfl_submission/output_check_ranges_d53.json` and are not `check`'s default; if the fallback ships, `check` is pointed at them and Derek runs it once, outside the AI session, on that file.

*D53 ranges (computed 2026-11-13, 12:52 EST, before any check of any new file).* Over the 28 external views the fallback at ownership cost 14 produced 1.58–2.83 possessions per graded minute, a dead share of 0.30–0.55 and 0–38.6 sub-second stoppages per 90 minutes; the ranges are 0.79–4.24, 0.15–0.82 and 0–57.9. They are not `check`'s default. [Ranges](dfl_submission/output_check_ranges_d53.json)

## D27 — Dead ball until the tagged second-half kick-off (2026-11-12, written before any D27 code runs on the competition bundle)

**Question.** FAQ.md says the second half is graded from its first frame, but play starts only at the kick-off about two minutes later. Every external half used for training or scoring starts at its kick-off, so D15's output on that stretch has never been measured. The Laws of the Game settle it without data: no team has possession before the kick-off. This is round 7's step 1.

**What was seen before this rule.** The records of D1–D26 and C2; DATA.md, FAQ.md and the organisers' `starter/load_smart_tagging.py`. No competition file, tag or label has been opened.

**Rule.** After D15 decodes the match, for each half:
- *Tags:* the Smart Tagging instances whose code is `SET PIECES` and whose `Type` label is `Kick Off (Start)`. Code, group and value are compared case-insensitively after trimming spaces, and `Kick Off (After Goal)` does not match. A tag's frames are round(seconds × 25), as DATA.md documents, rounded as the organisers' loader rounds.
- *Which half:* a tag belongs to a half if it ends at or after the half's first frame and starts no more than 5 minutes (7,500 frames) after it. A half uses the rule only if exactly one tag belongs to it; with none, or more than one, the half is untouched.
- *The change:* every frame from the half's first frame up to, but not including, the tag's start frame becomes `D`. If the tag starts at or before the half's first frame, nothing changes. Frames outside the halves stay `X`.

**Why the start of the tag.** The organisers' loader says each kick-off tag is a window of about 5 s that contains the real kick-off. Its start is therefore never later than the kick-off, so the rule cannot overwrite live play. D15 still decides the last seconds before the kick. In the first half, the published first graded frame is the kick-off itself and the tag contains it, so nothing changes there.

**No tuning and no external score.** There is one hand-set limit (5 minutes) and no grid. Every external half starts at its kick-off, so on every external match the rule does nothing by construction and cannot be scored. It is adopted as a constraint from the Laws of the Game, on tests and a rebuild. The Nov 3 email allows Smart Tagging as an input. The rule uses only the start and end times of the kick-off tags, not their team or any other tag.

**Adoption.**
- `tests/test_kickoff.py` passes, on a SportsCode XML fixture that the test writes itself, and so does the whole suite.
- The competition file is built twice with `python -m dfl_submission.kickoff build`. Both builds must pass the validator and give the same SHA-256. Only the validator line and the SHA are read.
- If the SHA equals D15's (`14d18a23…`), the rule changed no label. D27 is recorded as having no effect, and D15's build command stays.
- If the SHA differs, the D27 file becomes the submission after the release check (`release/check.sh`: twice on macOS, `linux/arm64` and `linux/amd64`, tests in the clone). Any change can only be on second-half frames before the tagged kick-off.
- Either way the rule stays in force for any later candidate's build (D28), because it is a constraint, not a tuned step.

Code: `dfl_submission/kickoff.py` (`read_tags`, `apply`, `build`). The build is D15's (`scale_trial`), then `apply`, then `build.write_csv`.

**Outcome (2026-11-12).** Adopted: **D27 replaces D15 as the submission**, subject to the release check.
- `tests/test_kickoff.py` (16 tests) passes, and so does the whole suite.
- Three builds with `python -m dfl_submission.kickoff build` passed the validator and gave the same SHA-256, `0ddbb34216fdab55752a5b46458639f745cc4e88c074ec0787d771f3630fceff`. D15's is `14d18a23…`.
- Pipeline check, also reading only the validator line and the SHA: D15's labels through the same code without the rule (`kickoff.d15_labels`) reproduced `14d18a23…`. The difference therefore comes from the rule alone.

- Release check (`release/check.sh`, 2026-11-12): a fresh clone at commit `17a0312` rebuilt it twice on macOS (fresh venvs, Homebrew Python 3.12.13) and in `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`; all four gave `0ddbb342…` and passed the validator, and 269 of 269 tests passed in the clone.

What it means, read from the rule and not from the file: D15 had given a team at least one second-half frame before the start of the kick-off tag, and those frames are now `D`. How many was not read. Under the rule, nothing else can change: the first half's tag contains its first graded frame (DATA.md's verified time base), so it starts at or before it. Only the validator line and the SHA of each build were read.

## D28 — Fill short ball dropouts (2026-11-12, written before any D28 fill is computed or scored)

**Question.** A hidden ball is D15's largest measured loss: going from clean to masked tracking costs 0.055 N = 3 spellF1 on DFL (D21) and 0.075 on Metrica (D24). The pipeline uses the ball only where it is seen and never fills a gap. Does filling short dropouts with a straight line help? This is round 7's step 2.

**What was seen before this rule.** The records of D1–D26, the D27 rule, and C2's ranges. To check how a missing ball is encoded, the share of frames with a missing ball was printed for one PFF development half (0.62) and one Metrica half (0.43). A missing ball is NaN in both coordinates. No fill has been computed and nothing has been scored.

**Candidate.** Frozen D15, with nothing retrained and the decoder unchanged. Before the models see a half, the ball is filled in each run of frames where it is missing if:
- the run is at most K frames long; and
- the ball is seen on the frame just before and the frame just after the run, in the same half.

Each filled frame gets the point at its time on the straight line between those two sightings. Runs that touch the half's first or last frame are left alone, as is a half with no seen ball. A ball is seen on a frame when both coordinates are finite. In the build, the fill runs per half on the ball exactly as the models receive it, after the existing validity mask.

**Grid.** K ∈ {6, 12, 25} frames (0.24, 0.48 and 1 s). Nothing above 1 s: over a longer gap the ball can change hands between sightings, and the synthetic masks, which are independent of play, would flatter long fills.

**Coverage first.** Before any scoring, `gap_trial coverage` records, per development group, the number of frames with a missing ball and the share of them each K fills. The groups are DFL WPY clean, masked, dead ball carried and team swap; SkillCorner development; PFF development; Metrica clean; and D24 masked, handled and team swap. These numbers are reported only, and nothing is selected on them.

**Pipeline checks, before any candidate score is read.** D15 must reproduce its development record (PFF development included), its D19 measurement and its D24 measurement exactly. With K = 0 the fill must leave every ball unchanged (tested). Otherwise D28 stops without a record.

**Development, once per K.** D15 with the K fill is forecast on DFL WPY in its four views, the four SkillCorner development matches, the eight PFF development matches, the three Metrica games (D19's clean view) and the D24 views, and decoded at D15's recipe. A K is eligible if, against D15:
- standard combined N = 3 spellF1 rises by at least +0.005;
- N = 1.75, 4 and 4.5 each drop by at most 0.01;
- DFL dead ball carried and team swap (N = 3) each drop by at most 0.01;
- mean N = 3 spellF1 on the PFF development matches drops by at most 0.01 (veto);
- the Metrica veto passes (mean N = 3 spellF1 at least 0.77004, D19), and the D24 mean is not lower.

The eligible K with the highest standard combined N = 3 spellF1 is selected, and a tie goes to the smaller K. Raw-label scores and per-group changes are reported, not gated.

**Confirmation, once,** if a K is selected. The same conditions apply, except the Metrica and D24 ones (same games), on:
- the reused DFL/SkillCorner holdout: WQQ and WR9 with their guard views, and the four SkillCorner holdout matches;
- PFF confirmation set B, whose mean N = 3 spellF1 may drop by at most 0.01. Set B is parsed only for this step, and D15 is forecast on it once.

D15 must first reproduce its holdout record (`scale_recipe/confirmation.json`). The holdout has confirmed D1, D3 and D15 before, so it is reused evidence; set B is fresh.

**If confirmed,** the build is the fill, then D15, then D27. It becomes the submission after the release check. C2's ranges are then recomputed from the candidate's external predictions, under an amendment written before the check. **If not,** D15 stays (with D27, if D27 changed the file). PFF and Metrica only veto; no value is chosen on them.

**Cut-off.** If D28 is not confirmed by the end of Fri 13 Nov (EST), it stops and is recorded as unfinished. Set B stays unparsed unless development passes.

Code: `dfl_submission/gap_trial.py` (`fill`, `coverage`, `develop`, `confirm`, `build`). `metrica_sample.reference_halves` and `metrica_views.view_halves` get an optional `signals_of` hook; at its default they must reproduce their caches and records exactly. Caches go under `data/d28/`, and records to `dfl_submission/gap_recipe/`, each written once. Every D28 command except `build` runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12).** Failed at development: no K was eligible, so **the submission stays D27** (D15 with the kick-off rule). Confirmation did not run: the holdout was not scored, and PFF confirmation set B was neither parsed nor scored, so it is still in reserve. The pipeline checks passed first: D15 reproduced its development record (PFF development included) and its D19 and D24 measurements, and its recipe reproduced every cached Metrica label.

*Coverage, recorded before scoring.* Share of hidden-ball frames on live play that each K fills:

| Group | Live frames with the ball hidden | K = 6 | K = 12 | K = 25 |
|---|---:|---:|---:|---:|
| DFL masked, dead ball carried, team swap | 32,011 of 81,460 | 0.009 | 0.029 | 0.095 |
| SkillCorner development | 23,200 of 319,119 | 0 | 0 | 0 |
| PFF development | 262,718 of 722,461 | 0.017 | 0.064 | 0.129 |
| Metrica clean | 694 of 269,644 | 0.414 | 0.624 | 1.000 |
| D24 views | 106,327 of 269,644 | 0.009 | 0.026 | 0.094 |

DFL clean has no hidden ball. Every SkillCorner gap is longer than 1 s.

*Development, against D15:*

| Condition | K = 6 | K = 12 | K = 25 | Required |
|---|---:|---:|---:|---|
| Standard combined N = 3 (D15 0.72280) | 0.72280 (+0.0000) | 0.72436 (**+0.0016**) | 0.72124 (−0.0016) | ≥ +0.005 |
| N = 1.75 / 4 / 4.5 | 0 / −0.0003 / −0.0003 | +0.0014 / +0.0014 / +0.0017 | −0.0011 / −0.0011 / −0.0009 | each ≥ −0.01 |
| DFL dead ball carried / team swap | +0.0022 / −0.0055 | +0.0029 / −0.0017 | −0.0095 / −0.0087 | each ≥ −0.01 |
| PFF development | −0.0002 | +0.0008 | −0.0034 | ≥ −0.01 |
| Metrica (D19), mean N = 3 | 0.78031 (passes) | 0.78055 (passes) | 0.78028 (passes) | ≥ 0.77004 |
| D24 mean (D15 0.67367) | **−0.0007** | **−0.0003** | **−0.0003** | ≥ 0 |

Reported: raw-label change on the standard views −0.0002, +0.0008 and −0.0002.

What it shows: a straight line through gaps of up to 1 s reaches at most a tenth of the hidden live ball on the masked views, and hardly moves spellF1. The best K gained a third of the required margin, and every K lowered the D24 mean slightly. The step is not adopted. Nothing is retrained or changed, and D27 stays the submission. The code is kept (`gap_trial.py`); its `confirm` and `build` paths have not run on real data. [Coverage](dfl_submission/gap_recipe/coverage.json) · [Development](dfl_submission/gap_recipe/development.json)

## C3 — What may change after C1 (2026-11-12, written before any round 8 step is built, fitted or scored)

**Why.** C1 passed on `0ddbb342…` (D27), and its record says the check does not run again. Round 8 (D29–D32) continues development until the hard freeze. This rule fixes what may change after C1 and how a changed file is checked without a second C1. Derek approved the plan that contains it, including shipping a changed file without a second C1.

- **C1's output is not used.** It reached an earlier AI session (C1's disclosure). Nothing from it is used in round 8, and it is not shown to the AI session again. Every round 8 step is selected on external data only.
- **Clear gain.** A step changes the submission only if its own external confirmation shows a gain of at least **+0.01** N = 3 spellF1 against the standing model, on matches that step's models never trained on. For D30 that is its leave-one-game-out comparison on the Metrica games (every held-out game is scored by models that never saw it), followed by a no-harm confirmation of the final fit; for D31 its holdout confirmation; for D32 PFF set C. D29 has no external score and ships only alongside a step that clears the bar.
- **C1 is not run again.** Every round 8 change sits behind a switch in the build. With every switch off, the build must reproduce `0ddbb342…` byte for byte, so the reading, team orientation, frame indexing and half spans that C1 checked are unchanged; a step can change only the labels its own switch controls.
- **C2 runs once more** on a new final candidate. Before it runs, an amendment recomputes its ranges from that candidate's predictions on external matches. Derek runs it once and reports PASS or FAIL.
- **Records.** If the file changes, C1's record and section 4 of the technical document are amended to say that development continued under this rule after C1, and what changed. If nothing clears its bar, `0ddbb342…` ships and C1's record stays true as written.
- **Cut-off.** Hard freeze Sun 15 Nov (EST). A step not confirmed by then stops and is recorded as unfinished.

**Outcome (2026-11-12).** No round 8 step cleared its bar: D30 failed leave one game out, D31 and its second attempt D31a failed before confirmation, D32 failed development, and D29 therefore could not ship. **`0ddbb342…` stays the submission** and C1's record stays true as written. Checks on the build path, whose code round 8 extended (`kickoff.model_labels`, the `merge_ball` hook in `build.tactical_tracking_halves`, `round8.py`): `round8 build` with every switch off gave `0ddbb342…`, and so did two rebuilds with the standing `kickoff build`, all with the validator passing; only the validator line and the SHA were read. C2 was not re-declared, because the file did not change.

**Release check (2026-11-12).** Because round 8 changed code on the build path, `release/check.sh` ran once on the final commit `05de6cc` (round 8 committed, the organisers' reply recorded, the pre-public tidy). A fresh clone was rebuilt twice on macOS (fresh venvs, Homebrew Python 3.12.13) and in `python:3.12.14-slim` under `linux/arm64` and `linux/amd64` (Python 3.12.14). All four builds gave `0ddbb342…` and passed the validator, and 352 of 352 tests passed in the clone. Only the validator lines and the SHA-256s were read. C1 and C2 were not run again.

## D30 — Metrica's sample games as a training source (2026-11-12, written before any Metrica training half is prepared or any D30 model is fitted or scored)

**Question.** Every model so far is trained on DFL and SkillCorner tracking. Metrica tracked the competition match, and its three public sample games have so far only been scored (D19, D24). Does training on them help? This is round 8's step 1.

**What was seen before this rule.** The records of D1–D28, C1's and C2's records, and C3. No Metrica half has been prepared for training and nothing has been fitted on these games.

**Data and targets.** The three games as D19 pinned and parsed them (the halves that passed D19's reader gate), with D19's raw labels as targets: event possession, and the dead ball as D19 marks it. D19 said nothing would be trained on these games; this rule amends that, as Derek approved in the plan. Metrica's README offers the data to the community and asks for credit; the README and the technical document give it.

**Training views,** per game g ∈ {1, 2, 3} and half i ∈ {0, 1}, mirroring each set's DFL training views with new seeds:
- *Original set* (as `external_benchmark.halves` builds DFL's training halves): the half as provided (`clean`); `masked`, `corrupt(positions, ball, 870000 + 10g + i)`; `stress`, `stress(positions, ball, 880000 + 10g + i)`.
- *Augmented set* (as `augment.training_halves`), with `rng = numpy.random.default_rng(890000 + 10g + i)`: D2's `handle_dead_ball` on D19's labels (`clean`); `masked`, `wide_masking` and then the training `swap_teams`; `stress`, `stress(…, 880000 + 10g + i)` on the handled ball and then the training `swap_teams`.
- Inputs `long_context.features`; targets `temporal_network.targets(labels[frames])`, as `final_trial.prepare_sequences` writes them.

**Weights.** The Metrica games count as matches of the DFL source: their records carry `source = "dfl"` (and `vendor = "metrica"`). The trees therefore keep half the weight for each of DFL and SkillCorner, shared equally by the matches within a source (four DFL training matches plus the Metrica games), and the networks keep 8 windows per source in each batch, drawing Metrica halves alongside DFL ones. D14 failed by giving a small third source a third of the weight.

**Candidate.** Both model sets refit on their stored training folders plus the Metrica halves: both trees at D3's settings (`touch_trial.fit_trees`), and all four networks at D15's setting (width 96, 5,400 steps, seeds 42 and 97, boundary boosts 0.5 and 1.4). D15's structure, blend and decoder recipe (ownership cost 8) are unchanged. No grid.

**Pipeline checks, before any D30 score is read.**
- With no Metrica half added, the same combine and fit path reproduces D15's four trees byte for byte and its four networks bit for bit (every exported array equal).
- D15 reproduces its development record (PFF development included), its D19 measurement and its D24 measurement from their caches. Before confirmation, it reproduces its holdout record.
- Otherwise D30 stops without a record.

**Leave one game out.** Three fits; fold g leaves out game g's halves. Each fold model is scored on its held-out game: D19's clean view and D24's three views, decoded at D15's recipe against D19's targets, and compared with D15's records for that game (`d15_measurement.json`, `d24_measurement.json`).

**Pass if, against D15:**
1. *D24 gain:* per held-out game, the mean over D24's three views of N = 3 spellF1. The mean change over the three games is at least **+0.01**, and no game's change is negative.
2. *Clean view:* the mean over the three games of clean-view N = 3 spellF1 is not lower (D15 0.78004).
3. *Other targets:* for N = 1.75, 4, 4.5 and control, the mean over the three games of the mean over the four views (clean and D24's three) drops by at most 0.01.
4. *Each fold model on development* (DFL WPY in its four views, the four SkillCorner development matches, the eight PFF development matches): standard combined N = 3 spellF1 drops by at most 0.005; DFL dead ball carried, team swap and mean N = 3 on PFF development each drop by at most 0.01.

Reported, not gated: raw labels, macroF1, and per-game and per-view changes.

**Final fit and confirmation, once,** if leave one game out passes. Both sets refit with all three games. On the reused DFL/SkillCorner holdout (WQQ and WR9 with their guard views, the four SkillCorner holdout matches): standard combined N = 3 drops by at most 0.005, dead ball carried and team swap by at most 0.01. On PFF confirmation set B, parsed only now and forecast by D15 once: mean N = 3 drops by at most 0.005. N = 1.75, 4 and 4.5 are reported. The holdout is reused evidence; set B is fresh. Under C3, D30's +0.01 is condition 1 above, and the confirmation is a no-harm check on the final fit.

**After a pass,** D30 goes into round 8's combine step. The final model is trained on all three Metrica games, so the D19 veto and the D24 mean can no longer measure it; any later Metrica evidence uses the fold models on their held-out games. **If it fails,** D15's models stay.

Code: `dfl_submission/metrica_trial.py` (`prepare`, `check`, `fit`, `logo`, `confirm`, `build`). Prepared halves go to `data/d30/` (gitignored), weights to `dfl_submission/metrica_models/` (`fold1`–`fold3`, `final`), and records to `dfl_submission/metrica_fit_recipe/`, each written once. Every command except `build` runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

## D29 — Dead ball from a tagged goal to the kick-off after it (2026-11-12, written before any D29 code runs on the competition bundle)

**Question.** After a goal no team has possession until the kick-off that restarts play (Laws of the Game). DATA.md lists 4 `GOALS` instances and 4 `SET PIECES` instances of type `Kick Off (After Goal)` in the match's Smart Tagging. D15 decides those stretches from the tracking alone. This is round 8's step 4; like D27 it is a constraint, not a tuned step.

**What was seen before this rule.** The records of D1–D28, C1–C3 and the D30 rule; DATA.md's Smart Tagging section and the D27 code. No competition file, tag or label has been opened.

**Rule.** After the model decodes the match (and with D27 applied):
- *Tags,* compared after trimming spaces and lower-casing as D27 does: kick-off tags are `SET PIECES` instances whose `Type` is `Kick Off (After Goal)`; goal tags are `GOALS` instances. A tag's frames are round(seconds × 25), as D27 rounds them. A tag belongs to the half whose graded block contains its start frame.
- *Pairing:* for each kick-off tag, the goal tag is the one in the same half whose end frame is the latest at or before the kick-off tag's start, and no more than 3 minutes (4,500 frames) before it.
- *Ambiguity:* the pair is not used if two goal tags share that latest end frame, if the same goal tag is the pair of another kick-off tag, or if another goal tag starts inside the stretch the rule would change (*amendment, written before any D29 code was written:* a second goal inside the stretch would otherwise be marked dead). A kick-off tag with no goal tag in reach changes nothing.
- *The change:* every graded frame from the goal tag's end frame up to, but not including, the kick-off tag's start frame becomes `D`.

**Why these ends.** The organisers' loader describes a tag as a window that contains its event, so a kick-off tag starts no later than the kick-off and the rule cannot overwrite play after it. The goal tag's end is taken to be no earlier than the goal itself. That cannot be checked without opening the tags, which this project does not do, so it is an assumption, disclosed here and in the technical document. If a goal tag ended before the ball crossed the line, the rule would mark live frames before that goal dead.

**No tuning and no external score.** Two hand-set limits (3 minutes and the half) and no frame offset. No external match carries Smart Tagging, so the rule cannot be scored. Under C3 it ships only alongside a round 8 step that clears its bar; on its own it never changes the submission.

**Adoption checks (if it ships).** `tests/test_kickoff.py` covers pairing, ambiguity, reach, halves and the change on a SportsCode fixture the test writes. With D29 switched off the build reproduces the standing SHA; the switched-on build passes the validator twice with one SHA. Only the validator line and the SHA are read.

Code: `kickoff.read_instances`, `kickoff.goal_spans`, `kickoff.apply_goals`.

## D31 — The broadcast ball as a second source (2026-11-12, written before any harness, aligner or merge code exists and before anything is scored)

**Question.** A hidden ball is D15's largest measured loss (clean to masked: −0.055 on DFL, −0.075 on Metrica). The tactical ball is missing on 41% of graded frames (DATA.md). The broadcast recording has its own tracking, with the ball seen on 50.5% of its frames and a usable homography on 65.8%, but it is not synchronised to the tactical timeline. Where the tactical ball is missing and the broadcast ball is seen, can the broadcast ball fill it? This is round 8's step 2.

**What was seen before this rule.** The records of D1–D30, C1–C3; DATA.md's broadcast and homography sections and `games.py`; the code of `experiments/broadcast_alignment.py`. No broadcast file, earlier research note or alignment output has been opened. If the competition check fails, only the names of the failed criteria are reported.

**Stretches and offsets (`broadcast_ball.py`).** All positions are centred on the pitch; an orientation is one of the four flips of x and y.
- A *stretch* is a maximal run of usable broadcast frames (valid homography and at least 4 players per team, as `frame_usable`), at least 125 frames (5 s) long.
- *Cost:* the same-team formation error of `formation_cost` (each broadcast player to the nearest tactical player of the same team, capped at 10 m, root mean square, averaged over the two teams), between a broadcast frame and a tactical frame.
- *Coarse search:* broadcast frames every 25 frames in the stretch (at most 120, evenly thinned), against tactical start frames every 25 frames, in all four orientations; a candidate's cost is the mean over the sampled frames with a finite cost, and needs finite costs on at least half of them.
- *Fine search:* the 51 offsets within ±25 frames of the coarse best, in its orientation, on broadcast frames every 5 frames (at most 600). The stretch's offset is the fine minimum.
- *Accepted* if the fine cost is at most MAX_COST, the margin is at least MIN_MARGIN, and no other fine offset is within 0.001 m of the minimum (a tie, rejected so that builds agree across platforms). The margin is the lowest coarse cost in any orientation more than 50 frames away from the best offset, or in another orientation at any offset, minus the fine cost.

**Merge.** For each accepted stretch, each broadcast frame with a seen ball maps to tactical frame b + offset. Where the tactical ball is missing there (it fails the build's validity test), it gets the broadcast ball in the stretch's orientation. A seen tactical ball is never overwritten, players stay tactical, and when two stretches map to one frame the earlier stretch wins. Teams are reconciled from the metadata (`reconcile_teams`).

**Synthetic two-camera harness (`two_camera.py`), external data only, every parameter fixed here.** From one external half's clean tracking (the truth) it builds a broadcast recording:
- *Timeline:* broadcast frame b shows tactical frame b + d₀, with d₀ uniform on [−3000, 3000]. Live segments last 125 frames plus an exponential with mean 900. Between them: a cut (probability 0.5; no homography, 50–500 frames), a replay at normal speed (0.3; valid homography, 75–200 frames, starting 125–375 frames before the live time), or a slow replay (0.2; the same at half speed). A replay is wrapped in 25-frame wipes without homography with probability 0.5; otherwise it touches the live frames.
- *Camera:* a window 50 m wide following the true ball's x (exponential smoothing with a 25-frame time constant, held while the ball is unseen), full pitch width. Players inside it are seen, each dropped for a 25-frame block with probability 0.1, with a per-segment bias N(0, 0.5 m) and per-frame noise N(0, 0.3 m) on each axis.
- *Ball:* seen inside the window when the true ball is seen, through a two-state switch (mean seen run 100 frames, mean hidden run 25 frames), with a radial error drawn from a log-normal with median 1.5 m and 90th percentile 5 m in a uniform direction, independently per frame.
- *Orientation:* one of the four, uniform per half.

**Alignment constants, set on the harness before any score.** Harness recordings of the 8 halves of the four DFL training matches (seeds 900000 + 10 × match index + half). Grid: MAX_COST ∈ {1.0, 1.5, 2.0, 3.0} m × MIN_MARGIN ∈ {0.25, 0.5, 1.0, 2.0} m. A frame is *wrong* if its mapped tactical frame is more than 2 frames from the frame it really shows (a slow-replay frame counts by its rounded true frame), or its orientation is wrong. Selected: the setting with the most accepted broadcast frames among those with wrong frames at most 1% of accepted frames; ties go to the larger margin, then the smaller cost. If no setting qualifies, D31 stops. Recorded once in `broadcast_recipe/constants.json`.

**Development (D15 frozen, nothing retrained).** DFL WPY's masked, dead-ball-carried and team-swap views, and D24's three views of the three Metrica games. Each half gets a harness recording of its clean truth (seeds 910000 + 100 × group + 10 × game or match + half) and the view's ball is merged through the real aligner at the selected constants. The merged ball is at least as bad as the competition check allows (the harness error above has median 1.5 m and 90th percentile 5 m before any alignment error). *Thinning:* recovery is cut to r = 10, 20, 30 and 40% of each half's frames with a missing view ball, by keeping recovered stretches in a seeded random order and cutting the last one; if a half recovers less, all of it is kept and the shortfall recorded. Gains are N = 3 spellF1 against D15 without the merge: the DFL group is the mean change over its three views; the Metrica group is the change of the D24 mean. **r\*** is the smallest r gaining at least +0.01 on both groups. If none does, D31 stops. Before any merged score is read, D15's labels without the merge must equal its cached labels on every development half.

**Confirmation, once, at r\*.** WQQ and WR9 in the same three views, with fresh harness seeds (920000 + …). Per match, the mean change over its views. Pass if the mean over the two matches is at least **+0.01** and neither match is lower. The holdout has confirmed earlier steps, so it is reused evidence.

**Competition check, run by Derek outside the AI session, PASS or FAIL with the failed names only** (`python -m dfl_submission.broadcast_trial check`):
- *teams:* the teams reconcile from the metadata;
- *coverage:* accepted stretches map onto at least 30% of graded frames;
- *agreement:* on at least 2,000 graded frames where both balls are seen, their distance has a median of at most 1.5 m and a 90th percentile of at most 5 m;
- *recovery:* the merge fills at least r\* of the graded frames whose tactical ball is missing.

**Adoption.** Only if confirmation and the competition check both pass. Then the build reads the three broadcast files and merges before D15, and `release/check.sh` copies them too. Otherwise D15's inputs stay.

Code: `dfl_submission/broadcast_ball.py`, `dfl_submission/two_camera.py`, `dfl_submission/broadcast_trial.py` (`calibrate`, `develop`, `confirm`, `check`, `build`). Records go to `dfl_submission/broadcast_recipe/`, each written once. Every command except `check` and `build` runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

## D32 — Label-free adaptation, proved on an unseen vendor (2026-11-12, written before any D32 code exists and before anything is fitted or scored)

**Question.** D6 continued one ownership tree on a match's own confident pseudo-labels and gained +0.0006 on SkillCorner 2020, short of its gate. Does the same per-match adaptation, with the standing model D15 as teacher and extended to the stoppage head, help on PFF, a vendor none of the models has seen? This is round 8's step 3. It is applied to the competition match only on a written organiser yes (below).

**What was seen before this rule.** The records of D1–D31 (D30 and D31 not yet scored), C1–C3, and D6's code and record. No D32 fit or score exists.

**Candidate, per match (its two halves), every constant D6's.**
- *Teacher:* D15 as built (both sets, the 0.5 blend, D15's networks and decoder recipe). Its four final decoder inputs at 25 Hz, and the original set's two long-context trees at their 5 Hz feature frames.
- *Pseudo-labels* at the feature frames, from D15's final inputs (`adaptation_trial.pseudo_labels`, confidence 0.90): ownership where the stoppage probability is at most 0.10 and one team's probability is at least 0.90; stoppage where the stoppage probability is at least 0.90 (dead) or at most 0.10 (live). A head whose pseudo-labels lack a class is not adapted on that match.
- *Students:* the original set's ownership tree and stoppage tree, each continued for 40 rounds with its training parameters. Replay is 75% of the weight, shared as in the original fit (`long_context.source_weights`), from that set's training halves at frames divisible by 25 (D6's ownership replay, `data/d6_replay.npz`, and a stoppage replay built the same way from the known frames); pseudo-labels are 25%, shared equally by class. Ownership rows are mirrored with the teams swapped, as in the original fit; stoppage rows are not, as in the original fit.
- *Adjustment:* each student's probability minus the original tree's, at the feature frames, interpolated to 25 Hz, times a multiplier m, added to D15's final home-ownership and stoppage probabilities and clipped to [0, 1]. Decoded at D15's recipe. Grid m ∈ {0.25, 0.5, 1.0}, one student per match for all three.

**Development.** The eight PFF development matches, each adapted on its own tracking only and scored against PFF's labels as D7 scores them, against D15 (m = 0, its development record). A multiplier is eligible if mean N = 3 spellF1 rises by at least **+0.01**, no match loses more than 0.02, and mean N = 1.75, 4 and 4.5 each drop by at most 0.01. The eligible m with the highest mean N = 3, then N = 3 macroF1, then grid order, is selected. If none is eligible, D32 stops.

**Confirmation, once.** PFF confirmation set C, parsed only for this step, D15 forecast on it once: at the selected m, mean N = 3 spellF1 rises by at least **+0.01**. Set C is fresh.

**Determinism.** The adapted labels for the first PFF development match must be byte-identical on macOS and in `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`. If not, D32 cannot ship.

**Applied only if** the organisers give a written yes by Sat 14 Nov (EST) to label-free fitting on the provided tracking, the question of the 11 November email (the follow-up can go from 13 Nov; Derek sends it). Without it D32 is recorded and not applied, whatever its scores. PFF only measures here: one multiplier is chosen on PFF development (Derek approved this in the plan), and no PFF value enters the submission unless D32 is applied, which the technical document would disclose.

Code: `dfl_submission/vendor_adaptation.py` (`replay`, `develop`, `confirm`, `determinism`, `build`). Records go to `dfl_submission/vendor_adaptation_recipe/`, each written once. Every command except `build` runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D32 outcome (2026-11-12).** Failed at development, so **D32 stops** and is not applied, whatever the organisers reply. Confirmation and the determinism check did not run, and PFF confirmation set C was neither parsed nor scored. The pipeline check passed first: at multiplier 0 the adaptation code reproduced D15's PFF development record exactly (mean N = 3 0.65615). Every match supplied both classes for both heads (ownership pseudo-labels 8,220–14,003 per match, stoppage 21,502–26,594), so every match was adapted.

| Multiplier | Mean N = 3 | Change | Worst match | N = 1.75 / 4 / 4.5 | Eligible |
|---:|---:|---:|---:|---|---|
| 0 (D15) | 0.65615 | — | — | — | — |
| 0.25 | 0.65448 | −0.0017 | −0.0073 | −0.0015 / −0.0017 / −0.0018 | No |
| 0.5 | 0.65578 | −0.0004 | −0.0063 | +0.0004 / −0.0004 / −0.0005 | No |
| 1.0 | 0.65477 | −0.0014 | −0.0063 | −0.0016 / −0.0018 / −0.0024 | No |

What it shows: as in D6, continuing the trees on the match's own confident outputs moves the decoded possessions very little, and here slightly down; the stoppage head did not change that. Nothing from PFF enters the submission. [Development](dfl_submission/vendor_adaptation_recipe/development.json)

**D31 outcome (2026-11-12).** Stopped at the constant setting: no setting qualified, so development, confirmation and the competition check did not run, and nothing was scored. On the 8 harness recordings (643 stretches) every stretch found the right orientation, but against the masked tactical view the formation cost at the true offset was about 3–4 m and nearly flat within ±8 frames, so many offsets landed 3–5 frames away. The strictest settings accepted only a few stretches, all wrong (max cost 1.0 m: 175 frames accepted, 175 wrong); the loosest accepted 47,522 frames with 17,598 wrong (37%), against the 1% allowed. A diagnosis on one recording found the cause in the cost, not in the code: (1) the masked view drops each player track in 50-frame blocks with probability 0.2, and every broadcast player whose tactical counterpart is missing adds a capped error of up to 10 m, which dominates the root mean square; (2) against the clean view the minimum lay 2–4 frames late, because a constant position bias can be traded against a small time shift when a team moves together. [Constants record](dfl_submission/broadcast_recipe/constants.json)

## D31a — The broadcast ball, second attempt with a robust cost (2026-11-12, written after D31's constant setting failed and before any D31a code exists or anything is scored)

**Why a second attempt, and what it may use.** D31 stopped at its constant setting, before any score: its development views, its holdout confirmation and the competition were not touched. Its diagnosis named two causes in the cost (above). D31a changes only the cost, in those two ways, and repeats D31 otherwise unchanged. It is a second attempt on the same eight DFL training halves for the constant setting, which is disclosed as weaker evidence for the constants; it uses fresh harness seeds there. Development and confirmation are first attempts. If D31a fails at any stage, D31 is finished and is not attempted again.

**The cost, changed in two ways.**
1. *Trimmed:* within each team, only the 75% smallest per-player errors (rounded up) count; the rest are dropped before the root mean square. A broadcast player whose tactical counterpart is missing no longer dominates.
2. *Translation per stretch, in the fine search:* for each candidate offset, the median displacement between the kept broadcast players and their nearest tactical teammates over the sampled frames is a translation τ; the cost is recomputed with the broadcast players moved by −τ. The stretch keeps the τ of its chosen offset, and its broadcast ball is moved by −τ when merged. The coarse search, margin and tie rule are D31's (the margin compares coarse costs with the translated fine cost).

**Constants.** Harness recordings of the same eight halves with seeds 905000 + 10 × match index + half. Grid: MAX_COST ∈ {0.75, 1.0, 1.5, 2.0, 3.0} m × MIN_MARGIN ∈ {0.25, 0.5, 1.0, 2.0} m. D31's definition of a wrong frame (more than 2 frames off, or the wrong orientation) and its selection rule (most accepted frames with at most 1% wrong; then larger margin; then smaller cost) are unchanged. If no setting qualifies, D31 is finished.

**Everything else is D31's:** the harness and its parameters, the merge, development (seeds 910000 + …, the thinning levels and r\*), confirmation (seeds 920000 + …, +0.01 with neither match lower), the competition check run by Derek (teams, coverage ≥ 30%, agreement on ≥ 2,000 frames with median ≤ 1.5 m and 90th percentile ≤ 5 m, recovery ≥ r\*), adoption and the build. Records go to `dfl_submission/broadcast_recipe/` with an `a_` prefix (`a_constants.json`, `a_development.json`, `a_confirmation.json`), each written once.

**D30 outcome (2026-11-12).** Failed at leave one game out, so **D15's models stay**. The final fit and confirmation did not run: the holdout was not scored, and PFF confirmation set B was neither parsed nor scored, so it is still in reserve. The pipeline checks passed first: with no Metrica half, the combine and fit path refit D15's four trees byte for byte and its four networks bit for bit, the held-out scoring path reproduced D15's D19 and D24 records exactly, and D15 reproduced its development record.

*Held-out games, against D15 (N = 3 spellF1 change):*

| Game | Clean | Masked | Handled | Team swap | D24 mean |
|---:|---:|---:|---:|---:|---:|
| 1 | +0.0100 | +0.0009 | −0.0059 | +0.0230 | +0.0060 |
| 2 | +0.0331 | +0.0281 | −0.0047 | +0.0042 | +0.0092 |
| 3 | +0.0141 | +0.0073 | −0.0067 | −0.0037 | **−0.0010** |

| Condition | Result | Required |
|---|---:|---|
| 1. D24 gain, mean over the games | **+0.0047** | ≥ +0.01, no game negative |
| 2. Clean view, mean | +0.0191 | ≥ 0 |
| 3. N = 1.75 / 4 / 4.5 / control, four views | +0.0081 / +0.0093 / +0.0108 / +0.0120 | each ≥ −0.01 |
| 4. Fold models on development: standard N = 3 | −0.0063 / −0.0083 / −0.0089 | each ≥ −0.005 |
| 4. Dead ball carried | **−0.0167 / −0.0593 / −0.0528** | each ≥ −0.01 |
| 4. Team swap / PFF development | +0.0010 to +0.0094 / −0.0018 to −0.0066 | each ≥ −0.01 |

What it shows: training on Metrica's corrected tracking helps where the ball is missing exactly when it is dead (clean +0.019) and on masked tracking, but it hurts wherever the ball is seen while dead: the handled view lost in every game and DFL's dead-ball-carried view lost in every fold. Metrica's sample games teach the dead-ball shortcut that D2 was built to remove, and the competition's tracking shows the ball while it is dead. Nothing from the Metrica games is trained into the submission. The fold weights are kept in `dfl_submission/metrica_models/` for the record. [Pipeline checks](dfl_submission/metrica_fit_recipe/check.json) · [Leave one game out](dfl_submission/metrica_fit_recipe/logo.json)

**D31a outcome (2026-11-12).** Failed at development: no level reached its gain, so **D31 is finished**. Confirmation and the competition check did not run, and no broadcast file was read.
- *Constants (second attempt on the calibration halves, fresh seeds).* Only the strictest corner of the grid qualified: max cost 0.75 m, margin 2.0 m, which accepted 4,770 broadcast frames with 1 wrong. Looser settings accepted far more frames but with 6–14% wrong (for example max cost 1.0 m and margin 1.0 m: 36,994 accepted, 2,361 wrong).
- *Development.* D15 reproduced its development and D24 records, and its labels without the merge equalled its cached ones on every view (D24 mean 0.67367). At the selected constants, 55 of 1,896 stretches were accepted and the merge filled 2,398 of 852,873 frames with a missing view ball (0.28%; DFL 0.36%, Metrica 0.26%), so every thinning level kept all of it, with a shortfall at every level. On the Metrica views 658 of 9,950 accepted frames (6.6%) were more than 2 frames off, against the 1% the constants were chosen for.
- *Gains, at every level:* DFL group +0.0003, Metrica D24 mean +0.0006, against +0.01 required.

What it shows: on the harness, an alignment strict enough to be right 99% of the time accepts so few stretches that the broadcast ball barely reaches the hidden frames, and the strict constants do not hold their error rate on other halves. With the competition's 41% missing tactical ball, the step would have needed at least ten times more accepted coverage than any 1%-error setting gave. Nothing from the broadcast recording enters the submission. [Constants](dfl_submission/broadcast_recipe/a_constants.json) · [Development](dfl_submission/broadcast_recipe/a_development.json)

## Note — The organisers' reply, and the end of the search (2026-11-12)

**The reply.** On 11 November we asked the organisers two questions (technical document, sections 7 and 8): whether the 3 November rule "If you need to train (or retrain), you MUST use only external data" withdraws RULES.md's allowance for learning without human labels on the provided game, and whether a synthetic ball-dropout rate may be set from DATA.md's published "ball missing on 41% of graded frames". They replied on 12 November without a yes or a no to either question. They repeated the 3 November clarification word for word and added: "Each team is responsible for making sure that its particular project, tool, platform or dataset complies with the competition rules. As a reference, anything that is not explicitly allowed is not permitted."

**What follows.**
1. *Label-free fitting on the provided tracking is closed for good.* It is not on the allowed list, so it is not permitted. D5 and D6 stay external-only results, D32 needed the same permission, and D10 (adaptation on the provided tracking, planned to wait for a written yes) is never written. Nothing is fitted, calibrated or tuned on the provided tracking.
2. *The 60% ball visibility stays, disclosed* (Derek, 12 November). The original model set's degraded view keeps the ball visible 60% of the time, which matches DATA.md's published figure; the retrained set draws visibility from 30–90%, and the two sets are blended equally. The constant is unchanged and is disclosed in the technical document (section 7) and CONSTANTS.md.
3. *The search is closed* (Derek, 12 November). No further step is written or run. PFF confirmation sets B and C, the untouched PFF set and the four 2019–20 SkillCorner confirmation matches stay unused. The submission stays `0ddbb342…`; C1 and C2 are not run again.

## Note — The search reopens (2026-11-12, 13:15 EST)

Derek reopened the search on 12 November ("the model is still nowhere good enough"), after the note above closed it. Everything else in that note stands: nothing is fitted on the provided tracking, the 60% constant stays, and C1 is never run again. Every new step follows C3: a rule written here first, development on external matches, one external confirmation with a gain of at least +0.01 N = 3 spellF1 against the standing model (`0ddbb342…`, D15 then D27), a switch in the build so that with every new step off the build still gives `0ddbb342…`, and C2 re-declared for any new file. C3's hard freeze (Sun 15 Nov, EST) stays.

*PFF (Derek, 12 November, 13:25 EST):* PFF data stays measurement only: a veto and a confirmation source. No network or tree is trained on PFF tracking, because Gradient Sports has not replied and its website terms bar derivative works. D9 stays unwritten.

## D33 — Spell-balanced network training (2026-11-12, 13:15 EST, written before any D33 network is trained or anything is scored)

**Question.** D21 places the loss in ownership: D15 recovers under half of the true spells shorter than 5 s. spellF1 counts every possession once, whatever its length, but each network's loss counts frames, so a 2 s possession weighs a tenth of a 20 s one. Does weighting each frame's ownership loss by the inverse of its possession's length, and giving the boundary heads more weight, raise spellF1? Neither has been tried: D13 changed the targets, D15, D20 and D22 the size, length and seeds.

**What was seen before this rule.** The records of D1–D32 and the D21 measurement. No D33 code has run.

**Reference.** D15 as it stands in the submission (D3's trees and blends, D15's four networks at width 96 and 5,400 steps, cutoff 0.53, ownership cost 8, stoppage cost 2, discount 0.8). D27 acts after decoding and is unchanged.

**Candidate networks.** For each objective setting, all four networks are retrained exactly as D15 trained them (both model sets; seed 42 with boundary weight 0.5 and seed 97 with 1.4; width 96; 5,400 steps; D15's prepared sequences, manifests `d3891c24…` and `2828b0c3…`; Apple MPS; last snapshot; the NumPy export and its check). Only the loss weights change. The trees are D3's and are not refit.
- *Spell weight, α.* At the prepared 5 Hz frames, each sequence's labels are rebuilt from its stored targets (dead, home, known, live). A–B–A spells whose interior lasts under 3 s are merged, inside live segments (`conventions.collapse_brief_changes` at 5 fps), as in the N = 3 targets. A live frame's possession length L is the length in seconds of its run of one team in those merged labels. Its raw weight is (10 / max(L, 2))^α when L < 10 s, otherwise 1. Raw weights are then divided by their mean over the live frames of all the set's sequences from the same source (DFL, SkillCorner), so the total ownership weight per source is unchanged and only its spread over frames changes. The weight multiplies the ownership head's and the ownership-boundary head's terms. The stoppage heads are unchanged. α = 0 gives D15's loss.
- *Boundary weight, e.* The weight of both boundary heads' terms, 0.2 in D15.
- *Pipeline check, before any candidate is trained.* The new trainer at α = 0 and e = 0.2 retrains D15's original-set seed-42 network on MPS. It must reproduce D15's published checkpoint bit for bit, or D33 stops and only the trainer may be fixed.

**Grid (36 points).** Objective (α, e) ∈ {0, 0.5, 1} × {0.2, 0.6}; (0, 0.2) is D15's own networks and is not retrained. Decoder: ownership cost ∈ {8, 10, 12} × discount ∈ {0.8, 0.9}, with D15's cutoff and stoppage cost. Every point is eligible, D15's networks at other decoder settings included.

**Pipeline check on scoring.** D15, decoded from its cached development inputs, must reproduce its development record exactly, and its Metrica and D24 records. Otherwise D33 stops without writing a record.

**Primary.** Standard combined N = 3 spellF1 on DFL WPY and the four SkillCorner development matches, as in D15.

**Eligible if,** compared with D15 on development:
- the primary rises by at least +0.01;
- standard combined spellF1 at N = 1.75, 4 and 4.5 s each drops by no more than 0.01;
- DFL dead ball carried and team swap, at N = 3, each drop by no more than 0.01;
- *PFF veto:* mean N = 3 spellF1 on the eight PFF development matches drops by no more than 0.01;
- *Metrica veto (D19, reference D15):* mean N = 3 spellF1 on the three Metrica sample matches is at least 0.77004;
- *D24 veto:* the D24 mean drops by no more than 0.01.

**Selection.** The eligible point with the highest primary, then N = 3 standard macroF1, then grid order. None eligible → D15 stays and nothing is confirmed.

**Confirmation, once,** on the reused DFL/SkillCorner holdout (WQQ, WR9 and their guard views; the four SkillCorner holdout matches) and PFF confirmation set B, parsed only for this step. D15 must first reproduce its holdout record (`scale_recipe/confirmation.json`); it is then scored on set B as the reference. It passes if, against D15: the holdout primary rises by at least **+0.01** (C3); N = 1.75, 4 and 4.5 each drop by no more than 0.01; dead ball carried and team swap each drop by no more than 0.01; and mean N = 3 spellF1 on set B drops by no more than 0.01. The holdout has confirmed D1, D3 and D15 before, so it is reused evidence; set B is fresh.

**If it passes,** the selected networks and decoder setting replace D15's inside the D27 build, behind a switch; with the switch off the build must still give `0ddbb342…`. The new file then gets the rebuild checks (twice on macOS in fresh virtual environments, then `python:3.12.14-slim` under `linux/arm64` and `linux/amd64`), C2 is re-declared with ranges recomputed from the new model's external predictions, and the records and technical document are updated. PFF and Metrica only veto; no value is chosen on them. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/objective_network.py` (the trainer; `temporal_network.py` stays byte-identical) and `dfl_submission/objective_trial.py` (`check`, `networks`, `develop`, `confirm`, `build`). Weights go to `dfl_submission/objective_models/`, records to `dfl_submission/objective_recipe/`, each written once. Every D33 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 13:55 EST).** No point was eligible, so **D15 stays** and nothing was confirmed. PFF confirmation set B was neither parsed nor scored.
- **Pipeline checks all passed.** The trainer at α = 0 and e = 0.2 retrained D15's original-set seed-42 network bit for bit on MPS (`objective_recipe/check.json`). D15 reproduced its development, Metrica and D24 records from its caches.
- **Development, compared with D15** (N = 3 standard 0.72280), at discount 0.8; the discount 0.9 rows, none of them better than +0.0031, are in the record:

| α | e | Cost | N = 3 standard | Change | DFL clean / masked | SkillCorner | PFF | Metrica | D24 |
|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 0.2 | 8 | 0.72280 | — (D15) | 0.778 / 0.723 | 0.704 | 0.656 | — | — |
| 0 | 0.2 | 12 | 0.71918 | −0.0036 | 0.782 / 0.698 | 0.713 | 0.652 | +0.0034 | +0.0070 |
| 0 | 0.6 | 10 | 0.71739 | −0.0054 | 0.784 / 0.705 | 0.703 | 0.653 | +0.0038 | −0.0000 |
| 0.5 | 0.2 | 12 | 0.72584 | +0.0030 | 0.789 / 0.720 | 0.709 | 0.649 | +0.0140 | +0.0028 |
| 0.5 | 0.6 | 12 | 0.71523 | −0.0076 | 0.793 / 0.702 | 0.698 | 0.650 | +0.0099 | −0.0018 |
| 1 | 0.2 | 10 | 0.72743 | +0.0046 | 0.781 / 0.736 | 0.704 | 0.643 | +0.0132 | −0.0040 |
| **1** | **0.6** | **10** | **0.73049** | **+0.0077** | 0.782 / 0.742 | 0.705 | **0.639** | +0.0178 | −0.0019 |

  The best primary, +0.0077, was under the +0.01 gate, and that point also failed the PFF veto (−0.0173). Every α = 1 point lost 0.012–0.021 on PFF. All 36 rows are in the record.

What it shows: weighting short possessions moves the networks towards finer segmentation, as D20 and D22 found for longer training. It gains where possessions are short and the labels follow them (DFL masked +0.019, Metrica +0.018) and loses where they are coarse (PFF's possession sequences, −0.017). SkillCorner is flat. A larger boundary weight helped only together with α = 1. The networks are kept in `dfl_submission/objective_models/` for reference. [Development](dfl_submission/objective_recipe/development.json) · [Trainer check](dfl_submission/objective_recipe/check.json)

## D34 — Optimisation hyperparameters by cross-validation (2026-11-12, 14:12 EST, written before any D34 network is trained or anything is scored)

**Question.** D15's gain came from a training value nobody had searched: the networks were under-trained at 1,800 steps. Four other optimisation values have never been searched either: the learning rate (a cosine schedule from 0.0015 to 0.0002), the batch (16 windows of 25.6 s), AdamW's weight decay (0.01) and the blocks' dropout (0.08). Does changing one of them raise spellF1? All four act only in training: the NumPy inference is unchanged.

**What was seen before this rule.** The records of D1–D33, including D33's outcome and D22's Stage A table. After D33, a univariate probe on the external training halves (AUC of direction-aware team features against home possession on live frames; nothing trained or decoded) found no possession signal without the ball. Nothing was built from it.

**Stage A — on the training matches only, D22 Stage A's method.**
- *Folds.* D20's four match-grouped folds of the 16 training matches, from the original set's prepared sequences (manifest `d3891c24…`). Each fold's network trains on the other three folds.
- *Network.* Original set, seed 42, boundary weight 0.5, width 96, 5,400 steps, last snapshot. It is decoded alone (no trees, no second set, no phase averaging) on its held-out training halves, exactly as D22 Stage A decoded D20's fold networks.
- *Anchor.* D20's fold networks at width 96 and 5,400 steps: D15's values, D22's anchor. They scored 0.71293 at cost 8 and 0.71409 at cost 10, their best.
- *Pipeline check, before any candidate is trained.* The D34 trainer at its defaults, on fold 0's training folder, must reproduce D20's fold-0 checkpoint bit for bit (`ac3620e0…`). Otherwise D34 stops and only the trainer may be fixed.
- *Grid A1 (7 settings), one change from D15's values at a time:* learning rate × 2, learning rate × 0.5, batch 32, batch 64, weight decay 0.1, dropout 0.2, dropout 0. The batch splits equally between DFL and SkillCorner, as now.
- *Grid A2 (at most 1 setting):* if two or more of the four hyperparameters have a level whose best primary over the costs beats the anchor's best (0.71409), one more setting combines each such hyperparameter's best level.
- *Costs.* Every setting is decoded at ownership cost 8, 10 and 12, with D15's cutoff, stoppage cost and discount.
- *Primary.* Standard combined N = 3 spellF1 over the 16 training matches, each scored once by the fold network that did not train on it (DFL 1/3 clean + 2/3 masked per match, then the mean over the four; the mean over the 12 SkillCorner matches; then the mean of the two). This is D22 Stage A's primary.
- *Selection.* The point with the highest primary, then N = 3 macroF1, then grid order (A1 in the order above, then A2; costs ascending). Stage B runs only if this point's primary is at least **+0.008** above the anchor's best, 0.71409. Otherwise D34 stops and D15 stays.

**Stage B — development, once.** All four networks (both model sets; seed 42 with boundary weight 0.5 and seed 97 with 1.4; width 96; 5,400 steps; D15's prepared sequences, manifests `d3891c24…` and `2828b0c3…`; MPS; last snapshot; the NumPy export and its check) are retrained at the selected hyperparameters. The trees are D3's. The grid is ownership cost ∈ {8, 10, 12}, discount 0.8. A point is eligible under D33's conditions against D15: primary +0.01; N = 1.75, 4 and 4.5 each ≥ −0.01; dead ball carried and team swap each ≥ −0.01; the PFF veto (≥ −0.01); the Metrica veto (≥ 0.77004); the D24 veto (≥ −0.01). The eligible point with the highest primary is selected, then macroF1, then cost order. None → D15 stays.

**Confirmation, once,** exactly as D33's would have run: the reused DFL/SkillCorner holdout and PFF confirmation set B (parsed only for this step), with D15 first reproducing its holdout record. It passes with a holdout primary gain of at least **+0.01** (C3), the conventions and guards each ≥ −0.01, and set B ≥ −0.01.

**If it passes,** the new networks and cost replace D15's inside the D27 build behind a switch (with the switch off the build still gives `0ddbb342…`), followed by the rebuild checks, a re-declared C2, and the records. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/hyper_network.py` (the trainer; `temporal_network.py` stays byte-identical) and `dfl_submission/hyper_trial.py` (`check`, `stage-a`, `networks`, `develop`, `confirm`, `build`). Weights go to `dfl_submission/hyper_models/`, records to `dfl_submission/hyper_recipe/`, each written once. Every D34 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

## D35 — The tree share, re-selected for D15's networks (2026-11-12, 14:24 EST, written before any D35 member output is computed or anything is scored)

**Question.** In each model set the trees' ownership and stoppage outputs are mixed with the seed-42 network's at 0.5 (`owner_mix`, `dead_mix`), and that mix with the phase-averaged seed-97 network at 0.5, so the trees carry a quarter of each output. The 0.5 was chosen in the original ensemble, when the networks were 48 wide and trained for 1,800 steps (section 4 of the technical document); D15 replaced the networks without re-selecting it. Does another tree share, with a re-selected ownership cost, raise spellF1? Nothing is trained.

**What was seen before this rule.** The records of D1–D34 as written so far (D34's Stage A is training; none of its points has been scored).

**Members.** For every half of the development matches (DFL WPY in its four views, the four SkillCorner development matches, the eight PFF development matches), the three Metrica games and the D24 views, the outputs of every member of both model sets (trees, seed-42 network, phase-averaged seed-97 network) are computed with D15's models and mixed across the two sets at D3's 0.5, leaf by leaf, exactly as the build mixes them (`mix_trial members`).

**Pipeline check, before anything is scored.** Re-blended at owner_mix = dead_mix = 0.5, the members must reproduce D15's cached decoder inputs exactly on every development half, every Metrica half and every D24 view. D15 must also reproduce its development, Metrica and D24 records. Otherwise D35 stops and only the pipeline may be fixed.

**Grid (48 points).** owner_mix ∈ {0.25, 0.5, 0.75, 1} × dead_mix ∈ {0.25, 0.5, 0.75, 1} × ownership cost ∈ {6, 8, 10}. owner_mix 1 drops the trees from ownership (their share 0.375, 0.25, 0.125, 0 in that order); dead_mix likewise for the stoppage output. The second-level mixes, the set weight, the boundary signals, cutoff 0.53, stoppage cost 2 and discount 0.8 stay. (0.5, 0.5, 8) is D15.

**Primary, gates and selection: D33's.** Standard combined N = 3 spellF1 on DFL WPY and the four SkillCorner development matches. A point is eligible against D15 with: primary +0.01; N = 1.75, 4 and 4.5 each ≥ −0.01; dead ball carried and team swap each ≥ −0.01; the PFF veto (≥ −0.01); the Metrica veto (≥ 0.77004); the D24 veto (≥ −0.01). The eligible point with the highest primary is selected, then macroF1, then grid order. None → D15 stays.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and PFF confirmation set C (set B stays with D34), parsed only for this step. D15 must first reproduce its holdout record, and the members at D15's mixes must reproduce D15's holdout inputs. It passes with a holdout primary gain of at least **+0.01** (C3), the conventions and guards each ≥ −0.01, and set C ≥ −0.01.

**If it passes,** the selected mixes and cost replace D15's inside the D27 build behind a switch (with the switch off the build still gives `0ddbb342…`), followed by the rebuild checks, a re-declared C2 and the records. If D34 is also confirmed, the two are not combined without a new rule. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/mix_trial.py` (`members`, `develop`, `confirm`, `build`). Records go to `dfl_submission/mix_recipe/`, each written once. Every D35 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 14:31 EST).** No point was eligible, so **D15 stays** and nothing was confirmed. PFF confirmation set C was neither parsed nor scored.
- **Pipeline checks passed.** Re-blended at D15's mixes, the members reproduced D15's cached inputs exactly on every development half, Metrica half and D24 view, and D15 reproduced its three records.
- **Development, compared with D15** (N = 3 standard 0.72280). The best point, owner_mix 1 (no trees in ownership), dead_mix 0.25, cost 10, gained +0.0030 (PFF −0.0008, Metrica +0.0036, D24 +0.0011), against the +0.01 required. The best point at each owner_mix: 0.25 −0.0017, 0.5 0.0000 (D15), 0.75 +0.0014, 1 +0.0030. All 48 rows are in the record.

What it shows: the trees' quarter share is close to the best mix for D15's networks; dropping them from ownership changes spellF1 by less than D15's seed spread. The ensemble's weights are not where the loss is. [Development](dfl_submission/mix_recipe/development.json)

**Outcome (2026-11-12, 15:23 EST).** Stage A stopped D34: the best point was +0.0075 over the anchor's best, under the +0.008 bar, so **D15 stays**. Stage B, development and confirmation did not run; PFF set B stays unparsed.
- **Pipeline checks passed.** The D34 trainer at its defaults reproduced D20's fold-0 checkpoint bit for bit (`hyper_recipe/check.json`), and D20's fold networks reproduced D22's Stage A anchor exactly (0.71293, 0.71409 and 0.71363 at costs 8, 10 and 12).
- **Stage A, standard combined N = 3 spellF1 over the 16 training matches, each network decoded alone:**

| Setting | Cost 8 | Cost 10 | Cost 12 | Best − anchor's best |
|---|---:|---:|---:|---:|
| Anchor (D15's values) | 0.71293 | **0.71409** | 0.71363 | — |
| Learning rate × 2 | 0.71328 | 0.71482 | 0.71510 | +0.0010 |
| Learning rate × 0.5 | 0.71141 | 0.71355 | 0.71477 | +0.0007 |
| Batch 32 | 0.71418 | 0.71816 | 0.71860 | +0.0045 |
| Batch 64 | 0.70918 | 0.71239 | 0.71635 | +0.0023 |
| Weight decay 0.1 | 0.71038 | 0.71246 | 0.71191 | −0.0016 |
| Dropout 0.2 | 0.71132 | 0.71296 | 0.71663 | +0.0025 |
| Dropout 0 | 0.71180 | 0.71343 | 0.71455 | +0.0005 |
| **A2: learning rate × 2, batch 32, dropout 0.2** | 0.71548 | 0.72034 | **0.72160** | **+0.0075** |

  Three hyperparameters beat the anchor's best (learning rate, batch, dropout), so the rule's A2 combination of their best levels was trained and scored; weight decay did not.

What it shows: no single optimisation value moved the held-out score by more than D15's seed spread; the combination of a higher learning rate, a larger batch and more dropout gained +0.0075, with the best cost at the grid's top (12). The fold runs are kept in `data/d34/cv` and the record in `dfl_submission/hyper_recipe/stage_a.json`. [Stage A](dfl_submission/hyper_recipe/stage_a.json) · [Trainer check](dfl_submission/hyper_recipe/check.json)

## D36 — D34's combined setting, taken to development (2026-11-12, 15:25 EST, written after D34's Stage A and before any D36 network is trained or anything is scored)

**Question.** D34's A2 combination (learning rate × 2, batch 32, dropout 0.2) gained +0.0075 on the held-out training matches with single networks, just under D34's +0.008 pre-filter, with its best cost at the top of the grid. Does it raise spellF1 with the full ensemble on development, and then on a fresh confirmation?

**Disclosure.** This is a second decision on D34's evidence, written after its Stage A results were seen; the setting was chosen on training-match cross-validation, not on development. It is weaker evidence than a step whose setting was fixed before any of its results existed, and the record says so. The development and confirmation bars stay where D33 set them.

**Candidate.** All four networks retrained at learning rate × 2, batch 32 and dropout 0.2, otherwise exactly as D15 (width 96, 5,400 steps, D15's seeds, boundary weights and prepared sequences, MPS, last snapshot, the NumPy export and its check), with `hyper_network`, whose defaults D34 checked bit for bit. The trees are D3's.

**Grid (4 points).** Ownership cost ∈ {8, 10, 12, 14}, discount 0.8, D15's cutoff and stoppage cost.

**Primary, gates and selection: D33's,** against D15 on development: primary +0.01; N = 1.75, 4 and 4.5 each ≥ −0.01; dead ball carried and team swap each ≥ −0.01; the PFF veto (≥ −0.01); the Metrica veto (≥ 0.77004); the D24 veto (≥ −0.01). The eligible point with the highest primary, then macroF1, then cost order. None → D15 stays.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and PFF confirmation set B (D34, whose set it was, never reached confirmation), parsed only for this step, with D15 first reproducing its holdout record. It passes with a holdout primary gain of at least **+0.01** (C3), the conventions and guards each ≥ −0.01, and set B ≥ −0.01.

**If it passes,** the networks and cost replace D15's inside the D27 build behind a switch (with the switch off the build still gives `0ddbb342…`), with the rebuild checks, a re-declared C2 and the records. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/second_trial.py` (`networks`, `develop`, `confirm`, `build`, reusing `hyper_trial`). Weights go to `dfl_submission/second_models/`, records to `dfl_submission/second_recipe/`, each written once. Every D36 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 15:37 EST).** No point was eligible, so **D15 stays** and nothing was confirmed. PFF confirmation set B is still unparsed.
- The four networks trained and passed their NumPy export checks. D15 reproduced its development, Metrica and D24 records.
- **Development, compared with D15** (N = 3 standard 0.72280; DFL clean 0.778, masked 0.723, SkillCorner 0.704, PFF 0.656):

| Cost | N = 3 standard | Change | DFL clean / masked | SkillCorner | PFF | Dead ball carried | Team swap | Metrica | D24 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **8** | **0.72984** | **+0.0070** | 0.780 / 0.725 | 0.717 | 0.655 | +0.0082 | +0.0053 | +0.0072 | +0.0072 |
| 10 | 0.72707 | +0.0043 | 0.778 / 0.720 | 0.715 | 0.652 | +0.0090 | +0.0018 | +0.0084 | +0.0082 |
| 12 | 0.72320 | +0.0004 | 0.767 / 0.719 | 0.711 | 0.650 | +0.0148 | −0.0007 | +0.0089 | +0.0098 |
| 14 | 0.72130 | −0.0015 | 0.773 / 0.708 | 0.713 | 0.650 | +0.0103 | −0.0094 | +0.0064 | +0.0055 |

  The best point, cost 8, gained +0.0070 against the +0.01 required.

What it shows: the combined setting carries over from single networks to the full ensemble at about the same size (+0.0075 held out, +0.0070 on development), and unlike D33 it gains on every guard and on Metrica and the D24 views with PFF flat. The gain sits almost entirely on SkillCorner (+0.0125); DFL changed little. It is still under the bar. The networks are kept in `dfl_submission/second_models/`. [Development](dfl_submission/second_recipe/development.json)

## D37 — D36's training with spell weighting and touch inputs (2026-11-12, 15:38 EST, written after D36's development and before any D37 network is trained or anything is scored)

**Question.** D36's training values gained mostly on SkillCorner (+0.0125) and D33's spell weighting mostly on DFL masked (+0.019 at α = 1, with PFF −0.017); D16's touch features were never given to the networks. Do D36's networks gain more with a milder spell weight (α = 0.5) or with the touch features?

**Disclosure.** Written after D33's, D34's, D35's and D36's results were seen; every element was chosen from them, and the development matches have been used by every step. This is the weakest kind of evidence in this record, and only the confirmation on fresh PFF matches can carry it.

**Candidates.** All four networks retrained with D36's training values (learning rate × 2, batch 32, dropout 0.2; width 96; 5,400 steps; D15's seeds and boundary weights; MPS; last snapshot; NumPy export and its check), in three variants:
- *spell:* D33's spell weight at α = 0.5 with boundary weight 0.2, on D15's prepared sequences;
- *touch:* α = 0, on D16's extended prepared sequences (the 162 columns plus D16's 22 touch columns, manifests checked against D16's);
- *touch + spell:* both.
The fourth corner (α = 0, 162 columns) is D36's networks, reused. The trees are D3's. Inference gives each model exactly the columns it was trained on (`touch_trial.half_signals`).

**Pipeline check, before any candidate is trained.** `hyper_network` with the new spell options at α = 0 and boundary weight 0.2 keeps D34's default path, so it must reproduce D20's fold-0 checkpoint bit for bit again.

**Grid (8 points).** The four variants × ownership cost ∈ {8, 10}, discount 0.8.

**Primary, gates and selection: D33's,** against D15 on development (primary +0.01; conventions and guards ≥ −0.01; PFF, Metrica and D24 vetoes). The eligible point with the highest primary, then macroF1, then grid order.

**Confirmation, once,** on the reused holdout and PFF confirmation set B (parsed only for this step), with D15 first reproducing its holdout record: holdout primary at least **+0.01** (C3), conventions and guards ≥ −0.01, set B ≥ −0.01.

**If it passes,** the networks and cost replace D15's inside the D27 build behind a switch, with the rebuild checks, a re-declared C2 and the records. **If it fails,** `0ddbb342…` stays, and round 9 ends.

Code: spell options in `dfl_submission/hyper_network.py` (`--alpha`, `--edge`), and `dfl_submission/third_trial.py` (`check`, `networks`, `develop`, `confirm`, `build`). Weights go to `dfl_submission/third_models/`, records to `dfl_submission/third_recipe/`, each written once. Every D37 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 16:19 EST).** No point was eligible, so **D15 stays**, nothing was confirmed and round 9 ends. PFF confirmation set B is still unparsed.
- **Pipeline checks passed.** `hyper_network` with the spell options at their defaults reproduced D20's fold-0 checkpoint bit for bit (`third_recipe/check.json`); every new network passed its NumPy export check; D15 reproduced its development, Metrica and D24 records.
- **Development, compared with D15** (N = 3 standard 0.72280), best cost per variant:

| Variant | Cost | N = 3 standard | Change | N = 1.75 / 4 / 4.5 | Dead ball carried | Team swap | PFF | Metrica | D24 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| plain (D36's networks) | 8 | 0.72984 | +0.0070 | +0.009 / +0.001 / +0.000 | +0.0082 | +0.0053 | −0.0014 | +0.0072 | +0.0072 |
| spell (α = 0.5) | 10 | 0.72963 | +0.0068 | +0.010 / +0.003 / +0.000 | +0.0150 | +0.0103 | −0.0019 | +0.0139 | +0.0057 |
| touch | 8 | 0.72561 | +0.0028 | +0.011 / −0.004 / −0.004 | +0.0097 | +0.0071 | −0.0096 | +0.0105 | +0.0071 |
| touch + spell | 8 | 0.72623 | +0.0034 | +0.017 / −0.004 / −0.008 | +0.0137 | +0.0108 | −0.0065 | +0.0174 | +0.0039 |

What it shows: neither addition improved on D36's networks. The milder spell weight kept D36's primary and raised the guards and Metrica without D33's PFF loss; the touch inputs raised N = 1.75 and Metrica but lost on PFF and at N = 4 and 4.5, as D16 found with the trees. The networks are kept in `dfl_submission/third_models/`. [Development](dfl_submission/third_recipe/development.json) · [Trainer check](dfl_submission/third_recipe/check.json)

## Note — Round 9 ends (2026-11-12, 16:19 EST)

Round 9 (D33–D37) ends with the submission unchanged: `0ddbb342…` (D15, then D27). No step reached +0.01 on development, so no confirmation ran and no PFF set was parsed. Every candidate stayed within +0.008 of D15 on development, inside the spread of single networks across seeds (about ±0.007, D23), whichever lever it moved: the loss weights (D33), the optimisation values (D34, D36), the ensemble weights (D35) and the inputs (D37). PFF confirmation sets B and C, the untouched PFF set and the four 2019–20 SkillCorner confirmation matches stay unused. The build path did not change, so no release check was needed: a rebuild with `kickoff build` after round 9 gave `0ddbb342…` with the validator passing (only the validator line and the SHA-256 were read). C1 and C2 stand as recorded. Tests: 382 of 382 (46 files).

## Note — Round 10 (2026-11-12, 16:25 EST)

The goal set for this session ("get as much of an improvement in the spellF1 as you can") stands after round 9, so development continues under the same rules: C3's +0.01 on a fresh confirmation, D33's development gates, a switch in the build, and the hard freeze on Sunday 15 November (EST). PFF stays measurement only.

## D38 — Beyond D34's combination: longer, more regularised, larger (2026-11-12, 16:25 EST, written before any D38 network is trained or anything is scored)

**Question.** Training scale is the only lever that has moved spellF1 by more than seed noise (D15: +0.04). D34's combination (learning rate × 2, batch 32, dropout 0.2) added +0.0075 on held-out training matches and +0.0070 on development (D36), gaining on every guard, with its best cross-validated cost at the top of the grid. Does going further in that direction (longer training, more dropout, a larger batch, a higher learning rate or a wider network) gain more, and does the result clear development and confirmation, alone or without the trees in the ownership blend (D35's best mix)?

**Disclosure.** Written after round 9's results, which chose the direction; development has been used by every step.

**Stage A — on the training matches only, D34's method.**
- Folds, network slot (original set, seed 42, boundary weight 0.5), decoding alone on held-out training halves and the primary are D34's.
- *Anchor.* D34's A2 fold networks (learning rate × 2, batch 32, dropout 0.2, width 96, 5,400 steps), which must first reproduce their D34 Stage A scores (0.71548, 0.72034 and 0.72160 at costs 8, 10 and 12).
- *Grid B1 (5 settings), one change from the anchor at a time:* 10,800 steps (the schedule stretched); dropout 0.3; batch 64; learning rate × 3; width 128.
- *Costs.* 10, 12, 14 and 16 for the anchor and every setting.
- *Selection.* The point with the highest primary, then N = 3 macroF1, then grid order (anchor first, then B1 in the order above; costs ascending). Stage B runs only if the selected point is a B1 setting whose primary is at least **+0.003** above the anchor's best over the same costs. Otherwise D38 stops (D36 already scored the anchor's full models).

**Pipeline check.** `hyper_network` already reproduced D20's fold-0 network at its defaults (D34, D37). The width and length options are `temporal_network`'s, checked in D15.

**Stage B — development, once.** All four networks retrained at the selected setting (both model sets, D15's seeds, boundary weights and prepared sequences, MPS, last snapshot, the NumPy export and its check); the trees are D3's. Member outputs are cached and re-blended (D35's method) so that the grid can include the tree share. Grid (8 points): owner_mix ∈ {0.5 (D15's), 1 (no trees in ownership)} × ownership cost ∈ {6, 8, 10, 12}; dead_mix 0.5, discount 0.8. D33's gates against D15 (primary +0.01; conventions and guards ≥ −0.01; PFF, Metrica and D24 vetoes); the eligible point with the highest primary is selected, then macroF1, then grid order.

**Confirmation, once,** on the reused holdout and PFF confirmation set B (parsed only for this step), with D15 first reproducing its holdout record: holdout primary at least **+0.01** (C3), conventions and guards ≥ −0.01, set B ≥ −0.01.

**If it passes,** the networks, mix and cost replace D15's inside the D27 build behind a switch, with the rebuild checks, a re-declared C2 and the records. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/fourth_trial.py` (`stage-a`, `networks`, `develop`, `confirm`, `build`), reusing `hyper_network`, `hyper_trial` and `mix_trial`. Weights go to `dfl_submission/fourth_models/`, records to `dfl_submission/fourth_recipe/`, each written once. Every D38 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

## D39 — Pressure and support around the ball (2026-11-12, 16:29 EST, written before any pressure column is computed on external data, any D39 network is trained or anything is scored)

**Question.** The 162 columns describe each team's nearest player to the ball (distance, contact strength, its speed relative to the ball) and whole-team shape, but nothing about the other players around the ball: how many of each team are close, how far the second player is, whether they are closing in, or whether the nearest player moves with the ball. Do such columns, given to the networks, raise spellF1?

**What was seen before this rule.** Round 9 and D38's rule (D38's Stage A is training; nothing of it is scored). The columns below were tested only on synthetic tracks (`tests/test_pressure.py`).

**New columns** (`dfl_submission/pressure_features.py`, 12 columns, an `h__`/`a__` pair each; computed at 25 Hz and sampled at the long-context frames; all missing where the ball is unseen at the frame, and the motion columns also where it was unseen five frames earlier; thresholds hand-set, not tuned):
- the distance from the ball to the team's second-nearest visible player;
- the number of the team's visible players within 5 m and within 10 m of the ball;
- the mean speed at which the team's three nearest players close in on the ball over the last 5 frames (m/s, negative when receding; displacements above 12 m/s ignored);
- the speed of the team's nearest player over the last 5 frames;
- the cosine between that player's velocity and the ball's over the last 5 frames (missing if either is under 0.5 m/s).

**Candidate.** D15's prepared training sequences of both sets, extended with the 12 columns (174 in all). All four networks retrained on them with D36's training values (learning rate × 2, batch 32, dropout 0.2; width 96; 5,400 steps; D15's seeds and boundary weights; MPS; last snapshot; the NumPy export and its check). The trees are D3's. Inference gives each model exactly the columns it was trained on.

**Pipeline check, before any network is trained.** For every prepared training half of both sets, the 162 re-extracted columns must equal the stored sequence exactly (D16's check). Otherwise D39 stops and only the pipeline may be fixed.

**Grid (2 points).** Ownership cost ∈ {8, 10}, discount 0.8, D15's mixes.

**Primary, gates and selection: D33's,** against D15 on development (primary +0.01; conventions and guards ≥ −0.01; PFF, Metrica and D24 vetoes).

**Confirmation, once,** on the reused holdout and PFF confirmation set C (set B is D38's), parsed only for this step, with D15 first reproducing its holdout record: holdout primary at least **+0.01** (C3), conventions and guards ≥ −0.01, set C ≥ −0.01.

**If it passes,** the columns, networks and cost replace D15's inside the D27 build behind a switch, with the rebuild checks, a re-declared C2 and the records. If D38 also passes, the two are not combined without a new rule. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/pressure_features.py` and `dfl_submission/pressure_trial.py` (`prepare`, `networks`, `develop`, `confirm`, `build`). Sequences go to `data/d39/`, weights to `dfl_submission/pressure_models/`, records to `dfl_submission/pressure_recipe/`, each written once. Every D39 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 16:52 EST).** No point was eligible, so **D15 stays** and nothing was confirmed. PFF confirmation set C is still unparsed.
- **Pipeline checks passed.** All 96 prepared training halves (48 per set) re-extracted their 162 columns exactly before the pressure columns were appended; the four networks passed their NumPy export checks; D15 reproduced its development, Metrica and D24 records.
- **Development, compared with D15** (N = 3 standard 0.72280; DFL clean 0.778, masked 0.723, SkillCorner 0.704, PFF 0.656):

| Cost | N = 3 standard | Change | DFL clean / masked | SkillCorner | PFF | N = 1.75 / 4 / 4.5 | Dead ball carried | Team swap | Metrica | D24 |
|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 8 | 0.72401 | +0.0012 | 0.777 / 0.711 | 0.715 | 0.654 | +0.006 / −0.002 / −0.003 | +0.0131 | +0.0155 | +0.0083 | +0.0141 |
| 10 | 0.72304 | +0.0002 | 0.780 / 0.705 | 0.716 | 0.655 | +0.001 / −0.002 / −0.003 | +0.0079 | +0.0086 | +0.0129 | +0.0171 |

What it shows: with D36's training values, the pressure columns kept SkillCorner's gain but lost it on DFL masked (0.725 → 0.711 against D36), so the primary fell from D36's +0.0070 to +0.0012. They raised both guard views and the D24 views more than any earlier step (+0.014 to +0.017), but the primary is the gate. The networks are kept in `dfl_submission/pressure_models/`. [Development](dfl_submission/pressure_recipe/development.json)

## D40 — D36's networks with the tree share re-selected (2026-11-12, 17:14 EST, written after D39's outcome, before any D40 member output is computed or anything is scored)

**Question.** D36's networks gained +0.0070 at D15's mixes, and dropping the trees from ownership gained +0.0030 with D15's networks (D35). Does re-selecting the tree share for D36's networks clear development and confirmation?

**Disclosure.** Both ingredients were chosen from development results already seen (D35, D36); this is a further look at reused development data and the record says so. Only the confirmation on fresh PFF matches can carry it.

**Members and pipeline check.** D36's member outputs on the development halves, the Metrica games and the D24 views are cached (D35's method, `fourth_trial`'s member functions with D36's networks). Re-blended at owner_mix = dead_mix = 0.5 they must reproduce D36's cached decoder inputs exactly (`data/d36/cache`); otherwise D40 stops.

**Grid (9 points).** owner_mix ∈ {0.5, 0.75, 1} × ownership cost ∈ {6, 8, 10}; dead_mix 0.5, discount 0.8.

**Primary, gates and selection: D33's,** against D15 on development (primary +0.01; conventions and guards ≥ −0.01; PFF, Metrica and D24 vetoes).

**Confirmation, once,** on the reused holdout and PFF confirmation set C (parsed only for this step; D39 never reached it), with D15 first reproducing its holdout record: holdout primary at least **+0.01** (C3), conventions and guards ≥ −0.01, set C ≥ −0.01.

**If it passes,** D36's networks with the selected mix and cost replace D15's inside the D27 build behind a switch, with the rebuild checks, a re-declared C2 and the records; it is not combined with D38 without a new rule. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/fifth_trial.py` (`develop`, `confirm`, `build`). Records go to `dfl_submission/fifth_recipe/`, each written once. Every D40 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 17:20 EST).** No point was eligible, so **D15 stays** and nothing was confirmed. PFF confirmation set C is still unparsed.
- **Pipeline checks passed.** Re-blended at D15's mixes, D36's member outputs reproduced D36's cached inputs exactly on every development half, Metrica half and D24 view; D15 reproduced its three records.
- **Development, compared with D15:** the best point stayed D36's own mix, owner_mix 0.5 at cost 8 (+0.0070). The best at owner_mix 0.75 was +0.0052 (cost 10) and at owner_mix 1 +0.0051 (cost 10). All 9 rows are in the record.

What it shows: dropping the trees from ownership does not add to D36's gain; with D36's networks the trees' quarter share is again the best of the grid. [Development](dfl_submission/fifth_recipe/development.json)

**Outcome (2026-11-12, 17:35 EST).** Stage A stopped D38: its best point was +0.0029 over the anchor's best, under the +0.003 bar, so **D15 stays**. Stage B, development and confirmation did not run; PFF set B stays unparsed.
- **Pipeline check passed.** D34's A2 fold networks reproduced their D34 scores (0.71548, 0.72034 and 0.72160 at costs 8, 10 and 12).
- **Stage A, standard combined N = 3 spellF1 over the 16 training matches, each network decoded alone:**

| Setting (one change from the anchor) | Cost 10 | Cost 12 | Cost 14 | Cost 16 | Best − anchor's best |
|---|---:|---:|---:|---:|---:|
| Anchor (learning rate × 2, batch 32, dropout 0.2) | 0.72034 | 0.72160 | **0.72168** | 0.72148 | — |
| 10,800 steps | 0.70991 | 0.71055 | 0.71259 | 0.71274 | −0.0089 |
| **Dropout 0.3** | 0.72160 | 0.72309 | 0.72342 | **0.72458** | **+0.0029** |
| Batch 64 | 0.71141 | 0.71196 | 0.71421 | 0.71510 | −0.0066 |
| Learning rate × 3 | 0.71233 | 0.71679 | 0.71875 | 0.71995 | −0.0017 |
| Width 128 | 0.70945 | 0.71079 | 0.71354 | 0.71566 | −0.0060 |

What it shows: the anchor is close to the best of its neighbourhood. Longer training, a larger batch, a higher learning rate and a wider network all lost; only more dropout gained, by less than the bar, again with its best cost at the top of the grid. The fold runs are kept in `data/d38/cv`. [Stage A](dfl_submission/fourth_recipe/stage_a.json)

## D41 — The best-evidenced candidate to confirmation (2026-11-12, 17:36 EST, written after D38's Stage A, before any D41 network is trained or anything new is scored)

**Question.** Rounds 9 and 10 found one direction with consistent evidence on two independent sets of matches: D34's training values gained +0.0075 on the 16 held-out training matches and +0.0070 on development (D36), gaining on every guard, Metrica and D24 view; more dropout added +0.0029 on the training matches (D38). Every step's own development pre-filter (+0.01) stopped them before C3's actual test, a fresh external confirmation. Does the best of these candidates clear C3's +0.01 on confirmation?

**Disclosure.** Written after every result above was seen. The development gate below (+0.005) is lower than D33–D40's (+0.01) and was set knowing that the candidates sit near +0.007; it decides only whether a confirmation set is spent. C3's bar on confirmation is unchanged, and so is every other gate. Development has been reused by every step.

**Candidates.**
- *D36:* D36's networks (learning rate × 2, batch 32, dropout 0.2) at ownership cost 8, as scored in D36 (+0.0070; not rescored).
- *D41:* all four networks retrained at learning rate × 2, batch 32, dropout 0.3 (width 96, 5,400 steps, D15's seeds, boundary weights and prepared sequences, MPS, last snapshot, the NumPy export and its check), trees D3's, at ownership cost ∈ {8, 10, 12}, discount 0.8, D15's mixes.

**Development, once (D41's three points; D36's row is taken from its record).** A point is eligible against D15 with: primary at least **+0.005**; N = 1.75, 4 and 4.5 each ≥ −0.01; dead ball carried and team swap each ≥ −0.01; the PFF veto (≥ −0.01); the Metrica veto (≥ 0.77004); the D24 veto (≥ −0.01). Of the eligible points, D36's included, the one with the highest primary is selected, then macroF1, then the order D36, then D41 by cost.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and PFF confirmation set B (parsed only for this step), with D15 first reproducing its holdout record. It passes with a holdout primary gain of at least **+0.01** (C3), N = 1.75, 4 and 4.5 each ≥ −0.01, dead ball carried and team swap each ≥ −0.01, and set B ≥ −0.01.

**If it passes,** the selected networks and cost replace D15's inside the D27 build behind a switch (with the switch off the build still gives `0ddbb342…`), followed by two identical macOS rebuilds, `release/check.sh`, a re-declared C2 run once by Derek, and the records and technical document. **If it fails,** `0ddbb342…` stays, set B is spent, and the record says so.

Code: `dfl_submission/sixth_trial.py` (`networks`, `develop`, `confirm`, `build`), reusing `hyper_trial` and `second_trial`. Weights go to `dfl_submission/sixth_models/`, records to `dfl_submission/sixth_recipe/`, each written once. Every D41 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 17:57 EST).** Development selected D36's networks at cost 8; confirmation **failed**, so **D15 stays**. PFF confirmation set B is now spent.
- **Development.** D36's row (from its record) +0.0070 was eligible under D41's gates. The dropout-0.3 networks scored +0.0027, +0.0035 and −0.0017 at costs 8, 10 and 12 (none eligible). D15 reproduced its development, Metrica and D24 records first, and D36's record was scored against the same D15 reference.
- **Confirmation, once** (D15 first reproduced its holdout record; set B parsed only for this step):

| | D15 | D36's networks, cost 8 | Change | Required |
|---|---:|---:|---:|---|
| N = 3 standard combined, holdout | 0.70874 | 0.71054 | **+0.0018** | ≥ +0.01 |
| DFL WQQ clean / masked | 0.764 / 0.727 | 0.753 / 0.705 | | |
| DFL WR9 clean / masked | 0.759 / 0.702 | 0.777 / 0.704 | | |
| SkillCorner holdout (mean of four) | 0.687 | 0.696 | +0.009 | |
| N = 1.75 / 4 / 4.5 | | | +0.014 / −0.003 / −0.006 | each ≥ −0.01 |
| Dead ball carried / team swap | | | +0.0038 / +0.0199 | each ≥ −0.01 |
| PFF set B (fresh) | 0.655 | 0.659 | +0.0039 | ≥ −0.01 |

What it shows: D36's training values carry a small real gain, positive on SkillCorner, on fresh PFF matches and on both guards, but about a quarter of their development gain on the holdout primary, far under C3's +0.01. The development pre-filters of D33–D40 were right to stop these candidates. [Development](dfl_submission/sixth_recipe/development.json) · [Confirmation](dfl_submission/sixth_recipe/confirmation.json)

## Note — Round 10 ends (2026-11-12, 17:59 EST)

Round 10 (D38–D41) ends with the submission unchanged: `0ddbb342…` (D15, then D27). D38 and D39 stopped at their gates, D40 found nothing to add, and D41 sent the best-evidenced candidate (D36's networks) to one confirmation, which it failed: +0.0018 on the holdout and +0.0039 on PFF set B against C3's +0.01. Across rounds 9 and 10, nine steps moved the loss weights, the optimisation values, the ensemble weights and the inputs; the one candidate measured on fresh matches is at most marginally better than D15. PFF set B is spent; PFF set C, the untouched PFF set and the four 2019–20 SkillCorner confirmation matches remain. The build path did not change (`kickoff build` still gives `0ddbb342…`). Tests: 395 of 395 (50 files).

## Note — Round 11 (2026-11-12, 18:04 EST)

The session goal still stands after round 10. Rounds 9 and 10 showed that re-weighting, re-tuning and hand-made inputs move spellF1 by less than seed noise, and that the best of them is at most marginally better on fresh matches (D41). Round 11 tries the one representational change not yet made: a network that reads the players themselves instead of team summaries. Same rules: C3's +0.01 on a fresh confirmation, D33's development gates, a build switch, the Sunday freeze, PFF measurement only.

## D42 — A player-set network (2026-11-12, 18:04 EST, written before any D42 code runs on external data, any network is trained or anything is scored)

**Question.** Every network input is a hand-made team summary (nearest-player distances, contact strength, team shape and motion). Does a network that also reads every nearby player's position and velocity relative to the ball, through a small learned per-player encoder pooled per team, raise spellF1?

**Disclosure.** Written after rounds 9 and 10; development has been reused by every step.

**Player tokens** (`dfl_submission/player_set.py`), at the long-context frames (every fifth frame). The reference point is the ball when it is seen at the frame, otherwise the mean position of all visible players. For each team, its visible players nearest the reference point, at most 11, each give eight values: offset from the reference point (x and y, ÷ 10 m), distance (÷ 10 m), velocity over the previous 5 frames (x and y, ÷ 5 m/s; 0 when the player was not visible then or moved faster than 12 m/s), position on the pitch (÷ half-length and half-width) and whether the ball is seen. Missing players are masked. Nothing here is fitted.

**Network.** D15's temporal network (width 96, six blocks), with one addition: a per-player encoder (8 → 32 → 32, GELU) whose outputs are pooled per team (masked mean and max), concatenated (home, away; 128 values), projected to the network width and added to the input projection. Swapping the teams swaps the two pools and the feature columns together, as the trainer's swap augmentation and the swap-averaged inference already do. Inference is NumPy only; the export must match PyTorch within D15's tolerance (2e-5 absolute plus 1e-5 relative).

**Training.** All four networks (both model sets; D15's seeds and boundary weights; D15's prepared sequences plus the tokens; D36's training values: learning rate × 2, batch 32, dropout 0.2; 5,400 steps; MPS; last snapshot). The trees are D3's.

**Pipeline checks, before any network is trained.** For every prepared training half of both sets, the 162 re-extracted columns must equal the stored sequence exactly, and the token frames must be the stored frames. Otherwise D42 stops and only the pipeline may be fixed.

**Grid (2 points).** Ownership cost ∈ {8, 10}, discount 0.8, D15's mixes.

**Primary, gates and selection: D33's,** against D15 on development (primary +0.01; conventions and guards ≥ −0.01; PFF, Metrica and D24 vetoes).

**Confirmation, once,** on the reused holdout and PFF confirmation set C (parsed only for this step), with D15 first reproducing its holdout record: holdout primary at least **+0.01** (C3), conventions and guards ≥ −0.01, set C ≥ −0.01.

**If it passes,** the player-set networks and cost replace D15's inside the D27 build behind a switch, with the rebuild checks, `release/check.sh`, a re-declared C2 and the records. **If it fails,** `0ddbb342…` stays.

Code: `dfl_submission/player_set.py` (tokens, network, trainer, export, NumPy inference) and `dfl_submission/player_trial.py` (`prepare`, `networks`, `develop`, `confirm`, `build`). Tokens go to `data/d42/`, weights to `dfl_submission/player_models/`, records to `dfl_submission/player_recipe/`, each written once. Every D42 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 18:35 EST).** No point was eligible, so **D15 stays** and nothing was confirmed under D42's own rule. PFF set C is still unparsed.
- **Pipeline checks passed.** All 96 prepared training halves re-extracted their 162 columns and frames exactly before their tokens were written; every network's NumPy export matched PyTorch (largest logit difference 7e-6); D15 reproduced its development, Metrica and D24 records.
- **Development, compared with D15** (DFL clean 0.778, masked 0.723, dead ball carried 0.608, team swap 0.659, SkillCorner 0.704, PFF 0.656, D24 mean 0.674):

| Cost | N = 3 standard | Change | DFL clean / masked | Dead ball carried | Team swap | SkillCorner | PFF | Metrica | D24 mean |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | 0.72859 | +0.0058 | 0.771 / 0.718 | 0.644 (+0.036) | 0.695 (+0.036) | 0.721 | 0.664 (+0.008) | +0.0034 | 0.703 (+0.029) |
| 10 | 0.72952 | +0.0067 | 0.780 / 0.722 | 0.643 (+0.034) | 0.694 (+0.035) | 0.718 | 0.663 (+0.007) | +0.0109 | 0.701 (+0.027) |

What it shows: reading the players directly is the first change that gains on every degraded or unseen view at once: both guards by about +0.035, PFF (a vendor no model trained on) by +0.007, Metrica under competition-like tracking by +0.027, SkillCorner by +0.014, with DFL WPY flat. The primary, which weighs DFL WPY as half, rose +0.0067, under the +0.01 gate. [Development](dfl_submission/player_recipe/development.json)

## D43 — The player-set network to confirmation (2026-11-12, 18:36 EST, written after D42's development, before anything new is scored)

**Question.** Does D42's player-set model clear C3's +0.01 on a fresh confirmation?

**Disclosure.** Written after D42's development results were seen, as D41 was after D36's. The development gate is lowered to +0.005, knowing D42 scored +0.0067; it decides only whether a confirmation set is spent. C3's bar on confirmation and every other gate are unchanged.

**Candidate.** D42's four networks with D3's trees at ownership cost 10 (D42's best development point; +0.0067, every other D33 gate passed). Nothing is retrained.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and PFF confirmation set C (parsed only for this step), with D15 first reproducing its holdout record. It passes with a holdout primary gain of at least **+0.01** (C3), N = 1.75, 4 and 4.5 each ≥ −0.01, dead ball carried and team swap each ≥ −0.01, and set C ≥ −0.01.

**If it passes,** the player-set networks at cost 10 replace D15's inside the D27 build behind a switch (with the switch off the build still gives `0ddbb342…`), followed by two identical macOS rebuilds, `release/check.sh`, a re-declared C2 run once by Derek, and the records and technical document. **If it fails,** `0ddbb342…` stays and set C is spent; the untouched PFF set remains.

Code: `player_trial confirm` and `player_trial build`, with the selection read from a D43 development record that takes D42's cost-10 row under the +0.005 gate (`dfl_submission/player_recipe/development_d43.json`).

**Outcome (2026-11-12, 18:45 EST).** **Passed.** The player-set model at ownership cost 10 replaces D15 in the submission, after the adoption steps below. PFF set C is spent; the untouched PFF set and the four 2019–20 SkillCorner confirmation matches remain.
- **Selection.** D42's cost-10 row (+0.0067) passed D43's gates (`player_recipe/development_d43.json`).
- **Confirmation, once** (D15 first reproduced its holdout record; set C parsed only for this step; `player_recipe/confirmation.json`):

| | D15 | Player-set model, cost 10 | Change | Required |
|---|---:|---:|---:|---|
| **N = 3 standard combined, holdout** | 0.70874 | **0.72761** | **+0.0189** | ≥ +0.01 |
| DFL WQQ clean / masked | 0.764 / 0.727 | 0.766 / 0.745 | +0.002 / +0.019 | |
| DFL WR9 clean / masked | 0.759 / 0.702 | 0.788 / 0.725 | +0.028 / +0.023 | |
| SkillCorner holdout (mean of four) | 0.687 | 0.706 | +0.019 | |
| N = 1.75 / 4 / 4.5 | | | +0.027 / +0.015 / +0.008 | each ≥ −0.01 |
| Dead ball carried / team swap | | | +0.033 / +0.061 | each ≥ −0.01 |
| PFF set C (eight fresh matches) | 0.711 | 0.721 | +0.0108 | ≥ −0.01 |
| Raw provider labels, standard combined (reported) | 0.618 | 0.640 | +0.022 | |

What it shows: the player-set model's development profile carried over and grew on the holdout: every DFL view, SkillCorner, the fresh PFF matches, raw labels and both guards gained, and the primary cleared C3's bar with room. It is the first step since D15 to do so. Development had understated it because DFL WPY, half of the development primary, was flat.

**Adoption (D43's rule, C3).** The build switch is the build command: `player_trial build` writes the new file; `kickoff build`, unchanged, still writes `0ddbb342…`. Then: two identical macOS rebuilds, `release/check.sh`, C2 re-declared with ranges recomputed from the new model's external predictions and run once by Derek, C1's record amended as C3 requires, and the README, CONSTANTS.md, technical document and SHA files updated.

*Adoption checks (2026-11-12, 18:56 EST).* Two builds with `player_trial build` in the repository venv gave `a56574fefd83af81747206e3aedcaf625cdd170e7376d4e00e5f0abee7c431e3` with the validator passing; `release/BUILD_COMMAND` and `SUBMISSION_SHA256.txt` now name it, and `data/submission.csv` and `Anfield.csv` hold it (the validator printed OK; only the OK line and the SHA-256 were read). `release/check.sh` clones `HEAD`, and round 9–11 is not yet committed, so its builds were first run on a copy of the working tree (its tracked and untracked, non-ignored files) with the same steps: two fresh macOS venvs from Homebrew `python3.12` and `dfl_submission/requirements.txt`, then the pinned `python:3.12.14-slim` image under `linux/arm64` and `linux/amd64`. All four gave `a56574fe…` and printed the validator's OK line. Tests: 398 of 398 (51 files). `release/check.sh` then ran on the commit `6d979c6` (rounds 9–11 committed with Derek's approval): two fresh macOS builds and the pinned image under `linux/arm64` and `linux/amd64` all gave `a56574fe…` with the validator's OK line, and 398 of 398 tests passed in the clone (only the validator lines and the SHA-256s were read). Still to run: C2 on the new file (Derek, once, outside the AI session).

## D44 — Player-set variants by cross-validation (2026-11-12, 19:00 EST, written before any D44 code runs on external data, any network is trained or anything is scored)

**Question.** D42/D43 tested one player-set design with one training setting. Does a variant (more players, a wider encoder, the players' motion relative to the ball, or D15's training values) score clearly higher, enough to justify the last fresh PFF set?

**Reference.** The standing submission, D43 (D42's player-set networks at ownership cost 10).

**Stage A — on the training matches only, D34's method.** D20's four folds of the 16 training matches, the original set's prepared sequences with player tokens, one network per fold (seed 42, boundary weight 0.5, width 96, 5,400 steps), decoded alone on its held-out training halves; primary is D22 Stage A's (standard combined N = 3 spellF1 over the 16 matches).
- *Anchor:* D42's design and training values (11 players per team, eight values each, encoder 32, learning rate × 2, batch 32, dropout 0.2), trained in this step.
- *Variants, one change each:* V1, 16 players per team; V2, encoder width 64; V3, two more values per player, its velocity relative to the ball's over the last 5 frames (x and y, ÷ 5 m/s; 0 where the ball was unseen at either frame); V4, D15's training values (learning rate × 1, batch 16, dropout 0.08).
- *Costs:* 8, 10 and 12 for every setting.
- *Selection.* The point with the highest primary, then macroF1, then order (anchor, V1–V4; costs ascending). Stage B runs only if it is a variant whose primary is at least **+0.005** above the anchor's best.

**Stage B — development, once** (only if Stage A selects a variant). All four networks retrained at the selected variant (both sets, D15's seeds and boundary weights), D3's trees, ownership cost ∈ {8, 10, 12}. Development is scored against D43 (D42's cached development inputs at cost 10, which must reproduce D42's record). Eligible with: primary at least +0.005 over D43 (the gate D43 used, disclosed then); N = 1.75, 4 and 4.5 each ≥ −0.01; dead ball carried and team swap ≥ −0.01; PFF ≥ −0.01; Metrica mean N = 3 at least D43's minus 0.01; D24 mean at least D43's minus 0.01.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and the untouched PFF set (parsed only for this step; the last fresh PFF matches), against D43, which must first reproduce its holdout record. It passes with a holdout primary gain of at least **+0.01** over D43 (C3), conventions and guards ≥ −0.01, and the untouched set ≥ −0.01.

**If it passes,** the variant replaces D43 in the build behind a switch (with the old command the build still gives `a56574fe…`), with rebuild checks, `release/check.sh`, C2 re-declared and the records. **If it fails or stops,** `a56574fe…` stays.

Code: `dfl_submission/player_set2.py` (configurable tokens, network, trainer and NumPy inference; `player_set.py`, on the build path, stays byte-identical) and `dfl_submission/player2_trial.py`. Tokens go to `data/d44/`, records to `dfl_submission/player2_recipe/`, each written once. Every D44 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**Outcome (2026-11-12, 20:48 EST).** Stage A stopped D44: D42's own design scored best, so **D43 stays** (`a56574fe…`). Stage B, development and confirmation did not run; the untouched PFF set stays unparsed.
- **Stage A, standard combined N = 3 spellF1 over the 16 training matches, each fold network decoded alone:**

| Setting | Cost 8 | Cost 10 | Cost 12 | Best − anchor's best |
|---|---:|---:|---:|---:|
| **Anchor (D42: 11 players, encoder 32, learning rate × 2, batch 32, dropout 0.2)** | 0.71809 | **0.72041** | 0.71894 | — |
| V1, 16 players per team | 0.71802 | 0.72008 | 0.71880 | −0.0003 |
| V2, encoder width 64 | 0.70783 | 0.71201 | 0.71081 | −0.0084 |
| V3, relative velocity added | 0.70914 | 0.71174 | 0.71110 | −0.0087 |
| V4, D15's training values | 0.71365 | 0.71686 | 0.71729 | −0.0031 |

What it shows: D42's design is the best of its neighbourhood, and its best cost on the held-out training matches is the cost D43 uses (10). More players changed nothing; a wider encoder and the relative velocity lost, as more inputs and capacity did elsewhere. On these matches a single player-set network scores about the same as a single network with D34's values alone (D34's A2, 0.72160), so the player-set model's confirmed gain (D43) lies on the views these matches do not contain: degraded and swapped tracking, the unseen vendor, and the holdout. The fold runs are kept in `data/d44/cv`. [Stage A](dfl_submission/player2_recipe/stage_a.json)

## D45 — The player-set model refit on the training and development matches (2026-11-12, 20:55 EST, written before any D45 sequence is prepared, any network is trained or anything is scored)

**Question.** D43's player-set networks read raw players and so have more to learn than the hand-made features; they trained on 16 matches. The five development matches (DFL WPY and the four SkillCorner development matches) are labelled, external and no longer needed for selection. Does refitting the four networks on all 21 raise spellF1 on matches neither model trained on?

**Disclosure.** Written after D42–D44. Development cannot score this step (it trains on those matches), so it is judged on vetoes and one confirmation only. D23 asked the same question for D15's networks and stopped at its evidence step; this is the player-set model's version.

**Candidate.** A `train_dev` split (DFL fitting matches plus WPY; SkillCorner training plus development matches), with every match given the training views its set uses (`external_benchmark.halves` for the original set, `augment.training_halves` for the augmented set; each match keeps its seeds). Both sets' prepared sequences and player tokens are built for it, and all four networks are retrained exactly as D42 trained them (seeds 42 and 97, boundary weights 0.5 and 1.4, D36's training values, width 96, 5,400 steps, MPS, last snapshot, the export check). The trees are D3's; the ownership cost is D43's (10). Nothing else changes.

**Pipeline check, before any network is trained.** Every prepared half of the 16 training matches must equal the stored training sequence of its set exactly (values and targets). Otherwise D45 stops and only the pipeline may be fixed.

**Vetoes, before confirmation** (against D43 at cost 10): mean N = 3 spellF1 on the eight PFF development matches, the Metrica mean and the D24 mean each drop by no more than 0.01. Nothing is selected.

**Confirmation, once,** on the reused DFL/SkillCorner holdout and the untouched PFF set (the last fresh PFF matches, parsed only for this step), against D43, which must first reproduce its holdout record. It passes with a holdout primary gain of at least **+0.01** over D43 (C3), N = 1.75, 4 and 4.5 each ≥ −0.01, dead ball carried and team swap each ≥ −0.01, and the untouched set ≥ −0.01.

**If it passes,** the refit networks replace D43's in the build behind a switch (the old command still gives `a56574fe…`), with rebuild checks, `release/check.sh`, C2 re-declared and the records. **If it fails,** `a56574fe…` stays, and the untouched set is spent.

Code: the `train_dev` split in `external_benchmark.py` (the existing splits are unchanged) and `dfl_submission/refit_trial.py` (`prepare`, `networks`, `vetoes`, `confirm`, `build`), reusing `player_set` and `player_trial`. Sequences and tokens go to `data/d45/`, weights to `dfl_submission/refit_models/`, records to `dfl_submission/refit_recipe/`, each written once. Every D45 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D45 status (2026-11-12, 21:26 EST): vetoes passed; confirmation held.** Both sets' `train_dev` sequences (62 halves each) reproduced every stored training half exactly (`refit_recipe/prepare_current.json`, `prepare_augmented.json`); the four refit networks trained and passed their export checks. Against D43 at cost 10: PFF development −0.0012, Metrica −0.0062, D24 +0.0014, so the vetoes passed (`refit_recipe/vetoes.json`). The confirmation has **not** run and the untouched PFF set is **not** parsed: at 21:26 the orchestration session Derek started tonight asked this session to hold it until Derek decides, so that the refit can be combined with other levers into one candidate and the last fresh PFF set spent once. D45's networks, sequences and tokens are kept unchanged for that use (`dfl_submission/refit_models/`, `data/d45/`). The submission stays `a56574fe…`.

## Note — Round 12 (2026-11-12, 21:28 EST)

Derek opened a second session for the same goal and made it the decision-maker ("orchestrator"): it writes the rules and the records here, and separate coding sessions build and run what the rules describe. Everything standing stays: nothing is fitted on the provided tracking, PFF is measurement only, C1 is never run again, C3's +0.01 on a confirmation and its Sunday freeze hold, and every command runs with `MTAH_DATA_DIR` set to a nonexistent path.

*What changes is the order of work.* Rounds 9 and 10 took levers to confirmation one at a time, and each landed under +0.01 (D41: +0.0018). Round 12 screens several levers on development with a no-harm rule (D47), stacks the ones that hold into one candidate, and spends a confirmation once on the stack (D48, to be written after D46 and D47). D45 belongs to the first session; the orchestrator asked it at 21:24 EST to finish its trainings and vetoes but hold its confirmation, so that the 21-match refit can join the stack and the untouched PFF set is spent once. Whatever D45 has done by then is recorded under D45.

## D46 — Where D43 loses spellF1, and how much of it lies where the ball is hidden (2026-11-12, 21:28 EST, written before anything is decoded or scored for D46)

**Question.** D21 measured D15. D43 reads the players directly and gained most on degraded views. Is ownership still the main loss, and is that loss on frames where the ball is hidden or where it is seen? The answer picks round 12's next lever: inference without the ball (last-seen-ball tokens, an auxiliary ball-location target) or better modelling where the ball is seen. Measurement only; it selects nothing and changes nothing in the submission.

**Candidate.** Frozen D43 (D42's networks, D3's trees, ownership cost 10) decoded afresh from `data/d42/cache` (`development`, `metrica`, `d24`). Holdout, confirmation, training and untouched matches are not read.

**Pipeline check, before any result is read.** D43 decoded from those caches must reproduce the cost-10 row of `player_recipe/development.json` exactly (PFF, Metrica and D24 included). Otherwise D46 stops without a record.

**What is measured,** per group (DFL WPY clean, masked, dead ball carried, team swap; SkillCorner; PFF; Metrica; the three D24 views) against the N = 3 targets:
1. D21's four measurements, unchanged (spell detection by IoU, the true-stoppage and true-ownership oracles, boundary offsets, recall by spell length).
2. *Ball visibility.* A frame's ball is seen when the view's tracking has a ball position at that frame. Per group: the share of target-live frames with the ball unseen; frame agreement of the decoded owner with the target on frames both call live, split by seen / unseen and by time since the ball was last seen (0, up to 1 s, 1–3 s, over 3 s).
3. *Recall by visibility.* Recall at IoU 0.5 of true spells by the share of the spell's frames with the ball seen (under 20%, 20–80%, 80% or more).
4. *Split oracle.* D11's true-ownership oracle applied only on frames where the ball is unseen, and only where it is seen: mean spellF1 per group for each.

**Record.** `dfl_submission/diagnosis_recipe/d46.json`, written once. Code: a new module `dfl_submission/diagnosis_d43.py` reusing `diagnosis.py` (unchanged).

**How it is read.** If, on the degraded groups (DFL masked, SkillCorner, PFF, D24 views), the unseen-frame oracle recovers at least half of the full ownership oracle's gain, D48's stack includes a ball-free lever; otherwise it does not, and the effort goes to the seen-ball side.

**D46 outcome (2026-11-12, 21:40 EST).** The reproduction check passed (D43's cost-10 development row, PFF, Metrica and D24 included, from `data/d42/cache`). Against the N = 3 targets:

| Group | D43 | True stoppages | True ownership | Live frames with ball unseen | Owner agreement, seen / unseen | Recall at IoU 0.5 by share of spell seen: under 20% / 20–80% / 80%+ (spells) |
|---|---:|---:|---:|---:|---|---|
| DFL clean | 0.780 | 0.780 | 0.921 | 0% | 0.929 / – | – / – / 0.77 (281) |
| DFL masked | 0.722 | 0.729 | 0.913 | 39% | 0.916 / 0.905 | 0.37 (30) / 0.79 (169) / 0.54 (82) |
| DFL dead ball carried | 0.643 | 0.739 | 0.788 | 39% | 0.917 / 0.907 | 0.37 / 0.79 / 0.49 |
| DFL team swap | 0.694 | 0.705 | 0.912 | 39% | 0.903 / 0.890 | 0.33 / 0.78 / 0.50 |
| SkillCorner | 0.718 | 0.766 | 0.842 | 7% | 0.944 / 0.842 | 0.00 (81) / 0.61 (115) / 0.72 (775) |
| PFF | 0.663 | 0.746 | 0.785 | 36% | 0.957 / 0.851 | 0.17 (414) / 0.72 (996) / 0.76 (373) |
| Metrica | 0.791 | 0.830 | 0.846 | 0.3% | 0.957 / 0.565 | – / 0.20 (5) / 0.77 (820) |
| D24 masked | 0.717 | 0.779 | 0.824 | 39% | 0.947 / 0.933 | 0.23 (87) / 0.79 (512) / 0.54 (226) |
| D24 handled | 0.725 | 0.780 | 0.831 | 39% | 0.947 / 0.932 | 0.23 / 0.80 / 0.57 |
| D24 team swap | 0.659 | 0.715 | 0.824 | 39% | 0.919 / 0.901 | 0.22 / 0.76 / 0.52 |
| Standard combined | 0.7295 | 0.7557 | 0.8789 | | | |

- **Ownership is still the main loss** (standard combined: true ownership +0.149, true stoppages +0.026; D15: +0.162 and +0.027). Stoppages weigh more where the dead ball is realistic: dead ball carried +0.096, PFF +0.082, the D24 views +0.055 to +0.062.
- **The split oracle (point 4) cannot be read, so the reading rule cannot be applied as written.** Putting the true owner on unseen frames only, or on seen frames only, lowered spellF1 on most groups (pooled share of the ownership gain −0.52), because every seam between true and decoded labels adds boundaries. It measures the seams, not the headroom. This is a flaw in the rule, not a finding. The decision below is read from points 2 and 3 instead, and that is a reading made after the results were seen.
- **On real automatic tracking the loss is where the ball is hidden; the synthetic masks do not show it.** On PFF the owner agrees with the target on 95.7% of seen-ball frames, 85.1% of unseen ones and 76.6% of those more than 3 s after the last sighting; SkillCorner 94.4%, 84.2% and 79.8%. On the DFL masked and D24 views, whose masks are independent of play with hidden runs of 2 s on average, the two figures differ by 0.010 to 0.017. Spells whose ball is seen on under 20% of their frames are 23% of PFF's true spells and are recalled at 0.17 (5–10 s: 0.23; 10–20 s: 0.475), against 0.72–0.76 for the rest.
- **PFF still has sub-second stoppages** (116 predicted under 1 s against 1 in the targets); D25 found removing them changed almost nothing.

What follows: the next levers act on long hidden-ball stretches (D49, D50), and D46a first measures how long those stretches are and whether the players are visible in them. [Record](dfl_submission/diagnosis_recipe/d46.json)

## D47 — Mirror symmetry, token noise and seed averaging for the player-set networks: screening on development (2026-11-12, 21:28 EST, written before any D47 code exists, any network is trained or anything is scored)

**Question.** D43's four networks train on 16 matches with one augmentation (team order) and one seed per slot. Three cheap levers have not been tried on them: the pitch's mirror symmetries (in training and at inference), noise on the player tokens during training, and averaging seeds. Does each at least do no harm on development, so that it may join D48's stack?

**Disclosure.** Written after D42–D44 and after seeing D43's per-group results. Development has been reused by every step. D22 averaged three seeds of D15's networks and lost 0.0059 on the development primary while both guards and Metrica gained; D47 repeats that question for the player-set networks with the ownership cost re-read, because averaging flattens probabilities.

**Reference.** D43 from `data/d42/cache` at ownership cost 10, which must first reproduce its development, Metrica and D24 records exactly.

**Composite (the screening metric).** The unweighted mean of N = 3 spellF1 over eight development groups: DFL WPY clean, masked, dead ball carried and team swap; the four SkillCorner development matches (mean); the eight PFF development matches (mean); the three Metrica sample matches (mean); the D24 mean. Reason, fixed before scoring: D43's confirmation showed that the development primary, half of which is DFL WPY, understated the holdout gain, and the competition tracking is automatic with a hidden ball, which the degraded and unseen-vendor groups resemble most. D33's primary is reported beside it.

**Candidates.** Each retrains all four networks exactly as D42 did (both sets; seed 42 with boundary weight 0.5 and seed 97 with 1.4; D36's training values; width 96; 5,400 steps; MPS; last snapshot; the export check) except for the stated change. Trees are D3's, mixes D15's.
- **S (seeds).** D42's recipe at seeds + 1000 and + 2000: two more complete models. With D43 they give three independent models (the seed spread of the composite and the primary at cost 10) and **M3**, the per-slot mean of the three (`zoo.SlotMean`).
- **F0 (mirror inference only).** D43's own networks, with inference averaged over the four mirrors of the pitch (identity, x, y, both) times the two team orders. Nothing is trained.
- **F (mirrors).** In training, each window is mirrored in x with probability ½ and in y with probability ½, independently: the token's x offset, x velocity and x position change sign (likewise y). Inference as F0.
- **N (token noise), training only.** Each present token is masked with probability 0.10 per frame; Gaussian noise of 0.3 m is added to each player's position (offset and pitch position together, distance recomputed) and of 0.5 m/s to each velocity component. These constants are set here by judgement and are not tuned.
- **FN.** F and N together.

**Pipeline checks, before any candidate is trained.**
- The new trainer with every switch off retrains D42's original-set seed-42 network on MPS bit for bit.
- *Mirror parity of the 162 columns.* The columns are re-extracted from mirrored tracking for every prepared training half of both sets and compared with the stored columns. Each column must be unchanged or a pure sign change under each mirror (tolerance 1e-4); the trainer and the inference then apply that sign map with the token signs. If any column is neither, F0, F and FN stop and the orchestrator decides; S and N go on.
- Every network's NumPy export matches PyTorch within D15's tolerance.

**Grid.** Ownership cost ∈ {8, 10, 12} for every candidate, and 6 as well for M3; cutoff, stoppage cost and discount are D43's.

**Screening rule (no harm).** A lever (M3, F0, F, N, FN) is carried to D48 if, at its best cost by composite, its composite is at least D43's and no group falls by more than 0.01 against D43. Of F0, F and FN at most one is carried: the highest composite. Nothing is confirmed here and the build does not change; D48 sets the gate for the stack.

**Records.** One JSON per candidate in `dfl_submission/screen_recipe/`, each written once, with every group's score at every cost. Code: `dfl_submission/player_set3.py` (the configurable trainer and mirror-averaged NumPy inference; `player_set.py`, `player_trial.py` and everything else on the build path stay byte-identical) and `dfl_submission/screen_trial.py`. Scratch, weights and caches go to `data/d47/`. Every D47 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D47 outcome (2026-11-13, 01:08 EST).** **Carried to D48: the three-seed average (M3) and mirror training with mirror-averaged inference (F).** F0 failed; N passed without a gain and FN scored below F, so under D48's amendment the token noise is left out.

*Checks.* The trainer with every switch off retrained D42's original-set seed-42 network bit for bit; all 162 columns are unchanged under both mirrors on all 96 prepared training halves (largest difference 0), so only the token signs change; every export matched PyTorch; D43 reproduced its development, Metrica and D24 records (composite 0.71392, primary 0.72952).

| Candidate | Cost | Composite | Change | D33's primary | DFL clean / masked | Dead ball carried / team swap | SkillCorner | PFF | Metrica | D24 mean |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| D43 | 10 | 0.7139 | | 0.7295 | 0.780 / 0.722 | 0.643 / 0.694 | 0.718 | 0.663 | 0.791 | 0.701 |
| S1000 (seeds + 1,000) | 10 | 0.7090 | −0.0049 | 0.7269 | 0.767 / 0.715 | 0.645 / 0.680 | 0.721 | 0.662 | 0.782 | 0.700 |
| S2000 (seeds + 2,000) | 10 | 0.7160 | +0.0020 | 0.7355 | 0.772 / 0.731 | 0.646 / 0.709 | 0.726 | 0.657 | 0.792 | 0.695 |
| **M3** (mean of the three) | 12 | **0.7172** | **+0.0033** | 0.7335 | 0.775 / 0.729 | 0.661 / 0.698 | 0.722 | 0.663 | 0.790 | 0.699 |
| M3 | 6 / 8 / 10 | 0.7080 / 0.7106 / 0.7114 | | | | | | | | |
| F0 (mirror inference only) | 10 | 0.7129 | −0.0010 | 0.7328 | 0.780 / 0.725 | 0.634 / 0.689 | 0.723 | 0.663 | 0.787 | 0.702 |
| **F** (mirror training and inference) | 8 | **0.7175** | **+0.0036** | 0.7362 | 0.783 / 0.731 | 0.637 / 0.713 | 0.724 | 0.664 | 0.786 | 0.702 |
| F | 10 / 12 | 0.7163 / 0.7152 | | 0.7345 / 0.7315 | | | | | | |
| N (token noise) | 10 | 0.7143 | +0.0004 | 0.7305 | 0.774 / 0.724 | 0.640 / 0.697 | 0.720 | 0.665 | 0.793 | 0.701 |
| N | 8 / 12 | 0.7117 / 0.7113 | | | | | | | | |
| FN | 12 | 0.7147 | +0.0008 | 0.7348 | 0.786 / 0.726 | 0.635 / 0.701 | 0.723 | 0.663 | 0.786 | 0.697 |
| FN | 8 / 10 | 0.7128 / 0.7143 | | 0.7333 / 0.7358 | | | | | | |

- **Seed spread.** The three models of D42's recipe score 0.7139, 0.7090 and 0.7160 at cost 10: a standard deviation of 0.0036 on the composite and 0.0044 on the primary; on PFF they lie within 0.006 of each other. A single-model difference under about 0.01 on the composite is not evidence.
- **M3** passed the no-harm rule at cost 12 (+0.0033; largest fall DFL clean −0.005). At cost 10 it scored below the mean of its three members, so the stack's cost is re-read in D48.
- **F0** failed (composite −0.0010; dead ball carried −0.009). **F** passed at cost 8 (+0.0036; largest fall dead ball carried −0.006) and is positive at all three costs, with the primary +0.0067 and team swap +0.019. **N** passed with no gain. **FN**'s best composite (0.7147) is below F's (0.7175), so F is the mirror lever and the noise is left out.
- **What it shows.** None of these levers is large: each sits within about one seed spread of D43 on the composite. They are carried because they do no harm and each has a reason to help (variance, symmetry); the evidence that the stack is better has to come from L and C and from D48's own gate. [Records](dfl_submission/screen_recipe/)

## D46a — How long the ball stays hidden, and whether the players are seen meanwhile (2026-11-12, 21:40 EST, written after D46's results were read and before anything below is computed)

Measurement only, on D46's matches and views, with D43 decoded as in D46.
1. *Hidden runs.* A hidden run is a maximal run of frames without a ball position inside a half. Per group: the share of unseen target-live frames lying in runs of up to 1 s, 1–3 s, 3–10 s and over 10 s, and the median and 90th percentile of the run length over those frames.
2. *Agreement by run length.* D46's owner agreement on unseen frames, by the same four bins of the run the frame lies in.
3. *Players seen (PFF and SkillCorner).* The share of unseen target-live frames with at least 16 visible players (both teams), owner agreement on unseen frames with at least 16 and with fewer, and recall at IoU 0.5 of the under-20%-seen spells split by whether the spell's mean visible-player count is at least 16.

Record: `dfl_submission/diagnosis_recipe/d46a.json`, written once, from `diagnosis_d43.py`. It selects nothing. It is read as context for D49: if PFF's long hidden runs are mostly frames with fewer than 16 players, PFF cannot judge a ball-free lever and D49's long-hidden guard views carry the decision.

**D46a outcome (2026-11-12, 21:48 EST).** Share of unseen target-live frames by the length of their hidden run, and D43's owner agreement there:

| Group | Up to 1 s | 1–3 s | 3–10 s | Over 10 s | Median / 90th percentile run |
|---|---|---|---|---|---|
| DFL masked (training-like mask) | 9% / 0.912 | 36% / 0.934 | 52% / 0.888 | 2.5% / 0.810 | 3.2 s / 6.8 s |
| D24 masked | 9% / 0.908 | 34% / 0.950 | 46% / 0.934 | 11% / 0.887 | 3.6 s / 12.3 s |
| SkillCorner (real) | 0.2% / 0.625 | 23% / 0.854 | 35% / 0.898 | 42% / 0.774 | 7.2 s / 29.1 s |
| PFF (real) | 13% / 0.902 | 18% / 0.897 | 36% / 0.838 | 33% / 0.776 | 5.4 s / 34.9 s |

- **Real automatic tracking loses the ball for long.** A third to two fifths of the hidden live frames on PFF and SkillCorner lie in runs over 10 s; under the training mask it is 2.5%. Agreement is lowest in runs over 10 s on every group except SkillCorner, whose few frames in runs up to 1 s (0.2%) score 0.625; in runs over 10 s it is 0.78 on real tracking and 0.81 even on DFL masked.
- **Players seen.** On PFF only 22% of unseen live frames have at least 16 visible players (SkillCorner 11%). Agreement there is 0.862 against 0.848 with fewer, and the under-20%-seen spells are recalled at 0.175 (57 spells) against 0.174 (357): no better with the players in view.
- **Reading.** PFF's hidden frames are mostly frames with few players, so PFF cannot by itself judge a ball-free lever; D49's long-hidden guard views, where every player is seen, carry that decision, as D49's rule already allows. [Record](dfl_submission/diagnosis_recipe/d46a.json)

## D49 — Long hidden-ball runs in training (2026-11-12, 21:40 EST, written after D46's results and before any D49 view is built, any network is trained or anything is scored)

**Question.** Every DFL training view hides the ball with a two-state process whose hidden runs last 2 s on average, independent of play (`augment.wide_masking`, `HIDDEN_RUN` = 50 frames), so a run over 10 s almost never occurs in training. D46 shows the model fails exactly there on real automatic tracking, and the competition tracking hides the ball on 41% of graded frames (DATA.md). Does adding training views with long hidden runs raise spellF1 where the ball is hidden for long, without harm elsewhere?

**Disclosure.** Written after D46, whose reading was made after its results were seen. Development has been reused by every step. The mask constants below are set by judgement before anything is scored and are not tuned; nothing is taken from the provided tracking.

**Long hide process (training).** On the 25 Hz frames of a half, visible and hidden runs alternate, starting visible with probability v. A hidden run lasts, with probability ½, a geometric number of frames with mean 50 and otherwise a uniform number between 125 and 750 frames (5–30 s). A visible run is geometric with mean 243.75 · v / (1 − v) frames, so the expected visibility is v; v is drawn per half, uniform on (0.3, 0.9), as in `augment.wide_masking`.

**Candidate L.** Both model sets gain one view per training match, with new seeds:
- *DFL training matches:* the set's own view construction (`external_benchmark.halves` for the original set; `augment.training_halves`, including its dead-ball handling, for the augmented set) with `wide_masking`'s coordinate noise and player dropout and the long hide process in place of its two-state process.
- *SkillCorner training matches:* the provider tracking with the long hide process applied to the detected ball, v uniform on (0.5, 0.9), nothing else changed.
The prepared sequences and tokens of each set are its existing records, which must be reproduced byte for byte, plus the new views. All four networks are retrained exactly as D42 trained them. Trees are D3's, mixes D15's.

**Long-hidden guard views (development only, built differently from training, as D2's guards are).** Hidden runs uniform on 200–500 frames (8–20 s), visible runs uniform on 300–750 frames (12–30 s), starting visible, fixed new seeds: (a) DFL WPY, otherwise as its masked development view; (b) the three Metrica games, otherwise as D24's masked view. D43 is scored on them first, as the reference.

**Grid.** Ownership cost ∈ {8, 10, 12}; the rest is D43's.

**Carried to D48 if,** at the cost with the highest mean of (long-hidden guard mean, PFF development mean): the long-hidden guard mean (the mean of a and b) rises by at least +0.015 over D43, or the PFF development mean by at least +0.005; and D47's composite is no lower than D43's minus 0.003; and no D47 group falls by more than 0.01. Nothing is confirmed here and the build does not change.

**Records.** `dfl_submission/hidden_recipe/` (the reference on the guard views, then the candidate), each written once. Code: `dfl_submission/hidden_views.py` (the hide process, the training views, the guard views) and `dfl_submission/hidden_trial.py`; existing modules stay byte-identical. Sequences, tokens, weights and caches go to `data/d49/`. Every D49 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

*Amendment (2026-11-12, 21:55 EST, before any D49 view of the augmented set is built and before anything is scored).* The coding session asked two questions the rule left open for the augmented set. (1) Its new DFL view keeps the set's whole construction, `swap_teams` included, with only the hide process replaced. (2) Its new SkillCorner view is the set's own SkillCorner construction (the dead ball handled by `handle_dead_ball`, as in that set's existing view) followed by the long hide process; "nothing else changed" stands for the original set only. Read literally, the rule would have shown the augmented set SkillCorner's undetected dead ball again, the shortcut D2 built that set to remove. The guard views draw noise, dropout and hide from new seeds, and the new views join their source's records, so each batch stays half DFL and half SkillCorner.

**D49 outcome (2026-11-12, 22:55 EST).** **L is carried to D48.** The checks passed: both sets' 48 stored records were reproduced byte for byte and 32 long-hidden views were added to each; D43 reproduced its development, Metrica and D24 records. The guard views as built: DFL WPY visibility 0.60 with hidden runs of 14 s on average (85% of hidden frames in runs over 10 s); Metrica visibility 0.38 (the long process on top of D24's mask), runs of 18 s.

| | D43, cost 10 | L, cost 8 | L, cost 10 | L, cost 12 |
|---|---:|---:|---:|---:|
| **Long-hidden guard mean** | 0.6748 | 0.7149 | **0.7152 (+0.040)** | 0.7174 |
| DFL WPY long-hidden / Metrica long-hidden | 0.674 / 0.675 | 0.732 / 0.698 | 0.732 / 0.699 | 0.737 / 0.698 |
| **PFF development mean** | 0.6632 | 0.684 | **0.6868 (+0.024)** | 0.685 |
| PFF recall of spells seen under 20% | 0.174 | 0.258 | 0.258 | 0.258 |
| D47's composite | 0.7139 | 0.7097 | 0.7129 (−0.001) | 0.7138 |
| D33's primary | 0.7295 | 0.7266 | 0.7257 | 0.7259 |
| DFL clean / masked | 0.780 / 0.722 | 0.769 / 0.712 | 0.774 / 0.716 | 0.779 / 0.716 |
| Dead ball carried / team swap | 0.643 / 0.694 | 0.631 / 0.679 | 0.635 / 0.690 | 0.639 / 0.688 |
| SkillCorner / Metrica / D24 mean | 0.718 / 0.791 / 0.701 | 0.722 / 0.777 / 0.703 | 0.716 / 0.783 / 0.702 | 0.715 / 0.790 / 0.699 |

- **The carry rule passed at cost 10** (the cost with the highest mean of guard and PFF): the guard mean rose +0.040 (+0.015 needed), PFF +0.024 (+0.005 needed), the composite fell 0.001 (−0.003 allowed), and no group fell by more than 0.01 (largest: dead ball carried −0.008).
- **What it shows.** Seeing long hidden runs in training is worth +0.057 on a DFL match whose ball is hidden for 8–20 s at a time and +0.024 on real PFF tracking, where the under-20%-seen spells are recalled at 0.26 instead of 0.17. The other groups moved by less than a second seed set of D43's own recipe did in D47 (S1000: DFL clean −0.013, team swap −0.014), so no harm is visible; PFF's frame agreement on unseen frames barely moved (0.851 → 0.853), so most of the hidden-ball loss on PFF remains. [Reference](dfl_submission/hidden_recipe/reference.json), [candidate](dfl_submission/hidden_recipe/candidate.json)

## D50 — Player tokens in the attacking frame (2026-11-12, 21:40 EST, written before any D50 code exists, any network is trained or anything is scored)

**Question.** Nothing the models read says which way a team attacks: the 162 columns are unsigned and the player tokens carry positions in pitch coordinates. Without the ball, which team is moving towards its opponent's goal is the plainest sign of possession. Does giving the tokens in each team's attacking frame raise spellF1, above all where the ball is hidden?

**Disclosure.** Written after D46. Before D34, a univariate probe of direction-aware team features found no possession signal without the ball; D50 tests the same idea inside the network, where it can combine with the players' positions. Development has been reused by every step.

**Attacking sign, per half, without labels or fitted values.** Over the half's long-context frames with at least seven visible players in each team, D is the mean of (mean x of the away players − mean x of the home players). The home team attacks towards +x when D > 0 and towards −x otherwise. *Pipeline check:* on every external half read (training and development only), the sign must equal the one given by the first frame of the half with at least seven visible players per team; any disagreement stops D50 and is reported.

*Amendment (2026-11-12, 21:58 EST, before any D50 network is scored).* One half read for D50 has no frame with seven visible players in each team (PFF development game 3852, second half: its home team never shows more than six). A half without such a frame takes the opposite of the sign of the same match's other half, because teams change ends at half time; if neither half has a sign, the sign is +1 (the tokens stay in pitch coordinates). The other 151 halves read (96 training, 31 development, 6 Metrica, 18 D24) all have a sign that agrees with their first qualifying frame; the smallest |D| is 2.1 m.

**Candidate C.** The token's x offset, x velocity and x position are multiplied by the half's sign, so the home team always attacks towards +x; when the teams are swapped (the trainer's augmentation and the swap-averaged inference), x is negated as well, so the first team still attacks towards +x. If D47's parity check finds columns that change sign under the x mirror, they are multiplied the same way. Nothing else changes: all four networks are retrained as D42 trained them on the same prepared sequences; trees are D3's, mixes D15's.

**Grid.** Ownership cost ∈ {8, 10, 12}; the rest is D43's.

**Carried to D48 if,** at its best cost by D47's composite, the composite is at least D43's and no group falls by more than 0.01. The PFF development mean, D49's long-hidden guard views (if built by then) and the owner agreement on unseen frames are reported beside it. Nothing is confirmed here and the build does not change.

**Records.** `dfl_submission/direction_recipe/`, each written once. Code: `dfl_submission/direction_tokens.py` and `dfl_submission/direction_trial.py`; existing modules stay byte-identical. Tokens, weights and caches go to `data/d50/`. Every D50 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D50 outcome (2026-11-12, 23:42 EST).** **C is carried to D48.** Checks: the trainer with the sign forced to +1 and the swap negation off reproduced D42's original-set seed-42 network bit for bit; 151 of 152 halves have a sign and all agree with their first qualifying frame (the one without took the amendment's fallback); D47's parity record lists no sign-changing column, so only the tokens changed; D43 reproduced its records and D47's reference composite.

| | Composite | Change | D33's primary | DFL clean / masked | Dead ball carried / team swap | SkillCorner | PFF | Metrica | D24 mean |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| D43, cost 10 | 0.7139 | | 0.7295 | 0.780 / 0.722 | 0.643 / 0.694 | 0.718 | 0.663 | 0.791 | 0.701 |
| C, cost 8 | **0.7208** | **+0.0069** | 0.7324 | 0.773 / 0.732 | 0.654 / 0.702 | 0.719 | 0.681 | 0.791 | 0.715 |
| C, cost 10 | 0.7200 | +0.0061 | 0.7298 | 0.775 / 0.722 | 0.650 / 0.705 | 0.720 | 0.681 | 0.791 | 0.717 |
| C, cost 12 | 0.7207 | +0.0068 | 0.7289 | 0.780 / 0.719 | 0.640 / 0.712 | 0.718 | 0.680 | 0.796 | 0.720 |

- **The carry rule passed** at cost 8: the composite rose +0.0069 and the largest fall was DFL clean −0.007.
- **What it shows.** The gain is the same at all three costs and sits where the hypothesis put it: PFF +0.018, the D24 mean +0.015 to +0.019, team swap +0.008 to +0.018, with the clean and seen-ball groups flat. Three seed sets of D43's recipe spread about ±0.004 on the composite and ±0.003 on PFF (D47), so PFF's gain is well outside noise and the composite's is about 1.7 of those spreads. D49's guard views were scored for C afterwards, outside the carry rule (2026-11-13, 00:13 EST, [guards](dfl_submission/direction_recipe/guards.json)): guard mean 0.6751 / 0.6838 / 0.6864 at cost 8 / 10 / 12 against D43's 0.6748 at cost 10. [Records](dfl_submission/direction_recipe/)

## D51 — An auxiliary ball-location target on the long-hidden views (2026-11-12, 22:55 EST, written after D49's outcome and before any D51 code exists, any network is trained or anything is scored)

**Question.** D49's views taught the networks to keep the owner through long hidden runs, but only through the possession labels. In those views the hidden ball's true position is known. Does also asking the network where the ball is, on the frames where the view hides it, raise spellF1 on hidden-ball tracking beyond L?

**Disclosure.** Written after D49's development results. Development has been reused by every step. The loss weight and scale below are set by judgement and are not tuned.

**Candidate LB.** L's sequences, tokens and training values, with two more network outputs: the ball's x and y offset from the token's reference point (the mean position of the visible players when the ball is hidden), divided by 10 m. They are trained with a Huber loss (δ = 1) on the frames of D49's new views where the view hides a ball the source tracking has; other frames and other views carry no auxiliary loss. The term is added to D42's loss with weight 0.5. Inference is unchanged: the export keeps the four original outputs, so the NumPy path and its check are D42's.

**Pipeline checks.** The auxiliary targets equal the hidden ball of the stored views exactly (re-derived from the same seeds); with the auxiliary weight at 0 the four original outputs' loss is D42's; every export matches PyTorch within D15's tolerance.

**Grid.** Ownership cost ∈ {8, 10, 12}; the rest is D43's.

**LB replaces L in D48's stack if,** at its best cost by the mean of (long-hidden guard mean, PFF development mean), that mean is at least L's best plus 0.005, D47's composite is no lower than L's minus 0.003, and no group falls by more than 0.01 against D43. Otherwise L stays. Nothing is confirmed here and the build does not change.

**Records.** `dfl_submission/hidden_recipe/ball_*.json`, each written once. Code: `dfl_submission/hidden_ball.py` (targets, the network with the extra outputs, the trainer) and a command in `dfl_submission/hidden_trial.py` or a new `hidden_ball_trial.py`; modules written before D51 stay byte-identical. Everything else goes to `data/d51/`. Every D51 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D51 outcome (2026-11-13, 00:48 EST).** **Failed: L stays, LB is not in the stack.** The checks passed (targets re-derived exactly for both sets; the trainer check; the exports).

| | Composite | Long-hidden guard mean | PFF | D33's primary |
|---|---:|---:|---:|---:|
| D43, cost 10 | 0.7139 | 0.6748 | 0.6632 | 0.7295 |
| L, cost 10 | 0.7129 | 0.7152 | 0.6868 | 0.7257 |
| LB, cost 8 (its best) | 0.6962 | 0.6999 | 0.6672 | 0.7106 |
| LB, cost 10 / 12 | 0.6930 / 0.6925 | 0.6969 / 0.6916 | 0.6660 / 0.6661 | 0.7068 / 0.7055 |

- **LB is worse than L everywhere:** the mean of guard and PFF −0.018 (+0.005 was needed), the composite −0.017, and against D43 six of eight groups fell by more than 0.01 (dead ball carried −0.036, D24 −0.025).
- **The network did learn where the ball is.** On hidden frames of development views its estimate is 5.9–7.6 m off at the median, against 19–23 m between the ball and the players' mean position. That knowledge did not help the possession outputs at this weight; it cost them. The weight was set by judgement and is not searched: a second weight would be a second attempt on the same matches. [Record](dfl_submission/hidden_recipe/ball_candidate.json), [auxiliary error](dfl_submission/hidden_recipe/ball_aux_error.json)

## D48 — The stack: long-hidden views, attacking-frame tokens and three seeds, confirmed once (2026-11-12, 23:42 EST, written after the outcomes of D49 and D50 and D47's seed results, before any stack network is trained or anything below is scored)

**Question.** L (D49) and C (D50) each gained on hidden-ball and unseen-vendor tracking, and averaging three seeds (D47's M3) did no harm. Do they add up, and does the stack clear C3's bar on fresh matches?

**Disclosure.** Written after the development results of D47 (S1000, S2000, M3, F0), D49 and D50 were seen. Development has been reused by every step; the holdout has been used by every confirmation since D3. The confirmation below is the last use of the untouched PFF set. The 21-match refit (D45) is not part of the stack: its vetoes showed no gain, and it would take development away from this step.

**The stack.** D42's player-set recipe with:
- *L:* D49's training sequences and tokens for both sets;
- *C:* D50's attacking-frame tokens (with its amendment for a half without a sign), on every view including L's;
- *three seeds per slot:* seeds 42 and 97 plus 0, 1,000 and 2,000, the three predictions of a slot averaged (`zoo.SlotMean`);
- *any other lever that passes its own rule before the stack is trained:* N, F or FN under D47's no-harm rule (of F and FN at most one, F0 has failed), and LB in place of L under D51's rule. A lever whose record is not written by then is left out.
Trees are D3's, mixes D15's, cutoff, stoppage cost and discount D43's.

*Amendment (2026-11-13, 00:17 EST, after D47's F and N were scored and before FN's or LB's score exists or any stack network is trained).* Two points the rule left open. (1) *Mirrors with C.* In the attacking frame a mirror in x changes no token, and no column changes under either mirror (D47's parity check), so with C the mirror lever acts in y only: each training window is mirrored in y with probability ½, and inference averages the two y mirrors times the two team orders. (2) *Token noise.* N alone passed the no-harm rule with no gain (composite +0.0004 at cost 10; −0.002 and −0.003 at the other costs). The rule as written could put F and N into the stack separately even if FN, which is exactly the two together, scores below F. So N enters the stack only as part of FN, and only if FN's best composite is at least F's best (0.7175); otherwise the mirror lever is F and the noise is left out.

**Pipeline checks, before any stack network is trained.** One trainer holds every switch. With only C on and D42's data it must reproduce D50's original-set seed-42 network bit for bit; with only the seed offset 1,000, D47's S1000 original-set seed-42 network; with only L's data, D49's. Every export matches PyTorch within D15's tolerance. D43 reproduces its development, Metrica, D24 and guard-view records.

**Development (it decides only whether the confirmation is spent).** The stack is scored at ownership cost 8, 10, 12 and 14 on D47's eight groups, D49's two guard views and D33's conventions; each seed set alone is scored too, as a measure of spread. The selection score is the mean of D47's composite and the long-hidden guard mean (D43 at cost 10: 0.6944). The cost with the highest selection score goes on if, against D43 at cost 10: the selection score rises by at least +0.015; the PFF development mean by at least +0.015; the composite does not fall; no group falls by more than 0.01; and standard combined spellF1 at N = 1.75, 4 and 4.5 each falls by no more than 0.01. Otherwise the confirmation is not spent and the orchestrator decides what is taken forward.

**Confirmation, once,** at the selected cost, against D43, which must first reproduce its holdout record:
- the reused DFL/SkillCorner holdout, as D43's confirmation scored it (primary, conventions, dead ball carried, team swap);
- long-hidden views of the two DFL holdout matches (WQQ, WR9), built as D49's guard view (a) with new fixed seeds;
- the untouched PFF set, parsed only for this step.

**It passes if all of these hold:** the untouched PFF mean rises by at least **+0.01** N = 3 spellF1 (C3's bar, on fresh real tracking the models never trained on, as PFF set C was D32's confirmation under C3); the holdout long-hidden mean rises by at least +0.01; the holdout primary falls by no more than 0.01; and N = 1.75, 4 and 4.5, dead ball carried and team swap each fall by no more than 0.01. The reason the bar sits on PFF and the long-hidden views: that is where the stack acts and what the competition tracking resembles (the ball is missing on 41% of graded frames); the standard holdout views hide the ball for 2 s at a time and can only show that nothing was lost.

**If it passes,** the stack replaces D43's networks in the build behind a switch (`player_trial build` still gives `a56574fe…`), with two identical rebuilds, `release/check.sh`, C2 re-declared, and the records. In the build the attacking sign of each graded half is computed from that half's tracking by D50's rule; it is a feature of the input, not a fitted value. **If it fails,** `a56574fe…` stays and the untouched PFF set is spent.

**Records.** `dfl_submission/stack_recipe/`, each written once. Code: `dfl_submission/stack_set.py` (the trainer and NumPy inference with every switch) and `dfl_submission/stack_trial.py` (`check`, `networks`, `develop`, `confirm`, `build`); modules written before D48 stay byte-identical. Scratch and caches go to `data/d48/`; the weights of an adopted stack go to `dfl_submission/stack_models/`. Every D48 command except `build` runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D48 development (2026-11-13, 03:18 EST).** **The gate passed at ownership cost 14; the confirmation may be spent, after D52 has reported.**

*Levers in the stack.* L, C, three seeds per slot and F acting in y (the 00:17 amendment). The noise is out (FN 0.7147 below F 0.7175) and so is LB (D51 failed). *Checks.* One trainer reproduced D50's, D47's S1000, D49's and D47's F original-set seed-42 networks bit for bit, each with only that switch on; every export matched PyTorch; D43 reproduced its development, Metrica, D24 and guard-view records; 517 of 517 tests passed.

| | Cost | Selection score | Composite | Long-hidden guard mean (DFL / Metrica) | PFF | D33's primary | DFL clean / masked | Dead ball carried / team swap | SkillCorner | Metrica | D24 mean |
|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|---:|
| D43 | 10 | 0.6944 | 0.7139 | 0.675 (0.674 / 0.675) | 0.663 | 0.7295 | 0.780 / 0.722 | 0.643 / 0.694 | 0.718 | 0.791 | 0.701 |
| Stack | 8 | 0.7229 | 0.7190 | 0.727 (0.742 / 0.711) | 0.699 | 0.7286 | 0.767 / 0.725 | 0.651 / 0.687 | 0.718 | 0.785 | 0.721 |
| Stack | 10 | 0.7236 | 0.7246 | 0.722 (0.730 / 0.715) | 0.698 | 0.7338 | 0.782 / 0.726 | 0.658 / 0.695 | 0.723 | 0.792 | 0.723 |
| Stack | 12 | 0.7251 | 0.7267 | 0.723 (0.731 / 0.716) | 0.694 | 0.7331 | 0.786 / 0.726 | 0.664 / 0.697 | 0.720 | 0.801 | 0.726 |
| **Stack** | **14** | **0.7265 (+0.032)** | **0.7279 (+0.014)** | **0.725 (0.731 / 0.720) (+0.050)** | **0.691 (+0.028)** | 0.7357 | 0.788 / 0.730 | 0.661 / 0.706 | 0.722 | 0.799 | 0.726 |
| Seed set 0 alone | 14 | 0.7223 | 0.7255 | 0.719 | 0.691 | 0.7325 | 0.784 / 0.725 | 0.656 / 0.709 | 0.721 | 0.792 | 0.726 |
| Seed set 1,000 alone | 14 | 0.7250 | 0.7260 | 0.724 | 0.695 | 0.7344 | 0.786 / 0.725 | 0.660 / 0.702 | 0.723 | 0.799 | 0.717 |
| Seed set 2,000 alone | 14 | 0.7265 | 0.7261 | 0.727 | 0.689 | 0.7355 | 0.793 / 0.735 | 0.645 / 0.711 | 0.717 | 0.795 | 0.724 |

- **Every gate condition held at cost 14** (the cost with the highest selection score): selection score +0.032 (+0.015 needed), PFF +0.028 (+0.015 needed), composite +0.014 (no fall allowed), all eight groups up (the smallest, SkillCorner, +0.004), N = 1.75, 4 and 4.5 up by +0.005, +0.008 and +0.006.
- **The levers added up.** L alone lifted the guard mean +0.040 and PFF +0.024 with the composite flat; C alone lifted the composite +0.007 and PFF +0.018. Together with the seeds and the mirror: guard +0.050, PFF +0.028, composite +0.014, and the standard primary +0.006. PFF's spells seen under 20% are recalled at 0.26–0.29 (D43: 0.17).
- **It does not hang on a seed.** Each seed set alone scores a composite of 0.7255–0.7261 at cost 14, each about +0.012 over D43, where three seed sets of D43's own recipe spread 0.0036.
- **The selected cost is the grid's largest.** The selection score rises by about 0.001 per step from 8 to 14, less than a seed spread; PFF alone prefers the low costs (0.699 at 8). The grid is not extended: the rule fixed it, and the differences are too small to act on.

D52 (T and X on seed set 0) is still running; under D52's rule the stack is retrained before the confirmation if one of them holds. [Development](dfl_submission/stack_recipe/development.json)

## D52 — Training length and temporal context for the stack's recipe (2026-11-13, 01:09 EST, written while D48's last trainer check runs and before any stack or D52 network is trained or anything of D48 or D52 is scored)

**Question.** D48's stack trains each network for D42's 5,400 steps on two thirds more sequences (D49 added 32 views to each set's 48) with one more augmentation, and D15's whole gain came from networks that had been under-trained. Its networks also see 12.6 s to each side, while D49's views hide the ball for up to 30 s. Does training longer, or seeing further, raise spellF1 for this recipe?

**Disclosure.** Written after D47, D49, D50 and D51. D38 found 10,800 steps worse than 5,400 for D36's values on the original data (by cross-validation); D20's longer-context networks did not help D15. Both were measured without the long-hidden views.

**Base.** The stack's first seed set alone (L, C and the y mirror; seeds 42 and 97), trained and scored by D48.

**Candidates,** each the base with one change, first seed set only (four networks):
- **T:** 8,100 steps (the schedule stretched, last snapshot), which gives each sequence about as many visits as D42's 5,400 steps gave its 48.
- **X:** a seventh temporal block with dilation 64, so the network sees 25.4 s to each side (windows of 128 steps plus 127 on each side); 5,400 steps.

**Grid.** Ownership cost ∈ {8, 10, 12, 14}; the rest is D43's. The selection score is D48's (the mean of D47's composite and the long-hidden guard mean).

**A candidate holds if,** at its best cost, its selection score is at least the base's best plus **0.008** (about two seed spreads of the composite), its PFF development mean is no more than 0.005 below the base's, and no group falls by more than 0.01 against the base. If both hold, the higher selection score is taken; they are not combined untested.

**If one holds,** the stack is retrained once with that change (three seeds), and D48's development gate and confirmation are read on the retrained stack; D48's record says so before its confirmation is run. **If neither holds,** D48 goes on as trained.

**Records.** `dfl_submission/stack_recipe/d52_*.json`, each written once. Code: new modules only (`dfl_submission/context_set.py` for the seven-block network and its NumPy inference, `dfl_submission/length_trial.py` for both candidates), reusing `stack_set.py` and `stack_trial.py` unchanged. Scratch goes to `data/d52/`. Every D52 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D52 outcome (2026-11-13, 03:54 EST).** **T holds; X does not. The stack is retrained at 8,100 steps before the confirmation.** D43 was recomputed bit for bit on all 64 halves; the seven-block network equals `player_set`'s forward at six blocks.

| First seed set alone | Cost | Selection score | Composite | Long-hidden guard mean (DFL / Metrica) | PFF | D33's primary | Dead ball carried / team swap | Metrica | D24 mean |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|
| Base (5,400 steps, six blocks) | 14 | 0.7223 | 0.7255 | 0.719 (0.726 / 0.712) | 0.691 | 0.7325 | 0.656 / 0.709 | 0.792 | 0.726 |
| **T (8,100 steps)** | **14** | **0.7311 (+0.0088)** | 0.7299 | 0.732 (0.743 / 0.721) | 0.695 | 0.7344 | 0.668 / 0.712 | 0.801 | 0.730 |
| T | 8 / 10 / 12 | 0.7268 / 0.7243 / 0.7271 | 0.7226 / 0.7226 / 0.7251 | 0.731 / 0.726 / 0.729 | 0.701 / 0.699 / 0.697 | | | | |
| X (seven blocks) | 12 | 0.7192 (−0.0030) | 0.7220 | 0.716 (0.734 / 0.699) | 0.695 | 0.7301 | 0.652 / 0.701 | 0.788 | 0.718 |

- **T passed all three conditions:** selection score +0.0088 over the base's best (+0.008 needed), PFF +0.003 (−0.005 allowed), and the largest fall of a group −0.001 (DFL clean). Its single seed set scores above the three-seed stack at 5,400 steps (0.7265) and above each of that stack's seed sets at any cost (best 0.7284).
- **X failed:** seeing 25 s to each side lowered the selection score by 0.003 and the Metrica long-hidden view by 0.013. The networks keep D15's six blocks.
- **A caution.** T clears its bar by 0.0008, and the three seed sets of the 5,400-step stack differ by up to 0.004 in selection score, so the gain is likely real but small. The retrained stack is scored on development again before anything is confirmed (amendment below). [T](dfl_submission/stack_recipe/d52_T.json), [X](dfl_submission/stack_recipe/d52_X.json), [decision](dfl_submission/stack_recipe/d52_decision.json)

*D48 amendment (2026-11-13, 03:54 EST, after D52's outcome and before any further stack network is trained or scored).* Under D52's rule the stack is retrained once at 8,100 steps: the same levers and seeds, with D52's four T networks as its first seed set if the stack's trainer reproduces one of them bit for bit, and eight new networks for the seed offsets 1,000 and 2,000. It is scored on development exactly as the 5,400-step stack was. The confirmation then goes to one stack only: the 8,100-step stack if it passes D48's gate and its selection score at its selected cost is at least the 5,400-step stack's (0.7265); otherwise the 5,400-step stack at cost 14, which has passed. The confirmation's rule is unchanged.

**D48 development of the 8,100-step stack (2026-11-13, 05:53 EST).** **The gate passed at cost 14 and the amendment selects the 8,100-step stack for the confirmation.** The stack's trainer at 8,100 steps reproduced D52's original-set seed-42 T network bit for bit, so D52's four networks are its first seed set; eight networks were trained for the other two; 547 of 547 tests passed.

| | Cost | Selection score | Composite | Long-hidden guard mean (DFL / Metrica) | PFF | D33's primary | DFL clean / masked | Dead ball carried / team swap | SkillCorner | Metrica | D24 mean |
|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|---:|
| D43 | 10 | 0.6944 | 0.7139 | 0.675 (0.674 / 0.675) | 0.663 | 0.7295 | 0.780 / 0.722 | 0.643 / 0.694 | 0.718 | 0.791 | 0.701 |
| 5,400-step stack | 14 | 0.7265 | 0.7279 | 0.725 (0.731 / 0.720) | 0.691 | 0.7357 | 0.788 / 0.730 | 0.661 / 0.706 | 0.722 | 0.799 | 0.726 |
| 8,100-step stack | 8 / 10 / 12 | 0.7258 / 0.7279 / 0.7311 | 0.7191 / 0.7233 / 0.7257 | 0.733 / 0.733 / 0.736 | 0.698 / 0.698 / 0.699 | | | | | | |
| **8,100-step stack** | **14** | **0.7326 (+0.038)** | **0.7283 (+0.014)** | **0.737 (0.755 / 0.719) (+0.062)** | **0.696 (+0.033)** | 0.7352 | 0.781 / 0.728 | 0.657 / 0.707 | 0.725 | 0.803 | 0.729 |
| Its seed sets alone | 14 | 0.7311 / 0.7300 / 0.7338 | 0.7299 / 0.7266 / 0.7270 | 0.732 / 0.733 / 0.741 | 0.695 / 0.694 / 0.691 | | | | | | |

- **Every gate condition held at cost 14:** selection score +0.038 (+0.015 needed), PFF +0.033 (+0.015), composite +0.014 (no fall), all eight groups up (the smallest, DFL clean, +0.001), N = 1.75, 4 and 4.5 up by +0.009, +0.007 and +0.005.
- **Against the 5,400-step stack** the selection score is +0.006 (0.7326 against 0.7265): the long-hidden guard mean +0.012 and PFF +0.005, with the composite level (0.7283 against 0.7279). All three seed sets at 8,100 steps score above the 5,400-step stack's selection score.
- **By the amendment the confirmation goes to the 8,100-step stack at ownership cost 14.** [Development](dfl_submission/stack_recipe/development_t.json)

**D48 confirmation (2026-11-13, 06:37 EST).** **Passed.** The stack (8,100 steps, ownership cost 14) is to replace D43 in the submission, after the adoption steps below. The untouched PFF set is spent; no fresh PFF matches remain.

*Before scoring.* D43 reproduced its holdout record. The untouched PFF set (games 3819, 3827, 3835, 3843, 3851, 3859, 10509, 10517) was parsed at 06:03 EST for this step and had never been parsed before. The long-hidden views of DFL WQQ and WR9 were built with new seeds (visibility 0.60, hidden runs of 14.2 s on average, 90% of hidden frames in runs over 10 s). Every half's attacking sign agreed with its first qualifying frame, with no fallback (24 holdout halves, 4 long-hidden, 16 PFF).

| | D43, cost 10 | Stack, cost 14 | Change | Required |
|---|---:|---:|---:|---|
| **Untouched PFF set, N = 3 mean (eight fresh matches)** | 0.6886 | **0.7236** | **+0.0350** | ≥ +0.01 |
| **Holdout long-hidden mean (WQQ / WR9)** | 0.6714 (0.689 / 0.654) | **0.7135** (0.719 / 0.708) | **+0.0421** | ≥ +0.01 |
| **Holdout primary, N = 3 standard combined** | 0.72761 | 0.73276 | +0.0052 | ≥ −0.01 |
| DFL WQQ clean / masked | 0.766 / 0.745 | 0.757 / 0.751 | −0.010 / +0.005 | |
| DFL WR9 clean / masked | 0.788 / 0.725 | 0.794 / 0.721 | +0.006 / −0.004 | |
| SkillCorner holdout (four matches) | 0.707, 0.670, 0.745, 0.703 | 0.716, 0.691, 0.741, 0.718 | mean +0.010 | |
| N = 1.75 / 4 / 4.5 | | | +0.016 / +0.001 / +0.001 | each ≥ −0.01 |
| Dead ball carried / team swap | 0.643 / 0.693 | 0.646 / 0.711 | +0.003 / +0.018 | each ≥ −0.01 |
| Raw provider labels, standard combined (reported) | | | +0.013 | |

- **Every condition held.** On the eight fresh PFF matches every match gained (+0.021 to +0.063; the weakest match, 3851, rose from 0.614 to 0.678). On the long-hidden views both DFL holdout matches gained (+0.030 and +0.054). The standard holdout did not fall: the primary rose +0.005, inside the seed spread, as the rule expected of views that hide the ball for 2 s at a time.
- **What it shows.** The gain development found on hidden-ball and unseen-vendor tracking held on matches no step had touched, at about the same size (development: PFF +0.033, long-hidden guard +0.062, primary +0.006). [Confirmation](dfl_submission/stack_recipe/confirmation.json)

*Adoption amendment (2026-11-13, 06:37 EST, written after the confirmation and before any stack build has run).* An independent review of `stack_trial build` (no build was run; `tests/test_stack_build_review.py`) found that everything upstream of the networks is the standing path and that the sign, swap, mirror and seed-averaging code is the code development scored, and three things to change before the build may run:
1. *Identity.* The build refuses unless all twelve networks, the trees, the code it runs and the lever record equal those written in `confirmation.json`, as `player_trial build` refuses against its record.
2. *Levers.* The switches are read from the confirmed lever record, not from `levers.json`.
3. *Signs, fail closed.* The build refuses if a graded half has no sign of its own, if a half's sign disagrees with its first qualifying frame, or if the two halves have the same sign. These are the checks every external half passed. If the build refuses on them, only that fact is read, the stack does not ship, and `a56574fe…` stays unless Derek decides otherwise.

Then, as D48's rule says: the twelve confirmed networks are copied to `dfl_submission/stack_models/`; two builds with `stack_trial build` must give the same SHA-256 with the validator's OK line (only that line and the SHA are read); `player_trial build` must still give `a56574fe…`; the four release builds (two fresh macOS environments and the pinned image under `linux/arm64` and `linux/amd64`) run on a copy of the working tree, and `release/check.sh` on the commit once Derek approves one; C2 is re-declared (amendment under C2) and run once by Derek; C1's record is amended as C3 requires; and the README, CONSTANTS.md, the technical document, `SUBMISSION_SHA256.txt` and `release/BUILD_COMMAND` are updated.

**D48 adoption, first build attempt (2026-11-13, 07:36 EST).** **The stack's build refused: "a half's sign disagrees with its first qualifying frame". No file was written and the submission is unchanged (`a56574fe…`).**

- *Done before the build.* `stack_trial build` now refuses unless the twelve networks, the trees, the lever record and the code it runs equal `confirmation.json` (the build driver itself is locked in `stack_recipe/build_identity.json`, written only after the edited code recomputed `development_t.json` and `confirmation.json` exactly from their caches and re-forecast the eight development guard halves bit for bit with the copied networks); it reads its switches from the confirmed lever record; it carries the three fail-closed sign checks; 553 of 553 tests passed. The twelve confirmed networks were copied to `dfl_submission/stack_models/` and verified against the confirmed identity (`stack_recipe/adoption_models.json`). A first attempt stopped in 0.6 s on a missing import before any data was read; the import order was fixed and the reproduction re-run.
- *The refusal.* On the second attempt the build stopped at the sign check quoted above. Only that message was read: no number from the tracking, no log, no file. The second stack build, the `player_trial build` check and the release builds did not run. `data/submission.csv`, `Anfield.csv`, `SUBMISSION_SHA256.txt` and `release/BUILD_COMMAND` are untouched.
- *What it means under the amendment.* The stack does not ship and `a56574fe…` stays, unless Derek decides otherwise. The check did what it was written for: something that held on every external half checked in D48–D50 (268 halves: 224 in `stack_recipe/signs.json` and 44 in the confirmation; the one half without a qualifying frame, PFF 3852's second, took D50's fallback) does not hold on the provided match as the build reads it.
- *A known difference between the build and the external halves, not examined on the provided data.* Every external half starts at its kick-off, where each team stands in its own half. The build applies D50's rule to a graded half as a whole, and D27 exists because the provided match's second graded half starts before its tagged kick-off. The review of the build had noted this (its finding 6). Whether this is the cause has not been looked at.

## D53 — A fallback: the stack without the attacking frame (2026-11-13, 07:38 EST, written after the stack's build refused and before any D53 network is trained or anything is scored)

**Question.** The attacking-frame tokens (C) are the one part of the confirmed stack that reads something from the provided match at build time, and the build refused on their check. If C cannot be used, how much of the confirmed gain remains without it? D53 prepares that answer while Derek decides about the build; it ships nothing by itself.

**Disclosure.** Written after D48's confirmation and after the build's refusal, of which only the message was read. The matches D53 is confirmed on were scored once for D48's stack, so its confirmation is a second attempt on the same matches and weaker evidence than D48's; the untouched PFF set was fresh for D48 and is not fresh here. Nothing was selected on those matches in D48, and D53's candidate and cost are fixed before it is scored on them.

**Candidate.** D48's stack without C: D49's long-hidden sequences and tokens, tokens in pitch coordinates, D47's mirror lever as screened (each training window mirrored in x and in y with probability ½ each; inference averaged over the four mirrors times the two team orders), three seeds per slot (offsets 0, 1,000, 2,000), 8,100 steps; trees D3's, mixes D15's, cutoff, stoppage cost and discount D43's. Its build needs no attacking sign and reads the provided match exactly as `player_trial build` does.

**Development.** Scored as D48's stack was (the averaged model and each seed set alone; ownership cost 8, 10, 12 and 14; D47's eight groups, D49's two guard views, D33's conventions), with D48's selection score and D48's gate against D43 at cost 10. If the gate fails, D53 stops.

**Confirmation, a second use (see the disclosure),** at the selected cost against D43, on the sets of D48's confirmation as they were built for it: the holdout, the long-hidden views of WQQ and WR9 with D48's seeds, and the eight PFF games parsed for D48. D48's pass rule applies unchanged: PFF mean at least +0.01, holdout long-hidden mean at least +0.01, holdout primary no more than 0.01 lower, N = 1.75, 4 and 4.5, dead ball carried and team swap each no more than 0.01 lower.

**What it is used for.** If D53 passes, it is a fallback that Derek may choose to ship if the stack with C does not ship (he does not approve another build of it, or that build refuses again). Shipping it would follow the same adoption steps as D48, with its weaker evidence stated in the technical document. If the stack with C ships, D53 is recorded and not used.

**Records.** `dfl_submission/stack_recipe/*_d53.json`, each written once. Code: `stack_set.py` and `stack_trial.py` through their switches (direction off, mirror in x and y), with any new command in a new module `dfl_submission/fallback_trial.py`; the confirmed stack's records and the locked build identity are not changed. Scratch goes to `data/d53/`. Every D53 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D53 outcome (2026-11-13, 11:29 EST).** **The fallback passed its development gate and its second-use confirmation at ownership cost 14.** It ships nothing by itself: it is available to Derek if the stack with C does not ship. No trainer check was added (D48's `check_mirror` and `check_hidden` cover its two switches; no reference exists for the combination at 8,100 steps).

| | D43, cost 10 | Stack with C (D48), cost 14 | Fallback without C (D53), cost 14 |
|---|---:|---:|---:|
| *Development:* selection score | 0.6944 | 0.7326 | 0.7249 (+0.031) |
| Composite / long-hidden guard mean | 0.7139 / 0.675 | 0.7283 / 0.737 | 0.7257 (+0.012) / 0.724 (+0.049) |
| PFF development / D24 mean | 0.663 / 0.701 | 0.696 / 0.729 | 0.684 (+0.020) / 0.712 |
| D33's primary | 0.7295 | 0.7352 | 0.7383 |
| *Confirmation sets:* PFF, the eight games parsed for D48 | 0.6886 | 0.7236 (+0.0350, first use) | 0.7091 (**+0.0204**, second use) |
| Holdout long-hidden mean | 0.6714 | 0.7135 (+0.0421) | 0.7037 (**+0.0323**) |
| Holdout primary | 0.72761 | 0.73276 (+0.0052) | 0.73036 (+0.0027) |
| N = 1.75 / 4 / 4.5 | | +0.016 / +0.001 / +0.001 | +0.003 / +0.002 / +0.004 |
| Dead ball carried / team swap | | +0.003 / +0.018 | +0.005 / +0.001 |

- **Development:** every gate condition held (selection +0.031, PFF +0.020, composite +0.012, the largest fall of a group Metrica −0.001, conventions +0.005, +0.011 and +0.010). The three seed sets alone score 0.7178–0.7238.
- **Confirmation sets, second use:** every condition of D48's rule held, and all eight PFF games gained (+0.002 to +0.045). As the rule's disclosure says, these matches had been scored once for D48's stack, so this is weaker evidence than D48's.
- **What C is worth inside the stack:** on the same sets the stack with C is ahead of the fallback by 0.015 on PFF, 0.010 on the holdout long-hidden views and 0.002 on the holdout primary; on development by 0.012 on PFF and 0.018 on the D24 mean. The fallback keeps about three fifths of the PFF gain and three quarters of the long-hidden gain. [Development](dfl_submission/stack_recipe/development_d53.json), [confirmation](dfl_submission/stack_recipe/confirmation_d53.json)

**D53 build preparation (2026-11-13, 12:48 EST).** In scratch only; nothing was adopted and `data/submission.csv`, `Anfield.csv`, `SUBMISSION_SHA256.txt` and `release/BUILD_COMMAND` are unchanged. The build runs the D53 record through the fixed build path with the same identity checks (its networks, copied to `dfl_submission/fallback_models/` and verified; the trees; the code; its lever record; the driver locked in `stack_recipe/build_identity_d53.json` after it recomputed `development_d53.json` and `confirmation_d53.json` exactly). With direction off no attacking sign is computed.
- Two builds in the repository venv gave `769655cad55cfddd00bf2e3a82e9f4330ff7703768a73633194927aa7264a32e` with the validator's OK line each time, and `player_trial build` still gave `a56574fe…`.
- The four release builds on a copy of the working tree (two fresh macOS venvs from Homebrew Python 3.12.13 with `dfl_submission/requirements.txt`; the pinned `python:3.12.14-slim` image under `linux/arm64` and `linux/amd64`) all gave `769655ca…` and validated. Only the validator lines and the SHA-256s were read.
- So the new inference (twelve networks averaged per slot, mirror-averaged) is byte-identical on six builds across three platforms. The kick-off-aligned build of the stack with C is prepared behind a switch and has not run.

**Audit of round 12's record (2026-11-13, 13:27 EST).** A coding session recomputed every table cell, gate and stated number of round 12 from the records they cite (716 checks; `data/d48/audit_round12.md`). No pass or fail statement was wrong. It found 25 discrepancies in this file, almost all a last digit from rounding twice, and they are corrected in place above: among them D46's and D46a's rounded cells, the count of signed development halves in D50 (31 of 32), C's scores on D49's guard views (they were scored afterwards and are now cited), the range of LB's ball error (5.9–7.6 m), the number of external halves the sign rule was checked on (268, not 196), and D45's status paragraph, which had been filed under D47 and now sits under D45. The whole test suite passes on the working tree as it stands: 581 tests in 71 files.

## D54 — Both candidates on the last unused matches (2026-11-13, 13:28 EST, written before any of the four matches is read or scored)

**Question.** D53's confirmation was a second use of its matches, and Derek has to choose between it, the stack with C and D43. The four 2019–20 SkillCorner confirmation matches (3442, 3518, 3749, 4039; D6) are the only external matches no model has trained on and no step has scored. How do D43, the stack with C and the D53 fallback score on them? Measurement only: it selects nothing, changes nothing, and is one more piece of evidence for Derek's choice.

**Disclosure.** Written after D48's confirmation, the stack's refused build and D53's outcome. These are the last unused external matches, and this spends them. They are a weaker test than the PFF set was: older broadcast tracking, provider ownership only (no dead-ball label, so stoppages are not scored), with a target on about 60–72% of the timeline on the release's development matches. D14 found the release harmful as training data, and D6 found label-free adaptation flat on its development matches; neither used these four.

**What is scored.** D43 at ownership cost 10, the stack with C at cost 14 and the D53 fallback at cost 14, each as confirmed (the stack with D50's whole-half sign and its fallback; whether each half's sign agrees with its first qualifying frame is reported, not enforced). Inputs, targets and scoring are D6's: roster players and the ball, the metadata pitch, 25 Hz interpolation and the five-metre outlier limit; provider home and away possession as the target, null or unrecognised ownership excluded, ownership gaps splitting the scoring runs; spellF1 per match over its two halves, then the mean of the four matches. The primary is N = 3; N = 1.75, 4 and 4.5 and the raw provider labels are reported, with the share of the timeline that has a target.

**How it is read.** A candidate is supported on fresh matches if its N = 3 mean is at least D43's, and clearly supported if it is at least D43's plus 0.01. Per-match changes are reported. Whatever the result, the choice of what ships stays Derek's.

**Record.** `dfl_submission/stack_recipe/fresh_2019.json`, written once. Code: a new module `dfl_submission/fresh_trial.py`, reusing `skillcorner_2020.py` and the readers and scoring D6 used, and the confirmed models' inference unchanged. Scratch goes to `data/d54/`. Every D54 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D54 outcome (2026-11-13, 13:52 EST).** **Neither candidate is supported on the four fresh matches: both score below D43 there.** The reproduction check passed first (frozen D3 through this reader and scoring gave D6's row on the five development matches: 0.62374, raw 0.54793). A target covers 62–73% of each match's timeline.

| N = 3 spellF1 | 3442 | 3518 | 3749 | 4039 | Mean | Change | Raw (mean) | N = 1.75 / 4 / 4.5 (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D43, cost 10 | 0.629 | 0.649 | 0.595 | 0.644 | 0.6293 | | 0.537 | 0.594 / 0.632 / 0.626 |
| Stack with C, cost 14 | 0.625 | 0.653 | 0.592 | 0.628 | 0.6248 | **−0.0046** | 0.531 | 0.588 / 0.629 / 0.623 |
| Fallback without C (D53), cost 14 | 0.629 | 0.646 | 0.568 | 0.632 | 0.6190 | **−0.0103** | 0.528 | 0.585 / 0.622 / 0.619 |

- **By D54's reading neither is supported** (a mean at least D43's was needed). The stack is 0.005 below D43, one match up and three down, the largest fall 0.017; the fallback is 0.010 below, with one match level and one down 0.026. Every convention and the raw labels point the same way, by 0.003–0.011.
- **How much weight it carries.** Four matches, whose per-match changes spread about 0.009 (stack) and 0.012 (fallback), so the stack's deficit is about one standard error and the fallback's under two. The test is ownership only on old broadcast tracking: no stoppage is scored, and a target is missing on about a third of the timeline. It is the least like the provided match of the external sets, which are tracked from a camera that sees every player. But it is fresh, and it is the only fresh evidence the fallback has.
- **Against the rest.** The stack gained on the eight fresh PFF games (+0.035, all eight up), on long-hidden views of the DFL holdout matches (+0.042) and on the standard holdout (+0.005); the fallback, on a second use of those sets, +0.020, +0.032 and +0.003. The stack is ahead of the fallback on every set, this one included (+0.006).
- **The sign rule on real tracking.** All eight halves have a sign (|D| 3.3–4.7 m), no half needed the fallback, and the two halves of every match have opposite signs. One half (4039, second) disagrees with its first qualifying frame, which lies 64 s into the half's tracking: on tracking that does not start at a kick-off, the first-frame check misfires while the whole-half sign is plain. That is the brittleness the kick-off alignment was written for.

The last unused external matches are now spent. The choice of what ships stays Derek's, with this result in front of him. [Record](dfl_submission/stack_recipe/fresh_2019.json)

*Second adoption amendment (2026-11-13, 13:56 EST, Derek's decision, written before any further build runs on the provided match).* Derek was given the three options (the stack with C after a build whose sign rule starts at the tagged kick-off; the D53 fallback; D43), with D54's result and the recommendation: try the kick-off-aligned build of the stack, and keep D43 if it refuses again. He answered "Yes then do this".
- *The change.* In the build, a graded half's attacking sign is D50's D taken over the half's long-context frames at or after the frame of that half's Smart Tagging `Kick Off (Start)` tag, and the first qualifying frame is the first such frame at or after the tag. The build refuses unless each graded half owns exactly one such tag, paired with its half as D27 pairs it. The three fail-closed checks are unchanged and are applied to those frames. The sign still applies to every frame of the half, the frames before the kick-off included, and D27 still sets the labels before the kick-off to D. The networks, weights, levers, cost and every other value are the confirmed ones.
- *Why this form.* It is the form the rule was validated in. Every external half the rule and its first-frame check were written for starts at its kick-off, where each team stands in its own half. The organisers' FAQ.md says the provided second half is graded from its first frame while play starts at the kick-off about two minutes later (D27's premise), and the review of the build noted before the first attempt that these frames enter the sign rule (its finding 6). D54 then showed the first-frame check misfiring on external tracking that does not start at a kick-off (one half of eight), with the whole-half sign plain. On an external half, which has no frames before its kick-off, the two forms are the same rule, so the confirmed records are unchanged: the edited driver recomputed `development_t.json` and `confirmation.json` exactly before its lock was written.
- *What has been read from the provided match so far.* One refusal message from the first attempt, and nothing else. This attempt adds either the validator's OK line and a SHA-256, or one more refusal message.
- *If this build refuses on any check,* the stack does not ship and `a56574fe…` stays (Derek's decision). The D53 fallback is not shipped either: its only fresh result (D54) is negative.
- *If it passes,* the adoption steps of the first amendment follow. C2's D48 ranges stand, because they come from external views, where the alignment changes nothing.

**D48 adoption, second build attempt (2026-11-13, 14:16 EST).** **The kick-off-aligned build passed.** The edited driver first recomputed `development_t.json` and `confirmation.json` exactly and re-forecast the eight development guard halves bit for bit, and its lock was written (`stack_recipe/build_identity.json`, the kick-off form; the earlier locks are kept in `data/d48/build/`). Then, with only the validator's OK line and the SHA-256 read:
- two builds with `stack_trial build` gave `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`, each with the validator's OK line; so every sign check passed (each graded half owns one kick-off tag and has a sign, each sign agrees with its first qualifying frame from the kick-off on, and the two halves have opposite signs);
- `player_trial build` still gave `a56574fefd83af81747206e3aedcaf625cdd170e7376d4e00e5f0abee7c431e3`.
The four release builds on a copy of the working tree follow; the standing files change only after them.

**Correction (2026-11-13, 14:20 EST): 24 PFF matches are still unused.** D48's confirmation record says "no fresh PFF matches remain" and D54's says "the last unused external matches are now spent". Both are wrong about PFF. D7's split deals the 64 games to eight slots, three of them named `train`; those 24 games were pinned but never parsed, because no model may train on PFF (Derek, 12 November). Checked now with `pff.split` over the 64 pinned ids against `data/pff/parsed`: development, sets A, B and C and the untouched set are parsed (8 each), and the 24 `train` games are not. A researcher session Derek started found this. What stands: the untouched set of eight is spent (D48), and the four 2019–20 SkillCorner matches are spent (D54). What changes: the 24 games remain available as measurement and confirmation matches under the same ruling, and a rule must name which of them it uses before any is parsed.

## Note — Round 13 (2026-11-13, 14:26 EST)

The stack is being adopted (its build passed; the release builds are finishing). The goal still stands, 24 PFF games are unused (the correction above), and a researcher session's brief (`data/research/RESEARCH_BRIEF_2026-11-13.md`) points at one thing no view simulates. The organisers write that the ball "is absent on 41% of graded frames, and disproportionately on the interesting ones — crowded boxes, aerial duels, tackles" (FAQ.md) and that "occlusions, crowded boxes and balls in the air all lose it" (DATA.md). Every mask so far, D49's long runs included, hides the ball at random times. Round 13 first measures whether that matters (D55) and trains for it only if it does. Everything standing stays: nothing is fitted on the provided tracking, constants come from judgement or external matches, PFF is measurement and confirmation only, C3's bar holds, and the Sunday freeze holds. The adopted stack is the reference from here on.

## D55 — The ball hidden where it really goes missing: a measurement on guard views (2026-11-13, 14:26 EST, written before any D55 view is built or anything is scored)

**Question.** Do D43 and the stack lose more when the ball is hidden in the situations the organisers name (in the air, in a crowd, around a change of possession) than when the same amount is hidden at random? Measurement only; it selects nothing and changes nothing in the submission.

**Disclosure.** Written after D48's confirmation and D54. The situations come from the organisers' published documents, not from the provided tracking. The constants below are set by judgement and are not tuned. DFL's tracking gives the ball's height, which this repository's reader has dropped until now; no other external source has it, so the views are on DFL WPY only.

**Views of DFL WPY,** each built as its masked development view is (the same coordinate noise and player dropout, new fixed seeds) with the hide process replaced:
- **(a) air:** the ball is hidden on every frame where its height is above 1.5 m;
- **(b) crowd:** the ball is hidden on every frame where at least three players are within 3 m of it;
- **(c) change:** the ball is hidden from 1 s before to 2 s after each ownership change of the N = 3 targets;
- **(d) air or crowd:** the union of (a) and (b), the two mechanisms that need no label;
- **(e) duplicates:** the masked development view with a duplicate track: in runs of 2–5 s covering about a tenth of the half, the player nearest the ball is given a second track of the same team within 1 m of the first.
For each of (a)–(d) a **matched random view** hides the same hidden runs (the same lengths, so the same visibility) at positions drawn at random in the same half, without overlap. Dead-ball frames are treated as in the masked development view.

**Scored.** D43 at cost 10 and the adopted stack at cost 14, N = 3 spellF1 on each view and its matched random view; the visibility and hidden-run statistics of each view; owner agreement on hidden frames.

**Pipeline checks.** The new reader's ball x and y equal the existing reader's exactly on every WPY frame; D43 and the stack reproduce their development records on the standing views.

**How it is read.** For the stack, the gap is its score on a view minus its score on that view's matched random view. If the gap on (d) is −0.02 or lower, hiding by situation is a real weakness and D56 will train with such views, confirmed on fresh PFF games; otherwise no training follows from D55. (c) and (e) are reported for understanding; a fall of 0.02 or more on (e) against the masked development view would call for its own rule.

**Record.** `dfl_submission/stack_recipe/situation_views.json`, written once. Code: new modules `dfl_submission/situation_views.py` (the height-aware DFL reader path and the views) and `dfl_submission/situation_trial.py`; existing modules stay byte-identical. Scratch goes to `data/d55/`. Every D55 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D48 adopted (2026-11-13, 14:28 EST). The submission is now `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`.**
- *Release builds on a copy of the working tree:* two fresh macOS venvs (Homebrew Python 3.12.13, `dfl_submission/requirements.txt`) and the pinned `python:3.12.14-slim` image under `linux/arm64` and `linux/amd64` all gave `f6d7b293…` with the validator's OK line. With the two builds in the repository venv that is six identical builds on three platforms. Only the validator lines and the SHA-256s were read.
- *Standing files:* `data/submission.csv` and `Anfield.csv` are copies of the built file (SHA-256 checked, not opened); `SUBMISSION_SHA256.txt` names it; `release/BUILD_COMMAND` is `-m dfl_submission.stack_trial build --out {out}`. `player_trial build` still writes `a56574fe…`, which stays the first fallback.
- *C1 and C2:* C1's record is amended (above, under C1). C2 is re-declared for D48 with `output_check_ranges_d48.json`, which `check` reads by default; the ranges come from external views, where the kick-off alignment changes nothing, so they stand.
- *Still to do:* the documents (README, CONSTANTS.md, the technical document, from the drafts in `data/d48/docs_draft/A/`); then Derek: approve a commit so that `release/check.sh` can run on it, run C2 once in a terminal outside the AI session, re-read the technical document, and decide on the push.
- *The evidence this rests on, in one place:* fresh PFF (eight games) +0.035 with every game up; long-hidden views of the DFL holdout matches +0.042; the standard holdout +0.005; and, against, the four 2019–20 SkillCorner games −0.005 (about one standard error), on ownership-only labels and the tracking least like the provided match's. The D53 fallback is recorded and not shipped.

**D55 outcome (2026-11-13, 15:03 EST).** **Hiding the ball by situation is a real weakness: the stack's gap on the air-or-crowd view is −0.028 (−0.02 was the line), so D56 follows.** The checks passed: the height-aware reader's ball x and y equal the existing reader's on every WPY frame (every ball record has a height; dead frames are at 0; 14% and 25% of the two halves' live frames are above 1.5 m), and D43 and the stack reproduced their standing WPY scores bit for bit.

| View of DFL WPY (N = 3 spellF1) | Live visibility | D43: view / matched random / gap | Stack: view / matched random / gap | Stack, owner agreement on hidden frames: view / matched random |
|---|---:|---|---|---|
| (a) air, height above 1.5 m | 0.81 | 0.731 / 0.769 / −0.039 | 0.737 / 0.795 / **−0.058** | 0.813 / 0.937 |
| (b) crowd, three players within 3 m | 0.95 | 0.735 / 0.748 / −0.012 | 0.748 / 0.750 / −0.002 | 0.795 / 0.832 |
| (c) change, 1 s before to 2 s after | 0.84 | 0.684 / 0.752 / −0.068 | 0.694 / 0.768 / **−0.073** | 0.663 / 0.911 |
| **(d) air or crowd** | 0.78 | 0.723 / 0.754 / −0.031 | 0.746 / 0.774 / **−0.028** | 0.821 / 0.928 |
| (e) duplicates, against the masked view | 0.61 | 0.722 / 0.722 / 0.000 | 0.724 / 0.728 / −0.004 | |

- **Where the ball is hidden matters more than how much.** The air view hides 19% of live frames and the stack scores 0.737 on it; the standing masked view hides 39% of live frames at random and the stack scores 0.728. On frames hidden while the ball is in the air the stack names the owner right 81% of the time, against 94% on frames hidden at random; around a change of possession 66% against 91%.
- **The crowd rule alone hides little** (5% of live frames, in runs of 0.7 s) and costs the stack nothing; the weakness is the ball in the air. **Duplicate tracks cost nothing** (−0.004 for the stack, none for D43), so no rule follows from (e).
- **A caution on the size of the gaps.** A matched random view hides the same runs anywhere in the half, dead ball included, so it hides about half as many live frames as its situation view (air: 7,895 against 15,445). The gaps therefore overstate the effect of *where*; the comparison with the standing masked view in the first bullet does not have this flaw and points the same way.
- **The stack did not close this gap.** Its gap on (a) is larger than D43's (−0.058 against −0.039) because it gained more on the random view than on the air view: the long-hidden training helped with hiding at random times, not with hiding in the air. [Record](dfl_submission/stack_recipe/situation_views.json)

## D56 — Training views that hide the ball in the air and in a crowd (2026-11-13, 15:03 EST, written after D55's outcome and before any D56 view is built, any network is trained or anything is scored)

**Question.** D55 shows the stack is weak where the organisers say the provided ball goes missing. Does adding training views that hide the ball by situation raise spellF1 on such views and on real automatic tracking (PFF), without harm elsewhere?

**Disclosure.** Written after D55, on the same development match (WPY) that D55's views use, so the situation view is no longer an untouched guard for this step; PFF, whose ball is lost by a real tracker, is the judge that was not built by us. The constants are set by judgement from the organisers' published text and D55's definitions, and are not tuned. The reference is the adopted stack.

**Situation hide (training).** On the frames of a half, the ball is hidden while its height is above 1.5 m (DFL only: no other source has the height) or at least three players are within 3 m of it; each such hidden run is extended by a delay drawn uniformly from 0 to 1 s, for a tracker's time to find the ball again; on top, the standing two-state random process (`augment.wide_masking`) hides the ball with a visibility drawn per half from 0.6 to 0.95.

**Candidates,** each the adopted stack's recipe (D49's views, attacking-frame tokens, the y mirror) on its first seed set only (four networks), with one more view per training match in each set:
- *DFL training matches:* the set's own view construction with `wide_masking`'s coordinate noise and player dropout and the situation hide;
- *SkillCorner training matches:* the set's own SkillCorner construction (the original set's provider tracking; the augmented set's handled dead ball) with the situation hide's crowd rule, delay and random part (no height).
The existing records of D49's folders must be reproduced byte for byte, and the new views added.
- **S:** 8,100 steps.
- **SL:** 10,800 steps, which gives each sequence about as many visits as 8,100 steps gave D49's 80 records.

**Base.** The adopted stack's first seed set alone (D52's T networks), scored on the same views.

**Scored** at ownership cost 8, 10, 12 and 14 on D47's eight groups, D49's two long-hidden guard views and D55's (d) view of WPY. The round-13 selection score is the mean of D47's composite, the long-hidden guard mean and the (d) view.

**A candidate holds if,** at its best cost by the selection score: the selection score is at least the base's best plus 0.008; the PFF development mean is at least the base's at its best cost; and no group falls by more than 0.01 against the base. If both hold, the higher selection score is taken.

**If one holds,** the stack is retrained with it (three seeds), scored on development against the adopted stack with the same three conditions, and then confirmed once on twelve fresh PFF games (set D: of the 24 unparsed games, the twelve dealt to the first and second `train` slots of D7's split, parsed only for that step) and the holdout: it replaces the adopted stack if set D's N = 3 mean rises by at least **+0.01** (C3), the holdout primary and the holdout long-hidden mean each fall by no more than 0.01, and N = 1.75, 4 and 4.5, dead ball carried and team swap each fall by no more than 0.01. The other twelve unparsed games stay in reserve. **If neither holds,** the adopted stack stands.

**Records.** `dfl_submission/stack_recipe/d56_*.json`, each written once. Code: new modules only (`dfl_submission/situation_train.py` for the training views and the prepare step, `dfl_submission/situation_fit.py` for the candidates), reusing `situation_views.py`, `hidden_views.py`, `stack_set.py`, `stack_trial.py` and `length_trial.py` unchanged. Scratch goes to `data/d56/`. Every D56 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

*Correction to D55 and D56 (2026-11-13, 16:06 EST, while D56's first trainings run; nothing of D56 is scored yet).* Both say that no external source other than DFL gives the ball's height. That is wrong: the researcher session found that the 2024–25 SkillCorner tracking carries a ball height on every detected frame (7–18% of detected frames above 1.5 m on the training matches, comparable to DFL's live frames), although its documentation does not say how the height is made and this repository's SkillCorner reader keeps x and y only. D56's rule is not changed by this: its SkillCorner views keep the crowd rule, the delay and the random part, as written before anything was scored. Using SkillCorner's height for the air rule, and hiding around changes of possession (D55's largest gap), would each need a rule of their own after D56's outcome.

*Disclosure for Derek (2026-11-13, 16:07 EST): PFF's part in round 12's selection.* Derek's ruling of 12 November makes PFF "a veto and a confirmation source". In round 12 the orchestrator used the eight PFF development games as more than a veto: their mean is one of the eight groups of D47's composite, D49's carry rule could pass on the PFF mean alone, D48's gate required a PFF gain, and the ownership cost (14) and the 8,100 steps were chosen with selection scores that contain the composite. No model was trained on PFF and the build reads none of it, but the shipped cost and training length were selected on scores that PFF is part of. The technical document now says so (sections 3, 7 and 10, replacing "no value in the submission comes from it"). Whether that is within the ruling and PFF's terms is Derek's judgement; the alternative wording and the numbers without PFF can be produced from the records if he wants them.

*Documents (2026-11-13, 14:31 EST).* The README, CONSTANTS.md, PROVENANCE.md and the technical document carry outcome A; `Anfield.md` is an exact copy of the technical document; 594 of 594 tests pass in 72 files. One placeholder remains in four places: the release check on a commit, pending Derek's approval of a commit.

## D57 — The air rule on the SkillCorner training matches too (2026-11-13, 16:19 EST, written while D56's last candidate trains, before anything of D56 or D57 is scored)

**Question.** D56's situation views hide the ball in the air on the four DFL training matches only, because this repository's SkillCorner reader drops the ball's height. The 2024–25 SkillCorner tracking has a height on every detected frame (the correction above), and on both providers the ball is in the air about twice as often in the seconds around a change of possession as elsewhere (the researcher session's count on the training matches: DFL 33% against 15%, SkillCorner 22% against 9%). Does applying the air rule to the twelve SkillCorner training matches as well do better than D56's S?

**Disclosure.** Written after D55's outcome and the researcher session's read-only counts on the training matches, before any score of D56. SkillCorner's documentation does not say how its ball height is made. The constants are D56's.

**Candidate SZ.** D56's S (8,100 steps, first seed set), with one change: in the new SkillCorner views the ball is also hidden on detected frames whose height is above 1.5 m, the height read from the provider's tracking and resampled to 25 Hz as x and y are. The crowd rule, the delay and the random part are D56's. The reader path is new; the existing records of D56's folders must be reproduced byte for byte.

**Scored** as D56's candidates are, plus, reported only, an air view of the four SkillCorner development matches (their tracking with the ball hidden on detected frames above 1.5 m), which D55's WPY view cannot give: a second view of the weakness, on another provider. The base, S and SL are scored on it too.

**SZ holds** under D56's three conditions against the base (selection score at least the base's best plus 0.008; PFF development mean at least the base's; no group more than 0.01 lower). Of the candidates of D56 and D57 that hold, the one with the highest selection score is taken to D56's retraining and its one confirmation on PFF set D; D56's rule for that confirmation is unchanged.

**Records.** `dfl_submission/stack_recipe/d57_*.json`, each written once. Code: new modules only (`dfl_submission/skill_height.py` for the reader path and the views, `dfl_submission/height_fit.py` for the candidate), reusing D56's modules unchanged. Scratch goes to `data/d57/`. Every D57 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

*Amendment to D56 and D57 (2026-11-13, 16:23 EST, before any candidate of either is scored and before any of the 24 games is parsed).* Two corrections, both prompted by the researcher session.
1. *Set D.* D56 says "twelve fresh PFF games … the twelve dealt to the first and second `train` slots", which is a contradiction: D7 deals eight games to each slot, so those two slots hold sixteen. **Set D is the sixteen games of the first and second `train` slots** (3816, 3824, 3832, 3840, 3848, 3856, 10506, 10514 and 3817, 3825, 3833, 3841, 3849, 3857, 10507, 10515), and **set E is the eight games of the third** (3818, 3826, 3834, 3842, 3850, 3858, 10508, 10516), kept in reserve. Both spread over the whole range of game ids.
2. *When set D is spent.* On the eight games of D48's confirmation the per-game changes spread 0.013, so on sixteen games the standard error of a mean change is about 0.003–0.004, and a candidate whose true gain equals C3's +0.01 would pass only about half the time. PFF development has forecast the fresh gain closely (D48: +0.033 on development, +0.035 fresh). So the three-seed retrain goes to set D only if, besides D56's three conditions against the adopted stack, **its PFF development mean is at least the adopted stack's plus 0.010**. Below that the confirmation is not spent and the adopted stack stands; the orchestrator may then write a further rule.

*Addition to the disclosure on PFF's part in round 12's selection (2026-11-13, 16:27 EST; arithmetic on the records, checked by the orchestrator for the cost).* The researcher session recomputed round 12's shipped choices without PFF, taking D47's composite as the mean of its other seven groups (the long-hidden guard has no PFF in it). None changes. The ownership cost: the stack's selection score without PFF is 0.7273, 0.7298, 0.7331 and 0.7349 at cost 8, 10, 12 and 14, still highest at 14, and PFF alone is lowest at 14 (0.6958), so PFF pulled slightly against the shipped cost. The 8,100 steps: T's gain without PFF is +0.0089 against its bar of +0.008. L: the long-hidden guard rose +0.040 against the +0.015 that carried it without PFF. So in every shipped choice PFF acted as a veto and a confirmation, which is what Derek's ruling allows; the wording in the technical document stays as it is unless Derek prefers otherwise.

**D56 outcome (2026-11-13, 17:05 EST).** **Neither candidate holds.** D49's records were reproduced byte for byte and 32 situation views added to each set; the base reproduced D52's T record. In the new views the DFL halves have the ball hidden by height on 10.8% of frames, the SkillCorner halves on none (crowd 2.5–3.1%, delay 2.0–4.0%, the random part 23–25%).

| First seed set alone | Best cost | Round-13 selection score | Composite | Long-hidden guard mean | (d) air or crowd | (a) air | PFF | D33's primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base (the adopted stack's first seed set) | 14 | 0.7378 | 0.7299 | 0.732 | 0.751 | 0.740 | 0.695 | 0.7344 |
| S (8,100 steps) | 10 | 0.7382 (+0.0004) | 0.7227 | 0.726 | 0.766 | 0.739 | 0.694 | 0.7307 |
| SL (10,800 steps) | 12 | 0.7402 (+0.0024) | 0.7267 | 0.736 | 0.758 | 0.750 | 0.695 | 0.7345 |
| SL at the base's cost | 14 | 0.7376 | 0.7327 | 0.737 | 0.743 | 0.755 | 0.697 | 0.7396 |

- **S failed all three conditions** (selection +0.0004 against +0.008; PFF −0.001; DFL clean and team swap −0.019 against the base at its best cost). **SL failed two** (selection +0.0024; team swap −0.018) and met the PFF condition by +0.0004.
- **What it shows.** The situation views lift the view they imitate (the (d) view +0.007 to +0.015 at the candidates' best costs) and move nothing else: PFF, whose tracker really loses the ball in the air, is unchanged, and the differences elsewhere are the size of a seed set's spread. The air rule reached only the eight DFL halves of the 32 new ones, so this was a small dose; D57's SZ applies it to all 32. [S](dfl_submission/stack_recipe/d56_S.json), [SL](dfl_submission/stack_recipe/d56_SL.json), [decision](dfl_submission/stack_recipe/d56_decision.json)

**D57 outcome (2026-11-13, 17:33 EST).** **SZ does not hold, so nothing is taken from D56 or D57: the adopted stack stands and PFF set D is not spent.** The height-aware SkillCorner reader reproduced the stored ball exactly on every training half, and D56's records were reproduced byte for byte with only the SkillCorner situation views changed (the ball hidden by height on 4–10% of their frames).

| First seed set alone | Cost | Round-13 selection score | Composite | Long-hidden guard mean | (d) air or crowd | PFF | D33's primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 14 (its best) | 0.7378 | 0.7299 | 0.732 | 0.751 | 0.695 | 0.7344 |
| SZ | 8 (its best) | 0.7392 (+0.0014) | 0.7207 | 0.730 | 0.767 | 0.707 | 0.7269 |
| SZ | 14 | 0.7365 | 0.7313 | 0.737 | 0.741 | 0.701 | 0.7371 |
| Base, PFF at cost 8 / 10 / 12 / 14 | | | | | | 0.7005 / 0.6989 / 0.6967 / 0.6946 | |
| SZ, PFF at cost 8 / 10 / 12 / 14 | | | | | | 0.7074 / 0.7052 / 0.7033 / 0.7011 | |

- **The rule's conditions:** selection +0.0014 (+0.008 needed); PFF +0.013 against the base at its best cost; DFL clean −0.037 against the base at its best cost. Two of three failed.
- **A flaw in the rule's third condition, noted for honesty and not acted on:** it compares the candidate at its best cost with the base at its best cost, so SZ at cost 8 is set against the base at cost 14 and the fall on DFL clean is mostly the cost. At equal cost DFL clean changes by −0.027, −0.010, +0.003 and +0.002 (cost 8 to 14). The first condition fails at every cost, so the outcome does not hang on this.
- **What it shows.** With the air rule on all 32 new halves, PFF rises by +0.006 to +0.007 at every cost: a steady gain on the tracker that really loses the ball in the air, where D56's two candidates were flat. It is about half of what a confirmation on set D would need (the amendment's +0.010 on PFF development), and nothing else moves by more than a seed set's spread. Round 13's situation views are a real but small lever, and it stops here. [SZ](dfl_submission/stack_recipe/d57_SZ.json)

## D58 — Where the adopted stack loses spellF1 now, and what kind of stoppage error costs most (2026-11-13, 17:33 EST, written before anything is decoded or scored for D58)

**Question.** Rounds 12 and 13 worked on ownership where the ball is hidden. D46 also showed that true stoppages are worth +0.06 to +0.10 where the dead ball is realistic (dead ball carried, PFF, the D24 views), and nothing since has aimed at them. For the adopted stack: how much is still lost in stoppages on those groups, and is it stoppages the model misses or stoppages it invents? Measurement only; it selects nothing and changes nothing in the submission.

**Candidate.** The adopted stack at ownership cost 14, decoded from the caches its development record was scored from (D47's eight groups, D49's two long-hidden guard views, the three D24 views separately).

**Pipeline check.** The decode must reproduce the cost-14 row of `stack_recipe/development_t.json` exactly, or D58 stops without a record.

**Measured,** per group, against the N = 3 targets:
1. D21's four measurements, unchanged (spell detection, the true-stoppage and true-ownership oracles, boundary offsets, recall by spell length).
2. *Stoppages by length* (under 3 s, 3–10 s, 10–30 s, over 30 s): of the true stoppages, the share found (a predicted stoppage covering at least half of it); of the predicted stoppages, the share that are real (at least half inside a true stoppage); and for the invented ones, the share whose ball is hidden on at least 80% of their frames.
3. *Two one-sided oracles:* the true-stoppage oracle applied only where it adds a stoppage the model missed, and only where it removes one the model invented; mean spellF1 for each.
4. Stop and restart offsets (median and quartiles) for the stoppages that are found.

**How it is read.** On the groups with a realistic dead ball (dead ball carried, PFF, the D24 views, the long-hidden guard views), if the true-stoppage oracle is worth at least 0.04 on average and one side (missed or invented) carries at least two thirds of that gain, a stoppage rule is written for that side; otherwise round 13 ends with the adopted stack.

**Record.** `dfl_submission/diagnosis_recipe/d58.json`, written once. Code: a new module `dfl_submission/diagnosis_stack.py`, reusing `diagnosis.py` and `diagnosis_d43.py` unchanged. Every D58 command runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

*Amendment to D58 (2026-11-13, 17:36 EST, before anything is decoded).* The coding session asked how the one-sided oracles are defined. They are at the level of whole stoppages, as point 2 counts them: the "missed" side puts the true-stoppage oracle's output on the frames of true stoppages that no predicted stoppage covers by at least half, and the "invented" side on the frames of predicted stoppages that are not at least half inside a true stoppage. Corrections to the edges of stoppages that were found fall on neither side, so the two shares need not add up to the full oracle's gain; the remainder is reported as "edges". The frame-level split (all live-to-dead changes against all dead-to-live changes, which add up exactly) is reported as supplementary and is not what D58 is read on.

*Second amendment to D58 (2026-11-13, 17:37 EST, before anything is decoded).* "On average" in D58's reading means the mean over four families, so that the views built from the Metrica games do not count five times: dead ball carried; PFF; the three D24 views averaged into one; the two long-hidden guard views averaged into one. Each side's share is its mean gain over those four divided by the full oracle's mean gain over the same four. The plain mean over the seven views is reported as supplementary.

## Note — Derek hands over the remaining steps (2026-11-13, 17:42 EST)

Derek's instruction, in full: "You may commit round 12 if that is what you recommend. From now on only do what you recommend. Do not leave anything for me." From here the orchestrator takes the steps that were held for him, on its own recommendation, and records each one here: committing, running `release/check.sh` on the commit, running C2, the final reading of the technical document, and pushing. What stays his because only he can do it: submitting the organisers' form with `Anfield.csv`, `Anfield.md` and the repository URL. The rules that protect the provided data do not change: nothing of the provided match, its outputs or its statistics is opened or printed in an AI session beyond a validator's OK line, a SHA-256 and a check's one-line verdict.

*C2's procedure, amended (same time, before C2 is run on `f6d7b293…`).* C2 was to be run by Derek in a terminal outside the AI session, reporting PASS or FAIL only. Under his instruction the orchestrator runs it once instead, after `shasum -a 256 -c SUBMISSION_SHA256.txt`, and reads exactly what he would have reported: the word PASS, or FAIL with the names of the statistics outside their ranges. The check prints no number, and none is read. Everything else in C2 stands: the ranges are D48's (`output_check_ranges_d48.json`), and after a FAIL only a pipeline investigation follows; models, recipes and thresholds do not change.

**D58 outcome (2026-11-13, 17:46 EST).** **No stoppage rule follows: round 13 ends with the adopted stack.** The decode reproduced the cost-14 row of `development_t.json` exactly.

| Family (realistic dead ball) | Stack | True stoppages | Gain | From stoppages missed | From stoppages invented | From the edges of found ones |
|---|---:|---:|---:|---:|---:|---:|
| DFL dead ball carried | 0.657 | 0.734 | +0.076 | +0.015 | 0.000 | +0.061 |
| PFF | 0.696 | 0.775 | +0.079 | +0.024 | +0.014 | +0.041 |
| D24 views (mean of three) | | | +0.060 | +0.025 | +0.012 | +0.023 |
| Long-hidden guard views (mean of two) | | | +0.043 | +0.015 | +0.007 | +0.021 |
| **Mean of the four families** | | | **+0.065** | **+0.020 (31%)** | **+0.008 (13%)** | **+0.037 (57%)** |

- **By D58's reading** the true stoppages are worth enough (+0.065 against the 0.04 line), but neither side carries two thirds of it: missed stoppages 31%, invented ones 13%. So no rule is written for either side.
- **Most of the loss is in when a found stoppage starts.** With the dead ball carried the model calls the stop a median of 25 frames late (a quarter of them more than 4 s late): the ball is still moving with a player, and the model waits. On PFF it calls the stop a median of 17 frames early (a quarter more than 1.7 s early), and predicts 917 stoppages for 748 true ones, 230 of them under 3 s of which 32 are real: there the ball vanishes and the model stops play. The two errors point in opposite directions, so no single shift or cost corrects both, and DATA.md does not say which behaviour the provided tracking has during a dead ball. Restarts are timed well everywhere (medians within 2 frames).
- **Ownership is still the larger loss** on the standard views (true ownership +0.144, true stoppages +0.025 on the standard combined; D43: +0.149 and +0.026).
- The frame-level split, supplementary: of the oracle's frame changes, live-to-dead carry 66% and dead-to-live 33% on the same families. [Record](dfl_submission/diagnosis_recipe/d58.json)

## Note — Round 13 ends (2026-11-13, 17:46 EST)

Round 13 measured two things and adopted nothing. D55: the stack is weaker where the ball is hidden in the air than where it is hidden at random. D56 and D57: training views that hide it there raise PFF by +0.006 to +0.007 at best and no candidate met its gate. D58: stoppages still cost about 0.065 on the realistic dead-ball views, mostly in when a stoppage starts, late on one kind of tracking and early on the other. The submission stays D48's stack (`f6d7b293…`). PFF sets D (sixteen games) and E (eight) are unparsed and unused. The search is closed by the orchestrator under Derek's instruction of 17:42 EST: what remains is the commit, the release check on it, the final reading of the documents and the push.

**Release check of `49cec0d` (2026-11-13, 18:06 EST). Passed.** Rounds 12 and 13 were committed as `49cec0d` at 17:48 EST (668 of 668 tests on the working tree first; the commit was scanned for data files, CSVs, internal documents, home paths and credentials, by the orchestrator and independently by the researcher session: none). `release/check.sh --bundle …` then ran on that commit: a fresh clone was rebuilt twice on macOS (fresh venvs, Homebrew Python 3.12.13, `dfl_submission/requirements.txt`) and in `python:3.12.14-slim@sha256:f77ac9e4…` under `linux/arm64` and `linux/amd64`; all four builds gave `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d` with the validator's OK line, and 668 of 668 tests passed in the clone (79 files). Only the validator lines and the SHA-256s were read.

*Documents, final (same time).* The README, CONSTANTS.md, PROVENANCE.md and the technical document carry round 13's outcome (rows D55–D58, two more limitations), C2's PASS, the hand-over and the release check; no placeholder remains. A last reading of the technical document as an organiser would read it found seven points, all corrected: the opening claim now says that some constants were set by judgement; the claim that every decision had fixed gates now names its exceptions; the sentence on the blend's 238 recipes is tied to the first ensemble; "Derek" is introduced as the team's lead and the decision times carry their date; the heading "The one correctness check" is now "Correctness checks"; the sentence that match data is out of any AI tool's reach now says what an AI session does read (a validator's OK line, a SHA-256, a check's verdict or refusal); and section 7 discloses that the AI session ran C2. `Anfield.md` is an exact copy of the technical document.

## Note — Round 14 (2026-11-13, 18:15 EST)

The stack is committed, release-checked and pushed (`1ddaeb7`). The goal still stands, Derek has left the decisions to the orchestrator, and one part of the adopted model has not been touched since D3: its trees. Half of the stack's ownership probability comes from them. They read summaries built around the ball, and they were fitted before any long hidden-ball view existed, while rounds 12 and 13 showed the networks, which read the players, carry what is known when the ball is hidden. Round 14 tests that one lever (D59) and nothing else. It needs no network training. Everything standing stays, the adopted stack is the reference, the Sunday freeze holds, and if D59 does not pass the search closes for good.

## D59 — The trees' share where the ball is hidden, and trees refitted on the long-hidden views (2026-11-13, 18:15 EST, written before any D59 code exists, any tree is refitted or anything is scored)

**Question.** Does lowering the trees' share of the ownership probability on frames where the ball is unseen, or refitting the trees on D49's views, raise spellF1 on hidden-ball and real automatic tracking without harm elsewhere?

**Disclosure.** Written after rounds 12 and 13. D35 and D40 re-selected one tree share for all frames (D15's and D36's networks) and found 0.5 best; neither tried a share that depends on whether the ball is seen, and neither had networks trained on long hidden runs. Development has been reused by every step.

**Candidates.** The adopted stack's twelve networks, unchanged, with D43's cutoff, stoppage cost and discount:
- **H(w):** the trees' share of the ownership probability is 0.5 on frames where the ball is seen and w on frames where it is not, w ∈ {0.25, 0}. The stoppage mix is unchanged.
- **R:** both sets' trees (ownership and stoppage) refitted exactly as D3 fitted them (`long_context.fit_head`, the same parameters and weights) on D49's prepared sequences, which add the long-hidden views; share 0.5 everywhere.
- **RH(w):** R with H(w), w ∈ {0.25, 0}.
Each at ownership cost 12, 14 and 16.

**Pipeline checks.** With D3's trees and a share of 0.5 everywhere the scorer reproduces the cost-14 row of `stack_recipe/development_t.json` exactly. The refit, run on the sequences D3's trees were fitted on, reproduces D3's trees byte for byte before it is run on D49's.

**Scored** on D47's eight groups, D49's two long-hidden guard views and D33's conventions, with D55's (d) view and PFF's visibility figures reported. The selection score is D48's (the mean of the composite and the long-hidden guard mean; the adopted stack at cost 14: 0.7326).

**A candidate holds if,** against the adopted stack at cost 14: its selection score is at least 0.008 higher; its PFF development mean at least 0.010 higher (the amendment of 16:23 EST on when set D is spent); the composite is not lower; no group falls by more than 0.01; and N = 1.75, 4 and 4.5 each fall by no more than 0.01. Of those that hold, the highest selection score is taken.

**Confirmation, once,** if one holds: against the adopted stack, on PFF set D (the sixteen games named in the amendment of 16:23 EST, parsed only for this step) and the holdout. It replaces the adopted stack if set D's N = 3 mean rises by at least **+0.01** (C3), the holdout primary and the holdout long-hidden mean each fall by no more than 0.01, and N = 1.75, 4 and 4.5, dead ball carried and team swap each fall by no more than 0.01. **If it passes,** the adoption steps of D48 follow (a build switch with the identity checks, two builds, the release check, C2 re-declared, the records, a commit and a push). **If none holds or the confirmation fails,** the adopted stack stands and the search is closed.

**Records.** `dfl_submission/stack_recipe/d59_*.json`, each written once. Code: new modules only (`dfl_submission/tree_share.py`, `dfl_submission/tree_share_trial.py`); existing modules stay byte-identical. Scratch goes to `data/d59/`. Every D59 command except a build runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

*Amendment to D59 (2026-11-13, 18:18 EST, before any member output is cached or anything is scored).* The researcher session read the saved trees' feature lists: over 90% of their gain is in ball-relative columns, many of them means over a few seconds or the last and next sighting, so on a hidden frame the trees still know roughly where the ball was across a short gap and almost nothing deep inside a long one. H(w) switches on the frame's own visibility and so also removes the trees in short flickers, where they still help. Two candidates are added, and nothing else changes:
- **HT(w):** the trees' share of the ownership probability is w on frames whose nearest ball sighting, earlier or later in the half, is more than 3 s away, and 0.5 on every other frame; w ∈ {0.25, 0}. The 3 s is set by judgement from the windows the trees' strongest columns average over, and is not tuned.
- **RHT(w):** R with HT(w), w ∈ {0.25, 0}.
The grid is therefore H(0.25), H(0), HT(0.25), HT(0), R, RH(0.25), RH(0), RHT(0.25), RHT(0), each at ownership cost 12, 14 and 16. The rule for what holds and the confirmation are unchanged.

*Correction to the round-14 note and D59 (2026-11-13, 18:58 EST, before anything is scored).* The note says half of the stack's ownership probability comes from the trees. It is a quarter. Each model set blends its trees with one of its two network slots at 0.5, and that blend is then mixed at 0.5 with the other slot (the seed-97 networks, phase-averaged), so the final ownership probability is ¼ trees, ¼ the first slot's networks and ½ the second slot's. The researcher session checked this numerically on the code D59's scorer uses. D59's shares are therefore shares of the first blend: H(w), HT(w) and their R forms move the trees' part of the final probability on the affected frames from 0.25 to 0.5 × w, that is to 0.125 or 0. The candidates, the grid and the rule are unchanged; only the wording was wrong.

**D59 outcome (2026-11-13, 19:25 EST).** **None of the nine candidates holds at any cost. The adopted stack stands, PFF set D is not spent, and the search is closed.** Both checks passed: with D3's trees and the standing mix the scorer reproduced the cost-14 row of `development_t.json` exactly from its cached members (68 halves), and the refit reproduced D3's trees byte for byte on their own sequences before it ran on D49's.

| Candidate, at its best cost by selection score | Selection score, change | PFF, change | Composite, change | Long-hidden guard mean, change | Largest fall of a group |
|---|---:|---:|---:|---:|---|
| Adopted stack, cost 14 | 0.7326 | 0.6958 | 0.7283 | 0.7369 | |
| H(0.25): trees halved on unseen frames | +0.0030 | +0.0019 | +0.0026 | +0.0034 | Metrica −0.001 |
| H(0): trees removed on unseen frames | −0.0006 | +0.0055 | −0.0001 | −0.0011 | team swap −0.004 |
| HT(0.25): halved beyond 3 s from a sighting | +0.0019 | +0.0006 | +0.0001 | +0.0038 | none |
| HT(0): removed beyond 3 s from a sighting | +0.0014 | −0.0020 | +0.0019 | +0.0008 | PFF −0.002 |
| R: trees refitted on D49's views | −0.0006 | +0.0032 | −0.0017 | +0.0005 | DFL masked −0.007 |
| RH(0.25) | +0.0011 | +0.0037 | +0.0019 | +0.0003 | Metrica −0.001 |
| RH(0) | +0.0005 | +0.0095 | +0.0005 | +0.0005 | DFL masked −0.007 |
| RHT(0.25) | −0.0011 | +0.0036 | −0.0015 | −0.0006 | DFL masked −0.007 |
| RHT(0) | +0.0008 | +0.0034 | −0.0016 | +0.0032 | DFL masked −0.007 |

- **The rule:** +0.008 on the selection score and +0.010 on PFF development were needed. The best selection gain is +0.0030 and the best PFF gain +0.0095; no candidate meets either.
- **What it shows.** Taking the trees out of the ownership probability on unseen frames (H, RH) raises PFF at every cost, by up to +0.010 with the refitted trees, which is the direction their dependence on the ball predicts; keyed to 3 s from a sighting (HT) it does not. Nothing else moves by more than a seed set's spread. The trees are a quarter of the final ownership probability (the correction above), and they cost little where the ball is hidden. [Scores](dfl_submission/stack_recipe/d59_scores.json), [amended scores](dfl_submission/stack_recipe/d59_scores_amended.json), [decisions](dfl_submission/stack_recipe/d59_decision_amended.json), [refit](dfl_submission/stack_recipe/d59_refit.json)

## Note — The search is closed (2026-11-13, 19:25 EST)

Round 14 tried its one lever and adopted nothing. The submission is D48's stack, `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`, release-checked on `49cec0d`. Rounds 13 and 14 leave three small, consistent signals on PFF that no rule could confirm at C3's bar (the air rule on every training match +0.006 to +0.007; refitted trees removed on unseen frames +0.010; 10,800 steps about +0.002 on the selection score); they are recorded for whoever continues, not combined untested. PFF sets D and E are unparsed. The orchestrator closes the search here under Derek's instruction of 17:42 EST; what remains is recording this round in the documents, a commit and a push.

## Note — Round 15, one combination (2026-11-13, 20:04 EST)

The note above closed the search. The orchestrator reopens it for one step, and says why: the goal Derek set is still open, he has left the decisions to the orchestrator, sixteen fresh PFF games are unused, and rounds 13 and 14 left two signals that point the same way on real automatic tracking and were each too small alone: the air rule on every training match (D57: PFF +0.006 to +0.007 at every cost) and the refitted trees taken out of the ownership blend on unseen frames (D59: PFF +0.0095). Neither moved the selection score, which was built for round 12's levers. D60 tests their combination once, on development and then on the fresh games, with C3's +0.01. The organisers put the noise of the full-match score at about 0.04, so a gain of this size is small against it; it is worth one clean test and no more. Whatever D60's result, no further step follows. To make room on a full disk the prepared sequences and tokens of D56's S and SL (not adopted; their records are in the repository and `situation_train prepare` regenerates them) were deleted from `data/d56/`.

## D60 — The air rule on every training match with refitted trees left out where the ball is unseen (2026-11-13, 20:04 EST, written before any D60 network is trained, any tree is refitted or anything is scored)

**Question.** Do D57's training views and D59's tree change, together, raise spellF1 on real automatic tracking by enough to confirm on fresh matches?

**Disclosure.** Written after D57 and D59, whose two components were each the best of their kind on the PFF development games; D59's was the best of nine candidates at three costs, so its +0.0095 is optimistic. Development has been reused by every step and PFF development has now guided three choices. The fresh games of set D are the test.

**Candidate P.** The adopted stack's recipe (attacking-frame tokens, the y mirror, 8,100 steps, three seeds per slot) trained on D57's prepared sequences and tokens (D49's views plus one situation view per match, with the air rule on DFL and SkillCorner); D57's four SZ networks are its first seed set if the trainer reproduces one of them bit for bit, and eight networks are trained for the seed offsets 1,000 and 2,000. Both sets' trees are refitted as D3 fitted them on D57's sequences. The trees' share of the first ownership blend is 0 on frames where the ball is unseen and 0.5 elsewhere (D59's H(0)); the stoppage mix is unchanged. Ownership cost 12, 14 and 16.

**Pipeline checks.** The trainer reproduces D57's original-set seed-42 SZ network bit for bit; the refit reproduces D3's trees byte for byte on their own sequences; the adopted stack reproduces the cost-14 row of `development_t.json` through the scorer.

**Development (it decides only whether set D is spent).** Scored as D59's candidates were. The cost with the highest PFF development mean goes on if, against the adopted stack at cost 14: the PFF development mean is at least **0.012** higher; D47's composite and the long-hidden guard mean are each no more than 0.005 lower; no group falls by more than 0.01; and N = 1.75, 4 and 4.5 each fall by no more than 0.01.

**Confirmation, once,** if the gate passes: against the adopted stack, on PFF set D (the sixteen games named in the amendment of 16:23 EST, parsed only for this step) and the holdout. P replaces the adopted stack if set D's N = 3 mean rises by at least **+0.01** (C3), the holdout primary and the holdout long-hidden mean each fall by no more than 0.01, and N = 1.75, 4 and 4.5, dead ball carried and team swap each fall by no more than 0.01. **If it passes,** the adoption steps of D48 follow. **If the gate or the confirmation fails,** the adopted stack stands and nothing further is tried.

**Records.** `dfl_submission/stack_recipe/d60_*.json`, each written once. Code: new modules only (`dfl_submission/combined_trial.py`), reusing `stack_set.py`, `stack_trial.py`, `skill_height.py`, `height_fit.py`, `tree_share.py` and `tree_share_trial.py` unchanged. Scratch goes to `data/d60/`. Every D60 command except a build runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

**D60 outcome (2026-11-13, 21:58 EST).** **The development gate failed on one condition, so P does not go to set D.** The checks passed (the trainer reproduced D57's original-set seed-42 SZ network bit for bit; the refit reproduced D3's trees on their own sequences; the adopted stack reproduced its cost-14 row through the scorer).

| | Cost | PFF | Composite | Long-hidden guard mean (DFL / Metrica) | D33's primary | Team swap | PFF recall of spells seen under 20% |
|---|---:|---:|---:|---|---:|---:|---:|
| Adopted stack | 14 | 0.6958 | 0.7283 | 0.7369 (0.755 / 0.719) | 0.7352 | 0.707 | 0.27 |
| **P: SZ networks, refitted trees, trees out on unseen frames** | 14 (highest PFF) | **0.7167 (+0.0209)** | 0.7296 (+0.0013) | **0.7295 (−0.0074)** (0.730 / 0.729) | 0.7369 | 0.699 | 0.35 |
| P | 12 / 16 | 0.7141 / 0.7153 | 0.7294 / 0.7292 | 0.7256 / 0.7285 | | | |
| *For information, outside the rule:* SZ networks with D3's trees and the standing mix | 14 | 0.7091 (+0.0133) | 0.7303 (+0.0020) | 0.7386 (+0.0018) (0.748 / 0.729) | 0.7353 | 0.705 | 0.31 |
| the same | 12 / 16 | 0.7097 / 0.7080 | 0.7301 / 0.7316 | 0.7280 / 0.7380 | | | |

- **The gate:** PFF +0.0209 (+0.012 needed), composite +0.0013, no group down by more than 0.01 (team swap −0.008), conventions up; but the long-hidden guard mean fell 0.0074 against the 0.005 allowed. The fall is all on the DFL long-hidden view (0.755 to 0.730); the Metrica one rose (0.719 to 0.729).
- **What it shows.** Taking the trees out on unseen frames buys PFF (+0.021, and the spells seen under 20% are recalled at 0.35 against 0.27) at the price of the DFL match whose ball is hidden for 8–20 s with every player in view, the view most like the provided tracking in that respect. That price is why P stops here.
- **The row outside the rule.** With three seeds, D57's networks alone, on D3's trees and the standing mix, gain +0.012 to +0.014 on PFF at every cost, twice what one seed set showed in D57, and at the adopted cost 14 nothing falls: composite +0.0020, guard +0.0018, the largest fall of a group Metrica −0.005. D60's rule would not have selected it (its highest PFF is at cost 12, where its guard falls 0.009). It was reported for understanding and was read after the results. [Development](dfl_submission/stack_recipe/d60_development.json)

## D61 — D57's networks at three seeds to the fresh PFF games (2026-11-13, 21:59 EST, written after D60's development results, the row outside its rule included, and before PFF set D is parsed or any holdout match is scored for it)

**Question.** Does the adopted stack retrained on D57's views (the air rule on every training match), with nothing else changed, clear C3's +0.01 on sixteen fresh PFF games?

**Disclosure, in full.** This is a second look, and it breaks a statement of the orchestrator's own. D60's rule said that if its gate failed "nothing further is tried", and the round-15 note said no step would follow D60. D60's gate failed on its guard condition. Its record also holds a row reported for understanding: D60's three-seed networks with D3's trees and the standing mix, which is D57's SZ at three seeds. That row was read after the results, and it is the reason for this rule; D60 could not have selected it. D57 judged SZ on one seed set with a selection score that gives PFF one sixteenth of the weight, and it failed there. The development numbers below are therefore known, not predicted, and they are the best of what rounds 13 to 15 produced on PFF development, which has guided several choices. The project has taken this path before and said so (D41, D43: a development gate set knowing the score, C3's bar on a fresh confirmation unchanged). What keeps the evidence honest is the same here: the candidate and its cost are fixed below before set D is parsed, the confirmation is on sixteen games nothing has scored, and its bar is not moved.

**Candidate Q.** The adopted stack's recipe (D49's views, attacking-frame tokens, the y mirror, three seeds per slot, 8,100 steps) trained on D57's prepared sequences and tokens: the twelve networks of D60 (D57's four and D60's eight). D3's trees, the standing mixes, D43's cutoff, stoppage cost and discount, and **ownership cost 14, the adopted stack's**, so that only the networks differ from the submission. The cost is not chosen by score: at cost 12 Q's PFF is higher and its guard falls; at 16 it is about the same as at 14.

**Development, as already recorded in D60** (Q against the adopted stack, both at cost 14): PFF 0.6958 → 0.7091 (+0.0133; the amendment of 16:23 EST asks +0.010 before set D is spent); composite +0.0020; long-hidden guard mean +0.0018; D48's selection score +0.0019; the largest fall of a group Metrica −0.005; D33's primary level. Nothing more is scored on development.

**Confirmation, once,** against the adopted stack, which must first reproduce its holdout numbers of D48's confirmation: on PFF set D (the sixteen games named in the amendment of 16:23 EST, parsed only for this step), the holdout as D48's confirmation scored it, and the long-hidden views of the DFL holdout matches with D48's seeds. **Q replaces the adopted stack if** set D's N = 3 mean rises by at least **+0.01** (C3); the holdout primary and the holdout long-hidden mean each fall by no more than 0.01; and N = 1.75, 4 and 4.5, dead ball carried and team swap each fall by no more than 0.01.

**If it passes,** the adoption steps of D48 follow, with Q's twelve networks in the build behind the same identity checks and the kick-off-aligned sign rule unchanged: two builds, the release check, C2 re-declared with ranges from Q's external predictions, the records and documents, a commit and a push. `stack_trial build` with the stack's record must still give `f6d7b293…`, which becomes the first fallback. **If it fails,** the adopted stack stands, set D is spent, set E (eight games) remains, and this time nothing follows: the orchestrator will not write another rule.

**Records.** `dfl_submission/stack_recipe/confirmation_d61.json`, written once. Code: `combined_trial.py` or a new module; existing modules stay byte-identical. Scratch goes to `data/d61/`. Every D61 command except a build runs with `MTAH_DATA_DIR` set to a nonexistent path and calls `require_external_only()`.

*Amendment to D61 (2026-11-13, 22:25 EST, before any score of Q or of the adopted stack on set D exists).* The standing PFF reader refused one of set D's games while parsing (3840: a repeated frame with differing content, a case D7's amendment did not cover). Six games had been parsed before it (3816, 3817, 3824, 3825, 3832, 3833) and nothing was forecast or scored; the pre-scoring check had passed (the adopted stack reproduced D48's holdout and long-hidden numbers bit for bit). The reader is not changed for a confirmation set. **A game of set D that the standing reader refuses is left out and named in the record; set D is the games it reads.** If fewer than twelve of the sixteen can be read, D61 stops without scoring and set D stays unspent. The rule's bar and conditions are unchanged.

*Second amendment to D61 (2026-11-13, 22:32 EST, before anything of Q or the adopted stack is forecast or scored on set D).* Fifteen of the sixteen games were read (3840 left out, as the first amendment says). The attacking-sign check that D48's confirmation ran then stopped on one half: 3841's second half has a whole-half sign of −1 (D −4.2 m over 283 qualifying frames) while its first qualifying frame, 190 frames in, gives +1; the match's first half is +1 (D +3.8 m), so its two halves are opposite, as they must be. This is the case D54 met on SkillCorner 4039: on tracking that does not begin at a kick-off, one frame can show the teams on either side while the half as a whole is plain. **For D61 the first-frame agreement is reported and not enforced; D50's whole-half sign is used, and the two halves of every match must have opposite signs (that check is enforced).** The sign rule and the tokens are the same for both models scored, so this cannot favour either. 3841 stays in set D.

**D61 outcome (2026-11-13, 23:28 EST).** **Failed. Q does not replace the adopted stack, and the search is closed.** Before scoring, the adopted stack reproduced D48's holdout and long-hidden numbers bit for bit. Of set D's sixteen games fifteen were read (3840 left out: the standing reader refuses it); none had been parsed before. One half (3841, second) disagreed with its first qualifying frame and was reported, not enforced; every match's two halves had opposite signs.

| | Adopted stack, cost 14 | Q, cost 14 | Change | Required |
|---|---:|---:|---:|---|
| **PFF set D, N = 3 mean (fifteen fresh games)** | 0.7039 | 0.7090 | **+0.0051** | ≥ +0.01 |
| Holdout long-hidden mean (WQQ / WR9) | 0.7135 | 0.7165 | +0.0030 | ≥ −0.01 |
| Holdout primary, N = 3 standard combined | 0.73276 | 0.72760 | −0.0052 | ≥ −0.01 |
| N = 1.75 / 4 / 4.5 | | | −0.005 / −0.006 / −0.005 | each ≥ −0.01 |
| Dead ball carried / team swap | 0.646 / 0.711 | 0.651 / 0.715 | +0.005 / +0.004 | each ≥ −0.01 |

- **Set D is the condition that failed.** Eight of the fifteen games rose and seven fell (from +0.025 on 3856 to −0.012 on 10515); the mean change is +0.0051 with a standard error of about 0.003. Every no-harm condition held.
- **What it shows.** The development figure this rule was written on, +0.0133 on PFF, shrank to +0.0051 on fresh games. That is what choosing the best row after the fact predicts, and it is unlike D48, whose development figure was not chosen that way and held (+0.033 on development, +0.035 fresh). The air rule on every training match is a real but small gain on real tracking, about +0.005, and not one that clears C3's bar; on the standard holdout it gave back about as much.
- **Where things stand.** The submission is D48's stack, `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`, unchanged since it was adopted. PFF set D is spent; set E (eight games) is unparsed. As D61 said, no further rule is written. [Confirmation](dfl_submission/stack_recipe/confirmation_d61.json)
