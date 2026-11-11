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

**Outcome.** *(Added after the check: PASS or FAIL, the date, and the SHA-256 of the file checked.)*

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

**Outcome.** *(Added after the sanity gate and the measurement.)*

*Amendment, 2026-11-11, after reading PFF's Tracking Data Specification v2.2 and before any PFF tracking file was downloaded or scored:*
- The specification lists exactly these game event types: `FIRSTKICKOFF`, `SECONDKICKOFF`, `END`, `G`, `OFF`, `ON`, `OTB`, `OUT`, `SUB` and `VID`. It defines `home_ball` as whether the home team has control of the ball during the possession sequence. The mapping above is unchanged.
- The specification has no foul, offside or other stoppage event apart from `OUT` and `END`. Stoppages where the ball stays on the pitch therefore remain live in the D7 targets, owned by the last team. The `D` targets underestimate dead ball, and a predicted stoppage at a foul counts against every candidate scored on PFF, including the baseline.
- The specification places the frame rate inside a `videos` dict, while the public sample has it at the top level. The reader accepts either.
