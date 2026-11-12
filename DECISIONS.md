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
