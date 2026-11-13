# Team possession from tracking: technical document

*More Than A Hack 2027, Initial Challenge.* Team: **Anfield**. Repository: https://github.com/Derekko-web/mtah2027-submission. Submission file: `Anfield.csv`, SHA-256 `f6d7b2933a6210ce0c8a1ee96019f86815288d97368d656f366782345e6fc65d`.

## 1. Summary

We label every frame of the tactical video as `H`, `A` or `D`, and write `X` outside the two official halves. The system reads the automatic tactical tracking, plus the start and end times of the Smart Tagging kick-off tags. It has four parts:

- a 162-feature description of the players and the ball, and, for each team, the eleven players nearest the ball described one by one, in that team's direction of attack;
- two sets of three models that estimate who controls the ball, whether play is stopped, and where possessions change. The networks among them read the individual players through a small learned encoder (D42) and were also trained on views where the ball stays hidden for up to 30 s (D49); each network slot averages three networks trained with different seeds (D48);
- a three-state Viterbi decoder that turns those estimates into possessions;
- one rule from the Laws of the Game (D27): no team has possession before the kick-off, so the graded frames before the second-half kick-off tag are `D`.

Which way each team attacks in each half is read from the players' positions in that half from its tagged kick-off on, with no label and no fitted value (D50, section 2).

**Every model, threshold and decoder setting was trained, tuned or selected on external matches from other competitions.** The public 5-minute ground truth, the provided video and any human-produced data about Barcelona–Athletic were not used for training, tuning, calibration or selection.

## 2. Approach

### Why this design

The ranking metric scores possessions as objects, so the main risks are splitting one possession into several and missing a real change. We therefore:

1. estimate per-frame probabilities with models that see several seconds of context;
2. decode with explicit costs for changing state, so that an isolated flicker cannot create a possession on its own.

Both parts are fitted on external data where the truth is known. The public 5-minute window cannot be used for tuning. It is also far too small to choose between versions: the challenge's own baselines show it picking the wrong smoothing window.

### Inputs

- **Read:** the tactical ATD tracking and its metadata (team of each track), the homography `valid` flag, and the start and end times of the Smart Tagging `Kick Off (Start)` tags (section 6; the direction of attack is read from them on, below).
- **Coordinates:** converted to metres with the 105 × 68 m pitch reported by the ATD metadata.
- **Cleaning:** frames whose homography is invalid are masked, and ball positions more than 5 m outside the pitch are discarded.
- **Halves:** each half is processed separately.
- **Direction of attack:** which way the home team attacks in each half is read from that half's tracking (D50). Over the half's frames at 5 Hz from the start of its Smart Tagging `Kick Off (Start)` tag on (the tag paired with the half as in the kick-off rule, section 6), with at least seven visible players in each team, we take the mean of (mean x of the away players − mean x of the home players); if it is positive, the home team attacks towards +x. The sign then applies to every frame of the half. Nothing in this rule is fitted and no label is used. The build refuses to run if a half does not own exactly one kick-off tag, if a half has no such frame, if a half's direction disagrees with the direction of its own first such frame, or if both halves give the same direction.
  - *On external data.* Every external half used for training, development and confirmation starts at its kick-off, so there the rule reads the whole half. On the 268 external halves and views checked in D48–D50 (training views, long hidden-ball views, development, Metrica and confirmation), every direction agreed with the half's first such frame, and all but one PFF half had such frames; in training and scoring that half took the opposite of the other half's direction, since teams change ends at half time. On the eight halves of D54's older SkillCorner tracking (section 4), every half had a direction and the two halves of each match differed, but in one half the first such frame, 64 s into its tracking, disagreed with the half's direction: the first-frame check can misfire on tracking that does not start at a kick-off.
  - *The build's history.* The first build of the stack applied the rule to each graded half as a whole, from its first graded frame. It refused at the check that a half's direction agrees with its first such frame. That refusal message is the only output of the build that was read: no number, no frame, not which half. The provided match differs from every external half in one known way: its second half is graded from about two minutes before the kick-off (section 6). On Derek's decision (13:56 EST), and without looking at the provided tracking, the rule was aligned to the tagged kick-off, which changes nothing on external data. The second attempt passed every check: two builds gave the same SHA-256, and the validator's OK line and that SHA-256 were the only output read. The three direction checks were not changed; pairing each half with its tag added the fourth.

### Features

The model sees 162 features, computed at 5 Hz and made symmetric between the teams. They cover:

- nearest-player distances and player counts around the ball, per team;
- sustained player-to-ball contact (strength and distance over 0.25 to 4 s);
- ball speed, speed change, distance to the touchlines and goal lines, and time spent outside the pitch;
- team shape and motion;
- summaries over context windows of 3, 9 and 18 s.

There are no player identities in the data, so every feature is collective.

### Models

| Model | What it estimates | Notes |
|---|---|---|
| Extended-context XGBoost (two heads) | home control given live play; probability of a stoppage | The ownership head trains and predicts in both team orders. Classes and matches are weighted equally. |
| Player-set temporal network (seeds 42, 1,042 and 2,042) | ownership, stoppage and boundary probabilities | Dilated 1-D convolutions over 5 Hz sequences: six blocks of 96 channels. Its input adds, to the 162 features, each team's eleven players nearest the ball (offset, distance and velocity relative to the ball, pitch position), encoded one by one (8 → 32 → 32) and pooled per team by mean and maximum (D42). The players' x values are given in each team's direction of attack (D50). Trained for 8,100 steps with a learning rate × 2, batch 32 and dropout 0.2 (D34, D36, D52), on each set's training views plus one long hidden-ball view of every training half (section 3, D49). Each training window is mirrored across the line joining the goals with probability ½ (D47, D48). The slot's output is the mean of three networks trained with seeds 42, 1,042 and 2,042 (D47, D48). |
| Second player-set network (seeds 97, 1,097 and 2,097) | same | The same, with a heavier loss weight near labelled boundaries. Predicts on three 5 Hz grids offset by 0, 2 and 4 frames, then averaged. |

Each network's prediction is averaged over both team orders and over the pitch and its mirror image across the line joining the goals. The tree and network outputs are blended with equal weights. This ensemble was chosen on external development matches from 238 recipes, then confirmed on external holdout matches.

**Two model sets.** The ensemble exists twice:
1. trained on the external data as provided;
2. retrained after removing the providers' dead-ball artifacts (section 5).

Every output of the two sets is averaged with equal weight. Inference uses NumPy exports of the network weights and needs no PyTorch.

### Decoder

A three-state (`H`, `A`, `D`) Viterbi decoder runs over each half.

- **State:** a frame is `D` when the stoppage probability passes 0.53.
- **Switch costs:** a direct change of possession costs 14; entering or leaving a stoppage costs 2. Each cost is cut by up to 80% when the matching boundary signal is strong.
- **How these values were chosen:**
  - The stoppage cost, cutoff and discount came from the external development grid (section 4).
  - The ownership cost rose from 2 to 6 in a retune against controlled-possession targets (D1), to 8 when it was re-selected for the larger networks (D15), to 10 with the player-set networks (D42, D43), and to 14 for the stack (D48), chosen from 8, 10, 12 and 14 on development. 14 is the grid's largest value: from 8 to 14 the stack's development score rose by 0.007 in all (0.7258 to 0.7326), about the spread between its three seed sets scored alone (0.7300–0.7338), and the grid was not extended because the rule had fixed it.

## 3. Training data

| Dataset | Source and licence | Who labelled it | Amount | Use |
|---|---|---|---|---|
| DFL / Sportec integrated tracking and event data (Bassek, Rein, Weber & Memmert, *Scientific Data* 12:195, 2025) | [doi:10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0 | The data provider. Ball status and ball-owning team come from the official position feed. | 7 Bundesliga matches, about 1.0 M frames. Fitting: WMX, WN1, WOH, WOY. Development: WPY. Confirmation: WQQ, WR9. | Training, tuning, validation. Long hidden-ball views of WPY (a development guard) and of WQQ and WR9 (D48's confirmation) were scored too. |
| SkillCorner Open Data | [revision 4340d274](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656), MIT | SkillCorner: phases of play and possession on its own detected broadcast tracking | 20 A-League matches: 12 fitting, 4 development, 4 confirmation | Training, tuning, validation |
| PFF FC 2022 FIFA World Cup broadcast tracking | Shared on request by PFF FC (Gradient Sports). No dataset licence; see DECISIONS.md D7 for the terms we found. | PFF: possession sequences in its game events | 40 of 64 matches: 8 for development, and four confirmation sets of 8 (A, B, C and the untouched set), each parsed and scored once; the other 24 were never parsed and remain available for measurement and confirmation | Measurement (D7), a weight selection that changed nothing (D8), and from D11 on a guard that could veto candidates. Set A was scored in D15's confirmation, set B in D41's, set C in D43's and the untouched set in D48's, which required +0.01 on it. In round 12 (D47–D52) the development matches were also one of the eight groups of the development score that chose the stack's changes and its ownership cost, and a gate of their own (section 4). D32 adapted models to the development matches' unlabelled tracking and was rejected. No model in the submission was trained or adapted on PFF data, and the build reads none of it. |
| SkillCorner Open Data, 2019–20 release | [revision 1b99d401](https://github.com/SkillCorner/opendata/tree/1b99d401ddc82e366c5c20d6860878668fd5858f), MIT | SkillCorner: home/away possession on its detected broadcast tracking; no dead-ball label | 9 matches: 5 development, 4 confirmation (scored once, in D54) | The D6 proxy, D14 training models that were not adopted, and D54's measurement of D43 and round 12's two candidates on the 4 confirmation matches (section 4). No model in the submission was trained on it. |
| Metrica Sports sample data, games 1–3 | [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data); no licence file, credit requested | Metrica: event data (passes, recoveries, ball out, set pieces) on anonymised matches; corrected tracking | 3 matches: event data, and in D19 also the tracking | To choose the controlled-possession convention (section 6), and from D19 on as a benchmark that can veto later candidates. In round 12 the games under competition-like tracking (D24) were one group of the development score, and a long hidden-ball view of them a guard. D30 trained fold models on it and was rejected. No model in the submission was trained on it. |

**None of the training data comes from the provided game.** Both halves of a match always stay in the same split.

- The XGBoost heads were fitted on 763,244 ownership rows and 696,959 stoppage rows, sampled at 5 Hz. The ownership count includes the team-swapped copies.
- The DFL training views include:
  - the clean tracking;
  - a degraded view: coordinate noise, 20% of player tracks dropped in 2 s blocks, and runs of missing ball;
  - a stress view: longer ball outages, false ball observations, heavier player dropout and camera outages.
- SkillCorner data is used as detected, at 25 Hz after interpolation between supported observations.
- The networks of both sets, not the XGBoost heads, also train on one long hidden-ball view of every training half (D49): 32 views per set, alongside its 48 existing training views. Half of the hidden runs last 2 s on average and the other half 5–30 s, and the visibility is drawn per half from 30–90% for DFL and from 50–90% for SkillCorner, on top of its own undetected frames. Each view otherwise keeps its set's construction (noise and dropped players for DFL; dead-ball handling and identity swaps in the retrained set). In these views 81–88% of the hidden-ball frames lie in runs over 10 s, against 2.5% in the masked DFL development view, which is built like the degraded view above.

## 4. External validation

The confirmation matches were reused across several decisions, so they are development evidence rather than an untouched test. Every decision after the first ensemble was governed by a rule written down before scoring ([DECISIONS.md](https://github.com/Derekko-web/mtah2027-submission/blob/main/DECISIONS.md)), with fixed gates and a single holdout check.

### Ensemble against provider labels

Combined spellF1 weights DFL (one-third clean, two-thirds degraded tracking) and SkillCorner equally.

| Holdout | DFL | SkillCorner | Combined |
|---|---:|---:|---:|
| Previous reference (XGBoost and first network) | 0.5672 | 0.5964 | 0.5818 |
| Ensemble | 0.5740 | 0.6094 | 0.5917 |

### Against controlled-possession targets (N = 3 s, section 6)

These are combined holdout spellF1 scores.

| Stage | Standard views | Dead ball carried (DFL) | Raw provider labels |
|---|---:|---:|---:|
| Ensemble, switch costs 2 / 2 | 0.6782 | — | 0.5917 |
| Decoder retune, ownership cost 6 (D1) | 0.6923 | 0.5364 | 0.5794 |
| Blend with the retrained models (D3) | 0.6873 | 0.5980 | 0.5746 |
| Larger, longer-trained networks, ownership cost 8 (D15) | 0.7087 | 0.6104 | 0.6179 |
| Player-set networks, ownership cost 10 (D42, D43) | 0.7276 | 0.6430 | 0.6401 |
| **The stack: long hidden-ball views, direction of attack, three seeds, mirror, 8,100 steps, ownership cost 14 (D48), submitted** | **0.7328** | **0.6460** | **0.6532** |

The decoder retune improved all eight holdout match views on the primary target. The blend trades 0.005 on views that still contain the provider dead-ball artifacts for +0.062 when the dead ball moves (section 5).

**D15: larger, longer-trained networks.** Each network had 48 channels per block and trained for 1,800 steps, values that were never searched. D15 retrained all four networks of both model sets at widths 48 and 96 and lengths of 1,800 and 5,400 steps, and re-selected the ownership cost from 4, 6 and 8, under a rule written before any candidate was trained. Every new setting passed every development gate. Width 96, 5,400 steps and cost 8 gained +0.041 on development. On its single confirmation it gained +0.021 on the holdout, +0.012 with the dead ball carried, +0.059 on team swaps and +0.019 on PFF confirmation set A, eight PFF matches never scored before. Training length mattered more than width. The holdout gain is about half the development gain, as expected after picking the best of nine eligible settings on reused matches.

**D42 and D43: a network that reads the players.** Every earlier input was a hand-made team summary. D42 gave each network, alongside the 162 features, the eleven players of each team nearest the ball, each described by its offset, distance and velocity relative to the ball and its position on the pitch, passed through a small shared encoder and pooled per team (mean and maximum). Everything else is D15's structure; the networks were trained with the optimisation values D34 and D36 found. On development it gained +0.0067, under the +0.01 gate, but on every degraded or unseen view at once: both guards by about +0.035, PFF (a vendor no model trained on) by +0.007, the Metrica games under competition-like tracking by +0.027 and SkillCorner by +0.014, with DFL WPY flat. D43 sent it to one confirmation, under a lower development gate written after those results and disclosed as such, with C3's bar unchanged. It gained **+0.0189** on the holdout (every DFL view and SkillCorner rose), +0.0108 on PFF confirmation set C (eight fresh matches), +0.033 with the dead ball carried, +0.061 on team swaps and +0.022 against raw provider labels. It replaced D15 in the submission, until D48 (below).

**Round 12: long hidden-ball stretches (D46–D54).** D46 measured where the player-set model still loses. On real automatic tracking the loss is concentrated where the ball stays hidden for long:
- on PFF and SkillCorner, a third to two fifths of the hidden-ball frames lie in runs over 10 s, against 2.5% in the masked DFL development view;
- on PFF, the decoded owner agrees with the target on 95.7% of seen-ball frames and 85.1% of hidden-ball ones;
- possessions whose ball is seen on under 20% of their frames are 23% of PFF's possessions, and D43 recovered 0.17 of them.

Round 12 screened changes on development under a no-harm rule, kept those that held, and spent one confirmation on their combination. Screening used a composite: the unweighted mean of eight development groups (DFL WPY clean, masked, dead ball carried and team swap; SkillCorner; PFF; Metrica; the D24 views). Four changes were kept:
- **Long hidden-ball views (D49):** one more training view per half, with half of the hidden runs lasting 5–30 s (section 3). Against D43: long-hidden guard views +0.040, PFF +0.024, composite −0.001. The guard views, of DFL WPY and the Metrica games, are built differently, with the ball hidden for 8–20 s at a time.
- **The direction of attack (D50):** the player tokens in each team's direction of attack (section 2). Composite +0.0069, PFF +0.018, the D24 mean +0.015.
- **Three seeds per slot and the mirror (D47):** each was within about one seed spread of D43 (three seeds of D43's recipe spread with a standard deviation of 0.0036 on the composite). They were kept because they did no harm and had a reason to help. With the direction of attack, only the mirror across the line joining the goals changes anything.
- **Longer training (D52):** on one seed set, 8,100 steps instead of 5,400 gained +0.0088 on the selection score below (+0.008 was required). A seventh temporal block lost.

D48 combined them and retrained every network at 8,100 steps. On development at ownership cost 14, against D43 at cost 10:
- selection score (the mean of the composite and the long-hidden guard mean): 0.6944 → 0.7326 (+0.038);
- composite: 0.7139 → 0.7283 (+0.014), with all eight groups up;
- long-hidden guard mean: 0.675 → 0.737 (+0.062);
- PFF development: 0.663 → 0.696 (+0.033);
- the standard primary: 0.7295 → 0.7352 (+0.006).

Each seed set scored alone reached a selection score of 0.7300–0.7338.

The confirmation's rule was written before any stack network was trained. It put C3's +0.01 where the stack acts and where the data most resemble the competition tracking. The standard holdout views hide the ball for 2 s at a time, so they had only to lose no more than 0.01. N = 3 spellF1 against D43:

| Confirmation set | Matches | D43, cost 10 | Stack, cost 14 | Change | Required |
|---|---|---:|---:|---:|---|
| Untouched PFF set (**fresh**: never parsed or scored before) | 8 World Cup matches | 0.6886 | 0.7236 | **+0.0350** | ≥ +0.01 |
| Long hidden-ball views of the DFL holdout (new views of **reused** matches) | WQQ, WR9 | 0.6714 | 0.7135 | **+0.0421** | ≥ +0.01 |
| Standard holdout views, primary (**reused**) | WQQ, WR9 and four SkillCorner matches | 0.7276 | 0.7328 | +0.0052 | ≥ −0.01 |
| Four 2019–20 SkillCorner matches (**fresh**; measured afterwards, D54) | 3442, 3518, 3749, 4039 | 0.6293 | 0.6248 | **−0.0046** | none; supported if ≥ 0 |

On the reused holdout the stack also gained +0.016, +0.001 and +0.001 at N = 1.75, 4 and 4.5, +0.003 with the dead ball carried, +0.018 on team swaps and +0.013 against raw provider labels. All eight fresh PFF matches gained (+0.021 to +0.063), and so did both long-hidden holdout matches (+0.030 and +0.054). Its changes and its ownership cost were chosen on development matches reused by every step since D3, so the development gains are optimistic. On the fresh PFF matches the gain was about the same size as on PFF development (+0.035 against +0.033). On the four 2019–20 SkillCorner confirmation matches, which no step had scored before, measured after D53 (D54, below), the stack was 0.005 below D43: one match up and three down, with every convention the same way. Those matches are fresh, but they are a weaker test than the PFF set: ownership only, on older broadcast tracking with a target on 62–73% of each timeline, and with four matches the deficit is about one standard error.

The stack's first build refused at one of its direction checks; only the refusal message was read (section 2). While the next step was decided, D53 prepared a fallback: the same stack without the direction of attack, its training windows mirrored in x and in y as D47 screened them. It passed D48's development gate (selection score 0.7249 against D43's 0.6944, PFF +0.020) and D48's pass rule on the same confirmation sets. That was their second use, so it is weaker evidence: PFF +0.0204, long-hidden holdout views +0.0323, holdout primary +0.0027. On D54's four fresh SkillCorner matches it was 0.010 below D43 (0.6190 against 0.6293). Its build gave one SHA-256 six times on three platforms. Then, on Derek's decision (13:56 EST), the direction rule was aligned to each half's tagged kick-off; the stack's second build attempt passed every check, two builds and four release builds gave one SHA-256, and the stack replaced D43 in the submission. D53 was not used: before that build, Derek had decided it would not ship, since its only fresh result is negative.

**D27: dead ball before the kick-off.** The second half is graded from about two minutes before its kick-off, a stretch no external half contains, so the models' output there had never been measured. D27 applies the Laws of the Game instead: every graded frame before the start of the half's Smart Tagging kick-off tag becomes `D` (section 6). It cannot be scored on external data, so it was adopted on tests and identical rebuilds. The rebuilt file differs from D15's, and D15's labels through the same code without the rule reproduce D15's file exactly, so the rule alone changed some second-half frames before the kick-off. How many was not looked at.

### Experiments that did not change the submission

Each followed a rule written before scoring, and each outcome is recorded in [DECISIONS.md](https://github.com/Derekko-web/mtah2027-submission/blob/main/DECISIONS.md) whether it passed or failed. Scores are spellF1 against the N = 3 targets.

| Decision | What was tested | Result | Effect on the submission |
|---|---|---|---|
| D4 | Set-piece restart timestamps as decoder evidence. The tags were simulated from the DFL event feed with missing, late and spurious tags. | Best of ten settings: 0.61376 against D3's 0.61382 on development. A gain of +0.003 was required. | None. The build reads only the kick-off tags, for D27 (section 6). |
| D5 | Label-free transfer: DFL-only heads adapted to unlabelled SkillCorner tracking, by speed alignment, matched corruption or self-training on confident pseudo-labels. | Self-training at 0.90 confidence passed. On the four confirmation matches it raised spellF1 from 0.493 to 0.576 over the DFL-only reference. | None. It shows that label-free adaptation transfers across tracking providers on external data. It is not applied to the provided game (section 8). |
| D6 | The same idea applied to the submitted ensemble, adapted separately for each of five SkillCorner 2019–20 development matches. | Best setting +0.0006, against a required +0.005. Confirmation was not scored. | None. |
| D7 | Frozen D3, then the submission, measured on PFF FC's 2022 World Cup broadcast tracking, a vendor none of the models has seen. Eight development matches, after a baseline sanity gate. | spellF1 0.626 against N = 3 controlled-possession targets and 0.549 against raw provider labels. The organisers' smooth50 baseline scores 0.339 and 0.345. | None. A measurement only; nothing was trained or tuned on PFF data. |
| D8 | The share of the retrained model set (0, 0.25, 0.5, 0.75, 1), chosen on the PFF development matches, with DFL/SkillCorner guard views. | Best +0.008 (share 0), against a required +0.02. That share also lost 0.068 when the dead ball moves. Confirmation was not scored. | None. The equal blend stays. |
| D11 | Where D3, then the submission, lost spellF1 on the development matches: per-IoU detection, two oracle runs, boundary offsets by transition type, recall by spell length. | With true ownership substituted, standard combined spellF1 rises from 0.682 to 0.887; with true stoppages, only to 0.714. D3 predicts 20–25% fewer spells than the targets, with high precision and low recall; it recovers under half of the true spells shorter than 5 s. Boundary offsets have medians of −2 to +2 frames on DFL and SkillCorner, with no sign the vendors agree on. | None. A measurement only. It handed no boundary shift to D12. |
| D12 | Decoder output refinements: a 1-frame decoder step instead of 5, and collapsing brief interior spells (1.75 s or 3 s) in the decoded labels. | Best change −0.001 on development, against a required +0.005. D3 predicts almost no brief interior spells, so collapsing has little to remove. Confirmation was not scored. | None. The 5-frame decoder stays. |
| D13 | Both model sets refit on controlled-possession targets (brief A–B–A spells under 3 s merged before training), with ownership costs 2–8 and three stoppage cutoffs. Checks first confirmed that the refit reproduces the published trees and networks when given the original targets. | Best change −0.00002 at N = 3, against a required +0.01. Every setting lost more than 0.01 at N = 1.75; at D3's costs the aligned models gain +0.011 at N = 4 and 4.5. Confirmation was not scored. | None. Training on the challenge's convention moves the score between conventions; it does not recover missed changes. |
| D16 | 22 last-touch and possession-chain features (ball turns next to players, time since and until each team's touches, touch counts and run lengths), with both trees refit first; the networks would follow only if the trees gained +0.003 against D15. Checks first confirmed that the 162 existing features were re-extracted exactly and that the trees refit byte for byte. | Best tree-only change −0.0004 against D15. The new trees matched more ownership changes on clean DFL (104 against 93 of 184) but not on SkillCorner or PFF, and lost at N = 4 and 4.5 and on team swaps. | None. The networks were not retrained; confirmation was not scored. |
| D14 | Five 2019–20 SkillCorner matches added to training as a third source, with ownership-only targets. | Every development gate failed: −0.018 at N = 3, −0.049 with the dead ball carried, −0.028 on team swaps, −0.023 on PFF. Confirmation was not scored. | None. |
| D18 | A published ball-free model, PathCRF (Kim et al., KDD 2026), against D3 on frames where our ball is hidden. PathCRF infers possession from player trajectories alone. We ran one released checkpoint unchanged. Its checkpoints were trained on DFL WPY, our development match, which favours it there; the four SkillCorner development matches are unseen. | Team-possession accuracy on ball-hidden frames against the N = 3 targets: D3 0.870 against PathCRF 0.595 on masked WPY, and 0.786 against 0.710 on SkillCorner. A diagnostic traced the masked-view collapse to the 0.2 m per-frame position noise, not to the dropped tracks. PathCRF's teams with D3's stoppages scored 0.337 combined spellF1 against D3's 0.682, because PathCRF changes team at every touch. | None. PathCRF lost on both sets, so the planned retrain did not run. |
| D19 | Metrica Sports' three public sample matches as a benchmark. Their corrected tracking has no ball while play is dead, and their events carry the team: the conventions CHALLENGE.md describes for the ground truth. A reader sanity gate came first. | D3 0.760 and D15 0.780; every game improved. The smooth50 baseline scores 0.423 against the raw targets. The tracking is corrected and the dead ball is exactly the missing ball, so these scores flatter every model and do not estimate the competition score. | None. A measurement and a veto: from D19 on, a candidate whose mean N = 3 spellF1 falls more than 0.01 below the submission's (line 0.770) is rejected. Nothing is trained or chosen on these games. |
| D20 | A model search beyond D15, tuned by match-grouped cross-validation on the training matches only (frame log-loss): the temporal network at widths 96 and 192 and 5,400 to 19,800 steps, a bidirectional GRU, a transformer encoder and deeper XGBoost tree heads. Each family's best setting was scored once on development, then a greedy equal-weight ensemble. | The temporal network selected D15's own setting. Against D15's 0.723 on development: GRU −0.003, trees −0.003, transformer −0.031. The best pair, D15 + GRU, scored 0.722, so the selection kept D15 alone. | None. Frame log-loss favoured the shortest training in every network family, the opposite of D15's spellF1 gain from longer training. |
| D21 | Where the submission loses spellF1: D11's measurements repeated for D15, with the Metrica sample matches added. | With true ownership substituted, standard combined spellF1 rises from 0.723 to 0.885; with true stoppages, to 0.750 (D3: 0.682, 0.887 and 0.714). D15 predicts 6–22% fewer spells than the targets (D3: 20–25% fewer). On SkillCorner it recovers 46–52% of the true spells lasting 1.75–5 s. | None. A measurement only. |
| D22 | Finishing D15's search, then averaging seeds. Stage A decoded D20's 24 fold networks (widths 96 and 192, 5,400 to 19,800 steps) on the training matches they had not seen, at ownership costs 8, 10 and 12. Stage B trained two more seeds for each of the four network slots, and each slot became the mean of three networks. | Stage A: the best point (width 96, 10,800 steps, cost 12) beat D15's own setting by +0.002 on the held-out training matches, against a required +0.005, so D15's setting and cost stayed. Three seeds per slot scored 0.717 against D15's 0.723 on development (+0.005 was required). They gained with the dead ball carried (+0.013), on team swaps (+0.011) and on Metrica (+0.002). | None. Confirmation was not scored. |
| D23 | Refitting the submission on all 27 labelled external matches. It would run only if, on development, a network trained on 16 matches beat four trained on 12 by +0.005. | The 16-match network scored 0.7114 against the 12-match networks' mean of 0.7135; single networks ranged from 0.708 to 0.722. | None. The refit did not run. The untouched PFF matches and the four 2019–20 confirmation matches were not scored. |
| D24 | The Metrica sample matches under competition-like tracking: the DFL masked degradation (ball hidden about 40% of the time, noisy and dropped player tracks); D2's dead-ball handling followed by the same masking, so a missing ball no longer marks the stoppages; and team swaps on top. | Against the N = 3 targets, D15 scored 0.705 masked, 0.706 with the dead ball handled and 0.611 with team swaps, against 0.780 on the clean tracking. The smooth50 baseline scored 0.14–0.19 on the same views. | None. A measurement only. The mean of the three views (0.674) became a gate for D25 and D26. |
| D25 | Removing every predicted stoppage shorter than one second that has a team on both sides. D15 predicted 87 of them on the PFF development matches, where the targets have 1. One hand-set limit; adopted only if nothing got worse. | All of them were removed (PFF 87 to 0). PFF gained +0.0007, the standard views lost 0.0001, and the mean of the D24 views lost 0.00008 because one game lost 0.0007. | None. The no-harm rule failed; confirmation was not scored. |
| D26 | Moving D15's ownership changes earlier to match Metrica's timing. On the Metrica games they land a median 2–6 frames after Metrica's marks, under two marking rules and on clean and masked tracking; the shift was the smallest median, 2 frames. | Leave one game out on the clean Metrica games: +0.0031, −0.0008 and −0.0014 (mean +0.0003; each game had to be at least 0 and the mean at least +0.003). Guards on DFL, SkillCorner and PFF lost at most 0.0042. | None. Nothing estimated on the Metrica games enters the submission. Confirmation was not scored, and the untouched PFF set is unused. |
| D28 | Filling short ball dropouts before the models see a half: each missing-ball run of at most K frames between two sightings in the same half gets the straight line between them, for K = 6, 12 and 25 frames (up to 1 s). Nothing retrained. | The fill reached at most 9.5% of hidden live-ball frames on the masked DFL views, 13% on PFF and none on SkillCorner. Best K (12): +0.0016 on development against a required +0.005, and every K lowered the mean of the D24 views by 0.0003–0.0007. | None. Confirmation was not scored, and PFF confirmation set B was not used. |
| D30 | Metrica's three sample games as training data for both model sets, in each set's DFL training views, counted as DFL matches in the weights. Checks first confirmed that the refit reproduces D15's trees and networks exactly when no Metrica half is added. Scored leaving one game out. | Clean Metrica view +0.019, but the D24 views only +0.0047 (+0.01 required; one game lower), and every fold model lost 0.017–0.059 on DFL with the dead ball carried. | None. The corrected tracking hides the ball exactly when it is dead, so training on it teaches the dead-ball shortcut. |
| D31, D31a | The broadcast ball where the tactical ball is missing: broadcast stretches aligned to the tactical timeline by formation matching, with every constant set on a synthetic two-camera harness built from external tracking. D31a was a disclosed second attempt with a trimmed cost and a translation per stretch, after D31's constants failed. | D31: no setting kept wrong alignments under 1%. D31a: only the strictest setting did; it filled 0.28% of hidden-ball frames on development, for +0.0003 (DFL) and +0.0006 (Metrica), against +0.01. | None. No broadcast file was read. |
| D32 | D6's per-match label-free adaptation with the submission as teacher, extended to the stoppage head, on the eight PFF development matches. | Every multiplier lowered mean N = 3 (best −0.0004, against +0.01 required). | None. It would also have needed the organisers' written permission. |
| D29 | Dead ball from each tagged goal to the kick-off after it (Smart Tagging `GOALS` and `Kick Off (After Goal)`). | Cannot be scored on external data. | None. Under C3 it could ship only alongside a step that cleared its bar. |
| D33 | Spell-balanced network training. spellF1 counts every possession once, but the networks' loss counts frames. So each live frame's ownership loss was weighted by the inverse of its possession's length under the N = 3 convention, and the boundary heads got more weight. All four networks were retrained, and the ownership cost and boundary discount re-selected. A check first confirmed that the new trainer at its defaults retrains a published network bit for bit. | Best +0.0077 on development (weight exponent 1, boundary weight 0.6, cost 10), against +0.01 required; that point also lost 0.017 on PFF. The weighting gained where possessions are short and the labels follow them (DFL masked +0.019, Metrica +0.018) and lost on PFF's coarser possession sequences at every setting. | None. Confirmation was not scored. |
| D34 | The optimisation values never searched before (learning rate, batch size, weight decay, dropout), one change at a time and then the best combination. Chosen by four-fold cross-validation on the 16 training matches, each network decoded on the matches it did not train on, as in D22. A check first confirmed that the new trainer at its defaults retrains a published fold network bit for bit. | No single change moved the held-out score by more than +0.0045 (batch 32). The combination of a learning rate × 2, batch 32 and dropout 0.2 gained +0.0075, under the +0.008 the rule required before retraining the full models. | None at this stage; see D36. |
| D35 | The trees' share of each model set's ownership and stoppage outputs (a quarter since the original ensemble), re-selected for D15's networks together with the ownership cost. Nothing was retrained; the re-blended member outputs reproduced D15's inputs exactly at the old weights. | Best +0.0030 (no trees in the ownership output, cost 10), against +0.01 required. | None. |
| D36 | D34's combination taken to the full models anyway, as a decision disclosed as a second look at D34's evidence: all four networks retrained with a learning rate × 2, batch 32 and dropout 0.2, and the ownership cost re-selected. | +0.0070 on development (cost 8), against +0.01 required. Unlike D33 it gained on every guard view and on Metrica and the D24 views, with PFF flat, but almost all of the gain was on SkillCorner. | None. Confirmation was not scored. |
| D37 | D36's training values combined with a milder spell weight (exponent 0.5), with D16's touch features as network inputs, and with both. | Best +0.0070, D36's own networks; the spell weight +0.0068, the touch inputs +0.0028, both +0.0034, against +0.01 required. The touch inputs lost on PFF and at N = 4 and 4.5, as with D16's trees. | None. Round 9 ended here. |
| D38 | Cross-validation on the training matches around D34's combination: 10,800 steps, dropout 0.3, batch 64, a learning rate × 3, width 128. | Only dropout 0.3 gained (+0.0029), under the +0.003 the rule required before retraining; the others lost 0.002–0.009. | None. |
| D39 | Twelve columns describing the players around the ball (second-nearest distance per team, players within 5 m and 10 m, how fast the three nearest close in, the nearest player's speed and whether it moves with the ball), given to all four networks trained with D36's values. A check first confirmed that the 162 existing columns were re-extracted exactly for every training half. | +0.0012 on development, against +0.01 required: SkillCorner kept D36's gain but DFL masked lost it. Both guard views and the Metrica views under competition-like tracking gained the most of any step (+0.014 to +0.017). | None. |
| D40 | D36's networks with the trees' share re-selected, as in D35. Nothing retrained. | D36's own mix stayed best (+0.0070), against +0.01 required. | None. |
| D41 | The best-evidenced candidate sent to one confirmation, with a development gate lowered to +0.005 (disclosed as set after the results) while C3's +0.01 on confirmation stayed: D36's networks, or the same training with dropout 0.3. | Development selected D36's networks (cost 8). On confirmation they gained +0.0018 on the holdout (SkillCorner +0.009, DFL mixed) and +0.0039 on PFF set B, eight fresh matches, with both guards up; C3 required +0.01. | None. PFF set B is spent. |
| D44 | Player-set variants after D43, by cross-validation on the 16 training matches: 16 players per team, a wider encoder, the players' velocity relative to the ball, D15's training values. | D42's own design scored best (0.7204 at cost 10, the cost D43 uses); the variants changed it by −0.0003 to −0.0087, so none went to development and the untouched PFF set stayed unused. | None. |
| D45 | The player-set model refit on all 21 labelled training and development matches (the 16 training matches plus DFL WPY and the four SkillCorner development matches), every network retrained exactly as D42 trained it. Development cannot score a model trained on it, so the refit was to be judged on vetoes and one confirmation. A check first confirmed that the halves of the 16 training matches were reproduced exactly. | Against D43: PFF development −0.0012, Metrica −0.0062, the D24 views +0.0014. The vetoes passed but showed no gain. The confirmation was held so that the untouched PFF set would be spent once, on round 12's combined candidate, which left the refit out. | None. Never confirmed. |
| D46, D46a | Where the submission loses spellF1, as in D21, now split by whether the ball is seen. Then: how long the ball stays hidden, and whether the players are seen meanwhile. | Ownership is still the main loss: true ownership lifts standard combined spellF1 from 0.730 to 0.879; true stoppages lift it to 0.756. On PFF, the decoded owner agrees with the target on 95.7% of seen-ball frames and 85.1% of hidden ones. Possessions seen on under 20% of their frames are 23% of PFF's possessions and are recovered at 0.17. A third (PFF) to two fifths (SkillCorner) of the hidden-ball frames lie in runs over 10 s, against 2.5% under our training mask. On PFF, only 22% of hidden-ball frames show at least 16 players. D46's planned reading turned out not to measure the headroom: an oracle putting the true owner only where the ball is hidden adds a boundary at every seam between true and decoded labels. So the next steps were chosen from the other measurements, after the results were seen; the record says so. | None directly. It pointed round 12 at long hidden-ball stretches. |
| D47 | Three cheap changes to the player-set networks, screened on development under a no-harm rule (composite not lower, no group down by more than 0.01): two more seeds per slot, averaged (M3); the pitch's mirror images at inference only (F0), or in training and inference (F); noise on the player tokens in training (N); F and N together (FN). A check first confirmed that all 162 features are unchanged under both mirrors. | Three single models of D42's recipe spread with a standard deviation of 0.0036 on the composite. M3 (+0.0033, cost 12) and F (+0.0036, cost 8) passed. F0 failed (−0.0010). N passed with no gain (+0.0004), and FN scored below F. | M3 and F (mirroring across the line joining the goals only) entered the submission through D48. F0, N and FN did not. |
| D51 | Asking the networks during training where the hidden ball is. On the long-hidden views, an extra output was trained (Huber loss, weight 0.5) on the hidden ball's offset from the visible players' mean position. Checks first confirmed that the targets were re-derived exactly, and that at weight 0 the trainer reproduces D42's bit for bit. | The networks learned to place the hidden ball: 5.9–7.6 m off at the median, against 19–23 m for the players' mean position. But possession got worse. Against D49's networks (the same views without the extra output), the composite fell 0.017 and the mean of the guard and PFF scores fell 0.018 (+0.005 required). Six of eight groups fell by more than 0.01 against D43. | None. The weight was set by judgement and not searched. |
| D52 (X) | On the stack's first seed set, a seventh temporal block, so the networks see 25 s on each side instead of 12.6 s. | Selection score −0.0030; the Metrica long-hidden view −0.013. | None. The networks keep six blocks. D52's other candidate, 8,100 training steps, is in the submission (above). |
| D53 | A fallback, prepared after the stack's first build refused: the stack without the direction of attack, its training windows mirrored in x and in y with probability ½ each, inference averaged over four mirrors and both team orders. Same data, seeds, length, trees and decoder. Its build reads the provided match exactly as D43's does. | Development at cost 14, against D43 at cost 10: selection score 0.6944 → 0.7249, composite +0.012, long-hidden guard mean +0.049, PFF +0.020. D48's confirmation sets, used a second time: PFF 0.6886 → 0.7091 (+0.0204), long-hidden holdout views 0.6714 → 0.7037 (+0.0323), holdout primary 0.7276 → 0.7304 (+0.0027). On the same sets the stack with the direction of attack was ahead by 0.015, 0.010 and 0.002. On D54's four fresh SkillCorner matches: 0.6293 → 0.6190 (−0.0103). | None. Built in scratch (one SHA-256 on six builds); not shipped, by Derek's decision before the aligned build, since its only fresh result is negative. |
| D54 | Measurement: D43 and both candidates on the four 2019–20 SkillCorner confirmation matches (3442, 3518, 3749, 4039), which no step had scored. Ownership targets only, on older broadcast tracking, with a target on 62–73% of each timeline; stoppages are not scored. A candidate was to count as supported there if its mean was at least D43's. | N = 3 means: D43 0.6293; the stack with the direction of attack 0.6248 (−0.0046; one match up, three down, the largest fall 0.017); D53 0.6190 (−0.0103). Every convention and the raw labels point the same way, by 0.003–0.011. With four matches, the stack's deficit is about one standard error and D53's under two. On these halves, whose tracking does not start at a kick-off, one half's first qualifying frame, 64 s in, disagreed with its whole-half direction. | None; it selects nothing. By its reading neither candidate is supported there. The stack shipped on the rest of its evidence (above). |

### The one correctness check

The public 5-minute window is used once, on the final file, under a procedure fixed in advance (DECISIONS.md C1). The check confirms that the file is well formed, that frames and halves are aligned, and that home and away are not swapped. It cannot choose between versions. A failure may only lead to a pipeline fix, never to a change of model or setting.

A second check uses no ground truth at all (DECISIONS.md C2). On the final file it computes three statistics: possessions per minute, the share of dead ball, and stoppages shorter than one second per 90 minutes. Each must lie within a range fixed in advance from the submission's predictions on 28 external match views: from half the smallest value to 1.5 times the largest. It reports only pass or fail, and a failure, as with C1, leads only to a pipeline investigation.

Both checks passed on the file submitted until D43 (SHA-256 `0ddbb342…`) on 12 November: C2 first, then C1. One disclosure: C1 was run from inside our AI coding session, so the AI assistant saw the evaluation output for the public window. That breaks C1's own rule that no AI tool sees it. By then the file was frozen and checked, and nothing was changed or chosen afterwards (DECISIONS.md C1).

After C1, development continued for one more round (D29–D32) under a rule written first (DECISIONS.md C3): a step could change the file only with a gain of at least +0.01 on its own external confirmation, and with every new step switched off the build had to reproduce the checked file byte for byte, which it did. The search was then reopened for rounds 9 and 10 (D33–D41) under the same rule. One candidate reached a confirmation (D41: +0.0018 on the holdout and +0.0039 on eight fresh PFF matches, against +0.01). Round 11 then gave the networks the individual players (D42) and sent the result to one confirmation (D43), which it passed (+0.0189 on the holdout, +0.0108 on eight fresh PFF matches). That changed the file. Round 12 (D45–D52) then screened changes on development under a no-harm rule and sent their combination to one confirmation (D48). Its rule put the +0.01 on the untouched PFF set and on long hidden-ball views of the holdout matches, and allowed no loss of more than 0.01 on the standard holdout views. The stack passed (+0.0350 on eight fresh PFF matches, +0.0421 on the long-hidden holdout views, +0.0052 on the standard holdout views). Measured afterwards on the four 2019–20 SkillCorner confirmation matches, which no step had scored, it was 0.005 below D43 (D54). The stack changed the file again. C1 is not run again (C3): the reading, team orientation, frame indexing, half spans and the kick-off rule are the same code as in the checked file, which the old build command still reproduces byte for byte. The networks and the ownership cost changed, and the build now also computes each half's direction of attack from its tracking, from the half's tagged kick-off on (section 2). That changes only the players' x values inside the networks' input, not which team is home or how frames are indexed. C2 was re-declared for D43's file and again for the stack's, each time with ranges recomputed from the new model's predictions on the same 28 external views (for the stack: 0.83–4.16 possessions per graded minute, a dead share of 0.15–0.82 and 0–56.5 sub-second stoppages per 90 minutes). Round 13 has started with a measurement (D55); it has no result yet.

## 5. Dead-ball artifacts and the retrained model set

Both providers mark dead ball in a way automatic tracking will not:
- DFL freezes the ball on every dead frame.
- SkillCorner detects the ball on about 18% of dead frames, against about 91% of live frames.

Models trained on them can learn that a still or missing ball means a stoppage. On external DFL matches where the dead-frame ball instead follows the nearest player, spellF1 against the N = 3 targets fell from about 0.68 to about 0.53.

We retrained every model on the same matches with three changes to the training data (D2):
1. **Dead-ball handling.** Each dead run keeps its ball (20%), loses it (20%), or has it handled (60%). Handled means it travels to the restart taker, follows the taker, then rests at the restart spot.
2. **Ball visibility.** The degraded view's ball visibility is drawn per half from 30–90%.
3. **Identity swaps.** Home and away tracks occasionally swap positions, as in a cross-team identity switch.

**Guard views.** Two guard views with deliberately different settings measure the effect:
- the dead-frame ball follows the player nearest to it;
- 10-second cross-team swaps.

**Result.**
- The retrained set on its own gained +0.090 with the dead ball carried, but lost 0.015 on the standard views, missing its 0.01 limit. It was not adopted alone.
- An equal blend with the original set, tested under a new rule (D3), passed on development and on the holdout:

| Blend vs current models | Holdout change |
|---|---:|
| Dead ball carried | +0.062 |
| Standard views | −0.005 |
| Team swap | +0.009 |

D3 was a second attempt on the same development matches, made after seeing the D2 result. We report it as weaker evidence for that reason.

## 6. Possession rules implemented

| Situation | What the system does |
|---|---|
| **What counts as possession** | Following CHALLENGE.md, only *controlled* possession changes hands. The external labels instead change the owning team on every opponent touch. We therefore tuned the decoder against external targets re-expressed under the challenge's definition: within a live segment, an A–B–A sequence whose interior B lasts under N seconds is merged into A. |
| **Choosing N** | External sources put N between 1.75 s and 4.5 s: Metrica sample events, raw (1.75 s) and with a pass/carry/hold control rule (4.25–4.5 s), and SkillCorner phases of play (4.0 s). We use N = 3 s, which minimises the worst-case loss over that interval. Without the Metrica data, SkillCorner alone gives 2.0–4.0 s, whose midpoint is also 3 s. N is used only to score external candidates; it is not a rule applied to the output. |
| **Loose balls and contested play** | The decoder's ownership cost (14) keeps the last team in possession unless the models give sustained evidence of a change. A brief touch, deflection or aerial duel therefore rarely creates a possession. |
| **Challenges** | A change is accepted when the other team's control probability stays high long enough to outweigh the switch cost. The cost is lower when the boundary network signals a genuine change. |
| **Stoppages** | `D` when the blended stoppage probability passes 0.53 under the decoder. The evidence includes ball speed, the ball leaving the pitch, distance to the lines, and player movement. |
| **Restarts** | Leaving `D` costs 2. The team after a restart follows the ownership evidence; there is no rule that hands the restart to a particular team. |
| **Halves** | Each half is decoded separately. Frames outside the published half spans are `X`. Inside the halves, every frame gets `H`, `A` or `D`. No constant is tied to particular frames. |
| **Before the kick-off** | No team has possession before the kick-off (D27). After decoding, every graded frame from the start of a half up to the start of that half's Smart Tagging `Kick Off (Start)` tag becomes `D`. The organisers' loader describes each kick-off tag as a window of about 5 s containing the real kick-off, so the tag's start is never later than the kick and the rule cannot overwrite live play. A half uses the rule only if exactly one such tag ends at or after its first frame and starts within 5 minutes of it. The second half is graded from about two minutes before its kick-off, so the rule acts there; the first half is graded from the kick-off itself, so it changes nothing. On external matches every half starts at its kick-off, so the rule cannot be scored there; it was adopted as a constraint, on tests and two identical rebuilds. |
| **Abstention** | We do not write `X` inside the halves. The ranking metric forward-fills it, so it would only lower macroF1. |

## 7. Assets used and dropped, and development history

**Used:** tactical ATD tracking, its metadata and the homography `valid` flag; from Smart Tagging, only the start and end times of the `Kick Off (Start)` tags (D27, section 6; the direction of attack is read from each half's tag on, D50, section 2).

**Dropped:**
- **Tactical and broadcast video.** Ball detectors and locally run vision models were explored during earlier research and dropped. The submitted pipeline does not read the video.
- **Broadcast ATD.** A camera-alignment prototype failed its external checks, and so did D31 and D31a, which tried to align the broadcast tracking to the tactical timeline on a synthetic harness built from external tracking. No broadcast file was read in D31 or D31a.
- **Smart Tagging, apart from the kick-off tags.** No external data matches its format. D4 tested restart timestamps simulated from DFL events, and no setting improved on D3 (section 4). The phases, teams and other set pieces are not read.
- **The public 5-minute ground truth.** The rules forbid training or tuning on it.

**Development history.** Before the organisers' clarification of 3 November, part of our research trained and tuned models on the public 5-minute window. That included a TabPFN-based pipeline, described in an earlier, now archived repository. We also ran a short hand-annotation pilot on the provided game on 24 October; it was rejected and never used. None of that work, its models, thresholds or outputs is used in this submission. The earlier research repository is available to the organisers on request. Some feature formulas were first written during that period; they were re-implemented and fitted on external data only. During development we used hosted coding assistants in a workspace that contained the match data. Match data is now kept out of any AI tool's reach, and competition builds are run locally.

Items we are disclosing:
- The original model set's degraded training view keeps the ball visible 60% of the time, which matches DATA.md's published figure of 41% missing. The retrained set uses 30–90% instead. On 11 November we asked the organisers whether a dropout rate may be set from that published figure. Their reply of 12 November gave no yes or no; it restated the clarification and added that anything not explicitly allowed is not permitted. We kept the constant and disclose it here.
- The organisers' `games.py` pitch size (105.3 × 68.0 m) was replaced at inference by the metadata's 105 × 68 m.
- The direction of attack of each half (D50, section 2) is computed at build time from that half's provided tracking: one sign per half, from the half's frames after its tagged kick-off with at least seven visible players per team. It uses no label and has no fitted value. Its one setting, seven players, was written in the rule before anything was scored, and it was checked only on external halves. We regard it as a reading of the input, like the 162 features, and not as learning on the provided tracking. Unlike the features, which use windows of up to 18 s, it summarises a whole half.
- The stack's first build on the provided match refused at a direction check, and the rule was then aligned to each half's tagged kick-off (section 2). The refusal message was the only output read: no number, frame, label or half. The change was chosen from a known difference between the provided match and the external halves (the second half's graded minutes before the kick-off), not from the provided tracking, and it leaves every external result unchanged. Still, it is the one change to the build made after a build had run on the provided match.
- PFF FC's terms for its World Cup data are unresolved (DECISIONS.md D7). No model was trained or adapted on it, and the build reads none of it. But in round 12 its development matches were one of the eight groups of the score that chose the stack's changes and its ownership cost, and a fresh PFF set was part of the stack's confirmation (sections 3 and 4).
- The rules of D49 and D48 cited DATA.md's published figure, the ball missing on 41% of graded frames, as one reason to work on long hidden-ball stretches. None of D49's values comes from that figure: the DFL visibility range is D2's 30–90% (50–90% for SkillCorner), and the run lengths were set by judgement before D46a measured hidden-run lengths on external tracking.

## 8. Limitations

- **Domain shift.** The training data come from other leagues and other tracking systems. The dead-ball simulation, the degraded views and the long hidden-ball views approximate automatic tracking but are not fitted to it.
- **No labelled validation in the target domain.** Every number in this document is external. The competition score is unknown.
- **Definition uncertainty.** If the ground truth counts every touch as a change of possession, our decoder settings cost about 0.01–0.02 on that convention, judging from the raw-label scores.
- **Short possessions.** Possessions of a second or two are hard to recover from noisy tracking. On external development matches D3, an earlier submission, recovered under half of the true possessions shorter than 5 s, and predicted 20–25% fewer possessions than the targets (D11). Neither a finer decoder (D12) nor training on the challenge's convention (D13) improved this. Longer training narrowed the gap (D15): on clean DFL D15 predicted 263 possessions against 281 true (D3: 219) and matched 93 of 184 ownership changes within one second (D3: 68).
- **Long hidden-ball stretches.** On real automatic tracking (PFF, SkillCorner), a third to two fifths of the hidden-ball frames lie in runs over 10 s, where the decoded owner is right less often: on PFF development, D43 agreed with the target owner on 95.7% of seen-ball frames and 85.1% of hidden-ball ones (D46, D46a). The stack raised the recovery of PFF's possessions whose ball is seen on under 20% of their frames from 0.17 to 0.27, but its agreement on hidden-ball frames only to 86.2% (D48's development). Most of that loss remains, and the long hidden-ball views are synthetic, not fitted to the competition tracking.
- **Identity errors.** Only cross-team swaps were simulated. Fragmented tracks and duplicate tracks are handled only through dropout in the degraded views.
- **Holdout reuse.** The confirmation matches were reused across decisions, which makes the gains optimistic. The stack's confirmation scored new long hidden-ball views of the holdout matches and spent the untouched PFF set, and D54 then scored the four 2019–20 SkillCorner confirmation matches, where the stack was 0.005 below D43. The only external matches still unused are the 24 PFF games of D7's three training slots, which were never parsed because no model may train on PFF; they remain available for measurement and confirmation.
- **No adaptation to the provided game.** D5 showed on external data that label-free adaptation helps across tracking providers. RULES.md allows learning without human labels on the provided game, but the organisers' 3 November clarification requires external data for all training. We asked on 11 November which applies. Their reply of 12 November restated the clarification and added that anything not explicitly allowed is not permitted. Label-free learning on the provided game is not on the allowed list, so nothing is fitted on the provided tracking.

What we would do with more time:
- train on more and noisier external tracking;
- model the ball's trajectory when it is missing: D28 showed that a straight line through gaps of up to 1 s is not enough, and D51 that a network can learn to place a hidden ball to within 6–8 m at the median without that helping possession;
- validate on a fully untouched external competition.

## 9. Reproduction

Verified to give the same file on Apple Silicon macOS (Homebrew Python 3.12.13) and in Linux containers (x86_64 and arm64, Python 3.12.14). Point `MTAH_DATA_DIR` at a folder containing `tactical/BAR-ATH_tactical_atd.xml`, `tactical/BAR-ATH_tactical_atd.txt`, `tactical/BAR-ATH_tactical_homography.csv` and `tactical/BAR-ATH_tactical_smart_tagging.xml`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r dfl_submission/requirements.txt
export MTAH_DATA_DIR=/path/to/competition-bundle
.venv/bin/python -m dfl_submission.stack_trial build --out data/submission.csv
.venv/bin/python evaluation/validate_submission.py data/submission.csv
shasum -a 256 -c SUBMISSION_SHA256.txt
```

Before reading the competition tracking, the build checks that the twelve networks, the trees, the code it runs and the switches are those recorded in D48's confirmation. It then pairs each half with its kick-off tag, checks each half's direction of attack (section 2), and refuses to run if any check fails. Inference runs the twelve networks on both team orders and on the pitch and its mirror image, then the decoder and the kick-off rule (section 6). The submitted file was rebuilt byte for byte twice in the repository's environment, and then on a copy of the working tree twice on macOS in fresh virtual environments and in the official `python:3.12.14-slim` image under `linux/arm64` and `linux/amd64`. `release/check.sh` on a commit is pending Derek's approval of a commit (`TBD-COMMIT`). The README gives the Docker command.

The README documents:
- how to retrain every model from the public external data;
- how to re-run every tuning and confirmation step;
- the guard that refuses to fit while the competition data is reachable.

Every constant and its source are listed in [CONSTANTS.md](https://github.com/Derekko-web/mtah2027-submission/blob/main/CONSTANTS.md).

## 10. Licences and credits

- The DFL data is credited to the DFL and to Bassek et al. (2025), under CC BY 4.0. It was parsed, resampled, synthetically degraded and used to fit our model weights.
- SkillCorner Open Data is used under the MIT licence.
- The Metrica Sports sample data is credited to Metrica Sports.
- PFF FC's 2022 FIFA World Cup tracking was shared on request by PFF FC (Gradient Sports). It was used for measurement, as a guard, as one group of the round 12 development score, and once in each of four confirmations (section 3); its terms are discussed in DECISIONS.md D7 and section 7. None of its data is in our repository, which holds only file hashes and scores, and no model in the submission was trained on it.
- PathCRF (Kim et al., KDD 2026) was run from its own repository under MPL-2.0 for the D18 comparison only. None of its code or weights is in our repository or in the submission.
- The challenge's evaluation code, `games.py` and starter code come from the organisers' repository under its MIT licence.
