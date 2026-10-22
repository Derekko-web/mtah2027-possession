# Directional ownership interactions and transition weighting

None of the 42 new evaluations passes the frozen promotion gates. The selected
eight-model candidate remains unchanged, SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
No seeds 43/44 follow-up, full candidate or promotion was made in this round.
The buffered short-precision feedback still fails: 3 matches at IoU≥.5, required 4.
This is a rejected research round, not a new accuracy improvement.

## Starting point and feedback

The source tree, selected entry and 1,249-row ledger were archived before the
plan; selected8 artifacts and confirmation were copied and hashed under
`data/control_ownership_graph_v1/`. All 1,249 preceding rows remain unchanged.
The ledger now contains 1,291 dependent score rows: 778 inherited and 513 added
in this chat. This round adds 21 original screen rows and 21 decoder-only rows.
Repeated references and reused predictions are not independent experiments.

The diagnosing-bugs skill supplied the feedback discipline. This command was
run before fitting and failed deterministically:

```sh
.venv/bin/python experiments/require_short_precision.py --scores data/control_pair_motion_v1/dead_trim_confirmation/scores.csv --model motion_dead_trim_seed 42
```

Output: `Short IoU>=.5 matches: 3; required: 4`, followed by the expected failure.
The same failure reproduces from a minimal three-column, one-row aggregate CSV.
This is a statistical model-quality symptom; no deterministic software defect
or manual endpoint correction was assumed. Synthetic tests cover the new code's
physical invariants; a hard-coded test requiring a particular public endpoint
would not establish a valid modeling fix. The feedback remains red after research.

The ranked hypotheses, shared before testing, were: signed interactions retain
information lost by distances/speed magnitudes; transition-focused fitting helps
brief controls; conservative proposals use that evidence more safely than broad
replacement. [OWNERSHIP_GRAPH_PLAN.json](OWNERSHIP_GRAPH_PLAN.json) was frozen
before feature extraction/fitting. Later decoder checks have their own plan and
explicit chronology below.

## New observations and fitting

The representation uses only local native tactical and strictly aligned broadcast
tracking. It reads no media or targets. Mutual nearest same-team matches at ±5
native frames require a second-best margin ≥ .25m and speed ≤ 15m/s. Ball velocities
are limited to 80m/s; the entire native interval must be valid. Broadcast alignment
must continuously enclose native ±5 over tactical ±10 without orientation changes.
Ambiguity and missing observations remain missing; slots are never player IDs.

For each player, features are incoming/outgoing ball-relative radial velocity,
absolute tangential relative velocity, player/ball heading cosines, linear
closest-approach distance over a past/future horizon of at most 1s, radial ball
impulse, nearest opposing-player distance and teammate support within 5m.
Gaussian ball-distance pooling at bandwidth 2/5m uses only players within 10m.
The extrapolation is a feature calculation, not an extra second of observed data.
Shared turn/coverage inputs and past/future 25/75-frame summaries produce 70
columns per camera, 140 total. Features are invariant to translation, orthogonal
rotation/reflection, slot permutation and team relabeling as tested synthetically.
There are no absolute coordinates, player IDs or clock/frame inputs.

The total new observation reach is at most 135 tactical frames, inside 250-frame
target exclusions. Actual feature/info construction through frames 18934 and 19059
agrees on every public row. Public complete-kinematic coverage is 87.47% tactical
and 38.88% broadcast. Input/cache/source hashes are retained in `inputs/manifest.json`.

Three joint XGBoost heads use the existing depth 3/160-tree settings and seed 42:

- Graph frame: original 102 ordered  + 68 broadcast + 140 graph inputs, 310 total.
- Graph transition: the same 310 inputs, transition-focused training weights.
- Ordered transition: the original 102 inputs, a matched weighting control.

Transition weights multiply eligible rows by 4 within 25 frames of an adjacent
training-only H/A change, confined to the same eligible contiguous component.
Augmented H/A/D class masses are then rebalanced equally. Target construction,
weights and augmentation use eligible fitting labels only. Every fitting index,
validation index, augmented target and weight is saved. Masking all other labels
toX exactly regenerates them. No fitted preprocessing, calibration, early stopping,
held-out weights, weak targets or reserved annotations are used.

There are 39 fitted checkpoints across 13 folds and 3 heads. Broad replacement
changes only the original ordered head inside the eight-model pipeline, where
new source coverage exists. Conservative short proposals use the existing
5-frame smoothing,8..125-frame duration, confidence ≥ .65, physical support and
opposite 13-frame reference flanks. Proposals preserve every reference D exactly.
All seven other head arrays remain unchanged.

## First screen: 21 rows

All cells are seed 42 spellF1. Full scores, official reports and diagnostics remain
under `screen/`.

| Head / decoder | Buffered minutes | Purged spells | Buffered100s |
|---|---:|---:|---:|
|Selected reference|.741176|.685714|.670270|
|Graph frame / replace current|.589474|.589474|.526316|
|Graph frame / short proposals|.741176|.685714|.670270|
|Graph transition / replace current|.589474|.578947|.540541|
|Graph transition / short proposals|.741176|.685714|.670270|
|Ordered transition / replace current|.741176|.651429|.688889|
|Ordered transition / short proposals|.741176|.685714|.670270|

Graph replacement produces 475/530 falseD buffered,476/517 purged and 412/472
longer-block, versus 88/142/16 reference. It raises D recall slightly but loses
primary and MacroF1. Graph proposals change no predictions in any protocol.
Ordered proposals change 3 buffered frames, reducing MacroF1 and short IoU despite
unchanged primary; other protocols are exact. Ordered replacement has a modest
longer-block gain but worsens purged primary and increases falseD in every protocol.
There are 15 primary regressions among 78 clipped-block comparisons.

The new graph heads choose H on 23 of 33 true short H frames in buffered/purged
joint raw argmax, versus 0 for ordered-transition. This is a local diagnostic
lead, not a matched possession. All three heads choose the wrong raw team on the
71-frame away interval; the 19-frame home interval is also unsupported by their
raw H/A decision. Existing rescue heads preserve that 19-frame recovery in the
full pipeline. The graph's local signal therefore does not replace the old models.

## Separately frozen decoder follow-up: 21 rows, no new fitting

[GRAPH_CONDITIONAL_PLAN.json](GRAPH_CONDITIONAL_PLAN.json) was frozen after
inspecting the first screen and before running any decoder follow-up. It reuses
all 39 seed 42 checkpoints/probabilities. It retains the original substantial +.04
and extra buffered short-match objectives and all secondary gates; it is not an
objective relaxation or an independent confirmation.

Conditional-current replacement holds the original ordered raw D probability
exactly and changes only its conditional H/A where observed. Downstream D can
still change through state competition, so final D outcomes remain audited.
Source-extent refinement starts from selected 8 and the unchanged six-model
baseline. It may only shorten an automatically added H/A component where the
two live references disagree. New conditional H/A events must be 8..125 frames,
fully observed over the clipped intersection and mean confidence ≥ .65. A uniquely
best overlap identifies one retained core; the baseline is restored on the
component outside it. Ties preserve the reference. D and all original decisions
outside source-added components stay exact. No true endpoint enters inference.

| Head / decoder | Buffered minutes | Purged spells | Buffered100s |
|---|---:|---:|---:|
|Selected reference|.741176|.685714|.670270|
|Graph frame / conditional current|.697143|.627027|.577778|
|Graph frame / source extent|.741176|.685714|.670270|
|Graph transition / conditional current|.741176|.655556|.617143|
|Graph transition / source extent|.741176|.685714|.670270|
|Ordered transition / conditional current|.741176|.651429|.688889|
|Ordered transition / source extent|.741176|.685714|.670270|

Holding the original raw D probability avoids most of the falseD increase.
Graph conditional variants reduce falseD to 85 buffered, 139 purged and 11–12
longer-block, but they regress primary/Macro or short precision. Even graph
transition's unchanged buffered primary has MacroF1 .869604 versus .870833,
outside the frozen .001 allowance. Every graph conditional method loses purged
and longer-block primary, and longer-block short matches fall 3→2.

The ordered conditional control improves only the longer-block result:
spellF1 .670270→.688889, MacroF1 .844383→.846347, falseD 16→15 and short IoU
.000616. It loses purged primary .685714→.651429 and worsens buffered short IoU
and falseD. It is a retained lead, not an eligible finalist.

Graph source-extent variants change 20 buffered frames and 19–20 purged frames;
longer-block outputs remain exact. They slightly raise MacroF1 but reduce mean
short IoU, without another short IoU.5 match or primary gain. Ordered extents
change nothing. There are 13 primary regressions among 78 clipped-block checks.
All source-extent D masks are exact.

The diagnostic explains the failed 33-frame lead: graph conditional decoding's
best same-team run is 104 frames long, with best IoU .201754. Correct local
argmax on 23 true frames does not locate a useful interval. The 71-frame away
miss and 19-frame overlong endpoint remain. Further arbitrary contraction of
known public cases is not justified by these results.

## Integrity, repairs and preservation

Both screens agree with all 42 official score reports and reproduce every saved
prediction through shared inference. All 39 fitted models reload to exact saved
probabilities. The decoder-only stage verifies all 39 parent checkpoint hashes;
it does not count them as new fits. Training buffers, target/weight isolation,
feature schema, actual padded-prefix parity and source/input/cache identities pass.
All 120 tests pass. There are no eligible finalists in either frozen check file.

An initial weighting test used training components so short that every row
received the same multiplier; class normalization correctly made its weights
identical to the control. Longer synthetic components distinguish the objectives.
The initial test source and failed log remain. An exploratory diagnostic initially
attached an unrelated run's length when IoU was zero; its original source/log
remain, and the corrected summary reports no matching run. Neither correction
changed fitted code, models, predictions, plans or score rows.

The selected CSV, its manifest, model/probability files and source hashes remain
unchanged. Only the selected entry's history count was updated; its original
bytes remain in the pre-round archive. Every prior candidate and all 1,249 prior
ledger rows are preserved. New evidence stays under ignored
`data/control_ownership_graph_v1/`; no commits, uploads, submissions, publication
or external contact occurred. No media, private labels or reserved annotation
content were read. The 24 unlabelled reserved-window metadata SHA is unchanged.

The fresh-workspace audit prepares the unchanged baseline and replays both
screens without historical session files. It reuses verified fit artifacts and
does not claim new independent fitting. Evidence includes `integrity.json`,
`conditional_integrity.json`, `fresh_reference_audit.json`, fitting records,
official reports, `prediction_change_counts.csv`, both diagnostic CSVs and logs.

## Reproduction

Use the existing Python3.12 environment and pinned research requirements. First
reproduce selected 8 and its upstream caches/confirmation following
[PAIR_MOTION_REPORT.md](PAIR_MOTION_REPORT.md). In a workspace with fresh graph
stage directories:

```sh
.venv/bin/python experiments/prepare_graph_baseline.py
.venv/bin/python experiments/prepare_ownership_graph.py
.venv/bin/python experiments/research_ownership_graph.py
.venv/bin/python experiments/finalize_ownership_graph.py
.venv/bin/python experiments/research_graph_conditional.py
.venv/bin/python experiments/research_graph_conditional.py --verify-only
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

With the verified caches/models present, the finalizer and conditional
`--verify-only` command replay both screens. Optional
`finalize_ownership_graph.py --audit-history` additionally verifies the original
archive/protected files; ordinary reproduction does not need them. The selected
short-feedback command above intentionally still fails. These remain adaptive
same-match checks on 7500 labels/17 possessions, not independent generalization,
private full-match accuracy or proof of the maximum score.
