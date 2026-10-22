# Native motion and conservative dead-ball trimming

The new native broadcast-motion head gives a small, reproducible false-dead-ball
improvement when it may change only the selected reference's D predictions.
All nine paired incremental checks pass. Buffered spellF1 stays .741176; purged
spellF1 improves .674286→.685714; longer-block predictions remain identical.
This does **not** meet the original substantial-gain or extra buffered short-match
objectives. Maximum score and private full-match accuracy remain unverified.

The full eight-model candidate is `data/candidate_motion_dead_trim/candidate.csv`,
SHA256 `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
The seven-model predecessor remains at
`data/candidate_broadcast_closure/candidate.csv`, SHA256
`dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce`;
its entry point is preserved in [BROADCAST_MODEL.md](BROADCAST_MODEL.md).
The candidate was selected after the final review recorded in
[MOTION_CODE_REVIEW.md](MOTION_CODE_REVIEW.md).

## What was preserved and what changed

Before this round, `data/control_pair_motion_v1/experiments_before.tar.gz`
preserved the source tree, selected entry and 1,177-row ledger. The selected
candidate and all-seed confirmation were copied into `baseline_candidate/` and
`baseline_confirmation/`; `baseline_manifest.json` records their identities.
All 1,177 prior rows remain unchanged. This round adds 39 original screen rows,
15 symmetric-decoder checks and 18 confirmation rows: 72 rows, bringing the total
to 1,249 (778 inherited, 471 added in this chat). These are dependent score rows,
including repeated references and reused predictions, not independent experiments.

No media was inspected or decoded this round. Local supplied tracking is processed
locally; source observations, features, checkpoints and outputs stay in ignored
`data/`. Feature construction reads no possession targets. No private labels,
manual labels, manual final-frame patches, new dependencies, commits, uploads,
submissions, publication or external contact were used. All 24 reserved annotation
windows remain unlabelled and unused; their metadata SHA stays
`5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c`.

## Frozen representations and chronology

[PAIR_MOTION_PLAN.json](PAIR_MOTION_PLAN.json) freezes four joint-frame heads and
three decoders over the original three protocols, initially at seed42. All heads
retain the original 102 ordered features and unchanged depth3/160-tree fitting:

| Head | Added inputs | Total inputs |
|---|---:|---:|
|Broadcast taps|68 broadcast geometry +42 fixed temporal taps|212|
|Tactical motion|49 native tactical-motion summaries|151|
|Broadcast motion|110 broadcast geometry/taps +49 native broadcast-motion summaries|261|
|Fused motion|208 combined source inputs|310|

At native offsets ±5 and ±25, mutual nearest same-team position matches require
a second-best margin of at least .25m and speed at most 15m/s. Ball speed is capped
at 80m/s; missing, ambiguous or cut observations stay missing. Features include
nearest-player relative velocity, minimum relative velocity within3m, matching
coverage and a distance/relative-speed kernel. Past/future kernel means use
25/75/125 tactical frames. No player IDs or absolute coordinates/timestamps become
features. These set matches approximate correspondence; identities are unverified.

Broadcast velocities are calculated on the native camera timeline, avoiding
mapped-frame rounding artifacts. The ±5 motion requires continuous same-orientation
alignment over tactical ±10 and enclosure of native ±5; the ±25 motion uses
tactical ±35 and encloses native ±25. Alignment, summary and enclosure together
reach at most 210 tactical frames, inside existing 250-frame target exclusions.
Actual feature construction through frames18934 and19059 agrees on every public
row. Public valid motion coverage is 87.47% tactical, 38.88% broadcast, 89.64% either.
Full-match broadcast inputs later agree exactly with the prefix on public rows.

Two synthetic repairs are retained explicitly:

- Before tracking-feature generation or fitting/scoring, an extent-refinement
  test showed that selected-reference flanks reject events inside their own broad
  added run. Closure/island protection now uses the six-model reference only for
  events overlapping a same-owner component added by model7; other gates retain
  model7. The original plan/source/test and failed log remain in
  `synthetic_gate_revision/`. The plan records the revision; score gates did not change.
- Before inspecting the original39 scores, a second synthetic case exposed
  team-order bias when opposite event pads overlap. The old decoder and every
  result remain. [PAIR_EVENT_SYMMETRY_PLAN.json](PAIR_EVENT_SYMMETRY_PLAN.json)
  separately freezes per-frame conditional probability×kernel resolution, with
  exact ties retaining the reference. Its15 checks reuse all52 original checkpoints;
  no new fitting occurs. The old event decoder is ineligible for promotion.
  All actual event predictions equal the old version here, although only the new
  version passes the synthetic overlap case. Actual swap invariance also passes.

## Original screen: all methods rejected

Each table cell is seed42 spellF1. Original39 rows are in `screen/scores.csv`;
the extra15 symmetric checks are in `symmetric_screen/scores.csv`.

| Head / decoder | Buffered minutes | Purged spells | Buffered100s |
|---|---:|---:|---:|
|Selected reference|.741176|.674286|.670270|
|Broadcast taps / replace current|.731429|.677778|.637838|
|Broadcast taps / live replacement|.675000|.606061|.560000|
|Broadcast taps / events|.717647|.651429|.637838|
|Tactical motion / replace current|.557895|.530000|.578947|
|Tactical motion / live replacement|.557895|.502564|.512821|
|Tactical motion / events|.705882|.640000|.648649|
|Broadcast motion / replace current|.741176|.685714|.648649|
|Broadcast motion / live replacement|.675000|.606061|.594286|
|Broadcast motion / events|.717647|.651429|.637838|
|Fused motion / replace current|.550000|.536585|.555556|
|Fused motion / live replacement|.536842|.492308|.529730|
|Fused motion / events|.717647|.651429|.648649|

The symmetric event variants have the same actual scores as their event rows.
No method clears the frozen +.04 buffered improvement and extra short IoU.5-match
gates. Among208 clipped-block comparisons,72 primary regressions remain recorded.
The event diagnostic counts779 proposed and47 accepted events; rejected gates
include closure492, confidence78, contacts77, duration60, referenceD23 and
protected island2. This is additional adaptive analysis, not independent evidence.

Broadcast-motion replacement reduces seed42 falseD92→88 buffered and152→142
purged, retaining exact D recall. The87-frame home interval improves recall
.7471→.7931 buffered and .5172→.6322 purged; purged IoU crosses .5. But broad
replacement loses longer-block ownership: the81-frame away recall falls
1→.6790 and the90-frame away recall .7111→.5889. The19-frame home endpoint and
33-frame home overextension remain; the71-frame away interval stays missed.
These failures motivate a narrower prospective test, without treating this
parent screen as a passing substantial improvement.

## Separately frozen D-only confirmation

[MOTION_DEAD_TRIM_PLAN.json](MOTION_DEAD_TRIM_PLAN.json) was frozen after the
seed42 screen and before any seed43/44 motion evaluation. It changes the objective
explicitly to incremental falseD reduction with strict protections. It keeps the
261-input broadcast-motion model and existing thresholds. The canonical decoder
computes model7 and the broad alternative, then accepts alternative H/A only
where model7 says D. Every original H/A/X stays exact; it cannot add D. Its
`replace_current` branch returns before the retained historical event decoder.

The confirmation refits39 checkpoints (13 folds×3 seeds), with seed42 probabilities
exactly reproducing the parent screen. All18 evaluations have official agreement.

| Protocol | Seed | SpellF1 before→after | MacroF1 before→after | FalseD before→after | D recall, unchanged |
|---|---:|---|---|---|---:|
|Buffered minutes|42|.741176→.741176|.870211→.870833|92→88|.748128|
|Buffered minutes|43|.741176→.741176|.869194→.869971|93→88|.744724|
|Buffered minutes|44|.741176→.741176|.870658→.871436|94→89|.750851|
|Purged spells|42|.674286→.685714|.870074→.871604|152→142|.784888|
|Purged spells|43|.674286→.685714|.870008→.872610|156→139|.786930|
|Purged spells|44|.674286→.685714|.871253→.872937|154→143|.790334|
|Buffered100s|42|.670270→.670270|.844383→.844383|16→16|.649421|
|Buffered100s|43|.670270→.670270|.844065→.844065|15→15|.649421|
|Buffered100s|44|.670270→.670270|.844083→.844083|15→15|.648741|

All nine pairs satisfy the frozen primary, MacroF1, short-IoU, mostly-recovered
short, falseD, D-recall, missed-stoppage, official spurious/missed, frame-preservation
and true-D-retention gates. Each buffered/purged seed strictly removes falseD.
There are0 primary regressions among39 clipped-block comparisons. Mean short IoU
gains .00647–.00817 buffered and .01603–.02724 purged; longer-block predictions
are identical. Buffered short IoU.5 matches stay3; purged improve2→3. Official
purged total matched/spurious/missed counts improve12/6/5→13/5/4; buffered13/4/4
and longer13/7/4 stay unchanged. No previously true-D prediction is lost.

The earlier +.04 buffered and additional buffered short-match criteria remain
**false**. The19-/33-frame overextensions and71-frame away miss remain. This is a
modest secondary improvement and a purged-primary gain, not the requested maximum.
The1000-draw90% spellF1 bootstrap intervals overlap: buffered [.58824,.88235]
unchanged; purged [.52571,.81143]→[.53714,.82286]; longer [.52973,.81081] unchanged.
Only7500 public labels and17 true possessions exist. Repeated public selection,
overlapping training sets, seeds and protocols make these correlated same-match
development checks; bootstrap does not correct adaptive selection bias.

## Full artifact and integrity

Two independent full builds match every file byte-for-byte, including manifests.
The output has199319 rows, frames0..199318,25fps,H/A/D/X. Eight model checkpoints
and six probability arrays verify; every retained model7 artifact stays byte-identical.
Only811 referenceD frames change:437 toH,374 toA. Final counts areD77197,H49936,
A31724,X40462. Every referenceH/A/X remains exact. These811 changes are **not**
811 verified full-match corrections: private targets were never accessed.

The new head fits only the7500 public targets after confirmation. Its full
training-set predictions are not reported as validation. Public/full features
and ordered inputs match; source, actual camera input, cache, checkpoint and
probability hashes pass. Both full CSVs pass the official format validator.
All112 tests pass. Independent Standards and Spec reviews find no source or
artifact or documentation issues; the two axes are recorded separately.

Evidence under `data/control_pair_motion_v1/` includes original/repaired synthetic
logs, `inputs/`, `screen/`, `symmetric_screen/`, `dead_trim_confirmation/`,
`full_inputs/`, `integrity.json`, `dead_trim_integrity.json`,
`full_artifact_integrity.json`, build/validator/test logs and the audit scripts.
Fresh baseline preparation and both ordinary finalizers pass without historical
session snapshots; `--audit-history` separately verifies preserved prior rows
and artifacts. The fresh-workspace audit reuses verified fit artifacts; it does
not claim a third full refit. The two full builds provide that separate check.

## Reproduction

Use the existing Python3.12 `.venv` and pinned research requirements. Reproduce
the upstream selected broadcast candidate, confirmation and original advanced
validation inputs first, following
[BROADCAST_CLOSURE_REPORT.md](BROADCAST_CLOSURE_REPORT.md). Then, with fresh stage
directories where scripts require them:

```sh
.venv/bin/python experiments/prepare_pair_baseline.py
.venv/bin/python experiments/prepare_pair_motion.py
.venv/bin/python experiments/research_pair_motion.py
.venv/bin/python experiments/research_pair_event_symmetry.py
.venv/bin/python experiments/confirm_motion_dead_trim.py
.venv/bin/python experiments/finalize_pair_motion.py
.venv/bin/python experiments/finalize_motion_dead_trim.py
.venv/bin/python experiments/prepare_full_pair_motion.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_motion_dead_trim_candidate.py --output data/candidate_motion_dead_trim_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_motion_dead_trim_fresh/candidate.csv
```

With existing verified caches/confirmation, only tests, the builder and validator
are needed for a fresh full output. Optional historical auditing uses
`finalize_pair_motion.py --audit-history` and requires the original local snapshot.
