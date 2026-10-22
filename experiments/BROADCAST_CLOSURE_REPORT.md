# Broadcast observation and source closure

The new broadcast-camera method improves public development spellF1 by about
.011 in all nine paired comparisons. This is an incremental gain. It does not
meet the original buffered +.04 goal, add a short IoU .5 match, or establish the
maximum or private full-match score. All generation and integrity checks pass, and independent Standards and Spec
reviews are complete with no unresolved findings. The seven-model candidate is
now selected; the protected six-model stoppage predecessor remains preserved.

## What was tried and retained

The preceding timing/contact/weak-ownership rounds are preserved and summarized
in [WORK_HISTORY.md](WORK_HISTORY.md). This round introduces a different local
observation source, the optional unsynchronized broadcast tracking. No video was
decoded, no new annotations or private labels were used, and no competition
media or tracking extracts were sent to a hosted model. Derived artifacts stay
under ignored `data/control_broadcast_v1/`.

Four plans record the adaptive search in order:

1. [BROADCAST_ALIGNMENT_PLAN.json](BROADCAST_ALIGNMENT_PLAN.json): two label-free
   physical alignment profiles, then four classifier heads and two decoders.
   There are 27 score rows, including references. All methods fail the original
   substantial/extra-short-match objective. Broad joint-head replacement helps
   buffered/purged scores slightly but regresses the longer-block protocol.
2. [BROADCAST_SOURCE_INTERVAL_PLAN.json](BROADCAST_SOURCE_INTERVAL_PLAN.json):
   physical source proposals, two smoothing widths and two learned joint heads.
   All 15 rows are retained. No proposal clears every gate; predictions remain
   unchanged. The two-opposite-flanks rule rejects useful source evidence beside
   a reference dead-ball boundary.
3. [BROADCAST_INTERSECTION_PLAN.json](BROADCAST_INTERSECTION_PLAN.json): allow
   a dead-ball or same-owner anchor on one side while requiring opposite control
   on the other; compare whole source closure and source/learned intersection
   with each joint head. All 15 rows are retained. Whole closure improves the
   primary score, but this plan still fails its additional short-match and
   fewer-missed-possession gates.
4. [BROADCAST_CLOSURE_CONFIRMATION_PLAN.json](BROADCAST_CLOSURE_CONFIRMATION_PLAN.json):
   a separate incremental-primary objective frozen after the seed 42 screens and
   before testing seeds 43/44. Only unchanged `frame_closure` is confirmed. All 18
   reference/candidate rows are retained. All nine pairs pass primary +.01,
   MacroF1 tolerance, mean short IoU, existing short recovery, identical D masks,
   and nonincreasing official IoU .5 spurious/missed counts. This does not convert
   the preceding failed plans into successes.

The 75 rows are dependent method/protocol/seed evaluations, not 75 independent
experiments. The full ledger has 1,156 rows: 778 inherited and 378 added in this
chat. Reproduction after the provenance repair repeats existing rows; it does
not inflate this inventory. One mistaken synthetic test fixture was corrected
without changing the algorithm; its failure log remains local.

## Alignment and model

The two sources have different frame counts and timelines. Club-name metadata
reconciles their team assignments without target labels. Every 25 tactical frames,
the aligner compares team-specific player formations with eligible broadcast
anchors under four pitch reflections. It caps nearest-player distances at 10m,
requires temporal neighborhood consistency, refines the match within 25 broadcast
frames, and rejects ambiguous distant alternatives. Linear mapping only joins
adjacent accepted anchors with the same orientation and rate .8..1.2. Cuts,
missing homographies, weak team coverage and uncertain matches remain unmapped.
The search covers the whole broadcast recording and is offline.

Strict alignment is chosen prospectively by physical feasibility, before target
labels are read. It maps 44.79% of the public window; an aligned broadcast ball is
observed in 41.05%. There are 2,926 paired ball observations, with median cross-camera
disagreement .661m. The relaxed profile maps 58.51% with .760m median disagreement;
it is preserved but not selected because strict passes first. No correspondence
answer key exists, so these are consistency checks, not alignment accuracy.

Same-camera ball/player distances and cross-camera distances provide 68 combined
features, including bounded temporal contact summaries. Their tactical observation
context reaches 175 frames. Existing 250-frame validation exclusions cover this
new context. The original supplied phase metadata and older models retain their
previously documented context. Prefix 18935 vs 19060 and full-match public features
must agree exactly. Camera mapping and feasibility use observations only.

The final method retains all six original models and their probabilities. A new
joint XGBoost head is trained only on eligible fold training labels. It contributes
where an aligned broadcast ball is observed. The canonical learned prediction
supports physical source proposals; it does not replace the reference globally.
Source runs are decoded within contiguous broadcast-observed episodes using the
fixed 11-frame smoothing and penalty3. Proposed runs 8..125 frames need mean
confidence>=.65, at least three exclusive close contacts, >=.5 learned agreement,
and the frozen reference-flank gates. Entirely enclosed opposite reference runs
of 8..250 frames are protected. Gates read the unchanged reference. No reference
D frame changes, no missing-camera interval is filled, and inference resets per
validation block and official half. Frame IDs and ground-truth endpoints are not
algorithm inputs.

## Paired results and costs

Primary values are identical across seeds 42/43/44 within each protocol:

| Protocol | Reference spellF1 | Broadcast closure | Gain | Seed 42 MacroF1 before→after |
|---|---:|---:|---:|---:|
| Buffered minutes |.729412|.741176|+.011765|.867912→.870211|
| Purged spells |.662857|.674286|+.011429|.867623→.870074|
| Buffered100 seconds |.659459|.670270|+.010811|.842747→.844383|

All nine MacroF1 and mean-short-IoU values improve. No primary regression occurs
among 39 clipped-block confirmation comparisons. D predictions, false-D frames
and D recall remain exactly unchanged. At official IoU .5, seed 42 matched/spurious/
missed counts remain 13/4/4, 12/6/5 and 13/7/4 respectively. The primary gain comes
from the 90-frame away possession crossing the scorer's IoU .7 threshold.

In buffered seed 42, that away possession's IoU improves .653226→.788889, while
its frame recall falls 90%→78.89%: the prediction is shorter and better bounded,
but loses ten correct frames. The missed 33-frame home possession reaches 100%
frame recall; its best prediction is still overlong, with IoU .266129. It does
not become an IoU .5 match. Mean short IoU rises .399285→.466250 and mostly
frame-recovered short runs increase 4→5, while short IoU .5 matches remain 3.
The 71-frame away possession remains missed. The 19-frame home possession remains
overlong at IoU .463415. The 81- and 87-frame cases are unchanged.

Broadcast observations are incomplete and can have wrong team assignments. The
71-frame case has no aligned ball coverage; the81-frame case has poor observed
same-camera nearest-team agreement. The 33-frame case has correct same-camera
nearest-team evidence but wrong cross-camera nearest-team evidence, supporting
the use of independent player geometry rather than assuming perfect tracking.

Only 7,500 public labels and 17 true possessions are available. Training sets and
seeds overlap, and the same public labels have repeatedly guided selection.
Bootstrap 90% intervals overlap: seed 42 buffered .576471–.870588 before and
.588235–.882353 after; purged .514286–.800000 and .525714–.811429; longer blocks
.518919–.789189 and .529730–.810811. They do not correct adaptive selection bias.
These results are neither independent generalization nor statistical proof.

## Reproduction and preservation

Use Python 3.12 in the existing `.venv` with the pinned research requirements.
The original selected candidate and OOF prerequisites must first be reproduced
using [STOPPAGE_RECOVERY_REPORT.md](STOPPAGE_RECOVERY_REPORT.md), including the
advanced temporal and anchored prerequisites it names. Raw tracking is resolved
through `games.require`; `MTAH_DATA_DIR` may point to a different local directory.
The ordinary broadcast chain requires no historical source archive:

```sh
.venv/bin/python experiments/prepare_broadcast_baseline.py
.venv/bin/python experiments/prepare_broadcast_alignment.py
.venv/bin/python experiments/research_broadcast_ownership.py
.venv/bin/python experiments/research_broadcast_source_intervals.py
.venv/bin/python experiments/research_broadcast_intersection.py
.venv/bin/python experiments/confirm_broadcast_closure.py
.venv/bin/python experiments/finalize_broadcast_closure.py
.venv/bin/python experiments/prepare_full_broadcast_features.py
.venv/bin/python experiments/build_broadcast_candidate.py --output data/candidate_broadcast_closure_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_broadcast_closure_fresh/candidate.csv
```

Research/output folders are exclusive to preserve results; run missing steps in
a fresh reproduction workspace rather than overwrite completed outputs. The
historical preservation audit is separately invoked with
`finalize_broadcast_closure.py --audit-history` in this original workspace, where
`experiments_before.tar.gz` and `baseline_manifest.json` exist. Fresh confirmation
reports historical audit fields as unperformed, rather than claiming preservation
it did not verify. `prepare_broadcast_baseline.py` copies/verifies the unchanged
stoppage confirmation, without fabricating a historical snapshot.

Independent Standards and Spec review both found machine-specific absolute paths
in the original camera manifest. All original sources, plans, caches, checkpoints,
scores and the first complete CSV were preserved under `provenance_before/`.
The repair uses six stable logical camera identities, resolved locally, and
rejects missing identities or changed bytes. The frozen confirmation plan records
a provenance revision and its original SHA; thresholds, folds, labels, features
and predictions must remain unchanged. Synthetic relocation/corruption tests
pass. All nine prefix-cache and four full-cache files are byte-identical. All
138 checkpoint/probability/score/prediction files compared across the four stages
also match exactly. The frozen confirmation configuration is unchanged; only
provenance metadata is revised. Actual camera input hashes agree under an
alternate configured data directory. A new workspace without either historical
session file successfully prepares its baseline and finalizes all 75 rows; its
historical-audit fields correctly remain unperformed. The original workspace
historical audit separately preserves all 1,081 preceding rows. Both independent full builds pass the official validator and match in every
file, including manifests. Their CSV, seven checkpoints and saved probabilities
exactly match the preserved pre-repair artifact. The six retained model artifacts
and probabilities remain unchanged.

The full CSV has 199,319 rows, 378 changed H/A frames and an identical D mask;
SHA256 `dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce`.
Counts are D 78,008, H 49,499, A 31,350 and X 40,462. Its original pre-repair
provenance is preserved. Current tests: 97 passing. Official scoring agrees for
all 75 evaluations; all 91 development models reload exactly; the original 1,081
ledger rows and frozen unlabelled annotation metadata are unchanged. Evidence
is in `integrity.json`, `provenance_parity.json`, `fresh_reference_audit.json`
and `full_artifact_integrity.json` under the round directory. Independent artifact review is complete;
[SELECTED_MODEL.md](SELECTED_MODEL.md) now identifies this seven-model candidate.
Source review is recorded in [BROADCAST_CODE_REVIEW.md](BROADCAST_CODE_REVIEW.md).
An intentional turn interruption stopped one full-alignment run. Its missing
process handle and process inventory were checked before retrying; the partial
folder and log remain in `interrupted_full_alignment/`. This retry adds no
new evaluation rows.
No commits, uploads, submissions, publication or external contact occurred.
