# Selected model: conservative native-motion dead-ball trimming

The current default retains the seven-model broadcast closure and adds one
native broadcast-motion head. It accepts H/A from the alternative only where
the reference predicts D; every existing H/A/X prediction stays exact.
All nine paired incremental checks pass. Buffered spellF1 stays .741176,
purged spellF1 improves .674286→.685714, and longer-block predictions remain
identical at spellF1 .670270. FalseD falls by 4–5 buffered and 10–17 purged frames
across seeds 42/43/44, with exact D recall and no lost previously true-D predictions.
There are 0 primary regressions among 39 clipped-block comparisons.

This is a modest incremental improvement. The original +.04 buffered improvement
and extra buffered short IoU .5-match criteria remain unmet. The 71-frame away miss
and 19-/33-frame home overextensions remain. Only 7,500 public labels exist; these
are adaptive, correlated within-match development checks. Private full-match
accuracy and maximum score remain unverified.

The selected full-match file is `data/candidate_motion_dead_trim/candidate.csv`,
with 199,319 rows and SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
The independent rebuild matches every file, including manifests. Eight model
checkpoints, six probability arrays and source/input/cache hashes verify. All
retained seven-model artifacts remain byte-identical. Only 811 reference D frames
change to H/A; these are not 811 verified full-match corrections. Both CSV builds
pass the official format validator. All 112 tests pass.

The complete method, 72 added evaluation rows, failed parent screens, synthetic
repair history, frozen objective changes, paired metrics and reproduction are in
[PAIR_MOTION_REPORT.md](PAIR_MOTION_REPORT.md). Independent Standards and Spec
reviews have 0 unresolved findings:
[MOTION_CODE_REVIEW.md](MOTION_CODE_REVIEW.md).

With verified full motion/broadcast caches, confirmation and the preserved
seven-model reference present, use the pinned Python 3.12 research environment:

```sh
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_motion_dead_trim_candidate.py --output data/candidate_motion_dead_trim_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_motion_dead_trim_fresh/candidate.csv
```

The report provides the full upstream/fresh-workspace chain. Deterministic
baseline preparation and both ordinary finalizers were checked without historical
session files. Optional preservation auditing remains separate. Evidence is under
`data/control_pair_motion_v1/`, including `integrity.json`,
`dead_trim_integrity.json`, `fresh_reference_audit.json`,
`full_artifact_integrity.json` and saved audit scripts/logs.

Every previous candidate remains. The exact preceding selected entry is preserved
in [BROADCAST_MODEL.md](BROADCAST_MODEL.md); its CSV remains
`data/candidate_broadcast_closure/candidate.csv`, SHA256
`dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce`.
The six-model entry remains in [STOPPAGE_MODEL.md](STOPPAGE_MODEL.md).
[WORK_HISTORY.md](WORK_HISTORY.md) distinguishes 778 inherited score rows from 552
added in this chat, for 1,330 dependent evaluations. The rejected directional
ownership follow-up is documented in
[OWNERSHIP_GRAPH_REPORT.md](OWNERSHIP_GRAPH_REPORT.md). Twelve additional weak
local ball-appearance trials also failed feasibility, with no public possession
fitting or ledger additions; see [BALL_APPEARANCE_REPORT.md](BALL_APPEARANCE_REPORT.md).
Selected artifacts remain unchanged. No commits, uploads,
submissions, publication or external contact occurred.

The subsequent [native pretrained detector follow-up](NATIVE_BALL_REPORT.md)
establishes local CoreML inference but rejects all six weak detector/crop
combinations. It adds no possession-score rows or public models. Selected8
remains unchanged; the latest full test run passes135 tests.

The subsequent [modern detector follow-up](MODERN_BALL_REPORT.md) rejects six
YOLO11/context variants on frozen weak location gates. It retains aggregate
location diagnostics, independent-compilation sample parity and a fresh ordinary
audit, with no public possession fit or confirmation images. Selected8 and all
1,291 possession-score rows remain exact. The latest full suite passes137 tests.

The [ball-location diagnosis](BALL_LOCATION_REPORT.md) finds no short temporal
offset or frame-extraction error and leaves predictions unchanged. It verifies a
previously discarded supplied ball-speed scalar as a new feature lead, without
claiming it is corrected truth. All140 tests pass; no possession rows are added.

The [supplied-speed preflight](PROVIDED_SPEED_REPORT.md) finds no scalar or scalar
context in the public supervised window for either source. It stops before any
fitting; no39-row screen was run. All prepared files rebuild exactly, and144 tests
pass. Selected8 and the1,291-row possession ledger remain unchanged.

The [detector-feature screen](DETECTOR_FEATURES_REPORT.md) adds30evaluations and
26new research checkpoints. Both new heads and all3decoders fail the frozen
promotion gates; all30official scores and39model reloads verify. Every feature
cache reproduces exactly in a separate workspace.150tests pass. Selected8 stays
unchanged; the ledger now has1,321rows, including all preceding bytes intact.

The separately frozen [matched-slot follow-up](DETECTOR_SLOT_REPORT.md) adds9
no-refit checks. Both detector substitutions leave every OOF label unchanged;
all9official metrics and isolated replay agree. No finalist advances. The ledger
now has1,330dependent rows:778inherited+552added, with all prior bytes preserved.

The [optical ball/color contact screen](COLOR_CONTACT_REPORT.md) adds12no-fit
checks. Both color variants keep all labels exact; the metadata control regresses.
No finalist advances. Source and scoring replay are exact,154tests pass, and all
selected artifact hashes are unchanged. The ledger has1,342dependent rows:
778inherited+564added, preserving every preceding byte.

The [trained contact-history screen](CONTACT_HEADS_REPORT.md) adds30evaluations
and26research checkpoints. Both new heads fail; conservative outputs remain
exact and direct live ownership regresses. All30official scores,26new/13control
reloads and isolated replay verify;157tests pass. The ledger now has1,372rows:
778inherited+594added. Selected8 remains unchanged.

The [full-team spatial experiment](SPATIAL_SHAPE_REPORT.md) adds30evaluations
and26research checkpoints. Both joint and independent spatial heads fail;
falseD removal sacrifices trueD recall and no extra short match is recovered.
All30official scores,26new/13control reloads and isolated replay verify;161tests
pass. Selected8 stays exact. The ledger has1,402dependent rows:778inherited+
624added, with preceding bytes preserved.

The [local temporal ownership probe](LOCAL_OWNERSHIP_REPORT.md) rejects3dependent
protocols(81competition calls). Truncation and input-layout issues are repaired,
but both valid-output runs classify all27samplesHOME,balancedrecall.333333.
Source/montage rebuilds and all3score replays are exact;170tests pass. A manifest
footer repair preserves all29cachedoutputs and uses0neural calls. No possession
fit,row or candidate is added. Selected8 and the1,402-row ledger stay exact.

The [pretrained football-action runtime](SPORTS_ACTION_RUNTIME_REPORT.md) passes
synthetic technical checks:600checkpoint entries load strictly, CPU outputs replay
byte-exact from an offline wheel-built runtime, and full-resolution MPS forwards
repeat exactly at4.27/1.53seconds.170production tests pass. No competition images,
new fits, possession-score rows or candidate changes occur. Actual possession
value is untested; the1,402-row ledger remains byte-identical.

The [football-action observation follow-up](SPORTS_ACTION_OBSERVATION_REPORT.md) completes 147 local clips and rejects both feature heads after 26 fits and 39 official/replayed score rows. Joint-slot predictions stay exact; other decoders fail the frozen gates. Ledger: 1,441 dependent rows (778 inherited, 663 added). Selected8 unchanged; 175 tests pass. Memory and decoder execution failures are preserved and documented.


The [Round22 broadcast-action screen](BROADCAST_ACTION_REPORT.md) is completed
and rejected.40safe broadcast clips pass the separately frozen sparse support
rule (2,003public frames,26peaks);13new fits and27official/replayed comparisons
produce no eligible finalist. Joint-slot labels remain exact; dead trimming loses
buffered D recall and adds spurious possessions. No extra short match,43/44
confirmation, full-match inference or promotion. All17protected artifact hashes
and the selected CSV remain unchanged;182tests pass. Ledger:1,468dependent rows
(778inherited+690added), preserving every preceding byte. No next round started;
private accuracy and maximum score remain unverified.


The [Round 23 strict-alignment coverage investigation](ALIGNMENT_COVERAGE_REPORT.md)
ends at metadata feasibility. Bounded inverse scheduling increases public source
coverage from 2,003 to 2,328 frames but fails its frozen 3,000 total / 750 net-gain
gate. No images, actual action outputs, possession targets, fits or score rows
are used. The complete new tactical dependency bound of 250 includes mapping
eligibility and missingness. Round 22's 225 was an image-content bound and did
not certify that full dependency; the prior round remains rejected. All 17
protected hashes, this selected CSV and all 1,468 ledger rows stay exact.
All 192 tests pass. No downstream action round started.


The [Round 24 anchor diagnosis](ANCHOR_FAILURE_REPORT.md) preserves every
production anchor/mapping array and the selected model. All 300 public tactical
anchor observations are usable; geometry and neighborhood gating dominate the
138 rejections. Missing mappings split into 3,791 endpoint-pair failures and 350
fine-time-rate failures. No image, action-head, label, fitting or scoring work;
no threshold changes or new alignment. Both actual prefixes and all 17 diagnostic
artifacts replay exactly; 202 tests pass. All 17 protected artifact hashes and
all 1,468 ledger rows remain unchanged. A sub-grid/neighbor hypothesis is recorded
for a future frozen protocol and has not been started.


The [Round 25 sub-grid refinement](SUBGRID_ALIGNMENT_REPORT.md) is completed and
rejected as a metadata opportunity. Preserving all old mapping fields adds454
mapped rows but only50 safe query frames (2,328->2,378), below the frozen750gain;
36new usable checkpoints also fall short of50. No images, actual action outputs,
labels, fits or scores are used. Both actual prefixes and37artifacts replay
exactly;219tests and17protected hashes pass. A preserved aggregate-prefix
assertion failure is repaired without changing the plan or alignment algorithm.
The selected CSV and all1,468ledger rows remain exact. No alignment or model is
promoted, and no next round has started. Accuracy and maximum-score limitations
and all promotion gates remain unchanged.


### Round 26 temporal global matching — completed, opportunity rejected

One frozen three-frame global formation search changes the winning moment before
neighbor gates, retaining individual geometry and temporal ambiguity checks.
Inverse100 pays for mapping context75: complete tactical bound 100+75+25+50=250.
Historical strict125 reproduces 2,328 safe queries; paired strict100 has 1,785.
The primary row-preserving extension reaches 1,785 (-543
vs historical, 76.68% retention), with 0 usable new
checkpoints. The frozen conjunction is rejected_metadata_opportunity; no fallback or further round.
Both actual prefixes and all 121synthetic columns agree; 45 fresh-process artifacts
replay byte-exact, 242 tests pass, 17 protected hashes and every 1,468 ledger row
stay exact. Source/input/cache hashes and both symlinks verify. No images,
action outputs, targets, fits, scores or selected changes; no alignment/model
promotion. See [TEMPORAL_GLOBAL_REPORT.md](TEMPORAL_GLOBAL_REPORT.md). This metadata
result establishes no possession/private accuracy or maximum score; broader
optimization remains unfinished. All existing promotion gates remain unchanged.


### Round 27 DFL ownership transfer — completed, promotion rejected

Seven external matches pass frozen direct-label/data feasibility (1,002,644
frames). One ownership teacher passes all three fixed external validation gates,
but its paired public integration fails all three promotion conjunctions:
buffered/purged/longer spellF1 .571429/.512821/.536842 versus selected8
.741176/.685714/.670270. Buffered short IoU.5 matches fall3->2. D stays exact.
No43/44 confirmation, full-match candidate or promotion follows.

All261 tests pass;17 XML,70 external and51 public artifacts replay byte-exact;
26 public checkpoints and the external teacher reload.17 protected hashes and
both symlinks verify. Exactly9 ledger rows append to1,477, preserving every
preceding byte. Selected CSV remains SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
See [DFL_TRANSFER_REPORT.md](DFL_TRANSFER_REPORT.md). This one XY-only ownership
transfer fails; data feasibility/external gains establish no competition/private
improvement or maximum score. Prior failures and all promotion gates remain;
no further round has started.


### Round 28 teacher signal audit — completed, opportunity rejected

The frozen audit covers all 17 public possessions, six short cases, matched
controls and every missed/spurious case in all three protocols. Raw teacher
balanced accuracy is .854971, but confident correction precision is only
.0875/.141176/.20: 14/24/24 potential corrections versus 146/146/96 errors on
previously correct frames. It adds no missing possession match and has zero
correct raw decisions on the 71-, 19- and 33-frame unmatched short cases.
The continuation gate fails; no integration fit, score row or candidate follows.

All 12 identity controls, 56 target-aware probes and both closure branches verify;
13 fresh-process artifacts match byte-for-byte. All 275 tests, 17 protected
hashes and source symlinks pass. Every byte of the 1,477-row ledger stays exact.
Selected8 remains unchanged at public spell F1 .741176/.685714/.670270.

Prefix evidence correction: Round27 compared 18,935 native rows with the same
18,935 rows, so its claimed 19,060-versus-18,935 check was unverified. Round28
actually reconstructs both prefixes and verifies exact public features, teacher
probabilities and frozen Round27 cache identity. No old score, model or candidate
changes. The initial guard failure and regression evidence remain preserved.
See [DFL_SIGNAL_REPORT.md](DFL_SIGNAL_REPORT.md). This sufficient test rejects one
teacher avenue; it establishes no private accuracy or maximum score.
No further round has started.


The [reviewed human annotation pilot](HUMAN_ANNOTATION_REPORT.md) uses a frozen
subset of reviewed clips1,3,4 (90seconds /2250frames), excluding the incomplete
unreviewed clip2 snapshot. One ordered102 head receives source-balanced added
supervision; thirteen fits fail all-protocol continuation. Public spellF1 becomes
.720000/.666667/.604878 versus selected8 .741176/.685714/.670270, with no new
possession match and more spurious predictions. Selected8 and all17 protected
artifacts stay unchanged. All296 tests pass; models, features, official scores and
fresh replay verify. Six dependent rows append:1483 total (778 inherited +705
continuation). Human snapshots are retained; the audit stays sealed. No candidate,
confirmation, promotion or further fit follows.


### Rounds 30–32 automatic-only score work — no selected change

Public-only temporal D/live heads, a fixed physical short-D bridge and a dependent
head-agreement rule fail their prospective gates. No manual annotations enter
this work. Selected8 remains unchanged at public spellF1
.741176/.685714/.670270 and CSV SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
All307 tests, saved-model/official/shared replays,17 protected hashes, source
links and reserved metadata pass. Exactly27 dependent rows append to1,510,
preserving all1,483 original rows/bytes. No full candidate, promotion or
submission follows. See [TEMPORAL_DEAD_REPORT.md](TEMPORAL_DEAD_REPORT.md) and
[DEAD_BRIDGE_REPORT.md](DEAD_BRIDGE_REPORT.md). Private accuracy and optimality
remain unestablished; further manual annotation is not required by the pipeline.


### Rounds33–34 direct DFL live/dead — no selected change

Direct public DFLstatus supervision passes external validation but yields no
native possession-score gain. A frozen rawsignal audit rejects safe transfer.
All315tests,60external/71public/4native data replays,checkpoint/shared/official
checks and17protected artifacts verify. Nine dependent rows append to1519,
preserving all1510oldrows/bytes. Selected8 remains unchanged. See
[DFL_DEAD_REPORT.md](DFL_DEAD_REPORT.md). The user's substantial-improvement goal
continues; no further manual annotation is required or used.


### Round35 bounded ball-free PathCRF — no selected change

Highaccuracyonpartialcoverage(.991balanced) adds no missing short possession:
all6short/all4unmatchedcases have0supportedcenters. Frozen continuation fails;
no newfit/possession-score/candidate. All320tests andfreshmodel/input/metric
replays verify, with every1519ledgerbyte and17protectedartifacts unchanged.
See[PATHCRF_REPORT.md](PATHCRF_REPORT.md). The user's substantial-gain goal remains
active; checkpoint/window/roster tuning is not continued within this avenue.


### Round36 masked player-set ownership — no selected change

Temporal external gates pass and native coverage reaches100%, but direct
corrections fail the safe-incremental-signal gate. No nativefit, newpossession
score, candidate or promotion. All325tests/replays and17protectedartifacts pass;
every1519ledgerrow/byte stays unchanged. See
[MASKED_PLAYER_REPORT.md](MASKED_PLAYER_REPORT.md). The broader user goal remains
active; scalar confidence thresholds are not relaxed.

### Round37 player-set representation — no selected change

Aggregate spellF1 improves across three protocols, but no additional short
possession matches and a buffered spurious prediction is added. The frozen
substantial/secondary gates fail. All327 tests and deterministic replays pass;
1531 ledger rows preserve their1519-row prefix. See
[PLAYER_REPRESENTATION_REPORT.md](PLAYER_REPRESENTATION_REPORT.md).

### Round38 soccer detector — no selected change

Public test-camera numerical localization gates pass, but visible-ball support
27/23 fails the frozen>=32 minimum. No competition processing or possession
candidate follows from that pilot. See[WASB_REPORT.md](WASB_REPORT.md). A separate
prospective disjoint confirmation is pending; R38 remains failed.

### Round39 relative-ball temporal encoder — no selected change

The extra ball geometry fails its external incremental-information gate before
native inference. No possession fits/scores or candidate. See
[BALL_CONTEXT_REPORT.md](BALL_CONTEXT_REPORT.md). Selected17 artifacts and1531
ledger rows remain exact.
