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
[WORK_HISTORY.md](WORK_HISTORY.md) distinguishes 778 inherited score rows from 513
added in this chat, for 1,291 dependent evaluations. The rejected directional
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
