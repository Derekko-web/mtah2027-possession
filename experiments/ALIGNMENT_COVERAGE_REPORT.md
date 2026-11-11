# Round 23: bounded strict-alignment coverage — rejected opportunity

Completed 2026-10-23. The metadata investigation is complete. It finds a safe but
insufficient coverage gain under the frozen opportunity gate. No images were
decoded, actual action outputs consumed, possession targets read, models fitted,
or possession scores added. The selected eight-model candidate and every ledger
byte remain unchanged. No further round has started.

## Result

| Schedule | Distinct source clips | Observed public frames | Coverage | Net additional frames |
|---|---:|---:|---:|---:|
| Retained Round 22, 50-frame grid | 40 | 2,003 | 26.71% | 0 |
| Bounded inverse support, 50-frame grid | 46 | 2,303 | 30.71% | 300 |
| Bounded inverse support, 25-frame grid | 90 | 2,328 | 31.04% | 325 |

The predeclared primary variant was the 25-frame grid. Its frozen engineering
opportunity gate required at least 3,000 observed public frames, at least 750 net
additional frames, at least 98% retention of original coverage, and all safety /
prefix checks. Retention is 100% and safety/prefix checks pass, but both coverage
requirements fail. No thresholds or fallback selection were changed after counts.
The primary result is **rejected metadata opportunity**, not a possession score.

The exact inverse test recovers 300 frames. Adding 44 more clips on the denser
grid adds only 25 more frames; clip count alone substantially overstates the
coverage benefit. Strict mapping itself exists for only 3,359/7,500 public frames
(44.79%). Of those, 1,031 still lack safe complete-clip query support in the
primary schedule; another 4,141 public frames have no strict mapping. These are
source availability diagnostics, not corrected correspondence or possession truth.

The predetermined 125-frame windows (starts every 25 frames) have full coverage in
65/296 cases for the retained schedule, 77/296 for inverse50 and 78/296 for
inverse25. At least 80% coverage rises 71->83->84 cases. These overlapping fixed
windows are label-free and are not counts of recovered short possessions.
No ground-truth possession boundaries influenced scheduling or support thresholds.

## Frozen protocol and implementation

[ALIGNMENT_COVERAGE_PLAN.json](ALIGNMENT_COVERAGE_PLAN.json) was frozen before any
new coverage processing, SHA256
`1001ff2799a43bc36f4f6611e31c06835f3c7bc3c3640991014f3977359d1a3c`.
The source-based design investigation is
[ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md](ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md).
The pre-round source archive, 1,468-row ledger snapshot and 17 protected artifact
identities are under ignored
`data/control_alignment_coverage_v1/protocol_freeze/`.

The original strict mapping and all formation cost, far-runner-up, uncertainty,
rate, orientation and usable-player guards stay unchanged. No relaxed profile,
new synchronization rule or gap filling was tried. Two fixed, nested grids
(50 and 25 frames, centers 11000..18950) use an inverse search confined to
`center +/-125` mapping rows. The exact 100 stride-two source samples must be
bracketed in the center's local continuous component, each with a mapped witness
within one broadcast frame. No whole-component search is permitted. Every one
of the 199 intervening source video indices must be usable and inside the actual
164,487-frame video; its last valid index is 164486, distinct from the 164,488-row
tracking timeline.

The full candidate decision interval `center +/-175` must lie inside one half
and the source prefix and avoid all 24 sealed windows and 60 held weak images,
each expanded by 250 frames. The source/image span is recorded separately.
Queries select the nearest eligible same-component center within 25 (earlier
ties); an invalid closer candidate cannot hide another valid candidate.
Broadcast source centers are deduplicated while preserving tactical associations.
The primary schedule has 49 odd and 41 even distinct source centers; any later
producer must preserve per-clip source parity.

The implementations are [alignment_coverage.py](alignment_coverage.py),
[prepare_alignment_coverage.py](prepare_alignment_coverage.py),
[audit_alignment_coverage.py](audit_alignment_coverage.py) and
[test_alignment_coverage.py](test_alignment_coverage.py).
All schedules, per-frame query plans and synthetic feature extracts remain under
ignored data; none are published or included in hosted context.

## Full dependency bound and Round 22 qualification

The independent source review found a distinction missing from the prior audit.
A strict mapped row, including its absence, can depend on tactical observations
within 50 frames through neighboring-anchor eligibility and fine matching. The
old scheduler inspects `center +/-150`; adding this alignment context, 25-frame
query reach and 50-frame summaries gives a generic 275-frame tactical-observation
dependency bound. The old 225-frame figure bounds source image content only.
See the code-derived analysis and citations in the
[design research](ALIGNMENT_COVERAGE_DESIGN_RESEARCH.md).

Round 22's successful image/source checks, exact prefix replay, official scoring
and model reloads remain recorded. They do not establish that its full
alignment-dependent missingness satisfied a general 250-frame tactical-input
bound. Keep those models and rows as rejected development evidence; do not reuse
them as a certified promotion. This qualification does not assert held-out label
use: eligibility never consumed possession targets. Round 22 was rejected and did
not alter the selected model.

For both new schedules, every map inspection, including failed searches and
missingness decisions, is restricted to 125 frames. With 50-frame tactical alignment
context, 25-frame nearest-center reach and 50-frame summaries, the complete
conservative tactical bound is **125+50+25+50=250**. Image content has the separate
bound **125+25+50=200**. The largest actual clip inverse radius in this assessment
is 106; the declared conservative bounds were not reduced after seeing that value.
The underlying formation matcher still searches the global broadcast tracking
timeline. The 250 bound is on the tactical observation axis and is not a claim
that all broadcast source dependencies are local.

## Verification

The new source/query plans agree exactly between the preserved mappings rebuilt
independently from 18,935 and 19,060 tactical source prefixes. The inherited mapping
reconstruction source and mapping hashes were verified. Query parity includes
50 frames on either side of the 7,500 public rows. A complete 121-column event
feature construction using fixed synthetic heads also agrees exactly on all 7,500
public rows, including future summaries, cuts and missingness.

This is **metadata and synthetic feature parity**. No new actual broadcast action
features were observed; actual-head full-feature prefix parity would still be
required in any later separately frozen observation/fitting round.

A fresh assessment in a separate ignored output directory reproduces all 22 files,
including manifests, byte-for-byte. The audit validates 272 inverse associations
across both grids and both prefixes, exact source witnesses, all intervening source
frames, query components, half/prefix/exclusion guards and the complete declared
tactical bounds. All 17 protected artifact hashes remain exact. All 192 production
unittests pass, including 10 new synthetic contracts. The entire 1,468-row ledger
is byte-identical to its pre-round snapshot. There are 0 new neural calls, fits,
score rows, confirmation runs or full-match inferences.

The initial tool output did not retain the first assessment's process-session
handle. A subsequent exact-command process check found no matching owned process,
and the complete stage output was present. The isolated replay's explicit handle
returned exit 0, as did the final audit and production tests. No unrelated process
was stopped or changed. Saved stage files alone were not treated as liveness
proof. Both complete generations and their logs remain preserved.

Evidence: `assessment/manifest.json`, `assessment_replay/manifest.json`,
`integrity.json`, `audit.log`, `production_tests.log`, `round_stage.json` and
`round_closeout.json` under `data/control_alignment_coverage_v1/`.

The selected CSV remains
`data/candidate_motion_dead_trim/candidate.csv`, SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
All original substantial-gain, short-match, D/false-D, macro, official scoring,
paired-seed, reproduction and independent review gates remain unchanged. No
candidate is promoted by a metadata feasibility result.

## Reproduction and next question

Repeat the ordinary audit without overwriting any stage:

```sh
.venv/bin/python experiments/audit_alignment_coverage.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -q
```

The original generation commands intentionally require fresh output directories:

```sh
.venv/bin/python experiments/prepare_alignment_coverage.py
.venv/bin/python experiments/prepare_alignment_coverage.py --output data/control_alignment_coverage_v1/assessment_replay
.venv/bin/python experiments/audit_alignment_coverage.py
```

The frozen plan binds every input hash, source prefix, source implementation and
ledger identity. The ordinary audit compares both complete generations and all
saved source/input/cache hashes. No historical fitting or inference is rerun.

The next useful investigation is why strict formation anchors are absent or
rejected, with independent observation consistency checks. This round suggests
that denser action sampling alone adds little coverage. That question needs its
own frozen protocol and has not been started. The supplied alignment remains
fallible and public possession checks remain extensively adaptive, correlated
single-match development on 17 labelled possessions. Private/full-match accuracy
and maximum score are unverified; the broader optimization objective is unfinished.
No images, tracking extracts or raw action arrays entered hosted context; no
manual labels, commits, uploads, submission, publication or external contact occurred.
