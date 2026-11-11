# Round 22: broadcast football-action features — rejected

Completed 2026-10-23. The selected eight-model candidate remains unchanged:
`data/candidate_motion_dead_trim/candidate.csv`, SHA256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
No eligible finalist advances to seeds 43/44 or full-match inference. This negative
result completes Round 22; no next experimental round has started.

## Frozen experiment

[BROADCAST_ACTION_PLAN.json](BROADCAST_ACTION_PLAN.json) was frozen before any new
broadcast decoding or model output. Its SHA256 is
`6543e35f4f131ce6b3743e2338d55cce6c0eef34b88eaec8419cbcf36cee90e1`.
The pre-round source archive, ledger snapshot and 17 protected artifact hashes are
under ignored `data/control_broadcast_action_v1/protocol_freeze/`.
The preceding [metadata feasibility](BROADCAST_ACTION_FEASIBILITY_REPORT.md)
supplies exactly 40 safe tactical-to-broadcast associations, with 40 distinct
source centers: 19 odd and 21 even. No centers were replaced after inspecting
outputs. The source alignment is fallible observation-derived evidence.

This round predeclared one event-only head. It appends 121 fixed features to the
170-column matched broadcast joint view. Actor/team geometry is omitted because
broadcast side logits require a separately validated camera-specific team mapping.
The support rule was frozen at >=1,500/7,500 observed public frames, >=8 local
nonbackground peaks >=.3 separated by >=25 tactical frames, peaks from >=3 distinct
source clips, and observed event-max standard deviation >=.01. These are sparse
source-support checks, not action recognition accuracy or promotion gates.

Extraction selects the exact union of each clip's `b-100+2*arange(100)` frames with
per-range parity, sorted/deduplicated source centers and preserved tactical
associations. FFmpeg seek and duration are input-side. The original offline
600-entry public checkpoint, MPS memory fraction .6, deterministic float32 model,
persistent uint8 input buffers, and inactive-cache release are retained. Images
are neither exported nor displayed. Three streamed input frames match independent
accurate-seek pixel bytes exactly. All raw heads remain under ignored data.

Dense outputs interpolate at each query's mapped **broadcast** frame from its
nearest eligible **tactical** center within 25 (earlier on a tie). Missing mapping,
invalid source queries and cuts are never filled. The 121 features are 13 class
probabilities, event max, entropy, displacement and seen, plus fixed past/future
25/50-frame means/maxima for the 12 actions and event max. Summary runs contain observed
queries in one alignment component/half only; missing rows retain NaN features
and seen=0. Source image reach is bounded by 150 + 25 + 50 = 225 tactical frames within
the existing 250-frame label buffer. All 24 reserved windows and 60 held weak images
retain their 250-frame image exclusions. Duplicate source centers cannot inflate
support counts; the implementation also tests that case synthetically.

An execution-only metadata check initially stopped **before image decoding**:
the actual video container reports 164,487 frames while broadcast tracking has
164,488 records. The adapter had assumed those counts equal. The failed source,
log and red/green regression evidence are preserved. Before the first neural
call, the metadata repair pinned 164,487 and bounded the same schedule to the
actual video count. No schedule, feature definition, support threshold, model,
fit setting or promotion gate changed. The original frozen plan remains exact.
This discrepancy is not a corrected synchronization or timing claim.

## Support and prefix evidence

The frozen support gate passes:

| Diagnostic | Result |
|---|---:|
| Observed public frames | 2,003 / 7,500 (26.71%) |
| Confident separated peaks | 26 |
| Distinct source clips providing peaks | 23 |
| Observed event-max standard deviation | .170970 |
| Mean observed entropy | .834731 |

Complete feature construction from 18,935 versus 19,060 source prefixes is exact
on all 7,500 public rows and all 121 columns, including future summaries and NaNs.
A second audit independently recomputes formation anchors and strict mappings
from each actual truncated tactical observation source, then reconstructs all
features. Both prefix features agree exactly and the rebuilt full mapping agrees
with every cached mapping array. No possession targets enter either support or
prefix reconstruction. Evidence: `inputs/`, `prefix_replay/summary.json`,
`source_integrity.json` and `pre_fit_checks.json` under the round data directory.

## Screen result

Exactly 13 new fold models were fitted under the existing three split protocols,
seed 42, unchanged depth 3 / 160-tree frame objective, team symmetry and 250-frame
training exclusions. Training-only targets/weights/indices are saved and replayed.
No calibration, feature selection or decoder tuning occurs. The 27 dependent
comparisons comprise 3 selected references, 12 reused matched controls and 12 new
head/decoder rows. These are not 27 independent experiments.

| Method | Buffered spell F1 | Purged spell F1 | 100-second spell F1 |
|---|---:|---:|---:|
| Selected reference | .741176 | .685714 | .670270 |
| Event head, replace current | .590000 | .590000 | .543590 |
| Event head, joint slot | .741176 | .685714 | .670270 |
| Event head, live replace | .588235 | .548571 | .482353 |
| Event head, dead trim | .731429 | .688889 | .652632 |

Joint-slot predictions remain exactly equal to selected predictions in all three
protocols. The required +.04 buffered gain and an additional buffered short
IoU 0.5 match are absent. Dead trimming reduces false-D counts 88->70, 142->121 and
16->0, but buffered D recall falls .748128->.706603 and macro F1 falls
.870833->.860036. It also adds official spurious possessions in every protocol.
Purged dead trim's small spell gain does not compensate for its macro/spurious
failures or the other protocol failures. All joint-slot/dead-trim short IoU 0.5
matches remain 3; the missed short possession is not recovered. Other replacements
regress, and live replacement loses previously recovered short possessions.
There are 20 primary regressions among 52 new-head clipped-block comparisons.

Every applicable seed 42 gate was evaluated against the selected reference and
matched same-decoder control. No model/decoder passes all protocols. The original
substantial-gain, short-match, macro, false-D, D-recall, stoppage, official
spurious/missed, confirmation and reproduction/review requirements were retained.
No seed 43/44 fits, confirmation scoring, full-match inference, new candidate CSV,
or promotion were run for these rejected heads. Independent Standards/Spec review
is a pre-promotion requirement; no promotion review is claimed for this rejection.

## Verification and reproduction

All 27 scores agree with the official scorer. All 13 new checkpoints and 13 reused
control checkpoints reproduce saved probabilities exactly. All four shared
decoders replay exactly. Training-only target/weight reconstruction, split
indices, buffers, feature schemas, source/input/cache hashes and selected
reference predictions pass. All 17 protected candidate/reference artifact hashes
remain exact. The latest full production suite passes 182 tests, including seven
new synthetic broadcast contract tests. The selected full CSV passes the official
199,319-row format validator and its SHA256 remains exact.

The ledger appends 27 rows once through `detector_ledger.py`, preserving all
preceding 1,441 rows byte-for-byte. An explicit repeat append adds 0 duplicates.
The ledger now contains 1,468 dependent rows: 778 inherited + 690 continuation rows.
`finalize.log`, `integrity.json`, `screen_checks.json` and
`ledger_idempotence.json` retain the outcome. The initial metadata failure and
all sources are retained rather than overwritten.

Run ordinary audits from the repository root with the main pinned Python 3.12
research environment:

```sh
.venv/bin/python experiments/audit_broadcast_action_sources.py
.venv/bin/python experiments/finalize_broadcast_action.py --audit-history --record-ledger
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -q
.venv/bin/python evaluation/validate_submission.py data/candidate_motion_dead_trim/candidate.csv
```

The original execution sequence below intentionally refuses to overwrite existing
stage directories. Reproduce in a fresh corresponding workspace with all pinned
upstream inputs, runtime and feasibility artifacts retained. Use the preserved
pre-round ledger in that fresh workspace because the metadata feasibility input
manifest predates the score append. Do not replace the current ledger. The ordinary
source audit verifies that historical identity against the immutable snapshot
and rebuilds complete features/support without changing any stage outputs:

```sh
data/control_sports_action_v1/runtime/bin/python experiments/prepare_broadcast_action_observations.py
.venv/bin/python experiments/prepare_broadcast_action_features.py
.venv/bin/python experiments/check_broadcast_action_prefix.py
.venv/bin/python experiments/research_broadcast_action.py
.venv/bin/python experiments/finalize_broadcast_action.py --audit-history --record-ledger
```

The source archives, dispatch hashes, video identity, input pixel hashes, head
cache hashes, feature caches, saved fitting inputs, model hashes and official
reports bind those commands to this run. No historical fitting/inference was
rerun merely to resume. All owned execution handles are terminal at closeout.

## Limits and next investigation

This rejects the frozen sparse event-feature pipeline; it does not show that all
broadcast action models are ineffective. Only 17 public possessions are labelled,
and these are extensively adaptive, correlated same-match development checks.
Maximum score and private/full-match accuracy remain unverified. The existing
71-frame away miss and 19-/33-frame home overextensions remain unresolved.

The next investigation should first measure whether target-blind alignment can
safely extend source coverage around short control changes, with its own frozen
protocol and unchanged promotion gates. It has not been started. All competition
media, tracking and raw outputs stayed local; no manual labels, commits, uploads,
submissions, publication or external contact occurred.
