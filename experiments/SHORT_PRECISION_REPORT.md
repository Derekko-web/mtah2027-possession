# Ownership precision research — 21 October 2026

The selected six-model stoppage consensus remains unchanged. Neither finer
ownership endpoints nor additional weak ownership supervision improved the
primary score enough to pass the frozen screen. Both complete screens, including
failures, are retained. The ledger grows from 967 to 1,027 dependent evaluation
rows; all previous 967 rows are preserved by identity.

The current buffered result remains spellF1 **0.729412**, MacroF1 **0.867912**,
D recall **74.81%**, with **92** false-D frames. Purged and longer-block spellF1
remain **0.662857** and **0.659459**. These are adaptive public development
results, not private/full-match accuracy or proof of a maximum score.

## Protected starting point and diagnostic

Before this round, `data/control_precision_v1/` received a source archive,
the complete selected candidate, confirmation outputs and baseline file hashes.
The selected CSV remains SHA256
`dac48e6be66f4395089cae4eee3cfb5daeadcd7963480f399fe079d871be534c`.
The preceding plans, reports, probabilities and candidates were not overwritten.

The initial deterministic feedback loop read the selected buffered seed42 score
and asserted at least four short live possessions with best same-label IoU>=.5.
It failed: **3 of 6**. The 19-frame H interval is recalled completely but the
predicted interval is 41 frames long, giving IoU .463415. The 71-frame A and
33-frame H possessions remain unrecognized. Public label intervals are used for
diagnosis and scoring only, never as inference inputs or final frame patches.

All new methods hold the selected D mask exactly fixed. This isolates ownership
precision and prevents a misleading gain from sacrificing the previous stoppage
recovery. It also means these screens cannot reduce false D.

## Fine endpoints: 27 evaluations

[SHORT_PRECISION_PLAN.json](SHORT_PRECISION_PLAN.json) froze eight variants plus
the selected reference, seed42, across all three existing protocols. Only the
boundaries of model-derived 8..125-frame ownership islands bracketed by opposite
live control may move. A joint dynamic program preserves the original run-label
sequence, all D frames, positive run lengths and at least eight frames for each
eligible island. Movement is limited to 13 or 25 frames. Evidence comes from the
older and/or fast head, with raw or 11-frame probability/log-probability pooling.

| Evidence / radius | Buffered spellF1 | Purged spellF1 | 100-second spellF1 |
|---|---:|---:|---:|
| Selected reference | .729412 | .662857 | .659459 |
| Previous raw /13 | .705882 | .651429 | .616216 |
| Previous mean11 /13 | .705882 | .640000 | .616216 |
| Fast raw /13 | .729412 | .662857 | .637838 |
| Fast mean11 /13 | .729412 | .651429 | .637838 |
| Joint mean11 /13 | .717647 | .651429 | .637838 |
| Joint log11 /13 | .717647 | .651429 | .637838 |
| Joint mean11 /25 | .705882 | .640000 | .627027 |
| Joint log11 /25 | .705882 | .640000 | .627027 |

None adds a short IoU>=.5 match. Fast raw evidence improves the 81-frame
possession's buffered IoU .648→.711, but worsens the 19-frame interval
.463→.345. The finer evidence extends its span rather than locating the true
ending. Joint log pooling offers no primary benefit over probability pooling.
There are 24 primary regressions among 104 method/block comparisons; the complete
`block_deltas.csv` records improvements and regressions rather than only averages.

The new aggregate head diagnostic helps separate two issues. In buffered
validation, raw fast predictions choose the correct team on all 19 true frames,
yet do not locate its ending. For the missed 71-frame possession, correct raw
argmax fractions are only 26.8% in the previous head and 23.9% in the fast head;
for the 33-frame possession they are 0% and 3.0%. This supports investigating
additional ownership evidence, while not establishing that the raw data lack
recoverable information. Topology-preserving endpoint changes cannot create an
entirely missing possession.

## Weak ownership supervision: 33 evaluations

[WEAK_OWNERSHIP_PLAN.json](WEAK_OWNERSHIP_PLAN.json) froze five binary H/A heads
and two ownership-only decoders, with the selected reference in each protocol.
Features are compact relative observations, existing contact memory and ordered
taps; semantic views add automatic attacking/defending/restart phase features.
This differs from prior phase-feature screens by adding outside-public weak
training targets and matching public-only controls.

Weak samples, every tenth qualifying frame inside official half bounds, exclude
the entire public window plus250 frames and every frozen annotation window
plus250 frames. No annotations or reserved media were read. The regimes are:

- Sustained exclusive moving-contact evidence: **755 H /381 A** samples.
- Stable exclusive phase support, observed moving ball, no restart or opposing
  sustained contact: **1,670 H /841 A** samples.
- Agreement of contact and phase evidence: **297 H /155 A** samples.

These are noisy automatic proxies, not additional corrected targets. Weak
classes are separately balanced with total mass .5 of eligible public live
training weight. Public targets and weights use eligible training indices only.
Each depth3/200-tree head uses team-swap augmentation and symmetrized inference.
No held-out preprocessing, calibration or early stopping is used. Bounded
features fit within125 frames; supplied phase membership can span a tagged phase.
Full/padded public feature parity passes exactly.

One decoder replaces ownership within each contiguous reference live segment,
resetting at D and block boundaries. The other accepts only brief model-derived
proposals with confidence>=.65, physical or phase support, opposite reference
flanks and no D overlap. Every proposal gate reads the unchanged reference.

| Head / decoder | Buffered spellF1 | Purged spellF1 | 100-second spellF1 |
|---|---:|---:|---:|
| Geometry public / replace | .533333 | .497297 | .492308 |
| Geometry public / proposals | .729412 | .662857 | .625641 |
| Semantic public / replace | .526316 | .494737 | .517073 |
| Semantic public / proposals | .652632 | .594872 | .575610 |
| Semantic contact / replace | .473684 | .441026 | .480000 |
| Semantic contact / proposals | .652632 | .594872 | .575610 |
| Semantic phase / replace | .533333 | .491429 | .541935 |
| Semantic phase / proposals | .729412 | .662857 | .659459 |
| Semantic agreement / replace | .511111 | .475676 | .510000 |
| Semantic agreement / proposals | .729412 | .662857 | .565854 |

Contact proposals improve the missed 71-frame interval's buffered best IoU
0→.146 and mean short IoU .399→.424, but add false possessions: predicted count
17→21 and primary .729→.653. A better short average therefore does not make this
a better possession solution. Phase proposals are identical to the reference in
all three protocols. Broad weak-head replacement loses existing short recoveries.
There are 75 primary regressions among130 method/block comparisons, with no
positive block primary changes in this family.

## Decision, checks and reproduction

No method passes the prospective gates: buffered primary gain>=.04, an additional
short IoU>=.5 match, no paired primary/Macro/short-precision regression, no loss of
previously recovered short control, and identical D masks. Accordingly no finalist
was sent to seeds43/44, no full-match candidate was built and no promotion was
made. This is a failed research round, not a new accuracy gain.

All60 official CLI score reports agree exactly with the research metrics.
Every stored prediction passes a fresh shared-inference replay. All65 fitted
weak-screen checkpoints reload to exactly their saved OOF probabilities. All74
tests pass, including exhaustive endpoint likelihood, adjacent-island topology,
team symmetry, weak/public/reserved exclusions, training-weight provenance and
D-mask preservation. Protected starting files and all967 old ledger rows verify.
The audit is `data/control_precision_v1/integrity.json`.

Use the existing Python3.12 `.venv` and pinned research dependencies, with prior
selected confirmation/cache artifacts present. Fresh output directories preserve
existing screens:

```sh
.venv/bin/python experiments/research_short_precision.py --output data/control_precision_v1/screen_reproduce
.venv/bin/python experiments/research_weak_ownership.py --output data/control_precision_v1/weak_ownership_screen_reproduce
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

`finalize_precision_screens.py` audits the original local screen paths and the
pre-round archive; it is a history-specific audit, not a fresh-checkout entry
point. `update_experiment_ledger.py` includes both screens. Both local output
folders retain plans, source/input hashes, predictions, official scorer inputs
and reports, fold/segment/block tables, gates, execution logs and all fitted
models/probabilities. The weak folder also retains feature tables and weak sample
arrays under ignored `data/`.

The target feedback loop remains red. Repeating these smoothing or weak-phase
variants without a new diagnostic reason is unsupported. Further work needs a
meaningfully different source of ownership information or representation. These
screens do not rule out better physical tracking or a different learned model.
Nothing was committed, submitted, uploaded or published, and no private data,
new labels, competition images or video descriptions entered hosted context.
