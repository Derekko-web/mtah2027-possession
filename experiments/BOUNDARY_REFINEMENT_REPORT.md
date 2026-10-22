# Anchored live rescue — 21 October 2026

Anchored rescue recovers most of a weakly recognized short possession and reduces
false dead-ball extensions. The preceding consensus model and all failed
experiments remain preserved. No submission, publication, commit or upload was
made. This is adaptive within-match development on the same 17 public possessions,
not an independent accuracy estimate.

## Paired development results

Official scorer, seed 42. The reference is the selected consensus model from
`CONTROL_REFINEMENT_REPORT.md`, not an earlier, weaker baseline.

| Buffered minute folds | Reference | Anchored rescue |
|---|---:|---:|
| Stitched spellF1 | 0.682353 | **0.729412** |
| MacroF1 | 0.837288 | **0.844580** |
| Mean clipped-minute spellF1 | 0.713333 | **0.748889** |
| False D frames | 140 | **92** |
| D recall | 66.712% | **66.712%** |
| Short-live mean frame recall | 51.590% | **60.785%** |
| Short-live mean best same-label IoU | 0.321615 | **0.399285** |
| Short-live runs with IoU >=0.5 | 2 of 6 | **3 of 6** |
| All live possessions matched at IoU 0.5 | 12 | **13** |
| Spurious / missed at IoU 0.5 | 5 / 5 | **4 / 4** |

The 48-frame false-D reduction is 1.92 seconds, or 34.3% of the reference's
false-D error. It comes entirely from the 87-frame H possession at
17293–17379. Its recall rises from 19.54% to 74.71%, and best same-label IoU
from 0.1650 to 0.6311. This is a better recognized and better bounded short
possession; it still loses 22 frames to D.

The 71-frame A and 33-frame H possessions still have zero recall in the main
protocol. Their correct-team evidence is weak even in the raw model outputs.
The recovered 19-frame H possession remains 15346–15386 against true
15346–15364: full frame recall but IoU only 0.4634. No endpoint repair for that
run was promoted. The final live run's false-D extension is unchanged.

| Seed-42 protocol | Reference spellF1 | Rescue spellF1 | Reference false D | Rescue false D |
|---|---:|---:|---:|---:|
| Five buffered minutes | 0.682353 | **0.729412** | 140 | **92** |
| Whole intersecting runs excluded | 0.640000 | **0.662857** | 175 | **152** |
| Three buffered 100-second blocks | 0.644444 | 0.644444 | 16 | 16 |

All three seeds have the same primary scores shown above. Buffered false-D
counts are 140/139/141 before and 92/93/94 after. Purged counts are 175/174/174
before and 152/156/154 after. Longer-block predictions are exactly unchanged.
D recall is exactly unchanged in all nine paired evaluations, as are mostly
missed stoppage counts (1/0/1 by protocol). No previously mostly recovered short
run becomes mostly missed. All nine frozen gates pass.

There are no primary-score regressions in any of the 39 clipped block/seed
comparisons. Only block 5 improves: seed-42 buffered 0.666667→0.844444 and
purged 0.520000→0.600000. All other blocks are unchanged. The live-run fragment
count also stays unchanged (12/14/11 by protocol). These correlated improvements
are concentrated on one possession, not broad recovery across the match.

Visible-ball stoppage robustness remains unsolved: the main protocol still
mostly misses one stoppage, and the longer-block protocol still completely
misses the final visible-ball stoppage. The existing recovery of other stoppages
is preserved; no new stoppage recovery is claimed.

The official 1,000-draw marginal 90% possession-bootstrap intervals overlap:
reference 0.517647–0.835294, rescue 0.576471–0.870588. Repeated seeds and protocols
reuse the same labels with overlapping training sets. These are adaptive stress
checks, not independent tests, selection-adjusted confidence intervals or proof
of better full-match/generalization accuracy.

## Mechanism and exclusions

The four saved models are unchanged. The preceding canonical consensus pipeline
still supplies the reference probabilities, contact attenuation and ordinary
penalty-3 decoder. The added mechanism addresses a specific inference gap: the
old enclosure gate accepts short opposite-team ownership islands but does not
extend a supported live possession into current D.

For each model-derived live proposal of 8–125 frames from the older ensemble:

1. Require the independent fast head's mean support >=0.65 and at least three
   contact-supported frames, using exactly the existing definitions.
2. Require current decoded D overlap and an existing same-team live anchor in
   the proposal or its 13 neighboring frames on either side.
3. Inside that proposal use the fast ownership probabilities and the lower of
   the current and older D probabilities, retaining the original bounded
   contact/motion attenuation. Decode again with penalty 3.

Every gate reads the unchanged baseline sequence, preventing cascading rescue.
There are no true-run endpoints, frame IDs, hand patches or label inputs in
inference. Future observations and global decoding within each block/half are
used offline. Current/older probabilities use 51-frame averaging and the fast
head uses 11 frames. Block-clipped proposals follow the same rule; their
context can differ from full-half proposals, as with any blocked decoding.

`boundary_refinement.decode_refinement` is shared by OOF verification and the
full builder. All fitting uses explicit eligible training indices. The new
micro-history heads also derive weights from training-only runs, but are
rejected. Padded development observations through frame 18934, 250-frame
exclusion buffers and existing whole-run exclusions are retained. No private
labels, new annotations or frozen audit windows were accessed.

## Failed hypotheses and retained artifacts

`BOUNDARY_REFINEMENT_PLAN.json` froze a seven-method seed-42 screen across three
protocols. `BOUNDARY_CONFIRMATION_PLAN.json` then froze just current and anchored
rescue for seeds 42/43/44. The six alternatives include two constrained boundary
relocations, anchored rescue, a rescue/relocation combination, and two learned
micro-history heads. All 21 screen and 18 confirmation evaluations are retained.
The ledger now has 817 dependent rows; the previous 778 rows remain unchanged
by identity. They are not 817 independent experiments.

- Boundary relocation within 13 or 25 frames uses joint likelihood optimization
  with fixed run labels and equal log weighting of coarse/fine evidence. Both
  radii yield identical screen predictions. They slightly improve mean short IoU
  but do not improve spellF1, and lose real D recall. The 19-frame endpoint stays
  unresolved. The combined rescue/relocation method also loses some D recall.
- New subsecond motion-history heads, with frame or equal-live-run weights,
  lose the previously recovered 19-frame possession in buffered validation.
  The frame head falls to 0.6125 spellF1; the run-weighted head is also 0.6125.
  Neither replaces the current fast head.
- The first full build failed an overbroad parity check: full-match raw features
  contain additional automatic event-category columns. The failed directory
  and log remain under `data/control_boundary_v1/failed_build_raw_schema/` and
  `build_raw_schema_failure.log`. The corrected check compares all base/ordered
  inputs and the eight raw columns actually used by shared inference.

Source and original confirmation/candidate snapshots are in
`data/control_boundary_v1/experiments_before.tar.gz`, `baseline_confirmation/`
and `baseline_candidate/`. Screen/confirmation outputs include full OOF
predictions, probabilities, models for rejected heads, diagnostics, official
reports, plan hashes, input/source hashes, paired gates and block deltas.
OOF scorer files are diagnostic arrays with X outside the public window, not
full-match submission candidates. Competition data and derived arrays stay in
ignored local `data/`.

## Reproduction

Use Python 3.12 in the existing `.venv`, pinned
`experiments/requirements-research.txt`, and the original local data symlinks.
From a fresh output directory, the complete chain is:

```sh
.venv/bin/python experiments/validate_temporal_candidate.py
.venv/bin/python experiments/finalize_temporal_validation.py
.venv/bin/python experiments/confirm_control_refinement.py
.venv/bin/python experiments/finalize_control_refinement.py
.venv/bin/python experiments/build_control_candidate.py
.venv/bin/python experiments/prepare_boundary_refinement.py
.venv/bin/python experiments/research_boundary_refinement.py
.venv/bin/python experiments/research_boundary_refinement.py --stage confirmation
.venv/bin/python experiments/finalize_boundary_refinement.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_anchored_candidate.py
.venv/bin/python evaluation/validate_submission.py data/candidate_anchored_control/candidate.csv
```

Existing prerequisite OOF/reference outputs can be reused; no historical search
is required. `prepare_boundary_refinement.py` deterministically copies those
references on a fresh checkout and verifies rather than overwrites existing
snapshots. The new screen, confirmation and full-builder commands refuse an
existing output directory. To repeat without disturbing frozen artifacts:

```sh
.venv/bin/python experiments/research_boundary_refinement.py --output data/control_boundary_v1/screen_reproduce
.venv/bin/python experiments/research_boundary_refinement.py --stage confirmation --output data/control_boundary_v1/confirmation_reproduce
.venv/bin/python experiments/finalize_boundary_refinement.py --output data/control_boundary_v1/confirmation_reproduce --screen data/control_boundary_v1/screen_reproduce
.venv/bin/python experiments/build_anchored_candidate.py --output data/candidate_anchored_control_rebuild
```

The new full builder checks actual competition input and public-label hashes,
all reference source/model hashes, padded feature parity, and exact saved-model
probability parity over the full match. It then applies only the confirmed
shared decoder. It requires the reproducible original consensus candidate;
its four checkpoints and probability arrays are copied without refitting.

## Final artifact and integrity

The selected full-match file is `data/candidate_anchored_control/candidate.csv`.
Two independent generations produce identical CSV bytes, SHA256
`973e1670e9e1240ecdde3bbabebf1e803f0d3b6259aad1b9b3223c413d26dcfd`.
The official validator accepts all 199,319 rows, frames 0–199318.
Counts are H=51,621, A=32,757, D=74,479, X=40,462. It reports 348 possessions,
versus 346 in the preceding consensus candidate. Those full-match count changes
cannot be judged accurate without unavailable labels.

Exactly 533 full-match labels differ: 430 D→live changes and 103 H↔A changes.
The raw probabilities and all four models remain byte-identical to the consensus
reference. All saved models reload to exactly those probabilities. Input/source,
model/probability and CSV hashes verify, and the final candidate/rebuild manifests
agree. The original consensus and ordered full CSV hashes are unchanged, as is
the frozen unlabelled annotation manifest.

The first successful build's manifest preceded a finalizer CLI-only change.
That entire build remains in `data/control_boundary_v1/first_build_before_finalizer_cli/`.
The second independent build generates the identical CSV using the final source;
its complete artifact and verified manifest populate the canonical directory.
This provenance correction changes no predictions.

Fifty-four tests pass, including exhaustive constrained-boundary likelihood,
team symmetry, training-target exclusion, padded micro-feature context, blocked
abstention/reset behavior, anchored-rescue evidence gates and promotion rejection
on secondary regressions. All 18 confirmation outputs pass shared-decoder parity;
seed-42 outputs exactly reproduce the screen, and all official metrics agree.
The original 45 tests, 27 score comparisons and consensus full rebuild also pass.

`data/control_boundary_v1/integrity.json` and `integrity.log` record the completed
local audit. `verify_boundary_artifacts.py` reruns it against the preserved
historical snapshots; those history-specific checks are not prerequisites for
regenerating a candidate on a fresh checkout. Independent Standards and Spec
review findings and resolutions are in [BOUNDARY_CODE_REVIEW.md](BOUNDARY_CODE_REVIEW.md).
