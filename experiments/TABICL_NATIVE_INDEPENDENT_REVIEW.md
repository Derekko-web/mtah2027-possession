# R54 native TabICL implementation review

Reviewed independently on 28 October 2026 while the frozen native screen was
running. This review reads implementation and synthetic prerequisite evidence;
it does not inspect native possession outcomes, change source, run an alternative
method, or establish model quality.

No blocking leakage, composition, or selection defect was found in the reviewed
paths. The native freeze contains 416 implementation files, and an independent
SHA-256 check found zero implementation mismatches. The R53 passed summary is
explicitly hash-bound by the native plan and matches that binding.

## Checks

- `tabicl_native.py:34–59` selects only outer-training frames on the fixed global
  5 Hz phase, filters H/A using training labels, augments paired team swaps with
  complementary targets, and fits the external median imputer on that augmented
  context. No validation labels enter this path. The controller additionally
  erases all labels outside training and requires identical context, targets,
  indices, and medians (`research_tabicl_native.py:100–108`).
- `rich_features.py:143–151` swaps all `h__`/`a__` columns, including the frozen
  latent columns. The worker predicts each query view separately, reverses the
  swapped H/A columns, and averages (`tabicl_native.py:106–120`). Fixed 512-row
  calls and last-row padding preserve the preregistered numerical interface.
- The acquired official estimator fits its unique-feature filter and both
  normalization pipelines on context only, then uses their transform methods for
  queries (`tabicl/_sklearn/classifier.py:502–524,729–735` and
  `preprocessing.py:1058–1091,1192–1203`). Query labels are absent from the worker
  payload, whose permitted keys are explicitly restricted.
- `factorized_latent.py:53–79` composes conditional H/A with the original slot-0
  live mass and retains the original D column. Unsupported latent frames retain
  the old slot-0 probability. The existing decoder runs unchanged; the final
  overlay asserts exact reference D/X support. This verifies the intended
  integration rather than a newly fitted D model.
- The source encoder and interpolation are frozen dependencies. Interpolation
  requires both adjacent endpoints to be supported; its established 79-frame
  encoder footprint lies within the unchanged 250-frame split exclusions.
- `research_tabicl_native.py:168–177` invokes the existing selected8 safeguard
  conjunction, adds strict improvement over the historical buffered record for
  seed 42, and requires all three protocol pairs. The historical substantial-gain
  and extra-short-match fields remain diagnostics. Seeds 43/44 are refused unless
  the screen passes (`:60–61`).
- Independent replay reconstructs context/imputation inputs and the official
  fitted estimator in fresh workers. It compares context tensors and final owner
  probabilities exactly (`tabicl_native.py:124–128`), followed by byte-identical
  scores, gates and decoded CSVs (`research_tabicl_native.py:179–186`). Cache
  fingerprints and checkpoint checks verify unchanged official cached inference.
- Fixed-pair scheduling limits concurrent workers to two. The shared registry
  checks their combined RSS, with separate per-worker memory, operation and wall
  limits. R53 independently passed the full-sized synthetic context before this
  screen; the native worker uses its one-thread initializer and `n_jobs=1`.

## Remaining limits and completion work

This is a source/protocol audit, not a completed native-result audit. Verify each
worker's actual exit, supervision record, persisted files and all final manifest
hashes after the screen. The replay compares the final symmetrized ownership
probability; the two individual query-view arrays are retained but are not
separately asserted against their parent arrays. Exact owner replay is the
quantity consumed by this decoder.

The runner writes score artifacts but does not append the shared experiment
ledger itself. Append every completed score row during closeout, preserving the
frozen ledger prefix. A successful screen still requires the planned independent
replays, two confirmation seeds, all nine safeguard comparisons, full/public
feature parity, two identical full CSV builds, format validation and final
independent review before promotion. Reused public possessions and correlated
development protocols do not establish private-match accuracy.
