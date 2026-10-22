# Anchored rescue code review

Two independent reviewers used the `code-review` skill's Standards and Spec axes
against `data/control_boundary_v1/review_reference/experiments/`, extracted from
the source archive captured before this work. Source is untracked; HEAD stays
`3de6b60`. No commits, publication or external messages were made.

## Standards

The initial review found two reproducibility issues: inherited input hashes were
not checked against actual current competition inputs, and the new runner/build
depended on an ad hoc reference snapshot. Both are resolved. The builder now
checks actual input and public-label hashes before building features, and
`prepare_boundary_refinement.py` plus the complete documented command chain
recreate/verify reference caches without overwriting history.

The reviewer verified that the raw parity check covers every column consumed by
shared inference while base/ordered parity remains complete. All 54 tests pass.
Training exclusion, shared inference, team symmetry and preservation logic were
sound. No substantive Fowler-smell findings. **0 unresolved actionable findings.**

## Spec

The initial review found that all-column raw parity prevented the full build
because full-match event vocabularies add columns, and that snapshot dependencies
were absent from the builder's short reproduction example. The corrected check
compares the eight actual raw inference inputs, and the preparation script/report
now provide the complete reproduction chain. The failed build and log are retained.

No algorithmic leakage, manual patches, scope creep or result/spec mismatch was
found. All nine D-recall deltas are zero; all 39 clipped-block primary scores avoid
regression; all previous 778 ledger rows and protected candidate hashes remain
unchanged. Candidate, model and probability hashes verify.

A final provenance check noticed that the first successful manifest preceded a
finalizer CLI change. That build is preserved, and an independent final-source
rebuild generates identical predictions. Final manifests and all source/input
hashes agree, as verified by `verify_boundary_artifacts.py` and recorded in
`data/control_boundary_v1/integrity.json`. **0 unresolved actionable findings.**

The reviews support correctness and reproducibility within the stated development
checks; they do not establish independent or full-match predictive accuracy.
