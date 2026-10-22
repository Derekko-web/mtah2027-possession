# Stoppage recovery code review

The `code-review` skill ran separate Standards and Spec reviews against
`data/control_renewal_v1/review_reference/experiments/`, extracted from the source
snapshot taken before this continuation. HEAD remains `3de6b60`; experiments
remain untracked and no commits or uploads were made.

## Standards

Two initial findings were resolved:

- The builder verified cache bytes but did not check all feature-cache sources.
  It now checks each recorded source against current code. The cache source list
  includes the restart-feature dependency `stoppage_features.py`.
- The report understated physical context as 125 frames. Nested displacement
  summaries reach 149 past frames, while restart kernels reach 250 frames either
  way and supplied phase metadata spans tagged intervals. The wording is corrected;
  existing 250-frame exclusions cover the physical inputs.

Original caches, confirmation, full builds and generating source/logs were
preserved before the provenance repair. Full features and weak samples were
regenerated and compared exactly, then confirmation and both full builds were
repeated. The reviewer independently ran all 66 tests. Training-only targets,
weak exclusions, retained ownership, artifact preservation and algorithmic inference
were sound. No substantive Fowler-smell findings. **0 unresolved actionable findings.**

## Spec

The initial context wording issue is resolved. No algorithmic leakage, manual
patches or promotion-rule mismatch was found. The report correctly distinguishes
the unmet original primary-gain criterion from the separately frozen adaptive
stoppage-recovery objective.

The reviewer independently verified all 1,554 weak-sample exclusions, all nine
recovery gates, preservation of the previous 817 ledger rows and protected
candidates, identical final CSVs/manifests, current source hashes and unchanged
predictions after the provenance repair. The reproduction chain includes required
preparation, and historical integrity auditing is separate from fresh generation.
**0 unresolved actionable findings.**

The reviews establish no independent or private full-match accuracy claim.
