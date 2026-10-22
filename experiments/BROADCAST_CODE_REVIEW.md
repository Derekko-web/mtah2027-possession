# Broadcast source closure code review

The `code-review` skill ran independent Standards and Spec reviews against
`data/control_broadcast_v1/review_reference/experiments/`, extracted from the
source archive saved before this round. The fixed reference is explicit because
`experiments/` was already untracked; HEAD remains `3de6b60`. No commits or
external tracker writes were made.

## Standards

Two actionable reproduction findings were resolved:

- Camera input manifests stored resolved absolute data paths. A frozen manifest
  SHA therefore depended on the machine's data location. The new six-asset
  logical identity resolver preserves byte checks across relocation and rejects
  changed content. Prefix/full feature preparation, finalizer and candidate
  builder now use those identities. Relocation/corruption tests pass.
- Research required a session-created reference directory, and the finalizer
  always required the historical source archive. A deterministic baseline
  preparation script now copies or verifies the unchanged stoppage confirmation.
  Historical archive and protected-snapshot auditing are explicitly optional
  through `--audit-history`. Fresh runs report those checks as unperformed.

The reviewer independently ran all 97 tests, then all 15 broadcast tests after
the reproduction repair. Inference, observation context, training-only targets,
unchanged D decisions, source/cache guards and preservation controls looked
sound. No substantive Fowler-smell findings. **0 unresolved actionable source
findings.** Final artifact review independently verifies byte-identical full builds and
manifests, seven model hashes, five probability hashes, all source/camera input
hashes, unchanged pre-repair predictions and 13 feature-cache files, the D mask,
378 changed H/A frames, both official format checks and all 97 tests. The new
workspace audit supports ordinary reproduction without historical session files.

## Spec

The independent Spec reviewer found the same two reproduction issues and
confirmed both repairs. No target leakage, manual patches, block/half errors or
promotion-rule mismatch was found. The report distinguishes the failed earlier
substantial/extra-short-match criteria from the separately frozen incremental
objective. It discloses lost correct frames, unchanged D masks and the absence
of an additional short IoU .5 match. **0 unresolved source/report findings.**
The reviewer also independently verifies the identical full artifacts, current
source/input/model hashes, all nine incremental gates, all 1,081 preceding ledger
rows, the 1,156-row total and unchanged predictions after the provenance repair.

The first review attempts encountered usage-limit failures; the same two agents
resumed and completed source review after the repairs. Those interruptions are
not represented as completed reviews. Review does not establish independent or
private full-match accuracy.
