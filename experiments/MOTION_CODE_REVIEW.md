# Native-motion code review

The [code-review skill](/Users/derekko/.codex/skills/code-review/SKILL.md) ran
independent Standards and Spec reviews against the preserved pre-round source
tree at `data/control_pair_motion_v1/review_reference/experiments/`.
The fixed diff was `diff -ru --exclude=__pycache__ data/control_pair_motion_v1/review_reference/experiments experiments`.
The user-provided handoff and three frozen plans supplied the specification.
HEAD remains `3de6b60`; `experiments/` was already untracked, so the explicit
source archive supplies the reference. No commits or external tracker writes occurred.

## Standards

No actionable source findings. Training uses eligible public targets only;
source, cache and schema checks cover reproduction. Production takes the
`replace_current` branch and never executes the historically biased event decoder.
Its final mask preserves every reference H/A/X and permits only D→H/A changes.
Baseline preparation and optional historical auditing support fresh reproduction.
The separately frozen objective retains the earlier substantial and extra-short
failures. Retained duplication supports historical comparisons; no substantive
Fowler-smell findings were identified.

The reviewer independently ran all 112 tests. Saved confirmation evidence includes
39 exact model reloads, shared inference parity and official agreement for 18
evaluations. Final artifact/documentation review independently verifies all 17
files match between the two full builds, including manifests; 8 model and 6
probability hashes, current source/cache/input hashes and retained artifacts match.
Exactly 811 changes are D→H/A; every reference H/A/X is unchanged. All 1,177 preceding
ledger rows are preserved; 72 new rows produce 1,249. Fresh preparation and both
ordinary finalizers pass without historical snapshot files. The official format
validator accepts 199,319 rows. Report wording and reproduction prerequisites agree
with the evidence. **0 unresolved Standards findings.**

## Spec

No actionable source or artifact findings. The D-only policy matches the frozen
plan, and the historical event decoder is separated from production. Training
exclusions, at most 210 frames of observation context, block/half decoding and
fresh preparation meet the specification. Both full builds and manifests match;
current hashes, retained artifacts, 199,319-row contract and 811 D→H/A-only changes
were independently checked. All nine prospective incremental gates pass; both
original criteria remain false. All 1,177 prior ledger rows remain unchanged.

The final report's scores, falseD reductions, preserved D recall, 8-model artifact
counts and 1,249-row inventory match the evidence. It distinguishes passing
incremental checks from failed original objectives, preserves repair history and
rejected methods, and does not treat full-match changes as verified corrections.
The fresh-workspace audit has completed successfully, and selection wording is
synchronized. **0 unresolved Spec findings.**

Standards: 0 findings. Spec: 0 findings. Review does not establish independent
generalization, private full-match accuracy or maximum score.
