# Expanded round code review

The `code-review` skill reviewed Standards and Spec independently. The fixed
reference is HEAD `3de6b60` plus the preserved original untracked experiment
snapshot at `data/continuation_20261021/preserved/experiments_before.tar.gz`.
No commits were created. New files were inspected directly because the original
`experiments/` directory was already untracked.

## Standards

No remaining actionable findings in the selected path. Three reproducibility
and preservation issues were found and fixed:

- Validation always rebuilds padded features; same-shaped stale caches cannot
  silently be reused after source/input changes. The actual final run started
  without a cache, and independent full builds also confirmed feature equality.
- Finalization accepts the same `--out` option as validation.
- Alternative model builds use separate default output directories, so choosing
  CatBoost cannot overwrite the selected XGBoost candidate by default. A
  regression test covers this case. The original candidate directory is protected.

The reviewer found no held-out-label leakage, symmetry violation or declared
context overrun. Validation and full-match inference share fitting, symmetric
probabilities and decoding. Source/input/output hashes are recorded. There were
no substantive Fowler-smell findings; rejected research prototypes were checked
for reproduction dependencies without inspecting restricted video or tracking.

## Spec

Zero actionable findings. Final scores and claims match aggregate evidence:
buffered spellF1 0.5231→0.6125, MacroF1 0.7278→0.8341, mean minute 0.5996→0.6500;
11 IoU-0.5 matches retained and unmatched predictions reduced from 11 to four.
Every paired XGBoost seed improves aggregate metrics across all three protocols.

The report discloses false-D frames increasing 2→164, minute-3 regression,
short-possession misses, incomplete stoppage recovery and a smaller longer-block
gain. Choosing the secondary finalist after stress checks is explicitly adaptive.
The old gates, revised frozen gates and later robustness assessment are distinct.
The old model is preserved; no private or independent superiority is claimed.

Final totals: Standards 0 unresolved (three fixed); Spec 0. Both axes support
the new model as a development default with the documented limitations.
