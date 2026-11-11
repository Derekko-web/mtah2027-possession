# R66: fixed 4-GiB allocator gate failed

R66 is terminally **FAILED**. The same two moderate invented stages pass with
exact canonical probability, retained-preprocessing and checkpoint reconstruction.
The full 3,000×166 context's first 512-row original query completes in
113.111666 seconds with 2,304 official native MPS wrapper calls and matching
dispatches. The following `swapped_0` query raises an allocator OOM. Full rebuild
never starts; there is no native TabPFN input, possession score or promotion.

The failure reports 4.34GiB allocated under the fixed 4GiB internal allowance,
plus an attempted 78MiB allocation. The supervisor records exit1 without a
timeout, monitor failure or total-memory abort. Its conservative owned-RSS plus
full-driver peak is 5,317,279,744bytes, about4.952GiB, below the unchanged6GiB
combined cap. This does not turn the internal allocator failure into a pass.

At the successful original-query boundary, driver allocation is3,631,349,760
bytes while live tensor allocation is28,978,432bytes. That gap is consistent
with retained execution resources, but it is **not** a measured reclaimable
cache size: the driver counter also includes framework allocations and partly
occupied heaps. The original-query result is not a completed full-stage
repeat/rebuild proof. Full probability artifacts are saved only after all
canonical queries, which this worker did not finish.

Evidence is preserved under
`data/control_tabpfn_mps_balanced_budget_v1/runtime_assessment/`: the top-level
summary, both moderate summaries, full first supervision/failure/memory and
operation records, stdout and stderr. The top-level summary SHA256 is
`2e3e22d1670d2a1b05fd52a5d5faacb4505094fd164f5c40574ae53ba5973c0a`.
No failed R66 file is repaired, overwritten or rerun.

## Source-supported prospective R67

The official public `torch.mps.empty_cache()` API releases unused allocator
resources. The installed Torch2.15 Python module is byte-identical to official
commit `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`. Its allocator implementation
synchronizes streams, releases only wholly free/unretained heaps and protects
live buffers. The public API's C++ bridge **also clears the framework MPSGraph
compilation cache**; that cache is separate from TabPFN's fitted preprocessing
and training context. Recompilation costs must remain inside the original
operation timer.
[Official API documentation](https://docs.pytorch.org/docs/stable/generated/torch.mps.empty_cache.html),
[pinned Python API](https://raw.githubusercontent.com/pytorch/pytorch/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/torch/mps/__init__.py),
[allocator](https://raw.githubusercontent.com/pytorch/pytorch/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/mps/MPSAllocator.mm),
[public API bridge](https://raw.githubusercontent.com/pytorch/pytorch/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/mps/MPSHooks.mm),
[graph cache clear](https://raw.githubusercontent.com/pytorch/pytorch/d0f2af8b66b20da4baa50644c0cd3f887a2eae2e/aten/src/ATen/native/mps/OperationUtils.h).

The allocator already tries cached-buffer release during an allocation failure.
Calling the public API at a quiescent prediction boundary may release resources
earlier and clear cached graphs, but source inspection does not guarantee it
will prevent this OOM. R67 therefore proposes one fixed
`synchronize(); empty_cache()` before every prediction, inside the unchanged
timed callback. It preserves all four stages, allocator/total/time limits,
512-row queries, eight views, float32 model mathematics and exact raw
cache/weight/probability reconstruction checks. Driver reduction is a diagnostic;
live tensor bytes must remain unchanged. Fit and fingerprint operations receive
no added flush. A failed prediction is never retried or flushed as a rescue.

Only new source preparation and pure mocks are authorized here. Independent
root review and freeze precede any R67 model calls. All four runtime stages must
pass before a separately reviewed native plan can process native inputs.
Selected8, the ledger and the higher-score objective remain unchanged by R66/R67.
