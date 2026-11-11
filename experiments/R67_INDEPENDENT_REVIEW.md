# R67 independent source review

APPROVED for prospective ROOT-reviewed freeze and the unchanged four-stage invented gate. No blocker found. This review performed no scientific imports, model calls, native-array reads or freeze actions. Approval is neither a runtime pass nor native-inference authorization.

Final bound SHA256:

- Wrapper: `bf69d5c22fae4694a571d95d6668abff80f6c186d6eaa0b2af6ef96cba3ae073`
- Plan: `1a325754ac59b4cb25b4520e9de94195d177784eb3c7da89f14552c8c442b8d6`
- Tests: `45b9ed3aadb09678ba7d68e8a4ef569f2d3b164bfefd5ed30bc45bad8665f99d`
- Source proof: `2c43b50b0680c30f7ec20a930dd01018e3b368094255cf2260ffc9bbc6dfa96b`

The final source preflight binds these exact hashes and passes. The reviewer independently reran all15 pure source/mocked-API tests successfully and independently verified the entire prior frozen chain. R66's terminal summary is still failed (`2e3e22d1670d2a1b05fd52a5d5faacb4505094fd164f5c40574ae53ba5973c0a`): moderate first/rebuild passed, full first failed on its second canonical query, full rebuild absent. No failed round is reclassified.

## Change and source basis

Exactly one external `torch.mps.synchronize()` plus one `torch.mps.empty_cache()` is added before each canonical or diagnostic prediction callback, inside the unchanged R57 timed callback. Before/after driver and live-tensor counters are read and logged under the supplied operation output directory. Changed live-tensor bytes fail before prediction. Driver reduction is diagnostic and never selects behavior. The original prediction callback is called exactly once. Fit/fingerprint callbacks are passed unchanged; cleanup or prediction errors propagate without retry or post-failure flush.

The retained eight official Torch source/license files are checksum-bound to installed Torch2.15.0 git revision `d0f2af8b66b20da4baa50644c0cd3f887a2eae2e`; installed MPS Python API bytes match the pinned author source. The Python API/C++ module/MPSHooks/allocator chain releases only wholly free unretained heaps after synchronization and prohibits release of in-use blocks. It also explicitly clears the framework MPSGraph compilation cache, deleting cached graphs. This permitted framework cache is distinct from TabPFN fitted preprocessors, retained training context and checkpoint tensors. The driver counter includes non-allocator framework memory; driver minus live tensor bytes is not a proven reclaimable-cache measurement. The allocator already retries cached-buffer release during allocation failure, so source evidence does not guarantee this experiment will prevent OOM.

## Preserved bindings and gates

The reviewer compared complete code objects against frozen R66/R57 for worker, main, supervise, canonical_predict, checkpoint_fingerprint, synthetic_data, fixed_environment, runtime_paths, probability_checks, array_record, exact_array, owned_rss and read_json_if_present. Every original MemoryMonitor and NativeDispatchGuard method is also byte-identical. The original operation's code object is unchanged, its globals are the returned execution module, and only the operation binding points to the new prediction wrapper. Thus its original timing, live memory and native-dispatch checks include cleanup, logging and graph-recompilation costs. Output-relative logging supports later native core reuse, but authorizes none now.

The original R63 one-time categorical validation boundary remains exact; its active core is the returned execution object and its original complete raw fingerprint is R60's unchanged function. No field is removed, reset or excluded. Original checkpoint equality, full retained preprocessing/context/configuration and embedded fitted RNG state hashes, repeated probabilities and fresh-context reconstruction remain mandatory. The cleanup code references no model/context object and calls no RNG or model-state mutation API. The live-byte check is an additional guard, not a replacement for full raw-state and weight equality.

R66 hardware and acquisition files are byte-identical copies:4GiB internal allocator allowance,2GiB planning RSS reservation and the same pre-import fraction. The enforced combined owned RSS plus FULL driver bound remains6GiB, with300seconds per operation and900seconds per worker. Original eight views, float32 MPS, author factory/checkpoint/settings, canonical512 padding, all invented data and serial scheduling are unchanged. All FOUR original moderate64x166/full3000x166 first/rebuild stages must pass; no waiver, fallback or partial-fold rescue exists.

The new freeze binds every retained R66 runtime file, its freeze manifests/runtime log and terminal report; recursive verification retains older failed evidence and read-only dependencies. ROOT must provide its own final reviewed metadata and schedule any model gate. R65 still requires an actual complete R67 technical pass and its separately reviewed freeze before native payloads, fitting or inference.
