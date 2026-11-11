# Full-context TabPFN allocation repair

The unchanged2352-row context and512-row native prediction failed under the4GiB MPS allocation ceiling in a clean process. Total conservative measured usage remained below6GiB. This reproduced the archived full-build failure and ruled out leftover-process allocations as its cause.

The separately frozen treatment reallocates the same6GiB total from4GPU+2host to5GPU+1host. No learner parameters, seed, checkpoint, paired context,166 features, eight views,float32 precision,512-row schedule or correction changed. Operation/worker limits remain900/3600seconds on both clocks.

Two fresh full-context fits passed, matched the pre-OOM fitted state/checkpoint exactly, and reproduced their probabilities exactly. All five original main folds were also freshly replayed at the revised allocation. Their original/swapped/combined/repeat probability files and fitted-state/checkpoint fingerprints match the saved qualification byte-for-byte. Recipes differ only in allocation fraction. Corrected labels are exact and main SpellF1 remains **0.777142857142857**. Maximum combined measured usage across the seven positive workers was **4.823GiB**, below6GiB.

Evidence: `data/control_tabpfn_full_memory_gate_v1/gate_passed.json`, frozen plan `TABPFN_FULL_MEMORY_GATE_PLAN.json`, unchanged prior failures in full-build-v1/v2, and the source-only experiment contracts. The actual native worker is still `tabpfn_native.py`.

Full production is now running under `data/control_tabpfn_full_build_v3/`:40 fresh context fits over158857 official-half query rows, producing a199319-row candidate. A fixed-source completion workflow will perform two independent full CSV exports, exact-state/probability/time/memory checks, full D/X preservation and the official validator before changing local selection. The old candidate and selected-model document will be preserved. No private full-match accuracy is claimed.

Prevention: production admission now tests the actual full-context footprint, rather than relying on a synthetic context of similar row count. There are no temporary debug hooks in model code; the allocation experiment and failed control remain preserved as regression/provenance evidence.
