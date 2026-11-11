# R59: loader repair succeeds; preprocessing audit assertion fails

The separately copied/resealed loader repair is frozen after independent
whole-byte/header/signature/source review. All29 corrected binaries retain
every original instruction and file-backed payload byte. All original R57
inputs and its import failure remain exact. [Plan](TABPFN_MPS_LOADER_REPAIR_PLAN.json).

The repaired runtime imports successfully, fits the invented64x166 context and
verifies translated checkpoint tensors exactly on MPSfloat32. It then exits1
after30.72seconds because R57's observational fingerprint function rejects every
non-None `gpu_preprocessor`. No model prediction or attention forward occurs.
The resource monitor stays below its cap; no runtime limit causes this failure.

That assertion misreads the published API. In the retained official source,
`preprocessing/ensemble.py:337–375` creates a legacy torch pipeline even when
`ENABLE_GPU_PREPROCESSING=False`; `preprocessing/torch/factory.py:139–156`
adds the default numerical soft-outlier clipping step. Bulk quantile/SVD
preprocessing remains on CPU. The implementation's assertion, rather than an
observed setting change, causes this rejection.

R59 remains failed, with all output and traceback retained under
`data/control_tabpfn_mps_loader_repair_v1/`. A separately frozen validation-only
correction must inspect and fingerprint this exact author-defined legacy step,
its parameters and non-retained cache. It must preserve all model mathematics,
inputs, outputs, limits and exact-replay requirements. No previous failure is
rewritten as a pass, and no possession score or selected-model change follows.
