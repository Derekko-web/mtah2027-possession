# TabPFN-v2 on current supported MPS: source-only feasibility

Prepared 28 October 2026. This investigation read official package metadata and
small official source files only. It acquired no package/checkpoint, installed
nothing, imported no scientific runtime, performed no inference, read no native
competition data or labels, and changed no frozen input or dependency tree.

## Decision

**Go for one separately frozen synthetic technical gate; no-go for native
scoring until that gate passes.** Current TabPFN explicitly supports the exact
existing v2 classifier checkpoint on an explicit MPS device. An isolated
TabPFN 9.2.0 + Torch 2.15.0 + float32 route is technically distinct from the
existing TabPFN 2.2.1 / Torch 2.5 CPU experiment: current v2 is a rewritten
architecture, and current MPS uses a different attention implementation.
Source establishes support, not this Mac's time, memory or byte replay result.
Retain the CPU experiment's complete outcome; never relabel an MPS rerun as a
CPU success or select backend/precision after a failure.

## Exact current identities

Official PyPI metadata reports **TabPFN 9.2.0**, released 12 October 2026. Its
wheel is 850,035 bytes with SHA256
`bf8edb860cf490ab021d0ba6e91a9798f1b549bdb220066599661c8fc6d6089f`.
The official `v9.2.0` tag resolves to commit
`f02b3b5941717472eafd078c15ae2043676b3fcf`. References below use that commit;
some web-rendered tag pages returned older line counts, so commit source was
read directly for the cited line numbers.
[PyPI metadata](https://pypi.org/pypi/tabpfn/9.2.0/json),
[official tag reference](https://api.github.com/repos/PriorLabs/TabPFN/git/ref/tags/v9.2.0).

Official PyPI metadata currently reports **Torch 2.15.0**, Python >=3.11.
Its CPython 3.12 / macOS 14+ arm64 wheel is 138,558,517 bytes, SHA256
`593e5fd0c204284890d1c3dff643be34e549e1d24523e13402510f960d4da438`.
This is a concrete future isolated runtime candidate, not an acquired or
locally verified installation. The existing Torch 2.5 tree must remain intact.
[Torch 2.15.0 metadata](https://pypi.org/pypi/torch/2.15.0/json).

## Supported v2 identity, API and access

- `TabPFNClassifier.create_default_for_version(ModelVersion.V2, **overrides)`
  explicitly chooses the v2 source. Supply `model_path` as the absolute existing
  checksum-bound `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt`, rather than
  depending on a global cache or changing package default. The factory's v2
  checkpoint default is exactly that filename from `Prior-Labs/TabPFN-v2-clf`.
  [Factory, lines 619–674](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/classifier.py#L619),
  [checkpoint source, lines 86–106](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/model_loading.py#L86).
- V2 is absent from the download license-acceptance map, which contains v2.5,
  v2.6, v3, v3.5 and v3.5-fast. A present explicit local path requires no hosted
  inference or newer model license acceptance. The existing v2 License 1.1 and
  exact checkpoint SHA256 from the R56 plan remain binding; this research does
  not replace their provenance or authorize any terms acceptance.
  [Access dispatch, lines 548–575](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/model_loading.py#L548).
- The loader recognizes original v2 checkpoints lacking `architecture_name`,
  selects `tabpfn_v2`, and obtains the v2 inference defaults. The rewritten v2
  class translates legacy state keys. This is explicit backwards support,
  rather than an inference from a shared product name.
  [Loader, lines 1194–1240](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/model_loading.py#L1194),
  [v2 config resolution, lines 1243–1272](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/model_loading.py#L1243),
  [legacy architecture/state translation](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/architectures/tabpfn_v2.py#L1).

## MPS and precision

Explicit `device="mps"` raises if MPS is unavailable or Torch <2.6, avoiding the
automatic CPU fallback that `device="auto"` permits. Package requirements alone
are weaker (`torch>=2.5`).
[Device checks, lines 110–204](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/utils.py#L110),
[package dependencies, lines 7–30](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/pyproject.toml#L7).

For this new gate, prospectively fix `inference_precision=torch.float32`.
That turns autocast off and forces the model inference dtype. Tensor conversion
on the inference path respects it. **CPU/sklearn preprocessing still uses its
official float64 default**; claiming all CPU-side arithmetic is float32 would
be incorrect. Leave GPU preprocessing disabled, as v2 defaults specify.
[Precision, lines 273–314](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/base.py#L273),
[tensor conversion, lines 1623–1638](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/inference.py#L1623),
[CPU dtype, line 53](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/constants.py#L53),
[GPU preprocessing default, lines 264–268](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/inference_config.py#L264).

Torch >=2.13.0.dev20260510 enables the official `torch-mps` attention wrapper;
it accepts float32 and pads head dimensions around native PyTorch SDPA. It is
consulted before MLX. Torch 2.6–2.12 supports MPS but does not establish this
same native flash route. Pinning 2.15.0 avoids that ambiguity. Record the
official `recorded_attention_backends()` output and reject an unexpected
backend; do not patch source to choose one after observing results. This
records the wrapper dispatch, not an independent profiler proof of every
internal PyTorch kernel.
[Torch MPS backend, lines 22–79](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/architectures/shared/torch_mps_backend.py#L22),
[registry order, lines 25–33](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/architectures/shared/scaled_dot_product_attention.py#L25),
[supported recorder, lines 242–254](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/architectures/shared/attention_backends.py#L242).

## Defaults that the new gate must bind

V2 still resolves to eight views at 166 input features; explicitly fixing
`n_estimators=8` prevents future automatic scaling. Its two preprocessing
families are coarse uniform quantile with original columns plus SVD, and
numeric identity; feature/class shuffling and fingerprint features are enabled,
sample subsampling is absent, and original v2 limits are 10,000 rows / 500
input features / 10 classes. Temperature remains 0.9. Internal transformed
widths can exceed 166 and must be recorded.
[V2 presets, lines 13–31](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/preprocessing/presets.py#L13),
[view count and scaling, lines 1508–1584](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/preprocessing/ensemble.py#L1508),
[v2 settings, lines 503–528](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/inference_config.py#L503),
[temperature, lines 24–28](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/inference_config.py#L24).

Use fixed `fit_mode="fit_preprocessors"`, `n_preprocessing_jobs=1`, seed42,
and no tuning. Current `n_jobs` has no effect. That engine retains transformed
contexts in RAM, runs neural forwards at predict time, and runs MPS ensemble
members singly. Do not silently substitute the different explicit KV-cache
engine or its precision defaults.
[Constructor, lines 258–291](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/classifier.py#L258),
[n_jobs deprecation, lines 607–615](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/classifier.py#L607),
[retained preprocessing engine, lines 845–856](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/inference.py#L845),
[MPS member grouping, lines 196–214](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/inference.py#L196).

## Resource and replay gate: unresolved, measurable requirements

The source always enables peak-memory saving on MPS unless an explicit bool
overrides it; select `memory_saving_mode=True` prospectively. Its chunked v2
layers make a bounded investigation credible, but no official source proves
3,000 x166 contexts / 512-row queries / 8 views fit **6 GiB and 300 seconds
per operation** on M1 Pro. A numeric memory-saving argument is not a hard MPS
allocation cap. The 27 GiB free disk is ample for the advertised individual
wheels, but a complete isolated dependency closure is still unacquired.
[Memory heuristic, lines 15–47](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/memory.py#L15),
[v2 chunked execution, lines 323–447](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/architectures/tabpfn_v2.py#L323).

Before importing TabPFN, fix `TABPFN_MPS_MEMORY_FRACTION` from a conservative
absolute Metal allocation budget and recorded `torch.mps.recommended_max_memory()`.
TabPFN's import initializes this setting and defaults to0.7, which is not a
6 GiB total-process guarantee. Keep a separate external RSS/descendant and wall
supervisor; record MPS current and driver allocations as well. Driver accounting
includes cached pools and MPSGraph allocations that current tensor accounting
omits. Conservative combined accounting may double-count unified pages but
must not omit GPU allocation. Include `torch.mps.synchronize()` inside timed
operation boundaries. Never disable the allocator upper limit.
[TabPFN import setting, lines 58–100](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/settings.py#L58),
[official Torch memory/synchronization API, lines 30–137](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/mps/__init__.py#L30).

Retain the existing invented-data shapes, canonical query/padding schedule,
fresh-process reconstruction, immutable checkpoint/model/preprocessor/cache
fingerprints and **byte-exact** probability/repeat replay requirements from the
CPU gate in a new independently frozen plan. Capture fingerprints after any
official one-time model-device initialization. A failed deterministic operation,
changed cache or weights, probability mismatch, resource abort or unexpected
backend fails this MPS gate. No tolerance or CSV reproducibility waiver follows
from official device support. Passing does not establish sports accuracy.

One compatibility preflight remains: current TabPFN passes `weights_only=None`
to `torch.load`, while Torch 2.15 resolves that to weights-only loading. Do not
infer exact checkpoint metadata compatibility from the architecture API alone.
If preserving legacy author checkpoint loading requires PyTorch's documented
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` setting, bind that choice before the new gate;
never patch a failed frozen loader or dynamically retry with different loading
semantics.
[TabPFN checkpoint loading, lines 67–73](https://github.com/PriorLabs/TabPFN/blob/f02b3b5941717472eafd078c15ae2043676b3fcf/src/tabpfn/checkpoint.py#L67),
[Torch loading semantics, lines 1479–1509](https://github.com/pytorch/pytorch/blob/v2.15.0/torch/serialization.py#L1479).
