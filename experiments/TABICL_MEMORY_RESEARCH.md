# TabICL CPU memory: source diagnosis and one bounded execution proposal

Prepared 28 October 2026. This is read-only source research and arithmetic,
followed by a proposed synthetic experiment. No model forward, competition
input, target, fitting, scoring, installation or existing-file change occurs.
Round45's failed configuration remains failed.

## Finding: the documented offload settings do not fix CPU execution

The acquired official TabICL 2.2.0 implementation explicitly bypasses batching
on CPU: `InferenceManager.__call__` returns `_run_forward` at local
`_model/inference.py:1297–1298`, before batch planning, output allocation and
offload resolution. Its docstring states this limitation. Therefore setting
`offload_mode="disk"`, a smaller `safety_factor`, or a `COL_CONFIG`/`ROW_CONFIG`
offload value alone cannot bound CPU activation memory. Public estimator
`batch_size=1` already limits the number of ensemble tables; it does not split
the columns or rows inside that table.

This behavior was checked against the acquired wheel and the current
[official inference implementation](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/inference.py).
The precise authority for this experiment is the unchanged acquired wheel,
already hashed by Round45; the online `main` branch is a reference, not an
unversioned replacement. The allowed settings and their semantics are in the
[official configuration source](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/inference_config.py)
and [estimator API](https://tabicl.readthedocs.io/en/latest/api.html).

## One proposed configuration: fixed CPU column/row stage chunks

Keep the exact checkpoint, CPU float32, two threads, one ensemble member,
ensemble batch size one, all preprocessing defaults and complete synthetic
15,000-training-row / 166-feature / 1,500-query workload. Implement a small,
separately frozen execution adapter in a **new file**, leaving the acquired
package and old runtime script untouched:

- Replace only `model_.col_embedder.inference_mgr` with a CPU manager that
  processes at most **eight column groups** per call.
- Replace only `model_.row_interactor.inference_mgr` with a CPU manager that
  processes at most **128 rows** per call.
- Reuse the official `_run_forward`, multidimensional input slicing and
  original forward callbacks; preallocate each complete output once and write
  chunks into their original slices. Keep the output dtype float32.
- Leave `model_.icl_predictor.inference_mgr` and all ICL attention unchanged.
  Do not split the dataset sequence or reduce its training context.

This is an execution adapter, **not an available configuration-only CPU fix**.
The official accelerator manager already splits these leading batch dimensions;
the proposed adapter supplies a deterministic CPU chunk schedule instead of
the CPU bypass. It does not alter weights, activations, attention masks,
feature grouping, target handling or neural operations within a chunk.

The decomposition follows the actual module interfaces. Column embedding
passes `(B, G+C, T, group_size)` to its manager; groups are independent leading
batch items (`embedding.py:580–606`). Row interaction passes
`(B, T, G+C, E)`; rows are independent leading batch items, while attention
still spans every feature token within each row (`interaction.py:236–250`).
The original ICL manager receives the full `(B,T,512)` row representation and
all training targets (`learning.py:326–335`). Splitting that sequence as if it
were independent rows would change the model and is excluded.
See the [official embedding](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/embedding.py),
[row interaction](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/interaction.py),
and [ICL learning code](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/learning.py).

## Memory arithmetic and its limits

The saved checkpoint configuration uses `same` feature grouping, group size
three, embedding width 128 and four reserved CLS tokens. `same` grouping
preserves the number of feature groups. For one table, `T=16,500` and
`G+C=170`:

| Allocation or planning estimate | Size |
|---|---:|
| Complete column output, `16500×170×128×4` bytes | 1,436,160,000 bytes = 1.338 GiB |
| Eight-group column output | 64.45 MiB |
| 128-row interaction input | 10.625 MiB |
| Complete row output, `16500×512×4` bytes | 32.23 MiB |
| All 27,552,258 float32 parameters | 105.10 MiB |
| Official profiler's eight-group column estimate, including input | 790.36 MiB |
| Official profiler's 128-row estimate, including input | 266.07 MiB |
| Official profiler's unchunked 170-group column estimate | 13,894.75 MiB |

The profiler formulas are in `inference.py:63–119`, use float32 coefficients
and report binary MiB despite the source naming its unit MB. Their profiling
does **not** constitute a CPU peak-memory guarantee. Retaining the 1.338-GiB
column output, a sub-GiB column chunk and the roughly 0.55-GiB observed small
runtime suggests an envelope around three GiB before allocator/runtime effects.
The untouched ICL stage, CPU attention backend, transient copies and allocator
retention remain unmeasured at full size. Only an actual supervised resource
run can establish that the **same six-GiB cap** and **300-second prediction
limit** pass. The memory cap should not be increased to rescue Round45.

## Numerical checks required before the large gate

Mathematical batch independence does not prove bitwise equality across different
batch shapes. Round45 already found up to `9.24e-7` probability differences when
changing query length. Chunking may also change CPU floating-point kernel
blocking. Do not promise byte-exact parity with the unchunked baseline.

Before any new forward, freeze one adapter, the fixed 8/128 schedule and its
parity tolerances. Compare against the unmodified runtime on the existing small
fixture and one fixed synthetic medium fixture large enough to exercise both
chunk boundaries. Check all checkpoint tensors remain identical, finite
normalized probabilities, tight declared probability/logit agreement, unchanged
class predictions, and correct final partial chunks. Check stage outputs as
well as final probabilities to localize errors. Require **byte-exact repeats
and independent reloads of the adapter's own execution**.

Only after these checks pass should the unchanged full synthetic workload run,
with the existing six-GiB supervisor and 300-second limits. Keep every failed
attempt; run no chunk-size, device, precision, feature-count or checkpoint sweep.
This proposal establishes no runtime feasibility or possession-score benefit
until those separate checks actually succeed.
