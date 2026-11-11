# R48 independent source and protocol review

Reviewed 28 October 2026. No actionable correctness or leakage finding in the
fixed CPU, float32, one-member, `kv_cache=False` path. This review reads source
and saved evidence; it runs hash verification only, with no new model inference,
native data processing, fitting or score computation. The completed large
synthetic run's input/output/log hashes, exact repeat and resource/supervision
proof now also verify. The fixed R48 technical gate passes.

## Chunking preserves the specified computation

The new [adapter](tabicl_chunked_runtime_gate.py) replaces only the column and
row inference managers. The acquired package remains unchanged. In the column
stage, the leading batch dimensions are table and feature-group dimensions;
each chunk retains the complete row sequence, complete training labels and
unchanged `train_size`. In the row stage, the leading dimensions are table and
row; each chunk retains every feature/class token for that row. These are
independent computations within their respective stages. The adapter calls
the original callbacks, `_run_forward` and multidimensional tensor slicer,
then places every chunk in one complete output tensor.

The acquired `embedding.py` passes `embed_with_test=False` and the original
training size into the shared column transformer. The acquired
`interaction.py` aggregates features within each row. The ICL predictor and
its inference manager are untouched, and the original transformer still
restricts key/value context to the training prefix. No training-row subsampling,
context splitting, attention-mask change, dtype change or weight update occurs.

The ignored `output_repeat` argument is an output-memory-estimation multiplier
in the original manager, not an instruction to repeat network results; its
omission does not change this pinned computation. Original store/use-cache
accumulation is not reproduced by the adapter and is outside this review's
scope. Keep `kv_cache=False`; enabling cache modes would require a separate
implementation and validation protocol.

Primary implementation references:
[inference manager](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/inference.py),
[column embedding](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/embedding.py),
[row interaction](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/interaction.py),
[model composition](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/tabicl.py).
The review uses the acquired 2.2.0 bytes fixed by the local manifests, rather
than assuming that these moving main-branch URLs retain the same contents.

## Verified evidence and remaining technical gate

The R48 plan/adapter hashes and all acquired, installed and read-only R45
dependency hashes verify. Saved artifact hashes in the completed small, medium
and independent reload stages also verify. Small and medium baseline versus
chunked column tensors, row tensors and probabilities are byte-identical;
both chunked repeats and the independent small estimator/model reload are
byte-identical. The medium case exercises partial column and row chunks.

Completed worker peaks are 575,963,136 bytes (small), 848,510,976 bytes (medium)
and 563,429,376 bytes (reload). The full 15,000-by-166 context and 1,500-query
workload completes in 98.094 and 97.587 seconds. Both saved probability arrays
are byte-identical, finite, nonnegative and normalized with shape `(1500, 3)`.
Full input arrays reproduce R45's saved workload exactly, including NaNs.

The full worker peak and largest sampled RSS both equal 3,166,289,920 bytes
(2.949 GiB), below the unchanged 6-GiB cap. The supervisor reports terminal
exit code zero, no abort and 198.55 seconds total wall time. Independent review
verifies all full-workload artifact hashes, stdout/stderr hashes and the
supervision-to-worker-summary hash binding. Every frozen check passes; no
large unchunked comparison is claimed because R45 never completed one.

The inherited supervisor retains the same 6-GiB process-RSS abort, 300-second
active-prediction abort and 1,200-second worker watchdog. It terminates and waits
for an over-limit worker. Worker peak RSS adds an unsampled-peak check. These
checks establish the specified technical workload, not native prediction value.

The fixed CPU callback path and local checkpoint do not invoke hosted inference;
workers disable downloads, use offline environment flags and reject socket
connections. The protocol does not turn R45's failed unchunked run into a pass.

## Conditions for a later native ownership protocol

- Fit the external median/empty-feature policy on the permitted training rows
  only, retain the fixed 166-column schema and save its statistics. Internal
  unique-feature filtering, normalization and outlier statistics are also fitted
  on the context passed to `fit`, so outer validation labels/rows must stay out
  of that context and all fitted preprocessing.
- Freeze the global five-Hz training schedule, chronological row ordering,
  live-label eligibility, H/A augmentation, binary class mapping and any source
  support fallback before processing. A TabICL `fit` supplies an in-context
  training table; unchanged neural weights do not make training-label exclusion
  optional. Uniform sampling and no `sample_weight` are different semantics
  from the current class-balanced XGBoost head and should be explicit.
- Keep the full training context, the reviewed chunk managers and `kv_cache=False`.
  Specify the same canonical 512-query batching, order and last-row padding for
  ordinary, swapped, replay and full-match inference. Arbitrary query-size
  byte equality is not proved; architecture-level query independence does not
  remove floating-point shape effects. Canonical batching controls those effects
  without reading validation labels.
- Apply explicit H/A prediction symmetrization using the same fitted context.
  Mirrored training examples alone do not prove team-swap equivariance. Preserve
  the selected D/X mask through the existing final live overlay, because
  retaining raw D probability alone does not guarantee unchanged decoder labels.

These are prospective requirements for reuse of the R48 uncached adapter, not
approval of an uninspected native implementation. Technical runtime success
establishes neither possession accuracy nor private-match improvement.

## Separate official KV-cache path: source inspection only

A future cache experiment can use the original acquired implementation without
the R48 adapter. `TabICLClassifier._build_kv_cache` obtains the training-only
transformed tables, processes each ensemble member with the full context, saves
its column and ICL cache, then combines member caches along the table dimension
within each normalization method. Prediction slices the corresponding member's
cache before each query forward. The official CPU manager calls the cache-aware
forward directly, so full cache mutation/collection remains intact.

For this checkpoint, feature grouping mode `same` creates one circular-neighbor
group per feature, not one group per three features. With all 166 features
retained, there are 166 feature groups plus four reserved class-token groups.
The three column layers cache K/V projections of 128 inducing summaries for
every group, conditioned on the entire training context. The twelve ICL layers
cache K/V for every training row; they do not select a smaller context. The
target-aware class embeddings are already baked into the matching member's
cached projections.

At 3,000 context rows, an upper-size count assuming all 170 groups survive is:

| Stored fp32 K/V tensors | Per member | Eight members |
|---|---:|---:|
| Column cache, 3 layers, width 128, 128 inducing points | 63.75 MiB | 510 MiB |
| ICL cache, 12 layers, width 512, 3,000 training rows | 140.625 MiB | 1,125 MiB |
| Total cache tensors | 204.375 MiB | 1,635 MiB (1.597 GiB) |

This is not a process-memory bound. Training/member activation buffers,
concatenation while old member caches remain alive, preprocessing, the model,
pickle buffers and simultaneous estimator reloads add to it. Measure the full
cache-building `fit`, query predictions, serialization and an independent-process
reload under a separately frozen resource gate. A 512-query batch does not bound
the memory used while initially building the training caches.

The official persistence code stores `model_kv_cache_` on CPU when
`save_kv_cache=True`, together with fitted transforms, feature/class permutations
and context-shape metadata. Source inspection finds no intended feature-group or
training-row truncation in that path. A synthetic test must nevertheless verify
all 3 column and 12 ICL layer entries, group/member dimensions and full context
length, tensor equality after serialization/reload, and cached-versus-uncached
predictions under frozen numerical tolerances. Cache artifacts contain training
information and must retain the same source exclusions as the original context.

Primary source references for this separate path:
[classifier cache collection/use](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/classifier.py),
[cache structures/concatenation](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/kv_cache.py),
[persistence](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/base.py).
No cache inference, cache feasibility claim or native run is performed by this
review.
