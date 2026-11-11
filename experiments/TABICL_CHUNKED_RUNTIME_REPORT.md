# Round48: fixed CPU stage chunks pass synthetic feasibility

Completed 28 October 2026. The separately frozen CPU execution adapter passes
the unchanged 15,000-training-row / 166-feature / 1,500-query resource test.
Round45's unchunked CPU configuration remains rejected. No competition
observations, images, labels, possession fits, scores or candidate changes occur.

## Verified result

| Synthetic stage | Prediction seconds | Worker peak GiB | Result |
|---|---:|---:|---|
| Small: 64 training ×8 features, 16 queries | .0520 / .0515 | .5364 | Original-versus-chunked column, row and probability outputs are byte-identical; chunked repeat exact |
| Medium: 512 training ×32 features, 64 queries | .5078 / .5153 | .7902 | Both partial chunk boundaries exercised; all stage/probability comparisons byte-identical; repeat exact |
| Independent small estimator/model reload | .0553 | .5247 | Output byte-identical to first chunked small prediction |
| Full: 15,000 training ×166 features, 1,500 queries | 98.0940 / 97.5869 | 2.9488 | Both predictions complete within 300 seconds and six GiB; full outputs byte-identical |

All outputs are finite, nonnegative normalized three-class probabilities.
All four supervised workers exit zero without an abort. The full worker's
reported peak and largest externally sampled RSS both equal 3,166,289,920 bytes.
The supervisor retains Round45's six-GiB, 300-second-per-prediction and
1,200-second-worker limits; no resource limit is relaxed.

The small and full synthetic inputs reproduce Round45's saved arrays exactly.
Medium inputs follow the same fixed generator with dimensions specified before
any forward. No synthetic accuracy metric is used. The original checkpoint's
391 tensors load exactly, with all model parameters remaining CPU float32.

## What changed

The [source diagnosis](TABICL_MEMORY_RESEARCH.md) establishes that the official
CPU inference manager bypasses its automatic batching and offload path.
The new [adapter](tabicl_chunked_runtime_gate.py) replaces only the column and
row stage manager objects. It calls the original neural forward functions and
the original tensor slicing/forward wrappers with a fixed maximum of eight
column groups and 128 rows per call. The complete output is allocated once.
The model's ICL manager and its entire training-prefix attention context are
untouched. No acquired package file, checkpoint, neural layer, model setting,
precision, feature count or device is changed.

Each full prediction processes 22 column chunks, whose last shape is
`[1,2,16500,128]`, and 129 row chunks, whose last output shape is `[1,116,512]`.
The partial groups/rows remain in their original positions. Invented-tensor
tests also verify multidimensional slicing and expanded target dimensions.

Small/medium probability parity was prospectively allowed max absolute error
`2e-5`, mean error `2e-6` and identical argmax. Column/row comparisons use the
same declared numeric limits. Every observed difference is exactly zero on
these fixtures. This is stronger observed evidence than the specified numeric
gate, but it does **not** establish bitwise equivalence for arbitrary table or
query shapes. Round45's query-prefix numerical discrepancy remains recorded.
The unchanged full-size baseline is not rerun after its known memory failure;
large-baseline equivalence is therefore not measured.

## Provenance and replay

[TABICL_CHUNKED_RUNTIME_PLAN.json](TABICL_CHUNKED_RUNTIME_PLAN.json) and the new
implementation were frozen before any model forward. The adapter's weights,
runtime and dependency authority is the original acquired TabICL2.2.0 wheel
and classifier checkpoint documented in
[TABICL_RUNTIME_REPORT.md](TABICL_RUNTIME_REPORT.md).
The old source, acquisition, installed and read-only dependency hashes still
verify exactly. Network connections are blocked throughout all synthetic
fit, prediction and reload workers.

The adapter reuses Round45's original resource supervisor, changing only its
spawned script path and combined verifier for the duration of supervision.
It does not edit the old script. Four stage summaries and all 30 saved stage
artifacts verify against their recorded hashes. Evidence is under ignored
`data/control_tabicl_chunked_v1/`, including `final_audit.json`, chunk traces,
stage outputs and complete supervision records.

The frozen plan SHA256 is
`ea004afef7f8e9bedd36a84acbbb174e5c10c2c391a1a74ad75c8cd9f654aaff`;
the adapter SHA256 is
`00f4ff8b1c08ce554a391bd5ac530cf3ab3b974df3e782edaee4b2804746632a`.

Verify source/runtime hashes without inference:

```sh
data/control_sports_action_v1/runtime/bin/python -B -c 'import sys; sys.path.insert(0,"experiments"); import tabicl_chunked_runtime_gate as gate; gate.verify()'
```

Reproduce the fixed synthetic gate in a fresh output folder:

```sh
data/control_sports_action_v1/runtime/bin/python -B experiments/tabicl_chunked_runtime_gate.py --output data/control_tabicl_chunked_v1/runtime_assessment_replay
```

This is a technical feasibility pass. A separately frozen native experiment
must still define training-only context, target-weight semantics, query schedule,
missingness behavior, scoring safeguards and reproducibility before any claim
of possession-score improvement or selected-model change.
