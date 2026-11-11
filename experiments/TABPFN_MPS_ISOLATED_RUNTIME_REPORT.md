# R70v2 isolated TabPFN technical gate

All four invented-data stages passed with terminal exit0. This is technical admission only; no native possession score or selected-model change occurred.

| Context | Run | Wall seconds | Peak owned RSS + full driver GiB |
|---|---|---:|---:|
| moderate | first | 68.372 | 1.783 |
| moderate | rebuild | 62.711 | 1.849 |
| full | first | 365.242 | 3.915 |
| full | rebuild | 362.729 | 4.265 |

Both contexts independently rebuilt exact canonical probabilities, imputation data, checkpoint tensors, full raw fitted preprocessing/context/configuration and initialization metadata. Every recorded operation was within300seconds, every worker within900seconds, and combined owned RSS + FULL MPS driver below6GiB (allocator4GiB plus planning RSS reserve2GiB). Exact source-supported cache release is retained before every prediction; no fitted state, context, weights or RNG is reset by it. Original8 V2 views/f32/512 schedule and failed-round evidence are unchanged.

The cross-runtime audit also compares every retained numeric array/checkpoint/raw-preprocessing/initialization file to historical R67: all exact. Recipe JSON differs in exactly its new ROUND-specific unused model-cache directory; all remaining fields are equal. Recipe bytes are exact between independent first/rebuild runs within this new protocol. Old strict runtime verification remains failed because its shared executable changed outside this work; no old hash or verifier was rewritten.

The first R70 setup attempt failed before any model because its snapshot included its own changing console log. That setup remains preserved; v2 changes only new source/plan/output identities, with identical function ASTs. Console logs for v2 are outside the bound round.

Current sources/runtime maps remain frozen. A separately reviewed native protocol must still bind this actual evidence and pass its own preflight/freeze before any released-label modeling. A full technical pass establishes neither a higher possession score nor private/full-match performance.

Actual evidence: `data/control_tabpfn_mps_isolated_runtime_v2/runtime_assessment/summary.json` SHA256 `5ad010c56855e2ec9fad2b4690046b240825ca206291ab592421bd38f210022e`; cross-runtime audit SHA256 `e238a94dac95f0498f7bbbb380fb9e9fcb34a4dd7a14750e9d226f74383d1cd9`. Built with PriorLabs-TabPFN.
