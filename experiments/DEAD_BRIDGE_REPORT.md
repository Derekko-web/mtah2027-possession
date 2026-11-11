# Round 31: automatic short-stoppage bridge

Completed 2026-10-25. **No improvement; selected8 stays selected.** The fixed
rule accepts no D island in any of the three public protocols. No model is fit,
no confirmation or full candidate follows, and no parameter is changed after
seeing results.

The [prospective plan](DEAD_BRIDGE_PLAN.json) was frozen while the independent
Round30 temporal model was fitting, before any bridge proposal count or score.
It uses only supplied tactical tracking, automatic restart intervals and the
existing public-only fast head. It does not use the temporal trial, manual
annotations, external labels, private targets or video interpretations.

Starting from complete selected8 inference, the rule fills a D run only when it
lasts at most63 frames, has13 consecutive same-team live frames on each side,
stays inside a validation block, contains no automatic restart, has at least50%
ball observations and three close/exclusive moving-ball contacts for that team,
has no opposing contact and has mean fast-head agreement at least.65. All
proposals use the original sequence so repairs cannot cascade.

| Protocol | D runs tested | Accepted | Changed frames | Reference and bridge spell F1 |
|---|---:|---:|---:|---:|
| Buffered minutes | 4 | 0 | 0 | .741176 |
| Purged spells | 6 | 0 | 0 | .685714 |
| Buffered100seconds | 4 | 0 | 0 | .670270 |

Macro F1, D recall, false D frames, short matches and all other scores remain
identical. The prospective+.01 main-protocol gain fails. The historical+.04
gain and additional-short-match goals also remain unmet. These are measurements
on the same17 repeatedly used public possessions, with no private-score claim.

All five targeted synthetic tests pass: supported repair, protected ambiguous
cases, block boundaries, team symmetry and long/unknown preservation. The full
selected8 identity control, shared inference and official evaluator agree.
A fresh process reproduces the six score rows, fold/segment diagnostics,
official matching counts, selection and proposal counts byte-for-byte. The
source/plan/input hashes,17 protected selected artifacts and both source links
verify. A duplicate run correctly refuses to overwrite the completed directory;
the fresh replay uses its own directory.

Six dependent score rows append to the ledger, preserving all1,483 preceding
rows and bytes:1,489 total. Idempotent append and preceding-byte preservation
were checked against a separate local ledger copy. Inference traces, public
rows and probabilities stay in ignored local data. No candidate is built or
selected from this rejected trial.

Plan SHA256:
`c4f696ffaaca56f8b893cf1623debb12e70f7a2806d89b2366975641fb66bb35`.
Local evidence: `data/control_dead_bridge_v1/`.
