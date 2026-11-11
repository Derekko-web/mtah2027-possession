# Round 49: cached-object reconstruction fails the resource gate

The original, unmodified TabICL2.2.0 eight-member cached CPU path was tested
before any competition data. Small/medium cached-versus-uncached numerical
parity, repeated predictions, full-cache reload and compact-object regeneration
passed. The largest fresh fit also passed, but independent reconstruction from
its serialized fitted object exceeded the prospective memory limit. Therefore
the complete technical protocol is rejected and does not authorize native work.

At 3,000 synthetic context rows ×166 features and512 queries, the original
cached fit took110.657seconds; predictions took14.409/14.326seconds and were
byte-identical. Its visible eight-member caches contain1,714,421,760bytes.
The measured worker peak was5,490,802,688bytes, within6GiB. The context-only
compact object occupied6,024,203bytes. All layer, member, context and checkpoint
checks passed for that fresh fit.

The independent largest compact-object regeneration was terminated after
44.605seconds at a sampled6,445,645,824bytes, above6GiB. It produced no completed
probability output or final resource/repeat proof. The memory ceiling is not
relaxed and the failed stage is not described as passing.

Source inspection and a small executable witness also identify a persistence
issue: the official `save_kv_cache=False` branch does not remove an existing
`model_kv_cache_` from the copied state. A separately declared narrow storage
wrapper temporarily binds that attribute toNone during official serialization,
then restores the same live object and verifies its tensor hashes. Acquired
package code, checkpoint weights and cache computation remain unchanged.
This storage wrapper passes the small/medium proofs but does not rescue the
largest reconstruction failure.

The primary implementation is the pinned acquired
[official base class](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/base.py)
and [classifier](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/classifier.py).
See [the frozen plan](TABICL_CACHED_RUNTIME_PLAN.json). Complete evidence and
supervised terminal statuses are under ignored `data/control_tabicl_cached_v1/`.
No possession fitting, scores, candidate changes, private/manual labels,
dependency mutation, uploads or submissions occur.
