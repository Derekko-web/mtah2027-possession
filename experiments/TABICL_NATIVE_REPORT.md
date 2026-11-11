# Round 54: native TabICL ownership rejected

The published TabICLv2 classifier was applied using its default eight-member
ensemble, fixed166 original/pretrained features and train-only5Hz live context.
The same imputer, symmetric team augmentation/query averaging, original live/D
mass and selected8 decoder were used in every held-out fold. R53's separately
frozen serial reconstruction prerequisite passed before native processing.
See [plan](TABICL_NATIVE_CONTEXT_PLAN.json) and
[independent source review](TABICL_NATIVE_INDEPENDENT_REVIEW.md).

| Protocol | Selected8 | TabICL |
|---|---:|---:|
| Buffered minutes | .741176 | .694118 |
| Purged spells | .685714 | .611111 |
| Buffered100seconds | .670270 | .651429 |

All three primary and mean-fold scores regress. Each protocol loses a prior
matched possession and reduces short matches from3 to2. Buffered/purged macroF1
also fail the tolerance; D/X remain exact. The strict historical .754286 record
is not exceeded. The frozen conjunction rejects this interface. No view-count,
context, feature, temperature, backend or decoder rescue follows, and no
confirmation/full candidate is authorized by these outcomes.

All13 workers finish without resource aborts. Maximum sampled worker/combined
RSS are4.551/5.925GiB; the worker's own high-water record reaches4.786GiB. Every
fit/predict operation and worker wall time passes the respective300/900second
limits. Source/input/protected-artifact hashes, saved caches and official/shared
accounting verify. Six score rows append without changing the previous1,576
rows/bytes. Selected8 and its source links remain exact.

Independent fresh-context replay is attempted and also fails a resource gate.
Buffered blocks1/2/3 rebuild with byte-exact cache tensors and original/swapped/
averaged probabilities. Block4's context fit exceeds300seconds and the worker
is terminated (307.27s total wall, exit-15). No completed block4 probability,
all13 replay or replay manifest exists. The controller exits1; no process is
still running. The sampled RSS remains below6GiB. No cause for the timing
increase is established, and no retry or time-limit waiver is used.

Both failures and every saved context/query/probability artifact are retained
in `data/control_tabicl_native_v1/`, with `closeout.json`. The native screen's
recorded outcomes and hashes still verify; a partial exact replay does not prove
complete reproducibility. Separately frozen R55 is also rejected; R56's CPU
synthetic gate fails its full-size prediction time limit. The user's higher-score
objective remains unachieved.

Primary source: [TabICLv2 paper](https://arxiv.org/html/2602.11139v1).
These are adaptively reused public development results on17possessions, not
private/full-match accuracy or evidence of an optimum.
