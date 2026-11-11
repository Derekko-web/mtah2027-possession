# Round56: fixed CPU TabPFN-v2 gate fails the time limit

The separately frozen official2.2.1/v2 eight-view CPUfloat32 interface was
tested only on invented data. All acquisition/dependency/source hashes verify;
existing scientific runtimes remain untouched. See
[plan](TABPFN_RUNTIME_PLAN.json) and
[primary-source research](TABPFN_ALTERNATIVE_RESEARCH_20261028.md).

The64x166 moderate context passes both first and independent fresh-process
reconstruction stages. Canonical512 original/swapped/averaged/repeat arrays,
fitted preprocessors and checkpoint tensors are exact. Wall times383.85/419.79s
and maximum sampled owned RSS2.127/2.217GiB pass their gates. Per-call prediction
times are58.9–76.0s. The deliberately noncanonical8-row prefix differs by
1.78814e-7; that diagnostic was prospectively excluded from selection. Canonical
shapes remain mandatory, and no tolerance substitutes for exact replay.

At the fixed3000x166 context, the first512-row original prediction exceeds the
300-second operation limit. The supervisor terminates the worker at307.02s
total wall time (exit-15), with2.211GiB sampled RSS. Its unfinished operation and
all logs are retained; no completed full-stage probability or summary exists.
The overall gate is failed. No native competition inputs, labels, feature
loader, possession score, alternate setting or cache mode was used. No ledger
score rows are added.

Evidence: `data/control_tabpfn_v1/runtime_assessment/summary.json`, its two
passing moderate stages, and `full/first.supervision.json`. The frozen-input
verifier is independently rerun after terminal failure. A separately researched
current official MPS interface requires a new plan/gate; it does not replace or
reinterpret this failure. The user's higher-score objective remains unachieved.
