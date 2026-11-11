# Research: a bounded TabPFN-v2 alternative

Prepared 28 October 2026 while the frozen R54 TabICL screen runs. This is
read-only feasibility research. No competition predictions, label comparisons,
checkpoint download, package installation, environment change or inference took
place. Small official package source was read in memory without installation.

## Decision

**Go for a separately frozen synthetic runtime investigation of TabPFN-v2;
no-go for a native score experiment yet.** There is a concrete anonymous author
checkpoint and a supported CPU interface. No source establishes that the
3,000-row,166-feature,8-view workload meets this machine's6GiB and300-second
limits. TabPFN2.5 and newer have a license-acceptance/authentication step in the
official runtime and do not fit this search's no-acceptance constraint. A
publicly accessible file listing does not remove that integration requirement.

TabM was not investigated because the requested fallback condition—no public
accessible TabPFN weights—is false for v2. This recommendation is not evidence
that TabPFN improves possession matching or deserves priority over the running
TabICL experiment.

## Primary scientific evidence and its limits

Hollmann et al.'s Nature paper describes synthetic-prior pretraining and
in-context prediction. Its architecture alternates attention across features
and samples; test samples attend to training samples, not to each other.
Training keys/values can be cached. The paper reports quadratic compute in
sample/feature counts and linear memory in table size, and evaluates up to
10,000 samples and500 features. These ranges include the proposed shape but
are not local CPU timing or RSS measurements. Its cell-attention architecture
and synthetic prior differ from TabICL's column/row/dataset pipeline; the
common ICL family also makes correlated failures plausible.
[Primary paper, architecture and inference methods](https://www.nature.com/articles/s41586-024-08328-6).

I read the first-party indexed article passages on ICL, architecture, inference
preprocessing and training details. Direct article/PDF opening redirected to a
publisher identity endpoint, so this is not a claim that every supplementary
page was retrieved. No third-party summary supports the recommendation.

The sports frames are highly dependent and this search has only17 public
possessions. My inference is that a different learned prior might move
boundaries differently from the rejected tree heads, but neither the paper's
benchmarks nor its Bayesian interpretation establishes calibration under this
tracking-data distribution or gains in the official interval-matching metric.

## Access, license and exact candidate identity

The current official repository defaults to TabPFN3.5. Its pinned9.2.0 loader
calls `ensure_license_accepted` for2.5,2.6,3,3.5 and3.5-fast; v2 is excluded
and retains an official direct-download option. Use an explicit v2 identity;
never allow a changing package default to choose the model.
[Official loader](https://raw.githubusercontent.com/PriorLabs/TabPFN/v9.2.0/src/tabpfn/model_loading.py),
[Official README](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/README.md).

Anonymous metadata and HEAD requests verified the following author-hosted
resource on28October. No binary body was fetched:

| Field | Observed value |
|---|---|
| Repository | `Prior-Labs/TabPFN-v2-clf` |
| Revision | `275f19d2add9c2c0de38f9ebaded3856599fa2e7` |
| API access metadata | `private=false`, `gated=false` |
| Candidate filename | `tabpfn-v2-classifier-finetuned-zk73skhh.ckpt` |
| Bytes advertised and anonymous HEAD |29,009,539; HTTP200 |
| Advertised LFS SHA256 | `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49` |

[Pinned author metadata](https://huggingface.co/api/models/Prior-Labs/TabPFN-v2-clf/tree/275f19d2add9c2c0de38f9ebaded3856599fa2e7),
[Pinned checkpoint](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/275f19d2add9c2c0de38f9ebaded3856599fa2e7/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt).

This is also the classifier default named by official package2.2.1. The author
card says the `finetuned-zk73skhh` and `v2_default` files are identical aliases;
their advertised LFS hashes agree. Do not infer unlisted checkpoint training
provenance from a filename. The checksum above remains advertised metadata,
not a locally recomputed hash until acquisition.
[Author model card](https://huggingface.co/Prior-Labs/TabPFN-v2-clf).

The v2 weights carry Prior Labs License1.1, an Apache2-derived license with
an additional attribution section. Preserve the exact license/notice and
appropriate TabPFN attribution with any derived artifact. This research does
not accept terms on the user's behalf or authorize a publication.
[Pinned weight license](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/275f19d2add9c2c0de38f9ebaded3856599fa2e7/LICENSE.txt).

## Runtime evidence

I inspected the official2.2.1 wheel source (`classifier.py`, `base.py`,
`model_loading.py` and `utils.py`) in memory. It defaults to8 ensemble prompts,
temperature.9 and `fit_preprocessors`; exposes explicit CPU, fp32,
`fit_with_cache` and CPU memory-saving modes; and rejects CPU contexts above
1,000 rows unless the documented override is set. Its fit acceptsX/y without
sample weights. The memory estimator is a mechanism, not a hard RSS guarantee.
The documented override `TABPFN_ALLOW_CPU_LARGE_DATASET=1` can retain separate
pretraining-range checks, unlike broadly ignoring all pretraining limits.
[Official2.2.1 package and source distribution](https://pypi.org/project/tabpfn/2.2.1/),
[Official release JSON](https://pypi.org/pypi/tabpfn/2.2.1/json).

The2.2.1 wheel is173,723bytes, SHA256
`9aa3a0df8ecebcae5f426b1fec9bb2e2a690880bc8ad079191e8d959bf0f157b`.
Its declared `scikit-learn>=1.2,<1.7` conflicts with the current research
environment's1.9.1. It also requires pydantic/settings and common-utils.
Therefore do not insert it into the running environment; a later acquisition
must resolve exact compatible dependencies in a separate target/runtime,
retain manifests, disable telemetry and prohibit network during inference.

Current9.2.0 source requires Torch>=2.5 but brings further dependencies; it is
not automatically a cheaper compatibility choice. Its MPS device checks
require Torch>=2.6. Existing Torch2.5 is consequently not sufficient evidence
for a supported MPS route. Keep CPU as the proposed single backend.
[Current pinned dependencies](https://raw.githubusercontent.com/PriorLabs/TabPFN/v9.2.0/pyproject.toml),
[Current pinned device logic](https://raw.githubusercontent.com/PriorLabs/TabPFN/v9.2.0/src/tabpfn/utils.py).

## Proposed next gate, not an executed experiment

Freeze the2.2.1 package, explicit default-v2 file above, resolved dependencies,
CPUfloat32,8views, n_jobs1, single-thread Torch/BLAS, fixed random state and
query schedule before acquisition/inference. Begin with author defaults for
preprocessing and `fit_preprocessors`, bounded numeric memory-saving budget,
and the documented large-CPU-context override. Do not try cached/uncached,
precision, ensemble-count or context-size variants selected by possession
scores. A different technical interface requires its own retained gate.

Use only invented data: small64x166 and full3,000x166 contexts with balanced
binary labels,512-row query batches and a padded last batch; include constant
and missing-feature cases. Compare repeated calls, prefix/order/padding effects
and independent fresh-context reconstruction. Record all resolved model and
preprocessor parameters, transformed feature sizes (defaults can expand
features), checkpoint/cache identities, CPU time and actual supervised RSS.
Enforce6GiB per worker,300seconds per operation and a bounded wall time.
Perform no compute while the current two-worker native screen owns the budget.

Only a passing gate justifies a native plan. A native method could reuse the
already frozen R54 context schedule, train-only imputation,166 features, team
augmentation/symmetry, original live/D mass, selected decoder and D/X overlay,
thereby changing the learner while preserving those interfaces. Any such
screen must first bind exact source/data hashes, retain all outcomes, exceed
the historical.754286 primary record and pass the existing secondary guards;
confirmations and full CSV reproducibility remain subsequent requirements.
These are proposed project choices, not author-validated sports settings.

## Subsequent authorized technical preparation — R56 frozen, not run

After the research-only stage above, the parent authorized isolated acquisition
and preparation of one synthetic gate. The checkpoint now downloads anonymously
and its actual29,009,539-byte content hashes to the advertised SHA256 above.
The complete author License1.1 was read and retained; section10 explicitly
exempts internal testing from added attribution, while later external model
distribution has attribution and naming conditions. No auth token, browser
acceptance or agreement flow was used.

Thirty-one exact official PyPI wheels (60,997,686bytes total) were verified and
installed with `--no-index --no-deps --no-compile --target` solely into new
`data/control_tabpfn_v1/runtime_deps/`. The untouched sports runtime provides
Torch2.5.0/NumPy2.1.2 and shared packages. All44 distributions in the resolved
dependency closure satisfy declared requirements. The author classifier,
model loading, retained-preprocessing inference engine, memory estimator,
fingerprint feature transform and telemetry/disable/state code were read.

[TABPFN_RUNTIME_PLAN.json](TABPFN_RUNTIME_PLAN.json) and
[tabpfn_runtime_gate.py](tabpfn_runtime_gate.py) implement the fixed gate.
It uses the author's default retained-preprocessing interface, not neural KV
caching; numeric memory-saving budget4GB leaves room beneath the6GiB hard cap.
This is one prospective supported execution choice. No alternate runtime,
precision, view count or context shape is selected from timings or scores.

Independent read-only source review found two pre-freeze orchestration defects:
the telemetry class is a decorated singleton factory, and a resource-aborted
worker must never count as passing. Both fixes were verified before freeze;
the reviewer found no remaining blocker. Actual scientific-package imports,
joblib cache-fingerprint stability, memory/time and numeric repeatability
remain untested.

The final standard-library-only source/import preflight passes in.0548seconds,
without importing Torch, NumPy, TabPFN or sklearn and without any forward.
`protocol_freeze/` binds6,192 source/acquisition/target files and14 read-only
runtime distributions. A subsequent complete frozen-input verification passes.
No heavy synthetic run, native input, score or claim of runtime success exists.
The parent will schedule after its active native CPU workers finish:

```sh
data/control_sports_action_v1/runtime/bin/python -B experiments/tabpfn_runtime_gate.py
```

Frozen-input-only verification, already passed:

```sh
data/control_sports_action_v1/runtime/bin/python -B experiments/tabpfn_runtime_gate.py --verify
```

Both commands run from the repository root; all results stay in the new round.
The first command should be executed once for the prospective assessment,
retaining any failure rather than rerunning into the same directory.
