# Research: randomized trees versus an open tabular foundation model

Prepared 27 October 2026 after the R43 proposal-projection screen. This note
reviews primary papers, official implementation documentation and public model
metadata. No model is fitted or scored; no package or checkpoint is downloaded
or installed. A read-only import confirms the repository's current research
environment already provides scikit-learn 1.9.1 and ExtraTreesClassifier.

## Recommendation

For the next **immediately feasible small experiment**, use a matched
ExtraTrees102 control and ExtraTrees166 pretrained-latent candidate with the
repository's existing randomized-tree settings. This changes the learning
algorithm from sequentially optimized boosting to independently randomized
partition averaging, while holding inputs, training exclusions, symmetry and
the selected8 decoder fixed. Run no new smoothing, blending or topology variant.

TabICLv2 is a credible, more distinct model family and appears obtainable
without a user agreement. Its published requirements accommodate Torch2.5,
and its API supports CPU and MPS. It needs a separately frozen synthetic
runtime/replay investigation before a native score experiment: the existing
runtime is not evidence that its full workload fits memory or finishes quickly.

| Candidate | Available here | Distinct information/learner | Main limitation |
|---|---|---|---|
| ExtraTrees on102/166 | Installed sklearn1.9.1; no acquisition | Random feature thresholds and probability averaging instead of boosted residual optimization | ExtraTrees has already been tried on older compact/geometry features; only its interaction with the transferred representation is new |
| TabICLv2 | Public package and110MB checkpoint; not installed or run in this research | Synthetic-pretrained transformer performs in-context classification from training examples | Unverified local speed/memory, no sample_weight fit interface, prior mismatch and preprocessing/replay details |

Neither method's literature establishes a possession-score gain. Comparators
must include the relevant historical R37 aggregate high, not only selected8.

## ExtraTrees: source and precise new hypothesis

Geurts, Ernst and Wehenkel's primary paper, read particularly sections2.1,
3.2–3.3 and the bias/variance discussion, selects candidate features randomly,
draws their thresholds randomly, chooses among those splits, and averages an
ensemble grown on the full learning sample. This differs from bootstrap
Random Forests and from boosted trees. Their analysis motivates variance
reduction through randomized aggregation while acknowledging increased bias
and problem-dependent behavior. The experiments concern classification and
regression, not temporal possession matching.
[Author university full paper](https://orbi.uliege.be/bitstream/2268/9357/1/geurts-mlj-advance.pdf),
[Publisher record](https://doi.org/10.1007/s10994-006-6226-1).

The official implementation permits explicit tree count, depth, leaf size,
feature subsampling, class weights and random state; predictions aggregate
per-tree class probabilities.
[ExtraTreesClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html).

The repository has already tested four compact/geometry ExtraTrees/Random
Forest heads, including an anchored variant:
[train_tree_ensembles.py](train_tree_ensembles.py). Therefore this is not a
claim of an unexplored general learner. The narrow question is whether the
same randomized learner handles the64 transferred player-set coordinates
better than R37's boosted split optimization, with a matched102-feature
control to identify whether the representation helps this learner.

### Proposed fixed screen

Use multiclass H/A/D targets and the exact original13 folds across the three
protocols. Both arms receive the same explicit training indices, team-swap
augmentation, class order H=0/A=1/D=2 and symmetrized prediction. The two
feature matrices are original102 and original102+pretrained64. Do not use the
failed random encoder or choose feature subsets from scores.

Reuse historical settings: `n_estimators=200`, `max_depth=8`,
`min_samples_leaf=20`, `max_features=.8`, `class_weight=balanced`,
`bootstrap=False`, `criterion=gini`, `random_state=42`, `n_jobs=2`.
Fit the training-only median imputer with missing indicators and retained
empty features, as in the historical script. Save the imputer and classifier
together and verify transformed-column order on reload. These depth/leaf
restrictions are project choices, not the primary paper's fully grown defaults.
No early stopping, hyperparameter search or learned blend is proposed.

Replace only slot0 in the unchanged selected8 inference and apply its existing
live-only overlay against a separately reproduced selected8 reference. Freeze
unsupported-latent fallback before fitting. Check class order, probability
validity, source/fold provenance, team symmetry, saved-model probability
replay and exact reference D/X preservation before score interpretation.
Report all protocols and secondary protections, including any losses.

This is26 new seed42 fits. The historical200-tree implementation and installed
runtime make it inexpensive relative to a new foundation-model installation,
but elapsed time has not been measured here. Confirmation/full-match work
must follow a successful prospective screen, rather than testing a parameter
grid after an adverse outcome.

## TabICLv2: acquisition and runtime evidence

The official repository currently identifies **TabICL2.2.0**, and PyPI's
release metadata records its September2,2026 release. Its pure-Python wheel
is310.4kB. The official `pyproject.toml` requires Python>=3.10,
Torch>=2.2, scikit-learn>=1.3, NumPy, SciPy, einops>=.7, psutil,
tqdm>=4.64, huggingface-hub and tokenizers>=.20. The existing Torch2.5 version
satisfies the declared Torch floor; complete dependency resolution and runtime
compatibility remain untested.
[Official package metadata](https://pypi.org/project/tabicl/),
[Official dependency declarations](https://raw.githubusercontent.com/soda-inria/tabicl/main/pyproject.toml),
[Official version source](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/__about__.py).

The code declares BSD3-Clause, and the author's public Hugging Face model card
also declares bsd-3-clause. The public file browser exposes
`tabicl-classifier-v2-20260212.ckpt`, size110MB, without a visible access gate
or acceptance flow. No binary was requested. This establishes a concrete
public source for a frozen acquisition plan; it does not certify a downloaded
file's checksum before acquisition.
[Code license](https://raw.githubusercontent.com/soda-inria/tabicl/main/LICENSE),
[Author model card](https://huggingface.co/jingang/TabICL),
[Checkpoint metadata](https://huggingface.co/jingang/TabICL/blob/main/tabicl-classifier-v2-20260212.ckpt).

The official API explicitly lists CPU and MPS devices, with automatic
selection CUDA→XPU→MPS→CPU. CPU automatic mixed precision remains off;
FlashAttention3 is off on CPU/MPS. `batch_size` controls simultaneous ensemble
members, **not** query rows. The documented default is8 views. Caching may
retain training representations or fuller KV tensors. This provides a route
to CPU inference without CUDA, not a benchmark for this machine.
[Official API](https://tabicl.readthedocs.io/en/latest/api.html),
[Classifier source](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/classifier.py).

The v2 paper, read particularly architecture/inference and limitations,
pretrains on synthetic data, separates column, row and dataset processing,
and evaluates strong tabular benchmarks. It also leaves distribution shift
handling to future work and notes mean imputation for missing values. Its
benchmark results are not evidence on correlated tracking frames or repeated
same-match possession validation.
[Primary v2 paper](https://arxiv.org/html/2602.11139).

### What a later bounded feasibility plan should freeze

Pin package2.2.0, the single v2 classifier checkpoint, source/model hashes
after acquisition, and the existing isolated Torch2.5 runtime. Use a dedicated
environment; do not replace the working research environment. Fix explicit
CPU fp32, two threads and ensemble batch_size1 for a reproducibility-first
synthetic gate, unless the plan separately names MPS as its one intended
device. Avoid choosing the device/precision from possession scores.

Benchmark both a small synthetic classification case and a synthetic shape
matching the largest prospective augmented fold and query batch. Measure peak
memory and elapsed time, repeated forward/save-load replay, class ordering,
team symmetry and any query-batch/prefix dependence. The documented column
embedding tensor has shape `(ensemble_batch, rows, columns,128)` for the
published embedding width. For12,000 rows and166 columns, float32 storage alone
is about.95GiB per simultaneous view, before attention workspace, checkpoint
or caches; eight simultaneous views multiply that intermediate by8. This is
an arithmetic estimate, not observed peak memory.

`TabICLClassifier.fit` accepts X and y without a sample_weight argument.
A native experiment therefore cannot silently replace current class-balanced
fitting by an unweighted context set. One possible separately frozen interface
is conditional H/A context with team augmentation (which balances classes
exactly), composed with original D mass, plus its matched102-feature control.
That is a new architecture test using the same conditional interface as R42,
not a claim that R42's failed conditional CatBoost route passed. Alternatively,
a joint-class context sampling rule needs explicit training-only definition
and a matched control. Do not choose the interface after comparing scores.

Use only eligible training labels in the context; freeze missingness handling,
preprocessing, ensemble views, source checkpoint, train/context selection and
query schedule before native evaluation. There is no evidence here that
8views, a single view, MPS or CPU provides the best score. A runtime gate must
precede native processing, and a failed native screen must remain a failure.

All current private-label exclusions and original candidate protections remain
applicable. More expressive tabular learning does not create new independent
possessions, remove public selection bias or establish an optimum.
