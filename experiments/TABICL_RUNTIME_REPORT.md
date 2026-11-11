# Round45: TabICLv2 synthetic CPU feasibility rejected

Completed 27 October 2026. The fixed CPU configuration fails its prospective
memory gate on the required full-size synthetic workload. No competition
observations, labels, manual annotations, possession fits, scores or candidate
changes are produced. The successful small forward does not rescue the failed
resource test or authorize a native trial.

## Result

| Synthetic workload | Result | Timing | Memory |
|---|---|---|---|
| 64 training rows ×8 features,16 queries | Valid normalized3-class probabilities; repeated outputs byte-identical | First/repeat .0610/.0498seconds | Worker-reported peak about.546GiB |
| Independent process, saved small estimator/model reload | Probabilities byte-identical to first small output | .0583seconds | About.527GiB |
| 15,000 training rows ×166 features,1,500 queries | **Rejected during first prediction**; no output completed | Supervisor terminated worker at25.99seconds total wall time | Sampled RSS reached6,975,193,088bytes, about6.50GiB, exceeding6GiB |

The supervisor sent SIGTERM and verified terminal exit−15 for the large worker;
the overall gate exits1. The observed6.50GiB is the largest sampled RSS, not a
complete measurement of the killed worker's instantaneous peak. The300-second
prediction requirement and large-output repeat parity are unestablished,
because the first prediction never completed. No data-size, device, precision,
cache, architecture or checkpoint fallback is run.

## Frozen configuration and public acquisition

[TABICL_RUNTIME_PLAN.json](TABICL_RUNTIME_PLAN.json) fixes the synthetic-only
stage and its6GiB/300-second gates. Its SHA256 is
`daf14abe3f22912123e998ac1628f398e95aa17e3590baaffcd06f336fad5e57`.
The implementation
[tabicl_runtime_gate.py](tabicl_runtime_gate.py) is frozen before any model
forward, SHA256
`d907f7321481e2ec4184aef6514e69b00b6e96dd6305fe9c631923c59642b24a`.

Only the official TabICL2.2.0 wheel, einops.8.1, psutil7.0.0 and
tokenizers.22.0 wheels are installed into the **new** round-local target
directory. Wheel bytes match the corresponding official PyPI release hashes.
The existing sports runtime supplies Torch2.5.0 and NumPy2.1.2. Missing
scientific dependencies are loaded read-only from the existing research
environment: SciPy1.18.1 and scikit-learn1.9.1, with their existing supporting
packages. Neither existing environment is changed or upgraded. Code and
distribution metadata for read-only dependencies are hashed and verified.

The author-hosted classifier checkpoint is pinned to public revision
`4dcd344ece2c00be9e831fdd35bed57b5ad83e19` in
[jingang/TabICL](https://huggingface.co/jingang/TabICL).
`tabicl-classifier-v2-20260212.ckpt` has110,368,038bytes and SHA256
`bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
The author API's LFS digest and length both agree with the actual file.
The original model card declares bsd-3-clause; the wheel's complete BSD3/forecast
Apache notices are retained. No login, gated agreement or hosted inference is
used. The plan's auxiliary general-forecast license URL was not used as the
checkpoint authority; acquisition records identify the canonical author card,
revision and wheel license explicitly.

Runtime is explicitCPU, float32, two threads, one ensemble member, ensemble
batch_size1, random_state42, use_ampFalse and use_fa3False. Other public API
defaults remain fixed and are recorded. The small checkpoint load uses the
unmodified official weights-only loader, then independently compares every
one of391state tensors. All27,552,258parameters are CPU float32 and all
checkpoint tensors match exactly. Network connections are blocked during
every synthetic fit/prediction/reload.

The two synthetic datasets use NumPy PCG64 seed42, Gaussian float32 features,
deterministic flat-index modulo37 missingness and labels cycling0/1/2.
Inputs, targets and completed probabilities are retained and hashed. Targets
are synthetic only; no synthetic accuracy score is computed. A TabICL `fit`
prepares the context/preprocessing and loads the frozen model rather than
optimizing its neural weights.

## Query independence: architecture and observed numerical limit

The official column embedding defaults `embed_with_test=False`; inducing
points attend only to training rows. The in-context transformer also restricts
key/value context to the training prefix, while each query has its own row
processing. The inspected implementation is in the acquired `embedding.py`,
`encoders.py` and `layers.py`, fixed by the dependency manifest.
[Official classifier source](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_sklearn/classifier.py),
[Official model source](https://raw.githubusercontent.com/soda-inria/tabicl/main/src/tabicl/_model/tabicl.py).

On the small synthetic witness, reversing query order and restoring it yields
byte-identical probabilities. Predicting only the first8queries, rather than
all16, differs from the original prefix by at most9.238719940185547e−7.
Thus **byte-exact query-prefix independence is not demonstrated**, despite
the training-only attention context. It would be incorrect to claim arbitrary
query batching reproduces a larger inference call exactly. This was a retained
diagnostic, not an added gate or a reason to change precision or query schedule.
The public API's all-NaN query-column masking is another reason a later native
protocol would need to specify and audit query/missingness behavior.

## Preserved evidence and reproduction

Ignored evidence is under `data/control_tabicl_v1/`:

- `acquisition.json`, authoritative PyPI/Hugging Face metadata, public wheel
  files, checkpoint, model card, license and installation logs;
- `protocol_freeze/` plan, implementation/acquisition hashes, target-file
  hashes and read-only dependency hashes;
- `runtime_assessment/` small inputs/outputs, estimator, independent reload,
  large synthetic inputs, exact stdout/stderr, resource supervision records
  and final failed summary;
- dispatch output retaining the memory abort and terminal status.

The ordinary verification passes after the rejected gate: all frozen acquired
files, target dependencies and read-only runtime code hashes remain exact.
Every owned model/supervisor process is terminal. Acquisition/setup and both
small workers succeeded; the large worker stopped for the prospective memory
violation. No source-math repair or experimental-setting change occurred.

To verify frozen source/runtime bytes without model inference:

```sh
data/control_sports_action_v1/runtime/bin/python -B -c 'import sys; sys.path.insert(0,"experiments"); import tabicl_runtime_gate as gate; gate.verify()'
```

The saved attempt can be inspected directly. For a deliberate same-protocol
reproduction in a fresh output directory, the command is:

```sh
data/control_sports_action_v1/runtime/bin/python -B experiments/tabicl_runtime_gate.py --output data/control_tabicl_v1/runtime_assessment_replay
```

No full-size repeat is run after the failure. The current fixed configuration
is rejected, with no possession-score claim, native protocol, model promotion
or implied private-match improvement.
