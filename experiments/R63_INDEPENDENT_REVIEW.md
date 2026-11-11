# R63 independent prepared-wrapper review

**APPROVED for root freeze and the unchanged four-stage invented gate, at the exact hashes below. No blocking issue found.** This is source/pure-sentinel approval, not a model/runtime pass. Root retains authority to freeze and schedule the gate.

| Artifact | SHA-256 |
|---|---|
| `experiments/tabpfn_mps_validation_init_gate.py` | `59823fca874ead58dbd1b8684b17647d2dea61cff53523466b70dde05657494f` |
| `experiments/TABPFN_MPS_VALIDATION_INIT_PLAN.json` | `5c8ecead817a74645218b1ea6455d481d4ec0da9290f6fd0a7d92c86c078e6fe` |

## Source-defined change

The reviewer independently read the complete R63 wrapper/plan, frozen R60 raw-fingerprint implementation, frozen R57 worker/supervisor/numerical functions, R59 loader-composite binding, R62 trace findings and the retained exact official categorical/base preprocessing sources.

The source proof checks two files at official commit `f02b3b5941717472eafd078c15ae2043676b3fcf`, each against retained, original and read-only repaired-runtime copies. Their hashes match. `EncodeCategoricalFeaturesStep.fit_transform` sets its fitted output schema and bypasses base `_validate_added_data`; both `_transform` return sites (lines 430–447) return `None,None` for the added array/modality. Reading all branches confirms this covers numeric/none, no-category identity, ordinal, ordinal-shuffled, one-hot and the large-one-hot identity fallback. One-hot output expansion is already represented in transformed data and its fitted output schema; it does not return a separate added array.

`PreprocessingStep._validate_added_data` (lines 304–326) with `(None,None)` writes only `n_added_columns_=None` and `modality_added_=None` on its first call. Later calls check those values. It does not fit, transform, alter output schema, consume data, advance RNG or change numerical parameters. The R63 AST source contract independently passes: exactly two categorical returns with `None,None`, exactly those two validator assignment fields, and no base-validation call in the categorical `fit_transform` override.

R63 invokes that exact inherited author method directly, once per selected member at its one-time initialization boundary. It requires exactly the author categorical class, exactly one categorical step in each of eight retained members, the unoverridden inherited method, a present/non-None fitted output schema and present fitted-transformer marker. Identity `categorical_transformer_=None` is valid; a non-None transformer must pass `check_is_fitted`. Both metadata fields must be absent or both already None. Asymmetric presence, any non-None value, missing fitted state, subclass/instance/class overrides and unrelated mutations fail. All pre-existing attribute raw hashes must remain unchanged; the only possible new keys are the two allowed None fields. Partial failures are recorded and propagated, without retry or restoration.

## Lifecycle and complete immutability

The actual inherited R57 worker calls `preprocessing_fingerprint` only after context fitting and its initial exact checkpoint comparison, before the first prediction. R63 replaces only that entry point with a boundary that initializes at its first invocation and immediately delegates to the original R60 fingerprint. Model identity and boundary state live outside the estimator. All subsequent fingerprint calls delegate without initialization; a second model in the same worker fails. Each separate first/rebuild process gets its own fresh boundary.

There is no query warmup, transform call, post-query reset, field removal, fingerprint normalization, hash exclusion, tolerance or cache exemption. R60's original complete raw CPU-pipeline joblib hashes include both newly initialized fields, and its original GPU-pipeline/input-preprocessor/context/configuration/schema/feature-index/seed checks remain intact. Later deletion or mutation is exposed by the unchanged equality assertion. An additional reviewer pure lifecycle sentinel deleted one field after the baseline: the next fingerprint differed and the field remained absent, proving the boundary did not silently repair it.

The initialization record covers source proof, all eight pipeline hashes before/after, per-member/step identities, field presence/None values, whole-step hashes, every existing attribute hash, identity/nonidentity fitted checks and partial failure details. The initialization executes while the original live memory monitor and worker deadline are active. No separate trace observer from R62 is installed.

## Unchanged numerical/runtime gate

The reviewer loaded definition-only wrappers and independently compared the configured core's Python code objects and defining source paths with a separately loaded original R57 module: **27 functions/methods are identical**, including `worker`, `main`, `supervise`, `operation`, `canonical_predict`, `checkpoint_fingerprint`, `synthetic_data`, memory monitor and native-dispatch guard. Exact limits, fixed environment, interpreter and checkpoint constants match. The original R60 fingerprint is bound by its frozen source; R63 only wraps its first invocation. Execution changes are confined to new round/plan/spawn identity, verification binding and that initialization callback.

The original gate still requires, sequentially, moderate64×166 first/rebuild then full3000×166 first/rebuild, with all four records passing. Moderate uses 513 queries and full uses 512. Context pairing, H/A-style feature-swap involution, train-only median/empty-zero imputation, canonical 512 padding, original/swapped symmetry, canonical repeat, noncanonical diagnostics, exact full cache and checkpoint invariance, byte-identical probabilities/recipes and fresh-process reconstruction remain unchanged.

The same trusted anonymous V2 checkpoint, author default preprocessing, eight views, temperature 0.9, probability averaging, seed42, official MPS wrapper, MPS float32, disabled fallback/autocast/compile, `fit_preprocessors`, memory saving and one preprocessing job remain required. Existing dependency versions and R59 loader-only composite are reused without modification. Hardware-probe bytes match R60 exactly, including pre-import memory fraction `0.2533781302073466` and 3 GiB MPS-driver allocation budget. Conservative owned RSS plus full driver memory remains ≤6 GiB, every original operation ≤300 seconds, every worker ≤900 seconds, with live monitor and serial workers. No model/data/backend/resource fallback or changed setting is introduced.

## Independent verification

The reviewer reran only definition-level verification and pure mock tests, without invoking preflight's artifact writer, freeze, worker or model. All **16 pure sentinels pass** using the AST-compiled exact author validator body. They cover absent/existing-None metadata, valid fitted transformer, unfitted/missing fitted state, non-None/asymmetric fields, exact-class and override rejection, unrelated fitted mutation and a single initialization across three original fingerprints with second-model rejection. The extra post-baseline deletion sentinel also passes. `torch`, `numpy`, `scipy`, `sklearn`, `tabpfn` and `joblib` remained absent from `sys.modules` throughout this independent check.

The frozen verification chain rehashed and verified the previous rounds and R59 composite symlinks:

| Preserved round | Frozen implementation bindings | Frozen plan bytes match |
|---|---:|---|
| R57 | 25,257 | yes |
| R59 | 2,172 | yes |
| R60 | 20 | yes |
| R62 | 22 | yes |

R60's moderate/first failure remains exit1 at 49.433634 seconds. R62 remains the failed trace-only moderate/first diagnostic, exit1 at 59.605048 seconds, with no full-context/rebuild/post-query-weights pass. Its evidence establishes only that the traced workload's first query lazily added sixteen None-valued fields in eight categorical steps; it did not establish complete runtime support. R63 neither revises those outcomes nor declares them passed.

## Scope limits

This review created only this new repository report; it did not alter wrapper, plan, dependencies, prior runs or frozen evidence. No scientific package/model import, forward call, data transformation, competition/native input read, performance scoring, freeze, upload or commit occurred. The prepared source is eligible for root to freeze and run the full original invented gate. Any new failure remains a failure with all evidence retained. Even an eventual synthetic pass permits no native score or promotion until a separately reviewed native plan exists.
