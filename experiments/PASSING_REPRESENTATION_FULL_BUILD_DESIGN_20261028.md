# Full-match construction design for a passing R54 or R55 method

Prepared 28 October 2026 by an independent implementation reviewer. This is a
technical design only. No encoder forward, feature extraction, model training,
native prediction, scoring or selected-model change was performed. It does not
authorize construction before the applicable public gates and replays pass.

## Entry condition and immutable inputs

Require the method's seed42 screen to pass all three protocol pairs, including
strict buffered spellF1 above `.7542857142857143`, then require the identical
seed43/44 method to pass every remaining pair. Verify all nine pairs and their
independent exact replays, shared/official score agreement, all manifests and
complete ledger closeout. The R55 diagnostic R37 proposal is never an eligible
replacement. Do not use scores from the fitted full-match candidate as new
validation evidence.

Freeze new construction sources and a construction-only plan after those gates.
Existing Python files are already frozen and must remain unchanged. Fresh helper
files may reuse the functions below. Use new output directories which must not
exist; preserve all earlier failed methods and selected artifacts.

Read-only checks independently performed for this design:

| Input | Verified state |
|---|---|
| `data/control_renewal_v1/stoppage_inputs/raw.pkl` | `(199319,177)`, `RangeIndex(0,199319)`, SHA-256 `f75702b6e720448236be119b69714aff130a470645c4955de83a25f9bf6f88ef` |
| R37 `inputs/xy.npy` | `(1,39864,46,2)`, float32 |
| R37 `inputs/observed.npy` | `(1,39864,46)`, bool |
| R37 `inputs/teams.npy` | `(39864,46)`, int8 |
| R37 `inputs/half_id.npy` | `(39864,)`, int16 |
| Existing R37 hidden cache | 1501 centers, `(1501,64)` float32; public prefix only |
| Six selected8 full arrays | 199319 rows; current/previous/broadcast/motion have 3 columns, fast/heads have 2 |
| Protected selected8 artifacts | All 17 hashes match the R54 frozen map; both source symlinks match |

R37 paths above are under `data/control_player_representation_v1/`. Audit their
existing `inputs/manifest.json` and `encodings/manifest.json` before use, rather
than inferring validity from shapes. Verify the external R37 checkpoint
`data/control_masked_player_v1/external_models/temporal/model.pt` against its
existing hash `c22debc44f35a02486b1fd2194d1dab176dcac1a62d92bba214f8dac3f575786`.
For R54, also bind the exact acquired TabICL runtime, dependency inventory,
author checkpoint, R53 prerequisite and all native confirmation evidence.

The exact 17-artifact set is in
`data/control_tabicl_native_v1/protocol_freeze/protected_artifacts.json`: eight
`.ubj` models, six probability arrays, `candidate.csv`, `manifest.json` and
`reference_manifest.json` under `data/candidate_motion_dead_trim/`. Reverify this
set and the frozen `symlinks.json` before and after each construction/replay.
The selected CSV must remain SHA-256
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.

## Shared full features

### Original 102 columns

`temporal_candidate.build_inputs(n)` is the authoritative existing composition
(`experiments/temporal_candidate.py:30–40`). To use the already verified full raw
cache without reconstructing its 177 columns, a new helper can reproduce that
composition explicitly:

1. Read the pinned full `raw.pkl`; verify its shape, index and hash first.
2. Form `base = raw[compact_columns(raw)].join(build_kinematics(n))`
   `.join(build_relative_motion(n))`, using the unchanged functions from
   `train_factorized_baseline.py:24`, `ball_kinematics.py:9`, and
   `relative_motion.py:10`.
3. Apply `temporal_candidate.ordered_inputs(base)` without changing column order,
   contact memory, temporal taps, dtypes or NaN conventions. It produces the
   exact 102-column schema, dropping the two pitch-boundary-distance columns.
4. Require `pd.testing.assert_frame_equal(..., check_exact=True)` for public
   frames `11185..18684` against `data/advanced_validation/ordered.pkl`.
   Also verify public raw parity against the retained validation raw cache.
   Feature hashes and column/dtype/index metadata belong in the full manifest.

The kinematics and relative-motion functions read only provided tactical
tracking inputs. Reuse their existing source hashes and tactical input hashes.
Do not derive public inputs from a truncated feature matrix lacking the original
future-context halo. A refitted model's public predictions will differ from its
OOF predictions; feature parity, not fitted/OOF probability equality, is required.

### Full frozen 64-dimensional representation

Do not run `encode_player_representation.main` unchanged: it reads the 1501
public centers. Reuse its `extract` and `reflected` helpers (`:11–27`) in a new
full extraction helper, with the same pinned CPU Torch2.5.0 runtime and unchanged
`masked_player_network.initialize()`/`MaskedPlayer` checkpoint loading. No
random encoder is needed for either passing method.

`prepare_player_representation.py` already saved cleaned inputs for all199319
native frames, sampled every five frames. Reuse those immutable tensors;
`masked_player_native_data.assemble_native` accepts centers in their 5Hz index
space and masks unavailable or different-half context. Each window is 31 samples
(center ±15 at5Hz), and the encoder's radius is75 native frames.

Derive centers from `games.half_video_blocks(GAME)` on the original global phase
`frame % 5 == 0`, rather than restarting the sampling phase at each half:

| Half | Inclusive native range | Inclusive center range | Centers |
|---|---|---|---:|
| 1 | 11185..90651 | 11185..90650 step5 | 15894 |
| 2 | 110808..190197 | 110810..190195 step5 | 15878 |
| Total | 158857 known-half frames | | 31772 |

For each ascending batch of64 centers, repeat its final index until the batch
has64 rows and discard the duplicate outputs. Call `assemble_native` with
`centers//5`. Extract pre-output hidden states for all four reflections in the
fixed order `(1,1),(-1,1),(1,-1),(-1,-1)`. Concatenate the two32-dimensional team
states; average reflections in float64 and cast once to float32, exactly as the
existing helper does. Preserve the original H/A latent column order
`player_representation.COLUMNS`. Do not change the observation-cleaning,
centering, team, half-padding, reflection or precision conventions.

Calculate each center's eligibility using the identical source rule:
at least four observed players from each team at the center, and mean observed
count at least four per team across its31 samples
(`prepare_player_representation.py:29–30`). This is target-free. Save full center
frames, hidden values and center eligibility.

Before full inference, select the1501 old center frames11185..18685 and require
their hidden vectors and eligibility to equal the original R37 caches exactly.
The old final batch used repeated endpoint padding, whereas the full batch may
contain later centers. Although the model has no intended cross-example
operation, do not assume numerical equality: assert it. Retain the existing
team-exchange and reflection checks and record checkpoint/source/runtime hashes.

Implement full interpolation in a new helper. The existing
`player_representation.interpolate` is deliberately public-only and cannot be
called unchanged for all halves. For each half independently, exact grid centers
use their own vector; other frames need both adjacent centers in the **same**
half and both must be eligible. Use the existing float64 weighted combination
and final float32 cast. Never bridge halftime, extrapolate an edge, or clamp a
missing neighbor into an apparently supported observation.

The five frames unsupported by grid geometry alone are90651,110808,110809,
190196,190197. There may additionally be unsupported observation windows; their
number must be measured rather than assumed zero. Those rows retain the original
selected slot0 probability exactly. Outside known halves, latent columns remain
NaN/unsupported. Require exact public interpolation/mask parity with the existing
R37 helper, including the last public frame's18685 endpoint. The resulting
ordered102-plus-latent64 DataFrame must have the exact166-column schema used in
validation.

## Full selected8 identity

Before applying either candidate, load the six immutable arrays from
`data/candidate_motion_dead_trim/` in this order:
`current`, `previous`, `fast`, `heads`, `broadcast`, `motion`, each suffixed
`_probabilities.npy`. Load broadcast context from
`data/control_broadcast_v1/full_inputs/combined_inputs.pkl` and motion info from
`data/control_pair_motion_v1/full_inputs/motion_info.pkl`; audit their manifests
and existing public/full parity metadata.

Use the half-open blocks `(11185,90652)` and `(110808,190198)` and call
`motion_dead_trim.decode_motion_dead_trim` unchanged with these arrays, context,
motion info and pinned full raw. Require exact equality of all199319 labels with
the protected selected8 CSV. This is the full identity pattern already used in
`build_temporal_dead_candidate.py:40–45`; do not refit any selected8 model.

## Method-specific full recipe

### If R54 passes

Build one context from **all and only the7500 public labeled frames**, retaining
only those divisible by5 and labeled H/A, followed by paired team augmentation
and the unchanged external median imputer. `tabicl_native.context_record` can
prepare the context with `train=PUBLIC` and an empty validation index; pass full
queries separately because that helper deliberately rejects nonpublic validation
indices. Save selected context frame indices, augmented targets/context, medians,
the imputer,166-column schema and exact resolved estimator recipe.

Use the same official8-member classifier, seed42, cached CPUfp32 execution,
single-thread environment and `n_jobs=1`. Fit only this saved public context.
Queries are all158857 known-half frames in a prospectively fixed ascending
schedule, including reference D frames and unsupported rows. Use512-row batches
with last-row padding, original and team-swapped calls separately, reverse the
swapped H/A columns and average. Changing queries or skipping D frames can change
the upstream alternate decoder, so any shortcut needs a proof and a separately
frozen construction interface; the direct complete schedule avoids that issue.

Use `factorized_latent.decode_owner` with the full eligibility mask. Its
`compose_owner` preserves original slot0 D exactly, allocates original live mass
to the new H/A conditional, restores original probabilities for unsupported
latent rows and applies the unchanged selected8 decoder. Assert final selected8
D/X masks remain exact. Context caches and author weights must remain unchanged.

The native fold supervisor has a900-second worker wall limit and is **not** a
ready full-match launcher. There are311 padded query calls per view,622 in total,
plus fit/repeat checks. Freeze a separate full construction workload/supervision
plan before running it, preserving the same mathematical interface, memory cap,
network block, per-operation timing checks and exact reconstruction requirements.
Do not silently waive a native guard or claim that the full workload has already
passed. At the subsequently reported native timing of roughly24–26 seconds per
512-row call,622 calls alone take4.15–4.49 hours. Allow roughly4.3–4.5 hours or
more per build for fit, repeats, integrity checks and I/O; two serial independent
builds are approximately a nine-hour workload. These estimates retain CPU1 and
the default8-member ensemble. This is planning arithmetic from measured calls,
not a completed full-workload benchmark.

In principle, two independently reconstructed full contexts could divide query
work and reduce its wall time toward half. That arrangement is not yet proven
feasible: it needs a separately frozen serial-fit/query-barrier schedule,
full-context combined-memory/resource evidence and byte-identical outputs under
the same canonical512 query calls. Do not introduce that concurrency or change
the ensemble, precision or number of views implicitly. R55 full construction,
if that method passes instead, uses the substantially cheaper tree-model path.

### If R55 passes

Fit one unchanged R37 `ordered_xgboost_frame` model, seed42, on all7500 public
H/A/D labels using the full166 features. Use `temporal_candidate.fit_models` and
its identical class weighting, team augmentation and parameters. This is a
standard full-data refit, **not** an average of13 OOF models or a fitted-public
score. Save the `.ubj` file and public fitting recipe.

Use `temporal_candidate.predict_models` for full team-symmetric H/A/D raw
probabilities, with a fixed construction batch schedule and an independently
loaded-model equality check. Restore original slot0 on every unsupported latent
row exactly. Create a valid neutral1/3 three-state array outside known halves,
because `decode_boundary_risk` validates every supplied row even outside blocks.
Within the two halves supply the refitted raw R37 probabilities with their
fallbacks, then call the unchanged `decode_boundary_risk` on the exact selected8
reference. It performs conditional H/A clipping itself; do not add smoothing,
temperature, passes or a different evidence head.

Assert D/X and every half/component exterior endpoint remain unchanged, and each
reference live component's ordered H/A transcript and positive run count are
identical. Blocks reset at both halves. Do not substitute R37's own decoded
proposal for the selected8 transcript.

## Two standalone constructions and promotion evidence

1. Build A starts in a fresh directory/process from verified immutable inputs.
   Save all new context/model/feature/probability/source provenance. Verify the
   unchanged baseline identity and candidate-specific structural assertions.
2. Build B starts in another fresh directory/process. It must independently
   regenerate probabilities and decoded labels; copying A's CSV or probability
   file is not a replay. An immutable full feature cache may be shared only after
   auditing every manifest hash, its public parity and encoder provenance. Record
   this reuse explicitly rather than claiming a second feature extraction.
3. For R54, B performs fresh official `fit(saved_context,saved_targets)` and
   verifies every reconstructed cache tensor fingerprint and full owner array
   exactly. Do not use fitted-object/cache pickle reconstruction. For R55, B
   independently loads the hash-bound full `.ubj` model and reproduces the entire
   raw probability array exactly. A second XGBoost fit is not required for this
   saved-model replay, unless the separately frozen construction plan asks for it.
4. Both builds create a fresh table with integer frames `np.arange(199319)` and
   exactly `frame,label` columns. Require
   `evaluation.possession_metrics.validate_submission(table,source_frames())`
   to return `None`, then write with `index=False`. Re-read both files, revalidate,
   and require byte-identical CSVs and hashes, exact predictions, and their
   structural invariants. Preserve the40462 reference X frames outside the halves.
5. Audit all full manifests, source/checkpoint/dependency/context/feature hashes,
   public label hash, reference CSV/protected hashes, source symlink targets,
   explicit public training indices, half/query schedules, model reload or cache
   reconstruction evidence, changed-frame/class counts, resources and all failure
   logs. Recheck the original17 protected artifacts and both source links after B.
6. Obtain independent final source/spec and artifact review. Only then append
   an accurate selected-model record pointing to the new local candidate while
   preserving the old selected artifacts. No submission, publication, upload,
   commit or external contact is implied. Public adaptive gains and full-match
   changed rows remain unverified private accuracy.

This checklist is reusable construction design; no method has been promoted by
writing it. Neither R54 nor R55 outcome was inspected for this document.
