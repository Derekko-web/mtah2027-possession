# Supplied ball-speed preflight: no supervised observations

The new supplied-speed feature hypothesis cannot be evaluated with the available
public labels: **neither aligned source has a finite scalar or scalar-context
observation anywhere in the7,500 public frames**. The branch stops before reading
possession targets or fitting models. No planned39-row screen,39 checkpoints,
confirmation, full candidate or ledger additions were produced. Selected8 stays
SHA256 `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
The ledger remains1,291 dependent rows:778 inherited+513 added during this chat.

## Prospective hypothesis

The preceding [ball-location diagnosis](BALL_LOCATION_REPORT.md) verified a third
ball field that earlier coordinate-only loaders ignore. Local metadata declares
x/y/s and speed in km/h. This is a supplied automatic estimate, not calibrated
physical truth or a ball-height channel.

[PROVIDED_SPEED_PLAN.json](PROVIDED_SPEED_PLAN.json) was frozen before new public
feature construction or possession fitting, SHA256
`68b4c827a4a256e55aacc8f54f3a89c7e8b198e608f9035f8f2fecd81892b2b8`.
It proposed three feature views—tactical, broadcast and fused—and unchanged joint
frame fitting, plus a reused no-scalar control with the same original feature view
and hyperparameters. Each would use three fixed decoders: raw-head replacement,
D-preserving live ownership replacement, and D-only trimming. The intended39
score rows include repeated references and controls; they are not independent
experiments. None of those supervised comparisons ran.

The original substantial-gain gates remain unchanged: at least+.04 buffered
spellF1 and an extra short IoU.5 match, with all protocol/seed protections,
matched-control checks, exact replay, a reproducible valid full candidate and
independent review. No smaller gain or preflight result is counted as passing.

## Prepared observations

The parser verifies declared x/y/s order and recognized speed units. Scalars and
player speeds convert nominally to m/s using metadata; negative/nonfinite values
stay missing. Both camera inputs and the unchanged strict formation mapping are
verified. Raw x/y reconstructed by the new reader matches the preceding observation
cache exactly. No tracking timeline or selected loader is changed.

Each source contributes35 columns: supplied scalar and availability, scalar-only
availability when x/y is absent, a centered5-frame coordinate-speed comparison,
bounded past/future25/75/125 statistics, and per-team scalar/player-speed gaps.
The fused view has74 columns, including four cross-source comparisons. No absolute
coordinates, timestamps, slot IDs or target-derived preprocessing are added.

Broadcast values use the existing5-native/10-tactical enclosure mask. Summaries
run on the tactical timeline within the official halves. The maximum declared
footprint is185frames, inside the existing250-frame target buffers. Each fixed
window has a stable summation order. Actual18935/19060 constructions agree
exactly on every public feature, not just within a floating-point tolerance.

The preflight finds:

| Source | Public scalar fraction | Public fraction with any scalar in context |
|---|---:|---:|
|Tactical|0|0|
|Aligned broadcast|0|0|

The full tactical recording does contain20,971 finite scalar values:2,116 in the
first official half,11,837 in the second and7,018 outside those halves. The public
five-minute interval contains none. These counts use the provided phase mapping;
they do not use private possession labels. The scalar's availability elsewhere
does not provide supervision in this public window.

The remaining nonconstant added features would be coordinate-derived motion.
Fitting this branch would therefore fail to test the proposed supplied-speed
evidence. The original plan did not specify an observability threshold. After
this unlabeled preflight, an explicit zero-support guard was added before target
loading and output-directory creation. This is an early stop, not a changed
acceptance gate. The original unrun research source and the decision are retained.
No supervised score was inspected to make this decision.

## Verification and preservation

The fourteenth round begins with a source archive, a complete selected8 copy,
saved reference OOF files and hashes. Current/earlier models, probabilities,
candidate CSVs, plans and all1,291 ledger rows remain unchanged. Generated scalar
arrays, feature caches and all source-derived artifacts stay under ignored
`data/control_provided_speed_v1/`. No video is decoded in this branch.

All prepared files, including both scalar arrays, all feature tables, info,
frozen plan and manifest, rebuild byte-for-byte in a separate workspace. That
workspace contains no public target files, supervised fold inputs, source archives
or baseline copies. The research entry point still reaches the exact expected
zero-support stop; there is no screen directory or checkpoint. Ordinary preflight
finalization reparses both sources and verifies the unit specifications, all
feature hashes, actual observation parity and absence of fitting outputs.

All144 tests pass. New synthetic controls verify scalar-only observations survive
missing x/y, declared-unit conversion, team-swap symmetry, half isolation and exact
bounded future statistics. The preceding metadata-order/unit guard remains.
The first historical audit incorrectly counted the preserved baseline model copy
as new fitting. Its source/log and a failing minimized regression are retained.
The repaired audit excludes only that preserved candidate subtree and still
rejects new checkpoints elsewhere; both historical and fresh ordinary audits
pass. This audit-scope repair changes no feature, scientific plan or model.
No private GT, manual annotations, frame patches, commits, uploads, submissions,
publication or external contact occurred. The24 reserved windows and60-center
confirmation images remain unused. Maximum score and private full-match accuracy
remain unverified.

## Reproduction

Use the existing pinned Python3.12 environment and verified upstream broadcast
observation cache. In a fresh `data/control_provided_speed_v1/inputs` directory:

```sh
.venv/bin/python experiments/prepare_provided_speed.py
.venv/bin/python experiments/finalize_provided_speed_preflight.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

`research_provided_speed.py` is intentionally guarded and exits with
`STOP BEFORE FITTING` on these inputs. It creates no supervised output. Optional
preflight `--audit-history` verifies the starting archive/candidate/ledger; it is
separate from ordinary reconstruction. `fresh_preflight_parity.json` records full
cache parity, and `preflight_decision.json` records why the proposed fitting was
not run. This closes the direct supplied-scalar hypothesis on the present public
labels; it does not change the selected model or the broader improvement objective.
