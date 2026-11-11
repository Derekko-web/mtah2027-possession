# DFL possession transfer: primary-source design review

Prepared 2026-10-23. **Design review only; not executed.** This reviewer read
published documentation, provider-parser source and local fitting/decoding source
and aggregate reports. No competition or external raw data, caches, targets,
media or arrays were opened; no downloads, processing, tests, fits or scores ran.
The research skill is applied by the already-dispatched background reviewer.
The executing agent separately verified the catalog; its frozen provenance and
the eventual immutable protocol, rather than this advisory document, own those
execution facts.

## Recommendation and stopping point

Freeze and run **Phase A only: source, label, schema and shared-feature
feasibility**. Do not train an external teacher until that conjunction passes.
Do not fit against public competition targets until a separately frozen external
holdout experiment passes. A feasible data source is useful new evidence, not a
model improvement or promotion.

The primary paper describes seven matches with 25 Hz player/ball coordinates,
plus explicit `BallPossession` (home 1, away 2) and `BallStatus` (inactive 0,
active 1). Ownership is defined through player control and inactivity through
referee interruptions. It documents CC-BY 4.0 release and no match video. This is
a promising supervision source, not proof of complete or correct labels in every
file. [Primary paper](https://www.nature.com/articles/s41597-025-04505-y),
[dataset DOI](https://doi.org/10.6084/m9.figshare.28196177).

The opportunity differs from the failed external Metrica teacher, which used
event-derived weak dead-ball proxies and reproduced the missing-ball shortcut.
It also differs from another feature, decoder or broadcast alignment variation.
[EXPANDED_RESEARCH.md:28,39-44](EXPANDED_RESEARCH.md#L28-L44),
[TEMPORAL_GLOBAL_REPORT.md:73-92,118-138](TEMPORAL_GLOBAL_REPORT.md#L73-L92).

## Phase A: freeze before any XML parsing

Use the seven catalog match identities, not sorted filesystem positions. The
executing agent's proposed fixed split is first four catalog IDs
`WMX/WN1/WOH/WOY` for training, `WPY` for development and `WQQ/WR9` as locked
whole-match tests. Store exact complete match IDs and matched information/position
file identities in the protocol; suffixes here are shorthand. Keep both halves
of every match together. Neither test is available for threshold, feature,
architecture or integration selection. Events and videos are unnecessary.

The author's example `data_processing.py` pairs information and position files
with `zip` over directory listings, without validating match identity. It is an
analysis example, not a safe ingestion contract. Match files by their embedded
identity and manifest, rejecting absent, duplicate or conflicting matches.
[Author's source](https://github.com/spoho-datascience/idsse-data/blob/main/data_processing.py).

The current floodlight parser exposes coordinates, possession and status
separately, but fills ball arrays by list order and player arrays using ranges
between first and last frame IDs. It estimates the frame rate. Therefore
explicitly validate keyed frame order, continuity and timestamps instead of
assuming that a successfully returned array establishes alignment. Pin any
parser version/source used.
[Official parser source](https://floodlight.readthedocs.io/en/latest/_modules/floodlight/io/dfl.html#read_position_data_xml).

Adopt the executing agent's prospective support gates unchanged:

| Requirement | Frozen threshold or rule |
|---|---|
| Source identity | Verified catalog/license, hashes, unique match IDs and exact information/position pairing |
| Match sections | Two halves per match, each with at least 25,000 canonical frame rows |
| Time axis | Unique `(match, half, N)` keys, 25 Hz timestamp alignment, no conflicting duplicates |
| Status coverage | Valid 0/1 status on at least .99 of the declared frame universe |
| Live ownership coverage | Valid owner 1/2 on at least .98 of active frames |
| Supervised support | At least 2,500 live H and 2,500 live A frames, and at least 10 spells of each team, per match |
| Live player geometry | At least four finite players per team on at least .95 of live frames |
| Live ball geometry | Finite XY on at least .90 of live frames |

Freeze whether coverage thresholds apply per match or per half, and their exact
denominators, before parsing. Use the canonical half/frame grid, including gaps
as missing, rather than dropping absent rows to make coverage appear complete.
Halftime is not an interpolated section. Preserve unknown timestamps/values and
report their counts. Thresholds are prospective feasibility decisions, not
estimates of label accuracy. A failed conjunction stops without deleting a
failed match, changing the split or lowering a cutoff.

Target conversion is independent of feature construction: a known inactive
status gives D; a known active status with known owner 1/2 gives H/A; everything
else remains unknown X. Never forward-fill ownership, smooth labels or close
unknown gaps. Count spells on genuinely contiguous known sections, resetting at
halves, missing keys and unknown labels. Report run-length, switch-density and
unknown-gap aggregates without using them for post hoc filtering or retuning.

This conversion resembles the challenge's mechanical D/owner rule, but ownership
through contested or loose balls and exact restart/transition timing still need
explicit compatibility analysis. The challenge retains the last controlling team
in such phases; similar names are not proof of identical conventions.
[CHALLENGE.md:42-45,55-80,97-103](../CHALLENGE.md#L55-L80).

The paper documents processed multi-camera tracking and quality controls;
competition inputs are automatic, fragmented and often missing. That domain gap
remains even if label conversion is syntactically exact. The available sources
do not validate every external possession boundary or establish transfer to this
fixture. [Primary paper](https://www.nature.com/articles/s41597-025-04505-y),
[CHALLENGE.md:109-118](../CHALLENGE.md#L109-L118).

## Shared observations: use an explicit XY-only interface

Build one small, fixed geometry view from ball XY, per-frame player XY sets and
home/away membership. Suggested ingredients are nearest three team distances,
within-radius counts, valid-player counts, inverse-distance control share,
ball displacement and bounded contact history/taps. Freeze every column,
formula, physical unit, validity predicate and temporal radius before processing.
Keep the total tactical observation reach inside 250 frames, with exact actual
prefix equality and half resets. Missing support stays missing.

Apply exactly the same builder to both domains. Convert coordinate origins from
documented pitch metadata, retain physical metres, and test translation,
reflection, rotation, team swap and player permutation. Do not use frame IDs,
match clocks, player/jersey identities, team names, events, tactical tags or
attacking direction. Player IDs may identify source membership during parsing;
they must not enter features or justify player motion continuity unavailable in
the competition inputs.

Exclude **BallPossession/BallStatus and every derivative** from features. Also
exclude provider speed, acceleration, covered-distance, ball height and provider
labels/flags whose observation semantics are not shared. Recompute any included
ball kinematics from the same bounded XY rule in both domains; never use supplied
speed in one and recomputed speed in the other. Do not make coordinates missing
because a target says dead or unknown. Build the feature table from observations
before joining supervision; a feature whitelist is safer than dropping two known
target columns.

The selected ordered102 view cannot simply be reused as the shared schema:
compact inputs include supplied nearest/player speeds and temporal taps inherit
them; the full raw loader also adds automatic tags. Borrow formulas without
silently importing those provider-dependent fields.
[train_factorized_baseline.py:24-31](train_factorized_baseline.py#L24-L31),
[temporal_control.py:7-30](temporal_control.py#L7-L30),
[rich_features.py:47-78,82-140](rich_features.py#L47-L78),
[temporal_candidate.py:30-40](temporal_candidate.py#L30-L40).

Any synthetic missingness must use one fixed, predeclared schedule generated
independently of possession/status, events and transition positions. Apply it
to observation coordinates **before** deriving histories or kinematics. Reusing
label-dependent ball gaps, concealing only dead-ball frames or masking after
feature computation would create a shortcut or leave concealed coordinates in
the features. Freeze clean and corrupted comparisons as diagnostics, not a
choice between alternatives after observing scores. They cannot certify a full
model of automatic tracking errors.

Phase A should require exact invariance of features when all target/provider
fields are randomized or removed, equality under permitted coordinate/team
transformations, bounded-context perturbation checks, actual-prefix parity and
fresh-process replay. Those are substantive interface checks; they do not turn
source coverage into evidence of predictive value.

## Later training: one simple teacher and one fixed integration

If Phase A passes, freeze Phase B separately before fitting. A defensible minimal
teacher is one binary H/A XGBoost model using the existing depth2/120-tree
`binary_model` settings, known active external rows, fixed training stride,
equal match/class weighting, team-swap augmentation and symmetrized probabilities.
Use native missing values, with no fitted imputer, calibration, early stopping
or architecture sweep. Fit weights and any preprocessing from training matches
only. Ball status determines supervised row eligibility, not model inputs.
[train_factorized_baseline.py:34-38,88-93](train_factorized_baseline.py#L34-L38),
[rich_features.py:143-151](rich_features.py#L143-L151).

Before any public competition-target fit, require a frozen external gate on the
development match and both locked test matches. Compare against the same simple
nearest-team ownership baseline and report class-balanced frame performance,
calibration, whole possession matching, brief runs and each match separately.
The exact numeric gate belongs in the Phase B protocol before fitting or inspecting model performance;
it must not be invented from obtained scores. For conditional ownership scoring,
use the same reference D/unknown sections for both teacher and baseline and label
the result explicitly: this is not measured full H/A/D inference. Test failures
stop this method; they do not select another feature or model.

One fixed later integration can add only the frozen teacher's conditional
probability to the shared geometry view for a public H/A head, with an otherwise
identical public-only head as its matched control. Fit only each existing fold's
eligible public live targets; preserve buffers, whole-run exclusions and training
weight provenance. The teacher itself never fits competition targets. No new
pseudo-labels outside the public window or use of reserved labels is implied.
Full-public fitting is a separate post-confirmation build step, not validation.
[temporal_candidate.py:43-74](temporal_candidate.py#L43-L74),
[build_motion_dead_trim_candidate.py:62-73](build_motion_dead_trim_candidate.py#L62-L73).

Apply one fixed existing D-preserving live-ownership decoder, resetting at
reference D, X, half and evaluation-block boundaries. Do not also try proposal,
joint-slot and dead-trim variants. Existing shared live replacement provides the
appropriate invariant but has failed with earlier inputs; new supervision is the
new hypothesis, not evidence that this decoder will now work.
[provided_speed.py:114-126](provided_speed.py#L114-L126),
[SHORT_PRECISION_REPORT.md:97-129](SHORT_PRECISION_REPORT.md#L97-L129),
[SPATIAL_SHAPE_REPORT.md:41-68](SPATIAL_SHAPE_REPORT.md#L41-L68).

## Preservation and claims

Phase A adds zero competition possession-score rows and changes no selected
artifact. Freeze hashes of the 17 protected artifacts, all existing ledger bytes,
source/input identities and old failures; write new evidence only to a fresh
ignored directory. If the phase stops, report the exact failed support condition
and retain all failed evidence. Do not restart with another conversion or split
under the same protocol.

All inherited substantial-gain, extra-short-match, paired-protocol, D/Macro,
official scorer, seeds42/43/44, feature/model/source replay and independent review
requirements remain in force for later promotion. External holdout success
would not erase adaptive use of the competition's 17 public possessions or prove
private/full-match accuracy. No current score gain, safe transfer or optimum is
claimed by this source-only review.
[PAIR_MOTION_REPORT.md:147-163](PAIR_MOTION_REPORT.md#L147-L163),
[WORK_HISTORY.md:340-355](WORK_HISTORY.md#L340-L355).
