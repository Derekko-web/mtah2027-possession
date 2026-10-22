# Local jersey-color research

The local color classifier passes feasibility, but its possession models do not
improve the selected solution. All 21 possession evaluations are retained. No
method clears the frozen substantial-gain/extra-short-match gates, so no new
seeds or full candidate are fitted. The seven-model broadcast candidate remains
selected, SHA256
`dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce`.

## Hypothesis and data boundaries

[JERSEY_COLOR_PLAN.json](JERSEY_COLOR_PLAN.json) was frozen before extracting
color features. The hypothesis was that independent jersey-color evidence could
correct tracking team assignments near the ball and recover short possessions.
The selected candidate, confirmation, source tree, selected entry and previous
1,156-row ledger were preserved under `data/control_jersey_v1/` before this work.

Only tactical video is decoded. Its local stream metadata agrees with the
contract: 1920x1080, 25fps, start time 0 and 199,319 frames. No images, crops,
tracking excerpts or generated scene descriptions were displayed or sent to a
hosted model. NumPy and local ffmpeg compute statistics directly; derived caches
and checkpoints stay under ignored `data/`. No dependencies, manual annotations,
private labels or public possession targets enter color feasibility.

Tracking team IDs supply noisy proxy jersey labels. Sampling uses 240 fitting
frames and 80 validation frames from both official halves, on a 250-frame grid
with seed 42. It excludes public frames 11185..18684 plus 250 frames, all 24 frozen
annotation windows plus 250, and fitting frames within 250 of any color validation
frame. The annotation metadata remains unlabelled and its SHA is unchanged:
`5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c`.

Training crops require moving in-pitch players at least 3m from another observed
player and away from pitch boundaries. Torso crops sit above projected tracked
feet, using local homography scale and a fixed 1.75m body-height approximation.
Generic green-pitch masking removes grass. A 124-dimensional normalized HSV/RGB
summary contains no frame, position, track identity or possession target.

## Color feasibility and numerical repair

All six predeclared combinations pass the proxy-label gates. The first in frozen
order, ratio 1.0 logistic regression, is selected; no choice uses possession scores.
There are 2,682 fitting crops and 875 validation crops for this profile, with 440
home and 435 away validation crops. Crop validity is 99.21%.

| Crop height ratio | Model | Balanced proxy accuracy | Confident coverage | Confident proxy agreement |
|---|---|---:|---:|---:|
|1.0|Scaled logistic|.98403|.98971|.98961|
|1.0|Random forest|.98857|.97486|.99062|
|.8|Scaled logistic|.98510|.98282|.99184|
|.8|Random forest|.98051|.97365|.99059|
|1.25|Scaled logistic|.98627|.98857|.98728|
|1.25|Random forest|.98283|.96114|.98930|

These are agreements with track-team proxies on a held-out color sample, not
human jersey accuracy, new possession ground truth or independent match accuracy.

The initial extraction completed all 320 frames, then an exact reload check failed
for parallel random-forest prediction: maximum difference 3.33e-16. The original
source, plan, two completed models, failure log and crop cache remain under
`reload_repair_before/`. Inference now uses one thread for deterministic tree
summation; fitting still uses two. The frozen plan records this numerical repair.
The crop cache is byte-identical, and the first profile's logistic parameters
and forest tree states are identical. All six repaired models reload exactly.
No possession score had been inspected before this repair.

## Observation construction and frozen possession screen

[JERSEY_OWNERSHIP_SCREEN_PLAN.json](JERSEY_OWNERSHIP_SCREEN_PLAN.json) was frozen
after feasibility and before fitting a possession head. Local video is sampled
every 5 frames over 10935..18935. All 1,601 samples are retained as numeric team
probabilities, with no saved images. Three stream samples match independent
accurate-seek decoding pixel-for-pixel.

Confidence>=.8 assigns each observed player to H/A independently of metadata.
Interpolation requires two valid endpoints exactly 5 frames apart; missing or
separated samples are not filled. Tracking slots associate these nearby
observations; slot identity is not a model feature. Corrected nearest distances,
speeds, contact counts/control share, coverage and reassignment fractions form
46 geometry features. Fixed past/future summaries reach 125 frames, plus at most
4 frames of interpolation context. Existing 250-frame fold exclusions cover this.
Actual feature constructions ending at 19060 and 18935 agree on every public row.

Usable color geometry covers 88.20% of the public window. Mean classified-player
fraction is 92.61%; mean reassignment fraction among classified players is 2.18%.
The frozen screen fits a binary fast head and a joint frame head on the original
eligible public training folds. The color classifier is never refitted on
possession labels. Each head is decoded with live ownership replacement, short
proposals, and conservative physical source closure. All reference D frames and
frames without valid color evidence remain unchanged. No frame IDs or true
endpoints enter inference.

## Possession results and rejected methods

| Method | Buffered minutes | Purged spells | Buffered 100 seconds |
|---|---:|---:|---:|
|Selected reference|.741176|.674286|.670270|
|Fast, replace live|.477778|.432432|.515789|
|Fast, short proposals|.741176|.674286|.670270|
|Fast, source closure|.741176|.674286|.659459|
|Joint, replace live|.675000|.606061|.571429|
|Joint, short proposals|.741176|.674286|.670270|
|Joint, source closure|.741176|.674286|.659459|

Broad replacement loses interval precision. Short proposals change no predictions.
Source closure leaves buffered/purged predictions unchanged and regresses the
longer-block primary score. No additional short IoU .5 match appears. All methods
retain D exactly. There are 20 primary regressions among 78 clipped-block
comparisons, including the broad replacements; every comparison remains saved.

The diagnostics explain why strong proxy color scores are insufficient. On
comparable observed public live frames, original nearest-team agreement with the
possession target is 77.84%, versus 76.05% for color geometry. Color changes 77
wrong target agreements into correct ones and 178 correct agreements into wrong
ones. These are possession-target comparisons, not independently verified
tracking-team errors.

| Short true possession | Original nearest-team agreement | Color nearest-team agreement |
|---|---:|---:|
|71-frame away|.23944|.14085|
|81-frame away|1.00000|1.00000|
|19-frame home|.42105|.42105|
|33-frame home|.00000|.00000|
|90-frame away|.46667|.46667|
|87-frame home|.80460|.68966|

Both distance sources are observed on all frames of these six cases. The color
classifier does not supply the missing ownership evidence for the 19- or 33-frame
cases, and worsens nearest-team target agreement for the 71- and 87-frame cases.
The experiment does not establish that its color assignments are the cause;
nearest-player team and possession are different targets. Accurate proxy-label
classification alone cannot justify replacing possession inference.

## Integrity and reproduction

The official scorer agrees with all 21 evaluations. All 26 new OOF ownership
checkpoints reproduce saved probabilities exactly; six color checkpoints rescore
their saved feasibility metrics. Shared inference reproduces every prediction.
Training buffers, D masks, missing-source fallback, source/input/cache hashes,
stream/seek parity and padded-feature parity pass. All 1,156 preceding ledger
rows remain unchanged. The ledger now has 1,177 dependent score rows: 778 inherited
and 399 added in this chat. Six color-feasibility rows are recorded separately and
are not represented as possession-score evaluations. All 103 tests pass.

The selected CSV, model/probability artifacts and manifest remain unchanged.
Only the selected entry's history count was updated; its original bytes remain
in the source archive. No commits, uploads, submissions, publication or external
contact occurred. No new independent/private accuracy or maximum score is known.

Use the existing Python 3.12 `.venv` and pinned research requirements. First
reproduce the selected broadcast candidate/confirmation and its prerequisites
as described in [BROADCAST_CLOSURE_REPORT.md](BROADCAST_CLOSURE_REPORT.md).
For a fresh workspace with no session snapshot:

```sh
.venv/bin/python experiments/prepare_jersey_baseline.py
.venv/bin/python experiments/prepare_jersey_color.py
.venv/bin/python experiments/prepare_jersey_geometry.py
.venv/bin/python experiments/research_jersey_ownership.py
.venv/bin/python experiments/finalize_jersey_ownership.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

Outputs are exclusive. After an extraction is confirmed stopped, `--resume`
reuses its crop cache under the unchanged plan; a timeout alone is insufficient
reason to restart. `finalize_jersey_ownership.py --audit-history` additionally
checks this workspace's original local source archive and protected baseline.
A new-workspace smoke test successfully creates the OOF reference and replays
all 21 scores without either historical session file. Ordinary fresh runs report
unperformed preservation checks as null. Updating the aggregate ledger requires all retained
historical outputs, using `update_experiment_ledger.py`; do not discard old rows
when an earlier output is unavailable.

Evidence under `data/control_jersey_v1/` includes `feasibility/scores.csv`, the
color cache/models, `reload_repair_parity.json`, `inputs/manifest.json`, all OOF
checkpoints/predictions, `screen/scores.csv`, `official_comparison.csv`,
`nearest_team_diagnostics.csv`, `block_deltas.csv`, `screen_checks.json`,
`integrity.json`, `fresh_reference_audit.json`, `runtime_versions.json` and
`tests.log`. The selected model remains the current best.
