# Full-team spatial shape and surrounding-space control

The two spatial heads fail the frozen improvement requirements. Selected8 and
all full-match candidate artifacts remain unchanged. The separately frozen plan,
[SPATIAL_SHAPE_PLAN.json](SPATIAL_SHAPE_PLAN.json), has SHA256
`a4dfcbf4e3852a6b69c04153c15ef27dd984e5e954616a8ab8a919cbf56deb61`.

## Hypothesis and inputs

An audit of the selected102ordered and68broadcast columns found that they omit
full-team formation and surrounding-space control. Earlier directional graph
features pooled players within10m of the ball; earlier receiver/contact models
retain local ball proximity. This round tests a distinct representation, while
preserving every preceding failure. It does not claim that all previous research
omitted formation statistics; early rich features included spread, but those
columns are absent from the selected compact views.

Native cached tactical positions supply88features: per-team track count,
regularized covariance scale/eigen ratio/logarea, mean pair distance, center-to-
nearest-opponent distance, centroid-to-ball distance and normalized distance,
Mahalanobis ball distance, Gaussian ball densities at2/5/10/20m, ball projection
and spreads along/across the line joining team centers, and soft nearest-team
control averaged over8ring points at5/10/20m. Selected fields have fixed
past/future25/75mean and standard deviation. Ball visibility and bounded5-frame
speed complete the view. Unsupported ball/formation values stay missing.

No player identity, slot continuity, absolute position, clock, team attack
orientation, target-derived statistic, tag, image or new annotation is used.
Features stay within each official half; observation reach is at most80frames,
inside the existing250-frame target exclusion. Source preparation reads no
possession labels. The prospective coverage gates pass:100%of public frames have
at least5players per team,88.2%have a valid ball. These are support checks, not
verified control observations. Actual18935vs19060public features match exactly.

Two new heads use the unchanged depth3/160tree,frame-objective,seed42 fitting:
shape_joint joins88features to the existing170columns; shape_reduced uses only
those88features. Both use team-symmetric augmentation and prediction, eligible
training labels only and saved actual targets/weights/indices. There are26new
checkpoints over13folds. Existing broadcast joint models provide a matched
170-column no-shape control, verified through13reloads without new control fits.
Three inherited decoders replace only the raw ordered probability slot, replace
live ownership with D exact, or trim referenceD only while preserving H/A/X.

## Results

| Variant / decoder | Buffered spellF1 | Purged spellF1 | Longer-block spellF1 |
|---|---:|---:|---:|
| Selected8 | .741176 | .685714 | .670270 |
| Shape joint / raw replacement | .508108 | .447059 | .488889 |
| Shape joint / live replacement | .412121 | .433333 | .412500 |
| Shape joint / D-only | .670270 | .688889 | .635897 |
| Shape reduced / raw replacement | .500000 | .500000 | .588889 |
| Shape reduced / live replacement | .427586 | .434286 | .477419 |
| Shape reduced / D-only | .662857 | .651429 | .635897 |

The small purged D-only primary gain fails the other protocols and secondary
protections. FalseD falls88/142/16→3/3/0for both D-only heads, but trueD recall
falls from.748128/.784888/.649421 to.526889/.684820/.624915for shape_joint and
.405718/.444520/.609939for shape_reduced. This is not an acceptable stoppage
improvement. D-only short matches remain3/3/3with no additional short recovery.
Every alternative fails promotion;56of78clipped-block primary comparisons regress.
No seeds43/44, full-match candidate generation, selection or review is warranted.

The known71-frame away miss has0correct raw argmax frames in every protocol for
both spatial heads and the matched control. It persists even in the independent
spatial representation, before conservative decoding. Aggregated feature-gain
and changed-frame diagnostics are retained; feature usage is not causal evidence
or new accuracy. No thresholds or individual frames were patched after inspection.

## Verification and retained evidence

All30official score rows agree. All26new and13existing models reload exactly;
shared decoder predictions, training-only targets/weights,250-frame buffers and
source/input/cache hashes verify. Four synthetic tests cover team/slot symmetry,
translation/reflection, complementary ring control, missingness, half boundaries
and exact bounded prefix behavior. All161tests pass.

A separate workspace regenerates every source feature cache and manifest
byte-for-byte, then repeats the ordinary saved-model/shared/official/gate audit.
All metrics, diagnostics and gates reproduce exactly. Cached native tracking and
saved models are reused; the26model fits are not repeated. No current-round
before archives or candidate copies are required for this replay.

Every selected artifact and all preceding ledger bytes remain intact.30dependent
rows append once:1,402total =778inherited +624added in this chat. Repeated
references, decoders, seeds and protocols are correlated development evaluations,
not independent experiments or proof of private accuracy / a maximum score.
No research commit, upload, submission, publication or contact was made.

Evidence stays local under `data/control_spatial_shape_v1/`: before snapshots,
frozen target-blind inputs,26newmodels/fittingarrays,30-row screen and official
comparison, `screen_checks.json`, `integrity.json`, `block_deltas.csv`,
`raw_head_short_diagnostics.csv`, `aggregate_diagnostics.csv`, test logs,
`fresh_shape_replay.py`, `fresh_feature_parity.json` and `round_closeout.json`.
The24reserved annotation windows and60held weak-ball images remain unused.
