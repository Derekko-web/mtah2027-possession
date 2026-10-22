# Ball-location diagnostics and unused scalar lead

The existing ball-location disagreement is not fixed by a short temporal offset,
different box reference point or corrected frame seek. No candidate, coordinate
alignment, weak feasibility threshold or possession score changes in this round.
Selected8 and the1,291-row ledger remain exact. The fast possession feedback remains
red:3 short IoU≥.5 matches,4 required.

## Prospective diagnostic and evidence

[BALL_LOCATION_DIAGNOSTIC_PLAN.json](BALL_LOCATION_DIAGNOSTIC_PLAN.json) was frozen
before the diagnostic run, SHA256
`0a281f0f5ca54816e7d4e0b7c8f721fdbb5eaf7c8c708cc66f4375bb2d31485e`.
It reuses only the60 previously excluded weak centers and immutable detections
from the modern detector round. Public/reserved250-frame buffers remain checked;
the separate confirmation images stay unused. These are fallible detector/tracker
comparisons, not corrected ball ground truth.

- **Short lag:** test every integer shift−10..10. Each source ball point is
  projected through the current frame's homography; all shifts use the same
  complete cases. Zero shift has the lowest median image residual in every
  variant, with31..56 complete cases. No offset is applied. This rules against
  the tested short global-lag explanation on this cohort, not every possible
  tracking/timing failure.
- **Box convention:** test center, bottom center and top center of the strongest
  ball detection. Bottom center gives small gains in wider crops but still has
  only.357..467 location-radius coverage among their detected positives. None is
  a sufficient location correction. Original model decisions remain unchanged.
- **Exact frame control:** at cohort indices0/29/59 compare the ordinary accurate
  seek with a decode that retains original timestamps, seeks two seconds earlier
  and selects the exact original PTS. All RGB bytes and hashes match. Source
  metadata confirms1920×1080,CFR25fps,start0. No pixels are displayed or uploaded.
- **Context consistency:** all nine eligible within-model crop pairs and
  same-size between-model pairs have96.97%..100% of both-detected centers with
  a mutual nearest ball match within8source pixels. Median matched separation is
  .156..250pixels. This makes gross crop/box-coordinate instability less likely.
  Correlated models can agree on wrong objects, so it does not establish accuracy.

These checks constrain the next hypothesis. They do not explain the remaining
spatial discrepancy or authorize moving the supplied tracking timeline. The
automatic ball track and pretrained detector can each be wrong.

## Follow-up source-schema observation

A separately recorded local schema inspection found that all199,319 raw ball
records have three fields. The prior research loaders consistently retain only
the first two. Locally parsed metadata declares the ordered fields x/y/s, identifies
s as speed and declares its unit km/h. No raw XML or tracking excerpt enters
hosted context; only these derived semantic checks and aggregate counts are read.

The supplied scalar has20,971 finite values. It coexists with finite x/y on9,909
frames, is present without x/y on11,062, and is missing on103,579 x/y-observed
frames. Only8 of the60 weak validation centers have it. These patterns are not
proof of calibrated physical speed or a ball-height interpretation. Seven cases
with both scalar and a strongest wider-crop detection are insufficient to explain
the image disagreement. The declared field is treated as a noisy supplied scalar.

The public
[Kloppy Metrica reader](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/metrica_epts/reader.py)
builds its parsing map from metadata, and its
[deserializer](https://github.com/PySport/kloppy/blob/master/kloppy/infra/serializers/tracking/metrica_epts/deserializer.py)
constructs ball coordinates without retaining a supplied speed. Exact public
reference files and hashes are archived locally. This helps explain why the
field was absent from the prior coordinate-only research; it does not validate
the field's values.

This is a distinct new feature lead: the scalar and its missingness could supply
evidence even when ball x/y is unavailable. No public target fitting or per-class
scalar scoring has occurred in this diagnostic. A separate prospective feature,
head and decoder plan is required before testing possession performance. No
previous failed visual gate is reclassified as passing.

## Preservation and verification

Before this round, the source tree and selected8 candidate were copied and their
hashes recorded in `data/control_ball_location_v1/`. The red feedback, frozen plan,
all lag curves, convention comparisons, context matches, source-schema summaries,
exact-PTS hashes and private arrays are preserved there. No fresh image centers,
manual labels, private truth, full CSV, commits, uploads, submissions, publication
or external contact were created. All24 annotation windows stay unlabelled and
unused. Maximum score and private full-match accuracy remain unverified.

The two new synthetic tests check that lag calculations use the current camera,
enforce common observation sets and do not count distant/context-missing matches.
All140 tests pass, including a metadata unit/order guard. The diagnostic reruns against immutable inputs in a separate
output workspace and verifies every numerical array/summary and exact frame hash.
The unchanged candidate and earlier scientific files are independently checked.

With the preceding modern detector screen present, reproduce into a fresh
`data/control_ball_location_v1/diagnostics` directory:

```sh
.venv/bin/python experiments/ball_location_diagnostics.py
.venv/bin/python experiments/inspect_ball_scalar.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
```

This is diagnosis, not a passing possession improvement. Current selected
reproduction remains [SELECTED_MODEL.md](SELECTED_MODEL.md).
