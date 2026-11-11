# After the reviewed annotation pilot: explanation and next decision

Prepared 2026-10-25. This is a read-only diagnosis of the frozen Round29 artifacts.
No fit, new possession score, threshold search, candidate change, human annotation
edit, media inspection or sealed-audit evaluation occurs.

The measured reason the public spell score fell is **additional false possession
fragments without additional recovered possessions**. Official matched counts
remain exactly equal at every scored IoU threshold (.3/.4/.5/.6/.7), in all three
protocols. Spurious counts at IoU.5 rise4→5,5→6 and7→11. Thus the possession F1
loss is explained by the larger prediction denominator, not by a gain hidden by
the aggregate score. This describes the error mechanism, not its training cause.

The buffered sequence changes162 frames:16 are corrections,89 change a previously
correct reference decision to an error, and57 switch between two wrong labels.
Purged counts are17/80/44; longer-block counts44/81/51. Final D/X masks are fixed,
so none of those changes repairs a reference live/dead decision.

Two training/integration concerns deserve checking before another model:

1. The three clips receive **one third of combined training weight** (.5 human
   mass per1 public mass). Their2250 adjacent frames represent only three
   windows, with three direct live H/A changes, seven clipped live runs and two
   dead runs. This is substantial influence for limited independent event
   coverage. Overweighting local patterns is plausible, not proved: the trial
   changes training examples, weights and sampled tree fits together.
2. The supervision mainly challenges live/dead decisions, while the final pilot
   can change H/A ownership only. Of619 disagreements between selected8 and the
   frozen human labels,399 (**64.46%**) involve D versus live. Clip3 has zero human
   D frames, but the selected model predicts326 D frames (**13.04seconds**). Only
   220 disagreements concern H versus A. The locked final D mask prevents using
   the dominant potential correction type.

Selected8 agrees with the human snapshot on72.49% of frames:82.13% in clip1,
51.73% in clip3 and83.60% in clip4. **These are agreement measurements, not an
accuracy judgment on the annotator.** Selected8 is not an answer key outside the
public interval. Clip3 might expose a substantial false-stoppage failure; it
might also contain waiting/restart frames that were labelled by the team holding
the ball rather than by whether play is live. Neither explanation is established
without local review. A simple global H/A swap agrees less with the model and is
not supported by this diagnostic; that does not prove kit identification correct.

The best-supported immediate next step is **annotation calibration and a focused
live/dead review**, before asking for more annotation volume or changing weights:

- Prepare a short, representative practice sample from the released public
  five-minute video, covering ordinary live control, a stoppage and a restart.
  The same annotators label it without seeing the released answers or model
  output; then compare automatically with the official public labels. This
  checks target interpretation, H/A mapping and transition timing. It is not
  untouched model validation.
- Recheck training clip3 locally for whether play is actually live throughout.
  Judge the football evidence, not agreement with selected8. Correct it only if
  the human review finds an annotation error; otherwise preserve the disagreement
  as evidence of a model failure. Do not replace it with model-generated labels.
- If semantics are consistent, freeze one **D-focused correction experiment**
  that protects existing H/A decisions. Use whole training clips for development
  exclusions and keep official public validation protections. Give the small
  human source a prospectively chosen influence; do not sweep weights against
  the same17 public possessions. Only after there is independent clip coverage
  should the untouched audit be used under a separately frozen comparison.

Calibration may show that both label consistency and limited sample diversity
need improvement. A D-focused fit is conditional on that result, not a guaranteed
score gain or a claim that every remaining error is a stoppage error. The known
short ownership misses remain unresolved. Selected8 stays the comparator, the
three human snapshots are retained and the audit remains sealed.

Aggregate evidence is in ignored
`data/human_annotation_diagnosis_20261025/summary.json`. All17 selected artifacts
and every byte of the1,483-row ledger remain unchanged. The original trial,
results and limitations are in [HUMAN_ANNOTATION_REPORT.md](HUMAN_ANNOTATION_REPORT.md).
