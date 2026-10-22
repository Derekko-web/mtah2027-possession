# Selected model: broadcast source closure

The current default retains the six-model anchored/stoppage consensus and adds
one joint broadcast-camera head with conservative physical source proposals.
Public development spellF1 improves .729412→.741176 in buffered minutes,
.662857→.674286 in purged spells, and .659459→.670270 in buffered 100-second
blocks. Each primary gain repeats across seeds 42/43/44. All nine MacroF1 and
mean-short-IoU values improve, with identical D masks and no additional official
IoU .5 spurious or missed possessions. No clipped-block primary score regresses.

This is a separately frozen adaptive incremental objective. The earlier buffered
+.04 and additional-short-match criteria remain unmet. The 33-frame home case
now has 100% frame recall but remains overlong at IoU .266129. The 90-frame away
case has better IoU but loses ten correct frames. The 71-frame miss and 19-frame
overlong endpoint remain. See [BROADCAST_CLOSURE_REPORT.md](BROADCAST_CLOSURE_REPORT.md)
for every failed screen, objective history, paired scores, costs and reproduction.
Independent Standards and Spec reviews have 0 unresolved findings:
[BROADCAST_CODE_REVIEW.md](BROADCAST_CODE_REVIEW.md).

The selected full-match CSV is `data/candidate_broadcast_closure/candidate.csv`,
with 199,319 rows and SHA256
`dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce`.
An independent build matches every artifact file, including the manifest.
All seven checkpoints and five probability arrays verify. Both builds pass the
submission format validator; all 97 tests pass. The path-portability repair
preserves feature, model and prediction bytes exactly. All previous candidates
remain, including `data/candidate_stoppage_consensus/candidate.csv`; its entry
point is preserved in [STOPPAGE_MODEL.md](STOPPAGE_MODEL.md).

With the verified broadcast feature cache, confirmation and unchanged six-model
reference present, use Python 3.12 in `.venv` with the pinned research requirements:

```sh
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_broadcast_candidate.py --output data/candidate_broadcast_closure_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_broadcast_closure_fresh/candidate.csv
```

The report gives the complete fresh-workspace chain, including deterministic
baseline preparation, label-free camera alignment, failed screens, confirmation,
official scoring and full feature generation. Historical preservation auditing
is separate from ordinary fresh reproduction. Evidence is under
`data/control_broadcast_v1/`, including `integrity.json`, `provenance_parity.json`,
`fresh_reference_audit.json` and `full_artifact_integrity.json`.
[WORK_HISTORY.md](WORK_HISTORY.md) distinguishes 778 inherited score rows from 399
added in this chat; these dependent evaluations are not independent experiments.

Only 7,500 public labels exist. These are adaptive within-match development
results, not independent generalization or verified private full-match accuracy.
Maximum score remains unverified. No commits, uploads, submissions, publication
or external contact occurred.
