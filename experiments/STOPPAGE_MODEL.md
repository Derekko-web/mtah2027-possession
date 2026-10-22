# Selected model: anchored ownership with stoppage consensus

The current default retains the four anchored ownership models and adds two
stoppage heads trained with weak labels. Buffered D recall improves 66.71%→74.81% and
MacroF1 improves 0.8446→0.8679, with false-D frames unchanged at 92 and spellF1
unchanged at 0.7294. Across all nine paired checks, D recall rises 6.88–12.80
percentage points, no false-D count increases, and no primary score regresses.
Mostly missed stoppages fall to zero in every pair. All public live-frame
predictions remain exactly unchanged.

The longer-block spellF1 improves 0.6444→0.6595; buffered/purged primary scores
remain unchanged. This passes a separately frozen adaptive stoppage-recovery
objective. None of the new screens met the original buffered +.04 primary-gain
criterion. Two short possessions and the overlong 19-frame endpoint remain;
stoppage recognition also remains incomplete. Read
[STOPPAGE_RECOVERY_REPORT.md](STOPPAGE_RECOVERY_REPORT.md) for all failed screens,
objective history, paired comparisons, limitations and complete reproduction.
Independent review is recorded in [STOPPAGE_CODE_REVIEW.md](STOPPAGE_CODE_REVIEW.md).

The selected full-match CSV is `data/candidate_stoppage_consensus/candidate.csv`,
with 199,319 rows and SHA256
`dac48e6be66f4395089cae4eee3cfb5daeadcd7963480f399fe079d871be534c`.
An independent build produces the same bytes. All previous candidates remain
preserved, including `data/candidate_anchored_control/candidate.csv`.
Weak labels are fallible proxies from other tracking observations, not additional
corrected ground truth. These are adaptive development results, not independent
or private full-match accuracy. Nothing was submitted, uploaded or committed.

With completed stoppage confirmation, its verified feature cache and the
reproducible anchored candidate, use Python 3.12 in `.venv` with
`experiments/requirements-research.txt`:

```sh
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_stoppage_candidate.py --output data/candidate_stoppage_consensus_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_stoppage_consensus_fresh/candidate.csv
```

For missing prerequisites or fresh confirmation, follow the complete chain in
the report. New output directories preserve frozen results. Current artifact
identity checks are in `data/control_renewal_v1/integrity.json`.
The preceding selection is preserved in [ANCHORED_MODEL.md](ANCHORED_MODEL.md).
