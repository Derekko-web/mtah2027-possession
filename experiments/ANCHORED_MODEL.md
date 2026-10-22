# Selected model: anchored short-live rescue

The current default adds anchored live rescue to the unchanged four-model
consensus pipeline. Buffered OOF spellF1 improves 0.6824→0.7294, MacroF1
0.8373→0.8446, and false-D frames fall 140→92. D recall is unchanged in all nine
paired seed/protocol checks. Six primary scores improve and three stay unchanged;
no clipped-block primary score regresses.

The 87-frame H possession's recall rises 19.5%→74.7% and IoU rises 0.165→0.631.
Two short possessions remain missed, the 19-frame recovery still ends too late,
and visible-ball stoppage robustness remains unresolved. Read
[BOUNDARY_REFINEMENT_REPORT.md](BOUNDARY_REFINEMENT_REPORT.md) for the frozen
comparisons, failed experiments, limitations and complete reproduction chain.
Independent review is recorded in [BOUNDARY_CODE_REVIEW.md](BOUNDARY_CODE_REVIEW.md).

The selected full-match CSV is `data/candidate_anchored_control/candidate.csv`,
with 199,319 rows and SHA256
`973e1670e9e1240ecdde3bbabebf1e803f0d3b6259aad1b9b3223c413d26dcfd`.
An independent build produces the same bytes. All previous candidates remain
preserved, including `data/candidate_control_consensus/candidate.csv`.
The gains come from one public possession. These are adaptive development
results, not independent or private full-match accuracy. Nothing was submitted,
uploaded or committed.

With existing prerequisite OOF outputs, completed boundary confirmation and the
reproducible consensus candidate, use Python 3.12 in `.venv` with
`experiments/requirements-research.txt`:

```sh
.venv/bin/python experiments/prepare_boundary_refinement.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_anchored_candidate.py --output data/candidate_anchored_control_fresh
.venv/bin/python evaluation/validate_submission.py data/candidate_anchored_control_fresh/candidate.csv
```

For missing prerequisites or fresh confirmation, follow the complete chain in
the report. New output directories preserve frozen results. Current artifact
identity checks are in `data/control_boundary_v1/integrity.json`.
The preceding selection is preserved in [CONSENSUS_MODEL.md](CONSENSUS_MODEL.md).
