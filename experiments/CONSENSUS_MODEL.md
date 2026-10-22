# Selected model: consensus-supported brief control

The selected development pipeline now adds consensus-supported short ownership
proposals to the retained ordered XGBoost model. Buffered OOF spellF1 improves
from 0.6125 to 0.6824, mean minute spellF1 from 0.6500 to 0.7133, and false-D
frames fall from 164 to 140. Dead-ball recall stays at 66.71%.

The previously missed 19-frame H possession now has 100% frame recall, although
the predicted interval extends too far. Other short misses remain. Read
[CONTROL_REFINEMENT_REPORT.md](CONTROL_REFINEMENT_REPORT.md) for the method,
regressions, seed/protocol checks, uncertainty and reproduction details.

The new full-match file is `data/candidate_control_consensus/candidate.csv`.
The preceding `data/candidate_temporal_xgb/candidate.csv` and original
`data/candidate/candidate.csv` remain preserved. These are adaptive public
results, not independent evidence or measured private full-match accuracy.
Nothing has been submitted or uploaded.

With existing `data/advanced_validation/` prerequisites, run from the repository
root using the existing Python 3.12 environment:

```sh
.venv/bin/python -m pip install -r experiments/requirements-research.txt
.venv/bin/python experiments/confirm_control_refinement.py
.venv/bin/python experiments/finalize_control_refinement.py
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/build_control_candidate.py
.venv/bin/python evaluation/validate_submission.py data/candidate_control_consensus/candidate.csv
```

To recreate prerequisite OOF probabilities, first run
`validate_temporal_candidate.py` and `finalize_temporal_validation.py`.
The preceding model is documented in [TEMPORAL_REPORT.md](TEMPORAL_REPORT.md),
and the handoff ensemble in [PREVIOUS_MODEL.md](PREVIOUS_MODEL.md).
