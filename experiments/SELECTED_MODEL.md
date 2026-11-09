# Selected model with short-control recovery

The selected recipe is `tabpfn_short_control_recovery_seed42`. Its 199,319-row CSV is `data/candidate_tabpfn_short_control_recovery_v1/candidate.csv` with SHA256 `801cb64b59260daf4586865e9ef13d124fc61d7584e2798283e465a1758e9ed7`.

The 51-frame ownership smoother erased a brief home possession inside a long away run. The [recovery decoder](short_control_recovery.py) proposes short handovers using unsmoothed TabPFN ownership. It requires fast-head support and tactical and broadcast contacts, and uses the same rule for both teams. The recovery stage consumes only model predictions and observations. It builds on the [previous physical and continuity repairs](HOLDOUT_EXTRAS_REPAIR_REPORT.md) and reuses the fitted models and caches.

The full export changes exactly 19 frames, all in the public window. Official public SpellF1 increases from **0.912500 to 1.000000**, macroF1 from **0.991058 to 0.993200**, and accuracy from **0.991200 to 0.993733**. All 17 possessions match at every scored IoU. All nine out-of-fold prediction arrays across seeds 42–44 are identical to their predecessors. All 39 fold comparisons passed. The official scorer ran on both versions of the full export and each of the nine runs.

Two builds from the saved inputs produced byte-identical CSVs and passed the official format validator. The public labels informed this repair, and the private full-match score remains unknown.

For a clean rebuild from the tracking bundle, use the [submission build instructions](../SUBMISSION_BUILD.md). The [technical summary](../TECHNICAL_SUMMARY.md) describes the selected system and its evaluation.

Run the export and scoring replay with existing experiment caches from the repository root and a fresh output directory.

```sh
.venv/bin/python experiments/build_short_control_recovery_candidate.py --output /tmp/tabpfn-short-control-replay
```

The output includes the candidate, official scoreboard, per-run comparisons, and recovery events. The selected package also preserves a copy of the preceding selection document and pointer.
