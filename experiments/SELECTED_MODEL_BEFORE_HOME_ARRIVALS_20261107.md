# Selected model: TabPFN with posterior-guarded possession boundaries

The selected recipe is `tabpfn_guarded_boundaries_seed42`. Its full199319-row CSV is `data/candidate_tabpfn_boundary_precision_v1/candidate.csv`,SHA256 `8eeb131d343fd34fea9ad77874960ef8047bccdd407a354f59a256f5de0260a0`.

Seed42 held-out SpellF1 main/purged/longer improves **.8648648649/.8216216216/.7487179487→.8972972973/.8540540541/.7897435897**. All9seed/protocol SpellF1,macroF1,accuracy,meanbestIoU and five per-IoUF1 checks are nonregressing;every previously matched true possession is retained at every threshold. All39clipped-fold primary/macro checks are nonregressing. Mainshortmatches remain6/6;all17true mainpossessions match atIoU0.6,15at0.7.

The71-frame away IoU improves.689189→.898734;the19-frame home improves.575758→.760000. Neighboring202-frame home and401-frame away overlaps also improve. MainmeanbestIoU .819672→.847564. Boundaries remain approximate;the33-/87-frame home intervals are still belowIoU0.7.

The rule uses13frame arrival/5frame release contact envelopes and accepts a boundary change only when saved conditional ownership probability for the new label averages>.5. Proven new-camera onsets may look back25frames for3consecutive bilateral3m approach observations. It receives no truth,protocol or frame-specific output exception and preserves D/X and run topology. No new models or TabPFN queries are required.

The fullCSV now changes49frames across8boundary refinements,all outside the labelled public window. These are not49verified corrections;private score effects remain unknown. Two full exports match byte-for-byte and pass the official validator. Independent reconstruction matches all33trialOOFoutputs and the entire fullCSV. All40affected tests pass,including900new differential cases. The initial unguarded screens and failures remain preserved;33dependent score rows append without changing historical ledger bytes.

See [BOUNDARY_PRECISION_REPORT.md](BOUNDARY_PRECISION_REPORT.md) for all methods,validation,limitations and reproduction. The previous selection document is preserved in [SELECTED_MODEL_BEFORE_BOUNDARY_PRECISION_20261107.md](SELECTED_MODEL_BEFORE_BOUNDARY_PRECISION_20261107.md). All earlier candidate packages remain unchanged.

The7500same-match public labels and17possessions have extensively informed development,including the new posterior guard. These results establish measured development nonregression and reproducibility;private generalization and maximum score remain unverified.
