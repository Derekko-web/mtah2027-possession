# Selected model: TabPFN with newly observed two-camera control recovery

The selected recipe is `tabpfn_new_camera_control_seed42`. Its199319-row CSV is `data/candidate_tabpfn_new_camera_v1/candidate.csv`,SHA256 `e2f550bec2ccaa0a381bfe1e227538f800fdf1b369005c526c70e6e68a8713a5`.

**The full-match CSV is byte-identical to the previous selected file.** The full-fit parent already labelled70of71diagnostic frames as away. The added stage fixes the held-out failure and fires nowhere in the full-fit export. No changed private submission predictions or improved private score is claimed.

Seed42 held-out SpellF1 main/purged/longer is **.8648648649/.8216216216/.7487179487**, versus .8114285714/.7657142857/.6918918919 before. All9matched seed/protocol SpellF1,macroF1,accuracy and each of five per-IoU F1 scores are nondecreasing;every previous IoU0.5GTmatch remains. All39clipped-fold SpellF1/macroF1 checks are nonregressing. Mainshortmatches5/6→6/6. The71-frame away possession is matched in all9pairs at IoU.689189;its boundaries remain approximate(51true-away frames identified,3late-home frames changed).

The unchanged TabPFN and broadcast short-envelope paths receive one source-gated stage using the original Round25sub-grid alignment. It only accepts bilateral contacts in long reference spells where the original camera path was unmapped and new observations are complete, and waits for13stable reacquisition frames. No frame/protocol-specific output patch or possession truth enters inference. Full/public mapping and all68broadcast feature columns match exactly;rebuilt tactical kernels match historical public values exactly. All D/X labels remain exact.

The audit independently reconstructs12OOFoutputs,21official candidate/reference scores,39folds,the full mapping and both full CSVexports. Both exports pass the official validator. All27affected tests pass,including200time-shift/jitter/missingness differential cases. No new model fits or TabPFN queries occur. The initial full-motion schema failure,original sources and separate repair plan remain preserved.

These are adaptive same-match development checks on7500public labels and17possessions. Private accuracy and maximum score remain unknown;the full-match score is necessarily unchanged while CSVbytes are unchanged. See [REMAINING_AWAY_REPAIR_REPORT.md](REMAINING_AWAY_REPAIR_REPORT.md) for diagnosis,all comparisons,production parity,limitations and reproduction.

The preceding selection document is preserved in [SELECTED_MODEL_BEFORE_NEW_CAMERA_20261107.md](SELECTED_MODEL_BEFORE_NEW_CAMERA_20261107.md). All earlier broadcast-short,TabPFN-short-control and motion packages remain unchanged.
