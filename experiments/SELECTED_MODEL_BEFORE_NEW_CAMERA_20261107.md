# Selected model: TabPFN with observed broadcast short-possession boundaries

The selected full-match candidate is `data/candidate_tabpfn_broadcast_short_v1/candidate.csv`,199319rows,SHA256 `e2f550bec2ccaa0a381bfe1e227538f800fdf1b369005c526c70e6e68a8713a5`.

Prespecified seed42 public OOF SpellF1 is **0.8114285714**, compared with0.7771428571 for the previous TabPFN model. Purged SpellF1 improves0.7314285714→0.7657142857; longer-block improves0.6810810811→0.6918918919. All9matched seed/protocol SpellF1,macroF1 and accuracy comparisons improve. None of39clipped-fold SpellF1 or macroF1 comparisons regresses.

The unchanged TabPFN system receives one target-free broadcast-contact envelope refinement. It only tightens fully observed8–250frame reference H/A islands bracketed by opposite live control, with3exclusive same-camera contacts,13frame padding and at most25frame inward movement per edge. Every D/X label stays exact. Main short IoU0.5matches improve3/6→5/6; the19-/33-frame home possessions are recovered at that threshold. The71-frame away miss remains unresolved. Complete short-error resolution is not claimed.

The full repair changes347frames across16islands. Two separate-process exports match exactly and pass the official validator. Public/full broadcast rows match exactly. A separate audit replays39official score rows,26new rejected-head checkpoint probabilities,all9selected predictions and the full output using an independent envelope implementation. All33affected tests pass. The two additional duration-balanced/continuation families fail and are excluded. No additional TabPFN fits or query inference are needed.

These are adaptive checks on7500same-match public labels and17possessions; private full-match accuracy remains unknown. See [SHORT_POSSESSION_REPAIR_REPORT.md](SHORT_POSSESSION_REPAIR_REPORT.md) for all failures,confirmation scores,validation and reproduction.

The prior TabPFN candidate remains unchanged under `data/candidate_tabpfn_short_control_v1/`,SHA256 `4632cfe2cb71ffb15e76784b257175168e1e634754a48a612edfd3cdf6baa58e`. Its preceding selection document is preserved in [SELECTED_MODEL_BEFORE_BROADCAST_SHORT_20261107.md](SELECTED_MODEL_BEFORE_BROADCAST_SHORT_20261107.md). The motion candidate also remains unchanged.
