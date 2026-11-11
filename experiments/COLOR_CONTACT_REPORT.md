# Optical ball and jersey-color contact recovery

The frozen three-variant screen fails the substantial-improvement gate. Selected8
is unchanged. This experiment reuses cached native detector observations and
per-player jersey-color probabilities; no new media decoding, classifier fitting,
manual annotation, tracking correction or private labels occurs.

The plan is [COLOR_CONTACT_PLAN.json](COLOR_CONTACT_PLAN.json), SHA256
`b3a22c3b05bce7dad4c78f74ba8f805a32ddee6f5553db01c0a3811d90c87d6b`.
The metadata control uses detected ball contacts with tracking team IDs; the
tracked-color control uses original ball positions with color probabilities;
the fused variant uses detected ball contacts with color probabilities. Fixed
contact support, bounded75-frame memory and conservative short-island rules
preserve every D prediction. All actual team-swap checks pass.

| Variant | Buffered spellF1 | Purged spellF1 | Longer-block spellF1 |
|---|---:|---:|---:|
| Selected8 | .741176 | .685714 | .670270 |
| Detected metadata | .705882 | .651429 | .637838 |
| Tracked color | .741176 | .685714 | .670270 |
| Detected color | .741176 | .685714 | .670270 |

Every row retains3short IoU.5matches. Both color variants produce exactly the
reference labels, while the metadata control changes30/29/30frames and regresses
primary and Macro scores. No variant qualifies for confirmation or promotion.
No additional seeds or full-match candidate were generated.

Source preparation passes its prospective observability requirement:
198strong fused public samples,148H/50A. Those are uncertain source cues, not
verified team/ball labels. Across protocols,190/189/193strong fused live votes
include28/28/24votes contrary to the reference owner. The fused decoder produces
17/16/14short components, only2/2/4with three supporting actual sample votes;
none survives all proposal requirements as a label change. These aggregate
source diagnostics explain inactivity without selecting new thresholds or
claiming that the rejected votes were correct. Prior jersey-only and detector
feasibility failures remain failures.

All12official scores agree with the shared scorer. All4new synthetic tests pass
(contact uncertainty and team symmetry,75-frame expiry/D reset, possible new
short island, protection of existing opposite short island);154tests pass in the
full suite. Actual18935vs19060source and memory-prefix checks pass. The initial
preparation implementation and its caches are retained; adding the required
prefix audit leaves all three full arrays byte-identical, without changing
scientific parameters or inspecting possession scores first.

An isolated workspace reproduces all source arrays and the preparation manifest
byte-for-byte using cached observations, then all12predictions, official metrics,
fold/segment tables, acceptance checks and idempotent ledger replay exactly.
This does not rerun the original neural-network inference or weak-color training.
Every selected candidate artifact remains unchanged. The12rows append once,
with every preceding ledger byte preserved:1,342dependent rows =778inherited +
564added in this chat. Rows are correlated method/protocol/reference evaluations,
not independent experiments. Public development cannot verify maximum or private
full-match accuracy.

Local evidence lives under `data/control_color_contact_v1/`: before source and
candidate snapshots, initial and final preparation, frozen inputs, screen,
`screen_checks.json`, `integrity.json`, `contact_diagnostics.json`,
`prefix_revision_parity.json`, `fresh_parity.json`, `fresh_source_parity.json`,
`preservation.json`, test logs and isolated replay code. All private arrays stay
local. No research commit, upload, submission or publication was made.
