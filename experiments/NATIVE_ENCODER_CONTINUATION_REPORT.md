# Round 61: source/native encoder continuation rejected

Both fixed continuation methods fail the seed42 public promotion conjunction.
Neither exceeds the historical buffered record, and no method is eligible for
43/44 confirmation, full-match inference or promotion. Selected8 remains
unchanged; the higher-score objective remains unachieved.

| Protocol | Selected8 | Frozen R37 comparator | Source continuation | Native continuation |
|---|---:|---:|---:|---:|
| Buffered minutes | .741176 | .754286 | .711111 | .635294 |
| Purged spells | .685714 | .720000 | .697143 | .606061 |
| Buffered100seconds | .670270 | .691892 | .605405 | .611765 |

The native method loses two previously matched possessions in every protocol:
matched IoU.5 possessions fall13→11 and missed possessions rise4→6. Short
possession matches fall3→2 throughout. Its primary, mean-fold primary, macro,
prior-match, prior-short-recall and short-IoU guards fail across all protocols;
buffered spurious predictions also rise4→6.

The source-only continuation improves purged spellF1, but its purged macroF1
falls below the frozen .001 tolerance. Buffered primary/mean-fold/macro/spurious
guards fail; longer-block primary, mean-fold and preservation guards fail,
with short matches3→2. Native-minus-source differences remain diagnostics.
The highest comparable buffered research result remains R37
.7542857142857143. The complete saved selection JSON, including every failed
check, independently recomputes exactly.

## What was tested

The separately reviewed R61 plan preserved the R58 architecture, data, schedules,
400 updates, optimizer and all quality guards; it prospectively extended only
the initial compute-projection allowance7200→9000seconds. R58 remains failed.
One source-only control and13 fold-specific source/native encoders start from
the same immutable external checkpoint. Native fitting uses only permitted
public owntrain H/A windows with complete79-frame footprints. Both methods use
the fixed102 ordered plus64 latent features and unchanged R37 heads, selected8
decoder and final live overlay.

All14 original encoder fits finish at400 updates with strict reload checks;
maximum fit time is87.116seconds and peak RSS .661GiB. The completed seed42
screen retains26 new heads and14 encoding manifests. Every protocol covers
all7,500 public frames, irrespective of label. All new outputs preserve the
selected D/X masks exactly.

## External gates and read-only verification

All84 required external match/view cells pass:6 source-control cells and78
native-fold cells, spanning WPY/WQQ/WR9 and clean/masked views. Their252 absolute
checks require balanced accuracy≥.75 and both H/A recalls≥.65. Saved
probabilities reproduce every metric exactly in the prescribed NumPy2.1.2
runtime. Across these cells, minimum source BA/H/A recall is
.802803/.811739/.716131; minimum native is .784688/.803084/.682345. Nine relative
retention diagnostic cells fail; those diagnostics were explicitly outside
eligibility before outcomes. External passage does not establish native
possession improvement.

The independent result audit recomputes all12 official reports exactly from the
saved scorer CSVs. It also verifies all12 shared score rows,52 fold-score rows
and252 segment diagnostics to1e-12, all9 D/X identities including the comparator,
reference/R37 saved-label identities and the complete gate JSON. It performs
no model fitting or inference. Source/input/runtime/cache hashes,14 model
hashes,17 protected selected artifacts and both source symlinks verify. The
audit covers256 input,425 implementation and2,175 runtime bindings, with34
completed manifests and343 cache entries.

Exactly12 ledger rows append,1591→1603. Their identities, values and score-source
hash match the actual screen CSV:
`b5787fc876fbff0d69fe07ff13df330dba07df39af41b8ed6f0b4229a5cf0ac7`.
All605,189 preceding bytes remain exact, SHA256
`6568a890a2a5fbc5ce428230af01a5b3ed9fc88351c370484134b0a23387f100`.
No selected artifact changes.

## Fresh replay remains incomplete

The prospectively mandated replay of both externally passing arms is running;
the failed public quality conjunction does not waive it. At
2026-10-30 13:45:17UTC, execution session42781/controller PID63505 is live with
five completed exact-refit manifests: source control and buffered native
blocks1–4. Complete external-replay and screen-replay manifests are absent.
The scheduled replay requires all14 fresh encoder fits, all14 external
assessments and full public head/shared/official/gate replay. No complete fresh
reproducibility claim is made here. This liveness statement is a timestamped
observation and must be checked again before continuing work.

No43/44 confirmation or final-selection artifact exists. The12 rows are
adaptively reused public development evidence on17 possessions, with no claim
of private/full-match accuracy, a new record or an optimum. No private/manual/
sealed targets, submission, commit, upload or external contact was used by
this audit.

Evidence: [frozen plan](NATIVE_ENCODER_CONTINUATION_EXTENDED_PLAN.json),
`data/control_native_encoder_continuation_extended_v1/screen/`,
`screen_independent_metric_audit.json`, `external_independent_audit.json`,
`screen_ledger_append.json` and the retained `neural_replay/`/`replay_seed42.log`.
