# Purged and longer holdout possession repair — 7 November 2026

The selected recipe is `tabpfn_physical_continuity_repair_seed42`. All nine seed/protocol development comparisons now have **SpellF1 1.000000**, with all 17 true possessions matched and **zero spurious or missed possessions at every scored IoU from 0.3 to 0.7**. Main performance is retained, while purged and longer holdouts improve. This is the ceiling of these heavily reused development checks, not exact frame labelling or verified private performance.

## Results against the immediately preceding selection

| Seed | Previous main / purged / longer SpellF1 | New main / purged / longer |
| --- | --- | --- |
| 42 | 1.000000 / 0.875676 / 0.821053 | **1.000000 / 1.000000 / 1.000000** |
| 43 | 1.000000 / 0.875676 / 0.873684 | **1.000000 / 1.000000 / 1.000000** |
| 44 | 1.000000 / 0.864865 / 0.873684 | **1.000000 / 1.000000 / 1.000000** |

For seed42, the three purged and five longer extras at IoU 0.5 disappear. Some were false live predictions; others were real possessions split into fragments or left below the matching threshold by poor boundaries. Deleting every unmatched prediction would lose real control.

| Seed42 metric | Main, previous → new | Purged, previous → new | Longer, previous → new |
| --- | --- | --- | --- |
| MacroF1 | 0.932449 → **0.938290** | 0.904345 → **0.934610** | 0.874799 → **0.955858** |
| Accuracy | 0.942533 → **0.947067** | 0.920933 → **0.944667** | 0.895067 → **0.961467** |
| False-D frames | 88 → **85** | 142 → **88** | 16 → **16** |
| Dead-ball recall | 0.857046 → **0.874064** | 0.784888 → **0.861130** | 0.671205 → **0.890402** |

All nine primary/macroF1/accuracy/five-IoU F1 comparisons are nondecreasing. Every previous match flag remains, and every one of the 153 individual true-possession best overlaps is nondecreasing. All 39 clipped-fold SpellF1/macroF1 comparisons pass. Recent short home/away gains are preserved.

## Diagnosis and final inference

The remaining errors have several causes:

- A short invisible owner flip beside a stoppage survived the previous eight-frame continuation refusal. The same two bounded gap anchors now include the existing five-frame dead-facing allowance. This can suppress the whole uncertain flip while preserving the previous owner's existing interval.
- Two live fragments occur while tactical ball coordinates are beyond the goal line. Three consecutive outside-pitch observations establish a dead suffix. A D/live/D interval is suppressed entirely only if that outside core is already at its start; otherwise its original live prefix remains. A spurious owner switch followed by the exit may continue its preceding owner until the confirmed exit.
- A short false D gap separates two home-control fragments. It can be bridged when both sides have the same owner, visible moving ball and fast-owner support agree, tactical contacts support that owner, no opposing broadcast control is established, and the ball stays inside the pitch.
- A short stationary fragment starts exactly at a decode-block edge. Its own saved dead-head and visibility/movement evidence can suppress it without borrowing a neighbor from another holdout block.
- The long 33-frame home prediction includes disconnected contact bouts. The terminal bout has the same cross-camera disagreement that supports the preceding home arrival correction. Its arrival can replace leading excess when the saved owner posterior supports the replacement. Other cross-source short intervals receive bounded inward likelihood trims; first/last runs use only their available opposing neighbor and their intrinsic evidence.
- A short home possession ends prematurely at D, or is followed by an unsupported opposing run, a tiny home trace and D. Broadcast owner contacts support a bounded release extension. An opposing/trace tail may become D after the supported owner release when its own broadcast contacts and owner probability are insufficient. Very short starts can look back to recent tactical contacts under the same bounded rule.

All rules apply to either team, use the unchanged reference for proposals, and stay in the original decode blocks. They receive no ground truth, seed, protocol, target frame number or interval-specific output exception. Constants come from the existing contact/duration/uncertainty windows: 3, 5, 8, 13, 25, 50, 125 and 250 frames, plus the existing 4.5m movement threshold and 0.5/0.65 probability gates. Combining two existing release allowances yields a 30-frame contact tail.

The selected entry point is `bounded_holdout_repair.refine`, wrapping the preserved initial `holdout_possession_repair.refine`. It reuses saved ownership, fast-owner and weak dead-ball probabilities. **No new model fits or TabPFN queries** were required.

Unlike earlier boundary-only rules, D may both grow and shrink to remove false live fragments and restore false-dead splits. The requirement is nondecreasing dead recall and no net increase in false-D count in every pair. It is not a claim that no newly assigned D frame is false. For example, some ambiguous unseen-flip frames remain wrong after reassignment; restoring false-D control and repairing releases reduces the total false-D count. Some previously correct live frames can also change while the overall metrics and each possession's overlap improve. X is exact throughout.

## Retained experiments and audit

The first frozen round, `data/control_holdout_extras_v1/`, retains all 15 seed42 score rows: physical, continuity, control, release and combined variants. The initial combined trial removed extras at IoU 0.5 but incorrectly consumed a real away prefix before a goal-line exit and left a block-edge possession below IoU 0.7. It was rejected by the original nonregression gates.

The separate `data/control_bounded_holdout_repair_v1/` round retains nine seed42 rows and six rows for the other two seeds. It changes only the two diagnosed mechanisms: live-prefix protection before pitch exit and cross-source inward trimming at decode edges. `bounded_combined` was the only eligible seed42 method before the two further paired checks. All 30 dependent score rows, including 21 unselected/rejected rows, append to the ledger without changing preceding bytes.

Verification includes:

- **14 focused tests**, including **800 randomized comparisons** against a separate implementation, covering every retained variant, both teams, missing/partial observations, likelihood ties, stationary/moving/outside balls, contact bouts, continuity and block edges.
- Independent reconstruction of **all 30 OOF files** and the complete full-match CSV.
- **32 official score checks**: all 30 trial outputs plus the preceding/new full files, agreeing on primary/macro/accuracy and per-IoU F1 and counts.
- All nine zero-extra/17-match results at all five thresholds; all 153 individual overlaps and all 39 fold comparisons verified.
- Two separate full builds with identical CSV, event, public-comparison and manifest bytes. Both pass the official submission validator.

Evidence is in the bounded round: frozen plans/source copies, `qualification_result.json`, `final_audit.json`, `fold_scores.csv`, `boundary_diagnostics.csv`, `affected_tests.log`, `official_scoring/`, and both full exports. The original red acceptance check and diagnostic rows remain in the initial round.

## Selected output and limits

CSV: `data/candidate_tabpfn_holdout_repair_v1/candidate.csv`, 199319 rows.

SHA256: `400c86edb5ba25bf0f9c40b29a7c8d1d305b0a4d4b7fd3ebc78a0fb04efe415b`.

The full-match file changes **832 unlabelled frames** through 43 recorded updates. These consist of 489 live-to-D changes, 71 D-to-live restorations and 272 home/away changes. Public full-fit predictions remain exactly unchanged, with SpellF1/macroF1/accuracy **0.912500/0.991058/0.991200**. The full fit already handles the targeted holdout errors differently. These 832 private changes are unverified; they are not 832 proven corrections, and private score effects remain unknown.

The previous candidate and pointer remain preserved, and the exact preceding selection document is saved as `SELECTED_MODEL_BEFORE_HOLDOUT_REPAIR_20261107.md`. Thirty ledger rows append 1837 → 1867 with preceding bytes intact.

The same 7500 public frames and 17 true possessions have extensively informed many rounds of adaptive development, including these rules. Reusing seed/protocol caches does not provide a new independent match or private test. SpellF1 1.0 means all possessions meet the scored overlap thresholds, while frame accuracy and exact boundaries still have room to improve. Further primary gains cannot be demonstrated on these now-saturated development comparisons.

Reproduce a full export into a fresh directory:

```sh
.venv/bin/python experiments/build_holdout_repair_candidate.py --output /tmp/holdout-possession-replay
```

Rerun acceptance:

```sh
.venv/bin/python experiments/reproduce_holdout_extras.py --candidate data/control_bounded_holdout_repair_v1/candidates/bounded_combined
```
