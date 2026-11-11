# TabPFN longer-block diagnosis and short-control fix

**The targeted development fix succeeds:** main spellF1 remains **0.777142857143**, purged remains **0.731428571429**, and longer-block spellF1 improves from **0.576470588235** to **0.681081081081**, above the current model’s **0.670270270270**. The main and purged prediction labels are byte-for-byte unchanged.

| Protocol | Current model | Uncorrected TabPFN | Corrected TabPFN |
|---|---:|---:|---:|
| buffered_minutes | 0.741176470588 | 0.777142857143 | 0.777142857143 |
| purged_spells | 0.685714285714 | 0.731428571429 | 0.731428571429 |
| buffered_100_seconds | 0.670270270270 | 0.576470588235 | 0.681081081081 |

## Why the longer-block score drops

The regression check reproduces the original symptom in under a second using the official scorer: main0.777143 remains good while longer0.576471 trails current0.670270. The frozen13-fold decoder replays exactly from its saved probabilities; there is no reconstruction mismatch.

The middle100-second holdout uses a different, smaller training context. The two largest short-control losses show up in the raw TabPFN ownership probabilities before the broadcast or final overlay stages:

| Public event | Main mean P(correct owner) | Longer mean P(correct owner) | Main / longer decoded recall |
|---|---:|---:|---:|
| 81-frame away possession | 0.842636 | 0.487326 | 100% / 0% |
| 33-frame home possession | 0.535801 | 0.337647 | 100% / 0% |

The longer fold has647 original5Hz context centers, versus772 and874 in the relevant main folds. Both augmented contexts are class-balanced; this is a change in the available examples and fitted conditional ownership behavior, not simply an unequal final H/A count. The decoder’s existing smoothing/transition logic turns the weaker brief-control evidence into uninterrupted possessions. This loses whole spells and merges neighboring spells even though frame macroF1 improves. The losses are already present after the anchored decoding stage. Later broadcast logic does not create them.

The first event begins19frames after a decode boundary; the second is386frames from its nearest boundary. A boundary-only explanation cannot account for both. A separate native, single-variable probe held the exact training context, targets, seed, checkpoint and preprocessing fixed while regrouping1024held-out query rows into different512-row batches. Original, swapped and averaged probabilities were **byte-identical**, with zero argmax changes and unchanged fitted fingerprints. Query packing is therefore ruled out as the large cause on these affected rows. The evidence supports context sensitivity amplified by the integration/decoder; it does not identify a defect inside the pretrained TabPFN implementation.

## The correction

The old hybrid fully replaces the original ownership probability slot with TabPFN’s owner estimate. The correction keeps that path and adds one target-free protection step for existing short-control signals. A baseline-predicted H/A island is restored only when it lasts8–250frames, TabPFN retains less than half of it, at least3frames have independently observed contact support, and every available13-frame flank has an opposite-team majority. At a decode edge, the available flank is used. The rule stays inside each original decode block and preserves all D/X labels exactly.

These duration, support and flank scales come from the existing shared control/protected-island inference vocabulary. The same fixed rule runs in every protocol. It receives predictions, contact observations and decode blocks; it has no ground-truth, protocol-name or special-frame input. Eligibility always reads the unchanged reference and alternative, so one restoration cannot enable another.

The rule restores three baseline control intervals in the longer-block protocol and changes373 labels. Two intervals recover the missed short-control structure; the third is an imperfect baseline signal during a true stoppage and is deliberately retained in the reported outcome rather than removed using truth labels. Longer-block short-spell matches rise from1to3. Main and purged make zero changes. Longer-block macroF1 changes from0.859014to0.856936, still above current0.844383; spellF1 is the user’s stated ranking priority.

A fixed50/50 probability blend was registered as the other causal candidate before either candidate was scored. It failed: main0.731429, purged0.688889, longer0.637838. No parameter sweep followed. The short-control rule above is the successful candidate.

## Validation and artifacts

The first two synthetic tests reproduced complete loss of a supported island, including one clipped at a decode edge; they failed before the correction. All12 focused tests now pass, covering recovery, contact requirement, retained-majority behavior, long-run exclusion, team-swap symmetry, input immutability, D/X integrity, shapes and block overlap. The original fast reproduction passes when pointed at the corrected output. The old frozen predictions remain a red negative control.

A separate read-only audit verifies all12 official aggregate rows and52 per-fold rows across the two candidates and two comparators, complete source/cache identities, D/X preservation, and exact parity between the shared reusable inference entrypoint and the evaluated outputs on all13 original folds. The correction needs zero TabPFN refits; it uses the frozen13-context probabilities. The native query-grouping diagnostic used one separate fresh fit for causal isolation.

An initial scorer export used the18935-row feature prefix instead of a full199319-row scorer table. That failed export and its source snapshot are preserved in `data/control_tabpfn_block_fix_v1/`. The corrected `v2` export uses the full public-label CSV shape. Scores, fold diagnostics and segment diagnostics are byte-identical between the failed export and corrected run; neither candidate’s inference changed during the repair.

- [Reusable corrected inference](tabpfn_short_control_inference.py)
- [Target-free protection rule](tabpfn_short_control.py)
- [Complete correction audit](../data/control_tabpfn_block_diagnosis_v1/correction_audit.json)
- [Native query-grouping result](../data/control_tabpfn_extended_v1/diagnosis_query_grouping_v1/result.json)
- [Registered two-candidate plan](../data/control_tabpfn_block_diagnosis_v1/fix_plan.json)
- [Detailed context and stage probes](../data/control_tabpfn_block_diagnosis_v1/analysis.json)

Twelve method/protocol/comparator rows append to the ledger,1639to1651, with all previous bytes preserved. These are dependent development evaluations of two correction rules using the same frozen model predictions, not independent experiments. Source/evidence identities are sealed under `data/control_tabpfn_block_diagnosis_v1/correction_freeze/`.

## Scope and current selection

This fix was designed after inspecting the same7,500 public labels and17 possessions, so its gains are adaptive development evidence. Matched fresh seed43/44 confirmation, independent full native replay and a full199319-frame candidate build have not been performed for this correction. The existing selected model and CSV are unchanged. The prior rejection of the uncorrected TabPFN method remains valid.

The requested diagnosis and fixed seed42 regression are complete. The correction is implemented and validated at the real inference seam. It is a candidate for fresh confirmation, rather than a claim of private/full-match accuracy or an already-deployed replacement.

Current selected CSV SHA256 remains `3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.
