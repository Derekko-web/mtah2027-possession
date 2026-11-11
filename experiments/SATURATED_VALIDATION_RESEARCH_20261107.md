# Saturated validation and the next independent evaluation

Research read on 7 November 2026 (local date), after the existing selected candidate reached the development metric ceiling. This note adds no model, predictions, scores, labels or promotion. Existing frozen evidence and selections are unchanged.

## Current evidence and its limit

`SELECTED_MODEL.md` and `HOLDOUT_EXTRAS_REPAIR_REPORT.md` record SpellF1 **1.0 in all nine seed/protocol development comparisons**, based on the same 7500 public frames and 17 true possessions. The full exported candidate's labelled public score is separately **0.912500**, and private full-match accuracy is unknown. These are different evaluation paths; the nine 1.0 values do not establish a full-match or unseen-match score of 1.0.

The repository scorer computes precision/recall/F1 from one-to-one possession matches at each threshold and averages those F1 values (`evaluation/possession_metrics.py`, `spell_scores`, lines 517–538). Every constituent F1 is at most 1. Therefore a strictly higher primary score on the already saturated nine comparisons is mathematically impossible. Exact boundaries and frame labels can still differ when all scored overlap thresholds pass. Secondary-metric progress would be a separate objective, not a higher SpellF1 claim.

## Primary sources read

**Cawley and Talbot (2010), JMLR, sections 5.3 and 6, pp. 2101–2103.** Their empirical comparisons show optimistic evaluation bias when model selection occurs outside the evaluation procedure; its size can rival differences between learners. They argue that selection belongs inside fitting and must be repeated independently within evaluation trials. Here, inspecting errors and changing decoder rules is part of selection even without refitting the classifier. The paper supports a fresh outer evaluation, rather than treating further same-label confirmation seeds as independent evidence. [Paper](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf).

**Dwork et al. (2015), Generalization in Adaptive Data Analysis and Holdout Reuse, sections 1 and 4.1, pp. 1–3 and 15–16.** Repeated adaptive holdout use can overfit the holdout itself. Their Thresholdout method protects validity through a controlled interface, noise and a finite overfitting budget. Ordinary repeated exact scores and event-level diagnostics do not implement that mechanism. Applying a protected interface now would not erase information already disclosed. This source provides a reason to stop describing the existing adaptively reused checks as untouched tests; it does not prove this candidate fails on unseen data. [Paper](https://arxiv.org/pdf/1506.02629).

**Scikit-learn's first-party cross-validation guide, grouped-data and time-series sections.** It explains that dependent groups require validation groups absent from paired training, and that temporal autocorrelation makes ordinary random splits unsuitable for independent-sample generalization estimates. Its group splitters enforce group separation; time-series splitters preserve ordering. Mapping a group to a complete match is our proposed domain choice, not a soccer-specific claim in the documentation. [Grouped data](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), [time series](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data).

## Proposed next evaluation

These are concrete protocol choices inferred from the sources, not results or a new experiment:

1. Keep this match and every already inspected label in the development pool. Do not recover independence by reseeding, repartitioning those frames, or hiding their labels after their errors have guided changes.
2. Obtain appropriately authorized, independently annotated labels for untouched matches. Keep each whole match in one outer partition, including every camera view and duplicate encoding of it. Split by match identity before preprocessing, representation fitting, calibration, model selection or decoder tuning.
3. Freeze the complete pipeline, input and source hashes, candidate set, inference rules, primary metric, secondary metrics and resource limits before any final-test labels or diagnostics are disclosed. If tuning remains necessary, do it using only development matches, with group-separated inner validation. All learned preprocessing must use its training partition only.
4. Evaluate the frozen finalist and a frozen comparator once on the same untouched matches. Report per-match and aggregate official SpellF1, the five IoU results, macroF1, accuracy, dead-ball errors, match counts and boundary diagnostics. Keep reproducibility checks separate from evidence of generalization. A small number of matches limits population claims regardless of frame count.
5. If final-test errors are inspected and used to revise the pipeline, label that set development thereafter and use different untouched matches for the next final comparison. Retain negative results and the initial frozen score.

No new test data were located or accessed for this note. No inference, fits, queries, labels, ledger rows or candidate files were produced. The current development result remains reproducible but adaptively selected; measuring further generalization requires new independent evidence.
