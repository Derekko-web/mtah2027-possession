# Research: conservative probability fusion of a transferred representation

Prepared 27 October 2026. This is source research and a proposed experiment;
it does not fit, score, select or promote a model. Existing selected artifacts,
plans and score records are unchanged.

## Recommended next candidate

Test **one fixed equal-weight arithmetic mean** of the original ordered
XGBoost head and the R37 pretrained-latent XGBoost head, before the unchanged
selected8 decoder. Preserve selected8 D/X decisions with R37's existing
`live_overlay` interface. Use no weight, threshold, checkpoint or decoder sweep.

This is a new fusion hypothesis, rather than a reinterpretation of R37's
rejected direct substitution. R37 improved aggregate public spellF1 across all
three protocols but introduced one buffered spurious possession and failed
its frozen substantial-improvement and short-possession gates. Its rejection
remains valid. The new hypothesis is that retaining half of the original head's
evidence may retain useful representation signal with fewer unsupported
ownership excursions. That is an inference to test, not an established result.
See [R37 report](PLAYER_REPRESENTATION_REPORT.md) and
[the selected reference](SELECTED_MODEL.md).

## Primary sources read and their relevance

1. **Kittler, Hatef, Duin and Matas (1998), On Combining Classifiers.**
   Read the author-hosted paper, especially the representation distinction in
   sections 1–2, the sum-rule derivation, section 6's estimation-error analysis
   and the conclusions. The paper studies combining classifiers that use
   different feature representations. It finds the sum rule less sensitive
   to probability-estimation errors than the product rule under its analysis,
   with supporting classification experiments. Its derivation has strong
   assumptions; the paper explicitly limits its theory's application to
   complex sequential decision systems. This supports a small arithmetic-fusion
   experiment, not a prediction that possession spellF1 must improve.
   [Author-hosted full paper](https://cmp.felk.cvut.cz/~matas/papers/kittler-pami98.pdf),
   [DOI](https://doi.org/10.1109/34.667881).

2. **Gorishniy, Rubachev, Khrulkov and Babenko (NeurIPS 2021), Revisiting
   Deep Learning Models for Tabular Data.** Read the experimental protocol and
   sections 4.4–4.5, including the ensemble definition and tables 3–4. Their
   protocol evaluates prediction averages of equal-size groups of separately
   seeded models. They find benefits from ensembling while retaining no
   universal winner between deep models and boosted trees. This is evidence
   for prediction averaging as a credible matched baseline. It does not test
   our two nested-feature XGBoost heads or segment matching and therefore
   supplies no task-specific guarantee.
   [Conference full paper](https://proceedings.neurips.cc/paper/2021/file/9d86d83f925f2149e9edb0ac3b49229c-Paper.pdf),
   [Official implementation](https://github.com/yandex-research/rtdl-revisiting-models).

3. **Ishikawa et al. (WACV 2021), Alleviating Over-segmentation Errors by
   Detecting Action Boundaries.** Read the full method sections 3.1–3.5 and
   refinement analysis. ASRF separates frame classification from a
   class-agnostic boundary predictor, then assigns one action class between
   predicted boundaries. Its similarity-weighted smoothing loss reduces
   smoothing across feature changes. This demonstrates why better frame
   predictions and better segment predictions can diverge. A faithful new
   boundary learner would need training evidence and another protocol, and
   does not justify reusing true public endpoints at inference.
   [Paper](https://arxiv.org/html/2007.06866),
   [Official implementation](https://github.com/yiskw713/asrf).

4. **Wang et al. (ECCV 2020), Boundary-Aware Cascade Networks for Temporal
   Action Segmentation.** Read sections 3.2–3.3, including local barrier
   pooling's equation 4. Learned boundary strengths attenuate temporal pooling
   weights across boundaries; ordinary average pooling is a special case
   without barriers. This gives a principled reason to avoid changing our
   already-tested temporal smoothing together with probability fusion: the
   two mechanisms would otherwise be confounded. The paper does not prove
   that our present smoothing is optimal.
   [Conference full paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700035.pdf).

The first two sources directly motivate the proposed candidate. The latter
two informed its limited scope: leave temporal inference fixed so the next
comparison tests probability fusion alone. Neither boundary method is proposed
as an untested automatic fallback after a failed score.

## Exact proposed interface

For aligned raw probability vectors in H/A/D order, define

`q(f) = 0.5 * p_original(f) + 0.5 * p_pretrained_latent(f)`.

Use R37's original fallback to `p_original` for unsupported latent rows. Replace
only slot0 by `q` when calling `decode_motion_dead_trim`; retain the other five
probability arrays, feature tables, smoothing, penalties and rescue rules.
Compute the original selected8 sequence independently, then apply exactly
`live_overlay(reference, alternate)`. D/X stay exact, and an alternative D/X
cannot overwrite reference live ownership. No labels enter this inference.

Arithmetic fusion guarantees a valid probability vector when both inputs are
valid and halves every component's displacement from the original head:
`q - p_original = 0.5 * (p_pretrained_latent - p_original)`.
It also preserves team-swap equivariance if both component heads have it.
These are algebraic properties, not scoring guarantees. The downstream
decoder is nonlinear, errors are correlated, the heads share labels and many
features, and neither output should be described as calibrated truth.

Existing code supports this experiment without source-model retraining:
[R37 inference](research_player_representation.py),
[latent interpolation and overlay](player_representation.py),
[selected8 decoder](motion_dead_trim.py), and
[shared fit/predict class order](temporal_candidate.py).
The 79-frame source footprint remains inside the existing 250-frame training
exclusion; averaging does not expand it.

## Small prospective protocol

Freeze the candidate, input/checkpoint hashes and objective **before** decoding
or evaluating its outputs. Reuse seed42 OOF probabilities from the exact saved
folds. Include selected8 and direct R37 substitution as named comparators in
all three unchanged protocols. Comparing against selected8 alone would omit
the already-observed R37 aggregate record (.754286/.720000/.691892 versus
selected8 .741176/.685714/.670270); any claim of a new high must identify which
comparable historical record it exceeds.

No fitted fusion weights, per-case exceptions, true endpoints or frame-number
rules. If the frozen screen permits confirmation, refit the same latent head
under the original settings for seeds43/44, replay saved-model probabilities,
and compare the same complete decoder. Check primary score, macroF1,
short-possession matches, spurious/missed possessions, D-mask identity and
per-block regressions. Preserve failures and report the adaptive history.

Historical substantial-gain gates must not be silently relaxed or described
as passed. If the current user objective is interpreted as an incremental
score target, write that distinct prospective objective and its protections
explicitly. The five-minute public window has only17 possessions and has been
reused extensively; confirmation seeds/protocols are correlated development
checks. They cannot establish private full-match improvement.

Only a candidate that meets its stated goal should advance to full-match
inference, two byte-identical builds, official format validation and a concise
reproduction report. No source above removes the need to run those checks.
