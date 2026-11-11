# Conditional boundary risk: a limited inference hypothesis

Prepared 28 October 2026. This note reads primary sources and existing
implementations. It runs no fit, prediction screen, public scoring, or label
inspection. R54 native TabICL validation remains the active model experiment.

## Verdict

Expected threshold-IoU utility is a distinct objective from the repository's
previous likelihood, hazard, endpoint-offset and hard-Hamming boundary methods.
The proposed CDF calculation is exact for **two adjacent spans with fixed outer
endpoints and a specified distribution of their shared boundary**. Applying it
once from left to right within a longer transcript is only a conditional local
surrogate. It must not be described as an exact optimizer of expected official
spellF1, or as a source-established route to a record score.

A single inexpensive, frozen screen is defensible if a further decoder
experiment is wanted after the native model finishes. Confidence is limited:
this reuses R37's existing evidence and S8's imperfect transcript, cannot add a
missing possession, and revisits a heavily explored family of boundary methods.
The main new hypothesis is the **utility**, not the structural constraint or a
new source of ownership information. There is no reason here to relax the
selected8 safeguards or to run a sweep after rejection.

## What the primary sources establish

Kumar and Byrne, NAACL 2004, section 3 and equations 3–6, select a decision by
expected application loss under a distribution over alternatives. Their
implementation approximates both the hypothesis space and the unavailable true
distribution using model-generated alternatives. This supports distinguishing
the most probable output from the output with the best expected task utility;
it does not supply calibrated possession-boundary probabilities.
[Conference paper](https://aclanthology.org/N04-1022.pdf).

Dembczyński et al., NeurIPS 2011, introduction and section 3, formulate expected
F-measure inference and show that their exact binary-label-vector solution
requires joint information beyond ordinary marginal probabilities. Their
theorem is not an interval-detection theorem: its numerator is an aligned binary
overlap, whereas this repository's scorer creates a greedy interval matching.
Therefore the GFM algorithm cannot simply be applied to frame H/A probabilities
and called an exact spellF1 decoder.
[Conference paper](https://proceedings.neurips.cc/paper/2011/file/71ad16ad2c4d81f348082ff6c4b20768-Paper.pdf).

The existing transcript research already reads the relevant ordering/alignment
sources. Their stay-or-advance constraint supplies useful machinery, but our
transcript is predicted rather than known. See
[PROPOSAL_PROJECTION_RESEARCH_20261027.md](PROPOSAL_PROJECTION_RESEARCH_20261027.md).

## Exact local calculation, derived here

Use half-open integer intervals. Fix `L < R`, left label `c0`, and opposite right
label `c1`. A possible shared boundary is `t in {L+1,...,R-1}`. Define positive
unary potentials `u_i(c)` from one frozen ownership head and a uniform prior on
these boundary positions:

```
log w(t) = sum_{i=L}^{t-1} log u_i(c0)
         + sum_{i=t}^{R-1} log u_i(c1)
q(t) = exp(log w(t) - logsumexp_s log w(s)).
```

This is a normalized, explicitly chosen conditional model. Products of
discriminative frame probabilities do not establish that it is the true
posterior. Consecutive frames share observations and derived features, so
interpreting their product as independent evidence can overstate certainty.
Use float64 cumulative log potentials and stable normalization. Clipping unary
potentials below `1e-12` follows the existing likelihood code's numerical
convention, not a fitted probability calibration.

This is particularly concerning at 25 Hz: adjacent rows and their temporal
features are strongly dependent, yet each row contributes another log-odds
term. For example, shifting a boundary across `m` repeated rows with odds `r`
changes the likelihood ratio by `r**m`. A long sequence of modest repeated
preferences can therefore concentrate this pseudo-posterior close to its
likelihood mode. There may be little useful distributional information left
beyond the old MAP evidence. This is a theoretical concern, not an inspected
native-data result. Even a point-mass posterior does not force the returned
boundary to equal MAP: threshold-IoU utility has plateaus, and the reference
tie rule can choose another position on the same maximal plateau.

For candidate `b`, compare its two spans to the corresponding spans under `t`.
At threshold `theta`, the anchored left IoU qualifies exactly when

```
ceil(theta * (b-L)) <= t-L <= floor((b-L)/theta).
```

The anchored right IoU qualifies exactly when

```
R - floor((R-b)/theta) <= t <= R - ceil(theta * (R-b)).
```

Intersect each interval with `[L+1,R-1]`. If `F(x)=sum_{t<=x} q(t)`, its
probability is `F(high)-F(low-1)`, or zero for an empty interval. Sum the left and
right probabilities over the five official thresholds and divide by ten. This
is the expected fraction of the two **corresponding** spans that qualify. The
normalizing constant does not affect the maximizing `b`.

For `theta=k/10`, `k in {3,4,5,6,7}`, derive integer bounds with integer division
and ceiling division, rather than floating-point `ceil` near an exact threshold.
Once `q` and its CDF are available, all `R-L-1` candidates cost linear time for
the fixed five thresholds. No radius or probability temperature is needed.

For exactly two opposite-team spans and correct fixed outer endpoints,
corresponding-span matches have no same-team competitors. If both actual and
predicted sequences contain only these two spans, their F1 is matched-count/2.
Under precisely those assumptions this calculation also maximizes expected
two-span official F1. Those assumptions do not hold for a whole imperfect
predicted match transcript.

## Why a longer-transcript pass is not global Bayes-risk optimization

The official implementation is the primary metric specification:
[`evaluation/possession_metrics.py`](../evaluation/possession_metrics.py),
`spell_overlaps`, `matched_gt_flags`, `match_spells`, and `spell_scores`.
It forms all overlapping same-team interval pairs, sorts by descending IoU,
then greedily consumes each ground-truth and prediction interval at most once.
It averages F1 at .3/.4/.5/.6/.7.

With a fixed predicted run count `K` and a true run count `G`, the official
F1 at any threshold is `2*TP/(K+G)`. Thus increasing the **actual greedy** match
count would improve that metric while those counts are fixed. But summing
indicators for a chosen run-index correspondence need not equal the greedy
match count: repeated same-team spans may overlap a different occurrence, and
predicted and true transcripts need not have the same run count or order.
Keeping the predicted transcript does not guarantee retained matches.

Even if correspondence were certain, a local pass freezes neighboring
boundaries instead of integrating their uncertainty. Moving one boundary
changes one end of each adjacent span. A later update moves another end of a
span whose earlier decision was already made. The objective, feasible local
domain, and conditional distribution used later therefore change. One pass
does not solve the joint optimization, and changing traversal order can change
the result. Repeating until convergence would not cure the missing global
posterior or matching treatment and would add another protocol choice.

Two invented counterexamples, independently checked mathematically:

- Let truth be H/A/H on `[0,30)` with boundaries 10 and 20, while the initial
  second predicted boundary is 5. The first update is constrained to positions
  1..4. Even if the second update later reaches 20, the first boundary cannot
  be repaired by a one-pass algorithm. The first true spell can qualify at no
  more than .3 and .4; the correct segmentation qualifies all three spells at
  all five thresholds.
- Let truth be `H:[0,1), A:[1,2), H:[2,12)` and prediction be
  `H:[0,9), A:[9,10), H:[10,12)`. Every same-index IoU is below .3. Nonetheless,
  official matching pairs the first predicted H with the last true H at IoU
  `7/12`, qualifying at .3/.4/.5. A sum of corresponding-index indicators is
  therefore not the official true-positive count, even with equal alternating
  label transcripts and positive run lengths.

No probability temperature should be chosen from the observed validation score
to make the local distribution more useful. That would be another adaptive
decoder search, not evidence supplied by the MBR papers.

## Relationship to failed repository methods

- `boundary_refinement.relocate_boundaries` jointly maximizes pooled coarse/fine
  likelihood within radii 13 or 25, with fixed labels. Its historical screen
  did not improve primary and lost D recall.
- `handoff_hazard.relocate_handoffs` adds trained directional hazard evidence to
  bounded likelihood relocation while retaining D positions/live topology.
- Nested endpoint calibration estimates timing corrections from eligible inner
  training predictions. Its report rejects all four prospective variants.
- R43 `transcript_projection.align_component` globally minimizes hard R37
  proposal disagreements, then reference disagreements. Its .752941 buffered
  result falls below R37's .754286 record, and mean-fold checks fail.

The mathematical distinction is real: likelihood selects the most probable
boundary, Hamming minimizes expected frame disagreement, and threshold-IoU
utility can select a different boundary. That does not make this an independent
model family or a fresh validation set. Existing failures reduce its priority.

## One concrete protocol, if implemented separately

Use one method only, with no radius, smoothing, temperature, duration penalty,
iteration count, or alternative utility sweep. This is a proposal, not an
executed or frozen score experiment.

The implied likelihood temperature is exactly one. It is never tuned.

1. Reproduce selected8 and R37's existing held-out raw probability arrays for
   each protocol/seed, using exact cached-checkpoint replay. Keep every original
   label exclusion and model fit unchanged. Do not refit for seed42.
2. Use conditional H/A probabilities from R37's saved raw three-class head:
   `u(H)=pH/(pH+pA)` and `u(A)=pA/(pH+pA)`, then clip/renormalize for numerical
   safety. Use raw head evidence, with no added temporal smoothing or gap
   correction. This intentionally tests an explicit unary model rather than
   claiming to reproduce all evidence inside the composite selected decoder.
3. Start from selected8 labels. Independently split each existing decode block
   into its selected8 live components, retaining every D/X frame exactly. Keep
   component endpoints and all ordered run labels/counts.
4. Within a component, process original transcript boundary identities once in
   chronological order. For boundary `j`, use the **currently assigned** left
   neighbor boundary `L` and the unchanged/right current neighbor boundary `R`.
   Compute the local distribution and expected threshold utility above over
   every integer position in `[L+1,R-1]`. This makes the specified greedy
   conditional algorithm deterministic and ensures positive run lengths.
5. Maximize utility; exact ties prefer the current boundary, then the closest
   position to it, then the smallest position. Do not introduce an unreported
   epsilon whose magnitude trades utility for proximity. Copy the result back
   into the fixed transcript. A one-run component remains unchanged.
6. Before any score, test CDF results against direct enumeration on small
   invented examples, rational threshold endpoints, uniform and concentrated
   boundary distributions, deterministic ties, team swaps, no crossing,
   independent components/blocks, and exact D/X/topology preservation.
7. Evaluate the single frozen candidate against selected8 and R37 across all
   three existing protocols. Apply the current strict-record and secondary
   guards without alteration. Append every score and rejection, then replay
   exact source/probabilities/labels/official metrics. Only a conjunction pass
   advances to seeds43/44 and the full candidate workflow.

If this screen fails, its natural follow-up is not another temperature or
boundary radius. A materially different ownership signal or properly trained
joint boundary model would be needed to justify more public development.
