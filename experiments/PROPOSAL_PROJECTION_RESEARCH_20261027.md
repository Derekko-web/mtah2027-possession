# Research: project the latent proposal onto the reference possession order

Prepared 27 October 2026. This is a source-based proposal for one cheap
inference experiment. It fits and scores no model, changes no prior verdict,
and does not establish an improvement.

## Candidate and rationale

Use the **final R37 pretrained-latent live-overlay prediction** as the proposal
and the independently computed selected8 prediction as the reference. Within
each contiguous reference H/A component, retain the reference's ordered run
labels and find their best assignment to frames. First minimize disagreement
with the proposal; among those optima, minimize disagreement with the reference.
Require every reference run to occupy at least one frame.

R37's final prediction already contains the complete selected8 decoding and
live-overlay logic. Using it as hard evidence preserves that composite
information rather than reconstructing endpoints from raw latent probabilities.
The proposed projection can discard an extra proposed ownership island while
retaining some endpoint changes. Whether it keeps useful changes rather than
wrong ones is an empirical hypothesis. Its stronger structural restriction
replaces R37's free modification of the run sequence; it does not train another
model or choose a mixture coefficient.

This merits one bounded screen because it is inexpensive and tests a precise
mechanism suggested by R37's aggregate improvement plus extra buffered segment.
It is not especially promising for recovering a possession absent from the
reference transcript, because the constraint makes such recovery impossible.

## Primary sources and scope of the adaptation

**Bojanowski et al., ECCV 2014, Weakly Supervised Action Labeling in Videos
Under Ordering Constraints.** Read sections 2.1 and 3.2. The admissible sequence
starts at transcript state 1, ends at state K, and either stays or advances
one state per frame. Their dynamic program minimizes arbitrary additive
frame/state costs in O(TK). This directly supports our constraint set and
solver; Hamming costs are our application-specific choice. Their transcript
comes from annotation, whereas ours is an imperfect algorithmic prediction.
Their learning method and reported improvements are not reproduced here.
[Author-hosted paper](https://www.di.ens.fr/~fbach/bojanowski14weakly.pdf),
[Author project](https://www.di.ens.fr/willow/research/actionordering/).

**Richard et al., CVPR 2018, NeuralNetwork-Viterbi: A Framework for Weakly
Supervised Video Learning.** Read sections 5.1 and 6.5. The action-alignment
task supplies an ordered action sequence at inference and estimates its
segment lengths through constrained decoding. The full method also models
class likelihoods, lengths and grammar probabilities. Our proposal uses
only the alignment idea: it is a hard-label nearest-feasible projection,
not a faithful implementation of their likelihood model, training loss or
duration prior. Their known-transcript results do not validate our predicted
transcript or guarantee better possession IoU.
[Conference paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Richard_NeuralNetwork-Viterbi_A_Framework_CVPR_2018_paper.pdf),
[Accessible method text](https://arxiv.org/html/1805.06875).

## Exact proposed objective and algorithm

Process every validation block independently. Full-match inference must reset
at the same canonical half/gap boundaries used by the reference. Split a block
into maximal contiguous components whose reference labels are H/A. Preserve
every reference D/X value exactly. No component can cross D/X or a block reset.

For one component, let its reference labels be `r[0:T]`, final R37 proposal
labels be `p[0:T]`, and reference run transcript be `c[0:K]`. Consecutive
transcript labels differ. Choose frame states `z[t]` with

`z[0]=0; z[T-1]=K-1; z[t]-z[t-1] in {0,1}`.

The output label is `c[z[t]]`. These conditions force all K states to occur
in order with positive duration. The reference itself is a feasible path,
so a path always exists. For K=1, the output is necessarily unchanged.

Minimize the ordered pair

`(sum_t 1[c[z[t]] != p[t]], sum_t 1[c[z[t]] != r[t]])`.

An exact equivalent scalar frame/state cost is

`cost(t,k) = (T+1)*1[c[k] != p[t]] + 1[c[k] != r[t]]`.

The factor T+1 is derived from lexicographic dominance: total secondary
cost is at most T, so one additional primary mismatch can never be compensated
by reference agreement. It is not a fitted regularization strength.

Set `D(0,0)=cost(0,0)` and other initial states to infinity. For later frames,

`D(t,k)=cost(t,k)+min(D(t-1,k),D(t-1,k-1))`.

Require the last state at the final frame, and backtrack saved predecessors.
Choose stay when predecessor costs are exactly equal. Use integer arithmetic
and an adequately bounded infinity sentinel; int64 avoids overflow when
component lengths make the quadratic-scale aggregate cost exceed int32.
No duration model, minimum length beyond one frame, transition penalty,
boundary radius, smoothing or post-projection free decoder is added.

## Relationship to existing repository work

No identical hard-label Hamming projection appeared in the inspected
experiment source, plans or reports. The closest implementations are:

- [Boundary relocation](boundary_refinement.py): fixes run order and optimizes
  old pooled coarse/fine log likelihood inside radii 13 or 25. It also allows
  boundaries involving D to move in that historical screen.
- [Directed handoff relocation](handoff_hazard.py): preserves D boundaries and
  live topology, but optimizes old emission likelihood plus a trained directed
  hazard inside a bounded radius.
- [Nested endpoint calibration](ENDPOINT_CALIBRATION_REPORT.md): learns
  training-only timing offsets and preserves topology with constraints.
- [Ordinary switch-penalty decoding](transition_penalty.py): changes run count
  and order by trading emission likelihood against state changes.

Thus topology-preserving dynamic programming is established project machinery.
The distinct hypothesis is specifically **global within-component projection
of the final latent proposal onto the current reference transcript**, using
the exact lexicographic Hamming objective above. This is not evidence that
prior boundary work failed merely because its radii were too small.

## Guarantees, failure modes and useful checks

The candidate preserves reference D/X, reference live-run count and order,
and the first/last owner of each component. Stitching blocks therefore also
preserves their reference boundary transitions. If proposal equals reference,
the output is exactly reference. If the proposal already has the same
transcript, it can be represented with zero primary cost and is retained
exactly. The solver never needs true public labels or endpoints.

Preserving run count does **not** preserve the number of matched or spurious
possessions under the official scorer: different endpoints can lose IoU
matches. It also cannot repair a missing run or remove an incorrect reference
run, and can shrink a required run to one frame. Unrestricted boundary motion
may align a reference run to the wrong repeated-label proposal island.
Hamming agreement weights long intervals heavily, whereas spellF1 weights
possessions and thresholds IoU. These are reasons to evaluate the candidate,
not to claim its constraint is a scoring guarantee.

Before scoring, compare the solver with exhaustive enumeration on small
sequences; verify identity, compatible-proposal retention, compulsory state
visitation, deterministic ties, team swaps, unchanged D/X, independent block
resets and exact topology. Keep one frozen candidate rather than trying
probability losses, radii, duration penalties and projection weights in a
hidden sweep. Report all three protocols against selected8 and the direct R37
proposal, including short matches, macroF1, unmatched/spurious cases and
per-block deltas. Existing historical substantial-gain rejections remain
unchanged; repeated public development is not private-match validation.
