# R55 boundary-risk implementation review

Reviewed independently on 28 October 2026 before the R55 freeze. No native
predictions, score screen, alternative model fit, or R54 outcome inspection was
performed. Existing implementation files were not modified.

No blocking implementation or mathematical defect was found in
`boundary_risk.py`, `research_boundary_risk.py`, their prospective plan and the
associated research note. All 13 tests on invented data pass (0.451 seconds).
An independent SHA-256 check of 84 accessible input bindings in the draft plan
found no mismatch. One inherited reserved-window binding was deliberately not
opened by this review.

## Mathematical and inference checks

The relative cumulative log-odds implementation in `boundary_risk.py:57–65`
equals the proposed full two-span likelihood after canceling a common positive
factor. Boundary candidates are precisely `1..N-1`; frame 0 and frame N-1 have
fixed owners in every candidate and correctly cancel. Stable max-shifted
normalization preserves the uniform prior interpretation. The conditional H/A
normalization and numerical clipping occur in the planned order.

For threshold `k/10`, the left bounds are
`ceil(k*b/10) <= t <= floor(10*b/k)`. The corresponding right bounds are
`N-floor(10*(N-b)/k) <= t <= N-ceil(k*(N-b)/10)`. Lines 95–102 compute these
using integer division and intersect them with the feasible interval before CDF
subtraction. These inequalities are correct for half-open anchored intervals.
Point-mass rational endpoint tests and independent direct enumeration verify the
calculation, including inclusive threshold comparisons.

The chronological pass keeps each original run identity and visits it once.
Previously updated left endpoints and still-original right endpoints define the
local conditioning interval. Every chosen boundary is strictly between its
neighbors. Exact computed utility ties use proximity to the original selected8
boundary, then the smallest position, matching the prospective plan. The helper
also handles an infeasible external tie anchor; the traversal itself retains a
feasible original endpoint at each update.

Live components are split independently within each existing decode block.
Their exterior endpoints, ordered labels, positive run count, D/X frames and
frames outside the blocks remain unchanged. The tests cover team exchange,
determinism, separate component/block decoding, reversed independent block order,
single runs and empty inputs. None of these structural invariants guarantees
retained official greedy matches; the plan correctly retains the selected8
matching safeguards.

## Evidence, scoring and provenance checks

`research_boundary_risk.py:106–117` uses the existing R37 raw three-state head,
including its unsupported-latent fallback, rather than its decoded labels as
unary evidence. The reused `latent_probability` function loads all 13 seed42
checkpoints, checks their original training-index/target/weight records, predicts
using the unchanged team-symmetric function, independently reloads each model,
and asserts exact saved raw probabilities. Seed42 performs zero new fits.
Neutral probabilities outside public decode blocks never enter boundary edits.
The unchanged R37 proposal is separately decoded and checked against its saved
seed42 labels for an honest diagnostic comparison.

The method receives no targets. Ground truth is used only by the unchanged
training-recipe verification, ordinary public validation and scorer, as in the
existing frozen experiments. Only the boundary-risk method's three pairs can
determine eligibility; the R37 comparator cannot be selected. The selected8
safeguards and strict seed42 historical-record check are retained. Confirmations
require a passing boundary-risk screen, and replay requires byte-identical raw
probabilities, final labels, scores and gates.

The runner refuses execution unless the R54 screen has a completed manifest and
a failed selection. R54's trigger artifacts are captured by hash at execution
and checked against those bindings on replay. Since its outcomes do not yet
exist at preparation time, complete the separate R54 artifact audit before
launching this conditional screen. The source and plan freeze must be made only
after this review and the completed synthetic checks; no native result may
change the proposed utility, clipping, evidence source, traversal or tie rule.

## Limits and closeout

This is the stated conditional local surrogate, not a global optimizer of
expected official spellF1. Repeated same-team occurrences, uncertain outer
endpoints and later adjacent-boundary updates break such a claim. Correlated
25 Hz unary products can overconcentrate the chosen pseudo-posterior. The
research note and implementation explicitly disclose these limitations.

This audit approves no outcome or promotion. Retain R54 failures, append every
R55 score row while preserving the ledger prefix, verify final manifests and
exact replay, and apply all confirmation/full-candidate requirements if this
single fixed screen passes. A failed conjunction closes the stated interface.
