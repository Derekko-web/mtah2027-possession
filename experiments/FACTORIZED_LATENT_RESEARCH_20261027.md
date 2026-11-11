# Research: conditional live ownership from transferred player features

Prepared 27 October 2026 after the fixed R41 arithmetic-fusion screen failed.
This note proposes a distinct learning interface. It performs no model fitting,
score computation or promotion, and changes no prior plan or selected artifact.

## Recommendation and distinction from prior failures

Fit a fixed **binary H/A classifier on eligible live training frames**, using
the original 102 ordered inputs plus R37's 64 frozen pretrained player-set
features. Retain the original slot0 D probability when composing the new
H/A/D input to the unchanged selected8 decoder. Apply the existing final
live-only overlay against an independently computed selected8 reference.

Use one matched original 102-feature binary CatBoost control and one 166-feature
binary CatBoost candidate. This separates the representation's contribution
from the change of learner/objective. A second XGBoost arm is possible, but
would enlarge the comparison; the two CatBoost arms are the smallest useful
screen. Freeze both before observing their outputs, and do not search depths,
iterations, weights, thresholds or smoothing windows.

Factorization itself is **not new to this repository**:
[the earlier factorized baseline](train_factorized_baseline.py) and
[the selected fast control head](control_refinement.py) already separate
live ownership from dead-ball evidence. The new hypothesis is specifically
whether the pretrained representation's signal works better under a
conditional, live-only loss than R37's joint three-class loss, using a fixed
Ordered CatBoost learner. Existing factorized and ownership failures remain
relevant negative evidence; this is not a claim that a generic binary split
has never been tried.

R37 jointly trained H/A/D using 102+64 features and replaced all three slot0
probabilities. R41 averaged that joint head with the original joint head.
The present method learns a different conditional target and prevents the new
head from assigning any raw D probability. It neither changes R41's averaging
weight nor reruns its rejected hypothesis under another name.

## Primary sources read

**Conditional decomposition.** Morin and Bengio's AISTATS 2005 paper constructs
a normalized multiclass distribution from binary conditional decisions along
a hierarchy; sections 3–4 give its probability decomposition. This supplies a
primary example of the general factorization principle, not soccer-specific
evidence. Their motivation was language-model speed and their hierarchical
model had somewhat worse generalization than the flat neural model. Therefore
the source supports a valid interface, not an assertion that binary hierarchy
will score better here.
[Conference paper](https://proceedings.mlr.press/r5/morin05a/morin05a.pdf).

**Football target semantics.** Bassek et al.'s public DFL data descriptor
separately defines ball possession (home/away control) and match state
(active/inactive according to referee interruptions). This makes a live/dead
plus owner decomposition meaningful for soccer. It does not establish exact
compatibility with this challenge's loose-ball rules or validate the frozen
R37 representation's transfer.
[Primary data descriptor](https://www.nature.com/articles/s41597-025-04505-y.pdf).

**Learner choice.** Prokhorenkova et al., NeurIPS 2018, section 4 describes
prediction shift in ordinary gradient boosting and the permutation-based
Ordered approach; section 6 compares Ordered and Plain modes and reports
stronger Ordered benefits on smaller datasets. The categorical-feature
algorithm is a separate contribution; no categorical advantage is claimed for
our numeric features. Their theory and experiments do not imply superior
possession interval scores, and the small number of independent possessions
remains a limitation even with thousands of correlated frame rows.
[Conference paper](https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf).

**Executable parameter semantics.** The official documentation lists Logloss
as a trainable classification objective, describes Ordered as usually useful
on small datasets, and defines `l2_leaf_reg` as the L2 cost coefficient.
These support an explicit binary configuration rather than relying on defaults.
Ordered refers to internal training permutations; it is not temporal
chronological validation and does not replace the existing blocked exclusions.
[Objectives](https://catboost.ai/docs/en/concepts/loss-functions-classification),
[Parameters](https://catboost.ai/docs/en/references/training-parameters/common).

## Exact factorization and its limits

Let `d0(f)` be the saved original slot0 raw D mass and let `h(f)` be the
symmetrized binary head's estimated home share conditional on live play.
Compose

`qH=(1-d0)*h; qA=(1-d0)*(1-h); qD=d0`.

This follows the probability chain rule for `live={H,A}`; it needs no conditional
independence assumption. Numerically, it guarantees nonnegative normalized
H/A/D vectors and exact original raw D mass when its inputs are valid. It does
not guarantee calibrated probabilities: the estimated components may be
miscalibrated, the source representation may transfer imperfectly, and
symmetry augmentation changes the training distribution.

The new binary loss has no D examples and no D output channel, so its trees
allocate fitting capacity to the H/A distinction instead of also predicting
stoppages. Whether that helps rather than removes useful regularization is
the experimental hypothesis. Preserving raw D mass does **not** prove the
complete decoder's D labels remain identical because rescue and consensus
logic also read ownership probabilities. The final `live_overlay` is therefore
essential: preserve all reference D/X and reject alternate D/X at reference
live frames.

## Fixed bounded fitting protocol

Reuse the original three protocols and exact fold indices. Within each fold,
select only its explicitly eligible H/A training labels; never filter validation
rows using their true class. Save the chosen indices/targets. Augment every
training row with its team swap and flipped binary label. With A coded 1 and
H coded 0, predict away share as
`a=.5*(pA(x)+1-pA(swap(x)))`, then `h=1-a`.
This follows the existing fast-head convention and balances augmented class
counts exactly without new class weights. Assert model class order and swap
the h__/a__ latent feature blocks along with all original paired features.

Both arms use the pinned installed CatBoost runtime and the historical
declarations: `loss_function=Logloss`, `boosting_type=Ordered`, depth 4,
iterations 300, learning_rate .04, l2_leaf_reg 10, random_seed 42, thread_count 2,
nan_mode Min, allow_writing_files False and verbose False. Train all 300 iterations;
no early stopping, held-out eval-set tuning, calibration, segment weighting,
class-weight search or feature selection. Keep all other runtime defaults
identical and record the resolved parameters. Use identical handling of
unsupported latent observations and specify fallback before the run.

Reuse the target-free R37 features, interpolation/support rules and 79-frame
source footprint within the existing 250-frame exclusion. Unsupported inference
rows use the full original probability vector. Recompute selected8 as the
reference, replace only slot0 with q, run its exact decoder and apply the final
live-only overlay. No inference stage may read true live status, true runs,
true endpoints, manual corrections or frame-specific exceptions.

Before score interpretation, verify saved/reloaded model predictions,
team-swap equivariance, probability normalization, raw D identity,
unsupported-row fallback, block resets and final reference D/X identity.
Report the matched 102-versus-166 contrast and all original primary/secondary
metrics, not only frame-level ownership accuracy. A new high must exceed the
relevant comparable recorded score, including R37, and satisfy the prospective
goal as written. Any incremental objective must be explicitly separate from
historical substantial-gain criteria; those previous rejections remain valid.

Repeated seeds and public protocols reuse the same 17 possessions. No source
or successful development comparison can establish private full-match
improvement or optimality. If the frozen screen passes, preserve its artifacts
and complete the same seed confirmation, replay, full-build and format checks
required elsewhere in this repository before changing the selected model.
