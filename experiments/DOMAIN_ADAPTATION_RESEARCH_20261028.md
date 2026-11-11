# Architecture-preserving native adaptation of the R37 encoder

Research-only recommendation, 2026-10-28. **Conditional go for one paired,
public-native-supervised fine-tuning experiment; do not prioritize unlabeled
Mean Teacher consistency when the permitted outer-training labels directly
supervise the desired H/A distinction.** The proposed method retains external
source BCE rehearsal and the exact R36 architecture. It is a new learning
interface, not a reinterpretation of R36/R37/R39 failures. No gain is claimed.

Only this new note was written for this research task. Local reports, plans and
implementation text and public primary papers were read. No native input or
target arrays, sealed/private/reserved content, model weights, or media were
opened in this research task. No acquisition, feature processing, training,
inference, scoring, dependency change, or ledger edit occurred. R56 and the
separate MPS preparation were not touched. The numeric recipe below is a
prospective design choice, not a paper-validated soccer setting; execution still
requires a separate frozen plan, implementation review, and synthetic gate.

## What the existing model actually learns

The name `MaskedPlayer` does **not** identify a masked autoencoder.
[The network](/Users/derekko/Projects/more-than-a-hack-2027/experiments/masked_player_network.py:12)
has an XY stem, null token, masked spatial attention, per-team mean/count pooling,
four temporal convolution blocks, and a scalar output. Lines 35–39 subtract the
two shared oriented team scores. Most decisively,
[`step`, lines 42–46](/Users/derekko/Projects/more-than-a-hack-2027/experiments/masked_player_network.py:42)
optimizes `binary_cross_entropy_with_logits` against external H/A targets.
There is no reconstruction head, coordinate decoder, masked-coordinate target,
or reconstruction loss to resume unchanged on unlabeled native observations.
Adding one would be a different architecture/objective and is not proposed.

The existing external corpus maps direct live home ownership to 1 and away to 0
in
[`prepare_masked_player_data.py`, lines 50–75](/Users/derekko/Projects/more-than-a-hack-2027/experiments/prepare_masked_player_data.py:50).
Masks/crops/bursts/jitter are observation corruption, created and hashed before
that target join. The four training matches are WMX/WN1/WOH/WOY; WPY/WQQ/WR9
are disjoint from fitting. Their historical reuse means they are not untouched
validation. The exact positive-logit-is-H convention must remain distinct from
the downstream XGBoost coding H=0, A=1, D=2.

R37 freezes the external temporal checkpoint:
[`encode_player_representation.py`, lines 30–51](/Users/derekko/Projects/more-than-a-hack-2027/experiments/encode_player_representation.py:30)
loads it, runs evaluation-only extraction, and captures the 32+32 states before
the scalar output. Four pitch reflections are averaged. The subsequent
[`research_player_representation.py`, lines 56–103](/Users/derekko/Projects/more-than-a-hack-2027/experiments/research_player_representation.py:56)
fits XGBoost on the fixed 102+64 columns; it never backpropagates native labels
through the encoder. Native encoder fine-tuning is therefore a distinct,
untried hypothesis in the inspected implementation/plans.

| Previous experiment | What changed | Why this proposal differs |
|---|---|---|
| R36 | External supervised temporal encoder; scalar native correction audit failed | Encoder would now receive permitted native outer-train H/A supervision; scalar override remains closed |
| R37 | Frozen encoder features plus public-trained XGBoost | The representation, not only the head, would adapt inside each outer fold |
| R38 | WASB ball-detector pilot | Different model, inputs, task, and supervision; no encoder adaptation |
| R39 | Ball-relative input features and new external fits; external gate failed | Exact player-only architecture and inputs retained; no ball feature rescue |
| R41 | Fixed fusion of unchanged R37 probabilities; rejected | No probability coefficient or decoder change is proposed |

Sources for those findings are the
[R36 report](/Users/derekko/Projects/more-than-a-hack-2027/experiments/MASKED_PLAYER_REPORT.md),
[R37 report/plan](/Users/derekko/Projects/more-than-a-hack-2027/experiments/PLAYER_REPRESENTATION_PLAN.json),
[R38 report](/Users/derekko/Projects/more-than-a-hack-2027/experiments/WASB_REPORT.md),
[R39 report](/Users/derekko/Projects/more-than-a-hack-2027/experiments/BALL_CONTEXT_REPORT.md),
and [R41 report](/Users/derekko/Projects/more-than-a-hack-2027/experiments/REPRESENTATION_ENSEMBLE_REPORT.md).

## Primary-source assessment

**Supervised adaptation is the more direct hypothesis here.** Yosinski et al.
distinguish fixed transferred features from fine-tuning and demonstrate that
source specialization can limit transfer. Kornblith et al. compare fixed-feature
classification with fine-tuning and find transfer gains depend on the dataset.
These are image studies, so they motivate an experiment, not an expected soccer
gain. R37's useful but imperfect frozen features are local evidence that makes
this distinction worth testing.
[Yosinski et al., NeurIPS 2014](https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html),
[Kornblith et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Kornblith_Do_Better_ImageNet_Models_Transfer_Better_CVPR_2019_paper.html).

Li, Grandvalet and Davoine explicitly study retaining source knowledge while
fine-tuning with limited target data, recommending a penalty toward pretrained
weights, commonly called L2-SP. Source rehearsal is a different retention
mechanism, not an implementation of L2-SP. The proposed single trial uses
source BCE and a small fixed adaptation budget; adding an L2-SP coefficient or
layer-freezing comparison would introduce another experiment and is excluded.
[Li et al., ICML 2018](https://proceedings.mlr.press/v80/li18a.html).

**Mean Teacher is technically applicable, but less direct.** Tarvainen and
Valpola construct a teacher by exponential averaging of student weights and
train against detached, perturbed teacher predictions. They use probability
MSE, with a consistency ramp, and identify incorrect targets/consistency weights
as hazards. This preserves network topology but adds an actual new training
objective; calling it reconstruction continuation would be wrong. French et al.
extend self-ensembling to source-labelled/target-unlabelled visual adaptation.
Their confidence and class-balance choices are not transferable defaults for
H/A, and their paper warns that balancing can hurt imbalanced targets. No such
extra mechanism is recommended in this trial.
[Mean Teacher, NeurIPS 2017](https://proceedings.neurips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf),
[Self-Ensembling, ICLR 2018, author-hosted paper](https://ueaeprints.uea.ac.uk/id/eprint/66432/1/1706.05208.pdf).

Gururangan et al. show gains from continuing language-model pretraining on
domain/task text. This is only an analogy for domain relevance: their language
models already have a token-prediction pretraining objective. It supplies no
missing reconstruction objective for this supervised soccer classifier and no
validation rule for it.
[Don't Stop Pretraining, ACL 2020](https://aclanthology.org/2020.acl-main.740/).

Relevant trajectory papers reinforce the distinction. Rudolph and Brefeld's
T-MAE uses an explicit encoder/decoder and reconstructs masked trajectory
segments. Its soccer experiment requires complete players/ball, speed, and role
information, unlike this anonymous partial-set input. Chen et al.'s Traj-MAE
also reconstructs masked trajectories/maps for driving. Those findings do not
make the existing R36 checkpoint a reconstruction model. AdaSports-Traj is a
later sports-domain-adaptation preprint with role/domain adapters and
contrastive objectives; those architectural additions are outside this proposal.
[T-MAE, Machine Learning 2025](https://link.springer.com/article/10.1007/s10994-024-06647-3),
[Traj-MAE, ICCV 2023](https://arxiv.org/abs/2303.06697),
[AdaSports-Traj, 2025 preprint](https://arxiv.org/abs/2509.16095).

## Why unlabeled consistency is not the selected next trial

This assessment is an inference from the code/data setting, not a finding that
Mean Teacher generally fails. A stable teacher can repeat a confident source
error, or become consistently neutral. Native missingness, duplicated/fragmented
slots, coordinate errors, and possession ambiguity differ from source corruption.
Further masking can remove the only discriminating player movement. Reflections
and team swaps have known mappings; arbitrary crop/burst views need not retain
enough evidence to make equal-confidence H/A predictions sensible.

The unlabeled native mixture also includes dead/loose-ball/restart states for
which this binary source head has no calibrated third class. Enforcing ownership
consistency there is not equivalent to learning possession. Confidence filtering
can preferentially keep confident errors, while uniform class balancing can
change a real ownership prior. Source rehearsal prevents neither target-specific
collapse nor semantic label mismatch by itself.

There is no supported rule here for choosing epochs by the lowest target
consistency, entropy, or highest agreement. UDA model selection is a separate
problem; even source validation need not rank target performance correctly.
General adaptation guarantees require assumptions about domain/label relations.
[You et al., ICML 2019](https://proceedings.mlr.press/v97/you19a.html),
[Ben-David et al., AISTATS 2010](https://proceedings.mlr.press/v9/david10a.html).

Outer-training native H/A labels are already authorized for the supervised
head. Applying the same permitted labels to encoder BCE is clearer than
discarding them and treating a teacher's guesses as an additional target.
No unlabeled-consistency arm, target metric, epoch choice, or fallback rescue is
included below.

## One prospective protocol: source/native supervised continuation

This is the sole recommended method, with one diagnostic control. Freeze it
before adaptation inputs, outputs, or public scores. Preserve every earlier
verdict and selected8 artifact. All constants are design priors fixed for a
bounded first test, not selected from target outcomes.

### Fixed learner and loss

Initialize every neural fit from the **same immutable passing R36 temporal
checkpoint**, with all 24,321 parameters trainable. Architecture, masks, null
token, pooling, 31-sample temporal context, scalar antisymmetry and 64-dimensional
extraction point remain exact. No decoder, extra head, ball, identities,
velocities, dropout rate, layer choice, temperature or feature count changes.

Candidate loss is

`L = .5 * mean(BCE(source_logits, source_HA)) + .5 * mean(BCE(native_logits, native_HA))`.

Use exactly 400 AdamW steps, seed 42, float32, learning rate .0001, weight decay
.01, gradient clipping 1, and batch 32 comprising 16 source plus 16 native
windows. No scheduler, early stopping, EMA teacher, best epoch, or intermediate
checkpoint selection. Use the final step-400 student only. These choices shorten
and slow the original .001/4,000-step fit; they do not guarantee retention.

The paired source-only control starts from the identical checkpoint and performs
the same 400 updates with 32 source windows. The candidate's 16 source examples
are the first 16 examples of each control source batch; the control's second 16
replace the candidate's native branch. Reuse the first 400 rows of the existing
frozen R36 32-example source schedule and its four observation views/reflections.
This matches total examples, optimizer steps, initialization, shapes, and
continued-training opportunity while isolating native supervision as the changed
domain. A single source-only control can be reused across all outer folds;
its inputs and schedule have no fold dependence.

Use fixed N46/B32 tensors, no batch-dependent node trimming, with explicit saved
Torch RNG seeds for each update and separate saved sampling/augmentation RNGs.
Both arms' shared source examples must have identical post-corruption inputs and
stochastic masks. The source architecture has no batch normalization, so no
target-population normalization state or domain-specific BN is added.

### Native training pool and label use

Fit **13 distinct candidate encoders**, one per existing outer split (5/5/3).
For each encoder, admit only global 5-Hz-phase centers in that split's original
public training indices whose true center label is H or A. Join labels after
the observation windows and observation-only support eligibility are fixed.
Sample 8 H and 8 A centers per candidate update, uniformly within each class
with replacement. A missing class or empty admissible pool fails the entire
protocol, rather than changing the window/budget.

Every non-padding observation used by a fitting window must lie within that
same split's permitted training indices. Keep the original 250-frame exclusions;
do not borrow neighboring validation observations to make a window complete.
Mask padding is permitted only at known half boundaries, as in the existing
assembler. No true episode cut, dead-state context reset, or label-dependent
observation mask is allowed. D frames remain observable in permitted context;
D centers are excluded only from the explicitly binary encoder loss. The
downstream three-state head still learns D from its ordinary permitted labels.

Retain the existing observation-only support rule: at least four observed
players per team at the center and on average over the window. Native windows
use their existing masks; add only joint random x/y reflections and random
team swapping, with H/A target inverted on a swap. Reuse source mask/jitter
corruptions through the unchanged source views. Do not add another native
mask/crop/jitter distribution in this first trial. No observation support or
hyperparameter is changed after examining native outcomes.

The fitting worker receives only source and its own permitted training-window
payloads/targets. It must not receive query windows, query labels, full native
arrays, current reference predictions, possession boundaries, or score files.
Prove that replacing every label and observation outside the permitted training
payload leaves its inputs/schedule/checkpoint unchanged. Native labels are
supervised training data here, so this is not an unsupervised-adaptation claim.

### External retention gate and checkpoint selection

Evaluate the single final control and all 13 final candidate encoders on the
unchanged clean and masked R36 validation schedules for WPY/WQQ/WR9. No
observation or label from those three matches enters continuation training.
Report all centers, BA, both recalls, Brier and log loss, including blackouts.
These are disjoint external matches with historical reuse, not a new holdout.

For **every model, match, and view**, require BA >= .75, both recalls >= .65,
BA >= the immutable R36 temporal model's corresponding BA minus .005, and Brier
<= its corresponding Brier plus .005. These are explicit retention tolerances,
not a target-performance estimate. Any failure closes the whole method before
native query-feature extraction or possession scoring. No fold may select an
earlier checkpoint, fall back to the source model, or use a different setting.

Checkpoint selection is therefore trivial: final step 400, or rejection.
Native validation BCE, possession metrics, teacher agreement, target entropy,
feature variance, and source validation ranking never select epochs. Public
query labels are used only in the final frozen evaluation after neural weights,
query features, and head outputs have been saved and hashed.

### Features, heads, score rows, and safeguards

For each fold, freeze its adapted encoder before extracting both its head's
training features and query features. A feature cache from a globally
native-fitted encoder is invalid OOF evidence. A fold's training/query features
must share that fold's same encoder and the existing extraction function.

Retain the original 31 samples at 5 Hz, half padding, reflection averaging,
support interpolation and h__/a__ pair mapping. Inference still reads a radius
75 window; adjacent 5-Hz interpolation reaches at most 79 frames. The learned
weights depend on all permitted training observations, as any fitted estimator
does: **79 is a per-query feature footprint, not a claim that adapted weights
depend only on the query's local window.** Fit exclusion and the original
250-frame gap are both needed.

Use exactly the original 102 ordered plus 64 latent columns and unchanged R37
XGBoost hyperparameters, class weights and H/A augmentation. Each new head
receives only its outer-training H/A/D labels. Unsupported latent rows retain
original slot0. Keep the complete selected8 decoder and final live overlay,
with exact D/X preservation.

The seed42 screen contains 12 prespecified score rows: selected8, unchanged R37
diagnostic comparator, source-only-continuation representation, and native-tuned
representation across all three protocols. The sole eligible method is the
native-tuned representation. This requires 14 new neural continuation fits and
26 new XGBoost heads; the unchanged R37 comparator reuses its 13 checkpoint
outputs after exact provenance checks. A control success cannot silently become
the eligible method.

Require the current strict buffered record > .7542857142857143 plus every
selected8 protection: primary and mean-fold nonregression in all protocols,
macro >= reference-.001, short IoU, all old matched possessions and prior short
recall retained, no extra missed/spurious matches, D recall, false-D allowance,
and the existing buffered +.01 gain check. Also require native tuning to
strictly exceed the paired source-only control's buffered primary, with
primary/mean-fold nonregression against it in the other protocols. Historical
+.04 and extra-short objectives remain separately reported; no old verdict is
rewritten. A failed screen closes this fixed learning recipe without step,
learning-rate, source ratio, layer, augmentation, loss, or decoder sweeps.

Only a passing seed42 method may receive unchanged head-seed 43/44 confirmations;
the seed42 fold encoders remain fixed, as R37's external encoder did. All nine
protocol/seed comparisons must meet the safeguards. These are correlated
development checks, not independent neural-seed evidence. Require exact fresh
CPU continuation replay, strict reload, feature/probability/label/shared/official
gate replay, and independent source/spec review.

Only after those passes may one final encoder be trained by the identical
recipe on the union of allowed public training centers, and one final public
head fitted. Its adaptation pool remains public-only; it must not expand to
unlabeled full-match frames. Training windows and features must reproduce from
actual 18,935/19,060/full-199,319 input paths. The full candidate must be rebuilt
twice exactly, validated, and independently reviewed before any selected-model
change. The final encoder is a new fit; OOF accuracy is not its fitted-public
training score.

## Resources, failure modes, and decision

Use existing CPU Torch 2.5.0, two threads, deterministic algorithms and the
current attention fast-path setting. The small R36 model previously used less
than 1 GiB RSS; that is feasibility evidence, not a new measurement. A separate
invented-data gate must demonstrate the complete mixed-source/native BCE update,
same-shape control, mapped team swaps, padding/NaN neutrality, finite gradients,
deterministic fresh refitting, strict reload, and outside-fold independence.
Require peak RSS <=2 GiB for one worker and a conservative measured projection
<=7,200 seconds for the initial 14 neural fits, all external retention checks,
and representation extraction. Run serially after current CPU work releases
resources. Failure of this frozen resource gate is a no-go, not permission for
a shorter native run or an unreviewed backend switch.

The available 16-GiB MPS memory does not establish deterministic operation or
speed for this network. Existing `probability()` calls `.numpy()` directly on
CPU output, and initialization is pinned to CPU threading/determinism. MPS would
need its own separately reviewed runtime/parity evidence; no MPS execution,
fallback, dependency change, or cross-backend exactness claim is part of this
single proposal.

The main scientific risks remain severe: only 17 adaptively reused public
possessions, far fewer independent situations than sampled 25-Hz frames;
native annotation/source control-definition mismatch; noisy team assignment and
missingness shortcuts; forgetting despite rehearsal; overfitting transient
geometry; and latent representations that improve BCE without improving spell
boundaries. External retention and a fixed source-only control make these risks
observable in part, not impossible. Labelled fine-tuning removes the need to
trust ownership pseudo-labels, not the need for outer exclusion and all
possession safeguards.

**Decision:** this direct, architecture-preserving supervised continuation is
scientifically defensible as one bounded prospective test. It is the clearer
untried next method than target-only consistency under the existing label
permissions. Proceed first to a separately frozen synthetic implementation and
resource gate, with no native fitting authorized by this research note alone.
If the implementation cannot enforce the 13-encoder training/query separation,
the fixed checkpoint rule, exact replay, or budget, reject the proposal. No
unchanged label-free reconstruction continuation exists for this checkpoint.

## Authorized pre-score implementation amendments for R58

Before any real-data processing, training, or new possession outcomes, the root
explicitly broadened eligibility to match the user's goal: **both** the fixed
source-only continuation and the fixed source/native supervised continuation may
advance if they independently beat the strict record and satisfy all selected8
guards. The native-minus-source contrast is reported separately and is not an
eligibility condition. Each arm has its own external retention conjunction; a
technical or retention failure closes that entire arm while the other passing
arm may continue. Frozen hash/provenance corruption remains fatal to the whole
experiment. No partial-fold fallback or failed-method rescue is permitted.

The source schedule is also made exactly match/class/team balanced rather than
taking the first 400 rows of the older random R36 schedule: new PCG64 seed
580042 generates two 16-example chunks per update, each containing four matches
times two classes and exact paired team swaps. The candidate shares the first
16 with the 32-source control. Native seed580043 supplies paired 4H+4A base
draws, giving 8H/8A after swaps. Canonical B32/N46 and per-step Torch seed
580000+i preserve identical stochastic masks for shared source examples. This
is a newly frozen draw schedule using the existing corruption tensors, not a
claim to reuse the old schedule's bytes.

Both seed42-screen-eligible methods receive seeds43/44 head confirmations when
both pass; failed seed42 methods never do. Final eligibility is the intersection
of all three seed method sets, requiring all nine pairs per method. Rank final
passers by main-seed buffered primary, minimum other-protocol gain, fewer total
changes, then lexical name. This replaces the earlier sole-native eligibility,
fine-must-beat-control rule, and single-winner-only confirmation recommendation.
No architecture, optimizer, 400-step budget, retention tolerance, or decoder
change follows. The concrete authoritative recipe is
[NATIVE_ENCODER_CONTINUATION_PLAN.json](/Users/derekko/Projects/more-than-a-hack-2027/experiments/NATIVE_ENCODER_CONTINUATION_PLAN.json).

The final pre-score scientific amendment retains R36's established external
absolute floors (BA >= .75 and both recalls >= .65) as required per-method
checks. The newly proposed relative BA -.005 and Brier +.005 bounds are
preserved as reported diagnostics, **not eligibility gates**. Those tolerances
were unsupported new design priors; the user's goal is stronger reproducible
native possession performance, and the permitted supervised adaptation can
change the source/native label convention. This choice was made before any
source/native processing or outcomes. All selected8 native safeguards, strict
record, all nine confirmation comparisons per method, replay and full-CSV
requirements remain required and may not be relaxed after outcomes.
