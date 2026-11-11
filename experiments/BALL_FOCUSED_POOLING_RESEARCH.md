# A local ball-query bottleneck for team ownership

Research only, 2026-10-25. This note proposes one conditional experiment; it does
not run or authorize a model, reopen Round39, or change any frozen criterion.
Only source, aggregate reports and primary publications were read. No `data/`
files, competition observations, possession targets, media or checkpoints were
opened, and no fitting, scoring, downloads or dependency changes occurred.

The bounded hypothesis is that **competition between nearby player tokens for
one ball-conditioned query, followed by team marginalization before temporal
processing**, learns useful contact evidence that global team-mean pooling did
not retain. The proposed model deliberately cannot use distant formations. It
could still fail because the ball is wrong, the controller is missing, local
geometry is ambiguous, or team supervision does not identify contact. Published
research supports the ingredients, not an improvement claim for this recipe.

## What is closed and what would be different

[BALL_CONTEXT_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/BALL_CONTEXT_REPORT.md)
now records Round39's final masked-source balanced accuracies:
.799187/.828833/.828201 versus .795096/.833657/.824940 with the ball masked.
It fails the .80 absolute gate on WPY and the +.02 paired gate on every match.
Clean-view gains do not rescue it. Its relative-ball inputs feed the original
shared spatial attention and team **mean/count** pooling in
[ball_context_network.py](/Users/derekko/Projects/more-than-a-hack-2027/experiments/ball_context_network.py)
and
[masked_player_network.py](/Users/derekko/Projects/more-than-a-hack-2027/experiments/masked_player_network.py).
Changing its training length, corruptions, widths or thresholds is not the
proposal here.

Earlier
[OWNERSHIP_GRAPH_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/OWNERSHIP_GRAPH_REPORT.md)
already tested ball-distance Gaussian pools, relative radial/tangential motion,
closest approach, impulses and opponent support in XGBoost heads. Those pools
were themselves local and invariant; locality is therefore not a new source of
information. The distinct test is an end-to-end **latent player competition**
with an explicit no-contact alternative, team probability mass and a constrained
temporal interface, rather than another collection of engineered contact
scalars. This distinction is a hypothesis worth falsifying, not evidence that
the previous failure was caused by XGBoost or averaging.

[CONTACT_DYNAMICS_REPORT.md](/Users/derekko/Projects/more-than-a-hack-2027/experiments/CONTACT_DYNAMICS_REPORT.md)
is a substantial counterargument: repaired/raw nearest-team geometry remained
wrong on important missed possessions. A sharper local selector can amplify
that error. The original WASB pilot remains failed for insufficient visible
cases, despite its reported temporal advantage; separately frozen Round40 is
still running according to the parent task. Neither outcome supplies evidence
for this ownership model, and no WASB output enters the proposed input contract.

## What the primary literature actually supports

* **Ball Radar** first predicts player-level possession and uses that intermediate
  representation for ball trajectory inference. Its team metric sums player
  probabilities by team. It explicitly separates possession probability from
  whether the ball is currently being touched, and motivates direction changes
  near players with a physical loss. That supports a player-to-team bottleneck
  and a separate no-current-contact state. It does not validate ball-query
  pooling: its player-wise recurrence needs persistent player trajectories, and
  its training includes player-possession supervision that the proposed direct
  team-label experiment lacks. Its released forward API also accepts optional
  masked target hints; none belong in a label-blind adapter. No Ball Radar model,
  loss or target-hint path is imported here. See the
  [paper, §§3.4–3.6 and 4.3](https://arxiv.org/html/2306.08206v1),
  [official repository](https://github.com/hyunsungkim-ds/ballradar) and
  [PlayerBall source](https://raw.githubusercontent.com/hyunsungkim-ds/ballradar/main/models/player_ball.py).
* **Link and Hoernig** distinguish individual contact/control from team possession
  that can continue between touches. Their 2D proximity detector can falsely
  assign a ball passing above a player; adding ball-acceleration evidence helps,
  while their limitations acknowledge touches without a strong acceleration.
  This supports an explicit flyover counterexample and forbids equating nearest
  player with team owner. Their thresholds, running flag, player identities and
  interval construction are not imported. See
  [Individual ball possession in soccer, §§3.2–3.6](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0179953).
* **Vidal-Codina et al.** compare near-player ball motion whose speed/direction
  changes with a constant-motion pass-by that is not a gain of control. Their
  rule-based framework uses more observation and event assumptions than this
  partial-roster problem. This motivates exposing bounded incoming/outgoing ball
  motion, not treating a learned attention weight as a verified touch. See
  [Automatic event detection in football using tracking data, §2.3 and Fig.3](https://link.springer.com/article/10.1007/s12283-022-00381-6).
* **Set Transformer** provides attention-based permutation-invariant pooling
  using learned seed queries. Conditioning one query on the observed ball is
  an original adaptation of that operation, not an architecture evaluated in
  that paper. It permits different influence for different set members; it
  supplies no football-specific accuracy guarantee. See
  [Lee et al., §3.2](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf).

These sources do not establish that direct H/A labels identify a unique latent
controller or meaningful null probability. The attention mass below is a
representation, not a calibrated annotation of who touched the ball.

## One concrete small model

The following constants are a prospective, unverified recipe to freeze before
implementation or any result access. They are not paper-derived optimal values.

Use the existing 31 samples at 5Hz and half-boundary padding, with player XY,
observed masks, team groups, ball XY/observed mask and pitch dimensions. Convert
field-normalized positions back to consistent metres before distances. There
are no player IDs, correspondences, keeper guesses, player velocities, absolute
pitch coordinates, team centroids, global roster counts, true live resets or
event features. Independent node reordering at every sample is valid.

For an observed ball at sample `t`, retain only finite observed players within
**8 metres**. For each retained player compute
`r=(playerXY-ballXY)/8`, `d=||r||`, then a shared `MLP(3,16,16)` on
`[r.x,r.y,d]` with ReLU. This produces `h_j`. A second shared linear map gives
16-dimensional keys. A six-input query MLP `MLP(6,16,16)` receives incoming and
outgoing ball velocity XY, clipped at 80m/s and divided by 80, plus their two
validity flags. Differences use only adjacent samples inside the same 31-sample
window; gaps, half boundaries and padded samples invalidate the corresponding
velocity. No derivative or sample outside the existing ±75-frame reach is used.

Use one shared query across both teams:

```text
u_j = dot(query, key_j)/sqrt(16) - (distance_metres/2)^2
u_null = Linear16to1(query)
a = softmax([u_null, u_1, ..., u_n])
E_H = sum_{j in H} a_j * concat(h_j, 1)       # 17 values
E_A = sum_{j in A} a_j * concat(h_j, 1)       # 17 values
z_t = concat(E_H, E_A, six_query_inputs, a_null)   # 41 values
```

The final coordinate of each `E_team` is that team's total candidate mass;
it is not divided by player count. Both teams compete in one denominator. A
far player cannot dilute a contact because it is masked before key/value
computation. The fixed distance term is a bias, not a hard nearest-player
decision. If the ball is missing, all player candidates are masked and null
mass is exactly one. The null token makes even an entirely missing set finite.
It means **no observed current contact evidence**, including occlusion, flight
and tracking error; it is not dead ball or a third possession target.

Feed the 41-channel sequence to the existing width32, four residual temporal
convolutions with dilations1/2/4/8, after a 1×1 projection. Use the shared score
for both team orientations and output `score(H,A)-score(A,H)` at the center.
This guarantees team-swap antisymmetry; entirely absent team evidence yields a
neutral logit. Temporal processing can retain evidence from a prior nearby
team while the center is a flyover or missing-ball sample. No hand-coded
last-controller state or true possession initialization is introduced.

This is approximately **14,802 parameters** with the explicitly listed biased
linear/conv layers, identity values `h_j`, and one final scalar linear layer.
Implementation must verify the exact count. Plain Torch2.5.0 linear layers,
masked softmax, sums and Conv1d suffice; NumPy2.1.2 is enough for existing input
preparation. No PyG, upstream package, checkpoint, MPS, installation or download
is needed. Runtime feasibility is unmeasured. A synthetic-only timing/RSS and
deterministic reload stage must precede source processing; retain two CPU
threads, ≤2GiB peak per model and ≤2400s projected total for the two full fits.

This handles fragmentation by never connecting player slots through time.
It does not reconstruct absent players. A missing controller can produce
neutral evidence or a wrong rival candidate; local count changes and duplicate
detections still change the attention denominator. Ball-location errors can
exclude the correct player entirely. The radius is not enlarged after failures.

## Falsify the contact explanation

The main paired fit is **true anchor versus anchor placebo**, both the model
above, identical parameter count, initialization, corruption schedule, batches,
optimizer and fixed budget. In the placebo, independently permute the finite
ball anchor XY values among the observed timestamps of each already bounded
window. Leave player sets, their order/masks, ball-observed schedule and the
six actual ball-motion query inputs unchanged. Recompute local eligibility and
relative keys with the substituted anchors. The permutation is target-blind,
hashed from sample identity/view, and fixed for validation; training permutations
are frozen in the schedule. With fewer than two finite anchors the placebo
cannot disrupt anything; count such windows explicitly. No anchor comes from
outside the permitted context. This controls capacity and ball visibility/motion
while disrupting synchronized local ball-player geometry. It is a deliberately
corrupted control, not an alternative deployable estimator or proof of causality.

Before DFL fitting, freeze and execute a small synthetic capability suite with
held-out generator seeds. These are prospective tests, not results:

1. **Identical formation, changed ball:** create balanced paired examples with
   exactly the same full player sets/masks at every time, but a slow ball within
   0.5m of H in one member and A in the other, with the other team ≥3m away.
   Randomly translate, reflect and permute these scenes. Toy-trained true-anchor
   inference should classify ≥95% of held-out pairs correctly. An input with
   the ball hidden is identical across each pair and cannot exceed50% paired
   accuracy. Failure rejects the claim that this implementation can learn local
   ball evidence even in a favorable case.
2. **Same center, different touch history:** the simulator stores a scripted
   last-touch team. Construct balanced trajectory pairs with identical player
   sets/masks and identical center ball position, incoming/outgoing center
   velocities and flags. One trajectory has the last H touch followed by a
   straight flyover near A; the paired trajectory has a later A touch and
   rejoins the same center segment. Their center snapshots and local query
   packets are therefore identical, but their earlier touch histories differ
   within the 31 samples. Require ≥90% paired correctness on new synthetic
   seeds and ≤55% for the same trained network given repeated center snapshots.
   A global formation, center-nearest-team or center-motion shortcut cannot
   solve both members. Scripted 2D touch labels test representational capacity;
   they are not a definition imposed on DFL or real airborne balls.
3. **Structural invariants:** move/add/remove arbitrary players that remain
   outside8m of the ball at every sample, without altering local candidates.
   Logits must agree to floating tolerance `1e-6`. Test fresh node permutations
   every frame, team swaps, joint translations, NaNs in masked values, empty
   teams, missing balls, half padding and strict checkpoint reload. Reflection
   augmentation must transform relative positions and ball motion consistently;
   the signed-coordinate MLP does not guarantee exact reflection invariance.
   Exactly unchanged distant-player output is stronger than a saliency
   plot and directly excludes a global formation route in this implementation.

The toy thresholds are suggested precommitments, not measured facts. Passing
them only establishes the intended capability and invariants. On real DFL
evaluation, additionally publish the frozen true-arm's outputs with repeated
center inputs and with anchor permutation as diagnostics, without selecting
between them or fitting more models. A source gain without anchor sensitivity
would not support the claimed contact mechanism.

## Prospective source and native decisions

Keep the existing official seven-match public DFL corpus, historical4/1/2
whole-match split, direct `BallPossession` H/A targets on eligible live centers,
and unknown-label exclusion. BallStatus/ownership are eligibility/supervision
only: never features, observation masks, context cuts, resets or anchor choices.
Both halves of a match stay together. The public release contains player/ball
XY and official events under CC-BY4.0; its high-quality multicamera domain is
different from partial native tracking.
[DFL data descriptor](https://www.nature.com/articles/s41597-025-04505-y).

The current partition has already been inspected repeatedly and is not a fresh
holdout. Reuse must be called adaptive research. Freeze all implementation/input
hashes, toy criteria, radius, fixed 4000×32 schedule, seed42, AdamW.001,
weight-decay.01 and gradient clip1 before fitting. Reuse the existing target-blind
player and ball corruption views from
[BALL_CONTEXT_EXTERNAL_PLAN.json](/Users/derekko/Projects/more-than-a-hack-2027/experiments/BALL_CONTEXT_EXTERNAL_PLAN.json),
without selecting corruption strength from this candidate's outcomes. No extra
contact labels, true player controller labels, weak native labels or target-
derived crops are created. Use center team BCE, not synthetic touch losses on
real data. There is no early stopping or checkpoint choice.

Require, in **each** WPY/WQQ/WR9 masked assessment, true-arm BA≥.80, both team
recalls≥.70 and BA≥placebo+.02, plus the toy, resource, provenance, deterministic
reload and boundary/prefix checks. Report all clean and masked results. A failed
conjunction closes this recipe; it does not start a radius/query/epoch/noise
sweep. These are retained strict research gates, not a statement that .80 source
BA predicts useful competition possession boundaries.

Only a source pass would justify a separately frozen native observation and
raw-signal audit **before any public possession fitting**. Keep all source-
eligible native centers in the denominator, including missing-ball/unsupported
cases; report physical support and the same ≥80% coverage aim without removing
hard cases after target access. Freeze native normalization, eligibility,
observed masks, array hashes, outputs and the existing .65/.35 abstention rule
before a one-time diagnostic target join. Do not infer true live intervals or
initialize from true episodes. Preserve selected D/X exactly. Do not combine a
new detector, change anchoring, or choose a confidence threshold after seeing
public errors.

The existing
[MASKED_PLAYER_SIGNAL_PLAN.json](/Users/derekko/Projects/more-than-a-hack-2027/experiments/MASKED_PLAYER_SIGNAL_PLAN.json)
provides the conservative native continuation test: in every protocol,
supported/editable true-live BA≥.70, at least25 corrections, correction
precision≥.55, corrections minus damage≥25, all prior matched possessions
retained, no extra spurious possession, and one common originally missed short
match recovered by the frozen diagnostic overlay. Unsupported cases remain in
the full17-possession/6-short/4-miss universe. Undefined or insufficient counts
fail. This is an opportunity gate, not possession-head fitting or a promotion.

The wider selected8 goal remains ≥+.04 buffered spellF1 plus an extra short
match, every secondary protection, seeds42/43/44, all protocols, exact replay,
two full builds and independent review. None is relaxed because this bottleneck
is physically appealing. Its strongest justification is that the contact
hypothesis becomes directly testable; its strongest risk is that the existing
ball observations simply do not contain the missing evidence.
