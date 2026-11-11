# Round39: relative-ball geometry in a masked temporal encoder

The ball-aware model fails its frozen external acceptance gate. It does not
proceed to native inference or public possession fitting. Selected8 and all1531
ledger rows remain unchanged.

The model adds player-to-ball XY and an explicit ball-observed flag to each
player's XY before the original R36 shared attention/team-mean pooling and
temporal scoring. Missing balls contribute zero relative geometry and flag.
An identical24,417-parameter control masks the ball throughout. Both train from
the same initialization and unchanged4000×32 equal-match/class schedule. No
epochs, noise levels or architecture are selected from validation outcomes.

| Masked validation match | Ball-aware BA | Ball-masked BA | Difference |
|---|---:|---:|---:|
| WPY | .799187 | .795096 | +.004090 |
| WQQ | .828833 | .833657 | −.004824 |
| WR9 | .828201 | .824940 | +.003261 |

The fixed conjunction requires BA>=.80, both class recalls>=.70 and at least
.02 BA over the matched control in **each** match. WPY fails the absolute BA
threshold; all three fail the incremental ball-information threshold. Both
class-recall checks pass. Clean BA is .862018/.881995/.895899 versus control
.853934/.873543/.873426; those clean gains do not rescue the masked gate.

Four label-blind ball views retain the original player corruption/sampling.
They add predetermined missing intervals, half- or one-metre noise, a visible
player-region constraint and occasional5–15m shifts. Observations are hashed
before target access. Half resets and31 samples at5Hz bound context to75frames.
Only existing publicDFL direct ownership labels are used. The historical4/1/2
match partition has already been reused in this project; it is not an untouched
generalization benchmark. Sources/terms follow
[the public DFL release](https://www.nature.com/articles/s41597-025-04505-y).

Synthetic ball-only information is learned at1.0 accuracy versus.5 with the
ball masked. Deterministic refitting, strict reload, team swap, node permutation,
missing-ball NaN safety and all-player-missing neutrality pass. Projected two-fit
time1889seconds and peak.541GiB meet the frozen2400seconds/2GiB limits. Actual
fits complete all4000 steps in768.5/787.6seconds with peaks.595/.584GiB.

All four complete external probability arrays match strict checkpoint reloads.
A fresh process replays192 boundary/middle witnesses from each array exactly.
Both new ball-observation arrays rebuild byte-for-byte without reading targets.
The full334-test suite passes, plus two subsequent synthetic native-padding/
normalization tests. All17 protected artifacts and source links remain unchanged.

A native representation-transfer implementation was prepared and synthetically
tested while source assessment ran. Its native plan was **not frozen** and its
real-data methods were **not run**, because the source model failed. The current
mean-pooled relative-ball architecture is closed without an epoch/noise/layer
sweep. The separate WASB detector confirmation continues; no result from it
enters this experiment. No manual annotations, private targets, native target
join, installations, submissions or contacts occur.

Plans: [synthetic](BALL_CONTEXT_RUNTIME_PLAN.json),
[external](BALL_CONTEXT_EXTERNAL_PLAN.json). Evidence remains under ignored
`data/control_ball_context_v1/`. The wider substantial-improvement goal is active.
