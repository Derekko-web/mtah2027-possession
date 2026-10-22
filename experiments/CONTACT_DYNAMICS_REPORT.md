# Contact dynamics research — 21 October 2026

The selected model remains unchanged. The new trajectory and touch features do
not pass the frozen substantial-gain criterion. Impulse-based proposals provide
a modest lead in the longer-block protocol; the main buffered score is unchanged.
All 39 evaluations, 52 fitted checkpoints and failed variants are retained. The
ledger now has 1,066 dependent rows, with all 1,027 preceding rows preserved.

## Hypotheses and observations

[CONTACT_DYNAMICS_PLAN.json](CONTACT_DYNAMICS_PLAN.json) fixed the filters,
features, four heads, three decoders, protocols and promotion gates before
scoring. The source archive, selected six-model candidate and confirmation were
preserved under `data/control_dynamics_v1/` first. The initial short-precision
feedback loop failed with 3 of 6 short possessions matched at IoU>=.5. It still
fails; `require_short_precision.py` is a reusable fast diagnostic for that symptom.

Earlier features summarize noisy ball motion but still measure proximity to the
original ball location. This round tested two local trajectory repairs: a
centered 7-frame coordinate median with whole-gap filling limited to 5 frames, and a
centered 25-frame robust local linear fit with Gaussian weights and three Huber
reweighting passes. Robust filling is limited to complete gaps<=13 frames;
unsupported missing intervals stay missing. No long trajectory interpolation,
player identities or absolute coordinates enter the new model inputs.

The nearest-team diagnostic rejects the simple jitter explanation for the two
missed controls. Both raw and repaired geometry choose the correct nearest team
on 23.94% of the 71-frame possession and 0% of the 33-frame possession. Across all
public live frames, observed nearest-team correctness is 77.84% raw, 77.80% median
and 77.29% robust. Observation coverage rises only 93.52%→93.60%. Nearest-team
correctness is a diagnostic proxy, not a possession score; nevertheless these
filters do not supply the missing control information by themselves.

The second representation adds 5-frame before/after ball velocities, vector
impulse magnitude, turn cosine and team-specific distance-weighted impulse
strength. Bounded past/future maxima and means carry brief touch evidence into
neighboring frames. Their total observation footprint, including trajectory
repair, fits within 125 frames. No labels enter feature construction. All raw
tracking, features and derived arrays stay under ignored local `data/`.

## Frozen screen

Each head uses unchanged live-only `fit_control` settings and team-symmetrized
inference. Only eligible public training live labels are used; no fitted
preprocessing, calibration or early stopping. The matched control uses the
existing ordered inputs. New heads add median geometry, robust geometry, or
robust geometry plus impulse/memory features.

Three decoders test each head: replacement of only the existing fast head inside
the unchanged six-model pipeline; brief physically supported proposals; and
replacement within each reference live segment. The latter two retain every D
decision exactly. Fast-head replacement can affect D competition during decoding
and is checked against the existing secondary gates.

| Head / decoder | Buffered spellF1 | Purged spellF1 | 100-second spellF1 |
|---|---:|---:|---:|
| Selected reference | .729412 | .662857 | .659459 |
| Ordered / fast replacement | .662500 | .593939 | .594286 |
| Ordered / proposals | .729412 | .662857 | .659459 |
| Ordered / live replacement | .518919 | .484211 | .492308 |
| Median / fast replacement | .662500 | .593939 | .594286 |
| Median / proposals | .729412 | .662857 | .625641 |
| Median / live replacement | .560000 | .511111 | .492308 |
| Robust / fast replacement | .662500 | .593939 | .594286 |
| Robust / proposals | .729412 | .662857 | .625641 |
| Robust / live replacement | .560000 | .511111 | .492308 |
| Impulse / fast replacement | .612500 | .569697 | .594286 |
| Impulse / proposals | .729412 | .662857 | **.670270** |
| Impulse / live replacement | .484211 | .463158 | .518919 |

Impulse proposals improve longer-block mean short IoU .396764→.429413 and
MacroF1 .842747→.843559. The 90-frame possession's IoU improves .637931→.725490,
crossing another matching threshold and accounting for the small primary gain.
The 33-frame possession's recall rises 21.21%→63.64%, but its predicted interval
remains poorly bounded: IoU .058333→.166667. There is no additional short match
at IoU .5. The 71-frame miss and 19-frame overlong ending remain.

Buffered impulse proposals leave primary unchanged and slightly reduce MacroF1
.867912→.867590 and mean short IoU .399285→.396753. Purged impulse proposals
are unchanged. These results do not establish a broad or substantial gain.
All three proposal comparisons preserve the selected D masks exactly. Broad
impulse fast replacement increases false-D frames 92→138 buffered and 152→170 purged,
while losing primary score; it is rejected.

Across the 12 alternatives there are 64 primary regressions among 156 clipped-block
comparisons. The complete `block_deltas.csv` retains every comparison. Buffered
and purged maximum block gains are zero; the longer protocol's largest block
gain is+.026667. All broad replacements regress stitched primary.

## Decision and verification

No method passes the frozen gates: buffered primary gain>=.04 for each seed,
positive gains in each other protocol, improved mean short IoU and an additional
buffered short IoU>=.5 match, with secondary protections. There is no finalist
for seeds43/44, no full-match build and no promotion. The existing CSV retains
SHA256 `dac48e6be66f4395089cae4eee3cfb5daeadcd7963480f399fe079d871be534c`.

All 39 score rows agree with the official CLI scorer and replay exactly through
shared inference. All 52 saved heads reload to identical OOF probabilities.
Actual public geometry/impulse features from the 18,935-frame padded cache equal
features built from 19,060 observations, exactly. Source/input/cache hashes,
selected candidate/source identity and all 1,027 old ledger rows verify. All 78
tests pass, including repair of synthetic outliers, complete-gap limits,
geometric/team/slot symmetry and bounded future context. The integrity evidence
is `data/control_dynamics_v1/integrity.json` and `tests.log`.

With the existing selected confirmation and advanced-validation caches present,
use Python 3.12 in `.venv` and fresh output folders:

```sh
.venv/bin/python experiments/prepare_contact_dynamics.py --output data/control_dynamics_v1/inputs_reproduce
.venv/bin/python experiments/research_contact_dynamics.py --inputs data/control_dynamics_v1/inputs_reproduce --output data/control_dynamics_v1/screen_reproduce
.venv/bin/python -m unittest discover -s experiments -p 'test_*.py' -v
.venv/bin/python experiments/require_short_precision.py
```

The final diagnostic intentionally exits 1 until at least four short possessions
match. `finalize_contact_dynamics.py` audits the original local screen, including
its pre-round archive and ledger preservation; it is separate from fresh screen
generation. Plans, source/input/cache hashes, all models/probabilities,
diagnostics, predictions, official inputs/reports, block scores and execution
logs remain in `data/control_dynamics_v1/`. No failed output was overwritten.

This is further adaptive development on the same 17 public possessions with
overlapping training sets. The modest longer-block result is a lead for further
investigation, not independent accuracy evidence or a verified maximum. No
private labels, new annotations or media were accessed; nothing was committed,
submitted, uploaded or published.
