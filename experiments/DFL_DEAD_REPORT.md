# Rounds 33–34: direct external live/dead supervision

Completed 2026-10-25. **No possession-score improvement; selected8 stays selected.**
The user requested continued work until a substantial gain, using provided data
and public labels without manual annotation. This closes one rejected avenue;
the broader goal remains active.

## Round33: direct-status teacher and paired native integration

Three stages were separately frozen before their respective processing/fitting:
[support audit](DFL_DEAD_FEASIBILITY_PLAN.json),
[external model](DFL_DEAD_TEACHER_PLAN.json), and
[paired public integration](DFL_DEAD_PUBLIC_PLAN.json).

The already downloaded DFL data contain7matches,449,539 dead frames and713
contiguous dead intervals. All7 pass the new support audit. Ball coordinates and
at least4 players per team are available on100% of their dead frames, avoiding
the earlier Metrica weak-label dataset's absent-ball-only supervision. Direct
`BallStatus` defines inactive/active in the
[primary publication](https://www.nature.com/articles/s41597-025-04505-y).
DFL/Bassek et al.2025, Scientific Data12:195; CC-BY4.0,
[versioned dataset](https://doi.org/10.6084/m9.figshare.28196177.v1).

One depth3/240-tree binary model uses183 shared XY-only, team-invariant features:
the previous91 geometry summaries made symmetric plus anonymous mutual-nearest
player-motion/shape summaries. Both domains use the same builder, with a100-frame
maximum footprint and half resets. Features exclude status, ownership, names,
clocks, supplied kinematic scalars and tactical tags. Label-independent noise and
dropout are applied before feature derivation. Four training matches, clean and
masked views, stride5 and equal match/class mass give226,916 fitting rows.
The historical4/1/2 match split is retained; these match sets were already used
for ownership research and are not claimed as untouched general validation.

All three fixed external conjunctions pass on masked observations:

| Validation match | Balanced accuracy | D precision | D recall | Live→D error rate |
|---|---:|---:|---:|---:|
| WPY | .971199 | .965229 | .970178 | .027781 |
| WQQ | .966759 | .960728 | .960512 | .026994 |
| WR9 | .968232 | .965548 | .961183 | .024720 |

Only then is the public integration frozen. Independent actual18,935-,19,060-
and199,319-row tactical inputs produce exact public features and teacher
probabilities. One probability is appended to each original physical/restart D
view. The original `.5` weak-source mass, training settings, public indices and
reserved/public exclusions remain exact. All26 matched original-head refits
reproduce saved probabilities and complete selected8 labels before26 augmented
fits begin. These52 public fits use no external rows as native supervision.
Only the two D heads are replaced; original H/A decisions wherever both outputs
are live, and all X labels, are preserved.

| Protocol | Reference spell F1 | Teacher spell F1 | Reference→teacher D recall |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .741176 | .748128→.746086 |
| Purged spells | .685714 | .685714 | .784888→.780123 |
| Buffered100seconds | .670270 | .670270 | .649421→.652825 |

False D counts remain88/142/16. No primary gain or additional short match occurs;
main/purged D recall also declines. The frozen substantial requirement is+.04
buffered spellF1 and one additional short IoU.5 match, plus all paired protection
checks and later43/44 confirmation. It fails; no confirmation, full build or
promotion follows. Strong external results establish no native transfer gain.

## Round34: raw native signal audit

The [diagnostic plan](DFL_DEAD_SIGNAL_PLAN.json) was frozen after Round33 rejection,
before target-conditioned native teacher statistics. It covers all7,500 public
frames and all4 public D intervals. It changes no possession predictions and
adds no possession score rows.

At the fixed.5 cutoff, native balanced accuracy is.603546; D precision.815190,
D recall.219197. Recall is0 on949 visible-ball D frames and.619231 on520 hidden-
ball D frames. This is a severe transfer failure. Differences in observations
or target conventions may explain it; this audit does not isolate their causes.

The fixed>=.8 D/<=.2 live proposals fail all three prospective correction gates:

| Selected8 baseline | Corrected D/live errors | Previously correct frames damaged | Correction precision |
|---|---:|---:|---:|
| Buffered minutes | 88 | 666 | .116711 |
| Purged spells | 142 | 716 | .165501 |
| Buffered100seconds | 16 | 527 | .029466 |

Required correction precision was at least.75 with at least25 net corrections
per baseline. No direct-inference experiment follows. Lowering a threshold or
bypassing the native learner is unsupported by this evidence.

## Verification and continuation

All315 tests pass, including symmetry/player permutation/coordinate invariance,
bounded context and half reset, missingness handling, supervision mass and held-
label independence. Checkpoints reload. Fresh processes reproduce60 external,
71 public and4 native common data artifacts byte-for-byte. Official/shared
scores agree. The support audit and raw-signal summary replay exactly.

Execution details retained: PhaseB synthetic tests were separated from the
immutable PhaseA test file before real feature processing; the PhaseA file is
exactly restored. Round34 adds an output-directory CLI for fresh replay only,
preserving first-execution source and unchanged target logic/gates. A closeout
assertion initially compared official logs containing different output paths;
the correction excludes execution logs while retaining every common data/score
artifact. No model, metric, threshold or selection changed in either correction.

Nine dependent score rows append, preserving every1,510 prior ledger row/byte:
1,519 total (778 inherited plus741 continuation). Round34 adds zero rows. All17
protected artifacts and source links verify. No human annotations, private
fixture targets, audit media, competition-image calls, new dependencies, commits,
uploads, submissions or contacts occur. Selected CSV SHA256 remains
`3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea`.

Local ignored evidence is under `data/control_dfl_dead_v1/` and
`data/control_dfl_dead_signal_v1/`. The active next research question is whether
a genuinely different ball-free learned player-motion channel transfers. The
research skill's background note assesses released PathCRF feasibility and its
limitations; this report does not authorize skipping its independent gates or
claim a score gain from that future method.
