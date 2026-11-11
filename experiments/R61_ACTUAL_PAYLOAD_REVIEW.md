# R61 actual source/native payload review

Read-only artifact audit completed 2026-10-30T00:48:55Z. **PASS for the actual preparation and 14 completed fitting payloads; no audit failures found.** This review created only this new report in the repository. It did not fit or load a model, run inference, calculate a new performance score, alter frozen Python/dependencies, or read sealed/reserved contents beyond permitted hash-only bindings.

## Scope and evidence

The review read `CURRENT_RESEARCH_CHECKPOINT.md`, the R61 plan, both frozen R61 implementation files, the inherited observation assembly/preparation sources, and retained source/pool/job/recipe/manifests/logs. Independent NumPy derivations checked all compact source arrays, every native pool array, every admissible center and every saved draw. The main audit passed 10,987 structural assertions (including one assertion per native center), not a new model test run. It hashed 2,977 unique bound file targets; the R61 synthetic summary cache was checked separately.

Plan SHA-256: `f3abd03312aa98af460ec05b2594b11974a02cac1a71e75afb38dca9301f5ec4`. Its bytes equal `protocol_freeze/plan.json`. The frozen implementation contains 425 files; the 422 inherited bindings and three new R61 Python files all match. The 12 R58 provenance/failure/freeze/source bindings in `ast_equivalence.json` also match. R58 remains failed; nothing here waives its gate.

## Actual source payload

The compact `source/` contains exactly four training matches—WMX, WN1, WOH and WOY—each with two halves, totaling **113,458 5 Hz rows**. Its XY `(4,113458,46,2)`, observed mask, teams, half ID and labels equal the exact eight original corpus segments, element for element. The six WPY/WQQ/WR9 validation halves begin after those source rows and are absent from every worker source array. The retained pre-target observation hashes match the arrays after the target join.

The source labels also independently match the eight inherited external half files after `[::5]` sampling: live owner 1 gives H=1, live owner 2 gives A=0, everything else gives -1. Counts are H=30,504, A=29,885, unused=-1=53,069. Only valid H/A centers at least 15 source samples inside their half enter the draw buckets; all eight match/class buckets are nonempty.

Independent PCG64 reconstruction with seed 580042 exactly matches every saved `schedule.npz` field: 400×32 examples, or 12,800 draws. Each of the two 16-example chunks has one base draw from each four-match×two-class bucket, followed by its exact team-swap partner. Center, corruption view and both reflections match within each pair; swaps complement the target. Each chunk has eight H and eight A, four examples per match. All 31-sample windows stay within a single half. Source control uses both chunks; each native fit uses the same first chunk.

## Actual native pools

All 13 pools exactly derive from the hash-bound R37 `inputs/` arrays and the unchanged saved outer splits: 5 buffered-minutes, 5 purged-spells and 3 buffered-100-seconds. The representation grid is exactly `11185:18690:5`, with 1,501 supported observation centers. Each pool contains only its own public outer-train H/A centers, with H=1/A=0 targets; no center or real same-half footprint position is in its outer validation set.

The audit independently rebuilt observation eligibility before joining labels, then required the entire real same-half ±79-frame footprint to belong to the outer train. It checked **10,652 fold-centers and 1,687,028 retained footprint positions**. Every pool XY, mask and team tensor exactly equals the corresponding R37 clean-view 31×46 window, including prescribed padding. Validation indices are retained only as provenance in `fitting.npz`; the frozen worker never loads that archive.

| Fold | Train frames | Validation frames | Observation eligible | H/A centers | H | A |
|---|---:|---:|---:|---:|---:|---:|
| buffered_minutes/block1 | 5750 | 1500 | 1119 | 853 | 421 | 432 |
| buffered_minutes/block2 | 5500 | 1500 | 1054 | 788 | 324 | 464 |
| buffered_minutes/block3 | 5500 | 1500 | 1054 | 786 | 467 | 319 |
| buffered_minutes/block4 | 5500 | 1500 | 1054 | 890 | 650 | 240 |
| buffered_minutes/block5 | 5750 | 1500 | 1135 | 1006 | 613 | 393 |
| purged_spells/block1 | 5750 | 1500 | 1119 | 853 | 421 | 432 |
| purged_spells/block2 | 5089 | 1500 | 972 | 706 | 257 | 449 |
| purged_spells/block3 | 5350 | 1500 | 1024 | 756 | 452 | 304 |
| purged_spells/block4 | 5430 | 1500 | 1040 | 876 | 648 | 228 |
| purged_spells/block5 | 5215 | 1500 | 1028 | 914 | 613 | 301 |
| buffered_100_seconds/block1 | 4750 | 2500 | 919 | 653 | 237 | 416 |
| buffered_100_seconds/block2 | 4500 | 2500 | 854 | 663 | 383 | 280 |
| buffered_100_seconds/block3 | 4750 | 2500 | 935 | 908 | 613 | 295 |

**Known-half padding caveat:** the frozen rule explicitly excludes frames outside the center’s known half from its footprint, because `assemble_native` replaces those context samples with zero XY/teams and an unobserved mask. For each of ten folds (every fold except buffered-minutes/block1, purged-spells/block1 and buffered-100-seconds/block1), 16 admitted centers at frames 11185–11260 have a truncated ±79 footprint at the known-half start, 11185. Each such fold has 120 padded 5 Hz window timesteps, all exactly verified zero/unobserved. The other three folds have no padding. This is the prospective known-half exception, not permission to consume unavailable or validation observations. All retained real same-half positions still pass the full outer-train check.

Independent PCG64 reconstruction with seed 580043 exactly matches all five native schedule fields in every pool. Each 400×16 schedule draws four A and four H base examples with replacement, adds their exact eight team-swap partners, repeats reflections within each pair, and complements targets. Every step therefore has eight H and eight A. No seed, schedule or center selection was chosen from a result.

## Actual workers and resources

All 14 exact job schemas pass their input allowlist and every input hash. The source job contains the seven expected source/initial-checkpoint bindings; each native job adds only its own pool, schedule and fitting-provenance hashes. Each actual job has `parent: null`; no validation observations, query arrays, validation targets or scores are supplied as training arrays. `check_inputs` hashes frozen validation bindings without loading them into training. All recipes retain the same initial checkpoint (`c22debc44f35a02486b1fd2194d1dab176dcac1a62d92bba214f8dac3f575786`), AdamW LR 0.0001 / decay 0.01, clip 1, CPU 2, B32/N46, all 24,321 parameters, 400 steps and final checkpoint only. Native BCE is 0.5 mean(source16)+0.5 mean(native16); source control is the algebraically identical two-half mean over source32.

Each retained loss array has exactly 400 finite entries. Every stdout records steps 100/200/300/400; every stderr is empty; every controller supervision has exit 0 and no-retry true. All four cached outputs per fit rehash exactly, including model, recipe, loss and canonical probe.

| Model | Fit seconds | Peak GiB | Supervisor seconds |
|---|---:|---:|---:|
| source | 85.792 | 0.660873 | 88.254 |
| native/buffered_minutes/block1 | 84.335 | 0.656357 | 86.579 |
| native/buffered_minutes/block2 | 84.487 | 0.655411 | 86.448 |
| native/buffered_minutes/block3 | 86.886 | 0.620346 | 89.560 |
| native/buffered_minutes/block4 | 87.116 | 0.630783 | 89.699 |
| native/buffered_minutes/block5 | 86.625 | 0.637527 | 88.642 |
| native/purged_spells/block1 | 85.553 | 0.632950 | 87.592 |
| native/purged_spells/block2 | 84.312 | 0.647125 | 86.234 |
| native/purged_spells/block3 | 84.245 | 0.641754 | 86.183 |
| native/purged_spells/block4 | 84.691 | 0.642075 | 86.590 |
| native/purged_spells/block5 | 84.505 | 0.632706 | 86.384 |
| native/buffered_100_seconds/block1 | 85.368 | 0.659775 | 87.255 |
| native/buffered_100_seconds/block2 | 84.379 | 0.631439 | 86.325 |
| native/buffered_100_seconds/block3 | 84.584 | 0.641006 | 86.440 |

Every worker is below 900 fitting seconds, 2 GiB peak RSS and the 1000-second controller deadline. The retained R61 invented benchmark also passes: 12 tests/OK in its log; summary cache exact; projected 7191.156675 seconds below the prospective 9000-second initial projection bound, peak 0.606323 GiB, maximum training step 0.225689 seconds, B64 forward 0.119060 seconds. No retained method-failure JSON exists in R61.

**Strict reload evidence:** all 14 hashed manifests say `strict_reload_exact: true`, and the immutable worker only seals them after strict state-dict reload and exact complete canonical-probe equality. This read-only review did not independently execute a reload/inference. Every manifest correctly says `independent_refit_exact: false`; the required later fresh-refit/feature/probability/score/full-candidate replay is not satisfied by this audit.

## Existing source-control external gate

The retained source assessment has exactly the six required cells: three disjoint external matches × clean/masked views, one final-step400 source model. Its assessment cache and exact model binding pass. Each cell has only the three eligibility checks BA≥0.75, H recall≥0.65, A recall≥0.65; all 18 booleans are true and agree with its saved metrics.

| Match | View | Saved BA | Saved H recall | Saved A recall |
|---|---|---:|---:|---:|
| DFL-MAT-J03WPY | clean | 0.861629007 | 0.924980438 | 0.798277575 |
| DFL-MAT-J03WQQ | clean | 0.875202520 | 0.844197998 | 0.906207042 |
| DFL-MAT-J03WR9 | clean | 0.876514831 | 0.914229286 | 0.838800375 |
| DFL-MAT-J03WPY | masked | 0.802803456 | 0.889475743 | 0.716131169 |
| DFL-MAT-J03WQQ | masked | 0.842160202 | 0.811739248 | 0.872581157 |
| DFL-MAT-J03WR9 | masked | 0.838138184 | 0.879337910 | 0.796938457 |

The BA-baseline−0.005 and Brier-baseline+0.005 comparisons are confined to `diagnostics`; each explicitly states `relative_checks_are_eligibility: false`. Their stored booleans agree with the saved comparison values. Frozen `retention_eligibility` uses only `checks`, requires 6 source cells / 78 native cells, and never uses those relative diagnostics for eligibility. This review did not recompute probabilities or performance metrics. External matches are historically reused development data. The source assessment alone says nothing about whether all 13 native encoders pass their separate 78-cell gate, which root is assessing separately.

## Preservation and hashes

| Binding group | Checked entries | Mismatches |
|---|---:|---:|
| R61 plan input SHA-256 | 256 | 0 |
| R61 frozen implementation | 425 | 0 |
| Established runtime SHA-256 | 2,175 | 0 |
| Selected artifacts | 17 | 0 |
| R58 immutable provenance | 12 | 0 |
| Source symlink targets | 2 | 0 |

Both `data/tactical` and `data/broadcast` remain symlinks to their frozen absolute Downloads source paths. Current ledger and frozen ledger are byte-identical: **1,591 data rows, 605,189 bytes**, SHA-256 `6568a890a2a5fbc5ce428230af01a5b3ed9fc88351c370484134b0a23387f100`. No score rows were appended by this review.

Every actual checkpoint SHA-256 was independently recomputed:

| Model | SHA-256 |
|---|---|
| source | `72e726a07609cabd38a59b70738d70a0f1bc3f61dad5d0c6df13c0c503839c04` |
| native/buffered_minutes/block1 | `616af3b30b010d5ef54f7f678a32d78fd750fddcfb98404a249f37c138538fd8` |
| native/buffered_minutes/block2 | `ef19c7c6b4c75881b59b8fd023bc81fc71d1c4dfe0cfa9fa4f605aca61c7cddc` |
| native/buffered_minutes/block3 | `d04fd25e329596b257e319098dd9c156aa7e678e0e55f002a11ba3983945b566` |
| native/buffered_minutes/block4 | `52b5011a558e6baba5fc42a637075784ca1a0ae09d5108582ab0e798f64d0fe2` |
| native/buffered_minutes/block5 | `85826dc6ca21a0780977750e9703a8f90852db3eb26f9b8262b1a7e7195119a5` |
| native/purged_spells/block1 | `616af3b30b010d5ef54f7f678a32d78fd750fddcfb98404a249f37c138538fd8` |
| native/purged_spells/block2 | `85105f67be3c667bed8dc2bfc9456717de81e2025e6677b44ec8bd40ddc70b81` |
| native/purged_spells/block3 | `c0ab8573388e9701fa00974997ba38e6316d74c049005269d7a74134ee6713a5` |
| native/purged_spells/block4 | `7d9a9c3d0531f3f87499e4453dc9726710ee8db55da43c76d1ed08c9bdc552cb` |
| native/purged_spells/block5 | `6176fedb566486fee0276eb41a1e09152bb2bb762ad804ecf2982d73f17a402c` |
| native/buffered_100_seconds/block1 | `0dcee0f658db87f855bcb28bc9f0711e110cce2ff6411814e22c84251ceea818` |
| native/buffered_100_seconds/block2 | `636478612a6f55638f57c71e3f6e0d71d986e1df2c80833e95cd6a9972f73d54` |
| native/buffered_100_seconds/block3 | `3ed21362c8139922d92e768befc64e9be14802ec9aa9d71b71e751f5e2b21b07` |

The identical buffered-minutes/block1 and purged-spells/block1 hashes are expected: those two splits have byte-identical fitting archives, pools and schedules, and the deterministic fits also have identical loss/probe hashes. They were separate worker runs; the protocols remain correlated development evidence.

## Remaining limits

No payload, provenance, schedule, resource or retained-source-gate failure was found. Native external assessment, seed42 head screening, any eligible 43/44 confirmations, final selection, fresh neural/feature/head/probability/official/gate replay, full/public parity, repeated 199,319-row CSV construction, validator and independent promotion review remain outside this report. This is not a new public result, model promotion, private accuracy claim or goal completion.
