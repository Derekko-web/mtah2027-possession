# Terminal R54/R55 artifact review

Independent, read-only artifact audit on 2026-10-28. Both frozen native screens
are complete and correctly rejected. No artifact mismatch or gate defect was
found in the original screens. **The later R54 fresh-context native replay
failed its frozen operation-time limit after three exact folds**; the addendum
below supersedes this audit's original pending-replay status. Neither candidate
is eligible for confirmation or a full candidate build.

Scope: the retained R54 `screen`, R55 `screen`/`screen_replay`, their frozen
plans, implementation/input/protected-artifact maps, saved fold inputs and
outputs, official reports, gate files, and ledger closeout evidence. No fitting,
model inference, decoder rerun, or model replay was performed. Matching and gate
arithmetic were independently checked using saved labels and metrics. No sealed,
manual, private, or reserved content was parsed; the reserved-file binding was
checked by hashing bytes only. R56 and its running CPU gate were not touched.

## Findings and open action

1. **Failed safeguard: the complete R54 native reconstruction replay did not
   pass.** The
   13 worker results, within-worker first-query repeats, immutable-cache checks,
   and passing R53 synthetic reconstruction prerequisite support the saved
   screen. They do not demonstrate independent native context regeneration.
   At the initial audit these records left full native reconstruction pending;
   the later run failed at buffered-minute block 4 under the unchanged
   300-second operation limit. There is no complete 13-fold replay or score/gate
   reconstruction, and no retry/time waiver is authorized for this rejected
   method. The observed score failure already rejects this interface.
2. **R55 replay is exact for substantive artifacts.** Its five aggregate
   score/diagnostic/gate files, all protocol OOF labels, raw probabilities,
   checkpoint/fitting records, scorer inputs, official scoreboards, and official
   JSON/transition reports are byte-exact between screen and replay. Of 71
   cache entries, only the three `official.log` files differ: each substitutes
   `screen_replay` for `screen` in the two output-path messages. All 68 other
   entries match. The separate manifests consequently differ, as expected.
3. **Selected artifacts and historical ledger remain intact.** No promotion or
   new score search is justified by these outcomes. The existing R54
   `active_run.json` is a retained launch observation, not present liveness or
   completion evidence; its own text requires revalidation. The terminal screen
   manifest and 13 successful supervision records establish completion here.

## Hash and provenance verification

Every recorded hash was recomputed from the current file. No mismatch or
missing file was found across 936 unique files. Both live plans equal their
frozen `plan.json` copies byte-for-byte.

| Binding | R54 screen | R55 screen | R55 replay |
|---|---:|---:|---:|
| Manifest input hashes | 91/91 | 88/88 | 88/88 |
| Frozen implementation/source hashes | 416/416 | 419/419 | 419/419 |
| Manifest cache hashes | 257/257 | 71/71 | 71/71 |
| All actual non-manifest files covered | exact | exact | exact |

The R55 plan itself binds 85 inputs; its terminal manifests additionally bind
R54's completed manifest, selection, and scores. All 17 protected selected
artifact hashes and both `data/tactical`/`data/broadcast` symlink targets equal
each round's frozen records. The shared protected hash maps are identical.

Useful terminal identities:

| File | SHA-256 |
|---|---|
| R54 plan | `53905855df022c57bdf545b5eed7087b3c9f5f4e4e3be300da9c3b1eca119032` |
| R54 screen manifest | `e76dc1dc27894def60a8beb150b08d21ae43bb298f9dacb46775ed3264f9fd83` |
| R55 plan | `85b388756e28daeb1463d3013c49347b9bba1c9ec2884d77cb667ef51fb0d5a1` |
| R55 screen manifest | `c92ffb8ab3f6392dbd61dd4ca22e4ac931f82fdedad58f2084a1976283dd3491` |
| R55 replay manifest | `557d6c259565b21eaab554a80d9b5bdc42b4658e0f81cc25fd7ea308560c77ad` |

## Fold, worker, and probability checks

The original splits contain 5 buffered-minute, 5 purged-spell, and 3
buffered-100-second folds. Within each protocol, their validation indices are
disjoint and cover exactly frames 11185–18684: 7,500 queries, including D frames.
R54 therefore retains 22,500 OOF query results across 13 context models.

For every R54 fold, saved training/validation indices equal the original split;
training and validation are disjoint. Saved context frames equal only the
permitted H/A training frames divisible by 5. Original-plus-swapped target
augmentation is exact and balanced. Contexts contain 1,368–2,042 finite rows of
166 features, below the 3,000-row cap. Original and swapped query arrays are
finite, have the expected full validation shape, and are the only query inputs
listed in the worker jobs; jobs contain no query labels or D mass. Job input
hashes and recipe context/target bindings verify.

All 13 worker summaries pass; all supervision records report exit code 0 and no
abort. The recorded eight-member CPU, one-job, temperature .9 recipes agree.
All 90 original/swapped query timing records follow ascending 512-row call
boundaries, including the final partial chunk. Worker output schemas, finite
nonnegative probabilities, probability mass, exact swapped-column averaging,
and assembly into the saved OOF owner arrays verify. Final operation records
show completed first-query repeats. Cache fingerprints are saved and bound by
both worker and screen hashes; independently rebuilding their tensors is the
   incomplete R54 safeguard, not a result of this audit.

| R54 recorded maximum | Value | Frozen limit |
|---|---:|---:|
| Context fit | 134.433 s | 300 s |
| Query call | 46.906 s | 300 s |
| Worker wall | 559.885 s | 900 s |
| Sampled worker RSS | 4.552 GiB | 6 GiB |
| Worker high-water RSS | 4.787 GiB | 6 GiB |
| Sampled combined live-worker RSS | 5.926 GiB | 9.5 GiB |

R55 reuses all 13 R37 checkpoint files and fitting records exactly in both
screen and replay. Their saved training/validation indices match the original
splits. Each retained `(7500, 3)` raw probability array is finite and exactly
equals R37's corresponding array; row-mass error is below 1.1e-7. The runner's
seed42 path reloads/copies the checkpoints and reports zero new models.

## Labels, official scores, and gates

Each protocol's retained current labels exactly equal the selected8 reference.
Both candidates preserve D/X on all 199,319 rows of their official scorer inputs,
including outside-public X rows. R55 also retains the H/A run-label sequence
within every original decode block and selected live component.

All 24 retained score rows (6 R54, 9 R55, 9 R55 replay) agree with their official
JSON spellF1/macroF1 and 0.5-IoU match/spurious/missed counts. Their fold counts
and arithmetic mean-fold spellF1 verify. Official reports grade 7,500 rows with
zero abstentions on graded frames.

Every saved R54/R55 gate flag, changed-frame count, score delta, and final failed
conjunction was checked against the saved labels, short-segment diagnostics,
official greedy matching, and frozen arithmetic. No mismatch was found.

| Protocol | Selected8 | R54 TabICL | R55 boundary risk |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .694118 | .635294 |
| Purged spells | .685714 | .611111 | .594286 |
| Buffered 100 seconds | .670270 | .651429 | .616216 |

All candidate primary and mean-fold checks fail. R54 loses one previously
matched GT possession in each protocol and has 2 short matches in each. R55
loses 2/2/1 prior matches and has 1/1/2 short matches. Both preserve D recall and
false-D counts. Neither exceeds the strict .7542857142857143 buffered research
record. R55's unchanged R37 comparator is diagnostic and does not grant the
boundary-risk method eligibility.

## Ledger verification and limits

The ledger contains exactly 1,591 data rows. The six R54 rows and nine R55 rows
match their retained score sources, source hashes, protocol/model identities,
spellF1, and macroF1. The first 1,576-row byte prefix has SHA-256
`69a92e5ed6f640f04f64f8e4feb95cc65af667304fe9502d260d162b0b625f4a`;
the first 1,582-row prefix has SHA-256
`3b3f5b583a85d73c858465cf8828f46319f4e907302b3388c13217aaad97dfa4`.
Both equal the recorded append/closeout bindings. Each round's frozen
1,576-row ledger snapshot is also an exact prefix of the current ledger.

This establishes integrity and consistency of retained public-development
results. It does not establish an optimum, independent holdout performance,
private/full-match accuracy, fresh R54 native reconstruction, or the absence of
unobserved behavior beyond the retained records. The user's strict-record plus
safeguards objective remains unachieved.

## Later R54 replay addendum, 2026-10-29

A lightweight read-only check of `data/control_tabicl_native_v1/screen_replay/`
confirms three completed buffered-minute folds. For blocks 1, 2, and 3, worker
summaries pass and the saved cache fingerprint, owner probability, original
probability, and swapped probability files each equal the original screen
byte-for-byte. Each covers 1,500 queries. Their worker walls are
465.759/454.373/571.944 seconds and their supervision records show no abort.

Block 4's supervision instead records `operation_over_300seconds`, exit code
`-15`, and worker wall 307.271 seconds. Its last operation is still marked
active `context_fit`; that stale active field records the operation interrupted
by termination, not a live worker. There is no block-4 success summary. Block 5
has prepared inputs but no supervision/operation result. No later protocol
worker or terminal replay manifest was produced. The controller log terminates
in the expected retained worker-failure exception; controller exit 1 and process
shutdown were independently reported by the parent.

The original screen and failed replay evidence are preserved. The failed
resource safeguard is not recast as numerical mismatch, nor are three exact
folds recast as a 13-fold success. The earlier public-performance rejection
remains unchanged. No rerun or larger timeout was performed by this reviewer.
