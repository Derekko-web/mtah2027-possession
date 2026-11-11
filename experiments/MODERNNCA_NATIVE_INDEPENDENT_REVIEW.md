# R68 independent source review

**Approved for root freeze at the exact hashes below. No unresolved blocking
findings remain.** This approves the prepared prospective protocol, not a native
score, model promotion or automatic execution before root schedules resources.

Reviewed the complete native core, controller, plan and pure tests against the
frozen R64 runtime, author source/defaults, exact 166-column schema, original
selected8 assessment and factorized/live-overlay interfaces. This review made
no implementation changes, scientific imports, array loads, model calls or
freeze. Only this review document is new. Pure fixtures used invented lists
and literal temporary files.

| Artifact | SHA256 |
|---|---|
| `experiments/modernnca_native.py` | `5ec3d47acd2967bc53b2a8a8eaff5ca9f7160eb1f4cecdd68daf31ec1b25863e` |
| `experiments/research_modernnca_native.py` | `2a5b64e936dbc502ffa827b9e00cfde416e034bf3d965f7cc29593d204a56fb5` |
| `experiments/test_modernnca_native.py` | `af0ffea357b6c14055389e92389ecc68f6d9feb0529a2488de856e01f4141aed` |
| `experiments/MODERNNCA_NATIVE_PLAN.json` | `5177bbca1efd3e390360abb0fd628078cb8fc17c12ef49ebdba3b77411a62c0a` |
| `data/control_modernnca_native_v1/source_closure.json` | `dd4779bb3b14b8654b2d8b95e7785831e637b08565728d1556353408d3fcec51` |

## Findings resolved before freeze

1. The initial 680-entry source closure unnecessarily bound the mutable
   `data/control_tabpfn_native_v1/pure_tests.log` from unfinished R65. Its author
   removed only that unrelated entry; the 679-entry closure now has no active
   R65/R67 round coupling. No learner, feature, decoder or criterion changed.
2. The initial final replay marker relied on replay-directory names and OOF
   equality without independently requiring all 39 refit proofs. A normal run
   directed to a `_replay` output path could therefore lack the intended parent
   comparisons. The final preparation explicitly records stage/seed/replay mode
   and refit counts, then verifies every expected worker and its evidence before
   writing the completion marker. The independent review inspected this complete
   patch and reran its positive and negative pure fixtures.

## Learner and temporal isolation

R68 binds the unchanged R64 source and plan, requires the exact successful
four-stage runtime summary, checks all four stage identities/exit statuses and
resource results, and verifies the retained independent-refit evidence. The
bound R64 summary is
`f00bc2d9fc05456c7decd795fc60898266cec962bf4894196b7aa2ebc3dad47e`.
I independently verified that summary and its four stage-summary hashes using
JSON metadata only: both full fits completed in 262.655/261.422seconds, with
4.019/3.992GiB measured peaks and successful independent refit flags. This audit
did not reopen the retained array binaries.

The native worker calls the frozen R64 author model/helper, exact extracted
training definitions, orbit-tied statistics, preprocessing and canonical
prediction functions. PLR-lite77/34, metric128, zero blocks, temperature1,
sampling.5, AdamW.01/.0002, batch1024 and float64 remain exact. Every fit runs
all200 epochs and uses the final checkpoint; there is no validation-driven
selection or new tuning arm.

The context selector checks the unchanged250-frame exclusion and complete
inclusive ±79-frame own-training footprint before H/A filtering. Its exclusion
test matches the frozen split builders, whose nearest permitted training frame
is251 frames from validation. Only ascending 5-Hz own-training H/A frames enter
the paired pool, with complemented H0/A1 targets and a hard3000 augmented-row
cap. Mean/scale derive only from that paired pool. The exact original102 plus
R37 pretrained64 column names and team permutation are checked; no R61 encoder
or later-selected representation enters this method. Validation-label masking
checks the context construction again before fitting.

Every validation frame is queried regardless of its label. Canonical512 calls,
last-row padding, independent original/swapped calls and reversed H/A averaging
reuse R64. Seed streams are prospectively fixed to public seed / seed+1 /
seed+2 for Python / NumPy / Torch. Initial/final model state, context arrays and
tensors, strict checkpoint reload, repeated probabilities, configuration and
recomputed symmetrization are checked. Runtime isolation retains the read-only
repaired Torch2.15/NumPy2.1.2 target without mixing the controller's packages.

## Selection, failures and replay

The controller covers all13 unchanged5/5/3 folds, queries all7500 public frames
per protocol, reproduces selected8 and uses R37 only as a named diagnostic.
Only the fixed ModernNCA candidate enters the exact required guard conjunction.
Seed42 must strictly exceed `.7542857142857143` buffered spellF1 and pass every
selected8 primary, mean-fold, macro, short-IoU, prior-match/short-recall,
spurious/missed, D-recall, falseD and buffered+.01 condition. Seeds43/44 require
a passing seed42 screen, and all nine exact seed/protocol pairs must pass.
Historical substantial-gain and extra-short failures remain separate history.

Integration calls the unchanged factorized decoder: original slot0 D and live
mass, unsupported-latent fallback, other five inputs, full selected8 decoder and
final live overlay remain intact. Both D and X decisions are asserted exact.
No full199319-row construction is authorized by these public-stage markers.

The new `verify_replay_evidence()` rejects missing or misidentified stages,
normal runs stored under replay names, incorrect mode/counts, incomplete worker
sets, false refit flags, nonzero/aborted or unbound supervision, wrong/self/
cross-seed parents, missing artifacts and altered state even when hashes are
re-sealed. It requires precisely13 distinct5/5/3 workers per seed, checks each
job's original same-seed/stage/fold parent, verifies all retained file bindings,
and compares context/query payload, provenance, initial/final state, optimizer,
configuration, recipe, loss and probability bytes with originals. The final
marker persists the counted39 proofs; Torch archive bytes are hashed for their
own integrity while tensor-state witnesses establish original/refit equality.
The existing replay controller also requires exact shared/official scores,
selection gates and OOF labels.

Resource limits remain one CPU2 worker, RSS6GiB, fit1800seconds,
prediction300seconds and worker2400seconds. A resource abort cannot pass via a
pre-existing success summary. Numerical/resource failures close the fixed
method; provenance, schema, state and replay errors abort globally. No partial
fold/seed rescue or retry is introduced. Selected17 artifacts, source links,
all old frozen sources/dependencies and the ledger prefix remain protected.

## Verification and limits

All **24 pure tests independently pass**, including the complete literal39-worker
acceptance fixture and its negative replay-evidence cases. I verified the final
three source/plan hashes, compiled their source, inspected the final bound
source-only preflight result, and independently rehashed426 Python sources in
the679-entry closure. The final preflight reports no scientific imports, native
array loads or model forwards and binds these same final hashes.

No native quality result exists as part of this review. Public protocols and
confirmation seeds still reuse17 possessions; even a later passing development
screen cannot establish private/full-match improvement or an optimum. Root
review/freeze and resource scheduling remain the next execution boundary.
