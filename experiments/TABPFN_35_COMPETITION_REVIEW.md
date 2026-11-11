# TabPFN 3.5 comparison — 3 November 2026

## Competition and model access

The local RULES.md permits pretrained and commercial models if organizers can
inspect and run the method and its technical document explains it. SUBMISSION.md
requires exact regeneration. The published event identifies Mobile World Capital
Barcelona as a co-organizer and describes a public predictive-model challenge.

The [3.5 license](https://huggingface.co/Prior-Labs/tabpfn_3_5/raw/main/LICENSE)
expressly includes qualifying data-science competitions in non-commercial purposes
(section 1c). This event appears to fit through its foundation co-organizer, but
this is an interpretation, not permission for commercial deployment. Sections 2d
and 3 govern outputs and distribution. The opening terms also exclude consumers
and require professional capacity. The team's enrollment agreement is not present
in this checkout; any conflicting obligations must be checked before submission.

[Local access instructions](https://docs.priorlabs.ai/models/accessing-model-weights)
require account sign-in and license acceptance for the weights download. Afterward
inference can run offline. No cached PriorLabs token or token environment variable
was found (existence checks only; no credential contents were read). No license has
been accepted or bypassed by this preparation.

## Base versus Thinking

[Official model availability](https://docs.priorlabs.ai/models) distinguishes
downloadable 3.5 Base from 3.5 Thinking, offered via hosted API/client or enterprise
deployment. Local Base is the proposed initial test. Hosted Thinking would require
a separate review of match-data upload permission under the enrollment terms and
provider terms; no dataset upload is authorized or performed here.

## Comparison prepared, not executed

`prepare_tabpfn35_comparison.py` records the five original main-fold input hashes
and comparison settings in `data/control_tabpfn35_v1/comparison_plan.json` using
only Python's standard library. It does not load a model, accept terms, download
weights, or start a queue. No new numerical score exists yet.

The test will reuse seed 42, identical 166-column features, the original paired
training contexts and all 7,500 held-out query frames, and the existing fixed
decoder. The sole intended model substitution is V2 to V3_5; version-specific
internal defaults will be recorded. The threshold is strictly above
0.7771428571428571 on the complete five-fold out-of-fold SpellF1. No public
within-sample fit may be reported as this comparison. If it wins, evaluate the
other existing protocols and seeds before any recommendation to replace V2.

## Protecting the active build

The existing full-match V2 controller and GPU worker were observed running and
have not been stopped or changed. No GPU or scientific model imports were used
for this preparation. The new test must wait until full prediction, exports and
finalization finish and the existing controller and workers exit. A separate
bounded synthetic memory/reproducibility gate precedes any 3.5 public evaluation.
Existing source, plans, runtime installations, candidate outputs and ledger are
read-only to this comparison. This preparation does not select a new model.

## Next dependency

The user must establish applicable PriorLabs license acceptance and sign in
through the official flow. Do not ask for passwords or API keys in chat. Once
access is established, acquire weights separately, bind their hash and runtime,
implement the isolated evaluator, and run only after the active workflow ends.
