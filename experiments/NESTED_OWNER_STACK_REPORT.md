# Round 46: nested ownership stacking rejected

The source-backed two-expert ownership stack is complete and rejected. Each
outer fold uses three fixed inner time blocks with 250-frame exclusions to train
the original102 and pretrained166 experts without outer validation labels.
A fixed two-coefficient logistic learner uses live inner-OOF log odds, mirrored
for team symmetry. It preserves original D probability mass and uses unchanged
selected8 decoding plus the live-only overlay. See the
[research note](NESTED_OWNER_STACK_RESEARCH.md) and
[frozen plan](NESTED_OWNER_STACK_PLAN.json).

| Protocol | Selected8 | Nested ownership stack |
|---|---:|---:|
| Buffered minutes | .741176 | .715152 |
| Purged spells | .685714 | .678788 |
| Buffered 100 seconds | .670270 | .628571 |

Primary scores regress in all three protocols, despite increased macroF1.
Short IoU.5 matches fall from three to two, two, and one respectively. No
candidate advances to confirmation, full-match inference or promotion. There
is no meta-parameter, input, expert-weight, inner-fold or decoder search.

All 72 inner expert checkpoints, 13 outer R37 checkpoints and 13 meta models
have exact saved-model probability replay. Training records, inner coverage,
shared predictions, official scores and gates replay exactly. Four synthetic
tests verify exclusions, target mirroring and log-odds inputs. All 17 selected
artifact hashes and source links stay unchanged. Six rows append while
preserving the prior 1,564 rows/bytes (1,570 total).

Evidence is under ignored `data/control_nested_owner_stack_v1/`. Reproduce:

```sh
.venv/bin/python experiments/research_nested_owner_stack.py --replay --output data/control_nested_owner_stack_v1/review_replay
```

Nested fitting prevents the identified label-leakage path; it does not remove
adaptive selection bias from repeatedly using 17 public possessions. No private
accuracy or maximum score is established. Selected8 remains the default and
the recorded buffered development high remains Round37's .754286.
