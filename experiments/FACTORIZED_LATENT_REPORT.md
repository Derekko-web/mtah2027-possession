# Round 42: conditional live-owner training rejected

Two fixed Ordered CatBoost binary ownership heads were trained only on H/A
examples within the existing public training folds. One used the original 102
ordered features; the other added the 64 immutable pretrained player-set
features. Their symmetrized H/A probabilities retained original slot0 D mass,
then used the unchanged selected8 decoder and live-only overlay.
See [research](FACTORIZED_LATENT_RESEARCH_20261027.md) and
[prospective plan](FACTORIZED_LATENT_PLAN.json).

| Protocol | Selected8 | Original-feature binary | Pretrained-feature binary |
|---|---:|---:|---:|
| Buffered minutes | .741176 | .705882 | .694737 |
| Purged spells | .685714 | .666667 | .633333 |
| Buffered 100 seconds | .670270 | .610526 | .581818 |

Both arms regress on every primary protocol. The pretrained model improves
macroF1 but loses short-possession matches in the purged and longer protocols.
Neither reaches the historical .754286 record or the prospective safeguards.
Both are rejected; no confirmation, full candidate or promotion follows.
No parameter, feature, loss-weight, encoder, threshold or decoder sweep follows.

All 26 saved models reload with exact probabilities. Training indices, augmented
targets/weights, official/shared metrics and full saved-model score replays
agree exactly. Independent source/spec review finds no actionable leakage or
composition error. All 17 selected artifact hashes and source links stay exact.
Nine dependent score rows append, preserving all previous 1,537 rows/bytes
(1,546 total). Selected8 remains unchanged.

Evidence is under ignored `data/control_factorized_latent_v1/`. Reproduce:

```sh
.venv/bin/python experiments/research_factorized_latent.py --replay --output data/control_factorized_latent_v1/review_replay
```

The 17 public possessions remain adaptively reused development evidence.
Private full-match accuracy and maximum score remain unverified. The broader
user-requested research continues under a separately frozen hypothesis.
