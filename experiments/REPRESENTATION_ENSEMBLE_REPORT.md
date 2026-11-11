# Round 41: fixed probability fusion rejected

On 27 October 2026, one source-backed equal-weight average of original slot0
and the Round37 pretrained-latent slot0 was frozen before scoring. The unchanged
selected8 decoder and live-only overlay preserved its D/X labels. No coefficient,
threshold, encoder or model sweep was performed. See the
[research note](RESEARCH_IMPROVEMENT_20261027.md) and
[prospective plan](REPRESENTATION_ENSEMBLE_PLAN.json).

| Protocol | Selected8 | Equal probability fusion |
|---|---:|---:|
| Buffered minutes | .741176 | .694118 |
| Purged spells | .685714 | .658824 |
| Buffered 100 seconds | .670270 | .655556 |

All three primary scores regress and short IoU.5 matches fall from three to two.
MacroF1 improves, but that does not repair the lost possessions. The candidate
also fails prior matched-possession and short-recall protections. It is rejected;
no confirmation fits, full-match inference, candidate or promotion follows.
The previous raw record remains Round37's .754286, and selected8 stays unchanged.

All 13 existing R37 heads reload with exact saved OOF probabilities. There are
zero new fits. Independent official/shared scoring and saved-model replay match
exactly, and all 17 selected artifact hashes and source links remain unchanged.
Six score rows append to the ledger, preserving its previous 1,531 rows/bytes
(1,537 total). Independent source review found no leakage or scoring error.

The implementation retains the older +.01 buffered-gain check as well as the
plan's strict-positive confirmation check. This is a stricter rejection condition;
it cannot explain away this failed screen and was not altered after scoring.
Historical +.04 and extra-short objectives also remain unmet. The coefficient
is not adjusted after this result.

Evidence is under ignored `data/control_representation_ensemble_v1/`. Reproduce:

```sh
.venv/bin/python experiments/research_representation_ensemble.py --replay --output data/control_representation_ensemble_v1/review_replay
```

All evidence uses the repeatedly adapted 17-possession public window. It does
not establish private full-match accuracy or a maximum achievable score. The
broader requested search continues with a separately frozen training hypothesis.
