# External-only possession experiments

The competition match was reserved for frozen inference. Fitting and recipe selection used external matches, with whole matches kept together for development and evaluation. The competition's public possession labels and video were excluded from this work. The external evaluations below measure how the recipes transfer across these datasets and tracking conditions.

## External evidence

| Recipe | DFL SpellF1 | SkillCorner SpellF1 | Combined | Decision |
| --- | ---: | ---: | ---: | --- |
| Original DFL control | 0.4730 | 0.3645 | 0.4188 | Reproducible reference |
| Richer DFL model | 0.5143 | 0.4277 | 0.4710 | Passed both external checks |
| Original DFL and SkillCorner blend | 0.4419 | 0.5258 | 0.4838 | Failed the DFL loss bound |
| **Richer DFL ownership with blended stoppages** | **0.5016** | **0.5380** | **0.5198** | **Selected** |

DFL reports average the two external validation matches with one-third clean and two-thirds masked tracking. SkillCorner reports average four match-disjoint validation matches using detected broadcast tracking. The combined column weights the two datasets equally. These external validation matches also informed the final model-family selection.

The richer DFL model adds ball speed, pitch-boundary, and sustained player-to-ball contact features. It uses clean tracking and two seeded corruption regimes in training. Its severe-tracking-loss SpellF1 increased from 0.2715 to 0.4232 across the two DFL validation matches. The heavier corruption introduces longer occlusions, false ball observations, and camera outages.

The switch-penalty decoder lost 0.0062 weighted SpellF1 on the external development match. The second-camera prototype failed the synthetic alignment gate. Both were excluded from submission inference.

The selected model keeps the richer DFL ownership probabilities and blends the stoppage probabilities with 25% from DFL and 75% from SkillCorner. It gains 0.1103 on the four SkillCorner validation matches relative to the richer DFL model, with DFL losses of 0.0191 and 0.0064, within the 0.02 guard. The initial confirmation completed scoring and hit a JSON serialization error. The successful confirmation replay used the same frozen recipe and checkpoint hashes.

## Verified output

The [submission CSV](data/external_only/submission.csv) has 199,319 rows and passed the official format validator. A separate rebuild from the [source-only release](data/external_only/release/README.md) reproduced it byte for byte. The replay used the pinned CPU environment and a competition bundle containing only the three tactical inputs. Six unit tests passed.

## Sources

- Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert, *An integrated dataset of spatiotemporal and event data in elite soccer*. The DFL position files supply ball status and possession targets. [Versioned dataset](https://doi.org/10.6084/m9.figshare.28196177.v1) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- SkillCorner Open Data, 20 Australian A-League matches at Git revision `4340d274572876239c154c90bc507a9b3250a656`. Phases of play and provider possession supply targets. Features use detected ball and player coordinates. [Pinned source and license](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656)
