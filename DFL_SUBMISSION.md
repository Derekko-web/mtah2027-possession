# Team possession from tactical tracking

This submission predicts home possession (`H`), away possession (`A`) or a stoppage (`D`) from the tactical tracking stream. It writes `X` outside the two half spans supplied by the organizers. The fitting and decoder selection use only external DFL matches.

## Data and model

The training data are the [versioned DFL dataset](https://doi.org/10.6084/m9.figshare.28196177.v1) by Manuel Bassek, Robert Rein, Hendrik Weber and Daniel Memmert, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The XML parser maps inactive ball status to `D`, active status with a known owner to `H` or `A`, and other cases to `X`. Status and owner supply training targets. Player and ball coordinates supply the features. The corpus manifest records the download URLs, file sizes and SHA-256 digests.

Two CPU XGBoost models use relative geometry and motion in metres. The ownership model estimates home control conditional on live play using nearest-player distances, local player counts and recent and upcoming motion. It trains and predicts with the teams exchanged as well as in their original order. The live/dead model adds team-invariant player motion and shape. Training uses every fifth frame from the four fitting matches, with equal weight per class and match and both clean and synthetically degraded tracking. The degraded view adds fixed coordinate noise, player dropout and runs of missing ball observations.

Competition inference reads three supplied files, the tactical ATD coordinates and team metadata and the tactical homography validity flag. It masks invalid geometry, computes probabilities separately for each official half span and applies the frozen decoder.

## Possession decisions and external validation

The decoder averages stoppage probabilities in a centered window and marks a stoppage when the average reaches its threshold. Within each predicted live spell, centered averaging of the ownership probability gives brief changes during loose balls and challenges less weight. Ownership windows stop at decoded stoppages and half boundaries. Restarts follow the tracking evidence for the new spell.

The fitting matches are `WMX`, `WN1`, `WOH` and `WOY`, with match IDs prefixed by `DFL-MAT-J03`. We tune on `WPY`, comparing 36 combinations of the stoppage threshold and the two smoothing radii. The selection metric is one-third clean and two-thirds degraded SpellF1, with weighted MacroF1 as a tiebreaker. We freeze the recipe before validating on `WQQ` and `WR9`, and require it to beat the raw nearest-player baseline on both views of both matches. The two validation matches were used in earlier DFL research and are held out from this fit and decoder selection.

The fit used 241,556 paired ownership rows and 226,916 live/dead rows. The chosen stoppage threshold is 0.50, and both centered windows extend up to 50 frames (2 seconds) on either side. The frozen recipe passed the external gate.

| DFL match and view | SpellF1 | Raw SpellF1 | MacroF1 |
| --- | ---: | ---: | ---: |
| WPY development, clean | 0.544 | 0.090 | 0.918 |
| WPY development, degraded | 0.404 | 0.017 | 0.843 |
| WQQ holdout, clean | 0.593 | 0.080 | 0.926 |
| WQQ holdout, degraded | 0.438 | 0.018 | 0.852 |
| WR9 holdout, clean | 0.531 | 0.101 | 0.920 |
| WR9 holdout, degraded | 0.419 | 0.016 | 0.869 |

The [external validation report](dfl_submission/models/external_validation.json) contains the full search, per-match metrics and holdout decision. The [frozen recipe](dfl_submission/models/recipe.json) records the selected settings and checkpoint digests.

The external metrics measure agreement with DFL definitions. Challenge tracking can lose the ball or switch player tracks differently from the artificial degradation, and the two competitions may define loose balls and stoppages differently. Smoothing can merge brief possessions or move their boundaries.

## Reproduce

The verified environment is Apple Silicon macOS with Python 3.12.14 and the pinned CPU dependencies. XGBoost uses the system OpenMP runtime on macOS. From the repository root, install the environment and point `MTAH_DATA_DIR` at the directory containing `tactical/`.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r dfl_submission/requirements.txt
export MTAH_DATA_DIR=/path/to/competition-bundle
.venv/bin/python -m dfl_submission.build --out data/dfl_only.csv
.venv/bin/python evaluation/validate_submission.py data/dfl_only.csv
```

The build checks the frozen checkpoint hashes and validates the exact CSV before writing it. To reproduce the fallback with the same frame numbering and the starter's raw nearest-player rule, use the same command with `--raw`.

```sh
.venv/bin/python -m dfl_submission.build --raw --out data/raw_fallback.csv
```

Expected SHA-256 digests for the supplied competition bundle are below. Each CSV has 199,319 data rows in tactical frame order.

| Output | SHA-256 |
| --- | --- |
| DFL submission | `fe34cb5d884509b54d282e6612cfd4ced3e9759b30b617bb61c771fc95a40e6f` |
| Raw fallback | `9471afeb95722454a96189481561adfb73b3963365a83969f04a986930f12511` |

To audit the full training run, download the 14 XML files listed in `dfl_submission/dfl_sources.json` into a single directory. The training command checks their sizes and hashes, fits the checkpoints, freezes the decoder and writes the external validation report. It takes a new output directory and can run with the competition bundle absent.

```sh
.venv/bin/python -m dfl_submission.train --dfl-raw /path/to/dfl-xml --out data/dfl_refit
.venv/bin/python -m dfl_submission.build --model-dir data/dfl_refit --out data/dfl_refit.csv
```
