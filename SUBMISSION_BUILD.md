# Rebuild the possession submission

The frozen recipe is `tabpfn_short_control_recovery_seed42`. The output has 199,319 rows and SHA256 `801cb64b59260daf4586865e9ef13d124fc61d7584e2798283e465a1758e9ed7`.

## Setup

Use Apple Silicon macOS with MPS, 16 GB of RAM, and Python 3.12.14. The historical TabPFN pass took about 16 hours on a 16 GB Mac. Keep several GB free for the environments, feature tables and query partitions. The build reads tactical and broadcast tracking, metadata and homography validity plus tactical Smart Tagging from the provided bundle. Set `MTAH_DATA_DIR` to the directory containing `tactical/` and `broadcast/`.

From the repository root, install the three pinned environments. The encoder uses Torch 2.5.0 on CPU and the TabPFN worker uses Torch 2.15.0 on MPS. Separate environments preserve the NumPy and scikit-learn versions used to produce the selected file.

```sh
python3.12 -m venv .venv
python3.12 -m venv .venv-encoder
python3.12 -m venv .venv-tabpfn
.venv/bin/python -m pip install -r experiments/requirements-research.txt
.venv-encoder/bin/python -m pip install -r experiments/submission_assets/requirements-encoder.txt
.venv-tabpfn/bin/python -m pip install --no-deps -r experiments/submission_assets/requirements-tabpfn.txt
```

## Build

Run the following command from a clean checkout. Use a new local output directory for the first run. The build fetches the pinned TabPFN checkpoint from its author and verifies it before loading. `--checkpoint` can supply a local copy of the same file.

```sh
caffeinate -i .venv/bin/python experiments/reproduce_submission.py \
  --work data/submission_rebuild \
  --encoder-python .venv-encoder/bin/python \
  --tabpfn-python .venv-tabpfn/bin/python
```

The build regenerates tracking features, both camera alignments, the player encoder outputs, the XGBoost heads and all 40 TabPFN partitions. It runs the frozen decoder, checks the output SHA256 and calls the official format validator. The final file is `data/submission_rebuild/submission/candidate.csv`. The command resumes from verified stages when rerun with the same arguments. A failed stage keeps its partial output in a hidden directory under the work directory for diagnosis.

The Git repository contains source, the public labels shipped by the organisers, and a small player encoder trained on public external data. Keep the competition data, generated features, predictions and submission CSV in local ignored directories. See the [competition rules](RULES.md) and the [encoder provenance](experiments/submission_assets/README.md).
