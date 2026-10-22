# Expanded research round — 21 October 2026

Historical expanded round. The subsequent refinement and current default are in
[CONTROL_REFINEMENT_REPORT.md](CONTROL_REFINEMENT_REPORT.md).

This records the broad search requested after the first continuation round did
not establish an improvement. All methods use the same repeatedly inspected
7,500 public labels; score rows are combinations of methods, objectives,
decoders, protocols and seeds, **not independent experiments**. The selected
pipeline and its final validation are documented in `SELECTED_MODEL.md`.

The final selection is ordered XGBoost. CatBoost was the initial primary finalist
and had a slightly higher headline score, but its longer-block gains did not hold
across fixed seeds. XGBoost improved every paired seed/protocol. This decision
and the accepted regressions are documented in `TEMPORAL_REPORT.md` and
`data/advanced_validation/selection_decision.json`.

## What was tried

| Question | Reproduction | Local output | Finding |
|---|---|---|---|
| Can a dedicated dead/live detector fix the visible-ball shortcut? | `research_stoppages.py` | `data/stoppage_research/` | Some stoppages recover, but false dead predictions and possession fragmentation reduce spellF1. |
| Do whistle/crowd cues help? | `research_stoppages.py --audio` | `data/audio_research/` | Local spectral cues help frame classification in some variants; primary interval scores fall. Audio is extracted locally with ffmpeg. |
| Can collective inactivity establish a stoppage? | `research_stoppages.py --tempo` | `data/tempo_research/` | Low player speed occurs during live play too; false dead intervals increase. |
| Can contact-supported changes or conservative probability fusion improve sequences? | `research_sequence.py` | `data/sequence_research/` | Adaptive changes give a modest gain, but leave both visible-ball stoppages missed. |
| Is camera pan/zoom informative? | `research_camera.py` | `data/camera_research/` | Homography-derived motion does not improve the primary metric. |
| Can visible-ball physics be constrained to generalize sensibly? | `research_monotone.py` | `data/monotone_research/` | Monotone movement/ball-distance heads recover more D, but add too many false dead frames. |
| Can other fixtures supply representative examples? | `external_stoppages.py`; `research_external_teacher.py` | `data/external_metrica/`, `data/external_teacher_research/` | Motion teachers transfer poorly. Event-derived dead labels in both external games almost always have missing ball, reproducing the same shortcut. |
| Can tagging semantics improve control continuity? | `research_phase_control.py` | `data/phase_control_research/` | Attacking/defending phase priors overmerge possessions. |
| Can a learned combination choose between detectors? | `research_stacked_dead.py` | `data/stacked_dead_research/` | Fully nested outer/inner exclusion avoids leakage but does not improve spellF1. |
| Does stronger dead/live persistence help? | `research_sequence.py --stoppage-penalty` | `data/duration_research/` | Better dead recall in some variants, worse or negligible primary gains. |
| Can asynchronous inactivity, far-ball and slow-ball evidence be pooled? | `research_pooled_stoppages.py`; add `--whole-episodes` | `data/pooled_stoppage_research/`, `data/episode_stoppage_research/` | Substantial MacroF1 gain, limited spellF1 gain. Unbounded whole-episode pooling remains a rejected prototype, not a feature in the selected model. |
| Do ordered observations distinguish approach, control and release better than averages? | `research_temporal_control.py` | `data/temporal_control_research/` | Yes. Short contact history and time taps improve primary scores. Dropping absolute goal/touchline distance helps visible-ball D detection. |
| Can the old and ordered models complement each other? | `research_temporal_fusion.py` | `data/temporal_fusion_research/` | Some large MacroF1 gains, less primary improvement than the selected ordered model. |
| Is the result merely a decoder choice? | `research_temporal_decoding.py`; `research_context_ablation.py` | `data/temporal_decoding_research/`, `data/context_ablation_research/` | Ordered inputs help with the original decoder. The original occlusion prior remains useful. |
| Should training emphasize boundaries or artificial occlusions? | `research_boundary_training.py`; `research_ball_dropout.py` | `data/boundary_training_research/`, `data/ball_dropout_research/` | Neither improves the strongest ordered model. Occluded training copies are created without labels; validation observations stay untouched. |
| Does the learner/objective matter? | `research_catboost.py`; `research_objectives.py` | `data/catboost_research/`, `data/objective_research/` | Class-balanced frame training performs better here than segment weighting or their average. Ordered CatBoost is the predeclared primary finalist. |

The external data is Metrica Sports' public anonymized
[sample-data repository](https://github.com/metrica-sports/sample-data), pinned
at commit `e706dd506b360d69d9d123d5b8026e7294b13996`. These are different
fixtures and weak event-derived labels, not the competition fixture's private
or corrected possession labels. Nothing from these games is included in the
selected model. Attribution and download hashes are retained locally.

## Local video feasibility probes

`local_vision_probe.py` uses 15 uniformly spaced public frames with a fixed
playing/stopped prompt. `local_vision_sequence_probe.py` uses the same sample
locations, three images one second apart, and automatic homography/ball-centred
crops. Neither changes final prediction frames or supplies pseudo-labels to the
selected model. Answers, crops and images stay under ignored `data/`.

Moondream 1.8B returned unusable binary answers. Qwen3-VL 4B labelled every static
sample live, missing all three sampled stoppages. The three-frame probe still
missed all three and returned seven unclear answers. These are small adaptive
feasibility probes, not evidence that video methods generally cannot work.

All inference used an Ollama server at `http://127.0.0.1:11434`, never a hosted
model. Model assets were downloaded separately before inference. Model tags and
digests, prompts and aggregate diagnostics are saved in the local summary files.
The [official FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md)
prohibits sending competition extracts to hosted/cloud LLMs; no such uploads
occurred. No new manual annotations were created. The frozen 24 annotation
windows remain unused and unlabelled.

For local-only reproduction, install Ollama and ffmpeg, pull `moondream:1.8b`
and `qwen3-vl:4b`, start the local server, then run:

```sh
.venv/bin/python experiments/local_vision_probe.py
.venv/bin/python experiments/local_vision_probe.py --model qwen3-vl:4b --out data/qwen_vision_probe
.venv/bin/python experiments/local_vision_sequence_probe.py
```

## Reproduction and provenance

The final comparison is much smaller than this search and builds its own tracking
features. To reproduce all 20 scored research families after `run_selected.py`
and `run_continuation.py` have populated their original caches:

```sh
.venv/bin/python -m pip install -r experiments/requirements-research.txt
.venv/bin/python experiments/run_expanded_research.py
.venv/bin/python experiments/validate_temporal_candidate.py
.venv/bin/python experiments/finalize_temporal_validation.py
.venv/bin/python experiments/update_experiment_ledger.py
```

For a separate validation output directory, pass the same `--out` path to both
`validate_temporal_candidate.py` and `finalize_temporal_validation.py`.

Each research directory contains complete OOF probabilities or predictions,
score/fold-score/true-segment tables and, where fitted, model checkpoints. The
aggregate `EXPERIMENT_LEDGER.csv` retains failed methods and source table hashes.
Two early scratch screens are retained separately under
`data/stoppage_fusion/screen_scores.csv` and `data/tempo_prior/screen_scores.csv`.
The 12 fusion rows were subsequently reproduced by `research_sequence.py`;
the three direct tempo-prior rows are preliminary diagnostics, not separately
counted independent evidence. Neither screen affects the selected candidate.

The broad search used the original cache ending at frame 18684. Future-looking
features near that cache edge were truncated. This was identified before final
validation: all finalists were rebuilt with observations through frame 18934,
without extra labels, and then compared again. The full-match build asserts that
every public feature value exactly matches those padded validation features.
The results reported for the final candidate use that corrected context.
