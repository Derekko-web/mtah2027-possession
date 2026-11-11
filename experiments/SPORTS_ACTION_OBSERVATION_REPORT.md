# Football action observations: ongoing follow-up

The completed runtime preflight in SPORTS_ACTION_RUNTIME_REPORT.md used synthetic inputs only. The subsequent observation stage has now processed competition images locally; those images and raw head outputs remain under ignored data/ and never enter hosted model context.

The frozen observation protocol schedules 147 clips. The original producer completed 93 before reaching the retained 0.6 MPS allocation cap. All completed outputs, dispatch source and logs are preserved in data/control_sports_action_v1/observations_initial_memory_failure/. No support statistics, possession targets, fitted models or new possession scores have been inspected in this stage.

A persistent float32 input-buffer and cache-release synthetic trial also reached the cap. Its plan, source, progress and terminal log are preserved separately. A second frozen execution-only trial uses persistent uint8 input storage, with the original upstream float32 normalization. It must pass numerical head parity and 187 changing-input full-resolution synthetic forwards under the same cap before restarting private extraction. Resolution, clip length, weights and selection gates are unchanged.

The fitting extension retaining the three original decoder comparisons and adding the matched joint-slot comparison was frozen before support statistics or fitting. Neither trial has changed the selected candidate or appended possession score rows. The ledger remains 1,402 dependent rows: 778 inherited and 624 added during this continuation.

The user suggested Colab T4 as a fallback. FAQ.md prohibits restricted inputs to third-party hosted/cloud LLMs; that paragraph alone does not establish whether generic Colab GPU processing is permitted. No Colab session or upload has been initiated.

The uint8-storage trial passed a small synthetic comparison with maximum absolute head difference 0.0. Full production unittest discovery passed 175 tests. The changing-input endurance test is still live; this numerical comparison is execution evidence, not possession accuracy.

The uint8-storage endurance completed successfully: 187/187 forwards, zero post-warmup allocation span, 2,428,289,024 bytes maximum driver allocation after clearing and 6,662,455,296 after a forward. The process exited 0. The bounded producer has been dispatched for a new complete 147-clip generation; the old 93 clips remain separate. These figures are sampled driver allocations, not measured intra-forward peaks.

At 146 completed clips, the current FFmpeg process remained live and decoding at high CPU. A synthetic reproduction confirmed an efficiency bug: placing -t after -i (output duration) with selective filtering decodes all 150 synthetic source frames, while input-side -t decodes 30, with identical 19,200 output bytes. A separate producer with an input-side duration limit is prepared for future generations; the active dispatch source is unchanged. This is a decoder execution issue, not a changed sampling protocol.

The owned decoder was interrupted with SIGINT after the synthetic diagnosis. Its flush allowed all 147 clip outputs to finish, then the producer exited 1 on the nonzero decoder status. The complete-but-unfinalized generation is preserved as observations_output_duration_interrupted/, alongside its exact producer source and interruption record. A new complete generation with input-side -t was dispatched; no outputs are mixed across generations. This restart is due to a confirmed execution bug, not an observation timeout.

The input-limited producer completed all 147 clips and exited 0. All input RGB hashes and all head arrays match exactly against both the original 93-clip generation and the interrupted 147-clip generation (maximum absolute differences 0.0). Three independent accurate-seek decoder checks passed. The actual 18,935-versus-19,060 source-prefix feature comparison is bit-exact over all 7,500 public rows.

Target-blind support passes: 7,351/7,500 public frames observed, 60 confident separated event peaks, event-max standard deviation 0.201613, and actor geometry coverage 0.978400. Both predeclared heads are eligible (121 event features; 169 event-plus-actor features). These support figures do not measure action or possession accuracy. The frozen 26-fit, 39-row screen has now been dispatched.

## Result: rejected

All 39 dependent score rows agree with the official scorer. All 26 new checkpoints and 13 existing control checkpoints reproduce their saved probabilities exactly; all four decoders replay exactly. Training-only targets, weights, 250-frame exclusions, source/input/cache hashes and actual prefix parity pass. No finalist passes the frozen gates. Across 104 new-head clipped-block comparisons, 38 primary regressions occur.

| Decoder | Buffered spellF1 | Purged spellF1 | Longer-block spellF1 |
|---|---:|---:|---:|
| Selected reference | .741176 | .685714 | .670270 |
| Event-only joint slot | .741176 | .685714 | .670270 |
| Event-plus-actor joint slot | .741176 | .685714 | .670270 |
| Event-only direct replacement | .622222 | .622222 | .460000 |
| Event-plus-actor direct replacement | .651429 | .651429 | .470000 |
| Either action head, dead trim | .729412 | .685714 | .652632 |

Both joint-slot alternatives leave all selected labels byte-exact across all three protocols. Dead trimming reduces falseD but violates other gates: buffered D recall falls from .748128 to about .705, and purged D recall from .784888 to .740640. All three short IoU .5 matches remain three in these joint-slot/dead-trim comparisons; there is no extra recovered short match. The original substantial-gain gate remains unmet. No seed43/44 confirmation, dense full-match inference or candidate promotion is justified.

The first finalizer run completed official scoring/replays and appended 39 rows, then failed in its optional history audit because the technical-stage baseline dictionary did not include a ledger hash key. Its exact source and log are retained. The repaired audit checks the archived 1,402-row ledger against the technical-stage row count and verifies every archived byte remains the current ledger prefix; the 17 protected candidate/reference artifacts also remain exact. The rerun passes and appends zero duplicates. No model, probability, prediction or score changed in that repair.

The ledger now contains 1,441 dependent rows: 778 inherited and 663 added in this continuation. This round adds 3 selected-reference rows, 12 reused matched-control rows and 24 new-head/decoder rows, not 39 independent experiments. The complete evidence is under ignored data/control_sports_action_v1/, including inputs/, screen/, integrity.json, screen_checks.json, all failed runtime sources/logs, and both execution-parity reports. All 175 production tests pass. Selected8 remains unchanged; private/full-match maximum accuracy is unverified. No images, arrays or restricted material were uploaded, and no submission was made.

A next hypothesis is broadcast-camera action inference because the checkpoint was trained on broadcast footage. The broadcast video is not synchronized by frame number to tactical video: it has 164,488 frames versus 199,319. Any next protocol must use the preserved target-blind alignment, reject unsafe clip mappings and respect the same sealed windows and target buffers. Merely changing the video path would be invalid.
