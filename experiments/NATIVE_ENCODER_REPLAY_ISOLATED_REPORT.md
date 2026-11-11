# R61 reproduction completed through an isolated-runtime recovery

R61 remains **rejected**: native continuation scores.635294/.606061/.611765,
and source continuation.711111/.697143/.605405. Neither qualifies for
confirmations or promotion. Selected8 and the1,603-row ledger remain unchanged.

The previously completed independent replay produced14 checkpoints,14 loss
arrays,14 probe arrays and28 external probability arrays byte-identical to their
originals. Its final public-screen replay was interrupted with68 matching files
retained and no terminal manifest. The cause remains unknown. Separately, the
shared Python executable changed, correctly invalidating its frozen runtime
binding. The original interrupted directory and strict verifier are preserved.

The new R71 recovery uses the isolated project-local Python executable and
standard library. It explicitly replaces the single obsolete executable binding
and validates all other original source, data, model and runtime files, plus
the isolated interpreter, library links and configurations. The original frozen
files are untouched. The screen's sole AST change selects a new output directory;
the original encoder and prediction bodies remain intact behind a declared
runtime entry shim. No refits, external reassessment, tuning or extra score rows
occur in this recovery. Seventeen pure tests and root review precede its freeze.

Execution session95684 finishes with exit0 and no resource abort. All14 query
encodings are freshly computed; all26 saved XGBoost heads are reused. The result
verifies121 artifacts byte-identical to the main screen, including features,
probabilities, OOF labels, shared/official scores and selection gates. The main
rejection remains exact. All bound old files, selected artifacts and source links
verify unchanged. The390.023-second recovery samples peak combined supervisor
and worker-tree RSS at.639GiB, below its2GiB bound; sampling is not an
instantaneous-peak guarantee.

Authoritative result:
`data/control_native_encoder_continuation_extended_v1/screen_replay_isolated_result_v1.json`.
The completed new stage is `screen_replay_isolated_v1/`; the old partial
`screen_replay/` remains preserved. Source SHA256:
`e89f40b1f21812bc19427484bbfe0004a9ba96e268fbebc1a924bcdd160de954`.
Plan SHA256:
`28c4ac87b935cc682b21d5c397b73726e1685ed0954ad4ffc6764d59d965a342`.

This closes the reproduction work for the rejected experiment. It supplies no
new possession-score record, selected candidate or private-accuracy claim.
