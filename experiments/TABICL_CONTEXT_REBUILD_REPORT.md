# Round 51: independent context reconstruction fails exact probability replay

Two fresh processes reconstructed the unchanged official eight-member cached
estimator from the exact saved synthetic context/targets and fixed parameters.
This tests the supported `fit(context,target)` path rather than regenerating a
serialized fitted object. There are no fitted neural weights to transfer: the
checkpoint remains unchanged and `fit` constructs preprocessing/context caches.
See [the frozen plan](TABICL_CONTEXT_REBUILD_PLAN.json).

Both workers stayed below6GiB and completed their fit and predictions within
300seconds. The first worker exactly matched the Round49 full-fit caches and
probabilities. The second also matched every original cached tensor and its
within-process repeated predictions, but differed from the original probability
array on628/1024 values, by at most1.013279e-6. Its required cross-process
bit equality fails. Thus this configuration is rejected; numerical proximity is
not substituted for the frozen exact-replay criterion.

An initial parent-only launcher error attempted to set Torch's one-time
inter-operation thread setting twice. Its source and failure log remain intact.
A bound orchestration repair uses import-only parent setup, preserves the
completed first worker, and invokes the unchanged remaining frozen worker in a
fresh subprocess. This changes no model, context, numerical setting, limits or
gate. The subsequent probability failure remains a failure.

Failed numerical output details survive in the original assertion traceback;
the second probability array was not saved because the assertion preceded its
save. No argmax or downstream-label equivalence is claimed for that failed
worker. Evidence is under ignored `data/control_tabicl_context_rebuild_v1/`.
The Round49 failure remains preserved. No native model, possession score,
candidate change, private/manual labels, upload or submission follows.
