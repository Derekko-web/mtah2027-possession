# Pretrained football-action runtime: technical preflight

The public team-action checkpoint is technically usable on this machine. It loads
strictly, runs locally at its trained image geometry, and reproduces synthetic
outputs. No competition image or possession target was processed during this technical preflight. The subsequent local observation follow-up is documented in [SPORTS_ACTION_OBSERVATION_REPORT.md](SPORTS_ACTION_OBSERVATION_REPORT.md).
Selected8 and every byte of the1,402-row possession ledger remain unchanged.
Football/possession accuracy is untested; this is a runtime result, not a new model
selection or score improvement.

The frozen technical plan is [SPORTS_ACTION_RUNTIME_PLAN.json](SPORTS_ACTION_RUNTIME_PLAN.json).
A separate full-resolution benchmark plan is retained under the local evidence.

## Primary-source investigation

The [official SoccerNet team-action baseline](https://github.com/SoccerNet/sn-teamspotting)
provides a T-DEED adaptation, configurations and an author-linked public checkpoint.
It predicts12ball-action types plus background and a left/right team head. Those
outputs are not H/A/D possession labels. The implementation uses a100-frame clip,
stride2for ball-action data, and796x448frame extraction by default. The resulting
198-source-frame span fits inside a250-frame validation exclusion when centered.
Further feature summarization and clip selection still need their own footprint
proof before any possession experiment.

The [T-DEED author repository](https://github.com/arturxe2/T-DEED) is also archived
for source comparison. The official data downloads describe separately gated
videos; no external video dataset, NDA-gated material or private/corrected
competition labels is acquired here. Only public source and model weights are
used. Code licenses are retained in the source archives.

Pinned sources:

- SoccerNet/sn-teamspotting commit091fed2fc35c33f7489f3596958a2fe385e37d65,
  archiveSHA2566d12b9c5459b6f5db3819d6d7fb16557d1aff8dfaeb5ff6cd7e1fb25fb442671.
- arturxe2/T-DEED commitd953b4de8a2ad941c565e1683c5b07255cbe644e,
  archiveSHA2561783f607caaaa32ac8cce4b3900a9de162b0d171706ddd5f10fbc8e64555e692.
- Author-linked team checkpoint49,581,474bytes,
  SHA256ed7c558d575eba1918be8f67813571723962eeff807dba68a12a2d794dd1c72a.
  Its download matches the public folder's file size and has the expected Torch ZIP
  structure; every extracted source file verifies against the pinned archive.

## Local adapter and checks

An isolated Python3.12.14environment uses the pinned torch2.5.0,torchvision0.20.0,
timm1.0.11,numpy2.1.2,Pillow11.0.0and tqdm4.66.5inference dependencies. Production
requirements and the selected model's environment are not used for these installs.
All27resolved packages have an exact wheel lock and SHA256 hashes; installed
archive hashes match the cached wheels. A second environment installs offline
with--no-indexand--require-hashesfrom that wheelhouse.

The adapter uses the upstream model implementation and its13/18-class joint
prediction heads. It changes CPU/device placement of the joint head rather than
calling the upstream forced-CUDA placement. All600checkpoint entries load with
strict=Trueandweights_only=True. Initial ImageNet downloading is disabled because
the checkpoint supplies every parameter. The training entrypoint and WandB are
not imported or run. Python network connections are forbidden during inference;
there is no hosted processing, model fallback or logging of competition extracts.

Seeded synthetic input has shape1x100x3x32x56. Checks pass for all output shapes,
finite values, normalized first13-class probabilities, exact CPU repetition, and
responses to temporal reversal and spatial reflection. Output heads are
1x100x31jointlogits and1x100team/displacementvalues. Apple MPS executes the same
model; small-input CPU/MPS differences stay below1.3e-5and pass1e-3tolerances.
These checks establish computation and input sensitivity, not event accuracy.

The full-resolution synthetic input is1x100x3x448x796. MPSfloat32forwards take
4.269sand1.526s, and all three output tensors repeat bit-exactly. Driver allocation
after each pass is6,661,390,336bytes; peak processRSSis1,029,537,792bytes. These are
separate measurements; the GPU figure is an after-pass allocation, not a measured
peak. A0.6per-process MPS memory fraction is used without global settings changes.
Full-resolution CPU parity, real-video decoding overhead, full-match throughput
and actual football accuracy have not been measured.

The independently wheel-built runtime reruns the synthetic preflight in a fresh
workspace. All6saved CPU output files match byte-for-byte, including repeat files;
all technical checks pass again. Full-resolution neural inference is not repeated
in that second environment. The selected production environment's170tests pass.

## Decision and next step

The technical preflight passes. A separate observation/feature/possession plan
must be frozen next, before competition-image processing. It must define source
coverage, fixed sampling/half boundaries, reserved/held-image exclusions, feature
footprint, and treatment of the model's side labels. Left/right must not silently
become home/away truth. Only a paired public possession experiment can establish
whether the new action signal helps recover short controls or falseD.

No classifier fit, possession-score row, manual annotation, candidate promotion,
commit, upload, submission, publication or contact occurs. The ledger remains
1,402dependent rows:778inherited +624added in this chat. The24reserved annotation
windows and60held weak-ball images remain unused. The original substantial gain,
extra short match, D protections, paired seeds, reproduction and review requirements
remain required. Maximum/private full-match accuracy is unverified.

Local evidence under `data/control_sports_action_v1/` includes pinned source
archives, public folder responses, checkpoint download/hash, before snapshots,
installation report, exact dependency/wheel lock, two isolated runtimes, synthetic
outputs and benchmark, fresh-runtime replay code/proof, production test log,
`technical_integrity.json`and`running_stage.json`. All tracking-derived baseline
artifacts remain local under ignoreddata.
