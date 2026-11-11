# Round38: public soccer ball detector pilot

The soccer-specific WASB detector meets both numerical localization thresholds,
but the pilot fails its minimum visible-ball support. It does not authorize
native competition processing or candidate promotion. The original plan/verdict
remain unchanged. A separately frozen, disjoint Round40 confirmation is pending.

| Camera | Arm | Visible/256 | Correct | F1 | Visible localization recall |
|---|---|---:|---:|---:|---:|
| 5 | Adjacent frames |27|22|.846154|.814815|
| 5 | Repeated center |27|16|.666667|.592593|
| 6 | Adjacent frames |23|22|.956522|.956522|
| 6 | Repeated center |23|13|.619048|.565217|

The frozen conjunction requires>=32 visible examples **in each camera**, as well
as F1>=.80 and recall>=.75 at strictly four original-image pixels. Neither camera
has sufficient visibility support. The temporal arm produces two invisible-ball
false positives in camera5, one in camera6, and one mislocalized visible-ball
prediction in camera5. Wrong locations count as both FP and FN. This is stricter
than the publisher's legacy recall, which omits mislocalized detections from its
recall denominator. All256 uniform samples per camera remain in the accounting.

Source: [official WASB-SBDT](https://github.com/nttcom/WASB-SBDT), pinned commit
`923462cacdeb3353b84ddebdedb3f4b7a8553b0f`. The1,481,427-parameter HRNet uses the
one author-published soccer checkpoint, strict weights-only loading, unchanged
network/config and512×288 three-frame input. Local checkpoint SHA256:
`d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c`.
All12 source files match their pinned Git blobs. Code notices are retained;
the author's Drive binary wording is distinguished from the code MIT license
and the original video's research-use terms in the source manifest.

The CUDA-only wrapper is bypassed by a small config adapter; network math is
unchanged. Synthetic forwards/reload are exact. Maximum steady CPU forward is
.592627seconds, projected4000-forward workload2370.5seconds<3600. Synthetic
peak1.842GiB and real external peak1.929GiB meet2GiB limits. Existing Torch2.5
CPU runtime is reused; no dependency is installed.

Preprocessing follows the unrotated source affine and RGB/ImageNet convention.
The NumPy sampler implements5-bit bilinear weights and10-bit affine increments;
synthetic identity/integer-scale/geometry tests pass. An installed OpenCV binary
is unavailable, so binary-to-binary OpenCV parity is not claimed. The algorithm
was checked against [OpenCV's primary implementation](https://github.com/opencv/opencv/blob/4.10.0/modules/imgproc/src/imgwarp.cpp).
The dependency-free eight-connected weighted-component reader exactly matches
the SciPy reference on synthetic fixtures. Only center heatmap1 and largest-mass
component are used; there is no crop, confidence, model or recursive-track sweep.

Only the two public test-camera video members are acquired using guarded HTTP206
ZIP ranges, with exact content ranges, DEFLATE/length/CRC checks. Revised XML
labels come from the author's87,744-byte archive. Every frame selection is saved
before XML target parsing, using actual video2999-frame counts and zero-based
decoding. The author likewise extracts zero-based frames. This is synchronized
same-play cross-camera reproduction, not unseen-match transfer evidence.
See [acquisition research](WASB_ACQUISITION_RESEARCH.md).

An initial deeply left-associated FFmpeg selector exceeds its expression parser
depth. Original source, input artifacts and failure log are preserved. Execution
revision1 balances the same exact selectors, with a768-frame synthetic decoder
test. Model, target sample, gates and verdict are unchanged. All334 project tests
pass. No possession fits/scores, ledger additions, manual annotations, private
targets, competition media, installs, submissions or contacts occur in this pilot.
Selected8 and1531 ledger rows are preserved.

The [Round40 plan](WASB_CONFIRMATION_PLAN.json) explicitly starts an additional
experiment after the underpowered pilot; it does not reinterpret Round38 as a
pass or amend its frozen no-extra-sample verdict. New512-center selections use
no labels and exclude every image window sampled in Round38. The checkpoint,
accuracy/support thresholds and readout remain fixed; no pooling with the pilot
can rescue a failed confirmation. Same-game temporal dependence is acknowledged.
Any later native opportunity and substantial-score gates remain separate.
