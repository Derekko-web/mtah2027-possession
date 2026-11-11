# Round 53: serial cached-context reconstruction passes

The fixed single-thread execution protocol passes before any native possession
data. It uses the original TabICL2.2.0 checkpoint and eight-member cached model,
with unchanged fp32 computation, full3,000×166 context and512 queries. Torch and
BLAS thread counts are fixed to one before import. See the
[prospective plan](TABICL_SERIAL_RUNTIME_PLAN.json).

Two independent processes reconstruct preprocessing and all caches by the same
supported `fit(context,target)` call. Both cache fingerprints and probability
arrays are byte-identical, including exact within-process repeats. The outputs
also match the saved original two-thread Round49 probabilities exactly, stronger
than the prospectively allowed backend-equivalence tolerance. Checkpoint tensors,
class order, complete caches, normalized probabilities and cache immutability
verify. Original package files and all earlier failed protocols remain unchanged.

| Independent process | Fit | First prediction | Repeat |
|---|---:|---:|---:|
| First | 212.738s | 26.118s | 26.174s |
| Reconstruction | 208.143s | 26.454s | 25.848s |

Each operation stays below300seconds. Supervised sampled peaks are4.425GiB
and5.182GiB, below6GiB; both workers exit0 with no abort. Saved probability and
cache-fingerprint files verify against the worker and supervision manifests.
The fitted-object reload failure from Round49 and cross-process numerical
failure from Round51 remain failures; neither is relabelled as passed.

Evidence: ignored `data/control_tabicl_serial_v1/`. This technical pass permits
a separately frozen native screen, not a possession-score or private-accuracy
claim. No competition features, labels, images, scores or candidates were used
in this protocol.
