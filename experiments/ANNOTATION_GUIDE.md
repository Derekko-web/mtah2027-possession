# Local possession annotation

The workspace is at **http://127.0.0.1:8765** while the local server is running.
It exposes only the twelve frozen training windows: six minutes, divided equally
between uniformly sampled and automatically tagged restart windows. The twelve
second-half audit windows remain sealed. Neither model predictions nor official
possession labels are shown.

The immediate task needs teammates to watch and label the training clips. Codex
has prepared the tool and media; it has not authored labels or fitted a new model.
Human video interpretations are additional supervision, not the challenge's
corrected-tracking ground truth or a new official validation score.

## Start locally

From the repository root, with the existing environment and local tactical video:

```sh
.venv/bin/python experiments/prepare_annotation_clips.py
.venv/bin/python experiments/annotation_server.py
```

Clip preparation has already completed in this workspace. **Run only the second
command to restart the server.** Preparation refuses to overwrite existing
outputs. A different machine needs the licensed original assets and `ffmpeg` /
`ffprobe`; Python's standard library is otherwise sufficient. `MTAH_DATA_DIR` is
supported through `games.py`. The server binds to loopback, with no remote assets,
telemetry or cloud requests. Open the address in a local browser.

## Annotate and review

1. Choose a training clip and enter the annotator's name. Watch the clip before
   making decisions; the source is the tactical camera. **H = Barcelona** and
   **A = Athletic**. Confirm visual team identification rather than guessing it
   from pitch direction.
2. Pause at a transition. Use the arrow buttons / arrow keys to move one frame,
   or Shift + arrow to move 25 frames. The counter is a zero-based **clip** frame
   from 0 through 749; the smaller counter gives its original tactical frame.
   Choose a range start with `I` and its end with `O`, or type the endpoints.
3. Choose **H**, **A**, **D** (dead ball), or **U** (uncertain), then **Apply range**.
   Endpoints are inclusive. A new range replaces only the labels inside it;
   portions outside it survive. Click an existing range to edit it. Undo restores
   a range change. Unapplied editor inputs are not a saved range.
4. Follow the [possession definition](../CHALLENGE.md): a loose or contested live
   ball stays with the last team in control; a deflection alone does not change
   possession. Use D for stopped play, not for merely slow movement or a hidden
   ball. At an unclear transition, mark the ambiguous **frames** U with a note,
   rather than forcing an exact boundary. Every U range needs an explanation.
5. Save a draft regularly. A second teammate watches the clip, checks boundaries
   and coverage, enters their different name, and selects **Mark reviewed**.
   All 750 frames need coverage, including explicit U spans. A partially labelled
   clip can be saved as a draft. Changing labels clears its review. Unsaved
   changes trigger a warning before navigation or closing the page.
6. After **all twelve** training clips are reviewed, **Export training labels**
   writes a versioned export under the ignored local work folder. U frames are
   excluded from the CSV and preserved with notes in provenance. Both sampling
   strata remain identifiable; do not treat the restart-enriched sample as an
   unbiased estimate of full-match performance.

`Space` plays / pauses. Playback speeds include 0.25× and 0.5×. `H/A/D/U` chooses
a label; Enter applies the range when focus is outside form inputs. Cmd/Ctrl + Z
undoes a range edit. Normal text-input keys keep their usual behavior.

Switching away pauses playback and retains the displayed frame locally. Returning
keeps that frame and your current edits; press Play to resume. This also prevents
a paused video from turning black when an embedded browser drops its video
surface. After updating the tool, save any current draft before refreshing the
page to load the new player code.

## Files and isolation

All media and annotations stay inside ignored `data/local_annotation_v1/`:

- `clips/manifest.json`: source-video hash, frozen-window hash and each clip's
  checksum / frame count. Absolute decoder-frame trimming produces exactly 750
  frames at 25 fps, with original resolution and no temporal resampling. The
  H.264 clips are compressed; they are not pixel-identical source copies.
- `annotations/train/window_*.json`: latest saved human labels, annotator,
  reviewer, source-frame origin, media identity, freeze hash and revision.
- `annotations/train/history/`: every saved revision, retained before the latest
  document is atomically replaced. Conflicting tabs cannot overwrite a newer
  revision. A disk error leaves the browser draft available for retry.
- `exports/<content-hash>/training_labels.csv`: `window_id,frame,label` training
  rows in original tactical coordinates. This is **not a submission CSV**.
  `provenance.json` retains complete annotations, U exclusions and reviewer names.

Keep these files local. The code and guide can be shared; the licensed clips and
derived frame labels must not enter a public repository or a hosted LLM. The
server serves only whitelisted training media, UI assets and final training CSVs.
Audit-window IDs cannot be read, saved, served or exported through its API. There
is no audit-unlock switch. Audit metadata itself remains in the original local
freeze for the future auditor.

## Before the next model

Once the human training export exists, check coverage and uncertainty by stratum,
then freeze one focused ownership / transition hypothesis and its comparison with
selected8. Train with the extra human labels clearly distinguished from official
public labels, preserving temporal exclusion buffers. Do not patch final
submission frames with these annotations.

The audit remains unopened until the complete candidate, selected8 comparator,
metrics, uncertainty policy and evaluation procedure are frozen. A separate
auditor then labels and runs the one comparison, reporting uniform / restart
strata separately. A video-based audit is not an independent copy of the private
corrected-tracking target and cannot establish the official full-match score.

## Verification

```sh
.venv/bin/python -m unittest discover -s experiments -p test_annotation_tool.py -v
```

These tests use only synthetic windows, labels and video. They cover exact
absolute-frame trimming, inclusive endpoints, byte-range playback, uncertainty
exclusion, complete independent review, revision conflicts, frozen provenance,
and refusal of audit or foreign-origin access. Browser QA also uses a separate
synthetic-only workspace. No competition image or new human label is inspected
in hosted model context.
