# Training annotation workspace — ready for human labels

Built 2026-10-23 (America/Toronto) after the rejected DFL signal audit. The user
authorized local annotation tooling for the existing reserved sample. The tool is
running at **http://127.0.0.1:8765**. See [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)
for the labelling workflow and restart command.

Only the twelve frozen **training** windows have been prepared: 750 frames each
at 25 fps, six minutes total, evenly divided between uniform and restart sampling.
The twelve audit windows have no media, read/write route or export in this tool.
The original 24-window freeze remains byte-exact, SHA256
`5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c`.

The local UI supports frame stepping, slow playback, inclusive range labels,
overlap replacement, undo, uncertainty notes, draft saves and a different person's
review. Complete coverage and second-person review of all twelve training clips
are required before export. U frames are preserved in provenance and excluded
from H/A/D training rows. Saved revisions are retained, conflicts cannot overwrite
newer work, and a failed atomic update can be retried without replacing history.

Clip preparation uses one sequential decode with absolute frame-index trimming;
it neither seeks approximately nor resamples time. All clips retain native
resolution, have exactly 750 decoded frames, start at timestamp zero and run at
25 fps. Source video and prepared clips are hashed. Synthetic frame-value tests
verify the first and last source indices and inclusive trim boundaries. H.264
re-encoding is lossy, so pixel identity to the source is not claimed.

Verification passed:

- **289 Python tests**, including 14 new annotation regressions. The new checks
  cover frame mapping, frozen provenance, inclusive range validation, U exclusion,
  complete review, revision conflicts / failed-write recovery, byte-range playback,
  and audit / foreign-origin / path access denial.
- Synthetic-only browser checks cover save/reload, overlap replacement, undo,
  uncertainty errors, distinct review, frame and one-second stepping, final-frame
  seeking, playback, all twelve reviews and export (8,998 labelled plus two U
  frames). Unsaved navigation and stale-tab preservation were also exercised.
- A mobile overflow found during QA was fixed; desktop and mobile screenshots
  were inspected using synthetic video. The final refreshed page has no console
  errors or warnings. No competition image or human annotation entered hosted
  model context.
- All 17 selected-model artifacts, every byte of the 1,477-row ledger, the
  selected CSV, selection/history documents and original freeze remain unchanged.
  All generated media, QA fixtures and later labels stay under ignored `data/`.

There are **no real human labels or training exports yet**. No new model was
fitted, no score was added and no candidate was generated. The next dependency
is teammates completing the training annotation and review. Human video labels
are additional supervision, not the official corrected-tracking target. After
that export, freeze one focused training hypothesis; keep the audit unopened
until the full candidate and one comparison procedure are frozen. No promise of
an official score gain follows from building this tool.

Local evidence is under `data/local_annotation_v1/`: the pre-work hashes and
tool plan, verified clip manifest, synthetic-only browser workspace and screenshots,
test logs, and `verification.json`. Only the real loopback annotation server is
left running intentionally; the QA server and browser are closed. No commit,
upload, submission, external contact or installation was performed.

## 2026-10-24: embedded-browser focus repair

The user reports a black clip after leaving Codex's built-in browser panel. The
player now retains the presented frame in a browser-local canvas, pauses on
focus/visibility loss and refreshes the decoder at the same frame on return.
Manual Play/step controls recover even if the host misses its focus event. It
does not reload the page, clear edits, automatically resume playback, write
images or transmit pixels.

The original player fails a synthetic renderer-fault test: hiding the native
video surface after blur replaces the paused picture. With the updated player,
the picture pixels inside its rounded border are exact across the same fault
and focus return. Frame position and an unsaved draft survive; manual controls,
playback pausing and five repeated focus cycles pass. An initial overly strict
whole-border PNG comparison detects corner antialiasing rather than lost video;
that evidence remains, and the final check compares the actual picture interior.
The final browser callback is `check_annotation_focus.js` and needs the separate
synthetic QA server at 127.0.0.1:8766. All 14 annotation Python tests and JS syntax
checks pass. A synthetic clip also remains visible after switching tabs in the
Codex panel. No real annotation is read, saved or reloaded during UI testing.

The exact native host failure was not reproduced by switching browser tabs, and
native app-level focus control is unavailable to the automation. The regression
therefore injects video-surface loss explicitly; it verifies this protective
behavior, not a proven root cause in Codex. The user should save their draft and
refresh the real annotation page to load the change. Real annotations, reserved
audit media, candidate artifacts and the score ledger are untouched. Before
sources, red/green results and current-source verification remain under ignored
`data/local_annotation_v1/focus_fix_20261024/`.
