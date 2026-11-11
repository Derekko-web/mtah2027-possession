"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const video = $("video");
  const pausedFrame = $("paused-frame");
  const pausedContext = pausedFrame.getContext("2d", {alpha: false});
  let catalog, documentState, selectedWindow, selectedLabel = null, dirty = false, saving = false;
  let frame = 0, undoStack = [];
  let cachedFrame = null, videoSuspended = false;
  const names = {H: "Barcelona", A: "Athletic", D: "Dead ball", U: "Uncertain"};
  const clone = value => JSON.parse(JSON.stringify(value));
  function message(text, error = false) {
    $("notice").textContent = text;
    $("notice").classList.toggle("error", error);
    $("notice").hidden = !text;
  }
  async function api(path, body) {
    const options = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json", "X-Annotation-Token": catalog.save_token}, body: JSON.stringify(body)};
    const response = await fetch(path, options);
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Request failed");
    return result;
  }
  function markDirty() {
    dirty = true;
    documentState.status = "draft";
    documentState.reviewer = "";
    $("reviewer").value = "";
    renderStatus();
  }
  function renderStatus() {
    $("save-state").textContent = saving ? "Saving…" : dirty ? "Unsaved changes" : "Saved";
    $("save-state").classList.toggle("dirty", dirty);
    $("clip-status").textContent = documentState.status === "reviewed" ? "Reviewed" : "Draft";
    $("clip-status").classList.toggle("reviewed", documentState.status === "reviewed");
    $("undo").disabled = !undoStack.length || saving;
    $("apply").disabled = !selectedLabel || saving;
  }
  function renderCatalog() {
    const reviewed = catalog.windows.filter(w => w.status === "reviewed").length;
    $("review-count").textContent = reviewed;
    $("overall-fill").style.width = `${reviewed / catalog.windows.length * 100}%`;
    $("export").disabled = reviewed !== 12 || dirty || saving;
    $("windows").replaceChildren(...catalog.windows.map((window, index) => {
      const button = document.createElement("button");
      button.className = "window-button";
      button.classList.toggle("active", window.window_id === selectedWindow?.window_id);
      if (window.window_id === selectedWindow?.window_id) button.setAttribute("aria-current", "true");
      button.disabled = saving;
      button.setAttribute("aria-label", `Clip ${index + 1}, ${window.stratum}, ${window.status}`);
      const number = document.createElement("span"); number.className = "window-number"; number.textContent = String(index + 1).padStart(2, "0");
      const copy = document.createElement("span"); copy.className = "window-copy";
      const title = document.createElement("strong"); title.textContent = window.stratum === "restart" ? "Restart window" : "Uniform window";
      const subtitle = document.createElement("small"); subtitle.textContent = `${window.covered} / 750 frames`;
      copy.append(title, subtitle);
      const status = document.createElement("span"); status.className = "window-state"; status.textContent = window.status === "reviewed" ? "✓" : window.covered ? "◐" : "·";
      button.append(number, copy, status);
      button.addEventListener("click", () => loadWindow(window));
      return button;
    }));
  }
  function renderSegments() {
    const segments = documentState.segments;
    $("range-count").textContent = segments.length;
    const covered = segments.reduce((count, segment) => count + segment.end - segment.start + 1, 0);
    $("coverage").textContent = `${covered} / 750 frames`;
    $("segments").replaceChildren();
    $("timeline").replaceChildren();
    if (!segments.length) {
      const empty = document.createElement("p"); empty.className = "empty"; empty.textContent = "Mark a range, then assign possession."; $("segments").append(empty);
    }
    for (const [index, segment] of segments.entries()) {
      const row = document.createElement("div"); row.className = "segment-row";
      const select = document.createElement("button"); select.className = "segment-select";
      select.setAttribute("aria-label", `Edit ${names[segment.label]} range ${segment.start} to ${segment.end}`);
      const label = document.createElement("span"); label.className = "segment-label"; label.textContent = segment.label;
      const time = document.createElement("span"); time.className = "segment-time"; time.textContent = `${String(segment.start).padStart(3, "0")} — ${String(segment.end).padStart(3, "0")}`;
      const note = document.createElement("span"); note.className = "segment-note"; note.textContent = segment.note || names[segment.label];
      select.append(label, time, note);
      select.addEventListener("click", () => edit(segment));
      const remove = document.createElement("button"); remove.className = "delete-range"; remove.textContent = "×"; remove.setAttribute("aria-label", `Delete range ${segment.start} to ${segment.end}`);
      remove.addEventListener("click", () => { if (saving) return; remember(); documentState.segments.splice(index, 1); markDirty(); renderSegments(); });
      row.append(select, remove); $("segments").append(row);
      const block = document.createElement("button"); block.className = `timeline-block ${segment.label}`;
      block.style.left = `${segment.start / 750 * 100}%`; block.style.width = `${(segment.end - segment.start + 1) / 750 * 100}%`;
      block.textContent = segment.end - segment.start > 15 ? segment.label : "";
      block.setAttribute("aria-label", `Timeline ${names[segment.label]}, frames ${segment.start} to ${segment.end}`);
      block.addEventListener("click", () => edit(segment)); $("timeline").append(block);
    }
    renderStatus();
    $("export").disabled = catalog.windows.some(w => w.status !== "reviewed") || dirty || saving;
  }
  function choose(label) {
    selectedLabel = label;
    for (const button of $("label-options").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.label === label));
    renderStatus();
  }
  function updateFrame(value) {
    frame = Math.max(0, Math.min(749, value));
    $("current-frame").textContent = String(frame).padStart(3, "0");
    $("seek").value = frame;
    if (selectedWindow) $("source-frame").textContent = `Source frame ${selectedWindow.start + frame}`;
  }
  function cacheVideoFrame(index = frame) {
    if (!pausedContext || video.seeking || videoSuspended || video.readyState < 2 || !video.videoWidth) return;
    if (pausedFrame.width !== video.videoWidth || pausedFrame.height !== video.videoHeight) {
      pausedFrame.width = video.videoWidth; pausedFrame.height = video.videoHeight;
    }
    // Keep pixels in this browser only. A normal DOM canvas survives loss of
    // the host's native video surface; no image is encoded, sent or saved.
    try { pausedContext.drawImage(video, 0, 0, pausedFrame.width, pausedFrame.height); }
    catch { return; } // Keep the previous still if the decoder is unavailable.
    cachedFrame = index;
    pausedFrame.hidden = !video.paused;
  }
  function suspendVideo() {
    if (!selectedWindow?.video_ready || videoSuspended) return;
    videoSuspended = true;
    video.pause();
    // Do not recapture after blur: the host may have already discarded the
    // video surface. Retain the last successfully presented frame instead.
    if (cachedFrame !== null) { updateFrame(cachedFrame); pausedFrame.hidden = false; }
  }
  function restoreVideo() {
    if (!videoSuspended || document.hidden || !selectedWindow?.video_ready) return;
    videoSuspended = false;
    const resumeFrame = cachedFrame ?? frame;
    if (video.readyState >= 1) {
      updateFrame(resumeFrame);
      video.currentTime = (resumeFrame + 0.1) / catalog.fps;
    }
  }
  function seek(value) {
    if (!selectedWindow?.video_ready) return;
    restoreVideo();
    video.pause();
    const target = Math.max(0, Math.min(749, Number(value)));
    // Seek inside the frame's presentation interval, avoiding boundary rounding.
    video.currentTime = (target + 0.1) / catalog.fps;
    updateFrame(target);
  }
  function edit(segment) {
    if (saving) return;
    $("range-start").value = segment.start; $("range-end").value = segment.end; $("note").value = segment.note;
    choose(segment.label); seek(segment.start);
  }
  function remember() {
    undoStack.push(clone(documentState.segments));
    if (undoStack.length > 50) undoStack.shift();
  }
  function applyRange() {
    if (saving || !selectedLabel) return;
    const start = Number($("range-start").value), end = Number($("range-end").value), note = $("note").value.trim();
    if (!$("range-start").value || !$("range-end").value || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > 749 || start > end) return message("Choose inclusive range endpoints from 0 to 749. The start must be at or before the end.", true);
    if (selectedLabel === "U" && !note) return message("Add a note explaining why this range is uncertain.", true);
    remember();
    const ranges = [];
    for (const old of documentState.segments) {
      if (old.end < start || old.start > end) ranges.push(old);
      else {
        if (old.start < start) ranges.push({...old, end: start - 1});
        if (old.end > end) ranges.push({...old, start: end + 1});
      }
    }
    ranges.push({start, end, label: selectedLabel, note});
    ranges.sort((a, b) => a.start - b.start);
    const merged = [];
    for (const range of ranges) {
      const prior = merged[merged.length - 1];
      if (prior && prior.end + 1 === range.start && prior.label === range.label && prior.note === range.note) prior.end = range.end;
      else merged.push({...range});
    }
    documentState.segments = merged; markDirty(); renderSegments(); message("");
  }
  let loading = false;
  async function loadWindow(clip) {
    if (saving || loading) return;
    if (dirty && !window.confirm("Discard unsaved changes to this clip?")) return;
    loading = true;
    video.pause();
    try {
      const incoming = await api(`/api/annotation/${clip.window_id}`);
      selectedWindow = clip; documentState = incoming; dirty = false; undoStack = []; selectedLabel = null;
      cachedFrame = null; videoSuspended = false; pausedFrame.hidden = true;
      $("clip-title").textContent = `Clip ${String(catalog.windows.indexOf(clip) + 1).padStart(2, "0")}`;
      $("clip-stratum").textContent = `${clip.stratum.toUpperCase()} WINDOW · 30 SECONDS`;
      $("annotator").value = incoming.annotator || localStorage.getItem("possession.annotator") || "";
      $("reviewer").value = incoming.reviewer || "";
      $("range-start").value = "0"; $("range-end").value = "0"; $("note").value = "";
      for (const button of $("label-options").querySelectorAll("button")) button.setAttribute("aria-pressed", "false");
      if (clip.video_ready) { video.src = `/media/${clip.window_id}.mp4`; $("video-message").textContent = "Loading local video…"; }
      else { video.removeAttribute("src"); video.load(); $("video-message").textContent = "Training clips are not prepared. Run the preparation command in the annotation guide, then restart the workspace."; }
      $("video-message").hidden = false;
      updateFrame(0); renderCatalog(); renderSegments(); message("");
    } catch (error) { message(error.message, true); }
    finally { loading = false; }
  }
  async function save(status) {
    if (saving) return;
    saving = true; $("annotation-fields").disabled = true; renderStatus(); renderCatalog();
    const body = {...documentState, annotator: $("annotator").value, reviewer: $("reviewer").value, status};
    try {
      documentState = await api(`/api/annotation/${selectedWindow.window_id}`, body);
      dirty = false; localStorage.setItem("possession.annotator", documentState.annotator);
      catalog = await api("/api/windows");
      message(status === "reviewed" ? "Clip reviewed and saved. Its frames will be eligible for the training export." : "Draft saved on this computer.");
    } catch (error) { message(error.message, true); }
    finally { saving = false; $("annotation-fields").disabled = false; renderStatus(); renderCatalog(); }
  }
  $("apply").addEventListener("click", applyRange);
  $("undo").addEventListener("click", () => { if (!undoStack.length || saving) return; documentState.segments = undoStack.pop(); markDirty(); renderSegments(); message(""); });
  $("save").addEventListener("click", () => save("draft"));
  $("review").addEventListener("click", () => save("reviewed"));
  $("annotator").addEventListener("input", () => { markDirty(); renderCatalog(); });
  $("reviewer").addEventListener("input", () => { dirty = true; renderStatus(); renderCatalog(); });
  $("mark-in").addEventListener("click", () => { $("range-start").value = frame; });
  $("mark-out").addEventListener("click", () => { $("range-end").value = frame; });
  for (const button of $("label-options").querySelectorAll("button")) button.addEventListener("click", () => choose(button.dataset.label));
  $("seek").addEventListener("input", event => seek(event.target.value));
  $("back").addEventListener("click", () => seek(frame - 1));
  $("forward").addEventListener("click", () => seek(frame + 1));
  $("play").addEventListener("click", async () => { if (!selectedWindow?.video_ready) return; restoreVideo(); if (video.paused) { try { await video.play(); } catch (error) { message(error.message, true); } } else video.pause(); });
  $("rate").addEventListener("change", () => { video.playbackRate = Number($("rate").value); });
  video.addEventListener("loadeddata", () => { $("video-message").hidden = true; video.playbackRate = Number($("rate").value); cacheVideoFrame(); });
  video.addEventListener("error", () => { if (video.hasAttribute("src")) { $("video-message").textContent = "This clip could not be played. Check that clip preparation completed and use a browser with H.264 video support."; $("video-message").hidden = false; } });
  video.addEventListener("timeupdate", () => { if (!video.seeking && !videoSuspended) { updateFrame(Math.floor(video.currentTime * (catalog?.fps || 25) + 0.0001)); if (!video.requestVideoFrameCallback) cacheVideoFrame(); } });
  video.addEventListener("seeked", () => { if (!videoSuspended) { updateFrame(Math.floor(video.currentTime * catalog.fps + 0.0001)); cacheVideoFrame(); } });
  video.addEventListener("play", () => {
    if (document.hidden || videoSuspended) { video.pause(); return; }
    pausedFrame.hidden = true;
    $("play").textContent = "Ⅱ"; $("play").setAttribute("aria-label", "Pause");
  });
  video.addEventListener("pause", () => { $("play").textContent = "▶"; $("play").setAttribute("aria-label", "Play"); if (!videoSuspended) cacheVideoFrame(); });
  if (video.requestVideoFrameCallback) {
    const presented = (_, metadata) => { if (!video.seeking && !videoSuspended) { updateFrame(Math.round(metadata.mediaTime * (catalog?.fps || 25))); cacheVideoFrame(frame); } video.requestVideoFrameCallback(presented); };
    video.requestVideoFrameCallback(presented);
  }
  window.addEventListener("blur", suspendVideo);
  window.addEventListener("focus", restoreVideo);
  window.addEventListener("pagehide", suspendVideo);
  window.addEventListener("pageshow", restoreVideo);
  document.addEventListener("visibilitychange", () => { if (document.hidden) suspendVideo(); else restoreVideo(); });
  $("export").addEventListener("click", async () => {
    try { const result = await api("/api/export", {}); message(`Training export saved: ${result.training_rows} labelled frames; ${result.uncertain_frames_excluded} uncertain frames excluded.\n${result.path}`); }
    catch (error) { message(error.message, true); }
  });
  window.addEventListener("beforeunload", event => { if (dirty) { event.preventDefault(); event.returnValue = ""; } });
  window.addEventListener("keydown", event => {
    if (!documentState || saving || event.target.closest("input,textarea,select,summary")) return;
    if (event.target.closest("button") && (event.code === "Space" || event.key === "Enter")) return;
    const key = event.key.toLowerCase();
    if (event.code === "Space") { event.preventDefault(); $("play").click(); }
    else if (key === "arrowleft" || key === "arrowright") { event.preventDefault(); seek(frame + (key === "arrowleft" ? -1 : 1) * (event.shiftKey ? 25 : 1)); }
    else if ((event.metaKey || event.ctrlKey) && key === "z") { event.preventDefault(); $("undo").click(); }
    else if (!event.metaKey && !event.ctrlKey && "hadu".includes(key) && key.length === 1) choose(key.toUpperCase());
    else if (key === "i") $("mark-in").click();
    else if (key === "o") $("mark-out").click();
    else if (key === "enter") { event.preventDefault(); applyRange(); }
  });
  (async () => { try { catalog = await api("/api/windows"); renderCatalog(); await loadWindow(catalog.windows[0]); } catch (error) { message(error.message, true); } })();
})();
