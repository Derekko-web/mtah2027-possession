"""Local, train-only possession annotation workspace (Python standard library).

Run: .venv/bin/python experiments/annotation_server.py
Audit windows have no route, media, annotation file or export in this tool.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import secrets
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FREEZE = ROOT / "data/continuation_20261021/annotation_freeze/windows.json"
DEFAULT_WORK = ROOT / "data/local_annotation_v1"
STATIC = Path(__file__).with_name("annotation_ui")
FPS = 25


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_freeze(path):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != path.with_suffix(".sha256").read_text().split()[0]:
        raise ValueError("Frozen window checksum differs")
    manifest = json.loads(raw)
    windows = manifest["windows"]
    ids = [w["window_id"] for w in windows]
    if len(ids) != len(set(ids)) or len(windows) != 24:
        raise ValueError("Expected 24 uniquely identified frozen windows")
    for window in windows:
        if not re.fullmatch(r"window_\d{2}", window["window_id"]):
            raise ValueError("Invalid window ID")
        if window["split"] not in ("train", "audit") or window["stratum"] not in ("uniform", "restart"):
            raise ValueError("Unknown frozen split or stratum")
        if type(window["start"]) is not int or type(window["end"]) is not int or window["start"] < 0 or window["end"] - window["start"] != 749:
            raise ValueError("Expected inclusive, 750-frame windows")
    for split in ("train", "audit"):
        for stratum in ("uniform", "restart"):
            if sum(w["split"] == split and w["stratum"] == stratum for w in windows) != 6:
                raise ValueError("Frozen sampling strata differ")
    ordered = sorted(windows, key=lambda w: w["start"])
    if any(a["end"] + 250 >= b["start"] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Frozen windows overlap their buffer")
    return manifest, digest


class Conflict(ValueError):
    pass


class AnnotationStore:
    def __init__(self, freeze=DEFAULT_FREEZE, work=DEFAULT_WORK):
        self.manifest, self.freeze_hash = load_freeze(freeze)
        self.windows = {w["window_id"]: w for w in self.manifest["windows"] if w["split"] == "train"}
        self.work = Path(work)
        self.labels = self.work / "annotations" / "train"
        self.labels.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        media_path = self.work / "clips" / "manifest.json"
        self.media = json.loads(media_path.read_text()) if media_path.exists() else None
        if self.media:
            if self.media["freeze_sha256"] != self.freeze_hash or set(self.media["clips"]) != set(self.windows):
                raise ValueError("Prepared media does not match the frozen training windows")
            for key in self.windows:
                if sha256(self.work / "clips" / (key + ".mp4")) != self.media["clips"][key]["sha256"]:
                    raise ValueError("Prepared clip checksum differs")

    def window(self, key):
        if key not in self.windows:
            raise KeyError("Unknown training window")
        return self.windows[key]

    def get(self, key):
        self.window(key)
        path = self.labels / (key + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if result["freeze_sha256"] != self.freeze_hash or result["window_id"] != key or result["split"] != "train":
                raise ValueError("Saved annotation provenance differs")
            normalized = self.validate(key, result)
            for field in ("source_start", "source_end", "fps", "label_source"):
                if result.get(field) != normalized[field]:
                    raise ValueError("Saved annotation frame provenance differs")
            return result
        return dict(schema_version=1, window_id=key, split="train", freeze_sha256=self.freeze_hash,
                    label_source="human_video", revision=0, annotator="", reviewer="", status="draft", segments=[])

    def validate(self, key, value):
        window = self.window(key)
        if not isinstance(value, dict):
            raise ValueError("Expected an annotation object")
        if type(value.get("revision")) is not int or value["revision"] < 0:
            raise ValueError("A valid save revision is required")
        for field in ("annotator", "reviewer"):
            if not isinstance(value.get(field, ""), str) or len(value.get(field, "")) > 120:
                raise ValueError("Names must contain at most 120 characters")
        annotator, reviewer = value.get("annotator", "").strip(), value.get("reviewer", "").strip()
        if not annotator:
            raise ValueError("Enter the annotator's name before saving")
        status = value.get("status")
        if status not in ("draft", "reviewed"):
            raise ValueError("Unknown annotation status")
        segments = value.get("segments")
        if not isinstance(segments, list) or len(segments) > 750:
            raise ValueError("Invalid segment list")
        clean = []
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("Invalid segment")
            start, end = segment.get("start"), segment.get("end")
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < 750:
                raise ValueError("Range endpoints must be inclusive integers from 0 to 749")
            if segment.get("label") not in ("H", "A", "D", "U"):
                raise ValueError("Use H, A, D or U")
            note = segment.get("note", "")
            if not isinstance(note, str) or len(note) > 2000:
                raise ValueError("Notes must contain at most 2000 characters")
            if segment["label"] == "U" and not note.strip():
                raise ValueError("Explain why an uncertain range cannot be labelled")
            clean.append(dict(start=start, end=end, label=segment["label"], note=note.strip()))
        clean.sort(key=lambda s: s["start"])
        if any(a["end"] >= b["start"] for a, b in zip(clean, clean[1:])):
            raise ValueError("Ranges overlap")
        if status == "reviewed":
            if not reviewer or reviewer.casefold() == annotator.casefold():
                raise ValueError("A different teammate must review this window")
            if sum(s["end"] - s["start"] + 1 for s in clean) != 750:
                raise ValueError("Cover every frame, using U for uncertainty, before marking reviewed")
        return dict(schema_version=1, window_id=key, split="train", freeze_sha256=self.freeze_hash,
                    source_start=window["start"], source_end=window["end"], fps=FPS,
                    clip_sha256=self.media["clips"][key]["sha256"] if self.media else None,
                    label_source="human_video", annotator=annotator, reviewer=reviewer if status == "reviewed" else "",
                    status=status, segments=clean)

    def save(self, key, value):
        with self.lock:
            clean = self.validate(key, value)
            current = self.get(key)
            if value["revision"] != current["revision"]:
                raise Conflict("Another tab saved this window. Reload it before saving again.")
            history = self.labels / "history" / key
            history.mkdir(parents=True, exist_ok=True)
            # A failed head-file update may leave a preserved revision behind.
            # Retrying must retain it and create a new revision, never get stuck
            # trying to overwrite that evidence.
            previous = max([current["revision"]] + [int(p.stem) for p in history.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json")])
            clean["revision"] = previous + 1
            clean["updated_at"] = datetime.now(timezone.utc).isoformat()
            revision = history / ("{:06d}.json".format(clean["revision"]))
            encoded = json.dumps(clean, indent=2, ensure_ascii=False) + "\n"
            # Preserve every revision before atomically updating the working document.
            with revision.open("x") as handle:
                handle.write(encoded)
            temporary = self.labels / (key + ".tmp")
            temporary.write_text(encoded)
            temporary.replace(self.labels / (key + ".json"))
            return clean

    def catalog(self):
        windows = []
        for key, window in self.windows.items():
            document = self.get(key)
            covered = sum(s["end"] - s["start"] + 1 for s in document["segments"])
            windows.append(dict(**window, frame_count=750, covered=covered, status=document["status"],
                                video_ready=self.media is not None))
        return dict(fps=FPS, freeze_sha256=self.freeze_hash, windows=windows, audit_windows_sealed=12,
                    purpose="Training annotations only; human video interpretation is not official corrected tracking truth")

    def export(self):
        with self.lock:
            if not self.media:
                raise ValueError("Prepare and verify the training clips before export")
            documents = [self.get(key) for key in self.windows]
            if any(d["status"] != "reviewed" for d in documents):
                raise ValueError("All 12 training windows need complete coverage and a second-person review")
            # Recheck saved files; never trust status alone.
            for document in documents:
                self.validate(document["window_id"], document)
                if document.get("clip_sha256") != self.media["clips"][document["window_id"]]["sha256"]:
                    raise ValueError("Annotation media identity differs")
            rows, uncertain = [], 0
            for document in documents:
                for segment in document["segments"]:
                    if segment["label"] == "U":
                        uncertain += segment["end"] - segment["start"] + 1
                        continue
                    for offset in range(segment["start"], segment["end"] + 1):
                        rows.append((document["window_id"], document["source_start"] + offset, segment["label"]))
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer, lineterminator="\n")
            writer.writerow(("window_id", "frame", "label"))
            writer.writerows(rows)
            evidence = dict(schema_version=1, purpose="model_training_only_not_a_submission",
                            freeze_sha256=self.freeze_hash, split="train", fps=FPS,
                            audit_windows_sealed=12, uncertain_frames_excluded=uncertain,
                            training_rows=len(rows), annotations=documents)
            digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
            folder = self.work / "exports" / digest
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "training_labels.csv").write_text(buffer.getvalue())
            (folder / "provenance.json").write_text(json.dumps(evidence, indent=2) + "\n")
            return dict(export_id=digest, training_rows=len(rows), uncertain_frames_excluded=uncertain,
                        path=str(folder), download="/exports/" + digest + "/training_labels.csv")


def make_server(store, port=8765):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # No annotation content or media paths in access logs.

        def headers_common(self):
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")

        def valid_host(self):
            return self.headers.get("Host") in ("127.0.0.1:" + str(self.server.server_port), "localhost:" + str(self.server.server_port))

        def json(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.headers_common()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def file(self, path, mime, range_ok=False):
            if not path.is_file():
                return self.json(404, dict(error="File is not available"))
            size, start, end = path.stat().st_size, 0, path.stat().st_size - 1
            header = self.headers.get("Range") if range_ok else None
            if header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
                if not match or not any(match.groups()):
                    return self.json(416, dict(error="Invalid byte range"))
                left, right = match.groups()
                if not left:
                    start = max(0, size - int(right))
                else:
                    start, end = int(left), min(int(right), end) if right else end
                if not 0 <= start <= end < size:
                    return self.json(416, dict(error="Byte range is outside this file"))
            self.send_response(206 if header else 200)
            self.headers_common()
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(end - start + 1))
            if range_ok:
                self.send_header("Accept-Ranges", "bytes")
            if header:
                self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
            self.end_headers()
            try:
                with path.open("rb") as handle:
                    handle.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        data = handle.read(min(remaining, 256 * 1024))
                        if not data:
                            break
                        self.wfile.write(data)
                        remaining -= len(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.valid_host():
                return self.json(403, dict(error="Use the local workspace address"))
            path = urlsplit(self.path).path
            try:
                if path == "/api/windows":
                    return self.json(200, dict(**store.catalog(), save_token=token))
                if path.startswith("/api/annotation/"):
                    return self.json(200, store.get(path.rsplit("/", 1)[1]))
                if path.startswith("/media/"):
                    key = path.removeprefix("/media/").removesuffix(".mp4")
                    store.window(key)
                    if not store.media:
                        raise KeyError("Training clips are not prepared")
                    return self.file(store.work / "clips" / (key + ".mp4"), "video/mp4", True)
                if re.fullmatch(r"/exports/[0-9a-f]{64}/training_labels\.csv", path):
                    return self.file(store.work / path.lstrip("/"), "text/csv")
                assets = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/favicon.svg": ("favicon.svg", "image/svg+xml"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if path in assets:
                    name, mime = assets[path]
                    return self.file(STATIC / name, mime)
                return self.json(404, dict(error="Route not found"))
            except KeyError:
                self.json(404, dict(error="Training window not found"))
            except (ValueError, OSError):
                self.json(500, dict(error="Local evidence could not be read; check the workspace files"))

        def do_POST(self):
            origin = self.headers.get("Origin")
            if not self.valid_host() or self.headers.get("X-Annotation-Token") != token or (origin and origin not in ("http://127.0.0.1:" + str(self.server.server_port), "http://localhost:" + str(self.server.server_port))):
                return self.json(403, dict(error="Save requests must come from this local workspace"))
            path = urlsplit(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 2_000_000:
                    raise ValueError("Request is too large")
                body = json.loads(self.rfile.read(length))
                if path.startswith("/api/annotation/"):
                    return self.json(200, store.save(path.rsplit("/", 1)[1], body))
                if path == "/api/export":
                    return self.json(200, store.export())
                return self.json(404, dict(error="Route not found"))
            except KeyError:
                self.json(404, dict(error="Training window not found"))
            except Conflict as error:
                self.json(409, dict(error=str(error)))
            except (ValueError, TypeError) as error:
                self.json(400, dict(error=str(error)))
            except OSError:
                self.json(500, dict(error="Save failed; your draft is still open. Check disk space and permissions."))

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = AnnotationStore(args.freeze, args.work)
    server = make_server(store, args.port)
    print("Annotation workspace: http://127.0.0.1:{}".format(server.server_port), flush=True)
    print("12 training windows available; 12 audit windows sealed. Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
