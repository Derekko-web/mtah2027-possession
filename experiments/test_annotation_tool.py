"""Synthetic-only regressions for annotation integrity, isolation and video frames."""
import csv
import hashlib
import http.client
import io
import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from annotation_server import AnnotationStore, Conflict, load_freeze, make_server, sha256
from prepare_annotation_clips import build_command, probe


def fixture(folder):
    """No competition metadata, pixels or labels are used by these fixtures."""
    folder = Path(folder)
    windows = [dict(window_id="window_{:02d}".format(i + 1), split="train" if i < 12 else "audit",
                    stratum="uniform" if i % 12 < 6 else "restart", start=i * 1250, end=i * 1250 + 749) for i in range(24)]
    path = folder / "windows.json"
    path.write_text(json.dumps(dict(windows=windows)))
    path.with_suffix(".sha256").write_text(sha256(path) + "  windows.json\n")
    work = folder / "work"
    clips = work / "clips"
    clips.mkdir(parents=True)
    media = dict(freeze_sha256=sha256(path), clips={})
    for window in windows[:12]:
        clip = clips / (window["window_id"] + ".mp4")
        clip.write_bytes(b"synthetic-video-content")
        media["clips"][window["window_id"]] = dict(sha256=sha256(clip))
    (clips / "manifest.json").write_text(json.dumps(media))
    return path, work


class StoreFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.freeze, self.work = fixture(self.temp.name)
        self.store = AnnotationStore(self.freeze, self.work)
        self.key = "window_01"

    def value(self, **changes):
        value = dict(revision=0, annotator="Teammate One", reviewer="", status="draft",
                     segments=[dict(start=0, end=749, label="H", note="")])
        value.update(changes)
        return value


class StoreTests(StoreFixture, unittest.TestCase):
    def test_freeze_checksum_and_clip_identity_are_verified(self):
        self.freeze.write_text(self.freeze.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_freeze(self.freeze)
        self.freeze.with_suffix(".sha256").write_text(sha256(self.freeze))
        (self.work / "clips/window_01.mp4").write_bytes(b"different")
        # The freeze is also bound into the prepared-media manifest.
        with self.assertRaises(ValueError):
            AnnotationStore(self.freeze, self.work)

    def test_audit_is_absent_from_catalog_store_and_clip_command(self):
        self.assertEqual(len(self.store.catalog()["windows"]), 12)
        self.assertTrue(all(w["split"] == "train" for w in self.store.catalog()["windows"]))
        with self.assertRaises(KeyError):
            self.store.get("window_13")
        with self.assertRaises(KeyError):
            self.store.save("window_13", self.value())
        with self.assertRaises(ValueError):
            build_command(Path("source"), [dict(split="audit")], Path("out"))
        self.assertFalse((self.work / "annotations/audit").exists())

    def test_range_validation_rejects_noninteger_outside_overlap_and_submission_x(self):
        invalid = [(True, 749, "H"), (0.5, 749, "H"), ("0", 749, "H"), (-1, 749, "H"),
                   (0, 750, "H"), (5, 4, "H"), (0, 749, "X")]
        for start, end, label in invalid:
            with self.subTest(start=start, end=end, label=label), self.assertRaises(ValueError):
                self.store.save(self.key, self.value(segments=[dict(start=start, end=end, label=label)]))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.store.save(self.key, self.value(segments=[dict(start=0, end=10, label="H"), dict(start=10, end=749, label="A")]))

    def test_review_requires_complete_coverage_and_distinct_reviewer(self):
        for reviewer in ("", "teammate one", " Teammate One "):
            with self.subTest(reviewer=reviewer), self.assertRaises(ValueError):
                self.store.save(self.key, self.value(status="reviewed", reviewer=reviewer))
        with self.assertRaisesRegex(ValueError, "every frame"):
            self.store.save(self.key, self.value(status="reviewed", reviewer="Teammate Two", segments=[dict(start=1, end=749, label="D")]))
        result = self.store.save(self.key, self.value(status="reviewed", reviewer="Teammate Two"))
        self.assertEqual(result["status"], "reviewed")

    def test_uncertainty_requires_reason_and_is_preserved(self):
        with self.assertRaisesRegex(ValueError, "uncertain"):
            self.store.save(self.key, self.value(segments=[dict(start=0, end=749, label="U")]))
        result = self.store.save(self.key, self.value(segments=[dict(start=0, end=749, label="U", note="Occluded")]))
        self.assertEqual(result["segments"][0]["note"], "Occluded")

    def test_revision_conflict_preserves_history_and_latest(self):
        one = self.store.save(self.key, self.value())
        with self.assertRaises(Conflict):
            self.store.save(self.key, self.value())
        two = self.store.save(self.key, self.value(revision=1, segments=[]))
        history = self.work / "annotations/train/history" / self.key
        self.assertEqual(json.loads((history / "000001.json").read_text()), one)
        self.assertEqual(json.loads((history / "000002.json").read_text()), two)
        self.assertEqual(self.store.get(self.key), two)

    def test_failed_atomic_update_can_retry_without_overwriting_history(self):
        with patch.object(Path, "replace", side_effect=OSError("synthetic disk error")):
            with self.assertRaises(OSError):
                self.store.save(self.key, self.value())
        history = self.work / "annotations/train/history" / self.key / "000001.json"
        preserved = history.read_bytes()
        result = self.store.save(self.key, self.value())
        self.assertEqual(result["revision"], 2)
        self.assertEqual(history.read_bytes(), preserved)
        self.assertEqual(self.store.get(self.key), result)

    def test_saved_source_coordinate_drift_is_rejected(self):
        self.store.save(self.key, self.value())
        path = self.work / "annotations/train" / (self.key + ".json")
        document = json.loads(path.read_text())
        document["source_start"] += 1
        path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "frame provenance"):
            self.store.get(self.key)

    def test_export_requires_all_reviews_and_excludes_u_and_audit(self):
        with self.assertRaisesRegex(ValueError, "All 12"):
            self.store.export()
        for key in self.store.windows:
            segments = [dict(start=0, end=749, label="D", note="")]
            if key == self.key:
                segments = [dict(start=0, end=9, label="H", note=""), dict(start=10, end=11, label="U", note="Ambiguous boundary"), dict(start=12, end=749, label="A", note="")]
            self.store.save(key, self.value(status="reviewed", reviewer="Teammate Two", segments=segments))
        result = self.store.export()
        rows = list(csv.DictReader(io.StringIO((Path(result["path"]) / "training_labels.csv").read_text())))
        self.assertEqual(len(rows), 8998)
        self.assertEqual(result["uncertain_frames_excluded"], 2)
        self.assertEqual(rows[0], dict(window_id="window_01", frame="0", label="H"))
        self.assertFalse(any(r["frame"] in ("10", "11") for r in rows))
        self.assertTrue(all(r["window_id"] in self.store.windows and r["label"] in "HAD" for r in rows))
        self.assertEqual(self.store.export(), result)


class HTTPTests(StoreFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.server = make_server(self.store, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        _, result = self.request("GET", "/api/windows")
        self.token = result["save_token"]

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, method, path, body=None, headers=None, raw=False):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        all_headers = {"X-Annotation-Token": getattr(self, "token", "")}
        all_headers.update(headers or {})
        connection.request(method, path, json.dumps(body) if body is not None else None, all_headers)
        response = connection.getresponse()
        data = response.read()
        status = response.status
        connection.close()
        return status, data if raw else json.loads(data)

    def test_http_audit_routes_and_traversal_are_denied(self):
        for path in ("/api/annotation/window_13", "/media/window_13.mp4", "/media/../windows.json", "/../windows.json"):
            self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("POST", "/api/annotation/window_13", self.value())[0], 404)

    def test_http_foreign_host_origin_and_missing_token_are_denied(self):
        self.assertEqual(self.request("GET", "/api/windows", headers={"Host": "elsewhere.invalid"})[0], 403)
        self.assertEqual(self.request("POST", "/api/annotation/window_01", self.value(), {"X-Annotation-Token": ""})[0], 403)
        self.assertEqual(self.request("POST", "/api/annotation/window_01", self.value(), {"Origin": "https://elsewhere.invalid"})[0], 403)

    def test_http_byte_ranges_and_invalid_ranges(self):
        original = b"synthetic-video-content"
        for header, expected in (("bytes=0-3", original[:4]), ("bytes=4-", original[4:]), ("bytes=-4", original[-4:])):
            status, data = self.request("GET", "/media/window_01.mp4", headers={"Range": header}, raw=True)
            self.assertEqual((status, data), (206, expected))
        for header in ("bytes=999-", "bytes=5-1", "bytes=", "bytes=-0"):
            self.assertEqual(self.request("GET", "/media/window_01.mp4", headers={"Range": header})[0], 416)

    def test_http_conflicts_and_validation_are_clear(self):
        self.assertEqual(self.request("POST", "/api/annotation/window_01", self.value())[0], 200)
        self.assertEqual(self.request("POST", "/api/annotation/window_01", self.value())[0], 409)
        self.assertEqual(self.request("POST", "/api/annotation/window_01", [])[0], 400)
        self.assertEqual(self.request("POST", "/api/export", {})[0], 400)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg is required for exact frame regression")
class ClipFrameTests(unittest.TestCase):
    def test_absolute_trim_and_inclusive_edges_match_synthetic_frame_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mkv"
            pixels = b"".join(bytes([10 + index * 3]) * (64 * 64) for index in range(70))
            subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray", "-video_size", "64x64", "-framerate", "25", "-i", "pipe:0", "-c:v", "ffv1", str(source)], input=pixels, check=True)
            windows = [dict(window_id="one", start=11, end=30, split="train"), dict(window_id="two", start=41, end=59, split="train")]
            subprocess.run(build_command(source, windows, root), check=True, capture_output=True)
            for window in windows:
                path = root / (window["window_id"] + ".mp4")
                self.assertEqual(int(probe(path, True)["nb_read_frames"]), window["end"] - window["start"] + 1)
                decoded = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1"])
                for offset, source_index in ((0, window["start"]), (window["end"] - window["start"], window["end"])):
                    sample = decoded[offset * 4096:(offset + 1) * 4096]
                    self.assertAlmostEqual(sum(sample) / len(sample), 10 + source_index * 3, delta=1.1)


if __name__ == "__main__":
    unittest.main()
