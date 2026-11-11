"""Prepare only frozen TRAIN clips locally, with exact absolute-frame trimming.

One sequential decode of the first-half prefix avoids keyframe-seek rounding.
No media or labels are sent to any service. Existing outputs are never replaced.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from annotation_server import DEFAULT_FREEZE, DEFAULT_WORK, FPS, load_freeze, sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, half_video_blocks, video_path


def probe(path, count=False):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
    if count:
        command += ["-count_frames"]
    command += ["-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,start_time,nb_frames,nb_read_frames,duration", "-of", "json", str(path)]
    return json.loads(subprocess.check_output(command))["streams"][0]


def build_command(source, windows, output):
    if not windows or any(window["split"] != "train" for window in windows):
        raise ValueError("Only training windows may be prepared")
    stages = ["[0:v]split={}".format(len(windows)) + "".join("[v{}]".format(i) for i in range(len(windows)))]
    for index, window in enumerate(windows):
        stages.append("[v{0}]trim=start_frame={1}:end_frame={2},setpts=PTS-STARTPTS,format=yuv420p[c{0}]".format(index, window["start"], window["end"] + 1))
    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
               "-t", "{:.2f}".format((max(w["end"] for w in windows) + 1) / FPS), "-i", str(source),
               "-filter_complex_threads", "1", "-filter_complex", ";".join(stages)]
    for index, window in enumerate(windows):
        command += ["-map", "[c{}]".format(index), "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast", "-crf", "18",
                    "-fps_mode", "passthrough", "-movflags", "+faststart", str(output / (window["window_id"] + ".mp4"))]
    return command


def prepare(freeze, work, source):
    manifest, freeze_hash = load_freeze(freeze)
    windows = [w for w in manifest["windows"] if w["split"] == "train"]
    _, half_start, half_end = half_video_blocks(GAME)[0]
    if any(not half_start <= w["start"] <= w["end"] <= half_end or w["start"] <= 18684 + 250 for w in windows):
        raise ValueError("Training windows must be inside the first half and outside the public buffer")
    metadata = probe(source)
    if metadata["r_frame_rate"] != "25/1" or metadata["avg_frame_rate"] != "25/1" or float(metadata.get("start_time", -1)) != 0:
        raise ValueError("Expected the original zero-origin 25fps tactical source")
    output = Path(work) / "clips"
    output.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(source)
    command = build_command(source, windows, output)
    print("Preparing 12 training clips locally; audit clips are excluded.", flush=True)
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (output / "preparation.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    clips = {}
    for window in windows:
        path = output / (window["window_id"] + ".mp4")
        info = probe(path, count=True)
        if int(info["nb_read_frames"]) != 750 or info["r_frame_rate"] != "25/1" or info["avg_frame_rate"] != "25/1" or float(info["start_time"]) != 0:
            raise ValueError("Prepared clip failed frame-count/cadence checks")
        clips[window["window_id"]] = dict(sha256=sha256(path), source_start=window["start"], source_end=window["end"], **info)
    if sha256(source) != source_hash:
        raise ValueError("Source video changed during preparation")
    result = dict(schema_version=1, freeze_sha256=freeze_hash, source_sha256=source_hash,
                  source_metadata=metadata, fps=FPS, split="train", audit_windows_sealed=12,
                  frame_mapping="Absolute zero-based decoder frame n; trim [start,end+1), rebase timestamps only; no resampling or seeking",
                  clips=clips)
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Verified all 12 clips: 750 frames each at 25fps. Audit media remains sealed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--source", type=Path, default=Path(video_path()))
    args = parser.parse_args()
    prepare(args.freeze, args.work, args.source)


if __name__ == "__main__":
    main()
