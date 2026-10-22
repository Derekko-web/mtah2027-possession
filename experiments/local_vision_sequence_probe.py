"""Local-only three-frame ball-scene feasibility probe, with fixed sampling/prompt."""

import base64
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from games import require, asset_paths
from local_vision_probe import local_request, ENDPOINT
from starter.load_homography import load, project

PROMPT = ("These three images show the same area near the soccer ball, in time order, "
          "one second apart. Determine whether the ball is being used in live play or "
          "being retrieved, carried by hand, placed, or held while preparing a restart. "
          "Walking with the ball in the hands or waiting to take a restart counts as stopped. "
          "Answer exactly one word: playing, stopped, or unclear.")


def read_ball(n_frames):
    ball = np.full((n_frames, 2), np.nan)
    with open(require(["atd_raw"])[0]) as handle:
        for frame, line in enumerate(handle):
            if frame >= n_frames:
                break
            index, _, raw = line.strip().split(":")
            if int(index) != frame+1:
                raise ValueError("Frame alignment failed")
            ball[frame] = np.fromstring(raw, sep=",")[:2]
    return ball


def scene(frame, center, matrix):
    result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(frame/25),
                             "-i", asset_paths()["video"], "-frames:v", "1", "-f", "image2pipe",
                             "-vcodec", "mjpeg", "pipe:1"], check=True, capture_output=True)
    image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
    if np.isfinite(center).all():
        # A 16x16-metre pitch area, with extra vertical room for bodies.
        x = center[0]+np.array([-8, 8, 8, -8])/105.3
        y = center[1]+np.array([-8, -8, 8, 8])/68
        px, py = project(matrix, x, y)
        left, right = np.clip([px.min(), px.max()], 0, image.width)
        top, bottom = np.clip([py.min()-40, py.max()+20], 0, image.height)
        if right-left >= 80 and bottom-top >= 80:
            image = image.crop((int(left), int(top), int(right), int(bottom)))
    image.thumbnail((768, 768))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=90)
    return encoded.getvalue()


def main():
    output = ROOT/"data/local_vision_sequence_probe"
    output.mkdir(exist_ok=True)
    frames = np.arange(11185, 18685, 500)
    ball = read_ball(18711)
    matrices, _, valid = load()
    with urllib.request.urlopen(ENDPOINT+"/api/tags") as response:
        model = next(x for x in json.load(response)["models"] if x["name"] == "qwen3-vl:4b")
    cache = output/"answers.json"
    completed = {r["frame"]: r for r in json.loads(cache.read_text())} if cache.exists() else {}
    rows = []
    for index in frames:
        frame = int(index)
        if frame in completed:
            rows.append(completed[frame])
            continue
        center = ball[frame]
        if not np.isfinite(center).all():
            choices = np.arange(frame-25, frame+26)
            choices = choices[np.isfinite(ball[choices]).all(axis=1)]
            if len(choices):
                center = ball[choices[np.argmin(np.abs(choices-frame))]]
        images = []
        for offset in (-25, 0, 25):
            target = frame+offset
            if not valid[target]:
                raise ValueError("Invalid homography at probe frame")
            path = output/f"frame_{target}_scene_{frame}.jpg"
            if not path.exists():
                path.write_bytes(scene(target, center, matrices[target]))
            images.append(base64.b64encode(path.read_bytes()).decode())
        answer = local_request("/api/generate", dict(model=model["name"], prompt=PROMPT, images=images,
                    stream=False, think=False, options=dict(temperature=0, seed=42, num_predict=1024, num_ctx=8192)))
        text = answer.get("response", "").strip().lower()
        words = set(re.findall(r"\b(playing|stopped|unclear)\b", text))
        prediction = "D" if words == {"stopped"} else "L" if words == {"playing"} else "X"
        rows.append(dict(frame=frame, prediction=prediction, response=text))
        cache.write_text(json.dumps(rows, indent=2)+"\n")
        print("Processed temporal sample", len(rows), "/", len(frames), flush=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    truth = np.where(labels[frames] == "D", "D", "L")
    predictions = np.array([r["prediction"] for r in rows])
    summary = dict(model=model["name"], digest=model["digest"], prompt=PROMPT, samples=len(rows),
                    accuracy=float((truth == predictions).mean()),
                    dead_recall=float((predictions[truth == "D"] == "D").mean()),
                    live_recall=float((predictions[truth == "L"] == "L").mean()),
                    unknown_count=int((predictions == "X").sum()),
                    warning="Adaptive small feasibility probe, not independent validation.")
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
