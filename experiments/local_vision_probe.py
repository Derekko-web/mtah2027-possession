"""Local-only vision probe; images and generated descriptions never leave localhost.

This does not generate a final submission or hand-label frames. Uniform public
samples test a pretrained model's automatic stopped/live judgments. Model answers
stay in ignored artifacts; stdout exposes aggregate accuracy only.
"""

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from games import asset_paths

ENDPOINT = "http://127.0.0.1:11434"
PROMPT = ("Is this soccer match currently in active play, or are players waiting for a restart? "
          "Answer with exactly one word: playing or stopped.")


def local_request(route, body):
    request = urllib.request.Request(ENDPOINT+route, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    # No configurable host: competition images may be sent only to loopback.
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def extract_image(frame, path):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-ss", str(frame/25),
                    "-i", asset_paths()["video"], "-frames:v", "1", "-vf", "scale=960:-1",
                    "-y", str(path)], check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="moondream:1.8b")
    parser.add_argument("--out", type=Path, default=ROOT/"data/local_vision_probe")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ENDPOINT+"/api/tags") as response:
        models = {m["name"]: m for m in json.load(response)["models"]}
    if args.model not in models:
        raise SystemExit("Pull the specified local model first")
    samples = np.arange(11185, 18685, 500)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    rows = []
    cache_path = args.out/"answers.json"
    cached = {r["frame"]: r for r in json.loads(cache_path.read_text())} if cache_path.exists() else {}
    for frame in samples:
        frame = int(frame)
        if frame in cached:
            rows.append(cached[frame])
            continue
        image = args.out/f"frame_{frame}.jpg"
        if not image.exists():
            extract_image(frame, image)
        answer = local_request("/api/generate", dict(
            model=args.model, prompt=PROMPT, images=[base64.b64encode(image.read_bytes()).decode()],
            stream=False, think=False, options=dict(temperature=0, seed=42, num_predict=1024)))
        text = answer["response"].strip().lower()
        words = set(re.findall(r"\b(playing|stopped)\b", text))
        prediction = "D" if words == {"stopped"} else "L" if words == {"playing"} else "X"
        rows.append(dict(frame=frame, response=text, prediction=prediction,
                         image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
        cache_path.write_text(json.dumps(rows, indent=2)+"\n")
        print("Processed uniform sample", len(rows), "/", len(samples), flush=True)
    truth = np.where(labels[samples] == "D", "D", "L")
    prediction = np.array([r["prediction"] for r in rows])
    summary = dict(model=models[args.model]["name"], digest=models[args.model]["digest"], prompt=PROMPT,
                   sample_count=len(samples), true_dead_count=int((truth == "D").sum()),
                   accuracy=float((truth == prediction).mean()),
                   dead_recall=float((prediction[truth == "D"] == "D").mean()),
                   live_recall=float((prediction[truth == "L"] == "L").mean()),
                   false_dead_count=int(((prediction == "D") & (truth == "L")).sum()),
                   unknown_count=int((prediction == "X").sum()),
                   note="Small adaptive feasibility probe, not validation or independent annotation.")
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
