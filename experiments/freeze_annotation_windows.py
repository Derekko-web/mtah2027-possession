"""Freeze local annotation windows without using predictions or private labels.

Six minutes for training (first half, excluding the public labels and buffers),
six for a future audit (second half). Each split has six random windows and six
around automatically tagged restarts. No media is clipped or uploaded. The audit
is for human/video labels, not an independent copy of the corrected-tracking target.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import GAME, half_video_blocks
from starter.load_smart_tagging import load as load_tags

SEED = 42
WINDOW_FRAMES = 30*25
BUFFER_FRAMES = 10*25
PER_STRATUM = 6


def sample_windows(events, halves, rng):
    selected = []
    for split, half in (("train", halves[0]), ("audit", halves[1])):
        _, left, right = half
        # The only existing labels end at 18684; exclude their full input buffer.
        if split == "train":
            left = max(left, 18685+BUFFER_FRAMES)
        last_start = right-WINDOW_FRAMES+1

        def can_add(start):
            end = start+WINDOW_FRAMES-1
            return left <= start <= last_start and all(
                end+BUFFER_FRAMES < other["start"] or start > other["end"]+BUFFER_FRAMES
                for other in selected)

        candidates = sorted(set(int(event["start_frame"])-WINDOW_FRAMES//2 for event in events
                                if event["code"].lower() == "set pieces"))
        counts = {"restart": 0, "uniform": 0}
        for index in rng.permutation(len(candidates)):
            start = candidates[index]
            if can_add(start):
                selected.append(dict(split=split, stratum="restart", start=start,
                                     end=start+WINDOW_FRAMES-1))
                counts["restart"] += 1
                if counts["restart"] == PER_STRATUM:
                    break
        if counts["restart"] != PER_STRATUM:
            raise ValueError("Insufficient nonoverlapping tagged restart windows")
        for _ in range(10000):
            start = int(rng.integers(left, last_start+1))
            if can_add(start):
                selected.append(dict(split=split, stratum="uniform", start=start,
                                     end=start+WINDOW_FRAMES-1))
                counts["uniform"] += 1
                if counts["uniform"] == PER_STRATUM:
                    break
        if counts["uniform"] != PER_STRATUM:
            raise ValueError("Insufficient nonoverlapping uniform windows")
    selected.sort(key=lambda row: (row["split"], row["start"]))
    for number, row in enumerate(selected, 1):
        row["window_id"] = f"window_{number:02d}"
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "data/continuation_20261021/annotation_freeze")
    args = parser.parse_args()
    windows = sample_windows(load_tags(), half_video_blocks(GAME), np.random.default_rng(SEED))
    manifest = dict(seed=SEED, window_frames=WINDOW_FRAMES, buffer_frames=BUFFER_FRAMES,
                    selection_inputs="Published half bounds and automatic Smart Tagging only; no model predictions.",
                    annotation_status="No new annotations exist; no audit results have been inspected.",
                    label_source="Future local human/video annotations, separate from official public and private labels.",
                    audit_policy="Freeze models before annotation; keep audit labels unavailable to training and tuning. "
                                 "An auditor runs one comparison after training decisions are frozen. "
                                 "Report uniform/restart strata separately; enriched sampling does not estimate full-match accuracy.",
                    windows=windows)
    encoded = json.dumps(manifest, indent=2)+"\n"
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "windows.json"
    if target.exists() and target.read_text() != encoded:
        raise SystemExit("Refusing to replace a frozen annotation sample")
    target.write_text(encoded)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    (args.out / "windows.sha256").write_text(digest+"  windows.json\n")
    print(f"Frozen {len(windows)} windows: six minutes train, six minutes audit; SHA256 {digest}")


if __name__ == "__main__":
    main()
