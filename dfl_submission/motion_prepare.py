"""Add player-set tensors to the verified external temporal training sequences."""

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_submission import external_benchmark, long_context, motion_features
from dfl_submission.temporal_network import targets
from dfl_submission.train import write_json
from dfl_transfer_data import digest


def prepare(dfl, skill, previous, output, *, raw=False):
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((previous / "manifest.json").read_text())
    if old["skill_manifest"] != digest(skill / "manifest.json"):
        raise ValueError("Prepared temporal features use a different external corpus")
    by_name = {item["file"]: item for item in old["records"]}
    records = []
    for half in external_benchmark.halves(dfl, skill, "train", raw=raw):
        filename = f"{half.key}.npz"
        item = by_name[filename]
        feature_path = previous / filename
        if digest(feature_path) != item["sha256"] or item["match"] != half.match or item["source"] != half.source:
            raise ValueError(f"External feature identity differs: {filename}")
        frames = np.arange(0, len(half.ball), long_context.STRIDE)
        with np.load(feature_path, allow_pickle=False) as data:
            if not np.array_equal(data["targets"], targets(half.labels[frames])) or len(data["values"]) != len(frames):
                raise ValueError(f"External training rows differ: {filename}")
        players = motion_features.features(half.positions, half.ball, half.teams, frames)
        np.savez_compressed(output / filename, players=players)
        records.append(dict(item, player_sha256=digest(output / filename)))
        print(f"Prepared player motion {half.key}", flush=True)
    write_json(output / "manifest.json", {"records": records, "base_manifest": digest(previous / "manifest.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl", type=Path, required=True)
    parser.add_argument("--dfl-raw", action="store_true")
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dfl, args.skill, args.previous, args.out, raw=args.dfl_raw)
