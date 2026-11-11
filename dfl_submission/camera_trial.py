"""Measure whether an independent tracking view recovers possessions on external DFL."""

import argparse
import json
from pathlib import Path

import numpy as np

from dfl_submission import models as shared
from dfl_submission.camera import camera_map, fill_ball
from dfl_submission.path_trial import cache, rank, score
from dfl_submission.decoder import decode
from dfl_submission.robust import CONTROL, verified_models
from dfl_submission.robust_features import stress
from dfl_submission.train import DEVELOPMENT, HOLDOUT, TRAIN, write_json
from dfl_transfer_data import SECTIONS
from dfl_transfer_features import corrupt
from evaluation import possession_metrics as pm


POLICY = {"max_error": 2.5, "min_margin": .6}


def second_camera(positions, ball, seed):
    rng = np.random.default_rng(seed)
    xy, noisy_ball = stress(positions, ball, seed)
    # A narrower shot sees fewer players, and replays reuse an earlier time span.
    far = np.abs(positions[:, :, 0] - ball[:, None, 0]) > 33
    xy[far] = np.nan
    keep = []
    for start in range(0, len(ball), 1500):
        first = start + int(rng.integers(0, 50))
        stop = min(len(ball), first + int(rng.integers(1000, 1470)))
        keep.append(np.arange(first, stop))
        if start >= 1500 and rng.random() < .25:
            replay = start - int(rng.integers(500, 1000))
            keep.append(np.arange(replay, replay + 100))
    clock = np.concatenate(keep)
    reflection = np.array([-1, 1])
    return xy[clock] * reflection, noisy_ball[clock] * reflection, clock


def evaluate(folder, output):
    frozen = json.loads((CONTROL / "recipe.json").read_text())
    models = verified_models(CONTROL, frozen["models"])
    hashes = json.loads((folder / "manifest.json").read_text())["cache_sha256"]
    results = {}
    for match in (DEVELOPMENT, *HOLDOUT):
        index = (*TRAIN, DEVELOPMENT, *HOLDOUT).index(match)
        baseline, enhanced, alignment = [], [], []
        for half_index, section in enumerate(SECTIONS):
            name = f"DFL-MAT-J03{match}_{section}.npz"
            with cache(folder / name, hashes) as data:
                positions, ball, teams, labels = (data[k] for k in ("positions", "ball", "teams", "labels"))
            seed = 270000 + 100 * index + half_index
            primary, primary_ball = corrupt(positions, ball, seed)
            secondary, secondary_ball, clock = second_camera(positions, ball, seed + 500000)
            mapping, reflection = camera_map(primary, secondary, teams, teams, **POLICY)
            filled = fill_ball(primary_ball, secondary_ball, mapping, reflection)
            new_ball = np.isfinite(filled).all(axis=1) & ~np.isfinite(primary_ball).all(axis=1)
            mapped = mapping >= 0
            error = np.abs(clock[mapping[mapped]] - np.flatnonzero(mapped))
            recovered_error = np.abs(clock[mapping[new_ball]] - np.flatnonzero(new_ball))
            summary = dict(coverage=float(mapped.mean()), recovered_ball_frames=int(new_ball.sum()),
                           alignment_within_5=float(np.mean(error <= 5)) if len(error) else 0,
                           recovered_within_5=float(np.mean(recovered_error <= 5)) if len(recovered_error) else 0)
            alignment.append(summary)
            truth = pm.label_codes(labels)
            baseline.append((truth, *shared.predict(*models, primary, primary_ball, teams)))
            enhanced.append((truth, *shared.predict(*models, primary, filled, teams)))
            print(f"Camera trial {match} {section}: {summary}", flush=True)
        before = score({"masked": baseline}, decode, frozen["recipe"])["masked"]
        after = score({"masked": enhanced}, decode, frozen["recipe"])["masked"]
        results[match] = dict(control=before, filled=after, halves=alignment,
                              gain=after["spell_f1"] - before["spell_f1"])
        print(f"Camera trial {match} SpellF1 gain {results[match]['gain']:.4f}", flush=True)
    passed = all(r["gain"] > 0 and all(half["recovered_within_5"] >= .98 for half in r["halves"])
                 for r in results.values())
    write_json(output, dict(holdout_passed=passed, policy=POLICY, matches=results,
                            gate="SpellF1 gain in every match and >=98% of recovered observations aligned within 5 frames"))
    print(f"Camera trial passed: {passed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dfl-cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    evaluate(args.dfl_cache, args.out)
