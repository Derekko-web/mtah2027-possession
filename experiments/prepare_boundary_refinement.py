"""Preserve reproducible control-confirmation caches without overwriting history.

Run confirm_control_refinement.py, finalize_control_refinement.py and
build_control_candidate.py first. This copies their immutable reference artifacts
into the directory expected by the boundary screen/confirmation/build. Existing
snapshots are checked, never rewritten. No match data leaves the machine.
"""

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def prepare(root):
    source, target = ROOT/"data/control_confirmation", root/"baseline_confirmation"
    candidate, saved = ROOT/"data/candidate_control_consensus", root/"baseline_candidate"
    expected = "19dd8ecac4fe0c3fcbdc11d18d41cd9a38d1cf995a0defb05ff816b6057a64f1"
    if digest(candidate/"candidate.csv") != expected:
        raise ValueError("Unexpected reference candidate")
    if (source/"plan.json").read_bytes() != (ROOT/"experiments/CONTROL_CONFIRMATION_PLAN.json").read_bytes():
        raise ValueError("Reference confirmation plan differs")
    root.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copytree(source, target)
    else:
        for name in ("raw", "base", "ordered"):
            pd.testing.assert_frame_equal(pd.read_pickle(source/(name+".pkl")),
                                          pd.read_pickle(target/(name+".pkl")))
        for protocol in ("buffered_minutes", "purged_spells", "buffered_100_seconds"):
            for seed in (42, 43, 44):
                name = f"fast_seed{seed}_probabilities.npy"
                np.testing.assert_array_equal(np.load(source/protocol/name), np.load(target/protocol/name))
            pd.testing.assert_frame_equal(pd.read_csv(source/protocol/"oof_predictions.csv"),
                                          pd.read_csv(target/protocol/"oof_predictions.csv"))
    if not saved.exists():
        shutil.copytree(candidate, saved)
    elif digest(saved/"candidate.csv") != expected:
        raise ValueError("Preserved candidate differs")
    print("Reference caches prepared and checked:", root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT/"data/control_boundary_v1")
    args = parser.parse_args()
    prepare(args.root.resolve())


if __name__ == "__main__":
    main()
