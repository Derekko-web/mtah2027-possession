"""Recreate or verify immutable anchored references for the stoppage research.

Run the anchored reproduction/confirmation first. Reserved annotation-window
metadata is separately recreated by freeze_annotation_windows.py; no annotations
or media are created by either preparation command.
"""

from pathlib import Path
import shutil

import pandas as pd

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT/"data/control_renewal_v1"
    root.mkdir(parents=True, exist_ok=True)
    candidate = ROOT/"data/candidate_anchored_control"
    expected = "973e1670e9e1240ecdde3bbabebf1e803f0d3b6259aad1b9b3223c413d26dcfd"
    if digest(candidate/"candidate.csv") != expected:
        raise ValueError("Anchored reference differs")
    target = root/"baseline_candidate"
    if not target.exists():
        shutil.copytree(candidate, target)
    elif digest(target/"candidate.csv") != expected:
        raise ValueError("Preserved anchored reference differs")
    source, target = ROOT/"data/control_boundary_v1/confirmation", root/"baseline_confirmation"
    if not target.exists():
        shutil.copytree(source, target)
    else:
        for protocol in ("buffered_minutes", "purged_spells", "buffered_100_seconds"):
            pd.testing.assert_frame_equal(pd.read_csv(source/protocol/"oof_predictions.csv"),
                                          pd.read_csv(target/protocol/"oof_predictions.csv"))
    print("Anchored reference artifacts prepared and verified.")


if __name__ == "__main__":
    main()
