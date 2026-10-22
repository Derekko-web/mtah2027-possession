"""Prepare the fixed broadcast reference for fresh pair-motion reproduction."""

import json
from pathlib import Path
import shutil

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    if digest(ROOT/"data/candidate_broadcast_closure/candidate.csv") != "dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce":
        raise ValueError("Reproduce selected broadcast candidate first")
    source = ROOT/"data/control_broadcast_v1/closure_confirmation"
    if not json.loads((source/"promotion_checks.json").read_text())["passed"]: raise ValueError("Broadcast confirmation has not passed")
    target = ROOT/"data/control_pair_motion_v1/baseline_confirmation"
    for path in sorted(source.rglob("*")):
        if not path.is_file(): continue
        out = target/path.relative_to(source)
        if out.exists():
            if digest(out) != digest(path): raise ValueError("Protected pair-motion reference differs")
        else:
            out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
    print("Unchanged broadcast reference prepared")


if __name__ == "__main__": main()
