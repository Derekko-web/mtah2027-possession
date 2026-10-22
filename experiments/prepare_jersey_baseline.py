"""Prepare the unchanged broadcast OOF reference without creating historical evidence."""

import json
from pathlib import Path
import shutil

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    candidate = ROOT/"data/candidate_broadcast_closure/candidate.csv"
    if digest(candidate) != "dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce":
        raise ValueError("Reproduce the selected broadcast candidate first")
    source = ROOT/"data/control_broadcast_v1/closure_confirmation"
    if not json.loads((source/"promotion_checks.json").read_text())["passed"]:
        raise ValueError("Broadcast confirmation has not passed")
    destination = ROOT/"data/control_jersey_v1/baseline_confirmation"
    for path in sorted(source.rglob("*")):
        if not path.is_file(): continue
        target = destination/path.relative_to(source)
        if target.exists():
            if digest(target) != digest(path): raise ValueError("Protected jersey baseline differs")
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    print("Unchanged broadcast OOF reference is ready")


if __name__ == "__main__": main()
