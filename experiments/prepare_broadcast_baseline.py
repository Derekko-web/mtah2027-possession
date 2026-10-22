"""Prepare the unchanged stoppage OOF reference for a fresh broadcast run."""

from pathlib import Path
import json
import shutil

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT/"data/control_renewal_v1/recovery_confirmation"
    destination = ROOT/"data/control_broadcast_v1/baseline_confirmation"
    if not (source/"promotion_checks.json").is_file():
        raise ValueError("First reproduce stoppage confirmation as described in STOPPAGE_RECOVERY_REPORT.md")
    if not json.loads((source/"promotion_checks.json").read_text())["passed"]:
        raise ValueError("Stoppage confirmation has not passed")
    for path in sorted(source.rglob("*")):
        if not path.is_file(): continue
        target = destination/path.relative_to(source)
        if target.exists():
            if digest(target) != digest(path):
                raise ValueError("Protected broadcast baseline differs: " + str(target.relative_to(ROOT)))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    print("Unchanged stoppage confirmation reference is ready")


if __name__ == "__main__":
    main()
