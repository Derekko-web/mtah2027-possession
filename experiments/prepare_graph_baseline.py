"""Prepare unchanged selected8 confirmation without historical session snapshots."""

import json
from pathlib import Path
import shutil

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    if digest(ROOT/'data/candidate_motion_dead_trim/candidate.csv') != '3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea':
        raise ValueError('Reproduce selected8-model candidate first')
    source = ROOT/'data/control_pair_motion_v1/dead_trim_confirmation'
    if not json.loads((source/'promotion_checks.json').read_text())['passed']:
        raise ValueError('Motion-trim confirmation has not passed')
    destination = ROOT/'data/control_ownership_graph_v1/baseline_confirmation'
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        target = destination/path.relative_to(source)
        if target.exists():
            if digest(target) != digest(path):
                raise ValueError('Protected graph baseline differs')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    print('Unchanged selected8-model confirmation prepared')


if __name__ == '__main__':
    main()
