"""Verify the supplied-speed branch stopped before any supervised fitting."""

import argparse
import json
from pathlib import Path
import sys
import tarfile
import hashlib

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from broadcast_provenance import verify_camera_inputs
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from inspect_ball_scalar import load_ball_fields


def unexpected_checkpoints(root):
    return sorted(path for path in root.rglob('*.ubj')
                  if not path.is_relative_to(root/'baseline_candidate'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-history', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_provided_speed_v1'
    cache = root/'inputs'
    manifest = audit_manifest(cache)
    verify_camera_inputs(manifest['camera_sha256'])
    audit_manifest(ROOT/'data/control_broadcast_v1/inputs')
    plan = ROOT/'experiments/PROVIDED_SPEED_PLAN.json'
    if manifest['plan_sha256'] != digest(plan) or (cache/'plan.json').read_bytes() != plan.read_bytes():
        raise ValueError('Supplied-speed plan changed')
    if not manifest['actual_padded_prefix_parity'] or not manifest['check_exact']:
        raise ValueError('Bounded observation parity missing')
    for name in ('tactical', 'broadcast'):
        saved = np.load(cache/(name+'_provided_speed.npy'))
        actual, spec = load_ball_fields(name, len(saved))
        np.testing.assert_array_equal(saved, actual[:, 2])
        if spec != manifest['metadata_specs'][name]:
            raise ValueError('Supplied-speed unit specification changed')
    info = pd.read_pickle(cache/'speed_info.pkl').iloc[11185:18685]
    if len(info) != 7500 or (info.fillna(0).to_numpy() != 0).any():
        raise AssertionError('Expected no supervised supplied-speed observations/context')
    if (root/'screen').exists() or unexpected_checkpoints(root):
        raise AssertionError('Unidentifiable supplied-speed branch produced supervised models/scores')
    if args.audit_history:
        baseline = json.loads((root/'baseline_manifest.json').read_text())
        for name, expected in baseline.items():
            if name in ('experiments/SELECTED_MODEL.md', 'experiments/WORK_HISTORY.md'):
                continue
            if digest(ROOT/name) != expected:
                raise AssertionError('Protected selected8/ledger/reference artifact changed')
        with tarfile.open(root/'experiments_before.tar.gz') as archive:
            for member in archive.getmembers():
                if not member.isfile() or '__pycache__' in Path(member.name).parts or member.name in ('experiments/SELECTED_MODEL.md', 'experiments/WORK_HISTORY.md'):
                    continue
                if digest(ROOT/member.name) != hashlib.sha256(archive.extractfile(member).read()).hexdigest():
                    raise AssertionError('Preceding scientific source or score changed')
    result = dict(preflight_rejected=True, supervised_rows=7500,
        tactical_scalar_and_context_observations=0, broadcast_scalar_and_context_observations=0,
        supplied_fields_reparsed_exact=True, metadata_units_reverified=True,
        actual_padded_prefix_parity=True, all_prepared_features_and_sources_verified=True,
        public_possession_models_fitted=0, possession_score_rows_added=0,
        no_screen_or_checkpoints=True, history_audited=args.audit_history,
        calibration_or_private_accuracy_claimed=False)
    (root/'integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
