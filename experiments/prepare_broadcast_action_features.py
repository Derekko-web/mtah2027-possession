"""Frozen sparse support and actual full future-feature prefix comparison."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME, half_video_blocks
from broadcast_action_sampling import schedule
from broadcast_action_features import dense_actions, feature_view, support_summary


def construct(observations, records, mapping, n, blocks, reserved, held):
    rows = schedule(n, mapping, blocks, reserved, held)
    dense, sources, components = dense_actions(observations, records, rows, mapping, n, blocks)
    return dense, feature_view(dense, components), sources


def main():
    root = ROOT/'data/control_broadcast_action_v1'
    observations = root/'observations'
    manifest = audit_manifest(observations)
    plan_path = ROOT/'experiments/BROADCAST_ACTION_PLAN.json'
    plan = json.loads(plan_path.read_text())
    if manifest['plan_sha256'] != digest(plan_path):
        raise ValueError('Broadcast observation plan differs')
    audit_manifest(root/'feasibility')
    audit_manifest(ROOT/'data/control_broadcast_v1/inputs')
    mapping = dict(np.load(ROOT/'data/control_broadcast_v1/inputs/strict_mapping.npz'))
    records = json.loads((observations/'progress.json').read_text())
    reserved = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    blocks = half_video_blocks(GAME)
    dense, full, sources = construct(observations, records, mapping, 19060, blocks, reserved, held)
    short_dense, short, _ = construct(observations, records, mapping, 18935, blocks, reserved, held)
    pd.testing.assert_frame_equal(full.iloc[11185:18685], short.iloc[11185:18685], check_exact=True)
    pd.testing.assert_frame_equal(dense.iloc[11185:18685], short_dense.iloc[11185:18685], check_exact=True)
    if len(full.columns) != 121:
        raise ValueError('Event feature schema differs')
    support = support_summary(dense, sources, plan['support'])
    output = root/'inputs'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    dense.to_pickle(output/'dense_actions.pkl')
    full.to_pickle(output/'events_only_inputs.pkl')
    (output/'support.json').write_text(json.dumps(support, indent=2)+'\n')
    names = ('prepare_broadcast_action_features.py','broadcast_action_features.py','broadcast_action_sampling.py',
             'sports_action_features.py','sports_action_sampling.py','provided_speed.py','../games.py')
    result = dict(plan_sha256=digest(plan_path), source_sha256={n:digest(ROOT/'experiments'/n) for n in names},
                  input_sha256={str(p.relative_to(ROOT)):digest(p) for p in (observations/'manifest.json',root/'feasibility/manifest.json',ROOT/'data/control_broadcast_v1/inputs/manifest.json')},
                  cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
                  actual_padded_prefix_parity=True, public_prefix_rows=7500, targets_not_read=True,
                  support_passed=support['support_passed'], eligible_heads=support['eligible_heads'], feature_columns={'events_only':121})
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(support, indent=2))


if __name__ == '__main__':
    main()
