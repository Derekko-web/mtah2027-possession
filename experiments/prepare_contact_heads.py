"""Target-blind preparation of the frozen contact-history feature experiment."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from contact_heads import feature_views
from finalize_broadcast_closure import audit_manifest
from games import GAME, half_video_blocks


def main():
    root = ROOT/'data/control_contact_heads_v1'
    plan_path = ROOT/'experiments/CONTACT_HEADS_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen contact history parent changed')
    parent = ROOT/'data/control_color_contact_v1/inputs'
    manifest = audit_manifest(parent)
    counts = manifest['diagnostics']['fused_strong_by_team']
    if sum(counts) < 100 or min(counts) < 10:
        raise SystemExit('STOP BEFORE TARGETS: parent support failed')
    contact = {name:np.load(parent/(name+'.npy')) for name in
               ('detected_metadata', 'tracked_color', 'detected_color')}
    n = len(contact['detected_color'])
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    views = feature_views(contact, n, blocks)
    short = feature_views(contact, 18935, blocks)
    for name in views:
        pd.testing.assert_frame_equal(views[name].iloc[11185:18685], short[name].iloc[11185:18685], check_exact=True)
    output = root/'inputs'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    for name, view in views.items():
        view.to_pickle(output/(name+'_inputs.pkl'))
    names = ('prepare_contact_heads.py', 'contact_heads.py', 'provided_speed.py')
    result = dict(plan_sha256=digest(plan_path),
        source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256=plan['input_sha256'],
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        actual_padded_prefix_parity=True, targets_not_read=True, support_passed=True,
        feature_columns={name:len(view.columns) for name, view in views.items()})
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['feature_columns']))


if __name__ == '__main__':
    main()
