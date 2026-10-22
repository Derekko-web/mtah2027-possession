"""Read sparse supplied ball-speed fields and verify bounded observation parity."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from broadcast_alignment import PITCH
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME, half_video_blocks, require
from inspect_ball_scalar import load_ball_fields
from prepare_broadcast_alignment import sliced
from provided_speed import speed_views


def main():
    root = ROOT/'data/control_provided_speed_v1'
    output = root/'inputs'
    output.mkdir()
    upstream = ROOT/'data/control_broadcast_v1/inputs'
    upstream_manifest = audit_manifest(upstream)
    plan_path = ROOT/'experiments/PROVIDED_SPEED_PLAN.json'
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    tactical, broadcast = [dict(np.load(upstream/(name+'_observations.npz'))) for name in ('tactical', 'broadcast')]
    mapping = dict(np.load(upstream/'strict_mapping.npz'))
    scalar, specs, inputs = {}, {}, {'data/control_broadcast_v1/inputs/manifest.json':digest(upstream/'manifest.json')}
    for name, source in [('tactical', tactical), ('broadcast', broadcast)]:
        fields, spec = load_ball_fields(name, len(source['ball']))
        np.testing.assert_array_equal((fields[:, :2]*PITCH).astype(np.float32), source['ball'])
        scalar[name], specs[name] = fields[:, 2], spec
        np.save(output/(name+'_provided_speed.npy'), fields[:, 2])
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    views, info = speed_views(tactical, broadcast, scalar['tactical'], scalar['broadcast'],
        specs['tactical']['raw_to_declared_meters_per_second'], specs['broadcast']['raw_to_declared_meters_per_second'], mapping, blocks)
    n = 18935
    short_mapping = {key:value[:n] if len(value) == len(tactical['ball']) else value for key, value in mapping.items()}
    short_views, short_info = speed_views(sliced(tactical, n), broadcast, scalar['tactical'][:n], scalar['broadcast'],
        specs['tactical']['raw_to_declared_meters_per_second'], specs['broadcast']['raw_to_declared_meters_per_second'], short_mapping, blocks)
    for name, table in views.items():
        pd.testing.assert_frame_equal(table.iloc[11185:18685], short_views[name].iloc[11185:18685], check_exact=True)
        table.to_pickle(output/(name+'_inputs.pkl'))
    pd.testing.assert_frame_equal(info.iloc[11185:18685], short_info.iloc[11185:18685], check_exact=True)
    info.to_pickle(output/'speed_info.pkl')
    public = info.iloc[11185:18685]
    diagnostics = dict(feature_columns={name:len(value.columns) for name, value in views.items()},
        public_tactical_scalar_fraction=float(public.tactical_seen.mean()),
        public_broadcast_scalar_fraction=float(public.broadcast_seen.mean()),
        public_tactical_context_fraction=float((public.tactical_context > 0).mean()),
        public_broadcast_context_fraction=float((public.broadcast_context > 0).mean()))
    names = ('provided_speed.py', 'prepare_provided_speed.py', 'inspect_ball_scalar.py', 'pair_motion.py')
    manifest = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256=inputs, camera_sha256=upstream_manifest['input_sha256'], metadata_specs=specs, targets_not_read=True, media_not_read=True,
        actual_padded_prefix_parity=True, check_exact=True, diagnostics=diagnostics,
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(diagnostics, indent=2))


if __name__ == '__main__':
    main()
