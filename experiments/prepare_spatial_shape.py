"""Target-blind native spatial feature preparation with actual prefix audit."""
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
from spatial_shape import shape_features


def main():
    root = ROOT/'data/control_spatial_shape_v1'
    plan_path = ROOT/'experiments/SPATIAL_SHAPE_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen spatial source changed')
    parent = ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(parent)
    source = dict(np.load(parent/'tactical_observations.npz'))
    blocks = [(start, end+1) for _, start, end in half_video_blocks(GAME)]
    arrays = [source[key] for key in ('positions', 'ball', 'teams', 'valid')]
    full = shape_features(*arrays, blocks)
    short = shape_features(*(a if i == 2 else a[:18935] for i, a in enumerate(arrays)), blocks)
    pd.testing.assert_frame_equal(full.iloc[11185:18685], short.iloc[11185:18685], check_exact=True)
    public = full.iloc[11185:18685]
    players = (public.h__shape_count >= 5) & (public.a__shape_count >= 5)
    diagnostics = dict(feature_columns=len(full.columns), public_player_coverage=float(players.mean()),
                       public_ball_coverage=float(public.shape_ball_seen.mean()))
    support = players.mean() >= .8 and public.shape_ball_seen.mean() >= .8
    output = root/'inputs'
    output.mkdir()
    full.to_pickle(output/'shape_inputs.pkl')
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    names = ('prepare_spatial_shape.py', 'spatial_shape.py', 'provided_speed.py')
    result = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256=plan['input_sha256'], cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        targets_not_read=True, actual_padded_prefix_parity=True, support_passed=bool(support), diagnostics=diagnostics)
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(support_passed=bool(support), **diagnostics), indent=2))


if __name__ == '__main__':
    main()
