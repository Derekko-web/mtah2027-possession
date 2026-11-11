"""Target-blind action-feature support screen and actual prefix comparison."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME,half_video_blocks
from sports_action_features import dense_actions,feature_views,support_summary
from starter.load_homography import load


def main():
    root=ROOT/'data/control_sports_action_v1'
    observations=root/'observations'
    manifest=audit_manifest(observations)
    plan_path=ROOT/'experiments/SPORTS_ACTION_FITTING_PLAN.json'
    plan=json.loads(plan_path.read_text())
    original=ROOT/'experiments/SPORTS_ACTION_OBSERVATION_PLAN.json'
    if manifest['plan_sha256']!=digest(original) or plan['observation_plan_sha256']!=digest(original):
        raise ValueError('Action source/fitting chronology differs')
    upstream=ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(upstream)
    source=dict(np.load(upstream/'tactical_observations.npz'))
    records=json.loads((observations/'progress.json').read_text())
    matrices,error,valid=load()
    blocks=[(left,right+1) for _,left,right in half_video_blocks(GAME)]
    full=dense_actions(observations,records,len(source['positions']))
    views=feature_views(full,source,matrices,error,valid,blocks)
    short=dense_actions(observations,records,18935)
    short_views=feature_views(short,source,matrices,error,valid,blocks)
    for name in views:
        pd.testing.assert_frame_equal(views[name].iloc[11185:18685],short_views[name].iloc[11185:18685],check_exact=True)
    support=support_summary(full,views)
    output=root/'inputs'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    full.to_pickle(output/'dense_actions.pkl')
    for name,view in views.items():
        view.to_pickle(output/(name+'_inputs.pkl'))
    (output/'support.json').write_text(json.dumps(support,indent=2)+'\n')
    names=('prepare_sports_action_features.py','sports_action_features.py','sports_action_sampling.py','provided_speed.py','../starter/load_homography.py','../games.py')
    result=dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in (observations/'manifest.json',upstream/'manifest.json',original)},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        targets_not_read=True,actual_padded_prefix_parity=True,support_passed=support['support_passed'],
        eligible_heads=support['eligible_heads'],diagnostics=support,
        feature_columns={name:len(view.columns) for name,view in views.items()})
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(**support,feature_columns=result['feature_columns']),indent=2))


if __name__=='__main__':
    main()
