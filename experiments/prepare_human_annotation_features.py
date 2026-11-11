"""Rebuild the original ordered102 representation for reviewed train windows."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from human_annotation_training import digest
from temporal_candidate import build_inputs


def main(output):
    root=ROOT/'data/control_human_annotations_v1';plan_path=ROOT/'experiments/HUMAN_ANNOTATION_PLAN.json'
    plan=json.loads(plan_path.read_text())
    if plan_path.read_bytes()!=(root/'protocol_freeze/plan.json').read_bytes():raise ValueError('Frozen plan changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen input changed')
    metadata=json.loads((root/'annotation_snapshot/windows.json').read_text())
    snapshot=json.loads((root/'annotation_snapshot/snapshot.json').read_text())
    ids={row['window_id'] for row in snapshot['included']}
    selected=[w for w in metadata['windows'] if w['window_id'] in ids]
    if len(selected)!=3 or any(w['split']!='train' for w in selected):raise ValueError('Training snapshot differs')
    end=max(w['end'] for w in selected)
    short_rows,long_rows=[end+extra for extra in plan['feature_prefix_extra_rows']]
    if long_rows>=min(w['start'] for w in metadata['windows'] if w['split']=='audit')-250:
        raise ValueError('Feature prefix reaches sealed audit context')
    human=pd.read_csv(root/'annotation_snapshot/training_labels.csv',usecols=['frame']).frame.to_numpy()
    public=np.arange(11185,18685)
    checked=np.r_[public,human]
    output.mkdir(parents=True,exist_ok=False)
    print('Rebuilding longer tracking feature prefix; no media or possession target inputs',flush=True)
    raw,_,long=build_inputs(long_rows)
    expected=pd.read_pickle(ROOT/'data/advanced_validation/ordered.pkl')
    pd.testing.assert_frame_equal(long.iloc[public],expected.iloc[public],check_exact=True)
    oldraw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl')
    pd.testing.assert_frame_equal(raw[list(oldraw)].iloc[public],oldraw.iloc[public],check_exact=True)
    del raw
    print('Rebuilding distinct shorter prefix for public and human padding parity',flush=True)
    _,_,short=build_inputs(short_rows)
    pd.testing.assert_frame_equal(long.iloc[checked],short.iloc[checked],check_exact=True)
    if len(long.columns)!=102 or list(long)!=list(expected):raise ValueError('Selected input schema changed')
    long.to_pickle(output/'ordered.pkl')
    source_names=['experiments/'+name for name in ('prepare_human_annotation_features.py','human_annotation_training.py','temporal_candidate.py','temporal_control.py','rich_features.py','ball_kinematics.py','relative_motion.py','train_factorized_baseline.py')]+['games.py','starter/load_smart_tagging.py','starter/baseline.py']
    result=dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/name) for name in source_names},input_sha256=plan['input_sha256'],cache_sha256={'ordered.pkl':digest(output/'ordered.pkl')},
        actual_short_rows=short_rows,actual_long_rows=long_rows,distinct_prefix_difference=long_rows-short_rows,
        actual_public_and_human_prefix_parity=True,public_selected102_feature_identity=True,feature_columns=list(long),
        training_frames=len(human),audit_context_excluded=True,no_media_or_target_features=True)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Public feature identity and distinct-prefix parity pass for all 2250 human rows; ordered102 cache saved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/control_human_annotations_v1/inputs')
    main(parser.parse_args().output)
