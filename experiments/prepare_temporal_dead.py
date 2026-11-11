"""Actual-prefix physical-context checks, no annotation or target loading."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from temporal_candidate import build_inputs
from stoppage_context import context_features,load_ball


def main(output):
    plan_path=ROOT/'experiments/TEMPORAL_DEAD_PLAN.json';round_root=ROOT/'data/control_temporal_dead_v1'
    if plan_path.read_bytes()!=(round_root/'protocol_freeze/plan.json').read_bytes():raise ValueError('Plan differs')
    plan=json.loads(plan_path.read_text())
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen input changed: '+name)
    output.mkdir(parents=True,exist_ok=False)
    print('Rebuilding physical features from actual19060rows,using provided tracking only',flush=True)
    raw,base,_=build_inputs(19060);long,_=context_features(raw,base,load_ball(19060))
    print('Rebuilding distinct18935row input and physical features',flush=True)
    raw,base,_=build_inputs(18935);short,_=context_features(raw,base,load_ball(18935))
    check=np.arange(11185-31,18685+31)
    pd.testing.assert_frame_equal(short.iloc[check],long.iloc[check],check_exact=True)
    full=pd.read_pickle(ROOT/'data/control_renewal_v1/stoppage_inputs/physical.pkl')
    pd.testing.assert_frame_equal(short.iloc[check],full.iloc[check],check_exact=True)
    short.to_pickle(output/'physical.pkl');np.save(output/'physical.npy',short.to_numpy(dtype=np.float32))
    names=['prepare_temporal_dead.py','temporal_dead.py','temporal_candidate.py','stoppage_context.py','stoppage_features.py','rich_features.py','ball_kinematics.py','relative_motion.py','temporal_control.py']
    manifest=dict(plan_sha256=digest(plan_path),actual_prefix_rows=[18935,19060],public_plus31_prefix_exact=True,full_public_context_exact=True,
        columns=list(short),feature_count=len(short.columns),source_sha256={'experiments/'+n:digest(ROOT/'experiments'/n) for n in names},
        input_sha256=plan['input_sha256'],cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()},
        targets_loaded=False,human_annotations_loaded=False,private_targets_loaded=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Actual prefix and full-cache parity pass; '+str(len(short.columns))+' team-invariant physical inputs saved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'data/control_temporal_dead_v1/inputs')
    main(parser.parse_args().output)
