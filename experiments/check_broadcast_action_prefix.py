"""Independent source-prefix mapping plus complete feature replay; aggregate output only."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import formation_anchors,map_anchors
from prepare_broadcast_action_features import construct
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME,half_video_blocks


def main():
    root=ROOT/'data/control_broadcast_action_v1'
    audit_manifest(root/'observations');audit_manifest(root/'inputs')
    upstream=ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(upstream)
    tactical=dict(np.load(upstream/'tactical_observations.npz'))
    broadcast=dict(np.load(upstream/'broadcast_observations.npz'))
    alignment=json.loads((ROOT/'experiments/BROADCAST_ALIGNMENT_PLAN.json').read_text())
    records=json.loads((root/'observations/progress.json').read_text())
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    output=root/'prefix_replay';output.mkdir()
    views=[];started=time.monotonic()
    for n in (18935,19060):
        # Recompute both mappings from actual truncated source inputs, without targets.
        source={k:v[:n] if k in ('positions','speeds','ball','valid') else v for k,v in tactical.items()}
        anchors=formation_anchors(source,broadcast,alignment['alignment']['profiles'])
        mapping=map_anchors(anchors,source,broadcast,alignment['alignment']['profiles']['strict'])
        np.savez_compressed(output/f'mapping_{n}.npz',**mapping)
        dense,view,_=construct(root/'observations',records,mapping,n,half_video_blocks(GAME),reserved,held)
        view.to_pickle(output/f'features_{n}.pkl')
        dense.to_pickle(output/f'dense_{n}.pkl')
        views.append(view.iloc[11185:18685])
        if n==19060:
            for key,value in dict(np.load(upstream/'strict_mapping.npz')).items():
                np.testing.assert_array_equal(value,mapping[key])
    pd.testing.assert_frame_equal(*views,check_exact=True)
    pd.testing.assert_frame_equal(views[1],pd.read_pickle(root/'inputs/events_only_inputs.pkl').iloc[11185:18685],check_exact=True)
    result=dict(actual_source_lengths=[18935,19060],independently_recomputed_mappings=True,full_mapping_exact=True,
                complete_future_features_exact=True,public_rows=7500,columns=121,targets_not_read=True,
                seconds=time.monotonic()-started,source_sha256={n:digest(ROOT/'experiments'/n) for n in ('check_broadcast_action_prefix.py','broadcast_alignment.py','broadcast_action_features.py','broadcast_action_sampling.py')},
                input_sha256={str(p.relative_to(ROOT)):digest(p) for p in (root/'inputs/manifest.json',upstream/'manifest.json')},
                cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('sha256')},indent=2))


if __name__=='__main__':main()
