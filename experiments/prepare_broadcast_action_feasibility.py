"""Metadata-only safe broadcast action-window inventory; no images or labels."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from games import GAME,half_video_blocks
from build_temporal_candidate import digest
from sports_action_sampling import clip_frames
from finalize_broadcast_closure import audit_manifest


def eligible(n,mapping,blocks,reserved,held):
    rows=[]
    counts=dict(grid=0,boundary_or_sealed=0,missing_or_changing_mapping=0,source_not_locally_covered=0)
    for center in range(11000,18951,50):
        counts['grid']+=1
        start,end=center-150,center+150
        bad=start<0 or end>=n or not any(a<=start and end<=b for _,a,b in blocks)
        bad=bad or any(start<=r['end']+250 and end>=r['start']-250 for r in reserved)
        bad=bad or np.any((held+250>=start)&(held-250<=end))
        if bad:
            counts['boundary_or_sealed']+=1;continue
        frames=mapping['frames'][start:end+1]
        orientation=mapping['orientation'][start:end+1]
        if np.any(frames<0) or len(np.unique(orientation))!=1 or np.any(np.diff(frames)<0):
            counts['missing_or_changing_mapping']+=1;continue
        source=clip_frames(mapping['frames'][center])
        if source[0]<frames.min() or source[-1]>frames.max() or np.any(np.min(np.abs(source[:,None]-frames[None,:]),axis=1)>1):
            counts['source_not_locally_covered']+=1;continue
        rows.append(dict(tactical_center=center,broadcast_center=int(mapping['frames'][center])))
    observed=np.zeros(n,bool)
    for row in rows:
        c=row['tactical_center'];observed[max(0,c-25):min(n,c+26)]=True
    return rows,dict(**counts,eligible_clips=len(rows),public_dense_frames=int(observed[11185:18685].sum()))


def main():
    path=ROOT/'experiments/BROADCAST_ACTION_FEASIBILITY_PLAN.json'
    plan=json.loads(path.read_text())
    for name,value in plan['input_sha256'].items():
        if digest(ROOT/name)!=value:raise ValueError('Frozen broadcast feasibility input changed')
    upstream=ROOT/'data/control_broadcast_v1/inputs'
    audit_manifest(upstream)
    mapping=dict(np.load(upstream/'strict_mapping.npz'))
    reserved=json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held=np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    rows,summary=eligible(len(mapping['frames']),mapping,half_video_blocks(GAME),reserved,held)
    short,short_summary=eligible(18935,mapping,half_video_blocks(GAME),reserved,held)
    assert [r for r in rows if r['tactical_center']<=18700]==[r for r in short if r['tactical_center']<=18700]
    output=ROOT/'data/control_broadcast_action_v1/feasibility';output.mkdir()
    (output/'schedule.json').write_text(json.dumps(rows,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(dict(**summary,actual_public_prefix_schedule_parity=True,no_images_decoded=True,no_targets_read=True),indent=2)+'\n')
    (output/'plan.json').write_bytes(path.read_bytes())
    names=('prepare_broadcast_action_feasibility.py','sports_action_sampling.py','../games.py')
    manifest=dict(plan_sha256=digest(path),source_sha256={n:digest(ROOT/'experiments'/n) for n in names},input_sha256=plan['input_sha256'],cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
