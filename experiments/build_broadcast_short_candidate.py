"""Rebuild the qualified broadcast-boundary fix over the frozen full candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from games import GAME,half_video_blocks,source_frames
from broadcast_short_envelope import refine

PARENT=ROOT/'data/candidate_tabpfn_short_control_v1'
ROUND=ROOT/'data/control_short_continuation_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(output):
    selection=json.loads((ROUND/'selection.json').read_text())
    if selection['selected']!='broadcast13':raise ValueError('Only the qualified broadcast-only finalist is supported')
    plan=json.loads((ROUND/'plan.json').read_text())
    for name in ('experiments/broadcast_short_envelope.py',):
        if digest(ROOT/name)!=plan['source_sha256'][name]:raise ValueError('Qualified inference changed')
    baseline=pd.read_csv(PARENT/'candidate.csv');n=source_frames()
    manifest=json.loads((PARENT/'manifest.json').read_text())
    if digest(PARENT/'candidate.csv')!=manifest['csv_sha256']:raise ValueError('Parent candidate changed')
    if len(baseline)!=n or pm.validate_submission(baseline,n):raise ValueError('Invalid parent candidate')
    all_scores=[]
    for seed in (42,43,44):
        rows=pd.read_csv(ROUND/f'seed{seed}_scores.csv');rows=rows[rows.variant=='broadcast13']
        if len(rows)!=3 or not (rows.spellF1>rows.before_spellF1).all():raise ValueError('Incomplete or regressing qualification')
        for protocol in rows.protocol:
            metrics=json.loads((ROUND/'candidates/broadcast13'/f'seed{seed}'/protocol/'metrics.json').read_text())
            if not all(metrics['summary']['checks'].values()):raise ValueError('A paired secondary check failed')
        all_scores.extend(rows.to_dict('records'))
    full_path=ROOT/'data/control_broadcast_v1/full_inputs/combined_inputs.pkl'
    broadcast=pd.read_pickle(full_path)
    prefix=pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl')
    public=np.arange(11185,18685)
    pd.testing.assert_frame_equal(broadcast.iloc[public].reset_index(drop=True),prefix.iloc[public].reset_index(drop=True),check_exact=True)
    if len(broadcast)!=n:raise ValueError('Full broadcast observations are incomplete')
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    result,events=refine(baseline.label.to_numpy(),broadcast,blocks,13)
    table=pd.DataFrame(dict(frame=np.arange(n),label=result))
    if pm.validate_submission(table,n):raise ValueError('Invalid repaired full candidate')
    for state in ('D','X'):np.testing.assert_array_equal(result==state,baseline.label.to_numpy()==state)
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'candidate.csv',index=False)
    (output/'repair_events.json').write_text(json.dumps(events,indent=2,sort_keys=True)+'\n')
    summary=dict(model='tabpfn_broadcast_short_seed42',seed=42,parent_model='tabpfn_short_control_seed42',
        full_frames=n,changed_frames=int((result!=baseline.label.to_numpy()).sum()),events=len(events),
        candidate_sha256=digest(output/'candidate.csv'),parent_candidate_path=str((PARENT/'candidate.csv').relative_to(ROOT)),
        parent_candidate_sha256=digest(PARENT/'candidate.csv'),parent_manifest_sha256=digest(PARENT/'manifest.json'),
        D_X_exact=True,public_full_broadcast_features_exact=True,full_model_refits=0,public_main_spellF1=.8114285714285714,
        private_full_match_accuracy_unknown=True,all_nine_primary_and_secondary_comparisons_pass=True,
        complete_short_error_resolution=False,remaining_main_short_miss_frames=[11553,11623],
        qualification_scores=all_scores,qualification_plan_sha256=digest(ROUND/'plan.json'),
        input_sha256={str(full_path.relative_to(ROOT)):digest(full_path)},
        source_sha256={p:digest(ROOT/p) for p in ('experiments/broadcast_short_envelope.py','experiments/build_broadcast_short_candidate.py')})
    (output/'manifest.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:summary[k] for k in ('candidate_sha256','full_frames','changed_frames','events','D_X_exact')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    build(parser.parse_args().output)
