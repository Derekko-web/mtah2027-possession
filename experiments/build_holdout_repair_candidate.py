"""Export the qualified physical/continuity rule over the current full recipe."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from games import GAME,half_video_blocks,source_frames
from bounded_holdout_repair import refine
from research_bounded_holdout_repair import ROUND
from research_holdout_extras import digest,write
from reproduce_holdout_extras import pair_checks
PARENT=ROOT/'data/candidate_tabpfn_stoppage_fragments_v1'


def build(output):
    plan=json.loads((ROUND/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        assert digest(ROOT/name)==expected,name
    assert json.loads((ROUND/'qualification_result.json').read_text())['passed']
    assert json.loads((ROUND/'selection.json').read_text())['selected']=='bounded_combined'
    parent_key=str((PARENT/'candidate.csv').relative_to(ROOT))
    assert digest(PARENT/'candidate.csv')==plan['protected_sha256'][parent_key]
    parent=pd.read_csv(PARENT/'candidate.csv');n=source_frames();assert not pm.validate_submission(parent,n)
    inputs=dict(raw=ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl',
        broadcast=ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl',
        heads=ROOT/'data/candidate_tabpfn_short_control_v1/heads_probabilities.npy',
        owner=ROOT/'data/candidate_tabpfn_short_control_v1/owner_probability.npy',
        fast=ROOT/'data/candidate_tabpfn_short_control_v1/fast_probabilities.npy')
    raw=pd.read_pickle(inputs['raw']);bc=pd.read_pickle(inputs['broadcast'])
    heads,owner,fast=[np.load(inputs[k],allow_pickle=False) for k in ('heads','owner','fast')]
    grand=json.loads((ROOT/'data/candidate_tabpfn_short_control_v1/manifest.json').read_text())
    for k in ('heads','owner','fast'):assert digest(inputs[k])==grand['cache_sha256'][inputs[k].name]
    public=np.arange(11185,18685)
    prefix_raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl')
    prefix_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl')
    cols=['ball_seen','ball_displacement_1s','ball_goal_line_distance','ball_touchline_distance','h__distance_1','a__distance_1']
    pd.testing.assert_frame_equal(raw.iloc[public][cols],prefix_raw.iloc[public][cols],check_exact=True)
    pd.testing.assert_frame_equal(bc.iloc[public],prefix_bc.iloc[public],check_exact=True)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    pred,events=refine(parent.label.to_numpy(),heads,owner,fast,raw,bc,blocks)
    table=pd.DataFrame(dict(frame=np.arange(n),label=pred));assert not pm.validate_submission(table,n)
    truth=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label
    public_table=pd.DataFrame(dict(ground_truth=truth.iloc[public].to_numpy(),prediction=parent.label.iloc[public].to_numpy()))
    before,after,checks=pair_checks(public_table,pred[public])
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'candidate.csv',index=False);write(output/'events.json',events)
    write(output/'public_comparison.json',dict(before=before,after=after,checks=checks,passed=all(checks.values())))
    changed=pred!=parent.label.to_numpy()
    manifest=dict(model='tabpfn_physical_continuity_repair_seed42',seed=42,full_frames=n,
        candidate_sha256=digest(output/'candidate.csv'),parent_candidate_path=parent_key,
        parent_candidate_sha256=digest(PARENT/'candidate.csv'),parent_manifest_sha256=digest(PARENT/'manifest.json'),
        changed_frames=int(changed.sum()),changed_public_frames=int(changed[public].sum()),events=len(events),
        X_exact=bool(np.array_equal(pred=='X',parent.label=='X')),public_observations_exact=True,
        full_public_checks_pass=all(checks.values()),all_nine_OOF_spellF1=1.,all_nine_OOF_spurious=0,
        new_model_fits=0,new_TabPFN_queries=0,private_full_match_accuracy_unknown=True,
        source_sha256={p:digest(ROOT/p) for p in ('experiments/build_holdout_repair_candidate.py',*plan['source_sha256'])},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs.values()},
        qualification_sha256=digest(ROUND/'qualification_result.json'))
    write(output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('candidate_sha256','changed_frames','changed_public_frames','events','full_public_checks_pass')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
