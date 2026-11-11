"""Apply the qualified stoppage-fragment rule to the preserved full candidate."""
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
from supported_gap_stoppage import refine
from research_supported_gap_stoppage import ROUND
from research_spurious_possessions import digest,write,pair_checks
PARENT=ROOT/'data/candidate_tabpfn_home_arrivals_v1'


def build(output):
    plan=json.loads((ROUND/'plan.json').read_text())
    for name,expected in plan['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Qualified source changed: '+name)
    if not json.loads((ROUND/'qualification_result.json').read_text())['passed']:raise ValueError('Qualification failed')
    parent_key=str((PARENT/'candidate.csv').relative_to(ROOT))
    assert digest(PARENT/'candidate.csv')==plan['protected_sha256'][parent_key]
    n=source_frames();parent=pd.read_csv(PARENT/'candidate.csv')
    assert not pm.validate_submission(parent,n)
    inputs=dict(raw=ROOT/'data/control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl',
        broadcast=ROOT/'data/control_remaining_away_v1/full_inputs/broadcast_extension.pkl',
        heads=ROOT/'data/candidate_tabpfn_short_control_v1/heads_probabilities.npy')
    raw=pd.read_pickle(inputs['raw']);bc=pd.read_pickle(inputs['broadcast']);heads=np.load(inputs['heads'],allow_pickle=False)
    grand=json.loads((ROOT/'data/candidate_tabpfn_short_control_v1/manifest.json').read_text())
    assert digest(inputs['heads'])==grand['cache_sha256']['heads_probabilities.npy']
    public=np.arange(11185,18685)
    prefix_raw=pd.read_pickle(ROOT/'data/advanced_validation/raw.pkl')
    prefix_bc=pd.read_pickle(ROOT/'data/control_remaining_away_v1/broadcast_extension_18935.pkl')
    cols=['ball_seen','ball_displacement_1s']
    pd.testing.assert_frame_equal(raw.iloc[public][cols],prefix_raw.iloc[public][cols],check_exact=True)
    pd.testing.assert_frame_equal(bc.iloc[public],prefix_bc.iloc[public],check_exact=True)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    pred,events=refine(parent.label.to_numpy(),heads,raw,bc,blocks)
    table=pd.DataFrame(dict(frame=np.arange(n),label=pred));assert not pm.validate_submission(table,n)
    np.testing.assert_array_equal(pred=='X',parent.label=='X')
    assert np.all(pred[parent.label=='D']=='D')
    truth=pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label
    public_table=pd.DataFrame(dict(ground_truth=truth.iloc[public].to_numpy(),prediction=parent.label.iloc[public].to_numpy()))
    before,after,checks=pair_checks(public_table,pred[public])
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    table.to_csv(output/'candidate.csv',index=False);write(output/'events.json',events)
    write(output/'public_comparison.json',dict(before=before,after=after,checks=checks,passed=all(checks.values())))
    manifest=dict(model='tabpfn_supported_stoppage_fragments_seed42',seed=42,full_frames=n,
        candidate_sha256=digest(output/'candidate.csv'),parent_candidate_path=parent_key,
        parent_candidate_sha256=digest(PARENT/'candidate.csv'),parent_manifest_sha256=digest(PARENT/'manifest.json'),
        changed_frames=int(np.sum(pred!=parent.label)),changed_public_frames=int(np.sum(pred[public]!=parent.label.iloc[public])),
        events=len(events),prior_D_retained=True,X_exact=True,public_observations_exact=True,
        full_public_checks_pass=all(checks.values()),public_OOF_spellF1=dict(main=1.,purged=.8756756756756756,longer=.8210526315789473),
        new_model_fits=0,new_TabPFN_queries=0,private_full_match_accuracy_unknown=True,
        source_sha256={p:digest(ROOT/p) for p in ('experiments/build_stoppage_fragment_candidate.py',*plan['source_sha256'])},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs.values()},
        qualification_sha256=digest(ROUND/'qualification_result.json'))
    write(output/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ('candidate_sha256','changed_frames','changed_public_frames','events','full_public_checks_pass')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);build(p.parse_args().output)
