"""Require removal of main extra spells without losing the current scores."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from reproduce_boundary_precision import PROTOCOLS,overlap_summary
BASE=ROOT/'data/control_leading_gap_boundaries_v1/candidates/leading_camera13'


def assess(candidate=None):
    rows=[]
    for seed in (42,43,44):
        for protocol in PROTOCOLS:
            a=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            b=a if candidate is None else pd.read_csv(candidate/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(a.frame,b.frame)
            np.testing.assert_array_equal(a.ground_truth,b.ground_truth)
            truth=pm.label_codes(a.ground_truth)
            before=pm.evaluate(truth,pm.label_codes(a.prediction),bootstrap_n=0)
            after=pm.evaluate(truth,pm.label_codes(b.prediction),bootstrap_n=0)
            old,new=overlap_summary(a.ground_truth,a.prediction),overlap_summary(b.ground_truth,b.prediction)
            checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
            for t in pm.SPELL_IOUS:
                checks[f'IoU{t}']=after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12
                checks[f'prior_matches_{t}']=bool(new['flags'][t][old['flags'][t]].all())
                checks[f'extras_{t}']=after['spell_f1'][t]['n_spurious']<=before['spell_f1'][t]['n_spurious']
            old_false_dead=int(np.sum((a.prediction=='D')&(a.ground_truth!='D')))
            new_false_dead=int(np.sum((b.prediction=='D')&(b.ground_truth!='D')))
            checks.update(mean_best_iou=new['mean_best_iou']>=old['mean_best_iou']-1e-12,
                prior_D_retained=bool((b.prediction[a.prediction=='D']=='D').all()),
                no_new_false_D=new_false_dead<=old_false_dead,
                X_exact=bool(np.array_equal(a.prediction=='X',b.prediction=='X')))
            targets={}
            for length in (33,87):
                index=next(i for i,(s,e,k) in enumerate(old['gt']) if e-s==length and k==pm.H)
                v0=max((v for v,j,_ in old['overlap'] if j==index),default=0.)
                v1=max((v for v,j,_ in new['overlap'] if j==index),default=0.)
                targets[str(length)]=dict(before_iou=v0,iou=v1)
                checks[f'home{length}']=v1>=v0-1e-12
            rows.append(dict(seed=seed,protocol=protocol,before_spellF1=before['spell_f1_mean'],
                spellF1=after['spell_f1_mean'],before_spurious=before['spell_f1'][.5]['n_spurious'],
                spurious=after['spell_f1'][.5]['n_spurious'],targets=targets,checks=checks,
                all_thresholds_no_spurious=all(after['spell_f1'][t]['n_spurious']==0 for t in pm.SPELL_IOUS)))
    checks=dict(all_nine_nonregressing=all(all(r['checks'].values()) for r in rows),
        main_spellF1_improves=rows[0]['spellF1']>rows[0]['before_spellF1']+1e-12,
        main_three_spurious_removed=rows[0]['all_thresholds_no_spurious'])
    return dict(passed=all(checks.values()),checks=checks,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=assess(a.candidate)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
