"""Acceptance check for the remaining33-/87-frame home interval boundaries."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from reproduce_boundary_precision import overlap_summary,PROTOCOLS
BASE=ROOT/'data/control_guarded_boundaries_v1/candidates/guarded_release_approach25'


def assess(candidate=None):
    rows=[]
    for seed in (42,43,44):
        for protocol in PROTOCOLS:
            a=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            b=a if candidate is None else pd.read_csv(candidate/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(a.frame,b.frame);np.testing.assert_array_equal(a.ground_truth,b.ground_truth)
            gt=pm.label_codes(a.ground_truth);before=pm.evaluate(gt,pm.label_codes(a.prediction),bootstrap_n=0)
            after=pm.evaluate(gt,pm.label_codes(b.prediction),bootstrap_n=0)
            old,new=overlap_summary(a.ground_truth,a.prediction),overlap_summary(b.ground_truth,b.prediction)
            checks={key:after[key]>=before[key]-1e-12 for key in ('spell_f1_mean','macro_f1','accuracy')}
            for t in pm.SPELL_IOUS:
                checks[f'IoU{t}']=after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12
                checks[f'prior_matches_{t}']=bool(new['flags'][t][old['flags'][t]].all())
            checks.update(mean_best_iou=new['mean_best_iou']>=old['mean_best_iou']-1e-12,
                D_exact=bool(np.array_equal(a.prediction=='D',b.prediction=='D')),
                X_exact=bool(np.array_equal(a.prediction=='X',b.prediction=='X')))
            targets={}
            for length in (33,87):
                candidates=[i for i,(s,e,k) in enumerate(old['gt']) if e-s==length and k==pm.H]
                if len(candidates)!=1:raise ValueError('Expected one home diagnostic interval of each length')
                i=candidates[0]
                v0=max((v for v,j,_ in old['overlap'] if j==i),default=0.)
                v1=max((v for v,j,_ in new['overlap'] if j==i),default=0.)
                targets[str(length)]=dict(before_iou=v0,iou=v1)
                checks[f'home{length}_nonregressing']=v1>=v0-1e-12
            rows.append(dict(seed=seed,protocol=protocol,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],targets=targets,checks=checks))
    main=rows[0]
    checks=dict(all_nine_nonregressing=all(all(r['checks'].values()) for r in rows),
        main_spellF1_improves=main['spellF1']>main['before_spellF1']+1e-12,
        both_main_home_intervals_above_IoU70=all(t['iou']>=.7 and t['iou']>t['before_iou']+1e-12 for t in main['targets'].values()))
    return dict(passed=all(checks.values()),checks=checks,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=assess(a.candidate)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2));raise SystemExit(0 if r['passed'] else 1)
