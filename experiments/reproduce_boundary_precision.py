"""Acceptance check against the frozen new-camera possession score floors."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
BASE=ROOT/'data/control_remaining_away_v1/candidates/stable13'
PROTOCOLS=('buffered_minutes','purged_spells','buffered_100_seconds')


def overlap_summary(truth,prediction):
    g,p=pm.label_codes(truth),pm.label_codes(prediction);runs=pm.valid_runs(g)
    gt,pr=pm.possession_spells(g,runs),pm.possession_spells(p,runs)
    overlap=pm.spell_overlaps(gt,pr)
    flags={t:pm.matched_gt_flags(overlap,len(gt),t) for t in pm.SPELL_IOUS}
    best=np.array([max((v for v,i,_ in overlap if i==index),default=0.) for index in range(len(gt))])
    return dict(flags=flags,mean_best_iou=float(best.mean()),gt=gt,predicted=pr,overlap=overlap)


def evaluate(candidate=None):
    rows=[]
    for seed in (42,43,44):
        for protocol in PROTOCOLS:
            before=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            after=before if candidate is None else pd.read_csv(candidate/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(before.frame,after.frame);np.testing.assert_array_equal(before.ground_truth,after.ground_truth)
            gt=pm.label_codes(before.ground_truth)
            a=pm.evaluate(gt,pm.label_codes(before.prediction),bootstrap_n=0)
            b=pm.evaluate(gt,pm.label_codes(after.prediction),bootstrap_n=0)
            old,new=overlap_summary(before.ground_truth,before.prediction),overlap_summary(after.ground_truth,after.prediction)
            checks={key:b[key]>=a[key]-1e-12 for key in ('spell_f1_mean','macro_f1','accuracy')}
            for threshold in pm.SPELL_IOUS:
                checks[f'IoU{threshold}']=b['spell_f1'][threshold]['f1']>=a['spell_f1'][threshold]['f1']-1e-12
                checks[f'prior_matches_{threshold}']=bool(new['flags'][threshold][old['flags'][threshold]].all())
            checks.update(mean_best_iou=new['mean_best_iou']>=old['mean_best_iou']-1e-12,
                D_exact=bool(np.array_equal(before.prediction=='D',after.prediction=='D')),
                X_exact=bool(np.array_equal(before.prediction=='X',after.prediction=='X')))
            rows.append(dict(seed=seed,protocol=protocol,before_spellF1=a['spell_f1_mean'],spellF1=b['spell_f1_mean'],
                before_mean_best_iou=old['mean_best_iou'],mean_best_iou=new['mean_best_iou'],checks=checks))
    checks=dict(all_nine_nonregressing=all(all(r['checks'].values()) for r in rows),
        main_spellF1_improves=rows[0]['spellF1']>rows[0]['before_spellF1']+1e-12,
        main_boundary_overlap_improves=rows[0]['mean_best_iou']>rows[0]['before_mean_best_iou']+1e-12)
    return dict(passed=all(checks.values()),checks=checks,rows=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    r=evaluate(a.candidate)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
    raise SystemExit(0 if r['passed'] else 1)
