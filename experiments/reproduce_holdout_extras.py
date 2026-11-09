"""Require zero extras in purged/longer holdouts while preserving current scores."""
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
BASE=ROOT/'data/control_supported_gap_stoppage_v1/candidates/supported_gap25'


def pair_checks(table,pred):
    before=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(table.prediction),bootstrap_n=0)
    after=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(pred),bootstrap_n=0)
    old,new=overlap_summary(table.ground_truth,table.prediction),overlap_summary(table.ground_truth,pred)
    checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
    for t in pm.SPELL_IOUS:
        checks[f'IoU{t}']=after['spell_f1'][t]['f1']>=before['spell_f1'][t]['f1']-1e-12
        checks[f'prior_matches_{t}']=bool(new['flags'][t][old['flags'][t]].all())
        checks[f'extras_{t}']=after['spell_f1'][t]['n_spurious']<=before['spell_f1'][t]['n_spurious']
    prior=[];current=[]
    for i in range(len(old['gt'])):
        prior.append(max((v for v,j,_ in old['overlap'] if j==i),default=0.))
        current.append(max((v for v,j,_ in new['overlap'] if j==i),default=0.))
    checks.update(all_true_overlaps_nondecreasing=bool(np.all(np.array(current)>=np.array(prior)-1e-12)),
        X_exact=bool(np.array_equal(table.prediction=='X',pred=='X')),
        no_new_false_D=bool(np.sum((pred=='D')&(table.ground_truth!='D'))<=np.sum((table.prediction=='D')&(table.ground_truth!='D'))),
        dead_recall=after['per_class']['D']['recall']>=before['per_class']['D']['recall']-1e-12)
    return before,after,checks


def assess(candidate=None):
    rows=[]
    for seed in (42,43,44):
        for protocol in PROTOCOLS:
            table=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            proposed=table if candidate is None else pd.read_csv(candidate/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(table.frame,proposed.frame)
            np.testing.assert_array_equal(table.ground_truth,proposed.ground_truth)
            before,after,checks=pair_checks(table,proposed.prediction.to_numpy())
            rows.append(dict(seed=seed,protocol=protocol,checks=checks,before_spellF1=before['spell_f1_mean'],
                spellF1=after['spell_f1_mean'],before_spurious=before['spell_f1'][.5]['n_spurious'],
                spurious=after['spell_f1'][.5]['n_spurious'],
                no_extras_all_thresholds=all(after['spell_f1'][t]['n_spurious']==0 for t in pm.SPELL_IOUS)))
    checks=dict(all_nine_nonregressing=all(all(r['checks'].values()) for r in rows),
        all_holdout_extras_removed=all(r['no_extras_all_thresholds'] for r in rows),
        purged_and_longer_improve=all(rows[i]['spellF1']>rows[i]['before_spellF1']+1e-12 for i in (1,2)),
        all_main_scores_retained=all(rows[i]['spellF1']>=1.-1e-12 for i in (0,3,6)))
    return dict(passed=all(checks.values()),checks=checks,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidate',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=assess(args.candidate)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
