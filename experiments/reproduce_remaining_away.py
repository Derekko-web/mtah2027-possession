"""Acceptance check against the selected broadcast-refined TabPFN scores."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm

BASE=ROOT/'data/control_short_continuation_v1/candidates/broadcast13'
PROTOCOLS=('buffered_minutes','purged_spells','buffered_100_seconds')


def evaluate(candidate=None):
    rows=[]
    for seed in (42,43,44):
        for protocol in PROTOCOLS:
            old=pd.read_csv(BASE/f'seed{seed}'/protocol/'oof_predictions.csv')
            new=old if candidate is None else pd.read_csv(candidate/f'seed{seed}'/protocol/'oof_predictions.csv')
            np.testing.assert_array_equal(new.frame,old.frame)
            np.testing.assert_array_equal(new.ground_truth,old.ground_truth)
            truth=pm.label_codes(old.ground_truth);runs=pm.valid_runs(truth);spells=pm.possession_spells(truth,runs)
            before=pm.evaluate(truth,pm.label_codes(old.prediction),bootstrap_n=0)
            after=pm.evaluate(truth,pm.label_codes(new.prediction),bootstrap_n=0)
            overlaps=pm.spell_overlaps(spells,pm.possession_spells(pm.label_codes(new.prediction),runs))
            previous=pm.spell_overlaps(spells,pm.possession_spells(pm.label_codes(old.prediction),runs))
            flags=pm.matched_gt_flags(overlaps,len(spells),.5)
            prior_flags=pm.matched_gt_flags(previous,len(spells),.5)
            target=[i for i,(a,b,k) in enumerate(spells) if b-a==71 and k==pm.A]
            if len(target)!=1:raise ValueError('Expected one71-frame away diagnostic target')
            checks={k:after[k]>=before[k]-1e-12 for k in ('spell_f1_mean','macro_f1','accuracy')}
            checks.update(D_exact=bool(np.array_equal(old.prediction=='D',new.prediction=='D')),
                          X_exact=bool(np.array_equal(old.prediction=='X',new.prediction=='X')),
                          previous_matches_retained=bool(flags[prior_flags].all()),
                          remaining_away_matched=bool(flags[target[0]]))
            rows.append(dict(seed=seed,protocol=protocol,before_spellF1=before['spell_f1_mean'],spellF1=after['spell_f1_mean'],checks=checks))
    return dict(passed=all(all(r['checks'].values()) for r in rows),rows=rows,
                all_nine_nonregressing=all(all(v for k,v in r['checks'].items() if k!='remaining_away_matched') for r in rows),
                target_recovered_pairs=sum(r['checks']['remaining_away_matched'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    result=evaluate(a.candidate)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    raise SystemExit(0 if result['passed'] else 1)
