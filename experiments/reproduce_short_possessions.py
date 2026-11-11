"""Fast acceptance check for short-control repairs against selected TabPFN.

With no candidate directory this runs the selected predictions as a red control.
Candidate directories contain seed42/seed43/seed44/<protocol>/oof_predictions.csv
with frame,ground_truth,prediction columns. No model is fitted by this check.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm

BASE = ROOT / 'data/control_tabpfn_fixed_confirmation_v2'
PROTOCOLS = ('buffered_minutes', 'purged_spells', 'buffered_100_seconds')


def assess(candidate=None):
    rows = []
    all_nonregressing = True
    for seed, stage in ((42, 'screen'), (43, 'confirmation_seed43'), (44, 'confirmation_seed44')):
        for protocol in PROTOCOLS:
            reference = pd.read_csv(BASE / stage / protocol / 'oof_predictions.csv')
            old = reference[f'tabpfn_short_control_seed{seed}'].to_numpy()
            if candidate is None:
                new = old.copy()
            else:
                table = pd.read_csv(candidate / f'seed{seed}' / protocol / 'oof_predictions.csv')
                np.testing.assert_array_equal(reference.frame, table.frame)
                np.testing.assert_array_equal(reference.ground_truth, table.ground_truth)
                new = table.prediction.to_numpy()
            truth = pm.label_codes(reference.ground_truth)
            before = pm.evaluate(truth, pm.label_codes(old), bootstrap_n=0)
            after = pm.evaluate(truth, pm.label_codes(new), bootstrap_n=0)
            runs = pm.valid_runs(truth)
            spells = pm.possession_spells(truth, runs)
            overlaps = pm.spell_overlaps(spells, pm.possession_spells(pm.label_codes(new), runs))
            short = [i for i, (a, b, _) in enumerate(spells) if b-a <= 125]
            flags = pm.matched_gt_flags(overlaps, len(spells), .5)
            short_matched = int(flags[short].sum())
            checks = dict(spellF1=after['spell_f1_mean'] >= before['spell_f1_mean']-1e-12,
                          macroF1=after['macro_f1'] >= before['macro_f1']-1e-12,
                          accuracy=after['accuracy'] >= before['accuracy']-1e-12,
                          D_exact=bool(np.array_equal(old == 'D', new == 'D')),
                          X_exact=bool(np.array_equal(old == 'X', new == 'X')))
            all_nonregressing &= all(checks.values())
            rows.append(dict(seed=seed, protocol=protocol,
                             before_spellF1=before['spell_f1_mean'], after_spellF1=after['spell_f1_mean'],
                             before_macroF1=before['macro_f1'], after_macroF1=after['macro_f1'],
                             short_matched_iou50=short_matched, short_total=len(short), checks=checks))
    main = rows[0]
    checks = dict(all_nine_scores_nonregressing=bool(all_nonregressing),
                  main_spellF1_improves=main['after_spellF1'] > main['before_spellF1']+1e-12,
                  all_main_short_spells_recovered=main['short_matched_iou50'] == main['short_total'])
    return dict(passed=all(checks.values()), checks=checks, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = assess(args.candidate)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], checks=result['checks'],
                         main_short_matches=f"{result['rows'][0]['short_matched_iou50']}/{result['rows'][0]['short_total']}"), indent=2))
    raise SystemExit(0 if result['passed'] else 1)
