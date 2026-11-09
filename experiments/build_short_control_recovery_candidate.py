"""Export short-control recovery and check it against the selected predictions."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from games import GAME, half_video_blocks
from reproduce_boundary_precision import PROTOCOLS
from reproduce_holdout_extras import pair_checks
from research_holdout_extras import digest, write
from short_control_recovery import recover


DATA = ROOT / 'data'
PARENT = DATA / 'candidate_tabpfn_holdout_repair_v1'
OOF = DATA / 'control_bounded_holdout_repair_v1/candidates/bounded_combined'
TRUTH = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
SCORER = ROOT / 'evaluation/run_evaluation.py'
MODEL = 'tabpfn_short_control_recovery_seed42'


def compare(truth, before, after, blocks):
    table = pd.DataFrame(dict(ground_truth=truth, prediction=before))
    old, new, checks = pair_checks(table, after)
    checks['D_exact'] = np.array_equal(before == 'D', after == 'D')
    folds = []
    for start, end in blocks:
        g = pm.label_codes(truth[start:end])
        a = pm.evaluate(g, pm.label_codes(before[start:end]), bootstrap_n=0)
        b = pm.evaluate(g, pm.label_codes(after[start:end]), bootstrap_n=0)
        folds.append(all(b[k] >= a[k] - 1e-12 for k in ('spell_f1_mean', 'macro_f1', 'accuracy')))
    keys = ('spell_f1_mean', 'macro_f1', 'accuracy', 'spell_f1')
    return dict(before={k: old[k] for k in keys}, after={k: new[k] for k in keys},
                checks=checks, folds_pass=folds, changed_frames=int((before != after).sum()),
                passed=all(checks.values()) and all(folds))


def build(output):
    parent_manifest = json.loads((PARENT / 'manifest.json').read_text())
    parent = PARENT / 'candidate.csv'
    if digest(parent) != parent_manifest['candidate_sha256']:
        raise ValueError('Selected predecessor CSV hash differs')
    truth = pm.read_label_csv(TRUTH).label.to_numpy()
    public = np.flatnonzero(truth != 'X')
    first = int(public[0])
    n = len(truth)
    before = pm.read_label_csv(parent).label.to_numpy()
    raw = pd.read_pickle(DATA / 'control_tabpfn_full_build_v3/full_inputs_v3/raw.pkl')
    broadcast = pd.read_pickle(DATA / 'control_remaining_away_v1/full_inputs/broadcast_extension.pkl')
    cache = DATA / 'candidate_tabpfn_short_control_v1'
    owner = np.load(cache / 'owner_probability.npy', allow_pickle=False)
    fast = np.load(cache / 'fast_probabilities.npy', allow_pickle=False)
    eligible = np.load(DATA / 'control_tabpfn_full_build_v3/encodings/full_eligible.npy', allow_pickle=False)
    blocks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    after, events = recover(before, owner, fast, eligible, raw, broadcast, blocks)
    full = compare(truth[public], before[public], after[public], [])
    raw = raw.iloc[public]
    broadcast = broadcast.iloc[public]
    splits = json.loads((DATA / 'advanced_validation/splits.json').read_text())

    output.mkdir(parents=True, exist_ok=False)
    candidate = output / 'candidate.csv'
    pd.DataFrame(dict(frame=np.arange(n), label=after)).to_csv(candidate, index=False)
    validation = {}
    with TemporaryDirectory() as temporary:
        scorer_inputs = Path(temporary)
        shutil.copyfile(parent, scorer_inputs / 'full_before.csv')
        shutil.copyfile(candidate, scorer_inputs / 'full_after.csv')
        for seed, stage in ((42, 'screen'), (43, 'confirmation_seed43'), (44, 'confirmation_seed44')):
            for protocol in PROTOCOLS:
                baseline = pd.read_csv(OOF / f'seed{seed}' / protocol / 'oof_predictions.csv')
                np.testing.assert_array_equal(baseline.frame, public)
                np.testing.assert_array_equal(baseline.ground_truth, truth[public])
                reference = baseline.prediction.to_numpy()
                blocks = [(s['validation'][0] - first, s['validation'][-1] + 1 - first)
                          for s in splits if s['protocol'] == protocol]
                fitted = DATA / 'control_tabpfn_fixed_confirmation_v2' / stage / protocol
                owner = np.load(fitted / 'owner_probability.npy', allow_pickle=False)
                fast = np.load(DATA / f'control_boundary_v1/baseline_confirmation/{protocol}'
                               / f'fast_seed{seed}_probabilities.npy', allow_pickle=False)
                revised, _ = recover(reference, owner, fast, eligible[public], raw, broadcast, blocks)
                name = f'seed{seed}_{protocol}'
                validation[name] = compare(truth[public], reference, revised, blocks)
                for side, values in (('before', reference), ('after', revised)):
                    labels = np.full(n, 'X')
                    labels[public] = values
                    pd.DataFrame(dict(frame=np.arange(n), label=labels)).to_csv(
                        scorer_inputs / f'{name}_{side}.csv', index=False)

        official = output / 'official'
        subprocess.run([sys.executable, str(SCORER), '--gt', 'public', '--bootstrap', '0',
                        '--results', str(scorer_inputs), '--out', str(official), '--no-detail'], check=True)

    for name, comparison in {'full': full, **validation}.items():
        for side in ('before', 'after'):
            report = json.loads((official / 'reports' / f'{name}_{side}.json').read_text())
            for key in ('spell_f1_mean', 'macro_f1', 'accuracy'):
                np.testing.assert_allclose(report[key], comparison[side][key], rtol=0, atol=1e-12)
            for threshold, score in comparison[side]['spell_f1'].items():
                assert report['spell_f1'][str(threshold)] == score

    subprocess.run([sys.executable, str(ROOT / 'evaluation/validate_submission.py'), str(candidate)], check=True)
    changed = before != after
    passed = full['passed'] and len(validation) == 9 and all(v['passed'] for v in validation.values())
    write(output / 'validation.json', dict(passed=passed, full=full, oof=validation))
    write(output / 'events.json', events)
    write(output / 'manifest.json', dict(model=MODEL, passed=passed,
          candidate_sha256=digest(candidate), parent_candidate_sha256=digest(parent),
          changed_frames=int(changed.sum()), changed_public_frames=int(changed[public].sum()),
          official_score_checks=2 * (len(validation) + 1),
          fold_checks=sum(len(v['folds_pass']) for v in validation.values())))
    if not passed:
        raise SystemExit('Short-control recovery regressed a score or validation check')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    build(parser.parse_args().output.resolve())
