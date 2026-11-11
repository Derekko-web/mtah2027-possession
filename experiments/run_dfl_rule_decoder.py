"""Evaluate the tracking-only rules on the project's fixed DFL match splits."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from zlib import crc32

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dfl_transfer_features import corrupt
from evaluation import possession_metrics as pm
from rule_based_decoder import Rules, decode_evidence, runs, stoppages, tracking_evidence
from starter.baseline import smooth


PLAN = ROOT / 'experiments/DFL_TRANSFER_FEASIBILITY_PLAN.json'
DFL = ROOT / 'data/control_dfl_transfer_v1/assessment_A'
HALVES = ('firstHalf', 'secondHalf')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def nearest_with_stoppages(nearest, dead):
    live_runs = list(runs(~dead))
    labels = np.full(len(nearest), 'D')
    for start, end in live_runs:
        live = smooth(nearest[start:end], 50)
        labels[start:end] = np.where(live == 'D', 'X', live)
    # Resolve gaps within live runs without borrowing ownership across a stop.
    return np.array(pm.LABELS)[pm.fill_abstentions(pm.label_codes(labels), live_runs)]


def evaluate_match(paths, view, rules):
    truth = []
    predictions = {'nearest_50': [], 'nearest_50_with_stoppages': [], 'rule_decoder': []}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            positions, teams, ball, pitch = (data[key] for key in ('positions', 'teams', 'ball', 'pitch'))
            truth.append(data['labels'])
        if view == 'stress':
            positions, ball = corrupt(positions, ball, crc32(path.stem.encode()))
        distances = np.linalg.norm(positions - ball[:, None, :], axis=2)
        distances[~np.isfinite(distances)] = np.inf
        nearest = np.where(np.isfinite(distances.min(axis=1)), teams[distances.argmin(axis=1)], -1)
        baseline = smooth(nearest, 50)
        evidence = tracking_evidence(positions, teams, ball, pitch)
        predictions['nearest_50'].append(baseline)
        predictions['nearest_50_with_stoppages'].append(nearest_with_stoppages(nearest, stoppages(evidence, rules)))
        predictions['rule_decoder'].append(decode_evidence(evidence, rules))

    # Use an ungraded separator to prevent possessions or transitions spanning halftime.
    gt = pm.label_codes(np.concatenate((truth[0], ['X'], truth[1])))
    return {name: pm.evaluate(gt, pm.label_codes(np.concatenate((first, ['X'], second))), bootstrap_n=0)
            for name, (first, second) in predictions.items()}


def run(input_dir, output, split):
    splits = json.loads(PLAN.read_text())['splits']
    expected = json.loads((input_dir / 'manifest.json').read_text())['cache_sha256']
    sources = [Path(__file__).resolve(), PLAN, ROOT / 'experiments/rule_based_decoder.py',
               ROOT / 'experiments/dfl_transfer_features.py', ROOT / 'starter/baseline.py',
               ROOT / 'evaluation/possession_metrics.py']
    rules = Rules()
    input_hashes = {}
    for match in splits[split]:
        for half in HALVES:
            file = input_dir / f'{match}_{half}.npz'
            actual = digest(file)
            if expected.get(file.name) != actual:
                raise ValueError(f'DFL input hash differs for {file.name}')
            input_hashes[file.name] = actual

    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(split=split, matches=splits[split], rules=asdict(rules), fps=25,
                    inputs=input_hashes, source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
                    stress='DFL tracking corruption, seeded by CRC32 of the half file stem')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    details = {}
    rows = []
    for match in splits[split]:
        paths = [input_dir / f'{match}_{half}.npz' for half in HALVES]
        details[match] = {}
        for view in ('clean', 'stress'):
            scores = evaluate_match(paths, view, rules)
            details[match][view] = scores
            for model, score in scores.items():
                row = dict(match=match, view=view, model=model, spell_f1=score['spell_f1_mean'],
                           macro_f1=score['macro_f1'], dead_precision=score['per_class']['D']['precision'],
                           dead_recall=score['per_class']['D']['recall'], true_spells=score['n_gt_spells'],
                           predicted_spells=score['n_pred_spells'])
                rows.append(row)
                print(f"{match} {view:6} {model:26} SpellF1 {row['spell_f1']:.4f} "
                      f"MacroF1 {row['macro_f1']:.4f} D recall {row['dead_recall']:.4f}", flush=True)
    (output / 'metrics.json').write_text(json.dumps(details, indent=2) + '\n')
    with (output / 'scores.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DFL)
    parser.add_argument('--split', choices=('train', 'development', 'held_out'), default='held_out')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output, args.split)
