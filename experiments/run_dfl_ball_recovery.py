"""Screen missing-ball rules using external DFL tracking and simulated second views."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from zlib import crc32

from run_dfl_rule_decoder import DFL, HALVES, PLAN, ROOT, corrupt, digest, pm
import numpy as np

from rule_ball_recovery import RecoveryRules, VARIANTS, recover_possession
from rule_based_decoder import decode_evidence, tracking_evidence


VIEWS = ('clean', 'stress_aligned', 'stress_shifted')


def donor_view(positions, ball, name):
    _, donor = corrupt(positions, ball, crc32(f'{name}:donor'.encode()))
    # Simulate a residual three-frame timing error and a small camera bias.
    shifted = np.full_like(donor, np.nan)
    shifted[3:] = donor[:-3] + (.6, -.4)
    return donor, shifted


def evaluate_match(paths, variants):
    predictions = {(view, variant): [] for view in VIEWS for variant in variants}
    counts = {key: dict(broadcast=0, interpolate=0, changed=0) for key in predictions}
    truth = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            positions, teams, ball, pitch = (data[key] for key in ('positions', 'teams', 'ball', 'pitch'))
            truth.append(data['labels'])
        stress_positions, stress_ball = corrupt(positions, ball, crc32(path.stem.encode()))
        donor, shifted = donor_view(positions, ball, path.stem)
        for view, xy, observation, other in zip(VIEWS, (positions, stress_positions, stress_positions),
                                                 (ball, stress_ball, stress_ball), (donor, donor, shifted)):
            evidence = tracking_evidence(xy, teams, observation, pitch)
            reference = decode_evidence(evidence)
            for variant in variants:
                key = (view, variant)
                labels, added = recover_possession(xy, teams, observation, other, pitch, evidence,
                                                   reference, variant=variant)
                assert np.array_equal(labels == 'D', reference == 'D')
                predictions[key].append(labels)
                for source, amount in added.items():
                    counts[key][source] += amount
                counts[key]['changed'] += int((labels != reference).sum())
    gt = pm.label_codes(np.concatenate((truth[0], ['X'], truth[1])))
    rows = []
    for (view, variant), (first, second) in predictions.items():
        score = pm.evaluate(gt, pm.label_codes(np.concatenate((first, ['X'], second))), bootstrap_n=0)
        rows.append(dict(view=view, variant=variant, spell_f1=score['spell_f1_mean'],
                         macro_f1=score['macro_f1'], predicted_spells=score['n_pred_spells'],
                         **counts[(view, variant)]))
    return rows


def run(split, variants, output):
    splits = json.loads(PLAN.read_text())['splits']
    matches = splits['train'] + splits['development'] if split == 'development' else splits['held_out']
    expected = json.loads((DFL / 'manifest.json').read_text())['cache_sha256']
    inputs = {}
    for match in matches:
        for half in HALVES:
            path = DFL / f'{match}_{half}.npz'
            inputs[path.name] = digest(path)
            if inputs[path.name] != expected[path.name]:
                raise ValueError(f'DFL input hash differs for {path.name}')

    source_paths = ('experiments/run_dfl_ball_recovery.py', 'experiments/run_dfl_rule_decoder.py',
                    'experiments/rule_ball_recovery.py', 'experiments/rule_based_decoder.py',
                    'experiments/dfl_transfer_features.py', 'evaluation/possession_metrics.py')
    manifest = dict(split=split, variants=variants, rules=asdict(RecoveryRules()), matches=matches,
                    source_sha256={name: digest(ROOT / name) for name in source_paths}, inputs=inputs)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    for match in matches:
        rows = evaluate_match([DFL / f'{match}_{half}.npz' for half in HALVES], variants)
        for row in rows:
            results.append(dict(match=match, **row))
            print(f"{match} {row['view']:14} {row['variant']:12} {row['spell_f1']:.5f} "
                  f"({row['changed']:,} label changes)", flush=True)
    (output / 'scores.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('development', 'held_out'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--finalist', choices=tuple(VARIANTS)[1:])
    args = parser.parse_args()
    if (args.split == 'held_out') != bool(args.finalist):
        parser.error('--finalist is required only for held-out evaluation')
    run(args.split, ('baseline', args.finalist) if args.finalist else tuple(VARIANTS), args.output)
