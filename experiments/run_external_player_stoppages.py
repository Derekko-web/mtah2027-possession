"""Evaluate missing-ball stoppages and restart ownership on external matches."""

import argparse
import json
from pathlib import Path
from zlib import crc32

import numpy as np

from run_external_contact_rules import DFL, HALVES, PLAN, ROOT, SKILLCORNER, corrupt, digest, pm
from dfl_submission import skillcorner
from rule_based_decoder import decode_evidence, tracking_evidence
from rule_player_stoppages import VARIANTS, adjust, player_speed
from smart_tag_stoppages import refine


def evaluate_match(paths, dataset, variants):
    predictions = {}
    truth = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            xy, teams, ball, pitch = (data[key] for key in ('positions', 'teams', 'ball', 'pitch'))
            truth.append(data['labels'])
        views = [('observed', xy, ball)]
        if dataset == 'dfl':
            views.append(('masked', *corrupt(xy, ball, crc32(path.stem.encode()))))
        for view, positions, observed in views:
            evidence = tracking_evidence(positions, teams, observed, pitch)
            reference, _ = refine(decode_evidence(evidence), observed, pitch, [])
            speed = player_speed(positions, evidence.fps)
            for variant in variants:
                labels = adjust(reference, evidence, observed, teams, speed, variant)
                predictions.setdefault((view, variant), []).append(labels)

    gt = pm.label_codes(np.concatenate((truth[0], ['X'], truth[1])))
    results = []
    for (view, variant), (first, second) in predictions.items():
        score = pm.evaluate(gt, pm.label_codes(np.concatenate((first, ['X'], second))), bootstrap_n=0)
        results.append(dict(view=view, variant=variant, spell_f1=score['spell_f1_mean'],
                            macro_f1=score['macro_f1'], dead=score['per_class']['D']))
    return results


def run(split, variants, output):
    splits = json.loads(PLAN.read_text())['splits']
    development = split == 'development'
    dfl_matches = splits['train'] + splits['development'] if development else splits['held_out']
    skill_matches = skillcorner.DEVELOPMENT if development else skillcorner.HOLDOUT
    dfl_hashes = json.loads((DFL / 'manifest.json').read_text())['cache_sha256']
    skill_hashes = {name: value[1] for name, value in
                    json.loads((SKILLCORNER / 'manifest.json').read_text())['parsed'].items()}
    groups = [('dfl', match, [DFL / f'{match}_{half}.npz' for half in HALVES], dfl_hashes)
              for match in dfl_matches]
    groups += [('skillcorner', match, [SKILLCORNER / f'{match}_{half}.npz' for half in (1, 2)], skill_hashes)
               for match in skill_matches]
    inputs = {}
    for _, _, paths, expected in groups:
        for path in paths:
            inputs[path.name] = digest(path)
            if inputs[path.name] != expected[path.name]:
                raise ValueError(f'Tracking cache hash differs for {path.name}')

    sources = ('experiments/run_external_player_stoppages.py', 'experiments/rule_player_stoppages.py',
               'experiments/rule_based_decoder.py', 'experiments/smart_tag_stoppages.py',
               'experiments/dfl_transfer_features.py', 'evaluation/possession_metrics.py')
    manifest = dict(split=split, variants=variants, inputs=inputs,
                    source_sha256={name: digest(ROOT / name) for name in sources})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    results = []
    for dataset, match, paths, _ in groups:
        rows = evaluate_match(paths, dataset, variants)
        results.extend(dict(dataset=dataset, match=match, **row) for row in rows)
        summary = ' '.join(f"{row['view']}/{row['variant']}={row['spell_f1']:.4f}" for row in rows)
        print(f'{dataset} {match} {summary}', flush=True)
    (output / 'scores.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=('development', 'held_out'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--finalist', choices=VARIANTS[1:])
    args = parser.parse_args()
    if (args.split == 'held_out') != bool(args.finalist):
        parser.error('--finalist is required only for held-out evaluation')
    run(args.split, ('baseline', args.finalist) if args.finalist else VARIANTS, args.output)
