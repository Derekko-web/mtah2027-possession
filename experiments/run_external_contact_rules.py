"""Compare frozen contact rules on DFL and observed SkillCorner tracking."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from zlib import crc32

import numpy as np

from run_dfl_rule_decoder import DFL, HALVES, PLAN, ROOT, corrupt, digest, pm
from dfl_submission import skillcorner
from dfl_submission.robust_features import stress
from rule_based_decoder import tracking_evidence
from rule_contact_variants import MAX_BALL_SPEED, VARIANTS, decode_contacts
from smart_tag_stoppages import refine


SKILLCORNER = ROOT / 'data/skillcorner_external'


def evaluate_match(paths, dataset, variants):
    predictions = {}
    truth = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            xy, teams, ball, pitch = (data[key] for key in ('positions', 'teams', 'ball', 'pitch'))
            truth.append(data['labels'])
        views = [('observed', xy, ball)]
        if dataset == 'dfl':
            seed = crc32(path.stem.encode())
            views += [('masked', *corrupt(xy, ball, seed)), ('severe', *stress(xy, ball, seed))]
        for view, positions, observed in views:
            evidence = tracking_evidence(positions, teams, observed, pitch)
            baseline = decode_contacts(positions, teams, observed, pitch, evidence, 'baseline')
            refined, _ = refine(baseline, observed, pitch, [])
            dead = refined == 'D'
            for variant in variants:
                labels = refined if variant == 'baseline' else decode_contacts(
                    positions, teams, observed, pitch, evidence, variant)
                predictions.setdefault((view, variant), []).append(np.where(dead, 'D', labels))

    gt = pm.label_codes(np.concatenate((truth[0], ['X'], truth[1])))
    rows = []
    for (view, variant), (first, second) in predictions.items():
        score = pm.evaluate(gt, pm.label_codes(np.concatenate((first, ['X'], second))), bootstrap_n=0)
        rows.append(dict(view=view, variant=variant, spell_f1=score['spell_f1_mean'],
                         macro_f1=score['macro_f1'], predicted_spells=score['n_pred_spells'],
                         dead_f1=score['per_class']['D']['f1']))
    return rows


def run(split, variants, output):
    dfl_splits = json.loads(PLAN.read_text())['splits']
    dfl_matches = dfl_splits['train'] + dfl_splits['development'] if split == 'development' else dfl_splits['held_out']
    skill_matches = skillcorner.DEVELOPMENT if split == 'development' else skillcorner.HOLDOUT
    dfl_hashes = json.loads((DFL / 'manifest.json').read_text())['cache_sha256']
    skill_hashes = json.loads((SKILLCORNER / 'manifest.json').read_text())['parsed']
    groups = [('dfl', match, [DFL / f'{match}_{half}.npz' for half in HALVES], dfl_hashes)
              for match in dfl_matches]
    groups += [('skillcorner', match, [SKILLCORNER / f'{match}_{half}.npz' for half in (1, 2)],
                {name: identity[1] for name, identity in skill_hashes.items()}) for match in skill_matches]
    inputs = {}
    for _, _, paths, expected in groups:
        for path in paths:
            inputs[path.name] = digest(path)
            if inputs[path.name] != expected[path.name]:
                raise ValueError(f'Tracking cache hash differs for {path.name}')

    source_paths = ('experiments/run_external_contact_rules.py', 'experiments/rule_contact_variants.py',
                    'experiments/rule_based_decoder.py', 'experiments/smart_tag_stoppages.py',
                    'experiments/dfl_transfer_features.py', 'dfl_submission/robust_features.py',
                    'evaluation/possession_metrics.py')
    manifest = dict(split=split, rules={name: asdict(VARIANTS[name]) for name in variants},
                    max_ball_speed=MAX_BALL_SPEED, inputs=inputs,
                    source_sha256={name: digest(ROOT / name) for name in source_paths})
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
    parser.add_argument('--finalist', choices=tuple(VARIANTS)[1:])
    args = parser.parse_args()
    if (args.split == 'held_out') != bool(args.finalist):
        parser.error('--finalist is required only for held-out evaluation')
    run(args.split, ('baseline', args.finalist) if args.finalist else tuple(VARIANTS), args.output)
