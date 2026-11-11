"""Export the frozen missing-ball stoppage extension on the competition match."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rule_stoppage_gaps import extend_stoppages
from run_rule_contact_candidate import BASE, BASE_FREEZE
from run_rule_decoder import (GAME, GAMES, ROOT, digest, half_video_blocks, load_source, require,
                              source_frames, stoppage_fingerprint, validate_submission)


FINALIST = ROOT / 'data/external_gap_stops_finalist_v1.json'


def fingerprint():
    if json.loads(BASE_FREEZE.read_text()) != stoppage_fingerprint():
        raise ValueError('The reference stoppage rules differ from their freeze')
    source_files = ('experiments/run_rule_stoppage_gap_candidate.py', 'experiments/rule_stoppage_gaps.py',
                    'experiments/run_rule_contact_candidate.py', 'data/rule_stoppage_gap_trial.py')
    evidence_files = (FINALIST, ROOT / 'data/external_gap_stops_development_v1.json',
                      ROOT / 'data/external_gap_stops_holdout_v1.json')
    seconds = json.loads(FINALIST.read_text())['finalist_seconds']
    return dict(max_extension_seconds=seconds, v2_freeze_sha256=digest(BASE_FREEZE),
                sources={name: digest(ROOT / name) for name in source_files},
                external_evidence={str(p.relative_to(ROOT)): digest(p) for p in evidence_files},
                public_label_tuning=False)


def export(freeze_path, output):
    freeze = json.loads(freeze_path.read_text())
    if freeze != fingerprint():
        raise ValueError('Stoppage extension differs from its freeze')

    keys = ('atd_meta', 'atd_raw', 'homography', 'smart_tagging')
    input_hashes = {key: digest(path) for key, path in zip(keys, require(keys))}
    reference_manifest = json.loads((BASE / 'manifest.json').read_text())
    reference_path = BASE / 'candidate.csv'
    reference_hash = digest(reference_path)
    if (reference_manifest['input_sha256'] != input_hashes
            or reference_manifest['candidate_sha256'] != reference_hash
            or reference_manifest['freeze_sha256'] != freeze['v2_freeze_sha256']
            or reference_manifest['ground_truth_used']):
        raise ValueError('Reference differs from the frozen v2 export')

    n = source_frames()
    table = pd.read_csv(reference_path)
    problem = validate_submission(table, n)
    if problem:
        raise ValueError(problem)
    reference = table.label.to_numpy(dtype='U1')
    labels = reference.copy()
    tracking = load_source('tactical')
    pitch = np.asarray(GAMES[GAME]['pitch'], dtype=np.float32)
    ball = tracking['ball'] - pitch / 2
    ball[~tracking['valid'] | ~np.all(np.abs(ball) <= pitch / 2 + 5, axis=1)] = np.nan
    counts = []
    for half, start, last in half_video_blocks(GAME):
        block = slice(start, last + 1)
        labels[block] = extend_stoppages(reference[block], ball[block], freeze['max_extension_seconds'],
                                          fps=GAMES[GAME]['fps'])
        counts.append(dict(half=half, added_dead_frames=int((labels[block] != reference[block]).sum())))
    changed = labels != reference
    if not np.all(labels[changed] == 'D') or np.isfinite(ball[changed]).all(axis=1).any():
        raise ValueError('Extension changed labels outside missing-ball observations')

    table['label'] = labels
    problem = validate_submission(table, n)
    if problem:
        raise ValueError(problem)
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / 'candidate.csv'
    table.to_csv(candidate, index=False)
    manifest = dict(game=GAME, frame_count=n, candidate_sha256=digest(candidate),
                    freeze_sha256=digest(freeze_path), v2_reference_sha256=reference_hash,
                    input_sha256=input_hashes, halves=counts, ground_truth_used=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Exported {n:,} frames to {candidate} ({manifest['candidate_sha256']})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', type=Path)
    mode.add_argument('--frozen-rules', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.freeze and not args.output:
        with args.freeze.open('x') as file:
            json.dump(fingerprint(), file, indent=2)
            file.write('\n')
        print(f'Frozen stoppage extension at {args.freeze} ({digest(args.freeze)})')
    elif args.frozen_rules and args.output:
        export(args.frozen_rules, args.output)
    else:
        parser.error('Pass --freeze or both --frozen-rules and --output')
