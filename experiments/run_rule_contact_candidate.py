"""Freeze and export the externally selected contact rule on Barcelona–Athletic."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_rule_decoder import (GAME, GAMES, ROOT, digest, half_video_blocks, load_source, require,
                              source_frames, stoppage_fingerprint, validate_submission)
from rule_based_decoder import tracking_evidence
from rule_contact_variants import MAX_BALL_SPEED, VARIANTS, decode_contacts, reject_ball_spikes


BASE = ROOT / 'data/candidate_rule_based_stoppages_v2'
BASE_FREEZE = ROOT / 'data/rule_based_stoppages_v2_export_freeze.json'
EXTERNAL = ROOT / 'data/external_contact_rules_holdout_v1'
DEVELOPMENT = ROOT / 'data/external_contact_rules_development_v1'
PROTOCOL = ROOT / 'data/external_contact_rules_protocol_v1.json'
FINALIST = 'reject_ball_spikes'


def fingerprint():
    holdout = json.loads((EXTERNAL / 'manifest.json').read_text())
    for name, expected in holdout['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'External validation source changed: {name}')
    if holdout['rules'].get(FINALIST) != asdict(VARIANTS[FINALIST]):
        raise ValueError('Contact rule differs from the external finalist')
    if json.loads(BASE_FREEZE.read_text()) != stoppage_fingerprint():
        raise ValueError('Reference stoppage sources changed')
    sources = ('experiments/run_rule_contact_candidate.py', 'experiments/rule_contact_variants.py')
    evidence = [PROTOCOL, *(directory / name for directory in (DEVELOPMENT, EXTERNAL)
                            for name in ('manifest.json', 'scores.json'))]
    return dict(variant=FINALIST, rules=asdict(VARIANTS[FINALIST]), max_ball_speed=MAX_BALL_SPEED,
                source_sha256={name: digest(ROOT / name) for name in sources},
                evidence_sha256={str(path.relative_to(ROOT)): digest(path) for path in evidence},
                stoppage_freeze_sha256=digest(BASE_FREEZE), public_label_tuning=False)


def export(freeze_path, output):
    if json.loads(freeze_path.read_text()) != fingerprint():
        raise ValueError('Contact rule differs from the freeze')
    reference_manifest = json.loads((BASE / 'manifest.json').read_text())
    keys = ('atd_meta', 'atd_raw', 'homography', 'smart_tagging')
    input_hashes = {key: digest(path) for key, path in zip(keys, require(keys))}
    reference_path = BASE / 'candidate.csv'
    reference_hash = digest(reference_path)
    if (reference_manifest['input_sha256'] != input_hashes
            or reference_manifest['candidate_sha256'] != reference_hash
            or reference_manifest['freeze_sha256'] != digest(BASE_FREEZE)
            or reference_manifest['ground_truth_used']):
        raise ValueError('Reference provenance differs from the frozen v2 export')
    n = source_frames()
    reference = pd.read_csv(reference_path)
    problem = validate_submission(reference, n)
    if problem:
        raise ValueError(problem)
    reference = reference.label.to_numpy(dtype='U1')
    labels = reference.copy()
    fps = GAMES[GAME]['fps']
    pitch = np.asarray(GAMES[GAME]['pitch'], dtype=np.float32)
    tracking = load_source('tactical')
    positions, ball = tracking['positions'] - pitch / 2, tracking['ball'] - pitch / 2
    positions[~tracking['valid']] = np.nan
    ball[~tracking['valid'] | ~np.all(np.abs(ball) <= pitch / 2 + 5, axis=1)] = np.nan

    counts = []
    for half, start, last in half_video_blocks(GAME):
        block = slice(start, last + 1)
        xy, observed = positions[block], ball[block]
        evidence = tracking_evidence(xy, tracking['teams'], observed, pitch, fps=fps)
        decoded = decode_contacts(xy, tracking['teams'], observed, pitch, evidence, FINALIST)
        labels[block] = np.where(reference[block] == 'D', 'D', decoded)
        rejected = np.isfinite(observed).all(axis=1) & ~np.isfinite(reject_ball_spikes(observed, fps)).all(axis=1)
        counts.append(dict(half=half, rejected_observations=int(rejected.sum()),
                           changed_labels=int((labels[block] != reference[block]).sum())))

    if not np.array_equal(labels == 'D', reference == 'D'):
        raise ValueError('Contact rule changed the v2 stoppage mask')
    table = pd.DataFrame({'frame': np.arange(n), 'label': labels})
    problem = validate_submission(table, n)
    if problem:
        raise ValueError(problem)
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / 'candidate.csv'
    table.to_csv(candidate, index=False)
    manifest = dict(game=GAME, freeze_sha256=digest(freeze_path), candidate_sha256=digest(candidate),
                    v2_reference_sha256=reference_hash, input_sha256=input_hashes, frame_count=n,
                    halves=counts, ground_truth_used=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Exported {n:,} frames to {candidate} ({manifest['candidate_sha256']})", flush=True)


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
        print(f'Frozen contact rules at {args.freeze} ({digest(args.freeze)})')
    elif args.frozen_rules and args.output:
        export(args.frozen_rules, args.output)
    else:
        parser.error('Pass --freeze or both --frozen-rules and --output')
