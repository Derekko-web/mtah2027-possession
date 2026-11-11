"""Export a frozen second-camera rule using the label-free formation anchors."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_rule_decoder import (GAME, GAMES, ROOT, digest, half_video_blocks, load_source, require,
                              source_frames, validate_submission)
from broadcast_alignment import map_anchors, reconcile_teams, transform
from rule_ball_recovery import recover_possession
from rule_based_decoder import tracking_evidence
from rule_camera_calibration import calibrate
from run_rule_stoppage_gap_candidate import fingerprint as stoppage_gap_fingerprint


REFERENCE = ROOT / 'data/candidate_rule_based_stoppage_gap'
REFERENCE_FREEZE = ROOT / 'data/rule_based_stoppage_gap_freeze.json'
ANCHORS = ROOT / 'data/control_broadcast_v1/full_inputs'
PROFILE = dict(maximum_cost=2.5, minimum_margin=.5, maximum_fine_span=10)


def fingerprint():
    if json.loads(REFERENCE_FREEZE.read_text()) != stoppage_gap_fingerprint():
        raise ValueError('Reference stoppage extension differs from its freeze')
    sources = ('experiments/run_rule_camera_candidate.py', 'experiments/rule_camera_calibration.py',
               'experiments/rule_ball_recovery.py', 'experiments/broadcast_alignment.py',
               'data/rule_camera_calibration_trial.py')
    evidence = ('data/rule_camera_calibration_protocol_v1.json', 'data/external_camera_calibration_development_v1.json',
                'data/external_camera_calibration_holdout_v1.json')
    return dict(alignment=PROFILE, reference_freeze_sha256=digest(REFERENCE_FREEZE),
                source_sha256={name: digest(ROOT / name) for name in sources},
                evidence_sha256={name: digest(ROOT / name) for name in evidence}, public_label_tuning=False)


def aligned_ball(tactical, broadcast, input_hashes, pitch):
    history = json.loads((ANCHORS / 'manifest.json').read_text())
    anchor_path = ANCHORS / 'anchors.npz'
    anchor_hash = digest(anchor_path)
    cameras = {key: value for key, value in input_hashes.items() if key != 'tactical/smart_tagging'}
    if (history['input_sha256'] != cameras or not history['no_target_labels_read']
            or history['cache_sha256']['anchors.npz'] != anchor_hash
            or history['source_sha256']['broadcast_alignment.py'] != digest(ROOT / 'experiments/broadcast_alignment.py')):
        raise ValueError('Formation anchor provenance does not match the supplied tracking')
    broadcast['teams'] = reconcile_teams(tactical['names'], broadcast['names'], broadcast['teams'])
    with np.load(anchor_path, allow_pickle=False) as anchors:
        mapping = map_anchors(anchors, tactical, broadcast, PROFILE)
    frames = mapping['frames']
    orientation = mapping['orientation']
    step = np.diff(frames)
    continuous = ((step >= 0) & (step <= 2) & (orientation[1:] == orientation[:-1])) | (frames[:-1] < 0)
    valid = (frames >= 0) & np.r_[True, continuous]
    donor = np.full_like(tactical['ball'], np.nan)
    for direction in range(4):
        use = valid & (orientation == direction)
        donor[use] = transform(broadcast['ball'][frames[use]], direction) - pitch / 2
    return donor, valid, anchor_hash


def export(freeze_path, output):
    freeze = json.loads(freeze_path.read_text())
    if freeze != fingerprint():
        raise ValueError('Camera rule differs from its freeze')
    input_hashes = {}
    for source in ('tactical', 'broadcast'):
        keys = ('atd_meta', 'atd_raw', 'homography')
        if source == 'tactical':
            keys += ('smart_tagging',)
        input_hashes.update({f'{source}/{key}': digest(path) for key, path in zip(keys, require(keys, source=source))})
    manifest = json.loads((REFERENCE / 'manifest.json').read_text())
    reference_path = REFERENCE / 'candidate.csv'
    reference_hash = digest(reference_path)
    if (manifest['candidate_sha256'] != reference_hash or manifest['ground_truth_used']
            or manifest['freeze_sha256'] != freeze['reference_freeze_sha256']
            or any(input_hashes[f'tactical/{key}'] != value for key, value in manifest['input_sha256'].items())):
        raise ValueError('Reference CSV does not match the frozen stoppage export')
    n = source_frames()
    table = pd.read_csv(reference_path)
    problem = validate_submission(table, n)
    if problem:
        raise ValueError(problem)
    reference = table.label.to_numpy(dtype='U1')
    labels = reference.copy()
    fps = GAMES[GAME]['fps']
    pitch = np.asarray(GAMES[GAME]['pitch'], dtype=np.float32)
    tactical, broadcast = load_source('tactical'), load_source('broadcast')
    donor, aligned, anchors_hash = aligned_ball(tactical, broadcast, input_hashes, pitch)
    xy, ball = tactical['positions'] - pitch / 2, tactical['ball'] - pitch / 2
    xy[~tactical['valid']] = np.nan
    ball[~tactical['valid'] | ~np.all(np.abs(ball) <= pitch / 2 + 5, axis=1)] = np.nan
    halves = []
    for half, start, last in half_video_blocks(GAME):
        block = slice(start, last + 1)
        camera = calibrate(ball[block], donor[block], aligned[block], fps=fps)
        evidence = tracking_evidence(xy[block], tactical['teams'], ball[block], pitch, fps=fps)
        revised, fills = recover_possession(xy[block], tactical['teams'], ball[block], camera, pitch,
                                            evidence, reference[block], variant='broadcast')
        labels[block] = revised
        halves.append(dict(half=half, aligned_frames=int(aligned[block].sum()),
                           usable_camera_observations=int(np.isfinite(camera).all(axis=1).sum()),
                           filled_ball_observations=fills['broadcast'],
                           changed_labels=int((revised != reference[block]).sum())))
    if not np.array_equal(labels == 'D', reference == 'D'):
        raise ValueError('Camera rule changed the reference stoppage mask')
    table['label'] = labels
    problem = validate_submission(table, n)
    if problem:
        raise ValueError(problem)
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / 'candidate.csv'
    table.to_csv(candidate, index=False)
    receipt = dict(game=GAME, frame_count=n, candidate_sha256=digest(candidate), input_sha256=input_hashes,
                   freeze_sha256=digest(freeze_path), reference_sha256=reference_hash,
                   anchor_cache_sha256=anchors_hash, halves=halves, ground_truth_used=False)
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f"Exported {n:,} frames to {candidate} ({receipt['candidate_sha256']})")


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
        print(f'Frozen camera rule at {args.freeze} ({digest(args.freeze)})')
    elif args.frozen_rules and args.output:
        export(args.frozen_rules, args.output)
    else:
        parser.error('Pass --freeze or both --frozen-rules and --output')
