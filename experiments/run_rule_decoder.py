"""Export the DFL-frozen rule decoder on the supplied tactical tracking."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from broadcast_alignment import load_source
from evaluation.possession_metrics import validate_submission
from games import GAME, GAMES, half_video_blocks, require, source_frames
from rule_based_decoder import Rules, decode
from smart_tag_stoppages import RESTARTS, StoppageRules, refine, restart_windows
from starter.load_smart_tagging import load as load_tags


# Recorded before DFL held-out scoring in data/dfl_rule_decoder_v1/manifest.json.
FROZEN_DECODER = '75f6ac7989f5f63752cf594d0903a2f6330b83a392df363334740dceb9f75db7'


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def stoppage_fingerprint():
    sources = ('experiments/run_rule_decoder.py', 'experiments/rule_based_decoder.py',
               'experiments/smart_tag_stoppages.py', 'experiments/test_smart_tag_stoppages.py',
               'experiments/broadcast_alignment.py', 'starter/load_smart_tagging.py',
               'starter/baseline.py', 'evaluation/possession_metrics.py', 'games.py')
    return dict(model='rule_decoder_smart_tag_stoppages_v2',
                source_sha256={name: digest(ROOT / name) for name in sources},
                base_rules=asdict(Rules()), stoppage_rules=asdict(StoppageRules()),
                restart_windows={kind: list(values) for kind, values in RESTARTS.items()},
                design_inputs=['Published Smart Tagging schema and timing semantics',
                               'Pitch geometry and general restart behaviour', 'Synthetic tests'],
                public_label_tuning=False)


def export(output, stoppage_freeze=None):
    if digest(Path(__file__).with_name('rule_based_decoder.py')) != FROZEN_DECODER:
        raise ValueError('Decoder differs from the DFL-frozen source')
    freeze = None
    if stoppage_freeze is not None:
        freeze = json.loads(stoppage_freeze.read_text())
        if freeze != stoppage_fingerprint():
            raise ValueError('Stoppage rules differ from the recorded freeze')
    rules = Rules()
    fps = GAMES[GAME]['fps']
    pitch = np.asarray(GAMES[GAME]['pitch'], dtype=np.float32)
    tracks = load_source('tactical')
    # Match the DFL adapter's centred metres and five-metre observation margin.
    positions = tracks['positions'] - pitch / 2
    ball = tracks['ball'] - pitch / 2
    positions[~tracks['valid']] = np.nan
    ball[~tracks['valid'] | ~np.all(np.abs(ball) <= pitch / 2 + 5, axis=1)] = np.nan

    n = source_frames()
    labels = np.full(n, 'X')
    blocks = [(start, last + 1) for _, start, last in half_video_blocks(GAME)]
    tags = load_tags(fps=fps) if freeze else []
    events = []
    added_dead_frames = 0
    for half, (start, end) in enumerate(blocks, 1):
        base = decode(positions[start:end], tracks['teams'], ball[start:end], pitch, fps=fps, rules=rules)
        if freeze:
            revised, proposed = refine(base, ball[start:end], pitch, restart_windows(tags, start, end), fps=fps)
            added_dead_frames += int(np.count_nonzero(revised != base))
            events.extend(dict(event, half=half, start=start + int(event['start']),
                               end_exclusive=start + int(event['end_exclusive'])) for event in proposed)
            labels[start:end] = revised
        else:
            labels[start:end] = base
    table = pd.DataFrame({'frame': np.arange(n), 'label': labels})
    problem = validate_submission(table, n)
    if problem:
        raise ValueError(problem)

    output.mkdir(parents=True, exist_ok=False)
    candidate = output / 'candidate.csv'
    table.to_csv(candidate, index=False)
    keys = ('atd_meta', 'atd_raw', 'homography')
    if freeze:
        keys += ('smart_tagging',)
    inputs = {key: digest(path) for key, path in zip(keys, require(keys))}
    manifest = dict(game=GAME, camera='tactical', rules=asdict(rules), decoder_sha256=FROZEN_DECODER,
                    fps=fps, pitch=pitch.tolist(), half_blocks=blocks, input_sha256=inputs,
                    candidate_sha256=digest(candidate), ground_truth_used=False)
    if freeze:
        manifest.update(freeze_sha256=digest(stoppage_freeze), added_dead_frames=added_dead_frames)
        (output / 'events.json').write_text(json.dumps(events, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f"Exported {n:,} frames to {candidate} ({manifest['candidate_sha256']})", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze-stoppages', type=Path, help='Record the rules without loading match data')
    mode.add_argument('--stoppage-freeze', type=Path, help='Apply the verified Smart Tagging stoppage rules')
    args = parser.parse_args()
    if args.freeze_stoppages:
        if args.output:
            parser.error('--output is used for exports only')
        with args.freeze_stoppages.open('x') as file:
            json.dump(stoppage_fingerprint(), file, indent=2)
            file.write('\n')
        print(f'Frozen stoppage rules at {args.freeze_stoppages} ({digest(args.freeze_stoppages)})')
    else:
        if not args.output:
            parser.error('--output is required for exports')
        export(args.output, args.stoppage_freeze)
