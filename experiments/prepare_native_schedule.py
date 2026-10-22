"""Regenerate the predeclared, unconsumed confirmation schedule from local metadata."""

from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import weak_eligibility
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from finalize_ball_appearance import audit_manifest
from games import GAME, half_video_blocks
from starter.load_homography import load


def main():
    root = ROOT/'data/control_native_ball_v1'
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT/'experiments/NATIVE_BALL_PLAN.json').read_text())
    audit_manifest(ROOT/'data/control_ball_appearance_v1/feasibility')
    source = load_source('tactical')
    _, error, valid = load()
    eligible = weak_eligibility(source, error, valid)
    inside = np.zeros(len(eligible), bool)
    for _, start, end in half_video_blocks(GAME):
        inside[start:end+1] = True
    eligible &= inside & (np.arange(len(eligible)) % 25 == 0)
    eligible[10935:18935] = False
    frozen = ROOT/'data/continuation_20261021/annotation_freeze/windows.json'
    if digest(frozen) != '5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c':
        raise ValueError('Reserved annotation metadata changed')
    for window in json.loads(frozen.read_text())['windows']:
        eligible[max(0, window['start']-250):min(len(eligible), window['end']+251)] = False
    old = np.load(ROOT/'data/control_ball_appearance_v1/feasibility/weak_ball_cache.npz')['frames']
    for frame in old:
        eligible[max(0, frame-250):min(len(eligible), frame+251)] = False
    candidates = np.flatnonzero(eligible)
    frames = np.sort(np.random.default_rng(43).choice(candidates, 60, replace=False))
    buffer = BytesIO()
    np.save(buffer, frames)
    path = root/'confirmation_frames.npy'
    if path.exists() and path.read_bytes() != buffer.getvalue():
        raise ValueError('Preserved native confirmation schedule differs')
    if not path.exists():
        path.write_bytes(buffer.getvalue())
    if digest(path) != plan['confirmation_frames_sha256']:
        raise ValueError('Regenerated confirmation schedule differs from frozen hash')
    print('Native confirmation schedule regenerated exactly;no images decoded;candidate centers', len(candidates))


if __name__ == '__main__':
    main()
