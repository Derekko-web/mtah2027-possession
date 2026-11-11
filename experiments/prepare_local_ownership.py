"""Automatic weak jersey references and uniform local temporal scenes; no targets."""
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME, half_video_blocks, require
from jersey_color import decode_frame, torso_bounds, WIDTH, HEIGHT, PITCH
from local_ownership import allowed_frame, sample_centers, require_model
from starter.load_homography import load, project


def scene(frame, source, matrix):
    choices = np.arange(max(0, frame-25), min(len(source['ball']), frame+26))
    choices = choices[np.isfinite(source['ball'][choices]).all(axis=1)]
    if not len(choices):
        return None
    nearest = int(choices[np.argmin(np.abs(choices-frame))])
    center = source['ball'][nearest]
    corners = center+np.array([[-8, -8], [8, -8], [8, 8], [-8, 8]])
    px, py = project(matrix, corners[:, 0]/PITCH[0], corners[:, 1]/PITCH[1])
    if not np.isfinite(np.r_[px, py]).all():
        return None
    left, right = np.clip([px.min()/2, px.max()/2], 0, WIDTH).astype(int)
    top, bottom = np.clip([py.min()/2-20, py.max()/2+10], 0, HEIGHT).astype(int)
    if right-left < 40 or bottom-top < 40:
        return None
    image = Image.fromarray(decode_frame(frame)).crop((left, top, right, bottom))
    image.thumbnail((768, 768))
    return image, dict(frame=frame, ball_source_frame=nearest, bounds=[int(v) for v in (left, top, right, bottom)])


def main():
    root = ROOT/'data/control_local_ownership_v1'
    plan_path = ROOT/'experiments/LOCAL_OWNERSHIP_PLAN.json'
    plan = json.loads(plan_path.read_text())
    require_model(plan)
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen ownership probe parent changed')
    parent = ROOT/'data/control_jersey_v1/feasibility'
    manifest = json.loads((parent/'manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if digest(ROOT/'experiments'/name) != expected:
            raise ValueError('Weak reference producer changed')
    for identity, expected in manifest['input_sha256'].items():
        camera, asset = identity.split('/')
        if digest(require([asset], source=camera)[0]) != expected:
            raise ValueError('Weak reference source changed')
    for name, expected in manifest['cache_sha256'].items():
        if digest(parent/name) != expected:
            raise ValueError('Weak reference artifact changed')
    audit_manifest(ROOT/'data/control_broadcast_v1/inputs')
    reserved = json.loads((ROOT/'data/continuation_20261021/annotation_freeze/windows.json').read_text())['windows']
    held = np.load(ROOT/'data/control_native_ball_v1/confirmation_frames.npy')
    source = load_source('tactical')
    matrices, _, valid = load()
    cache = np.load(parent/'weak_color_cache.npz')
    model = joblib.load(parent/'profile0_logistic.joblib')
    output = root/'inputs'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    board = Image.new('RGB', (768, 360), 'white')
    draw, font = ImageDraw.Draw(board), ImageFont.load_default(size=24)
    references = []
    for team, label in enumerate(('HOME', 'AWAY')):
        draw.text((10, team*180+5), label, fill='black', font=font)
        used = set()
        for index, frame in enumerate(cache['frames'][:240]):
            frame = int(frame)
            if not valid[frame] or not allowed_frame(frame, reserved, held, exclude_public=True):
                continue
            for slot in np.flatnonzero(cache['weak_mask'][index] & (cache['teams'] == team)):
                features = cache['features'][0, index, slot]
                if not np.isfinite(features).all() or model.predict_proba(features[None])[0, team] < .99:
                    continue
                bounds = torso_bounds(matrices[frame], source['positions'][frame, slot], 1.)
                if bounds is None or bounds[3]-bounds[1] < 8:
                    continue
                crop = Image.fromarray(decode_frame(frame)).crop(bounds)
                crop.thumbnail((200, 125))
                crop = crop.resize((max(1, round(crop.width*125/crop.height)), 125))
                column = len(used)
                board.paste(crop, (160+column*200, team*180+42))
                references.append(dict(team=team, frame=frame, slot=int(slot), bounds=list(bounds)))
                used.add(frame)
                break
            if len(used) == 3:
                break
        if len(used) != 3:
            raise ValueError('Not enough prospective weak jersey references')
    board.save(output/'jersey_reference.png')
    rows, omitted = [], 0
    for center in sample_centers(valid, half_video_blocks(GAME), reserved, held):
        images, metadata = [], []
        for frame in (center-12, center, center+12):
            value = scene(frame, source, matrices[frame])
            if value is None:
                break
            image, record = value
            images.append(image)
            metadata.append(record)
        if len(images) != 3:
            omitted += 1
            continue
        names = []
        for image, record in zip(images, metadata):
            name = f'scene_{center}_{record["frame"]}.png'
            image.save(output/name)
            names.append(name)
        rows.append(dict(center=center, images=names, source=metadata))
    (output/'samples.json').write_text(json.dumps(rows, indent=2)+'\n')
    (output/'references.json').write_text(json.dumps(references, indent=2)+'\n')
    sources = ('prepare_local_ownership.py', 'local_ownership.py', 'jersey_color.py')
    record = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in sources},
        input_sha256=plan['input_sha256'], cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        targets_not_read=True, no_manual_references=True, samples=len(rows), omitted_missing_scene=omitted,
        reserved_and_held_images_excluded=True, reference_public_interval_excluded=True)
    (output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(samples=len(rows), references=len(references), omitted_missing_scene=omitted), indent=2))


if __name__ == '__main__':
    main()
