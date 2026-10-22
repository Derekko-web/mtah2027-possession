"""Local native-detector crop geometry, strict result parsing and repeatable batches."""

import json
from pathlib import Path
import subprocess

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def verify_native_assets(plan, manifest, root=ROOT):
    """Verify current compiler artifacts without requiring historical metadata bytes."""
    for name, expected in manifest['source_sha256'].items():
        if digest(root/name) != expected:
            raise ValueError('Native asset-generation source changed')
    by_name = {model['model']:model for model in manifest['models']}
    for model in plan['models']:
        local = by_name[model['name']]
        if digest(root/model['source_model']) != model['sha256'] or local['source_sha256'] != model['sha256']:
            raise ValueError('Pinned native raw model differs')
        compiled = root/model['compiled_model']
        actual = {str(path.relative_to(compiled)):digest(path) for path in sorted(compiled.rglob('*')) if path.is_file()}
        if actual != local['compiled_sha256']:
            raise ValueError('Local compiled native artifact changed')


def square_crop(image, x, y, side):
    width, height = image.size
    half = side/2
    if not np.isfinite([x, y]).all() or x-half < 0 or y-half < 0 or x+half > width or y+half > height:
        return None
    from PIL import Image
    return image.resize((416, 416), Image.Resampling.LANCZOS, box=(x-half, y-half, x+half, y+half))


def image_box(box, side, center=(0., 0.)):
    x, y, width, height = np.asarray(box, float)
    return np.array([center[0]-side/2+x*side, center[1]-side/2+(1-y-height)*side,
                     width*side, height*side])


def canonical_objects(objects):
    result = []
    for obj in objects:
        box = np.asarray(obj['box'], float)
        if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
            raise ValueError('Invalid native normalized box')
        labels = []
        for value in obj['labels']:
            confidence = float(value['confidence'])
            label = value['label']
            if not isinstance(label, str) or not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Invalid native label/confidence')
            labels.append(dict(label=label, confidence=confidence))
        labels.sort(key=lambda x:(-x['confidence'], x['label']))
        result.append(dict(labels=labels, box=box.tolist()))
    result.sort(key=lambda x:(tuple(x['box']), json.dumps(x['labels'], sort_keys=True)))
    return result


def canonical_reply(reply, expected_ids):
    if reply['compute_units'] != 0 or reply['confidence_floor'] != .01 or reply['iou_threshold'] != .5:
        raise ValueError('Native detector configuration changed')
    rows = reply['rows']
    ids = [row['id'] for row in rows]
    if ids != list(expected_ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing, reordered or duplicate native image IDs')
    return [dict(id=row['id'], objects=canonical_objects(row['objects'])) for row in rows]


def ball_scores(objects, side, diameter):
    near, any_score = 0., 0.
    radius = max(4., min(12., 2*diameter))
    for obj in objects:
        score = max((value['confidence'] for value in obj['labels'] if value['label'] == 'sports ball'), default=0.)
        any_score = max(any_score, score)
        box = image_box(obj['box'], side)
        center = box[:2]+box[2:]/2
        if np.linalg.norm(center) <= radius:
            near = max(near, score)
    return near, any_score


def detection_metrics(score, target, validation_frames):
    score, target = np.asarray(score), np.asarray(target)
    predicted = score >= .25
    confidence = np.where(predicted, score, 1-score)
    confident = confidence >= .8
    row = dict(positive_validation=int((target == 1).sum()), negative_validation=int((target == 0).sum()),
        positive_valid_fraction=float((target == 1).sum()/validation_frames),
        balanced_accuracy=float(balanced_accuracy_score(target, predicted)),
        positive_recall=float(predicted[target == 1].mean()), negative_false_positive=float(predicted[target == 0].mean()),
        confident_coverage=float(confident.mean()), confident_agreement=float((predicted[confident] == target[confident]).mean()) if confident.any() else 0.)
    row['feasible'] = bool(row['positive_validation'] >= 40 and row['negative_validation'] >= 120 and row['positive_valid_fraction'] >= .5 and
        row['balanced_accuracy'] >= .8 and row['confident_coverage'] >= .5 and row['confident_agreement'] >= .9)
    return row


def native_batches(compiled_model, images, output, resume=False, batch_size=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    requests, raw = output/'requests', output/'native_outputs'
    requests.mkdir(exist_ok=resume)
    raw.mkdir(exist_ok=resume)
    all_rows = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start+batch_size]
        request = dict(compiled_model=str(Path(compiled_model).relative_to(ROOT)), images=batch)
        request_path = requests/f'batch{start//batch_size:04d}.json'
        payload = json.dumps(request, indent=2)+'\n'
        if request_path.exists() and request_path.read_text() != payload:
            raise ValueError('Native resumed request differs')
        request_path.write_text(payload)
        response = raw/f'batch{start//batch_size:04d}.json'
        if not (resume and response.exists()):
            with response.open('w') as destination, response.with_suffix('.log').open('w') as log:
                subprocess.run(['osascript', '-l', 'JavaScript', str(ROOT/'experiments/native_detector_predict.js'),
                    str(request_path.relative_to(ROOT))], cwd=ROOT, stdout=destination, stderr=log, check=True)
        reply = json.loads(response.read_text())
        all_rows.extend(canonical_reply(reply, [item['id'] for item in batch]))
        print('Local native crops processed', min(start+batch_size, len(images)), '/', len(images), flush=True)
    (output/'predictions.json').write_text(json.dumps(all_rows, sort_keys=True, separators=(',', ':'))+'\n')
    return all_rows
