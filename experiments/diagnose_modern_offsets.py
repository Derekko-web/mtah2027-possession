"""Aggregate-only location diagnostics; never changes frozen scores or selection."""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_appearance import negative_positions
from broadcast_alignment import load_source
from modern_ball import unletterbox_objects
from native_detector import image_box

def summary(offsets):
    if not offsets:
        return dict(count=0)
    values = np.asarray(offsets)
    return dict(count=len(values), median_offset_pixels=np.median(values, axis=0).tolist(),
                fraction_above_projected=float((values[:, 1] < 0).mean()),
                median_distance_pixels=float(np.median(np.linalg.norm(values, axis=1))),
                fraction_within32pixels=float((np.linalg.norm(values, axis=1) <= 32).mean()))


def main():
    root = ROOT/'data/control_modern_ball_v1'
    folder = root/'screen'
    if not (folder/'manifest.json').exists():
        raise ValueError('Finish the frozen screen before diagnostic interpretation')
    records = json.loads((folder/'crop_metadata.json').read_text())
    source = load_source('tactical')
    result = []
    for path in sorted(folder.glob('*/predictions.json')):
        side = int(path.parent.name.split('roi')[1])
        subset = [r for r in records if r['side'] == side]
        arrays = np.load(path.parent/'score_arrays.npz')
        unlocalized = arrays['any_score'] >= .25
        positive = arrays['target'] == 1
        unlocalized_recall = float(unlocalized[positive].mean())
        unlocalized_fpr = float(unlocalized[~positive].mean())
        ball, offcenter, feet = [], [], []
        for record, prediction in zip(subset, json.loads(path.read_text())):
            objects = unletterbox_objects(prediction['objects'])
            if record['label']:
                scored = [(max((v['confidence'] for v in obj['labels'] if v['label'] == 'sports ball'), default=0.), obj)
                          for obj in objects]
                if not scored:
                    continue
                confidence, obj = max(scored, key=lambda item:item[0])
                if confidence < .25:
                    continue
                box = image_box(obj['box'], side)
                offset = box[:2]+box[2:]/2
                ball.append(offset)
                if np.linalg.norm(offset) > max(4., min(12., 2*record['diameter'])):
                    offcenter.append(offset)
            else:
                frame = record['frame']
                point = negative_positions(source['ball'][frame], source['positions'][frame])[record['point']-1]
                distance = np.linalg.norm(source['positions'][frame]-point, axis=1)
                if not (distance <= 1e-6).any():
                    continue
                candidates = []
                for obj in objects:
                    if max((v['confidence'] for v in obj['labels'] if v['label'] == 'person'), default=0.) < .25:
                        continue
                    box = image_box(obj['box'], side)
                    candidates.append(box[:2]+box[2:]*[.5, 1.])
                if candidates:
                    feet.append(min(candidates, key=lambda offset:np.linalg.norm(offset)))
        result.append(dict(variant=path.parent.name,
            unlocalized_diagnostic=dict(positive_recall=unlocalized_recall, negative_false_positive=unlocalized_fpr,
                                        balanced_accuracy=(unlocalized_recall+1-unlocalized_fpr)/2),
            highest_confidence_ball_positive=summary(ball), offcenter_highest_confidence_ball_positive=summary(offcenter),
            nearest_person_box_foot_on_tracked_foot_negative=summary(feet)))
    (root/'offset_diagnostic.json').write_text(json.dumps(dict(
        diagnostic_only=True, selection_or_thresholds_changed=False,
        not_corrected_ball_or_person_ground_truth=True,
        ball_statistic='Highest-confidence sports ball per positive crop; can differ from the primary maximum-near score.',
        foot_statistic='Nearest bottom-center of a person box on negatives exactly coincident with an existing tracked player.',
        variants=result), indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
