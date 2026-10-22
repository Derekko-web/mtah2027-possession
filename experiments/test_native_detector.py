import unittest
import hashlib
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from native_detector import ball_scores, canonical_reply, detection_metrics, image_box, square_crop, verify_native_assets


class NativeDetectorTests(unittest.TestCase):
    def test_crop_and_vision_bottom_left_coordinates(self):
        image = Image.new('RGB', (100, 80), (20, 120, 30))
        self.assertEqual(square_crop(image, 50., 40., 32).size, (416, 416))
        self.assertIsNone(square_crop(image, 5., 5., 32))
        np.testing.assert_array_equal(image_box([0, .75, .5, .25], 40, (100, 100)), [80, 80, 20, 10])

    def test_near_ball_score_rejects_a_distant_ball_and_other_classes(self):
        objects = [dict(box=[.8, .8, .1, .1], labels=[dict(label='sports ball', confidence=.9)]),
                   dict(box=[.45, .45, .1, .1], labels=[dict(label='sports ball', confidence=.4), dict(label='person', confidence=.99)])]
        near, any_score = ball_scores(objects, 64, 2.)
        self.assertEqual(near, .4)
        self.assertEqual(any_score, .9)

    def test_reply_validation_rejects_missing_ids_or_nonfinite_boxes(self):
        reply = dict(compute_units=0, confidence_floor=.01, iou_threshold=.5, rows=[dict(id='one', objects=[])])
        self.assertEqual(canonical_reply(reply, ['one']), [dict(id='one', objects=[])])
        with self.assertRaises(ValueError):
            canonical_reply(reply, ['two'])
        reply['rows'][0]['objects'] = [dict(box=[0, np.nan, .1, .1], labels=[])]
        with self.assertRaises(ValueError):
            canonical_reply(reply, ['one'])

    def test_weak_gate_does_not_pass_a_confident_all_negative_detector(self):
        target = np.r_[np.ones(60), np.zeros(300)]
        row = detection_metrics(np.zeros(len(target)), target, 60)
        self.assertEqual(row['balanced_accuracy'], .5)
        self.assertFalse(row['feasible'])
        row = detection_metrics(target*.9, target, 60)
        self.assertTrue(row['feasible'])

    def test_local_compiler_generation_is_bound_to_raw_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model').write_bytes(b'public raw source')
            (root/'writer').write_bytes(b'known compiler adapter')
            (root/'compiled').mkdir()
            (root/'compiled/metadata').write_bytes(b'new generation')
            sha = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
            plan = dict(models=[dict(name='detector', source_model='model', sha256=sha(root/'model'),
                compiled_model='compiled', compiled_sha256={'old historical metadata':'old'})])
            manifest = dict(source_sha256={'writer':sha(root/'writer')}, models=[dict(model='detector',
                source_sha256=sha(root/'model'), compiled_sha256={'metadata':sha(root/'compiled/metadata')})])
            verify_native_assets(plan, manifest, root)
            (root/'compiled/metadata').write_bytes(b'corruption')
            with self.assertRaises(ValueError):
                verify_native_assets(plan, manifest, root)

    def test_altered_raw_source_cannot_use_a_valid_compiler_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'model').write_bytes(b'changed source')
            (root/'compiled').mkdir()
            plan = dict(models=[dict(name='detector', source_model='model', sha256='expected', compiled_model='compiled')])
            manifest = dict(source_sha256={}, models=[dict(model='detector', source_sha256='expected', compiled_sha256={})])
            with self.assertRaises(ValueError):
                verify_native_assets(plan, manifest, root)


if __name__ == '__main__':
    unittest.main()
