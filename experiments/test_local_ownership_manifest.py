import json
from pathlib import Path
import tempfile
import unittest
from local_ownership_manifest import generation_manifest


class LocalOwnershipManifestTests(unittest.TestCase):
    def test_actual_input_path_and_completed_resume_are_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, output = root/'data/probe/inputs_montage', root/'data/probe/inference_montage'
            inputs.mkdir(parents=True)
            output.mkdir()
            (inputs/'manifest.json').write_text('{}')
            (output/'answers.json').write_text('[]')
            plan_path = root/'plan.json'
            plan_path.write_text('{}')
            plan = dict(model_digest='test', runtime_version='test')
            before = generation_manifest(root, output, inputs, plan_path, {}, plan, 0)
            (output/'manifest.json').write_text(json.dumps(before))
            after = generation_manifest(root, output, inputs, plan_path, {}, plan, 0)
            self.assertEqual(before, after)
            self.assertEqual(list(after['input_sha256']), ['data/probe/inputs_montage/manifest.json'])
            self.assertNotIn('manifest.json', after['cache_sha256'])
            self.assertIn('answers.json', after['cache_sha256'])


if __name__ == '__main__':
    unittest.main()
