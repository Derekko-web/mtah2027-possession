"""Preserved reference checkpoints must not be classified as new fitting."""

from pathlib import Path
import tempfile
import unittest

from finalize_provided_speed_preflight import unexpected_checkpoints


class ProvidedSpeedAuditTests(unittest.TestCase):
    def test_preserved_baseline_is_allowed_but_new_checkpoint_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            preserved = root/'baseline_candidate/models/reference.ubj'
            preserved.parent.mkdir(parents=True)
            preserved.write_text('preserved synthetic checkpoint')
            self.assertEqual(unexpected_checkpoints(root), [])
            added = root/'screen/model.ubj'
            added.parent.mkdir()
            added.write_text('new synthetic checkpoint')
            self.assertEqual(unexpected_checkpoints(root), [added])


if __name__ == '__main__':
    unittest.main()
