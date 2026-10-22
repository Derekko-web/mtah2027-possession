"""Camera manifests identify bytes independently of their local directory."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from broadcast_provenance import (CAMERA_ASSETS, camera_asset_path,
                                  camera_input_hashes, verify_camera_inputs)


class CameraProvenanceTests(unittest.TestCase):
    def test_relocated_assets_have_identical_identity_and_hashes(self):
        with TemporaryDirectory() as tmp:
            roots = [Path(tmp)/name for name in ("first", "relocated")]
            for root in roots:
                for identity in CAMERA_ASSETS:
                    path = root/identity
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(identity.encode())
            def resolve(root):
                return lambda assets, source: [str(root/source/asset) for asset in assets]
            with patch("broadcast_provenance.require", side_effect=resolve(roots[0])):
                original = camera_input_hashes()
            with patch("broadcast_provenance.require", side_effect=resolve(roots[1])):
                self.assertEqual(original, camera_input_hashes())
                verify_camera_inputs(original)
                (roots[1]/CAMERA_ASSETS[0]).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Camera input changed"):
                    verify_camera_inputs(original)

    def test_rejects_missing_unknown_and_machine_path_identities(self):
        with self.assertRaisesRegex(ValueError, "all six"):
            verify_camera_inputs({})
        for identity in ("/absolute/path", "tactical/video", "tactical/../atd_raw"):
            with self.assertRaisesRegex(ValueError, "Unknown camera"):
                camera_asset_path(identity)


if __name__ == "__main__":
    unittest.main()
