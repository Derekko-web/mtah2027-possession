"""Path-independent identity checks for the six local camera tracking assets."""

from pathlib import Path

from build_temporal_candidate import digest
from games import require

CAMERA_ASSETS = tuple(f"{source}/{asset}" for source in ("tactical", "broadcast")
                      for asset in ("atd_meta", "atd_raw", "homography"))


def camera_asset_path(identity):
    """Resolve a whitelisted logical identity through the current data directory."""
    if identity not in CAMERA_ASSETS:
        raise ValueError("Unknown camera asset identity: " + identity)
    source, asset = identity.split("/")
    return Path(require([asset], source=source)[0])


def camera_input_hashes():
    return {identity: digest(camera_asset_path(identity)) for identity in CAMERA_ASSETS}


def verify_camera_inputs(expected):
    if set(expected) != set(CAMERA_ASSETS):
        raise ValueError("Camera input manifest must identify all six assets")
    for identity, sha in expected.items():
        if digest(camera_asset_path(identity)) != sha:
            raise ValueError("Camera input changed: " + identity)
