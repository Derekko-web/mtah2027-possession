"""Full local broadcast-motion cache; label-free and exact against public prefix."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from pair_motion import pair_feature_views


def main():
    root = ROOT/"data/control_pair_motion_v1";output = root/"full_inputs";output.mkdir(exist_ok=False)
    cm = audit_manifest(root/"inputs")
    cache = ROOT/"data/control_broadcast_v1";audit_manifest(cache/"inputs");full = audit_manifest(cache/"full_inputs")
    if not full["public_mapping_feature_parity"] or not cm["actual_padded_prefix_parity"]: raise ValueError("Public prefix is not verified")
    print("Loading full local observations; no possession targets",flush=True)
    tactical = load_source("tactical")
    broadcast = dict(np.load(cache/"inputs/broadcast_observations.npz"))
    mapping = dict(np.load(cache/"full_inputs/mapping.npz"))
    bc = pd.read_pickle(cache/"full_inputs/combined_inputs.pkl")
    print("Computing full native pair motion and ordered source taps",flush=True)
    views,info = pair_feature_views(tactical,broadcast,mapping,bc)
    features = views["broadcast_motion"]
    columns = [name for name in info if name.startswith("bm_") or "__bm_" in name]
    info = info[columns]
    pd.testing.assert_frame_equal(features.iloc[11185:18685],pd.read_pickle(root/"inputs/broadcast_motion_inputs.pkl").iloc[11185:18685])
    pd.testing.assert_frame_equal(info.iloc[11185:18685],pd.read_pickle(root/"inputs/motion_info.pkl")[columns].iloc[11185:18685])
    features.to_pickle(output/"broadcast_motion_inputs.pkl");info.to_pickle(output/"motion_info.pkl")
    names = ("prepare_full_pair_motion.py","pair_motion.py","broadcast_alignment.py")
    result = dict(plan_sha256=digest(ROOT/"experiments/PAIR_MOTION_PLAN.json"),
        source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        input_sha256={"data/control_pair_motion_v1/inputs/manifest.json":digest(root/"inputs/manifest.json"),
            "data/control_broadcast_v1/full_inputs/manifest.json":digest(cache/"full_inputs/manifest.json")},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        n_frames=len(features),feature_columns=list(features),public_motion_feature_parity=True,targets_not_read=True)
    (output/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Full broadcast-motion inputs saved; all public features/info match exactly",flush=True)


if __name__ == "__main__": main()
