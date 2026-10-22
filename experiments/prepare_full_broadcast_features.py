"""Build full-match label-free broadcast features with exact public-cache parity."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import formation_anchors,load_source,map_anchors,reconcile_teams
from broadcast_features import aligned_features
from build_temporal_candidate import digest
from broadcast_provenance import verify_camera_inputs


def main():
    root = ROOT/"data/control_broadcast_v1"
    output = root/"full_inputs"
    output.mkdir(parents=True,exist_ok=False)
    prefix_manifest = json.loads((root/"inputs/manifest.json").read_text())
    for name,expected in prefix_manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected: raise ValueError("Public feature source changed")
    verify_camera_inputs(prefix_manifest["input_sha256"])
    plan_path = ROOT/"experiments/BROADCAST_ALIGNMENT_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if digest(plan_path) != prefix_manifest["plan_sha256"]: raise ValueError("Alignment plan changed")
    print("Loading full-match local observations; no labels or media",flush=True)
    tactical,broadcast = load_source("tactical"),load_source("broadcast")
    broadcast["teams"] = reconcile_teams(tactical["names"],broadcast["names"],broadcast["teams"])
    def progress(index,total): print("Full formation search",index,"/",total,flush=True)
    anchors = formation_anchors(tactical,broadcast,plan["alignment"]["profiles"],progress)
    np.savez(output/"anchors.npz",**anchors)
    profile = prefix_manifest["selected_profile"]
    mapping = map_anchors(anchors,tactical,broadcast,plan["alignment"]["profiles"][profile])
    np.savez(output/"mapping.npz",**mapping)
    same,combined = aligned_features(tactical,broadcast,mapping)
    public = np.arange(11185,18685)
    previous_mapping = np.load(root/"inputs"/(profile+"_mapping.npz"))
    np.testing.assert_array_equal(mapping["frames"][public],previous_mapping["frames"][public])
    for name,features in (("same",same),("combined",combined)):
        pd.testing.assert_frame_equal(features.iloc[public],pd.read_pickle(root/"inputs"/(name+"_inputs.pkl")).iloc[public])
        features.to_pickle(output/(name+"_inputs.pkl"))
    sources = ("prepare_full_broadcast_features.py","broadcast_alignment.py","broadcast_features.py","contact_dynamics.py","broadcast_provenance.py")
    manifest = dict(selected_profile=profile,plan_sha256=digest(plan_path),public_prefix_manifest_sha256=digest(root/"inputs/manifest.json"),
        source_sha256={name:digest(ROOT/"experiments"/name) for name in sources},input_sha256=prefix_manifest["input_sha256"],
        n_frames=len(tactical["positions"]),public_mapping_feature_parity=True,no_target_labels_read=True,
        cache_sha256={path.name:digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Full broadcast features saved; public mapping and features match exactly",flush=True)


if __name__ == "__main__": main()
