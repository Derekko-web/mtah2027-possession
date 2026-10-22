"""Build local matched-camera motion caches and verify actual padded-prefix parity."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_features import aligned_features
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from pair_motion import pair_feature_views
from prepare_broadcast_alignment import sliced


def main():
    root = ROOT/"data/control_pair_motion_v1";output = root/"inputs";output.mkdir(exist_ok=False)
    cache = ROOT/"data/control_broadcast_v1/inputs"; cm = audit_manifest(cache)
    if cm["selected_profile"] != "strict" or not cm["actual_padded_feature_parity"]: raise ValueError("Broadcast prerequisites differ")
    plan = ROOT/"experiments/PAIR_MOTION_PLAN.json";(output/"plan.json").write_bytes(plan.read_bytes())
    tactical,broadcast = [dict(np.load(cache/(name+"_observations.npz"))) for name in ("tactical","broadcast")]
    mapping = dict(np.load(cache/"strict_mapping.npz")); bc = pd.read_pickle(cache/"combined_inputs.pkl")
    print("Computing native matched-set motion on both local timelines; no targets or media",flush=True)
    views,info = pair_feature_views(tactical,broadcast,mapping,bc)
    for name,value in views.items(): value.to_pickle(output/(name+"_inputs.pkl"))
    info.to_pickle(output/"motion_info.pkl")
    n = 18935;shorter = sliced(tactical,n)
    short_mapping = {key:value[:n] if len(value) == len(tactical["positions"]) else value for key,value in mapping.items()}
    _,short_bc = aligned_features(shorter,broadcast,short_mapping)
    shorter_views,short_info = pair_feature_views(shorter,broadcast,short_mapping,short_bc)
    for name in views: pd.testing.assert_frame_equal(views[name].iloc[11185:18685],shorter_views[name].iloc[11185:18685])
    pd.testing.assert_frame_equal(info.iloc[11185:18685],short_info.iloc[11185:18685])
    diagnostics = dict(tactical_observed_fraction=float(info.tc_observed.iloc[11185:18685].mean()),
        broadcast_observed_fraction=float(info.bm_observed.iloc[11185:18685].mean()),
        either_observed_fraction=float(((info.tc_observed > 0)|(info.bm_observed > 0)).iloc[11185:18685].mean()),
        feature_columns={name:len(value.columns) for name,value in views.items()})
    names = ("pair_motion.py","pair_motion_inference.py","prepare_pair_motion.py","broadcast_features.py","contact_dynamics.py","prepare_broadcast_alignment.py")
    manifest = dict(plan_sha256=digest(plan),source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        input_sha256={str((cache/"manifest.json").relative_to(ROOT)):digest(cache/"manifest.json")},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        actual_padded_prefix_parity=True,targets_not_read=True,diagnostics=diagnostics)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(diagnostics,indent=2),flush=True)


if __name__ == "__main__": main()
