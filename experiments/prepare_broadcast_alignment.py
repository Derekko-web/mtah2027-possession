"""Run prospective local alignment and observation-only feasibility checks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import (formation_anchors,frame_usable,load_source,map_anchors,
                                  reconcile_teams,transform)
from broadcast_features import aligned_features
from build_temporal_candidate import digest
from broadcast_provenance import camera_input_hashes


def sliced(source,n):
    return {key:value[:n] if key in ("positions","speeds","ball","valid") else value for key,value in source.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/control_broadcast_v1/inputs")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True,exist_ok=False)
    plan_path = ROOT/"experiments/BROADCAST_ALIGNMENT_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    print("Loading local tracking from both sources; no media or target labels",flush=True)
    tactical,broadcast = load_source("tactical",19060),load_source("broadcast")
    broadcast["teams"] = reconcile_teams(tactical["names"],broadcast["names"],broadcast["teams"])
    for name,source in (("tactical",tactical),("broadcast",broadcast)):
        np.savez(output/(name+"_observations.npz"),**{key:value for key,value in source.items() if key != "names"})
    profiles = plan["alignment"]["profiles"]
    def progress(index,total):
        print("Formation search",index,"/",total,flush=True)
    anchors = formation_anchors(tactical,broadcast,profiles,progress)
    np.savez(output/"anchors.npz",**anchors)
    public = np.arange(11185,18685)
    rows,mappings = [],{}
    for name,profile in profiles.items():
        mapping = map_anchors(anchors,tactical,broadcast,profile)
        mappings[name] = mapping
        np.savez(output/(name+"_mapping.npz"),**mapping)
        valid = mapping["frames"] >= 0
        bball = np.full_like(tactical["ball"],np.nan)
        for orientation in range(4):
            use = valid & (mapping["orientation"] == orientation)
            bball[use] = transform(broadcast["ball"][mapping["frames"][use]],orientation)
        bseen = np.isfinite(bball).all(axis=1)
        both = bseen & np.isfinite(tactical["ball"]).all(axis=1)
        disagreement = np.linalg.norm(bball[public]-tactical["ball"][public],axis=1)
        paired = int(both[public].sum())
        median = float(np.median(disagreement[both[public]])) if paired else None
        row = dict(profile=name,mapping_coverage=float(valid[public].mean()),broadcast_ball_coverage=float(bseen[public].mean()),
                   paired_ball_frames=paired,median_ball_disagreement=median,accepted_anchors=int(mapping["anchor_allowed"].sum()))
        row["feasible"] = bool(row["mapping_coverage"] >= .1 and row["broadcast_ball_coverage"] >= .05 and paired >= 100 and median <= 5)
        rows.append(row)
    selected = next((row["profile"] for row in rows if row["feasible"]),None)
    pd.DataFrame(rows).to_csv(output/"feasibility.csv",index=False)
    diag = dict(tactical_eligible_anchors=int(frame_usable(tactical)[anchors["tactical_frames"]].sum()),
                broadcast_eligible_frames=int(frame_usable(broadcast).sum()),
                coarse_queries=int(np.isfinite(anchors["coarse"][:,0]).sum()),
                fine_queries=int(np.isfinite(anchors["fine"][:,0]).sum()),
                median_coarse_cost=float(np.nanmedian(anchors["coarse"][:,2])),
                median_far_margin=float(np.nanmedian(anchors["coarse"][:,3])),
                median_fine_span=float(np.nanmedian(anchors["fine"][:,2])) if np.isfinite(anchors["fine"][:,2]).any() else None)
    prefix_parity = None
    if selected is not None:
        same,combined = aligned_features(tactical,broadcast,mappings[selected])
        same.to_pickle(output/"same_inputs.pkl")
        combined.to_pickle(output/"combined_inputs.pkl")
        print("Verifying actual padded-prefix alignment and feature parity",flush=True)
        prefix = sliced(tactical,18935)
        prefix_anchors = formation_anchors(prefix,broadcast,profiles,progress)
        prefix_mapping = map_anchors(prefix_anchors,prefix,broadcast,profiles[selected])
        np.testing.assert_array_equal(prefix_mapping["frames"][public],mappings[selected]["frames"][public])
        prefix_same,prefix_combined = aligned_features(prefix,broadcast,prefix_mapping)
        pd.testing.assert_frame_equal(same.iloc[public],prefix_same.iloc[public])
        pd.testing.assert_frame_equal(combined.iloc[public],prefix_combined.iloc[public])
        prefix_parity = True
    sources = ("broadcast_alignment.py","broadcast_features.py","prepare_broadcast_alignment.py","contact_dynamics.py","broadcast_provenance.py")
    inputs = camera_input_hashes()
    manifest = dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/"experiments"/name) for name in sources},
                    input_sha256=inputs,selected_profile=selected,diagnostics=diag,feasibility=rows,
                    actual_padded_feature_parity=prefix_parity,no_target_labels_read=True,
                    cache_sha256={path.name:digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(selected_profile=selected,diagnostics=diag,feasibility=rows),indent=2),flush=True)


if __name__ == "__main__":
    main()
