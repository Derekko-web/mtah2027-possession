"""Construct frozen local trajectory features and aggregate diagnostic evidence."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from contact_dynamics import geometry_features, impulse_features, load_observations, repair_ball
from games import require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"data/control_dynamics_v1/inputs")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    plan_path = ROOT/"experiments/CONTACT_DYNAMICS_PLAN.json"
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    n = 18935
    xy, speeds, ball, teams = load_observations(n)
    np.savez(output/"observations.npz", positions=xy, speeds=speeds, ball=ball, teams=teams)
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt).label.to_numpy()[:n]
    reference = pd.read_csv(ROOT/"data/control_dynamics_v1/baseline_confirmation/segment_diagnostics.csv")
    short = reference[(reference.protocol == "buffered_minutes") & (reference.model == "stoppage_consensus_seed42") &
                      (reference.label != "D") & (reference.frames <= 125)]
    cases = [("all_public_live", np.flatnonzero(np.isin(labels, ["H", "A"])))]
    cases += [(f"short_{int(span.frames)}", np.arange(int(span.start), int(span.end)+1)) for span in short.itertuples()]
    rows = []
    for name in ("raw", "median", "robust"):
        corrected = ball if name == "raw" else repair_ball(ball, name)
        geometry = geometry_features(xy, speeds, corrected, ball, teams)
        geometry.to_pickle(output/f"{name}_geometry.pkl")
        if name == "robust":
            impulse = impulse_features(corrected, geometry)
            impulse.to_pickle(output/"impulse.pkl")
        nearest = geometry[[prefix+"clean_distance_1" for prefix in ("h__", "a__")]].to_numpy()
        seen = np.isfinite(nearest).all(axis=1)
        predicted = np.where(nearest[:, 0] <= nearest[:, 1], "H", "A")
        for case, frames in cases:
            observed = seen[frames]
            rows.append(dict(method=name, case=case, frames=len(frames), observed_fraction=float(observed.mean()),
                             correct_observed_fraction=float(np.mean((predicted[frames] == labels[frames]) & observed)),
                             accuracy_when_observed=float(np.mean(predicted[frames][observed] == labels[frames][observed])) if observed.any() else None))
    pd.DataFrame(rows).to_csv(output/"nearest_team_diagnostics.csv", index=False)
    sources = ("contact_dynamics.py", "prepare_contact_dynamics.py")
    manifest = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/"experiments"/name) for name in sources},
                    input_sha256={str(Path(p).resolve()):digest(p) for p in require(["atd_meta", "atd_raw"])},
                    public_gt_sha256=digest(gt), n_frames=n,
                    cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
