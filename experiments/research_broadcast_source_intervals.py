"""Fixed source-interval follow-up reusing the initial broadcast screen's heads."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from broadcast_source_intervals import source_intervals
from build_temporal_candidate import digest
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from xgboost_baseline import CLASSES


def codes(labels):
    return np.array([np.flatnonzero(CLASSES == label)[0] for label in labels])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/control_broadcast_v1/source_screen")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True,exist_ok=False)
    plan_path = ROOT/"experiments/BROADCAST_SOURCE_INTERVAL_PLAN.json"
    plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    root = ROOT/"data/control_broadcast_v1"
    features_path = root/"inputs/combined_inputs.pkl"
    features = pd.read_pickle(features_path)
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != "X")
    splits_path = ROOT/"data/advanced_validation/splits.json"
    splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    rows,details,segments,proposals = [],[],[],[]
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in (gt,splits_path,features_path,root/"inputs/manifest.json",root/"screen/manifest.json")}
    for protocol in plan["screen"]["protocols"]:
        folder = output/protocol
        folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        path = root/"screen"/protocol/"oof_predictions.csv"
        hashes[str(path.relative_to(ROOT))] = digest(path)
        original = pd.read_csv(path)
        table = pd.DataFrame({"frame":public,"ground_truth":labels[public]})
        selected = original.current_seed42.to_numpy()
        reference = codes(selected)
        for name,config in {"current":None,**plan["screen"]["methods"]}.items():
            prediction = selected.copy()
            if config is not None:
                learned = codes(original[f"{config['model']}_replace_head_seed42"].to_numpy())
                for block,_,validation in folds:
                    ix = validation-public[0]
                    states,records = source_intervals(reference[ix],learned[ix],features.iloc[validation],config["half_window"],True)
                    prediction[ix] = CLASSES[states]
                    proposals.extend(dict(protocol=protocol,method=name,block=block,**record) for record in records)
            method = f"{name}_seed42"
            table[method] = prediction
            np.testing.assert_array_equal(prediction == "D",selected == "D")
            record_predictions(output,protocol,method,public,labels,folds,prediction,rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,method)
        add_short_metrics(rows,segments)
        table.to_csv(folder/"oof_predictions.csv",index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
        pd.DataFrame(proposals).to_csv(output/"proposals.csv",index=False)
    names = ("research_broadcast_source_intervals.py","broadcast_source_intervals.py","broadcast_consensus.py")
    (output/"manifest.json").write_text(json.dumps(dict(plan_sha256=digest(plan_path),input_sha256=hashes,
        source_sha256={name:digest(ROOT/"experiments"/name) for name in names},no_new_models_fitted=True,
        all_D_masks_unchanged=True),indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol","model","spellF1","macroF1","short_live_mean_best_iou","short_live_matched_iou50"]].round(6).to_string(index=False))


if __name__ == "__main__": main()
