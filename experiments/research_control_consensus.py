"""Use bounded brief proposals only when two complementary models agree.

Proposals come from the previous model's decoded sequence, never true intervals.
The current model retains dead/live evidence and ownership outside proposals.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from control_refinement import compose, contact_support, control_evidence, smooth
from evaluate_physics_context import apply_context
from research_control_refinement import add_short_metrics
from research_stoppages import record_predictions
from temporal_candidate import gap_sides
from transition_penalty import decode
from xgboost_baseline import CLASSES
from consensus_control import consensus_team


def main():
    source = ROOT/"data/advanced_validation"
    output = ROOT/"data/control_consensus_research"
    output.mkdir(exist_ok=True)
    raw = pd.read_pickle(source/"raw.pkl")
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    side = gap_sides(raw)[public]
    support, evidence = contact_support(raw)[public], control_evidence(raw)[public]
    splits = json.loads((source/"splits.json").read_text())
    configs = {"current": None}
    for threshold in (.6, .65, .75):
        for strength in (0, .25, .5, .75):
            configs[f"consensus{threshold}_contact{strength}"] = dict(threshold=threshold, strength=strength)
    (output/"configs.json").write_text(json.dumps(configs,indent=2)+"\n")
    rows, details, segments = [], [], []
    for protocol in ("buffered_minutes", "purged_spells", "buffered_100_seconds"):
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        folds = [(s["block"], np.array(s["train"]), np.array(s["validation"]))
                 for s in splits if s["protocol"] == protocol]
        p = np.load(source/protocol/"ordered_xgboost_frame_seed42_probabilities.npy")
        p = apply_context(p, side, np.zeros(7500), False)
        prev = np.load(source/protocol/"previous_ensemble_seed42_probabilities.npy")
        prev = apply_context(prev, side, np.zeros(7500), False)
        fast = np.load(ROOT/"data/control_refinement"/protocol/"fast_probabilities.npy")
        fast = apply_context(np.column_stack((fast, np.zeros(7500))), side, np.zeros(7500), False)[:, :2]
        fast /= fast.sum(axis=1,keepdims=True)
        table = pd.DataFrame({"frame":public,"ground_truth":labels[public]})
        for name, config in configs.items():
            prediction = np.empty(7500,dtype="<U1")
            for _, _, validation in folds:
                ix = validation-public[0]
                slow, old, short = smooth(p[ix],25),smooth(prev[ix],25),smooth(fast[ix],5)
                if config is None:
                    emission = slow
                else:
                    team = consensus_team(slow,old,short,support[ix],config["threshold"])
                    dead = slow[:,2]*(1-config["strength"]*evidence[ix].max(axis=1))
                    emission = compose(dead,team)
                prediction[ix] = CLASSES[decode(emission)]
            table[name] = prediction
            record_predictions(output,protocol,name,public,labels,folds,prediction,rows,details,segments)
        add_short_metrics(rows,segments)
        table.to_csv(folder/"oof_predictions.csv",index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
