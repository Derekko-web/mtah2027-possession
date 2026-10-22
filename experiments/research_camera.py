"""Add camera evidence to the retained ensemble and to independent dead/live heads."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from camera_features import camera_features
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import gap_context, apply_context
from research_sequence import adaptive_decode, control_support
from research_stoppages import fit_head, compose_sequence, record_predictions
from run_continuation import fit_ensemble, purged_spell_blocks
from xgboost_baseline import CLASSES, probability_smoothing
from transition_penalty import decode


def main():
    output = ROOT/"data/camera_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    camera = camera_features(len(base))
    camera.to_pickle(output/"camera_inputs.pkl")
    physical = pd.read_pickle(ROOT/"data/stoppage_research/physical_inputs.pkl")
    inputs = {"camera_only": camera, "camera_physical": physical.join(camera),
              "camera_ensemble": base.join(camera)}
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    side = gap_context(raw)[public]
    support = control_support(raw)[public]
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        team = apply_context(np.load(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"),
                             side, np.zeros(len(public)), False)
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, features in inputs.items():
            print(protocol, name, flush=True)
            p = np.full((len(public), 3), np.nan) if name == "camera_ensemble" else np.full(len(public), np.nan)
            for minute, train, validation in blocks:
                ix = validation-public[0]
                if name == "camera_ensemble":
                    p[ix] = fit_ensemble(features, labels, train, validation, folder, minute)
                else:
                    config = dict(family="tree", weights="segments")
                    model, p[ix] = fit_head(features, labels, train, validation, config)
                    model.save_model(folder/f"{name}_fold_{minute}.ubj")
            np.save(folder/f"{name}_probabilities.npy", p)
            for variant in (False, True):
                method = name+("_adaptive" if name == "camera_ensemble" and variant else
                               "_hierarchical" if variant else "_joint")
                if name == "camera_ensemble":
                    adjusted = apply_context(p, side, np.zeros(len(public)), False)
                    prediction = np.empty(len(public), dtype="<U1")
                    for _, _, validation in blocks:
                        ix = validation-public[0]
                        smooth = probability_smoothing(adjusted[ix])
                        path = adaptive_decode(smooth, support[ix]) if variant else decode(smooth)
                        prediction[ix] = CLASSES[path]
                else:
                    prediction = compose_sequence(team, p, blocks, public, variant)
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"features.json").write_text(json.dumps({n: list(x.columns) for n, x in inputs.items()}, indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
