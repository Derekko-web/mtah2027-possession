"""Use semantic phase evidence as model inputs or a soft occlusion control prior."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from compare_time_blocks import buffered_blocks
from evaluate_physics_context import apply_context, gap_context
from phase_control import phase_control
from research_sequence import adaptive_decode, control_support
from research_stoppages import record_predictions
from run_continuation import fit_ensemble, purged_spell_blocks
from transition_penalty import decode
from xgboost_baseline import CLASSES, probability_smoothing


def main():
    output = ROOT/"data/phase_control_research"
    output.mkdir(exist_ok=True)
    labels = pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.flatnonzero(labels != "X")
    raw = pd.read_pickle(ROOT/"data/rich_baseline/input_features.pkl")
    base = pd.read_pickle(ROOT/"data/relative_motion_baseline/input_features.pkl")
    phase = phase_control(len(base))
    phase.to_pickle(output/"phase_inputs.pkl")
    teacher = np.load(ROOT/"data/external_teacher_research/teacher_probability.npy")
    features = {"semantic_ensemble": base.join(phase),
                "semantic_teacher": base.join(phase).assign(external_teacher_dead=teacher)}
    folds = buffered_blocks(int(public[0]), int(public[-1]), 25)
    protocols = {"buffered_minutes": folds, "purged_spells": purged_spell_blocks(labels, folds)}
    rows, details, segments = [], [], []
    side = gap_context(raw)[public]
    support = control_support(raw)[public]
    h, a = phase.h__phase_control_support.to_numpy()[public], phase.a__phase_control_support.to_numpy()[public]
    ph_phase = (h+.1)/(h+a+.2)
    for protocol, blocks in protocols.items():
        folder = output/protocol
        folder.mkdir(exist_ok=True)
        parent = apply_context(np.load(ROOT/"data/continuation_20261021"/protocol/"selected/oof_probabilities.npy"),
                               side, np.zeros(len(public)), False)
        ph = parent[:, 0]/parent[:, :2].sum(axis=1)
        probabilities = {}
        for name, inputs in features.items():
            print(protocol, name, flush=True)
            target = folder/name
            target.mkdir(exist_ok=True)
            p = np.full((len(public), 3), np.nan)
            for minute, train, validation in blocks:
                p[validation-public[0]] = fit_ensemble(inputs, labels, train, validation, target, minute)
            np.save(target/"oof_probabilities.npy", p)
            probabilities[name] = apply_context(p, side, np.zeros(len(public)), False)
        for mode in ("occluded", "all"):
            use = ((h+a) > 0) & ((base.ball_seen.to_numpy()[public] == 0) if mode == "occluded" else True)
            new_ph = ph.copy()
            new_ph[use] = .5*ph[use]+.5*ph_phase[use]
            dead = parent[:, 2]
            probabilities["semantic_prior_"+mode] = np.column_stack(((1-dead)*new_ph, (1-dead)*(1-new_ph), dead))
        table = pd.DataFrame({"frame": public, "ground_truth": labels[public]})
        for name, p in probabilities.items():
            for adaptive in (False, True):
                method = name+("_adaptive" if adaptive else "_fixed")
                prediction = np.empty(len(public), dtype="<U1")
                for _, _, validation in blocks:
                    ix = validation-public[0]
                    smooth = probability_smoothing(p[ix])
                    prediction[ix] = CLASSES[adaptive_decode(smooth, support[ix]) if adaptive else decode(smooth)]
                table[method] = prediction
                record_predictions(output, protocol, method, public, labels, blocks, prediction, rows, details, segments)
        table.to_csv(folder/"oof_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/"scores.csv", index=False)
    pd.DataFrame(details).to_csv(output/"fold_scores.csv", index=False)
    pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv", index=False)
    (output/"feature_names.json").write_text(json.dumps({k: list(v.columns) for k,v in features.items()}, indent=2)+"\n")
    print(scores.sort_values(["protocol", "spellF1"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
