"""Prospective directional-interaction and transition-weighted21-row screen."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from motion_dead_trim import decode_motion_dead_trim
from ownership_graph import decode_graph, fit_graph
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_pair_motion import reference_arrays
from research_stoppages import record_predictions
from temporal_candidate import predict_models


def selected_arrays(protocol, seed, n):
    arrays, hashes = reference_arrays(protocol, seed, n)
    path = ROOT/'data/control_pair_motion_v1/dead_trim_confirmation'/protocol/f'broadcast_motion_seed{seed}_probabilities.npy'
    probability = np.full((n, 3), np.nan)
    probability[11185:18685] = np.load(path)
    hashes[str(path.relative_to(ROOT))] = digest(path)
    return arrays+[probability], hashes


def main():
    root = ROOT/'data/control_ownership_graph_v1'
    output = root/'screen'
    output.mkdir(exist_ok=False)
    cache = root/'inputs'
    cm = audit_manifest(cache)
    plan_path = ROOT/'experiments/OWNERSHIP_GRAPH_PLAN.json'
    plan = json.loads(plan_path.read_text())
    if cm['plan_sha256'] != digest(plan_path) or not cm['actual_padded_prefix_parity']:
        raise ValueError('Graph cache plan/parity differs')
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    upstream = ROOT/'data/control_pair_motion_v1'
    audit_manifest(upstream/'inputs')
    audit_manifest(upstream/'dead_trim_confirmation')
    source = ROOT/'data/advanced_validation'
    raw, ordered = [pd.read_pickle(source/(name+'.pkl')) for name in ('raw', 'ordered')]
    bc = pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:len(raw)]
    motion_info = pd.read_pickle(upstream/'inputs/motion_info.pkl').iloc[:len(raw)]
    info = pd.read_pickle(cache/'graph_info.pkl').iloc[:len(raw)]
    features = pd.read_pickle(cache/'graph_inputs.pkl').iloc[:len(raw)]
    views = dict(ordered=ordered, graph=ordered.join(bc).join(features))
    gt = ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != 'X')
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    split_path = source/'splits.json'
    splits = json.loads(split_path.read_text())
    (output/'splits.json').write_bytes(split_path.read_bytes())
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in (gt, split_path, source/'raw.pkl', source/'ordered.pkl', cache/'manifest.json')}
    rows, details, segments = [], [], []
    for protocol in plan['screen']['protocols']:
        folder = output/protocol
        folder.mkdir()
        folds = [(s['block'], np.array(s['train']), np.array(s['validation'])) for s in splits if s['protocol'] == protocol]
        blocks = [(int(val[0]), int(val[-1])+1) for _, _, val in folds]
        arrays, identities = selected_arrays(protocol, 42, len(raw))
        hashes.update(identities)
        reference = decode_motion_dead_trim(*arrays, bc, motion_info, raw, blocks)[public]
        prior = root/'baseline_confirmation'/protocol/'oof_predictions.csv'
        hashes[str(prior.relative_to(ROOT))] = digest(prior)
        np.testing.assert_array_equal(reference, pd.read_csv(prior).motion_dead_trim_seed42)
        table = pd.DataFrame(dict(frame=public, ground_truth=labels[public], current_seed42=reference))
        record_predictions(output, protocol, 'current_seed42', public, labels, folds, reference, rows, details, segments)
        add_overlap_metrics(rows, segments, table, protocol, 'current_seed42')
        for head, config in plan['heads'].items():
            inputs = views[config['view']]
            probability = np.full((len(raw), 3), np.nan)
            for block, train, validation in folds:
                print(protocol, head, 'block', block, flush=True)
                model, y, weight = fit_graph(inputs, labels, train, config['transition_weight'], 42)
                model.save_model(folder/f'{head}_seed42_block{block}.ubj')
                np.savez(folder/f'{head}_seed42_block{block}_fitting.npz', train=train, validation=validation, augmented_target=y, augmented_weight=weight)
                probability[validation] = predict_models([model], inputs.iloc[validation])
            np.save(folder/f'{head}_seed42_probabilities.npy', probability[public])
            for mode in plan['decoders']:
                prediction = decode_graph(arrays, probability, bc, motion_info, info, raw, blocks, mode)[public]
                name = f'{head}_{mode}_seed42'
                table[name] = prediction
                record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
                add_overlap_metrics(rows, segments, table, protocol, name)
        add_short_metrics(rows, segments)
        table.to_csv(folder/'oof_predictions.csv', index=False)
        pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
        pd.DataFrame(details).to_csv(output/'fold_scores.csv', index=False)
        pd.DataFrame(segments).to_csv(output/'segment_diagnostics.csv', index=False)
    names = ('research_ownership_graph.py', 'ownership_graph.py', 'motion_dead_trim.py', 'weak_ownership.py', 'research_pair_motion.py', 'temporal_candidate.py')
    manifest = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names}, input_sha256=hashes,
        feature_columns={name:list(value) for name, value in views.items()}, training_only_public_targets=True, selected_eight_reference_exact=True,
        model_sha256={str(path.relative_to(output)):digest(path) for path in sorted(output.rglob('*.ubj'))}, models=39)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(pd.DataFrame(rows)[['protocol', 'model', 'spellF1', 'macroF1', 'false_dead_frames', 'short_live_mean_best_iou', 'short_live_matched_iou50']].round(6).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
