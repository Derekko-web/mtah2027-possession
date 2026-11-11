"""Prospective30-row detector-feature screen with a matched no-vision control."""

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
from ownership_graph import fitting_weights
from provided_speed import decode_provided_speed
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_control_renewal import fit_renewal
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from temporal_candidate import predict_models


def input_views(root, ordered, bc):
    base = ordered.join(bc)
    views = {'matched_control':base}
    shape = pd.read_pickle(root/'inputs/shape_inputs.pkl').iloc[:len(ordered)]
    views['shape_joint'] = base.join(shape)
    views['shape_reduced'] = shape
    return views


def main():
    root = ROOT/'data/control_spatial_shape_v1'
    cache = audit_manifest(root/'inputs')
    plan_path = ROOT/'experiments/SPATIAL_SHAPE_PLAN.json'
    plan = json.loads(plan_path.read_text())
    if cache['plan_sha256'] != digest(plan_path) or not cache['actual_padded_prefix_parity']:
        raise ValueError('Detector-feature cache plan/parity differs')
    if not cache['support_passed']:
        raise SystemExit('STOP BEFORE FITTING: detector observations fail frozen support counts')
    output = root/'screen'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    names = ('research_spatial_shape.py', 'provided_speed.py', 'spatial_shape.py', 'motion_dead_trim.py',
             'research_control_renewal.py', 'ownership_graph.py', 'research_ownership_graph.py', 'research_pair_motion.py',
             'temporal_candidate.py', 'rich_features.py', 'build_candidate.py', 'weak_ownership.py',
             'pair_candidate.py', 'broadcast_refinement.py', 'stoppage_consensus.py')
    sources = {name:digest(ROOT/'experiments'/name) for name in names}
    (output/'source_at_dispatch.json').write_text(json.dumps(sources, indent=2)+'\n')
    upstream = ROOT/'data/control_pair_motion_v1'
    audit_manifest(upstream/'inputs')
    audit_manifest(upstream/'dead_trim_confirmation')
    audit_manifest(ROOT/'data/control_broadcast_v1/inputs')
    control_manifest = audit_manifest(ROOT/'data/control_broadcast_v1/closure_confirmation')
    source = ROOT/'data/advanced_validation'
    raw, ordered = [pd.read_pickle(source/(name+'.pkl')) for name in ('raw', 'ordered')]
    bc = pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:len(raw)]
    info = pd.read_pickle(upstream/'inputs/motion_info.pkl').iloc[:len(raw)]
    views = input_views(root, ordered, bc)
    if control_manifest['feature_columns'] != list(views['matched_control']):
        raise ValueError('Reused no-scalar control has a different feature view')
    gt = ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    labels = pm.read_label_csv(gt).label.to_numpy()
    public = np.flatnonzero(labels != 'X')
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    split_path = source/'splits.json'
    splits = json.loads(split_path.read_text())
    (output/'splits.json').write_bytes(split_path.read_bytes())
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in (
        gt, split_path, source/'raw.pkl', source/'ordered.pkl', root/'inputs/manifest.json',
        ROOT/'data/control_broadcast_v1/closure_confirmation/manifest.json')}
    rows, details, segments = [], [], []
    for protocol in plan['screen']['protocols']:
        folder = output/protocol
        folder.mkdir()
        folds = [(s['block'], np.array(s['train']), np.array(s['validation'])) for s in splits if s['protocol'] == protocol]
        blocks = [(int(val[0]), int(val[-1])+1) for _, _, val in folds]
        arrays, identities = selected_arrays(protocol, 42, len(raw))
        hashes.update(identities)
        reference = decode_motion_dead_trim(*arrays, bc, info, raw, blocks)[public]
        prior = upstream/'dead_trim_confirmation'/protocol/'oof_predictions.csv'
        hashes[str(prior.relative_to(ROOT))] = digest(prior)
        np.testing.assert_array_equal(reference, pd.read_csv(prior).motion_dead_trim_seed42)
        table = pd.DataFrame(dict(frame=public, ground_truth=labels[public], current_seed42=reference))
        record_predictions(output, protocol, 'current_seed42', public, labels, folds, reference, rows, details, segments)
        add_overlap_metrics(rows, segments, table, protocol, 'current_seed42')
        for head in ['matched_control', *plan['heads']]:
            inputs = views[head]
            probability = arrays[4].copy() if head == 'matched_control' else np.full((len(raw), 3), np.nan)
            if head != 'matched_control':
                for block, train, val in folds:
                    if np.any((train >= val.min()-250) & (train <= val.max()+250)):
                        raise AssertionError('Detector-feature target buffer violation')
                    hidden = np.full(len(labels), 'X', dtype='<U1')
                    hidden[train] = labels[train]
                    print(protocol, head, 'block', block, flush=True)
                    model = fit_renewal(inputs, hidden, train, 'frame', 42)
                    model.save_model(folder/f'{head}_seed42_block{block}.ubj')
                    y, weight = fitting_weights(hidden, train, False)
                    np.savez(folder/f'{head}_seed42_block{block}_fitting.npz',
                             train=train, validation=val, augmented_target=y, augmented_weight=weight)
                    probability[val] = predict_models([model], inputs.iloc[val])
            np.save(folder/f'{head}_seed42_probabilities.npy', probability[public])
            for mode in plan['screen']['decoders']:
                prediction = decode_provided_speed(arrays, probability, bc, info, raw, blocks, mode)[public]
                name = f'{head}_{mode}_seed42'
                table[name] = prediction
                record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
                add_overlap_metrics(rows, segments, table, protocol, name)
        add_short_metrics(rows, segments)
        table.to_csv(folder/'oof_predictions.csv', index=False)
        pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
        pd.DataFrame(details).to_csv(output/'fold_scores.csv', index=False)
        pd.DataFrame(segments).to_csv(output/'segment_diagnostics.csv', index=False)
    if sources != {name:digest(ROOT/'experiments'/name) for name in names}:
        raise ValueError('Source changed during detector-feature screen')
    cache_paths = [p for p in sorted(output.rglob('*')) if p.is_file() and p.suffix in ('.csv', '.npy', '.npz', '.json')]
    manifest = dict(plan_sha256=digest(plan_path), source_sha256=sources, input_sha256=hashes,
        cache_sha256={str(p.relative_to(output)):digest(p) for p in cache_paths},
        model_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*.ubj'))},
        feature_columns={name:list(value) for name, value in views.items()}, new_models=26,
        matched_control_no_new_fitting=True, training_only_public_targets=True, selected_eight_reference_exact=True)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(pd.DataFrame(rows)[['protocol', 'model', 'spellF1', 'macroF1', 'false_dead_frames',
        'short_live_mean_best_iou', 'short_live_matched_iou50']].round(6).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
