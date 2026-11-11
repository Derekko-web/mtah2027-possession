"""Official scores, complete model replay and frozen detector-feature gates."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import hashlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_broadcast_candidate import check_sources
from build_temporal_candidate import digest
from detector_ledger import append_score_rows
from finalize_broadcast_closure import audit_manifest, official_scores
from motion_dead_trim import decode_motion_dead_trim
from ownership_graph import fitting_weights
from provided_speed import decode_provided_speed
from research_spatial_shape import input_views
from research_ownership_graph import selected_arrays
from temporal_candidate import predict_models


def same(a, b, message):
    numeric = np.asarray(a).dtype.kind not in 'OUS'
    equal = np.array_equal(a, b, equal_nan=True) if numeric else np.array_equal(a, b)
    if not equal:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-history', action='store_true')
    parser.add_argument('--record-ledger', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_spatial_shape_v1'
    folder = root/'screen'
    manifest, cache = audit_manifest(folder), audit_manifest(root/'inputs')
    plan_path = ROOT/'experiments/SPATIAL_SHAPE_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for stage, record in ((folder, manifest), (root/'inputs', cache)):
        if record['plan_sha256'] != digest(plan_path) or (stage/'plan.json').read_bytes() != plan_path.read_bytes():
            raise ValueError('Detector feature plan changed')
    for name, expected in manifest['model_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Detector feature checkpoint changed')
    prior = json.loads((ROOT/'data/candidate_motion_dead_trim/manifest.json').read_text())
    check_sources(prior['source_sha256'])
    check_sources(prior['reference_source_sha256'])
    upstream = ROOT/'data/control_pair_motion_v1'
    audit_manifest(upstream/'inputs')
    audit_manifest(upstream/'dead_trim_confirmation')
    control_folder = ROOT/'data/control_broadcast_v1/closure_confirmation'
    control_manifest = audit_manifest(control_folder)
    source = ROOT/'data/advanced_validation'
    raw, ordered = [pd.read_pickle(source/(name+'.pkl')) for name in ('raw', 'ordered')]
    bc = pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:len(raw)]
    info = pd.read_pickle(upstream/'inputs/motion_info.pkl').iloc[:len(raw)]
    views = input_views(root, ordered, bc)
    for name, view in views.items():
        if list(view) != manifest['feature_columns'][name]:
            raise ValueError('Detector feature schema changed')
    if list(views['matched_control']) != control_manifest['feature_columns']:
        raise ValueError('Matched control schema differs')
    scores = pd.read_csv(folder/'scores.csv')
    segments = pd.read_csv(folder/'segment_diagnostics.csv')
    details = pd.read_csv(folder/'fold_scores.csv')
    if len(scores) != 30 or scores.duplicated(['protocol', 'model']).any() or len(manifest['model_sha256']) != 26:
        raise ValueError('Incomplete detector feature experiment')
    official = official_scores(folder, scores)
    index, counts = scores.set_index(['protocol', 'model']), official.set_index(['protocol', 'model'])
    splits = json.loads((folder/'splits.json').read_text())
    public = np.arange(11185, 18685)
    gates, deltas, diagnostics = [], [], []
    reloads, control_reloads, control_hashes = 0, 0, {}
    for protocol in plan['screen']['protocols']:
        folds = [s for s in splits if s['protocol'] == protocol]
        blocks = [(min(s['validation']), max(s['validation'])+1) for s in folds]
        arrays, _ = selected_arrays(protocol, 42, len(raw))
        table = pd.read_csv(folder/protocol/'oof_predictions.csv')
        same(table.frame.to_numpy(), public, 'Public prediction frame identity differs')
        reference = decode_motion_dead_trim(*arrays, bc, info, raw, blocks)[public]
        same(reference, table.current_seed42.to_numpy(), 'Selected8 reference replay differs')
        labels = np.full(199319, 'X', dtype='<U1')
        labels[public] = table.ground_truth
        for head in ['matched_control', *plan['heads']]:
            inputs = views[head]
            probability = np.full((len(raw), 3), np.nan)
            for fold in folds:
                train, val = np.array(fold['train']), np.array(fold['validation'])
                if np.any((train >= val.min()-250) & (train <= val.max()+250)):
                    raise AssertionError('Detector training exclusion violated')
                if head == 'matched_control':
                    path = control_folder/protocol/f"broadcast_joint_frame_seed42_block{fold['block']}.ubj"
                    control_hashes[str(path.relative_to(ROOT))] = digest(path)
                    control_reloads += 1
                else:
                    path = folder/protocol/f"{head}_seed42_block{fold['block']}.ubj"
                    saved = np.load(folder/protocol/f"{head}_seed42_block{fold['block']}_fitting.npz")
                    same(saved['train'], train, 'Training indices differ')
                    same(saved['validation'], val, 'Validation indices differ')
                    hidden = np.full(len(labels), 'X', dtype='<U1')
                    hidden[train] = labels[train]
                    y, weight = fitting_weights(hidden, train, False)
                    same(saved['augmented_target'], y, 'Training-only target reconstruction differs')
                    same(saved['augmented_weight'], weight, 'Training-only weights differ')
                    reloads += 1
                model = XGBClassifier()
                model.load_model(path)
                probability[val] = predict_models([model], inputs.iloc[val])
            same(probability[public], np.load(folder/protocol/f'{head}_seed42_probabilities.npy'), 'Model probability reload differs')
            if head == 'matched_control':
                same(probability[public], arrays[4][public], 'Reused no-vision control differs from source models')
            short_cases = segments[(segments.protocol == protocol) & (segments.model == 'current_seed42') & (segments.label != 'D') & (segments.frames <= 125)]
            for case in short_cases.itertuples():
                use = (public >= case.start) & (public <= case.end)
                expected = 0 if case.label == 'H' else 1
                values = probability[public][use]
                diagnostics.append(dict(protocol=protocol, head=head, label=case.label, frames=case.frames,
                    raw_argmax_correct_fraction=float((values.argmax(axis=1) == expected).mean()),
                    mean_conditional_correct_probability=float((values[:, expected]/values[:, :2].sum(axis=1)).mean()),
                    mean_dead_probability=float(values[:, 2].mean())))
            for mode in plan['screen']['decoders']:
                name = f'{head}_{mode}_seed42'
                prediction = decode_provided_speed(arrays, probability, bc, info, raw, blocks, mode)[public]
                same(prediction, table[name].to_numpy(), 'Shared detector decoder replay differs')
                if head == 'matched_control':
                    continue
                old, new = index.loc[(protocol, 'current_seed42')], index.loc[(protocol, name)]
                control = index.loc[(protocol, f'matched_control_{mode}_seed42')]
                a, b = counts.loc[(protocol, 'current_seed42')], counts.loc[(protocol, name)]
                short = segments[(segments.protocol == protocol) & (segments.label != 'D') & (segments.frames <= 125)]
                before = short[short.model == 'current_seed42'].set_index(['start', 'end', 'label'])
                after = short[short.model == name].set_index(['start', 'end', 'label'])
                pd.testing.assert_index_equal(before.index, after.index)
                flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),
                    matched_control=bool(new.spellF1 >= control.spellF1-1e-12),
                    macro=bool(new.macroF1 >= old.macroF1-.001-1e-12),
                    short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                    previous_short=bool((after.recall[before.recall >= .5] >= .5).all()),
                    falseD=bool(new.false_dead_frames <= old.false_dead_frames),
                    Drecall=bool(new.dead_recall >= old.dead_recall-.02-1e-12),
                    stoppages=bool(new.missed_stoppages <= old.missed_stoppages),
                    spurious=bool(b.spurious_iou50 <= a.spurious_iou50), missed=bool(b.missed_iou50 <= a.missed_iou50))
                if protocol == 'buffered_minutes':
                    flags.update(substantial=bool(new.spellF1 >= old.spellF1+.04-1e-12),
                                 extra_short=bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1))
                gates.append(dict(protocol=protocol, model=name, checks=flags, primary_delta=float(new.spellF1-old.spellF1)))
                joined = details[(details.protocol == protocol) & (details.model == 'current_seed42')].merge(
                    details[(details.protocol == protocol) & (details.model == name)], on=['protocol', 'minute'],
                    suffixes=('_before', '_after'), validate='one_to_one')
                deltas.extend(dict(protocol=protocol, model=name, block=r.minute, delta=r.spellF1_after-r.spellF1_before) for r in joined.itertuples())
    eligible = [name for name in sorted({row['model'] for row in gates})
                if len([row for row in gates if row['model'] == name]) == 3 and
                all(all(row['checks'].values()) for row in gates if row['model'] == name)]
    (root/'screen_checks.json').write_text(json.dumps(dict(eligible_finalists=eligible, pairs=gates), indent=2)+'\n')
    pd.DataFrame(deltas).to_csv(root/'block_deltas.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(root/'raw_head_short_diagnostics.csv', index=False)
    ledger_result = None
    if args.record_ledger:
        ledger_result = append_score_rows(ROOT/'experiments/EXPERIMENT_LEDGER.csv', folder/'scores.csv',
            'spatial_shape_v1', 'data/control_spatial_shape_v1/screen/scores.csv')
    if args.audit_history:
        baseline = json.loads((root/'baseline_manifest.json').read_text())
        for name, expected in baseline.items():
            if name.startswith('experiments/'):
                continue
            if digest(ROOT/name) != expected:
                raise AssertionError('Protected selected/reference candidate artifact changed')
        with tarfile.open(root/'experiments_before.tar.gz') as archive:
            original = archive.extractfile('experiments/EXPERIMENT_LEDGER.csv').read()
            if hashlib.sha256(original).hexdigest() != baseline['experiments/EXPERIMENT_LEDGER.csv'] or not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith(original):
                raise AssertionError('Previous ledger bytes were not preserved')
    result = dict(official_score_rows=30, official_score_agreement=True, new_model_reload_exact=reloads,
        existing_control_model_reload_exact=control_reloads, control_model_sha256=control_hashes,
        shared_decoder_predictions_exact=True, training_targets_weights_and_buffers_verified=True,
        actual_padded_prefix_parity=cache['actual_padded_prefix_parity'], source_input_cache_hashes_verified=True,
        eligible_finalists=eligible, block_comparisons=len(deltas),
        block_primary_regressions=sum(r['delta'] < -1e-12 for r in deltas),
        ledger_append=ledger_result, history_audited=args.audit_history)
    (root/'integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k, v in result.items() if k != 'control_model_sha256'}, indent=2))


if __name__ == '__main__':
    main()
