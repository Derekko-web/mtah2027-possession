"""Official replay, training isolation and frozen ownership-graph acceptance gates."""

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_broadcast_candidate import check_sources
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest, official_scores
from motion_dead_trim import decode_motion_dead_trim
from ownership_graph import decode_graph, fitting_weights
from research_ownership_graph import selected_arrays
from temporal_candidate import predict_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-history', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_ownership_graph_v1'
    folder = root/'screen'
    manifest = audit_manifest(folder)
    cm = audit_manifest(root/'inputs')
    plan_path = ROOT/'experiments/OWNERSHIP_GRAPH_PLAN.json'
    plan = json.loads(plan_path.read_text())
    if manifest['plan_sha256'] != digest(plan_path) or cm['plan_sha256'] != digest(plan_path):
        raise ValueError('Frozen graph plan changed')
    for stage in (folder, root/'inputs'):
        if (stage/'plan.json').read_bytes() != plan_path.read_bytes():
            raise ValueError('Stage graph plan differs')
    for name, expected in manifest['model_sha256'].items():
        if digest(folder/name) != expected:
            raise ValueError('Graph checkpoint changed')
    prior = json.loads((ROOT/'data/candidate_motion_dead_trim/manifest.json').read_text())
    check_sources(prior['source_sha256'])
    check_sources(prior['reference_source_sha256'])
    upstream = ROOT/'data/control_pair_motion_v1'
    if digest(upstream/'inputs/manifest.json') != prior['prefix_cache_manifest_sha256']:
        raise ValueError('Selected motion input changed')
    audit_manifest(upstream/'inputs')
    audit_manifest(upstream/'dead_trim_confirmation')
    audit_manifest(ROOT/'data/control_broadcast_v1/inputs')
    source = ROOT/'data/advanced_validation'
    raw, ordered = [pd.read_pickle(source/(name+'.pkl')) for name in ('raw', 'ordered')]
    bc = pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:len(raw)]
    motion_info = pd.read_pickle(upstream/'inputs/motion_info.pkl').iloc[:len(raw)]
    features = pd.read_pickle(root/'inputs/graph_inputs.pkl').iloc[:len(raw)]
    info = pd.read_pickle(root/'inputs/graph_info.pkl').iloc[:len(raw)]
    views = dict(ordered=ordered, graph=ordered.join(bc).join(features))
    for name, value in views.items():
        if list(value) != manifest['feature_columns'][name]:
            raise ValueError('Graph feature schema changed')
    public = np.arange(11185, 18685)
    scores = pd.read_csv(folder/'scores.csv')
    segments = pd.read_csv(folder/'segment_diagnostics.csv')
    details = pd.read_csv(folder/'fold_scores.csv')
    if len(scores) != 21 or scores.duplicated(['protocol', 'model']).any():
        raise ValueError('Incomplete or duplicated graph screen')
    official = official_scores(folder, scores)
    index, counts = scores.set_index(['protocol', 'model']), official.set_index(['protocol', 'model'])
    splits = json.loads((folder/'splits.json').read_text())
    reloads, gates, deltas, diagnostics = 0, [], [], []
    for protocol in plan['screen']['protocols']:
        folds = [s for s in splits if s['protocol'] == protocol]
        blocks = [(min(s['validation']), max(s['validation'])+1) for s in folds]
        arrays, _ = selected_arrays(protocol, 42, len(raw))
        table = pd.read_csv(folder/protocol/'oof_predictions.csv')
        reference = decode_motion_dead_trim(*arrays, bc, motion_info, raw, blocks)[public]
        np.testing.assert_array_equal(reference, table.current_seed42)
        labels = np.full(199319, 'X', dtype='<U1')
        labels[public] = table.ground_truth
        for head, config in plan['heads'].items():
            inputs = views[config['view']]
            probability = np.full((len(raw), 3), np.nan)
            for fold in folds:
                train, val = np.array(fold['train']), np.array(fold['validation'])
                if np.any((train >= val.min()-250) & (train <= val.max()+250)):
                    raise AssertionError('Graph target buffer violated')
                saved = np.load(folder/protocol/f"{head}_seed42_block{fold['block']}_fitting.npz")
                np.testing.assert_array_equal(saved['train'], train)
                np.testing.assert_array_equal(saved['validation'], val)
                hidden = np.full(len(labels), 'X', dtype='<U1')
                hidden[train] = labels[train]
                y, weight = fitting_weights(hidden, train, config['transition_weight'])
                np.testing.assert_array_equal(saved['augmented_target'], y)
                np.testing.assert_array_equal(saved['augmented_weight'], weight)
                model = XGBClassifier()
                model.load_model(folder/protocol/f"{head}_seed42_block{fold['block']}.ubj")
                probability[val] = predict_models([model], inputs.iloc[val])
                reloads += 1
            np.testing.assert_array_equal(probability[public], np.load(folder/protocol/f'{head}_seed42_probabilities.npy'))
            true_short = segments[(segments.protocol == protocol) & (segments.model == 'current_seed42') & (segments.label != 'D') & (segments.frames <= 125)]
            for case in true_short.itertuples():
                selection = (public >= case.start) & (public <= case.end)
                expected = 0 if case.label == 'H' else 1
                value = probability[public][selection]
                diagnostics.append(dict(protocol=protocol, head=head, label=case.label, frames=case.frames,
                    raw_argmax_correct_fraction=float(np.mean(value.argmax(axis=1) == expected))))
            for mode in plan['decoders']:
                name = f'{head}_{mode}_seed42'
                prediction = decode_graph(arrays, probability, bc, motion_info, info, raw, blocks, mode)[public]
                np.testing.assert_array_equal(prediction, table[name])
                if mode == 'short_proposals':
                    np.testing.assert_array_equal(prediction == 'D', reference == 'D')
                old, new = index.loc[(protocol, 'current_seed42')], index.loc[(protocol, name)]
                a, b = counts.loc[(protocol, 'current_seed42')], counts.loc[(protocol, name)]
                short = segments[(segments.protocol == protocol) & (segments.label != 'D') & (segments.frames <= 125)]
                before = short[short.model == 'current_seed42'].set_index(['start', 'end', 'label'])
                after = short[short.model == name].set_index(['start', 'end', 'label'])
                pd.testing.assert_index_equal(before.index, after.index)
                flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12), macro=bool(new.macroF1 >= old.macroF1-.001-1e-12),
                    short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                    previous_short=bool((after.recall[before.recall >= .5] >= .5).all()),
                    falseD=bool(new.false_dead_frames <= old.false_dead_frames), Drecall=bool(new.dead_recall >= old.dead_recall-.02-1e-12),
                    stoppages=bool(new.missed_stoppages <= old.missed_stoppages), spurious=bool(b.spurious_iou50 <= a.spurious_iou50),
                    missed=bool(b.missed_iou50 <= a.missed_iou50))
                if protocol == 'buffered_minutes':
                    flags['substantial'] = bool(new.spellF1 >= old.spellF1+.04-1e-12)
                    flags['extra_short'] = bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1)
                gates.append(dict(protocol=protocol, model=name, checks=flags, primary_delta=float(new.spellF1-old.spellF1)))
                old_blocks = details[(details.protocol == protocol) & (details.model == 'current_seed42')]
                new_blocks = details[(details.protocol == protocol) & (details.model == name)]
                joined = old_blocks.merge(new_blocks, on=['protocol', 'minute'], suffixes=('_before', '_after'), validate='one_to_one')
                for row in joined.itertuples():
                    deltas.append(dict(protocol=protocol, model=name, block=row.minute, delta=row.spellF1_after-row.spellF1_before))
    eligible = [name for name in sorted(set(r['model'] for r in gates))
                if all(all(r['checks'].values()) for r in gates if r['model'] == name)]
    (root/'screen_checks.json').write_text(json.dumps(dict(eligible_finalists=eligible, pairs=gates), indent=2)+'\n')
    pd.DataFrame(deltas).to_csv(root/'block_deltas.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(root/'raw_head_short_diagnostics.csv', index=False)
    ledger = pd.read_csv(ROOT/'experiments/EXPERIMENT_LEDGER.csv')
    previous_count = None
    if args.audit_history:
        with tarfile.open(root/'experiments_before.tar.gz') as archive:
            before = pd.read_csv(io.BytesIO(archive.extractfile('experiments/EXPERIMENT_LEDGER.csv').read()))
        keys = ['round', 'protocol', 'model', 'source']
        joined = ledger.merge(before[keys], on=keys, how='inner', validate='one_to_one')
        pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True), joined[before.columns].sort_values(keys).reset_index(drop=True))
        previous_count = len(before)
        for name, expected in json.loads((root/'baseline_manifest.json').read_text()).items():
            if name == 'experiments/EXPERIMENT_LEDGER.csv':
                continue
            if name == 'experiments/SELECTED_MODEL.md' and digest(ROOT/name) != expected:
                with tarfile.open(root/'experiments_before.tar.gz') as archive:
                    original = archive.extractfile('experiments/SELECTED_MODEL.md').read()
                import hashlib
                if hashlib.sha256(original).hexdigest() != expected or prior['csv_sha256'] not in (ROOT/name).read_text():
                    raise ValueError('Protected selected entry lost')
            elif digest(ROOT/name) != expected:
                raise ValueError('Protected graph reference changed')
    if digest(ROOT/'data/continuation_20261021/annotation_freeze/windows.json') != '5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c':
        raise ValueError('Reserved annotation metadata changed')
    result = dict(official_score_rows=21, official_score_agreement=True, shared_inference_replay=True,
        models_reload_exactly=reloads, training_weights_and_targets_eligible_only=True, training_buffers_verified=True,
        padded_prefix_parity=cm['actual_padded_prefix_parity'], source_input_cache_hashes_verified=True,
        unchanged_eight_model_reference_verified=True, proposal_D_masks_exact=True, historical_snapshot_audited=args.audit_history,
        old_ledger_rows=previous_count, old_ledger_rows_preserved=True if args.audit_history else None,
        ledger_rows=len(ledger), selected_artifact_unchanged=True if args.audit_history else None,
        clipped_block_comparisons=len(deltas), clipped_block_primary_regressions=sum(r['delta'] < -1e-12 for r in deltas), eligible_finalists=eligible)
    (root/'integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
