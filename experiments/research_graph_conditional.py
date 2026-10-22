"""Reuse frozen graph heads for21 conditional/source-extent decoder checks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest, official_scores
from graph_conditional import decode_graph_conditional
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = ROOT/'data/control_ownership_graph_v1'
    output = root/'conditional_screen'
    parent = audit_manifest(root/'screen')
    audit_manifest(root/'inputs')
    for name, expected in parent['model_sha256'].items():
        if digest(root/'screen'/name) != expected:
            raise ValueError('Parent graph model changed')
    plan_path = ROOT/'experiments/GRAPH_CONDITIONAL_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen conditional input changed')
    if args.verify_only:
        saved = audit_manifest(output)
        if saved['plan_sha256'] != digest(plan_path) or (output/'plan.json').read_bytes() != plan_path.read_bytes():
            raise ValueError('Conditional screen plan differs')
    else:
        output.mkdir(exist_ok=False)
        (output/'plan.json').write_bytes(plan_path.read_bytes())
        (output/'splits.json').write_bytes((root/'screen/splits.json').read_bytes())
    source = ROOT/'data/advanced_validation'
    raw = pd.read_pickle(source/'raw.pkl')
    bc = pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:len(raw)]
    motion_info = pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:len(raw)]
    info = pd.read_pickle(root/'inputs/graph_info.pkl').iloc[:len(raw)]
    public = np.arange(11185, 18685)
    labels = np.full(199319, 'X', dtype='<U1')
    splits = json.loads((root/'screen/splits.json').read_text())
    hashes = {name:digest(ROOT/name) for name in plan['input_sha256']}
    rows, details, segments = [], [], []
    for protocol in plan['protocols']:
        folder = output/protocol
        if not args.verify_only:
            folder.mkdir()
        folds = [(s['block'], np.array(s['train']), np.array(s['validation'])) for s in splits if s['protocol'] == protocol]
        blocks = [(int(val[0]), int(val[-1])+1) for _, _, val in folds]
        arrays, identity = selected_arrays(protocol, 42, len(raw))
        hashes.update(identity)
        original_table = pd.read_csv(root/'screen'/protocol/'oof_predictions.csv')
        labels[public] = original_table.ground_truth
        reference = decode_motion_dead_trim(*arrays, bc, motion_info, raw, blocks)[public]
        np.testing.assert_array_equal(reference, original_table.current_seed42)
        table = pd.read_csv(folder/'oof_predictions.csv') if args.verify_only else pd.DataFrame(dict(frame=public, ground_truth=labels[public], current_seed42=reference))
        np.testing.assert_array_equal(table.current_seed42, reference)
        if not args.verify_only:
            record_predictions(output, protocol, 'current_seed42', public, labels, folds, reference, rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, 'current_seed42')
        for head in plan['heads']:
            path = root/'screen'/protocol/f'{head}_seed42_probabilities.npy'
            hashes[str(path.relative_to(ROOT))] = digest(path)
            probability = np.full((len(raw), 3), np.nan)
            probability[public] = np.load(path)
            for mode in plan['methods']:
                name = f'{head}_{mode}_seed42'
                prediction = decode_graph_conditional(arrays, probability, bc, motion_info, info, raw, blocks, mode)[public]
                if args.verify_only:
                    np.testing.assert_array_equal(prediction, table[name])
                else:
                    table[name] = prediction
                    record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
                    add_overlap_metrics(rows, segments, table, protocol, name)
                if mode == 'source_extent':
                    np.testing.assert_array_equal(prediction == 'D', reference == 'D')
        if not args.verify_only:
            add_short_metrics(rows, segments)
            table.to_csv(folder/'oof_predictions.csv', index=False)
            pd.DataFrame(rows).to_csv(output/'scores.csv', index=False)
            pd.DataFrame(details).to_csv(output/'fold_scores.csv', index=False)
            pd.DataFrame(segments).to_csv(output/'segment_diagnostics.csv', index=False)
    if not args.verify_only:
        names = ('research_graph_conditional.py', 'graph_conditional.py')
        result = dict(plan_sha256=digest(plan_path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
            input_sha256=hashes, new_model_fits=0, reused_checkpoints=39, reference_eight_exact=True)
        (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    scores, segments, details = [pd.read_csv(output/name) for name in ('scores.csv', 'segment_diagnostics.csv', 'fold_scores.csv')]
    if len(scores) != 21 or scores.duplicated(['protocol', 'model']).any():
        raise ValueError('Incomplete conditional decoder screen')
    official = official_scores(output, scores)
    index, counts = scores.set_index(['protocol', 'model']), official.set_index(['protocol', 'model'])
    gates, deltas = [], []
    for protocol in plan['protocols']:
        for head in plan['heads']:
            for mode in plan['methods']:
                name = f'{head}_{mode}_seed42'
                old, new = index.loc[(protocol, 'current_seed42')], index.loc[(protocol, name)]
                a, b = counts.loc[(protocol, 'current_seed42')], counts.loc[(protocol, name)]
                short = segments[(segments.protocol == protocol) & (segments.label != 'D') & (segments.frames <= 125)]
                before = short[short.model == 'current_seed42'].set_index(['start', 'end', 'label'])
                after = short[short.model == name].set_index(['start', 'end', 'label'])
                pd.testing.assert_index_equal(before.index, after.index)
                flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12), macro=bool(new.macroF1 >= old.macroF1-.001-1e-12),
                    short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                    prior_short=bool((after.recall[before.recall >= .5] >= .5).all()), falseD=bool(new.false_dead_frames <= old.false_dead_frames),
                    Drecall=bool(new.dead_recall >= old.dead_recall-.02-1e-12), missed_stoppages=bool(new.missed_stoppages <= old.missed_stoppages),
                    spurious=bool(b.spurious_iou50 <= a.spurious_iou50), missed=bool(b.missed_iou50 <= a.missed_iou50))
                if protocol == 'buffered_minutes':
                    flags['substantial'] = bool(new.spellF1 >= old.spellF1+.04-1e-12)
                    flags['extra_short'] = bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1)
                gates.append(dict(protocol=protocol, model=name, checks=flags, primary_delta=float(new.spellF1-old.spellF1)))
                pair = details[(details.protocol == protocol) & (details.model == 'current_seed42')].merge(
                    details[(details.protocol == protocol) & (details.model == name)], on=['protocol', 'minute'], suffixes=('_before', '_after'), validate='one_to_one')
                for row in pair.itertuples():
                    deltas.append(dict(protocol=protocol, model=name, block=row.minute, delta=row.spellF1_after-row.spellF1_before))
    eligible = [name for name in sorted(set(r['model'] for r in gates)) if all(all(r['checks'].values()) for r in gates if r['model'] == name)]
    (root/'conditional_checks.json').write_text(json.dumps(dict(eligible_finalists=eligible, pairs=gates), indent=2)+'\n')
    pd.DataFrame(deltas).to_csv(root/'conditional_block_deltas.csv', index=False)
    result = dict(official_score_rows=21, official_score_agreement=True, shared_inference_replay=True,
                  new_model_fits=0, parent_model_hashes_verified=39, source_extent_D_masks_exact=True,
                  clipped_block_comparisons=len(deltas), clipped_block_primary_regressions=sum(r['delta'] < -1e-12 for r in deltas), eligible_finalists=eligible)
    (root/'conditional_integrity.json').write_text(json.dumps(result, indent=2)+'\n')
    print(scores[['protocol', 'model', 'spellF1', 'macroF1', 'false_dead_frames', 'short_live_mean_best_iou', 'short_live_matched_iou50']].round(6).to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
