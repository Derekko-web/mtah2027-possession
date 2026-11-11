"""Frozen twelve-row optical/color contact comparison; no new fitting."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from build_temporal_candidate import digest
from color_contact import predict_contact_block
from pair_candidate import codes
from xgboost_baseline import CLASSES
from detector_ledger import append_score_rows
from finalize_broadcast_closure import audit_manifest, official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions


def main():
    root = ROOT/'data/control_color_contact_v1'
    plan_path = ROOT/'experiments/COLOR_CONTACT_PLAN.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen parent detector evidence changed')
    parent = ROOT/'data/control_detector_features_v1'
    audit_manifest(parent/'screen')
    audit_manifest(parent/'inputs')
    audit_manifest(root/'inputs')
    prepared = json.loads((root/'inputs/manifest.json').read_text())
    if not prepared['support_passed'] or not prepared['actual_padded_prefix_parity']:
        raise ValueError('Contact support/prefix preflight failed')
    contact = {name:np.load(root/'inputs'/(name+'.npy')) for name in plan['variants']}
    output = root/'screen'
    output.mkdir()
    (output/'plan.json').write_bytes(plan_path.read_bytes())
    source = ROOT/'data/advanced_validation'
    raw = pd.read_pickle(source/'raw.pkl')
    bc = pd.read_pickle(ROOT/'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:len(raw)]
    info = pd.read_pickle(ROOT/'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:len(raw)]
    labels = pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    public = np.flatnonzero(labels != 'X')
    if not np.array_equal(public, np.arange(11185, 18685)):
        raise ValueError('Public labels differ')
    split_path = source/'splits.json'
    splits = json.loads(split_path.read_text())
    (output/'splits.json').write_bytes(split_path.read_bytes())
    hashes = dict(plan['input_sha256'])
    hashes['data/control_color_contact_v1/inputs/manifest.json'] = digest(root/'inputs/manifest.json')
    for path in (source/'raw.pkl', split_path, ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'):
        hashes[str(path.relative_to(ROOT))] = digest(path)
    rows, details, segments = [], [], []
    for protocol in plan['protocols']:
        (output/protocol).mkdir()
        folds = [(s['block'], np.array(s['train']), np.array(s['validation'])) for s in splits if s['protocol'] == protocol]
        blocks = [(int(v[0]), int(v[-1])+1) for _, _, v in folds]
        arrays, identities = selected_arrays(protocol, 42, len(raw))
        hashes.update(identities)
        reference = decode_motion_dead_trim(*arrays, bc, info, raw, blocks)[public]
        old = pd.read_csv(parent/'screen'/protocol/'oof_predictions.csv')
        if not np.array_equal(reference, old.current_seed42):
            raise AssertionError('Identity slot does not reproduce selected8')
        table = pd.DataFrame(dict(frame=public, ground_truth=labels[public], current_seed42=reference))
        for head in ['current', *plan['variants']]:
            prediction = reference.copy()
            name = 'current_seed42' if head == 'current' else head+'_seed42'
            if head != 'current':
                for start, end in blocks:
                    mask = (public >= start) & (public < end)
                    r = codes(reference[mask])
                    evidence = contact[head][start:end]
                    result = predict_contact_block(r, evidence)
                    swapped = np.where(r == 2, 2, 1-r)
                    mirror = predict_contact_block(swapped, evidence[:, ::-1])
                    if not np.array_equal(mirror, np.where(result == 2, 2, 1-result)):
                        raise AssertionError('Contact team symmetry failed')
                    # Actual source-prefix extent must leave the entire public block exact.
                    prefix = contact[head][:18935][start:end]
                    if not np.array_equal(result, predict_contact_block(r, prefix)):
                        raise AssertionError('Contact memory prefix parity failed')
                    prediction[mask] = CLASSES[result]
            if not np.array_equal(prediction == 'D', reference == 'D'):
                raise AssertionError('Contact proposals changed D')
            table[name] = prediction
            record_predictions(output, protocol, name, public, labels, folds, prediction, rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, name)
        add_short_metrics(rows, segments)
        table.to_csv(output/protocol/'oof_predictions.csv', index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'scores.csv', index=False)
    pd.DataFrame(details).to_csv(output/'fold_scores.csv', index=False)
    segment_table = pd.DataFrame(segments)
    segment_table.to_csv(output/'segment_diagnostics.csv', index=False)
    official = official_scores(output, scores)
    index, counts = scores.set_index(['protocol', 'model']), official.set_index(['protocol', 'model'])
    gates = []
    for protocol in plan['protocols']:
        old, a = index.loc[(protocol, 'current_seed42')], counts.loc[(protocol, 'current_seed42')]
        short = segment_table[(segment_table.protocol == protocol) & (segment_table.label != 'D') & (segment_table.frames <= 125)]
        before = short[short.model == 'current_seed42'].set_index(['start', 'end', 'label'])
        for head in plan['variants']:
            name = head+'_seed42'
            new, b = index.loc[(protocol, name)], counts.loc[(protocol, name)]
            after = short[short.model == name].set_index(['start', 'end', 'label'])
            pd.testing.assert_index_equal(before.index, after.index)
            flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),
                macro=bool(new.macroF1 >= old.macroF1-.001-1e-12),
                short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                previous_short=bool((after.recall[before.recall >= .5] >= .5).all()),
                falseD=bool(new.false_dead_frames <= old.false_dead_frames),
                Drecall=bool(new.dead_recall >= old.dead_recall-.02-1e-12),
                stoppages=bool(new.missed_stoppages <= old.missed_stoppages),
                spurious=bool(b.spurious_iou50 <= a.spurious_iou50), missed=bool(b.missed_iou50 <= a.missed_iou50))
            if head == 'detected_color':
                flags['matched_controls'] = all(new.spellF1 >= index.loc[(protocol, control+'_seed42')].spellF1-1e-12
                    for control in ('detected_metadata', 'tracked_color'))
            if protocol == 'buffered_minutes':
                flags.update(substantial=bool(new.spellF1 >= old.spellF1+.04-1e-12),
                             extra_short=bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1))
            gates.append(dict(protocol=protocol, model=name, checks=flags, primary_delta=float(new.spellF1-old.spellF1)))
    eligible = [head+'_seed42' for head in plan['variants']
                if all(all(r['checks'].values()) for r in gates if r['model'] == head+'_seed42')]
    (root/'screen_checks.json').write_text(json.dumps(dict(eligible_finalists=eligible, pairs=gates), indent=2)+'\n')
    names = ('research_color_contact.py', 'color_contact.py', 'motion_dead_trim.py', 'broadcast_refinement.py', 'pair_candidate.py', 'stoppage_consensus.py')
    (output/'manifest.json').write_text(json.dumps(dict(plan_sha256=digest(plan_path),
        source_sha256={name:digest(ROOT/'experiments'/name) for name in names}, input_sha256=hashes,
        cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*.csv'))},
        no_new_fitting=True, identity_control_exact=True, all_D_masks_exact=True), indent=2)+'\n')
    ledger = append_score_rows(ROOT/'experiments/EXPERIMENT_LEDGER.csv', output/'scores.csv',
        'color_contact_v1', 'data/control_color_contact_v1/screen/scores.csv')
    (root/'integrity.json').write_text(json.dumps(dict(official_rows=12, official_agreement=True,
        identity_control_exact=True, all_D_masks_exact=True, new_fits=0,
        eligible_finalists=eligible, ledger_append=ledger), indent=2)+'\n')
    print(scores[['protocol', 'model', 'spellF1', 'macroF1', 'short_live_matched_iou50']].round(6).to_string(index=False))
    print('Eligible finalists:', eligible)


if __name__ == '__main__':
    main()
