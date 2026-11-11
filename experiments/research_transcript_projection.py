"""Project a fixed latent proposal onto the selected model's predicted transcript."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation import possession_metrics as pm
from player_representation import (
    ROUND as REPRESENTATION, COLUMNS, PROTOCOLS, audit_cache,
    check_inputs as check_upstream, digest, interpolate, write_json,
)
from research_representation_ensemble import latent_probability
from research_player_representation import decode as decode_proposal
from research_ownership_graph import selected_arrays
from research_stoppages import record_predictions
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_temporal_dead import assess
from dfl_signal_audit import matching
from finalize_broadcast_closure import official_scores
from motion_dead_trim import decode_motion_dead_trim
from transcript_projection import project_transcript

ROUND = ROOT / 'data/control_transcript_projection_v1'
PLAN = ROOT / 'experiments/TRANSCRIPT_PROJECTION_PLAN.json'


def check_inputs():
    check_upstream()
    freeze = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (freeze / 'plan.json').read_bytes():
        raise ValueError('Projection plan changed')
    plan = json.loads(PLAN.read_text())
    for group in (plan['input_sha256'],
                  json.loads((freeze / 'implementation.json').read_text()),
                  json.loads((freeze / 'protected_artifacts.json').read_text())):
        for name, expected in group.items():
            if digest(ROOT / name) != expected:
                raise ValueError('Frozen dependency changed: ' + name)
    if not (ROOT / 'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith(
            (freeze / 'EXPERIMENT_LEDGER.csv').read_bytes()):
        raise ValueError('Previous ledger bytes changed')
    for name, target in json.loads((freeze / 'symlinks.json').read_text()).items():
        path = ROOT / name
        if not path.is_symlink() or str(path.readlink()) != target:
            raise ValueError('Source symlink changed')
    return plan


def run(seed, replay, override):
    plan = check_inputs()
    for folder in ('inputs', 'encodings', 'screen'):
        audit_cache(REPRESENTATION / folder)
    stage = 'screen' if seed == 42 else f'confirmation_seed{seed}'
    saved = ROUND / stage
    if seed != 42 and not json.loads((ROUND / 'screen/selection.json').read_text())['passed']:
        raise ValueError('No eligible projected candidate')
    if replay:
        audit_cache(saved)
    output = override or ROUND / (stage + '_replay' if replay else stage)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'plan.json').write_bytes(PLAN.read_bytes())
    source = ROOT / 'data/advanced_validation'
    raw, ordered = [pd.read_pickle(source / (name + '.pkl')) for name in ('raw', 'ordered')]
    n = len(raw)
    enc = REPRESENTATION / 'encodings'
    latent, eligible = interpolate(np.load(enc / 'frames.npy'),
        np.load(enc / 'pretrained_latent.npy'), np.load(enc / 'eligible.npy'), n)
    inputs = ordered.join(pd.DataFrame(latent, columns=COLUMNS, index=ordered.index))
    broadcast = pd.read_pickle(ROOT / 'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:n]
    motion = pd.read_pickle(ROOT / 'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:n]
    labels = pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    public = np.flatnonzero(labels != 'X')
    np.testing.assert_array_equal(public, np.arange(11185, 18685))
    splits = json.loads((source / 'splits.json').read_text())
    rows, details, segments, states, changes = [], [], [], {}, {}
    new_models = 0
    input_hashes = dict(plan['input_sha256'])
    for protocol in PROTOCOLS:
        folder = output / protocol
        folder.mkdir()
        folds = [(s['block'], np.asarray(s['train']), np.asarray(s['validation']))
                 for s in splits if s['protocol'] == protocol]
        blocks = [(int(v[0]), int(v[-1]) + 1) for _, _, v in folds]
        arrays, hashes = selected_arrays(protocol, seed, n)
        input_hashes.update(hashes)
        reference = decode_motion_dead_trim(*arrays, broadcast, motion, raw, blocks)
        expected = pd.read_csv(ROOT / 'data/control_pair_motion_v1/dead_trim_confirmation'
            / protocol / 'oof_predictions.csv')[f'motion_dead_trim_seed{seed}']
        np.testing.assert_array_equal(reference[public], expected)
        probability, fitted = latent_probability(seed, replay, saved, folder, protocol,
            folds, inputs, labels, public, eligible, arrays[0])
        new_models += fitted
        proposal = decode_proposal(arrays, probability, broadcast, motion, raw, blocks)
        if seed == 42:
            old = pd.read_csv(REPRESENTATION / 'screen' / protocol / 'oof_predictions.csv')
            np.testing.assert_array_equal(proposal[public], old.pretrained_latent_seed42)
        prediction = project_transcript(reference, proposal, blocks)
        table = pd.DataFrame(dict(frame=public, ground_truth=labels[public]))
        current = f'current_seed{seed}'
        for name, value in ((current, reference), (f'latent_proposal_seed{seed}', proposal),
                            (f'projected_seed{seed}', prediction)):
            table[name] = value[public]
            record_predictions(output, protocol, name, public, labels, folds,
                               value[public], rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, name)
            if name != current:
                states[(protocol, name)] = (matching(labels[public], reference[public]),
                                            matching(labels[public], value[public]))
                changes[(protocol, name)] = int((value[public] != reference[public]).sum())
        table.to_csv(folder / 'oof_predictions.csv', index=False)
    add_short_metrics(rows, segments)
    scores, diagnostics = pd.DataFrame(rows), pd.DataFrame(segments)
    scores.to_csv(output / 'scores.csv', index=False)
    pd.DataFrame(details).to_csv(output / 'fold_scores.csv', index=False)
    diagnostics.to_csv(output / 'segment_diagnostics.csv', index=False)
    official = official_scores(output, scores)
    selection = assess(scores, diagnostics, official, states, changes, seed)
    for pair in selection['pairs']:
        if pair['protocol'] == 'buffered_minutes' and seed == 42:
            value = scores[(scores.protocol == pair['protocol']) &
                           (scores.model == pair['model'])].spellF1.item()
            pair['checks']['strict_historical_best_gain'] = bool(
                value > plan['historical_best_buffered'] + 1e-12)
    method = f'projected_seed{seed}'
    pairs = [p for p in selection['pairs'] if p['model'] == method]
    passed = len(pairs) == 3 and all(all(p['checks'].values()) for p in pairs)
    selection.update(passed=passed, eligible=[method] if passed else [],
                     selected=method if passed else None,
                     historical_substantial_and_extra_short_required=False)
    write_json(output / 'selection.json', selection)
    if replay:
        for name in ('scores.csv', 'fold_scores.csv', 'segment_diagnostics.csv',
                     'official_comparison.csv', 'selection.json'):
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError('Score replay differs: ' + name)
        for protocol in PROTOCOLS:
            name = Path(protocol) / 'oof_predictions.csv'
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError('Prediction replay differs')
    write_json(output / 'manifest.json', dict(plan_sha256=digest(PLAN),
        new_models=new_models, manual_labels_used=False, targets_in_projection=False,
        input_sha256=input_hashes,
        source_sha256=json.loads((ROUND / 'protocol_freeze/implementation.json').read_text()),
        cache_sha256={str(p.relative_to(output)): digest(p)
                      for p in sorted(output.rglob('*')) if p.is_file()}))
    check_inputs()
    print(scores[['protocol', 'model', 'spellF1', 'macroF1',
                  'short_live_matched_iou50']].to_string(index=False), flush=True)
    print('Eligible:', selection['eligible'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.seed, args.replay, args.output)
