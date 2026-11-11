"""Frozen native TabICLv2 context screen and independent-context score replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tabicl_native import ROUND, PLAN, PUBLIC, context_record, digest, write_json, supervise
from player_representation import (ROUND as REPRESENTATION, COLUMNS, PROTOCOLS,
    check_inputs as check_representation, audit_cache, interpolate)
from evaluation import possession_metrics as pm
from rich_features import swap_teams
from research_ownership_graph import selected_arrays
from motion_dead_trim import decode_motion_dead_trim
from factorized_latent import decode_owner
from research_stoppages import record_predictions
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_temporal_dead import assess
from dfl_signal_audit import matching
from finalize_broadcast_closure import official_scores


def check_inputs():
    check_representation()
    freeze = ROUND / 'protocol_freeze'
    if PLAN.read_bytes() != (freeze / 'plan.json').read_bytes():
        raise ValueError('Native context plan changed')
    plan = json.loads(PLAN.read_text())
    for group in (plan['input_sha256'], json.loads((freeze / 'implementation.json').read_text()),
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
            raise ValueError('Source link changed')
    proof = ROOT / 'data/control_tabicl_serial_v1/runtime_assessment/summary.json'
    if not json.loads(proof.read_text())['passed']:
        raise ValueError('Independent context reconstruction prerequisite failed')
    return plan


def run(seed, replay, override):
    plan = check_inputs()
    for name in ('inputs', 'encodings'):
        audit_cache(REPRESENTATION / name)
    stage = 'screen' if seed == 42 else f'confirmation_seed{seed}'
    saved = ROUND / stage
    if seed != 42 and not json.loads((ROUND / 'screen/selection.json').read_text())['passed']:
        raise ValueError('No eligible seed42 method')
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
    if inputs.shape[1] != 166:
        raise ValueError('Fixed166-feature schema differs')
    broadcast = pd.read_pickle(ROOT / 'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:n]
    motion = pd.read_pickle(ROOT / 'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:n]
    labels = pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels != 'X'), PUBLIC)
    splits = json.loads((source / 'splits.json').read_text())
    rows, details, segments, states, changes = [], [], [], {}, {}
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
        previous = pd.read_csv(ROOT / 'data/control_pair_motion_v1/dead_trim_confirmation' /
            protocol / 'oof_predictions.csv')[f'motion_dead_trim_seed{seed}']
        np.testing.assert_array_equal(reference[PUBLIC], previous)
        owner = np.full((n, 2), np.nan)
        jobs = []
        for block, train, validation in folds:
            job_dir = folder / f'block{block}'
            job_dir.mkdir()
            context, target, imputer, record = context_record(inputs, labels, train, validation)
            masked = np.full_like(labels, 'X')
            masked[train] = labels[train]
            second_context, second_target, second_imputer, second_record = context_record(
                inputs, masked, train, validation)
            np.testing.assert_array_equal(context, second_context)
            np.testing.assert_array_equal(target, second_target)
            for key, value in record.items():
                np.testing.assert_array_equal(value, second_record[key])
            joblib.dump(imputer, job_dir / 'imputer.joblib')
            np.savez_compressed(job_dir / 'fitting.npz', **record)
            queries = (imputer.transform(inputs.iloc[validation]).astype(np.float32),
                       imputer.transform(swap_teams(inputs.iloc[validation])).astype(np.float32))
            payloads = dict(context=context, target=target, original=queries[0], swapped=queries[1])
            for name, value in payloads.items():
                if not np.isfinite(value).all():
                    raise ValueError('Unimputed worker input')
                np.save(job_dir / (name + '.npy'), value)
            parent = saved / protocol / f'block{block}'
            if replay:
                with np.load(parent / 'fitting.npz') as old:
                    for key, value in record.items():
                        np.testing.assert_array_equal(old[key], value)
                for name, value in payloads.items():
                    np.testing.assert_array_equal(value, np.load(parent / (name + '.npy')))
            job = {name: str(job_dir / (name + '.npy')) for name in payloads}
            job.update(seed=seed, input_sha256={str(job_dir / (name + '.npy')):
                                               digest(job_dir / (name + '.npy')) for name in payloads})
            proof = ROOT / 'data/control_tabicl_serial_v1/runtime_assessment/summary.json'
            job['input_sha256'][str(proof)] = digest(proof)
            if replay:
                job['parent'] = str(parent / 'worker')
            write_json(job_dir / 'job.json', job)
            jobs.append((block, validation, job_dir, len(context)))
        # Fit independent folds in fixed pairs, keeping at most two one-thread
        # workers and bounding their combined RSS before scheduling another pair.
        for start in range(0, len(jobs), 2):
            pair = jobs[start:start+2]
            with ThreadPoolExecutor(max_workers=2) as executor:
                pending = []
                for block, validation, job_dir, count in pair:
                    print('Rebuild' if replay else 'Fit', 'TabICLv2', protocol, 'block', block,
                          'context rows', count, 'queries', len(validation), flush=True)
                    pending.append(executor.submit(supervise, job_dir / 'job.json', job_dir / 'worker'))
                for future in pending:
                    future.result()
            for block, validation, job_dir, count in pair:
                value = np.load(job_dir / 'worker/owner.npy')
                if value.shape != (len(validation), 2):
                    raise ValueError('Worker rows differ')
                owner[validation] = value
        np.save(folder / 'owner_probability.npy', owner[PUBLIC])
        prediction = decode_owner(arrays, owner, eligible, broadcast, motion, raw, blocks)
        table = pd.DataFrame(dict(frame=PUBLIC, ground_truth=labels[PUBLIC]))
        current, method = f'current_seed{seed}', f'tabicl_context_seed{seed}'
        for name, value in ((current, reference), (method, prediction)):
            table[name] = value[PUBLIC]
            record_predictions(output, protocol, name, PUBLIC, labels, folds, value[PUBLIC], rows, details, segments)
            add_overlap_metrics(rows, segments, table, protocol, name)
        states[(protocol, method)] = (matching(labels[PUBLIC], reference[PUBLIC]),
                                      matching(labels[PUBLIC], prediction[PUBLIC]))
        changes[(protocol, method)] = int((prediction[PUBLIC] != reference[PUBLIC]).sum())
        table.to_csv(folder / 'oof_predictions.csv', index=False)
    add_short_metrics(rows, segments)
    scores, diagnostics = pd.DataFrame(rows), pd.DataFrame(segments)
    scores.to_csv(output / 'scores.csv', index=False)
    diagnostics.to_csv(output / 'segment_diagnostics.csv', index=False)
    pd.DataFrame(details).to_csv(output / 'fold_scores.csv', index=False)
    official = official_scores(output, scores)
    selection = assess(scores, diagnostics, official, states, changes, seed)
    for pair in selection['pairs']:
        if pair['protocol'] == 'buffered_minutes' and seed == 42:
            value = scores[(scores.protocol == pair['protocol']) & (scores.model == pair['model'])].spellF1.item()
            pair['checks']['strict_historical_best_gain'] = bool(value > plan['historical_best_buffered'] + 1e-12)
    method = f'tabicl_context_seed{seed}'
    passed = len(selection['pairs']) == 3 and all(all(p['checks'].values()) for p in selection['pairs'])
    selection.update(passed=passed, eligible=[method] if passed else [], selected=method if passed else None,
                     historical_substantial_and_extra_short_required=False)
    write_json(output / 'selection.json', selection)
    if replay:
        for name in ('scores.csv', 'fold_scores.csv', 'segment_diagnostics.csv', 'official_comparison.csv', 'selection.json'):
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError('Score/gate replay differs: ' + name)
        for protocol in PROTOCOLS:
            name = Path(protocol) / 'oof_predictions.csv'
            if (output / name).read_bytes() != (saved / name).read_bytes():
                raise ValueError('Prediction replay differs')
    write_json(output / 'manifest.json', dict(plan_sha256=digest(PLAN), seed=seed, context_models=13,
        manual_labels_used=False, validation_labels_supplied_to_worker=False,
        retained_D_X_exact=True, context_regeneration_used=True, fitted_pickle_reconstruction_used=False,
        input_sha256=input_hashes,
        source_sha256=json.loads((ROUND / 'protocol_freeze/implementation.json').read_text()),
        cache_sha256={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    check_inputs()
    print(scores[['protocol', 'model', 'spellF1', 'macroF1', 'short_live_matched_iou50']].to_string(index=False), flush=True)
    print('Eligible:', selection['eligible'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.seed, args.replay, args.output)
