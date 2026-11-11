"""R58 staged continuation: synthetic, source control, 13 fold encoders, then heads."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from native_encoder_continuation import (ROOT, ROUND, PLAN, PYTHON, HEAD_PYTHON, PUBLIC, PROTOCOLS,
    SOURCE, REPRESENTATION, INITIAL, CONFIG, audit, check_inputs, digest, write_json,
    seal, ha_targets, source_schedule, native_schedule, permitted_centers, metrics,
    retention, retention_diagnostics, retention_eligibility, rank_eligible, MethodFailure, verify_runtime)

sys.path.insert(0, str(ROOT))


def method_closed(method):
    path = ROUND / (method+'_method_failure.json')
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get('method') != method or not record.get('closed'):
        raise ValueError('Malformed retained method-failure evidence')
    for name, expected in record['evidence_sha256'].items():
        if digest(ROUND / name) != expected:
            raise ValueError('Retained failed-method evidence changed')
    return True


def close_method(method, stage, error):
    check_inputs()
    path = ROUND / (method+'_method_failure.json')
    if path.exists():
        raise ValueError('A closed method cannot be retried')
    folders = [ROUND / 'neural' / method]
    if method == 'native':
        folders.append(ROUND / 'training_inputs')
    files = [p for folder in folders if folder.exists() for p in folder.rglob('*') if p.is_file()]
    if method == 'source' and (ROUND / 'neural').exists():
        files += [p for p in (ROUND / 'neural').glob('source*') if p.is_file()]
    write_json(path, dict(method=method, stage=stage, closed=True, reason=str(error),
        no_retry=True, no_partial_fold_fallback=True,
        evidence_sha256={str(p.relative_to(ROUND)): digest(p) for p in sorted(files)}))


def require(path):
    if not Path(path).exists() or not json.loads(Path(path).read_text())['passed']:
        raise ValueError('Required frozen stage has not passed: ' + str(path))


def prepare_source():
    plan = check_inputs()
    verify_runtime('head')
    require(ROUND / 'synthetic/summary.json')
    original = SOURCE / 'corpus'
    audit(original)
    segments = json.loads((original / 'segments.json').read_text())
    selected = [s for s in segments if s['match'] in plan['source_train_matches']]
    if len(selected) != 8:
        raise ValueError('Four source matches / eight halves required')
    indices = np.concatenate([np.arange(s['start'], s['end']) for s in selected])
    out = ROUND / 'source'
    out.mkdir(parents=True, exist_ok=False)
    # Copy only source-training observations; no validation observations enter workers.
    for name, axis in [('xy', 1), ('observed', 1), ('teams', 0), ('half_id', 0)]:
        value = np.load(original / (name+'.npy'), mmap_mode='r')
        np.save(out / (name+'.npy'), np.take(value, indices, axis=axis))
    observation_hashes = {n: digest(out / (n+'.npy')) for n in ('xy', 'observed', 'teams', 'half_id')}
    write_json(out / 'observations_before_targets.json', observation_hashes)
    labels = np.load(original / 'labels.npy')[indices]
    np.save(out / 'labels.npy', labels)
    offset, buckets, record = 0, {}, []
    for segment in selected:
        length = segment['end']-segment['start']
        centers = np.arange(offset+15, offset+length-15)
        game = segment['match']
        for target in (0, 1):
            buckets.setdefault((game, target), []).extend(centers[labels[centers] == target])
        record.append(dict(match=game, start=offset, end=offset+length,
                           original_start=segment['start'], original_end=segment['end']))
        offset += length
    draws = source_schedule({k: np.asarray(v, int) for k, v in buckets.items()})
    np.savez_compressed(out / 'schedule.npz', **draws)
    write_json(out / 'segments.json', record)
    for name, expected in observation_hashes.items():
        if digest(out / (name+'.npy')) != expected:
            raise ValueError('Target join changed observations')
    seal(out, source_matches=plan['source_train_matches'], validation_matches_excluded=True,
         source_schedule_match_class_team_balanced=True, source_label_orientation='H=1,A=0')
    check_inputs()


def prepare_native():
    check_inputs()
    verify_runtime('head')
    audit(ROUND / 'control_assessment')
    if method_closed('native'):
        raise ValueError('Native method already closed')
    audit(REPRESENTATION / 'inputs')
    from masked_player_native_data import assemble_native
    from evaluation import possession_metrics as pm
    source = REPRESENTATION / 'inputs'
    x, seen, teams = [np.load(source / (name+'.npy'), mmap_mode='r')
                      for name in ('xy', 'observed', 'teams')]
    half = np.load(source / 'half_id.npy')
    frames, support = [np.load(source / (name+'.npy')) for name in ('frames', 'eligible')]
    labels = pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    splits = json.loads((ROOT / 'data/advanced_validation/splits.json').read_text())
    out = ROUND / 'training_inputs'
    out.mkdir(parents=True, exist_ok=False)
    fold_count = 0
    for split in splits:
        protocol, block = split['protocol'], split['block']
        if protocol not in PROTOCOLS:
            continue
        train, validation = np.asarray(split['train']), np.asarray(split['validation'])
        centers, observation_eligible = permitted_centers(frames, train, validation, labels, half, support, len(labels))
        if not len(centers) or not all((labels[centers] == k).any() for k in ('H', 'A')):
            close_method('native', 'prepare', MethodFailure('Empty admissible native H/A class pool: '+protocol+f'/block{block}'))
            return
        masked_labels = np.full_like(labels, 'X')
        masked_labels[train] = labels[train]
        again, _ = permitted_centers(frames, train, validation, masked_labels, half, support, len(labels))
        np.testing.assert_array_equal(centers, again)
        value = assemble_native(x, seen, teams, half, centers // 5)
        folder = out / protocol / f'block{block}'
        folder.mkdir(parents=True)
        target = ha_targets(labels[centers])
        np.savez_compressed(folder / 'pool.npz', xy=value[0], observed=value[1], teams=value[2], target=target)
        np.savez_compressed(folder / 'fitting.npz', train=train, validation=validation,
                            centers=centers, observation_eligible=observation_eligible, target=target)
        # Same sampling rule/seed in every fold, rather than selecting a seed from outcomes.
        np.savez_compressed(folder / 'schedule.npz', **native_schedule(target))
        fold_count += 1
    if fold_count != 13:
        raise ValueError('Exactly thirteen outer splits required')
    seal(out, folds=13, query_observations_supplied_to_workers=False,
         outside_train_label_invariance=True, radius=79, public_train_HA_only=True)
    check_inputs()


def fit_one(mode, folder, destination, parent=None):
    check_inputs()
    audit(ROUND / 'source')
    source = ROUND / 'source'
    payload = dict(mode=mode, source=str(source), source_schedule=str(source / 'schedule.npz'),
                   pool=str(folder / 'pool.npz') if folder else None,
                   native_schedule=str(folder / 'schedule.npz') if folder else None,
                   parent=str(parent) if parent else None)
    files = [source / (n+'.npy') for n in ('xy', 'observed', 'teams', 'half_id', 'labels')]
    files += [source / 'schedule.npz', INITIAL]
    if parent:
        audit(parent)
        files += [parent / n for n in ('model.pt', 'training_loss.npy', 'probe_probability.npy')]
    if folder:
        audit(ROUND / 'training_inputs')
        files += [folder / n for n in ('pool.npz', 'schedule.npz', 'fitting.npz')]
    payload['input_sha256'] = {str(p): digest(p) for p in files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    job = destination.parent / (destination.name+'_job.json')
    write_json(job, payload)
    stdout = destination.parent / (destination.name+'.stdout.log')
    stderr = destination.parent / (destination.name+'.stderr.log')
    started = time.monotonic()
    try:
        with stdout.open('wb') as a, stderr.open('wb') as b:
            result = subprocess.run([str(PYTHON), '-B', str(ROOT / 'experiments/native_encoder_continuation.py'),
                '--job', str(job), '--output', str(destination)], cwd=ROOT,
                env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                     'OPENBLAS_NUM_THREADS': '2', 'PYTHONHASHSEED': '42'}, stdout=a, stderr=b, timeout=1000)
    except subprocess.TimeoutExpired:
        write_json(destination.parent / (destination.name+'_supervision.json'),
                   dict(exit_code=None, aborted='worker_over_1000seconds', seconds=time.monotonic()-started, no_retry=True))
        raise MethodFailure('Fitting worker exceeded 1000-second supervision deadline')
    write_json(destination.parent / (destination.name+'_supervision.json'),
               dict(exit_code=result.returncode, seconds=time.monotonic()-started, no_retry=True))
    if result.returncode == 3:
        raise MethodFailure('Numerical/resource worker failure; retained evidence: '+str(destination))
    if result.returncode:
        raise RuntimeError('Continuation worker failed; evidence retained: ' + str(destination))
    audit(destination)


def fit_native(replay=False):
    check_inputs()
    audit(ROUND / 'control_assessment')
    output = ROUND / ('neural_replay' if replay else 'neural')
    eligible = json.loads((ROUND / 'external_assessment/selection.json').read_text())['eligible'] if replay else []
    if replay and 'source_continuation' in eligible:
        fit_one('source', None, output / 'source', ROUND / 'neural/source')
    if replay and 'native_continuation' not in eligible:
        return
    if method_closed('native'):
        if replay:
            raise ValueError('Cannot replay a closed native method')
        return
    audit(ROUND / 'training_inputs')
    for protocol in PROTOCOLS:
        for folder in sorted((ROUND / 'training_inputs' / protocol).iterdir()):
            target = output / 'native' / protocol / folder.name
            parent = ROUND / 'neural/native' / protocol / folder.name if replay else None
            try:
                fit_one('native', folder, target, parent)
            except MethodFailure as error:
                if replay:
                    raise
                close_method('native', 'fit', error)
                return


def assess_external(control_only=False, replay=False):
    check_inputs()
    verify_runtime('torch')
    import torch
    from masked_player_network import MaskedPlayer, initialize
    from fit_masked_player import infer
    initialize()
    corpus = SOURCE / 'corpus'
    audit(corpus)
    root = ROUND / ('neural_replay' if replay else 'neural')
    previous = json.loads((ROUND / 'external_assessment/selection.json').read_text()) if replay else None
    source_available = not method_closed('source')
    native_available = not method_closed('native')
    models = [('source', root / 'source')] if source_available and (not replay or 'source_continuation' in previous['eligible']) else []
    if not control_only and native_available and (not replay or 'native_continuation' in previous['eligible']):
        models += [(f'{protocol}_{folder.name}', folder)
            for protocol in PROTOCOLS for folder in sorted((root / 'native' / protocol).iterdir()) if folder.is_dir()]
    expected = int(source_available) if control_only else (sum(1 if m == 'source_continuation' else 13 for m in previous['eligible']) if replay else int(source_available)+13*int(native_available))
    if len(models) != expected:
        raise ValueError('Incomplete neural continuation collection')
    out = ROUND / ('control_assessment' if control_only else 'external_assessment' + ('_replay' if replay else ''))
    out.mkdir(parents=True, exist_ok=False)
    values = {k: np.load(corpus / (k+'.npy'), mmap_mode='r') for k in ('xy', 'observed', 'teams', 'half_id')}
    validation = dict(np.load(corpus / 'validation.npz'))
    matches = json.loads((ROOT / 'experiments/MASKED_PLAYER_EXTERNAL_PLAN.json').read_text())['matches']
    report, model_hashes = [], {}
    for name, folder in models:
        audit(folder)
        model_hashes[name] = digest(folder / 'model.pt')
        model = MaskedPlayer()
        model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        loaded = MaskedPlayer()
        loaded.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        for view, choices in [('clean', np.zeros(len(validation['centers']), int)),
                              ('masked', 1+((validation['local']//10) % 3))]:
            print('External final-checkpoint assessment', name, view, flush=True)
            args = [values[k] for k in ('xy', 'observed', 'teams', 'half_id')]
            p = infer(model, *args, validation['centers'], choices, 'temporal')
            np.testing.assert_array_equal(p, infer(loaded, *args, validation['centers'], choices, 'temporal'))
            np.save(out / (name+'_'+view+'_probability.npy'), p)
            baseline = np.load(SOURCE / 'external_models/temporal' / (view+'_probability.npy'))
            for group in np.unique(validation['match']):
                use = validation['match'] == group
                new, old = metrics(p[use], validation['target'][use]), metrics(baseline[use], validation['target'][use])
                report.append(dict(model=name, match=matches[int(group)], view=view,
                                   metrics=new, baseline=old, checks=retention(new, old),
                                   diagnostics=retention_diagnostics(new, old)))
    methods, eligible = retention_eligibility(report)
    selection = dict(passed=bool(eligible), eligible=eligible, methods=methods,
        models=expected, final_checkpoint_only=True, pairs=report, model_sha256=model_hashes)
    write_json(out / 'selection.json', selection)
    seal(out, strict_reload_exact=True, disjoint_external_matches=True,
         source_validation_ranks_checkpoints=False, native_query_labels_loaded=False)
    if replay:
        old = ROUND / 'external_assessment'
        for p in out.glob('*_probability.npy'):
            if p.read_bytes() != (old / p.name).read_bytes():
                raise ValueError('External continuation replay differs')
        # Serialized file metadata may differ; compare actual metrics/checks too.
        expected_rows = [r for r in previous['pairs'] if (r['model'] == 'source') == ('source_continuation' in previous['eligible'])] if len(previous['eligible']) == 1 else previous['pairs']
        if selection['pairs'] != expected_rows or selection['eligible'] != previous['eligible']:
            raise ValueError('External retention gate replay differs')
    if not selection['passed'] and not control_only:
        raise ValueError('Final checkpoint external retention gate failed; no rescue or head scoring')
    check_inputs()


def encode_worker(folder, destination, parent=None):
    check_inputs()
    verify_runtime('torch')
    folder, destination = Path(folder).resolve(), Path(destination).resolve()
    replay = folder.is_relative_to(ROUND / 'neural_replay')
    model_root = ROUND / ('neural_replay' if replay else 'neural')
    gate = json.loads((ROUND / ('external_assessment_replay' if replay else 'external_assessment') / 'selection.json').read_text())
    if folder == model_root / 'source':
        method, key = 'source_continuation', 'source'
    elif folder.is_relative_to(model_root / 'native'):
        relative = folder.relative_to(model_root / 'native')
        if len(relative.parts) != 2:
            raise ValueError('Unexpected fold encoder path')
        method, key = 'native_continuation', '_'.join(relative.parts)
    else:
        raise ValueError('Only fixed R58 encoder checkpoints may be encoded')
    if method not in gate['eligible'] or digest(folder / 'model.pt') != gate['model_sha256'][key]:
        raise ValueError('Encoder did not pass its own external retention gate')
    if not destination.is_relative_to(ROUND):
        raise ValueError('Encoding destination must remain in the R58 evidence folder')
    import torch
    from masked_player_network import MaskedPlayer, initialize
    from masked_player_native_data import assemble_native
    from encode_player_representation import reflected
    initialize()
    audit(folder)
    source = REPRESENTATION / 'inputs'
    audit(source)
    frames = np.load(source / 'frames.npy')
    half = np.load(source / 'half_id.npy')
    values = [np.load(source / (k+'.npy'), mmap_mode='r') for k in ('xy', 'observed', 'teams')]
    prefix = [np.load(source / ('prefix_'+k+'.npy'), mmap_mode='r') for k in ('xy', 'observed', 'teams')]
    model = MaskedPlayer()
    model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
    destination.mkdir(parents=True, exist_ok=False)
    result = []
    for start in range(0, len(frames), 64):
        indices = np.arange(start, min(start+64, len(frames)))
        size = len(indices)
        if size < 64:
            indices = np.r_[indices, np.repeat(indices[-1], 64-size)]
        full = assemble_native(*values, half, frames[indices]//5)
        small = [np.asarray(v[indices]) for v in prefix]
        for a, b in zip(full, small):
            np.testing.assert_array_equal(a, b)
        hidden = reflected(model, full)
        np.testing.assert_array_equal(hidden, reflected(model, small))
        if start == 0:
            swapped = reflected(model, (full[0], full[1], 1-full[2]))
            np.testing.assert_array_equal(swapped, hidden[:, np.r_[32:64, 0:32]])
        result.append(hidden[:size])
    hidden = np.concatenate(result)
    np.save(destination / 'hidden.npy', hidden)
    np.save(destination / 'frames.npy', frames)
    np.save(destination / 'eligible.npy', np.load(source / 'eligible.npy'))
    if parent:
        for name in ('hidden.npy', 'frames.npy', 'eligible.npy'):
            if (destination / name).read_bytes() != (parent / name).read_bytes():
                raise ValueError('Adapted representation replay differs: ' + name)
    seal(destination, checkpoint_sha256=digest(folder / 'model.pt'), prefix_hidden_exact=True,
         team_swap_exact=True, targets_loaded=False, radius=79)
    return hidden


def encode(folder, destination, parent=None):
    """Head controller delegates Torch inference to its existing isolated runtime."""
    command = [str(PYTHON), '-B', str(Path(__file__).resolve()), '--stage', 'encode',
               '--model-folder', str(folder), '--destination', str(destination)]
    if parent is not None:
        command += ['--parent', str(parent)]
    subprocess.run(command, cwd=ROOT, check=True, env={**os.environ,
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'PYTHONHASHSEED': '42'})
    audit(destination)
    return np.load(destination / 'hidden.npy')


def screen(seed=42, replay=False):
    plan = check_inputs()
    verify_runtime('head')
    require(ROUND / ('external_assessment_replay' if replay else 'external_assessment') / 'selection.json')
    if seed != 42:
        require(ROUND / 'screen/selection.json')
    import pandas as pd
    from xgboost import XGBClassifier
    from player_representation import COLUMNS, interpolate
    from evaluation import possession_metrics as pm
    from build_candidate import objective_weights
    from temporal_candidate import fit_models, predict_models
    from research_player_representation import decode
    from research_ownership_graph import selected_arrays
    from motion_dead_trim import decode_motion_dead_trim
    from research_stoppages import record_predictions
    from research_boundary_refinement import add_overlap_metrics
    from research_control_refinement import add_short_metrics
    from research_temporal_dead import assess
    from dfl_signal_audit import matching
    from finalize_broadcast_closure import official_scores
    stage = 'screen' if seed == 42 else f'confirmation_seed{seed}'
    old = ROUND / stage
    if replay:
        audit(old)
    out = ROUND / (stage+'_replay' if replay else stage)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_bytes(PLAN.read_bytes())
    models = ROUND / ('neural_replay' if replay else 'neural')
    base = ROOT / 'data/advanced_validation'
    raw, ordered = [pd.read_pickle(base / (k+'.pkl')) for k in ('raw', 'ordered')]
    n = len(raw)
    bc = pd.read_pickle(ROOT / 'data/control_broadcast_v1/inputs/combined_inputs.pkl').iloc[:n]
    info = pd.read_pickle(ROOT / 'data/control_pair_motion_v1/inputs/motion_info.pkl').iloc[:n]
    labels = pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels != 'X'), PUBLIC)
    splits = json.loads((base / 'splits.json').read_text())
    frames = np.load(REPRESENTATION / 'inputs/frames.npy')
    support = np.load(REPRESENTATION / 'inputs/eligible.npy')
    source_hidden = None
    eligible_methods = json.loads((ROUND / ('external_assessment_replay' if replay else 'external_assessment') / 'selection.json').read_text())['eligible']
    if seed != 42:
        main = json.loads((ROUND / 'screen/selection.json').read_text())
        eligible_methods = [name.rsplit('_seed', 1)[0] for name in main['eligible']]
    if seed == 42 and 'source_continuation' in eligible_methods:
        source_hidden = encode(models / 'source', out / 'encodings/source', old / 'encodings/source' if replay else None)
    elif seed != 42 and 'source_continuation' in eligible_methods:
        audit(ROUND / 'screen/encodings/source')
        source_hidden = np.load(ROUND / 'screen/encodings/source/hidden.npy')
    rows, detail, segments, states, changes = [], [], [], {}, {}
    for protocol in PROTOCOLS:
        folder = out / protocol
        folder.mkdir()
        folds = [(s['block'], np.asarray(s['train']), np.asarray(s['validation'])) for s in splits if s['protocol'] == protocol]
        blocks = [(int(v[0]), int(v[-1])+1) for _, _, v in folds]
        arrays, _ = selected_arrays(protocol, seed, n)
        reference = decode_motion_dead_trim(*arrays, bc, info, raw, blocks)
        expected = pd.read_csv(ROOT / 'data/control_pair_motion_v1/dead_trim_confirmation' / protocol / 'oof_predictions.csv')[f'motion_dead_trim_seed{seed}']
        np.testing.assert_array_equal(reference[PUBLIC], expected)
        views = tuple(eligible_methods)
        probabilities = {view: np.full((n, 3), np.nan) for view in views}
        for block, train, validation in folds:
            key = Path(protocol) / f'block{block}'
            if seed == 42 and 'native_continuation' in views:
                native_hidden = encode(models / 'native' / key, out / 'encodings/native' / key,
                    old / 'encodings/native' / key if replay else None)
            elif 'native_continuation' in views:
                enc = ROUND / 'screen/encodings/native' / key
                audit(enc)
                native_hidden = np.load(enc / 'hidden.npy')
            for view in views:
                hidden = source_hidden if view == 'source_continuation' else native_hidden
                latent, eligible = interpolate(frames, hidden, support, n)
                inputs = ordered.join(pd.DataFrame(latent, columns=COLUMNS, index=ordered.index))
                if inputs.shape[1] != 166:
                    raise ValueError('R37 fixed 166-column interface changed')
                target = np.asarray([{'H': 0, 'A': 1, 'D': 2}[v] for v in labels[train]])
                y, weight = objective_weights(target, train, False)
                fitting_labels = np.full_like(labels, 'X')
                fitting_labels[train] = labels[train]
                path = folder / f'{view}_seed{seed}_block{block}.ubj'
                record = dict(train=train, validation=validation, target=y, weight=weight)
                np.savez_compressed(str(path)+'.fitting.npz', **record)
                if replay:
                    with np.load(str(old / protocol / path.name)+'.fitting.npz') as prior:
                        for k, value in record.items():
                            np.testing.assert_array_equal(value, prior[k])
                    model = XGBClassifier()
                    model.load_model(old / protocol / path.name)
                    shutil.copy2(old / protocol / path.name, path)
                else:
                    model = fit_models('ordered_xgboost_frame', inputs, fitting_labels, train, seed)[0]
                    model.save_model(path)
                p = predict_models([model], inputs.iloc[validation])
                loaded = XGBClassifier()
                loaded.load_model(path)
                np.testing.assert_array_equal(p, predict_models([loaded], inputs.iloc[validation]))
                probabilities[view][validation] = p
        table = pd.DataFrame(dict(frame=PUBLIC, ground_truth=labels[PUBLIC]))
        methods = {f'current_seed{seed}': reference}
        if seed == 42:
            r37 = np.full((n, 3), np.nan)
            r37[PUBLIC] = np.load(REPRESENTATION / 'screen' / protocol / 'pretrained_latent_probability.npy')
            methods['r37_pretrained_seed42'] = decode(arrays, r37, bc, info, raw, blocks)
            previous_r37 = pd.read_csv(REPRESENTATION / 'screen' / protocol / 'oof_predictions.csv').pretrained_latent_seed42
            np.testing.assert_array_equal(methods['r37_pretrained_seed42'][PUBLIC], previous_r37)
        for view, p in probabilities.items():
            p[PUBLIC[~eligible[PUBLIC]]] = arrays[0][PUBLIC[~eligible[PUBLIC]]]
            np.save(folder / (view+'_probability.npy'), p[PUBLIC])
            methods[f'{view}_seed{seed}'] = decode(arrays, p, bc, info, raw, blocks)
        for name, prediction in methods.items():
            np.testing.assert_array_equal(prediction[~np.isin(reference, ['H', 'A'])], reference[~np.isin(reference, ['H', 'A'])])
            table[name] = prediction[PUBLIC]
            record_predictions(out, protocol, name, PUBLIC, labels, folds, prediction[PUBLIC], rows, detail, segments)
            add_overlap_metrics(rows, segments, table, protocol, name)
            if name != f'current_seed{seed}':
                states[(protocol, name)] = (matching(labels[PUBLIC], reference[PUBLIC]), matching(labels[PUBLIC], prediction[PUBLIC]))
                changes[(protocol, name)] = int((prediction[PUBLIC] != reference[PUBLIC]).sum())
        table.to_csv(folder / 'oof_predictions.csv', index=False)
    add_short_metrics(rows, segments)
    scores, diagnostics = pd.DataFrame(rows), pd.DataFrame(segments)
    scores.to_csv(out / 'scores.csv', index=False)
    diagnostics.to_csv(out / 'segment_diagnostics.csv', index=False)
    pd.DataFrame(detail).to_csv(out / 'fold_scores.csv', index=False)
    official = official_scores(out, scores)
    selection = assess(scores, diagnostics, official, states, changes, seed)
    methods = [f'{m}_seed{seed}' for m in eligible_methods]
    pairs = [p for p in selection['pairs'] if p['model'] in methods]
    for pair in pairs:
        protocol = pair['protocol']
        index = scores[scores.protocol == protocol].set_index('model')
        if seed == 42 and protocol == 'buffered_minutes':
            pair['checks']['strict_historical_best_gain'] = bool(index.loc[pair['model']].spellF1 > plan['historical_best_buffered'])
    passing = rank_eligible(pairs, scores.to_dict('records'), methods)
    contrast = None
    if seed == 42 and len(eligible_methods) == 2:
        contrast = {protocol: float(scores[(scores.protocol == protocol) & (scores.model == 'native_continuation_seed42')].spellF1.item() -
                                   scores[(scores.protocol == protocol) & (scores.model == 'source_continuation_seed42')].spellF1.item()) for protocol in PROTOCOLS}
    selection.update(passed=bool(passing), eligible=passing, selected=passing[0] if passing else None,
                     native_minus_source_primary=contrast, contrast_is_not_eligibility=True,
                     historical_substantial_and_extra_short_required=False)
    write_json(out / 'selection.json', selection)
    if replay:
        for name in ('scores.csv', 'segment_diagnostics.csv', 'fold_scores.csv', 'official_comparison.csv', 'selection.json'):
            if (out / name).read_bytes() != (old / name).read_bytes():
                raise ValueError('Shared/official/gate replay differs: ' + name)
        for protocol in PROTOCOLS:
            if (out / protocol / 'oof_predictions.csv').read_bytes() != (old / protocol / 'oof_predictions.csv').read_bytes():
                raise ValueError('OOF label replay differs')
    seal(out, seed=seed, new_heads=0 if replay else 13*len(eligible_methods),
         fold_encoders=13, shared_source_control=True, retained_D_X_exact=True,
         public_train_labels_only=True, source_sha256=json.loads((ROUND / 'protocol_freeze/implementation.json').read_text()))
    check_inputs()


def final_selection():
    """Predeclared intersection of all three seeds; rank using main seed42 only."""
    check_inputs()
    stages = ['screen', 'confirmation_seed43', 'confirmation_seed44']
    selections, hashes = [], {}
    for stage in stages:
        audit(ROUND / stage)
        path = ROUND / stage / 'selection.json'
        selections.append(json.loads(path.read_text()))
        hashes[stage] = digest(ROUND / stage / 'manifest.json')
    main_order = selections[0]['eligible']
    confirmed = [set(name.rsplit('_seed', 1)[0] for name in s['eligible']) for s in selections]
    common = set.intersection(*confirmed)
    passing = [name for name in main_order if name.rsplit('_seed', 1)[0] in common]
    write_json(ROUND / 'final_selection.json', dict(passed=bool(passing), eligible=passing,
        selected=passing[0] if passing else None, all9_pairs_required_per_method=True,
        confirmation_sets=[sorted(v) for v in confirmed], stage_manifest_sha256=hashes,
        ranking='Frozen seed42 buffered primary/minimum other gain/fewer changes/lexical',
        no_failed_seed42_method_confirmed=True, full_build_authorized=False))
    check_inputs()


def synthetic_gate():
    """Invented data only; independent of every source/native observation/label."""
    check_inputs()
    verify_runtime('torch')
    import torch
    from masked_player_network import MaskedPlayer, initialize, probability
    from native_encoder_continuation import mixed_bce, peak_gib
    initialize()
    output = ROUND / 'synthetic'
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([str(PYTHON), '-B', '-m', 'unittest', 'discover', '-s', 'experiments',
                    '-p', 'test_native_encoder_continuation.py'], cwd=ROOT, check=True)
    rng = np.random.default_rng(58)
    xy = torch.from_numpy(rng.uniform(-1, 1, (32, 31, 46, 2)).astype(np.float32))
    seen = torch.from_numpy(rng.random((32, 31, 46)) > .2)
    teams = torch.from_numpy(np.broadcast_to(np.arange(46) % 2, (32, 31, 46)).copy())
    target = torch.from_numpy((np.arange(32) % 2).astype(np.float32))
    model = MaskedPlayer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
    times = []
    for i in range(13):
        start = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = mixed_bce(model(xy, seen, teams), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if i >= 3:
            times.append(time.monotonic()-start)
    args = (xy.repeat(2, 1, 1, 1), seen.repeat(2, 1, 1), teams.repeat(2, 1, 1))
    forward = []
    for _ in range(5):
        start = time.monotonic()
        probability(model, *args)
        forward.append(time.monotonic()-start)
    train_time, eval_time = max(times), max(forward)
    # Two complete canonical external passes (original/strict reload), two views,
    # 14 models; full/prefix x four-reflection public encoding also counted twice.
    projected = 14*400*train_time + 15*2*2*int(np.ceil(50148/64))*eval_time + 14*(8*24+4)*eval_time
    summary = dict(passed=projected <= 7200 and peak_gib() <= 2, invented_only=True,
        actual_observations_loaded=False, actual_targets_loaded=False,
        steps_benchmarked=10, maximum_train_step_seconds=train_time,
        maximum_B64_forward_seconds=eval_time, projected_initial_seconds=projected,
        peak_GiB=peak_gib(), parameters=sum(p.numel() for p in model.parameters()),
        training_config=CONFIG, torch=torch.__version__, numpy=np.__version__)
    write_json(output / 'summary.json', summary)
    seal(output, actual_data_loaded=False)
    if not summary['passed']:
        raise ValueError('Fixed synthetic resource gate failed')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['synthetic', 'prepare-source', 'fit-control',
        'assess-control', 'prepare-native', 'fit-native', 'assess-native', 'screen', 'replay', 'final-selection', 'encode'])
    p.add_argument('--seed', type=int, default=42, choices=[42, 43, 44])
    p.add_argument('--model-folder', type=Path)
    p.add_argument('--destination', type=Path)
    p.add_argument('--parent', type=Path)
    p.add_argument('--replay-external', action='store_true')
    args = p.parse_args()
    if args.stage == 'synthetic':
        synthetic_gate()
    elif args.stage == 'prepare-source':
        prepare_source()
    elif args.stage == 'fit-control':
        check_inputs(); require(ROUND / 'synthetic/summary.json')
        if method_closed('source'):
            raise ValueError('Source method already closed')
        try:
            fit_one('source', None, ROUND / 'neural/source')
        except MethodFailure as error:
            close_method('source', 'fit', error)
    elif args.stage == 'assess-control':
        assess_external(control_only=True)
    elif args.stage == 'prepare-native':
        prepare_native()
    elif args.stage == 'fit-native':
        fit_native()
    elif args.stage == 'assess-native':
        assess_external(replay=args.replay_external)
    elif args.stage == 'encode':
        if args.model_folder is None or args.destination is None:
            raise ValueError('Encoding subprocess requires fixed checkpoint and destination paths')
        encode_worker(args.model_folder, args.destination, args.parent)
    elif args.stage == 'screen':
        screen(args.seed)
    elif args.stage == 'final-selection':
        final_selection()
    elif args.stage == 'replay':
        fit_native(replay=True)
        subprocess.run([str(PYTHON), '-B', str(Path(__file__).resolve()), '--stage', 'assess-native', '--replay-external'], cwd=ROOT, check=True)
        screen(args.seed, replay=True)
