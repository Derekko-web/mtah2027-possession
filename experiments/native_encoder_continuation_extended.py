"""R61 fixed source/native BCE continuation; real-data work requires a freeze."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_native_encoder_continuation_extended_v1'
PLAN = ROOT / 'experiments/NATIVE_ENCODER_CONTINUATION_EXTENDED_PLAN.json'
PYTHON = ROOT / 'data/control_sports_action_v1/runtime/bin/python'
HEAD_PYTHON = ROOT / '.venv/bin/python'
PUBLIC = np.arange(11185, 18685)
PROTOCOLS = ('buffered_minutes', 'purged_spells', 'buffered_100_seconds')
SOURCE = ROOT / 'data/control_masked_player_v1'
REPRESENTATION = ROOT / 'data/control_player_representation_v1'
INITIAL = SOURCE / 'external_models/temporal/model.pt'
STEPS = 400
CONFIG = dict(steps=400, batch_size=32, source_batch=16, native_batch=16,
              learning_rate=.0001, weight_decay=.01, gradient_clip=1., seed=42,
              cpu_threads=2, final_checkpoint_only=True)


class MethodFailure(RuntimeError):
    """A bounded numerical/resource failure, distinct from provenance corruption."""


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
        default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist()) + '\n')


def audit(folder):
    manifest = json.loads((Path(folder) / 'manifest.json').read_text())
    for name, expected in manifest['cache_sha256'].items():
        if digest(Path(folder) / name) != expected:
            raise ValueError('Changed retained artifact: ' + str(Path(folder) / name))
    return manifest


def seal(folder, **metadata):
    folder = Path(folder)
    write_json(folder / 'manifest.json', dict(metadata,
        plan_sha256=digest(PLAN), cache_sha256={str(p.relative_to(folder)): digest(p)
            for p in sorted(folder.rglob('*')) if p.is_file() and p != folder / 'manifest.json'}))


def check_inputs():
    freeze = ROUND / 'protocol_freeze'
    if not (freeze / 'plan.json').exists():
        raise ValueError('R61 must be independently reviewed and frozen before real-data processing')
    if PLAN.read_bytes() != (freeze / 'plan.json').read_bytes():
        raise ValueError('R61 plan changed')
    plan = json.loads(PLAN.read_text())
    if plan['training'] != CONFIG:
        raise ValueError('Fixed continuation recipe changed')
    for group in (plan['input_sha256'], json.loads((freeze / 'implementation.json').read_text()),
                  json.loads((freeze / 'protected_artifacts.json').read_text())):
        for name, expected in group.items():
            if digest(ROOT / name) != expected:
                raise ValueError('Frozen dependency changed: ' + name)
    if not (ROOT / 'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith(
            (freeze / 'EXPERIMENT_LEDGER.csv').read_bytes()):
        raise ValueError('Historical ledger prefix changed')
    for name, expected in json.loads((freeze / 'symlinks.json').read_text()).items():
        path = ROOT / name
        if not path.is_symlink() or str(path.readlink()) != expected:
            raise ValueError('Source symlink changed')
    for name, expected in plan.get('runtime_sha256', {}).items():
        if digest(name) != expected:
            raise ValueError('Established runtime file changed: ' + name)
    return plan


def verify_runtime(role):
    """Separate fixed processes; never append the other runtime's site packages."""
    import importlib
    record = json.loads(PLAN.read_text())['runtime_roles'][role]
    for name, expected in record['modules'].items():
        module = importlib.import_module(name)
        if str(module.__version__) != expected['version'] or str(Path(module.__file__).resolve()) != expected['path']:
            raise ValueError('Use the fixed '+role+' interpreter; module identity differs: '+name)


def ha_targets(labels):
    """The inherited neural scalar is p(H): H=1, A=0, unlike XGB H=0."""
    labels = np.asarray(labels)
    if not np.isin(labels, ['H', 'A']).all():
        raise ValueError('Binary encoder targets must be H/A only')
    return (labels == 'H').astype(np.float32)


def source_schedule(buckets, steps=STEPS, seed=580042):
    """Two independent 16-example chunks, each 4 matches x 2 classes x swap pair."""
    keys = sorted(buckets)
    if len(keys) != 8 or len({k[0] for k in keys}) != 4 or any(
            {k[1] for k in keys if k[0] == match} != {0, 1} for match in {k[0] for k in keys}):
        raise ValueError('Require exactly four source matches and both classes')
    if any(not len(buckets[k]) for k in keys):
        raise ValueError('Empty source match/class bucket')
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ('centers', 'views', 'flip_x', 'flip_y', 'swap', 'target', 'match')}
    for _ in range(steps):
        record = {k: [] for k in out}
        for _chunk in range(2):
            c = np.array([rng.choice(buckets[k]) for k in keys], np.int64)
            y = np.array([k[1] for k in keys], np.float32)
            v = rng.integers(0, 4, 8, dtype=np.int8)
            fx, fy = rng.random(8) < .5, rng.random(8) < .5
            for k, value in dict(centers=np.r_[c, c], views=np.r_[v, v],
                    flip_x=np.r_[fx, fx], flip_y=np.r_[fy, fy],
                    swap=np.r_[np.zeros(8, bool), np.ones(8, bool)],
                    target=np.r_[y, 1-y], match=np.tile([k[0] for k in keys], 2)).items():
                record[k].extend(value)
        for k in out:
            out[k].append(record[k])
    return {k: np.asarray(v) for k, v in out.items()}


def native_schedule(target, steps=STEPS, seed=580043):
    """Eight base examples (four/class), each paired with its exact team swap."""
    target = np.asarray(target, np.float32)
    if not np.isin(target, [0., 1.]).all() or not all((target == k).any() for k in (0, 1)):
        raise ValueError('Both native training classes required')
    rng = np.random.default_rng(seed)
    rows = np.array([np.r_[rng.choice(np.flatnonzero(target == 0), 4),
                           rng.choice(np.flatnonzero(target == 1), 4)] for _ in range(steps)])
    fx, fy = rng.random((steps, 8)) < .5, rng.random((steps, 8)) < .5
    y = target[rows]
    return dict(rows=np.concatenate([rows, rows], axis=1),
        flip_x=np.concatenate([fx, fx], axis=1), flip_y=np.concatenate([fy, fy], axis=1),
        swap=np.tile(np.r_[np.zeros(8, bool), np.ones(8, bool)], (steps, 1)),
        target=np.concatenate([y, 1-y], axis=1))


def permitted_centers(frames, train, validation, labels, half_id, support, n):
    """Conservative complete +/-79 fitting footprint, except known-half padding."""
    train, validation, frames = map(lambda x: np.asarray(x, np.int64), (train, validation, frames))
    if np.intersect1d(train, validation).size or not np.isin(train, PUBLIC).all():
        raise ValueError('Invalid outer training split')
    allowed = np.zeros(n, bool)
    allowed[train] = True
    selected = []
    for i, center in enumerate(frames):
        if center >= n or center < 0 or center % 5 or not allowed[center] or not support[i]:
            continue
        h = half_id[center // 5]
        if h < 0:
            continue
        footprint = np.arange(center-79, center+80)
        in_bounds = (footprint >= 0) & (footprint < n)
        real = footprint[in_bounds]
        real = real[half_id[real // 5] == h]
        if allowed[real].all():
            selected.append(int(center))
    # Observation-only eligibility is determined before the supervised target join.
    observed_eligible = np.asarray(selected, np.int64)
    return observed_eligible[np.isin(np.asarray(labels)[observed_eligible], ['H', 'A'])], observed_eligible


def apply_native(pool, schedule, step):
    rows = schedule['rows'][step]
    value = [np.array(pool[k][rows], copy=True) for k in ('xy', 'observed', 'teams')]
    value[0][..., 0] *= np.where(schedule['flip_x'][step, :, None, None], -1., 1.)
    value[0][..., 1] *= np.where(schedule['flip_y'][step, :, None, None], -1., 1.)
    value[2] = np.where(schedule['swap'][step, :, None, None], 1-value[2], value[2]).astype(np.int8)
    expected = np.where(schedule['swap'][step], 1-pool['target'][rows], pool['target'][rows])
    np.testing.assert_array_equal(expected, schedule['target'][step])
    return value


def apply_source(source, schedule, step, count=32):
    from masked_player_data import assemble
    args = [schedule[k][step, :count] for k in ('centers', 'views', 'flip_x', 'flip_y')]
    value = list(assemble(source['xy'], source['observed'], source['teams'], source['half_id'],
                          *args, repeat=False, trim=False))
    swap = schedule['swap'][step, :count, None, None]
    value[2] = np.where(swap, 1-value[2], value[2]).astype(np.int8)
    expected = np.where(schedule['swap'][step, :count], 1-source['labels'][args[0]], source['labels'][args[0]])
    np.testing.assert_array_equal(expected, schedule['target'][step, :count])
    return value


def mixed_bce(logits, target):
    import torch
    if logits.shape != (32,) or target.shape != (32,):
        raise ValueError('Fixed 16+16 domain batch required')
    return .5 * torch.nn.functional.binary_cross_entropy_with_logits(logits[:16], target[:16]) + .5 * \
        torch.nn.functional.binary_cross_entropy_with_logits(logits[16:], target[16:])


def peak_gib():
    divisor = 1024**3 if sys.platform == 'darwin' else 1024**2
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def train_fixed(model, source, source_draws, pool=None, native_draws=None, steps=STEPS):
    """Pure fixed-budget loop; callers control provenance and never supply queries."""
    import torch
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
    losses, started = [], time.monotonic()
    for i in range(steps):
        first = apply_source(source, source_draws, i, 32 if pool is None else 16)
        target = source_draws['target'][i].astype(np.float32)
        if pool is not None:
            second = apply_native(pool, native_draws, i)
            first = [np.concatenate([a, b]) for a, b in zip(first, second)]
            target = np.r_[target[:16], native_draws['target'][i]].astype(np.float32)
        torch.manual_seed(580000 + i)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        tensors = [torch.from_numpy(np.ascontiguousarray(v)) for v in first]
        loss = mixed_bce(model(*tensors), torch.from_numpy(target))
        if not torch.isfinite(loss):
            raise MethodFailure('Nonfinite loss')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            raise MethodFailure('Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if peak_gib() > 2 or time.monotonic()-started > 900:
            raise MethodFailure('Frozen continuation resource limit exceeded')
        if (i+1) % 100 == 0:
            print('step', i+1, 'loss', losses[-1], 'seconds', time.monotonic()-started, flush=True)
    return np.asarray(losses), first


def metrics(probability, target):
    p, y = np.asarray(probability, float), np.asarray(target, int)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid external probabilities')
    recall = [float(((p >= .5)[y == k] == k).mean()) if (y == k).any() else float('nan') for k in (0, 1)]
    q = np.clip(p, 1e-12, 1-1e-12)
    return dict(balanced_accuracy=float(np.mean(recall)), away_recall=recall[0], home_recall=recall[1],
                brier=float(np.mean((p-y)**2)), logloss=float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q))))


def retention(new, baseline):
    return dict(absolute_BA=new['balanced_accuracy'] >= .75,
        home_recall=new['home_recall'] >= .65, away_recall=new['away_recall'] >= .65)


def retention_diagnostics(new, baseline):
    return dict(retained_BA=new['balanced_accuracy'] >= baseline['balanced_accuracy']-.005-1e-12,
        retained_Brier=new['brier'] <= baseline['brier']+.005+1e-12,
        BA_delta=new['balanced_accuracy']-baseline['balanced_accuracy'],
        Brier_delta=new['brier']-baseline['brier'], relative_checks_are_eligibility=False)


def retention_eligibility(report):
    source = [r for r in report if r['model'] == 'source']
    native = [r for r in report if r['model'] != 'source']
    methods = dict(source_continuation=len(source) == 6 and all(all(r['checks'].values()) for r in source),
                   native_continuation=len(native) == 78 and all(all(r['checks'].values()) for r in native))
    return methods, sorted(k for k, passed in methods.items() if passed)


def rank_eligible(pairs, scores, methods):
    passing = [name for name in methods if
        {p['protocol'] for p in pairs if p['model'] == name} == set(PROTOCOLS) and
        len([p for p in pairs if p['model'] == name]) == 3 and
        all(all(p['checks'].values()) for p in pairs if p['model'] == name)]
    index = {(r['protocol'], r['model']): float(r['spellF1']) for r in scores}
    def rank(name):
        rows = [p for p in pairs if p['model'] == name]
        return (-index['buffered_minutes', name],
                -min(p['spellF1_delta'] for p in rows if p['protocol'] != 'buffered_minutes'),
                sum(p['changed_frames'] for p in rows), name)
    return sorted(passing, key=rank)


def worker(job_path, output):
    check_inputs()
    verify_runtime('torch')
    import torch
    from masked_player_network import MaskedPlayer, initialize, probability
    initialize()
    job = json.loads(Path(job_path).read_text())
    if set(job) != {'mode', 'source', 'source_schedule', 'pool', 'native_schedule', 'input_sha256', 'parent'}:
        raise ValueError('Unexpected fitting payload; query/score inputs prohibited')
    if job['mode'] not in ('source', 'native') or bool(job['pool']) != (job['mode'] == 'native'):
        raise ValueError('Wrong fixed continuation arm')
    for name, expected in job['input_sha256'].items():
        if digest(name) != expected:
            raise ValueError('Fitting payload changed')
    source = {k: np.load(Path(job['source']) / (k+'.npy'), mmap_mode='r')
              for k in ('xy', 'observed', 'teams', 'half_id', 'labels')}
    draws = dict(np.load(job['source_schedule']))
    pool = dict(np.load(job['pool'])) if job['pool'] else None
    native = dict(np.load(job['native_schedule'])) if pool is not None else None
    if draws['centers'].shape != (400, 32) or (native is not None and native['rows'].shape != (400, 16)):
        raise ValueError('Fixed 400-step schedule required')
    if source['xy'].ndim != 4 or source['xy'].shape[0] != 4 or source['xy'].shape[2:] != (46, 2):
        raise ValueError('Fixed source four-view/N46 schema required')
    if pool is not None and (pool['xy'].ndim != 4 or pool['xy'].shape[1:] != (31, 46, 2)):
        raise ValueError('Fixed native T31/N46 schema required')
    if not np.isin(draws['target'], [0, 1]).all() or not np.all(draws['target'][:, :16].sum(1) == 8) or not np.all(draws['target'][:, 16:].sum(1) == 8):
        raise ValueError('Source schedule team/class balance changed')
    if native is not None and (not np.isin(native['target'], [0, 1]).all() or not np.all(native['target'].sum(1) == 8)):
        raise ValueError('Native schedule team/class balance changed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = MaskedPlayer()
    if sum(p.numel() for p in model.parameters()) != 24321:
        raise ValueError('Original architecture parameter count changed')
    model.load_state_dict(torch.load(INITIAL, map_location='cpu', weights_only=True), strict=True)
    started = time.monotonic()
    with patch.object(socket.socket, 'connect', side_effect=RuntimeError('Network disabled')):
        losses, probe = train_fixed(model, source, draws, pool, native)
        torch.save(model.state_dict(), output / 'model.pt')
        np.save(output / 'training_loss.npy', losses)
        values = [torch.from_numpy(np.ascontiguousarray(v)) for v in probe]
        p = probability(model, *values)
        loaded = MaskedPlayer()
        loaded.load_state_dict(torch.load(output / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        np.testing.assert_array_equal(p, probability(loaded, *values))
        if job['parent']:
            prior = torch.load(Path(job['parent']) / 'model.pt', map_location='cpu', weights_only=True)
            for k, value in model.state_dict().items():
                if not torch.equal(value, prior[k]):
                    raise ValueError('Independent continuation weights differ: ' + k)
            np.testing.assert_array_equal(losses, np.load(Path(job['parent']) / 'training_loss.npy'))
            np.testing.assert_array_equal(p, np.load(Path(job['parent']) / 'probe_probability.npy'))
        np.save(output / 'probe_probability.npy', p)
    write_json(output / 'recipe.json', dict(training=CONFIG, mode=job['mode'],
        initialization_sha256=digest(INITIAL), input_sha256=job['input_sha256'],
        neural_label_orientation='H=1,A=0', query_windows_supplied=False,
        checkpoint='final_step400', parameters=sum(p.numel() for p in model.parameters())))
    seal(output, passed=True, mode=job['mode'], steps=400, seconds=time.monotonic()-started,
         peak_GiB=peak_gib(), strict_reload_exact=True, independent_refit_exact=bool(job['parent']))
    check_inputs()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        worker(args.job, args.output)
    except MethodFailure as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'method_failure.json', dict(passed=False, reason=str(error), no_retry=True))
        raise SystemExit(3)
