"""Regenerate the frozen representation and a single native TabPFN partition."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def encode(tracking, checkpoint, output):
    import numpy as np
    import torch

    from encode_player_representation import reflected
    from games import GAME, GAMES, half_video_blocks
    from masked_player_data import observation_views
    from masked_player_native_data import assemble_native
    from masked_player_network import MaskedPlayer, initialize
    from tabpfn_full_features import interpolate_halves

    with np.load(tracking, allow_pickle=False) as source:
        positions, teams = source['positions'], source['teams']
    x, observed, groups = observation_views(positions[::5], teams, GAMES[GAME]['pitch'], 0, centered=False)
    grid = np.arange(0, len(positions), 5)
    half = np.full(len(grid), -1, np.int16)
    blocks = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    for number, (start, end) in enumerate(blocks):
        half[(grid >= start) & (grid < end)] = number
    centers = np.flatnonzero(half >= 0)
    initialize()
    model = MaskedPlayer()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    hidden = np.empty((len(centers), 64), dtype=np.float32)
    supported = np.empty(len(centers), dtype=bool)
    for start in range(0, len(centers), 64):
        batch = centers[start:start + 64]
        count = len(batch)
        if count < 64:
            batch = np.pad(batch, (0, 64 - count), mode='edge')
        values = assemble_native(x[:1], observed[:1], groups, half, batch)
        hidden[start:start + count] = reflected(model, values)[:count]
        counts = np.stack([(values[1] & (values[2] == team)).sum(-1) for team in (0, 1)], axis=-1)[:count]
        supported[start:start + count] = (counts[:, 15] >= 4).all(1) & (counts.mean(1) >= 4).all(1)
        if start % 4096 == 0:
            print('Encoded', start + count, '/', len(centers), 'player sets', flush=True)
    features, eligible = interpolate_halves(grid[centers], hidden, supported, len(positions), blocks)
    np.save(output / 'full_latent.npy', features)
    np.save(output / 'full_eligible.npy', eligible)


def predict(queries, start, checkpoint, output):
    import joblib
    import numpy as np
    from sklearn.utils.validation import check_is_fitted
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.preprocessing.pipeline_interface import PreprocessingStep
    from tabpfn.preprocessing.steps.encode_categorical_features_step import EncodeCategoricalFeaturesStep

    import tabpfn_mps_runtime_gate as runtime
    from tabpfn_mps_allocator_flush_gate import flushed_callback, record_flush
    from tabpfn_mps_validation_init_gate import initialize_step

    if torch.__version__ != '2.15.0' or np.__version__ != '2.1.2' or not torch.backends.mps.is_available():
        raise RuntimeError('The frozen native run requires Torch 2.15.0, NumPy 2.1.2 and Apple MPS')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')

    context, target = [np.load(queries / f'{name}.npy', allow_pickle=False) for name in ('context', 'target')]
    original, swapped = [np.load(queries / f'{start}-{name}.npy', allow_pickle=False)
                         for name in ('original', 'swapped')]
    model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, model_path=checkpoint,
        n_estimators=8, softmax_temperature=.9, balance_probabilities=False, average_before_softmax=False,
        device='mps', ignore_pretraining_limits=False, inference_precision=torch.float32,
        fit_mode='fit_preprocessors', memory_saving_mode=True, random_state=42,
        n_preprocessing_jobs=1, enable_torch_compile=False, tuning_config=None)

    operation = runtime.operation

    def with_flush(folder, name, callback):
        if not name.startswith('fit_'):
            callback = flushed_callback(callback, torch.mps, lambda event: record_flush(folder, name, event))
        return operation(folder, name, callback)

    with ExitStack() as stack:
        runtime.MONITOR = runtime.MemoryMonitor(output)
        runtime.MONITOR.start()
        stack.callback(runtime.MONITOR.close)
        runtime.DISPATCH = runtime.NativeDispatchGuard(stack)
        runtime.OPERATION_SECONDS = 900
        stack.enter_context(patch.object(runtime, 'operation', with_flush))
        runtime.operation(output, 'fit_context', lambda: model.fit(context, target))
        np.testing.assert_array_equal(model.classes_, [0, 1])
        bindings = (EncodeCategoricalFeaturesStep, PreprocessingStep, check_is_fitted)
        for member in model.executor_.ensemble_members:
            step, = (step for step, _ in member.cpu_preprocessor.steps
                     if type(step) is EncodeCategoricalFeaturesStep)
            initialize_step(step, bindings, lambda value: joblib.hash(value, hash_name='sha1'))
        p0, _ = runtime.canonical_predict(model, original, output, 'original')
        p1, _ = runtime.canonical_predict(model, swapped, output, 'swapped')
        repeated, _ = runtime.canonical_predict(model, original[:512], output, 'repeat_first')
        np.testing.assert_array_equal(repeated, p0[:512])
    np.save(output / 'owner.npy', (p0 + p1[:, [1, 0]]) / 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    encoder = sub.add_parser('encode')
    encoder.add_argument('tracking', type=Path)
    native = sub.add_parser('predict')
    native.add_argument('queries', type=Path)
    native.add_argument('start', type=int)
    for command in (encoder, native):
        command.add_argument('checkpoint', type=Path)
        command.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.stage == 'encode':
        encode(args.tracking, args.checkpoint, args.output)
    else:
        predict(args.queries, args.start, args.checkpoint, args.output)
