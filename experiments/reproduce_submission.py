"""Rebuild the frozen possession submission from the competition bundle."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import mkdtemp
from urllib.request import urlopen
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'experiments'
ASSETS = EXPERIMENTS / 'submission_assets'
sys.path.insert(0, str(ROOT))
from games import GAME, half_video_blocks, require


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def verify(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f'{path} has SHA256 {actual}; expected {expected}')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def stage(destination, command, env, timeout=None):
    """Publish a completed stage atomically and verify it on a resumed build."""
    receipt = destination / 'files.json'
    if destination.exists():
        for name, expected in json.loads(receipt.read_text()).items():
            verify(destination / name, expected)
        print('Verified existing stage', destination.name, flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(dir=destination.parent, prefix=f'.{destination.name}-'))
    print('Starting', destination, flush=True)
    try:
        subprocess.run([*map(str, command), str(temporary)], cwd=ROOT, env=env,
                       check=True, timeout=timeout)
        write_json(temporary / 'files.json', {str(p.relative_to(temporary)): digest(p)
                   for p in temporary.rglob('*') if p.is_file()})
        temporary.rename(destination)
    except BaseException:
        print('Partial stage preserved at', temporary, file=sys.stderr, flush=True)
        raise
    print('Completed', destination, flush=True)


def checkpoint(work, config, local):
    expected = config['tabpfn_sha256']
    path = local.resolve() if local else work / 'tabpfn-v2.ckpt'
    if local is None and not path.exists():
        temporary = path.with_suffix('.download')
        with urlopen(config['tabpfn_url'], timeout=120) as response, temporary.open('wb') as file:
            shutil.copyfileobj(response, file)
        verify(temporary, expected)
        temporary.replace(path)
    verify(path, expected)
    return path


def build(args):
    config_path = ASSETS / 'build.json'
    config = json.loads(config_path.read_text())
    for name, expected in config['input_sha256'].items():
        source, key = name.split('/')
        verify(Path(require([key], source=source)[0]), expected)
    verify(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv', config['public_gt_sha256'])
    encoder_model = ASSETS / 'player_encoder.pt'
    verify(encoder_model, config['encoder_sha256'])

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with (work / '.build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_list = ASSETS / 'sources.txt'
        inputs = [config_path, source_list, ASSETS / 'requirements-encoder.txt',
                  ASSETS / 'requirements-tabpfn.txt', EXPERIMENTS / 'requirements.txt',
                  EXPERIMENTS / 'requirements-research.txt',
                  *(ROOT / name for name in source_list.read_text().splitlines())]
        identity = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
        receipt = work / 'build_sources.json'
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise ValueError('Build sources changed. Choose a fresh --work directory.')
        if not receipt.exists():
            write_json(receipt, identity)
        model = checkpoint(work, config, args.checkpoint)

        env = dict(os.environ, PYTHONHASHSEED='0', PYTHONNOUSERSITE='1', PYTHONPATH='')
        cpu = [sys.executable, '-B', EXPERIMENTS / 'submission_cpu.py']
        encoder = [args.encoder_python.absolute(), '-B', EXPERIMENTS / 'submission_models.py']
        stage(work / 'tactical', [*cpu, 'tactical', work], env)
        stage(work / 'encoder', [*encoder, 'encode', work / 'tactical/encoder_input.npz', encoder_model], env)
        stage(work / 'queries', [*cpu, 'queries', work], env)
        for name in ('context', 'target'):
            verify(work / 'queries' / f'{name}.npy', config[f'tabpfn_{name}_sha256'])
        stage(work / 'cameras', [*cpu, 'cameras', work], env)
        stage(work / 'heads', [*cpu, 'heads', work], env)

        native = args.tabpfn_python.absolute()
        probe = ('import torch; assert torch.backends.mps.is_available(), "Apple MPS is required"; '
                 'print(5 * 1024**3 / torch.mps.recommended_max_memory())')
        memory_fraction = subprocess.check_output([native, '-B', '-c', probe], env=env, text=True).strip()
        native_env = dict(env, TABPFN_MPS_MEMORY_FRACTION=memory_fraction,
            TABPFN_MODEL_CACHE_DIR=str(work / 'unused_model_cache'), TABPFN_MODEL_CACHE_SIZE='0',
            TABPFN_DISABLE_TELEMETRY='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
            HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1', WANDB_MODE='disabled',
            TOKENIZERS_PARALLELISM='false', PYTORCH_ENABLE_MPS_FALLBACK='0', PYTORCH_MPS_FAST_MATH='0',
            TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', TORCH_FORCE_WEIGHTS_ONLY_LOAD='0',
            VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        for _, start, last in half_video_blocks(GAME):
            for left in range(start, last + 1, 4096):
                stage(work / 'native' / str(left), [native, '-B', EXPERIMENTS / 'submission_models.py',
                      'predict', work / 'queries', left, model], native_env, timeout=3600)

        stage(work / 'submission', [*cpu, 'export', work], env)
        candidate = work / 'submission/candidate.csv'
        verify(candidate, config['expected_candidate_sha256'])
        subprocess.run([sys.executable, '-B', str(ROOT / 'evaluation/validate_submission.py'),
                        str(candidate)], check=True, env=env)
        print('Exact selected CSV reproduced:', candidate, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True, type=Path, help='Local output and resume directory')
    parser.add_argument('--encoder-python', required=True, type=Path, help='Pinned Torch 2.5 CPU interpreter')
    parser.add_argument('--tabpfn-python', required=True, type=Path, help='Pinned Torch 2.15 MPS interpreter')
    parser.add_argument('--checkpoint', type=Path, help='Optional local copy of the pinned TabPFN checkpoint')
    build(parser.parse_args())
