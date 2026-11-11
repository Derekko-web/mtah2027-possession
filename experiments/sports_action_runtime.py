"""Strict offline loading and synthetic-only checks of a public T-DEED checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tarfile
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['WANDB_MODE'] = 'disabled'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'data/control_sports_action_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_public_inputs():
    plan_path = ROOT/'experiments/SPORTS_ACTION_RUNTIME_PLAN.json'
    plan = json.loads(plan_path.read_text())
    checkpoint = ROOT/plan['checkpoint']['path']
    if digest(checkpoint) != plan['checkpoint']['sha256'] or checkpoint.stat().st_size != plan['checkpoint']['bytes']:
        raise ValueError('Public action checkpoint changed')
    record = next(r for r in plan['sources'] if r['repository'] == 'SoccerNet/sn-teamspotting')
    archive = EVIDENCE/'public_sources/sn-teamspotting.tar.gz'
    if digest(archive) != record['archive_sha256']:
        raise ValueError('Public source archive changed')
    vendor = EVIDENCE/'public_sources/sn-teamspotting'
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            if member.isfile():
                relative = Path(*Path(member.name).parts[1:])
                if (vendor/relative).read_bytes() != handle.extractfile(member).read():
                    raise ValueError('Extracted public source differs from pinned archive')
    return plan_path, plan, checkpoint, vendor


def load_model(checkpoint, vendor):
    import torch
    import timm
    sys.path.insert(0, str(vendor))
    from model.model import TDEEDModel
    from model.modules import FC2Layers
    config = json.loads((vendor/'config/SoccerNetBall/SoccerNetBall_baseline.json').read_text())
    config['crop_dim'] = None
    args = SimpleNamespace(**config)
    original = timm.create_model

    def offline_create(*a, **kw):
        kw['pretrained'] = False
        return original(*a, **kw)

    with patch.object(timm, 'create_model', offline_create):
        model = TDEEDModel.Impl(args=args)
    # Same upstream joint head, with CPU placement instead of update_pred_head's forced CUDA.
    model._pred_fine = FC2Layers(model._feat_dim, [13, 18])
    model._double_head = True
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor-only public state dictionary')
    if all(k.startswith('module.') for k in state):
        state = {k[7:]:v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval(), len(state)


def main():
    import numpy as np
    import torch
    import timm
    import torchvision
    plan_path, plan, checkpoint, vendor = verify_public_inputs()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    def no_network(*args, **kwargs):
        raise RuntimeError('Network connections forbidden during action inference')

    with patch.object(socket.socket, 'connect', no_network):
        model, keys = load_model(checkpoint, vendor)
        x = torch.randint(0, 256, (1, 100, 3, 32, 56), dtype=torch.uint8).float()
        started = time.monotonic()
        with torch.inference_mode():
            first, _ = model(x, inference=True)
            second, _ = model(x, inference=True)
            reverse, _ = model(x.flip(1), inference=True)
            reflect, _ = model(x.flip((3, 4)), inference=True)
        elapsed = time.monotonic()-started
        expected = dict(im_feat=(1, 100, 31), team_feat=(1, 100), displ_feat=(1, 100))
        shapes = {name:list(value.shape) for name,value in first.items()}
        if {name:tuple(value.shape) for name,value in first.items()} != expected:
            raise ValueError('Public model output schema differs')
        finite = all(torch.isfinite(value).all().item() for value in first.values())
        repeat = all(torch.equal(first[name], second[name]) for name in first)
        time_sensitive = not torch.equal(first['im_feat'], reverse['im_feat'])
        spatial_sensitive = not torch.equal(first['im_feat'], reflect['im_feat'])
        probability = torch.softmax(first['im_feat'][:, :, :13], dim=2)
        normalized = bool(torch.allclose(probability.sum(2), torch.ones((1, 100)), atol=1e-6))
        output = EVIDENCE/'runtime_preflight'
        output.mkdir()
        for name, value in first.items():
            np.save(output/(name+'.npy'), value.numpy())
            np.save(output/(name+'_repeat.npy'), second[name].numpy())
        mps = dict(available=bool(torch.backends.mps.is_available()), tested=False)
        if mps['available']:
            try:
                model.to('mps')
                with torch.inference_mode():
                    other, _ = model(x.to('mps'), inference=True)
                differences = {name:float((value.cpu()-first[name]).abs().max()) for name,value in other.items()}
                mps.update(tested=True, supported=True, max_absolute_difference=differences,
                    allclose=all(torch.allclose(other[name].cpu(), first[name], rtol=1e-3, atol=1e-3) for name in first))
            except Exception as error:
                mps.update(tested=True, supported=False, error_type=type(error).__name__, error=str(error)[:600])
        checks = dict(strict_state_load=True, finite=finite, cpu_repeat_exact=repeat,
            temporal_input_sensitive=time_sensitive, spatial_input_sensitive=spatial_sensitive,
            first13_softmax_normalized=normalized)
        result = dict(checks=checks, passed=all(checks.values()), output_shapes=shapes,
            state_keys=keys, cpu_four_forward_seconds=elapsed, synthetic_shape=list(x.shape),
            mps=mps, versions=dict(torch=torch.__version__, torchvision=torchvision.__version__, timm=timm.__version__, numpy=np.__version__),
            plan_sha256=digest(plan_path), checkpoint_sha256=digest(checkpoint),
            source_sha256=digest(Path(__file__)), network_connections_forbidden=True,
            competition_images_used=False, possession_targets_read=False, new_fits=0,
            full_resolution_performance_not_yet_tested=True)
        (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
        if not result['passed']:
            raise SystemExit('Synthetic action runtime preflight failed')


if __name__ == '__main__':
    main()
