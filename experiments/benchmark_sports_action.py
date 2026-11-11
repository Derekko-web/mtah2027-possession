"""Synthetic full-resolution benchmark; no competition data or labels."""
import hashlib
import json
from pathlib import Path
import resource
import socket
import time
from unittest.mock import patch
from sports_action_runtime import EVIDENCE, ROOT, digest, load_model, verify_public_inputs


def main():
    import numpy as np
    import torch
    path = EVIDENCE/'full_resolution_benchmark_plan.json'
    settings = json.loads(path.read_text())
    _, _, checkpoint, vendor = verify_public_inputs()
    output = EVIDENCE/'full_resolution_benchmark'
    output.mkdir()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if not torch.backends.mps.is_available():
        raise SystemExit('Planned MPS benchmark unavailable')
    torch.mps.set_per_process_memory_fraction(.6)

    def no_network(*a, **kw):
        raise RuntimeError('Network forbidden during action inference')

    with patch.object(socket.socket, 'connect', no_network):
        model, keys = load_model(checkpoint, vendor)
        model.to('mps')
        raw = torch.randint(0, 256, (1, 100, 3, 448, 796), dtype=torch.uint8)
        input_sha = hashlib.sha256(raw.numpy().tobytes()).hexdigest()
        x = raw.to(device='mps', dtype=torch.float32)
        times, values, memories = [], [], []
        for index in range(2):
            torch.mps.synchronize()
            start = time.monotonic()
            with torch.inference_mode():
                result, _ = model(x, inference=True)
            torch.mps.synchronize()
            times.append(time.monotonic()-start)
            values.append({name:value.cpu().numpy() for name,value in result.items()})
            memories.append(dict(current_allocated=int(torch.mps.current_allocated_memory()),
                                 driver_allocated=int(torch.mps.driver_allocated_memory())))
            print('Completed synthetic full-resolution forward', index+1, '/2', flush=True)
        shapes = {name:list(value.shape) for name,value in values[0].items()}
        differences = {name:float(np.abs(values[0][name]-values[1][name]).max()) for name in values[0]}
        finite = all(np.isfinite(value).all() for group in values for value in group.values())
        expected = dict(im_feat=[1, 100, 31], team_feat=[1, 100], displ_feat=[1, 100])
        for index, group in enumerate(values):
            for name,value in group.items():
                np.save(output/f'{name}_{index}.npy', value)
        passed = bool(finite and shapes == expected and max(differences.values()) <= 1e-5)
        summary = dict(passed=passed, finite=bool(finite), shapes=shapes, forward_seconds=times,
            repeat_max_absolute_difference=differences, memory_bytes=memories,
            peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            synthetic_input_sha256=input_sha, state_keys=keys, plan_sha256=digest(path),
            source_sha256=digest(Path(__file__)), checkpoint_sha256=digest(checkpoint),
            competition_images_used=False, possession_targets_read=False, network_connections_forbidden=True,
            no_global_settings_changed=True, full_resolution_cpu_parity_not_tested=True)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2))
        if not passed:
            raise SystemExit('Synthetic full-resolution benchmark failed')


if __name__ == '__main__':
    main()
