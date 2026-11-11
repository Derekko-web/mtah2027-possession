"""Relative input identity and non-self-referential local generation manifests."""
from pathlib import Path
from build_temporal_candidate import digest


def generation_manifest(root, output, inputs, plan_path, sources, plan, count):
    root, output, inputs = Path(root), Path(output), Path(inputs)
    return dict(plan_sha256=digest(plan_path), source_sha256=dict(sources),
        input_sha256={str((inputs/'manifest.json').relative_to(root)):digest(inputs/'manifest.json')},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'},
        targets_not_read=True, local_model_digest=plan['model_digest'], runtime_version=plan['runtime_version'],
        samples=count, proxy_bypass=True, redirects_forbidden=True, endpoint='http://127.0.0.1:11434')
