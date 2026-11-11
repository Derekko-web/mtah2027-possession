"""Compose immutable local scenes into the synthetically verified single image."""
import json
from pathlib import Path
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from ownership_montage import make_montage


def main():
    root = ROOT/'data/control_local_ownership_v1'
    path = ROOT/'experiments/LOCAL_OWNERSHIP_MONTAGE_PLAN.json'
    plan = json.loads(path.read_text())
    for name, expected in plan['input_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Frozen montage source changed')
    control = json.loads((root/'synthetic_production_layout_summary.json').read_text())
    if not control['all_correct'] or control['layout_source_sha256'] != digest(ROOT/'experiments/ownership_montage.py'):
        raise ValueError('Exact montage layout control failed')
    original = root/'inputs'
    source = audit_manifest(original)
    if digest(original/'manifest.json') != plan['base_input_manifest_sha256'] or not source['targets_not_read']:
        raise ValueError('Original montage scenes changed')
    output = root/'inputs_montage'
    output.mkdir()
    (output/'plan.json').write_bytes(path.read_bytes())
    rows = json.loads((original/'samples.json').read_text())
    reference = Image.open(original/'jersey_reference.png')
    for row in rows:
        images = [Image.open(original/name) for name in row['images']]
        row['image'] = f'montage_{row["center"]}.png'
        make_montage(reference, images).save(output/row['image'])
    (output/'samples.json').write_text(json.dumps(rows, indent=2)+'\n')
    names = ('prepare_local_ownership_montage.py', 'ownership_montage.py')
    result = dict(plan_sha256=digest(path), source_sha256={name:digest(ROOT/'experiments'/name) for name in names},
        input_sha256=plan['input_sha256'], cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        targets_not_read=True, no_new_video_decode=True, same_samples_and_reference=True,
        synthetic_exact_layout_passed=True, samples=len(rows), images_per_request=1)
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Prepared', len(rows), 'single-image montages from immutable source scenes; no targets read')


if __name__ == '__main__':
    main()
