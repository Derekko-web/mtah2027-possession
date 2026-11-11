"""Pinned loopback-only VLM inference; raw text and all images remain local."""
import argparse
import base64
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from local_ownership import local_json, require_model, parse_ownership


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Only after the previous producer is confirmed terminal')
    args = parser.parse_args()
    root = ROOT/'data/control_local_ownership_v1'
    plan_path = ROOT/'experiments/LOCAL_OWNERSHIP_INSTRUCT_PLAN.json'
    plan = json.loads(plan_path.read_text())
    inputs = root/'inputs'
    manifest = audit_manifest(inputs)
    if manifest['plan_sha256'] != plan['source_inputs_plan_sha256'] or not manifest['targets_not_read']:
        raise ValueError('Local ownership inputs/plan changed')
    if digest(inputs/'manifest.json') != plan['input_manifest_sha256']:
        raise ValueError('Instruct source inputs changed')
    preflight = json.loads((root/'instruct_preflight.json').read_text())
    if not preflight['passed'] or preflight['plan_sha256'] != digest(plan_path):
        raise ValueError('Instruction model output preflight failed')
    samples = json.loads((inputs/'samples.json').read_text())
    if len(samples) < 15:
        raise SystemExit('STOP BEFORE INFERENCE: fewer than15source samples')
    output = root/'inference_instruct'
    output.mkdir(exist_ok=args.resume)
    saved_plan = output/'plan.json'
    if saved_plan.exists() and saved_plan.read_bytes() != plan_path.read_bytes():
        raise ValueError('Resumed ownership plan differs')
    saved_plan.write_bytes(plan_path.read_bytes())
    names = ('run_local_ownership_instruct.py', 'local_ownership.py')
    sources = {name:digest(ROOT/'experiments'/name) for name in names}
    cached = output/'answers.json'
    rows = json.loads(cached.read_text()) if cached.exists() else []
    completed = {row['center'] for row in rows}
    expected = {row['center'] for row in samples}
    if not completed <= expected or len(completed) != len(rows):
        raise ValueError('Resumed ownership sample identities differ')
    for sample in samples:
        center = sample['center']
        if center in completed:
            continue
        require_model(plan)
        images = ['jersey_reference.png', *sample['images']]
        if any(digest(inputs/name) != manifest['cache_sha256'][name] for name in images):
            raise ValueError('Prepared local image changed')
        generation = plan['generation']
        body = dict(model=plan['model'], prompt=plan['prompt'],
            images=[base64.b64encode((inputs/name).read_bytes()).decode() for name in images],
            stream=generation['stream'], think=generation['think'],
            options={k:v for k,v in generation.items() if k not in ('stream', 'think')})
        start = time.monotonic()
        answer = local_json('/api/generate', body)
        (output/f'answer_{center}.json').write_text(json.dumps(answer, indent=2)+'\n')
        rows.append(dict(center=center, prediction=parse_ownership(answer.get('response', '')),
                         elapsed_seconds=time.monotonic()-start,
                         raw_answer_sha256=digest(output/f'answer_{center}.json')))
        cached.write_text(json.dumps(rows, indent=2)+'\n')
        print('Completed local ownership sample', len(rows), '/', len(samples), flush=True)
    require_model(plan)
    if sources != {name:digest(ROOT/'experiments'/name) for name in names}:
        raise ValueError('Ownership inference source changed during generation')
    result = dict(plan_sha256=digest(plan_path), source_sha256=sources,
        input_sha256={'data/control_local_ownership_v1/inputs/manifest.json':digest(inputs/'manifest.json')},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},
        targets_not_read=True, local_model_digest=plan['model_digest'], runtime_version=plan['runtime_version'],
        samples=len(rows), proxy_bypass=True, redirects_forbidden=True, endpoint='http://127.0.0.1:11434')
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Local ownership inference complete; no possession targets read', flush=True)


if __name__ == '__main__':
    main()
