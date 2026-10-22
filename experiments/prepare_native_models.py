"""Fetch checksum-pinned public Apple models and compile them with native Core ML."""

import argparse
import json
import platform
from pathlib import Path
import shutil
import subprocess

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/control_native_ball_v1/model_assets')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'data'):
        raise ValueError('Keep native model artifacts under ignored data/')
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT/'experiments/NATIVE_BALL_PLAN.json').read_text())
    rows = []
    for model in plan['models']:
        name = model['name']
        original = ROOT/model['source_model']
        path = output/(name+'.mlmodel')
        url = f'https://ml-assets.apple.com/coreml/models/Image/ObjectDetection/{name}/{name}.mlmodel'
        if not path.exists():
            if original.exists() and digest(original) == model['sha256']:
                shutil.copy2(original, path)
            else:
                temporary = output/(name+'.download')
                subprocess.run(['curl', '-fL', '--no-progress-meter', '--show-error', url, '-o', str(temporary)], check=True)
                if digest(temporary) != model['sha256']:
                    raise ValueError('Public native model checksum differs')
                temporary.replace(path)
        if digest(path) != model['sha256']:
            raise ValueError('Native source model changed')
        compiled = output/(name+'.mlmodelc')
        if not compiled.exists():
            with (output/(name+'_metadata.json')).open('w') as metadata, (output/(name+'_compile.log')).open('w') as log:
                subprocess.run(['osascript', '-l', 'JavaScript', str(ROOT/'experiments/native_detector_bootstrap.js'),
                    str(path.relative_to(ROOT)), str(compiled.relative_to(ROOT))], cwd=ROOT,
                    stdout=metadata, stderr=log, check=True)
        actual = {str(p.relative_to(compiled)):digest(p) for p in sorted(compiled.rglob('*')) if p.is_file()}
        previous = {str(Path(key).relative_to(Path(model['compiled_model']))):value for key, value in model['compiled_sha256'].items()}
        rows.append(dict(model=name, source_url=url, source_sha256=digest(path), compiled_sha256=actual,
                         compiled_matches_recorded_generation=actual == previous))
        print('Public native model prepared', name, 'compiled generation matches:', actual == previous, flush=True)
    sources = ('experiments/prepare_native_models.py', 'experiments/native_detector_bootstrap.js')
    (output/'asset_manifest.json').write_text(json.dumps(dict(models=rows, no_competition_inputs=True,
        raw_model_sources_pinned=True, macos_version=platform.mac_ver()[0],
        source_sha256={name:digest(ROOT/name) for name in sources}), indent=2)+'\n')


if __name__ == '__main__':
    main()
