"""Prepare checksum-pinned public YOLO11 Core ML packages locally."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile

from build_temporal_candidate import digest

ROOT = Path(__file__).resolve().parents[1]


def verify_packages(plan, asset_root):
    for model in plan['models']:
        archive = asset_root/(model['name']+'.mlpackage.zip')
        if digest(archive) != model['archive_sha256']:
            raise ValueError('Public model archive checksum changed')
        package = asset_root/(model['name']+'.mlpackage')
        with zipfile.ZipFile(archive) as zipped:
            expected = {}
            for entry in zipped.infolist():
                if not (package/entry.filename).resolve().is_relative_to(package.resolve()):
                    raise ValueError('Unsafe public package member')
                if not entry.is_dir():
                    expected[entry.filename] = hashlib.sha256(zipped.read(entry)).hexdigest()
        actual = {str(p.relative_to(package)):digest(p) for p in sorted(package.rglob('*')) if p.is_file()}
        if actual != expected:
            raise ValueError('Public package differs from pinned archive')


def verify_assets(plan, manifest, root=ROOT):
    asset_root = root/manifest['asset_root']
    verify_packages(plan, asset_root)
    for name, expected in manifest['source_sha256'].items():
        if digest(root/name) != expected:
            raise ValueError('Modern asset-generation source changed')
    for model in plan['models']:
        compiled = asset_root/(model['name']+'.mlmodelc')
        actual = {str(p.relative_to(compiled)):digest(p) for p in sorted(compiled.rglob('*')) if p.is_file()}
        if actual != manifest['compiled_sha256'][model['name']]:
            raise ValueError('Modern compiled model changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/control_modern_ball_v1/model_assets')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT/'data'):
        raise ValueError('Keep native model artifacts under ignored data/')
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT/'experiments/MODERN_BALL_PLAN.json').read_text())
    previous = output/'asset_manifest.json'
    if previous.exists():
        verify_assets(plan, json.loads(previous.read_text()))
        print('Existing modern model assets verified')
        return
    metadata, compiled_hashes = {}, {}
    for model in plan['models']:
        name = model['name']
        archive = output/(name+'.mlpackage.zip')
        original = ROOT/model['archive']
        if not archive.exists():
            if original.exists() and digest(original) == model['archive_sha256']:
                shutil.copy2(original, archive)
            else:
                subprocess.run(['curl', '-fL', '--no-progress-meter', '--show-error',
                                model['url'], '-o', str(archive)], check=True)
        if digest(archive) != model['archive_sha256']:
            raise ValueError('Public model archive checksum differs')
        package = output/(name+'.mlpackage')
        if not package.exists():
            package.mkdir()
            with zipfile.ZipFile(archive) as zipped:
                for entry in zipped.infolist():
                    if not (package/entry.filename).resolve().is_relative_to(package):
                        raise ValueError('Unsafe public package member')
                zipped.extractall(package)
        compiled = output/(name+'.mlmodelc')
        if not compiled.exists():
            with (output/(name+'_metadata.json')).open('w') as dest, (output/(name+'_compile.log')).open('w') as log:
                subprocess.run(['osascript', '-l', 'JavaScript', str(ROOT/'experiments/native_detector_bootstrap.js'),
                                str(package.relative_to(ROOT)), str(compiled.relative_to(ROOT))],
                               cwd=ROOT, stdout=dest, stderr=log, check=True)
        reply = subprocess.run(['osascript', '-l', 'JavaScript', str(ROOT/'experiments/modern_detector_metadata.js'),
                                str(compiled.relative_to(ROOT))], cwd=ROOT, text=True, capture_output=True, check=True)
        shape = json.loads(reply.stdout)
        if (shape['width'], shape['height']) != (plan['input_width'], plan['input_height']) or set(shape['outputs']) != {'coordinates', 'confidence'}:
            raise ValueError('Modern model interface differs')
        metadata[name] = shape
        compiled_hashes[name] = {str(p.relative_to(compiled)):digest(p) for p in sorted(compiled.rglob('*')) if p.is_file()}
        print('Modern public model prepared', name, flush=True)
    verify_packages(plan, output)
    sources = ('experiments/prepare_modern_models.py', 'experiments/modern_detector_metadata.js', 'experiments/native_detector_bootstrap.js')
    previous.write_text(json.dumps(dict(asset_root=str(output.relative_to(ROOT)),
        source_sha256={name:digest(ROOT/name) for name in sources}, compiled_sha256=compiled_hashes,
        interfaces=metadata, macos_version=platform.mac_ver()[0], no_competition_inputs=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
