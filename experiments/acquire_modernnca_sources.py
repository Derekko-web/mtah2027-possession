"""Acquire only small pinned author text for R64; never install or load models."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'data/control_modernnca_v1'
REVISION = '39b54024902955e51063a34bce17d514cd68ff42'
TABR_REVISION = '17baa9082506f8e7a0f8d11bb1e08212926a1507'
SOURCES = {
    'TALENT/model/models/modernNCA.py': (4470, '02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f'),
    'TALENT/model/methods/modernNCA.py': (15409, '608c85e94714718fd5690a8096890b6232f2da94d0c94d5489734278df112d3e'),
    'TALENT/model/lib/tabr/utils.py': (12259, '46f26474f559cdda4eda92c5d28d504905c3180ad0ead1bc3f296af72c74230b'),
    'TALENT/model/utils.py': (32779, '74a48976d4230793a41d96e743c25c0685fe8f9b91cd1b0e85750fe202d79ead'),
    'TALENT/configs/default/modernNCA.json': (585, 'ea4dc58f92ea9c5d0be7e1c35b38d28d247b6d3c589da10e16c2dcd951a74700'),
    'TALENT/configs/deep_configs.json': (529, 'fcaafb28503aa72d8170c87851c582704acfac1fbbeed9da8b06e53949d9aad6'),
    'LICENSE': (1101, '93cf90600c386cb5719779090fbc37080d5833eeec8d1f66cc370ba48af2d422'),
    'pyproject.toml': (705, '5a8b2fcf178bb30670a162fab2fd95fc77d40a1c329c5371df1f4a0a232bb544'),
    'TABR_LICENSE': (1127, 'e2a0b9b891e143bf0512e1a862af308779228e973d61e75fbb774ef32bc32761'),
}


def source_url(name):
    if name == 'TABR_LICENSE':
        return f'https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/{TABR_REVISION}/LICENSE'
    return f'https://raw.githubusercontent.com/LAMDA-Tabular/TALENT/{REVISION}/{name}'


def acquire():
    ROUND.mkdir(exist_ok=False)
    folder = ROUND / 'public_sources'
    folder.mkdir()
    records = {}
    for name, (count, expected) in SOURCES.items():
        url = source_url(name)
        with urllib.request.urlopen(url, timeout=30) as response:
            body = response.read(100_001)
        if len(body) != count or hashlib.sha256(body).hexdigest() != expected:
            raise ValueError('Pinned text identity mismatch: ' + name)
        destination = folder / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(body)
        records[str(destination.relative_to(ROOT))] = dict(url=url, bytes=count, sha256=expected)
    (ROUND / 'acquisition.json').write_text(json.dumps(dict(
        revision=REVISION, tabr_revision=TABR_REVISION, files=records,
        source_text_only=True, model_weights=False, installs=False,
        scientific_imports=False, native_inputs_read=False,
    ), indent=2, sort_keys=True) + '\n')
    print('Acquired', len(records), 'small exact author text files; no models or dependencies')


if __name__ == '__main__':
    acquire()
