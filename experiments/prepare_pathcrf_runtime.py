"""Keep pinned model math intact while extracting its one dependency-free helper."""
import ast
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from research_temporal_dead import write_json


def main():
    folder=ROOT/'data/control_pathcrf_v1';source=folder/'upstream';output=folder/'inference_source';output.mkdir(exist_ok=False)
    manifest=json.loads((folder/'source_manifest.json').read_text())
    for r in manifest['source_files']:
        if digest(source/r['path'])!=r['sha256']:raise ValueError('Pinned upstream file changed')
    names=['models/set_lstm.py','models/static_dense_crf.py','models/static_sparse_crf.py','models/dynamic_dense_crf.py','models/dynamic_sparse_crf.py',
        'set_transformer/attention.py','set_transformer/blocks.py','set_transformer/model.py','datatools/config.py','LICENSE']
    for name in names:
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,path)
    for package in ('models','set_transformer','datatools'):(output/package/'__init__.py').write_text('')
    original=(source/'models/utils.py').read_text();tree=ast.parse(original)
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='build_edge_compression_maps')
    exact=ast.get_source_segment(original,function)
    utility='# Extracted verbatim from pinned PathCRF models/utils.py; MPL-2.0, see ../LICENSE.\nfrom typing import Tuple\nimport torch\n\n'+exact+'\n'
    (output/'models/utils.py').write_text(utility)
    extracted=next(n for n in ast.parse(utility).body if isinstance(n,ast.FunctionDef))
    if ast.dump(function,include_attributes=False)!=ast.dump(extracted,include_attributes=False):raise ValueError('Extracted helper math changed')
    (output/'ADAPTATION.md').write_text('PathCRF commit '+manifest['commit']+'; MPL-2.0 license retained.\nNine model/attention/config files copied byte-for-byte. models/utils.py contains only the verbatim build_edge_compression_maps helper plus Torch/typing imports. Unused PyG-dependent batching/training utilities are omitted. No model/CRF math or state keys are changed. Empty package initializers added.\n')
    write_json(folder/'runtime_source_manifest.json',dict(upstream_manifest_sha256=digest(folder/'source_manifest.json'),helper_ast_exact=True,
        unchanged_files=names,cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        source_sha256=digest(Path(__file__))))
    print('Pure-Torch source closure prepared; exact edge helper; original model math retained',flush=True)


if __name__=='__main__':main()
