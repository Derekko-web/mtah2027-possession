"""Download one pinned public model and inspectable source, excluding upstream data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from research_temporal_dead import write_json


def main():
    folder=ROOT/'data/control_pathcrf_v1';plan_path=ROOT/'experiments/PATHCRF_FEASIBILITY_PLAN.json';plan=json.loads(plan_path.read_text())
    if plan_path.read_bytes()!=(folder/'protocol_freeze_A/plan.json').read_bytes():raise ValueError('PathCRF source plan changed')
    tree=json.loads((folder/'source_catalog/tree.json').read_text())
    if tree['sha']!=plan['commit'] or tree.get('truncated'):raise ValueError('Incomplete or different pinned source tree')
    source=folder/'upstream';source.mkdir(exist_ok=False)
    checkpoint=plan['checkpoint'];records=[]
    for r in tree['tree']:
        name=r['path']
        allowed=(name.endswith('.py') or name in ('README.md','LICENSE','requirements.txt','saved/140/args.json',checkpoint['path']))
        if r['type']=='blob' and allowed:records.append(r)
    item=next(r for r in records if r['path']==checkpoint['path'])
    if item['sha']!=checkpoint['git_blob_sha1'] or item['size']!=checkpoint['bytes']:raise ValueError('Predeclared checkpoint identity differs')
    def fetch(record):
        name=record['path'];url='https://raw.githubusercontent.com/hyunsungkim-ds/pathcrf/'+plan['commit']+'/'+name
        with urllib.request.urlopen(url,timeout=120) as response:content=response.read()
        git_sha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if len(content)!=record['size'] or git_sha!=record['sha']:raise ValueError('Publisher source/checkpoint identity failed: '+name)
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle:handle.write(content)
        return dict(path=name,bytes=len(content),git_blob_sha1=git_sha,sha256=digest(path))
    with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(fetch,records))
    write_json(folder/'source_manifest.json',dict(repo=plan['repo'],commit=plan['commit'],plan_sha256=digest(plan_path),
        catalog_sha256=digest(folder/'source_catalog/tree.json'),license='MPL-2.0',source_files=result,checkpoints_downloaded=1,
        source_sha256=digest(Path(__file__)),competition_data_loaded=False,external_observations_loaded=False))
    print('Verified pinned public source files',len(result),'checkpoint bytes',checkpoint['bytes'],flush=True)


if __name__=='__main__':main()
