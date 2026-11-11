"""Download pinned public DFL files and assess only the frozen Phase A protocol."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dfl_transfer_data import digest,fetch_file,parse_match,quality_summary

ROUND=ROOT/'data/control_dfl_transfer_v1'
PLAN=ROOT/'experiments/DFL_TRANSFER_FEASIBILITY_PLAN.json'


def validate():
    plan=json.loads(PLAN.read_text());freeze=ROUND/'protocol_freeze_A'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Phase A protocol changed')
    for name,expected in plan['input_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen Phase A input changed: '+name)
    for name,expected in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Frozen Phase A source changed: '+name)
    for name,expected in json.loads((freeze/'protected_artifacts.json').read_text()).items():
        if digest(ROOT/name)!=expected:raise ValueError('Protected selected/reference changed')
    if (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes()!=(freeze/'ledger_before.csv').read_bytes():
        raise ValueError('Phase A ledger changed')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        p=ROOT/name
        if not p.is_symlink() or str(p.readlink())!=target:raise ValueError('Source symlink changed')
    return plan


def download():
    plan=validate();records=plan['files']
    def one(record):
        path=fetch_file(record,ROUND/'raw')
        print('Verified public source',record['id'],path.name,flush=True)
        return dict(name=record['name'],size=path.stat().st_size,md5=digest(path,'md5'),sha256=digest(path))
    with ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(one,records))
    manifest=dict(catalog_sha256=plan['input_sha256']['data/control_dfl_transfer_v1/catalog/article_28196177.json'],
        files=result,license='CC-BY-4.0',source='https://doi.org/10.6084/m9.figshare.28196177.v1',observations_processed=0)
    with (ROUND/'download_manifest.json').open('x') as handle:handle.write(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(files=len(result),bytes=sum(r['size'] for r in result),all_publisher_checksums_verified=True),indent=2))


def assess(output):
    plan=validate();output.mkdir(parents=True)
    (output/'plan.json').write_bytes(PLAN.read_bytes())
    download=json.loads((ROUND/'download_manifest.json').read_text())
    for record in download['files']:
        if digest(ROUND/'raw'/record['name'])!=record['sha256']:raise ValueError('Downloaded source changed')
    games={};artifacts={}
    for game in plan['matches']:
        print('Assessing public DFL match',game,flush=True)
        records=[r for r in plan['files'] if game in r['name']]
        info=next(r for r in records if 'matchinformation' in r['name'])
        pos=next(r for r in records if 'positions_raw' in r['name'])
        outputs,identity=parse_match(ROUND/'raw'/info['name'],ROUND/'raw'/pos['name'],game)
        summary=quality_summary(outputs,plan['quality_gate']);summary['identity']=identity
        summary['split']=next(name for name,values in plan['splits'].items() if game in values)
        games[game]=summary
        for section,current in outputs.items():
            file=output/f'{game}_{section}.npz'
            np.savez_compressed(file,**current);artifacts[file.name]=digest(file)
        print('DFL aggregate',game,json.dumps(dict(frames=summary['canonical_frames'],passed=summary['passed'],checks=summary['checks'])),flush=True)
        del outputs
    passed=all(v['passed'] for v in games.values())
    summary=dict(round=27,phase='A',passed=passed,result='accepted_data_feasibility' if passed else 'rejected_data_feasibility',
        matches=games,total_canonical_frames=sum(g['canonical_frames'] for g in games.values()),
        splits=plan['splits'],new_fits=0,new_competition_target_reads=0,new_score_rows=0,selected_unchanged=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    validate()
    manifest=dict(plan_sha256=digest(PLAN),input_sha256=plan['input_sha256'],
        raw_files=download['files'],source_sha256=json.loads((ROUND/'protocol_freeze_A/implementation.json').read_text()),
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='matches'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',action='store_true')
    p.add_argument('--output',type=Path,default=ROUND/'assessment_A')
    args=p.parse_args()
    if args.download:download()
    else:assess(args.output.resolve())
