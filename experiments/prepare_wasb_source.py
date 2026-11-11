"""Acquire one pinned official soccer detector, preserving all source notices."""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request
from player_representation import ROOT,digest,write_json

FOLDER=ROOT/'data/control_wasb_v1'
PLAN=ROOT/'experiments/WASB_FEASIBILITY_PLAN.json'


def config_yaml(text):
    result={};stack=[(-1,result)]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        indent=len(line)-len(line.lstrip());key,value=line.strip().split(':',1);value=value.strip()
        while stack[-1][0]>=indent:stack.pop()
        if not value:
            child={};stack[-1][1][key]=child;stack.append((indent,child))
        else:
            try:parsed=ast.literal_eval(value)
            except (ValueError,SyntaxError):parsed=value
            stack[-1][1][key]=parsed
    return result


def main():
    plan=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze_A'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('WASB source plan changed')
    for name,sha in json.loads((freeze/'implementation.json').read_text()).items():
        if digest(ROOT/name)!=sha:raise ValueError('WASB source adapter changed')
    output=FOLDER/'source';output.mkdir(parents=True,exist_ok=False)
    base='https://raw.githubusercontent.com/nttcom/WASB-SBDT/'+plan['commit']+'/'
    with urllib.request.urlopen('https://api.github.com/repos/nttcom/WASB-SBDT/git/trees/'+plan['commit']+'?recursive=1',timeout=60) as response:tree=json.load(response)
    if tree.get('truncated') or tree['sha']!=plan['commit']:raise ValueError('Incomplete official tree')
    write_json(output/'tree.json',tree);catalog={x['path']:x for x in tree['tree']};records=[]
    for name in plan['source_files']:
        with urllib.request.urlopen(base+name,timeout=60) as response:value=response.read()
        item=catalog[name];sha=hashlib.sha1(b'blob '+str(len(value)).encode()+b'\0'+value).hexdigest()
        if sha!=item['sha'] or len(value)!=item['size']:raise ValueError('Pinned source differs')
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value);records.append(dict(path=name,sha256=digest(path),git_blob_sha1=sha,bytes=len(value)))
    checkpoint=output/'wasb_soccer_best.pth.tar'
    with urllib.request.urlopen(plan['checkpoint_url'],timeout=120) as response:value=response.read(plan['checkpoint_bytes']+1)
    if len(value)!=plan['checkpoint_bytes']:raise ValueError('Unexpected checkpoint length')
    checkpoint.write_bytes(value)
    cfg=config_yaml((output/'src/configs/model/wasb.yaml').read_text());write_json(output/'config.json',cfg)
    if cfg['frames_in']!=3 or cfg['frames_out']!=3 or cfg['inp_width']!=512 or cfg['inp_height']!=288 or cfg['out_scales']!=[0]:raise ValueError('Unexpected published interface')
    write_json(FOLDER/'source_manifest.json',dict(plan_sha256=digest(PLAN),repo=plan['repo'],commit=plan['commit'],source_files=records,checkpoint_sha256=digest(checkpoint),checkpoint_bytes=len(value),competition_data_loaded=False,external_media_loaded=False,
        terms=plan['terms'],cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))
    print('Pinned WASB source and one soccer checkpoint verified; no media processed',flush=True)


if __name__=='__main__':main()
