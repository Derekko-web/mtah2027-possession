"""Frozen R37 interfaces: masked-set representations and live-owner fusion."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/'data/control_player_representation_v1'
PLAN=ROOT/'experiments/PLAYER_REPRESENTATION_PLAN.json'
PROTOCOLS=('buffered_minutes','purged_spells','buffered_100_seconds')
VIEWS=('pretrained_latent','random_latent')
COLUMNS=[f'{team}__set_latent_{i:02}' for team in ('h','a') for i in range(32)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


def check_inputs():
    plan=json.loads(PLAN.read_text());freeze=ROUND/'protocol_freeze'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('R37 plan changed')
    for source in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text()),json.loads((freeze/'protected_artifacts.json').read_text())):
        for name,sha in source.items():
            if digest(ROOT/name)!=sha:raise ValueError('Frozen dependency differs: '+name)
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((freeze/'EXPERIMENT_LEDGER.csv').read_bytes()):raise ValueError('Earlier ledger changed')
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink())!=target:raise ValueError('Source symlink changed')
    return plan


def audit_cache(folder):
    record=json.loads((folder/'manifest.json').read_text())
    for name,sha in record['cache_sha256'].items():
        if digest(folder/name)!=sha:raise ValueError('Saved R37 artifact differs: '+name)
    return record


def interpolate(frames,hidden,supported,n):
    """Adjacent 5 Hz endpoints; never bridge an unsupported observation."""
    frames=np.asarray(frames);hidden=np.asarray(hidden);supported=np.asarray(supported,bool)
    if len(frames)!=len(hidden) or len(frames)!=len(supported) or not np.all(np.diff(frames)==5):raise ValueError('Nonuniform representation grid')
    public=np.arange(11185,18685)
    if frames[0]!=public[0] or frames[-1]<public[-1]:raise ValueError('Missing interpolation endpoint')
    left=np.searchsorted(frames,public,side='right')-1;exact=frames[left]==public
    right=np.minimum(left+1,len(frames)-1);right[exact]=left[exact]
    weight=((public-frames[left])/5).astype(np.float64)
    value=(hidden[left].astype(np.float64)*(1-weight[:,None])+hidden[right].astype(np.float64)*weight[:,None]).astype(np.float32)
    eligible=supported[left]&supported[right]
    features=np.full((n,64),np.nan,np.float32);features[public[eligible]]=value[eligible]
    mask=np.zeros(n,bool);mask[public]=eligible
    return features,mask


def live_overlay(reference,alternate):
    result=reference.copy();keep=np.isin(reference,['H','A'])&np.isin(alternate,['H','A']);result[keep]=alternate[keep]
    return result
