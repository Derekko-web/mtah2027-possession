"""Target-free full-match extraction with the unchanged R37 fixed encoder."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/control_tabpfn_full_build_v1'
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from games import source_frames,half_video_blocks,GAME
    from player_representation import ROUND,check_inputs,audit_cache,digest,write_json,interpolate
    from encode_player_representation import reflected
    from masked_player_network import initialize,MaskedPlayer
    from masked_player_native_data import assemble_native
    from tabpfn_full_features import interpolate_halves
    plan=json.loads((OUT/'feature_plan.json').read_text())
    for name,sha in plan['sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Full feature binding changed: '+name)
    confirmed=ROOT/'data/control_tabpfn_fixed_confirmation_v2'
    for name in ('final_selection.json','replay_complete.json'):
        if json.loads((confirmed/name).read_text()).get('passed') is not True:
            raise ValueError('Fresh qualification/replay incomplete')
    check_inputs();source=ROUND/'inputs';audit_cache(source);audit_cache(ROUND/'encodings')
    output=OUT/'encodings';output.mkdir(exist_ok=False)
    half=np.load(source/'half_id.npy');frames=np.flatnonzero(half>=0)*5
    x,mask,teams=[np.load(source/(name+'.npy'),mmap_mode='r') for name in ('xy','observed','teams')]
    initialize();model=MaskedPlayer()
    model.load_state_dict(torch.load(ROOT/'data/control_masked_player_v1/external_models/temporal/model.pt',map_location='cpu',weights_only=True),strict=True)
    parts=[];supported=[]
    oldframes=np.load(ROUND/'encodings/frames.npy');oldhidden=np.load(ROUND/'encodings/pretrained_latent.npy')
    oldeligible=np.load(ROUND/'encodings/eligible.npy')
    for start in range(0,len(frames),64):
        index=np.arange(start,min(start+64,len(frames)));size=len(index)
        if size<64:index=np.r_[index,np.repeat(index[-1],64-size)]
        values=assemble_native(x,mask,teams,half,frames[index]//5)
        z=reflected(model,values)[:size]
        counts=np.stack([(values[1]&(values[2]==team)).sum(-1) for team in (0,1)],axis=-1)[:size]
        eligible=(counts[:,15]>=4).all(1)&(counts.mean(1)>=4).all(1)
        current=frames[index[:size]]
        for local,frame in enumerate(current):
            old=np.searchsorted(oldframes,frame)
            if old<len(oldframes) and oldframes[old]==frame:
                np.testing.assert_array_equal(z[local],oldhidden[old])
                np.testing.assert_array_equal(eligible[local],oldeligible[old])
        parts.append(z);supported.append(eligible)
        if start%1024==0:print('Encoded',min(start+64,len(frames)),'of',len(frames),'centers',flush=True)
    hidden=np.concatenate(parts);eligible=np.concatenate(supported)
    public=np.searchsorted(frames,oldframes)
    np.testing.assert_array_equal(frames[public],oldframes)
    np.testing.assert_array_equal(hidden[public],oldhidden)
    np.testing.assert_array_equal(eligible[public],oldeligible)
    blocks=[(a,b+1) for _,a,b in half_video_blocks(GAME)]
    features,valid=interpolate_halves(frames,hidden,eligible,source_frames(),blocks)
    expected=interpolate(oldframes,oldhidden,oldeligible,source_frames())
    for actual,previous in zip((features,valid),expected):
        np.testing.assert_array_equal(actual[11185:18685],previous[11185:18685])
    for name,value in (('frames',frames),('pretrained_latent',hidden),('eligible',eligible),
                       ('full_latent',features),('full_eligible',valid)):
        np.save(output/(name+'.npy'),value)
    check_inputs()
    write_json(output/'manifest.json',dict(targets_loaded=False,encoder_trained=False,full_frames=source_frames(),
        public_hidden_and_eligibility_exact=True,public_interpolation_exact=True,
        encoder_batch_size=64,reflection_views=4,feature_plan_sha256=digest(OUT/'feature_plan.json'),
        cache_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    print('Full fixed encoder complete; public parity exact; no targets read',flush=True)


if __name__=='__main__':main()
