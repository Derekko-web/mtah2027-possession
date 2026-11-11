"""Conditional fixed optical-contact opportunity; all native cases remain visible."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import numpy as np
from player_representation import ROOT,digest,write_json
from wasb_adapter import warp
from wasb_decode import selection_expression
from wasb_components import normalize_warped,components

FOLDER=ROOT/'data/control_wasb_native_v1'
PLAN=ROOT/'experiments/WASB_NATIVE_PLAN.json'


def validate():
    plan=json.loads(PLAN.read_text());freeze=FOLDER/'protocol_freeze'
    if PLAN.read_bytes()!=(freeze/'plan.json').read_bytes():raise ValueError('Native optical plan changed')
    confirmation=ROOT/'data/control_wasb_confirmation_v1/assessment/selection.json';acceleration=ROOT/'data/control_wasb_acceleration_v1/assessment/summary.json'
    if not json.loads(confirmation.read_text())['passed']:raise ValueError('Disjoint external detector confirmation failed')
    if not json.loads(acceleration.read_text())['passed']:raise ValueError('Fixed MPS numerical/runtime gate failed')
    for group in (plan['input_sha256'],json.loads((freeze/'implementation.json').read_text()),json.loads((freeze/'protected_artifacts.json').read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('Native optical dependency differs: '+name)
    ledger=ROOT/'experiments/EXPERIMENT_LEDGER.csv'
    if not ledger.read_bytes().startswith((freeze/'EXPERIMENT_LEDGER.csv').read_bytes()):raise ValueError('Earlier ledger changed')
    binding=freeze/'external_gate_binding.json';value={str(p.relative_to(ROOT)):digest(p) for p in (confirmation,acceleration)}
    if binding.exists():
        if json.loads(binding.read_text())!=value:raise ValueError('External gate binding differs')
    else:write_json(binding,value)
    return plan


def check_cache(folder):
    record=json.loads((folder/'manifest.json').read_text())
    for name,sha in record['cache_sha256'].items():
        if digest(folder/name)!=sha:raise ValueError('Immutable native optical artifact differs: '+name)
    return record


def decode_one(video,frame,timebase,start_pts):
    pts=Fraction(int(frame),25)/timebase+start_pts
    if pts.denominator!=1:raise ValueError('Nonintegral original timestamp')
    command=['ffmpeg','-nostdin','-v','error','-threads','1','-copyts','-ss',f'{max(0.,int(frame)/25-2):.8f}','-i',str(video),'-vf',f'select=eq(pts\\,{int(pts)})','-filter_threads','1','-frames:v','1','-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    pixels=subprocess.check_output(command)
    if len(pixels)!=1920*1080*3:raise ValueError('Native timestamp witness missing')
    return np.frombuffer(pixels,np.uint8).reshape(1080,1920,3)


def prepare():
    import pandas as pd
    from games import GAME,half_video_blocks,require
    from broadcast_alignment import load_source
    from starter.load_homography import load
    plan=validate();output=FOLDER/'inputs';output.mkdir(parents=True,exist_ok=False);frames=np.arange(11185,18685,5)
    reserved=json.loads((ROOT/plan['reserved_metadata']).read_text())['windows'];held=np.load(ROOT/plan['held_image_centers']);allowed=np.zeros(len(frames),bool)
    for _,a,b in half_video_blocks(GAME):allowed|=(frames-1>=a)&(frames+1<=b)
    for row in reserved:allowed&=~((frames+1>=row['start']-250)&(frames-1<=row['end']+250))
    allowed&=~(np.abs(frames[:,None]-held[None])<=251).any(1)
    with np.load(ROOT/'data/control_dfl_dead_v1/native_inputs/native_observations.npz') as source:positions=source['positions'];teams=source['teams'];ball=source['ball']
    for n in (18935,19060):
        source=load_source('tactical',n);value=source['positions'];value[~source['valid']]=np.nan
        np.testing.assert_array_equal(value[11185:18685],positions[11185:18685]);np.testing.assert_array_equal(source['teams'],teams)
        np.testing.assert_array_equal(source['ball'][11185:18685],ball[11185:18685])
    matrices,error,valid=load();count=np.stack([np.isfinite(positions[frames][:,teams==team]).all(-1).sum(-1) for team in (0,1)],axis=1)
    eligible=allowed&valid[frames]&np.isfinite(matrices[frames]).all((1,2))&(count>=4).all(1)
    video=Path(require(['video'],source='tactical')[0]);metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,time_base,start_pts,r_frame_rate','-of','json',str(video)]))['streams'][0]
    if metadata['width']!=1920 or metadata['height']!=1080 or metadata['r_frame_rate']!='25/1':raise ValueError('Native video schema changed')
    needed=np.unique((frames[eligible,None]+[-1,0,1]).ravel());timebase=Fraction(metadata['time_base']);start_pts=int(metadata.get('start_pts',0));pts=[Fraction(int(f),25)/timebase+start_pts for f in needed]
    if any(p.denominator!=1 for p in pts):raise ValueError('Nonintegral native original timestamps')
    np.save(output/'frames.npy',frames);np.save(output/'eligible.npy',eligible);np.save(output/'decoded_frames.npy',needed)
    np.save(output/'positions.npy',positions[frames]);np.save(output/'teams.npy',teams);np.save(output/'tracked_ball.npy',ball[frames]);np.save(output/'homography.npy',matrices[frames])
    write_json(output/'selection_before_targets.json',dict(frames_sha256=digest(output/'frames.npy'),eligible_sha256=digest(output/'eligible.npy'),targets_loaded=False,metadata=metadata))
    pixels=np.lib.format.open_memmap(output/'rgb.npy',mode='w+',dtype=np.uint8,shape=(len(needed),288,512,3));select=selection_expression([int(p) for p in pts]).replace('eq(n\\,','eq(pts\\,')
    command=['ffmpeg','-nostdin','-v','error','-threads','2','-copyts','-ss',f'{max(0.,int(needed[0])/25-2):.8f}','-i',str(video),'-vf','select='+select,'-filter_threads','1','-frames:v',str(len(needed)),'-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    witness=set([0,len(needed)//2,len(needed)-1]);witnesses=[]
    with (output/'decoder.log').open('wb') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for i,frame in enumerate(needed):
                data=bytearray();size=1920*1080*3
                while len(data)<size:
                    part=process.stdout.read(size-len(data))
                    if not part:raise ValueError('Native selected video ended early')
                    data.extend(part)
                image=np.frombuffer(data,np.uint8).reshape(1080,1920,3);pixels[i]=warp(image)
                if i in witness:
                    direct=decode_one(video,frame,timebase,start_pts);np.testing.assert_array_equal(image,direct);witnesses.append(dict(frame=int(frame),exact=True))
                if (i+1)%500==0:print('Native target-blind decoded frames',i+1,'/',len(needed),flush=True)
            if process.stdout.read(1) or process.wait()!=0:raise ValueError('Native decoder count/status differs')
        finally:
            process.stdout.close()
            if process.poll() is None:process.terminate();process.wait()
    pixels.flush();coverage={}
    for protocol in ('buffered_minutes','purged_spells','buffered_100_seconds'):
        table=pd.read_csv(ROOT/'data/control_dfl_dead_v1/public_trial'/protocol/'oof_predictions.csv',usecols=['frame','current_seed42']);np.testing.assert_array_equal(table.frame,np.arange(11185,18685))
        live=np.isin(table.current_seed42.to_numpy()[frames-11185],['H','A']);coverage[protocol]=float((eligible&live).sum()/max(live.sum(),1))
    passed=all(v>=.8 for v in coverage.values());write_json(output/'summary.json',dict(passed=passed,uniform_centers=len(frames),eligible_centers=int(eligible.sum()),input_coverage=coverage,actual_prefix_tracking_exact=True,exact_timestamp_witnesses=witnesses,targets_loaded=False,no_images_displayed=True))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));validate()
    if not passed:raise ValueError('Native physical video/metadata coverage failed')


def infer():
    import torch,resource,time
    from check_wasb_runtime import load_model
    from masked_player_network import initialize
    validate();source=FOLDER/'inputs';check_cache(source)
    if not json.loads((source/'summary.json').read_text())['passed']:raise ValueError('Native physical support failed')
    output=FOLDER/'outputs';output.mkdir(parents=True,exist_ok=False);frames=np.load(source/'frames.npy');eligible=np.load(source/'eligible.npy');decoded=np.load(source/'decoded_frames.npy');indices=np.flatnonzero(eligible);initialize();model=load_model().to('mps');started=time.monotonic()
    def predict(model,rgb,center,mode):
        selected=center+np.asarray([-1,0,1]) if mode=='temporal' else np.repeat(center,3);rows=np.searchsorted(decoded,selected);np.testing.assert_array_equal(decoded[rows],selected)
        value=np.concatenate([normalize_warped(rgb[r]) for r in rows],axis=0)[None]
        with torch.inference_mode():hm=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(value)).to('mps'))[0])[0,1].cpu().numpy()
        result=components(hm,1920,1080)
        return max(result,key=lambda x:x[2]) if result else (np.nan,np.nan,np.nan)
    results={mode:np.full((len(frames),3),np.nan) for mode in ('temporal','frame')}
    for i,index in enumerate(indices):
        rgb=np.load(source/'rgb.npy',mmap_mode='r')
        for mode in results:results[mode][index]=predict(model,rgb,frames[index],mode)
        rgb._mmap.close()
        if (i+1)%100==0:print('Native immutable paired detections',i+1,'/',len(indices),flush=True)
    loaded=load_model().to('mps')
    for index in indices[np.rint(np.linspace(0,len(indices)-1,16)).astype(int)]:
        rgb=np.load(source/'rgb.npy',mmap_mode='r')
        for mode in results:np.testing.assert_array_equal(results[mode][index],predict(loaded,rgb,frames[index],mode))
        rgb._mmap.close()
    for mode,value in results.items():np.save(output/(mode+'_detections.npy'),value)
    write_json(output/'summary.json',dict(targets_loaded=False,new_fits=0,device='mps',MPS_fallback_disabled=True,strict_fresh_reload_witnesses=32,elapsed_seconds=time.monotonic()-started,cpu_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,MPS_driver_GiB=torch.mps.driver_allocated_memory()/1024**3))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));validate()


def contact_arrays(points,positions,teams,matrices,eligible):
    from color_contact import contact_scores
    from starter.load_homography import unproject
    from games import GAME,GAMES
    pitch=np.asarray(GAMES[GAME]['pitch']);metadata=np.eye(2)[teams];probability=np.full(len(points),np.nan);support=np.zeros(len(points),bool);strong=np.zeros(len(points),bool)
    for i,point in enumerate(points):
        if not eligible[i] or not np.isfinite(point).all():continue
        xy=np.asarray(unproject(matrices[i],*point))*pitch
        if not np.isfinite(xy).all() or (xy< -5).any() or (xy>pitch+5).any():continue
        scores=contact_scores(xy,positions[i],metadata);total=scores.sum()
        if not np.isfinite(scores).all() or total<=0:continue
        support[i]=True;probability[i]=scores[0]/total
        strong[i]=scores.max()>=.5 and abs(scores[0]-scores[1])>=.2 and max(probability[i],1-probability[i])>=.65
    return probability,support,strong


def contacts():
    from starter.load_homography import project
    from games import GAME,GAMES
    validate();source=FOLDER/'inputs';check_cache(source);check_cache(FOLDER/'outputs');output=FOLDER/'contacts';output.mkdir(parents=True,exist_ok=False)
    positions=np.load(source/'positions.npy');teams=np.load(source/'teams.npy');matrices=np.load(source/'homography.npy');eligible=np.load(source/'eligible.npy');ball=np.load(source/'tracked_ball.npy');pitch=np.asarray(GAMES[GAME]['pitch'])
    points={mode:np.load(FOLDER/'outputs'/(mode+'_detections.npy'))[:,:2] for mode in ('temporal','frame')};tracked=np.full_like(ball,np.nan)
    for i,point in enumerate(ball):
        if np.isfinite(point).all():tracked[i]=project(matrices[i],*(point/pitch))
    points['tracked']=tracked
    for mode,point in points.items():
        p,s,q=contact_arrays(point,positions,teams,matrices,eligible);np.savez_compressed(output/(mode+'.npz'),probability=p,support=s,strong=q)
    np.save(output/'frames.npy',np.load(source/'frames.npy'))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),targets_loaded=False,readout='Unchanged contact_scores,metadata onehot,Gaussian2m,max>=.5,margin>=.2,posteriorconfidence>=.65',cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));validate()


def audit():
    import pandas as pd
    from audit_pathcrf_signal import overlay
    from dfl_signal_audit import matching
    validate();source=FOLDER/'contacts';check_cache(source);output=FOLDER/'signal_audit';output.mkdir(parents=True,exist_ok=False)
    binding={str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',source/'manifest.json']}
    write_json(FOLDER/'protocol_freeze/target_join_binding.json',binding)
    full=pd.read_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').label.to_numpy();public=np.arange(11185,18685);np.testing.assert_array_equal(np.flatnonzero(full!='X'),public);labels=full[public];frames=np.load(source/'frames.npy');relative=frames-11185;splits=json.loads((ROOT/'data/advanced_validation/splits.json').read_text());arms={}
    for mode in ('temporal','frame','tracked'):
        with np.load(source/(mode+'.npz')) as d:p=d['probability'];support=d['support'];strong=d['strong']
        prediction=np.where(p>=.5,'H','A');prediction[~support]='X';protocols={};new_cases=[]
        for protocol in ('buffered_minutes','purged_spells','buffered_100_seconds'):
            table=pd.read_csv(ROOT/'data/control_dfl_dead_v1/public_trial'/protocol/'oof_predictions.csv',usecols=['frame','current_seed42']);np.testing.assert_array_equal(table.frame,public);reference=table.current_seed42.to_numpy();old=reference[relative];target=labels[relative]
            editable=np.isin(old,['H','A']);known=support&editable&np.isin(target,['H','A']);all_live=editable&np.isin(target,['H','A'])
            recall=[float((prediction[known&(target==k)]==k).mean()) if (known&(target==k)).any() else None for k in ('H','A')];balanced=sum(recall)/2 if all(v is not None for v in recall) else None
            qualified=editable&support&strong;changes=qualified&(prediction!=old);corrected=int((changes&(prediction==target)).sum());damage=int((changes&(old==target)).sum());precision=corrected/(corrected+damage) if corrected+damage else None
            diagnostic=reference.copy()
            for split in splits:
                if split['protocol']!=protocol:continue
                a,b=min(split['validation']),max(split['validation'])+1;q=qualified&(frames>=a)&(frames<b);partial=overlay(reference,frames,prediction,q);where=(public>=a)&(public<b);diagnostic[where]=partial[where]
            np.testing.assert_array_equal(diagnostic[~np.isin(reference,['H','A'])],reference[~np.isin(reference,['H','A'])])
            before=matching(labels,reference);after=matching(labels,diagnostic);np.testing.assert_array_equal(before['truth'],after['truth']);short=np.asarray([b-a<=125 for a,b,_ in before['truth']]);new=np.flatnonzero(~before['gt_flags']&after['gt_flags']&short).tolist();new_cases.append(set(new));sp0=int((~before['pred_flags']).sum());sp1=int((~after['pred_flags']).sum());retained=bool(np.all(after['gt_flags'][before['gt_flags']]))
            checks=dict(balanced=balanced is not None and balanced>=.70,corrections=corrected>=25,precision=precision is not None and precision>=.55,net=corrected-damage>=25,prior_matches=retained,no_extra_spurious=sp1<=sp0)
            cases=[]
            for i,(a,b,label) in enumerate(before['truth']):
                group=(relative>=a)&(relative<b);cases.append(dict(gt_id=i,frames=b-a,label=label,short=bool(short[i]),sampled_centers=int(group.sum()),source_observed=int((group&support).sum()),confident=int((group&qualified).sum()),correct_observed=int((group&support&(prediction==labels[a])).sum()),old_matched=bool(before['gt_flags'][i]),diagnostic_matched=bool(after['gt_flags'][i])))
            if sum(c['sampled_centers'] for c in cases)!=int(np.isin(target,['H','A']).sum()):raise ValueError('Unsupported case denominator lost')
            spurious=[]
            for i,(a,b,label) in enumerate(before['predicted']):
                if before['pred_flags'][i]:continue
                group=(relative>=a)&(relative<b);spurious.append(dict(frames=b-a,label=label,sampled_centers=int(group.sum()),source_observed=int((group&support).sum()),confident=int((group&qualified).sum())))
            protocols[protocol]=dict(conditional_observed_live_balanced_accuracy=balanced,all_live_accuracy_missing_counted_wrong=float((prediction[all_live]==target[all_live]).mean()),observed_live_centers=int(known.sum()),all_live_centers=int(all_live.sum()),corrected=corrected,damage=damage,precision=precision,net=corrected-damage,spurious_before=sp0,spurious_after=sp1,common_short_candidates=new,checks=checks,all_truth_cases=cases,all_spurious_cases=spurious)
            pd.DataFrame(dict(frame=public,reference=reference,diagnostic=diagnostic)).to_csv(output/(mode+'_'+protocol+'.csv'),index=False)
        common=sorted(set.intersection(*new_cases));arms[mode]=dict(passed=all(all(v['checks'].values()) for v in protocols.values()) and bool(common),common_new_short_ids=common,protocols=protocols)
    result=dict(passed=arms['temporal']['passed'],eligible_arm='temporal' if arms['temporal']['passed'] else None,arms=arms,only_temporal_eligible=True,new_fits=0,new_possession_score_rows=0,diagnostic_only=True)
    write_json(output/'summary.json',result);write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}));print('Native optical opportunity gate:',result['passed'],flush=True)
    for mode in arms:
        for protocol,r in arms[mode]['protocols'].items():print(mode,protocol,'C/H',r['corrected'],r['damage'],'net',r['net'],'precision',r['precision'],'newshort',r['common_short_candidates'],'spurious',r['spurious_before'],r['spurious_after'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer','contacts','audit']);args=p.parse_args();globals()[args.phase]()
