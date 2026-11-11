"""Frozen external acceptance after both fixed-budget masked-player fits finish."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from prepare_masked_player_data import validate,FOLDER,PLAN
from research_temporal_dead import write_json


def metrics(target,p):
    target=np.asarray(target,int);p=np.asarray(p,float)
    if set(target)!={0,1} or p.shape!=target.shape or not np.isfinite(p).all():raise ValueError('Invalid external assessment support')
    prediction=p>=.5;recalls=[float((prediction[target==k]==k).mean()) for k in (0,1)]
    q=np.clip(p,1e-6,1-1e-6)
    return dict(frames=len(target),balanced_accuracy=sum(recalls)/2,away_recall=recalls[0],home_recall=recalls[1],
        brier=float(((p-target)**2).mean()),log_loss=float(-(target*np.log(q)+(1-target)*np.log(1-q)).mean()))


def main(output,replay=False):
    plan=validate();output.mkdir(parents=True,exist_ok=False);source=FOLDER/'corpus';validation=dict(np.load(source/'validation.npz'))
    models=FOLDER/('external_models_replay' if replay else 'external_models');scores={};eligible=[];input_hashes={}
    for mode in ('temporal','frame'):
        model_folder=models/mode;manifest=json.loads((model_folder/'manifest.json').read_text())
        if manifest['steps']!=4000 or manifest['parameters']!=24321 or not manifest['checkpoint_reload_exact']:raise ValueError('Wrong fitting budget/architecture')
        if manifest['schedule_sha256']!=digest(source/'schedule.npz'):raise ValueError('Control/candidate fitting schedules differ')
        for name,expected in manifest['cache_sha256'].items():
            if digest(model_folder/name)!=expected:raise ValueError('Model output changed')
        scores[mode]={}
        for view in ('clean','masked'):
            p=np.load(model_folder/(view+'_probability.npy'));input_hashes[str((model_folder/(view+'_probability.npy')).relative_to(ROOT))]=digest(model_folder/(view+'_probability.npy'))
            for game in plan['splits']['development']+plan['splits']['held_out']:
                group=validation['match']==plan['matches'].index(game)
                scores[mode].setdefault(game,{})[view]=metrics(validation['target'][group],p[group])
        flags=[]
        for game,row in scores[mode].items():
            m=row['masked'];row['checks']=dict(balanced=m['balanced_accuracy']>=.75,home=m['home_recall']>=.65,away=m['away_recall']>=.65)
            row['passed']=all(row['checks'].values());flags.append(row['passed'])
        if all(flags):eligible.append(mode)
    def rank(mode):
        values=[r['masked']['balanced_accuracy'] for r in scores[mode].values()]
        return (-min(values),-float(np.mean(values)),0 if mode=='temporal' else 1)
    eligible.sort(key=rank)
    result=dict(passed=bool(eligible),selected=eligible[0] if eligible else None,eligible=eligible,scores=scores,
        external_models_in_trial=2,new_fits_in_assessment=0,native_targets_loaded=False,manual_labels_used=False,same_fitting_schedule=True)
    write_json(output/'selection.json',result)
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),source_sha256=digest(Path(__file__)),input_sha256=input_hashes,selection_sha256=digest(output/'selection.json')))
    if replay and (output/'selection.json').read_bytes()!=(FOLDER/'external_assessment/selection.json').read_bytes():raise ValueError('External selection replay differs')
    for mode,values in scores.items():
        for game,row in values.items():print(mode,game,json.dumps(row['masked']),'passes',row['passed'],flush=True)
    print('External eligible',eligible,'selected',result['selected'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args();main(a.output or FOLDER/('external_assessment_replay' if a.replay else 'external_assessment'),a.replay)
