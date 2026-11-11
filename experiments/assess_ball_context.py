"""Fixed external absolute and paired ball-information gates, all three matches."""
import json
import numpy as np
from player_representation import ROOT,digest,write_json
from prepare_ball_context_data import FOLDER,PLAN,validate
from evaluate_masked_player import metrics


def main():
    plan=validate();output=FOLDER/'external_assessment';output.mkdir(parents=True,exist_ok=False)
    original=ROOT/'data/control_masked_player_v1/corpus';validation=dict(np.load(original/'validation.npz'));scores={};counts=[]
    for mode in ('ball','masked'):
        folder=FOLDER/'external_models'/mode;manifest=json.loads((folder/'manifest.json').read_text());counts.append(manifest['parameters'])
        if manifest['steps']!=4000 or manifest['schedule_sha256']!=digest(original/'schedule.npz') or not manifest['strict_reload_exact']:raise ValueError('Unequal control fitting interface')
        for name,sha in manifest['cache_sha256'].items():
            if digest(folder/name)!=sha:raise ValueError('External ball model output changed')
        scores[mode]={}
        for view in ('clean','masked'):
            p=np.load(folder/(view+'_probability.npy'))
            for game in plan['validation_matches']:
                group=validation['match']==plan['matches'].index(game);scores[mode].setdefault(game,{})[view]=metrics(validation['target'][group],p[group])
    if len(set(counts))!=1:raise ValueError('Control parameter counts differ')
    flags={}
    for game in plan['validation_matches']:
        a=scores['ball'][game]['masked'];b=scores['masked'][game]['masked'];flags[game]=dict(balanced=a['balanced_accuracy']>=.80,home=a['home_recall']>=.70,away=a['away_recall']>=.70,paired_ball_gain=a['balanced_accuracy']>=b['balanced_accuracy']+.02)
    passed=all(all(f.values()) for f in flags.values());write_json(output/'selection.json',dict(passed=passed,selected='ball' if passed else None,checks=flags,scores=scores,native_targets_loaded=False))
    write_json(output/'manifest.json',dict(plan_sha256=digest(PLAN),cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    for game in plan['validation_matches']:print(game,'ball',scores['ball'][game]['masked']['balanced_accuracy'],'masked',scores['masked'][game]['masked']['balanced_accuracy'],'checks',flags[game],flush=True)


if __name__=='__main__':main()
