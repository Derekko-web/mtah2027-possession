"""Replay the frozen DFL public trial, score officially and enforce promotion gates."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dfl_transfer_data import digest
from dfl_transfer_features import geometry
from dfl_transfer_teacher import probability
from detector_ledger import append_score_rows
from finalize_broadcast_closure import audit_manifest,official_scores
from games import GAME,GAMES,half_video_blocks
from motion_dead_trim import decode_motion_dead_trim
from provided_speed import decode_provided_speed
from research_ownership_graph import selected_arrays


def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    equal=np.array_equal(a,b,equal_nan=True) if a.dtype.kind not in 'OUS' else np.array_equal(a,b)
    if not equal:
        raise AssertionError('Replay differs')


def main(record_ledger=False):
    root=ROOT/'data/control_dfl_transfer_v1';folder=root/'public_trial'
    plan_path=ROOT/'experiments/DFL_TRANSFER_TEACHER_PLAN.json'
    plan=json.loads(plan_path.read_text());freeze=root/'protocol_freeze_B'
    assert plan_path.read_bytes()==(freeze/'plan.json').read_bytes()
    manifest=json.loads((folder/'manifest.json').read_text())
    for key,base in (('source_sha256',ROOT),('input_sha256',ROOT),('cache_sha256',folder)):
        for name,expected in manifest[key].items():assert digest(base/name)==expected,name
    for name,expected in json.loads((root/'public_dispatch.json').read_text())['input_sha256'].items():
        assert digest(ROOT/name)==expected,name
    original=(freeze/'ledger_before.csv').read_bytes()
    assert (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith(original)
    for name,expected in json.loads((freeze/'protected_artifacts.json').read_text()).items():
        assert digest(ROOT/name)==expected,name
    for name,target in json.loads((freeze/'symlinks.json').read_text()).items():
        path=ROOT/name;assert path.is_symlink() and str(path.readlink())==target,name
    assert json.loads((root/'external_integrity.json').read_text())['passed']
    replay=root/'public_replay';replay.mkdir()
    parent=ROOT/'data/control_broadcast_v1/inputs';audit_manifest(parent)
    source=ROOT/'data/advanced_validation'
    raw=pd.read_pickle(source/'raw.pkl');n=len(raw)
    with np.load(parent/'tactical_observations.npz') as data:
        positions,ball,teams,valid=[data[k] for k in ('positions','ball','teams','valid')]
    positions=positions[:n].copy();ball=ball[:n].copy();valid=valid[:n]
    positions[~valid]=np.nan;ball[~valid]=np.nan;pitch=np.array(GAMES[GAME]['pitch'])
    ball[~(np.isfinite(ball).all(axis=1)&(ball>=-5).all(axis=1)&(ball<=pitch+5).all(axis=1))]=np.nan
    blocks=[(start,end+1) for _,start,end in half_video_blocks(GAME)]
    inputs=geometry(positions,ball,teams,blocks)
    short=geometry(positions[:18935],ball[:18935],teams,blocks)
    pd.testing.assert_frame_equal(inputs.iloc[11185:18685],short.iloc[11185:18685],check_exact=True)
    teacher=XGBClassifier();teacher.load_model(root/'external_trial/teacher_seed42.ubj')
    p=probability(teacher,inputs)
    views=dict(matched_control=inputs,external_teacher=inputs.assign(h__teacher=p,a__teacher=1-p))
    for name,view in views.items():
        path=replay/(name+'_inputs.pkl');view.to_pickle(path)
        assert digest(path)==digest(folder/path.name)
    public=np.arange(11185,18685)
    bc=pd.read_pickle(parent/'combined_inputs.pkl').iloc[:n]
    upstream=ROOT/'data/control_pair_motion_v1'
    audit_manifest(upstream/'inputs');audit_manifest(upstream/'dead_trim_confirmation')
    info=pd.read_pickle(upstream/'inputs/motion_info.pkl').iloc[:n]
    splits=json.loads((folder/'splits.json').read_text())
    scores=pd.read_csv(folder/'scores.csv');segments=pd.read_csv(folder/'segment_diagnostics.csv')
    assert len(scores)==9 and not scores.duplicated(['protocol','model']).any()
    assert len(list(folder.rglob('*.ubj')))==26
    reloaded=0;references=0;diagnostics=[];identities={}
    for protocol in plan['public_screen']['protocols']:
        folds=[s for s in splits if s['protocol']==protocol]
        blocks=[(min(f['validation']),max(f['validation'])+1) for f in folds]
        arrays,hashes=selected_arrays(protocol,42,n);identities.update(hashes)
        table=pd.read_csv(folder/protocol/'oof_predictions.csv');same(table.frame,public)
        reference=decode_motion_dead_trim(*arrays,bc,info,raw,blocks)[public]
        same(reference,table.current_seed42);references+=1
        labels=np.full(199319,'X',dtype='<U1');labels[public]=table.ground_truth
        for head,features in views.items():
            pp=np.full((n,3),np.nan)
            for fold in folds:
                train,val=np.array(fold['train']),np.array(fold['validation'])
                assert not np.any((train>=val.min()-250)&(train<=val.max()+250))
                live=train[np.isin(labels[train],['H','A'])];target=(labels[live]=='H').astype(int)
                weight=np.zeros(len(target))
                for label in (0,1):weight[target==label]=len(target)/(2*(target==label).sum())
                prefix=folder/protocol/f'{head}_seed42_block{fold["block"]}'
                with np.load(str(prefix)+'_fitting.npz') as saved:
                    for key,value in dict(train=train,live=live,validation=val,target=target,weight=weight).items():same(saved[key],value)
                model=XGBClassifier();model.load_model(str(prefix)+'.ubj')
                assert model.get_booster().num_boosted_rounds()==120
                assert model.get_booster().feature_names==list(features)
                home=probability(model,features.iloc[val]);pp[val]=np.c_[home,1-home,np.zeros(len(val))]
                reloaded+=1
            same(pp[public],np.load(folder/protocol/f'{head}_seed42_probabilities.npy'))
            prediction=decode_provided_speed(arrays,pp,bc,info,raw,blocks,'live_replace')[public]
            same(prediction,table[head+'_seed42']);same(prediction=='D',reference=='D')
            path=replay/f'{protocol}_{head}_probabilities.npy';np.save(path,pp[public])
            assert digest(path)==digest(folder/protocol/f'{head}_seed42_probabilities.npy')
            for case in segments[(segments.protocol==protocol)&(segments.model=='current_seed42')&(segments.label!='D')&(segments.frames<=125)].itertuples():
                use=(public>=case.start)&(public<=case.end)
                expected=0 if case.label=='H' else 1
                diagnostics.append(dict(protocol=protocol,head=head,frames=case.frames,
                    raw_correct_fraction=float((pp[public][use].argmax(axis=1)==expected).mean())))
    official=official_scores(folder,scores)
    index=scores.set_index(['protocol','model']);counts=official.set_index(['protocol','model']);gates=[]
    for protocol in plan['public_screen']['protocols']:
        old=index.loc[(protocol,'current_seed42')];new=index.loc[(protocol,'external_teacher_seed42')]
        control=index.loc[(protocol,'matched_control_seed42')]
        a=counts.loc[(protocol,'current_seed42')];b=counts.loc[(protocol,'external_teacher_seed42')]
        short=segments[(segments.protocol==protocol)&(segments.label!='D')&(segments.frames<=125)]
        before=short[short.model=='current_seed42'].set_index(['start','end','label'])
        after=short[short.model=='external_teacher_seed42'].set_index(['start','end','label'])
        pd.testing.assert_index_equal(before.index,after.index)
        flags=dict(primary=bool(new.spellF1>=old.spellF1-1e-12),matched_control=bool(new.spellF1>=control.spellF1-1e-12),
            macro=bool(new.macroF1>=old.macroF1-.001-1e-12),short_iou=bool(new.short_live_mean_best_iou>=old.short_live_mean_best_iou-1e-12),
            previous_short=bool((after.recall[before.recall>=.5]>=.5).all()),falseD=bool(new.false_dead_frames<=old.false_dead_frames),
            Drecall=bool(new.dead_recall>=old.dead_recall-.02-1e-12),stoppages=bool(new.missed_stoppages<=old.missed_stoppages),
            spurious=bool(b.spurious_iou50<=a.spurious_iou50),missed=bool(b.missed_iou50<=a.missed_iou50))
        if protocol=='buffered_minutes':flags.update(substantial=bool(new.spellF1>=old.spellF1+.04-1e-12),extra_short=bool(new.short_live_matched_iou50>=old.short_live_matched_iou50+1))
        gates.append(dict(protocol=protocol,checks=flags,passed=all(flags.values()),primary_delta=float(new.spellF1-old.spellF1)))
    promotable=all(g['passed'] for g in gates)
    (folder/'promotion_gate.json').write_text(json.dumps(dict(passed=promotable,gates=gates),indent=2)+'\n')
    pd.DataFrame(diagnostics).to_csv(folder/'raw_short_diagnostics.csv',index=False)
    for name,expected in identities.items():assert digest(ROOT/name)==expected
    result=dict(round=27,external_gate_passed=True,public_promotion_gate_passed=promotable,gates=gates,
        external_models=1,public_models_reloaded=reloaded,selected_references_exact=references,
        native_feature_files_byte_exact=2,native_probability_files_byte_exact=6,actual_prefix_parity=True,
        fitting_rows_targets_weights_exact=True,D_exact_all6_new_sequences=True,official_shared_score_rows_exact=9,
        protected_hashes=17,source_input_cache_hashes_exact=True,symlinks_exact=True,
        supplemental_input_sha256=identities,verification_source_sha256=digest(__file__))
    if record_ledger:
        result['ledger']=append_score_rows(ROOT/'experiments/EXPERIMENT_LEDGER.csv',folder/'scores.csv',
            'dfl_ownership_transfer','data/control_dfl_transfer_v1/public_trial/scores.csv')
    with (root/'public_integrity.json').open('x') as output:output.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='supplemental_input_sha256'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--record-ledger',action='store_true')
    main(parser.parse_args().record_ledger)
