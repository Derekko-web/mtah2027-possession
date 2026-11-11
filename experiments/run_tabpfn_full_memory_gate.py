"""Replay the exact allocation failure, then prove five main folds unchanged."""
import argparse
import json
from pathlib import Path
import sys
import traceback

import tabpfn_full_memory_gate as native

ROOT,ROUND,PLAN=native.ROOT,native.ROUND,native.PLAN
PARENT=ROOT/'data/control_tabpfn_full_build_v2'
SEMANTIC=('original_probability.npy','swapped_probability.npy','owner.npy','repeat_first.npy',
          'checkpoint_fingerprint.json','preprocessing_fingerprint.json')


def prepare():
    native.qualification.verify()
    native.qualification.require_private_runtime(sys.prefix,sys.base_prefix,sys.executable,controller=True)
    if ROUND.exists():raise ValueError('Preserve existing allocation experiment')
    import numpy as np
    ROUND.mkdir()
    profiles={};old_hardware=native.read(PARENT/'hardware_probe.json')
    specifications=[('baseline4',4,None),('fixed5_first',5,None),('fixed5_rebuild',5,None)]
    specifications += [(f'main_block{b}',5,b) for b in range(1,6)]
    for block,(name,gpu,comparison) in enumerate(specifications,1):
        folder=ROUND/name;folder.mkdir()
        source=PARENT/'training' if comparison is None else native.qualification.ROUND/'screen/buffered_minutes'/f'block{comparison}'
        for item in ('context','target'):(folder/(item+'.npy')).write_bytes((source/(item+'.npy')).read_bytes())
        query=PARENT/'jobs/full_half1/block1' if comparison is None else source
        for item in ('original','swapped'):
            if comparison is None:np.save(folder/(item+'.npy'),np.load(query/(item+'.npy'),allow_pickle=False)[:512])
            else:(folder/(item+'.npy')).write_bytes((query/(item+'.npy')).read_bytes())
        hardware=dict(old_hardware)
        if gpu==5:
            hardware.update(mps_allocator_limit_bytes=5*1024**3,reserved_owned_rss_budget_bytes=1024**3,
                tabpfn_mps_memory_fraction=repr((5*1024**3)/hardware['recommended_max_memory_bytes']))
        native.write_json(folder/'hardware_probe.json',hardware)
        (folder/'acquisition.json').write_bytes((PARENT/'acquisition.json').read_bytes())
        paths={item:folder/(item+'.npy') for item in ('context','target','original','swapped')}
        job={item:str(path) for item,path in paths.items()}
        job.update(seed=42,protocol='allocation_gate',block=block,
            input_sha256={str(path):native.digest(path) for path in paths.values()})
        native.write_json(folder/'job.json',job)
        profiles[name]=dict(block=block,gpu_GiB=gpu,query_rows=512 if comparison is None else 1500,
            context_rows=2352 if comparison is None else len(np.load(folder/'context.npy',allow_pickle=False)),
            original_main_block=comparison,job_sha256=native.digest(folder/'job.json'))
    plan=native.read(native.qualification.PLAN)
    plan.update(round='round80_full_context_allocation_gate',profiles=profiles,
        purpose='Only reallocate the same6GiB total from4GPU+2host to5GPU+1host; do not change numerical model/context/query schedule',
        negative_control='baseline4 must reproduce the actual2352-context/512-query GPU OOM below6GiB total',
        fixed_profile='Two independent fresh full-context fits and complete saved main5fold reconstruction at5GiB GPU',
        mandatory='All native fitted state/checkpoint/repeat/integrity/time/6GiB resource bounds; main probabilities and labels byte-exact to saved0.7771428571428571',
        operation_seconds=900,worker_seconds=3600,combined_limit_bytes=6*1024**3,
        no_model_context_precision_batch_or_correction_change=True,baseline_expected_failure_not_a_retry=True)
    files=[p for p in ROUND.rglob('*') if p.is_file()]
    files.extend([PARENT/'production_failure.json',PARENT/'jobs/full_half1/block1/supervision.json',
                  PARENT/'jobs/full_half1/block1/worker.stderr.log',
                  PARENT/'jobs/full_half1/block1/worker/checkpoint_fingerprint.json',
                  PARENT/'jobs/full_half1/block1/worker/preprocessing_fingerprint.json',
                  native.qualification.ROUND/'final_selection.json',native.qualification.ROUND/'replay_complete.json'])
    for b in range(1,6):
        source=native.qualification.ROUND/'screen/buffered_minutes'/f'block{b}'/'worker'
        files.extend(source/item for item in (*SEMANTIC,'recipe.json'))
    plan['input_sha256']={str(p.relative_to(ROOT)):native.digest(p) for p in files}
    with PLAN.open('x') as f:f.write(json.dumps(plan,indent=2,sort_keys=True)+'\n')
    print('Prepared exact negative control, two full-context allocation fits, and all five main-fold comparisons',flush=True)


def compare_recipe(actual,expected):
    actual=json.loads(json.dumps(actual));expected=json.loads(json.dumps(expected))
    for recipe in (actual,expected):
        recipe['fixed_environment'].pop('TABPFN_MPS_MEMORY_FRACTION')
    if actual!=expected:raise ValueError('Numerical recipe changed beyond allocation fraction')


def run():
    plan=native.verify();native.require_admission()
    native.qualification.require_private_runtime(sys.prefix,sys.base_prefix,sys.executable,controller=True)
    for name in plan['profiles']:
        native.configure(name);folder=ROUND/name
        print('Allocation experiment',name,flush=True)
        try:native.supervise(folder/'job.json',folder/'worker')
        except native.NativeMethodFailure:
            if name!='baseline4':raise
            record=native.read(folder/'supervision.json');error=(folder/'worker.stderr.log').read_text()
            if (record['exit_code']==0 or record['aborted'] is not None
                    or record['peak_owned_rss_plus_full_driver_bytes']>6*1024**3
                    or 'TabPFNMPSOutOfMemoryError' not in error
                    or '2,352 train / 512 test samples, 166 features' not in error):
                raise ValueError('Negative control failed for a different reason')
            native.write_json(ROUND/'baseline_reproduced.json',dict(passed=True,expected_oom=True,
                clean_process=True,supervision_sha256=native.digest(folder/'supervision.json')))
            print('Reproduced4GiB allocation failure in a clean process below6GiB total',flush=True)
        else:
            if name=='baseline4':raise ValueError('Negative control unexpectedly passed; reassess allocation hypothesis')
            record=native.read(folder/'worker/summary.json')
            if not all(record.get(k) is True for k in ('passed','complete_raw_preprocessing_immutable',
                    'checkpoint_exact','resolved_model_configuration_immutable','repeat_byte_exact')):
                raise ValueError('Native fitted-state or repeat proof incomplete')
            if name=='fixed5_first':
                for item in ('checkpoint_fingerprint.json','preprocessing_fingerprint.json'):
                    if (folder/'worker'/item).read_bytes()!=(PARENT/'jobs/full_half1/block1/worker'/item).read_bytes():
                        raise ValueError('Changing allocation changed initial full-context fitted state')
            elif name=='fixed5_rebuild':
                for item in (*SEMANTIC,'recipe.json'):
                    if (folder/'worker'/item).read_bytes()!=(ROUND/'fixed5_first/worker'/item).read_bytes():
                        raise ValueError('Independent full-context allocation reconstruction differs')
            else:
                b=plan['profiles'][name]['original_main_block']
                old=native.qualification.ROUND/'screen/buffered_minutes'/f'block{b}'/'worker'
                for item in SEMANTIC:
                    if (folder/'worker'/item).read_bytes()!=(old/item).read_bytes():
                        raise ValueError('Higher allocation changed a saved main-fold output: '+item)
                compare_recipe(native.read(folder/'worker/recipe.json'),native.read(old/'recipe.json'))
            print('Passed',name,'with exact fitted state and prediction checks',flush=True)
    audit()


def audit():
    plan=native.verify()
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(ROOT))
    from evaluation import possession_metrics as pm
    from analyze_tabpfn_blocks import load_inputs
    from research_ownership_graph import selected_arrays
    from tabpfn_short_control_inference import decode_tabpfn_short_control
    raw,bc,motion,eligible,splits=load_inputs()
    owner=np.full((len(raw),2),np.nan)
    folds=[s for s in splits if s['protocol']=='buffered_minutes']
    for split in folds:
        name=f"main_block{split['block']}"
        value=np.load(ROUND/name/'worker/owner.npy',allow_pickle=False)
        owner[split['validation']]=value
    arrays,_=selected_arrays('buffered_minutes',42,len(raw))
    blocks=[(min(s['validation']),max(s['validation'])+1) for s in folds]
    labels,events=decode_tabpfn_short_control(arrays,owner,eligible,bc,motion,raw,blocks)
    old=pd.read_csv(native.qualification.ROUND/'screen/buffered_minutes/oof_predictions.csv')
    public=old.frame.to_numpy()
    np.testing.assert_array_equal(labels[public],old.tabpfn_short_control_seed42)
    metrics=pm.evaluate(pm.label_codes(old.ground_truth),pm.label_codes(labels[public]),bootstrap_n=0)
    if abs(metrics['spell_f1_mean']-.7771428571428571)>1e-12:raise ValueError('Main0.7771 score changed')
    workers={}
    for name in plan['profiles']:
        if name=='baseline4':continue
        folder=ROUND/name;supervision=native.read(folder/'supervision.json')
        if not supervision['passed'] or supervision['exit_code']!=0 or supervision['aborted'] is not None:
            raise ValueError('Allocation treatment cohort incomplete')
        native.validate_operations(folder/'worker')
        workers[name]=dict(supervision_sha256=native.digest(folder/'supervision.json'),
            summary_sha256=native.digest(folder/'worker/summary.json'),
            peak_combined_bytes=supervision['peak_owned_rss_plus_full_driver_bytes'])
    native.write_json(ROUND/'gate_passed.json',dict(passed=True,clean4GiB_control_oom_reproduced=True,
        memory_policy=dict(gpu_bytes=5*1024**3,reserved_host_bytes=1024**3,combined_limit_bytes=6*1024**3),
        full_context_rows=2352,query_batch_rows=512,independent_full_context_fits=2,
        full_context_fitted_state_exact_to_pre_oom=True,independent_full_context_probabilities_exact=True,
        all_five_main_fold_native_probabilities_exact=True,main_corrected_labels_exact=True,
        main_spellF1=metrics['spell_f1_mean'],model_context_views_precision_batch_and_correction_unchanged=True,
        workers=workers,source_sha256=native.source_identity(),plan_sha256=native.digest(PLAN)))
    print('Allocation gate passed; full-context exact reconstruction and unchanged main0.777142857',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    try:
        if args.prepare:prepare()
        elif args.run:run()
        else:parser.error('Choose preparation or experiment')
    except BaseException as error:
        if ROUND.exists() and not (ROUND/'gate_failure.json').exists():
            native.write_json(ROUND/'gate_failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
