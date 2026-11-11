"""Source-only causal isolation and resource policy contracts."""
from pathlib import Path


def run_tests(subject):
    results=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        results.append(dict(name=name,passed=True))
    plan=subject.read(subject.PLAN)
    check('one_negative_two_full_fits_five_complete_main_folds',set(plan['profiles'])=={'baseline4','fixed5_first','fixed5_rebuild',*[f'main_block{b}' for b in range(1,6)]})
    for item in ('context.npy','target.npy','original.npy','swapped.npy'):
        values=[(subject.ROUND/name/item).read_bytes() for name in ('baseline4','fixed5_first','fixed5_rebuild')]
        check('identical_full_probe_'+item,values[0]==values[1]==values[2])
    baseline=subject.read(subject.ROUND/'baseline4/hardware_probe.json')
    for name in plan['profiles']:
        hardware=subject.read(subject.ROUND/name/'hardware_probe.json')
        gpu=hardware['mps_allocator_limit_bytes'];host=hardware['reserved_owned_rss_budget_bytes']
        check(name+'_same6GiB_total',gpu+host==6*1024**3)
        check(name+'_allocator_matches_fraction',abs(float(hardware['tabpfn_mps_memory_fraction'])*hardware['recommended_max_memory_bytes']-gpu)<1)
        old=dict(baseline);new=dict(hardware)
        for key in ('mps_allocator_limit_bytes','reserved_owned_rss_budget_bytes','tabpfn_mps_memory_fraction'):
            old.pop(key);new.pop(key)
        check(name+'_no_other_hardware_environment_change',old==new)
    subject.configure('fixed5_first')
    check('unchanged_native_worker',Path(subject.engine.legacy().worker.__code__.co_filename).resolve()==subject.ROOT/'experiments/tabpfn_native.py')
    check('same6GiB_900s_3600s_supervision',(subject.engine.LIMIT_BYTES,subject.engine.OPERATION_SECONDS,subject.engine.WORKER_SECONDS)==(6*1024**3,900,3600))
    check('no_predecessor_model_reuse',subject.engine.reuse_completed(None,None) is False)
    return results
