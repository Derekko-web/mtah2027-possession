"""Source-only contracts for fresh corrected-TabPFN qualification."""
import copy
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch


def run_tests(subject):
    results=[]
    def check(name,fn):fn();results.append(dict(name=name,passed=True))
    def require(value):
        if not value:raise AssertionError("Fresh confirmation contract failed")
    check("all_workers_are_fresh_no_predecessor_reuse",lambda:require(subject.engine.reuse_completed(Path('/invented/job'),Path('/invented/out')) is False))
    for name in ('worker','context_frames','context_record','query_indices'):
        check('unchanged_original_'+name,lambda n=name:require(Path(getattr(subject.engine.legacy(),n).__code__.co_filename).resolve()==subject.ROOT/'experiments/tabpfn_native.py'))
    check('same_extended_limits_and_memory',lambda:require((subject.engine.OPERATION_SECONDS,subject.engine.WORKER_SECONDS,subject.engine.LIMIT_BYTES)==(900,3600,6*1024**3)))
    check('canonical512_padding_unchanged',lambda:require(list(subject.engine.query_indices(513))[-1]==(512,513,[512]*512)))
    spec=importlib.util.spec_from_file_location('fixed_confirmation_policy_subject',subject.ROOT/'experiments/tabpfn_fixed_policy.py')
    policy=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'tabpfn_fixed_confirmation':subject}):spec.loader.exec_module(policy)
    rows=[]
    for seed in (42,43,44):
        for protocol,before,after in [('buffered_minutes',.741176470588,.7771428571428571 if seed==42 else .78),
                                      ('purged_spells',.685714285714,.731428571429),('buffered_100_seconds',.670270270270,.70)]:
            rows.extend([dict(protocol=protocol,model=f'current_seed{seed}',spellF1=before),
                         dict(protocol=protocol,model=f'tabpfn_short_control_seed{seed}',spellF1=after)])
    selections=[policy.rank_rows(rows,s) for s in (42,43,44)]
    check('all9_matched_primary_pairs_pass',lambda:require(policy.all_nine(selections)))
    def loss():
        bad=copy.deepcopy(selections);bad[1]['pairs'][2]['spellF1']=.6
        require(not policy.all_nine(bad))
    check('one_stress_protocol_regression_rejects_promotion',loss)
    def changed_main():
        bad=copy.deepcopy(rows)
        next(r for r in bad if r['model']=='tabpfn_short_control_seed42' and r['protocol']=='buffered_minutes')['spellF1']-=.001
        require(not policy.rank_rows(bad,42)['passed'])
    check('seed42_must_preserve_the_saved_main_result',changed_main)
    def wrong_seed():
        bad=copy.deepcopy(selections);bad[2]['seed']=42;require(not policy.all_nine(bad))
    check('same_seed_cannot_replace_confirmation_seed44',wrong_seed)
    return results
