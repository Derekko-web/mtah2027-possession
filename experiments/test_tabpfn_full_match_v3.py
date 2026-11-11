"""Source-only production schedule, isolation and no-query-label contracts."""
from pathlib import Path
import json
from tabpfn_host_interruption import is_host_sleep_interruption


def run_tests(subject):
    outcomes=[]
    def require(condition):
        if not condition:raise AssertionError('Full-match source contract failed')
    def check(name,callback):callback();outcomes.append(dict(name=name,passed=True))
    def must_raise(callback):
        try:callback()
        except ValueError:return
        raise AssertionError('Invalid full-match input was accepted')
    blocks=[(11185,90652),(110808,190198)]
    schedule=list(subject.partitions(blocks))
    def full_coverage():
        require(len(schedule)==40)
        require(sum(end-start for _,_,start,end in schedule)==158857)
        actual=[frame for _,_,start,end in schedule for frame in range(start,end)]
        expected=[frame for start,end in blocks for frame in range(start,end)]
        require(actual==expected)
    check('exact40_partitions_cover_both_official_halves_once',full_coverage)
    def canonical():
        for half,(start,end) in enumerate(blocks,1):
            whole=[(start+a,start+b,[start+i for i in ids]) for a,b,ids in subject.engine.query_indices(end-start)]
            partitioned=[]
            for protocol,_,left,right in schedule:
                if protocol==f'full_half{half}':
                    partitioned.extend((left+a,left+b,[left+i for i in ids]) for a,b,ids in subject.engine.query_indices(right-left))
            require(whole==partitioned)
    check('partitioned512_queries_identical_to_unpartitioned_half_schedule',canonical)
    check('unaligned_partition_size_rejected',lambda:must_raise(lambda:list(subject.partitions(blocks,1025))))
    check('overlapping_halves_rejected',lambda:must_raise(lambda:list(subject.partitions([(0,100),(99,200)]))))
    check('same_qualified_native_worker_executable',lambda:require(Path(subject.engine.legacy().worker.__code__.co_filename).resolve()==subject.ROOT/'experiments/tabpfn_native.py'))
    check('extended_resource_limits_unchanged',lambda:require((subject.engine.OPERATION_SECONDS,subject.engine.WORKER_SECONDS,subject.engine.LIMIT_BYTES)==(900,3600,6*1024**3)))
    job=dict(context='',target='',original='',swapped='',seed=43,protocol='full_half1',block=1,input_sha256={})
    check('confirmation_best_seed_cannot_replace_prespecified42',lambda:must_raise(lambda:subject.validate_job(job)))
    labelled=dict(job,seed=42,query_labels='forbidden')
    check('query_labels_forbidden_in_production_worker_payload',lambda:must_raise(lambda:subject.validate_job(labelled)))
    interrupted=subject.ROOT/'data/control_tabpfn_fixed_confirmation_v1/screen/buffered_minutes/block2'
    sleep=json.loads((interrupted/'supervision.json').read_text())
    check('captured_host_suspension_recognized',lambda:require(is_host_sleep_interruption(sleep,(interrupted/'worker.stderr.log').read_text())))
    failed=subject.ROOT/'data/control_tabpfn_full_build_v2/jobs/full_half1/block1'
    oom=json.loads((failed/'supervision.json').read_text())
    check('actual_model_memory_failure_never_retried',lambda:require(not is_host_sleep_interruption(oom,(failed/'worker.stderr.log').read_text())))
    no_sleep=dict(sleep,elapsed=dict(sleep['elapsed'],utc_ns=sleep['elapsed']['monotonic_ns']))
    check('real_active_clock_overrun_never_retried',lambda:require(not is_host_sleep_interruption(no_sleep,'')))
    return outcomes
