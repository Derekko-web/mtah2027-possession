"""Full-match scheduling around the unchanged, qualified native TabPFN worker."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import tabpfn_fixed_confirmation_v2 as qualification
import tabpfn_full_memory_gate as memory_gate

ROOT=qualification.ROOT
ROUND=ROOT/'data/control_tabpfn_full_build_v3'
PLAN=ROOT/'experiments/TABPFN_FULL_MATCH_PLAN_V3.json'
SOURCES=('experiments/tabpfn_full_match_native_v3.py','experiments/build_tabpfn_full_candidate_v3.py',
         'experiments/test_tabpfn_full_match_v3.py','experiments/tabpfn_full_features.py',
         'experiments/tabpfn_host_interruption.py')
spec=importlib.util.spec_from_file_location('full_match_qualified_execution',ROOT/'experiments/tabpfn_extended_evaluation.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
engine.verify()
engine.ROUND,engine.PLAN,engine.__file__=ROUND,PLAN,str(Path(__file__).resolve())
engine._LEGACY=None
engine.reuse_completed=lambda job_path,output:False


def read(path):return json.loads(Path(path).read_text())


def partitions(blocks,maximum=4096):
    if maximum<=0 or maximum%512:raise ValueError('Partitions must align to canonical512 batches')
    previous=0
    for half,(start,end) in enumerate(blocks,1):
        if not 0<=previous<=start<end:raise ValueError('Invalid half boundaries')
        previous=end
        for block,left in enumerate(range(start,end,maximum),1):
            yield f'full_half{half}',block,left,min(left+maximum,end)


def source_identity():return {n:engine.digest(ROOT/n) for n in SOURCES}


def verify_memory_gate():
    memory_gate.verify()
    folder=memory_gate.ROUND;proof=read(folder/'gate_passed.json')
    if (proof.get('passed') is not True or proof.get('main_corrected_labels_exact') is not True
            or proof.get('all_five_main_fold_native_probabilities_exact') is not True
            or proof.get('independent_full_context_probabilities_exact') is not True
            or abs(proof.get('main_spellF1',0)-.7771428571428571)>1e-12
            or proof.get('memory_policy')!=dict(gpu_bytes=5*1024**3,reserved_host_bytes=1024**3,combined_limit_bytes=6*1024**3)):
        raise ValueError('Allocation and unchanged-main-score gate incomplete')
    if proof['source_sha256']!=memory_gate.source_identity() or proof['plan_sha256']!=engine.digest(memory_gate.PLAN):
        raise ValueError('Allocation gate source or plan differs')
    expected={'fixed5_first','fixed5_rebuild',*[f'main_block{b}' for b in range(1,6)]}
    if set(proof['workers'])!=expected:raise ValueError('Incomplete allocation experiment cohort')
    baseline=read(folder/'baseline4/supervision.json')
    if (baseline['passed'] or baseline['exit_code']==0 or baseline['aborted'] is not None
            or baseline['peak_owned_rss_plus_full_driver_bytes']>6*1024**3
            or 'TabPFNMPSOutOfMemoryError' not in (folder/'baseline4/worker.stderr.log').read_text()):
        raise ValueError('Invalid exact allocation-failure negative control')
    for name,witness in proof['workers'].items():
        profile=folder/name;summary=read(profile/'worker/summary.json');supervision=read(profile/'supervision.json')
        if (not summary['passed'] or not supervision['passed'] or supervision['exit_code']!=0
                or engine.digest(profile/'supervision.json')!=witness['supervision_sha256']
                or engine.digest(profile/'worker/summary.json')!=witness['summary_sha256']
                or not engine.within(supervision['elapsed'],3600)
                or supervision['peak_owned_rss_plus_full_driver_bytes']>6*1024**3):
            raise ValueError('Invalid allocation fit/resource proof')
        for item,sha in summary['artifact_sha256'].items():
            if engine.digest(profile/'worker'/item)!=sha:raise ValueError('Allocation worker evidence changed')
    return proof


def verify():
    qualification.verify()
    verify_memory_gate()
    folder=ROUND/'protocol_freeze'
    if PLAN.read_bytes()!=(folder/'plan.json').read_bytes():raise ValueError('Full plan changed')
    for group in ('implementation.json','input_sha256.json','protected_artifacts.json','head_runtime.json'):
        for name,sha in read(folder/group).items():
            if engine.digest(ROOT/name)!=sha:raise ValueError('Full production binding changed: '+name)
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((folder/'EXPERIMENT_LEDGER.csv').read_bytes()):
        raise ValueError('Ledger history changed')
    for name in ('global_failure.json','method_failure.json','production_failure.json'):
        if (ROUND/name).exists():raise ValueError('Full production attempt is closed')
    return read(PLAN)


def validate_job(job):
    expected={'context','target','original','swapped','seed','protocol','block','input_sha256'}
    if set(job)!=expected or job['seed']!=42 or type(job['block']) is not int:
        raise ValueError('Only the prespecified full-context seed42 is allowed')
    key=f"{job['protocol']}/block{job['block']}"
    plan=read(PLAN)
    if key not in plan['jobs']:raise ValueError('Unregistered full query partition')
    folder=ROUND/'jobs'/key
    paths={name:ROUND/'training'/(name+'.npy') for name in ('context','target')}
    paths.update({name:folder/(name+'.npy') for name in ('original','swapped')})
    if {str(p) for p in paths.values()}!=set(job['input_sha256']):raise ValueError('Unexpected full payload set')
    for name,path in paths.items():
        if Path(job[name]).resolve()!=path.resolve() or engine.digest(path)!=job['input_sha256'][str(path)]:
            raise ValueError('Full payload identity differs')
    if engine.digest(folder/'job.json')!=plan['jobs'][key]['job_sha256']:
        raise ValueError('Registered full query job changed')


engine.verify=verify
engine.legacy().validate_job=validate_job


def freeze():
    qualification.verify()
    verify_memory_gate()
    proof=qualification.verify_replay_evidence()
    if proof['independent_process_fits']!=39:raise ValueError('Incomplete independent reconstruction proof')
    for name in SOURCES:compile((ROOT/name).read_bytes(),name,'exec')
    tests_spec=importlib.util.spec_from_file_location('full_match_source_tests',ROOT/SOURCES[2])
    tests=importlib.util.module_from_spec(tests_spec);tests_spec.loader.exec_module(tests)
    outcomes=tests.run_tests(sys.modules[__name__])
    if any(n in sys.modules for n in ('numpy','torch','pandas','tabpfn')):raise ValueError('Source freeze imported scientific modules')
    review=read(ROUND/'production_root_review.json')
    if not review['approved_for_freeze'] or review['source_sha256']!=source_identity() or review['plan_sha256']!=engine.digest(PLAN):
        raise ValueError('Exact full production source/plan review required')
    folder=ROUND/'protocol_freeze';folder.mkdir(exist_ok=False)
    (folder/'plan.json').write_bytes(PLAN.read_bytes())
    engine.write_json(folder/'implementation.json',{**source_identity(),
        'experiments/TABPFN_FULL_MATCH_PLAN_V3.json':engine.digest(PLAN),
        str((ROUND/'production_root_review.json').relative_to(ROOT)):engine.digest(ROUND/'production_root_review.json')})
    engine.write_json(folder/'input_sha256.json',read(PLAN)['input_sha256'])
    for name in ('protected_artifacts.json','head_runtime.json'):
        (folder/name).write_bytes((qualification.ROUND/'protocol_freeze'/name).read_bytes())
    (folder/'EXPERIMENT_LEDGER.csv').write_bytes((ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes())
    engine.write_json(folder/'freeze.json',dict(source_sha256=source_identity(),plan_sha256=engine.digest(PLAN),
        source_tests=outcomes,unchanged_native_worker=True,operation_seconds=900,worker_seconds=3600,
        seed=42,context_fits_fresh=True,full_match_queries=True,model_calls=False))
    verify()


def __getattr__(name):return getattr(engine,name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--job',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.freeze:freeze()
    elif args.verify:verify()
    elif args.worker:
        engine.require_admission();engine.legacy().worker(args.job,args.output)
    else:parser.error('Choose freeze, verify or worker')
