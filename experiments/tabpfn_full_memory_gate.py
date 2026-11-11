"""One-variable allocation experiment with the unchanged native TabPFN worker."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import tabpfn_fixed_confirmation_v2 as qualification

ROOT=qualification.ROOT
ROUND=ROOT/'data/control_tabpfn_full_memory_gate_v1'
PLAN=ROOT/'experiments/TABPFN_FULL_MEMORY_GATE_PLAN.json'
SOURCES=('experiments/tabpfn_full_memory_gate.py','experiments/run_tabpfn_full_memory_gate.py',
         'experiments/test_tabpfn_full_memory_gate.py')
spec=importlib.util.spec_from_file_location('full_allocation_execution',ROOT/'experiments/tabpfn_extended_evaluation.py')
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
engine.verify()
engine.PLAN,engine.__file__=PLAN,str(Path(__file__).resolve())
engine.reuse_completed=lambda job_path,output:False
PROFILE=None


def read(path):return json.loads(Path(path).read_text())


def source_identity():return {name:engine.digest(ROOT/name) for name in SOURCES}


def verify():
    qualification.verify()
    folder=ROUND/'protocol_freeze'
    if PLAN.read_bytes()!=(folder/'plan.json').read_bytes():raise ValueError('Allocation plan changed')
    for group in ('implementation.json','input_sha256.json'):
        for name,sha in read(folder/group).items():
            if engine.digest(ROOT/name)!=sha:raise ValueError('Allocation gate binding changed: '+name)
    if (ROUND/'gate_failure.json').exists():raise ValueError('Allocation gate is closed; preserve evidence')
    return read(PLAN)


def require_admission():
    value=read(ROUND/'resource_admission.json')
    if not value['approved'] or not value['serial_native_workers'] or value['freeze_sha256']!=engine.digest(ROUND/'protocol_freeze/freeze.json'):
        raise ValueError('Matching allocation-gate admission required')


def configure(profile):
    global PROFILE
    if profile not in read(PLAN)['profiles']:raise ValueError('Unknown allocation profile')
    PROFILE=profile;engine.ROUND=ROUND/profile;engine._LEGACY=None
    engine.legacy().validate_job=validate_job


def validate_job(job):
    fields={'context','target','original','swapped','seed','protocol','block','input_sha256'}
    if set(job)!=fields or job['seed']!=42 or job['protocol']!='allocation_gate' or type(job['block']) is not int:
        raise ValueError('Unchanged seed42 and no query labels required')
    plan=read(PLAN);profile=plan['profiles'][PROFILE];folder=ROUND/PROFILE
    if job['block']!=profile['block']:raise ValueError('Wrong allocation experiment role')
    expected={str(folder/(name+'.npy')) for name in ('context','target','original','swapped')}
    if set(job['input_sha256'])!=expected:raise ValueError('Unexpected allocation payload paths')
    for name in ('context','target','original','swapped'):
        path=folder/(name+'.npy')
        if Path(job[name]).resolve()!=path.resolve() or engine.digest(path)!=job['input_sha256'][str(path)]:
            raise ValueError('Allocation payload bytes changed')
    if engine.digest(folder/'job.json')!=profile['job_sha256']:raise ValueError('Allocation role changed')


engine.verify=verify
engine.require_admission=require_admission


def freeze():
    qualification.verify()
    proof=qualification.verify_replay_evidence()
    if proof['independent_process_fits']!=39:raise ValueError('Qualified reconstruction proof incomplete')
    for name in SOURCES:compile((ROOT/name).read_bytes(),name,'exec')
    tests_spec=importlib.util.spec_from_file_location('allocation_source_tests',ROOT/SOURCES[2])
    tests=importlib.util.module_from_spec(tests_spec);tests_spec.loader.exec_module(tests)
    outcomes=tests.run_tests(sys.modules[__name__])
    if any(n in sys.modules for n in ('numpy','torch','pandas','tabpfn')):raise ValueError('Scientific import before allocation source freeze')
    review=read(ROUND/'root_review.json')
    if not review['approved'] or review['source_sha256']!=source_identity() or review['plan_sha256']!=engine.digest(PLAN):
        raise ValueError('Exact allocation experiment source/plan review required')
    folder=ROUND/'protocol_freeze';folder.mkdir(exist_ok=False)
    (folder/'plan.json').write_bytes(PLAN.read_bytes())
    bindings=source_identity()
    for path in (PLAN,ROUND/'root_review.json'):
        bindings[str(path.relative_to(ROOT))]=engine.digest(path)
    engine.write_json(folder/'implementation.json',bindings)
    engine.write_json(folder/'input_sha256.json',read(PLAN)['input_sha256'])
    engine.write_json(folder/'freeze.json',dict(source_sha256=source_identity(),plan_sha256=engine.digest(PLAN),
        tests=outcomes,worker_source_unchanged=True,model_calls=False,combined_limit_bytes=6*1024**3,
        operation_seconds=900,worker_seconds=3600,only_allocation_split_changed=True))
    verify()


def __getattr__(name):return getattr(engine,name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--job',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze()
    elif args.worker:
        profile=args.job.resolve().relative_to(ROUND.resolve()).parts[0];configure(profile)
        require_admission();engine.legacy().worker(args.job,args.output)
    else:parser.error('Choose freeze or worker')
