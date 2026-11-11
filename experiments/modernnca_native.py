"""R68 fixed final200-epoch native ModernNCA; preparation imports no science.
All native work requires root freeze plus the complete successful frozen R64 gate.
"""
import argparse
import ast
from bisect import bisect_left
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/'data/control_modernnca_native_v1'
PLAN=ROOT/'experiments/MODERNNCA_NATIVE_PLAN.json'
GATE_SOURCE=ROOT/'experiments/modernnca_isolated_runtime_gate.py'
GATE_ROUND=ROOT/'data/control_modernnca_isolated_runtime_v1'
GATE_SHA256='1db961c0b5051a49f47d2c26466368afcc4e1068073aeac81267f5d0ca34f1a7'
GATE_PLAN_SHA256='a26c40faba319fec47e06d75af9cbc184c80450dc3259f70887c1570a4691ed9'
PUBLIC_START,PUBLIC_END=11185,18685
PROTOCOLS=('buffered_minutes','purged_spells','buffered_100_seconds')
SOURCES=('experiments/modernnca_native.py','experiments/research_modernnca_native.py','experiments/test_modernnca_native.py')
LIMIT_BYTES=6*1024**3
_PAYLOAD_ACCESS_STARTED=False

def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):result.update(block)
    return result.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def source_identity():return {name:digest(ROOT/name) for name in SOURCES}

def seed_streams(seed):
    if type(seed) is not int or seed not in (42,43,44):raise ValueError('Unexpected prospective seed')
    return dict(python=seed,numpy_global=seed+1,torch=seed+2)

def validate_job(job):
    required={"context","target","original","swapped","seed","protocol","block","input_sha256"}
    if set(job)-required-{"parent"} or not required<=set(job):raise ValueError("Unexpected job fields; query labels forbidden")
    if job["seed"] not in (42,43,44) or job["protocol"] not in PROTOCOLS or type(job["block"]) is not int:
        raise ValueError("Unexpected fixed seed/protocol/fold")
    if not 1<=job['block']<=(3 if job['protocol']=='buffered_100_seconds' else 5):
        raise ValueError('Unexpected original outer-fold block')
    paths={str(Path(job[name]).resolve()) for name in ("context","target","original","swapped")}
    if set(job["input_sha256"])!=paths:raise ValueError("Payload provenance must bind exactly four arrays")
    for name in paths:
        if not Path(name).is_relative_to(ROUND.resolve()):raise ValueError("Payload path escapes native round")
        if digest(name)!=job["input_sha256"][name]:raise ValueError("Native payload bytes changed")
    if "parent" in job and not Path(job["parent"]).resolve().is_relative_to(ROUND.resolve()):
        raise ValueError("Replay parent escapes native round")
    if 'parent' in job:
        stage='screen' if job['seed']==42 else f"confirmation_seed{job['seed']}"
        expected=ROUND/stage/job['protocol']/f"block{job['block']}"/'worker'
        if Path(job['parent']).resolve()!=expected.resolve():
            raise ValueError('Replay parent must be the original same-seed/stage/fold worker')

def context_frames(train,validation,labels):
    """Pure owntrain selection with complete real +/-79-frame dependency footprint."""
    train,validation=list(map(int,train)),list(map(int,validation));allowed=set(train)
    if not train or len(allowed)!=len(train) or len(set(validation))!=len(validation) or allowed&set(validation):
        raise ValueError("Invalid outer split")
    if any(f<PUBLIC_START or f>=PUBLIC_END for f in train+validation) or train!=sorted(train) or validation!=sorted(validation):
        raise ValueError("Only ascending original public splits are permitted")
    if not validation:raise ValueError("Empty outer validation set")
    for frame in train:
        at=bisect_left(validation,frame)
        if any(abs(frame-validation[k])<=250 for k in (at-1,at) if 0<=k<len(validation)):
            raise ValueError("Unchanged250-frame outer exclusion is required")
    if any(labels[f] not in ("H","A","D") for f in train):raise ValueError("Invalid owntrain public label")
    observed=[f for f in train if f%5==0 and all(x in allowed for x in range(f-79,f+80))]
    frames=[f for f in observed if labels[f] in ("H","A")]
    if not frames or 2*len(frames)>3000:raise ValueError("Invalid fixed <=3000-row live paired context")
    return frames,observed

def query_indices(length):
    if length<=0:raise ValueError("Empty query set")
    for start in range(0,length,512):
        end=min(start+512,length);index=list(range(start,end));index += [index[-1]]*(512-len(index))
        yield start,end,index

def required_guards(seed,protocol):
    names={"primary","mean_fold_primary","macro","short_iou","prior_matched","prior_short_recall",
        "no_extra_spurious","no_extra_missed","D_recall","false_D"}
    if protocol=="buffered_minutes":names.add("incremental_primary_gain")
    if protocol=="buffered_minutes" and seed==42:names.add("strict_historical_best_gain")
    return names

def all_nine(selections):
    if len(selections)!=3:return False
    for seed,selection in zip((42,43,44),selections):
        pairs=selection.get("pairs",[])
        if not selection.get("passed") or len(pairs)!=3 or {p["protocol"] for p in pairs}!=set(PROTOCOLS):return False
        if any(p["seed"]!=seed or p["model"]!=f"modernnca_context_seed{seed}" or
                set(p["checks"])!=required_guards(seed,p["protocol"]) or not all(p["checks"].values()) for p in pairs):return False
    return True

def audit(folder):
    record=json.loads((folder/"manifest.json").read_text())
    for name,sha in record["cache_sha256"].items():
        if digest(folder/name)!=sha:raise ValueError("Retained stage artifact changed: "+name)
    return record

def require_confirmations():
    selections=[]
    for stage in ("screen","confirmation_seed43","confirmation_seed44"):
        audit(ROUND/stage);selections.append(json.loads((ROUND/stage/"selection.json").read_text()))
    if not all_nine(selections):raise ValueError("All nine seed/protocol conjunctions must pass")

def verify_replay_evidence():
    """Count actual independently refitted workers, never infer proof from folder names."""
    stages=(('screen',42),('confirmation_seed43',43),('confirmation_seed44',44))
    expected={f'{protocol}/block{block}/worker' for protocol in PROTOCOLS
        for block in range(1,4 if protocol=='buffered_100_seconds' else 6)}
    compared=('initial_state_fingerprint.json','state_fingerprint.json','recipe.json',
        'resolved_optimizer.json','resolved_configuration.json','training_losses.npy','epoch_losses.npy',
        'original_probability.npy','swapped_probability.npy','owner.npy','repeat_first.npy')
    def record(path):
        if not path.is_file():raise ValueError('Missing retained replay proof: '+str(path))
        return json.loads(path.read_text())
    proofs=[];stage_hashes={}
    for stage,seed in stages:
        original=ROUND/stage;replay=ROUND/(stage+'_replay')
        for folder,is_replay in ((original,False),(replay,True)):
            manifest=record(folder/'manifest.json');audit(folder)
            if (manifest.get('stage')!=stage or manifest.get('seed')!=seed or manifest.get('replay') is not is_replay
                    or manifest.get('context_models')!=13 or manifest.get('independent_refits')!=(13 if is_replay else 0)):
                raise ValueError('Stage lacks explicit correct replay/seed/count provenance')
            retained={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=folder/'manifest.json'}
            if set(manifest['cache_sha256'])!=retained:raise ValueError('Stage cache does not bind every retained file')
            actual={str(p.relative_to(folder)) for p in folder.rglob('worker') if p.is_dir()}
            if actual!=expected:raise ValueError('Expected exactly thirteen distinct original5/5/3 workers')
            stage_hashes[folder.name]=digest(folder/'manifest.json')
        for relative in sorted(expected):
            worker=replay/relative;parent=original/relative;job_dir=worker.parent
            protocol,block_name,_=Path(relative).parts;block=int(block_name[5:])
            job=record(job_dir/'job.json');validate_job(job)
            if (job['seed']!=seed or job['protocol']!=protocol or job['block']!=block
                    or Path(job.get('parent','')).resolve()!=parent.resolve()):
                raise ValueError('Replay worker is not bound to its original seed/stage/fold parent')
            for name in ('context','target','original','swapped'):
                if Path(job[name]).resolve()!=(job_dir/(name+'.npy')).resolve():
                    raise ValueError('Replay job payload is not its own exact fold payload')
                if (job_dir/(name+'.npy')).read_bytes()!=(parent.parent/(name+'.npy')).read_bytes():
                    raise ValueError('Final replay context/query payload differs: '+name)
            if (job_dir/'context_provenance.json').read_bytes()!=(parent.parent/'context_provenance.json').read_bytes():
                raise ValueError('Final replay context provenance differs')
            summary=record(worker/'summary.json');supervision=record(job_dir/'supervision.json')
            if (not summary.get('passed') or summary.get('seed')!=seed or
                    summary.get('checks',{}).get('independent_refit_exact') is not True or
                    not all(summary['checks'].values())):
                raise ValueError('Worker lacks successful independent200-epoch refit proof')
            if (not supervision.get('passed') or supervision.get('exit_code')!=0 or supervision.get('aborted') is not None
                    or supervision.get('summary_sha256')!=digest(worker/'summary.json')):
                raise ValueError('Replay supervision is missing, aborted, nonzero or unbound')
            for kind in ('stdout','stderr'):
                if supervision.get(kind+'_sha256')!=digest(job_dir/('worker.'+kind+'.log')):
                    raise ValueError('Replay supervision log binding differs')
            artifacts=summary['artifact_sha256']
            actual={p.name for p in worker.iterdir() if p.is_file() and p.name not in ('summary.json','operation.json')}
            if set(artifacts)!=actual or not set(compared+('model.pt',))<=set(artifacts):
                raise ValueError('Replay worker artifacts are incomplete or not fully bound')
            for name,sha in artifacts.items():
                if Path(name).name!=name or digest(worker/name)!=sha:raise ValueError('Replay worker artifact hash changed')
            for name in compared:
                if (worker/name).read_bytes()!=(parent/name).read_bytes():
                    raise ValueError('Final original/replay state/loss/config/probability bytes differ: '+name)
            proofs.append(dict(seed=seed,stage=stage,worker=relative,parent=str(parent.relative_to(ROUND)),
                job_sha256=digest(job_dir/'job.json'),summary_sha256=digest(worker/'summary.json'),
                supervision_sha256=digest(job_dir/'supervision.json')))
    if len(proofs)!=39:raise ValueError('Exactly39 proven independent200-epoch refits are required')
    return dict(independent_process_fits=len(proofs),stage_manifest_sha256=stage_hashes,workers=proofs)

def verify():
    """Global provenance failure always aborts; never rescue a partial native method."""
    folder=ROUND/"protocol_freeze"
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes():raise ValueError("R68 plan changed")
    for group in ("implementation.json","input_sha256.json","protected_artifacts.json","technical_gate.json","head_runtime.json"):
        for name,sha in json.loads((folder/group).read_text()).items():
            if digest(ROOT/name)!=sha:raise ValueError("Frozen global source/input/runtime changed: "+name)
    for name,target in json.loads((folder/"symlinks.json").read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink())!=target:raise ValueError("Frozen source symlink changed")
    if not (ROOT/"experiments/EXPERIMENT_LEDGER.csv").read_bytes().startswith((folder/"EXPERIMENT_LEDGER.csv").read_bytes()):
        raise ValueError("Inherited ledger bytes changed")
    technical_gate()
    if (ROUND/"global_failure.json").exists():raise ValueError("A retained global provenance/replay failure blocks this experiment")
    if (ROUND/"method_failure.json").exists():
        failure=json.loads((ROUND/"method_failure.json").read_text())
        if not failure.get("closed"):raise ValueError("Malformed retained method failure")
        for name,sha in failure["evidence_sha256"].items():
            if digest(ROUND/name)!=sha:raise ValueError("Retained failed-method evidence changed")
        raise ValueError("This fixed method is closed; no retry/partial-fold rescue")
    return json.loads(PLAN.read_text())

class NativeMethodFailure(RuntimeError):pass

def technical_gate(require_pass=True):
    """Verify exact frozen R64; no native access or numerical import is needed."""
    if digest(GATE_SOURCE)!=GATE_SHA256 or digest(ROOT/'experiments/MODERNNCA_ISOLATED_RUNTIME_PLAN.json')!=GATE_PLAN_SHA256:
        raise ValueError('Frozen R64 source/plan differs')
    module=load_module(GATE_SOURCE,'r68_frozen_r64_gate');module.verify()
    if require_pass:
        folder=GATE_ROUND/'runtime_assessment';summary=json.loads((folder/'summary.json').read_text())
        if digest(folder/'summary.json')!=json.loads(PLAN.read_text())['technical_prerequisite']['actual_all_four_stage_summary_sha256']:
            raise ValueError('Plan-bound actual R64 all-four-stage evidence changed')
        expected=[('moderate','first'),('moderate','rebuild'),('full','first'),('full','rebuild')]
        records=summary.get('records',[])
        if (not summary.get('passed') or summary.get('implementation_sha256')!=GATE_SHA256 or
                summary.get('plan_sha256')!=GATE_PLAN_SHA256 or
                [(r.get('size'),r.get('stage')) for r in records]!=expected):
            raise ValueError('All four exact R64 stages must actually pass')
        for (size,stage),record in zip(expected,records):
            path=folder/size/stage/'summary.json';retained=json.loads(path.read_text())
            if (not record.get('passed') or record.get('aborted') is not None or record.get('exit_code')!=0
                    or record.get('summary_sha256')!=digest(path) or not retained.get('passed')):
                raise ValueError('Incomplete or aborted R64 stage')
            checks=retained['checks']
            if (any(not v for k,v in checks.items() if k!='independent_refit_exact')
                    or checks.get('independent_refit_exact')!=(stage=='rebuild')):
                raise ValueError('R64 exactness/refit conjunction differs')
            if (record['wall_seconds']>2400 or retained['fit_seconds']>1800 or
                    max(record['largest_sampled_rss_bytes'],retained['peak_bytes'])>LIMIT_BYTES or
                    any(t['seconds']>300 for t in retained['query_timings'])):
                raise ValueError('R64 resource bound failed')
            for name,sha in retained['artifact_sha256'].items():
                if digest(path.parent/name)!=sha:raise ValueError('R64 retained artifact changed')
            for kind in ('stdout','stderr'):
                if digest(folder/size/(stage+'.'+kind+'.log'))!=record[kind+'_sha256']:
                    raise ValueError('R64 retained log changed')
        for size in ('moderate','full'):
            for name in ('initial_state_fingerprint.json','state_fingerprint.json','recipe.json','resolved_optimizer.json',
                    'training_losses.npy','epoch_losses.npy','original_probability.npy','swapped_probability.npy','owner.npy'):
                if (folder/size/'first'/name).read_bytes()!=(folder/size/'rebuild'/name).read_bytes():
                    raise ValueError('R64 actual independent200-epoch refit differs: '+name)
    return module


class OrbitScaler:
    """Already-fitted paired-owntrain statistics; queries cannot alter them."""
    def __init__(self,mean,scale,execution):self.mean,self.scale,self.execution=mean,scale,execution
    def transform(self,values):
        import numpy as np
        value=np.asarray(values,dtype=np.float64)
        if value.ndim!=2 or value.shape[1]!=166:raise ValueError('Query column schema differs')
        return self.execution.preprocess(value,self.mean,self.scale)


def context_record(inputs,labels,train,validation):
    import numpy as np
    import pandas as pd
    from rich_features import swap_teams
    execution=technical_gate();contract=execution.source_contract();schema=contract['schema']
    if list(inputs.columns)!=schema['columns']:raise ValueError('Immutable166 columns/order differs')
    frames,observed=context_frames(train,validation,labels)
    values=inputs.iloc[frames]
    augmented=pd.concat([values,swap_teams(values)],ignore_index=True).to_numpy(dtype=np.float64)
    permutation=np.asarray(schema['permutation']);signs=np.asarray(schema['signs'],dtype=np.float64)
    np.testing.assert_array_equal(augmented[len(frames):],execution.swap_array(augmented[:len(frames)],permutation,signs))
    mean,scale=execution.symmetric_statistics(augmented,permutation,signs)
    context=execution.preprocess(augmented,mean,scale)
    target=(labels[frames]=='A').astype(np.int64);target=np.r_[target,1-target]
    np.testing.assert_array_equal(context[len(frames):],execution.swap_array(context[:len(frames)],permutation,signs))
    if (context.shape!=(2*len(frames),166) or not np.isfinite(context).all() or
            not np.array_equal(np.bincount(target),[len(frames)]*2)):
        raise ValueError('Paired owntrain context/schema differs')
    record=dict(split_train=np.asarray(train),validation=np.asarray(validation),context_frames=np.asarray(frames),
        observation_eligible=np.asarray(observed),target=target,weight=np.ones(len(target)),
        mean=mean,scale=scale,raw_context=augmented,permutation=permutation,signs=signs,
        footprint_radius=np.asarray(79),outer_exclusion=np.asarray(250))
    return context,target,OrbitScaler(mean,scale,execution),record


def numerical_call(callback):
    """Numerical/resources close the method; provenance and replay failures abort globally."""
    try:return callback()
    except (MemoryError,FloatingPointError,OverflowError,RuntimeError) as error:
        raise NativeMethodFailure(str(error)) from error
    except ValueError as error:
        if str(error) in ('Nonfinite training loss','Invalid binary probability vector'):
            raise NativeMethodFailure(str(error)) from error
        raise


def configuration_record(model):
    return {name:getattr(model,name) for name in
        ('d_in','d_out','d_num','dim','dropout','d_block','n_blocks','T','sample_rate')}


def worker(job_path,output):
    global _PAYLOAD_ACCESS_STARTED
    verify();execution=technical_gate();job=json.loads(job_path.read_text());validate_job(job)
    if sys.flags.ignore_environment or sys.flags.hash_randomization or not sys.flags.no_site or not sys.flags.safe_path:
        raise ValueError('Expected isolated -P -S -B worker and hashseed0')
    output.mkdir(parents=True,exist_ok=False)
    denied=[]
    def deny(*args,**kwargs):denied.append('network_call');raise RuntimeError('Network forbidden during native fit')
    with ExitStack() as stack:
        for owner,name in ((socket.socket,'connect'),(socket.socket,'connect_ex'),(socket,'create_connection'),(urllib.request,'urlopen')):
            stack.enter_context(patch.object(owner,name,deny))
        np,torch=execution.initialize()
        seed=int(job['seed']);streams=seed_streams(seed)
        random.seed(streams['python']);np.random.seed(streams['numpy_global']);torch.manual_seed(streams['torch'])
        contract=execution.source_contract();schema=contract['schema']
        permutation=np.asarray(schema['permutation']);signs=np.asarray(schema['signs'],dtype=np.float64)
        _PAYLOAD_ACCESS_STARTED=True
        data={name:np.load(job[name],mmap_mode='r',allow_pickle=False) for name in ('context','target','original','swapped')}
        context_array,target_array,original,swapped=[data[k] for k in ('context','target','original','swapped')]
        if (context_array.ndim!=2 or context_array.shape[1]!=166 or not 2<=len(context_array)<=3000 or len(context_array)%2
                or target_array.shape!=(len(context_array),) or target_array.dtype!=np.int64
                or original.ndim!=2 or original.shape!=swapped.shape or original.shape[1]!=166 or not len(original)
                or any(v.dtype!=np.float64 or not np.isfinite(v).all() for v in (context_array,original,swapped))
                or not np.array_equal(np.bincount(target_array),[len(context_array)//2]*2)
                or not np.array_equal(target_array[len(target_array)//2:],1-target_array[:len(target_array)//2])):
            raise ValueError('Native context/query schema differs')
        half=len(context_array)//2
        np.testing.assert_array_equal(context_array[half:],execution.swap_array(context_array[:half],permutation,signs))
        np.testing.assert_array_equal(swapped,execution.swap_array(original,permutation,signs))
        array_before={name:execution.array_record(v) for name,v in data.items()}
        context=torch.tensor(context_array,dtype=torch.float64,device='cpu')
        target=torch.tensor(target_array,dtype=torch.int64,device='cpu')
        tensor_before=dict(context=execution.array_record(context),target=execution.array_record(target))
        Model=execution.official_model_class()
        model=Model(d_in=166,d_num=166,d_out=2,**contract['model_config']).to('cpu').double()
        if any(p.device.type!='cpu' or p.dtype!=torch.float64 for p in model.parameters()):raise ValueError('Expected CPUfloat64')
        configuration=configuration_record(model)
        if configuration!=dict(d_in=166*34,d_out=2,d_num=166,dim=128,dropout=.1,d_block=512,n_blocks=0,T=1,sample_rate=.5):
            raise ValueError('Resolved author model configuration differs')
        initial=execution.state_record(model)
        (losses,epoch_losses),fitting=numerical_call(lambda:execution.operation(output,'complete_200_epoch_fit','fit',
            lambda:execution.train_complete(model,torch,context,target,contract,output)))
        if not np.isfinite(losses).all() or epoch_losses.shape!=(200,):raise NativeMethodFailure('Invalid complete200-epoch losses')
        state=execution.state_record(model)
        torch.save({'params':model.state_dict()},output/'model.pt')
        for name,value in (('training_losses',losses),('epoch_losses',epoch_losses)):
            np.save(output/(name+'.npy'),value)
        for name,value in (('initial_state_fingerprint.json',initial),('state_fingerprint.json',state)):
            write_json(output/name,value)
        p0,t0=numerical_call(lambda:execution.canonical_predict(model,torch,context,target,original,output,'original'))
        p1,t1=numerical_call(lambda:execution.canonical_predict(model,torch,context,target,swapped,output,'swapped'))
        owner=.5*(p0+p1[:,[1,0]])
        r0,t2=numerical_call(lambda:execution.canonical_predict(model,torch,context,target,swapped,output,'equivariant_original'))
        r1,t3=numerical_call(lambda:execution.canonical_predict(model,torch,context,target,original,output,'equivariant_swapped'))
        np.testing.assert_array_equal(.5*(r0+r1[:,[1,0]]),owner[:,[1,0]])
        repeated,tr=numerical_call(lambda:execution.canonical_predict(model,torch,context,target,original[:512],output,'repeat_first'))
        np.testing.assert_array_equal(repeated,p0[:512])
        reloaded=Model(d_in=166,d_num=166,d_out=2,**contract['model_config']).to('cpu').double()
        reloaded.load_state_dict(torch.load(output/'model.pt',map_location='cpu',weights_only=True)['params'],strict=True)
        if execution.state_record(reloaded)!=state or configuration_record(reloaded)!=configuration:
            raise ValueError('Strict saved reload changed state/configuration')
        reload0,t4=numerical_call(lambda:execution.canonical_predict(reloaded,torch,context,target,original,output,'reload_original'))
        reload1,t5=numerical_call(lambda:execution.canonical_predict(reloaded,torch,context,target,swapped,output,'reload_swapped'))
        np.testing.assert_array_equal(reload0,p0);np.testing.assert_array_equal(reload1,p1)
        if execution.state_record(model)!=state:raise ValueError('Query changed fitted model state')
        if configuration_record(model)!=configuration:raise ValueError('Training/query changed model configuration')
        if tensor_before!=dict(context=execution.array_record(context),target=execution.array_record(target)):
            raise ValueError('Training/query changed context tensors')
        if array_before!={name:execution.array_record(v) for name,v in data.items()}:
            raise ValueError('Native read-only payload arrays changed')
        recipe=dict(features=166,context_rows=len(context_array),query_rows=len(original),class_order=['H','A'],
            epochs=200,batches_per_epoch=math.ceil(len(context_array)/1024),batch_size=1024,
            precision='float64',backend='cpu',cpu_threads=2,seed=streams,
            model_config=contract['model_config'],training_config=contract['training_config'],
            resolved_configuration=configuration,definitions=contract['definitions'],schema=schema,arrays=array_before,context_tensors=tensor_before,
            preprocessing='Paired-owntrain-only orbit-tied mean/empty0 population scale/constant1',
            checkpoint='Final complete200epochs only; no outer labels or checkpoint selection',
            source_parity='Exact frozen R64 author model/helper/extracted train_epoch and candidate self-exclusion',
            fitted_estimator_pickle_used=False)
        write_json(output/'recipe.json',recipe)
        write_json(output/'resolved_configuration.json',configuration)
        for name,value in (('original_probability',p0),('swapped_probability',p1),('owner',owner),('repeat_first',repeated)):
            np.save(output/(name+'.npy'),value)
        if 'parent' in job:
            parent=Path(job['parent'])
            for name in ('initial_state_fingerprint.json','state_fingerprint.json','recipe.json','resolved_optimizer.json','resolved_configuration.json',
                    'training_losses.npy','epoch_losses.npy','original_probability.npy','swapped_probability.npy','owner.npy','repeat_first.npy'):
                if (output/name).read_bytes()!=(parent/name).read_bytes():raise ValueError('Fresh200-epoch replay differs: '+name)
        if denied:raise ValueError('Native worker attempted network')
        verify()
        checks=dict(complete200_epochs=True,finite_losses=True,probabilities_valid=execution.probability_checks(owner,len(original)),
            paired_H0_A1=True,swap_exact=True,repeat_exact=True,strict_saved_reload_exact=True,
            fitted_model_immutable=True,resolved_model_configuration_immutable=True,context_arrays_tensors_immutable=True,
            independent_refit_exact='parent' in job,rss_within_limit=execution.peak_bytes()<=LIMIT_BYTES)
        write_json(output/'summary.json',dict(passed=all(v for k,v in checks.items() if k!='independent_refit_exact'),
            checks=checks,seed=seed,context_rows=len(context_array),query_rows=len(original),fit_seconds=fitting,
            query_timings=t0+t1+t2+t3+tr+t4+t5,peak_bytes=execution.peak_bytes(),query_labels_supplied=False,
            original_D_mass_supplied_to_worker=False,native_public_processing=True,competition_inputs_read=True,
            synthetic_only=False,artifact_sha256={p.name:digest(p) for p in sorted(output.iterdir())
                if p.is_file() and p.name!='operation.json'}))


def supervise(job_path,output):
    """Serial isolated CPU worker with R64's same prospective limits."""
    verify();execution=technical_gate();job=json.loads(job_path.read_text());validate_job(job)
    start,peak,aborted=time.monotonic(),0,None
    log,error=output.parent/'worker.stdout.log',output.parent/'worker.stderr.log'
    with log.open('xb') as stdout,error.open('xb') as stderr:
        process=subprocess.Popen([str(execution.PYTHON),'-P','-S','-B',str(Path(__file__).resolve()),
            '--worker','--job',str(job_path),'--output',str(output)],cwd=ROOT,
            env={**os.environ,**execution.ENVIRONMENT},stdout=stdout,stderr=stderr)
        while process.poll() is None:
            sample=subprocess.run(['/bin/ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True)
            if sample.returncode==0 and sample.stdout.strip():peak=max(peak,int(sample.stdout.strip())*1024)
            status=output/'operation.json';current=json.loads(status.read_text()) if status.exists() else {}
            if peak>LIMIT_BYTES:aborted='rss_over_6GiB'
            elif time.monotonic()-start>2400:aborted='worker_over_2400seconds'
            elif current.get('active') and time.monotonic()-current['started']>current['limit_seconds']:
                aborted=current['kind']+'_over_'+str(current['limit_seconds'])+'seconds'
            if aborted:process.terminate();break
            time.sleep(.1)
        try:code=process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();code=process.wait(timeout=10)
    path=output/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {}
    elapsed=time.monotonic()-start;peak=max(peak,summary.get('peak_bytes',0))
    if aborted is None and elapsed>2400:aborted='worker_over_2400seconds'
    if aborted is None and peak>LIMIT_BYTES:aborted='rss_over_6GiB'
    passed=code==0 and aborted is None and summary.get('passed',False)
    record=dict(passed=passed,exit_code=code,aborted=aborted,peak_sampled_bytes=peak,elapsed_seconds=elapsed,
        stdout_sha256=digest(log),stderr_sha256=digest(error),summary_sha256=digest(path) if path.exists() else None)
    write_json(output.parent/'supervision.json',record);verify()
    if aborted or code==3:raise NativeMethodFailure('Numerical/resource worker failed; no retry/partial-fold rescue')
    if not passed:raise ValueError('Native global provenance/state/replay worker failure; retained evidence')
    for name,sha in summary['artifact_sha256'].items():
        if digest(output/name)!=sha:raise ValueError('Worker artifact hash changed')
    return record
def preflight():
    """Only hashes, source AST and pure invented lists. Never native/scientific work."""
    started=time.monotonic()
    for name in SOURCES:
        payload=(ROOT/name).read_bytes();tree=ast.parse(payload);compile(payload,str(ROOT/name),'exec')
        for statement in tree.body:
            if isinstance(statement,(ast.Import,ast.ImportFrom)):
                names=[n.name.split('.')[0] for n in statement.names] if isinstance(statement,ast.Import) else [(statement.module or '').split('.')[0]]
                if set(names)&{'numpy','pandas','torch','scipy','sklearn','TALENT','joblib'}:
                    raise ValueError('Scientific top-level import: '+name)
    tests=load_module(ROOT/SOURCES[2],'r68_pure_tests').run_tests()
    plan=json.loads(PLAN.read_text())
    for group in (plan['input_sha256'],json.loads((ROUND/'protected_artifacts.json').read_text()),
            json.loads((ROUND/'source_closure.json').read_text()),json.loads((ROUND/'head_runtime.json').read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError('Prepared source/input/runtime changed: '+name)
    for name,target in json.loads((ROUND/'symlinks.json').read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink())!=target:raise ValueError('Source symlink changed')
    if not (ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes().startswith((ROUND/'preparation_ledger.csv').read_bytes()):
        raise ValueError('Historical ledger prefix changed during preparation')
    execution=technical_gate()
    if (execution.EPOCHS,execution.BATCH_SIZE,execution.QUERY_SIZE,execution.CPU_THREADS,
            execution.FIT_SECONDS,execution.PREDICT_SECONDS,execution.WORKER_SECONDS,execution.LIMIT_BYTES)!=(200,1024,512,2,1800,300,2400,LIMIT_BYTES):
        raise ValueError('R64 prospective numerical/resource constants differ')
    execution.assert_no_scientific_imports()
    write_json(ROUND/'source_preflight.json',dict(passed=True,seconds=time.monotonic()-started,
        source_sha256=source_identity(),plan_sha256=digest(PLAN),pure_tests=tests,
        native_arrays_loaded=False,scientific_imports=False,model_forward=False,
        actual_R64_all_four_stage_pass_verified=True,
        technical_summary_sha256=digest(GATE_ROUND/'runtime_assessment/summary.json'),
        native_processing_authorized=False,freeze_authorized=False))
    print('R68 source/pure preflight passed; no native arrays/scientific imports/model calls',flush=True)


def freeze():
    """Only root's exact review plus all-four actual R64 PASS may freeze native work."""
    review=json.loads((ROUND/'independent_review.json').read_text());pre=json.loads((ROUND/'source_preflight.json').read_text())
    if (not review.get('approved_for_freeze') or review.get('source_sha256')!=source_identity()
            or review.get('plan_sha256')!=digest(PLAN)):
        raise ValueError('Independent ROOT review must bind final three sources/plan')
    if (not pre['passed'] or pre['source_sha256']!=source_identity() or pre['plan_sha256']!=digest(PLAN)
            or pre['technical_summary_sha256']!=digest(GATE_ROUND/'runtime_assessment/summary.json')):
        raise ValueError('Final source-only pure preflight and technical binding are required')
    technical_gate()
    closure=json.loads((ROUND/'source_closure.json').read_text());closure.update(source_identity())
    for path in (PLAN,ROUND/'source_preflight.json',ROUND/'independent_review.json',ROUND/'source_closure.json',
            ROUND/'protected_artifacts.json',ROUND/'symlinks.json',ROUND/'head_runtime.json',ROUND/'preparation_ledger.csv'):
        closure[str(path.relative_to(ROOT))]=digest(path)
    technical={str(GATE_SOURCE.relative_to(ROOT)):digest(GATE_SOURCE)}
    for folder in (GATE_ROUND/'protocol_freeze',GATE_ROUND/'runtime_assessment'):
        for path in sorted(folder.rglob('*')):
            if path.is_file():technical[str(path.relative_to(ROOT))]=digest(path)
    folder=ROUND/'protocol_freeze';folder.mkdir(exist_ok=False)
    (folder/'plan.json').write_bytes(PLAN.read_bytes())
    write_json(folder/'implementation.json',closure);write_json(folder/'technical_gate.json',technical)
    write_json(folder/'input_sha256.json',json.loads(PLAN.read_text())['input_sha256'])
    for name in ('protected_artifacts.json','symlinks.json','head_runtime.json'):
        (folder/name).write_bytes((ROUND/name).read_bytes())
    (folder/'EXPERIMENT_LEDGER.csv').write_bytes((ROOT/'experiments/EXPERIMENT_LEDGER.csv').read_bytes())
    write_json(folder/'freeze.json',dict(source_sha256=source_identity(),plan_sha256=digest(PLAN),
        technical_gate_source_sha256=GATE_SHA256,all_four_R64_stages_actually_passed=True,
        no_native_arrays_loaded_before_freeze=True,model_forward_started=False,selected17_protected=True))
    verify();print('Frozen R68 after ROOT review and actual R64 four-stage pass; no model call',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--preflight',action='store_true');actions.add_argument('--freeze',action='store_true')
    actions.add_argument('--verify',action='store_true');actions.add_argument('--worker',action='store_true')
    parser.add_argument('--job',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.preflight:preflight()
    elif args.freeze:freeze()
    elif args.verify:verify();print('R68 frozen provenance verified',flush=True)
    else:
        if args.job is None or args.output is None:parser.error('Worker needs job/output')
        if not args.output.resolve().is_relative_to(ROUND.resolve()):parser.error('Worker output escapes R68')
        try:worker(args.job,args.output)
        except BaseException as error:
            if args.output.exists():
                write_json(args.output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                    method_numerical_resource_failure=isinstance(error,NativeMethodFailure),
                    native_payload_access_started=_PAYLOAD_ACCESS_STARTED,no_partial_fold_rescue=True))
            if isinstance(error,NativeMethodFailure):raise SystemExit(3)
            raise
