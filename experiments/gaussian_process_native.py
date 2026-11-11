"""Fixed native Laplace-GP ownership with frozen technical admission and evidence.

Source-only actions use the standard library. The numerical method delegates
only reviewed frozen GPv2 technical primitives, never the invented worker or its admission.
"""
import argparse
import ast
from bisect import bisect_left
from contextlib import ExitStack
import datetime
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import struct
import sys
import time
import traceback
import urllib.request
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_gaussian_process_native_v1"
PLAN=ROOT/"experiments/GAUSSIAN_PROCESS_NATIVE_PLAN.json"
GATE_SOURCE=ROOT/"experiments/gaussian_process_runtime_gate_v2.py"
GATE_ROUND=ROOT/"data/control_gaussian_process_runtime_v2"
GATE_PLAN=ROOT/"experiments/GAUSSIAN_PROCESS_RUNTIME_PLAN_V2.json"
ISOLATED=ROOT/"data/control_isolated_python_v1"
PUBLIC_START,PUBLIC_END=11185,18685
PROTOCOLS=("buffered_minutes","purged_spells","buffered_100_seconds")
SOURCES=("experiments/gaussian_process_native.py","experiments/research_gaussian_process_native.py","experiments/test_gaussian_process_native.py", "experiments/audit_gaussian_process_runtime_v2.py")
PAYLOADS=("context","target","original","swapped","base_frames")
LIMIT_BYTES=6*1024**3
_PAYLOAD_ACCESS_STARTED=False
TECHNICAL_SEMANTIC_ARTIFACTS=("recipe.json","fit_record.json","full_state.json","posterior_traces.json",
    "original_probability.npy","swapped_probability.npy","owner.npy")
SEMANTIC_ARTIFACTS=TECHNICAL_SEMANTIC_ARTIFACTS+(
    "repeat_first.npy","repeat_first_swapped.npy","classes.json","payload_immutability.json","head_seed.json")
WORKER_CHECKS=("one_actual_author_fit","two_actual_posterior_calls","actual_convergence_observed",
    "full_fitted_preprocessing_kernel_RNG_state_immutable","raw_payloads_immutable","canonical_reload_exact","semantic_reload_exact",
    "repeat_byte_exact","query_labels_forbidden","no_network_attempts")
REQUIRED_WORKER_ARTIFACTS=set(SEMANTIC_ARTIFACTS)|{
    "gp_model.pkl","threadpools.json","operation.json","operations.jsonl","code_identity_audit.json"}

def digest(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):result.update(block)
    return result.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+".writing")
    write_json(temporary,value);temporary.replace(path)

def duration_ok(value,limit):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<limit

def validate_fit_evidence(execution, folder, seed=None, query_rows=None):
    """Validate the one actual GP fit and its two observed posterior calls."""
    record=json.loads((folder/"fit_record.json").read_text())
    rows=record.get("raw_X",{}).get("shape",[0])[0]
    roles=execution.training_roles(rows)
    if query_rows is not None:roles["queries"]=f"feature_only_all{query_rows}"
    state=json.loads((folder/"full_state.json").read_text())
    state_sha=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if (record.get("actual_author_fit") is not True or record.get("actual_posterior_calls")!=2
            or record.get("roles")!=roles or record.get("optimizer") is not None
            or record.get("unit_amplitude")!=1.0 or record.get("class_order")!=[0,1]
            or record.get("posterior_contract")!=execution.posterior_source_contract()
            or record.get("semantic_sha256")!=state_sha):
        raise ValueError("Actual fixed GP fit/roles/full-state evidence differs")
    traces=json.loads((folder/"posterior_traces.json").read_text())
    if len(traces)!=2:raise ValueError("Exactly two actual posterior calculations are required")
    for ordinal,trace in enumerate(traces):
        if trace.get("ordinal")!=ordinal:raise ValueError("Posterior invocation order differs")
        execution.validate_posterior_trace(trace)
    if seed is not None:
        witness=json.loads((folder/"head_seed.json").read_text())
        if witness!=dict(requested_seed=seed,public_seed=seed,binary_seed=seed,actual_author_fits=1,actual_posterior_calls=2):
            raise ValueError("Actual GP random-state/fit provenance differs from the job")
    return record

def load_module(path,name):
    path=Path(path).resolve()
    existing=sys.modules.get(name)
    if existing is not None:
        if not getattr(existing,"__file__",None) or Path(existing.__file__).resolve()!=path:
            raise ValueError("Canonical module name is already bound to a different file: "+name)
        return existing
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:del sys.modules[name]
        raise
    return module

def source_identity():return {name:digest(ROOT/name) for name in SOURCES}

def require_private_runtime(prefix,base_prefix,executable,controller):
    expected=ISOLATED/("head_runtime" if controller else "python")
    if (Path(prefix).resolve()!=expected.resolve() or Path(base_prefix).resolve()!=(ISOLATED/"python").resolve()
            or Path(executable).resolve()!=(ISOLATED/"python/bin/python3.12").resolve()):
        raise ValueError("Expected the bound private controller or model runtime")

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
        if any(p["seed"]!=seed or p["model"]!=f"gaussian_process_context_seed{seed}" or
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

def technical_header(summary,source_sha,plan_sha):
    expected=[(size,stage) for size in ("moderate","full") for stage in ("first","rebuild")]
    records=summary.get("records",[])
    if (summary.get("passed") is not True or summary.get("implementation_sha256")!=source_sha
            or summary.get("plan_sha256")!=plan_sha
            or [(r.get("size"),r.get("stage")) for r in records]!=expected):
        raise ValueError("Actual GPv2 technical all-four-stage PASS with exact identities is required")
    if any(r.get("passed") is not True or r.get("exit_code")!=0 or r.get("aborted") is not None for r in records):
        raise ValueError("Every technical stage needs exit0 and no resource abort")
    return expected

def technical_gate(require_pass=True):
    plan=json.loads(PLAN.read_text());prerequisite=plan["technical_prerequisite"]
    if require_pass and (not prerequisite.get("actual_all_four_stage_summary_sha256")
            or prerequisite.get("status")!="ACTUAL_ALL_FOUR_STAGE_PASS_BOUND"):
        raise ValueError("GPv2 technical evidence is pending; native admission is forbidden")
    if digest(GATE_SOURCE)!=prerequisite["source_sha256"] or digest(GATE_PLAN)!=prerequisite["plan_sha256"]:
        raise ValueError("Frozen GPv2 source or plan differs")
    if digest(GATE_ROUND/"protocol_freeze/freeze.json")!=prerequisite["source_freeze_sha256"]:
        raise ValueError("Frozen GPv2 source freeze differs")
    execution=load_module(GATE_SOURCE,"gaussian_process_runtime_gate_v2")
    execution.verify()
    if not require_pass:return execution
    assessment=GATE_ROUND/"runtime_assessment";path=assessment/"summary.json"
    if digest(path)!=prerequisite["actual_all_four_stage_summary_sha256"]:
        raise ValueError("Plan-bound actual GPv2 summary differs")
    review_path=ROOT/prerequisite["actual_independent_review_path"]
    if digest(review_path)!=prerequisite["actual_independent_review_sha256"]:
        raise ValueError("Actual GPv2 continuation audit changed")
    independent=json.loads(review_path.read_text())
    if (independent.get("passed") is not True or independent.get("actual_all_four_stage_pass") is not True
            or independent.get("technical_summary_sha256")!=prerequisite["actual_all_four_stage_summary_sha256"]
            or independent.get("actual_author_fits")!=4 or independent.get("actual_posterior_calls")!=8
            or independent.get("semantic_refit_comparisons")!=14
            or independent.get("source_or_dependencies_changed") is not False or independent.get("native_processing") is not False
            or independent.get("model_calls") is not False
            or independent.get("memory_measures",{}).get("conservative_observed_max_bytes",LIMIT_BYTES+1)>LIMIT_BYTES
            or independent.get("resource_admission_sha256")!=digest(GATE_ROUND/"resource_admission.json")):
        raise ValueError("Actual GPv2 audit does not establish the complete bounded result")
    calendar_path=ROOT/prerequisite["actual_calendar_audit_path"]
    if (digest(calendar_path)!=prerequisite["actual_calendar_audit_sha256"]
            or independent.get("prior_actual_audit_sha256")!=digest(calendar_path)):
        raise ValueError("Retained actual GPv2 prior audit differs")
    if not prerequisite["metadata_sha256"]:raise ValueError("Complete technical evidence must be bound")
    current={str(p.relative_to(ROOT)) for p in GATE_ROUND.rglob("*") if p.is_file()}
    if not current<=set(prerequisite["metadata_sha256"]):raise ValueError("Unbound technical evidence file")
    for name,sha in prerequisite["metadata_sha256"].items():
        if digest(ROOT/name)!=sha:raise ValueError("Bound technical evidence changed: "+name)
    summary=json.loads(path.read_text());expected=technical_header(summary,digest(GATE_SOURCE),digest(GATE_PLAN))
    for (size,stage),supervision in zip(expected,summary["records"]):
        folder=assessment/size/stage;record=json.loads((folder/"summary.json").read_text())
        if (record.get("passed") is not True or record.get("size")!=size or record.get("stage")!=stage
                or record.get("synthetic_only") is not True or record.get("native_arrays_read") is not False
                or set(record.get("checks",{}))!={"one_actual_author_fit","two_actual_posterior_calls","actual_convergence_observed",
                    "finite_normalized_probabilities","exact_first_repeat","exact_team_swap","raw_arrays_immutable",
                    "full_fitted_preprocessing_kernel_RNG_state_immutable","semantic_reload_exact","canonical_reload_exact","independent_refit_exact"}
                or any(value is not True for key,value in record["checks"].items() if key!="independent_refit_exact")
                or record["checks"]["independent_refit_exact"] is not (stage=="rebuild")):
            raise ValueError("Technical numerical/state/reload conjunction is incomplete")
        if (json.loads((assessment/size/(stage+".supervision.json")).read_text())!=supervision
                or supervision["summary_sha256"]!=digest(folder/"summary.json")
                or supervision["elapsed"]!=execution.clock_elapsed(supervision["started"],supervision["ended"])
                or any(not execution.duration_ns_ok(v,900) for v in supervision["elapsed"].values())
                or max(record["peak_bytes"],supervision["largest_sampled_rss_bytes"])>LIMIT_BYTES):
            raise ValueError("Technical resource or supervision binding differs")
        execution.validate_operation_evidence(folder,record)
        for kind in ("stdout","stderr"):
            if digest(assessment/size/(stage+"."+kind+".log"))!=supervision[kind+"_sha256"]:raise ValueError("Technical log changed")
        artifacts=record["artifact_sha256"]
        actual={p.name for p in folder.iterdir() if p.is_file() and p.name not in ("summary.json","operation.json")}
        if set(artifacts)!=actual or set(record["exact_artifacts"])!=set(TECHNICAL_SEMANTIC_ARTIFACTS):
            raise ValueError("Technical artifact closure differs")
        for name,sha in artifacts.items():
            if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError("Technical artifact changed")
        validate_fit_evidence(execution,folder)
    for size in ("moderate","full"):
        for name in TECHNICAL_SEMANTIC_ARTIFACTS:
            if (assessment/size/"first"/name).read_bytes()!=(assessment/size/"rebuild"/name).read_bytes():
                raise ValueError("Technical fresh refit differs: "+name)
    return execution

def validate_job(job):
    required=set(PAYLOADS)|{"seed","protocol","block","input_sha256"}
    if set(job)-required-{"parent"} or not required<=set(job):raise ValueError("Exactly five payloads; query labels forbidden")
    if job["seed"] not in (42,43,44) or job["protocol"] not in PROTOCOLS or type(job["block"]) is not int:
        raise ValueError("Unexpected fixed seed/protocol/fold")
    if job["block"] not in range(1,4 if job["protocol"]=="buffered_100_seconds" else 6):
        raise ValueError("Only the original thirteen protocol blocks are permitted")
    paths={str(Path(job[name]).resolve()) for name in PAYLOADS}
    if len(paths)!=5 or set(job["input_sha256"])!=paths:raise ValueError("Payload provenance must bind exactly five distinct arrays")
    for name in paths:
        if not Path(name).is_relative_to(ROUND.resolve()) or digest(name)!=job["input_sha256"][name]:
            raise ValueError("Payload escaped its round or changed bytes")
    if "parent" in job:
        stage="screen" if job["seed"]==42 else f"confirmation_seed{job['seed']}"
        expected=ROUND/stage/job["protocol"]/f"block{job['block']}"/"worker"
        if Path(job["parent"]).resolve()!=expected.resolve():raise ValueError("Replay parent must match original seed/stage/fold")

def context_record(inputs,labels,train,validation):
    import numpy as np
    import pandas as pd
    from rich_features import swap_teams
    if inputs.shape[1]!=166:raise ValueError("Expected original102 plus frozen R37latent64")
    frames,observed=context_frames(train,validation,labels)
    values=inputs.iloc[frames]
    augmented=pd.concat([values,swap_teams(values)],ignore_index=True)
    context=np.ascontiguousarray(augmented.to_numpy(dtype=np.float64,copy=True))
    target=(labels[frames]=="A").astype(np.int64);target=np.r_[target,1-target]
    base_frames=np.asarray(frames,dtype=np.int64)
    if context.shape!=(2*len(frames),166) or not np.array_equal(np.bincount(target),[len(frames)]*2):
        raise ValueError("Raw paired context shape or class balance differs")
    return context,target,base_frames,dict(split_train=np.asarray(train),validation=np.asarray(validation),
        context_frames=base_frames,observation_eligible=np.asarray(observed),target=target,
        weight=np.ones(len(target)),footprint_radius=np.asarray(79))

def verify():
    technical_gate()  # Must fail pending/null before any native scientific work.
    folder=ROUND/"protocol_freeze"
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes():raise ValueError("GP native plan changed")
    for group in ("implementation.json","input_sha256.json","protected_artifacts.json","technical_gate.json","head_runtime.json"):
        for name,sha in json.loads((folder/group).read_text()).items():
            if digest(ROOT/name)!=sha:raise ValueError("Frozen global source/input/runtime changed: "+name)
    for name,target in json.loads((folder/"symlinks.json").read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink())!=target:raise ValueError("Frozen source link changed")
    if not (ROOT/"experiments/EXPERIMENT_LEDGER.csv").read_bytes().startswith((folder/"EXPERIMENT_LEDGER.csv").read_bytes()):
        raise ValueError("Preserved ledger prefix changed")
    if (ROUND/"global_failure.json").exists():raise ValueError("Retained global failure blocks this experiment")
    if (ROUND/"method_failure.json").exists():
        failure=json.loads((ROUND/"method_failure.json").read_text())
        if failure.get("closed") is not True:raise ValueError("Malformed retained method failure")
        for name,sha in failure["evidence_sha256"].items():
            if digest(ROUND/name)!=sha:raise ValueError("Failed-method evidence changed")
        raise ValueError("This one fixed method is closed; no retry or partial-fold rescue")
    return json.loads(PLAN.read_text())

def require_admission():
    approval=json.loads((ROUND/"resource_admission.json").read_text())
    if (approval.get("approved_for_native_workers") is not True or approval.get("other_model_workers_stopped") is not True
            or approval.get("predecessor_r73_closed_or_passed") is not True
            or approval.get("freeze_sha256")!=digest(ROUND/"protocol_freeze/freeze.json")):
        raise ValueError("Separate root admission and free model resources are required")

class NativeMethodFailure(RuntimeError):pass

def native_query_checks(execution,model,data,payload,before,output):
    """Unchanged GPv2 technical checks, plus retained native repeat/classes/raw-state proof."""
    import numpy as np
    diagnostics=execution.query_and_reload_checks(model,data,output)
    for name,query,probability in (("repeat_first",payload["original"],"original_probability.npy"),
            ("repeat_first_swapped",payload["swapped"],"swapped_probability.npy")):
        repeated=execution.canonical_predict(model,query[:512])
        reference=np.load(output/probability,mmap_mode="r",allow_pickle=False)[:512]
        if execution.array_record(repeated)!=execution.array_record(reference):raise ValueError("Retained native first-query repeat differs")
        np.save(output/(name+".npy"),repeated,allow_pickle=False)
    after={name:execution.array_record(value) for name,value in payload.items()}
    if before!=after:raise ValueError("Raw native payload bytes changed during queries")
    write_json(output/"classes.json",dict(classes=model.classifier.classes_.tolist(),meaning=["H","A"]))
    write_json(output/"payload_immutability.json",dict(before=before,after=after,exact=True))
    return diagnostics

def worker(job_path,output):
    global _PAYLOAD_ACCESS_STARTED
    verify();require_admission();execution=technical_gate()
    job=json.loads(job_path.read_text());validate_job(job)
    output.mkdir(parents=True,exist_ok=False)
    denied=[]
    def deny(*args,**kwargs):denied.append(True);raise RuntimeError("Network forbidden in native GP worker")
    with ExitStack() as stack:
        for owner,name in ((socket.socket,"connect"),(socket.socket,"connect_ex"),(socket,"create_connection"),(urllib.request,"urlopen")):
            stack.enter_context(patch.object(owner,name,deny))
        np,thread_limits,thread_info=execution.initialize()
        stack.enter_context(thread_limits(limits=1))
        pools=thread_info()
        if any(pool["num_threads"]!=1 for pool in pools):raise ValueError("Expected one numerical thread")
        _PAYLOAD_ACCESS_STARTED=True
        payload={name:np.load(job[name],mmap_mode="r",allow_pickle=False) for name in PAYLOADS}
        context,target,original,swapped,base_frames=[payload[name] for name in PAYLOADS]
        if (context.ndim!=2 or context.shape[1]!=166 or not 2<=len(context)<=3000 or len(context)%2
                or target.dtype!=np.int64 or target.shape!=(len(context),)
                or base_frames.dtype!=np.int64 or base_frames.shape!=(len(context)//2,)
                or original.ndim!=2 or original.shape!=swapped.shape or original.shape[1]!=166 or len(original)<512
                or any(v.dtype!=np.float64 or not v.flags.c_contiguous for v in (context,original,swapped))
                or not np.array_equal(np.bincount(target),[len(context)//2]*2)
                or not np.array_equal(target[len(target)//2:],1-target[:len(target)//2])
                or not np.all(np.diff(base_frames)>0) or np.any(base_frames%5)
                or np.any(base_frames<PUBLIC_START) or np.any(base_frames>=PUBLIC_END)):
            raise ValueError("Native raw166/five-payload/H0A1 schema differs")
        permutation=np.asarray(execution.load_schema()["permutation"],dtype=np.int64)
        if (execution.array_record(context[len(base_frames):])!=execution.array_record(context[:len(base_frames),permutation])
                or execution.array_record(swapped)!=execution.array_record(original[:,permutation])):
            raise ValueError("Actual raw team-swap bytes differ")
        before={name:execution.array_record(value) for name,value in payload.items()}
        write_json(output/"threadpools.json",pools)
        recipe=dict(head_seed=job["seed"],protocol=job["protocol"],block=job["block"],
            context_shape=list(context.shape),query_shape=list(original.shape),class_order=["H","A"],
            payloads=before,source_contract=execution.source_contract(),query_batch=512,
            raw_context_refit=True,foreign_estimator_pickle_used=False,local_pickle_reload_audit=True,
            actual_author_fits=1,actual_posterior_calls=2,calibration=None)
        write_json(output/"recipe.json",recipe)
        data=dict(raw_context=context,target=target,raw_query=original,permutation=permutation)
        import sklearn.gaussian_process as author
        constructor=author.GaussianProcessClassifier
        roles=execution.training_roles
        def seeded_classifier(*args,**kwargs):
            if kwargs.get("random_state")!=42:raise ValueError("Frozen GP constructor default changed")
            kwargs["random_state"]=job["seed"]
            return constructor(*args,**kwargs)
        def native_roles(rows):
            value=roles(rows);value["queries"]=f"feature_only_all{len(original)}";return value
        with patch.object(author,"GaussianProcessClassifier",seeded_classifier),patch.object(execution,"training_roles",native_roles):
            model,fit_elapsed=execution.operation(output,execution.FIT_OPERATION,300,
                lambda:execution.fit_with_author_recording(data,output))
        base=model.classifier.base_estimator_
        if model.classifier.random_state!=job["seed"] or base.random_state!=job["seed"]:
            raise ValueError("Actual public/binary GP seed differs from matched jobseed")
        write_json(output/"head_seed.json",dict(requested_seed=job["seed"],public_seed=model.classifier.random_state,
            binary_seed=base.random_state,actual_author_fits=1,actual_posterior_calls=2))
        fitted=execution.state_digest(model)
        diagnostics,query_elapsed=execution.operation(output,execution.QUERY_OPERATION,300,
            lambda:native_query_checks(execution,model,data,payload,before,output))
        if fitted!=execution.state_digest(model) or before!={name:execution.array_record(value) for name,value in payload.items()}:
            raise ValueError("Complete native payload or learned state changed")
        validate_fit_evidence(execution,output,job["seed"],len(original))
        if "parent" in job:
            parent=Path(job["parent"])
            for name in SEMANTIC_ARTIFACTS:
                if (output/name).read_bytes()!=(parent/name).read_bytes():raise ValueError("Fresh raw-context refit differs: "+name)
        if denied:raise ValueError("Network access was attempted")
        verify()
        write_json(output/"summary.json",dict(passed=True,seed=job["seed"],protocol=job["protocol"],block=job["block"],
            fresh_context_replay="parent" in job,fresh_refit_parent=job.get("parent"),actual_author_fits=1,actual_posterior_calls=2,
            context_rows=len(context),query_rows=len(original),fit_elapsed=fit_elapsed,query_elapsed=query_elapsed,
            peak_bytes=execution.peak_bytes(),diagnostics=diagnostics,
            checks={name:True for name in WORKER_CHECKS},fitted_estimator_pickle_used=False,
            native_public_processing=True,competition_inputs_read=True,synthetic_only=False,
            artifact_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))

def supervise(job_path,output):
    verify();require_admission();execution=technical_gate();job=json.loads(job_path.read_text());validate_job(job)
    log,error=output.parent/"worker.stdout.log",output.parent/"worker.stderr.log"
    started=execution.clock_pair();peak,aborted=0,None
    with log.open("xb") as stdout,error.open("xb") as stderr:
        process=subprocess.Popen([str(execution.PYTHON),"-P","-S","-B",str(Path(__file__).resolve()),"--worker",
            "--job",str(job_path),"--output",str(output)],cwd=ROUND,
            env=execution.worker_environment(os.environ),stdout=stdout,stderr=stderr,start_new_session=True)
        while process.poll() is None:
            sample=subprocess.run(["/bin/ps","-o","rss=","-p",str(process.pid)],capture_output=True,text=True)
            if sample.returncode or not sample.stdout.strip():
                if process.poll() is None:aborted="rss_monitor_failed"
            else:peak=max(peak,int(sample.stdout.strip())*1024)
            marker=json.loads((output/"operation.json").read_text()) if (output/"operation.json").exists() else {}
            if peak>LIMIT_BYTES:aborted="rss_over_6GiB"
            if aborted is None:aborted=execution.clock_abort_reason(started,execution.clock_pair(),marker)
            if aborted:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                break
            time.sleep(.1)
        try:code=process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=process.wait(timeout=5)
    summary=json.loads((output/"summary.json").read_text()) if (output/"summary.json").exists() else {}
    ended=execution.clock_pair();elapsed=execution.clock_elapsed(started,ended)
    peak=max(peak,summary.get("peak_bytes",0))
    if aborted is None:aborted=execution.clock_abort_reason(started,ended,{})
    passed=(code==0 and aborted is None and summary.get("passed") is True and peak<=LIMIT_BYTES
        and all(execution.duration_ns_ok(v,900) for v in elapsed.values())
        and all(set(summary.get(key,{}))=={"utc_ns","monotonic_ns"}
            and all(execution.duration_ns_ok(v,300) for v in summary[key].values()) for key in ("fit_elapsed","query_elapsed")))
    if passed:execution.validate_operation_evidence(output,summary)
    record=dict(passed=passed,exit_code=code,aborted=aborted,peak_bytes=peak,started=started,ended=ended,elapsed=elapsed,
        summary_sha256=digest(output/"summary.json") if (output/"summary.json").exists() else None,
        stdout_sha256=digest(log),stderr_sha256=digest(error))
    write_json(output.parent/"supervision.json",record);verify()
    if not passed:raise NativeMethodFailure("Native GP numerical/resource gate failed; no retry or partial rescue")
    return record

def verify_replay_evidence():
    expected={f"{p}/block{b}/worker" for p in PROTOCOLS for b in range(1,4 if p=="buffered_100_seconds" else 6)}
    proofs=[];stage_hashes={};execution=technical_gate()
    for stage,seed in (("screen",42),("confirmation_seed43",43),("confirmation_seed44",44)):
        original,replay=ROUND/stage,ROUND/(stage+"_replay")
        for folder,is_replay in ((original,False),(replay,True)):
            manifest=audit(folder)
            if (manifest.get("stage")!=stage or manifest.get("seed")!=seed or manifest.get("replay") is not is_replay
                    or manifest.get("context_models")!=13 or manifest.get("actual_author_fits")!=13 or manifest.get("actual_posterior_calls")!=26
                    or manifest.get("independent_context_rebuild_count")!=(13 if is_replay else 0)):
                raise ValueError("Stage lacks thirteen-fit/twenty-six-posterior-call provenance")
            actual={str(p.relative_to(folder)) for p in folder.rglob("worker") if p.is_dir()}
            retained={str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p!=folder/"manifest.json"}
            if actual!=expected or set(manifest["cache_sha256"])!=retained:raise ValueError("Missing/repeated worker or incomplete stage cache")
            stage_hashes[folder.name]=digest(folder/"manifest.json")
        for relative in sorted(expected):
            parent,worker=original/relative,replay/relative
            protocol,block_name,_=Path(relative).parts;block=int(block_name[5:])
            for candidate,is_replay in ((parent,False),(worker,True)):
                job_dir=candidate.parent;job=json.loads((job_dir/"job.json").read_text());validate_job(job)
                if (job["seed"]!=seed or job["protocol"]!=protocol or job["block"]!=block or ("parent" in job)!=is_replay
                        or any(Path(job[name]).resolve()!=(job_dir/(name+".npy")).resolve() for name in PAYLOADS)):
                    raise ValueError("Worker payload/parent is not its own same-seed original/replay fold")
                summary=json.loads((candidate/"summary.json").read_text());supervision=json.loads((job_dir/"supervision.json").read_text())
                if (summary.get("passed") is not True or summary.get("seed")!=seed or summary.get("protocol")!=protocol
                        or summary.get("block")!=block or summary.get("fresh_context_replay") is not is_replay
                        or summary.get("actual_author_fits")!=1 or summary.get("actual_posterior_calls")!=2 or summary.get("fitted_estimator_pickle_used") is not False
                        or summary.get("fresh_refit_parent")!=(str(parent) if is_replay else None)
                        or set(summary.get("checks",{}))!=set(WORKER_CHECKS) or not all(summary["checks"].values())):
                    raise ValueError("Worker lacks actual GP-fit/fresh-context/full-state proof")
                if (supervision.get("passed") is not True or supervision.get("exit_code")!=0 or supervision.get("aborted") is not None
                        or supervision.get("summary_sha256")!=digest(candidate/"summary.json")
                        or supervision.get("peak_bytes",LIMIT_BYTES+1)>LIMIT_BYTES
                        or supervision.get("elapsed")!=execution.clock_elapsed(supervision["started"],supervision["ended"])
                        or set(supervision.get("elapsed",{}))!={"utc_ns","monotonic_ns"}
                        or any(not execution.duration_ns_ok(v,900) for v in supervision["elapsed"].values())
                        or any(set(summary.get(key,{}))!={"utc_ns","monotonic_ns"}
                            or any(not execution.duration_ns_ok(v,300) for v in summary[key].values())
                            for key in ("fit_elapsed","query_elapsed"))):
                    raise ValueError("Unbound, aborted, nonzero or over-budget replay worker")
                for kind in ("stdout","stderr"):
                    if supervision[kind+"_sha256"]!=digest(job_dir/("worker."+kind+".log")):raise ValueError("Replay worker log differs")
                artifacts=summary["artifact_sha256"];actual={p.name for p in candidate.iterdir() if p.is_file() and p.name!="summary.json"}
                if set(artifacts)!=actual or not REQUIRED_WORKER_ARTIFACTS<=actual:raise ValueError("Incomplete worker artifact closure")
                for name,sha in artifacts.items():
                    if Path(name).name!=name or digest(candidate/name)!=sha:raise ValueError("Replay artifact changed")
                execution.validate_operation_evidence(candidate,summary)
                validate_fit_evidence(execution,candidate,seed,summary["query_rows"])
            for name in tuple(n+".npy" for n in PAYLOADS)+("fitting.npz","context_provenance.json"):
                if (worker.parent/name).read_bytes()!=(parent.parent/name).read_bytes():raise ValueError("Five-payload/fitting/provenance replay differs: "+name)
            for name in SEMANTIC_ARTIFACTS:
                if (worker/name).read_bytes()!=(parent/name).read_bytes():raise ValueError("Full GP-state/posterior/probability replay differs: "+name)
            proofs.append(dict(seed=seed,stage=stage,worker=relative,parent=str(parent.relative_to(ROUND)),actual_author_fits=1,actual_posterior_calls=2,
                job_sha256=digest(worker.parent/"job.json"),summary_sha256=digest(worker/"summary.json"),
                supervision_sha256=digest(worker.parent/"supervision.json")))
    if len(proofs)!=39:raise ValueError("Exactly39 complete fresh-context GP workers/78 posterior calls are required")
    return dict(independent_process_fits=39,actual_author_fits=39,actual_posterior_calls=78,stage_manifest_sha256=stage_hashes,workers=proofs)

def pure_source_checks():
    for name in SOURCES:
        source=(ROOT/name).read_bytes();tree=ast.parse(source);compile(source,str(ROOT/name),"exec")
        for statement in tree.body:
            if isinstance(statement,(ast.Import,ast.ImportFrom)):
                names=[n.name.split(".")[0] for n in statement.names] if isinstance(statement,ast.Import) else [(statement.module or "").split(".")[0]]
                if set(names)&{"numpy","pandas","torch","scipy","sklearn","joblib"}:raise ValueError("Scientific top-level import")
    tests=load_module(ROOT/SOURCES[2],"gp_native_pure_tests").run_tests()
    if any(name in sys.modules for name in ("numpy","pandas","torch","scipy","sklearn","joblib")):
        raise ValueError("Scientific imports occurred in a source-only action")
    return tests

def verify_preparation_snapshot(plan):
    folder=ROUND/"before_actual_binding";path=folder/"snapshot.json"
    if digest(path)!=plan["timing_amendment"]["preserved_snapshot_sha256"]:
        raise ValueError("Initial source-only preparation snapshot changed")
    for relative,sha in json.loads(path.read_text())["files_sha256"].items():
        if digest(folder/relative)!=sha:raise ValueError("Preserved initial preparation byte changed: "+relative)

def verify_record_ceiling(plan):
    provenance=plan["historical_best_provenance"];path=ROOT/provenance["path"]
    if digest(path)!=provenance["sha256"]:raise ValueError("Observed higher buffered diagnostic changed")
    record=json.loads(path.read_text());value=record["scores"]["tabpfn_context_seed42"]["spell_f1_mean"]
    if (plan["historical_best_buffered"]!=136/175 or provenance["canonical_numerator"]!=136
            or provenance["canonical_denominator"]!=175 or abs(value-136/175)>1e-12
            or record.get("protocol")!="buffered_minutes" or record.get("scored_public_rows")!=7500
            or record.get("screen_complete") is not False or record.get("selected_changed") is not False
            or record.get("no_retry") is not True):
        raise ValueError("Strict ceiling must retain the complete observed136/175 buffered score without admitting its failed method")

def preflight():
    # Pending evidence cannot produce a source_preflight PASS or freeze.
    technical_gate();tests=pure_source_checks();plan=json.loads(PLAN.read_text())
    verify_preparation_snapshot(plan);verify_record_ceiling(plan)
    for name,sha in plan["metadata_sha256"].items():
        if digest(ROOT/name)!=sha:raise ValueError("Prepared metadata or clock evidence changed: "+name)
    for group in (plan["input_sha256"],json.loads((ROUND/"source_closure.json").read_text()),
            json.loads((ROUND/"protected_artifacts.json").read_text()),json.loads((ROUND/"head_runtime.json").read_text())):
        for name,sha in group.items():
            if digest(ROOT/name)!=sha:raise ValueError("Prepared inherited source/input/runtime changed: "+name)
    for name,target in json.loads((ROUND/"symlinks.json").read_text()).items():
        path=ROOT/name
        if not path.is_symlink() or str(path.readlink())!=target:raise ValueError("Prepared link changed")
    if not (ROOT/"experiments/EXPERIMENT_LEDGER.csv").read_bytes().startswith((ROUND/"preparation_ledger.csv").read_bytes()):
        raise ValueError("Preserved1621-row ledger prefix changed")
    write_json(ROUND/"source_preflight.json",dict(passed=True,source_sha256=source_identity(),plan_sha256=digest(PLAN),
        pure_tests=tests,actual_GPv2_all_four_stage_pass_verified=True,scientific_imports=False,native_arrays_loaded=False,
        technical_summary_sha256=digest(GATE_ROUND/"runtime_assessment/summary.json"),freeze_authorized=False))

def freeze():
    preflight();review=json.loads((ROUND/"independent_review.json").read_text())
    if (review.get("approved_for_freeze") is not True or review.get("source_sha256")!=source_identity()
            or review.get("plan_sha256")!=digest(PLAN) or review.get("unresolved_findings")!=0):
        raise ValueError("Independent root review must bind final GP native sources/plan")
    folder=ROUND/"protocol_freeze";folder.mkdir(exist_ok=False);(folder/"plan.json").write_bytes(PLAN.read_bytes())
    closure=json.loads((ROUND/"source_closure.json").read_text());closure.update(source_identity())
    for name in (PLAN,ROUND/"source_preflight.json",ROUND/"independent_review.json",ROUND/"source_preparation.json",ROUND/"PREPARATION.md",
            ROUND/"TIMING_AND_RECORD_AMENDMENT.md",ROUND/"source_amendment.json",
            ROUND/"source_closure.json",ROUND/"protected_artifacts.json",ROUND/"head_runtime.json",ROUND/"symlinks.json",ROUND/"preparation_ledger.csv"):
        closure[str(name.relative_to(ROOT))]=digest(name)
    for path in sorted((ROUND/"before_actual_binding").rglob("*")):
        if path.is_file():closure[str(path.relative_to(ROOT))]=digest(path)
    technical={str(GATE_SOURCE.relative_to(ROOT)):digest(GATE_SOURCE),str(GATE_PLAN.relative_to(ROOT)):digest(GATE_PLAN)}
    for branch in (GATE_ROUND/"protocol_freeze",GATE_ROUND/"runtime_assessment"):
        for path in sorted(branch.rglob("*")):
            if path.is_file():technical[str(path.relative_to(ROOT))]=digest(path)
    write_json(folder/"implementation.json",closure);write_json(folder/"technical_gate.json",technical)
    write_json(folder/"input_sha256.json",json.loads(PLAN.read_text())["input_sha256"])
    for name in ("protected_artifacts.json","head_runtime.json","symlinks.json"):(folder/name).write_bytes((ROUND/name).read_bytes())
    # Bind only the preserved immutable prefix; later R65/other appended rows may continue.
    (folder/"EXPERIMENT_LEDGER.csv").write_bytes((ROUND/"preparation_ledger.csv").read_bytes())
    write_json(folder/"freeze.json",dict(source_sha256=source_identity(),plan_sha256=digest(PLAN),
        actual_GPv2_all_four_stage_pass=True,native_arrays_loaded=False,model_calls=False,full_build_authorized=False))
    verify()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    for name in ("pure-tests","preflight","freeze","verify","worker"):actions.add_argument("--"+name,action="store_true")
    parser.add_argument("--job",type=Path);parser.add_argument("--output",type=Path);args=parser.parse_args()
    if args.pure_tests:print(json.dumps(dict(passed=True,tests=pure_source_checks(),technical_pass_claim=False)))
    elif args.preflight:preflight()
    elif args.freeze:freeze()
    elif args.verify:verify()
    else:
        if args.job is None or args.output is None or not args.output.resolve().is_relative_to(ROUND.resolve()):
            parser.error("Worker requires its own GP native job and new output")
        try:worker(args.job,args.output)
        except BaseException as error:
            if args.output.exists():write_json(args.output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                competition_inputs_read=_PAYLOAD_ACCESS_STARTED,native_payload_access_started=_PAYLOAD_ACCESS_STARTED,
                synthetic_only=False,no_partial_fold_rescue=True))
            raise

