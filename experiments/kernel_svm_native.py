"""R73 native RBF-SVC ownership preparation; pending R72 forbids data/model access.

Source-only actions use the standard library. The numerical method delegates
only reviewed frozen R72 primitives, never the invented worker or its admission.
"""
import argparse
import ast
from bisect import bisect_left
from contextlib import ExitStack
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_kernel_svm_native_v1"
PLAN=ROOT/"experiments/KERNEL_SVM_NATIVE_PLAN.json"
GATE_SOURCE=ROOT/"experiments/kernel_svm_runtime_gate.py"
GATE_ROUND=ROOT/"data/control_kernel_svm_runtime_v1"
GATE_PLAN=ROOT/"experiments/KERNEL_SVM_RUNTIME_PLAN.json"
ISOLATED=ROOT/"data/control_isolated_python_v1"
PUBLIC_START,PUBLIC_END=11185,18685
PROTOCOLS=("buffered_minutes","purged_spells","buffered_100_seconds")
SOURCES=("experiments/kernel_svm_native.py","experiments/research_kernel_svm_native.py","experiments/test_kernel_svm_native.py")
PAYLOADS=("context","target","original","swapped","base_frames")
LIMIT_BYTES=6*1024**3
_PAYLOAD_ACCESS_STARTED=False
TECHNICAL_SEMANTIC_ARTIFACTS=("recipe.json","inner_folds.json","fit_records.json","full_state.json","calibrator_state.json",
    "fit_0_state.json","fit_1_state.json","fit_2_state.json","fit_3_state.json",
    "inner_0_margin.npy","inner_1_margin.npy","inner_2_margin.npy","oof_margin.npy",
    "original_probability.npy","swapped_probability.npy","owner.npy")
SEMANTIC_ARTIFACTS=TECHNICAL_SEMANTIC_ARTIFACTS+(
    "repeat_first.npy","repeat_first_swapped.npy","classes.json","payload_immutability.json")
WORKER_CHECKS=("four_actual_author_fits","three_purged_inner_margin_calls","single_final_classifier",
    "full_fitted_state_immutable","raw_payloads_immutable","canonical_reload_exact","semantic_reload_exact",
    "repeat_byte_exact","query_labels_forbidden","no_network_attempts")
REQUIRED_WORKER_ARTIFACTS=set(SEMANTIC_ARTIFACTS)|{
    "fit_0.pkl","fit_1.pkl","fit_2.pkl","fit_3.pkl","calibrated_model.pkl","threadpools.json","operation.json",
    "calendar_operation.json","calendar_operations.jsonl"}

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

def calendar_operation(execution,output,name,callback):
    """Add UTC evidence around the unchanged frozen monotonic operation."""
    wall_started=time.time()
    marker=dict(name=name,active=True,wall_started=wall_started,limit_seconds=300,
        wall_started_utc=datetime.datetime.fromtimestamp(wall_started,datetime.timezone.utc).isoformat())
    path=output/"calendar_operation.json";atomic_json(path,marker)
    def finish(monotonic_seconds,status):
        wall_ended=time.time();elapsed=wall_ended-wall_started
        record=dict(marker,active=False,wall_ended=wall_ended,wall_elapsed_seconds=elapsed,
            wall_ended_utc=datetime.datetime.fromtimestamp(wall_ended,datetime.timezone.utc).isoformat(),
            monotonic_seconds=monotonic_seconds,status=status,calendar_guard_passed=duration_ok(elapsed,300))
        atomic_json(path,record)
        with (output/"calendar_operations.jsonl").open("a") as handle:
            handle.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
        return elapsed
    try:
        result,monotonic_seconds=execution.operation(output,name,300,callback)
    except BaseException:
        finish(None,"delegate_failed")
        raise
    elapsed=finish(monotonic_seconds,"complete")
    if not duration_ok(elapsed,300):raise NativeMethodFailure("UTC operation elapsed negative/nonfinite or at least300seconds")
    return result,monotonic_seconds,elapsed

def calendar_abort_reason(wall_now,wall_started,marker):
    elapsed=wall_now-wall_started
    if not math.isfinite(elapsed) or elapsed<0:return "wall_clock_negative_or_nonfinite"
    if elapsed>=900:return "wall_worker_over_900seconds"
    if marker.get("active"):
        elapsed=wall_now-marker["wall_started"]
        if not math.isfinite(elapsed) or elapsed<0:return "wall_operation_clock_negative_or_nonfinite"
        if elapsed>=300:return "wall_operation_over_300seconds"
    return None

def validate_calendar_evidence(folder,summary):
    events=[json.loads(line) for line in (folder/"calendar_operations.jsonl").read_text().splitlines()]
    expected=(("complete_three_inner_and_final_fit","wall_fit_seconds"),
        ("all_native_queries_repeat_symmetry_reload","wall_all_query_seconds"))
    if len(events)!=2:raise ValueError("Exactly two complete UTC operation records are required")
    for event,(name,field) in zip(events,expected):
        if (event.get("name")!=name or event.get("active") is not False or event.get("status")!="complete"
                or event.get("calendar_guard_passed") is not True
                or not duration_ok(event.get("wall_elapsed_seconds"),300)
                or not duration_ok(event.get("monotonic_seconds"),300)
                or event.get("wall_elapsed_seconds")!=event.get("wall_ended",0)-event.get("wall_started",0)
                or summary.get(field)!=event.get("wall_elapsed_seconds")):
            raise ValueError("UTC operation evidence is incomplete, inconsistent or outside300seconds")
    if json.loads((folder/"calendar_operation.json").read_text())!=events[-1]:
        raise ValueError("Retained UTC operation marker differs from the final complete event")

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
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
        if any(p["seed"]!=seed or p["model"]!=f"kernel_svm_context_seed{seed}" or
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
        raise ValueError("Actual R72 all-four-stage PASS with exact identities is required")
    if any(r.get("passed") is not True or r.get("exit_code")!=0 or r.get("aborted") is not None for r in records):
        raise ValueError("Every technical stage needs exit0 and no resource abort")
    return expected

def technical_calendar_header(audit,summary_sha):
    expected=[(size,stage) for size in ("moderate","full") for stage in ("first","rebuild")]
    if (audit.get("passed") is not True or audit.get("actual_all_four_stage_pass") is not True
            or audit.get("technical_summary_sha256")!=summary_sha or audit.get("actual_classifier_fits")!=16
            or audit.get("exact_fresh_rebuild_file_comparisons")!=32 or audit.get("native_processing") is not False
            or audit.get("source_or_dependencies_changed") is not False
            or [(r.get("size"),r.get("stage")) for r in audit.get("records",[])]!=expected):
        raise ValueError("Complete actual R72 calendar audit with exact four-stage identity is required")
    for record in audit["records"]:
        elapsed=record.get("utc_stage_upper_bound_seconds")
        if (record.get("passed") is not True or record.get("actual_classifier_fits")!=4
                or record.get("both_operation_utc_caps_proven_by_whole_stage_upper_bound") is not True
                or not duration_ok(elapsed,300) or elapsed<=0
                or elapsed!=record.get("supervision_complete_epoch",0)-record.get("stdout_birth_epoch",0)):
            raise ValueError("Every technical complete-worker UTC bound must be positive and below300seconds")
    return audit["records"]

def validate_four_fit_records(records,folds):
    if len(records)!=4 or len(folds)!=3:raise ValueError("Exactly three genuine inner fits plus final fit are required")
    for ordinal,record in enumerate(records):
        if (record.get("ordinal")!=ordinal or record.get("role")!=("final" if ordinal==3 else "inner")
                or not record.get("semantic_sha256") or not record.get("input_sha256")):
            raise ValueError("Actual classifier fit provenance is incomplete")
        learned=record.get("learned_state",{})
        if learned.get("fit_status")!=0 or learned.get("effective_probability") is not False or not learned.get("full_dict"):
            raise ValueError("Actual fit lacks successful complete learned-state proof")
        if ordinal<3 and (record.get("train_indices")!=folds[ordinal]["train"]
                or record.get("validation_indices")!=folds[ordinal]["validation"]):
            raise ValueError("Actual inner fit does not match its purged role indices")
        if ordinal==3 and (record.get("validation_indices") is not None
                or record.get("train_indices")!=list(range(sum(len(f["validation"]) for f in folds)))):
            raise ValueError("Fourth actual classifier must fit the complete permitted context")

def technical_gate(require_pass=True):
    plan=json.loads(PLAN.read_text());prerequisite=plan["technical_prerequisite"]
    # Pending null is a hard boundary, before scientific imports/native payloads.
    if require_pass and (not prerequisite.get("actual_all_four_stage_summary_sha256")
            or prerequisite.get("status")!="ACTUAL_ALL_FOUR_STAGE_PASS_BOUND"):
        raise ValueError("R72 actual all-four-stage evidence is pending/null; native admission is forbidden")
    if digest(GATE_SOURCE)!=prerequisite["source_sha256"] or digest(GATE_PLAN)!=prerequisite["plan_sha256"]:
        raise ValueError("Reviewed frozen R72 source or plan differs")
    if digest(GATE_ROUND/"protocol_freeze/freeze.json")!=prerequisite["source_freeze_sha256"]:
        raise ValueError("Reviewed R72 source freeze identity differs")
    execution=load_module(GATE_SOURCE,"r73_frozen_kernel_numerics")
    execution.verify()
    if not require_pass:return execution
    assessment=GATE_ROUND/"runtime_assessment";path=assessment/"summary.json"
    if digest(path)!=prerequisite["actual_all_four_stage_summary_sha256"]:
        raise ValueError("Plan-bound actual R72 top-level evidence differs")
    calendar_path=ROOT/prerequisite["actual_calendar_audit_path"]
    if digest(calendar_path)!=prerequisite["actual_calendar_audit_sha256"]:
        raise ValueError("Actual R72 calendar audit changed")
    calendar_audit=json.loads(calendar_path.read_text())
    calendar_records=technical_calendar_header(calendar_audit,prerequisite["actual_all_four_stage_summary_sha256"])
    independent_path=ROOT/prerequisite["actual_independent_review_path"]
    if digest(independent_path)!=prerequisite["actual_independent_review_sha256"]:
        raise ValueError("Actual R72 independent evidence review changed")
    independent=json.loads(independent_path.read_text())
    if (independent.get("passed") is not True or independent.get("actual_all_four_stage_pass") is not True
            or independent.get("technical_summary_sha256")!=prerequisite["actual_all_four_stage_summary_sha256"]
            or independent.get("actual_classifier_fits")!=16 or independent.get("semantic_refit_comparisons")!=32
            or independent.get("source_or_dependencies_changed") is not False or independent.get("native_processing") is not False
            or independent.get("memory_measures",{}).get("conservative_observed_max_bytes",LIMIT_BYTES+1)>LIMIT_BYTES):
        raise ValueError("Actual independent R72 technical review does not establish the complete bounded result")
    if calendar_audit.get("resource_admission_sha256")!=digest(GATE_ROUND/"resource_admission.json"):
        raise ValueError("Actual technical calendar audit does not bind the retained resource admission")
    if not prerequisite["metadata_sha256"]:raise ValueError("Complete actual R72 technical evidence must be bound by root")
    current_files={str(p.relative_to(ROOT)) for p in GATE_ROUND.rglob("*") if p.is_file()}
    if not current_files<=set(prerequisite["metadata_sha256"]):
        raise ValueError("Every retained actual R72 round file must be included in its evidence binding")
    for name,sha in prerequisite["metadata_sha256"].items():
        if digest(ROOT/name)!=sha:raise ValueError("Bound technical evidence changed: "+name)
    summary=json.loads(path.read_text());expected=technical_header(summary,digest(GATE_SOURCE),digest(GATE_PLAN))
    for (size,stage),supervision in zip(expected,summary["records"]):
        folder=assessment/size/stage;stage_path=folder/"summary.json";record=json.loads(stage_path.read_text())
        calendar_record=next(r for r in calendar_records if (r["size"],r["stage"])==(size,stage))
        if (calendar_record.get("summary_sha256")!=digest(stage_path)
                or calendar_record.get("supervision_sha256")!=digest(assessment/size/(stage+".supervision.json"))):
            raise ValueError("Technical UTC proof is not bound to its actual numerical stage/supervision")
        checks=record.get("checks",{})
        if (record.get("passed") is not True or record.get("size")!=size or record.get("stage")!=stage
                or set(checks)!={"four_actual_author_fits","three_purged_inner_margin_calls","single_final_classifier",
                    "finite_normalized_probabilities","exact_first_repeat","exact_team_swap","raw_arrays_immutable",
                    "full_fitted_state_immutable","semantic_reload_exact","canonical_reload_exact","independent_refit_exact"}
                or any(value is not True for key,value in checks.items() if key!="independent_refit_exact")
                or checks["independent_refit_exact"] is not (stage=="rebuild")):
            raise ValueError("Technical stage lacks its complete numerical/replay conjunction")
        if (json.loads((assessment/size/(stage+".supervision.json")).read_text())!=supervision
                or supervision.get("summary_sha256")!=digest(stage_path)
                or supervision.get("wall_seconds",900)>=900 or supervision.get("largest_sampled_rss_bytes",LIMIT_BYTES+1)>LIMIT_BYTES
                or record.get("fit_seconds",300)>=300 or record.get("all_query_seconds",300)>=300
                or record.get("peak_bytes",LIMIT_BYTES+1)>LIMIT_BYTES):
            raise ValueError("Technical stage resources or supervision binding differ")
        for kind in ("stdout","stderr"):
            if digest(assessment/size/(stage+"."+kind+".log"))!=supervision[kind+"_sha256"]:
                raise ValueError("Technical log changed")
        artifacts=record.get("artifact_sha256",{})
        actual={p.name for p in folder.iterdir() if p.is_file() and p.name not in ("summary.json","operation.json")}
        if set(artifacts)!=actual or set(record.get("exact_artifacts",[]))!=set(TECHNICAL_SEMANTIC_ARTIFACTS):
            raise ValueError("Technical artifact set is incomplete")
        for name,sha in artifacts.items():
            if Path(name).name!=name or digest(folder/name)!=sha:raise ValueError("Technical artifact changed")
        operation=json.loads((folder/"operation.json").read_text())
        if operation.get("active") is not False or operation.get("seconds",300)>=300:
            raise ValueError("Technical timed operation is incomplete")
        records=json.loads((folder/"fit_records.json").read_text());folds=json.loads((folder/"inner_folds.json").read_text())
        validate_four_fit_records(records,folds)
        for ordinal in range(4):
            if json.loads((folder/f"fit_{ordinal}_state.json").read_text())!=records[ordinal]:
                raise ValueError("Technical per-fit state/provenance differs")
    for size in ("moderate","full"):
        first,rebuild=[assessment/size/stage for stage in ("first","rebuild")]
        for name in TECHNICAL_SEMANTIC_ARTIFACTS:
            if (first/name).read_bytes()!=(rebuild/name).read_bytes():
                raise ValueError("Complete independent technical fresh refit differs: "+name)
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
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes():raise ValueError("R73 plan changed")
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
            or approval.get("predecessor_r65_closed_or_passed") is not True
            or approval.get("freeze_sha256")!=digest(ROUND/"protocol_freeze/freeze.json")):
        raise ValueError("Separate root admission and free model resources are required")

class NativeMethodFailure(RuntimeError):pass

def native_query_checks(execution,model,data,payload,before,output):
    """Unchanged R72 checks, plus retained native repeat/classes/raw-state proof."""
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
    write_json(output/"classes.json",dict(classes=model.classes_.tolist(),meaning=["H","A"]))
    write_json(output/"payload_immutability.json",dict(before=before,after=after,exact=True))
    return diagnostics

def worker(job_path,output):
    global _PAYLOAD_ACCESS_STARTED
    verify();require_admission();execution=technical_gate()
    job=json.loads(job_path.read_text());validate_job(job)
    output.mkdir(parents=True,exist_ok=False)
    denied=[]
    def deny(*args,**kwargs):denied.append(True);raise RuntimeError("Network forbidden in native R73 worker")
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
        folds=execution.chronological_splits(base_frames.tolist())
        write_json(output/"inner_folds.json",folds);write_json(output/"threadpools.json",pools)
        recipe=dict(head_seed=job["seed"],protocol=job["protocol"],block=job["block"],
            context_shape=list(context.shape),query_shape=list(original.shape),class_order=["H","A"],
            payloads=before,source_contract=execution.source_contract(),query_batch=512,
            raw_context_refit=True,foreign_estimator_pickle_used=False,local_pickle_reload_audit=True,
            actual_classifier_fits=4,inner_purge_frames=250,ensemble=False,sigmoid=True)
        write_json(output/"recipe.json",recipe)
        data=dict(raw_context=context,target=target,raw_query=original,permutation=permutation)
        make=execution.make_estimator
        with patch.object(execution,"make_estimator",lambda:make().set_params(svc__random_state=job["seed"])):
            model,fit_seconds,wall_fit_seconds=calendar_operation(execution,output,"complete_three_inner_and_final_fit",
                lambda:execution.fit_with_author_recording(data,folds,output))
        fitted=execution.state_digest(model)
        diagnostics,query_seconds,wall_query_seconds=calendar_operation(execution,output,"all_native_queries_repeat_symmetry_reload",
            lambda:native_query_checks(execution,model,data,payload,before,output))
        if fitted!=execution.state_digest(model) or before!={name:execution.array_record(value) for name,value in payload.items()}:
            raise ValueError("Complete native payload or learned state changed")
        records=json.loads((output/"fit_records.json").read_text());validate_four_fit_records(records,folds)
        if "parent" in job:
            parent=Path(job["parent"])
            for name in SEMANTIC_ARTIFACTS:
                if (output/name).read_bytes()!=(parent/name).read_bytes():raise ValueError("Fresh raw-context refit differs: "+name)
        if denied:raise ValueError("Network access was attempted")
        verify()
        write_json(output/"summary.json",dict(passed=True,seed=job["seed"],protocol=job["protocol"],block=job["block"],
            fresh_context_replay="parent" in job,fresh_refit_parent=job.get("parent"),actual_classifier_fits=4,
            context_rows=len(context),query_rows=len(original),fit_seconds=fit_seconds,all_query_seconds=query_seconds,
            wall_fit_seconds=wall_fit_seconds,wall_all_query_seconds=wall_query_seconds,
            peak_bytes=execution.peak_bytes(),diagnostics=diagnostics,
            checks={name:True for name in WORKER_CHECKS},fitted_estimator_pickle_used=False,
            native_public_processing=True,competition_inputs_read=True,synthetic_only=False,
            artifact_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}))

def supervise(job_path,output):
    verify();require_admission();execution=technical_gate();job=json.loads(job_path.read_text());validate_job(job)
    log,error=output.parent/"worker.stdout.log",output.parent/"worker.stderr.log"
    started,wall_started,peak,aborted=time.monotonic(),time.time(),0,None
    with log.open("xb") as stdout,error.open("xb") as stderr:
        process=subprocess.Popen([str(execution.PYTHON),"-P","-S","-B",str(Path(__file__).resolve()),"--worker",
            "--job",str(job_path),"--output",str(output)],cwd=ROUND,
            env=execution.worker_environment(os.environ),stdout=stdout,stderr=stderr,start_new_session=True)
        while process.poll() is None:
            sample=subprocess.run(["/bin/ps","-o","rss=","-p",str(process.pid)],capture_output=True,text=True)
            if sample.returncode or not sample.stdout.strip():
                if process.poll() is None:aborted="rss_monitor_failed"
            else:peak=max(peak,int(sample.stdout.strip())*1024)
            status=json.loads((output/"operation.json").read_text()) if (output/"operation.json").exists() else {}
            calendar=json.loads((output/"calendar_operation.json").read_text()) if (output/"calendar_operation.json").exists() else {}
            if peak>LIMIT_BYTES:aborted="rss_over_6GiB"
            elif time.monotonic()-started>=900:aborted="worker_over_900seconds"
            elif status.get("active") and time.monotonic()-status["started"]>=300:aborted="operation_over_300seconds"
            if aborted is None:aborted=calendar_abort_reason(time.time(),wall_started,calendar)
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
    elapsed=time.monotonic()-started;wall_ended=time.time();wall_elapsed=wall_ended-wall_started
    peak=max(peak,summary.get("peak_bytes",0))
    if aborted is None:aborted=calendar_abort_reason(wall_ended,wall_started,{})
    passed=(code==0 and aborted is None and summary.get("passed") is True and elapsed<900 and peak<=LIMIT_BYTES
        and summary.get("fit_seconds",300)<300 and summary.get("all_query_seconds",300)<300
        and duration_ok(wall_elapsed,900) and duration_ok(summary.get("wall_fit_seconds"),300)
        and duration_ok(summary.get("wall_all_query_seconds"),300))
    record=dict(passed=passed,exit_code=code,aborted=aborted,peak_bytes=peak,elapsed_seconds=elapsed,
        wall_started=wall_started,wall_ended=wall_ended,wall_elapsed_seconds=wall_elapsed,
        summary_sha256=digest(output/"summary.json") if (output/"summary.json").exists() else None,
        stdout_sha256=digest(log),stderr_sha256=digest(error))
    write_json(output.parent/"supervision.json",record);verify()
    if not passed:raise NativeMethodFailure("Native SVC numerical/resource gate failed; no retry or partial rescue")
    return record

def verify_replay_evidence():
    expected={f"{p}/block{b}/worker" for p in PROTOCOLS for b in range(1,4 if p=="buffered_100_seconds" else 6)}
    proofs=[];stage_hashes={}
    for stage,seed in (("screen",42),("confirmation_seed43",43),("confirmation_seed44",44)):
        original,replay=ROUND/stage,ROUND/(stage+"_replay")
        for folder,is_replay in ((original,False),(replay,True)):
            manifest=audit(folder)
            if (manifest.get("stage")!=stage or manifest.get("seed")!=seed or manifest.get("replay") is not is_replay
                    or manifest.get("context_models")!=13 or manifest.get("actual_classifier_fits")!=52
                    or manifest.get("independent_context_rebuild_count")!=(13 if is_replay else 0)):
                raise ValueError("Stage lacks explicit thirteen-worker/fifty-two-fit provenance")
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
                        or summary.get("actual_classifier_fits")!=4 or summary.get("fitted_estimator_pickle_used") is not False
                        or summary.get("fresh_refit_parent")!=(str(parent) if is_replay else None)
                        or set(summary.get("checks",{}))!=set(WORKER_CHECKS) or not all(summary["checks"].values())):
                    raise ValueError("Worker lacks actual four-fit/fresh-context/full-state proof")
                if (supervision.get("passed") is not True or supervision.get("exit_code")!=0 or supervision.get("aborted") is not None
                        or supervision.get("summary_sha256")!=digest(candidate/"summary.json")
                        or supervision.get("elapsed_seconds",900)>=900 or supervision.get("peak_bytes",LIMIT_BYTES+1)>LIMIT_BYTES
                        or summary.get("fit_seconds",300)>=300 or summary.get("all_query_seconds",300)>=300
                        or not duration_ok(supervision.get("wall_elapsed_seconds"),900)
                        or supervision.get("wall_elapsed_seconds")!=supervision.get("wall_ended",0)-supervision.get("wall_started",0)
                        or not duration_ok(summary.get("wall_fit_seconds"),300)
                        or not duration_ok(summary.get("wall_all_query_seconds"),300)):
                    raise ValueError("Unbound, aborted, nonzero or over-budget replay worker")
                for kind in ("stdout","stderr"):
                    if supervision[kind+"_sha256"]!=digest(job_dir/("worker."+kind+".log")):raise ValueError("Replay worker log differs")
                artifacts=summary["artifact_sha256"];actual={p.name for p in candidate.iterdir() if p.is_file() and p.name!="summary.json"}
                if set(artifacts)!=actual or not REQUIRED_WORKER_ARTIFACTS<=actual:raise ValueError("Incomplete worker artifact closure")
                for name,sha in artifacts.items():
                    if Path(name).name!=name or digest(candidate/name)!=sha:raise ValueError("Replay artifact changed")
                validate_calendar_evidence(candidate,summary)
                fits=json.loads((candidate/"fit_records.json").read_text());folds=json.loads((candidate/"inner_folds.json").read_text())
                validate_four_fit_records(fits,folds)
                for ordinal in range(4):
                    if json.loads((candidate/f"fit_{ordinal}_state.json").read_text())!=fits[ordinal]:raise ValueError("Actual per-fit state differs")
            for name in tuple(n+".npy" for n in PAYLOADS)+("fitting.npz","context_provenance.json"):
                if (worker.parent/name).read_bytes()!=(parent.parent/name).read_bytes():raise ValueError("Five-payload/fitting/provenance replay differs: "+name)
            for name in SEMANTIC_ARTIFACTS:
                if (worker/name).read_bytes()!=(parent/name).read_bytes():raise ValueError("Full learned/calibrator/OOF/probability replay differs: "+name)
            proofs.append(dict(seed=seed,stage=stage,worker=relative,parent=str(parent.relative_to(ROUND)),actual_classifier_fits=4,
                job_sha256=digest(worker.parent/"job.json"),summary_sha256=digest(worker/"summary.json"),
                supervision_sha256=digest(worker.parent/"supervision.json")))
    if len(proofs)!=39:raise ValueError("Exactly39 complete fresh-context workers/156 actual classifiers are required")
    return dict(independent_process_fits=39,actual_classifier_fits=156,stage_manifest_sha256=stage_hashes,workers=proofs)

def pure_source_checks():
    for name in SOURCES:
        source=(ROOT/name).read_bytes();tree=ast.parse(source);compile(source,str(ROOT/name),"exec")
        for statement in tree.body:
            if isinstance(statement,(ast.Import,ast.ImportFrom)):
                names=[n.name.split(".")[0] for n in statement.names] if isinstance(statement,ast.Import) else [(statement.module or "").split(".")[0]]
                if set(names)&{"numpy","pandas","torch","scipy","sklearn","joblib"}:raise ValueError("Scientific top-level import")
    tests=load_module(ROOT/SOURCES[2],"r73_pure_tests").run_tests()
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
        raise ValueError("Preserved1612-row ledger prefix changed")
    write_json(ROUND/"source_preflight.json",dict(passed=True,source_sha256=source_identity(),plan_sha256=digest(PLAN),
        pure_tests=tests,actual_R72_all_four_stage_pass_verified=True,scientific_imports=False,native_arrays_loaded=False,
        technical_summary_sha256=digest(GATE_ROUND/"runtime_assessment/summary.json"),freeze_authorized=False))

def freeze():
    preflight();review=json.loads((ROUND/"independent_review.json").read_text())
    if (review.get("approved_for_freeze") is not True or review.get("source_sha256")!=source_identity()
            or review.get("plan_sha256")!=digest(PLAN) or review.get("unresolved_findings")!=0):
        raise ValueError("Independent root review must bind final R73 sources/plan")
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
        actual_R72_all_four_stage_pass=True,native_arrays_loaded=False,model_calls=False,full_build_authorized=False))
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
            parser.error("Worker requires its own R73 job and new output")
        try:worker(args.job,args.output)
        except BaseException as error:
            if args.output.exists():write_json(args.output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                competition_inputs_read=_PAYLOAD_ACCESS_STARTED,native_payload_access_started=_PAYLOAD_ACCESS_STARTED,
                synthetic_only=False,no_partial_fold_rescue=True))
            raise

