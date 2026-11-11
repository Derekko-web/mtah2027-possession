"""R65: prospective native conditional H/A TabPFN worker and protocol guards.
No scientific imports or native-array reads occur during preparation/preflight.
"""
import argparse
import ast
from bisect import bisect_left
from contextlib import ExitStack
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import urllib.request
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_tabpfn_native_v1"
PLAN=ROOT/"experiments/TABPFN_NATIVE_PLAN.json"
GATE_SOURCE=ROOT/"experiments/tabpfn_mps_isolated_runtime_gate_v2.py"
GATE_ROUND=ROOT/"data/control_tabpfn_mps_isolated_runtime_v2"
GATE_PLAN=ROOT/"experiments/TABPFN_MPS_ISOLATED_RUNTIME_PLAN_V2.json"
GATE_SHA256="16804adff92ba4308dddb693faa53f1763fb8cad4686be18487feaa9a4a67596"
GATE_PLAN_SHA256="c54e006a10909c57a341c235531bd826f269ab2c0d0ab830bd1f5ccb1a896f73"
ISOLATED=ROOT/"data/control_isolated_python_v1"
PUBLIC_START,PUBLIC_END=11185,18685
PROTOCOLS=("buffered_minutes","purged_spells","buffered_100_seconds")
SOURCES=("experiments/tabpfn_native.py","experiments/research_tabpfn_native.py","experiments/test_tabpfn_native.py")
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

def technical_gate(require_pass=True):
    """R67 is historical; the new isolated R70v2 all-four-stage pass is required."""
    if digest(GATE_SOURCE)!=GATE_SHA256 or digest(GATE_PLAN)!=GATE_PLAN_SHA256:
        raise ValueError("Frozen R70v2 source/plan differs")
    module=load_module(GATE_SOURCE,"r65_frozen_technical_gate")
    module.verify()
    if require_pass:
        folder=GATE_ROUND/"runtime_assessment"
        prerequisite=json.loads(PLAN.read_text())["technical_prerequisite"]
        if digest(folder/"summary.json")!=prerequisite["actual_all_four_stage_summary_sha256"]:
            raise ValueError("Plan-bound actual R70v2 all-four-stage evidence changed")
        for name,sha in prerequisite["metadata_sha256"].items():
            if digest(ROOT/name)!=sha:raise ValueError("Plan-bound R70v2 metadata changed: "+name)
        summary=json.loads((folder/"summary.json").read_text())
        expected=[(s,t) for s in ("moderate","full") for t in ("first","rebuild")]
        records=summary.get("records",[])
        if (summary.get("passed") is not True or summary.get("implementation_sha256")!=GATE_SHA256
                or summary.get("plan_sha256")!=GATE_PLAN_SHA256
                or [(r.get("size"),r.get("stage")) for r in records]!=expected):
            raise ValueError("R70v2 all four original technical stages have not passed")
        for (size,stage),supervision in zip(expected,records):
            path=folder/size/stage/"summary.json";stage_record=json.loads(path.read_text())
            if (stage_record.get("passed") is not True or not all(stage_record["checks"].values())
                    or stage_record.get("size")!=size or stage_record.get("stage")!=stage
                    or supervision.get("passed") is not True or supervision.get("exit_code")!=0
                    or supervision.get("aborted") is not None or supervision.get("monitor_failed") is not False):
                raise ValueError("Incomplete R70v2 stage exactness/resource conjunction")
            if json.loads((folder/size/(stage+".supervision.json")).read_text())!=supervision:
                raise ValueError("R70v2 top-level supervision differs")
            if (supervision["elapsed_wall_seconds"]>900 or supervision["peak_conservative_owned_rss_plus_driver_bytes"]>LIMIT_BYTES
                    or stage_record["fit_seconds"]>300 or stage_record["memory"]["peak_conservative_combined_bytes"]>LIMIT_BYTES):
                raise ValueError("R70v2 recorded resource bound failed")
            for kind in ("stdout","stderr"):
                if digest(folder/size/(stage+"."+kind+".log"))!=supervision[kind+"_sha256"]:
                    raise ValueError("R70v2 retained log changed")
            for name,sha in stage_record["artifact_sha256"].items():
                if Path(name).name!=name or digest(path.parent/name)!=sha:raise ValueError("Technical stage artifact changed")
            operations=[json.loads(line) for line in (path.parent/"operations.jsonl").read_text().splitlines()]
            if not operations or any(op["active"] or op["elapsed"]>300 for op in operations):
                raise ValueError("R70v2 timed operation did not complete within bound")
            predictions=[op["name"] for op in operations if module.previous().is_prediction(op["name"])]
            flushes=[json.loads(line) for line in (path.parent/"allocator_flush.jsonl").read_text().splitlines()]
            if [event["operation"] for event in flushes]!=predictions:
                raise ValueError("R70v2 allocator flush missing for a prediction")
            if any(event.get("status")!="complete" or event.get("live_tensor_bytes_unchanged") is not True
                    or event["before"]["mps_tensor_bytes"]!=event["after"]["mps_tensor_bytes"]
                    or event.get("empty_cache_calls")!=1 or event.get("synchronization_calls")!=1
                    or event.get("model_context_or_rng_mutation_apis_called") is not False for event in flushes):
                raise ValueError("R70v2 allocator flush altered live state or failed")
        for size in ("moderate","full"):
            first,rebuild=[folder/size/stage for stage in ("first","rebuild")]
            names={p.name for p in first.glob("*.npy")}
            if names!={p.name for p in rebuild.glob("*.npy")} or len(names)!=(14 if size=="moderate" else 12):
                raise ValueError("R70v2 complete retained array set differs")
            for name in names|{"recipe.json","checkpoint_fingerprint.json","preprocessing_fingerprint.json","validation_metadata_initialization.json"}:
                if (first/name).read_bytes()!=(rebuild/name).read_bytes():raise ValueError("R70v2 independent rebuild bytes differ: "+name)
        hardware=json.loads((GATE_ROUND/"hardware_probe.json").read_text())
        if hardware["mps_allocator_limit_bytes"]!=4*1024**3 or hardware["reserved_owned_rss_budget_bytes"]!=2*1024**3:
            raise ValueError("Prospective balanced 4GiB/2GiB allocation differs")
        if module.core().LIMIT_BYTES!=LIMIT_BYTES:raise ValueError("Combined 6GiB gate differs")
    return module

def source_identity():return {name:digest(ROOT/name) for name in SOURCES}

def require_private_runtime(prefix,base_prefix,executable,controller):
    expected=ISOLATED/("head_runtime" if controller else "python")
    if (Path(prefix).resolve()!=expected.resolve() or Path(base_prefix).resolve()!=(ISOLATED/"python").resolve()
            or Path(executable).resolve()!=(ISOLATED/"python/bin/python3.12").resolve()):
        raise ValueError("Expected the bound private controller or model runtime")

def validate_job(job):
    required={"context","target","original","swapped","seed","protocol","block","input_sha256"}
    if set(job)-required-{"parent"} or not required<=set(job):raise ValueError("Unexpected job fields; query labels forbidden")
    if job["seed"] not in (42,43,44) or job["protocol"] not in PROTOCOLS or type(job["block"]) is not int:
        raise ValueError("Unexpected fixed seed/protocol/fold")
    paths={str(Path(job[name]).resolve()) for name in ("context","target","original","swapped")}
    if set(job["input_sha256"])!=paths:raise ValueError("Payload provenance must bind exactly four arrays")
    for name in paths:
        if not Path(name).is_relative_to(ROUND.resolve()):raise ValueError("Payload path escapes native round")
        if digest(name)!=job["input_sha256"][name]:raise ValueError("Native payload bytes changed")
    if "parent" in job and not Path(job["parent"]).resolve().is_relative_to(ROUND.resolve()):
        raise ValueError("Replay parent escapes native round")
    if "parent" in job:
        stage="screen" if job["seed"]==42 else f"confirmation_seed{job['seed']}"
        expected=ROUND/stage/job["protocol"]/f"block{job['block']}"/"worker"
        if Path(job["parent"]).resolve()!=expected.resolve():
            raise ValueError("Replay parent must be the original same-seed/stage/fold worker")

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

def context_record(inputs,labels,train,validation):
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from rich_features import swap_teams
    if inputs.shape[1]!=166:raise ValueError("Expected immutable original102 plus R37 fixed64")
    frames,observed=context_frames(train,validation,labels)
    values=inputs.iloc[frames]
    augmented=pd.concat([values,swap_teams(values)],ignore_index=True)
    imputer=SimpleImputer(strategy="median",keep_empty_features=True,add_indicator=False)
    context=imputer.fit_transform(augmented).astype(np.float32)
    target=(labels[frames]=="A").astype(np.int64);target=np.r_[target,1-target]
    if context.shape!=(2*len(frames),166) or not np.isfinite(context).all() or not np.array_equal(np.bincount(target),[len(frames)]*2):
        raise ValueError("Paired context/finite fixed-width imputation differs")
    return context,target,imputer,dict(split_train=np.asarray(train),validation=np.asarray(validation),
        context_frames=np.asarray(frames),observation_eligible=np.asarray(observed),target=target,
        weight=np.ones(len(target)),median=imputer.statistics_,footprint_radius=np.asarray(79))

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
        if any(p["seed"]!=seed or p["model"]!=f"tabpfn_context_seed{seed}" or
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
    """Prove39 fresh same-seed/fold context fits from workers and bytes, not names."""
    stages=(("screen",42),("confirmation_seed43",43),("confirmation_seed44",44))
    expected={f"{protocol}/block{block}/worker" for protocol in PROTOCOLS
        for block in range(1,4 if protocol=="buffered_100_seconds" else 6)}
    compared=("checkpoint_fingerprint.json","preprocessing_fingerprint.json","recipe.json",
        "original_probability.npy","swapped_probability.npy","owner.npy","repeat_first.npy")
    def record(path):
        if not path.is_file():raise ValueError("Missing retained replay proof: "+str(path))
        return json.loads(path.read_text())
    proofs=[];stage_hashes={}
    for stage,seed in stages:
        original=ROUND/stage;replay=ROUND/(stage+"_replay")
        for folder,is_replay in ((original,False),(replay,True)):
            manifest=record(folder/"manifest.json");audit(folder)
            if (manifest.get("stage")!=stage or manifest.get("seed")!=seed or manifest.get("replay") is not is_replay
                    or manifest.get("context_models")!=13
                    or manifest.get("independent_context_rebuild_count")!=(13 if is_replay else 0)):
                raise ValueError("Stage lacks explicit correct replay/seed/count provenance")
            retained={str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p!=folder/"manifest.json"}
            if set(manifest["cache_sha256"])!=retained:raise ValueError("Stage cache does not bind every retained file")
            actual={str(p.relative_to(folder)) for p in folder.rglob("worker") if p.is_dir()}
            if actual!=expected:raise ValueError("Expected exactly thirteen distinct original5/5/3 workers")
            stage_hashes[folder.name]=digest(folder/"manifest.json")
        for relative in sorted(expected):
            worker=replay/relative;parent=original/relative
            protocol,block_name,_=Path(relative).parts;block=int(block_name[5:])
            for candidate,is_replay in ((parent,False),(worker,True)):
                job_dir=candidate.parent;job=record(job_dir/"job.json");validate_job(job)
                if (job["seed"]!=seed or job["protocol"]!=protocol or job["block"]!=block
                        or ("parent" in job)!=is_replay
                        or (is_replay and Path(job["parent"]).resolve()!=parent.resolve())):
                    raise ValueError("Worker is not bound to its original seed/stage/fold parent")
                for name in ("context","target","original","swapped"):
                    if Path(job[name]).resolve()!=(job_dir/(name+".npy")).resolve():
                        raise ValueError("Worker payload is not its own exact fold payload")
                summary=record(candidate/"summary.json");supervision=record(job_dir/"supervision.json")
                checks=("complete_raw_preprocessing_immutable","checkpoint_exact","resolved_model_configuration_immutable","repeat_byte_exact")
                if (summary.get("passed") is not True or summary.get("seed")!=seed
                        or summary.get("protocol")!=protocol or summary.get("block")!=block
                        or summary.get("fresh_context_replay") is not is_replay
                        or any(summary.get(name) is not True for name in checks)):
                    raise ValueError("Worker lacks successful fresh-context replay and full-state proof")
                if (supervision.get("passed") is not True or supervision.get("exit_code")!=0 or supervision.get("aborted") is not None
                        or supervision.get("summary_sha256")!=digest(candidate/"summary.json")):
                    raise ValueError("Worker supervision is missing, aborted, nonzero or unbound")
                for kind in ("stdout","stderr"):
                    if supervision.get(kind+"_sha256")!=digest(job_dir/("worker."+kind+".log")):
                        raise ValueError("Worker supervision log binding differs")
                artifacts=summary["artifact_sha256"]
                actual={p.name for p in candidate.iterdir() if p.is_file() and p.name not in ("summary.json","memory.json","memory.tmp")}
                if set(artifacts)!=actual or not set(compared)<=set(artifacts):
                    raise ValueError("Worker artifacts are incomplete or not fully bound")
                for name,sha in artifacts.items():
                    if Path(name).name!=name or digest(candidate/name)!=sha:raise ValueError("Worker artifact hash changed")
            for name in ("context.npy","target.npy","original.npy","swapped.npy","fitting.npz","context_provenance.json"):
                if (worker.parent/name).read_bytes()!=(parent.parent/name).read_bytes():
                    raise ValueError("Final replay context/target/query/provenance bytes differ: "+name)
            for name in compared:
                if (worker/name).read_bytes()!=(parent/name).read_bytes():
                    raise ValueError("Final original/replay preprocessing/checkpoint/recipe/probability bytes differ: "+name)
            proofs.append(dict(seed=seed,stage=stage,worker=relative,parent=str(parent.relative_to(ROUND)),
                job_sha256=digest(worker.parent/"job.json"),summary_sha256=digest(worker/"summary.json"),
                supervision_sha256=digest(worker.parent/"supervision.json")))
    if len(proofs)!=39:raise ValueError("Exactly39 proven independent context rebuilds are required")
    return dict(independent_process_fits=len(proofs),stage_manifest_sha256=stage_hashes,workers=proofs)

def verify():
    """Global provenance failure always aborts; never rescue a partial native method."""
    folder=ROUND/"protocol_freeze"
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes():raise ValueError("R65 plan changed")
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

def native_core():
    module=technical_gate();execution=module.core()
    execution.ROUND=ROUND;execution.PLAN=PLAN;execution.verify=verify
    # Numerical operations, MPS dispatch, full fingerprints and one-time R63
    # initialization are reused. The synthetic worker/recipe/supervisor are not.
    return execution

class NativeMethodFailure(RuntimeError):pass

def worker(job_path,output):
    global _PAYLOAD_ACCESS_STARTED
    verify();execution=native_core();job=json.loads(job_path.read_text());validate_job(job)
    output.mkdir(parents=True,exist_ok=False)
    for name,value in execution.fixed_environment().items():
        if os.environ.get(name)!=value:raise ValueError("Fixed isolated environment differs: "+name)
    if sys.flags.ignore_environment or sys.flags.hash_randomization or not sys.flags.no_site or not sys.flags.safe_path:
        raise ValueError("Expected isolated SportsTorch -P -S -B worker and fixed hash seed")
    require_private_runtime(sys.prefix,sys.base_prefix,sys.executable,controller=False)
    execution.runtime_paths()
    denied=[]
    def deny(*args,**kwargs):denied.append("network_call");raise RuntimeError("Network forbidden during native gate")
    with ExitStack() as stack:
        for owner,name in ((socket.socket,"connect"),(socket.socket,"connect_ex"),(socket,"create_connection"),(urllib.request,"urlopen")):
            stack.enter_context(patch.object(owner,name,deny))
        import numpy as np
        import torch
        if str(torch.__version__)!="2.15.0" or np.__version__!="2.1.2":raise ValueError("Pinned Torch/NumPy differs")
        seed=int(job["seed"]);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision("highest")
        hardware=json.loads((ROUND/"hardware_probe.json").read_text())
        if not torch.backends.mps.is_available() or torch.mps.recommended_max_memory()!=hardware["recommended_max_memory_bytes"]:
            raise ValueError("MPS hardware probe differs")
        if hash("TabPFN-MPS-R57-fixed-hash-seed")!=hardware["hash_probe"]:raise ValueError("Hash seed differs")
        execution.MONITOR=execution.MemoryMonitor(output);execution.MONITOR.start();stack.callback(execution.MONITOR.close)
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from tabpfn.settings import settings
        if repr(settings.tabpfn.mps_memory_fraction)!=hardware["tabpfn_mps_memory_fraction"]:raise ValueError("MPS allocator fraction differs")
        execution.DISPATCH=execution.NativeDispatchGuard(stack)
        acquisition=json.loads((ROUND/"acquisition.json").read_text())
        for name,record in acquisition["resolved_runtime_distributions"].items():
            dist=metadata.distribution(name)
            if dist.version!=record["version"] or Path(dist.locate_file("")).resolve()!=execution.DEPS.resolve():
                raise ValueError("Dependency escaped isolated R59 target: "+name)
        if any(name.startswith(("posthog","tabpfn_common_utils")) for name in sys.modules):raise ValueError("Telemetry package imported")
        _PAYLOAD_ACCESS_STARTED=True
        context,target,original,swapped=[np.load(job[name],mmap_mode="r",allow_pickle=False) for name in ("context","target","original","swapped")]
        if (context.ndim!=2 or context.shape[1]!=166 or not 2<=len(context)<=3000 or len(context)%2 or target.shape!=(len(context),)
                or original.ndim!=2 or original.shape!=swapped.shape or original.shape[1]!=166 or not len(original)
                or any(not np.isfinite(v).all() for v in (context,original,swapped))
                or any(v.dtype!=np.float32 for v in (context,original,swapped)) or target.dtype!=np.int64
                or not np.array_equal(np.unique(target),[0,1]) or not np.array_equal(np.bincount(target),[len(context)//2]*2)
                or not np.array_equal(target[len(target)//2:],1-target[:len(target)//2])):
            raise ValueError("Native context/query schema differs")
        model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,model_path=execution.MODEL,
            n_estimators=8,softmax_temperature=.9,balance_probabilities=False,average_before_softmax=False,
            device="mps",ignore_pretraining_limits=False,inference_precision=torch.float32,
            fit_mode="fit_preprocessors",memory_saving_mode=True,random_state=seed,
            n_preprocessing_jobs=1,enable_torch_compile=False,tuning_config=None)
        _,fitting=execution.operation(output,"fit_context",lambda:model.fit(context,target))
        execution.exact_array(model.classes_,np.array([0,1]),"H0/A1 class order")
        checkpoint,_=execution.operation(output,"fingerprint_initial_weights",lambda:execution.checkpoint_fingerprint(model))
        preprocessing=execution.preprocessing_fingerprint(model)
        def resolved_recipe():
            return dict(parameters=execution.plain(model.get_params(deep=False)),architecture_configurations=execution.plain(model.configs_),
                inference_configuration=execution.plain(model.inference_config_),devices=execution.plain(model.devices_),
                forced_dtype=str(model.forced_inference_dtype_),autocast=bool(model.use_autocast_),n_estimators=int(model.n_estimators_),
                context_shape=list(context.shape),query_shape=list(original.shape),model_sha256=digest(execution.MODEL),
                context_sha256=digest(job["context"]),target_sha256=digest(job["target"]),head_seed=seed,
                fixed_environment=execution.fixed_environment(),class_order=["H","A"],fitted_estimator_pickle_used=False,
                native_public_processing=True,synthetic_only=False,competition_inputs_read=True,
                reconstruction="Fresh process fit of saved paired owntrain context and targets; no estimator pickle",
                attribution="Built with PriorLabs-TabPFN")
        recipe=resolved_recipe()
        if recipe["autocast"] or recipe["forced_dtype"]!="torch.float32" or recipe["n_estimators"]!=8:raise ValueError("Resolved settings differ")
        for name,value in (("checkpoint_fingerprint.json",checkpoint),("preprocessing_fingerprint.json",preprocessing),("recipe.json",recipe)):
            write_json(output/name,value)
        p0,t0=execution.canonical_predict(model,original,output,"original")
        p1,t1=execution.canonical_predict(model,swapped,output,"swapped")
        repeated,tr=execution.canonical_predict(model,original[:512],output,"repeat_first")
        execution.exact_array(repeated,p0[:512],"repeat_first")
        owner=(p0+p1[:,[1,0]])/2
        if preprocessing!=execution.preprocessing_fingerprint(model):raise ValueError("Query changed complete retained preprocessing/context state")
        after,_=execution.operation(output,"fingerprint_final_weights",lambda:execution.checkpoint_fingerprint(model))
        if checkpoint!=after:raise ValueError("Query changed checkpoint tensors")
        if recipe!=resolved_recipe():raise ValueError("Query changed resolved model/configuration state")
        for name,value in (("original_probability",p0),("swapped_probability",p1),("owner",owner),("repeat_first",repeated)):
            np.save(output/(name+".npy"),value)
            if "parent" in job:execution.exact_array(value,np.load(Path(job["parent"])/(name+".npy"),allow_pickle=False),name+" fresh replay")
        if "parent" in job:
            for name,value in (("checkpoint_fingerprint.json",checkpoint),("preprocessing_fingerprint.json",preprocessing),("recipe.json",recipe)):
                if value!=json.loads((Path(job["parent"])/name).read_text()):raise ValueError("Fresh context reconstruction differs: "+name)
        memory=execution.MONITOR.sample()
        if denied:raise ValueError("Network was attempted")
        write_json(output/"native_dispatch.json",execution.DISPATCH.record())
        verify()
        write_json(output/"summary.json",dict(passed=True,seed=seed,protocol=job["protocol"],block=job["block"],context_rows=len(context),query_rows=len(original),
            fit_seconds=fitting,query_timings=dict(original=t0,swapped=t1,repeat=tr),memory=memory,
            complete_raw_preprocessing_immutable=True,checkpoint_exact=True,resolved_model_configuration_immutable=True,repeat_byte_exact=True,
            fresh_context_replay="parent" in job,fitted_estimator_pickle_used=False,query_labels_supplied=False,
            native_public_processing=True,competition_inputs_read=True,synthetic_only=False,
            no_network_attempts=True,native_wrapper_exclusive=True,
            artifact_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in {"memory.json","memory.tmp"}}))

def supervise(job_path,output):
    """One isolated worker at a time, full owned RSS plus full driver <=6GiB."""
    verify();execution=native_core();job=json.loads(job_path.read_text());validate_job(job)
    environment={k:v for k,v in os.environ.items() if not k.startswith(("PYTHON","HF_","TABPFN_","TORCH_","PYTORCH_"))}
    environment.update(execution.fixed_environment())
    log,error=output.parent/"worker.stdout.log",output.parent/"worker.stderr.log"
    started,peak,aborted=time.monotonic(),0,None
    with log.open("wb") as stdout,error.open("wb") as stderr:
        process=subprocess.Popen([str(execution.PYTHON),"-P","-S","-B",str(Path(__file__).resolve()),"--worker",
            "--job",str(job_path),"--output",str(output)],cwd=ROUND,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
        while process.poll() is None:
            rss,_=execution.owned_rss(process.pid);memory=execution.read_json_if_present(output/"memory.json")
            peak=max(peak,rss+memory.get("mps_driver_bytes",0),memory.get("peak_conservative_combined_bytes",0))
            record=execution.read_json_if_present(output/"operation.json");elapsed=time.monotonic()-started
            if peak>LIMIT_BYTES:aborted="owned_rss_plus_full_driver_over_6GiB"
            elif record.get("active") and time.monotonic()-record["started"]>300:aborted="operation_over_300seconds"
            elif elapsed>900:aborted="worker_over_900seconds"
            elif memory and time.monotonic()-memory["sampled"]>5:aborted="memory_monitor_stale"
            if aborted:
                import signal
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                break
            time.sleep(.1)
        try:code=process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            import signal
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=process.wait(timeout=5)
    summary=execution.read_json_if_present(output/"summary.json");memory=execution.read_json_if_present(output/"memory.json")
    peak=max(peak,memory.get("peak_conservative_combined_bytes",0))
    record=dict(exit_code=code,aborted=aborted,peak_owned_rss_plus_full_driver_bytes=peak,elapsed_seconds=time.monotonic()-started,
        passed=code==0 and aborted is None and summary.get("passed",False) and bool(memory) and peak<=LIMIT_BYTES and not (output/"monitor_failure.json").exists(),
        native_public_processing=True,competition_inputs_read=True,synthetic_only=False,stdout_sha256=digest(log),stderr_sha256=digest(error),
        summary_sha256=digest(output/"summary.json") if (output/"summary.json").is_file() else None)
    write_json(output.parent/"supervision.json",record)
    verify()
    if not record["passed"]:raise NativeMethodFailure("Native numerical/resource gate failed; no retry or partial-fold rescue")
    return record

def preflight():
    """Source hashes and invented pure lists only; no native payload or scientific import."""
    started=time.monotonic()
    for name in SOURCES:
        source=(ROOT/name).read_text();tree=ast.parse(source);compile(source,str(ROOT/name),"exec")
        for statement in tree.body:
            if isinstance(statement,(ast.Import,ast.ImportFrom)):
                names=[n.name.split(".")[0] for n in statement.names] if isinstance(statement,ast.Import) else [(statement.module or "").split(".")[0]]
                if set(names)&{"numpy","pandas","torch","scipy","sklearn","tabpfn","joblib"}:raise ValueError("Scientific top-level import: "+name)
    test=load_module(ROOT/SOURCES[2],"r65_pure_tests");tests=test.run_tests()
    plan=json.loads(PLAN.read_text());protected=json.loads((ROUND/"protected_artifacts.json").read_text())
    for name,sha in {**plan["input_sha256"],**protected}.items():
        if digest(ROOT/name)!=sha:raise ValueError("Inherited input/protected source bytes changed")
    # R70v2 verifies every preserved R63/R67 binding except the explicitly
    # superseded shared executable, and binds the private replacement in full.
    # The old strict verifier remains unchanged and still rejects that drift.
    for name,sha in json.loads((ROUND/"source_closure.json").read_text()).items():
        if digest(ROOT/name)!=sha:raise ValueError("Inherited source closure changed")
    for name,sha in json.loads((ROUND/"head_runtime.json").read_text()).items():
        if digest(ROOT/name)!=sha:raise ValueError("Controller runtime changed")
    if not (ROOT/"experiments/EXPERIMENT_LEDGER.csv").read_bytes().startswith((ROUND/"preparation_ledger.csv").read_bytes()):
        raise ValueError("Historical preparation ledger prefix changed")
    technical_gate()
    if any(name in sys.modules for name in ("numpy","pandas","torch","scipy","sklearn","tabpfn","joblib")):
        raise ValueError("Scientific library imported during source-only preparation")
    write_json(ROUND/"source_preflight.json",dict(passed=True,seconds=time.monotonic()-started,
        source_sha256=source_identity(),plan_sha256=digest(PLAN),pure_tests=tests,
        native_arrays_loaded=False,scientific_imports=False,model_forward=False,
        technical_prerequisite="Exact frozen isolated R70v2 source/plan and actual all-four-stage PASS are bound; old strict runtime failure is preserved",
        technical_summary_sha256=digest(GATE_ROUND/"runtime_assessment/summary.json"),
        technical_pass_claim=True,freeze_authorized=False))
    print("R65 source/pure preflight passed; no native arrays or scientific imports",flush=True)

def freeze():
    """Available only after root-bound review and actual all-four-stage R70v2 pass."""
    review=json.loads((ROUND/"independent_review.json").read_text());pre=json.loads((ROUND/"source_preflight.json").read_text())
    if not review.get("approved_for_freeze") or review.get("source_sha256")!=source_identity() or review.get("plan_sha256")!=digest(PLAN):
        raise ValueError("Independent ROOT review must bind final three sources/plan")
    if (not pre["passed"] or pre["source_sha256"]!=source_identity() or pre["plan_sha256"]!=digest(PLAN)
            or pre["technical_summary_sha256"]!=digest(GATE_ROUND/"runtime_assessment/summary.json")):
        raise ValueError("Final pure preflight and actual technical binding required")
    gate=technical_gate();gate.verify();folder=ROUND/"protocol_freeze";folder.mkdir(exist_ok=False)
    (folder/"plan.json").write_bytes(PLAN.read_bytes())
    closure=json.loads((ROUND/"source_closure.json").read_text());closure.update(source_identity())
    for path in (PLAN,ROUND/"source_preflight.json",ROUND/"independent_review.json",ROUND/"source_closure.json",ROUND/"protected_artifacts.json",ROUND/"symlinks.json",ROUND/"head_runtime.json",ROUND/"preparation_ledger.csv"):
        closure[str(path.relative_to(ROOT))]=digest(path)
    technical={str(GATE_SOURCE.relative_to(ROOT)):digest(GATE_SOURCE)}
    for path in sorted((GATE_ROUND/"protocol_freeze").rglob("*")):
        if path.is_file():technical[str(path.relative_to(ROOT))]=digest(path)
    for path in sorted((GATE_ROUND/"runtime_assessment").rglob("*")):
        if path.is_file():technical[str(path.relative_to(ROOT))]=digest(path)
    for name in ("hardware_probe.json","acquisition.json"):
        (ROUND/name).write_bytes((GATE_ROUND/name).read_bytes());closure[str((ROUND/name).relative_to(ROOT))]=digest(ROUND/name)
    write_json(folder/"implementation.json",closure);write_json(folder/"technical_gate.json",technical)
    write_json(folder/"input_sha256.json",json.loads(PLAN.read_text())["input_sha256"])
    for name in ("protected_artifacts.json","symlinks.json","head_runtime.json"):(folder/name).write_bytes((ROUND/name).read_bytes())
    (folder/"EXPERIMENT_LEDGER.csv").write_bytes((ROOT/"experiments/EXPERIMENT_LEDGER.csv").read_bytes())
    write_json(folder/"freeze.json",dict(source_sha256=source_identity(),plan_sha256=digest(PLAN),
        technical_gate_source_sha256=digest(GATE_SOURCE),all_four_technical_stages_passed=True,
        no_native_arrays_loaded_before_freeze=True,model_forward_started=False,selected17_protected=True))
    verify();print("Frozen R65 after actual isolated R70v2 all-four-stage pass; no native model call",flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight",action="store_true");actions.add_argument("--freeze",action="store_true")
    actions.add_argument("--verify",action="store_true");actions.add_argument("--worker",action="store_true")
    parser.add_argument("--job",type=Path);parser.add_argument("--output",type=Path);args=parser.parse_args()
    if args.preflight:preflight()
    elif args.freeze:freeze()
    elif args.verify:verify();print("R65 frozen provenance verified",flush=True)
    else:
        if args.job is None or args.output is None:parser.error("Worker needs job/output")
        if not args.output.resolve().is_relative_to(ROUND.resolve()):parser.error("Output must stay in R65")
        try:worker(args.job,args.output)
        except BaseException as error:
            if args.output.exists():write_json(args.output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                native_public_processing=True,competition_inputs_read=_PAYLOAD_ACCESS_STARTED,
                native_payload_access_started=_PAYLOAD_ACCESS_STARTED,synthetic_only=False,no_partial_fold_rescue=True))
            raise
