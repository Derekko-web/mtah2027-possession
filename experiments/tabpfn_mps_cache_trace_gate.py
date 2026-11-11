"""R62: one synthetic moderate cache trace; preserve R60's exact failing assertion.
All numerical worker/model/query functions remain the frozen R57/R60 functions.
"""
import argparse
from enum import Enum
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_tabpfn_mps_cache_trace_v1"
PREVIOUS=ROOT/"data/control_tabpfn_mps_audit_fix_v1"
PREVIOUS_SOURCE=ROOT/"experiments/tabpfn_mps_audit_fix_gate.py"
PREVIOUS_SHA256="32a1cecb752de4dcced9e736f0e3d2179ef25fea6427cd3d804199ed8f09e584"
PLAN=ROOT/"experiments/TABPFN_MPS_CACHE_TRACE_PLAN.json"
_CORE=None
_MODEL=None
_OUTPUT=None
_ORIGINAL_FP=None
_ORIGINAL_OPERATION=None
_AUDIT_COUNT=0
_QUERY_COUNT=0

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")

def qtype(value):
    return type(value).__module__+"."+type(value).__qualname__

def observed_state(value,path="$",seen=None):
    """Read vars directly; never call estimator repr/getstate/properties/transform.
    Array bytes and declared RNG state are the only runtime-specific observations.
    """
    if seen is None:seen={}
    if value is None or type(value) in (str,int,bool):return value
    if type(value) is float:return value if math.isfinite(value) else {"float":str(value)}
    if isinstance(value,Enum):return {"enum":qtype(value),"value":observed_state(value.value,path,seen)}
    if isinstance(value,(bytes,bytearray)):
        return dict(type=qtype(value),bytes=len(value),sha256=hashlib.sha256(value).hexdigest())
    np=sys.modules.get("numpy");torch=sys.modules.get("torch")
    if np is not None and isinstance(value,np.ndarray):
        if value.dtype.hasobject:
            content=observed_state(value.tolist(),path+".object_values",seen)
            record=dict(shape=list(value.shape),dtype=str(value.dtype),object_values=content)
        else:record=_CORE.array_record(value)
        return dict(array=record,strides=list(value.strides),c_contiguous=bool(value.flags.c_contiguous),
            f_contiguous=bool(value.flags.f_contiguous),writeable=bool(value.flags.writeable))
    if np is not None and isinstance(value,np.generic):return observed_state(value.item(),path,seen)
    if np is not None and isinstance(value,np.dtype):return {"numpy_dtype":str(value)}
    if torch is not None and isinstance(value,torch.Tensor):
        return dict(tensor=_CORE.array_record(value),device=str(value.device),requires_grad=bool(value.requires_grad))
    if np is not None and isinstance(value,np.random.RandomState):
        return dict(rng=qtype(value),state=observed_state(value.get_state(),path+".rng_state",seen))
    if np is not None and isinstance(value,np.random.Generator):
        return dict(rng=qtype(value),state=observed_state(value.bit_generator.state,path+".rng_state",seen))
    if isinstance(value,random.Random):return dict(rng=qtype(value),state=observed_state(value.getstate(),path+".rng_state",seen))
    if isinstance(value,Path):return {"path":str(value)}
    if isinstance(value,slice):return {"slice":[value.start,value.stop,value.step]}
    if isinstance(value,type):return {"class":value.__module__+"."+value.__qualname__}
    if callable(value) and type(value).__name__ in ("function","builtin_function_or_method","method"):
        return {"callable_module":getattr(value,"__module__",None),"qualname":getattr(value,"__qualname__",None)}
    identity=id(value)
    if identity in seen:return {"alias_of":seen[identity]}
    seen[identity]=path
    if type(value) is dict:
        if all(type(key) is str for key in value):
            return {key:observed_state(value[key],path+"."+key,seen) for key in sorted(value)}
        return {"mapping":[[observed_state(k,path+".key",seen),observed_state(v,path+".value",seen)] for k,v in value.items()]}
    if type(value) in (list,tuple):
        return {"sequence_type":qtype(value),"items":[observed_state(v,path+f"[{i}]",seen) for i,v in enumerate(value)]}
    if type(value) in (set,frozenset):
        values=[observed_state(v,path+".set_item",{}) for v in value]
        return {"set_type":qtype(value),"items":sorted(values,key=lambda x:json.dumps(x,sort_keys=True))}
    try:attributes=vars(value)
    except TypeError:return {"opaque_type":qtype(value),"no_instance_vars":True}
    return dict(object_type=qtype(value),attributes={key:observed_state(attributes[key],path+"."+key,seen)
        for key in sorted(attributes)},declared_fitted_attribute_names=[key for key in sorted(attributes) if key.endswith("_")])

def changes(before,after,path="$"):
    if type(before) is dict and type(after) is dict:
        result=[]
        for key in sorted(set(before)|set(after)):
            if key not in before:result.append(dict(path=path+"."+key,change="added",after=after[key]))
            elif key not in after:result.append(dict(path=path+"."+key,change="removed",before=before[key]))
            else:result.extend(changes(before[key],after[key],path+"."+key))
        return result
    if type(before) is list and type(after) is list and len(before)==len(after):
        return [change for i,(a,b) in enumerate(zip(before,after)) for change in changes(a,b,path+f"[{i}] ".rstrip())]
    return [] if before==after else [dict(path=path,change="value",before=before,after=after)]

def component_state(model):
    """Hash descriptions of vars, not live estimators, to avoid extra pickle hooks."""
    import joblib
    members=[]
    for index,member in enumerate(model.executor_.ensemble_members):
        steps=[]
        for step_index,(step,modalities) in enumerate(member.cpu_preprocessor.steps):
            descriptor=observed_state(step)
            steps.append(dict(index=step_index,class_name=qtype(step),modalities=observed_state(modalities),
                complete_instance_vars=descriptor,
                observed_state_joblib_sha1=joblib.hash(descriptor,hash_name="sha1"),
                hash_scope="Primitive observed descriptor; no extra live-object pickle/getstate calls"))
        members.append(dict(index=index,X_train=observed_state(member.X_train),y_train=observed_state(member.y_train),
            configuration=observed_state(member.config),feature_schema=observed_state(member.feature_schema),
            feature_indices=observed_state(member.feature_indices),
            complete_cpu_pipeline=observed_state(member.cpu_preprocessor),cpu_steps=steps,
            complete_gpu_pipeline=observed_state(member.gpu_preprocessor)))
    inputs=("ordinal_encoder_","date_transformer_","text_transformer_","label_encoder_",
            "inferred_feature_schema_","categorical_features_indices_")
    return dict(engine=qtype(model.executor_),members=members,
        input_preprocessors={name:observed_state(getattr(model,name)) for name in inputs},
        pipeline_seeds=observed_state(model.ensemble_preprocessor_.pipeline_seeds),
        ensemble_configurations=observed_state(model.ensemble_configs_),
        global_rng_read_or_advanced=False,stored_declared_rng_states_recorded=True)

def trace_folder(output):
    folder=output/"component_trace";folder.mkdir(exist_ok=True)
    return folder

def traced_fingerprint(model):
    global _MODEL,_AUDIT_COUNT
    _MODEL=model;_AUDIT_COUNT+=1
    folder=trace_folder(_OUTPUT);stem=f"audit_{_AUDIT_COUNT:03d}"
    before=component_state(model);save(folder/(stem+".before.json"),before)
    try:
        result=_ORIGINAL_FP(model)  # Same live joblib hash calls/count as the frozen audit.
        save(folder/(stem+".original_fingerprint.json"),result)
        return result
    finally:
        after=component_state(model);save(folder/(stem+".after.json"),after)
        save(folder/(stem+".observer_diff.json"),changes(before,after))

def traced_operation(output,name,callback):
    global _OUTPUT,_QUERY_COUNT
    _OUTPUT=output
    query=name.startswith(("original_","swapped_","repeat_first_","diagnostic_"))
    if not query:return _ORIGINAL_OPERATION(output,name,callback)
    if _MODEL is None:raise ValueError("No fitted model captured before query tracing")
    _QUERY_COUNT+=1;stem=f"query_{_QUERY_COUNT:03d}_{name}"
    folder=trace_folder(output)
    before=component_state(_MODEL);save(folder/(stem+".before.json"),before)
    try:
        result,elapsed=_ORIGINAL_OPERATION(output,name,callback)
        import numpy as np
        np.save(folder/(stem+".probability.npy"),result)
        save(folder/(stem+".probability_record.json"),_CORE.array_record(result))
        return result,elapsed
    finally:
        after=component_state(_MODEL);save(folder/(stem+".after.json"),after)
        save(folder/(stem+".diff.json"),changes(before,after))

def previous():
    if digest(PREVIOUS_SOURCE)!=PREVIOUS_SHA256:raise ValueError("Frozen R60 audit source changed")
    spec=importlib.util.spec_from_file_location("r62_frozen_r60_wrapper",PREVIOUS_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def core():
    global _CORE,_ORIGINAL_FP,_ORIGINAL_OPERATION
    execution=previous().core();execution.ROUND=ROUND;execution.PLAN=PLAN
    execution.__file__=str(Path(__file__).resolve());execution.verify=verify
    _ORIGINAL_FP=execution.preprocessing_fingerprint;_ORIGINAL_OPERATION=execution.operation
    execution.preprocessing_fingerprint=traced_fingerprint;execution.operation=traced_operation
    _CORE=execution
    return execution

def pure_tests():
    class Sentinel:
        def __init__(self):self.fitted_=[1,2];self.cache={"counter":0}
        def __repr__(self):raise AssertionError("Observer must not call repr")
        def __getstate__(self):raise AssertionError("Observer must not pickle live objects")
    a=Sentinel();first=observed_state(a);second=observed_state(a)
    if changes(first,second):raise AssertionError("Reading vars changed sentinel")
    a.cache["counter"]=1;diff=changes(first,observed_state(a))
    if len(diff)!=1 or not diff[0]["path"].endswith("cache.counter"):raise AssertionError("Nested state diff not localized")
    a.fitted_.append(3)
    if not any("fitted_" in d["path"] for d in changes(first,observed_state(a))):raise AssertionError("Fitted state change missed")
    cyclic={};cyclic["self"]=cyclic
    if observed_state(cyclic)["self"]!={"alias_of":"$"}:raise AssertionError("Cycle handling differs")
    rng=random.Random(42);state=rng.getstate();observed_state(rng)
    if state!=rng.getstate():raise AssertionError("RNG observation advanced stream")
    return dict(passed=True,tests=5,repr_pickle_transform_called=False,rng_advanced=False)

def preflight():
    started=time.monotonic();compile(Path(__file__).read_bytes(),str(Path(__file__)),"exec")
    previous().verify();tests=pure_tests();configured=core()
    if configured.operation is not traced_operation or configured.preprocessing_fingerprint is not traced_fingerprint:
        raise ValueError("Trace callback binding differs")
    if Path(configured.worker.__code__.co_filename).resolve()!=ROOT/"experiments/tabpfn_mps_runtime_gate.py":
        raise ValueError("Original worker code changed")
    for record in json.loads((ROUND/"source_proof.json").read_text()):
        if digest(ROUND/"public_sources"/record["path"])!=record["sha256"] or digest(record["original"])!=record["sha256"]:
            raise ValueError("Retained source proof changed")
    if any(name in sys.modules for name in ("torch","numpy","scipy","sklearn","tabpfn","joblib")):
        raise ValueError("Scientific package imported before freeze")
    save(ROUND/"source_preflight.json",dict(passed=True,seconds=time.monotonic()-started,
        implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),pure_tests=tests,
        all_prior_frozen_inputs_verified=True,original_worker_unchanged=True,
        scientific_imports=False,model_forward=False))
    print("R62 trace source/pure tests passed; no model call",flush=True)

def freeze():
    review=json.loads((ROUND/"independent_review.json").read_text())
    if not review["approved_for_freeze"] or review["implementation_sha256"]!=digest(__file__) or review["plan_sha256"]!=digest(PLAN):
        raise ValueError("Independent review must bind final trace source/plan")
    previous().verify()
    checks={str(Path(__file__).relative_to(ROOT)):digest(__file__),str(PLAN.relative_to(ROOT)):digest(PLAN)}
    for path in sorted(ROUND.rglob("*")):
        if "protocol_freeze" not in path.parts and "runtime_assessment" not in path.parts and path.is_file():
            checks[str(path.relative_to(ROOT))]=digest(path)
    for path in [PREVIOUS/"protocol_freeze/implementation.json",PREVIOUS/"protocol_freeze/freeze.json",
        PREVIOUS/"runtime_assessment/summary.json",PREVIOUS/"runtime_assessment/moderate/first/failure.json",
        PREVIOUS/"runtime_assessment/moderate/first.stderr.log"]:
        checks[str(path.relative_to(ROOT))]=digest(path)
    folder=ROUND/"protocol_freeze";folder.mkdir(exist_ok=False)
    (folder/"plan.json").write_bytes(PLAN.read_bytes());save(folder/"implementation.json",checks)
    save(folder/"freeze.json",dict(implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),
        prior_source_sha256=PREVIOUS_SHA256,bound_files=len(checks),diagnostic_only=True,
        original_immutability_assertion_unchanged=True,model_math_settings_changed=False,model_forward_started=False))
    print("Frozen R62 trace; no new model call",flush=True)

def verify():
    folder=ROUND/"protocol_freeze"
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes():raise ValueError("Trace plan changed")
    for path,expected in json.loads((folder/"implementation.json").read_text()).items():
        if digest(ROOT/path)!=expected:raise ValueError("Trace frozen input changed: "+path)
    previous().verify()

def main(output):
    verify();output.mkdir(parents=True,exist_ok=False)
    execution=core();record=execution.supervise("moderate","first",output)
    save(output/"summary.json",dict(diagnostic_only=True,worker_result=record,
        original_immutability_assertion_unchanged=True,prior_failures_retained=True,
        complete_runtime_or_model_gate_pass_claim=False,full_context_run=False,
        native_processing_started=False,competition_inputs_read=False,
        implementation_sha256=digest(__file__),plan_sha256=digest(PLAN)))
    verify()
    if not record["passed"]:raise SystemExit(1)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument("--preflight",action="store_true");action.add_argument("--freeze",action="store_true")
    action.add_argument("--verify",action="store_true");action.add_argument("--worker",action="store_true")
    parser.add_argument("--size",choices=("moderate",));parser.add_argument("--stage",choices=("first",))
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    output=(args.output or ROUND/"runtime_assessment").resolve()
    if not output.is_relative_to(ROUND.resolve()):parser.error("Artifacts must stay in R62")
    if args.preflight:preflight()
    elif args.freeze:freeze()
    elif args.verify:verify();print("R62 and all prior inputs verified",flush=True)
    elif args.worker:
        if args.size!="moderate" or args.stage!="first":parser.error("Trace is exactly one moderate/first worker")
        try:core().worker(args.size,args.stage,output)
        except BaseException as error:
            if output.exists():save(output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                diagnostic_only=True,competition_inputs_read=False,prior_failures_retained=True))
            raise
    else:main(output)
