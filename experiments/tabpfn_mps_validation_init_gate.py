"""R63: initialize source-defined categorical output-validation metadata once.
Retain R60's complete raw fingerprints and every original numerical/runtime gate.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_tabpfn_mps_validation_init_v1"
PREVIOUS=ROOT/"data/control_tabpfn_mps_audit_fix_v1"
PREVIOUS_SOURCE=ROOT/"experiments/tabpfn_mps_audit_fix_gate.py"
PREVIOUS_SHA256="32a1cecb752de4dcced9e736f0e3d2179ef25fea6427cd3d804199ed8f09e584"
TRACE=ROOT/"data/control_tabpfn_mps_cache_trace_v1"
TRACE_SOURCE=ROOT/"experiments/tabpfn_mps_cache_trace_gate.py"
TRACE_SHA256="ad48e32a5c8a11243fe91041bcce06ad3fe9707a3ab3d0104e15782f9ad1bd4f"
PLAN=ROOT/"experiments/TABPFN_MPS_VALIDATION_INIT_PLAN.json"
FIELDS=("n_added_columns_","modality_added_")
_CORE=None
_ORIGINAL_FP=None

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")

def load_wrapper(path,name,expected):
    if digest(path)!=expected:raise ValueError("Frozen wrapper source changed: "+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def previous():
    return load_wrapper(PREVIOUS_SOURCE,"r63_frozen_r60_wrapper",PREVIOUS_SHA256)

def verify_prior():
    previous().verify()
    load_wrapper(TRACE_SOURCE,"r63_retained_r62_wrapper",TRACE_SHA256).verify()

def source_proof():
    records=json.loads((ROUND/"source_proof.json").read_text())
    if {r["path"] for r in records}!={"preprocessing/pipeline_interface.py",
            "preprocessing/steps/encode_categorical_features_step.py"}:
        raise ValueError("Unexpected categorical validation source proof")
    for record in records:
        paths=(ROUND/"public_sources"/record["path"],Path(record["original"]),
            ROOT/"data/control_tabpfn_mps_loader_repair_v1/runtime_deps/tabpfn"/record["path"])
        if any(digest(path)!=record["sha256"] for path in paths):
            raise ValueError("Official validation source proof changed")
    return records

def source_contract():
    """Parse exact retained author source without importing scientific packages."""
    source_proof()
    def method(path,class_name,method_name):
        tree=ast.parse(path.read_bytes())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
        return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method_name)
    base=ROUND/"public_sources/preprocessing/pipeline_interface.py"
    categorical=ROUND/"public_sources/preprocessing/steps/encode_categorical_features_step.py"
    validator=method(base,"PreprocessingStep","_validate_added_data")
    transform=method(categorical,"EncodeCategoricalFeaturesStep","_transform")
    fit_transform=method(categorical,"EncodeCategoricalFeaturesStep","fit_transform")
    returns=[n for n in ast.walk(transform) if isinstance(n,ast.Return)]
    if len(returns)!=2 or any(not isinstance(n.value,ast.Tuple) or len(n.value.elts)!=3 or
            any(not isinstance(v,ast.Constant) or v.value is not None for v in n.value.elts[1:]) for n in returns):
        raise ValueError("Categorical branches no longer all return no added data/modality")
    assigned={n.attr for n in ast.walk(validator) if isinstance(n,ast.Attribute) and
        isinstance(n.ctx,ast.Store) and isinstance(n.value,ast.Name) and n.value.id=="self"}
    if assigned!=set(FIELDS):raise ValueError("Author validator writes unexpected instance state")
    if any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
            n.func.attr=="_validate_added_data" for n in ast.walk(fit_transform)):
        raise ValueError("Categorical fit_transform validation source changed")
    return dict(categorical_transform_returns=len(returns),all_added_data_and_modality_none=True,
        validator_only_assigned_fields=sorted(assigned),categorical_fit_transform_skips_base_validation=True,
        validator_lines=[validator.lineno,validator.end_lineno],
        transform_lines=[transform.lineno,transform.end_lineno]),validator

def initialize_step(step,bindings,raw_hash):
    """One exact author validation call; reject overwrites and unrelated mutation."""
    categorical_type,base_type,check_fitted=bindings
    if type(step) is not categorical_type:raise ValueError("Categorical step is not the exact author class")
    before=dict(vars(step))
    if categorical_type._validate_added_data is not base_type._validate_added_data or "_validate_added_data" in before:
        raise ValueError("Categorical validation method differs from inherited author method")
    if "feature_schema_updated_" not in before or before["feature_schema_updated_"] is None:
        raise ValueError("Categorical step has no fitted output schema")
    if "categorical_transformer_" not in before:raise ValueError("Categorical step has not been fitted")
    present=[name in before for name in FIELDS]
    if any(present) and not all(present):raise ValueError("Asymmetric validation metadata is forbidden")
    if any(before[name] is not None for name in FIELDS if name in before):
        raise ValueError("Non-None validation metadata must never be reset")
    full_before=raw_hash(step)
    old_hashes={name:raw_hash(value) for name,value in before.items()}
    if before["categorical_transformer_"] is not None:check_fitted(before["categorical_transformer_"])
    base_type._validate_added_data(step,X_added=None,modality_added=None)
    after=vars(step)
    if set(after)!=set(before)|set(FIELDS) or any(after[name] is not None for name in FIELDS):
        raise ValueError("Validator did not produce exactly the approved None metadata")
    if any(raw_hash(after[name])!=expected for name,expected in old_hashes.items()):
        raise ValueError("Initialization changed an existing fitted/configuration attribute")
    return dict(class_name=type(step).__module__+"."+type(step).__qualname__,
        method="PreprocessingStep._validate_added_data(X_added=None, modality_added=None)",
        before_fields={name:dict(present=name in before,value=None) for name in FIELDS},
        after_fields={name:dict(present=True,value=None) for name in FIELDS},
        raw_step_joblib_sha1_before=full_before,raw_step_joblib_sha1_after=raw_hash(step),
        all_existing_attribute_raw_hashes_unchanged=True,existing_attribute_raw_hashes=old_hashes,
        categorical_transformer_is_identity=before["categorical_transformer_"] is None,
        fitted_nonidentity_transformer_checked=before["categorical_transformer_"] is not None,
        added_fields=[name for name in FIELDS if name not in before],author_validation_calls=1)

class InitializationBoundary:
    """Keep lifecycle state outside the estimator; never initialize after queries."""
    def __init__(self):self.model=None
    def fingerprint(self,model,initialize,fingerprint):
        if self.model is None:
            initialize(model);self.model=model
        elif self.model is not model:raise ValueError("Unexpected second model in one worker")
        return fingerprint(model)

_BOUNDARY=InitializationBoundary()

def initialize_model_metadata(model):
    import joblib
    from sklearn.utils.validation import check_is_fitted
    from tabpfn.preprocessing.pipeline_interface import PreprocessingStep
    from tabpfn.preprocessing.steps.encode_categorical_features_step import EncodeCategoricalFeaturesStep
    output=_CORE.MONITOR.output
    path=output/"validation_metadata_initialization.json"
    if path.exists():raise ValueError("One-time initialization record already exists")
    raw_hash=lambda value:joblib.hash(value,hash_name="sha1")
    members=model.executor_.ensemble_members
    if type(model.executor_).__name__!="InferenceEngineCachePreprocessing" or len(members)!=8:
        raise ValueError("Unexpected retained preprocessing/eight-view interface")
    record=dict(status="started",pre_first_original_fingerprint=True,query_or_transform_called=False,
        fields_excluded_from_original_fingerprint=[],source_proof=source_proof(),events=[],
        raw_cpu_pipeline_joblib_sha1_before=[raw_hash(m.cpu_preprocessor) for m in members])
    save(path,record)
    try:
        for member_index,member in enumerate(members):
            selected=[(i,pair[0]) for i,pair in enumerate(member.cpu_preprocessor.steps)
                if type(pair[0]) is EncodeCategoricalFeaturesStep]
            if len(selected)!=1:raise ValueError("Expected exactly one author categorical step per member")
            step_index,step=selected[0]
            event=initialize_step(step,(EncodeCategoricalFeaturesStep,PreprocessingStep,check_is_fitted),raw_hash)
            event.update(member_index=member_index,step_index=step_index)
            record["events"].append(event);save(path,record)
        record.update(status="complete",initialized_members=8,
            raw_cpu_pipeline_joblib_sha1_after=[raw_hash(m.cpu_preprocessor) for m in members],
            no_non_none_state_reset=True,no_subsequent_initialization_permitted=True)
        save(path,record)
    except BaseException as error:
        record.update(status="failed",error=repr(error));save(path,record)
        raise

def initialized_fingerprint(model):
    return _BOUNDARY.fingerprint(model,initialize_model_metadata,_ORIGINAL_FP)

def core():
    global _CORE,_ORIGINAL_FP
    execution=previous().core()
    execution.ROUND=ROUND;execution.PLAN=PLAN
    execution.__file__=str(Path(__file__).resolve());execution.verify=verify
    _ORIGINAL_FP=execution.preprocessing_fingerprint
    execution.preprocessing_fingerprint=initialized_fingerprint
    _CORE=execution
    return execution

def pure_tests(validator):
    """Use the parsed author validation body; no scientific imports or transforms."""
    function=copy.deepcopy(validator);function.decorator_list=[]
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),function],type_ignores=[])
    namespace={};exec(compile(ast.fix_missing_locations(module),"<retained-author-validator>","exec"),namespace)
    class Base:pass
    Base._validate_added_data=namespace["_validate_added_data"]
    class Categorical(Base):
        def __init__(self):self.feature_schema_updated_="fitted_schema";self.categorical_transformer_=None;self.fitted_numeric_=[1,2]
    class Transformer:
        def __init__(self,fitted=True):self.fitted_=fitted
    def fitted(value):
        if not value.fitted_:raise ValueError("Not fitted")
    def raw_hash(value):
        return hashlib.sha1(json.dumps(value,sort_keys=True,default=lambda v:vars(v)).encode()).hexdigest()
    bindings=(Categorical,Base,fitted);outcomes=[]
    def expect(name,step,valid=True,custom=bindings):
        try:result=initialize_step(step,custom,raw_hash);accepted=True
        except (ValueError,AttributeError):accepted=False;result=None
        if accepted!=valid:raise AssertionError("Pure sentinel failed: "+name)
        outcomes.append(dict(name=name,passed=True,expected_valid=valid));return result
    expect("missing_metadata_identity_is_fitted",Categorical())
    step=Categorical();step.n_added_columns_=None;step.modality_added_=None
    result=expect("existing_none_is_validation_only",step)
    if result["raw_step_joblib_sha1_before"]!=result["raw_step_joblib_sha1_after"]:raise AssertionError("Existing None state changed")
    step=Categorical();step.categorical_transformer_=Transformer();expect("fitted_transformer_accepted",step)
    step=Categorical();step.categorical_transformer_=Transformer(False);expect("unfitted_transformer_rejected",step,False)
    for name,value in (("n_added_columns_",1),("modality_added_","numerical")):
        step=Categorical();step.n_added_columns_=None;step.modality_added_=None;setattr(step,name,value)
        expect("non_none_"+name+"_rejected",step,False)
    for name in FIELDS:
        step=Categorical();setattr(step,name,None);expect("asymmetric_"+name+"_rejected",step,False)
    for name in ("feature_schema_updated_","categorical_transformer_"):
        step=Categorical();delattr(step,name);expect("missing_fitted_"+name+"_rejected",step,False)
    step=Categorical();step.feature_schema_updated_=None;expect("none_schema_rejected",step,False)
    class Subclass(Categorical):pass
    expect("subclass_rejected",Subclass(),False)
    step=Categorical();step._validate_added_data=lambda **kwargs:None;expect("instance_override_rejected",step,False)
    class Changed(Categorical):
        def _validate_added_data(self,**kwargs):pass
    expect("class_override_rejected",Changed(),False,(Changed,Base,fitted))
    class MutatingBase:
        def _validate_added_data(self,**kwargs):
            self.n_added_columns_=None;self.modality_added_=None;self.fitted_numeric_.append(3)
    class Mutating(Categorical,MutatingBase):_validate_added_data=MutatingBase._validate_added_data
    expect("unrelated_fitted_mutation_rejected",Mutating(),False,(Mutating,MutatingBase,fitted))
    boundary=InitializationBoundary();model=object();counts=dict(init=0,fp=0)
    def init(value):counts["init"]+=1
    def fingerprint(value):counts["fp"]+=1;return "complete_raw_fingerprint"
    for _ in range(3):boundary.fingerprint(model,init,fingerprint)
    if counts!={"init":1,"fp":3}:raise AssertionError("Initialization repeated after baseline")
    try:boundary.fingerprint(object(),init,fingerprint)
    except ValueError:pass
    else:raise AssertionError("Second model silently reinitialized")
    outcomes.append(dict(name="one_time_boundary_never_reinitializes",passed=True))
    return outcomes

def preflight():
    started=time.monotonic();compile(Path(__file__).read_bytes(),str(Path(__file__)),"exec")
    verify_prior();contract,validator=source_contract();tests=pure_tests(validator);configured=core()
    if configured.preprocessing_fingerprint is not initialized_fingerprint or configured.verify is not verify:
        raise ValueError("Initialization/verifier binding differs")
    for name in ("worker","main","supervise","operation","canonical_predict","checkpoint_fingerprint","synthetic_data"):
        if Path(getattr(configured,name).__code__.co_filename).resolve()!=ROOT/"experiments/tabpfn_mps_runtime_gate.py":
            raise ValueError("Original numerical/runtime function changed: "+name)
    if Path(_ORIGINAL_FP.__code__.co_filename).resolve()!=PREVIOUS_SOURCE:
        raise ValueError("Complete original R60 fingerprint is not retained")
    if (ROUND/"hardware_probe.json").read_bytes()!=(PREVIOUS/"hardware_probe.json").read_bytes():
        raise ValueError("Hardware memory/seed settings changed")
    if any(name in sys.modules for name in ("torch","numpy","scipy","sklearn","tabpfn","joblib")):
        raise ValueError("Scientific package imported during pure preflight")
    save(ROUND/"source_preflight.json",dict(passed=True,seconds=time.monotonic()-started,
        implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),source_contract=contract,
        pure_sentinel_tests=tests,all_prior_frozen_inputs_verified=True,
        original_full_raw_fingerprint_retained=True,unchanged_numerical_core_verified=True,
        scientific_imports=False,model_forward=False,competition_inputs_read=False))
    print("R63 source/pure-sentinel preflight passed; no model call",flush=True)

def freeze():
    review=json.loads((ROUND/"independent_review.json").read_text())
    if not review["approved_for_freeze"] or review["implementation_sha256"]!=digest(__file__) or review["plan_sha256"]!=digest(PLAN):
        raise ValueError("Independent root review must bind final R63 source/plan")
    preflight_record=json.loads((ROUND/"source_preflight.json").read_text())
    if not preflight_record["passed"] or preflight_record["implementation_sha256"]!=digest(__file__) or preflight_record["plan_sha256"]!=digest(PLAN):
        raise ValueError("Final source/pure-sentinel preflight must precede freeze")
    verify_prior();source_proof()
    checks={str(Path(__file__).relative_to(ROOT)):digest(__file__),str(PLAN.relative_to(ROOT)):digest(PLAN)}
    for path in sorted(ROUND.rglob("*")):
        if "protocol_freeze" not in path.parts and "runtime_assessment" not in path.parts and path.is_file():
            if path.suffix==".pyc" or "__pycache__" in path.parts:raise ValueError("Unexpected generated bytecode")
            checks[str(path.relative_to(ROOT))]=digest(path)
    for folder in (PREVIOUS,TRACE):
        for relative in ("protocol_freeze/implementation.json","protocol_freeze/freeze.json",
                "runtime_assessment/summary.json","runtime_assessment/moderate/first/failure.json",
                "runtime_assessment/moderate/first.stderr.log"):
            path=folder/relative;checks[str(path.relative_to(ROOT))]=digest(path)
    checks[str((TRACE/"TRACE_FINDINGS.md").relative_to(ROOT))]=digest(TRACE/"TRACE_FINDINGS.md")
    checks[str(TRACE_SOURCE.relative_to(ROOT))]=TRACE_SHA256
    folder=ROUND/"protocol_freeze";folder.mkdir(exist_ok=False)
    (folder/"plan.json").write_bytes(PLAN.read_bytes());save(folder/"implementation.json",checks)
    save(folder/"freeze.json",dict(implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),
        previous_r60_source_sha256=PREVIOUS_SHA256,retained_r62_source_sha256=TRACE_SHA256,bound_files=len(checks),
        one_time_validation_metadata_initialization_only=True,full_raw_fingerprint_unchanged=True,
        prior_failed_rounds_retained=True,model_math_settings_changed=False,model_forward_started=False))
    print("Frozen R63 metadata initialization; no model call",flush=True)

def verify():
    folder=ROUND/"protocol_freeze"
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes():raise ValueError("R63 plan changed")
    for path,expected in json.loads((folder/"implementation.json").read_text()).items():
        if digest(ROOT/path)!=expected:raise ValueError("Frozen R63 input changed: "+path)
    verify_prior();source_proof()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group()
    action.add_argument("--preflight",action="store_true");action.add_argument("--freeze",action="store_true")
    action.add_argument("--verify",action="store_true");action.add_argument("--worker",action="store_true")
    parser.add_argument("--size",choices=("moderate","full"));parser.add_argument("--stage",choices=("first","rebuild"))
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    output=(args.output or ROUND/"runtime_assessment").resolve()
    if not output.is_relative_to(ROUND.resolve()):parser.error("Artifacts must stay in R63")
    if args.preflight:preflight()
    elif args.freeze:freeze()
    elif args.verify:verify();print("R63 and all prior frozen inputs verified",flush=True)
    else:
        execution=core()
        if args.worker:
            if not args.size or not args.stage:parser.error("Worker requires size/stage")
            try:execution.worker(args.size,args.stage,output)
            except BaseException as error:
                if output.exists():save(output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                    synthetic_only=True,competition_inputs_read=False,prior_failures_retained=True))
                raise
        else:execution.main(output)
