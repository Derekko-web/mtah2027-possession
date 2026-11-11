"""R60: observational preprocessing audit correction; frozen R57 math/R59 loader.
No estimator/preprocessing/forward implementation or model parameter is changed.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_tabpfn_mps_audit_fix_v1"
PREVIOUS=ROOT/"data/control_tabpfn_mps_loader_repair_v1"
PREVIOUS_SOURCE=ROOT/"experiments/tabpfn_mps_loader_repair_gate.py"
PREVIOUS_SHA256="65a1cad41bd54d33d28f08ca867f7416aedd8d50889be5a25a165b44bdcf035c"
PLAN=ROOT/"experiments/TABPFN_MPS_AUDIT_FIX_PLAN.json"
DEPS=PREVIOUS/"runtime_deps"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def qualified(obj):
    return type(obj).__module__+"."+type(obj).__qualname__

def audit_legacy_gpu_pipeline(pipeline,expected_std,bindings):
    """Read-only, source-defined audit. Bindings permit pure mock sentinel tests."""
    pipeline_type,step_type,remover_type,numerical=bindings
    if expected_std is None:
        if pipeline is not None:raise ValueError("Unexpected legacy GPU pipeline without outlier clipping")
        state=dict(pipeline=None,outlier_removal_std=None)
    else:
        if type(pipeline) is not pipeline_type:raise ValueError("Unexpected legacy GPU pipeline class")
        expected_keys={"steps","keep_fitted_cache","fitted_cache","record_timings","step_timings_"}
        if set(vars(pipeline))!=expected_keys:raise ValueError("Unexpected GPU pipeline state fields")
        if pipeline.keep_fitted_cache is not False or pipeline.record_timings is not False or pipeline.step_timings_ is not None:
            raise ValueError("Legacy outlier pipeline retained cache/timings unexpectedly")
        if type(pipeline.fitted_cache) is not list or len(pipeline.fitted_cache)!=1 or pipeline.fitted_cache[0] is not None:
            raise ValueError("Legacy outlier cache must remain exactly[None]")
        if type(pipeline.steps) is not list or len(pipeline.steps)!=1:
            raise ValueError("Only the official single soft-clip GPU step is allowed")
        pair=pipeline.steps[0]
        if type(pair) is not tuple or len(pair)!=2:raise ValueError("Unexpected GPU step registration")
        step,modalities=pair
        if type(step) is not step_type or type(modalities) is not set or modalities!={numerical}:
            raise ValueError("GPU step must be exactly numerical TorchSoftClipOutliersStep")
        if set(vars(step))!={"_outlier_remover"}:raise ValueError("Unexpected outlier step state fields")
        remover=step._outlier_remover
        if type(remover) is not remover_type or set(vars(remover))!={"n_sigma"}:
            raise ValueError("Unexpected outlier-remover state")
        sigma=remover.n_sigma
        if type(sigma) not in (int,float) or not math.isfinite(sigma) or sigma<=0 or sigma!=expected_std:
            raise ValueError("Outlier n_sigma differs from ensemble configuration")
        state=dict(pipeline_class=qualified(pipeline),keep_fitted_cache=False,fitted_cache=[None],
            record_timings=False,step_timings_=None,outlier_removal_std=expected_std,
            steps=[dict(step_class=qualified(step),modalities=[numerical.value],
                attributes=dict(_outlier_remover=dict(class_name=qualified(remover),attributes=dict(n_sigma=sigma))))])
    encoded=json.dumps(state,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return dict(state=state,state_sha256=hashlib.sha256(encoded).hexdigest())

def sentinel_tests():
    """No Torch, NumPy, native package, model, preprocessing or forward call."""
    class Remover:
        def __init__(self,sigma):self.n_sigma=sigma
    class Step:
        def __init__(self,sigma=12.0):self._outlier_remover=Remover(sigma)
    class Pipeline:
        def __init__(self):
            self.steps=[(Step(),{numerical})];self.keep_fitted_cache=False
            self.fitted_cache=[None];self.record_timings=False;self.step_timings_=None
    class Modality:
        value="numerical"
    numerical=Modality();bindings=(Pipeline,Step,Remover,numerical)
    outcomes=[]
    def expect(name,callback,valid):
        try: callback();accepted=True
        except ValueError:accepted=False
        if accepted!=valid:raise AssertionError("Pure sentinel failed: "+name)
        outcomes.append(dict(name=name,expected_valid=valid,passed=True))
    expect("none_without_clipping",lambda:audit_legacy_gpu_pipeline(None,None,bindings),True)
    expect("exact_default_outlier_pipeline",lambda:audit_legacy_gpu_pipeline(Pipeline(),12.0,bindings),True)
    expect("missing_expected_outlier",lambda:audit_legacy_gpu_pipeline(None,12.0,bindings),False)
    expect("unexpected_pipeline_without_clipping",lambda:audit_legacy_gpu_pipeline(Pipeline(),None,bindings),False)
    def mutated(name,mutation):
        pipeline=Pipeline();mutation(pipeline)
        expect(name,lambda:audit_legacy_gpu_pipeline(pipeline,12.0,bindings),False)
    mutated("wrong_sigma",lambda p:setattr(p.steps[0][0]._outlier_remover,"n_sigma",4.0))
    mutated("extra_gpu_step",lambda p:p.steps.append((Step(),{numerical})))
    mutated("all_modalities_forbidden",lambda p:p.steps.__setitem__(0,(Step(),None)))
    mutated("extra_modality",lambda p:p.steps.__setitem__(0,(Step(),{numerical,"categorical"})))
    mutated("retained_fitted_cache",lambda p:setattr(p,"keep_fitted_cache",True))
    mutated("populated_fitted_cache",lambda p:setattr(p,"fitted_cache",[{"lower":1}]))
    mutated("cache_length_changed",lambda p:setattr(p,"fitted_cache",[None,None]))
    mutated("unexpected_step_state",lambda p:setattr(p.steps[0][0],"extra",1))
    mutated("unexpected_pipeline_state",lambda p:setattr(p,"extra",1))
    mutated("timing_collection",lambda p:setattr(p,"record_timings",True))
    first=Pipeline();second=Pipeline()
    if audit_legacy_gpu_pipeline(first,12.0,bindings)!=audit_legacy_gpu_pipeline(second,12.0,bindings):
        raise AssertionError("Independent default audit states/hash differ")
    outcomes.append(dict(name="fresh_object_state_hash_exact",passed=True))
    return outcomes

def previous():
    if digest(PREVIOUS_SOURCE)!=PREVIOUS_SHA256:raise ValueError("Frozen R59 source changed")
    spec=importlib.util.spec_from_file_location("r60_frozen_r59_wrapper",PREVIOUS_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def preprocessing_fingerprint(model):
    """Same R57 retained-cache fingerprint, with the corrected additional GPU audit."""
    import joblib
    from tabpfn.preprocessing.datamodel import FeatureModality
    from tabpfn.preprocessing.torch.pipeline_interface import TorchPreprocessingPipeline
    from tabpfn.preprocessing.torch.steps import TorchSoftClipOutliersStep
    from tabpfn.preprocessing.torch.torch_soft_clip_outliers import TorchSoftClipOutliers
    from tabpfn.preprocessing.steps.reshape_feature_distribution_step import ReshapeFeatureDistributionsStep
    from tabpfn.preprocessing.steps.add_svd_features_step import AddSVDFeaturesStep
    engine=model.executor_;ensemble=model.ensemble_preprocessor_
    if type(engine).__name__!="InferenceEngineCachePreprocessing" or len(engine.ensemble_members)!=8:
        raise ValueError("Unexpected retained preprocessing/eight-view interface")
    if model.inference_config_.ENABLE_GPU_PREPROCESSING is not False or ensemble.enable_gpu_preprocessing is not False:
        raise ValueError("Quantile/SVD GPU preprocessing was enabled")
    if ensemble.keep_fitted_cache is not False:raise ValueError("Unexpected fitted GPU cache interface")
    bindings=(TorchPreprocessingPipeline,TorchSoftClipOutliersStep,TorchSoftClipOutliers,FeatureModality.NUMERICAL)
    execution=_ACTIVE_CORE
    members=[]
    for member in engine.ensemble_members:
        cpu_steps=[pair[0] for pair in member.cpu_preprocessor.steps]
        reshapers=[step for step in cpu_steps if type(step) is ReshapeFeatureDistributionsStep]
        svds=[step for step in cpu_steps if type(step) is AddSVDFeaturesStep]
        config=member.config.preprocess_config
        if len(reshapers)!=1 or reshapers[0].transform_name!=config.name or reshapers[0].schedule_gpu_transform is not None:
            raise ValueError("Official CPU distribution/quantile path differs")
        has_svd=config.global_transformer_name is not None and config.global_transformer_name!="None" and not config.differentiable
        if len(svds)!=(1 if has_svd else 0) or (has_svd and svds[0].global_transformer_name!=config.global_transformer_name):
            raise ValueError("Official CPU SVD path differs")
        if any(feature.scheduled_gpu_transform is not None for feature in member.feature_schema.features):
            raise ValueError("Unexpected scheduled quantile/SVD GPU transform")
        gpu=audit_legacy_gpu_pipeline(member.gpu_preprocessor,member.config.outlier_removal_std,bindings)
        gpu["full_pipeline_joblib_sha1"]=joblib.hash(member.gpu_preprocessor,hash_name="sha1")
        members.append(dict(X_train=execution.array_record(member.X_train),y_train=execution.array_record(member.y_train),
            config=execution.plain(member.config),feature_schema=execution.plain(member.feature_schema),
            feature_indices=execution.plain(member.feature_indices),
            cpu_preprocessor_joblib_sha1=joblib.hash(member.cpu_preprocessor,hash_name="sha1"),
            legacy_outlier_gpu_pipeline=gpu,cpu_quantile_svd_path_verified=True))
    input_names=("ordinal_encoder_","date_transformer_","text_transformer_","label_encoder_",
                 "inferred_feature_schema_","categorical_features_indices_")
    return dict(engine=type(engine).__name__,kv_cache_used=False,members=members,
        input_preprocessors={name:joblib.hash(getattr(model,name),hash_name="sha1") for name in input_names},
        pipeline_seeds=execution.array_record(ensemble.pipeline_seeds),
        ensemble_configs=execution.plain(model.ensemble_configs_),
        enable_gpu_quantile_svd_preprocessing=False,legacy_outlier_gpu_audit="exact_stateless_source_default")

_ACTIVE_CORE=None
def core():
    global _ACTIVE_CORE
    execution=previous().core()
    execution.ROUND=ROUND;execution.PLAN=PLAN;execution.DEPS=DEPS
    execution.__file__=str(Path(__file__).resolve());execution.verify=verify
    execution.preprocessing_fingerprint=preprocessing_fingerprint
    _ACTIVE_CORE=execution
    return execution

def preflight():
    started=time.monotonic()
    compile(Path(__file__).read_bytes(),str(Path(__file__)),"exec")
    previous().verify()
    tests=sentinel_tests()
    configured=core()
    if configured.preprocessing_fingerprint is not preprocessing_fingerprint or configured.verify is not verify:
        raise ValueError("Audit/verifier binding differs")
    if Path(configured.worker.__code__.co_filename).resolve()!=ROOT/"experiments/tabpfn_mps_runtime_gate.py":
        raise ValueError("Numerical worker is not frozen R57")
    if any(name in sys.modules for name in ("torch","numpy","scipy","sklearn","tabpfn")):
        raise ValueError("Scientific import occurred during pure preflight")
    for record in json.loads((ROUND/"source_proof.json").read_text()):
        if digest(ROUND/"public_sources"/record["path"])!=record["sha256"]:
            raise ValueError("Retained author source proof changed")
        original=ROOT/"data/control_tabpfn_mps_v1/runtime_deps/tabpfn"/record["path"]
        if digest(original)!=record["sha256"]:raise ValueError("Author source/wheel proof differs")
    if (ROUND/"hardware_probe.json").read_bytes()!=(PREVIOUS/"hardware_probe.json").read_bytes():
        raise ValueError("Original hardware memory/seed settings changed")
    save(ROUND/"source_preflight.json",dict(passed=True,seconds=time.monotonic()-started,
        implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),pure_sentinel_tests=tests,
        all_original_r57_r59_frozen_inputs_verified=True,unchanged_core_binding_verified=True,
        scientific_imports=False,model_forward=False))
    print("R60 source/pure-sentinel preflight passed; no scientific imports/model calls",flush=True)

def freeze():
    review=json.loads((ROUND/"independent_review.json").read_text())
    if not review["approved_for_freeze"] or review["implementation_sha256"]!=digest(__file__) or review["plan_sha256"]!=digest(PLAN):
        raise ValueError("Independent review must bind final R60 validation source/plan")
    previous().verify()
    checks={str(Path(__file__).relative_to(ROOT)):digest(__file__),str(PLAN.relative_to(ROOT)):digest(PLAN)}
    for path in sorted(ROUND.rglob("*")):
        if "protocol_freeze" in path.parts or "runtime_assessment" in path.parts:continue
        if path.is_file():checks[str(path.relative_to(ROOT))]=digest(path)
    for path in [PREVIOUS/"protocol_freeze/implementation.json",PREVIOUS/"protocol_freeze/symlinks.json",
        PREVIOUS/"protocol_freeze/freeze.json",PREVIOUS/"runtime_assessment/summary.json",
        PREVIOUS/"runtime_assessment/moderate/first/failure.json",
        PREVIOUS/"runtime_assessment/moderate/first.stderr.log"]:
        checks[str(path.relative_to(ROOT))]=digest(path)
    folder=ROUND/"protocol_freeze";folder.mkdir(exist_ok=False)
    (folder/"plan.json").write_bytes(PLAN.read_bytes());save(folder/"implementation.json",checks)
    save(folder/"freeze.json",dict(implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),
        previous_r59_source_sha256=PREVIOUS_SHA256,bound_files=len(checks),
        original_r57_r59_failures_retained=True,validation_only_change=True,
        dependency_model_math_settings_changed=False,scientific_imports_started=False,model_forward_started=False))
    print("Frozen R60 observational audit correction; no model call",flush=True)

def verify():
    folder=ROUND/"protocol_freeze"
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes():raise ValueError("R60 plan changed")
    for path,expected in json.loads((folder/"implementation.json").read_text()).items():
        if digest(ROOT/path)!=expected:raise ValueError("Frozen R60 audit input changed: "+path)
    previous().verify()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument("--preflight",action="store_true");action.add_argument("--freeze",action="store_true")
    action.add_argument("--verify",action="store_true");action.add_argument("--worker",action="store_true")
    parser.add_argument("--size",choices=("moderate","full"));parser.add_argument("--stage",choices=("first","rebuild"))
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    output=(args.output or ROUND/"runtime_assessment").resolve()
    if not output.is_relative_to(ROUND.resolve()):parser.error("Artifacts must stay in R60")
    if args.preflight:preflight()
    elif args.freeze:freeze()
    elif args.verify:verify();print("R60 and all prior frozen inputs verified",flush=True)
    else:
        execution=core()
        if args.worker:
            if not args.size or not args.stage:parser.error("Worker requires size/stage")
            try:execution.worker(args.size,args.stage,output)
            except BaseException as error:
                if output.exists():save(output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                    synthetic_only=True,competition_inputs_read=False,previous_failures_retained=True))
                raise
        else:execution.main(output)
