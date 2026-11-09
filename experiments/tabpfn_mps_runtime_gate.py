"""R57: synthetic-only supported TabPFN-v2 MPS gate. No competition input imports.
Supervisor/freeze/preflight are standard-library-only. No automatic fallback.
Built with PriorLabs-TabPFN; exact author checkpoint and license remain read-only.
"""
import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "data/control_tabpfn_mps_v1"
PLAN = ROOT / "experiments/TABPFN_MPS_RUNTIME_PLAN.json"
PYTHON = ROOT / "data/control_sports_action_v1/runtime/bin/python"
DEPS = ROUND / "runtime_deps"
MODEL = ROOT / "data/control_tabpfn_v1/public_sources/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
LIMIT_BYTES = 6 * 1024**3
OPERATION_SECONDS = 300
WORKER_SECONDS = 900
MONITOR = None
DISPATCH = None
SERIAL_ENV = {name: "1" for name in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
SERIAL_ENV.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="",
    PYTHONNOUSERSITE="1", TABPFN_DISABLE_TELEMETRY="1", HF_HUB_OFFLINE="1",
    HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
    WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false", PYTORCH_ENABLE_MPS_FALLBACK="0",
    PYTORCH_MPS_FAST_MATH="0", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    TORCH_FORCE_WEIGHTS_ONLY_LOAD="0", TABPFN_MODEL_CACHE_SIZE="0")

def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)

def array_record(value):
    import numpy as np
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())

def synthetic_data(size):
    import numpy as np
    from sklearn.impute import SimpleImputer
    rows, queries = (64, 513) if size == "moderate" else (3000, 512)
    generator = np.random.default_rng(42)
    permutation = np.r_[np.arange(83, 166), np.arange(83)]
    base = generator.normal(size=(rows // 2, 166)).astype(np.float32)
    target = (base[:, 0] + base[:, 1] > 0).astype(np.int64)
    query = generator.normal(size=(queries, 166)).astype(np.float32)
    base.flat[::37], query.flat[::41] = np.nan, np.nan
    # Paired constant and all-empty columns exercise fixed-width imputation and
    # author constant/unique feature handling without selecting a real feature.
    for values in (base, query):
        values[:, [80, 163]] = 1
        values[:, [81, 164]] = 0
        values[:, [82, 165]] = np.nan
    raw_context = np.concatenate([base, base[:, permutation]])
    target = np.r_[target, 1 - target]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=False)
    context = imputer.fit_transform(raw_context).astype(np.float32)
    original = imputer.transform(query).astype(np.float32)
    swapped = imputer.transform(query[:, permutation]).astype(np.float32)
    np.testing.assert_array_equal(permutation[permutation], np.arange(166))
    np.testing.assert_array_equal(original[:, permutation], swapped)
    np.testing.assert_array_equal(np.bincount(target), [rows // 2, rows // 2])
    if context.shape != (rows, 166) or not np.isfinite(context).all():
        raise ValueError("Invented context/imputation differs")
    return dict(raw_context=raw_context, context=context, target=target, raw_query=query,
        original=original, swapped=swapped, imputer_medians=imputer.statistics_, permutation=permutation)

def probability_checks(probability, rows):
    import numpy as np
    return dict(shape=list(probability.shape) == [rows, 2], finite=bool(np.isfinite(probability).all()),
        nonnegative=bool((probability >= 0).all()),
        normalized=bool(np.allclose(probability.sum(axis=1), 1, atol=1e-6, rtol=0)))

def canonical_predict(model, query, output, name):
    import numpy as np
    probabilities, timings = [], []
    for start in range(0, len(query), 512):
        values = query[start:start + 512]
        count = len(values)
        if count < 512:
            values = np.concatenate([values, np.repeat(values[-1:], 512 - count, axis=0)])
        value, elapsed = operation(output, f"{name}_{start}", lambda: model.predict_proba(values))
        if not all(probability_checks(value, 512).values()):
            raise ValueError("Invalid canonical prediction")
        probabilities.append(value[:count])
        timings.append(dict(start=start, real_rows=count, call_rows=512, seconds=elapsed))
    return np.concatenate(probabilities), timings


def runtime_paths():
    """-P -S preserves PYTHONHASHSEED while excluding all existing package trees."""
    base = Path(sys.base_prefix) / "lib"
    sys.path[:] = [str(DEPS), str(base / "python312.zip"), str(base / "python3.12"),
                   str(base / "python3.12/lib-dynload")]
    sys.dont_write_bytecode = True


def fixed_environment():
    hardware = json.loads((ROUND / "hardware_probe.json").read_text())
    return {**SERIAL_ENV, "TABPFN_MPS_MEMORY_FRACTION": hardware["tabpfn_mps_memory_fraction"],
            "TABPFN_MODEL_CACHE_DIR": str(ROUND / "unused_model_cache")}


def freeze():
    acquisition = json.loads((ROUND / "acquisition.json").read_text())
    if not acquisition["dependency_closure_valid"] or acquisition["native_data_read"]:
        raise ValueError("Invalid isolated synthetic acquisition")
    review = json.loads((ROUND / "independent_review.json").read_text())
    if not review["approved_for_freeze"]:
        raise ValueError("Independent review is required before freezing")
    if review["implementation_sha256"] != digest(__file__) or review["plan_sha256"] != digest(PLAN):
        raise ValueError("Review does not bind the current source and plan")
    checks = {str(Path(__file__).relative_to(ROOT)): digest(__file__),
              str(PLAN.relative_to(ROOT)): digest(PLAN)}
    for name, record in acquisition["files"].items():
        if digest(ROOT / name) != record["sha256"]:
            raise ValueError("Acquisition changed: " + name)
    for path in sorted(ROUND.rglob("*")):
        if path.is_file() and "protocol_freeze" not in path.parts and "runtime_assessment" not in path.parts:
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                raise ValueError("Unexpected generated bytecode in isolated target")
            checks[str(path.relative_to(ROOT))] = digest(path)
    for name, record in acquisition["files"].items():
        checks[name] = record["sha256"]
    folder = ROUND / "protocol_freeze"
    folder.mkdir(exist_ok=False)
    (folder / "plan.json").write_bytes(PLAN.read_bytes())
    write_json(folder / "implementation.json", checks)
    write_json(folder / "freeze.json", dict(plan_sha256=digest(PLAN),
        implementation_sha256=digest(__file__), acquisition_sha256=digest(ROUND / "acquisition.json"),
        bound_files=len(checks), independent_review_sha256=digest(ROUND / "independent_review.json"),
        synthetic_only=True, competition_inputs_read=False, model_inference_started=False))
    print("Frozen", len(checks), "files; model inference has not started", flush=True)


def verify():
    folder = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (folder / "plan.json").read_bytes():
        raise ValueError("Synthetic MPS plan changed")
    for name, expected in json.loads((folder / "implementation.json").read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source/dependency changed: " + name)


def preflight():
    """Compile source and inspect metadata without importing a scientific package."""
    started = time.monotonic()
    compile(Path(__file__).read_text(), str(Path(__file__)), "exec")
    imports = {n.names[0].name.split(".")[0] for n in ast.parse(Path(__file__).read_text()).body
               if isinstance(n, ast.Import)}
    if imports & {"torch", "numpy", "sklearn", "tabpfn", "psutil"}:
        raise ValueError("Supervisor has a scientific top-level import")
    acquisition = json.loads((ROUND / "acquisition.json").read_text())
    distributions = {d.metadata["Name"].lower().replace("_", "-"): d
                     for d in metadata.distributions(path=[str(DEPS)])}
    for name, item in acquisition["resolved_runtime_distributions"].items():
        if distributions[name].version != item["version"]:
            raise ValueError("Metadata resolution changed: " + name)
    source = DEPS / "tabpfn"
    for path in source.rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    comparisons = json.loads((ROUND / "official_source_comparison.json").read_text())
    if not all(row["matches"] for row in comparisons):
        raise ValueError("Official wheel/source comparison failed")
    if any(name in sys.modules for name in ("torch", "numpy", "sklearn", "tabpfn", "psutil")):
        raise ValueError("A scientific library was imported during source preflight")
    write_json(ROUND / "source_preflight.json", dict(passed=True,
        seconds=time.monotonic()-started, scientific_imports=False, model_inference=False,
        implementation_sha256=digest(__file__), plan_sha256=digest(PLAN),
        isolated_distributions=len(distributions), official_sources_compared=len(comparisons)))
    print("Source/metadata preflight passed; no scientific imports or inference", flush=True)


def exact_array(actual, expected, label):
    import numpy as np
    if array_record(actual) != array_record(expected):
        raise ValueError("Byte-exact replay failed: " + label)
    np.testing.assert_array_equal(actual, expected)


class MemoryMonitor:
    """Monitor conservative owned RSS plus this process's full Metal driver memory."""
    def __init__(self, output):
        import psutil
        self.process = psutil.Process()
        self.output = output
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.peak = 0
        self.driver_peak = 0
        self.rss_peak = 0
        self.last = {}
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        import torch
        owned = [self.process, *self.process.children(recursive=True)]
        rss = sum(p.memory_info().rss for p in owned if p.is_running())
        driver = torch.mps.driver_allocated_memory()
        current = torch.mps.current_allocated_memory()
        combined = rss + driver
        with self.lock:
            self.peak = max(self.peak, combined)
            self.driver_peak = max(self.driver_peak, driver)
            self.rss_peak = max(self.rss_peak, rss)
            self.last = dict(sampled=time.monotonic(), owned_rss_bytes=rss,
                mps_driver_bytes=driver, mps_tensor_bytes=current, conservative_combined_bytes=combined,
                peak_conservative_combined_bytes=self.peak, peak_driver_bytes=self.driver_peak,
                peak_owned_rss_bytes=self.rss_peak)
            temporary = self.output / "memory.tmp"
            write_json(temporary, self.last)
            temporary.replace(self.output / "memory.json")
        if combined > LIMIT_BYTES:
            raise MemoryError("Owned RSS plus MPS driver allocation exceeded6GiB")
        return dict(self.last)

    def loop(self):
        while not self.stop_event.is_set():
            try:
                self.sample()
            except BaseException as error:
                write_json(self.output / "monitor_failure.json", dict(error=repr(error),
                    traceback=traceback.format_exc(), last=self.last))
                os._exit(86)
            self.stop_event.wait(.05)

    def start(self):
        self.sample()
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        return self.sample()


class NativeDispatchGuard:
    """Observe official dispatch/run calls and delegate unchanged to the author code."""
    def __init__(self, stack):
        self.calls = 0
        self.dispatches = 0
        self.shapes = Counter()
        self.lock = threading.Lock()
        sdpa = importlib.import_module("tabpfn.architectures.shared.scaled_dot_product_attention")
        backend_module = importlib.import_module("tabpfn.architectures.shared.torch_mps_backend")
        backend = backend_module.TORCH_MPS_BACKEND
        if not backend.is_available():
            raise ValueError("The frozen native torch-mps wrapper is unavailable")
        original_record = sdpa.record_attention_backend
        original_run = type(backend).run
        def checked_record(selected):
            if selected is not backend:
                raise ValueError("Unexpected attention wrapper; no MLX/CPU fallback is allowed")
            with self.lock: self.dispatches += 1
            return original_record(selected)
        def checked_run(instance, q, k, v, **kwargs):
            import torch
            if instance is not backend or any(t is None or t.device.type != "mps" or
                    t.dtype != torch.float32 for t in (q, k, v)):
                raise ValueError("Attention inputs are not the fixed MPSfloat32 interface")
            shape = json.dumps([list(q.shape), list(k.shape), list(v.shape)], separators=(",", ":"))
            with self.lock:
                self.calls += 1
                self.shapes[shape] += 1
            return original_run(instance, q, k, v, **kwargs)
        stack.enter_context(patch.object(sdpa, "record_attention_backend", checked_record))
        stack.enter_context(patch.object(type(backend), "run", checked_run))

    def record(self):
        with self.lock:
            return dict(wrapper="torch-mps", calls=self.calls, dispatches=self.dispatches,
                shapes=dict(sorted(self.shapes.items())), delegated_unchanged=True,
                internal_pytorch_kernel_profiled=False)


def operation(output, name, callback):
    import torch
    path, temporary = output / "operation.json", output / "operation.tmp"
    started = time.monotonic()
    record = dict(name=name, started=started, active=True)
    write_json(temporary, record)
    temporary.replace(path)
    print("Start", name, flush=True)
    torch.mps.synchronize()
    before = DISPATCH.record()
    MONITOR.sample()
    value = callback()
    torch.mps.synchronize()
    memory = MONITOR.sample()
    elapsed = time.monotonic() - started
    after = DISPATCH.record()
    calls, dispatches = after["calls"]-before["calls"], after["dispatches"]-before["dispatches"]
    if calls != dispatches or (not name.startswith(("fit_", "fingerprint_")) and calls == 0):
        raise ValueError("Prediction did not execute exclusively through the observed native MPS wrapper")
    record.update(active=False, elapsed=elapsed, memory=memory, native_wrapper_calls=calls,
                  native_dispatches=dispatches)
    write_json(temporary, record)
    temporary.replace(path)
    with (output / "operations.jsonl").open("a") as log:
        log.write(json.dumps(record, sort_keys=True) + "\n")
    if elapsed > OPERATION_SECONDS:
        raise TimeoutError("Operation exceeded300seconds: " + name)
    print("Complete", name, round(elapsed, 3), "seconds", flush=True)
    return value, elapsed


def preprocessing_fingerprint(model):
    import joblib
    engine = model.executor_
    if type(engine).__name__ != "InferenceEngineCachePreprocessing" or len(engine.ensemble_members) != 8:
        raise ValueError("Unexpected retained preprocessing/eight-view interface")
    members = []
    for member in engine.ensemble_members:
        if member.gpu_preprocessor is not None:
            raise ValueError("GPU preprocessing was unexpectedly enabled")
        members.append(dict(X_train=array_record(member.X_train), y_train=array_record(member.y_train),
            config=plain(member.config), feature_schema=plain(member.feature_schema),
            feature_indices=plain(member.feature_indices),
            cpu_preprocessor_joblib_sha1=joblib.hash(member.cpu_preprocessor, hash_name="sha1")))
    input_names = ("ordinal_encoder_", "date_transformer_", "text_transformer_", "label_encoder_",
                   "inferred_feature_schema_", "categorical_features_indices_")
    return dict(engine=type(engine).__name__, kv_cache_used=False, members=members,
        input_preprocessors={name: joblib.hash(getattr(model,name), hash_name="sha1") for name in input_names},
        pipeline_seeds=array_record(model.ensemble_preprocessor_.pipeline_seeds),
        ensemble_configs=plain(model.ensemble_configs_))


def checkpoint_fingerprint(model):
    import torch
    from tabpfn.architectures.tabpfn_v2 import _replace_keys_from_base_architecture
    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    expected = {k:v for k,v in checkpoint["state_dict"].items() if "criterion." not in k}
    if any(k.startswith(("transformer_encoder.", "decoder_dict.")) for k in expected):
        expected = _replace_keys_from_base_architecture(expected)
    if len(model.models_) != 1:
        raise ValueError("Unexpected number of checkpoint architectures")
    actual = model.models_[0].state_dict()
    if set(actual) != set(expected):
        raise ValueError("Rewritten v2 checkpoint tensor keys differ")
    for name in actual:
        exact_array(actual[name].detach().cpu().numpy(), expected[name].numpy(), "checkpoint:"+name)
    if not all(p.device.type == "mps" and p.dtype == torch.float32 for p in model.models_[0].parameters()):
        raise ValueError("The model is not MPSfloat32")
    return {name:array_record(value) for name,value in sorted(actual.items())}


def recipe_for(model, values):
    return dict(parameters=plain(model.get_params(deep=False)),
        architecture_configurations=plain(model.configs_), inference_configuration=plain(model.inference_config_),
        devices=plain(model.devices_), forced_dtype=str(model.forced_inference_dtype_),
        autocast=bool(model.use_autocast_), n_estimators=int(model.n_estimators_),
        context_shape=list(values["context"].shape), query_shape=list(values["original"].shape),
        model_sha256=digest(MODEL), fixed_environment=fixed_environment(),
        reconstruct_from="Fresh process: identical invented compact context,target,imputer,checkpoint,seed and isolated packages",
        kv_cache_used=False, fitted_estimator_pickle_used=False, synthetic_only=True,
        attribution="Built with PriorLabs-TabPFN", synthetic_swap="Feature i exchanges with i+83 modulo166")


def worker(size, stage, output):
    global MONITOR, DISPATCH
    verify()
    output.mkdir(parents=True, exist_ok=False)
    environment = fixed_environment()
    for name,value in environment.items():
        if os.environ.get(name) != value:
            raise ValueError("Missing fixed runtime environment: "+name)
    if sys.flags.ignore_environment or sys.flags.hash_randomization or not sys.flags.no_site or not sys.flags.safe_path:
        raise ValueError("Interpreter isolation or fixed hash seed differs")
    runtime_paths()
    denied = []
    def deny(*args, **kwargs):
        denied.append("network_call")
        raise RuntimeError("Network forbidden during the synthetic MPS gate")
    with ExitStack() as stack:
        for owner,name in ((socket.socket,"connect"),(socket.socket,"connect_ex"),
                (socket,"create_connection"),(urllib.request,"urlopen")):
            stack.enter_context(patch.object(owner,name,deny))
        import numpy as np
        import torch
        if str(torch.__version__) != "2.15.0" or np.__version__ != "2.1.2":
            raise ValueError("Frozen Torch/NumPy differs")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        hardware = json.loads((ROUND / "hardware_probe.json").read_text())
        if not torch.backends.mps.is_available() or torch.mps.recommended_max_memory() != hardware["recommended_max_memory_bytes"]:
            raise ValueError("MPS availability or recommended hardware memory changed")
        if hash("TabPFN-MPS-R57-fixed-hash-seed") != hardware["hash_probe"]:
            raise ValueError("Interpreter hash seed differs")
        MONITOR = MemoryMonitor(output)
        MONITOR.start()
        stack.callback(MONITOR.close)
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from tabpfn.settings import settings
        if repr(settings.tabpfn.mps_memory_fraction) != hardware["tabpfn_mps_memory_fraction"]:
            raise ValueError("TabPFN changed the pre-import memory fraction")
        DISPATCH = NativeDispatchGuard(stack)
        acquisition = json.loads((ROUND / "acquisition.json").read_text())
        for name,record in acquisition["resolved_runtime_distributions"].items():
            dist = metadata.distribution(name)
            if dist.version != record["version"] or Path(dist.locate_file("")).resolve() != DEPS.resolve():
                raise ValueError("Dependency escaped isolated target: "+name)
        if any(name.startswith(("posthog", "tabpfn_common_utils")) for name in sys.modules):
            raise ValueError("Unexpected telemetry package imported")
        values = synthetic_data(size)
        parent = output.parent / "first"
        for name,value in values.items():
            if stage == "rebuild": exact_array(value,np.load(parent / (name+".npy")),name)
            np.save(output / (name+".npy"),value)
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V2,
            model_path=MODEL, n_estimators=8, softmax_temperature=.9,
            balance_probabilities=False, average_before_softmax=False, device="mps",
            ignore_pretraining_limits=False, inference_precision=torch.float32,
            fit_mode="fit_preprocessors", memory_saving_mode=True, random_state=42,
            n_preprocessing_jobs=1, enable_torch_compile=False, tuning_config=None)
        _, fitting = operation(output,"fit_context",lambda:model.fit(values["context"],values["target"]))
        exact_array(model.classes_,np.array([0,1]),"class order")
        checkpoint,_ = operation(output,"fingerprint_initial_weights",lambda:checkpoint_fingerprint(model))
        preprocessing = preprocessing_fingerprint(model)
        recipe = recipe_for(model,values)
        if recipe["autocast"] or recipe["n_estimators"] != 8 or recipe["forced_dtype"] != "torch.float32":
            raise ValueError("Resolved inference settings differ")
        original,timing_original = canonical_predict(model,values["original"],output,"original")
        swapped,timing_swapped = canonical_predict(model,values["swapped"],output,"swapped")
        repeated,timing_repeat = canonical_predict(model,values["original"][:512],output,"repeat_first")
        exact_array(repeated,original[:512],"repeat_first")
        probability = (original+swapped[:,[1,0]])/2
        if preprocessing != preprocessing_fingerprint(model):
            raise ValueError("Query prediction changed retained preprocessing/context cache")
        after,_ = operation(output,"fingerprint_final_weights",lambda:checkpoint_fingerprint(model))
        if checkpoint != after: raise ValueError("Query prediction changed checkpoint tensors")
        for name,value in (("original_probability",original),("swapped_probability",swapped),
                ("probability",probability),("repeat_first",repeated)):
            np.save(output / (name+".npy"),value)
            if stage == "rebuild": exact_array(value,np.load(parent / (name+".npy")),name)
        for name,value in (("checkpoint_fingerprint.json",checkpoint),
                ("preprocessing_fingerprint.json",preprocessing),("recipe.json",recipe)):
            write_json(output / name,value)
            if stage == "rebuild" and value != json.loads((parent/name).read_text()):
                raise ValueError("Fresh-context reconstruction differs: "+name)
        query_audit = dict(tested=False)
        if size == "moderate":
            prefix,_ = operation(output,"diagnostic_prefix8",lambda:model.predict_proba(values["original"][:8]))
            reverse,_ = operation(output,"diagnostic_reverse512",lambda:model.predict_proba(values["original"][:512][::-1].copy()))
            query_audit = dict(tested=True,prefix_exact=array_record(prefix)==array_record(original[:8]),
                prefix_max_abs=float(np.max(np.abs(prefix-original[:8]))),
                reverse_exact=array_record(reverse[::-1])==array_record(original[:512]),
                reverse_max_abs=float(np.max(np.abs(reverse[::-1]-original[:512]))))
            np.save(output/"diagnostic_prefix8.npy",prefix)
            np.save(output/"diagnostic_reverse512.npy",reverse)
            if stage == "rebuild":
                exact_array(prefix,np.load(parent/"diagnostic_prefix8.npy"),"diagnostic_prefix8")
                exact_array(reverse,np.load(parent/"diagnostic_reverse512.npy"),"diagnostic_reverse512")
        final_weights,_ = operation(output,"fingerprint_after_diagnostics",lambda:checkpoint_fingerprint(model))
        if preprocessing != preprocessing_fingerprint(model) or checkpoint != final_weights:
            raise ValueError("Diagnostic calls changed retained cache or checkpoint tensors")
        if denied: raise ValueError("Offline model attempted network access")
        dispatch = DISPATCH.record()
        if dispatch["calls"] <= 0 or dispatch["calls"] != dispatch["dispatches"]:
            raise ValueError("Native wrapper dispatch was not demonstrated")
        write_json(output/"native_dispatch.json",dispatch)
        memory = MONITOR.sample()
        checks = probability_checks(probability,len(values["original"]))
        checks.update(class_order=True,eight_official_views=True,checkpoint_exact=True,
            repeat_byte_exact=True,retained_cache_immutable=True,native_wrapper_exclusive=True,
            telemetry_disabled=True,no_network_attempts=True,
            combined_memory_within_6GiB=memory["peak_conservative_combined_bytes"] <= LIMIT_BYTES)
        if stage == "rebuild": checks["fresh_process_rebuild_byte_exact"] = True
        write_json(output/"summary.json",dict(passed=all(checks.values()),checks=checks,size=size,stage=stage,
            memory=memory,fit_seconds=fitting,query_audit=query_audit,
            query_timings=dict(original=timing_original,swapped=timing_swapped,repeat=timing_repeat),
            synthetic_only=True,competition_inputs_read=False,
            artifact_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()
                             and p.name not in {"memory.json","memory.tmp"}}))
        # Keep the memory heartbeat live through final input verification too.
        verify()


def read_json_if_present(path):
    return json.loads(path.read_text()) if path.exists() else {}


def owned_rss(pid):
    """macOS process inventory: standard library supervisor has no Torch import."""
    result = subprocess.run(["/bin/ps","-axo","pid=,ppid=,rss="],check=True,capture_output=True,text=True)
    rows = [tuple(map(int,line.split())) for line in result.stdout.splitlines() if line.strip()]
    owned = {pid}
    for _ in range(len(rows)):
        before = len(owned)
        owned.update(p for p,parent,_ in rows if parent in owned)
        if len(owned) == before: break
    return sum(rss*1024 for p,_,rss in rows if p in owned),owned


def supervise(size,stage,output):
    folder = output/size
    folder.mkdir(exist_ok=True)
    destination = folder/stage
    log,error = folder/(stage+".stdout.log"),folder/(stage+".stderr.log")
    started,peak,aborted = time.monotonic(),0,None
    environment = {k:v for k,v in os.environ.items() if not k.startswith(("PYTHON", "HF_", "TABPFN_", "TORCH_", "PYTORCH_"))}
    environment.update(fixed_environment())
    with log.open("wb") as stdout,error.open("wb") as stderr:
        process = subprocess.Popen([str(PYTHON),"-P","-S","-B",str(Path(__file__).resolve()),
            "--worker","--size",size,"--stage",stage,"--output",str(destination)],
            cwd=ROUND,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
        while process.poll() is None:
            rss,owned = owned_rss(process.pid)
            memory = read_json_if_present(destination/"memory.json")
            driver = memory.get("mps_driver_bytes",0)
            combined = rss+driver
            peak = max(peak,combined,memory.get("peak_conservative_combined_bytes",0))
            record = read_json_if_present(destination/"operation.json")
            elapsed = time.monotonic()-started
            if combined > LIMIT_BYTES or peak > LIMIT_BYTES:
                aborted = "owned_rss_plus_mps_driver_over_6GiB"
            elif record.get("active") and time.monotonic()-record["started"] > OPERATION_SECONDS:
                aborted = "operation_over_300seconds"
            elif elapsed > WORKER_SECONDS:
                aborted = "worker_over_900seconds"
            elif memory and time.monotonic()-memory["sampled"] > 5:
                aborted = "metal_memory_monitor_stale_over_5seconds"
            if aborted:
                import signal
                try: os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                break
            time.sleep(.1)
        try: code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            import signal
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            code = process.wait(timeout=5)
    summary = read_json_if_present(destination/"summary.json")
    last_memory = read_json_if_present(destination/"memory.json")
    peak = max(peak,last_memory.get("peak_conservative_combined_bytes",0))
    monitor_failed = (destination/"monitor_failure.json").exists()
    record = dict(size=size,stage=stage,exit_code=code,aborted=aborted,
        passed=code==0 and aborted is None and not monitor_failed and summary.get("passed",False)
            and peak<=LIMIT_BYTES and bool(last_memory),
        peak_conservative_owned_rss_plus_driver_bytes=peak,last_memory=last_memory,
        elapsed_wall_seconds=time.monotonic()-started,monitor_failed=monitor_failed,
        stdout_sha256=digest(log),stderr_sha256=digest(error))
    write_json(folder/(stage+".supervision.json"),record)
    print(size,stage,"passed",record["passed"],"combinedPeakGiB",round(peak/1024**3,3),flush=True)
    return record


def main(output):
    verify()
    output.mkdir(parents=True,exist_ok=False)
    records = []
    for size in ("moderate","full"):
        for stage in ("first","rebuild"):
            records.append(supervise(size,stage,output))
            if not records[-1]["passed"]: break
        if not records[-1]["passed"]: break
    passed = len(records)==4 and all(r["passed"] for r in records)
    write_json(output/"summary.json",dict(passed=passed,records=records,plan_sha256=digest(PLAN),
        implementation_sha256=digest(__file__),synthetic_only=True,competition_inputs_read=False,
        native_processing_started=False,failed_stages_retained=True,alternate_settings_run=False,
        tolerance_or_csv_reproducibility_waiver=False))
    verify()
    if not passed: raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--freeze",action="store_true")
    action.add_argument("--verify",action="store_true")
    action.add_argument("--preflight",action="store_true")
    action.add_argument("--worker",action="store_true")
    parser.add_argument("--size",choices=("moderate","full"))
    parser.add_argument("--stage",choices=("first","rebuild"))
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    output = (args.output or ROUND/"runtime_assessment").resolve()
    if not output.is_relative_to(ROUND.resolve()): parser.error("Artifacts must stay in this round")
    if args.freeze: freeze()
    elif args.verify:
        verify()
        print("Frozen R57 MPS inputs verified; no model inference",flush=True)
    elif args.preflight: preflight()
    elif args.worker:
        if not args.size or not args.stage: parser.error("A worker requires size and stage")
        try: worker(args.size,args.stage,output)
        except BaseException as error:
            if output.exists():
                write_json(output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                    synthetic_only=True,competition_inputs_read=False))
            raise
    else: main(output)
