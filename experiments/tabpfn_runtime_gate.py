"""Synthetic-only TabPFN-v2 gate with isolated dependencies and fresh-context replay.

No competition feature loader, labels, model, candidate or scorer is imported.
The supervisor and freeze/verification paths use the standard library only.
Built with PriorLabs-TabPFN; exact author license is retained with acquisition.
"""

import argparse
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import urllib.request
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "data/control_tabpfn_v1"
PLAN = ROOT / "experiments/TABPFN_RUNTIME_PLAN.json"
PYTHON = ROOT / "data/control_sports_action_v1/runtime/bin/python"
SHARED = PYTHON.parent.parent / "lib/python3.12/site-packages"
DEPS = ROUND / "runtime_deps"
MODEL = ROUND / "public_sources/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
LIMIT_BYTES = 6 * 1024**3
OPERATION_SECONDS = 300
WORKER_SECONDS = 900
SERIAL_ENV = {name: "1" for name in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
SERIAL_ENV.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1",
    TABPFN_DISABLE_TELEMETRY="1", TABPFN_ALLOW_CPU_LARGE_DATASET="1",
    HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
    WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze():
    """Bind sources, every acquired/target file, and read-only runtime contents."""
    acquisition = json.loads((ROUND / "acquisition.json").read_text())
    if not acquisition["dependency_closure_valid"] or acquisition["native_data_read"]:
        raise ValueError("Invalid synthetic acquisition")
    checks = {str(Path(__file__).relative_to(ROOT)): digest(__file__),
              str(PLAN.relative_to(ROOT)): digest(PLAN),
              str((ROUND / "acquisition.json").relative_to(ROOT)): digest(ROUND / "acquisition.json")}
    for name, record in acquisition["files"].items():
        if digest(ROOT / name) != record["sha256"]:
            raise ValueError("Acquisition changed: " + name)
        checks[name] = record["sha256"]
    for path in sorted(DEPS.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            checks[str(path.relative_to(ROOT))] = digest(path)
    installed = {}
    for name, record in acquisition["resolved_runtime_distributions"].items():
        base = Path(record["base"])
        dist = next((d for d in metadata.distributions(path=[str(base)])
            if d.metadata["Name"].lower().replace("_", "-") == name), None)
        if dist is None or dist.version != record["version"]:
            raise ValueError("Resolved distribution changed: " + name)
        if base.resolve() == DEPS.resolve():
            continue
        files = {}
        for entry in dist.files or []:
            path = Path(dist.locate_file(entry)).resolve()
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[str(path)] = digest(path)
        if not files:
            raise ValueError("No source provenance for " + name)
        installed[name] = dict(version=dist.version, base=str(base), files=files)
    folder = ROUND / "protocol_freeze"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "plan.json").write_bytes(PLAN.read_bytes())
    write_json(folder / "implementation.json", checks)
    write_json(folder / "readonly_dependencies.json", installed)
    write_json(folder / "freeze.json", dict(plan_sha256=digest(PLAN),
        implementation_sha256=digest(__file__), acquisition_sha256=digest(ROUND / "acquisition.json"),
        bound_files=len(checks), readonly_distributions=len(installed),
        synthetic_only=True, competition_inputs_read=False, model_inference_started=False))
    print("Frozen", len(checks), "files and", len(installed), "read-only distributions", flush=True)


def verify():
    folder = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (folder / "plan.json").read_bytes():
        raise ValueError("Synthetic runtime plan changed")
    for name, expected in json.loads((folder / "implementation.json").read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen source/acquisition changed: " + name)
    for record in json.loads((folder / "readonly_dependencies.json").read_text()).values():
        for name, expected in record["files"].items():
            if digest(name) != expected:
                raise ValueError("Read-only runtime changed: " + name)


def runtime_paths():
    sys.path.insert(0, str(DEPS))
    # Never append .venv: its NumPy/sklearn/pandas differ from this resolution.
    if str(SHARED) not in sys.path:
        sys.path.append(str(SHARED))
    sys.dont_write_bytecode = True


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def operation(output, name, callback):
    path, temporary = output / "operation.json", output / "operation.tmp"
    start = time.monotonic()
    write_json(temporary, dict(name=name, started=start, active=True))
    temporary.replace(path)
    print("Start", name, flush=True)
    value = callback()
    elapsed = time.monotonic() - start
    write_json(temporary, dict(name=name, started=start, active=False, elapsed=elapsed))
    temporary.replace(path)
    if elapsed > OPERATION_SECONDS:
        raise ValueError("Operation exceeded300seconds: " + name)
    print("Complete", name, round(elapsed, 3), "seconds", flush=True)
    return value, elapsed


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


def preprocessing_fingerprint(model):
    """Fingerprint only the official retained training/preprocessor cache, no KV."""
    import joblib
    executor = model.executor_
    if type(executor).__name__ != "InferenceEngineCachePreprocessing":
        raise ValueError("Unexpected inference/cache interface")
    if not (len(executor.X_trains) == len(executor.y_trains) == len(executor.preprocessors) == 8):
        raise ValueError("The default eight prompts changed")
    return dict(engine=type(executor).__name__, kv_cache_used=False,
        X_trains=[array_record(x) for x in executor.X_trains],
        y_trains=[array_record(y) for y in executor.y_trains],
        categorical_indices=plain(executor.cat_ixs),
        configurations=plain(executor.ensemble_configs),
        preprocessors_joblib_sha1=[joblib.hash(p, hash_name="sha1") for p in executor.preprocessors],
        input_preprocessor_joblib_sha1=joblib.hash(model.preprocessor_, hash_name="sha1"))


def checkpoint_fingerprint(model):
    import torch
    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    actual, expected = model.model_.state_dict(), checkpoint["state_dict"]
    if set(actual) != set(expected) or not all(torch.equal(actual[k], expected[k]) for k in actual):
        raise ValueError("Official checkpoint tensors changed")
    if not all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.model_.parameters()):
        raise ValueError("The model is not CPUfloat32")
    return {name: array_record(value) for name, value in actual.items()}


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


def worker(size, stage, output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    for name, value in SERIAL_ENV.items():
        if os.environ.get(name) != value:
            raise ValueError("Missing fixed runtime environment: " + name)
    os.environ["TABPFN_STATE_DIR"] = str(output / "disabled_telemetry_state")
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(ROUND / "unused_model_cache")
    runtime_paths()
    denied = []
    def deny(*args, **kwargs):
        denied.append("network_call")
        raise RuntimeError("Network forbidden during synthetic inference")
    with ExitStack() as blocked:
        for owner, name in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                            (socket, "create_connection"), (urllib.request, "urlopen")):
            blocked.enter_context(patch.object(owner, name, deny))
        import numpy as np
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn_common_utils.telemetry.core.service import ProductTelemetry
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        if torch.__version__ != "2.5.0" or np.__version__ != "2.1.2":
            raise ValueError("Original Torch/NumPy changed")
        telemetry = ProductTelemetry()
        if telemetry.telemetry_enabled() or telemetry._posthog_client is not None:
            raise ValueError("Telemetry was not disabled")
        acquisition = json.loads((ROUND / "acquisition.json").read_text())
        for name, record in acquisition["resolved_runtime_distributions"].items():
            if metadata.version(name) != record["version"]:
                raise ValueError("Wrong resolved dependency: " + name)
        values = synthetic_data(size)
        parent = output.parent / "first"
        for name, value in values.items():
            if stage == "rebuild":
                np.testing.assert_array_equal(value, np.load(parent / (name + ".npy")))
            np.save(output / (name + ".npy"), value)
        # Refit the public API from the identical compact context; no estimator,
        # neural-cache or preprocessing pickle is loaded or written.
        model = TabPFNClassifier(n_estimators=8, softmax_temperature=.9,
            balance_probabilities=False, average_before_softmax=False,
            model_path=MODEL, device="cpu", ignore_pretraining_limits=False,
            inference_precision=torch.float32, fit_mode="fit_preprocessors",
            memory_saving_mode=4.0, random_state=42, n_jobs=1)
        _, fitting = operation(output, "fit_context", lambda: model.fit(values["context"], values["target"]))
        np.testing.assert_array_equal(model.classes_, [0, 1])
        checkpoint = checkpoint_fingerprint(model)
        preprocessing = preprocessing_fingerprint(model)
        original, timing_original = canonical_predict(model, values["original"], output, "original")
        swapped, timing_swapped = canonical_predict(model, values["swapped"], output, "swapped")
        repeated, timing_repeat = canonical_predict(model, values["original"][:512], output, "repeat_first")
        np.testing.assert_array_equal(repeated, original[:512])
        probability = (original + swapped[:, [1, 0]]) / 2
        if preprocessing != preprocessing_fingerprint(model):
            raise ValueError("Query prediction mutated the retained training/preprocessing cache")
        if checkpoint != checkpoint_fingerprint(model):
            raise ValueError("Query prediction changed checkpoint state")
        for name, value in (("original_probability", original), ("swapped_probability", swapped),
                            ("probability", probability), ("repeat_first", repeated)):
            np.save(output / (name + ".npy"), value)
            if stage == "rebuild":
                np.testing.assert_array_equal(value, np.load(parent / (name + ".npy")))
        write_json(output / "checkpoint_fingerprint.json", checkpoint)
        write_json(output / "preprocessing_fingerprint.json", preprocessing)
        if stage == "rebuild":
            for name, value in (("checkpoint_fingerprint.json", checkpoint),
                                ("preprocessing_fingerprint.json", preprocessing)):
                if value != json.loads((parent / name).read_text()):
                    raise ValueError("Independent context reconstruction differs: " + name)
        # Noncanonical prefix/order calls are reported only on the moderate
        # workload. They never choose settings or replace fixed512 final replay.
        query_audit = dict(tested=False)
        if size == "moderate":
            prefix, _ = operation(output, "diagnostic_prefix8", lambda: model.predict_proba(values["original"][:8]))
            reverse, _ = operation(output, "diagnostic_reverse512", lambda: model.predict_proba(values["original"][:512][::-1].copy()))
            query_audit = dict(tested=True, prefix_exact=bool(np.array_equal(prefix, original[:8])),
                prefix_max_abs=float(np.max(np.abs(prefix - original[:8]))),
                reverse_exact=bool(np.array_equal(reverse[::-1], original[:512])),
                reverse_max_abs=float(np.max(np.abs(reverse[::-1] - original[:512]))))
            np.save(output / "diagnostic_prefix8.npy", prefix)
            np.save(output / "diagnostic_reverse512.npy", reverse)
        if denied:
            raise ValueError("The offline model attempted network access")
        if preprocessing != preprocessing_fingerprint(model) or checkpoint != checkpoint_fingerprint(model):
            raise ValueError("Diagnostic query calls changed retained context or weights")
        checks = probability_checks(probability, len(values["original"]))
        checks.update(class_order=True, eight_official_prompts=True, checkpoint_exact=True,
            repeat_exact=True, context_cache_immutable=True,
            peak_within_6GiB=peak_bytes() <= LIMIT_BYTES, telemetry_disabled=True, no_network_attempts=True)
        if stage == "rebuild":
            checks["independent_rebuild_exact"] = True
        recipe = dict(parameters=plain(model.get_params(deep=False)),
            architecture_configuration=plain(model.config_),
            context_shape=list(values["context"].shape), query_shape=list(values["original"].shape),
            model_sha256=digest(MODEL), fixed_environment=SERIAL_ENV,
            reconstruct_from="Fresh fit(saved context,target), exact package/checkpoint/seed and serial environment",
            kv_cache_used=False, fitted_estimator_pickle_used=False,
            attribution="Built with PriorLabs-TabPFN", synthetic_swap="Feature i exchanges with i+83 modulo166")
        if stage == "rebuild" and recipe != json.loads((parent / "recipe.json").read_text()):
            raise ValueError("Reconstructed resolved parameters/architecture differ")
        write_json(output / "recipe.json", recipe)
        write_json(output / "summary.json", dict(passed=all(checks.values()), checks=checks,
            size=size, stage=stage, independent_rebuild_compared=stage == "rebuild",
            peak_bytes=peak_bytes(), fit_seconds=fitting,
            query_timings=dict(original=timing_original, swapped=timing_swapped, repeat=timing_repeat),
            query_audit=query_audit, synthetic_only=True, competition_inputs_read=False,
            artifact_sha256={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}))
    verify()


def supervise(size, stage, output):
    runtime_paths()
    import psutil
    folder = output / size
    folder.mkdir(exist_ok=True)
    destination = folder / stage
    log, error = folder / (stage + ".stdout.log"), folder / (stage + ".stderr.log")
    start, peak, aborted = time.monotonic(), 0, None
    with log.open("wb") as stdout, error.open("wb") as stderr:
        process = subprocess.Popen([str(PYTHON), "-B", str(Path(__file__).resolve()),
            "--worker", "--size", size, "--stage", stage, "--output", str(destination)],
            cwd=ROUND, env={**os.environ, **SERIAL_ENV}, stdout=stdout, stderr=stderr)
        handle = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                owned = [handle, *handle.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in owned if p.is_running())
            except psutil.NoSuchProcess:
                continue
            peak = max(peak, rss)
            status = destination / "operation.json"
            record = json.loads(status.read_text()) if status.exists() else {}
            if rss > LIMIT_BYTES:
                aborted = "owned_rss_over_6GiB"
            elif record.get("active") and time.monotonic() - record["started"] > OPERATION_SECONDS:
                aborted = "operation_over_300seconds"
            elif time.monotonic() - start > WORKER_SECONDS:
                aborted = "worker_over_900seconds"
            if aborted:
                for child in reversed(owned):
                    try:
                        if child.is_running():
                            child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(.1)
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=10)
    path = destination / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    record = dict(size=size, stage=stage, exit_code=code, aborted=aborted,
        passed=code == 0 and aborted is None and summary.get("passed", False) and peak <= LIMIT_BYTES,
        largest_sampled_owned_rss_bytes=peak, elapsed_wall_seconds=time.monotonic() - start,
        summary_sha256=digest(path) if path.exists() else None,
        stdout_sha256=digest(log), stderr_sha256=digest(error))
    write_json(folder / (stage + ".supervision.json"), record)
    print(size, stage, "passed", record["passed"], "peakGiB", round(peak / 1024**3, 3), flush=True)
    return record


def main(output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for size in ("moderate", "full"):
        for stage in ("first", "rebuild"):
            records.append(supervise(size, stage, output))
            if not records[-1]["passed"]:
                break
        if not records[-1]["passed"]:
            break
    passed = len(records) == 4 and all(record["passed"] for record in records)
    write_json(output / "summary.json", dict(passed=passed, records=records,
        plan_sha256=digest(PLAN), implementation_sha256=digest(__file__),
        synthetic_only=True, competition_inputs_read=False, native_processing_started=False,
        failed_stages_retained=True, alternate_settings_or_cache_modes_run=False))
    verify()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--worker", action="store_true")
    parser.add_argument("--size", choices=("moderate", "full"))
    parser.add_argument("--stage", choices=("first", "rebuild"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or ROUND / "runtime_assessment").resolve()
    if not output.is_relative_to(ROUND.resolve()):
        parser.error("All synthetic artifacts must stay within this round")
    if args.freeze:
        freeze()
    elif args.verify:
        verify()
        print("Frozen TabPFN synthetic inputs verified", flush=True)
    elif args.worker:
        if not args.size or not args.stage:
            parser.error("A worker needs --size and --stage")
        worker(args.size, args.stage, output)
    else:
        main(output)
