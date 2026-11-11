"""R45 synthetic-only TabICL feasibility, offline replay and resource supervision.

Only files under this round, its frozen plan, and installed runtime dependencies
are read. No competition loader, label CSV or possession scorer is imported.
The supervisor is deliberately standard-library-only.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "data/control_tabicl_v1"
PLAN = ROOT / "experiments/TABICL_RUNTIME_PLAN.json"
PYTHON = ROOT / "data/control_sports_action_v1/runtime/bin/python"
DEPS = ROUND / "runtime_deps"
FALLBACK = ROOT / ".venv/lib/python3.12/site-packages"
MODEL = ROUND / "public_sources/tabicl-classifier-v2-20260212.ckpt"
LIMIT_BYTES = 6 * 1024**3
PREDICT_LIMIT_SECONDS = 300


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze():
    """Freeze exact acquired files and this implementation before any forward."""
    folder = ROUND / "protocol_freeze"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "plan.json").write_bytes(PLAN.read_bytes())
    acquisition = json.loads((ROUND / "acquisition.json").read_text())
    checks = {"experiments/tabicl_runtime_gate.py": digest(__file__),
              "experiments/TABICL_RUNTIME_PLAN.json": digest(PLAN),
              "data/control_tabicl_v1/acquisition.json": digest(ROUND / "acquisition.json")}
    for name, record in acquisition["files"].items():
        path = ROOT / name
        if digest(path) != record["sha256"]:
            raise ValueError("Acquired source changed: " + name)
        checks[name] = record["sha256"]
    trees = {}
    for label, base in (("target_dependencies", DEPS),):
        trees[label] = {str(path.relative_to(ROOT)): digest(path)
                        for path in sorted(base.rglob("*"))
                        if path.is_file() and "__pycache__" not in path.parts}
    checks.update(trees["target_dependencies"])
    # RECORD files and imported dependency content are both fixed. Resolve the
    # fallback distributions without adding their directory ahead of old NumPy.
    import importlib.metadata as metadata
    packages = ("numpy", "torch", "scipy", "scikit-learn", "joblib", "threadpoolctl",
                "tqdm", "huggingface-hub", "packaging", "typing_extensions", "pandas")
    installed = {}
    normal = list(sys.path)
    sys.path.append(str(FALLBACK))
    for name in packages:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        record = {"version": dist.version, "files": {}}
        for entry in dist.files or []:
            path = Path(dist.locate_file(entry))
            if path.is_file() and "__pycache__" not in path.parts and (
                path.suffix in (".py", ".so", ".dylib") or path.name in ("METADATA", "RECORD")
            ):
                record["files"][str(path.resolve())] = digest(path)
        installed[name] = record
    sys.path[:] = normal
    write_json(folder / "implementation.json", checks)
    write_json(folder / "readonly_dependencies.json", installed)
    write_json(folder / "freeze.json", {
        "plan_sha256": digest(PLAN), "implementation_sha256": digest(__file__),
        "acquisition_sha256": digest(ROUND / "acquisition.json"),
        "files": len(checks), "competition_data_used": False,
        "model_inference_started": False,
    })
    print(json.dumps({"event": "frozen", "files": len(checks)}), flush=True)


def verify():
    folder = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (folder / "plan.json").read_bytes():
        raise ValueError("R45 plan changed")
    for name, expected in json.loads((folder / "implementation.json").read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError("Frozen dependency changed: " + name)
    for package in json.loads((folder / "readonly_dependencies.json").read_text()).values():
        for name, expected in package["files"].items():
            if digest(name) != expected:
                raise ValueError("Read-only runtime dependency changed: " + name)


def runtime_paths():
    # Target-only packages take precedence, followed by the untouched sports
    # runtime's normal paths. Missing scientific dependencies come from .venv
    # last, keeping the sports runtime's NumPy2.1.2 and Torch2.5.0.
    sys.path.insert(0, str(DEPS))
    sys.path.append(str(FALLBACK))
    sys.dont_write_bytecode = True


def event(kind, **fields):
    print(json.dumps({"event": kind, "monotonic": time.monotonic(), **fields}), flush=True)


def peak_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def probability_checks(probability, rows):
    import numpy as np
    return {
        "shape": list(probability.shape) == [rows, 3],
        "finite": bool(np.isfinite(probability).all()),
        "nonnegative": bool((probability >= 0).all()),
        "normalized": bool(np.allclose(probability.sum(1), 1, atol=1e-6, rtol=0)),
    }


def predict(classifier, values, label):
    event("predict_start", label=label, rows=len(values))
    started = time.monotonic()
    result = classifier.predict_proba(values)
    elapsed = time.monotonic() - started
    event("predict_end", label=label, elapsed=elapsed, peak_bytes=peak_bytes())
    return result, elapsed


def worker(size, output):
    verify()
    runtime_paths()
    import importlib.metadata as metadata
    import numpy as np
    import torch
    from tabicl import TabICLClassifier

    for name, value in (("HF_HUB_OFFLINE", "1"), ("HF_DATASETS_OFFLINE", "1"),
                        ("WANDB_MODE", "disabled")):
        os.environ[name] = value
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.__version__ != "2.5.0" or np.__version__ != "2.1.2":
        raise ValueError("Original runtime Torch/NumPy versions changed")
    if metadata.version("tabicl") != "2.2.0":
        raise ValueError("Wrong TabICL release")

    output.mkdir(parents=True, exist_ok=False)

    def deny_network(*args, **kwargs):
        raise RuntimeError("Network forbidden during synthetic inference")

    with patch.object(socket.socket, "connect", deny_network):
        if size == "reload":
            first_folder = output.parent / "small"
            values = np.load(first_folder / "query.npy")
            saved = np.load(first_folder / "probability_first.npy")
            classifier = TabICLClassifier.load(first_folder / "estimator.pkl", device="cpu")
            actual, elapsed = predict(classifier, values, "independent_small_reload")
            np.save(output / "probability.npy", actual)
            checks = probability_checks(actual, len(values))
            checks["independent_reload_exact"] = bool(np.array_equal(saved, actual))
            checks["peak_within_6GiB"] = peak_bytes() <= LIMIT_BYTES
            summary = {"passed": all(checks.values()), "checks": checks,
                       "elapsed_seconds": elapsed, "peak_bytes": peak_bytes(),
                       "competition_inputs_used": False, "synthetic_only": True}
        else:
            train_rows, columns, query_rows = (64, 8, 16) if size == "small" else (15000, 166, 1500)
            generator = np.random.default_rng(42)
            train = generator.normal(size=(train_rows, columns)).astype(np.float32)
            values = generator.normal(size=(query_rows, columns)).astype(np.float32)
            train.flat[::37] = np.nan
            values.flat[::37] = np.nan
            target = (np.arange(train_rows) % 3).astype(np.int64)
            for name, value in (("train", train), ("query", values), ("target", target)):
                np.save(output / (name + ".npy"), value)
            classifier = TabICLClassifier(
                n_estimators=1, batch_size=1, n_jobs=2, device="cpu",
                use_amp=False, use_fa3=False, random_state=42,
                checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
                model_path=str(MODEL), allow_auto_download=False,
            )
            event("fit_start", size=size, train_shape=list(train.shape))
            started = time.monotonic()
            classifier.fit(train, target)
            fit_seconds = time.monotonic() - started
            event("fit_end", size=size, elapsed=fit_seconds)
            checks = {
                "class_order": bool(np.array_equal(classifier.classes_, np.arange(3))),
                "all_parameters_cpu_fp32": all(
                    p.device.type == "cpu" and p.dtype == torch.float32
                    for p in classifier.model_.parameters()),
            }
            # The unmodified public loader calls load_state_dict with its strict
            # default. Additionally compare every checkpoint tensor explicitly.
            checkpoint = torch.load(MODEL, map_location="cpu", weights_only=True)
            state = classifier.model_.state_dict()
            checks["strict_checkpoint_tensor_identity"] = (
                set(state) == set(checkpoint["state_dict"]) and all(
                    torch.equal(state[name], checkpoint["state_dict"][name]) for name in state))
            del checkpoint
            first, seconds_first = predict(classifier, values, size + "_first")
            np.save(output / "probability_first.npy", first)
            second, seconds_second = predict(classifier, values, size + "_repeat")
            np.save(output / "probability_repeat.npy", second)
            checks.update(probability_checks(first, query_rows))
            checks["repeat_exact"] = bool(np.array_equal(first, second))
            checks["peak_within_6GiB"] = peak_bytes() <= LIMIT_BYTES
            if size == "large":
                checks["first_predict_within_300s"] = seconds_first <= PREDICT_LIMIT_SECONDS
                checks["repeat_predict_within_300s"] = seconds_second <= PREDICT_LIMIT_SECONDS
            query_audit = {"tested": False}
            if size == "small":
                classifier.save(output / "estimator.pkl", save_model_weights=False,
                                save_training_data=True, save_kv_cache=False)
                left, _ = predict(classifier, values[:8], "query_prefix")
                reverse, _ = predict(classifier, values[::-1].copy(), "query_reverse")
                np.save(output / "probability_prefix.npy", left)
                np.save(output / "probability_reverse.npy", reverse)
                query_audit = {
                    "tested": True,
                    "prefix_exact": bool(np.array_equal(first[:8], left)),
                    "prefix_max_absolute_difference": float(np.max(np.abs(first[:8] - left))),
                    "permutation_exact": bool(np.array_equal(first, reverse[::-1])),
                    "permutation_max_absolute_difference": float(np.max(np.abs(first - reverse[::-1]))),
                    "not_an_accuracy_or_native_gate": True,
                }
            summary = {
                "passed": all(checks.values()), "checks": checks,
                "size": size, "train_shape": list(train.shape), "query_shape": list(values.shape),
                "synthetic_generator": "PCG64_rng42_Gaussian_float32_missing_flat_mod37_y_cycle3",
                "fit_seconds": fit_seconds, "predict_seconds": [seconds_first, seconds_second],
                "peak_bytes": peak_bytes(), "query_independence_diagnostic": query_audit,
                "versions": {n: metadata.version(n) for n in (
                    "tabicl", "torch", "numpy", "scipy", "scikit-learn", "einops", "psutil")},
                "model_config": classifier.model_config_,
                "checkpoint_state_keys": len(state),
                "parameter_count": sum(p.numel() for p in classifier.model_.parameters()),
                "configuration": {k: str(v) for k, v in classifier.get_params().items()},
                "checkpoint_sha256": digest(MODEL), "plan_sha256": digest(PLAN),
                "implementation_sha256": digest(__file__), "synthetic_only": True,
                "competition_inputs_used": False, "possession_scores_computed": False,
                "network_forbidden": True,
            }
        summary["artifact_sha256"] = {
            path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()}
        write_json(output / "summary.json", summary)
        event("worker_complete", size=size, passed=summary["passed"], peak_bytes=peak_bytes())
        if not summary["passed"]:
            raise SystemExit("Frozen synthetic feasibility checks failed")


def supervise(size, output):
    """Abort a worker if RSS or an active prediction exceeds the frozen gate."""
    verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout = output.parent / (size + ".stdout.log")
    stderr = output.parent / (size + ".stderr.log")
    if stdout.exists() or stderr.exists() or output.exists():
        raise ValueError("Refusing to overwrite existing worker artifacts")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                       HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="2",
                       OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    environment.pop("PYTHONPATH", None)
    started = time.monotonic()
    largest_rss = 0
    aborted = None
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen([str(PYTHON), "-B", str(Path(__file__).resolve()),
                                    "--worker", size, "--output", str(output)],
                                   cwd=ROOT, env=environment, stdout=out, stderr=err,
                                   start_new_session=True)
        while process.poll() is None:
            try:
                value = subprocess.check_output(["ps", "-o", "rss=", "-p", str(process.pid)], text=True)
                rss = int(value.strip() or 0) * 1024
            except (subprocess.CalledProcessError, ValueError):
                rss = 0
            largest_rss = max(largest_rss, rss)
            active = None
            for line in stdout.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("event") == "predict_start":
                    active = item
                elif item.get("event") == "predict_end":
                    active = None
            if rss > LIMIT_BYTES:
                aborted = {"reason": "rss_over_6GiB", "observed_bytes": rss}
            elif active is not None and time.monotonic() - active["monotonic"] > PREDICT_LIMIT_SECONDS:
                aborted = {"reason": "predict_over_300s", "active_predict": active["label"]}
            elif time.monotonic() - started > 1200:
                aborted = {"reason": "worker_wall_time_watchdog_1200s"}
            if aborted:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(.2)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    record = {
        "size": size, "exit_code": process.returncode, "aborted": aborted,
        "elapsed_wall_seconds": time.monotonic() - started,
        "largest_sampled_rss_bytes": largest_rss,
        "passed": process.returncode == 0 and summary is not None and summary["passed"] and aborted is None,
        "summary_sha256": digest(summary_path) if summary_path.exists() else None,
        "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr),
        "synthetic_only": True, "competition_inputs_used": False,
    }
    write_json(output.parent / (size + ".supervision.json"), record)
    print(json.dumps(record), flush=True)
    return record


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for size in ("small", "reload", "large"):
        record = supervise(size, output / size)
        records.append(record)
        if not record["passed"]:
            break
    result = {
        "passed": len(records) == 3 and all(r["passed"] for r in records),
        "records": records, "plan_sha256": digest(PLAN),
        "implementation_sha256": digest(__file__), "synthetic_only": True,
        "native_processing_started": False, "possession_scores_computed": False,
    }
    write_json(output / "summary.json", result)
    if not result["passed"]:
        raise SystemExit("R45 fixed CPU configuration rejected; no native trial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--worker", choices=("small", "reload", "large"))
    parser.add_argument("--output", type=Path, default=ROUND / "runtime_assessment")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.worker:
        worker(args.worker, args.output)
    else:
        run(args.output)
