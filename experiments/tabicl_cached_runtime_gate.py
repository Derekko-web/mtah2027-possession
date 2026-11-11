"""R49: official eight-member cache runtime and compact cache regeneration.

No neural/cache implementation is modified. The sole persistence wrapper clears
one cache reference during official serialization and restores it in finally.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from unittest.mock import patch
import warnings

import tabicl_runtime_gate as original

ROOT = original.ROOT
ROUND = ROOT / "data/control_tabicl_cached_v1"
PLAN = ROOT / "experiments/TABICL_CACHED_RUNTIME_PLAN.json"
PYTHON = original.PYTHON
LIMIT_BYTES = original.LIMIT_BYTES
OPERATION_LIMIT_SECONDS = 300
COMPACT_LIMIT_BYTES = 64 * 1024**2
digest, write_json, event = original.digest, original.write_json, original.event


def freeze():
    original.verify()
    folder = ROUND / "protocol_freeze"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "plan.json").write_bytes(PLAN.read_bytes())
    paths = (PLAN, Path(__file__).resolve(),
        original.ROUND / "protocol_freeze/implementation.json",
        original.ROUND / "protocol_freeze/readonly_dependencies.json")
    write_json(folder / "implementation.json", {str(p.relative_to(ROOT)): digest(p) for p in paths})
    write_json(folder / "freeze.json", dict(plan_sha256=digest(PLAN), script_sha256=digest(__file__),
        model_inference_started=False, synthetic_only=True, acquired_implementation_unchanged=True))
    print("R49 fixed eight-member cache/storage protocol frozen before forwards", flush=True)


def verify():
    original.verify()
    folder = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (folder / "plan.json").read_bytes():
        raise ValueError("R49 plan changed")
    for name, expected in json.loads((folder / "implementation.json").read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError("R49 frozen source changed: " + name)


def median_fill(train, query):
    """Train-only medians, including an explicit zero for all-empty columns."""
    import numpy as np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.asarray(train, np.float64), axis=0).astype(np.float32)
    median[~np.isfinite(median)] = 0
    filled_train = np.where(np.isfinite(train), train, median).astype(np.float32)
    filled_query = np.where(np.isfinite(query), query, median).astype(np.float32)
    return filled_train, filled_query, median


def logic_test():
    original.verify()
    original.runtime_paths()
    import numpy as np
    train = np.array([[1., np.nan, 3.], [3., np.nan, np.nan]], np.float32)
    query = np.array([[np.nan, np.nan, np.nan], [99., 88., 77.]], np.float32)
    first, second, median = median_fill(train, query)
    np.testing.assert_array_equal(median, [2., 0., 3.])
    np.testing.assert_array_equal(first, [[1., 0., 3.], [3., 0., 3.]])
    np.testing.assert_array_equal(second, [[2., 0., 3.], [99., 88., 77.]])
    np.testing.assert_array_equal(median_fill(train, query * 7)[2], median)
    print("Train-only median and all-empty-column synthetic logic checks pass", flush=True)


def timed(label, callback):
    event("operation_start", label=label)
    started = time.monotonic()
    result = callback()
    elapsed = time.monotonic() - started
    event("operation_end", label=label, elapsed=elapsed, peak_bytes=original.peak_bytes())
    if elapsed > OPERATION_LIMIT_SECONDS:
        raise ValueError("Operation exceeds frozen 300-second limit: " + label)
    return result, elapsed


def probability_checks(probability, rows):
    import numpy as np
    return dict(shape=list(probability.shape) == [rows, 2], finite=bool(np.isfinite(probability).all()),
        nonnegative=bool((probability >= 0).all()),
        normalized=bool(np.allclose(probability.sum(1), 1., atol=1e-6, rtol=0)))


def cache_fingerprint(classifier, rows, columns):
    """Validate complete official cache structure, hashing one tensor at a time."""
    import hashlib
    import torch
    result, total_members, total_bytes = {}, 0, 0
    if not classifier.kv_cache or classifier.cache_mode_ != "kv" or not classifier.model_kv_cache_:
        raise ValueError("Official full KV cache is missing")
    for method, cache in classifier.model_kv_cache_.items():
        members, n, h = cache.train_shape
        if ((n, h, cache.num_classes) != (rows, columns, 2)
                or cache.row_repr is not None or cache.cache_type != "kv"):
            raise ValueError("Cache context/schema differs")
        if set(cache.col_cache.kv) != set(range(3)) or set(cache.icl_cache.kv) != set(range(12)):
            raise ValueError("Missing cache layer")
        group = dict(train_shape=list(cache.train_shape), num_classes=cache.num_classes,
                     cache_type=cache.cache_type, tensors={})
        total_members += members
        for label, entries, shape in (
            ("column", cache.col_cache.kv, (members, columns+4, 8, 128, 16)),
            ("icl", cache.icl_cache.kv, (members, 8, rows, 64))):
            for layer, entry in sorted(entries.items()):
                for part in ("key", "value"):
                    tensor = getattr(entry, part)
                    if tuple(tensor.shape) != shape or tensor.device.type != "cpu" or tensor.dtype != torch.float32:
                        raise ValueError("Cache tensor context/dtype/shape differs")
                    array = tensor.detach().contiguous().numpy()
                    sha = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
                    total_bytes += tensor.numel() * tensor.element_size()
                    group["tensors"][f"{label}.{layer}.{part}"] = dict(shape=list(shape), sha256=sha)
                    del array
        result[method] = group
    if total_members != 8:
        raise ValueError("Expected eight total ensemble members")
    return dict(methods=result, total_members=total_members, visible_tensor_bytes=total_bytes)


def checkpoint_check(classifier):
    import torch
    from tabicl._model.inference import InferenceManager
    model = classifier.model_
    for module in (model.col_embedder, model.row_interactor, model.icl_predictor):
        if type(module.inference_mgr) is not InferenceManager:
            raise ValueError("A non-official manager or chunk adapter was installed")
    state = torch.load(original.MODEL, map_location="cpu", weights_only=True)["state_dict"]
    actual = model.state_dict()
    if set(state) != set(actual) or not all(torch.equal(state[k], actual[k]) for k in state):
        raise ValueError("Model checkpoint tensors changed")
    if not all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("Model dtype/device changed")
    return True


def save_compact(classifier, path, rows, columns):
    before = cache_fingerprint(classifier, rows, columns)
    retained = classifier.model_kv_cache_
    try:
        classifier.model_kv_cache_ = None
        classifier.save(path, save_training_data=True, save_model_weights=False, save_kv_cache=False)
    finally:
        classifier.model_kv_cache_ = retained
    if classifier.model_kv_cache_ is not retained or cache_fingerprint(classifier, rows, columns) != before:
        raise ValueError("Persistence wrapper altered the live cache")
    if path.stat().st_size > COMPACT_LIMIT_BYTES:
        raise ValueError("Compact estimator exceeds 64MiB")
    return path.stat().st_size


def compare(first, second):
    import numpy as np
    error = np.abs(first.astype(float) - second.astype(float))
    result = dict(max_absolute=float(error.max()), mean_absolute=float(error.mean()),
                  argmax_exact=bool(np.array_equal(first.argmax(1), second.argmax(1))))
    result["passed"] = result["max_absolute"] <= 2e-5 and result["mean_absolute"] <= 2e-6 and result["argmax_exact"]
    return result


def make_classifier(cached):
    from tabicl import TabICLClassifier
    return TabICLClassifier(n_estimators=8, batch_size=1, kv_cache=cached,
        n_jobs=2, device="cpu", use_amp=False, use_fa3=False, random_state=42,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        model_path=str(original.MODEL), allow_auto_download=False)


def worker(size, output):
    verify()
    original.runtime_paths()
    import numpy as np
    import torch
    from tabicl import TabICLClassifier
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if torch.__version__ != "2.5.0" or np.__version__ != "2.1.2":
        raise ValueError("Original runtime versions changed")
    output.mkdir(parents=True, exist_ok=False)
    def deny_network(*args, **kwargs):
        raise RuntimeError("Network blocked during synthetic execution")
    with patch.object(socket.socket, "connect", deny_network):
        times, checks = {}, {}
        if size.endswith("_reload"):
            stage = size.removesuffix("_reload")
            source = output.parent / stage
            query = np.load(source / "query.npy")
            expected = np.load(source / "cached_first.npy")
            fingerprint = json.loads((source / "cache_fingerprint.json").read_text())
            source_summary = json.loads((source / "summary.json").read_text())
            rows, columns = source_summary["train_shape"]
            if stage != "large":
                classifier, times["load_full_cache"] = timed("load_full_cache", lambda: TabICLClassifier.load(source / "cached_estimator.pkl", device="cpu"))
                if cache_fingerprint(classifier, rows, columns) != fingerprint:
                    raise ValueError("Serialized full cache tensor identity differs")
                loaded, times["saved_cache_predict"] = timed("saved_cache_predict", lambda: classifier.predict_proba(query))
                checks["saved_cache_reload_exact"] = bool(np.array_equal(loaded, expected))
                checks["saved_cache_prediction_does_not_mutate_cache"] = cache_fingerprint(classifier, rows, columns) == fingerprint
                np.save(output / "saved_cache_probability.npy", loaded)
                del classifier
                gc.collect()
            classifier, times["load_compact"] = timed("load_compact", lambda: TabICLClassifier.load(source / "compact_estimator.pkl", device="cpu"))
            checks["compact_contains_no_cache"] = classifier.model_kv_cache_ is None
            _, times["rebuild_cache"] = timed("rebuild_cache", classifier._build_kv_cache)
            checks["regenerated_cache_tensors_exact"] = cache_fingerprint(classifier, rows, columns) == fingerprint
            actual, times["regenerated_predict"] = timed("regenerated_predict", lambda: classifier.predict_proba(query))
            np.save(output / "regenerated_probability.npy", actual)
            checks.update(probability_checks(actual, len(query)))
            checks["compact_regeneration_probability_exact"] = bool(np.array_equal(actual, expected))
            checks["checkpoint_tensors_exact"] = checkpoint_check(classifier)
            checks["prediction_does_not_mutate_cache"] = cache_fingerprint(classifier, rows, columns) == fingerprint
            details = dict(source_stage=stage, train_shape=[rows, columns], query_shape=list(query.shape))
        else:
            rows, columns, queries = {"small": (64, 8, 16), "medium": (512, 32, 64),
                                       "large": (3000, 166, 512)}[size]
            generator = np.random.default_rng(42)
            raw_train = generator.normal(size=(rows, columns)).astype(np.float32)
            raw_query = generator.normal(size=(queries, columns)).astype(np.float32)
            raw_train.flat[::37] = np.nan
            raw_query.flat[::37] = np.nan
            target = (np.arange(rows) % 2).astype(np.int64)
            train, query, median = median_fill(raw_train, raw_query)
            for name, array in (("raw_train", raw_train), ("raw_query", raw_query),
                                ("train", train), ("query", query), ("target", target), ("median", median)):
                np.save(output / (name + ".npy"), array)
            baseline = None
            if size != "large":
                model = make_classifier(False)
                _, times["uncached_fit"] = timed("uncached_fit", lambda: model.fit(train, target))
                baseline, times["uncached_predict"] = timed("uncached_predict", lambda: model.predict_proba(query))
                np.save(output / "uncached_probability.npy", baseline)
                del model
                gc.collect()
            classifier = make_classifier(True)
            _, times["cached_fit"] = timed("cached_fit", lambda: classifier.fit(train, target))
            if not np.array_equal(classifier.classes_, [0, 1]):
                raise ValueError("Binary synthetic class order differs")
            fingerprint = cache_fingerprint(classifier, rows, columns)
            write_json(output / "cache_fingerprint.json", fingerprint)
            first, times["cached_first"] = timed("cached_first", lambda: classifier.predict_proba(query))
            second, times["cached_repeat"] = timed("cached_repeat", lambda: classifier.predict_proba(query))
            np.save(output / "cached_first.npy", first)
            np.save(output / "cached_repeat.npy", second)
            checks.update(probability_checks(first, queries))
            checks["cached_repeat_exact"] = bool(np.array_equal(first, second))
            checks["checkpoint_tensors_exact"] = checkpoint_check(classifier)
            checks["prediction_does_not_mutate_cache"] = cache_fingerprint(classifier, rows, columns) == fingerprint
            parity = compare(baseline, first) if baseline is not None else None
            if parity is not None:
                checks["uncached_cached_parity"] = parity["passed"]
            storage = {}
            if size == "small":
                _, times["prove_official_false_flag"] = timed("prove_official_false_flag", lambda: classifier.save(output / "official_false_flag.pkl", save_training_data=True, save_model_weights=False, save_kv_cache=False))
                loaded, _ = timed("load_false_flag_proof", lambda: TabICLClassifier.load(output / "official_false_flag.pkl", device="cpu"))
                checks["official_false_flag_retains_cache_as_inspected"] = loaded.model_kv_cache_ is not None
                storage["official_false_flag_bytes"] = (output / "official_false_flag.pkl").stat().st_size
                del loaded
                gc.collect()
            if size != "large":
                _, times["save_full_cache"] = timed("save_full_cache", lambda: classifier.save(output / "cached_estimator.pkl", save_training_data=True, save_model_weights=False, save_kv_cache=True))
                storage["full_cache_bytes"] = (output / "cached_estimator.pkl").stat().st_size
            compact_bytes, times["save_compact"] = timed("save_compact", lambda: save_compact(classifier, output / "compact_estimator.pkl", rows, columns))
            storage["compact_bytes"] = compact_bytes
            checks["compact_within_64MiB"] = compact_bytes <= COMPACT_LIMIT_BYTES
            checks["all_persistence_preserves_original_cache"] = cache_fingerprint(classifier, rows, columns) == fingerprint
            details = dict(train_shape=list(train.shape), query_shape=list(query.shape), parity=parity,
                storage=storage, visible_cache_tensor_bytes=fingerprint["visible_tensor_bytes"],
                configuration={k: str(v) for k, v in classifier.get_params().items()},
                official_managers_only=True, model_implementation_unchanged=True)
        checks["peak_within_6GiB"] = original.peak_bytes() <= LIMIT_BYTES
        summary = dict(passed=all(checks.values()), checks=checks, times=times, **details,
            peak_bytes=original.peak_bytes(), plan_sha256=digest(PLAN), script_sha256=digest(__file__),
            synthetic_only=True, competition_inputs_used=False, possession_scores_computed=False,
            network_forbidden=True)
        summary["artifact_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
        write_json(output / "summary.json", summary)
        event("worker_complete", size=size, passed=summary["passed"], peak_bytes=summary["peak_bytes"])
        if not summary["passed"]:
            raise SystemExit("R49 fixed cache/storage gate failed")


def supervise(size, output):
    verify()
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = output.parent / (size + ".stdout.log"), output.parent / (size + ".stderr.log")
    if output.exists() or stdout.exists() or stderr.exists():
        raise ValueError("Refusing to overwrite synthetic artifacts")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2", VECLIB_MAXIMUM_THREADS="2")
    environment.pop("PYTHONPATH", None)
    started, largest, aborted = time.monotonic(), 0, None
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen([str(PYTHON), "-B", str(Path(__file__).resolve()), "--worker", size,
            "--output", str(output)], cwd=ROOT, env=environment, stdout=out, stderr=err, start_new_session=True)
        while process.poll() is None:
            try:
                rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(process.pid)], text=True).strip() or 0) * 1024
            except (subprocess.CalledProcessError, ValueError):
                rss = 0
            largest = max(largest, rss)
            active = None
            for line in stdout.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("event") == "operation_start":
                    active = item
                elif item.get("event") == "operation_end":
                    active = None
            if rss > LIMIT_BYTES:
                aborted = dict(reason="rss_over_6GiB", observed_bytes=rss)
            elif active is not None and time.monotonic() - active["monotonic"] > OPERATION_LIMIT_SECONDS:
                aborted = dict(reason="operation_over_300s", operation=active["label"])
            elif time.monotonic() - started > 1200:
                aborted = dict(reason="worker_wall_time_over_1200s")
            if aborted:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(.2)
    path = output / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else None
    record = dict(size=size, exit_code=process.returncode, aborted=aborted,
        elapsed_wall_seconds=time.monotonic()-started, largest_sampled_rss_bytes=largest,
        passed=process.returncode == 0 and summary is not None and summary["passed"] and aborted is None,
        summary_sha256=digest(path) if path.exists() else None, stdout_sha256=digest(stdout),
        stderr_sha256=digest(stderr), synthetic_only=True, competition_inputs_used=False)
    write_json(output.parent / (size + ".supervision.json"), record)
    print(json.dumps(record), flush=True)
    return record


def run(output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for size in ("small", "medium", "small_reload", "medium_reload", "large", "large_reload"):
        record = supervise(size, output / size)
        records.append(record)
        if not record["passed"]:
            break
    result = dict(passed=len(records) == 6 and all(v["passed"] for v in records), records=records,
        plan_sha256=digest(PLAN), script_sha256=digest(__file__), synthetic_only=True,
        native_processing_started=False, possession_scores_computed=False,
        prior_R45_R48_unchanged=True)
    write_json(output / "summary.json", result)
    if not result["passed"]:
        raise SystemExit("R49 fixed official cache/storage configuration rejected; no native trial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--logic-test", action="store_true")
    parser.add_argument("--worker", choices=("small", "medium", "small_reload", "medium_reload", "large", "large_reload"))
    parser.add_argument("--output", type=Path, default=ROUND / "runtime_assessment")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.logic_test:
        logic_test()
    elif args.worker:
        worker(args.worker, args.output)
    else:
        run(args.output)
