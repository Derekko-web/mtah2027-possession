"""R48: fixed CPU stage chunks, synthetic parity and unchanged resource limits.

The acquired implementation is never edited. Only column/row inference managers
receive a fixed execution schedule; ICL retains the full training context.
Resource supervision follows the already frozen R45 supervisor.
"""
import argparse
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch

import tabicl_runtime_gate as original

ROOT = original.ROOT
ROUND = ROOT / "data/control_tabicl_chunked_v1"
PLAN = ROOT / "experiments/TABICL_CHUNKED_RUNTIME_PLAN.json"
PYTHON = original.PYTHON
LIMIT_BYTES = original.LIMIT_BYTES
PREDICT_LIMIT_SECONDS = original.PREDICT_LIMIT_SECONDS
MAX_DIFFERENCE = 2e-5
MEAN_DIFFERENCE = 2e-6
digest = original.digest
write_json = original.write_json
event = original.event


def freeze():
    original.verify()
    folder = ROUND / "protocol_freeze"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "plan.json").write_bytes(PLAN.read_bytes())
    paths = [PLAN, Path(__file__).resolve(), ROOT / "experiments/TABICL_MEMORY_RESEARCH.md",
             original.ROUND / "protocol_freeze/implementation.json",
             original.ROUND / "protocol_freeze/readonly_dependencies.json"]
    for size in ("small", "large"):
        for name in ("train.npy", "query.npy", "target.npy"):
            paths.append(original.ROUND / "runtime_assessment" / size / name)
    paths.append(original.ROUND / "runtime_assessment/small/probability_first.npy")
    write_json(folder / "implementation.json", {
        str(path.relative_to(ROOT)): digest(path) for path in paths})
    write_json(folder / "freeze.json", {
        "plan_sha256": digest(PLAN), "script_sha256": digest(__file__),
        "original_configuration_still_failed": True,
        "model_inference_started": False, "synthetic_only": True})
    print("R48 plan, adapter, original dependencies and synthetic witnesses frozen", flush=True)


def verify():
    original.verify()
    folder = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (folder / "plan.json").read_bytes():
        raise ValueError("R48 plan changed")
    for name, expected in json.loads((folder / "implementation.json").read_text()).items():
        if digest(ROOT / name) != expected:
            raise ValueError("R48 frozen dependency changed: " + name)


def manager_class():
    """Define the adapter after runtime paths are configured, not in supervisor."""
    import torch
    from tabicl._model.inference import InferenceManager

    class CPUChunkManager(InferenceManager):
        def __init__(self, enc_name, out_dim, out_no_seq, limit):
            super().__init__(enc_name=enc_name, out_dim=out_dim, out_no_seq=out_no_seq)
            self.limit = limit
            self.trace = []

        def __call__(self, forward_fn, inputs, auto_batch=True, output_repeat=1):
            if not self._is_configured:
                raise RuntimeError("Chunk manager is not configured")
            if self.exe_device.type != "cpu" or self.use_amp or self.use_fa3:
                raise ValueError("R48 supports CPU float32 without AMP/FA3 only")
            if not auto_batch:
                return self._run_forward(forward_fn, self._prepare_inputs(inputs))
            first = next(iter(inputs.values()))
            if not isinstance(first, torch.Tensor) or first.ndim < 3 or first.dtype != torch.float32:
                raise ValueError("Expected CPU float32 stage input")
            batch_dims = tuple(first.shape[:-2])
            sequence, _ = first.shape[-2:]
            shape = (*batch_dims, self.out_dim) if self.out_no_seq else (*batch_dims, sequence, self.out_dim)
            splits = self.compute_split_sizes(batch_dims, self.limit)
            output = torch.empty(shape, dtype=torch.float32, device="cpu")
            chunks = []
            for values, index in self.create_multidim_batches(inputs, batch_dims, splits):
                result = self._run_forward(forward_fn, values)
                if result.dtype != torch.float32 or result.shape != output[index].shape:
                    raise ValueError("Chunk output shape/dtype differs")
                output[index].copy_(result)
                chunks.append(list(result.shape))
                del result, values
            self.trace.append({"input_shape": list(first.shape), "output_shape": list(shape),
                               "limit": self.limit, "split_sizes": splits, "chunks": chunks})
            return output

    return CPUChunkManager


def install_adapter(classifier):
    cls = manager_class()
    for module, expected, limit in ((classifier.model_.col_embedder, "tf_col", 8),
                                     (classifier.model_.row_interactor, "tf_row", 128)):
        old = module.inference_mgr
        if old.enc_name != expected:
            raise ValueError("Unexpected original stage manager")
        module.inference_mgr = cls(old.enc_name, old.out_dim, old.out_no_seq, limit)
    if classifier.model_.icl_predictor.inference_mgr.enc_name != "tf_icl":
        raise ValueError("Unexpected ICL manager")


def logic_test():
    """Invented tensor callbacks verify slicing/coverage without a model forward."""
    original.verify()
    original.runtime_paths()
    import numpy as np
    import torch
    cls = manager_class()
    col = cls("tf_col", 7, False, 8)
    col.configure(device="cpu", use_amp=False, use_fa3=False)
    features = torch.arange(2*19*9*3, dtype=torch.float32).reshape(2, 19, 9, 3)
    labels = torch.arange(2*19*5, dtype=torch.float32).reshape(2, 19, 5)
    def columns(features, y_train, train_size):
        return (features[..., :1] + y_train.mean(-1)[..., None, None]).repeat_interleave(7, -1)
    args = OrderedDict(features=features, y_train=labels, train_size=5)
    np.testing.assert_array_equal(col(columns, args).numpy(), columns(**args).numpy())
    row = cls("tf_row", 5, True, 128)
    row.configure(device="cpu", use_amp=False, use_fa3=False)
    values = torch.arange(2*133*7*5, dtype=torch.float32).reshape(2, 133, 7, 5)
    np.testing.assert_array_equal(row(lambda embeddings: embeddings.sum(-2),
        OrderedDict(embeddings=values)).numpy(), values.sum(-2).numpy())
    print("Synthetic tensor-only chunk coverage, extra target dimensions and partial chunks pass", flush=True)


def capture_prediction(classifier, values, label, output, capture):
    import numpy as np
    stages, handles = {}, []
    if capture:
        for name, module in (("column", classifier.model_.col_embedder),
                             ("row", classifier.model_.row_interactor)):
            def hook(module, args, result, name=name):
                stages[name] = result.detach().cpu().numpy().copy()
            handles.append(module.register_forward_hook(hook))
    try:
        probability, elapsed = original.predict(classifier, values, label)
    finally:
        for handle in handles:
            handle.remove()
    np.save(output / (label + ".npy"), probability)
    for name, value in stages.items():
        np.save(output / (label + "_" + name + ".npy"), value)
    return probability, elapsed, stages


def difference(first, second, predictions=False):
    import numpy as np
    error = np.abs(np.asarray(first, float) - np.asarray(second, float))
    result = dict(max_absolute=float(error.max()), mean_absolute=float(error.mean()),
                  shape_exact=list(first.shape) == list(second.shape))
    result["within_tolerance"] = (result["shape_exact"] and
        result["max_absolute"] <= MAX_DIFFERENCE and result["mean_absolute"] <= MEAN_DIFFERENCE)
    if predictions:
        result["argmax_exact"] = bool(np.array_equal(first.argmax(-1), second.argmax(-1)))
        result["within_tolerance"] &= result["argmax_exact"]
    return result


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
        if size == "reload":
            first = output.parent / "small"
            values = np.load(first / "query.npy")
            classifier = TabICLClassifier.load(first / "estimator.pkl", device="cpu")
            install_adapter(classifier)
            actual, elapsed, _ = capture_prediction(classifier, values, "reload", output, False)
            checks = original.probability_checks(actual, len(values))
            checks["independent_reload_exact"] = bool(np.array_equal(actual, np.load(first / "chunked_first.npy")))
            summary = dict(checks=checks, predict_seconds=[elapsed])
        else:
            rows, columns, queries = {"small": (64, 8, 16), "medium": (512, 32, 64),
                                       "large": (15000, 166, 1500)}[size]
            generator = np.random.default_rng(42)
            train = generator.normal(size=(rows, columns)).astype(np.float32)
            values = generator.normal(size=(queries, columns)).astype(np.float32)
            train.flat[::37] = np.nan
            values.flat[::37] = np.nan
            target = (np.arange(rows) % 3).astype(np.int64)
            for name, value in (("train", train), ("query", values), ("target", target)):
                np.save(output / (name + ".npy"), value)
                if size in ("small", "large"):
                    np.testing.assert_array_equal(value, np.load(original.ROUND / "runtime_assessment" / size / (name + ".npy")))
            classifier = TabICLClassifier(n_estimators=1, batch_size=1, n_jobs=2, device="cpu",
                use_amp=False, use_fa3=False, random_state=42,
                checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
                model_path=str(original.MODEL), allow_auto_download=False)
            event("fit_start", size=size, train_shape=list(train.shape))
            classifier.fit(train, target)
            event("fit_end", size=size)
            checkpoint = torch.load(original.MODEL, map_location="cpu", weights_only=True)["state_dict"]
            state = classifier.model_.state_dict()
            exact = set(state) == set(checkpoint) and all(torch.equal(state[key], checkpoint[key]) for key in state)
            del checkpoint
            baseline, baseline_stages = None, {}
            if size != "large":
                baseline, _, baseline_stages = capture_prediction(classifier, values, "baseline", output, True)
                if size == "small":
                    np.testing.assert_array_equal(baseline, np.load(original.ROUND / "runtime_assessment/small/probability_first.npy"))
            install_adapter(classifier)
            first, seconds_first, stages = capture_prediction(classifier, values, "chunked_first", output, size != "large")
            second, seconds_second, _ = capture_prediction(classifier, values, "chunked_repeat", output, False)
            checks = original.probability_checks(first, queries)
            checks.update(checkpoint_tensors_exact=bool(exact),
                parameters_cpu_fp32=all(p.device.type == "cpu" and p.dtype == torch.float32 for p in classifier.model_.parameters()),
                repeat_exact=bool(np.array_equal(first, second)))
            parity = {}
            if baseline is not None:
                parity["probability"] = difference(baseline, first, True)
                for name in ("column", "row"):
                    parity[name] = difference(baseline_stages[name], stages[name])
                checks["baseline_probability_and_stages_within_tolerance"] = all(v["within_tolerance"] for v in parity.values())
            else:
                checks["first_predict_within_300s"] = seconds_first <= PREDICT_LIMIT_SECONDS
                checks["repeat_predict_within_300s"] = seconds_second <= PREDICT_LIMIT_SECONDS
            if size == "small":
                classifier.save(output / "estimator.pkl", save_model_weights=False,
                                save_training_data=True, save_kv_cache=False)
            trace = {name: module.inference_mgr.trace for name, module in
                (("column", classifier.model_.col_embedder), ("row", classifier.model_.row_interactor))}
            write_json(output / "chunk_trace.json", trace)
            summary = dict(checks=checks, parity=parity,
                train_shape=list(train.shape), query_shape=list(values.shape),
                predict_seconds=[seconds_first, seconds_second],
                configuration={k: str(v) for k, v in classifier.get_params().items()},
                chunk_limits={"column_groups": 8, "rows": 128},
                icl_full_context_unchanged=True)
        summary["checks"]["peak_within_6GiB"] = original.peak_bytes() <= LIMIT_BYTES
        summary.update(passed=all(summary["checks"].values()), peak_bytes=original.peak_bytes(),
            plan_sha256=digest(PLAN), script_sha256=digest(__file__),
            synthetic_only=True, competition_inputs_used=False, possession_scores_computed=False,
            network_forbidden=True)
        summary["artifact_sha256"] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
        write_json(output / "summary.json", summary)
        event("worker_complete", size=size, passed=summary["passed"], peak_bytes=summary["peak_bytes"])
        if not summary["passed"]:
            raise SystemExit("R48 fixed synthetic checks failed")


def supervise(size, output):
    """Reuse R45's exact watchdog by changing only the spawned script and verifier."""
    saved_file, saved_verify = original.__file__, original.verify
    # original.verify is called inside this verifier, so retain a non-recursive
    # binding while reusing the unchanged standard-library resource supervisor.
    def combined_verify():
        saved_verify()
        folder = ROUND / "protocol_freeze"
        if PLAN.read_bytes() != (folder / "plan.json").read_bytes():
            raise ValueError("R48 plan changed")
        for name, expected in json.loads((folder / "implementation.json").read_text()).items():
            if digest(ROOT / name) != expected:
                raise ValueError("R48 frozen source changed: " + name)
    original.__file__, original.verify = __file__, combined_verify
    try:
        return original.supervise(size, output)
    finally:
        original.__file__, original.verify = saved_file, saved_verify


def run(output):
    verify()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for size in ("small", "medium", "reload", "large"):
        record = supervise(size, output / size)
        records.append(record)
        if not record["passed"]:
            break
    result = dict(passed=len(records) == 4 and all(v["passed"] for v in records),
        records=records, plan_sha256=digest(PLAN), script_sha256=digest(__file__),
        synthetic_only=True, native_processing_started=False,
        possession_scores_computed=False, original_R45_remains_failed=True)
    write_json(output / "summary.json", result)
    if not result["passed"]:
        raise SystemExit("R48 fixed CPU chunk schedule rejected; no native trial")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--logic-test", action="store_true")
    parser.add_argument("--worker", choices=("small", "medium", "reload", "large"))
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
