"""User-authorized extension of the preserved TabPFN evaluation.

The original worker and numerical primitives are loaded unchanged. This wrapper
provides a separate round, longer dual-clock limits and verified reuse of the
nine successful original folds. The failed original attempt remains immutable.
"""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "data/control_tabpfn_extended_v1"
PLAN = ROOT / "experiments/TABPFN_EXTENDED_EVALUATION_PLAN.json"
PREVIOUS = ROOT / "data/control_tabpfn_native_v1"
LEGACY_SOURCE = ROOT / "experiments/tabpfn_native.py"
OPERATION_SECONDS, WORKER_SECONDS = 900, 3600
LIMIT_BYTES = 6 * 1024 ** 3
SOURCES = ("experiments/tabpfn_extended_evaluation.py", "experiments/research_tabpfn_extended.py",
           "experiments/test_tabpfn_extended_evaluation.py")
PUBLIC_START, PUBLIC_END = 11185, 18685
PROTOCOLS = ("buffered_minutes", "purged_spells", "buffered_100_seconds")
_LEGACY = None


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    write_json(temporary, value)
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def source_identity():
    return {name: digest(ROOT / name) for name in SOURCES}


def clock_pair():
    return dict(monotonic_ns=time.monotonic_ns(), utc_ns=time.time_ns())


def elapsed(start, end):
    return {key: end[key] - start[key] for key in ("monotonic_ns", "utc_ns")}


def within(values, limit):
    return (set(values) == {"monotonic_ns", "utc_ns"}
            and all(type(v) is int and 0 <= v < limit * 1_000_000_000 for v in values.values()))


def clock_abort(start, now, marker):
    if not within(elapsed(start, now), WORKER_SECONDS):
        return "worker_clock_invalid_or_over_3600seconds"
    if marker and marker.get("active"):
        if (marker.get("limit_seconds") != OPERATION_SECONDS
                or not within(elapsed(marker["started"], now), OPERATION_SECONDS)):
            return "operation_clock_invalid_or_over_900seconds"
    return None


def extended_operation(original, output, name, callback):
    """Keep the original synchronized/flush/dispatch operation inside both clocks."""
    start = clock_pair()
    marker = dict(name=name, active=True, started=start, limit_seconds=OPERATION_SECONDS)
    path = output / "calendar_operation.json"
    atomic_json(path, marker)
    status = "failed"
    try:
        value, active_seconds = original(output, name, callback)
        status = "complete"
    finally:
        end = clock_pair()
        record = dict(marker, active=False, ended=end, elapsed=elapsed(start, end), status=status)
        atomic_json(path, record)
        with (output / "calendar_operations.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    if not within(record["elapsed"], OPERATION_SECONDS):
        raise TimeoutError("Complete synchronized operation exceeded its extended dual-clock limit")
    return value, active_seconds


def validate_operations(folder):
    events = [json.loads(line) for line in (folder / "calendar_operations.jsonl").read_text().splitlines()]
    original = [json.loads(line) for line in (folder / "operations.jsonl").read_text().splitlines()]
    if not events or len(events) != len(original):
        raise ValueError("Every original operation needs one complete dual-clock record")
    for event, old in zip(events, original):
        if (event.get("name") != old["name"] or event.get("active") is not False
                or event.get("status") != "complete" or event.get("limit_seconds") != OPERATION_SECONDS
                or event.get("elapsed") != elapsed(event["started"], event["ended"])
                or not within(event["elapsed"], OPERATION_SECONDS)
                or old.get("active") is not False or not 0 <= old["elapsed"] < OPERATION_SECONDS):
            raise ValueError("Incomplete or over-limit operation evidence")
    if read(folder / "calendar_operation.json") != events[-1]:
        raise ValueError("Atomic dual-clock marker differs from its final completed event")
    return events


def legacy():
    global _LEGACY
    if _LEGACY is None:
        spec = importlib.util.spec_from_file_location("tabpfn_extended_preserved_native", LEGACY_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROUND, module.PLAN = ROUND, PLAN
        module.verify = verify
        module.native_core = native_core
        _LEGACY = module
    return _LEGACY


def technical_gate():
    return legacy().technical_gate()


def native_core():
    execution = technical_gate().core()
    execution.ROUND, execution.PLAN, execution.verify = ROUND, PLAN, verify
    execution.OPERATION_SECONDS, execution.WORKER_SECONDS = OPERATION_SECONDS, WORKER_SECONDS
    original = execution.operation
    execution.operation = lambda output, name, callback: extended_operation(original, output, name, callback)
    environment = execution.fixed_environment
    def same_environment():
        value = environment()
        value["TABPFN_MODEL_CACHE_DIR"] = str(PREVIOUS / "unused_model_cache")
        return value
    execution.fixed_environment = same_environment
    return execution


def verify_predecessor():
    plan = read(PLAN)
    for name, sha in plan["predecessor_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Preserved original TabPFN evidence changed: " + name)
    if read(PREVIOUS / "method_failure.json").get("closed") is not True:
        raise ValueError("Original TabPFN failed attempt must remain closed")


def verify():
    folder = ROUND / "protocol_freeze"
    if PLAN.read_bytes() != (folder / "plan.json").read_bytes():
        raise ValueError("Extended evaluation plan changed")
    for group in ("implementation.json", "input_sha256.json", "protected_artifacts.json", "head_runtime.json"):
        for name, sha in read(folder / group).items():
            if digest(ROOT / name) != sha:
                raise ValueError("Extended evaluation frozen binding changed: " + name)
    for name, target in read(folder / "symlinks.json").items():
        path = ROOT / name
        if not path.is_symlink() or str(path.readlink()) != target:
            raise ValueError("Preserved runtime/source link changed")
    if not (ROOT / "experiments/EXPERIMENT_LEDGER.csv").read_bytes().startswith((folder / "EXPERIMENT_LEDGER.csv").read_bytes()):
        raise ValueError("Preserved ledger prefix changed")
    verify_predecessor()
    technical_gate()
    if (ROUND / "global_failure.json").exists() or (ROUND / "method_failure.json").exists():
        raise ValueError("This extended evaluation attempt is closed; preserve its evidence")
    return read(PLAN)


def require_admission():
    value = read(ROUND / "resource_admission.json")
    if (value.get("approved_for_extended_evaluation") is not True
            or value.get("other_model_workers_stopped") is not True
            or value.get("freeze_sha256") != digest(ROUND / "protocol_freeze/freeze.json")):
        raise ValueError("Separate matching extended-evaluation admission is required")


def predecessor_worker(protocol, block):
    folder = PREVIOUS / "screen" / protocol / f"block{block}"
    if not (folder / "worker/summary.json").is_file():
        return None
    supervision, summary = read(folder / "supervision.json"), read(folder / "worker/summary.json")
    if (supervision.get("passed") is not True or supervision.get("exit_code") != 0
            or supervision.get("aborted") is not None or summary.get("passed") is not True
            or supervision.get("summary_sha256") != digest(folder / "worker/summary.json")):
        return None
    return folder


def reuse_completed(job_path, output):
    """Reuse only an exit0 original worker with regenerated byte-identical payloads."""
    job = read(job_path)
    if "parent" in job or job["seed"] != 42:
        return False
    origin = predecessor_worker(job["protocol"], job["block"])
    if origin is None:
        return False
    summary = read(origin / "worker/summary.json")
    for name in ("context", "target", "original", "swapped"):
        if Path(job[name]).read_bytes() != (origin / (name + ".npy")).read_bytes():
            raise ValueError("Regenerated predecessor payload differs: " + name)
    import numpy as np
    with np.load(job_path.parent / "fitting.npz", allow_pickle=False) as new, np.load(origin / "fitting.npz", allow_pickle=False) as old:
        if set(new.files) != set(old.files):
            raise ValueError("Predecessor fitting-role record differs")
        for key in old.files:
            np.testing.assert_array_equal(new[key], old[key])
    if (job_path.parent / "context_provenance.json").read_bytes() != (origin / "context_provenance.json").read_bytes():
        raise ValueError("Predecessor context provenance differs")
    for name, sha in summary["artifact_sha256"].items():
        if digest(origin / "worker" / name) != sha:
            raise ValueError("Successful predecessor worker artifact changed: " + name)
    shutil.copytree(origin / "worker", output)
    for name in ("supervision.json", "worker.stdout.log", "worker.stderr.log"):
        shutil.copyfile(origin / name, output.parent / name)
    write_json(output.parent / "reuse.json", dict(reused=True, fresh_model_fit=False,
        source=str(origin.relative_to(ROOT)), original_supervision_sha256=digest(origin / "supervision.json"),
        original_summary_sha256=digest(origin / "worker/summary.json"),
        regenerated_payloads_byte_exact=True, original_resource_timing_is_historical=True,
        original_monotonic_timing_is_not_claimed_as_UTC=True))
    return True


class NativeMethodFailure(RuntimeError):
    pass


def supervise(job_path, output):
    verify()
    require_admission()
    legacy().validate_job(read(job_path))
    if reuse_completed(job_path, output):
        print("Reused verified original worker", output.parent.parent.name, output.parent.name, flush=True)
        return read(output.parent / "supervision.json")
    execution = native_core()
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(("PYTHON", "HF_", "TABPFN_", "TORCH_", "PYTORCH_"))}
    environment.update(execution.fixed_environment())
    log, error = output.parent / "worker.stdout.log", output.parent / "worker.stderr.log"
    start, peak, aborted = clock_pair(), 0, None
    with log.open("xb") as stdout, error.open("xb") as stderr:
        process = subprocess.Popen([str(execution.PYTHON), "-P", "-S", "-B", str(Path(__file__).resolve()),
            "--worker", "--job", str(job_path), "--output", str(output)], cwd=ROUND, env=environment,
            stdout=stdout, stderr=stderr, start_new_session=True)
        while process.poll() is None:
            rss, _ = execution.owned_rss(process.pid)
            memory = execution.read_json_if_present(output / "memory.json")
            peak = max(peak, rss + memory.get("mps_driver_bytes", 0), memory.get("peak_conservative_combined_bytes", 0))
            marker = execution.read_json_if_present(output / "calendar_operation.json")
            aborted = clock_abort(start, clock_pair(), marker)
            if peak > LIMIT_BYTES:
                aborted = "owned_rss_plus_full_driver_over_6GiB"
            elif memory and time.monotonic() - memory["sampled"] > 5:
                aborted = "memory_monitor_stale"
            if aborted:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                break
            time.sleep(.1)
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait(timeout=5)
    summary = execution.read_json_if_present(output / "summary.json")
    memory = execution.read_json_if_present(output / "memory.json")
    end = clock_pair()
    peak = max(peak, memory.get("peak_conservative_combined_bytes", 0))
    passed = (code == 0 and aborted is None and summary.get("passed") is True and bool(memory)
              and peak <= LIMIT_BYTES and within(elapsed(start, end), WORKER_SECONDS)
              and not (output / "monitor_failure.json").exists())
    if passed:
        validate_operations(output)
    record = dict(exit_code=code, aborted=aborted, passed=passed, started=start, ended=end,
        elapsed=elapsed(start, end), peak_owned_rss_plus_full_driver_bytes=peak,
        elapsed_seconds=(end["monotonic_ns"] - start["monotonic_ns"]) / 1e9,
        native_public_processing=True, competition_inputs_read=True, synthetic_only=False,
        stdout_sha256=digest(log), stderr_sha256=digest(error),
        summary_sha256=digest(output / "summary.json") if (output / "summary.json").exists() else None)
    write_json(output.parent / "supervision.json", record)
    verify()
    if not passed:
        raise NativeMethodFailure("Extended TabPFN worker failed; complete evidence is retained")
    return record


def preflight():
    verify_predecessor()
    technical_gate()
    for name in SOURCES:
        source = (ROOT / name).read_bytes()
        compile(source, str(ROOT / name), "exec")
    spec = importlib.util.spec_from_file_location("tabpfn_extended_pure_tests", ROOT / SOURCES[2])
    tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tests)
    outcomes = tests.run_tests()
    if any(name in sys.modules for name in ("numpy", "pandas", "torch", "sklearn", "tabpfn")):
        raise ValueError("Source-only preflight imported a scientific library")
    write_json(ROUND / "source_preflight.json", dict(passed=True, source_sha256=source_identity(),
        plan_sha256=digest(PLAN), tests=outcomes, scientific_imports=False, model_calls=False,
        original_worker_source_unchanged=True, operation_seconds=OPERATION_SECONDS, worker_seconds=WORKER_SECONDS))


def freeze():
    preflight()
    review = read(ROUND / "root_review.json")
    if (review.get("approved_for_freeze") is not True or review.get("source_sha256") != source_identity()
            or review.get("plan_sha256") != digest(PLAN)):
        raise ValueError("Root review must bind the exact extension source and plan")
    folder = ROUND / "protocol_freeze"
    folder.mkdir(exist_ok=False)
    (folder / "plan.json").write_bytes(PLAN.read_bytes())
    closure = read(PREVIOUS / "protocol_freeze/implementation.json")
    closure.update(source_identity())
    for name in (PLAN, ROUND / "source_preflight.json", ROUND / "root_review.json", ROUND / "preparation_ledger.csv"):
        closure[str(name.relative_to(ROOT))] = digest(name)
    write_json(folder / "implementation.json", closure)
    write_json(folder / "input_sha256.json", read(PLAN)["input_sha256"])
    for name in ("protected_artifacts.json", "head_runtime.json", "symlinks.json"):
        (folder / name).write_bytes((PREVIOUS / "protocol_freeze" / name).read_bytes())
    (folder / "EXPERIMENT_LEDGER.csv").write_bytes((ROUND / "preparation_ledger.csv").read_bytes())
    write_json(folder / "freeze.json", dict(source_sha256=source_identity(), plan_sha256=digest(PLAN),
        user_authorized_longer_evaluation=True, old_attempt_unchanged=True, model_calls=False,
        operation_seconds=OPERATION_SECONDS, worker_seconds=WORKER_SECONDS))
    verify()


def __getattr__(name):
    return getattr(legacy(), name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight", "freeze", "verify", "worker"):
        actions.add_argument("--" + name, action="store_true")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        preflight()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
    else:
        require_admission()
        try:
            legacy().worker(args.job, args.output)
        except BaseException as error:
            if args.output and args.output.exists():
                write_json(args.output / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(),
                    native_public_processing=legacy()._PAYLOAD_ACCESS_STARTED, old_attempt_unchanged=True))
            raise
