"""Read-only continuation audit of the frozen GP v2 invented-data evidence.

Uses the standard library and the frozen source-only validators. Never imports a
scientific library, decodes a model pickle, or starts a model fit.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "data/control_gaussian_process_runtime_v2"
SOURCE = ROOT / "experiments/gaussian_process_runtime_gate_v2.py"
SEMANTIC = ("recipe.json", "fit_record.json", "full_state.json", "posterior_traces.json",
            "original_probability.npy", "swapped_probability.npy", "owner.npy")
CHECKS = {"one_actual_author_fit", "two_actual_posterior_calls", "actual_convergence_observed",
          "finite_normalized_probabilities", "exact_first_repeat", "exact_team_swap",
          "raw_arrays_immutable", "full_fitted_preprocessing_kernel_RNG_state_immutable",
          "semantic_reload_exact", "canonical_reload_exact", "independent_refit_exact"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def npy_record(path):
    """Parse inert numeric NPY bytes, without scientific imports or pickle."""
    with Path(path).open("rb") as handle:
        require(handle.read(6) == b"\x93NUMPY", "Expected a numeric NPY file")
        version = handle.read(2)
        require(version in (b"\x01\x00", b"\x02\x00"), "Unexpected NPY version")
        width, fmt = (2, "<H") if version[0] == 1 else (4, "<I")
        length = struct.unpack(fmt, handle.read(width))[0]
        header = ast.literal_eval(handle.read(length).decode("latin1"))
        require(header["fortran_order"] is False and header["descr"] in ("<f8", "<i8"),
                "Only contiguous float64/int64 evidence is allowed")
        payload = handle.read()
    count = 1
    for dim in header["shape"]:
        count *= dim
    require(len(payload) == count * 8, "NPY payload length differs")
    return dict(shape=list(header["shape"]), dtype=header["descr"],
                sha256=hashlib.sha256(payload).hexdigest()), payload


def audit():
    spec = importlib.util.spec_from_file_location("gp_v2_readonly_audit_source", SOURCE)
    execution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(execution)
    execution.verify()
    summary_path = ROUND / "runtime_assessment/summary.json"
    summary = read(summary_path)
    require(summary["passed"] is True and summary["synthetic_only"] is True
            and summary["native_processing_started"] is False
            and summary["implementation_sha256"] == digest(SOURCE)
            and summary["plan_sha256"] == digest(execution.PLAN), "Top-level evidence differs")
    expected = [(size, stage) for size in ("moderate", "full") for stage in ("first", "rebuild")]
    require([(r["size"], r["stage"]) for r in summary["records"]] == expected,
            "Exactly four ordered stages are required")
    records = []
    for (size, stage), supervision in zip(expected, summary["records"]):
        folder = ROUND / "runtime_assessment" / size / stage
        record = read(folder / "summary.json")
        require(read(folder.with_suffix(".supervision.json")) == supervision
                and supervision["passed"] is True and supervision["exit_code"] == 0
                and supervision["aborted"] is None
                and supervision["summary_sha256"] == digest(folder / "summary.json"),
                "Unbound or unsuccessful supervision")
        require(supervision["elapsed"] == execution.clock_elapsed(supervision["started"], supervision["ended"])
                and all(execution.duration_ns_ok(v, 900) for v in supervision["elapsed"].values()),
                "Worker UTC/monotonic limits differ")
        checks = record["checks"]
        require(record["passed"] is True and record["size"] == size and record["stage"] == stage
                and record["synthetic_only"] is True and record["native_arrays_read"] is False
                and set(checks) == CHECKS and checks["independent_refit_exact"] is (stage == "rebuild")
                and all(v is True for k, v in checks.items() if k != "independent_refit_exact"),
                "The numerical/state/reload conjunction is incomplete")
        execution.validate_operation_evidence(folder, record)
        for kind in ("stdout", "stderr"):
            require(digest(folder.with_suffix("." + kind + ".log")) == supervision[kind + "_sha256"],
                    "Worker log changed")
        actual = {p.name for p in folder.iterdir() if p.is_file()} - {"summary.json", "operation.json"}
        require(set(record["artifact_sha256"]) == actual and set(record["exact_artifacts"]) == set(SEMANTIC),
                "Incomplete actual artifact closure")
        for name, sha in record["artifact_sha256"].items():
            require(Path(name).name == name and digest(folder / name) == sha, "Artifact changed: " + name)
        fit, traces, state = [read(folder / name) for name in ("fit_record.json", "posterior_traces.json", "full_state.json")]
        rows = 600 if size == "moderate" else 3000
        require(fit["actual_author_fit"] is True and fit["actual_posterior_calls"] == 2
                and fit["roles"] == execution.training_roles(rows) and fit["optimizer"] is None
                and fit["unit_amplitude"] == 1.0 and fit["class_order"] == [0, 1]
                and fit["posterior_contract"] == execution.posterior_source_contract(), "Actual GP recipe differs")
        for name, field in (("raw_context", "raw_X"), ("target", "y")):
            require(npy_record(folder / (name + ".npy"))[0] == fit[field], "Actual raw fit provenance differs")
        require(fit["semantic_sha256"] == hashlib.sha256(json.dumps(state, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest(), "Full fitted state digest differs")
        require(len(traces) == 2, "Exactly two actual posterior calls are required")
        for ordinal, trace in enumerate(traces):
            require(trace["ordinal"] == ordinal, "Posterior call order differs")
            execution.validate_posterior_trace(trace)
        probabilities = []
        for name in ("original_probability", "swapped_probability", "owner"):
            shape, payload = npy_record(folder / (name + ".npy"))
            require(shape["shape"] == [7500, 2], "Every invented query must be retained")
            values = list(struct.iter_unpack("<dd", payload))
            require(all(0 <= a <= 1 and 0 <= b <= 1 and abs(a + b - 1) < 1e-12 for a, b in values),
                    "Nonfinite or unnormalized probability")
            probabilities.append(values)
        require(all(c == (a + y) / 2 and d == (b + x) / 2
                    for ((a, b), (x, y), (c, d)) in zip(*probabilities)), "Swap-average differs")
        peak = max(record["peak_bytes"], supervision["largest_sampled_rss_bytes"])
        require(peak <= 6 * 1024 ** 3, "Conservative memory limit exceeded")
        records.append(dict(size=size, stage=stage, passed=True, actual_author_fits=1,
            actual_posterior_calls=2, iterations=[len(t["history"]) for t in traces],
            summary_sha256=digest(folder / "summary.json"),
            supervision_sha256=digest(folder.with_suffix(".supervision.json")),
            worker_elapsed=supervision["elapsed"], fit_elapsed=record["fit_elapsed"],
            query_elapsed=record["query_elapsed"], conservative_peak_bytes=peak))
    for size in ("moderate", "full"):
        for name in SEMANTIC:
            require((ROUND / "runtime_assessment" / size / "first" / name).read_bytes()
                    == (ROUND / "runtime_assessment" / size / "rebuild" / name).read_bytes(),
                    "Fresh semantic refit differs: " + name)
    require(not any(n in sys.modules for n in ("numpy", "pandas", "sklearn", "scipy", "torch", "joblib")),
            "Audit must remain source/evidence only")
    return dict(passed=True, actual_all_four_stage_pass=True, actual_author_fits=4,
        actual_posterior_calls=8, semantic_refit_comparisons=14, records=records,
        technical_summary_sha256=digest(summary_path), resource_admission_sha256=digest(ROUND / "resource_admission.json"),
        prior_actual_audit_sha256=digest(ROUND / "actual_gate_audit.json"),
        source_or_dependencies_changed=False, native_processing=False, scientific_imports=False,
        model_calls=False, review_kind="continuation engineer independent audit of prior engineer retained evidence",
        reviewer_source_sha256=digest(Path(__file__)),
        memory_measures=dict(conservative_observed_max_bytes=max(r["conservative_peak_bytes"] for r in records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit()
    if args.output:
        with args.output.open("x") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, sort_keys=True))
