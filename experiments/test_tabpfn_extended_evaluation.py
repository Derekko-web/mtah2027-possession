"""Pure contract tests for the TabPFN execution-only time extension."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch


def run_tests():
    path = Path(__file__).with_name("tabpfn_extended_evaluation.py")
    spec = importlib.util.spec_from_file_location("extended_tabpfn_subject", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outcomes = []
    def check(name, callback):
        callback()
        outcomes.append(dict(name=name, passed=True))
    def require(value):
        if not value:
            raise AssertionError("Extended evaluation contract failed")
    start = dict(monotonic_ns=10_000_000_000_000, utc_ns=10_000_000_000_000)
    for clock in start:
        for seconds in (-1, 3600):
            now = dict(start)
            now[clock] += seconds * 1_000_000_000
            check(f"{clock}_worker_{seconds}_rejects", lambda n=now: require(module.clock_abort(start, n, {}) is not None))
        now = dict(start)
        now[clock] += 901 * 1_000_000_000
        marker = dict(active=True, started=start, limit_seconds=900)
        check(f"{clock}_operation_cap_rejects", lambda n=now: require(module.clock_abort(start, n, marker) is not None))
    check("subhour_worker_allowed", lambda: require(module.clock_abort(start, {k:v+1800_000_000_000 for k,v in start.items()}, {}) is None))
    def operation():
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            delegate = Mock(return_value=("unchanged numerical result", .5))
            end = {k:v+1_000_000_000 for k,v in start.items()}
            with patch.object(module, "clock_pair", side_effect=[start,end]):
                assert module.extended_operation(delegate, output, "original_0", lambda:None) == ("unchanged numerical result",.5)
            assert delegate.call_count == 1
            event = json.loads((output/"calendar_operations.jsonl").read_text())
            assert event["status"] == "complete" and event["elapsed"] == dict(monotonic_ns=1_000_000_000,utc_ns=1_000_000_000)
            (output/"operations.jsonl").write_text(json.dumps(dict(name="original_0",active=False,elapsed=.5))+"\n")
            assert len(module.validate_operations(output)) == 1
    check("original_operation_called_once_and_both_clocks_retained", operation)
    def overdue():
        with tempfile.TemporaryDirectory() as folder:
            for clock in start:
                end = dict(start)
                end[clock] += 901_000_000_000
                with patch.object(module,"clock_pair",side_effect=[start,end]):
                    try:module.extended_operation(Mock(return_value=(1,.1)),Path(folder),"original_0",lambda:None)
                    except TimeoutError:pass
                    else:raise AssertionError("Either clock must reject an overrun")
    check("either_clock_operation_overrun_rejects_fast_original_timer", overdue)
    def delegate_exception():
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(module,"clock_pair",return_value=start):
                try:module.extended_operation(Mock(side_effect=RuntimeError("invented")),Path(folder),"fit_context",lambda:None)
                except RuntimeError:pass
                else:raise AssertionError("Original operation failure cannot be rescued")
            assert json.loads((Path(folder)/"calendar_operations.jsonl").read_text())["status"] == "failed"
    check("original_exception_propagates_and_failure_is_retained", delegate_exception)
    legacy = module.legacy()
    for name in ("worker","context_frames","context_record","query_indices","required_guards","verify_replay_evidence"):
        check(f"unchanged_original_{name}_source", lambda n=name: require(Path(getattr(legacy,n).__code__.co_filename).resolve()==module.LEGACY_SOURCE))
    def no_failed_reuse():
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"PREVIOUS",Path(folder)):
            original=Path(folder)/"screen/purged_spells/block5";original.mkdir(parents=True)
            module.write_json(original/"supervision.json",dict(passed=False,exit_code=-15))
            assert module.predecessor_worker("purged_spells",5) is None
            (original/"worker").mkdir();module.write_json(original/"worker/summary.json",dict(passed=True))
            assert module.predecessor_worker("purged_spells",5) is None
    check("failed_or_incomplete_predecessor_cannot_be_reused",no_failed_reuse)
    return outcomes


if __name__ == "__main__":
    print(json.dumps(dict(passed=True,tests=run_tests()),indent=2))
