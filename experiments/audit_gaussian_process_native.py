"""Audit completed GP screen evidence and optionally append its nine ledger rows.

No training, model unpickling, or restricted labels. Recomputes public official
scores, fold and segment diagnostics, all guards and worker evidence.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

import gaussian_process_native as native

ROOT, ROUND = native.ROOT, native.ROUND
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def audit():
    native.require_private_runtime(sys.prefix, sys.base_prefix, sys.executable, True)
    execution = native.technical_gate()
    freeze = ROUND / "protocol_freeze"
    require(native.PLAN.read_bytes() == (freeze / "plan.json").read_bytes(), "Frozen plan differs")
    for group in ("implementation.json", "input_sha256.json", "protected_artifacts.json", "technical_gate.json", "head_runtime.json"):
        for name, sha in read(freeze / group).items():
            require(native.digest(ROOT / name) == sha, "Frozen source/input/runtime changed: " + name)
    for name, target in read(freeze / "symlinks.json").items():
        require((ROOT / name).is_symlink() and str((ROOT / name).readlink()) == target, "Source link differs")
    ledger = ROOT / "experiments/EXPERIMENT_LEDGER.csv"
    require(ledger.read_bytes().startswith((freeze / "EXPERIMENT_LEDGER.csv").read_bytes()), "Ledger prefix differs")
    stage = ROUND / "screen"
    manifest = native.audit(stage)
    require(manifest["context_models"] == 13 and manifest["actual_author_fits"] == 13
            and manifest["actual_posterior_calls"] == 26 and manifest["seed"] == 42
            and manifest["replay"] is False, "Complete original GP screen provenance is missing")
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file() and p.name != "manifest.json"}
    require(set(manifest["cache_sha256"]) == actual, "Stage closure is incomplete")
    import numpy as np
    import pandas as pd
    from evaluation import possession_metrics as pm
    from dfl_signal_audit import matching

    labels = pm.read_label_csv(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    public = np.arange(native.PUBLIC_START, native.PUBLIC_END)
    np.testing.assert_array_equal(np.flatnonzero(labels != "X"), public)
    splits = read(ROOT / "data/advanced_validation/splits.json")
    workers = []
    for protocol in native.PROTOCOLS:
        for split in (s for s in splits if s["protocol"] == protocol):
            job_dir = stage / protocol / f"block{split['block']}"
            folder = job_dir / "worker"
            job, summary, supervision = [read(p) for p in (job_dir / "job.json", folder / "summary.json", job_dir / "supervision.json")]
            native.validate_job(job)
            require(job["seed"] == 42 and job["protocol"] == protocol and job["block"] == split["block"]
                    and "parent" not in job, "Original worker job provenance differs")
            require(summary["passed"] is True and summary["seed"] == 42 and summary["protocol"] == protocol
                    and summary["block"] == split["block"] and summary["actual_author_fits"] == 1
                    and summary["actual_posterior_calls"] == 2 and summary["fresh_context_replay"] is False
                    and set(summary["checks"]) == set(native.WORKER_CHECKS)
                    and all(v is True for v in summary["checks"].values()), "Incomplete actual GP worker checks")
            require(supervision["passed"] is True and supervision["exit_code"] == 0 and supervision["aborted"] is None
                    and supervision["summary_sha256"] == native.digest(folder / "summary.json")
                    and supervision["elapsed"] == execution.clock_elapsed(supervision["started"], supervision["ended"])
                    and all(execution.duration_ns_ok(v, 900) for v in supervision["elapsed"].values())
                    and max(supervision["peak_bytes"], summary["peak_bytes"]) <= native.LIMIT_BYTES,
                    "Unbound or out-of-budget worker")
            execution.validate_operation_evidence(folder, summary)
            native.validate_fit_evidence(execution, folder, 42, len(split["validation"]))
            artifacts = {p.name for p in folder.iterdir() if p.is_file()} - {"summary.json"}
            require(set(summary["artifact_sha256"]) == artifacts and native.REQUIRED_WORKER_ARTIFACTS <= artifacts,
                    "Worker artifact closure differs")
            for name, sha in summary["artifact_sha256"].items():
                require(native.digest(folder / name) == sha, "Worker artifact changed: " + name)
            for kind in ("stdout", "stderr"):
                require(supervision[kind + "_sha256"] == native.digest(job_dir / ("worker." + kind + ".log")), "Log changed")
            require(all(p["num_threads"] == 1 for p in read(folder / "threadpools.json")), "Numerical pool is not one thread")
            with np.load(job_dir / "fitting.npz", allow_pickle=False) as fitting:
                centers, observed = native.context_frames(split["train"], split["validation"], labels)
                for name, value in (("split_train", split["train"]), ("validation", split["validation"]),
                                    ("context_frames", centers), ("observation_eligible", observed)):
                    np.testing.assert_array_equal(fitting[name], value)
                targets = np.r_[(labels[centers] == "A").astype(np.int64), (labels[centers] != "A").astype(np.int64)]
                np.testing.assert_array_equal(fitting["target"], targets)
                np.testing.assert_array_equal(np.load(job["target"], allow_pickle=False), targets)
            fit = read(folder / "fit_record.json")
            for key, field in (("context", "raw_X"), ("target", "y")):
                require(execution.array_record(np.load(job[key], allow_pickle=False)) == fit[field], "Actual fit raw input differs")
            original, swapped, owner = [np.load(folder / (name + ".npy"), allow_pickle=False)
                                        for name in ("original_probability", "swapped_probability", "owner")]
            for value in (original, swapped, owner):
                execution.probability_checks(value, len(split["validation"]))
            np.testing.assert_array_equal(owner, (original + swapped[:, ::-1]) / 2)
            np.testing.assert_array_equal(np.load(folder / "repeat_first.npy", allow_pickle=False), original[:512])
            np.testing.assert_array_equal(np.load(folder / "repeat_first_swapped.npy", allow_pickle=False), swapped[:512])
            workers.append(dict(worker=str(folder.relative_to(stage)), actual_author_fits=1, actual_posterior_calls=2,
                context_rows=summary["context_rows"], query_rows=summary["query_rows"],
                worker_UTC_seconds=supervision["elapsed"]["utc_ns"] / 1e9,
                fit_UTC_seconds=summary["fit_elapsed"]["utc_ns"] / 1e9,
                query_UTC_seconds=summary["query_elapsed"]["utc_ns"] / 1e9,
                conservative_peak_bytes=max(summary["peak_bytes"], supervision["peak_bytes"])))
    require(len(workers) == 13, "Thirteen complete workers are required")
    scores, folds, segments = [pd.read_csv(stage / name) for name in ("scores.csv", "fold_scores.csv", "segment_diagnostics.csv")]
    require(len(scores) == 9 and len(folds) == 39, "Incomplete metric cohort")
    selection = read(stage / "selection.json")
    comparisons = []
    for protocol in native.PROTOCOLS:
        table = pd.read_csv(stage / protocol / "oof_predictions.csv")
        np.testing.assert_array_equal(table.frame, public)
        np.testing.assert_array_equal(table.ground_truth, labels[public])
        reference = pd.read_csv(ROOT / "data/control_pair_motion_v1/dead_trim_confirmation" / protocol / "oof_predictions.csv").motion_dead_trim_seed42
        np.testing.assert_array_equal(table.current_seed42, reference)
        np.testing.assert_array_equal(table.r37_pretrained_seed42, pd.read_csv(ROOT / "data/control_player_representation_v1/screen" / protocol / "oof_predictions.csv").pretrained_latent_seed42)
        evaluated = {}
        for model in ("current_seed42", "gaussian_process_context_seed42", "r37_pretrained_seed42"):
            pred = table[model].to_numpy()
            result = pm.evaluate(pm.label_codes(table.ground_truth), pm.label_codes(pred), bootstrap_n=0)
            saved = read(stage / protocol / "official/reports" / (model + ".json"))
            row = scores[(scores.protocol == protocol) & (scores.model == model)].iloc[0]
            for a, b in ((row.spellF1, result["spell_f1_mean"]), (row.macroF1, result["macro_f1"]),
                         (saved["spell_f1_mean"], result["spell_f1_mean"]), (saved["macro_f1"], result["macro_f1"])):
                require(abs(a - b) < 1e-12, "Official headline metric differs")
            require(row.predicted_possessions == result["n_pred_spells"]
                    and abs(row.dead_recall - result["per_class"]["D"]["recall"]) < 1e-12
                    and row.false_dead_frames == np.count_nonzero((table.ground_truth.to_numpy() != "D") & (pred == "D")),
                    "Possession/dead-ball metrics differ")
            detail = folds[(folds.protocol == protocol) & (folds.model == model)]
            wanted = [s for s in splits if s["protocol"] == protocol]
            require(sorted(detail.minute) == sorted(s["block"] for s in wanted), "Fold coverage differs")
            for split in wanted:
                ids = np.asarray(split["validation"]) - native.PUBLIC_START
                local = pm.evaluate(pm.label_codes(table.ground_truth.to_numpy()[ids]), pm.label_codes(pred[ids]), bootstrap_n=0)
                actual = detail[detail.minute == split["block"]].iloc[0]
                require(abs(actual.spellF1 - local["spell_f1_mean"]) < 1e-12
                        and abs(actual.macroF1 - local["macro_f1"]) < 1e-12, "Fold metric differs")
            require(abs(row.mean_minute_spellF1 - detail.spellF1.mean()) < 1e-12, "Mean-fold metric differs")
            spans = segments[(segments.protocol == protocol) & (segments.model == model)].copy()
            truth_codes = pm.label_codes(table.ground_truth)
            expected_spans = [(native.PUBLIC_START + left, native.PUBLIC_START + right - 1, pm.SCORED[code])
                              for left, right, code in pm.segments(truth_codes, pm.valid_runs(truth_codes))]
            require(list(zip(spans.start, spans.end, spans.label)) == expected_spans,
                    "Complete ground-truth segment coverage differs")
            boundaries = np.r_[0, np.flatnonzero(pred[1:] != pred[:-1]) + 1, len(pred)]
            for span in spans.itertuples():
                start, end = span.start - native.PUBLIC_START, span.end - native.PUBLIC_START + 1
                require(span.frames == end - start and abs(span.recall - np.mean(pred[start:end] == span.label)) < 1e-12,
                        "Ground-truth segment recall differs")
                if span.label != "D" and span.frames <= 125:
                    best = max((max(0, min(end, r) - max(start, l)) / (end - start + r - l - max(0, min(end, r) - max(start, l)))
                                for l, r in zip(boundaries[:-1], boundaries[1:]) if pred[l] == span.label), default=0.)
                    require(abs(span.best_same_label_iou - best) < 1e-12, "Short-spell IoU differs")
            short = spans[(spans.label != "D") & (spans.frames <= 125)]
            require(abs(row.short_live_mean_best_iou - short.best_same_label_iou.mean()) < 1e-12
                    and row.short_live_matched_iou50 == (short.best_same_label_iou >= .5).sum()
                    and abs(row.short_live_mean_recall - short.recall.mean()) < 1e-12
                    and row.short_live_mostly_recovered == (short.recall >= .5).sum(), "Short-spell aggregate differs")
            evaluated[model] = (row, result, matching(table.ground_truth.to_numpy(), pred), short)
        old, a, before, s0 = evaluated["current_seed42"]
        new, b, after, s1 = evaluated["gaussian_process_context_seed42"]
        mask = table.current_seed42.isin(["D", "X"])
        np.testing.assert_array_equal(table.gaussian_process_context_seed42[mask], table.current_seed42[mask])
        checks = dict(primary=bool(new.spellF1 >= old.spellF1 - 1e-12),
            mean_fold_primary=bool(new.mean_minute_spellF1 >= old.mean_minute_spellF1 - 1e-12),
            macro=bool(new.macroF1 >= old.macroF1 - .001 - 1e-12),
            short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou - 1e-12),
            prior_matched=bool(np.all(after["gt_flags"][before["gt_flags"]])),
            prior_short_recall=bool((s1.recall.to_numpy()[s0.recall.to_numpy() >= .5] >= .5).all()),
            no_extra_spurious=bool(b["spell_f1"][.5]["n_spurious"] <= a["spell_f1"][.5]["n_spurious"]),
            no_extra_missed=bool(b["spell_f1"][.5]["n_missed"] <= a["spell_f1"][.5]["n_missed"]),
            D_recall=bool(new.dead_recall >= old.dead_recall - 1e-12),
            false_D=bool(new.false_dead_frames <= old.false_dead_frames + 10))
        if protocol == "buffered_minutes":
            checks.update(incremental_primary_gain=bool(new.spellF1 >= old.spellF1 + .01 - 1e-12),
                          strict_historical_best_gain=bool(new.spellF1 > read(native.PLAN)["historical_best_buffered"]))
        pair = next(p for p in selection["pairs"] if p["protocol"] == protocol)
        require(checks == pair["checks"], "Recomputed quality conjunction differs")
        comparisons.append(dict(protocol=protocol, spellF1=float(new.spellF1), macroF1=float(new.macroF1),
            reference_spellF1=float(old.spellF1), checks=checks, failed_checks=[k for k, v in checks.items() if not v]))
    passed = all(all(r["checks"].values()) for r in comparisons)
    require(selection["passed"] is passed, "Selection disagrees with recomputed conjunction")
    if not passed:
        failure = read(ROUND / "method_failure.json")
        require(failure["closed"] is True and failure["no_retry"] is True, "Failed fixed method is not closed")
        for name, sha in failure["evidence_sha256"].items():
            require(native.digest(ROUND / name) == sha, "Closed failure evidence differs")
        require(selection["eligible"] == [] and selection["selected"] is None, "Failed screen cannot select a model")
    return dict(passed=True, quality_passed=passed, context_fits=13, actual_author_fits=13, actual_posterior_calls=26,
        score_rows=9, official_headlines_recomputed=9, fold_scores_recomputed=39,
        all_worker_and_stage_hashes_verified=True, all_UTC_and_monotonic_bounds_passed=True,
        retained_D_X_exact=True, selected_model_changed=False, model_refits_by_audit=0,
        reviewer_source_sha256=native.digest(Path(__file__)),
        max_supervised_peak_bytes=max(w["conservative_peak_bytes"] for w in workers),
        max_worker_UTC_seconds=max(w["worker_UTC_seconds"] for w in workers),
        scores_sha256=native.digest(stage / "scores.csv"), screen_manifest_sha256=native.digest(stage / "manifest.json"),
        comparisons=comparisons, workers=workers), scores, folds


def append_ledger(scores, folds):
    target = ROOT / "experiments/EXPERIMENT_LEDGER.csv"
    before = target.read_bytes()
    records = list(csv.DictReader(io.StringIO(before.decode())))
    require(not any(r["round"] == "round75_gaussian_process" for r in records), "GP rows already appended")
    fields = list(records[0])
    source = ROUND / "screen/scores.csv"
    addition = io.StringIO(newline="")
    writer = csv.DictWriter(addition, fieldnames=fields, lineterminator="\n")
    for score in scores.to_dict("records"):
        row = {k: score[k] for k in fields if k in score}
        row.update(round="round75_gaussian_process", record_type="prospective_fixed_native_gaussian_process",
            stitched_spellF1=score["spellF1"], source=str(source.relative_to(ROOT)), source_sha256=native.digest(source))
        detail = folds[(folds.protocol == score["protocol"]) & (folds.model == score["model"])]
        row.update({f"block_{int(r['minute'])}": r["spellF1"] for r in detail.to_dict("records")})
        writer.writerow(row)
    with target.open("ab") as handle:
        handle.write(addition.getvalue().encode())
    require(target.read_bytes().startswith(before), "Ledger bytes were not preserved")
    return dict(ledger_rows_before=len(records), ledger_rows_after=len(records) + len(scores),
                prior_bytes_preserved=True, prior_prefix_sha256=__import__("hashlib").sha256(before).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append-ledger", action="store_true")
    args = parser.parse_args()
    result, scores, folds = audit()
    if args.append_ledger:
        result.update(append_ledger(scores, folds))
    with (ROUND / "screen_closeout.json").open("x") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("workers", "comparisons")}, sort_keys=True))
