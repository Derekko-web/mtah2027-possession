"""Audit a complete fresh corrected-TabPFN qualification stage.

No training, model unpickling, or restricted labels. Recomputes public official
scores, fold and segment diagnostics, all guards and worker evidence.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

import tabpfn_fixed_confirmation_v2 as native
import tabpfn_fixed_policy_v2 as ranking

ROOT, ROUND = native.ROOT, native.ROUND
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(seed=42,replay=False):
    native.require_private_runtime(sys.prefix, sys.base_prefix, sys.executable, True)
    native.verify();ranking.verify_policy()
    freeze = ROUND / "protocol_freeze"
    require(native.PLAN.read_bytes() == (freeze / "plan.json").read_bytes(), "Frozen plan differs")
    for group in ("implementation.json", "input_sha256.json", "protected_artifacts.json", "head_runtime.json"):
        for name, sha in read(freeze / group).items():
            require(native.digest(ROOT / name) == sha, "Frozen source/input/runtime changed: " + name)
    for name, target in read(freeze / "symlinks.json").items():
        require((ROOT / name).is_symlink() and str((ROOT / name).readlink()) == target, "Source link differs")
    ledger = ROOT / "experiments/EXPERIMENT_LEDGER.csv"
    require(ledger.read_bytes().startswith((freeze / "EXPERIMENT_LEDGER.csv").read_bytes()), "Ledger prefix differs")
    stage_name="screen" if seed==42 else f"confirmation_seed{seed}"
    stage=ROUND/(stage_name+"_replay" if replay else stage_name)
    expected_reused=0
    manifest = native.audit(stage)
    require(manifest["context_models"] == 13 and manifest["seed"] == seed
            and manifest["replay"] is replay and manifest["reused_predecessor_workers"] == expected_reused
            and manifest["fresh_context_fits"] == 13-expected_reused, "Extended screen provenance is incomplete")
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
            require(job["seed"] == seed and job["protocol"] == protocol and job["block"] == split["block"]
                    and ("parent" in job) is replay, "Initial worker role differs")
            require(summary["passed"] is True and summary["seed"] == seed and summary["protocol"] == protocol
                    and summary["block"] == split["block"] and summary["fresh_context_replay"] is replay
                    and all(summary.get(k) is True for k in ("complete_raw_preprocessing_immutable", "checkpoint_exact",
                        "resolved_model_configuration_immutable", "repeat_byte_exact", "no_network_attempts", "native_wrapper_exclusive")),
                    "Worker state/repeat/native execution checks are incomplete")
            require(supervision["passed"] is True and supervision["exit_code"] == 0 and supervision["aborted"] is None
                    and supervision["summary_sha256"] == native.digest(folder / "summary.json")
                    and supervision["peak_owned_rss_plus_full_driver_bytes"] <= native.LIMIT_BYTES,
                    "Unbound or out-of-memory worker")
            reused = (job_dir / "reuse.json").exists()
            require(not reused, "Fresh confirmation cannot reuse any predecessor worker")
            if seed == 42 and not replay: native.compare_development_worker(job_dir)
            if reused:
                origin = native.PREVIOUS / "screen" / protocol / f"block{split['block']}"
                witness = read(job_dir / "reuse.json")
                require(witness["reused"] is True and witness["fresh_model_fit"] is False
                        and witness["source"] == str(origin.relative_to(ROOT))
                        and witness["original_summary_sha256"] == native.digest(origin / "worker/summary.json")
                        and witness["original_supervision_sha256"] == native.digest(origin / "supervision.json")
                        and (folder / "summary.json").read_bytes() == (origin / "worker/summary.json").read_bytes()
                        and (job_dir / "supervision.json").read_bytes() == (origin / "supervision.json").read_bytes(),
                        "Historical worker reuse is not bound to the successful original")
            else:
                require(native.within(supervision["elapsed"], native.WORKER_SECONDS)
                        and supervision["elapsed"] == native.elapsed(supervision["started"], supervision["ended"]),
                        "New worker dual-clock limits differ")
                native.validate_operations(folder)
            artifacts = {p.name for p in folder.iterdir() if p.is_file()} - {"summary.json", "memory.json", "memory.tmp"}
            require(set(summary["artifact_sha256"]) == artifacts, "Worker artifact closure differs")
            for name, sha in summary["artifact_sha256"].items():
                require(native.digest(folder / name) == sha, "Worker artifact changed: " + name)
            for kind in ("stdout", "stderr"):
                require(supervision[kind + "_sha256"] == native.digest(job_dir / ("worker." + kind + ".log")), "Worker log differs")
            operations = [json.loads(line) for line in (folder / "operations.jsonl").read_text().splitlines()]
            predictions = [op for op in operations if not op["name"].startswith(("fit_", "fingerprint_"))]
            flushes = [json.loads(line) for line in (folder / "allocator_flush.jsonl").read_text().splitlines()]
            require([op["name"] for op in predictions] == [event["operation"] for event in flushes], "Prediction flush schedule differs")
            for op in predictions:
                require(op["active"] is False and op["native_wrapper_calls"] == op["native_dispatches"] > 0,
                        "Prediction did not exclusively use native MPS dispatch")
            for event in flushes:
                require(event["status"] == "complete" and event["live_tensor_bytes_unchanged"] is True
                        and event["model_context_or_rng_mutation_apis_called"] is False, "Flush changed fitted state")
            with np.load(job_dir / "fitting.npz", allow_pickle=False) as fitting:
                centers, observed = native.context_frames(split["train"], split["validation"], labels)
                for name, value in (("split_train", split["train"]), ("validation", split["validation"]),
                                    ("context_frames", centers), ("observation_eligible", observed)):
                    np.testing.assert_array_equal(fitting[name], value)
                targets = np.r_[(labels[centers] == "A").astype(np.int64), (labels[centers] != "A").astype(np.int64)]
                np.testing.assert_array_equal(fitting["target"], targets)
                np.testing.assert_array_equal(np.load(job["target"], allow_pickle=False), targets)
            original, swapped, owner = [np.load(folder / (name + ".npy"), allow_pickle=False)
                                        for name in ("original_probability", "swapped_probability", "owner")]
            for value in (original, swapped, owner):
                require(value.shape == (len(split["validation"]),2) and np.isfinite(value).all()
                        and np.all(value >= 0) and np.all(value <= 1)
                        and np.max(np.abs(value.sum(axis=1)-1)) < 1e-6, "Raw probabilities differ")
            np.testing.assert_array_equal(owner, (original + swapped[:, ::-1]) / 2)
            np.testing.assert_array_equal(np.load(folder / "repeat_first.npy", allow_pickle=False), original[:512])
            workers.append(dict(worker=str(folder.relative_to(stage)), reused_predecessor=reused,
                fresh_model_fit=not reused, context_rows=summary["context_rows"], query_rows=summary["query_rows"],
                worker_UTC_seconds=None if reused else supervision["elapsed"]["utc_ns"] / 1e9,
                historical_monotonic_seconds=supervision["elapsed_seconds"] if reused else None,
                conservative_peak_bytes=supervision["peak_owned_rss_plus_full_driver_bytes"]))
    require(len(workers)==13 and sum(w["reused_predecessor"] for w in workers)==expected_reused, "Complete original/replay cohort required")
    scores, folds, segments = [pd.read_csv(stage / name) for name in ("scores.csv", "fold_scores.csv", "segment_diagnostics.csv")]
    require(len(scores) == (9 if seed==42 else 6) and len(folds) == (39 if seed==42 else 26), "Incomplete metric cohort")
    selection = read(stage / "selection.json")
    comparisons = []
    for protocol in native.PROTOCOLS:
        table = pd.read_csv(stage / protocol / "oof_predictions.csv")
        np.testing.assert_array_equal(table.frame, public)
        np.testing.assert_array_equal(table.ground_truth, labels[public])
        reference = pd.read_csv(ROOT / "data/control_pair_motion_v1/dead_trim_confirmation" / protocol / "oof_predictions.csv")[f"motion_dead_trim_seed{seed}"]
        np.testing.assert_array_equal(table[f"current_seed{seed}"], reference)
        if seed==42:np.testing.assert_array_equal(table.r37_pretrained_seed42, pd.read_csv(ROOT / "data/control_player_representation_v1/screen" / protocol / "oof_predictions.csv").pretrained_latent_seed42)
        from analyze_tabpfn_blocks import load_inputs
        from research_ownership_graph import selected_arrays
        from tabpfn_short_control_inference import decode_tabpfn_short_control
        raw,bc,motion,supported,split_records=load_inputs()
        arrays,_=selected_arrays(protocol,seed,len(raw))
        blocks=[(min(s["validation"]),max(s["validation"])+1) for s in split_records if s["protocol"]==protocol]
        owner=np.full((len(raw),2),np.nan)
        owner[public]=np.load(stage/protocol/"owner_probability.npy",allow_pickle=False)
        for split in (s for s in splits if s["protocol"]==protocol):
            np.testing.assert_array_equal(owner[split["validation"]],np.load(stage/protocol/f"block{split['block']}"/"worker/owner.npy",allow_pickle=False))
        decoded,events=decode_tabpfn_short_control(arrays,owner,supported,bc,motion,raw,blocks)
        np.testing.assert_array_equal(decoded[public],table[f"tabpfn_short_control_seed{seed}"])
        require(events==read(stage/protocol/"restored_events.json"),"Restored controls differ from the shared fixed inference")
        if seed==42:
            developed=pd.read_csv(native.DEVELOPMENT/protocol/"oof_predictions.csv")
            np.testing.assert_array_equal(decoded[public],developed.tabpfn_short_control_seed42)
        evaluated = {}
        for model in ([f"current_seed{seed}",f"tabpfn_short_control_seed{seed}"]+(["r37_pretrained_seed42"] if seed==42 else [])):
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
        old, a, before, s0 = evaluated[f"current_seed{seed}"]
        new, b, after, s1 = evaluated[f"tabpfn_short_control_seed{seed}"]
        mask = table[f"current_seed{seed}"].isin(["D", "X"])
        np.testing.assert_array_equal(table[f"tabpfn_short_control_seed{seed}"][mask], table[f"current_seed{seed}"][mask])
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
                          **(dict(strict_historical_best_gain=bool(new.spellF1 > read(native.PLAN)["historical_best_buffered"])) if seed==42 else {}))
        pair = next(p for p in selection["pairs"] if p["protocol"] == protocol)
        require(checks == pair["checks"], "Recomputed quality conjunction differs")
        comparisons.append(dict(protocol=protocol, spellF1=float(new.spellF1), macroF1=float(new.macroF1),
            reference_spellF1=float(old.spellF1), checks=checks, failed_checks=[k for k, v in checks.items() if not v]))
    passed = all(all(r["checks"].values()) for r in comparisons)
    require(selection["passed"] is passed, "Selection disagrees with recomputed conjunction")
    main=next(r for r in comparisons if r["protocol"]=="buffered_minutes")
    require(seed!=42 or abs(main["spellF1"]-read(native.PLAN)["expected_predecessor_main_score"])<1e-12,
            "Completed main-protocol result differs from the saved0.777142857 diagnostic")
    one_extra=all(all(v for k,v in r["checks"].items() if k!="no_extra_spurious")
        and next(p for p in selection["pairs"] if p["protocol"]==r["protocol"])["spurious_after"]
        <=next(p for p in selection["pairs"] if p["protocol"]==r["protocol"])["spurious_before"]+1
        for r in comparisons)
    require(read(stage/"one_extra_spurious_diagnostic.json")["passed"] is one_extra,
            "Bounded-spurious trade-off diagnostic differs")
    decision=ranking.rank_rows(scores.to_dict("records"),seed)
    recorded=read(ranking.POLICY/(stage.name+"_selection.json"))
    require(decision["passed"] is recorded["passed"] and decision["pairs"]==recorded["pairs"]
            and recorded["scores_sha256"]==native.digest(stage/"scores.csv")
            and recorded["source_manifest_sha256"]==native.digest(stage/"manifest.json"), "Ranking decision differs from complete public metrics")
    return dict(passed=True, quality_passed=passed, ranking_policy_passed=decision["passed"],seed=seed,replay=replay,
        logical_context_models=13,reused_contexts=expected_reused,fresh_context_fits=13-expected_reused,one_extra_spurious_policy_passed=one_extra,
        score_rows=len(scores), official_headlines_recomputed=len(scores), fold_scores_recomputed=len(folds),
        all_worker_and_stage_hashes_verified=True, all_fresh_worker_UTC_and_monotonic_bounds_passed=True, historical_reused_timings_not_claimed_as_UTC=True,
        retained_D_X_exact=True, selected_model_changed=False, model_refits_by_audit=0, shared_fixed_inference_parity=True, development_seed42_exact=(seed==42),
        reviewer_source_sha256=native.digest(Path(__file__)),
        max_supervised_peak_bytes=max(w["conservative_peak_bytes"] for w in workers),
        max_worker_UTC_seconds=max(w["worker_UTC_seconds"] for w in workers if w["worker_UTC_seconds"] is not None),
        scores_sha256=native.digest(stage / "scores.csv"), screen_manifest_sha256=native.digest(stage / "manifest.json"),
        comparisons=comparisons, workers=workers), scores, folds



def append_ledger(result,scores,folds):
    seed,replay=result["seed"],result["replay"]
    stage="screen" if seed==42 else f"confirmation_seed{seed}"
    if replay:stage+="_replay"
    round_name=f"round78b_tabpfn_fixed_{stage}"
    ledger=ROOT/"experiments/EXPERIMENT_LEDGER.csv";before=ledger.read_bytes()
    existing=list(csv.DictReader(io.StringIO(before.decode())))
    require(not any(r["round"]==round_name for r in existing),"Stage ledger rows already exist")
    columns=list(existing[0]);buffer=io.StringIO(newline="")
    writer=csv.DictWriter(buffer,fieldnames=columns,lineterminator="\n")
    source=ROUND/stage/"scores.csv"
    for score in scores.to_dict("records"):
        row={k:score[k] for k in columns if k in score}
        row.update(round=round_name,record_type="fresh_fixed_TabPFN_confirmation_replay" if replay else "fresh_fixed_TabPFN_confirmation",
                   stitched_spellF1=score["spellF1"],source=str(source.relative_to(ROOT)),source_sha256=native.digest(source))
        detail=folds[(folds.protocol==score["protocol"])&(folds.model==score["model"])]
        row.update({f"block_{int(r['minute'])}":r["spellF1"] for r in detail.to_dict("records")})
        writer.writerow(row)
    with ledger.open("ab") as handle:handle.write(buffer.getvalue().encode())
    require(ledger.read_bytes().startswith(before),"Historical ledger bytes changed")
    result.update(ledger_rows_before=len(existing),ledger_rows_after=len(existing)+len(scores),prior_ledger_bytes_preserved=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=(42,43,44),default=42)
    parser.add_argument("--replay",action="store_true");parser.add_argument("--append-ledger",action="store_true")
    args=parser.parse_args()
    result,scores,folds=audit(args.seed,args.replay)
    if args.append_ledger:append_ledger(result,scores,folds)
    stage="screen" if args.seed==42 else f"confirmation_seed{args.seed}"
    if args.replay:stage+="_replay"
    path=ROUND/(stage+"_audit.json")
    with path.open("x") as handle:handle.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("workers","comparisons")},sort_keys=True))
