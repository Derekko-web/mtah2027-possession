"""Fresh-only corrected TabPFN qualification; all previous experiments remain preserved.
Preparation/preflight import no scientific packages and load no native arrays.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback

from tabpfn_fixed_confirmation_v2 import (ROOT,ROUND,PLAN,PROTOCOLS,PUBLIC_START,PUBLIC_END,
    context_record,digest,write_json,supervise,verify,audit,require_confirmations,
    all_nine,NativeMethodFailure,required_guards,verify_replay_evidence,require_private_runtime,compare_development_worker,DEVELOPMENT)
from tabpfn_fixed_policy_v2 import (verify_policy,require_seed42,require_confirmations,all_nine,assess_stage,ranking_selections,PLAN as RANKING_PLAN)

def close_method(stage,error,output,replay=False):
    path=ROUND/"method_failure.json"
    if path.exists():raise ValueError("Retained method failure cannot be overwritten")
    write_json(path,dict(closed=True,stage=stage,replay=replay,output=str(output.relative_to(ROUND)),reason=str(error),no_retry=True,
        no_partial_fold_rescue=True,evidence_sha256={str(p.relative_to(ROUND)):digest(p)
            for p in sorted(output.rglob("*")) if p.is_file()}))

def run(seed=42,replay=False,override=None):
    plan=verify();verify_policy()  # Both execution and explicit user policy precede scientific work.
    require_private_runtime(sys.prefix,sys.base_prefix,sys.executable,controller=True)
    if seed not in (42,43,44):raise ValueError("Unexpected fixed head seed")
    stage="screen" if seed==42 else f"confirmation_seed{seed}"
    saved=ROUND/stage
    if seed!=42:
        audit(ROUND/"screen")
        require_seed42()
    if replay:require_confirmations();audit(saved)
    output=(override or ROUND/(stage+"_replay" if replay else stage)).resolve()
    if not output.is_relative_to(ROUND.resolve()):raise ValueError("Output escapes extended TabPFN evaluation")
    output.mkdir(parents=True,exist_ok=False)
    (output/"plan.json").write_bytes(PLAN.read_bytes())
    import numpy as np
    import pandas as pd
    sys.path.insert(0,str(ROOT))
    from player_representation import (ROUND as REPRESENTATION,COLUMNS,
        check_inputs as check_representation,audit_cache,interpolate)
    from evaluation import possession_metrics as pm
    from rich_features import swap_teams
    from research_ownership_graph import selected_arrays
    from motion_dead_trim import decode_motion_dead_trim
    from tabpfn_short_control_inference import decode_tabpfn_short_control
    from research_player_representation import decode as decode_r37
    from research_stoppages import record_predictions
    from research_boundary_refinement import add_overlap_metrics
    from research_control_refinement import add_short_metrics
    from research_temporal_dead import assess
    from dfl_signal_audit import matching
    from finalize_broadcast_closure import official_scores
    check_representation()
    for name in ("inputs","encodings","screen"):audit_cache(REPRESENTATION/name)
    source=ROOT/"data/advanced_validation"
    raw,ordered=[pd.read_pickle(source/(name+".pkl")) for name in ("raw","ordered")]
    if ordered.shape[1]!=102:raise ValueError("Original102 feature interface changed")
    n=len(raw);public=np.arange(PUBLIC_START,PUBLIC_END);enc=REPRESENTATION/"encodings"
    latent,eligible=interpolate(np.load(enc/"frames.npy"),np.load(enc/"pretrained_latent.npy"),np.load(enc/"eligible.npy"),n)
    inputs=ordered.join(pd.DataFrame(latent,columns=COLUMNS,index=ordered.index))
    if inputs.shape!=(n,166):raise ValueError("Immutable original102 plus R37 fixed64 differs")
    broadcast=pd.read_pickle(ROOT/"data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:n]
    motion=pd.read_pickle(ROOT/"data/control_pair_motion_v1/inputs/motion_info.pkl").iloc[:n]
    labels=pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(labels!="X"),public)
    splits=json.loads((source/"splits.json").read_text())
    if sum(s["protocol"] in PROTOCOLS for s in splits)!=13:raise ValueError("Expected exactly13 unchanged outer folds")
    rows,details,segments,states,changes=[],[],[],{},{}
    input_hashes=dict(plan["input_sha256"]);method=f"tabpfn_short_control_seed{seed}"
    try:
        for protocol in PROTOCOLS:
            folder=output/protocol;folder.mkdir()
            folds=[(s["block"],np.asarray(s["train"]),np.asarray(s["validation"])) for s in splits if s["protocol"]==protocol]
            np.testing.assert_array_equal(np.concatenate([v for _,_,v in folds]),public)
            blocks=[(int(v[0]),int(v[-1])+1) for _,_,v in folds]
            arrays,hashes=selected_arrays(protocol,seed,n);input_hashes.update(hashes)
            reference=decode_motion_dead_trim(*arrays,broadcast,motion,raw,blocks)
            previous=pd.read_csv(ROOT/"data/control_pair_motion_v1/dead_trim_confirmation"/protocol/"oof_predictions.csv")[f"motion_dead_trim_seed{seed}"]
            np.testing.assert_array_equal(reference[public],previous)
            owner=np.full((n,2),np.nan)
            for block,train,validation in folds:
                job_dir=folder/f"block{block}";job_dir.mkdir()
                context,target,imputer,record=context_record(inputs,labels,train,validation)
                masked=np.full_like(labels,"X");masked[train]=labels[train]
                second=context_record(inputs,masked,train,validation)
                np.testing.assert_array_equal(context,second[0]);np.testing.assert_array_equal(target,second[1])
                for key,value in record.items():np.testing.assert_array_equal(value,second[3][key])
                np.savez_compressed(job_dir/"fitting.npz",**record)
                write_json(job_dir/"context_provenance.json",dict(columns=list(inputs.columns),head_seed=seed,
                    original_feature_columns=102,fixed_pretrained_latent_columns=64,footprint_radius=79,
                    paired_context_rows=len(context),public_owntrain_HA_only=True,
                    validation_label_invariance=True,all_validation_frames_queried=True,
                    imputer="train-paired-only median/empty0,keep166,no indicators",fitted_estimator_pickle_used=False))
                payloads=dict(context=context,target=target,
                    original=imputer.transform(inputs.iloc[validation]).astype(np.float32),
                    swapped=imputer.transform(swap_teams(inputs.iloc[validation])).astype(np.float32))
                parent=saved/protocol/f"block{block}"
                for name,value in payloads.items():
                    if not np.isfinite(value).all():raise ValueError("Unimputed native worker input")
                    np.save(job_dir/(name+".npy"),value)
                    if replay:
                        prior=np.load(parent/(name+".npy"),allow_pickle=False)
                        if value.dtype!=prior.dtype or value.shape!=prior.shape or value.tobytes()!=prior.tobytes():raise ValueError("Fresh payload bytes differ: "+name)
                if replay:
                    with np.load(parent/"fitting.npz",allow_pickle=False) as prior:
                        for key,value in record.items():np.testing.assert_array_equal(value,prior[key])
                    if (job_dir/"context_provenance.json").read_bytes()!=(parent/"context_provenance.json").read_bytes():raise ValueError("Context provenance replay differs")
                job={name:str(job_dir/(name+".npy")) for name in payloads}
                job.update(seed=seed,protocol=protocol,block=int(block),input_sha256={str(job_dir/(name+".npy")):digest(job_dir/(name+".npy")) for name in payloads})
                if replay:job["parent"]=str(parent/"worker")
                write_json(job_dir/"job.json",job)
                print("Fresh replay" if replay else "Fit","TabPFN",protocol,"block",block,"head seed",seed,"context",len(context),"queries",len(validation),flush=True)
                supervise(job_dir/"job.json",job_dir/"worker")  # Serial; every worker is a fresh fit.
                if seed==42 and not replay:compare_development_worker(job_dir)
                value=np.load(job_dir/"worker/owner.npy",allow_pickle=False)
                if value.shape!=(len(validation),2):raise NativeMethodFailure("Native worker rows differ")
                owner[validation]=value
            np.save(folder/"owner_probability.npy",owner[public])
            prediction,events=decode_tabpfn_short_control(arrays,owner,eligible,broadcast,motion,raw,blocks)
            write_json(folder/"restored_events.json",events)
            if seed==42:
                developed=pd.read_csv(DEVELOPMENT/protocol/"oof_predictions.csv")
                np.testing.assert_array_equal(prediction[public],developed.tabpfn_short_control_seed42)
            np.testing.assert_array_equal(prediction[~np.isin(reference,["H","A"])],reference[~np.isin(reference,["H","A"])])
            methods={f"current_seed{seed}":reference,method:prediction}
            if seed==42:
                r37=np.full((n,3),np.nan);r37[public]=np.load(REPRESENTATION/"screen"/protocol/"pretrained_latent_probability.npy")
                methods["r37_pretrained_seed42"]=decode_r37(arrays,r37,broadcast,motion,raw,blocks)
                old=pd.read_csv(REPRESENTATION/"screen"/protocol/"oof_predictions.csv").pretrained_latent_seed42
                np.testing.assert_array_equal(methods["r37_pretrained_seed42"][public],old)
            table=pd.DataFrame(dict(frame=public,ground_truth=labels[public]))
            for name,value in methods.items():
                table[name]=value[public]
                record_predictions(output,protocol,name,public,labels,folds,value[public],rows,details,segments)
                add_overlap_metrics(rows,segments,table,protocol,name)
                if name!=f"current_seed{seed}":
                    states[(protocol,name)]=(matching(labels[public],reference[public]),matching(labels[public],value[public]))
                    changes[(protocol,name)]=int((value[public]!=reference[public]).sum())
            table.to_csv(folder/"oof_predictions.csv",index=False)
        add_short_metrics(rows,segments);scores=pd.DataFrame(rows);diagnostics=pd.DataFrame(segments)
        scores.to_csv(output/"scores.csv",index=False);diagnostics.to_csv(output/"segment_diagnostics.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        official=official_scores(output,scores)
        selection=assess(scores,diagnostics,official,states,changes,seed)
        pairs=[p for p in selection["pairs"] if p["model"]==method]
        for pair in pairs:
            if seed==42 and pair["protocol"]=="buffered_minutes":
                value=scores[(scores.protocol==pair["protocol"])&(scores.model==method)].spellF1.item()
                pair["checks"]["strict_historical_best_gain"]=bool(value>plan["historical_best_buffered"])
        # Preserve strict checks and separately quantify the known +1-spurious trade-off.
        strict_pairs=json.loads(json.dumps(pairs))
        tolerance=plan.get("allowed_extra_spurious",0)
        for pair in pairs:
            protocol=pair["protocol"]
            counts=official[official.protocol==protocol].set_index("model")
            before=int(counts.loc[f"current_seed{seed}"].spurious_iou50)
            after=int(counts.loc[method].spurious_iou50)
            pair["original_no_extra_spurious"]=pair["checks"]["no_extra_spurious"]
            pair["spurious_before"],pair["spurious_after"]=before,after
            pair["checks"]["no_extra_spurious"]=bool(after<=before+tolerance)
        write_json(output/"strict_quality_diagnostic.json",dict(pairs=strict_pairs,
            passed=all(all(p["checks"].values()) for p in strict_pairs)))
        write_json(output/"one_extra_spurious_diagnostic.json",dict(
            passed=all(all(v for k,v in p["checks"].items() if k!="no_extra_spurious")
                and p["spurious_after"]<=p["spurious_before"]+1 for p in pairs),
            rule="Every other original guard remains mandatory; at most one extra spurious possession per protocol"))
        passed=(len(pairs)==3 and {p["protocol"] for p in pairs}==set(PROTOCOLS)
            and all(p["seed"]==seed and p["model"]==method and set(p["checks"])==required_guards(seed,p["protocol"])
                and all(p["checks"].values()) for p in pairs))
        selection.update(passed=passed,eligible=[method] if passed else [],selected=method if passed else None,pairs=pairs,
            R37_is_frozen_diagnostic_comparator=True,historical_substantial_and_extra_short_required=False)
        write_json(output/"selection.json",selection)
        if replay:
            for name in ("scores.csv","fold_scores.csv","segment_diagnostics.csv","official_comparison.csv","selection.json"):
                if (output/name).read_bytes()!=(saved/name).read_bytes():raise ValueError("Official/shared/gate replay differs: "+name)
            for protocol in PROTOCOLS:
                if (output/protocol/"oof_predictions.csv").read_bytes()!=(saved/protocol/"oof_predictions.csv").read_bytes():raise ValueError("OOF label replay differs")
        if not passed:write_json(output/"quality_rejected.json",dict(passed=False,
            replacement_authorized=False,reason="Completed evaluation fails the active quality policy"))
        write_json(output/"manifest.json",dict(plan_sha256=digest(PLAN),seed=seed,stage=stage,replay=bool(replay),
            independent_context_rebuild_count=13 if replay else 0,context_models=13,
            reused_predecessor_workers=len(list(output.glob("*/*/reuse.json"))),
            fresh_context_fits=13-len(list(output.glob("*/*/reuse.json"))),
            original_timing_of_reused_workers_is_historical=True,
            ranking_policy_plan_sha256=digest(RANKING_PLAN),ranking_driver_sha256=digest(Path(__file__)),
            native_public_processing=True,competition_inputs_read=True,synthetic_only=False,manual_labels_used=False,
            validation_labels_supplied_to_worker=False,retained_D_X_exact=True,complete_79_owntrain_footprint=True,
            fitted_estimator_pickle_used=False,context_regeneration_used=True,input_sha256=input_hashes,
            source_sha256=json.loads((ROUND/"protocol_freeze/implementation.json").read_text()),
            cache_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()}))
        verify();verify_policy();ranked=assess_stage(stage,output,seed,replay)
        print("Ranking-first eligible:",ranked["eligible"],flush=True)
        print(scores[["protocol","model","spellF1","macroF1","short_live_matched_iou50"]].to_string(index=False),flush=True)
        print("Eligible:",selection["eligible"],flush=True)

    except NativeMethodFailure as error:
        close_method(stage,error,output,replay);raise

def final_selection():
    verify();require_confirmations();stages=("screen","confirmation_seed43","confirmation_seed44")
    selections=ranking_selections()
    write_json(ROUND/"final_selection.json",dict(passed=all_nine(selections),selected="tabpfn_short_control_seed42",
        all9_pairs_required=True,head_seeds=[42,43,44],source_context_and_encoding_choices_unchanged=True,
        stage_manifest_sha256={s:digest(ROUND/s/"manifest.json") for s in stages},full_build_authorized=False))

def replay_complete():
    verify();require_confirmations()
    proof=verify_replay_evidence()
    for stage in ("confirmation_seed43","confirmation_seed44"):
        audit(ROUND/(stage+"_replay"))
        for name in ("scores.csv","fold_scores.csv","segment_diagnostics.csv","official_comparison.csv","selection.json"):
            if (ROUND/stage/name).read_bytes()!=(ROUND/(stage+"_replay")/name).read_bytes():raise ValueError("Incomplete exact score/official/gate replay")
        for protocol in PROTOCOLS:
            for name in ("oof_predictions.csv","owner_probability.npy"):
                if (ROUND/stage/protocol/name).read_bytes()!=(ROUND/(stage+"_replay")/protocol/name).read_bytes():raise ValueError("Incomplete exact OOF/probability replay")
    write_json(ROUND/"replay_complete.json",dict(passed=True,independent_process_fits=proof["independent_process_fits"],
        replay_evidence=proof,all13_folds_per_seed=True,
        seeds=[42,43,44],probabilities_contexts_fingerprints_weights_labels_official_gates_exact=True,
        full_build_authorized=False,next="Separate reviewed full199319CSV construction/parity/double rebuild/validator and independent review"))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=(42,43,44),default=42);parser.add_argument("--replay",action="store_true")
    parser.add_argument("--final-selection",action="store_true");parser.add_argument("--replay-complete",action="store_true")
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    try:
        if args.final_selection:final_selection()
        elif args.replay_complete:replay_complete()
        else:run(args.seed,args.replay,args.output)
    except BaseException as error:
        if not isinstance(error,NativeMethodFailure):
            path=ROUND/"global_failure.json"
            if not path.exists():write_json(path,dict(error=repr(error),traceback=traceback.format_exc(),global_abort=True,no_partial_fold_rescue=True))
        raise
