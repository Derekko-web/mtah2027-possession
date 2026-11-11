"""R73 pure invented-list tests. No scientific packages, model or native arrays."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

def run_tests():
    path=Path(__file__).with_name("kernel_svm_native.py")
    spec=importlib.util.spec_from_file_location("r73_pure_subject",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    outcomes=[]
    def check(name,callback):callback();outcomes.append(dict(name=name,passed=True))
    def require(value):
        if not value:raise AssertionError("Pure invariant failed")
    def reject(callback):
        try:callback()
        except ValueError:return
        raise AssertionError("Expected fail-closed rejection")
    base=module.ISOLATED/"python";head=module.ISOLATED/"head_runtime";exe=base/"bin/python3.12"
    check("private_head_controller_requires_prefix_base_and_executable",lambda:module.require_private_runtime(head,base,exe,True))
    check("private_model_worker_requires_base_prefix",lambda:module.require_private_runtime(base,base,exe,False))
    check("base_python_cannot_impersonate_head_controller",lambda:reject(lambda:module.require_private_runtime(base,base,exe,True)))
    check("old_shared_controller_is_rejected",lambda:reject(lambda:module.require_private_runtime(module.ROOT/".venv",base,exe,True)))
    check("wrong_base_is_rejected_independently",lambda:reject(lambda:module.require_private_runtime(head,Path("/invented/old/base"),exe,True)))
    check("wrong_executable_is_rejected_independently",lambda:reject(lambda:module.require_private_runtime(head,base,Path("/invented/old/python"),True)))
    labels=["X"]*19000
    train=list(range(12000,12500));validation=list(range(13000,13100))
    for f in train:labels[f]="H" if f%2 else "A"
    labels[12100]="D"
    frames,observed=module.context_frames(train,validation,labels)
    def footprint():
        assert all(f%5==0 and all(x in train for x in range(f-79,f+80)) for f in observed)
        assert 12080==observed[0] and 12420==observed[-1]
        assert 12100 not in frames and len(frames)==len(observed)-1
    check("5Hz_whole79_owntrain_before_HA_target_filter",footprint)
    changed=labels.copy()
    for f in validation:changed[f]="A"
    check("validation_labels_do_not_change_context",lambda:require(module.context_frames(train,validation,changed)==(frames,observed)))
    missing=[f for f in train if f!=12201]
    def hole():
        result,_=module.context_frames(missing,validation,labels)
        assert not any(abs(f-12201)<=79 for f in result)
    check("single_missing_footprint_frame_excludes_centers",hole)
    check("train_validation_overlap_rejected",lambda:reject(lambda:module.context_frames(train,[12000],labels)))
    check("duplicate_training_frames_rejected",lambda:reject(lambda:module.context_frames(train+[train[-1]],validation,labels)))
    check("unsorted_validation_rejected",lambda:reject(lambda:module.context_frames(train,list(reversed(validation)),labels)))
    check("outside_public_context_rejected",lambda:reject(lambda:module.context_frames([1],validation,labels)))
    dead=labels.copy()
    for f in train:dead[f]="D"
    check("empty_live_context_rejected",lambda:reject(lambda:module.context_frames(train,validation,dead)))
    # Up to1500 base rows is a hard maximum, no adaptive downsampling.
    # Public7500 at5Hz is naturally bounded; function still rejects >3000 pairs
    # when a synthetic public range is enlarged for this specific guard test.
    old_end=module.PUBLIC_END;module.PUBLIC_END=30000
    huge=list(range(11200,20000));huge_labels=["H"]*31000
    check("oversized_context_is_rejected_without_resampling",lambda:reject(lambda:module.context_frames(huge,[22000],huge_labels)))
    module.PUBLIC_END=old_end
    def padding():
        batches=list(module.query_indices(513))
        assert batches[0]==(0,512,list(range(512))) and batches[1]==(512,513,[512]*512)
        assert list(module.query_indices(3))==[(0,3,[0,1,2]+[2]*509)]
    check("canonical512_pad_last_and_keep_all_queries",padding)
    check("empty_query_rejected",lambda:reject(lambda:list(module.query_indices(0))))
    selections=[]
    for seed in (42,43,44):
        selections.append(dict(passed=True,pairs=[dict(protocol=p,seed=seed,model=f"kernel_svm_context_seed{seed}",checks={name:True for name in module.required_guards(seed,p)}) for p in module.PROTOCOLS]))
    check("all9_fixed_seed_protocol_pairs_required",lambda:require(module.all_nine(selections)))
    bad=copy.deepcopy(selections);bad[1]["pairs"][0]["seed"]=42
    check("confirmation_head_seed_must_change_to43",lambda:require(not module.all_nine(bad)))
    bad=copy.deepcopy(selections);bad[2]["pairs"][1]["checks"]["primary"]=False
    check("one_failed_confirmation_closes_conjunction",lambda:require(not module.all_nine(bad)))
    bad=copy.deepcopy(selections);bad[0]["pairs"][0]["checks"].pop("macro")
    check("missing_selected8_guard_cannot_pass",lambda:require(not module.all_nine(bad)))
    check("partial_seed_protocol_results_cannot_pass",lambda:require(not module.all_nine(selections[:2])))
    check("250frame_outer_exclusion_is_required",lambda:reject(lambda:module.context_frames(train,[12600],labels)))
    check("empty_validation_is_rejected",lambda:reject(lambda:module.context_frames(train,[],labels)))
    def pending_technical_is_hard_boundary():
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"PLAN",Path(folder)/"plan.json"):
            module.write_json(module.PLAN,dict(technical_prerequisite=dict(
                actual_all_four_stage_summary_sha256=None,status="PENDING_ACTUAL_ALL_FOUR_STAGE_PASS")))
            with patch.object(module,"load_module",side_effect=AssertionError("No technical module/native access allowed")):
                reject(module.technical_gate)
            with patch.object(module,"ROUND",Path(folder)),patch.object(module,"pure_source_checks",side_effect=AssertionError("No preflight before prerequisites")):
                reject(module.preflight);reject(module.freeze)
                assert not (Path(folder)/"source_preflight.json").exists() and not (Path(folder)/"protocol_freeze").exists()
    check("pending_null_technical_evidence_blocks_before_any_module_or_native_access",pending_technical_is_hard_boundary)
    def technical_stages():
        records=[dict(size=s,stage=t,passed=True,exit_code=0,aborted=None)
            for s in ("moderate","full") for t in ("first","rebuild")]
        summary=dict(passed=True,implementation_sha256="source",plan_sha256="plan",records=records)
        assert len(module.technical_header(summary,"source","plan"))==4
        for altered in (records[:3],records[:3]+[records[2]],records[::-1]):
            bad=dict(summary,records=altered);reject(lambda:module.technical_header(bad,"source","plan"))
        for fields in (dict(passed=False),dict(exit_code=1),dict(aborted="rss_limit")):
            bad=copy.deepcopy(summary);bad["records"][3].update(fields)
            reject(lambda:module.technical_header(bad,"source","plan"))
    check("missing_duplicate_reordered_failed_aborted_technical_stages_cannot_admit_native",technical_stages)
    def calendar_operation_uses_both_clocks():
        def operation(output,name,limit,callback):
            assert limit==300
            return callback(),1.0  # Source-defined active duration remains under its limit.
        delegate=Mock(side_effect=operation);execution=SimpleNamespace(operation=delegate)
        callback=Mock(return_value="unchanged-result")
        for started,ended,should_pass in ((1000.,1001.,True),(1000.,1301.,False),(1000.,999.,False)):
            with tempfile.TemporaryDirectory() as folder,patch.object(module.time,"time",side_effect=[started,ended]):
                output=Path(folder)
                if should_pass:
                    assert module.calendar_operation(execution,output,"invented-operation",callback)==("unchanged-result",1.,1.)
                else:
                    try:module.calendar_operation(execution,output,"invented-operation",callback)
                    except module.NativeMethodFailure:pass
                    else:raise AssertionError("UTC overrun/backward clock must reject despite active1second")
                event=json.loads((output/"calendar_operations.jsonl").read_text())
                assert event["wall_started"]==started and event["wall_ended"]==ended
                assert event["wall_elapsed_seconds"]==ended-started and event["monotonic_seconds"]==1.
                assert event["calendar_guard_passed"] is should_pass
                assert json.loads((output/"calendar_operation.json").read_text())==event
        assert delegate.call_count==3 and callback.call_count==3
    check("UTC_operation_limit_and_backward_clock_reject_even_when_monotonic_is_under_limit",calendar_operation_uses_both_clocks)
    def calendar_supervisor_guards():
        active=dict(active=True,wall_started=1000.)
        assert module.calendar_abort_reason(1301.,1000.,active)=="wall_operation_over_300seconds"
        assert module.calendar_abort_reason(1900.,1000.,{})=="wall_worker_over_900seconds"
        assert module.calendar_abort_reason(999.,1000.,active)=="wall_clock_negative_or_nonfinite"
        assert module.calendar_abort_reason(1100.,1000.,dict(active=True,wall_started=1200.))=="wall_operation_clock_negative_or_nonfinite"
        assert module.calendar_abort_reason(1100.,1000.,active) is None
        assert not module.duration_ok(float('nan'),300) and not module.duration_ok(True,300)
    check("UTC_supervisor_observes_active_operation_worker_and_negative_clock_bounds",calendar_supervisor_guards)
    def technical_calendar_audit():
        records=[dict(size=s,stage=t,passed=True,actual_classifier_fits=4,
            both_operation_utc_caps_proven_by_whole_stage_upper_bound=True,
            stdout_birth_epoch=1000.,supervision_complete_epoch=1001.,utc_stage_upper_bound_seconds=1.)
            for s in ("moderate","full") for t in ("first","rebuild")]
        audit=dict(passed=True,actual_all_four_stage_pass=True,technical_summary_sha256="bound-summary",
            actual_classifier_fits=16,exact_fresh_rebuild_file_comparisons=32,native_processing=False,
            source_or_dependencies_changed=False,records=records)
        assert len(module.technical_calendar_header(audit,"bound-summary"))==4
        for changes in (dict(records=records[:3]),dict(records=records[:3]+[records[2]]),dict(passed=False),
                dict(technical_summary_sha256="different-summary")):
            reject(lambda:module.technical_calendar_header(dict(audit,**changes),"bound-summary"))
        for duration in (-1.,0.,300.,301.):
            bad=copy.deepcopy(audit);bad["records"][0].update(utc_stage_upper_bound_seconds=duration,supervision_complete_epoch=1000.+duration)
            reject(lambda:module.technical_calendar_header(bad,"bound-summary"))
    check("technical_calendar_proof_requires_four_positive_sub300_whole_worker_bounds",technical_calendar_audit)
    def raw_pipeline_contract():
        import ast
        native=ast.parse(path.read_text());worker=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=="worker")
        text=ast.unparse(worker)
        assert text.index('verify()')<text.index('execution.initialize()')<text.index('np.load(')
        assert 'execution.fit_with_author_recording(data, folds, output)' in text
        assert 'native_query_checks(execution, model, data, payload, before, output)' in text
        assert 'make().set_params(svc__random_state=job[\'seed\'])' in text
        context=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=="context_record")
        assert not any(isinstance(n,ast.Name) and n.id in ("SimpleImputer","StandardScaler") for n in ast.walk(context))
        assert 'np.float64' in ast.unparse(context)
        controller=path.with_name("research_kernel_svm_native.py").read_text()
        assert 'imputer.transform' not in controller and 'base_frames=base_frames' in controller
        assert 'all_validation_frames_queried=True' in controller
    check("raw_payloads_have_no_outer_preimputation_and_frozen_author_methods_are_reused",raw_pipeline_contract)
    def unchanged_pure_scientific_interfaces():
        import ast
        current=ast.parse(path.read_text());prior=ast.parse(path.with_name("tabpfn_native.py").read_text())
        for name in ("context_frames","query_indices","required_guards","require_private_runtime"):
            a=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in prior.body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    check("R65_outer_footprint_query_and_selected8_guard_contracts_are_AST_identical",unchanged_pure_scientific_interfaces)
    def replay_finalization_proofs():
        # Literal invented bytes/JSON only; no NumPy file or model is deserialized.
        stages=(("screen",42),("confirmation_seed43",43),("confirmation_seed44",44))
        folds=[dict(train=[j for j in range(6) if j not in (i,i+3)],validation=[i,i+3]) for i in range(3)]
        fits=[]
        for i in range(4):
            fits.append(dict(ordinal=i,role="final" if i==3 else "inner",
                train_indices=list(range(6)) if i==3 else folds[i]["train"],
                validation_indices=None if i==3 else folds[i]["validation"],
                input_sha256={"X":"literal-invented-raw-input","y":"literal-target"},semantic_sha256=f"literal-fit-{i}",
                learned_state=dict(fit_status=0,effective_probability=False,full_dict={"literal":"complete-state-witness"})))
        module.validate_four_fit_records(fits,folds)
        def put(path,value):module.write_json(path,value)
        def read(path):return json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"ROUND",Path(folder).resolve()):
            root=Path(folder).resolve()
            for stage,seed in stages:
                for replay in (False,True):
                    output=root/(stage+"_replay" if replay else stage);output.mkdir()
                    for name in ("scores.csv","fold_scores.csv","segment_diagnostics.csv","official_comparison.csv","selection.json"):
                        (output/name).write_bytes(f"literal-score-{stage}-{name}".encode())
                    for protocol in module.PROTOCOLS:
                        (output/protocol).mkdir()
                        for name in ("oof_predictions.csv","owner_probability.npy"):
                            (output/protocol/name).write_bytes(f"literal-OOF-{stage}-{protocol}-{name}".encode())
                        for block in range(1,4 if protocol=="buffered_100_seconds" else 6):
                            job_dir=output/protocol/f"block{block}";worker=job_dir/"worker";worker.mkdir(parents=True)
                            witness=f"literal-proof-seed{seed}-{protocol}-{block}".encode()
                            for name in module.PAYLOADS:(job_dir/(name+".npy")).write_bytes(witness+name.encode())
                            (job_dir/"fitting.npz").write_bytes(witness+b"raw-fitting-record")
                            put(job_dir/"context_provenance.json",dict(seed=seed,protocol=protocol,block=block,outer_preimputation=False))
                            for name in module.REQUIRED_WORKER_ARTIFACTS:(worker/name).write_bytes(witness+name.encode())
                            put(worker/"inner_folds.json",folds);put(worker/"fit_records.json",fits)
                            for ordinal in range(4):put(worker/f"fit_{ordinal}_state.json",fits[ordinal])
                            # Replay deliberately has different UTC timestamps: they
                            # must be validated/bound locally, never byte-compared.
                            epoch=1000.+(100000. if replay else 0.)
                            events=[]
                            for name,start,end in (("complete_three_inner_and_final_fit",epoch,epoch+.125),
                                    ("all_native_queries_repeat_symmetry_reload",epoch+1.,epoch+1.25)):
                                events.append(dict(name=name,active=False,status="complete",calendar_guard_passed=True,
                                    wall_started=start,wall_ended=end,wall_elapsed_seconds=end-start,monotonic_seconds=.1))
                            (worker/"calendar_operations.jsonl").write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in events))
                            put(worker/"calendar_operation.json",events[-1])
                            for kind in ("stdout","stderr"):(job_dir/("worker."+kind+".log")).write_text("")
                            job={name:str(job_dir/(name+".npy")) for name in module.PAYLOADS}
                            job.update(seed=seed,protocol=protocol,block=block,
                                input_sha256={str(job_dir/(name+".npy")):module.digest(job_dir/(name+".npy")) for name in module.PAYLOADS})
                            parent=root/stage/protocol/f"block{block}"/"worker"
                            if replay:job["parent"]=str(parent)
                            put(job_dir/"job.json",job)
                            put(worker/"summary.json",dict(passed=True,seed=seed,protocol=protocol,block=block,
                                fresh_context_replay=replay,fresh_refit_parent=str(parent) if replay else None,
                                actual_classifier_fits=4,fitted_estimator_pickle_used=False,
                                checks={name:True for name in module.WORKER_CHECKS},fit_seconds=.1,all_query_seconds=.1,
                                wall_fit_seconds=.125,wall_all_query_seconds=.25,artifact_sha256={}))
                            put(job_dir/"supervision.json",dict(passed=True,exit_code=0,aborted=None,elapsed_seconds=1,peak_bytes=1024,
                                wall_started=epoch,wall_ended=epoch+2.,wall_elapsed_seconds=2.))
                    put(output/"manifest.json",dict(stage=stage,seed=seed,replay=replay,context_models=13,actual_classifier_fits=52,
                        independent_context_rebuild_count=13 if replay else 0,cache_sha256={}))
            def seal():
                for stage,seed in stages:
                    for replay in (False,True):
                        output=root/(stage+"_replay" if replay else stage)
                        for worker in output.rglob("worker"):
                            if not worker.is_dir():continue
                            summary=worker/"summary.json";value=read(summary)
                            value["artifact_sha256"]={p.name:module.digest(p) for p in worker.iterdir() if p.is_file() and p.name!="summary.json"};put(summary,value)
                            supervision=worker.parent/"supervision.json"
                            if supervision.exists():
                                value=read(supervision);value["summary_sha256"]=module.digest(summary)
                                for kind in ("stdout","stderr"):value[kind+"_sha256"]=module.digest(worker.parent/("worker."+kind+".log"))
                                put(supervision,value)
                        manifest=output/"manifest.json";value=read(manifest)
                        value["cache_sha256"]={str(p.relative_to(output)):module.digest(p) for p in output.rglob("*") if p.is_file() and p!=manifest};put(manifest,value)
            seal();proof=module.verify_replay_evidence()
            assert proof["independent_process_fits"]==39 and proof["actual_classifier_fits"]==156
            assert len({(p["seed"],p["worker"]) for p in proof["workers"]})==39
            def mutate(path,callback):
                before=path.read_bytes()
                try:callback(path);seal();reject(module.verify_replay_evidence)
                finally:path.write_bytes(before);seal()
            def fields(path,callback):value=read(path);callback(value);put(path,value)
            manifest=root/"screen_replay/manifest.json"
            for change in (dict(replay=False),dict(context_models=12),dict(actual_classifier_fits=51),dict(seed=44)):
                mutate(manifest,lambda p,change=change:fields(p,lambda v:v.update(change)))
            job_dir=root/"screen_replay/buffered_minutes/block1";worker=job_dir/"worker"
            for change in (dict(fresh_context_replay=False),dict(actual_classifier_fits=3),dict(fitted_estimator_pickle_used=True),
                    dict(fresh_refit_parent=str(worker)),dict(wall_fit_seconds=301.),dict(wall_all_query_seconds=-1.)):
                mutate(worker/"summary.json",lambda p,change=change:fields(p,lambda v:v.update(change)))
            for change in (dict(aborted="resource_abort"),dict(exit_code=1),dict(elapsed_seconds=901),
                    dict(wall_elapsed_seconds=901.,wall_ended=101901.),dict(wall_elapsed_seconds=-1.,wall_ended=100999.)):
                mutate(job_dir/"supervision.json",lambda p,change=change:fields(p,lambda v:v.update(change)))
            def bad_calendar(path):
                values=[json.loads(line) for line in path.read_text().splitlines()]
                values[0].update(wall_ended=values[0]["wall_started"]+301.,wall_elapsed_seconds=301.)
                path.write_text(''.join(json.dumps(v,sort_keys=True)+'\n' for v in values))
            mutate(worker/"calendar_operations.jsonl",bad_calendar)
            mutate(job_dir/"job.json",lambda p:fields(p,lambda v:v.update(parent=str(root/"confirmation_seed43/buffered_minutes/block1/worker"))))
            mutate(worker/"fit_records.json",lambda p:put(p,fits[:3]))
            mutate(worker/"fit_3_state.json",lambda p:put(p,dict(fits[3],role="inner")))
            for name in ("full_state.json","calibrator_state.json","oof_margin.npy","original_probability.npy"):
                mutate(worker/name,lambda p:p.write_bytes(b"changed with every enclosing hash resealed"))
            payload=job_dir/"base_frames.npy";job=job_dir/"job.json";oldjob=job.read_bytes()
            def alter_payload(p):
                p.write_bytes(b"changed original-center payload")
                fields(job,lambda v:v["input_sha256"].update({str(p):module.digest(p)}))
            try:mutate(payload,alter_payload)
            finally:job.write_bytes(oldjob);seal()
            renamed=worker.with_name("missing_worker")
            try:worker.rename(renamed);seal();reject(module.verify_replay_evidence)
            finally:renamed.rename(worker);seal()
            assert module.verify_replay_evidence()["actual_classifier_fits"]==156
            spec=importlib.util.spec_from_file_location("r73_pure_controller",path.with_name("research_kernel_svm_native.py"))
            controller=importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules,{"kernel_svm_native":module}):spec.loader.exec_module(controller)
            with patch.object(controller,"ROUND",root),patch.object(controller,"verify"),patch.object(controller,"require_confirmations"):
                controller.replay_complete()
            completed=read(root/"replay_complete.json")
            assert completed["passed"] is True and completed["actual_independent_classifier_fits"]==156
            assert completed["full_build_authorized"] is False
    check("39workers_156actual_fit_records_fivepayloads_and_complete_semantic_replay_are_required",replay_finalization_proofs)
    return outcomes

if __name__=="__main__":
    print(json.dumps(dict(passed=True,tests=run_tests(),scientific_imports=False,native_arrays_loaded=False),indent=2))
