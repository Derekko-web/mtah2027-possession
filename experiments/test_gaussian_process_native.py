"""Source-only native GP contract and adversarial retained-evidence tests."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from unittest.mock import patch


def run_tests():
    path=Path(__file__).with_name("gaussian_process_native.py")
    spec=importlib.util.spec_from_file_location("gp_native_pure_subject",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    execution=module.load_module(module.GATE_SOURCE,"gaussian_process_runtime_gate_v2")
    outcomes=[]
    def check(name,callback):
        callback();outcomes.append(dict(name=name,passed=True))
    def require(value):
        if not value:raise AssertionError("Contract failed")
    def reject(callback):
        try:callback()
        except (ValueError,KeyError):return
        raise AssertionError("Expected rejection")
    base=module.ISOLATED/"python";head=module.ISOLATED/"head_runtime";exe=base/"bin/python3.12"
    check("private_controller_runtime",lambda:module.require_private_runtime(head,base,exe,True))
    check("private_worker_runtime",lambda:module.require_private_runtime(base,base,exe,False))
    for name,prefix,origin,binary in (("wrong_prefix",base,base,exe),("wrong_base",head,Path("/invented"),exe),
            ("wrong_executable",head,base,Path("/invented/python"))):
        check(name,lambda p=prefix,o=origin,b=binary:reject(lambda:module.require_private_runtime(p,o,b,True)))
    labels=["X"]*19000;train=list(range(12000,12500));validation=list(range(13000,13100))
    for f in train:labels[f]="H" if f%2 else "A"
    labels[12100]="D"
    frames,observed=module.context_frames(train,validation,labels)
    check("whole79_owntrain_footprint_5Hz_then_live_filter",lambda:require(
        observed[0]==12080 and observed[-1]==12420 and 12100 not in frames
        and all(f%5==0 and all(x in train for x in range(f-79,f+80)) for f in frames)))
    changed=labels.copy()
    for f in validation:changed[f]="A"
    check("validation_label_invariance",lambda:require(module.context_frames(train,validation,changed)==(frames,observed)))
    check("missing_training_frame_removes_entire_encoder_footprint",lambda:require(
        not any(abs(f-12201)<=79 for f in module.context_frames([x for x in train if x!=12201],validation,labels)[0])))
    for name,t,v in (("overlap",train,[12000]),("outer_purge",train,[12600]),("duplicate",train+[train[-1]],validation),
            ("unsorted",train,validation[::-1]),("outside_public",[1],validation),("empty_validation",train,[])):
        check(name,lambda t=t,v=v:reject(lambda:module.context_frames(t,v,labels)))
    check("empty_live_context",lambda:reject(lambda:module.context_frames(train,validation,["D"]*19000)))
    check("canonical512_last_padding",lambda:require(list(module.query_indices(513))[-1]==(512,513,[512]*512)))
    check("empty_query",lambda:reject(lambda:list(module.query_indices(0))))
    selections=[dict(passed=True,pairs=[dict(protocol=p,seed=seed,model=f"gaussian_process_context_seed{seed}",
        checks={name:True for name in module.required_guards(seed,p)}) for p in module.PROTOCOLS]) for seed in (42,43,44)]
    check("all9_pairs",lambda:require(module.all_nine(selections)))
    for name,change in (("failed_guard",lambda x:x[2]["pairs"][1]["checks"].update(primary=False)),
            ("wrong_seed",lambda x:x[1]["pairs"][0].update(seed=42)),
            ("missing_guard",lambda x:x[0]["pairs"][0]["checks"].pop("macro"))):
        def bad_selection(change=change):
            value=copy.deepcopy(selections);change(value);require(not module.all_nine(value))
        check(name,bad_selection)
    check("partial_confirmations",lambda:require(not module.all_nine(selections[:2])))
    def pending():
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"PLAN",Path(folder)/"plan.json"):
            module.write_json(module.PLAN,dict(technical_prerequisite=dict(actual_all_four_stage_summary_sha256=None,status="PENDING")))
            with patch.object(module,"load_module",side_effect=AssertionError("No module/data access before admission")):
                reject(module.technical_gate)
                with patch.object(module,"ROUND",Path(folder)):reject(module.preflight);reject(module.freeze)
    check("pending_evidence_blocks_all_admission_before_import",pending)
    def exact_context_interfaces():
        current=ast.parse(path.read_text());prior=ast.parse(path.with_name("kernel_svm_native.py").read_text())
        for name in ("context_frames","query_indices","required_guards","require_private_runtime","context_record"):
            a=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in prior.body if isinstance(n,ast.FunctionDef) and n.name==name)
            assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
        worker=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name=="worker")
        attributes={n.attr for n in ast.walk(worker) if isinstance(n,ast.Attribute)}
        assert not {"chronological_splits","make_estimator","validate_four_fit_records"}&attributes
    check("unchanged_outer_features_footprint_queries_guards_and_no_SVM_calls",exact_context_interfaces)
    def two_clocks():
        start=dict(utc_ns=1000000000000,monotonic_ns=1000000000000)
        for clock in start:
            now=dict(start);now[clock]+=900000000000
            require(execution.clock_abort_reason(start,now,{}) is not None)
            now=dict(start);now[clock]-=1
            require(execution.clock_abort_reason(start,now,{}) is not None)
            now=dict(start);now[clock]+=300000000000
            marker=dict(name=execution.FIT_OPERATION,active=True,limit_seconds=300,started=start)
            require(execution.clock_abort_reason(start,now,marker) is not None)
    check("either_clock_worker_operation_or_backward_jump_rejects",two_clocks)
    def gp_wrapper_query_contract():
        payload=dict(context=[[1.0]],target=[0,1],original=[[.7,.3]],swapped=[[.3,.7]],base_frames=[12000])
        fake_numpy=SimpleNamespace(load=lambda *a,**kw:payload["swapped"] if "swapped" in str(a[0]) else payload["original"],
            save=lambda *a,**kw:None)
        model=SimpleNamespace(classifier=SimpleNamespace(classes_=SimpleNamespace(tolist=lambda:[0,1])))
        delegate=SimpleNamespace(query_and_reload_checks=lambda *a:dict(passed=True),
            canonical_predict=lambda m,q:q,array_record=lambda v:v)
        with tempfile.TemporaryDirectory() as folder,patch.dict(sys.modules,{"numpy":fake_numpy}):
            result=module.native_query_checks(delegate,model,{},payload,copy.deepcopy(payload),Path(folder))
            assert result==dict(passed=True)
            assert json.loads((Path(folder)/"classes.json").read_text())==dict(classes=[0,1],meaning=["H","A"])
    check("GP_wrapper_uses_classifier_classes_and_preserves_query_payloads",gp_wrapper_query_contract)
    def retained_proof():
        # Literal invented evidence; no model/array decoding or scientific imports.
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"ROUND",Path(folder).resolve()),patch.object(module,"technical_gate",return_value=execution):
            root=Path(folder).resolve()
            def put(p,v):module.write_json(p,v)
            def read(p):return json.loads(p.read_text())
            def state_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
            for stage,seed in (("screen",42),("confirmation_seed43",43),("confirmation_seed44",44)):
                for replay in (False,True):
                    output=root/(stage+"_replay" if replay else stage);output.mkdir()
                    for name in ("scores.csv","fold_scores.csv","segment_diagnostics.csv","official_comparison.csv","selection.json"):
                        (output/name).write_bytes(b"literal-identical-score")
                    for protocol in module.PROTOCOLS:
                        (output/protocol).mkdir()
                        for name in ("oof_predictions.csv","owner_probability.npy"):(output/protocol/name).write_bytes(b"literal-identical-OOF")
                        for block in range(1,4 if protocol=="buffered_100_seconds" else 6):
                            job_dir=output/protocol/f"block{block}";worker=job_dir/"worker";worker.mkdir(parents=True)
                            witness=f"seed{seed}-{protocol}-{block}".encode()
                            for name in module.PAYLOADS:(job_dir/(name+".npy")).write_bytes(witness+name.encode())
                            (job_dir/"fitting.npz").write_bytes(witness);put(job_dir/"context_provenance.json",dict(seed=seed))
                            for name in module.REQUIRED_WORKER_ARTIFACTS:(worker/name).write_bytes(witness+name.encode())
                            state=dict(class_name="invented-full-GP-state",seed=seed);put(worker/"full_state.json",state)
                            roles=execution.training_roles(4);roles["queries"]="feature_only_all1500"
                            fit=dict(actual_author_fit=True,actual_posterior_calls=2,raw_X=dict(shape=[4,166]),roles=roles,
                                optimizer=None,unit_amplitude=1.0,class_order=[0,1],posterior_contract=execution.posterior_source_contract(),semantic_sha256=state_sha(state))
                            put(worker/"fit_record.json",fit)
                            traces=[dict(ordinal=i,zero_initialization=True,returned=True,return_temporaries=True,exception=False,break_iteration=1,
                                history=[dict(iteration=0,lml=-2.,previous="-inf",delta="+inf"),dict(iteration=1,lml=-2.,previous=-2.,delta=0.)]) for i in range(2)]
                            put(worker/"posterior_traces.json",traces)
                            put(worker/"head_seed.json",dict(requested_seed=seed,public_seed=seed,binary_seed=seed,actual_author_fits=1,actual_posterior_calls=2))
                            epoch=1000000000000+(1000000000000 if replay else 0)
                            events=[]
                            for i,name in enumerate((execution.FIT_OPERATION,execution.QUERY_OPERATION)):
                                started=dict(utc_ns=epoch+i*1000000000,monotonic_ns=epoch+i*1000000000)
                                ended={k:v+100000000 for k,v in started.items()}
                                events.append(dict(name=name,active=False,started=started,ended=ended,elapsed=execution.clock_elapsed(started,ended),limit_seconds=300,status="complete"))
                            (worker/"operations.jsonl").write_text(''.join(json.dumps(e)+'\n' for e in events));put(worker/"operation.json",events[-1])
                            for kind in ("stdout","stderr"):(job_dir/("worker."+kind+".log")).write_text("")
                            parent=root/stage/protocol/f"block{block}"/"worker"
                            job={name:str(job_dir/(name+".npy")) for name in module.PAYLOADS}
                            job.update(seed=seed,protocol=protocol,block=block,input_sha256={v:module.digest(v) for v in job.values()})
                            if replay:job["parent"]=str(parent)
                            put(job_dir/"job.json",job)
                            put(worker/"summary.json",dict(passed=True,seed=seed,protocol=protocol,block=block,
                                fresh_context_replay=replay,fresh_refit_parent=str(parent) if replay else None,query_rows=1500,
                                actual_author_fits=1,actual_posterior_calls=2,fitted_estimator_pickle_used=False,
                                checks={n:True for n in module.WORKER_CHECKS},fit_elapsed=events[0]["elapsed"],query_elapsed=events[1]["elapsed"],artifact_sha256={}))
                            started=dict(utc_ns=epoch,monotonic_ns=epoch);ended={k:v+2000000000 for k,v in started.items()}
                            put(job_dir/"supervision.json",dict(passed=True,exit_code=0,aborted=None,peak_bytes=1024,started=started,ended=ended,elapsed=execution.clock_elapsed(started,ended)))
                    put(output/"manifest.json",dict(stage=stage,seed=seed,replay=replay,context_models=13,actual_author_fits=13,actual_posterior_calls=26,
                        independent_context_rebuild_count=13 if replay else 0,cache_sha256={}))
            def seal():
                for output in root.iterdir():
                    if not output.is_dir():continue
                    for worker in output.rglob("worker"):
                        value=read(worker/"summary.json");value["artifact_sha256"]={p.name:module.digest(p) for p in worker.iterdir() if p.is_file() and p.name!="summary.json"};put(worker/"summary.json",value)
                        value=read(worker.parent/"supervision.json");value["summary_sha256"]=module.digest(worker/"summary.json")
                        for kind in ("stdout","stderr"):value[kind+"_sha256"]=module.digest(worker.parent/("worker."+kind+".log"))
                        put(worker.parent/"supervision.json",value)
                    value=read(output/"manifest.json");value["cache_sha256"]={str(p.relative_to(output)):module.digest(p) for p in output.rglob("*") if p.is_file() and p.name!="manifest.json"};put(output/"manifest.json",value)
            seal();proof=module.verify_replay_evidence()
            assert proof["independent_process_fits"]==39 and proof["actual_author_fits"]==39 and proof["actual_posterior_calls"]==78
            worker=root/"screen_replay/buffered_minutes/block1/worker"
            def mutate(p,callback):
                original=p.read_bytes()
                try:callback(p);seal();reject(module.verify_replay_evidence)
                finally:p.write_bytes(original);seal()
            def update(p,changes):value=read(p);value.update(changes);put(p,value)
            for change in (dict(actual_author_fits=4),dict(actual_posterior_calls=1),dict(fresh_context_replay=False),dict(fitted_estimator_pickle_used=True)):
                mutate(worker/"summary.json",lambda p,c=change:update(p,c))
            for change in (dict(actual_posterior_calls=1),dict(optimizer="search"),dict(unit_amplitude=2.),dict(roles={})):
                mutate(worker/"fit_record.json",lambda p,c=change:update(p,c))
            mutate(worker/"posterior_traces.json",lambda p:put(p,read(p)[:1]))
            mutate(worker/"head_seed.json",lambda p:update(p,dict(binary_seed=44)))
            for name in ("full_state.json","original_probability.npy","payload_immutability.json"):
                mutate(worker/name,lambda p:p.write_bytes(b'{"tampered":true}'))
            for change in (dict(exit_code=1),dict(aborted="cap"),dict(elapsed=dict(utc_ns=900000000000,monotonic_ns=1))):
                mutate(worker.parent/"supervision.json",lambda p,c=change:update(p,c))
            def bad_operation(p):
                values=[json.loads(x) for x in p.read_text().splitlines()];values[0]["elapsed"]["utc_ns"]=300000000000
                p.write_text(''.join(json.dumps(v)+'\n' for v in values))
            mutate(worker/"operations.jsonl",bad_operation)
            mutate(worker.parent/"job.json",lambda p:update(p,dict(parent=str(root/"confirmation_seed43/buffered_minutes/block1/worker"))))
            mutate(root/"screen_replay/manifest.json",lambda p:update(p,dict(actual_author_fits=52)))
    check("all39_fresh_GP_fits_78_posterior_calls_and_resealed_tamper_rejection",retained_proof)
    return outcomes


if __name__=="__main__":
    print(json.dumps(dict(passed=True,tests=run_tests(),scientific_imports=False,native_arrays_loaded=False),indent=2))
