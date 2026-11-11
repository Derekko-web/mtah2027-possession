"""R68 pure invented-list tests. No scientific packages, model or native arrays."""
import ast
import copy
import json
import tempfile
import types
from unittest.mock import Mock, patch
import importlib.util
from pathlib import Path

def run_tests():
    path=Path(__file__).with_name("modernnca_native.py")
    spec=importlib.util.spec_from_file_location("r68_pure_subject",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    outcomes=[]
    def check(name,callback):callback();outcomes.append(dict(name=name,passed=True))
    def require(value):
        if not value:raise AssertionError("Pure invariant failed")
    def reject(callback):
        try:callback()
        except ValueError:return
        raise AssertionError("Expected fail-closed rejection")
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
        selections.append(dict(passed=True,pairs=[dict(protocol=p,seed=seed,model=f"modernnca_context_seed{seed}",checks={name:True for name in module.required_guards(seed,p)}) for p in module.PROTOCOLS]))
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
    check("prospective_seed_streams_are_fixed",lambda:require([module.seed_streams(seed) for seed in (42,43,44)]==[
        dict(python=42,numpy_global=43,torch=44),dict(python=43,numpy_global=44,torch=45),dict(python=44,numpy_global=45,torch=46)]))
    check("unplanned_seed_rejected",lambda:reject(lambda:module.seed_streams(45)))
    def classification():
        def bad_numeric():raise RuntimeError("numerical failure")
        try:module.numerical_call(bad_numeric)
        except module.NativeMethodFailure:pass
        else:raise AssertionError("Numerical failure did not close method")
        def bad_provenance():raise ValueError("Frozen payload hash changed")
        try:module.numerical_call(bad_provenance)
        except module.NativeMethodFailure:raise AssertionError("Provenance failure hidden as method failure")
        except ValueError:pass
        else:raise AssertionError("Provenance corruption was not raised")
    check("numerical_closes_method_and_provenance_stays_global",classification)
    def resource_abort():
        process=Mock(pid=680068);process.poll.return_value=None;process.wait.return_value=0
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);output=base/"worker";output.mkdir();job=base/"job.json";job.write_text("{}")
            (output/"summary.json").write_text('{"passed":true,"peak_bytes":0}')
            execution=types.SimpleNamespace(PYTHON="never-run",ENVIRONMENT={})
            oversized=types.SimpleNamespace(returncode=0,stdout=str(module.LIMIT_BYTES//1024+1))
            with patch.object(module,"verify"),patch.object(module,"technical_gate",return_value=execution), \
                    patch.object(module,"validate_job"),patch.object(module.subprocess,"Popen",return_value=process), \
                    patch.object(module.subprocess,"run",return_value=oversized),patch.object(module.time,"sleep"):
                try:module.supervise(job,output)
                except module.NativeMethodFailure:pass
                else:raise AssertionError("Aborted worker passed")
            record=json.loads((base/"supervision.json").read_text())
            assert not record["passed"] and record["aborted"]=="rss_over_6GiB" and record["exit_code"]==0
            process.terminate.assert_called_once()
    check("resource_abort_cannot_accept_success_summary",resource_abort)
    def source_boundaries():
        closure=json.loads((module.ROUND/"source_closure.json").read_text())
        assert not any(name.startswith("data/control_tabpfn_native_v1/") for name in closure)
        forbidden="data/control_tabpfn_native_v1/pure_tests.log"
        for name in ("protected_artifacts.json","head_runtime.json"):
            assert forbidden not in (module.ROUND/name).read_text()
        assert forbidden not in module.PLAN.read_text()
        tree=ast.parse(path.read_text());definitions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        worker=ast.unparse(definitions["worker"]);context=ast.unparse(definitions["context_record"])
        assert "execution.train_complete(model, torch, context, target, contract, output)" in worker
        assert "execution.canonical_predict" in worker and "weights_only=True" in worker and "strict=True" in worker
        assert "validation" not in ast.unparse(next(n for n in ast.walk(definitions["worker"]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="train_complete"))
        assert "execution.symmetric_statistics(augmented, permutation, signs)" in context
        assert "execution.preprocess(augmented, mean, scale)" in context
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="save" for n in ast.walk(definitions["context_record"]))
        controller=ast.parse(path.with_name("research_modernnca_native.py").read_text())
        run=next(n for n in controller.body if isinstance(n,ast.FunctionDef) and n.name=="run")
        assert ast.unparse(run.body[0]).startswith("plan = verify()")
        runtext=ast.unparse(run)
        assert "decode_owner(arrays, owner, eligible, broadcast, motion, raw, blocks)" in runtext
        assert "set(p['checks']) == required_guards(seed, p['protocol'])" in runtext
    check("exact_author_training_preprocess_and_controller_guards_reused",source_boundaries)
    def replay_finalization_proofs():
        # Literal bytes/JSON only: no arrays, estimator, fit or scientific import.
        compared=("initial_state_fingerprint.json","state_fingerprint.json","recipe.json",
            "resolved_optimizer.json","resolved_configuration.json","training_losses.npy","epoch_losses.npy",
            "original_probability.npy","swapped_probability.npy","owner.npy","repeat_first.npy")
        stages=(("screen",42),("confirmation_seed43",43),("confirmation_seed44",44))
        def put(path,value):module.write_json(path,value)
        def read(path):return json.loads(path.read_text())
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"ROUND",Path(folder).resolve()):
            root=Path(folder).resolve()
            for stage,seed in stages:
                for replay in (False,True):
                    output=root/(stage+"_replay" if replay else stage);output.mkdir()
                    for protocol in module.PROTOCOLS:
                        for block in range(1,4 if protocol=="buffered_100_seconds" else 6):
                            job_dir=output/protocol/f"block{block}";worker=job_dir/"worker";worker.mkdir(parents=True)
                            witness=f"literal-proof-seed{seed}-{protocol}-{block}".encode()
                            for name in ("context","target","original","swapped"):
                                (job_dir/(name+".npy")).write_bytes(witness+name.encode())
                            put(job_dir/"context_provenance.json",dict(seed=seed,protocol=protocol,block=block))
                            for name in compared:(worker/name).write_bytes(witness+name.encode())
                            (worker/"model.pt").write_bytes(witness+(b"new-archive" if replay else b"old-archive"))
                            for kind in ("stdout","stderr"):(job_dir/("worker."+kind+".log")).write_text("")
                            job={name:str(job_dir/(name+".npy")) for name in ("context","target","original","swapped")}
                            job.update(seed=seed,protocol=protocol,block=block,
                                input_sha256={str(job_dir/(name+".npy")):module.digest(job_dir/(name+".npy")) for name in ("context","target","original","swapped")})
                            if replay:job["parent"]=str(root/stage/protocol/f"block{block}"/"worker")
                            put(job_dir/"job.json",job)
                            put(worker/"summary.json",dict(passed=True,seed=seed,
                                checks=dict(independent_refit_exact=replay,strict_saved_reload_exact=True),artifact_sha256={}))
                            put(job_dir/"supervision.json",dict(passed=True,exit_code=0,aborted=None))
                    put(output/"manifest.json",dict(stage=stage,seed=seed,replay=replay,context_models=13,
                        independent_refits=13 if replay else 0,cache_sha256={}))
            def seal():
                for stage,seed in stages:
                    for replay in (False,True):
                        output=root/(stage+"_replay" if replay else stage)
                        for worker in output.rglob("worker"):
                            if not worker.is_dir():continue
                            summary=worker/"summary.json";record=read(summary)
                            record["artifact_sha256"]={p.name:module.digest(p) for p in worker.iterdir()
                                if p.is_file() and p.name not in ("summary.json","operation.json")};put(summary,record)
                            supervision=worker.parent/"supervision.json"
                            if supervision.exists():
                                record=read(supervision);record["summary_sha256"]=module.digest(summary)
                                for kind in ("stdout","stderr"):record[kind+"_sha256"]=module.digest(worker.parent/("worker."+kind+".log"))
                                put(supervision,record)
                        manifest=output/"manifest.json";record=read(manifest)
                        record["cache_sha256"]={str(p.relative_to(output)):module.digest(p) for p in output.rglob("*")
                            if p.is_file() and p!=manifest};put(manifest,record)
            seal();proof=module.verify_replay_evidence()
            assert proof["independent_process_fits"]==39 and len(proof["workers"])==39
            assert len({(p["seed"],p["worker"]) for p in proof["workers"]})==39
            def mutate(path,callback):
                before=path.read_bytes()
                try:
                    callback(path);seal();reject(module.verify_replay_evidence)
                finally:path.write_bytes(before);seal()
            def fields(path,callback):
                value=read(path);callback(value);put(path,value)
            manifest=root/"screen_replay/manifest.json"
            # A normal run renamed to *_replay cannot certify a single fresh refit.
            mutate(manifest,lambda p:fields(p,lambda v:v.update(replay=False,independent_refits=0)))
            mutate(manifest,lambda p:fields(p,lambda v:v.pop("replay")))
            mutate(manifest,lambda p:fields(p,lambda v:v.pop("independent_refits")))
            mutate(manifest,lambda p:fields(p,lambda v:v.update(context_models=12)))
            mutate(manifest,lambda p:fields(p,lambda v:v.update(seed=44)))
            job_dir=root/"screen_replay/buffered_minutes/block1";worker=job_dir/"worker"
            mutate(worker/"summary.json",lambda p:fields(p,lambda v:v["checks"].update(independent_refit_exact=False)))
            mutate(job_dir/"supervision.json",lambda p:p.unlink())
            mutate(job_dir/"supervision.json",lambda p:fields(p,lambda v:v.update(aborted="resource_abort")))
            mutate(job_dir/"job.json",lambda p:fields(p,lambda v:v.update(parent=str(worker))))
            mutate(job_dir/"job.json",lambda p:fields(p,lambda v:v.update(parent=str(root/"confirmation_seed43/buffered_minutes/block1/worker"))))
            # Re-seal every hash after tampering; actual comparison must still fail.
            mutate(worker/"state_fingerprint.json",lambda p:p.write_text("changed fitted-state witness"))
            renamed=worker.with_name("missing_worker")
            try:worker.rename(renamed);seal();reject(module.verify_replay_evidence)
            finally:renamed.rename(worker);seal()
            assert module.verify_replay_evidence()["independent_process_fits"]==39
    check("final_marker_requires39_actual_bound_refits_and_rejects_named_normal_runs",replay_finalization_proofs)
    return outcomes

if __name__=="__main__":
    import json
    print(json.dumps(dict(passed=True,tests=run_tests()),indent=2))
