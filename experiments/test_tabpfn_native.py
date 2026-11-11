"""R65 pure invented-list tests. No scientific packages, model or native arrays."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

def run_tests():
    path=Path(__file__).with_name("tabpfn_native.py")
    spec=importlib.util.spec_from_file_location("r65_pure_subject",path)
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
    def incomplete_technical_evidence():
        # Source-only invented JSON; all variants must fail before stage payloads.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);assessment=root/"runtime_assessment";assessment.mkdir()
            plan=root/"plan.json";summary=assessment/"summary.json"
            normal_digest=module.digest;calls=[]
            fake=SimpleNamespace(verify=lambda:calls.append("preserved_bindings_checked"))
            def checksum(path):
                if Path(path)==module.GATE_SOURCE:return module.GATE_SHA256
                if Path(path)==module.GATE_PLAN:return module.GATE_PLAN_SHA256
                return normal_digest(path)
            records=[dict(size=s,stage=t) for s in ("moderate","full") for t in ("first","rebuild")]
            for passed,retained in ((False,records),(True,records[:3]),(True,records[:3]+[records[2]])):
                module.write_json(summary,dict(passed=passed,implementation_sha256=module.GATE_SHA256,
                    plan_sha256=module.GATE_PLAN_SHA256,records=retained))
                module.write_json(plan,dict(technical_prerequisite=dict(
                    actual_all_four_stage_summary_sha256=normal_digest(summary),metadata_sha256={})))
                with patch.object(module,"GATE_ROUND",root),patch.object(module,"PLAN",plan), \
                        patch.object(module,"digest",checksum),patch.object(module,"load_module",return_value=fake):
                    reject(module.technical_gate)
            assert calls==["preserved_bindings_checked"]*3
    check("failed_missing_or_duplicated_technical_stages_cannot_admit_native_work",incomplete_technical_evidence)
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
        selections.append(dict(passed=True,pairs=[dict(protocol=p,seed=seed,model=f"tabpfn_context_seed{seed}",checks={name:True for name in module.required_guards(seed,p)}) for p in module.PROTOCOLS]))
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
    def replay_finalization_proofs():
        # Literal invented bytes/JSON only, never parse or load scientific arrays.
        compared=("checkpoint_fingerprint.json","preprocessing_fingerprint.json","recipe.json",
            "original_probability.npy","swapped_probability.npy","owner.npy","repeat_first.npy")
        stages=(("screen",42),("confirmation_seed43",43),("confirmation_seed44",44))
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
                            for name in ("context","target","original","swapped"):
                                (job_dir/(name+".npy")).write_bytes(witness+name.encode())
                            (job_dir/"fitting.npz").write_bytes(witness+b"invented preprocessing record")
                            put(job_dir/"context_provenance.json",dict(seed=seed,protocol=protocol,block=block))
                            for name in compared:(worker/name).write_bytes(witness+name.encode())
                            for kind in ("stdout","stderr"):(job_dir/("worker."+kind+".log")).write_text("")
                            job={name:str(job_dir/(name+".npy")) for name in ("context","target","original","swapped")}
                            job.update(seed=seed,protocol=protocol,block=block,
                                input_sha256={str(job_dir/(name+".npy")):module.digest(job_dir/(name+".npy")) for name in ("context","target","original","swapped")})
                            if replay:job["parent"]=str(root/stage/protocol/f"block{block}"/"worker")
                            put(job_dir/"job.json",job)
                            put(worker/"summary.json",dict(passed=True,seed=seed,protocol=protocol,block=block,
                                fresh_context_replay=replay,complete_raw_preprocessing_immutable=True,checkpoint_exact=True,
                                resolved_model_configuration_immutable=True,repeat_byte_exact=True,artifact_sha256={}))
                            put(job_dir/"supervision.json",dict(passed=True,exit_code=0,aborted=None))
                    put(output/"manifest.json",dict(stage=stage,seed=seed,replay=replay,context_models=13,
                        independent_context_rebuild_count=13 if replay else 0,cache_sha256={}))
            def seal():
                for stage,seed in stages:
                    for replay in (False,True):
                        output=root/(stage+"_replay" if replay else stage)
                        for worker in output.rglob("worker"):
                            if not worker.is_dir():continue
                            summary=worker/"summary.json";record=read(summary)
                            record["artifact_sha256"]={p.name:module.digest(p) for p in worker.iterdir()
                                if p.is_file() and p.name not in ("summary.json","memory.json","memory.tmp")};put(summary,record)
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
                try:callback(path);seal();reject(module.verify_replay_evidence)
                finally:path.write_bytes(before);seal()
            def fields(path,callback):
                value=read(path);callback(value);put(path,value)
            manifest=root/"screen_replay/manifest.json"
            # Includes normal --output *_replay with no --replay flag: no fresh fits.
            mutate(manifest,lambda p:fields(p,lambda v:v.update(replay=False,independent_context_rebuild_count=0)))
            mutate(manifest,lambda p:fields(p,lambda v:v.pop("replay")))
            mutate(manifest,lambda p:fields(p,lambda v:v.pop("independent_context_rebuild_count")))
            mutate(manifest,lambda p:fields(p,lambda v:v.update(stage="confirmation_seed43")))
            mutate(manifest,lambda p:fields(p,lambda v:v.update(context_models=12)))
            mutate(manifest,lambda p:fields(p,lambda v:v.update(seed=44)))
            job_dir=root/"screen_replay/buffered_minutes/block1";worker=job_dir/"worker"
            mutate(worker/"summary.json",lambda p:fields(p,lambda v:v.update(fresh_context_replay=False)))
            mutate(worker/"summary.json",lambda p:fields(p,lambda v:v.update(passed=False)))
            mutate(job_dir/"supervision.json",lambda p:p.unlink())
            mutate(job_dir/"supervision.json",lambda p:fields(p,lambda v:v.update(aborted="resource_abort")))
            mutate(job_dir/"supervision.json",lambda p:fields(p,lambda v:v.update(exit_code=1)))
            mutate(job_dir/"job.json",lambda p:fields(p,lambda v:v.update(parent=str(worker))))
            mutate(job_dir/"job.json",lambda p:fields(p,lambda v:v.update(parent=str(root/"confirmation_seed43/buffered_minutes/block1/worker"))))
            # Re-seal every hash after tampering; actual byte comparison must reject.
            for name in compared:
                mutate(worker/name,lambda p:p.write_bytes(b"changed proof with all hashes re-sealed"))
            for name in ("fitting.npz","context_provenance.json"):
                mutate(job_dir/name,lambda p:p.write_bytes(b"changed context proof with all hashes re-sealed"))
            payload=job_dir/"context.npy";job=job_dir/"job.json";prior_job=job.read_bytes()
            def change_payload(path):
                path.write_bytes(b"changed context bytes");fields(job,lambda v:v["input_sha256"].update({str(path):module.digest(path)}))
            try:mutate(payload,change_payload)
            finally:job.write_bytes(prior_job);seal()
            renamed=worker.with_name("missing_worker")
            try:worker.rename(renamed);seal();reject(module.verify_replay_evidence)
            finally:renamed.rename(worker);seal()
            assert module.verify_replay_evidence()["independent_process_fits"]==39
            spec=importlib.util.spec_from_file_location("r65_pure_full_controller",path.with_name("research_tabpfn_native.py"))
            controller=importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules,{"tabpfn_native":module}):spec.loader.exec_module(controller)
            with patch.object(controller,"ROUND",root),patch.object(controller,"verify"),patch.object(controller,"require_confirmations"):
                controller.replay_complete()
            completed=read(root/"replay_complete.json")
            assert completed["passed"] is True and completed["independent_process_fits"]==39
            assert len(completed["replay_evidence"]["workers"])==39 and completed["full_build_authorized"] is False
    check("final_marker_requires39_actual_bound_context_rebuilds_and_rejects_forged_proofs",replay_finalization_proofs)
    def renamed_normal_run_is_not_replay():
        # Exercise the real finalizer with matching OOF bytes but no replay workers.
        spec=importlib.util.spec_from_file_location("r65_pure_controller",path.with_name("research_tabpfn_native.py"))
        controller=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{"tabpfn_native":module}):spec.loader.exec_module(controller)
        with tempfile.TemporaryDirectory() as folder,patch.object(module,"ROUND",Path(folder)),patch.object(controller,"ROUND",Path(folder)), \
                patch.object(controller,"verify"),patch.object(controller,"require_confirmations"):
            for stage in ("screen","confirmation_seed43","confirmation_seed44"):
                for suffix in ("","_replay"):
                    output=Path(folder)/(stage+suffix);output.mkdir()
                    module.write_json(output/"manifest.json",dict(cache_sha256={}))
                    for protocol in module.PROTOCOLS:
                        target=output/protocol;target.mkdir();(target/"oof_predictions.csv").write_text("invented identical OOF\n")
            reject(controller.replay_complete)
            assert not (Path(folder)/"replay_complete.json").exists()
    check("matching_OOF_in_named_replay_folders_without_workers_is_rejected",renamed_normal_run_is_not_replay)
    return outcomes

if __name__=="__main__":
    import json
    print(json.dumps(dict(passed=True,tests=run_tests()),indent=2))
