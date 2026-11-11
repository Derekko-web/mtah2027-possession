"""Fresh-only qualification of the frozen corrected TabPFN hybrid.

Uses the unchanged native worker, longer dual-clock limits and fixed correction.
All original attempts and completed development artifacts remain immutable.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_tabpfn_fixed_confirmation_v2"
PLAN=ROOT/"experiments/TABPFN_FIXED_CONFIRMATION_PLAN_V2.json"
ANCESTOR=ROOT/"data/control_tabpfn_extended_v1"
DEVELOPMENT=ROOT/"data/control_tabpfn_block_fix_v2"
SOURCES=("experiments/tabpfn_fixed_confirmation_v2.py","experiments/research_tabpfn_fixed_confirmation_v2.py",
         "experiments/tabpfn_fixed_policy_v2.py","experiments/test_tabpfn_fixed_confirmation_v2.py")

spec=importlib.util.spec_from_file_location("tabpfn_fixed_frozen_execution",ROOT/"experiments/tabpfn_extended_evaluation.py")
engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
# Verify the preserved original execution before changing in-memory aliases.
engine.verify()
engine.ROUND,engine.PLAN,engine.__file__=ROUND,PLAN,str(Path(__file__).resolve())
engine._LEGACY=None
engine.reuse_completed=lambda job_path,output:False


def source_identity():
    return {name:engine.digest(ROOT/name) for name in SOURCES}


def read(path):return json.loads(Path(path).read_text())


def compare_development_worker(job_dir):
    """Fresh seed42 fits must exactly reproduce the original native model output."""
    job=read(job_dir/"job.json")
    if job["seed"]!=42:return
    original=ANCESTOR/"screen"/job["protocol"]/f"block{job['block']}"
    for name in ("context.npy","target.npy","original.npy","swapped.npy","fitting.npz","context_provenance.json"):
        if (job_dir/name).read_bytes()!=(original/name).read_bytes():
            raise ValueError("Fresh context/query/provenance differs from development: "+name)
    for name in ("checkpoint_fingerprint.json","preprocessing_fingerprint.json","recipe.json",
                 "original_probability.npy","swapped_probability.npy","owner.npy","repeat_first.npy"):
        if (job_dir/"worker"/name).read_bytes()!=(original/"worker"/name).read_bytes():
            raise ValueError("Fresh native model reconstruction differs: "+name)


def verify_replay_evidence():
    """13 fresh development refits plus 26 independent confirmation refits."""
    engine.verify()
    expected={f"{p}/block{b}/worker" for p in engine.PROTOCOLS
              for b in range(1,4 if p=="buffered_100_seconds" else 6)}
    semantic=("checkpoint_fingerprint.json","preprocessing_fingerprint.json","recipe.json",
              "original_probability.npy","swapped_probability.npy","owner.npy","repeat_first.npy")
    proofs=[];stage_hashes={}
    for stage,seed in (("screen",42),("confirmation_seed43",43),("confirmation_seed44",44)):
        names=[stage] if seed==42 else [stage,stage+"_replay"]
        for name in names:
            folder=ROUND/name;manifest=engine.audit(folder)
            actual={str(p.relative_to(folder)) for p in folder.rglob("worker") if p.is_dir()}
            files={str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and p.name!="manifest.json"}
            is_replay=name.endswith("_replay")
            if (actual!=expected or set(manifest["cache_sha256"])!=files or manifest.get("seed")!=seed
                    or manifest.get("context_models")!=13 or manifest.get("fresh_context_fits")!=13
                    or manifest.get("reused_predecessor_workers")!=0 or manifest.get("replay") is not is_replay):
                raise ValueError("Incomplete all-fresh qualification/replay stage")
            stage_hashes[name]=engine.digest(folder/"manifest.json")
            for relative in sorted(expected):
                worker=folder/relative;job_dir=worker.parent;job=read(job_dir/"job.json")
                engine.validate_job(job)
                summary=read(worker/"summary.json");supervision=read(job_dir/"supervision.json")
                if (job["seed"]!=seed or ("parent" in job) is not is_replay
                        or summary.get("passed") is not True or summary.get("seed")!=seed
                        or summary.get("fresh_context_replay") is not is_replay
                        or any(summary.get(key) is not True for key in ("complete_raw_preprocessing_immutable",
                            "checkpoint_exact","resolved_model_configuration_immutable","repeat_byte_exact"))
                        or supervision.get("passed") is not True or supervision.get("exit_code")!=0
                        or supervision.get("aborted") is not None
                        or supervision.get("summary_sha256")!=engine.digest(worker/"summary.json")
                        or supervision.get("peak_owned_rss_plus_full_driver_bytes",engine.LIMIT_BYTES+1)>engine.LIMIT_BYTES
                        or not engine.within(supervision.get("elapsed",{}),3600)
                        or supervision["elapsed"]!=engine.elapsed(supervision["started"],supervision["ended"])):
                    raise ValueError("Invalid fresh native fit/state/resource proof")
                engine.validate_operations(worker)
                artifacts={p.name for p in worker.iterdir() if p.is_file()}-{"summary.json","memory.json","memory.tmp"}
                if set(summary["artifact_sha256"])!=artifacts or not set(semantic)<=artifacts:
                    raise ValueError("Incomplete fresh worker artifact closure")
                for item,sha in summary["artifact_sha256"].items():
                    if engine.digest(worker/item)!=sha:raise ValueError("Fresh artifact changed")
                for kind in ("stdout","stderr"):
                    if supervision[kind+"_sha256"]!=engine.digest(job_dir/("worker."+kind+".log")):
                        raise ValueError("Fresh worker log changed")
                if seed==42:
                    compare_development_worker(job_dir)
                    proofs.append(dict(seed=42,worker=relative,reference="preserved development context",fresh_fit=True))
                elif is_replay:
                    parent=ROUND/stage/relative
                    for item in ("context.npy","target.npy","original.npy","swapped.npy","fitting.npz","context_provenance.json"):
                        if (job_dir/item).read_bytes()!=(parent.parent/item).read_bytes():
                            raise ValueError("Independent replay regenerated inputs differ")
                    for item in semantic:
                        if (worker/item).read_bytes()!=(parent/item).read_bytes():
                            raise ValueError("Independent fresh model replay differs")
                    proofs.append(dict(seed=seed,worker=relative,reference=str(parent.relative_to(ROUND)),fresh_fit=True))
    if len(proofs)!=39:raise ValueError("Exactly39 independent native context rebuilds are required")
    return dict(independent_process_fits=39,actual_fresh_fits_total=65,
                seed42_exact_development_refits=13,seed43_44_independent_refits=26,
                stage_manifest_sha256=stage_hashes,workers=proofs)


def preflight():
    engine.verify_predecessor();engine.technical_gate()
    for name in SOURCES:compile((ROOT/name).read_bytes(),str(ROOT/name),"exec")
    tests_path=ROOT/SOURCES[3]
    spec=importlib.util.spec_from_file_location("fixed_tabpfn_pure_tests",tests_path)
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    outcomes=tests.run_tests(sys.modules.get(__name__) or globals())
    if any(n in sys.modules for n in ("numpy","torch","pandas","sklearn","tabpfn")):
        raise ValueError("Scientific import during source-only confirmation preflight")
    engine.write_json(ROUND/"source_preflight.json",dict(passed=True,source_sha256=source_identity(),
        plan_sha256=engine.digest(PLAN),tests=outcomes,scientific_imports=False,model_calls=False,
        fresh_fits_only=True,original_native_worker_unchanged=True))


def freeze():
    preflight()
    review=read(ROUND/"root_review.json")
    if (review.get("approved_for_freeze") is not True or review.get("source_sha256")!=source_identity()
            or review.get("plan_sha256")!=engine.digest(PLAN)):
        raise ValueError("Exact source/plan review required before fresh model fits")
    folder=ROUND/"protocol_freeze";folder.mkdir(exist_ok=False)
    (folder/"plan.json").write_bytes(PLAN.read_bytes())
    closure=read(ANCESTOR/"protocol_freeze/implementation.json");closure.update(source_identity())
    closure.update(read(ROOT/"data/control_tabpfn_block_diagnosis_v1/correction_freeze/implementation.json"))
    for path in (PLAN,ROUND/"source_preflight.json",ROUND/"root_review.json",ROUND/"preparation_ledger.csv",
                 ROUND/"hardware_probe.json",ROUND/"acquisition.json",ROUND/"user_ranking_direction.json"):
        closure[str(path.relative_to(ROOT))]=engine.digest(path)
    engine.write_json(folder/"implementation.json",closure)
    engine.write_json(folder/"input_sha256.json",read(PLAN)["input_sha256"])
    for name in ("protected_artifacts.json","head_runtime.json","symlinks.json"):
        (folder/name).write_bytes((ANCESTOR/"protocol_freeze"/name).read_bytes())
    (folder/"EXPERIMENT_LEDGER.csv").write_bytes((ROUND/"preparation_ledger.csv").read_bytes())
    engine.write_json(folder/"freeze.json",dict(source_sha256=source_identity(),plan_sha256=engine.digest(PLAN),
        fresh_fits_only=True,correction_parameters_frozen=True,model_calls=False,
        operation_seconds=900,worker_seconds=3600,full_build_authorized=False))
    engine.verify()


def __getattr__(name):return getattr(engine,name)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);actions=parser.add_mutually_exclusive_group(required=True)
    for name in ("preflight","freeze","verify","worker"):actions.add_argument("--"+name,action="store_true")
    parser.add_argument("--job",type=Path);parser.add_argument("--output",type=Path);args=parser.parse_args()
    if args.preflight:preflight()
    elif args.freeze:freeze()
    elif args.verify:engine.verify()
    else:
        engine.require_admission()
        try:engine.legacy().worker(args.job,args.output)
        except BaseException as error:
            if args.output and args.output.exists():engine.write_json(args.output/"failure.json",
                dict(error=repr(error),traceback=traceback.format_exc(),fresh_fixed_confirmation=True))
            raise
