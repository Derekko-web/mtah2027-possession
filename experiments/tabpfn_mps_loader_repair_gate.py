"""R59: reuse frozen R57 gate with independently copied/resealed loader metadata.
No numerical function, model setting, dependency version or replay guard is changed.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/"data/control_tabpfn_mps_loader_repair_v1"
OLD=ROOT/"data/control_tabpfn_mps_v1"
SOURCE=ROOT/"experiments/tabpfn_mps_runtime_gate.py"
PLAN=ROOT/"experiments/TABPFN_MPS_LOADER_REPAIR_PLAN.json"
DEPS=ROUND/"runtime_deps"
SOURCE_SHA256="518dd0ef0b765e4337b8f036243cc4fe624bce1ef58a454900129a5038ea6d93"
OLD_PLAN_SHA256="a11e21bdcb903adc39bb58e4e6cdf6145c5560202a9163439c08de52564cc4bb"

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def original_checks():
    if digest(SOURCE)!=SOURCE_SHA256:
        raise ValueError("Frozen R57 source changed")
    if digest(ROOT/"experiments/TABPFN_MPS_RUNTIME_PLAN.json")!=OLD_PLAN_SHA256:
        raise ValueError("Frozen R57 plan changed")
    original=json.loads((OLD/"protocol_freeze/implementation.json").read_text())
    for path,expected in original.items():
        if digest(ROOT/path)!=expected:
            raise ValueError("Original frozen R57 input changed: "+path)
    return original

def helper():
    path=ROUND/"prepare_loader_repair.py"
    spec=importlib.util.spec_from_file_location("r59_static_repair",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def preflight():
    """All proofs here use stdlib parsing, byte hashes and codesign verification only."""
    started=time.monotonic()
    for path in [Path(__file__),ROUND/"prepare_loader_repair.py",SOURCE]:
        compile(path.read_bytes(),str(path),"exec")
    original=original_checks()
    manifest=json.loads((ROUND/"repair_manifest.json").read_text())
    static=helper()
    patch_map={r["path"]:r for r in manifest["records"]}
    checked=0
    for path,expected in original.items():
        old=ROOT/path
        try: relative=str(old.relative_to(OLD/"runtime_deps"))
        except ValueError: continue
        new=DEPS/relative
        wanted=patch_map.get(relative,{}).get("new_sha256",expected)
        if digest(new)!=wanted: raise ValueError("Unexpected composite-file difference: "+relative)
        checked+=1
    for record in manifest["records"]:
        old=(OLD/"runtime_deps"/record["path"]).read_bytes()
        new=(DEPS/record["path"]).read_bytes()
        proof=static.prove_diff(old,new,static.parse(old),static.parse(new))
        if proof!=record["proof"]: raise ValueError("Byte-complete repair proof changed")
        result=subprocess.run(["/usr/bin/codesign","--verify","--strict",str(DEPS/record["path"])],
            capture_output=True,text=True)
        if result.returncode: raise ValueError("Repaired signature verification failed: "+record["path"])
    distributions={d.metadata["Name"].lower().replace("_","-"):d.version
                   for d in metadata.distributions(path=[str(DEPS)])}
    acquisition=json.loads((ROUND/"acquisition.json").read_text())
    if distributions!={k:v["version"] for k,v in acquisition["resolved_runtime_distributions"].items()}:
        raise ValueError("Composite metadata versions differ")
    if (ROUND/"hardware_probe.json").read_bytes()!=(OLD/"hardware_probe.json").read_bytes():
        raise ValueError("Hardware fraction/seed probe changed")
    configured=core()  # Definitions and stdlib imports only; no worker or model call.
    if configured.ROUND!=ROUND or configured.DEPS!=DEPS or configured.PLAN!=PLAN:
        raise ValueError("Frozen core execution-location binding differs")
    if Path(configured.__file__).resolve()!=Path(__file__).resolve() or configured.verify is not verify:
        raise ValueError("Worker spawn or complete verification binding differs")
    if Path(configured.worker.__code__.co_filename).resolve()!=SOURCE.resolve():
        raise ValueError("Worker is not the original frozen R57 function")
    if any(name in sys.modules for name in ("torch","numpy","sklearn","scipy","tabpfn")):
        raise ValueError("Scientific import occurred before freeze")
    save(ROUND/"source_preflight.json",dict(passed=True,seconds=time.monotonic()-started,
        implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),
        repair_manifest_sha256=digest(ROUND/"repair_manifest.json"),
        original_frozen_files_verified=len(original),composite_files_verified=checked,
        patched_files_verified=len(manifest["records"]),signatures_strictly_verified=True,
        unchanged_core_definition_binding_verified=True,scientific_imports=False,model_forward=False))
    print("R59 static/header/signature/source preflight passed; no scientific imports",flush=True)

def freeze():
    review=json.loads((ROUND/"independent_review.json").read_text())
    if not review["approved_for_freeze"] or review["implementation_sha256"]!=digest(__file__) or review["plan_sha256"]!=digest(PLAN):
        raise ValueError("Independent approval must bind final wrapper/plan")
    if review["repair_manifest_sha256"]!=digest(ROUND/"repair_manifest.json"):
        raise ValueError("Approval must bind complete metadata/payload proof")
    original_checks()
    checks={str(Path(__file__).relative_to(ROOT)):digest(__file__),str(PLAN.relative_to(ROOT)):digest(PLAN)}
    links={}
    for path in sorted(ROUND.rglob("*")):
        if "protocol_freeze" in path.parts or "runtime_assessment" in path.parts: continue
        name=str(path.relative_to(ROOT))
        if path.is_symlink(): links[name]=os.readlink(path)
        elif path.is_file():
            if path.suffix==".pyc" or "__pycache__" in path.parts: raise ValueError("Unexpected generated bytecode")
            checks[name]=digest(path)
    for path in [OLD/"protocol_freeze/implementation.json",OLD/"protocol_freeze/freeze.json",
                 OLD/"runtime_assessment/summary.json",
                 OLD/"runtime_assessment/moderate/first.stderr.log",
                 OLD/"runtime_assessment/moderate/first.supervision.json"]:
        checks[str(path.relative_to(ROOT))]=digest(path)
    folder=ROUND/"protocol_freeze";folder.mkdir(exist_ok=False)
    (folder/"plan.json").write_bytes(PLAN.read_bytes())
    save(folder/"implementation.json",checks);save(folder/"symlinks.json",links)
    save(folder/"freeze.json",dict(implementation_sha256=digest(__file__),plan_sha256=digest(PLAN),
        repair_manifest_sha256=digest(ROUND/"repair_manifest.json"),
        original_r57_source_sha256=SOURCE_SHA256,original_plan_sha256=OLD_PLAN_SHA256,
        physical_files=len(checks),symlink_entries=len(links),
        all_original_r57_inputs_still_bound=True,original_r57_remains_failed=True,
        numerical_interface_changed=False,scientific_imports_started=False,model_forward_started=False))
    print("Frozen R59 loader-only repair; no model import or forward",flush=True)

def verify():
    folder=ROUND/"protocol_freeze"
    if PLAN.read_bytes()!=(folder/"plan.json").read_bytes(): raise ValueError("R59 plan changed")
    for path,expected in json.loads((folder/"implementation.json").read_text()).items():
        if digest(ROOT/path)!=expected: raise ValueError("R59 frozen physical file changed: "+path)
    for path,target in json.loads((folder/"symlinks.json").read_text()).items():
        link=ROOT/path
        if not link.is_symlink() or os.readlink(link)!=target: raise ValueError("Composite symlink changed: "+path)
    original_checks()

def core():
    if digest(SOURCE)!=SOURCE_SHA256: raise ValueError("R57 implementation binding changed")
    spec=importlib.util.spec_from_file_location("r59_frozen_r57_core",SOURCE)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Only execution locations and verification binding change. All math, workers,
    # dispatch, datasets, resource limits and exact-replay functions are the original.
    module.ROUND=ROUND;module.DEPS=DEPS;module.PLAN=PLAN
    module.__file__=str(Path(__file__).resolve())  # Original supervisor spawns this wrapper.
    module.verify=verify
    return module

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group()
    action.add_argument("--preflight",action="store_true")
    action.add_argument("--freeze",action="store_true")
    action.add_argument("--verify",action="store_true")
    action.add_argument("--worker",action="store_true")
    parser.add_argument("--size",choices=("moderate","full"))
    parser.add_argument("--stage",choices=("first","rebuild"))
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    output=(args.output or ROUND/"runtime_assessment").resolve()
    if not output.is_relative_to(ROUND.resolve()): parser.error("R59 artifacts must remain in this round")
    if args.preflight: preflight()
    elif args.freeze: freeze()
    elif args.verify:
        verify();print("R59 and all original R57 frozen inputs verified",flush=True)
    else:
        execution=core()
        if args.worker:
            if not args.size or not args.stage: parser.error("Worker requires size and stage")
            try: execution.worker(args.size,args.stage,output)
            except BaseException as error:
                if output.exists(): save(output/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),
                    synthetic_only=True,competition_inputs_read=False,original_r57_remains_failed=True))
                raise
        else: execution.main(output)
