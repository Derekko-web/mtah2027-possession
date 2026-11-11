"""Serial continuation of the already started fresh confirmation; no retries."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "data/control_tabpfn_fixed_confirmation_v2"
PYTHON = ROOT / "data/control_isolated_python_v1/head_runtime/bin/python"
DRIVER = "experiments/research_tabpfn_fixed_confirmation_v2.py"
AUDITOR = "experiments/audit_tabpfn_fixed_confirmation_v2.py"


def run(label, script, *args):
    print("Starting", label, flush=True)
    with (ROUND / (label + ".log")).open("x") as handle:
        subprocess.run([str(PYTHON), "-B", script, *args], cwd=ROOT,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print("Completed", label, flush=True)


def main():
    # The original model controller is the only process running seed42.
    while True:
        try:
            os.kill(65788, 0)
        except ProcessLookupError:
            break
        time.sleep(5)
    if not (ROUND / "screen/manifest.json").is_file():
        raise RuntimeError("Initial fresh seed42 did not finish; no retry or partial rescue")
    run("audit_seed42", AUDITOR, "--seed", "42", "--append-ledger")
    for seed in (43, 44):
        run(f"fresh_seed{seed}", DRIVER, "--seed", str(seed))
        run(f"audit_seed{seed}", AUDITOR, "--seed", str(seed), "--append-ledger")
    selections = [json.loads((ROUND / "ranking_policy" / (s + "_selection.json")).read_text())
                  for s in ("screen", "confirmation_seed43", "confirmation_seed44")]
    if not all(s["passed"] for s in selections):
        (ROUND / "workflow_quality_rejected.json").write_text(json.dumps(dict(
            complete_initial_confirmation=True, all9_pairs_passed=False,
            replacement_authorized=False, selected_model_changed=False,
            next="Report complete fresh qualification; preserve current model"), indent=2) + "\n")
        print("Complete confirmation failed qualification; current model retained", flush=True)
        return
    run("final_selection", DRIVER, "--final-selection")
    for seed in (43, 44):
        run(f"replay_seed{seed}", DRIVER, "--seed", str(seed), "--replay")
        run(f"audit_replay_seed{seed}", AUDITOR, "--seed", str(seed), "--replay", "--append-ledger")
    run("verify_complete_replay", DRIVER, "--replay-complete")
    print("Fresh qualification and independent replay complete; full build remains required", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        path = ROUND / "workflow_failure.json"
        with path.open("x") as handle:
            json.dump(dict(error=repr(error), traceback=traceback.format_exc(),
                           retries=False, selected_model_changed=False), handle, indent=2)
            handle.write("\n")
        raise
