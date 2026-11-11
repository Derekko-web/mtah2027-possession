"""Prepare a read-only comparison manifest; never import or run a model.

This does not accept a license, download weights, or start queued inference.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/control_tabpfn35_v1"
PREVIOUS = ROOT / "data/control_tabpfn_fixed_confirmation_v2/screen"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    inputs = {}
    jobs = []
    for block in range(1, 6):
        folder = PREVIOUS / "buffered_minutes" / f"block{block}"
        job = json.loads((folder / "job.json").read_text())
        if job["seed"] != 42 or job["block"] != block or job["protocol"] != "buffered_minutes":
            raise ValueError("Unexpected original main-fold identity")
        for name in ("context", "target", "original", "swapped"):
            path = Path(job[name])
            actual = digest(path)
            if job["input_sha256"].get(str(path)) != actual:
                raise ValueError("Original confirmation input hash differs")
            inputs[str(path.relative_to(ROOT))] = actual
        for name in ("job.json", "fitting.npz", "context_provenance.json", "worker/owner.npy"):
            path = folder / name
            inputs[str(path.relative_to(ROOT))] = digest(path)
        jobs.append(str((folder / "job.json").relative_to(ROOT)))
    for name in (
        "evaluation/gt/BAR-ATH_possession_gt_public.csv",
        "evaluation/possession_metrics.py",
        "data/advanced_validation/splits.json",
        "data/control_tabpfn_fixed_confirmation_v2/screen/buffered_minutes/oof_predictions.csv",
        "data/control_tabpfn_fixed_confirmation_v2/screen/scores.csv",
        "experiments/tabpfn_short_control.py",
        "experiments/tabpfn_short_control_inference.py",
        "experiments/prepare_tabpfn35_comparison.py",
    ):
        inputs[name] = digest(ROOT / name)
    plan = dict(
        status="PREPARED_NOT_STARTED",
        model="TabPFN-3.5 Base local; not hosted Thinking",
        purpose="Compare model version on the identical public out-of-fold task",
        comparator_main_spellF1=0.7771428571428571,
        metric="Official spell_f1_mean; macroF1 and spurious counts reported as diagnostics",
        folds=jobs,
        seed=42,
        retained_configuration=dict(n_estimators=8, softmax_temperature=0.9,
            balance_probabilities=False, average_before_softmax=False,
            inference_precision="float32", canonical_batch_rows=512,
            paired_home_away_queries=True),
        comparisons="Use same 166 features, fold-specific paired contexts/targets/imputation and fixed short-control decoder",
        prohibitions=["No full-match or private ground truth", "No within-sample score presented as out-of-fold", "No best seed or fold selection", "No changes to current build, dependencies or selected model", "No match-derived data sent to hosted services"],
        admission=["User's applicable license acceptance and legitimate model access established", "Exact downloaded model and runtime hashes bound before evaluation", "Existing full build AND its finalization terminal; no old worker/controller remains", "Separate bounded synthetic resource/reproducibility check passes before public fits"],
        resource_policy="Serial jobs only after the current workflow completes; combined 6 GiB ceiling, 5 GiB GPU allocation, 900-second operation and 3600-second worker limits. Fail rather than silently change model or precision.",
        outcome="Require all five main folds and complete 7500-frame out-of-fold predictions before reporting a comparable primary score. Strictly above comparator means advance to the other eight existing folds and seeds 43/44; no automatic model replacement in this experiment.",
        interpretation="Adaptive comparison on the same 17 public possessions, not an independent full-match accuracy estimate",
        input_sha256=inputs,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "comparison_plan.json"
    content = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise ValueError("Refusing to overwrite a different prepared plan")
    path.write_text(content)
    print(json.dumps(dict(plan=str(path), folds=len(jobs), bound_files=len(inputs), model_run_started=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
