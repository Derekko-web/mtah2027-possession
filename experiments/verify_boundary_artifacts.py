"""Verify final candidate identity, preserved history, and independent rebuild."""

import json
from pathlib import Path

import pandas as pd

from build_temporal_candidate import digest
from games import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT/"data/control_boundary_v1"
    candidate = ROOT/"data/candidate_anchored_control"
    rebuild = ROOT/"data/candidate_anchored_control_rebuild"
    manifest = json.loads((candidate/"manifest.json").read_text())
    for group in ("source_sha256", "reference_source_sha256"):
        for path, expected in manifest[group].items():
            if digest(ROOT/path) != expected:
                raise ValueError("Source hash differs: "+path)
    for group in ("reference_model_sha256", "probability_sha256"):
        for name, expected in manifest[group].items():
            if digest(candidate/name) != expected or digest(rebuild/name) != expected:
                raise ValueError("Model/probability hash differs: "+name)
    inputs = {Path(path).name: digest(path) for path in require(["atd_meta", "atd_raw", "smart_tagging"])}
    if inputs != manifest["input_sha256"]:
        raise ValueError("Competition input hashes differ")
    if digest(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv") != manifest["public_gt_sha256"]:
        raise ValueError("Public labels differ")
    csv_hash = digest(candidate/"candidate.csv")
    if csv_hash != manifest["csv_sha256"] or digest(rebuild/"candidate.csv") != csv_hash:
        raise ValueError("Rebuilt candidate differs")
    if (candidate/"manifest.json").read_bytes() != (rebuild/"manifest.json").read_bytes():
        raise ValueError("Final manifests differ")
    # First successful generation predates a finalizer CLI-only edit. Its stale
    # manifest remains preserved; its CSV is an independent generation check.
    first = output/"first_build_before_finalizer_cli"
    if digest(first/"candidate.csv") != csv_hash:
        raise ValueError("Independent first-generation CSV differs")
    old = pd.read_csv(output/"review_reference/experiments/EXPERIMENT_LEDGER.csv")
    new = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    keys = ["round", "record_type", "protocol", "model", "source"]
    before = old.set_index(keys)
    pd.testing.assert_frame_equal(before, new.set_index(keys).loc[before.index, before.columns], check_dtype=False)
    reference_hash = "19dd8ecac4fe0c3fcbdc11d18d41cd9a38d1cf995a0defb05ff816b6057a64f1"
    for path in (ROOT/"data/candidate_control_consensus/candidate.csv", output/"baseline_candidate/candidate.csv",
                 output/"baseline_rebuild/candidate.csv"):
        if digest(path) != reference_hash:
            raise ValueError("Consensus reference differs")
    if digest(ROOT/"data/candidate_temporal_xgb/candidate.csv") != "bcbd287f437b2404d46a86e6d3a7c0b9fc44c30feb58ed5fdd44f84482d4ec72":
        raise ValueError("Preceding ordered candidate differs")
    annotation_hash = digest(ROOT/"data/continuation_20261021/annotation_freeze/windows.json")
    if annotation_hash != "5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c":
        raise ValueError("Frozen annotation manifest differs")
    checks = dict(csv_sha256=csv_hash, csv_rebuild_identical=True, independent_first_csv_identical=True,
                  final_manifests_identical=True, source_hashes_match=True, input_hashes_match=True,
                  model_hashes_match=True, probability_hashes_match=True,
                  saved_model_predictions_match=manifest["saved_model_probability_parity"],
                  padded_feature_parity=manifest["padded_feature_parity"], prior_candidates_preserved=True,
                  ledger_rows=len(new), prior_ledger_rows_preserved=len(old), annotation_manifest_sha256=annotation_hash)
    (output/"integrity.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
