"""Local historical audit of final identity, weak exclusion and reproducibility.

Requires this run's preserved history; ordinary fresh candidate generation does
not require the historical snapshots checked here.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from build_temporal_candidate import digest
from games import require

ROOT = Path(__file__).resolve().parents[1]


def main():
    root = ROOT/"data/control_renewal_v1"
    candidate, rebuild = ROOT/"data/candidate_stoppage_consensus", ROOT/"data/candidate_stoppage_consensus_rebuild"
    manifest = json.loads((candidate/"manifest.json").read_text())
    for group in ("source_sha256", "reference_source_sha256"):
        for name, expected in manifest[group].items():
            if digest(ROOT/name) != expected:
                raise ValueError("Current source differs: "+name)
    for group in ("model_sha256", "probability_sha256"):
        for name, expected in manifest[group].items():
            if digest(candidate/name) != expected or digest(rebuild/name) != expected:
                raise ValueError("Final model/probabilities differ: "+name)
    csv_hash = digest(candidate/"candidate.csv")
    if csv_hash != manifest["csv_sha256"] or digest(rebuild/"candidate.csv") != csv_hash:
        raise ValueError("Independent full CSV differs")
    if (candidate/"manifest.json").read_bytes() != (rebuild/"manifest.json").read_bytes():
        raise ValueError("Final manifests differ")
    if digest(root/"provenance_before/candidate_stoppage_consensus/candidate.csv") != csv_hash:
        raise ValueError("Provenance-only repair changed predictions")
    actual_inputs = {Path(p).name: digest(p) for p in require(["atd_meta", "atd_raw", "smart_tagging"])}
    if actual_inputs != manifest["input_sha256"]:
        raise ValueError("Actual competition inputs differ")
    cache = root/"stoppage_inputs"
    for name, expected in manifest["cache_sha256"].items():
        if digest(cache/name) != expected:
            raise ValueError("Current feature/weak cache differs")
    cache_manifest = json.loads((cache/"manifest.json").read_text())
    for name, expected in cache_manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected:
            raise ValueError("Cached feature source differs")
    for name in ("raw.pkl", "physical.pkl", "restart.pkl"):
        pd.testing.assert_frame_equal(pd.read_pickle(cache/name), pd.read_pickle(root/"provenance_before/stoppage_inputs"/name))
    weak = np.load(cache/"weak_samples.npz")
    before_weak = np.load(root/"provenance_before/stoppage_inputs/weak_samples.npz")
    for name in weak.files:
        np.testing.assert_array_equal(weak[name], before_weak[name])
    frames = weak["frames"]
    if np.any((frames >= 10935) & (frames <= 18934)):
        raise ValueError("Weak rows touch buffered public window")
    reserved = ROOT/"data/continuation_20261021/annotation_freeze/windows.json"
    if digest(reserved) != manifest["annotation_manifest_sha256"]:
        raise ValueError("Reserved metadata differs")
    for window in json.loads(reserved.read_text())["windows"]:
        if np.any((frames >= window["start"]-250) & (frames <= window["end"]+250)):
            raise ValueError("Weak rows touch reserved windows")
    for protocol in ("buffered_minutes", "purged_spells", "buffered_100_seconds"):
        name = Path(protocol)/"oof_predictions.csv"
        table = pd.read_csv(root/"recovery_confirmation"/name)
        pd.testing.assert_frame_equal(table,
                                      pd.read_csv(root/"provenance_before/recovery_confirmation"/name))
        for seed in (42, 43, 44):
            live = table.ground_truth.to_numpy() != "D"
            np.testing.assert_array_equal(table.loc[live, f"current_seed{seed}"].to_numpy(),
                                          table.loc[live, f"stoppage_consensus_seed{seed}"].to_numpy())
            name = Path(protocol)/f"heads_seed{seed}_probabilities.npy"
            np.testing.assert_array_equal(np.load(root/"recovery_confirmation"/name),
                                          np.load(root/"provenance_before/recovery_confirmation"/name))
    heads = np.load(candidate/"heads_probabilities.npy")
    for index, (name, features) in enumerate((("weak_physical_model.ubj", "physical.pkl"),
                                             ("weak_restart_model.ubj", "restart.pkl"))):
        model = XGBClassifier()
        model.load_model(candidate/name)
        inputs = pd.read_pickle(cache/features)
        for start in range(0, len(inputs), 20000):
            end = min(len(inputs), start+20000)
            np.testing.assert_array_equal(heads[start:end, index], model.predict_proba(inputs.iloc[start:end])[:, 1])
    reference = json.loads((ROOT/"data/candidate_anchored_control/manifest.json").read_text())
    for group in ("reference_model_sha256", "probability_sha256"):
        for name, expected in reference[group].items():
            if digest(candidate/name) != expected:
                raise ValueError("Retained model/probability changed")
    protected = {
        "candidate_anchored_control": "973e1670e9e1240ecdde3bbabebf1e803f0d3b6259aad1b9b3223c413d26dcfd",
        "candidate_control_consensus": "19dd8ecac4fe0c3fcbdc11d18d41cd9a38d1cf995a0defb05ff816b6057a64f1",
        "candidate_temporal_xgb": "bcbd287f437b2404d46a86e6d3a7c0b9fc44c30feb58ed5fdd44f84482d4ec72",
    }
    for folder, expected in protected.items():
        if digest(ROOT/"data"/folder/"candidate.csv") != expected:
            raise ValueError("Protected candidate changed")
    initial = ROOT/"data/candidate"
    if digest(initial/"candidate.csv") != json.loads((initial/"manifest.json").read_text())["csv_sha256"]:
        raise ValueError("Original ensemble changed")
    before = pd.read_csv(root/"review_reference/experiments/EXPERIMENT_LEDGER.csv")
    after = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    keys = ["round", "record_type", "protocol", "model", "source"]
    old = before.set_index(keys)
    pd.testing.assert_frame_equal(old, after.set_index(keys).loc[old.index, old.columns], check_dtype=False)
    checks = dict(csv_sha256=csv_hash, csv_rebuild_identical=True, final_manifests_identical=True,
                  provenance_repair_prediction_parity=True, regenerated_feature_and_weak_parity=True,
                  regenerated_all18_OOF_prediction_parity=True, current_source_input_cache_hashes_match=True,
                  all9_public_live_prediction_parity=True,
                  model_probability_hashes_match=True, new_models_reload_exactly=True, retained_models_probabilities_unchanged=True,
                  weak_exclusions_pass=True, weak_sample_count=len(frames), reserved_manifest_sha256=digest(reserved),
                  prior_candidates_preserved=True, old_ledger_rows_preserved=len(before), ledger_rows=len(after))
    (root/"integrity.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
