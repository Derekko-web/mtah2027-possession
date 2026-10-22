"""Reproduce the seven-model incremental broadcast-closure candidate locally."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from broadcast_refinement import decode_broadcast_closure
from broadcast_provenance import verify_camera_inputs
from build_temporal_candidate import digest
from games import GAME,half_video_blocks,require,source_frames
from research_control_renewal import fit_renewal
from stoppage_consensus import decode_stoppage_consensus
from temporal_candidate import build_inputs,predict_models


def check_sources(mapping):
    for name,expected in mapping.items():
        path = ROOT/name
        if not path.is_file(): path = ROOT/"experiments"/name
        if digest(path) != expected: raise ValueError("Source changed: "+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/candidate_broadcast_closure")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): parser.error("Choose a fresh output directory to preserve all candidates")
    root = ROOT/"data/control_broadcast_v1"
    reference = ROOT/"data/candidate_stoppage_consensus"
    prior = json.loads((reference/"manifest.json").read_text())
    expected = "dac48e6be66f4395089cae4eee3cfb5daeadcd7963480f399fe079d871be534c"
    if digest(reference/"candidate.csv") != expected or prior["csv_sha256"] != expected: raise ValueError("Unexpected selected reference")
    check_sources(prior["source_sha256"]); check_sources(prior["reference_source_sha256"])
    if {Path(path).name:digest(path) for path in require(["atd_meta","atd_raw","smart_tagging"])} != prior["input_sha256"]:
        raise ValueError("Original tactical inputs changed")
    for group in ("model_sha256","probability_sha256"):
        for name,sha in prior[group].items():
            if digest(reference/name) != sha: raise ValueError("Reference artifact differs: "+name)
    confirmation = root/"closure_confirmation"
    checks_path = confirmation/"promotion_checks.json"
    checks = json.loads(checks_path.read_text())
    if not checks["passed"] or not checks["shared_inference_parity"] or not checks["official_score_agreement"]:
        raise ValueError("Incremental confirmation has not passed")
    plan_path = ROOT/"experiments/BROADCAST_CLOSURE_CONFIRMATION_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if (confirmation/"plan.json").read_bytes() != plan_path.read_bytes(): raise ValueError("Confirmation plan changed")
    check_sources(plan["source_sha256"])
    for name,sha in plan["input_sha256"].items():
        if digest(ROOT/name) != sha: raise ValueError("Confirmation input changed")
    cache = root/"full_inputs"
    cm = json.loads((cache/"manifest.json").read_text())
    if cm["plan_sha256"] != digest(ROOT/"experiments/BROADCAST_ALIGNMENT_PLAN.json"):
        raise ValueError("Full alignment plan changed")
    if not cm["public_mapping_feature_parity"] or not cm["no_target_labels_read"] or cm["n_frames"] != source_frames():
        raise ValueError("Full broadcast feature cache is not verified")
    check_sources(cm["source_sha256"])
    verify_camera_inputs(cm["input_sha256"])
    for name,sha in cm["cache_sha256"].items():
        if digest(cache/name) != sha: raise ValueError("Full broadcast cache changed")
    prefix = root/"inputs"
    if digest(prefix/"manifest.json") != cm["public_prefix_manifest_sha256"]: raise ValueError("Public prefix provenance differs")
    pmf = json.loads((prefix/"manifest.json").read_text())
    if pmf["plan_sha256"] != cm["plan_sha256"] or pmf["selected_profile"] != cm["selected_profile"]:
        raise ValueError("Full/public alignment configuration differs")
    check_sources(pmf["source_sha256"])
    for name,sha in pmf["cache_sha256"].items():
        if digest(prefix/name) != sha: raise ValueError("Public prefix cache changed")
    print("Rebuilding full original inputs for retained/source-model parity",flush=True)
    raw,base,ordered = build_inputs(source_frames())
    public = np.arange(11185,18685)
    source = ROOT/"data/advanced_validation"
    pd.testing.assert_frame_equal(base.iloc[public],pd.read_pickle(source/"base.pkl").iloc[public])
    pd.testing.assert_frame_equal(ordered.iloc[public],pd.read_pickle(source/"ordered.pkl").iloc[public])
    features = pd.read_pickle(cache/"combined_inputs.pkl")
    pd.testing.assert_frame_equal(features.iloc[public],pd.read_pickle(prefix/"combined_inputs.pkl").iloc[public])
    inputs = ordered.join(features)
    confirmed = json.loads((confirmation/"manifest.json").read_text())
    if list(inputs) != confirmed["feature_columns"]: raise ValueError("Full model schema differs from OOF schema")
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    if digest(gt) != prior["public_gt_sha256"]: raise ValueError("Public targets changed")
    labels = pm.read_label_csv(gt).label.to_numpy()
    train = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(train,public)
    output.mkdir(parents=True)
    print("Fitting only the new joint broadcast head on 7500 public targets",flush=True)
    model = fit_renewal(inputs,labels,train,"frame",42)
    model.save_model(output/"broadcast_joint_frame.ubj")
    reloaded = XGBClassifier(); reloaded.load_model(output/"broadcast_joint_frame.ubj")
    probability = np.empty((len(raw),3))
    for start in range(0,len(raw),20000):
        end = min(start+20000,len(raw))
        probability[start:end] = predict_models([model],inputs.iloc[start:end])
        np.testing.assert_array_equal(probability[start:end],predict_models([reloaded],inputs.iloc[start:end]))
    np.save(output/"broadcast_probabilities.npy",probability)
    arrays = [np.load(reference/(name+"_probabilities.npy")) for name in ("current","previous","fast","heads")]
    blocks = [(start,end+1) for _,start,end in half_video_blocks(GAME)]
    previous = decode_stoppage_consensus(*arrays,raw,blocks)
    np.testing.assert_array_equal(previous,pd.read_csv(reference/"candidate.csv").label.to_numpy())
    print("Decoding reference and source closure independently in each half",flush=True)
    prediction = decode_broadcast_closure(*arrays,probability,features,raw,blocks)
    np.testing.assert_array_equal(prediction == "D",previous == "D")
    table = pd.DataFrame(dict(frame=np.arange(len(raw)),label=prediction))
    reason = pm.validate_submission(table,source_frames())
    if reason: raise ValueError(reason)
    table.to_csv(output/"candidate.csv",index=False)
    for name in list(prior["model_sha256"])+list(prior["probability_sha256"]): shutil.copy2(reference/name,output/name)
    (output/"reference_manifest.json").write_bytes((reference/"manifest.json").read_bytes())
    sources = ("build_broadcast_candidate.py","broadcast_refinement.py","broadcast_consensus.py","broadcast_intersection.py",
        "broadcast_alignment.py","broadcast_features.py","broadcast_provenance.py","contact_dynamics.py","prepare_broadcast_baseline.py","prepare_broadcast_alignment.py","prepare_full_broadcast_features.py",
        "confirm_broadcast_closure.py","finalize_broadcast_closure.py","research_control_renewal.py",
        "BROADCAST_ALIGNMENT_PLAN.json","BROADCAST_SOURCE_INTERVAL_PLAN.json","BROADCAST_INTERSECTION_PLAN.json","BROADCAST_CLOSURE_CONFIRMATION_PLAN.json")
    manifest = dict(model="Selected six models plus joint broadcast head and source closure",seed=42,
        csv_sha256=digest(output/"candidate.csv"),reference_csv_sha256=expected,
        source_sha256={str(Path("experiments")/name):digest(ROOT/"experiments"/name) for name in sources},
        reference_source_sha256={**prior["source_sha256"],**prior["reference_source_sha256"]},
        model_sha256={path.name:digest(path) for path in sorted(output.glob("*.ubj"))},
        probability_sha256={path.name:digest(path) for path in sorted(output.glob("*.npy"))},
        input_sha256=cm["input_sha256"],public_gt_sha256=digest(gt),
        full_cache_manifest_sha256=digest(cache/"manifest.json"),full_cache_sha256=cm["cache_sha256"],
        public_cache_manifest_sha256=digest(prefix/"manifest.json"),confirmation_plan_sha256=digest(plan_path),promotion_checks_sha256=digest(checks_path),
        public_training_count=7500,retained_six_model_artifacts_unchanged=True,reference_decode_identical=True,new_model_reload_exact=True,
        reference_D_mask_identical=True,output_class_counts=pd.Series(prediction).value_counts().to_dict(),
        changed_frames=int((prediction != previous).sum()),original_substantial_primary_criterion_met=False,
        selected_objective="Separately frozen adaptive incremental-primary confirmation, not prior substantial/extra-short-match gates.",
        limitations=["All accuracy evidence is adaptive same-match public development; private/full-match accuracy is unknown.",
            "No additional short IoU.5 match was obtained in the main protocol.","Alignment has physical/synthetic checks but no correspondence answer key.",
            "Nothing committed, submitted, uploaded or published."])
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Saved",output/"candidate.csv",flush=True)
    print("CSV SHA256",manifest["csv_sha256"],"changed frames",manifest["changed_frames"],flush=True)


if __name__ == "__main__": main()
