"""Build the eight-model D-only motion candidate from verified full local features."""

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
from broadcast_provenance import verify_camera_inputs
from broadcast_refinement import decode_broadcast_closure
from build_broadcast_candidate import check_sources
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest
from games import GAME,half_video_blocks,source_frames
from motion_dead_trim import decode_motion_dead_trim
from research_control_renewal import fit_renewal
from temporal_candidate import build_inputs,predict_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/candidate_motion_dead_trim")
    args = parser.parse_args();output = args.output.resolve()
    if output.exists(): parser.error("Choose a fresh output directory to preserve candidates")
    root = ROOT/"data/control_pair_motion_v1";reference = ROOT/"data/candidate_broadcast_closure"
    prior = json.loads((reference/"manifest.json").read_text())
    expected = "dcb3a8d2ec3fbc1d2308c85303a995f6102112d4976e466942556626223a05ce"
    if prior["csv_sha256"] != expected or digest(reference/"candidate.csv") != expected: raise ValueError("Selected reference differs")
    check_sources(prior["source_sha256"]);check_sources(prior["reference_source_sha256"]);verify_camera_inputs(prior["input_sha256"])
    for group in ("model_sha256","probability_sha256"):
        for name,sha in prior[group].items():
            if digest(reference/name) != sha: raise ValueError("Retained reference artifact changed")
    confirmation = root/"dead_trim_confirmation";confirmed = audit_manifest(confirmation)
    checks_path = confirmation/"promotion_checks.json";checks = json.loads(checks_path.read_text())
    if not checks["passed"] or not checks["shared_inference_parity"] or not checks["official_score_agreement"]:
        raise ValueError("D-only confirmation has not passed")
    plan_path = ROOT/"experiments/MOTION_DEAD_TRIM_PLAN.json"
    if confirmed["plan_sha256"] != digest(plan_path) or (confirmation/"plan.json").read_bytes() != plan_path.read_bytes():
        raise ValueError("Dead-trim confirmation plan differs")
    cache = root/"full_inputs";cm = audit_manifest(cache);prefix = audit_manifest(root/"inputs")
    if not cm["public_motion_feature_parity"] or not cm["targets_not_read"] or cm["n_frames"] != source_frames():
        raise ValueError("Full motion cache is not verified")
    if cm["plan_sha256"] != prefix["plan_sha256"] or cm["plan_sha256"] != digest(ROOT/"experiments/PAIR_MOTION_PLAN.json"):
        raise ValueError("Public/full motion configurations differ")
    upstream = ROOT/"data/control_broadcast_v1";audit_manifest(upstream/"full_inputs");audit_manifest(upstream/"inputs")
    print("Rebuilding original full inputs for parity with unchanged selected models",flush=True)
    raw,base,ordered = build_inputs(source_frames());public = np.arange(11185,18685)
    pd.testing.assert_frame_equal(ordered.iloc[public],pd.read_pickle(ROOT/"data/advanced_validation/ordered.pkl").iloc[public])
    features = pd.read_pickle(cache/"broadcast_motion_inputs.pkl")
    pd.testing.assert_frame_equal(features.iloc[public],pd.read_pickle(root/"inputs/broadcast_motion_inputs.pkl").iloc[public])
    info = pd.read_pickle(cache/"motion_info.pkl")
    pd.testing.assert_frame_equal(info.iloc[public],pd.read_pickle(root/"inputs/motion_info.pkl")[list(info)].iloc[public])
    inputs = ordered.join(features)
    if list(inputs) != confirmed["feature_columns"]: raise ValueError("Full and confirmed feature schemas differ")
    gt = ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv"
    if digest(gt) != prior["public_gt_sha256"]: raise ValueError("Public targets changed")
    labels = pm.read_label_csv(gt).label.to_numpy();train = np.flatnonzero(labels != "X")
    np.testing.assert_array_equal(train,public)
    output.mkdir(parents=True)
    print("Fitting only new motion joint head on7500 public targets",flush=True)
    model = fit_renewal(inputs,labels,train,"frame",42);model.save_model(output/"broadcast_motion.ubj")
    loaded = XGBClassifier();loaded.load_model(output/"broadcast_motion.ubj")
    probability = np.empty((len(raw),3))
    for start in range(0,len(raw),20000):
        end = min(start+20000,len(raw));probability[start:end] = predict_models([model],inputs.iloc[start:end])
        if not np.array_equal(probability[start:end],predict_models([loaded],inputs.iloc[start:end])): raise AssertionError("Full motion model reload differs")
    np.save(output/"motion_probabilities.npy",probability)
    arrays = [np.load(reference/(name+"_probabilities.npy")) for name in ("current","previous","fast","heads","broadcast")]
    bc = pd.read_pickle(upstream/"full_inputs/combined_inputs.pkl")
    blocks = [(start,end+1) for _,start,end in half_video_blocks(GAME)]
    original = decode_broadcast_closure(*arrays,bc,raw,blocks)
    np.testing.assert_array_equal(original,pd.read_csv(reference/"candidate.csv").label.to_numpy())
    print("Decoding each official half with reference nonD preserved",flush=True)
    prediction = decode_motion_dead_trim(*arrays,probability,bc,info,raw,blocks)
    np.testing.assert_array_equal(prediction[original != "D"],original[original != "D"])
    if ((prediction == "D")&(original != "D")).any(): raise AssertionError("Full D trim addedD")
    table = pd.DataFrame(dict(frame=np.arange(len(raw)),label=prediction));reason = pm.validate_submission(table,source_frames())
    if reason: raise ValueError(reason)
    table.to_csv(output/"candidate.csv",index=False)
    for name in list(prior["model_sha256"])+list(prior["probability_sha256"]): shutil.copy2(reference/name,output/name)
    (output/"reference_manifest.json").write_bytes((reference/"manifest.json").read_bytes())
    names = ("build_motion_dead_trim_candidate.py","motion_dead_trim.py","pair_candidate.py","pair_motion.py","pair_motion_inference.py",
        "prepare_pair_motion.py","prepare_full_pair_motion.py","confirm_motion_dead_trim.py","finalize_motion_dead_trim.py",
        "research_control_renewal.py","PAIR_MOTION_PLAN.json","PAIR_EVENT_SYMMETRY_PLAN.json","MOTION_DEAD_TRIM_PLAN.json")
    result = dict(model="Selected7-model broadcast closure plus native motion head,only D->H/A decisions",seed=42,
        csv_sha256=digest(output/"candidate.csv"),reference_csv_sha256=expected,
        source_sha256={str(Path("experiments")/name):digest(ROOT/"experiments"/name) for name in names},
        reference_source_sha256={**prior["source_sha256"],**prior["reference_source_sha256"]},
        model_sha256={p.name:digest(p) for p in sorted(output.glob("*.ubj"))},probability_sha256={p.name:digest(p) for p in sorted(output.glob("*.npy"))},
        input_sha256=prior["input_sha256"],public_gt_sha256=digest(gt),
        full_cache_manifest_sha256=digest(cache/"manifest.json"),full_cache_sha256=cm["cache_sha256"],
        prefix_cache_manifest_sha256=digest(root/"inputs/manifest.json"),confirmation_plan_sha256=digest(plan_path),promotion_checks_sha256=digest(checks_path),
        public_training_count=7500,retained_seven_models_probabilities_unchanged=True,new_model_reload_exact=True,reference_decode_identical=True,
        reference_nonD_predictions_exact=True,D_mask_subset_of_reference=True,changed_frames=int((prediction != original).sum()),
        output_class_counts=pd.Series(prediction).value_counts().to_dict(),
        selected_objective="Separate frozen incremental falseD reduction; original substantial/extra-short-match criteria remain unmet",
        original_substantial_primary_criterion_met=False,original_extra_buffered_short_match_criterion_met=False,
        limitations=["Adaptive public same-match evidence,not independent/private accuracy or verified maximum score",
            "Old asymmetric event decoder retained for historical comparison; unused by D-only production inference",
            "No commits,uploads,submissions,publication or external contact"])
    (output/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Saved",output/"candidate.csv",flush=True);print("CSV SHA256",result["csv_sha256"],"changed frames",result["changed_frames"],flush=True)


if __name__ == "__main__": main()
