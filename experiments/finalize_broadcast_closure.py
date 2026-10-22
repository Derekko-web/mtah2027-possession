"""Official scores, complete replay and prospective incremental-primary checks."""

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import map_anchors
from broadcast_consensus import decode_broadcast_consensus
from broadcast_features import aligned_features
from broadcast_intersection import intersect_broadcast
from broadcast_refinement import decode_broadcast_closure
from broadcast_provenance import CAMERA_ASSETS,verify_camera_inputs
from broadcast_source_intervals import source_intervals
from build_temporal_candidate import digest
from control_refinement import fast_columns,predict_control
from stoppage_consensus import decode_stoppage_consensus
from temporal_candidate import predict_models
from xgboost_baseline import CLASSES


def codes(labels): return np.array([np.flatnonzero(CLASSES == x)[0] for x in labels])


def official_scores(folder,scores,bootstrap=0):
    rows = []
    for protocol in sorted(set(scores.protocol)):
        with (folder/protocol/"official.log").open("w") as log:
            subprocess.run([sys.executable,str(ROOT/"evaluation/run_evaluation.py"),"--gt","public","--results",str(folder/protocol/"scorer_inputs"),
                "--out",str(folder/protocol/"official"),"--bootstrap",str(bootstrap),"--no-detail"],check=True,stdout=log,stderr=subprocess.STDOUT)
        for score in scores[scores.protocol == protocol].to_dict("records"):
            report = json.loads((folder/protocol/"official/reports"/(score["model"]+".json")).read_text())
            if abs(report["spell_f1_mean"]-score["spellF1"]) > 1e-12 or abs(report["macro_f1"]-score["macroF1"]) > 1e-12:
                raise AssertionError("Official scoring disagrees")
            match = report["spell_f1"]["0.5"]
            rows.append(dict(protocol=protocol,model=score["model"],spellF1=report["spell_f1_mean"],macroF1=report["macro_f1"],
                matched_iou50=match["n_matched"],spurious_iou50=match["n_spurious"],missed_iou50=match["n_missed"],
                bootstrap90_low=report.get("spell_bootstrap_low"),bootstrap90_high=report.get("spell_bootstrap_high")))
    result = pd.DataFrame(rows)
    result.to_csv(folder/"official_comparison.csv",index=False)
    return result


def promotion_checks(scores,segments,official,plan):
    index,counts = scores.set_index(["protocol","model"]),official.set_index(["protocol","model"])
    expected = len(plan["protocols"])*len(plan["seeds"])*2
    if len(scores) != expected or index.index.has_duplicates or len(official) != expected or counts.index.has_duplicates:
        raise ValueError("Incomplete/duplicate confirmation")
    pairs = []
    for protocol in plan["protocols"]:
        for seed in plan["seeds"]:
            old_name,new_name = f"current_seed{seed}",f"broadcast_closure_seed{seed}"
            old,new = index.loc[(protocol,old_name)],index.loc[(protocol,new_name)]
            a,b = counts.loc[(protocol,old_name)],counts.loc[(protocol,new_name)]
            short = segments[(segments.protocol == protocol) & (segments.label != "D") & (segments.frames <= 125)]
            before = short[short.model == old_name].set_index(["start","end","label"])
            after = short[short.model == new_name].set_index(["start","end","label"])
            pd.testing.assert_index_equal(before.index,after.index)
            flags = dict(primary_increment=bool(new.spellF1 >= old.spellF1+.01-1e-12),
                macro=bool(new.macroF1 >= old.macroF1-.001-1e-12),short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                prior_short_recovery=bool((after.recall[before.recall >= .5] >= .5).all()),
                no_extra_spurious=bool(b.spurious_iou50 <= a.spurious_iou50),no_extra_missed=bool(b.missed_iou50 <= a.missed_iou50))
            pairs.append(dict(protocol=protocol,seed=seed,checks=flags,primary_delta=float(new.spellF1-old.spellF1),
                macro_delta=float(new.macroF1-old.macroF1),short_iou_delta=float(new.short_live_mean_best_iou-old.short_live_mean_best_iou)))
    means = {protocol:float(np.mean([p["primary_delta"] for p in pairs if p["protocol"] == protocol])) for protocol in plan["protocols"]}
    substantial = all(p["primary_delta"] >= .04-1e-12 for p in pairs if p["protocol"] == "buffered_minutes")
    return dict(passed=all(all(p["checks"].values()) for p in pairs) and all(v > 0 for v in means.values()),
                original_buffered_substantial_criterion_met=substantial,pairs=pairs,mean_protocol_primary_gain=means)


def audit_manifest(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name,expected in manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != expected: raise ValueError("Saved source differs: "+name)
    inputs = manifest.get("input_sha256",{})
    if set(inputs).intersection(CAMERA_ASSETS):
        verify_camera_inputs(inputs)
    else:
        for name,expected in inputs.items():
            if Path(name).is_absolute(): raise ValueError("Input identity must be path independent")
            if digest(ROOT/name) != expected: raise ValueError("Saved input differs: "+name)
    for name,expected in manifest.get("cache_sha256",{}).items():
        if digest(folder/name) != expected: raise ValueError("Saved cache differs: "+name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-history",action="store_true",help="Also verify the original local source snapshot and protected artifacts")
    args = parser.parse_args()
    root,source = ROOT/"data/control_broadcast_v1",ROOT/"data/advanced_validation"
    cache = root/"inputs"
    cm = audit_manifest(cache)
    for family in ("screen","source_screen","intersection_screen","closure_confirmation"): audit_manifest(root/family)
    for name,expected in (json.loads((root/"baseline_manifest.json").read_text()).items() if args.audit_history else []):
        if name == "experiments/EXPERIMENT_LEDGER.csv": continue
        path = ROOT/name
        if name == "experiments/SELECTED_MODEL.md" and digest(path) != expected:
            path = ROOT/"experiments/STOPPAGE_MODEL.md"
        if digest(path) != expected: raise ValueError("Protected artifact changed")
    selected = json.loads((ROOT/"data/candidate_stoppage_consensus/manifest.json").read_text())
    for name,expected in selected["source_sha256"].items():
        path = ROOT/name
        if not path.is_file(): path = ROOT/"experiments"/name
        if digest(path) != expected: raise ValueError("Selected source changed")
    after = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    if args.audit_history:
        with tarfile.open(root/"experiments_before.tar.gz") as archive:
            before = pd.read_csv(io.BytesIO(archive.extractfile("experiments/EXPERIMENT_LEDGER.csv").read()))
        keys = ["round","protocol","model","source"]
        preserved = after.merge(before[keys],on=keys,how="inner",validate="one_to_one")
        pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True),preserved[before.columns].sort_values(keys).reset_index(drop=True))
    t,b = [dict(np.load(cache/(name+"_observations.npz"))) for name in ("tactical","broadcast")]
    alignment_plan = json.loads((cache/"plan.json").read_text())
    anchors = dict(np.load(cache/"anchors.npz"))
    mapping = map_anchors(anchors,t,b,alignment_plan["alignment"]["profiles"][cm["selected_profile"]])
    saved_mapping = np.load(cache/(cm["selected_profile"]+"_mapping.npz"))
    for name,value in mapping.items(): np.testing.assert_array_equal(value,saved_mapping[name])
    same,combined = aligned_features(t,b,mapping)
    for name,value in (("same",same),("combined",combined)):
        pd.testing.assert_frame_equal(value,pd.read_pickle(cache/(name+"_inputs.pkl")))
    raw,base,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","base","ordered")]
    same,combined = same.iloc[:len(raw)],combined.iloc[:len(raw)]
    views = dict(broadcast_fast_same=base[fast_columns(base)].join(same),broadcast_fast_cross=base[fast_columns(base)].join(combined),
        broadcast_joint_frame=ordered.join(combined),broadcast_joint_runs=ordered.join(combined))
    public = np.arange(11185,18685)
    splits = json.loads((source/"splits.json").read_text())
    confirmation_plan = json.loads((ROOT/"experiments/BROADCAST_CLOSURE_CONFIRMATION_PLAN.json").read_text())
    model_reloads,stage_rows,all_deltas = 0,{},{}
    for family in ("screen","source_screen","intersection_screen","closure_confirmation"):
        folder = root/family
        scores = pd.read_csv(folder/"scores.csv")
        deltas = []
        for protocol in sorted(set(scores.protocol)):
            folds = [s for s in splits if s["protocol"] == protocol]
            blocks = [(min(s["validation"]),max(s["validation"])+1) for s in folds]
            table = pd.read_csv(folder/protocol/"oof_predictions.csv")
            original = pd.read_csv(root/"baseline_confirmation"/protocol/"oof_predictions.csv")
            seeds = confirmation_plan["seeds"] if family == "closure_confirmation" else [42]
            for seed in seeds:
                paths = [source/protocol/f"{name}_seed{seed}_probabilities.npy" for name in ("ordered_xgboost_frame","previous_ensemble")]
                paths += [ROOT/"data/control_boundary_v1/baseline_confirmation"/protocol/f"fast_seed{seed}_probabilities.npy",
                    root/"baseline_confirmation"/protocol/f"heads_seed{seed}_probabilities.npy"]
                arrays = []
                for path,width in zip(paths,(3,3,2,2)):
                    value = np.full((len(raw),width),np.nan); value[public] = np.load(path); arrays.append(value)
                reference = decode_stoppage_consensus(*arrays,raw,blocks)[public]
                np.testing.assert_array_equal(reference,original[f"stoppage_consensus_seed{seed}"])
                np.testing.assert_array_equal(reference,table[f"current_seed{seed}"])
                if family in ("screen","closure_confirmation"):
                    names = list(views) if family == "screen" else ["broadcast_joint_frame"]
                    for name in names:
                        joint = "joint" in name
                        probability = np.full((len(raw),3 if joint else 2),np.nan)
                        for fold in folds:
                            model = XGBClassifier(); model.load_model(folder/protocol/f"{name}_seed{seed}_block{fold['block']}.ubj")
                            val = np.array(fold["validation"])
                            probability[val] = predict_models([model],views[name].iloc[val]) if joint else predict_control(model,views[name].iloc[val])
                            model_reloads += 1
                        np.testing.assert_array_equal(probability[public],np.load(folder/protocol/f"{name}_seed{seed}_probabilities.npy"))
                        if family == "screen":
                            for mode in ("replace_head","short_proposals"):
                                prediction = decode_broadcast_consensus(*arrays,probability,combined,raw,blocks,joint,mode)[public]
                                np.testing.assert_array_equal(prediction,table[f"{name}_{mode}_seed{seed}"])
                        else:
                            prediction = decode_broadcast_closure(*arrays,probability,combined,raw,blocks)[public]
                            np.testing.assert_array_equal(prediction,table[f"broadcast_closure_seed{seed}"])
                            np.testing.assert_array_equal(prediction == "D",reference == "D")
                else:
                    plan = json.loads((folder/"plan.json").read_text())
                    learned_table = pd.read_csv(root/"screen"/protocol/"oof_predictions.csv")
                    for name,config in plan["screen"]["methods"].items():
                        learned = codes(learned_table[f"{config['model']}_replace_head_seed42"])
                        prediction = reference.copy(); states = codes(reference)
                        for fold in folds:
                            val = np.array(fold["validation"]); ix = val-public[0]
                            result = (source_intervals(states[ix],learned[ix],combined.iloc[val],config["half_window"]) if family == "source_screen" else
                                      intersect_broadcast(states[ix],learned[ix],combined.iloc[val],config["intersection"]))
                            prediction[ix] = CLASSES[result]
                        np.testing.assert_array_equal(prediction,table[f"{name}_seed42"])
                        np.testing.assert_array_equal(prediction == "D",reference == "D")
                folds_table = pd.read_csv(folder/"fold_scores.csv")
                old = folds_table[(folds_table.protocol == protocol) & (folds_table.model == f"current_seed{seed}")]
                for method in scores[(scores.protocol == protocol) & scores.model.str.endswith(f"_seed{seed}")].model:
                    if method == f"current_seed{seed}": continue
                    new = folds_table[(folds_table.protocol == protocol) & (folds_table.model == method)]
                    pair = old.merge(new,on=["protocol","minute"],suffixes=["_before","_after"],validate="one_to_one")
                    for row in pair.to_dict("records"):
                        deltas.append(dict(protocol=protocol,model=method,block=row["minute"],before=row["spellF1_before"],after=row["spellF1_after"],delta=row["spellF1_after"]-row["spellF1_before"]))
        official = official_scores(folder,scores,1000 if family == "closure_confirmation" else 0)
        pd.DataFrame(deltas).to_csv(folder/"block_deltas.csv",index=False)
        stage_rows[family],all_deltas[family] = len(scores),dict(comparisons=len(deltas),primary_regressions=sum(r["delta"] < -1e-12 for r in deltas))
        if family == "closure_confirmation":
            checks = promotion_checks(scores,pd.read_csv(folder/"segment_diagnostics.csv"),official,confirmation_plan)
            checks.update(shared_inference_parity=True,identical_D_masks=True,official_score_agreement=True)
            (folder/"promotion_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    audit = dict(historical_snapshot_audited=args.audit_history,protected_baseline_unchanged=True if args.audit_history else None,
        old_ledger_rows=len(before) if args.audit_history else None,old_ledger_rows_preserved=True if args.audit_history else None,ledger_rows=len(after),
        alignment_mapping_features_regenerated_exactly=True,physical_profile_selection_without_targets=cm["no_target_labels_read"],
        public_prefix_parity=cm["actual_padded_feature_parity"],source_input_cache_match=True,models_reload_exactly=model_reloads,
        official_shared_replay_rows=stage_rows,block_deltas=all_deltas,incremental_confirmation_passed=checks["passed"],
        original_substantial_criterion_met=checks["original_buffered_substantial_criterion_met"])
    (root/"integrity.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__": main()
