"""Replay all jersey experiments, official scores and frozen substantial-gain gates."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_temporal_candidate import digest
from control_refinement import fast_columns,predict_control
from finalize_broadcast_closure import official_scores
from games import GAME,half_video_blocks,require
from jersey_color import weak_frame_split
from jersey_ownership import decode_jersey_ownership
from temporal_candidate import predict_models
from xgboost_baseline import CLASSES


def codes(labels): return np.array([np.flatnonzero(CLASSES == value)[0] for value in labels])


def audit_manifest(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name,sha in manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != sha: raise ValueError("Source changed: "+name)
    for name,sha in manifest.get("input_sha256",{}).items():
        if name.startswith("tactical/"): path = Path(require([name.split("/")[1]])[0])
        else: path = ROOT/name
        if digest(path) != sha: raise ValueError("Input changed: "+name)
    for name,sha in manifest.get("cache_sha256",{}).items():
        if digest(folder/name) != sha: raise ValueError("Cache changed: "+name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-history",action="store_true",help="Also verify the original local source archive and protected baseline")
    args = parser.parse_args()
    root = ROOT/"data/control_jersey_v1"; folder = root/"screen"
    fm,cm,sm = [audit_manifest(root/name) for name in ("feasibility","inputs","screen")]
    plan = json.loads((folder/"plan.json").read_text())
    if sm["plan_sha256"] != digest(ROOT/"experiments/JERSEY_OWNERSHIP_SCREEN_PLAN.json"): raise ValueError("Screen plan changed")
    if fm["plan_sha256"] != digest(ROOT/"experiments/JERSEY_COLOR_PLAN.json") or cm["plan_sha256"] != sm["plan_sha256"]:
        raise ValueError("Feature/feasibility plan changed")
    if fm["selected_model"] != plan["feasibility_model"] or digest(root/"feasibility/manifest.json") != plan["feasibility_manifest_sha256"]:
        raise ValueError("Frozen color model differs")
    reference_manifest = json.loads((ROOT/"data/candidate_broadcast_closure/manifest.json").read_text())
    for group in ("source_sha256","reference_source_sha256"):
        for name,sha in reference_manifest[group].items():
            path = ROOT/name
            if not path.is_file(): path = ROOT/"experiments"/name
            if digest(path) != sha: raise ValueError("Retained model source changed")
    frozen = ROOT/"data/continuation_20261021/annotation_freeze/windows.json"
    if digest(frozen) != fm["annotation_metadata_sha256"]: raise ValueError("Reserved metadata changed")
    weak = np.load(root/"feasibility/weak_color_cache.npz")
    train,validation = weak_frame_split(199319,half_video_blocks(GAME),json.loads(frozen.read_text())["windows"])
    np.testing.assert_array_equal(weak["frames"],np.r_[train,validation])
    if not weak["done"].all(): raise ValueError("Weak extraction incomplete")
    target = np.broadcast_to(weak["teams"],weak["weak_mask"].shape)
    color_scores = pd.read_csv(root/"feasibility/scores.csv")
    for row in color_scores.itertuples():
        profile = int(row.model.split("_")[0].removeprefix("profile"))
        use = weak["weak_mask"] & np.isfinite(weak["features"][profile]).all(axis=2)
        use[:len(train)] = False
        model = joblib.load(root/"feasibility"/(row.model+".joblib"))
        p = model.predict_proba(weak["features"][profile][use]); y = target[use]
        confident = p.max(axis=1) >= .8
        if abs(balanced_accuracy_score(y,p.argmax(axis=1))-row.balanced_accuracy) > 1e-12: raise AssertionError("Color score differs")
        if abs((p.argmax(axis=1)[confident] == y[confident]).mean()-row.confident_accuracy) > 1e-12: raise AssertionError("Confident score differs")
    scores = pd.read_csv(folder/"scores.csv"); official = official_scores(folder,scores)
    source = ROOT/"data/advanced_validation"
    raw,base,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","base","ordered")]
    features = pd.read_pickle(root/"inputs/jersey_inputs.pkl").iloc[:len(raw)]
    views = dict(jersey_fast=base[fast_columns(base)].join(features),jersey_joint_frame=ordered.join(features))
    public = np.arange(11185,18685); splits = json.loads((folder/"splits.json").read_text())
    reloads,block_deltas,diagnostics = 0,[],[]
    details = pd.read_csv(folder/"fold_scores.csv")
    segments = pd.read_csv(folder/"segment_diagnostics.csv")
    for protocol in plan["protocols"]:
        folds = [s for s in splits if s["protocol"] == protocol]
        table = pd.read_csv(folder/protocol/"oof_predictions.csv")
        reference = table.current_seed42.to_numpy(); states = codes(reference)
        np.testing.assert_array_equal(reference,pd.read_csv(root/"baseline_confirmation"/protocol/"oof_predictions.csv").broadcast_closure_seed42)
        for head in plan["heads"]:
            joint = head == "jersey_joint_frame"; probability = np.full((len(raw),3 if joint else 2),np.nan)
            for fold in folds:
                val,train = np.array(fold["validation"]),np.array(fold["train"])
                if np.any((train >= val.min()-250)&(train <= val.max()+250)): raise AssertionError("Fold target buffer violated")
                model = XGBClassifier();model.load_model(folder/protocol/f"{head}_seed42_block{fold['block']}.ubj")
                probability[val] = predict_models([model],views[head].iloc[val]) if joint else predict_control(model,views[head].iloc[val])
                reloads += 1
            if not np.array_equal(probability[public],np.load(folder/protocol/f"{head}_seed42_probabilities.npy")):
                raise AssertionError("Ownership checkpoint reload differs")
            conditional = probability[:,:2]/probability[:,:2].sum(axis=1,keepdims=True)
            for mode in plan["decoders"]:
                name = f"{head}_{mode}_seed42"; prediction = reference.copy()
                for fold in folds:
                    val = np.array(fold["validation"]); ix = val-public[0]
                    prediction[ix] = CLASSES[decode_jersey_ownership(states[ix],conditional[val],features.iloc[val],mode)]
                np.testing.assert_array_equal(prediction,table[name])
                np.testing.assert_array_equal(prediction == "D",reference == "D")
                missing = features.jc_ball_seen.iloc[public].to_numpy() == 0
                np.testing.assert_array_equal(prediction[missing],reference[missing])
                old = details[(details.protocol == protocol)&(details.model == "current_seed42")]
                new = details[(details.protocol == protocol)&(details.model == name)]
                pair = old.merge(new,on=["protocol","minute"],suffixes=["_before","_after"],validate="one_to_one")
                for row in pair.itertuples(): block_deltas.append(dict(protocol=protocol,model=name,block=row.minute,
                    delta=row.spellF1_after-row.spellF1_before,before=row.spellF1_before,after=row.spellF1_after))
        if protocol == "buffered_minutes":
            labels = table.ground_truth.to_numpy()
            short = segments[(segments.protocol == protocol)&(segments.model == "current_seed42")&(segments.label != "D")&(segments.frames <= 125)]
            cases = [("all_live",np.flatnonzero(np.isin(labels,["H","A"])))]
            cases += [(f"short_{int(s.frames)}",np.arange(int(s.start)-public[0],int(s.end)-public[0]+1)) for s in short.itertuples()]
            original_distance = raw[[p+"distance_1" for p in ("h__","a__")]].iloc[public].to_numpy()
            color_distance = features[[p+"jc_distance_1" for p in ("h__","a__")]].iloc[public].to_numpy()
            a,b = CLASSES[np.argmin(original_distance,axis=1)],CLASSES[np.argmin(color_distance,axis=1)]
            for case,ix in cases:
                good = np.isfinite(color_distance[ix]).all(axis=1)&np.isfinite(original_distance[ix]).all(axis=1)
                chosen = ix[good]
                diagnostics.append(dict(case=case,coverage=float(good.mean()),original_correct=float((a[chosen] == labels[chosen]).mean()) if len(chosen) else None,
                    color_correct=float((b[chosen] == labels[chosen]).mean()) if len(chosen) else None,
                    corrected_wrong_original=int(((a[chosen] != labels[chosen])&(b[chosen] == labels[chosen])).sum()),
                    spoiled_correct_original=int(((a[chosen] == labels[chosen])&(b[chosen] != labels[chosen])).sum())))
    pd.DataFrame(block_deltas).to_csv(folder/"block_deltas.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(folder/"nearest_team_diagnostics.csv",index=False)
    gate_rows = []
    index = scores.set_index(["protocol","model"]); count = official.set_index(["protocol","model"])
    for head in plan["heads"]:
        for mode in plan["decoders"]:
            name = f"{head}_{mode}_seed42"; checks = []
            for protocol in plan["protocols"]:
                old,new = index.loc[(protocol,"current_seed42")],index.loc[(protocol,name)]
                a,b = count.loc[(protocol,"current_seed42")],count.loc[(protocol,name)]
                flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),macro=bool(new.macroF1 >= old.macroF1-.001),
                    short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                    spurious=bool(b.spurious_iou50 <= a.spurious_iou50),missed=bool(b.missed_iou50 <= a.missed_iou50))
                if protocol == "buffered_minutes":
                    flags["substantial"] = bool(new.spellF1 >= old.spellF1+.04-1e-12)
                    flags["extra_short_match"] = bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1)
                checks.append(dict(protocol=protocol,primary_delta=float(new.spellF1-old.spellF1),checks=flags))
            gate_rows.append(dict(model=name,passes_screen=all(all(x["checks"].values()) for x in checks),pairs=checks))
    (folder/"screen_checks.json").write_text(json.dumps(dict(eligible_finalists=[r["model"] for r in gate_rows if r["passes_screen"]],methods=gate_rows),indent=2)+"\n")
    after = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    if args.audit_history:
        with tarfile.open(root/"experiments_before.tar.gz") as archive:
            before = pd.read_csv(io.BytesIO(archive.extractfile("experiments/EXPERIMENT_LEDGER.csv").read()))
            old_entry = archive.extractfile("experiments/SELECTED_MODEL.md").read()
        keys = ["round","protocol","model","source"]
        joined = after.merge(before[keys],on=keys,how="inner",validate="one_to_one")
        pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True),joined[before.columns].sort_values(keys).reset_index(drop=True))
        for name,sha in json.loads((root/"baseline_manifest.json").read_text()).items():
            if name == "experiments/EXPERIMENT_LEDGER.csv": continue
            if name == "experiments/SELECTED_MODEL.md" and digest(ROOT/name) != sha:
                if hashlib.sha256(old_entry).hexdigest() != sha: raise AssertionError("Original selected entry lost")
                if json.loads((root/"baseline_candidate/manifest.json").read_text())["csv_sha256"] not in (ROOT/name).read_text():
                    raise AssertionError("Selected artifact changed")
            elif digest(ROOT/name) != sha: raise AssertionError("Protected baseline changed")
    audit = dict(official_score_rows=len(scores),official_score_agreement=True,ownership_models_reload_exactly=reloads,
        color_models_rescored=6,shared_inference_replay=True,reference_D_identical=True,missing_source_reference_unchanged=True,
        color_and_possession_training_target_buffers_verified=True,source_input_cache_hashes_match=True,
        actual_padded_prefix_parity=cm["actual_padded_prefix_parity"],stream_seek_pixel_parity=cm["stream_seek_pixel_parity"],
        historical_snapshot_audited=args.audit_history,old_ledger_rows=len(before) if args.audit_history else None,
        old_ledger_rows_preserved=True if args.audit_history else None,ledger_rows=len(after),
        selected_artifact_unchanged=True if args.audit_history else None,
        clipped_block_comparisons=len(block_deltas),clipped_block_primary_regressions=sum(x["delta"] < -1e-12 for x in block_deltas),
        eligible_finalists=[r["model"] for r in gate_rows if r["passes_screen"]])
    (root/"integrity.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2),flush=True)
    print(pd.DataFrame(diagnostics).round(6).to_string(index=False),flush=True)


if __name__ == "__main__": main()
