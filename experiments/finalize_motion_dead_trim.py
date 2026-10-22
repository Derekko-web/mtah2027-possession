"""Official nine-pair D-only gates and exact source/model/inference replay."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_refinement import decode_broadcast_closure
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest,official_scores
from motion_dead_trim import decode_motion_dead_trim
from research_pair_motion import reference_arrays
from temporal_candidate import predict_models


def promotion_checks(scores,segments,official,tables,plan):
    expected = len(plan["protocols"])*len(plan["seeds"])*2
    index = scores.set_index(["protocol","model"]);count = official.set_index(["protocol","model"])
    if len(scores) != expected or len(official) != expected or index.index.has_duplicates or count.index.has_duplicates:
        raise ValueError("Incomplete or duplicated dead-trim confirmation")
    pairs = []
    for protocol in plan["protocols"]:
        for seed in plan["seeds"]:
            old_name,new_name = f"current_seed{seed}",f"motion_dead_trim_seed{seed}"
            old,new = index.loc[(protocol,old_name)],index.loc[(protocol,new_name)]
            a,b = count.loc[(protocol,old_name)],count.loc[(protocol,new_name)]
            table = tables[protocol];reference,changed = table[old_name].to_numpy(),table[new_name].to_numpy()
            short = segments[(segments.protocol == protocol)&(segments.label != "D")&(segments.frames <= 125)]
            before = short[short.model == old_name].set_index(["start","end","label"])
            after = short[short.model == new_name].set_index(["start","end","label"])
            pd.testing.assert_index_equal(before.index,after.index)
            flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),macro=bool(new.macroF1 >= old.macroF1-1e-12),
                short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                previous_short=bool((after.recall[before.recall >= .5] >= .5).all()),falseD=bool(new.false_dead_frames <= old.false_dead_frames),
                Drecall=bool(new.dead_recall >= old.dead_recall-1e-12),missed_stoppages=bool(new.missed_stoppages <= old.missed_stoppages),
                spurious=bool(b.spurious_iou50 <= a.spurious_iou50),missed=bool(b.missed_iou50 <= a.missed_iou50),
                reference_nonD_exact=bool(np.array_equal(changed[reference != "D"],reference[reference != "D"])),
                no_new_D=bool(not ((changed == "D")&(reference != "D")).any()),
                reference_true_D_retained=bool((changed[(reference == "D")&(table.ground_truth.to_numpy() == "D")] == "D").all()))
            if protocol != "buffered_100_seconds": flags["strict_falseD_reduction"] = bool(new.false_dead_frames < old.false_dead_frames)
            pairs.append(dict(protocol=protocol,seed=seed,checks=flags,primary_delta=float(new.spellF1-old.spellF1),
                macro_delta=float(new.macroF1-old.macroF1),falseD_before=int(old.false_dead_frames),falseD_after=int(new.false_dead_frames),
                short_iou_delta=float(new.short_live_mean_best_iou-old.short_live_mean_best_iou),
                changed_frames=int((changed != reference).sum())))
    substantial = all(p["primary_delta"] >= .04-1e-12 for p in pairs if p["protocol"] == "buffered_minutes")
    extra_short = all(index.loc[("buffered_minutes",f"motion_dead_trim_seed{s}"),"short_live_matched_iou50"] >=
        index.loc[("buffered_minutes",f"current_seed{s}"),"short_live_matched_iou50"]+1 for s in plan["seeds"])
    return dict(passed=all(all(p["checks"].values()) for p in pairs),pairs=pairs,
        original_substantial_primary_criterion_met=bool(substantial),original_extra_buffered_short_match_criterion_met=bool(extra_short),
        objective="Separate prospectively frozen incremental falseD reduction, not original substantial/extra-short gates")


def main():
    root = ROOT/"data/control_pair_motion_v1";folder = root/"dead_trim_confirmation"
    manifest = audit_manifest(folder);audit_manifest(root/"inputs")
    plan_path = ROOT/"experiments/MOTION_DEAD_TRIM_PLAN.json";plan = json.loads(plan_path.read_text())
    if manifest["plan_sha256"] != digest(plan_path) or (folder/"plan.json").read_bytes() != plan_path.read_bytes(): raise ValueError("Dead-trim plan differs")
    source = ROOT/"data/advanced_validation";raw,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","ordered")]
    inputs = ordered.join(pd.read_pickle(root/"inputs/broadcast_motion_inputs.pkl").iloc[:len(raw)])
    if list(inputs) != manifest["feature_columns"]: raise ValueError("Confirmed feature schema differs")
    info = pd.read_pickle(root/"inputs/motion_info.pkl").iloc[:len(raw)]
    bc = pd.read_pickle(ROOT/"data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:len(raw)]
    splits = json.loads((folder/"splits.json").read_text());public = np.arange(11185,18685)
    scores = pd.read_csv(folder/"scores.csv");segments = pd.read_csv(folder/"segment_diagnostics.csv")
    tables = {};reloads,block_deltas = 0,[]
    details = pd.read_csv(folder/"fold_scores.csv")
    for protocol in plan["protocols"]:
        table = pd.read_csv(folder/protocol/"oof_predictions.csv");tables[protocol] = table
        folds = [s for s in splits if s["protocol"] == protocol]
        blocks = [(min(s["validation"]),max(s["validation"])+1) for s in folds]
        for seed in plan["seeds"]:
            probability = np.full((len(raw),3),np.nan)
            for fold in folds:
                val,train = np.array(fold["validation"]),np.array(fold["train"])
                if np.any((train >= val.min()-250)&(train <= val.max()+250)): raise AssertionError("Target exclusion buffer violated")
                model = XGBClassifier();model.load_model(folder/protocol/f"broadcast_motion_seed{seed}_block{fold['block']}.ubj")
                probability[val] = predict_models([model],inputs.iloc[val]);reloads += 1
            if not np.array_equal(probability[public],np.load(folder/protocol/f"broadcast_motion_seed{seed}_probabilities.npy")):
                raise AssertionError("Dead-trim model reload differs")
            if seed == 42: np.testing.assert_array_equal(probability[public],np.load(root/"screen"/protocol/"broadcast_motion_seed42_probabilities.npy"))
            arrays,_ = reference_arrays(protocol,seed,len(raw))
            reference = decode_broadcast_closure(*arrays,bc,raw,blocks)[public]
            prediction = decode_motion_dead_trim(*arrays,probability,bc,info,raw,blocks)[public]
            np.testing.assert_array_equal(reference,table[f"current_seed{seed}"])
            np.testing.assert_array_equal(prediction,table[f"motion_dead_trim_seed{seed}"])
            old = details[(details.protocol == protocol)&(details.model == f"current_seed{seed}")]
            new = details[(details.protocol == protocol)&(details.model == f"motion_dead_trim_seed{seed}")]
            pair = old.merge(new,on=["protocol","minute"],suffixes=["_before","_after"],validate="one_to_one")
            for row in pair.itertuples(): block_deltas.append(dict(protocol=protocol,seed=seed,block=row.minute,delta=row.spellF1_after-row.spellF1_before))
    official = official_scores(folder,scores,1000)
    checks = promotion_checks(scores,segments,official,tables,plan)
    checks.update(official_score_agreement=True,shared_inference_parity=True,models_reload_exactly=reloads,seed42_raw_probability_parity=True)
    (folder/"promotion_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    pd.DataFrame(block_deltas).to_csv(folder/"block_deltas.csv",index=False)
    result = dict(confirmation_rows=len(scores),models_reload_exactly=reloads,shared_inference_parity=True,official_score_agreement=True,
        reference_nonD_preserved=True,training_target_buffers_verified=True,seed42_probability_screen_parity=True,
        clipped_block_comparisons=len(block_deltas),clipped_block_primary_regressions=sum(r["delta"] < -1e-12 for r in block_deltas),
        incremental_objective_passed=checks["passed"],original_substantial_criterion_met=checks["original_substantial_primary_criterion_met"],
        original_extra_buffered_short_match_criterion_met=checks["original_extra_buffered_short_match_criterion_met"])
    (root/"dead_trim_integrity.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
