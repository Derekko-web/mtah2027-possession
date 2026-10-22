"""Official replay of original and symmetric motion screens, with optional history."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_refinement import decode_broadcast_closure
from build_broadcast_candidate import check_sources
from build_temporal_candidate import digest
from finalize_broadcast_closure import audit_manifest,official_scores
from pair_candidate import codes,decode_pair_candidate
from pair_motion_inference import motion_source
from pair_motion_symmetric import symmetric_motion_events
from research_pair_motion import reference_arrays
from stoppage_consensus import decode_stoppage_consensus
from temporal_candidate import predict_models
from xgboost_baseline import CLASSES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-history",action="store_true")
    args = parser.parse_args()
    root = ROOT/"data/control_pair_motion_v1"
    cm = audit_manifest(root/"inputs")
    audit_manifest(ROOT/"data/control_broadcast_v1/inputs")
    for stage in ("screen","symmetric_screen"): audit_manifest(root/stage)
    plan_path = ROOT/"experiments/PAIR_MOTION_PLAN.json";plan = json.loads(plan_path.read_text())
    if cm["plan_sha256"] != digest(plan_path): raise ValueError("Motion feature plan changed")
    for stage,name in (("screen","PAIR_MOTION_PLAN.json"),("symmetric_screen","PAIR_EVENT_SYMMETRY_PLAN.json")):
        if (root/stage/"plan.json").read_bytes() != (ROOT/"experiments"/name).read_bytes(): raise ValueError("Frozen screen plan differs")
    reference_manifest = json.loads((ROOT/"data/candidate_broadcast_closure/manifest.json").read_text())
    check_sources(reference_manifest["source_sha256"]);check_sources(reference_manifest["reference_source_sha256"])
    source = ROOT/"data/advanced_validation";raw,ordered = [pd.read_pickle(source/(name+".pkl")) for name in ("raw","ordered")]
    info = pd.read_pickle(root/"inputs/motion_info.pkl").iloc[:len(raw)]
    bc = pd.read_pickle(ROOT/"data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:len(raw)]
    views = {head:ordered.join(pd.read_pickle(root/"inputs"/(head+"_inputs.pkl")).iloc[:len(raw)]) for head in plan["heads"]}
    public = np.arange(11185,18685);splits = json.loads((root/"screen/splits.json").read_text())
    reloads,delta_rows,gate_rows,diagnostics = 0,[],[],[]
    event_parity = True
    for stage in ("screen","symmetric_screen"):
        folder = root/stage;scores = pd.read_csv(folder/"scores.csv");official = official_scores(folder,scores)
        expected = 39 if stage == "screen" else 15
        if len(scores) != expected or scores.duplicated(["protocol","model"]).any(): raise ValueError("Incomplete screen")
        score_index = scores.set_index(["protocol","model"]);counts = official.set_index(["protocol","model"])
        segments = pd.read_csv(folder/"segment_diagnostics.csv");details = pd.read_csv(folder/"fold_scores.csv")
        for protocol in plan["screen"]["protocols"]:
            folds = [s for s in splits if s["protocol"] == protocol]
            blocks = [(min(s["validation"]),max(s["validation"])+1) for s in folds]
            arrays,_ = reference_arrays(protocol,42,len(raw))
            reference = decode_broadcast_closure(*arrays,bc,raw,blocks)[public]
            base = decode_stoppage_consensus(*arrays[:4],raw,blocks)[public]
            table = pd.read_csv(folder/protocol/"oof_predictions.csv")
            np.testing.assert_array_equal(reference,table.current_seed42)
            for head in plan["heads"]:
                probability = np.full((len(raw),3),np.nan)
                if stage == "screen":
                    for fold in folds:
                        val,train = np.array(fold["validation"]),np.array(fold["train"])
                        if np.any((train >= val.min()-250)&(train <= val.max()+250)): raise AssertionError("Target exclusion buffer violated")
                        model = XGBClassifier();model.load_model(folder/protocol/f"{head}_seed42_block{fold['block']}.ubj")
                        probability[val] = predict_models([model],views[head].iloc[val]);reloads += 1
                    if not np.array_equal(probability[public],np.load(folder/protocol/f"{head}_seed42_probabilities.npy")):
                        raise AssertionError("Checkpoint reload differs")
                else: probability[public] = np.load(root/"screen"/protocol/f"{head}_seed42_probabilities.npy")
                conditional = probability[public,:2]/probability[public,:2].sum(axis=1,keepdims=True)
                modes = plan["screen"]["decoders"] if stage == "screen" else ["symmetric_events"]
                for mode in modes:
                    name = f"{head}_{mode}_seed42"
                    if stage == "screen": prediction = decode_pair_candidate(*arrays,probability,bc,info,raw,blocks,head,mode)[public]
                    else:
                        prediction = reference.copy();kernel,distance,observed = motion_source(info.iloc[public],head)
                        for fold in folds:
                            ix = np.array(fold["validation"])-public[0]
                            result = symmetric_motion_events(codes(reference[ix]),codes(base[ix]),conditional[ix],kernel[ix],distance[ix],observed[ix])
                            swapped = symmetric_motion_events(codes(reference[ix])^np.where(reference[ix] == "D",0,1),
                                codes(base[ix])^np.where(base[ix] == "D",0,1),conditional[ix,::-1],kernel[ix,::-1],distance[ix,::-1],observed[ix])
                            desired = np.where(result == 2,2,1-result)
                            np.testing.assert_array_equal(swapped,desired)
                            prediction[ix] = CLASSES[result]
                        prior = pd.read_csv(root/"screen"/protocol/"oof_predictions.csv")[f"{head}_motion_events_seed42"]
                        event_parity &= np.array_equal(prediction,prior)
                    np.testing.assert_array_equal(prediction,table[name])
                    if mode != "replace_current": np.testing.assert_array_equal(prediction == "D",reference == "D")
                    old,new = score_index.loc[(protocol,"current_seed42")],score_index.loc[(protocol,name)]
                    a,b = counts.loc[(protocol,"current_seed42")],counts.loc[(protocol,name)]
                    short = segments[(segments.protocol == protocol)&(segments.label != "D")&(segments.frames <= 125)]
                    before = short[short.model == "current_seed42"].set_index(["start","end","label"])
                    after = short[short.model == name].set_index(["start","end","label"])
                    pd.testing.assert_index_equal(before.index,after.index)
                    flags = dict(primary=bool(new.spellF1 >= old.spellF1-1e-12),macro=bool(new.macroF1 >= old.macroF1-.001),
                        short_iou=bool(new.short_live_mean_best_iou >= old.short_live_mean_best_iou-1e-12),
                        prior_short=bool((after.recall[before.recall >= .5] >= .5).all()),falseD=bool(new.false_dead_frames <= old.false_dead_frames),
                        dead_recall=bool(new.dead_recall >= old.dead_recall-.02),stoppages=bool(new.missed_stoppages <= old.missed_stoppages),
                        spurious=bool(b.spurious_iou50 <= a.spurious_iou50),missed=bool(b.missed_iou50 <= a.missed_iou50),
                        implementation_eligible=not(stage == "screen" and mode == "motion_events"))
                    if protocol == "buffered_minutes":
                        flags["substantial"] = bool(new.spellF1 >= old.spellF1+.04-1e-12)
                        flags["extra_short_match"] = bool(new.short_live_matched_iou50 >= old.short_live_matched_iou50+1)
                    gate_rows.append(dict(stage=stage,protocol=protocol,model=name,checks=flags,primary_delta=float(new.spellF1-old.spellF1)))
                    before = details[(details.protocol == protocol)&(details.model == "current_seed42")]
                    after = details[(details.protocol == protocol)&(details.model == name)]
                    pair = before.merge(after,on=["protocol","minute"],suffixes=["_before","_after"],validate="one_to_one")
                    for row in pair.itertuples(): delta_rows.append(dict(stage=stage,protocol=protocol,model=name,block=row.minute,delta=row.spellF1_after-row.spellF1_before))
    pd.DataFrame(delta_rows).to_csv(root/"block_deltas.csv",index=False)
    eligible = []
    for stage in ("screen","symmetric_screen"):
        for name in sorted(set(r["model"] for r in gate_rows if r["stage"] == stage)):
            pairs = [r for r in gate_rows if r["stage"] == stage and r["model"] == name]
            if len(pairs) == 3 and all(all(r["checks"].values()) for r in pairs): eligible.append(dict(stage=stage,model=name))
    (root/"screen_checks.json").write_text(json.dumps(dict(eligible_finalists=eligible,pairs=gate_rows),indent=2)+"\n")
    ledger = pd.read_csv(ROOT/"experiments/EXPERIMENT_LEDGER.csv")
    if args.audit_history:
        with tarfile.open(root/"experiments_before.tar.gz") as archive:
            before = pd.read_csv(io.BytesIO(archive.extractfile("experiments/EXPERIMENT_LEDGER.csv").read()))
            entry = archive.extractfile("experiments/SELECTED_MODEL.md").read()
        keys = ["round","protocol","model","source"]
        joined = ledger.merge(before[keys],on=keys,how="inner",validate="one_to_one")
        pd.testing.assert_frame_equal(before.sort_values(keys).reset_index(drop=True),joined[before.columns].sort_values(keys).reset_index(drop=True))
        for name,sha in json.loads((root/"baseline_manifest.json").read_text()).items():
            if name == "experiments/EXPERIMENT_LEDGER.csv": continue
            if name == "experiments/SELECTED_MODEL.md" and digest(ROOT/name) != sha:
                if hashlib.sha256(entry).hexdigest() != sha or reference_manifest["csv_sha256"] not in (ROOT/name).read_text(): raise AssertionError("Selected entry lost")
            elif digest(ROOT/name) != sha: raise AssertionError("Protected artifact changed")
    frozen = ROOT/"data/continuation_20261021/annotation_freeze/windows.json"
    if digest(frozen) != "5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c": raise AssertionError("Reserved windows changed")
    result = dict(official_score_rows=54,official_score_agreement=True,shared_inference_replay=True,models_reload_exactly=reloads,
        unchanged_seven_model_reference_verified=True,training_target_buffers_verified=True,actual_padded_prefix_parity=cm["actual_padded_prefix_parity"],
        input_cache_source_hashes_verified=True,ownership_event_D_masks_identical=True,real_data_event_swap_invariance=True,
        original_and_symmetric_event_predictions_identical=bool(event_parity),historical_snapshot_audited=args.audit_history,
        old_ledger_rows=len(before) if args.audit_history else None,old_ledger_rows_preserved=True if args.audit_history else None,
        ledger_rows=len(ledger),selected_artifact_unchanged=True if args.audit_history else None,
        clipped_block_comparisons=len(delta_rows),clipped_block_primary_regressions=sum(r["delta"] < -1e-12 for r in delta_rows),eligible_finalists=eligible)
    (root/"integrity.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
