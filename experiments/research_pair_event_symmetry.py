"""Separate frozen symmetric-event comparison reusing all original OOF models."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_refinement import decode_broadcast_closure
from build_temporal_candidate import digest
from pair_candidate import codes
from pair_motion_inference import motion_source
from pair_motion_symmetric import symmetric_motion_events
from research_boundary_refinement import add_overlap_metrics
from research_control_refinement import add_short_metrics
from research_pair_motion import reference_arrays
from research_stoppages import record_predictions
from stoppage_consensus import decode_stoppage_consensus
from xgboost_baseline import CLASSES


def main():
    root = ROOT/"data/control_pair_motion_v1";output = root/"symmetric_screen";output.mkdir(exist_ok=False)
    original = root/"screen";plan_path = ROOT/"experiments/PAIR_EVENT_SYMMETRY_PLAN.json";plan = json.loads(plan_path.read_text())
    (output/"plan.json").write_bytes(plan_path.read_bytes())
    raw = pd.read_pickle(ROOT/"data/advanced_validation/raw.pkl")
    info = pd.read_pickle(root/"inputs/motion_info.pkl").iloc[:len(raw)]
    bc = pd.read_pickle(ROOT/"data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:len(raw)]
    public = np.arange(11185,18685);splits_path = original/"splits.json";splits = json.loads(splits_path.read_text())
    (output/"splits.json").write_bytes(splits_path.read_bytes())
    hashes = {str(p.relative_to(ROOT)):digest(p) for p in (original/"manifest.json",root/"inputs/manifest.json",splits_path)}
    rows,details,segments,events = [],[],[],[]
    for protocol in plan["protocols"]:
        folder = output/protocol;folder.mkdir()
        folds = [(s["block"],np.array(s["train"]),np.array(s["validation"])) for s in splits if s["protocol"] == protocol]
        blocks = [(int(val[0]),int(val[-1])+1) for _,_,val in folds]
        arrays,source_hashes = reference_arrays(protocol,42,len(raw));hashes.update(source_hashes)
        reference = decode_broadcast_closure(*arrays,bc,raw,blocks)[public]
        base = decode_stoppage_consensus(*arrays[:4],raw,blocks)[public]
        old_path = original/protocol/"oof_predictions.csv";hashes[str(old_path.relative_to(ROOT))] = digest(old_path)
        old = pd.read_csv(old_path);np.testing.assert_array_equal(reference,old.current_seed42)
        labels = np.full(199319,"X",dtype="<U1");labels[public] = old.ground_truth
        table = pd.DataFrame(dict(frame=public,ground_truth=labels[public],current_seed42=reference))
        record_predictions(output,protocol,"current_seed42",public,labels,folds,reference,rows,details,segments)
        add_overlap_metrics(rows,segments,table,protocol,"current_seed42")
        for head in plan["methods"]:
            path = original/protocol/f"{head}_seed42_probabilities.npy";hashes[str(path.relative_to(ROOT))] = digest(path)
            p = np.load(path)[:,:2];p = p/p.sum(axis=1,keepdims=True)
            kernel,distance,observed = motion_source(info.iloc[public],head)
            prediction = reference.copy()
            for block,_,val in folds:
                ix = val-public[0]
                result,records = symmetric_motion_events(codes(reference[ix]),codes(base[ix]),p[ix],kernel[ix],distance[ix],observed[ix],True)
                prediction[ix] = CLASSES[result]
                events.extend(dict(protocol=protocol,head=head,block=block,**record) for record in records)
            name = f"{head}_symmetric_events_seed42";table[name] = prediction
            record_predictions(output,protocol,name,public,labels,folds,prediction,rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,name)
        add_short_metrics(rows,segments)
        table.to_csv(folder/"oof_predictions.csv",index=False)
        pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
        pd.DataFrame(details).to_csv(output/"fold_scores.csv",index=False)
        pd.DataFrame(segments).to_csv(output/"segment_diagnostics.csv",index=False)
    pd.DataFrame(events).to_csv(output/"events.csv",index=False)
    names = ("research_pair_event_symmetry.py","pair_motion_symmetric.py","pair_motion_inference.py","pair_candidate.py","research_pair_motion.py")
    manifest = dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        input_sha256=hashes,new_models=0,existing_OOF_models_reused=True,reference_D_identical=True)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(pd.DataFrame(rows)[["protocol","model","spellF1","macroF1","short_live_mean_best_iou","short_live_matched_iou50"]].round(6).to_string(index=False),flush=True)


if __name__ == "__main__": main()
