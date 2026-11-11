"""Evaluate the two registered uniform corrections using frozen probabilities."""
import json
from pathlib import Path
import sys

from analyze_tabpfn_blocks import ROOT, SOURCE, PROTOCOLS, load_inputs, replay, digest, write_json

OUT = ROOT / "data/control_tabpfn_block_fix_v2"
PLAN = ROOT / "data/control_tabpfn_block_diagnosis_v1/fix_plan.json"


def run():
    import numpy as np
    import pandas as pd
    from control_refinement import contact_support
    from factorized_latent import decode_owner
    from tabpfn_short_control import restore_supported_islands
    from research_stoppages import record_predictions
    from research_boundary_refinement import add_overlap_metrics
    from research_control_refinement import add_short_metrics
    from finalize_broadcast_closure import official_scores
    from evaluation import possession_metrics as pm
    from diagnose_tabpfn_blocks import check
    if OUT.exists():
        raise ValueError("Choose a new registered experiment; preserve completed outcomes")
    OUT.mkdir()
    (OUT/"plan.json").write_bytes(PLAN.read_bytes())
    shared=load_inputs();raw,bc,motion,eligible,_=shared
    support=contact_support(raw)
    rows,details,segments,events=[],[],[],[]
    labels=pm.read_label_csv(ROOT/"evaluation/gt/BAR-ATH_possession_gt_public.csv").label.to_numpy()
    for protocol in PROTOCOLS:
        arrays,owner,old,split_records,blocks=replay(protocol,shared)
        public=old.frame.to_numpy()
        reference=np.full(len(raw),"X",dtype="<U1");reference[public]=old.current_seed42
        original=np.full(len(raw),"X",dtype="<U1");original[public]=old.tabpfn_context_seed42
        # Candidate A: one fixed, uniform probability blend; no parameter sweep.
        blended=owner.copy()
        for start,end in blocks:
            base=arrays[0][start:end,:2]
            total=base.sum(axis=1,keepdims=True)
            conditional=np.divide(base,total,out=np.full_like(base,.5),where=total>0)
            blended[start:end]=.5*owner[start:end]+.5*conditional
        half=decode_owner(arrays,blended,eligible,bc,motion,raw,blocks)
        # Candidate B: same rule for every protocol; sees no truth labels.
        protected,accepted=restore_supported_islands(reference,original,support,blocks)
        events.extend(dict(protocol=protocol,**e) for e in accepted)
        methods=dict(current_seed42=reference,tabpfn_context_seed42=original,
                     tabpfn_halfblend_seed42=half,tabpfn_short_control_seed42=protected)
        table=pd.DataFrame(dict(frame=public,ground_truth=old.ground_truth))
        np.testing.assert_array_equal(labels[public],old.ground_truth)
        folds=[(s["block"],np.asarray(s["train"]),np.asarray(s["validation"])) for s in split_records]
        folder=OUT/protocol;folder.mkdir()
        for name,value in methods.items():
            np.testing.assert_array_equal(value=="D",reference=="D")
            np.testing.assert_array_equal(value=="X",reference=="X")
            table[name]=value[public]
            record_predictions(OUT,protocol,name,public,labels,folds,value[public],rows,details,segments)
            add_overlap_metrics(rows,segments,table,protocol,name)
        table.to_csv(folder/"oof_predictions.csv",index=False)
    add_short_metrics(rows,segments)
    scores=pd.DataFrame(rows)
    scores.to_csv(OUT/"scores.csv",index=False)
    pd.DataFrame(details).to_csv(OUT/"fold_scores.csv",index=False)
    pd.DataFrame(segments).to_csv(OUT/"segment_diagnostics.csv",index=False)
    official=official_scores(OUT,scores)
    write_json(OUT/"restored_events.json",events)
    verdict={name:check(OUT,name) for name in ("tabpfn_halfblend_seed42","tabpfn_short_control_seed42")}
    write_json(OUT/"regression_checks.json",verdict)
    write_json(OUT/"manifest.json",dict(plan_sha256=digest(PLAN),classifier_fits=0,
        frozen_context_models_reused=13,seed=42,uniform_rules_across_protocols=True,
        validation_labels_used_for_inference=False,manual_labels_used=False,retained_D_X_exact=True,
        main_preservation_requirement=.7771428571428571,
        predecessor_manifest_sha256=digest(SOURCE/"manifest.json"),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__),ROOT/"experiments/tabpfn_short_control.py",ROOT/"experiments/analyze_tabpfn_blocks.py",
             ROOT/"experiments/diagnose_tabpfn_blocks.py",ROOT/"experiments/test_tabpfn_short_control.py")},
        cache_sha256={str(p.relative_to(OUT)):digest(p) for p in sorted(OUT.rglob("*")) if p.is_file()}))
    print(scores[["protocol","model","spellF1","macroF1","short_live_matched_iou50"]].to_string(index=False))
    print(json.dumps(verdict,indent=2))


if __name__=="__main__":
    run()
