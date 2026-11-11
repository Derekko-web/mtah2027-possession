"""Independent official/role/rule/parity audit of the two fixed candidates."""
import csv
import io
import json
from pathlib import Path

from analyze_tabpfn_blocks import ROOT, SOURCE, PROTOCOLS, load_inputs, replay, digest, write_json

OUT=ROOT/"data/control_tabpfn_block_fix_v2"
AUDIT=ROOT/"data/control_tabpfn_block_diagnosis_v1/correction_audit.json"


def require(value,message):
    if not value:raise ValueError(message)


def run(append=False):
    import numpy as np
    import pandas as pd
    from evaluation import possession_metrics as pm
    from tabpfn_short_control_inference import decode_tabpfn_short_control
    manifest=json.loads((OUT/"manifest.json").read_text())
    for group,base in (("source_sha256",ROOT),("cache_sha256",OUT)):
        for name,sha in manifest[group].items():require(digest(base/name)==sha,"Bound correction artifact changed: "+name)
    require(set(manifest["cache_sha256"])=={str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and p.name!="manifest.json"},"Incomplete correction artifact closure")
    # The repair changed only scorer-file length/output path, never predictions.
    failed=ROOT/"data/control_tabpfn_block_fix_v1"
    for name in ("scores.csv","fold_scores.csv","segment_diagnostics.csv"):
        require((OUT/name).read_bytes()==(failed/name).read_bytes(),"Export-only repair changed scientific results")
    scores=pd.read_csv(OUT/"scores.csv");folds=pd.read_csv(OUT/"fold_scores.csv")
    shared=load_inputs();parity=[];official=[]
    for protocol in PROTOCOLS:
        arrays,owner,old,split_records,blocks=replay(protocol,shared)
        raw,bc,motion,supported,_=shared
        actual,events=decode_tabpfn_short_control(arrays,owner,supported,bc,motion,raw,blocks)
        table=pd.read_csv(OUT/protocol/"oof_predictions.csv");public=table.frame.to_numpy()
        np.testing.assert_array_equal(table.frame,old.frame);np.testing.assert_array_equal(table.ground_truth,old.ground_truth)
        np.testing.assert_array_equal(actual[public],table.tabpfn_short_control_seed42)
        np.testing.assert_array_equal(table.tabpfn_context_seed42,old.tabpfn_context_seed42)
        np.testing.assert_array_equal(table.current_seed42,old.current_seed42)
        parity.append(dict(protocol=protocol,shared_inference_exact=True,restored_events=events,
                           changed_frames=int((table.tabpfn_short_control_seed42!=old.tabpfn_context_seed42).sum())))
        if protocol!="buffered_100_seconds":np.testing.assert_array_equal(actual[public],old.tabpfn_context_seed42)
        for model in ("current_seed42","tabpfn_context_seed42","tabpfn_halfblend_seed42","tabpfn_short_control_seed42"):
            prediction=table[model].to_numpy();truth=table.ground_truth.to_numpy()
            for label in ("D","X"):np.testing.assert_array_equal(prediction==label,table.current_seed42.to_numpy()==label)
            result=pm.evaluate(pm.label_codes(truth),pm.label_codes(prediction),bootstrap_n=0)
            saved=json.loads((OUT/protocol/"official/reports"/(model+".json")).read_text())
            row=scores[(scores.protocol==protocol)&(scores.model==model)].iloc[0]
            require(abs(row.spellF1-result["spell_f1_mean"])<1e-12 and abs(row.macroF1-result["macro_f1"])<1e-12,"Shared headline metric mismatch")
            require(abs(saved["spell_f1_mean"]-result["spell_f1_mean"])<1e-12 and abs(saved["macro_f1"]-result["macro_f1"])<1e-12,"Official headline metric mismatch")
            for split in split_records:
                ids=np.asarray(split["validation"])-11185
                local=pm.evaluate(pm.label_codes(truth[ids]),pm.label_codes(prediction[ids]),bootstrap_n=0)
                value=folds[(folds.protocol==protocol)&(folds.model==model)&(folds.minute==split["block"])].iloc[0]
                require(abs(value.spellF1-local["spell_f1_mean"])<1e-12 and abs(value.macroF1-local["macro_f1"])<1e-12,"Per-fold metric mismatch")
            official.append(dict(protocol=protocol,model=model,spellF1=float(row.spellF1),macroF1=float(row.macroF1)))
    require(len(scores)==12 and len(folds)==52,"Incomplete four-method13fold comparison")
    result=dict(passed=True,official_metric_rows=12,per_fold_rows=52,original_context_models_reused=13,
        classifier_refits_for_correction=0,shared_inference_parity=parity,
        main_and_purged_labels_byte_exact_to_original_TabPFN=True,retained_D_X_exact=True,
        export_repair_scientific_results_byte_exact=True,metrics=official,
        main_score_preserved=True,longer_score_above_current=True,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/"experiments/tabpfn_short_control_inference.py")},
        original_selected_model_changed=False,adaptive_public_development_evidence=True)
    if append:
        ledger=ROOT/"experiments/EXPERIMENT_LEDGER.csv";before=ledger.read_bytes();existing=list(csv.DictReader(io.StringIO(before.decode())))
        require(not any(r["round"]=="round77_tabpfn_short_control" for r in existing),"Correction ledger rows already exist")
        columns=list(existing[0]);buffer=io.StringIO(newline="");writer=csv.DictWriter(buffer,fieldnames=columns,lineterminator="\n")
        for score in scores.to_dict("records"):
            row={k:score[k] for k in columns if k in score}
            row.update(round="round77_tabpfn_short_control",record_type="adaptive_fixed_TabPFN_decoder_correction",
                stitched_spellF1=score["spellF1"],source=str((OUT/"scores.csv").relative_to(ROOT)),source_sha256=digest(OUT/"scores.csv"))
            detail=folds[(folds.protocol==score["protocol"])&(folds.model==score["model"])]
            row.update({f"block_{int(r['minute'])}":r["spellF1"] for r in detail.to_dict("records")});writer.writerow(row)
        with ledger.open("ab") as handle:handle.write(buffer.getvalue().encode())
        require(ledger.read_bytes().startswith(before),"Historical ledger bytes changed")
        result.update(ledger_rows_before=len(existing),ledger_rows_after=len(existing)+12,prior_ledger_bytes_preserved=True)
    with AUDIT.open("x") as handle:handle.write(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("metrics","shared_inference_parity","source_sha256")},sort_keys=True))


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--append-ledger",action="store_true");run(p.parse_args().append_ledger)
