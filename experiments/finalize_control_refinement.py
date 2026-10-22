"""Check paired promotion gates, decoder parity and official scorer results."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation import possession_metrics as pm
from consensus_control import decode_consensus
from confirm_control_refinement import PLAN


def main():
    output=ROOT/"data/control_confirmation";source=ROOT/"data/advanced_validation"
    plan=json.loads(PLAN.read_text())
    if (output/"plan.json").read_bytes()!=PLAN.read_bytes():raise ValueError("Plan changed")
    scores=pd.read_csv(output/"scores.csv");index=scores.set_index(["protocol","model"])
    if len(scores)!=27 or index.index.has_duplicates:raise ValueError("Incomplete confirmation")
    checks={}
    for name in plan["finalists"]:
        paired=[]
        for protocol in plan["protocols"]:
            for seed in plan["seeds"]:
                old=index.loc[(protocol,f"current_seed{seed}")];new=index.loc[(protocol,f"{name}_seed{seed}")]
                flags=dict(spellF1=bool(new.spellF1>=old.spellF1-1e-12),
                           dead_recall=bool(new.dead_recall>=old.dead_recall-.02),
                           missed_stoppages=bool(new.missed_stoppages<=old.missed_stoppages),
                           macroF1=bool(new.macroF1>=old.macroF1-.005))
                if protocol=="buffered_minutes":
                    flags.update(false_dead=bool(new.false_dead_frames<old.false_dead_frames),
                                 short_recall=bool(new.short_live_mean_recall>old.short_live_mean_recall))
                paired.append(dict(protocol=protocol,seed=seed,checks=flags,
                                   spellF1_delta=float(new.spellF1-old.spellF1),
                                   false_dead_delta=int(new.false_dead_frames-old.false_dead_frames)))
        checks[name]=dict(passed=all(all(p["checks"].values()) for p in paired),paired=paired)
    selected=plan["preferred"] if checks[plan["preferred"]]["passed"] else None
    (output/"promotion_checks.json").write_text(json.dumps(dict(selected=selected,methods=checks),indent=2)+"\n")
    raw=pd.read_pickle(output/"raw.pkl");public=np.arange(11185,18685)
    splits=json.loads((source/"splits.json").read_text());reports=[]
    for protocol in plan["protocols"]:
        blocks=[(min(s["validation"]),max(s["validation"])+1) for s in splits if s["protocol"]==protocol]
        table=pd.read_csv(output/protocol/"oof_predictions.csv")
        for seed in plan["seeds"]:
            arrays=[]
            for path,width in ((source/protocol/f"ordered_xgboost_frame_seed{seed}_probabilities.npy",3),
                               (source/protocol/f"previous_ensemble_seed{seed}_probabilities.npy",3),
                               (output/protocol/f"fast_seed{seed}_probabilities.npy",2)):
                full=np.full((len(raw),width),np.nan);full[public]=np.load(path);arrays.append(full)
            for name,config in plan["finalists"].items():
                prediction=decode_consensus(*arrays,raw,blocks,**config)[public]
                np.testing.assert_array_equal(prediction,table[f"{name}_seed{seed}"].to_numpy())
        with (output/protocol/"official.log").open("w") as log:
            subprocess.run([sys.executable,str(ROOT/"evaluation/run_evaluation.py"),"--gt","public",
                            "--results",str(output/protocol/"scorer_inputs"),"--out",str(output/protocol/"official"),
                            "--bootstrap","1000","--no-detail"],check=True,stdout=log,stderr=subprocess.STDOUT)
        for row in scores[scores.protocol==protocol].to_dict("records"):
            report=json.loads((output/protocol/"official/reports"/(row["model"]+".json")).read_text())
            if abs(report["spell_f1_mean"]-row["spellF1"])>1e-12 or abs(report["macro_f1"]-row["macroF1"])>1e-12:
                raise ValueError("Official score mismatch")
            reports.append(dict(protocol=protocol,model=row["model"],bootstrap90_low=report["spell_bootstrap_low"],
                                bootstrap90_high=report["spell_bootstrap_high"],
                                matched_iou50=report["spell_f1"]["0.5"]["n_matched"],
                                spurious_iou50=report["spell_f1"]["0.5"]["n_spurious"],
                                missed_iou50=report["spell_f1"]["0.5"]["n_missed"]))
    scores.merge(pd.DataFrame(reports),on=["protocol","model"],validate="one_to_one").to_csv(
        output/"official_comparison.csv",index=False)
    print("Official metrics and shared-decoder parity verified for all 27 evaluations.")
    print("Selected:",selected)
    if selected is None:raise SystemExit("No candidate passes the frozen promotion checks")


if __name__=="__main__":main()
