"""Read-only causal probes of the complete TabPFN public OOF predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/control_tabpfn_extended_v1/screen"
OUT = ROOT / "data/control_tabpfn_block_diagnosis_v1"
PROTOCOLS = ("buffered_minutes", "purged_spells", "buffered_100_seconds")
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_inputs():
    import numpy as np
    import pandas as pd
    from player_representation import interpolate
    raw = pd.read_pickle(ROOT / "data/advanced_validation/raw.pkl")
    enc = ROOT / "data/control_player_representation_v1/encodings"
    _, eligible = interpolate(np.load(enc / "frames.npy"), np.load(enc / "pretrained_latent.npy"),
                              np.load(enc / "eligible.npy"), len(raw))
    bc = pd.read_pickle(ROOT / "data/control_broadcast_v1/inputs/combined_inputs.pkl").iloc[:len(raw)]
    motion = pd.read_pickle(ROOT / "data/control_pair_motion_v1/inputs/motion_info.pkl").iloc[:len(raw)]
    splits = json.loads((ROOT / "data/advanced_validation/splits.json").read_text())
    return raw, bc, motion, eligible, splits


def replay(protocol, shared):
    import numpy as np
    import pandas as pd
    from factorized_latent import decode_owner
    from research_ownership_graph import selected_arrays
    raw, bc, motion, eligible, splits = shared
    folds = [s for s in splits if s["protocol"] == protocol]
    blocks = [(min(s["validation"]), max(s["validation"]) + 1) for s in folds]
    arrays, _ = selected_arrays(protocol, 42, len(raw))
    table = pd.read_csv(SOURCE / protocol / "oof_predictions.csv")
    public = table.frame.to_numpy()
    owner = np.full((len(raw), 2), np.nan)
    owner[public] = np.load(SOURCE / protocol / "owner_probability.npy", allow_pickle=False)
    actual = decode_owner(arrays, owner, eligible, bc, motion, raw, blocks)
    np.testing.assert_array_equal(actual[public], table.tabpfn_context_seed42)
    return arrays, owner, table, folds, blocks


def analyze():
    import numpy as np
    import pandas as pd
    from evaluation import possession_metrics as pm
    from factorized_latent import compose_owner
    from stoppage_consensus import decode_stoppage_consensus
    from broadcast_refinement import decode_broadcast_closure
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    shared = load_inputs()
    raw, bc, motion, eligible, _ = shared
    spans = [(13704,13785,"A"), (15766,15799,"H"), (15799,15889,"A"), (17293,17380,"H")]
    details, coverage, scores = [], [], []
    for protocol in PROTOCOLS:
        arrays, owner, table, folds, blocks = replay(protocol, shared)
        replacement = arrays[0].copy()
        for start,end in blocks:
            value = compose_owner(arrays[0][start:end], owner[start:end])
            replacement[start:end][eligible[start:end]] = value[eligible[start:end]]
        intermediate = dict(anchored=decode_stoppage_consensus(replacement,*arrays[1:4],raw,blocks),
                            broadcast=decode_broadcast_closure(replacement,*arrays[1:5],bc,raw,blocks))
        for start,end,label in spans:
            local = slice(start-11185,end-11185)
            column = 0 if label=="H" else 1
            probability = owner[start:end,column]
            original = arrays[0][start:end,:2]
            original = original / original.sum(axis=1,keepdims=True)
            split = next(s for s in folds if start in s["validation"])
            folder = SOURCE / protocol / f"block{split['block']}"
            with np.load(folder/"fitting.npz",allow_pickle=False) as fitting:
                centers = fitting["context_frames"]
                targets = fitting["target"]
                train = fitting["split_train"]
                validation = fitting["validation"]
            context = np.load(folder/"context.npy",allow_pickle=False)
            query = np.load(folder/"original.npy",allow_pickle=False)[start-int(validation[0]):end-int(validation[0])]
            scaler = StandardScaler().fit(context)
            X,Q=scaler.transform(context),scaler.transform(query)
            nearest={}
            for group,columns in (("all166",slice(None)),("original102",slice(0,102)),("latent64",slice(102,None))):
                nearest[group]={}
                for target in (0,1):
                    nn=NearestNeighbors(n_neighbors=1).fit(X[targets==target,columns])
                    distance,_=nn.kneighbors(Q[:,columns])
                    nearest[group]["H" if target==0 else "A"]=float(distance.mean())
            details.append(dict(protocol=protocol,block=split["block"],start=start,end_exclusive=end,label=label,
                context_rows=len(context),base_context_rows=len(centers),
                distance_to_nearest_fold_edge=min(start-int(validation[0]),int(validation[-1])+1-end),
                P_true_mean=float(probability.mean()),P_true_min=float(probability.min()),P_true_max=float(probability.max()),
                raw_owner_recall=float((owner[start:end].argmax(1)==column).mean()),
                original_slot_owner_recall=float((original.argmax(1)==column).mean()),
                anchored_recall=float((intermediate["anchored"][start:end]==label).mean()),
                broadcast_recall=float((intermediate["broadcast"][start:end]==label).mean()),
                final_recall=float((table.tabpfn_context_seed42.iloc[local]==label).mean()),
                reference_recall=float((table.current_seed42.iloc[local]==label).mean()),
                nearest_training_distance=nearest))
        for split in folds:
            folder = SOURCE/protocol/f"block{split['block']}"
            with np.load(folder/"fitting.npz",allow_pickle=False) as f:
                centers=f["context_frames"];target=f["target"][:len(centers)]
            segments=pm.segments(pm.label_codes(table.ground_truth),pm.valid_runs(pm.label_codes(table.ground_truth)))
            covered=[]
            for a,b,code in segments:
                left,right=11185+a,11185+b
                count=int(((centers>=left)&(centers<right)).sum())
                if count:covered.append(dict(start=left,end_exclusive=right,label=pm.SCORED[code],sampled_centers=count))
            coverage.append(dict(protocol=protocol,block=split["block"],base_context_rows=len(centers),
                original_H=int((target==0).sum()),original_A=int((target==1).sum()),
                sampled_truth_segments=covered))
        for model in ("current_seed42","tabpfn_context_seed42"):
            result=pm.evaluate(pm.label_codes(table.ground_truth),pm.label_codes(table[model]),bootstrap_n=0)
            scores.append(dict(protocol=protocol,model=model,spellF1=result["spell_f1_mean"],macroF1=result["macro_f1"]))
    OUT.mkdir(exist_ok=False)
    write_json(OUT/"analysis.json",dict(scores=scores,segment_probes=details,context_coverage=coverage,
        actual_frozen_decoder_replayed_exact=True,model_refits=0,
        source_sha256=digest(Path(__file__)),
        predecessor_manifest_sha256=digest(SOURCE/"manifest.json")))
    print(json.dumps(dict(scores=scores,segment_probes=[{k:v for k,v in row.items() if k!="nearest_training_distance"} for row in details]),indent=2))


if __name__=="__main__":
    analyze()
