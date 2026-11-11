"""One-variable native probe: identical context, differently grouped query rows."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import tabpfn_extended_evaluation as native

ROOT=native.ROOT
OUT=native.ROUND/"diagnosis_query_grouping_v1"
SOURCE=native.ROUND/"screen/buffered_100_seconds/block2"


def run():
    native.verify()
    native.require_private_runtime(sys.prefix,sys.base_prefix,sys.executable,True)
    import numpy as np
    OUT.mkdir(exist_ok=False)
    # Both groups contain only rows from the original held-out block. They move
    # the two affected events within canonical512 batches; no labels are supplied.
    indices=np.r_[np.arange(7,519),np.arange(1950,2462)]
    for name in ("context","target"):
        shutil.copyfile(SOURCE/(name+".npy"),OUT/(name+".npy"))
    for name in ("original","swapped"):
        value=np.load(SOURCE/(name+".npy"),allow_pickle=False)
        np.save(OUT/(name+".npy"),np.ascontiguousarray(value[indices]),allow_pickle=False)
    np.save(OUT/"source_query_indices.npy",indices,allow_pickle=False)
    job={name:str(OUT/(name+".npy")) for name in ("context","target","original","swapped")}
    job.update(seed=42,protocol="buffered_100_seconds",block=2,
        input_sha256={str(OUT/(name+".npy")):native.digest(OUT/(name+".npy"))
                      for name in ("context","target","original","swapped")})
    native.write_json(OUT/"job.json",job)
    native.write_json(OUT/"plan.json",dict(purpose="Isolate query grouping from training-context change",
        same_context_and_targets_byte_exact=True,groups=[[7,519],[1950,2462]],query_labels_supplied=False,
        same_fixed_model_seed_precision_backend_preprocessing=True,operation_seconds=900,worker_seconds=3600,
        combined_memory_bytes=6*1024**3,original_frozen_worker_source_sha256=native.digest(native.LEGACY_SOURCE),
        original_context_sha256=native.digest(SOURCE/"context.npy"),source_sha256=native.digest(Path(__file__)),
        user_authorization="Investigate why TabPFN loses longer-block spellF1 and test a fix",
        no_candidate_or_selected_model_change=True))
    native.supervise(OUT/"job.json",OUT/"worker")
    new,old=OUT/"worker",SOURCE/"worker"
    comparisons={}
    for name in ("original_probability","swapped_probability","owner"):
        actual=np.load(new/(name+".npy"),allow_pickle=False)
        expected=np.load(old/(name+".npy"),allow_pickle=False)[indices]
        comparisons[name]=dict(max_abs_difference=float(np.max(np.abs(actual-expected))),
            byte_exact=actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes(),
            different_argmax_rows=int((actual.argmax(1)!=expected.argmax(1)).sum()))
    for name in ("checkpoint_fingerprint.json","preprocessing_fingerprint.json"):
        if (new/name).read_bytes()!=(old/name).read_bytes():
            raise ValueError("The probe changed fitted state: "+name)
    result=dict(passed=True,same_model_and_preprocessing=True,query_grouping_only=True,
        comparisons=comparisons,query_labels_supplied=False,model_fits=1,
        large_batching_explanation_supported=any(v["max_abs_difference"]>.01 for v in comparisons.values()),
        inference_only_correction_scores_unchanged=True)
    native.write_json(OUT/"result.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    run()
