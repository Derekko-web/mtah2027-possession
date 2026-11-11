"""Target-free full102+fixed64 features with exact public-prefix parity."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/control_tabpfn_full_build_v1'
sys.path.insert(0,str(ROOT))


def main():
    import tabpfn_fixed_confirmation_v2 as native
    native.verify()
    native.require_private_runtime(sys.prefix,sys.base_prefix,sys.executable,controller=True)
    plan=json.loads((OUT/'input_preparation_plan_v3.json').read_text())
    for name,sha in plan['sha256'].items():
        if native.digest(ROOT/name)!=sha:raise ValueError('Full input preparation binding changed: '+name)
    import numpy as np
    import pandas as pd
    from temporal_candidate import build_inputs
    from games import source_frames
    from player_representation import COLUMNS
    enc=OUT/'encodings';manifest=json.loads((enc/'manifest.json').read_text())
    for name,sha in manifest['cache_sha256'].items():
        if native.digest(enc/name)!=sha:raise ValueError('Full encoder artifact changed')
    if not manifest['public_hidden_and_eligibility_exact'] or not manifest['public_interpolation_exact']:
        raise ValueError('Full fixed encoder/public parity incomplete')
    output=OUT/'full_inputs_v3';output.mkdir(exist_ok=False)
    print('Rebuilding full original102 label-free inputs',flush=True)
    raw,base,ordered=build_inputs(source_frames())
    public=np.arange(11185,18685)
    old=ROOT/'data/advanced_validation'
    pd.testing.assert_frame_equal(ordered.iloc[public],pd.read_pickle(old/'ordered.pkl').iloc[public])
    prefix_raw=pd.read_pickle(old/'raw.pkl')
    pd.testing.assert_frame_equal(raw.loc[:,prefix_raw.columns].iloc[public],prefix_raw.iloc[public])
    latent=np.load(enc/'full_latent.npy',allow_pickle=False)
    eligible=np.load(enc/'full_eligible.npy',allow_pickle=False)
    if ordered.shape!=(source_frames(),102) or latent.shape!=(source_frames(),64) or eligible.shape!=(source_frames(),):
        raise ValueError('Full input shape differs')
    inputs=ordered.join(pd.DataFrame(latent,columns=COLUMNS,index=ordered.index))
    if inputs.shape!=(source_frames(),166):raise ValueError('Immutable166-feature interface differs')
    raw.to_pickle(output/'raw.pkl');ordered.to_pickle(output/'ordered.pkl');inputs.to_pickle(output/'inputs.pkl')
    native.write_json(output/'feature_columns.json',list(inputs.columns))
    for name,sha in plan['sha256'].items():
        if native.digest(ROOT/name)!=sha:raise ValueError('Preparation source changed during execution')
    native.verify()
    native.write_json(output/'manifest.json',dict(target_labels_loaded=False,model_fits=0,
        n_frames=source_frames(),columns=166,public_original102_and_raw_exact=True,
        fixed64_encoder_public_parity=True,input_preparation_plan_sha256=native.digest(OUT/'input_preparation_plan_v3.json'),
        cache_sha256={p.name:native.digest(p) for p in output.iterdir() if p.is_file()}))
    print('Full166 inputs complete; original102/raw and fixed64 public parity exact',flush=True)


if __name__=='__main__':main()
