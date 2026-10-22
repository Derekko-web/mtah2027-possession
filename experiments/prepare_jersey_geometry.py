"""Decode only local padded development video and build color-team geometry."""

import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from games import require
from jersey_color import HEIGHT,WIDTH,decode_frame,player_statistics
from jersey_ownership import jersey_geometry
from starter.load_homography import load


def main():
    root = ROOT/"data/control_jersey_v1"
    output = root/"inputs"; output.mkdir(exist_ok=False)
    feasibility = root/"feasibility"
    manifest = json.loads((feasibility/"manifest.json").read_text())
    plan_path = ROOT/"experiments/JERSEY_OWNERSHIP_SCREEN_PLAN.json"
    plan = json.loads(plan_path.read_text())
    if digest(feasibility/"manifest.json") != plan["feasibility_manifest_sha256"]: raise ValueError("Color feasibility changed")
    for name,sha in manifest["source_sha256"].items():
        if digest(ROOT/"experiments"/name) != sha: raise ValueError("Color source changed")
    for identity,sha in manifest["input_sha256"].items():
        source,asset = identity.split("/")
        if digest(require([asset],source=source)[0]) != sha: raise ValueError("Local color input changed")
    for name,sha in manifest["cache_sha256"].items():
        if digest(feasibility/name) != sha: raise ValueError("Color cache changed")
    if manifest["selected_model"] is None: raise ValueError("No feasible color model")
    row = pd.read_csv(feasibility/"scores.csv").set_index("model").loc[manifest["selected_model"]]
    model = joblib.load(feasibility/(manifest["selected_model"]+".joblib"))
    if not np.array_equal(model.classes_,[0,1]): raise ValueError("Color team order differs")
    source = load_source("tactical",19060); matrices,_,_ = load()
    frames = np.arange(10935,18936,5)
    probability = np.full((len(frames),len(source["teams"]),2),np.nan)
    command = ["ffmpeg","-nostdin","-v","error","-threads","2","-ss",f"{frames[0]/25:.8f}","-i",require(["video"])[0],
        "-t",f"{(frames[-1]-frames[0]+1)/25:.8f}","-filter_threads","1","-vf","select='not(mod(n,5))',scale=960:540:flags=bilinear",
        "-fps_mode","passthrough","-pix_fmt","rgb24","-f","rawvideo","pipe:1"]
    print("Extracting local5-frame color observations; no possession targets",flush=True)
    with (output/"decoder.log").open("wb") as log:
        process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for index,frame in enumerate(frames):
                data = bytearray()
                while len(data) < WIDTH*HEIGHT*3:
                    part = process.stdout.read(WIDTH*HEIGHT*3-len(data))
                    if not part: raise ValueError("Local video stream ended early")
                    data.extend(part)
                image = np.frombuffer(data,np.uint8).reshape(HEIGHT,WIDTH,3)
                if index in (0,len(frames)//2,len(frames)-1):
                    if not np.array_equal(image,decode_frame(frame)): raise AssertionError("Stream and accurate-seek frames differ")
                features = player_statistics(image,matrices[frame],source["positions"][frame],row.ratio)
                valid = np.isfinite(features).all(axis=1)
                if valid.any(): probability[index,valid] = model.predict_proba(features[valid])
                if index%100 == 0: print("Color samples processed",index,"/",len(frames),flush=True)
            if process.stdout.read(1): raise ValueError("Local video stream returned extra frames")
            if process.wait() != 0: raise ValueError("Local decoder failed")
        finally:
            process.stdout.close()
            if process.poll() is None: process.terminate(); process.wait()
    np.savez_compressed(output/"sampled_team_probability.npz",frames=frames,probability=probability)
    features = jersey_geometry(source["positions"],source["speeds"],source["ball"],source["teams"],frames,probability)
    features.to_pickle(output/"jersey_inputs.pkl")
    n = 18935
    prefix = jersey_geometry(source["positions"][:n],source["speeds"][:n],source["ball"][:n],source["teams"],frames[frames < n],probability[frames < n])
    pd.testing.assert_frame_equal(features.iloc[11185:18685],prefix.iloc[11185:18685])
    observations = dict(sample_frames=len(frames),feature_columns=len(features.columns),public_observed_fraction=float(features.jc_ball_seen.iloc[11185:18685].mean()),
        public_mean_classified_fraction=float(features.jc_classified_fraction.iloc[11185:18685].mean()),
        public_mean_reassigned_fraction=float(features.jc_reassigned_fraction.iloc[11185:18685].mean()))
    names = ("prepare_jersey_geometry.py","jersey_color.py","jersey_ownership.py","contact_dynamics.py","weak_ownership.py","broadcast_intersection.py")
    result = dict(plan_sha256=digest(plan_path),source_sha256={name:digest(ROOT/"experiments"/name) for name in names},
        input_sha256={"data/control_jersey_v1/feasibility/manifest.json":digest(feasibility/"manifest.json")},
        cache_sha256={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()},observations=observations,
        public_possession_labels_not_read=True,actual_padded_prefix_parity=True,stream_seek_pixel_parity=True)
    (output/"manifest.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(observations,indent=2),flush=True)


if __name__ == "__main__": main()
