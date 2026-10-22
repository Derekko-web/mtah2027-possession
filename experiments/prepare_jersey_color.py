"""Local-only color feasibility using excluded, weakly team-labelled player crops."""

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from broadcast_alignment import load_source
from build_temporal_candidate import digest
from games import GAME,half_video_blocks,require
from jersey_color import decode_frame,isolated_players,player_statistics,weak_frame_split
from starter.load_homography import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",action="store_true",help="Resume a confirmed stopped extraction from its saved local cache")
    args = parser.parse_args()
    output = ROOT/"data/control_jersey_v1/feasibility"
    output.mkdir(parents=True,exist_ok=args.resume)
    plan_path = ROOT/"experiments/JERSEY_COLOR_PLAN.json"
    plan = json.loads(plan_path.read_text())
    saved_plan = output/"plan.json"
    if saved_plan.exists() and saved_plan.read_bytes() != plan_path.read_bytes(): raise ValueError("Frozen color plan changed")
    saved_plan.write_bytes(plan_path.read_bytes())
    frozen = ROOT/"data/continuation_20261021/annotation_freeze/windows.json"
    if digest(frozen) != "5499723c34d803fbc2d15270510cab494687e2e2e37c59d87e246ce302cf4d1c":
        raise ValueError("Reserved windows changed")
    reserved = json.loads(frozen.read_text())["windows"]
    print("Loading local observations and excluded weak frame schedule; no possession labels",flush=True)
    source = load_source("tactical")
    matrices,_,valid = load()
    video = require(["video"])[0]
    info = json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "stream=width,height,r_frame_rate,start_time,nb_frames","-of","json",video],text=True))["streams"][0]
    if info != dict(width=1920,height=1080,r_frame_rate="25/1",start_time="0.000000",nb_frames="199319"):
        raise ValueError("Tactical video frame contract differs")
    train_frames,validation_frames = weak_frame_split(len(source["positions"]),half_video_blocks(GAME),reserved)
    frames = np.r_[train_frames,validation_frames]
    profiles = plan["crop"]["profiles"]
    mask = np.zeros((len(frames),len(source["teams"])),bool)
    for index,frame in enumerate(frames):
        mask[index,isolated_players(source["positions"][frame],source["speeds"][frame])] = True
    checkpoint = output/"weak_color_cache.npz"
    values = np.full((len(profiles),len(frames),len(source["teams"]),124),np.nan,np.float32)
    done = np.zeros(len(frames),bool)
    if checkpoint.exists():
        saved = np.load(checkpoint)
        np.testing.assert_array_equal(saved["frames"],frames)
        np.testing.assert_array_equal(saved["weak_mask"],mask)
        values,done = saved["features"],saved["done"]
    def save():
        temp = output/"cache_in_progress.npz"
        np.savez_compressed(temp,features=values,frames=frames,weak_mask=mask,teams=source["teams"],done=done)
        temp.replace(checkpoint)
    def extract(index):
        frame = int(frames[index])
        if not valid[frame]: return index,np.full_like(values[:,index],np.nan)
        image = decode_frame(frame)
        return index,np.stack([player_statistics(image,matrices[frame],source["positions"][frame],ratio) for ratio in profiles])
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(extract,i) for i in np.flatnonzero(~done)]
        for future in as_completed(futures):
            index,features = future.result()
            values[:,index],done[index] = features,True
            if done.sum()%20 == 0 or done.all():
                save(); print("Local color frames processed",int(done.sum()),"/",len(done),flush=True)
    target = np.broadcast_to(source["teams"],mask.shape)
    train_mask = np.arange(len(frames))[:,None] < len(train_frames)
    validation_mask = ~train_mask
    rows,selected = [],None
    for profile,ratio in enumerate(profiles):
        observed = mask & np.isfinite(values[profile]).all(axis=2)
        training = observed & train_mask
        testing = observed & validation_mask
        x,y = values[profile][training],target[training]
        vx,vy = values[profile][testing],target[testing]
        for kind in ("logistic","forest"):
            name = f"profile{profile}_{kind}"
            model = (make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight="balanced",max_iter=1000,random_state=42))
                if kind == "logistic" else RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=20,
                    class_weight="balanced_subsample",max_features="sqrt",n_jobs=2,random_state=42))
            if set(y) != {0,1} or set(vy) != {0,1}: raise ValueError("Both weak teams required")
            model.fit(x,y)
            if kind == "forest": model.set_params(n_jobs=1)
            probability = model.predict_proba(vx)
            prediction,confidence = probability.argmax(axis=1),probability.max(axis=1)
            confident = confidence >= .8
            joblib.dump(model,output/(name+".joblib"))
            if not np.array_equal(probability,joblib.load(output/(name+".joblib")).predict_proba(vx)):
                raise AssertionError("Color model reload probability parity failed")
            row = dict(model=name,ratio=ratio,training_crops=len(y),validation_crops=len(vy),
                validation_home=int((vy == 0).sum()),validation_away=int((vy == 1).sum()),
                valid_crop_fraction=float(testing.sum()/(mask & validation_mask).sum()),
                balanced_accuracy=float(balanced_accuracy_score(vy,prediction)),
                confident_coverage=float(confident.mean()),
                confident_accuracy=float((prediction[confident] == vy[confident]).mean()) if confident.any() else 0.)
            row["feasible"] = bool(min(row["validation_home"],row["validation_away"]) >= 250 and row["valid_crop_fraction"] >= .5 and
                row["balanced_accuracy"] >= .9 and row["confident_coverage"] >= .5 and row["confident_accuracy"] >= .95)
            if selected is None and row["feasible"]: selected = name
            rows.append(row)
    pd.DataFrame(rows).to_csv(output/"scores.csv",index=False)
    inputs = {f"tactical/{asset}":digest(require([asset])[0]) for asset in ("atd_meta","atd_raw","homography","video")}
    sources = ("jersey_color.py","prepare_jersey_color.py","broadcast_alignment.py")
    manifest = dict(plan_sha256=digest(plan_path),selected_model=selected,all_models_reload_exact=True,
        no_possession_targets_read=True,no_media_uploaded_or_displayed=True,weak_labels="Tracking team proxies, not human jersey or possession ground truth",
        train_frames=len(train_frames),validation_frames=len(validation_frames),validation_buffer=250,
        public_reserved_windows_excluded=True,annotation_metadata_sha256=digest(frozen),video_contract=info,
        source_sha256={name:digest(ROOT/"experiments"/name) for name in sources},input_sha256=inputs,
        cache_sha256={path.name:digest(path) for path in sorted(output.iterdir()) if path.is_file()})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(pd.DataFrame(rows).round(5).to_string(index=False),flush=True)
    print("Prospectively selected color model:",selected,flush=True)


if __name__ == "__main__": main()
