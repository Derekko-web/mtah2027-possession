"""Ball-relative motion from unambiguous team position sets on each native timeline."""

import numpy as np
import pandas as pd


def native_pair_features(source,frames):
    frames = np.asarray(frames,dtype=int)
    positions,ball,teams = source["positions"],source["ball"],source["teams"]
    result = {}
    n = len(positions)
    bad = np.r_[0,np.cumsum(~source["valid"])]
    for team,prefix in ((0,"h__"),(1,"a__")):
        xy = positions[:,teams == team]
        current = xy[frames]
        distances = np.linalg.norm(current-ball[frames,None,:],axis=2)
        safe_distance = np.where(np.isfinite(distances),distances,np.inf)
        nearest = safe_distance.argmin(axis=1)
        d = safe_distance.min(axis=1);d[~np.isfinite(d)] = np.nan
        result[prefix+"distance_1"] = d
        for step in (5,25):
            for direction,sign in (("past",-1),("future",1)):
                other_frames = frames+sign*step
                valid = (other_frames >= 0)&(other_frames < n)
                clipped = np.clip(other_frames,0,n-1)
                low,high = np.minimum(frames,clipped),np.maximum(frames,clipped)
                valid &= bad[high+1]-bad[low] == 0
                outputs = {name:np.full(len(frames),np.nan) for name in ("nearest_relative","minimum_relative_3m","control_kernel","match_fraction")}
                for start in range(0,len(frames),256):
                    end = min(start+256,len(frames)); rows = np.arange(end-start)
                    now,old = current[start:end],xy[clipped[start:end]]
                    delta = now[:,:,None,:]-old[:,None,:,:]
                    cost = np.linalg.norm(delta,axis=3)
                    cost[~np.isfinite(cost)] = np.inf
                    match = cost.argmin(axis=2)
                    best_two = np.sort(cost,axis=2)[:,:,:2]
                    best = best_two[:,:,0]
                    margin = np.full_like(best,np.nan)
                    finite_best = np.isfinite(best)
                    np.subtract(best_two[:,:,1],best,out=margin,where=finite_best)
                    reverse = cost.argmin(axis=1)
                    mutual = reverse[rows[:,None],match] == np.arange(now.shape[1])[None]
                    good = mutual & finite_best & (margin >= .25) & (best/(step/25) <= 15) & valid[start:end,None]
                    matched = old[rows[:,None],match]
                    velocity = (now-matched)/(-sign*step/25)
                    velocity[~good] = np.nan
                    bv = (ball[frames[start:end]]-ball[clipped[start:end]])/(-sign*step/25)
                    bv[np.linalg.norm(bv,axis=1) > 80] = np.nan
                    relative = np.linalg.norm(velocity-bv[:,None,:],axis=2)
                    relative[~valid[start:end]] = np.nan
                    local_distance = distances[start:end]
                    outputs["nearest_relative"][start:end] = relative[rows,nearest[start:end]]
                    close = np.where((local_distance <= 3)&np.isfinite(relative),relative,np.inf)
                    minimum = close.min(axis=1);minimum[~np.isfinite(minimum)] = np.nan
                    outputs["minimum_relative_3m"][start:end] = minimum
                    kernel = np.exp(-.5*(local_distance/2)**2-.5*(relative/5)**2)
                    maximum = np.max(np.where(np.isfinite(kernel),kernel,-1),axis=1)
                    maximum[maximum < 0] = np.nan
                    outputs["control_kernel"][start:end] = maximum
                    observed = np.isfinite(now).all(axis=2).sum(axis=1)
                    fraction = good.sum(axis=1)/np.maximum(observed,1)
                    fraction[~valid[start:end] | (observed == 0)] = np.nan
                    outputs["match_fraction"][start:end] = fraction
                for name,value in outputs.items(): result[prefix+f"{direction}{step}_"+name] = value
    return pd.DataFrame(result).replace([np.inf,-np.inf],np.nan).astype(np.float32)


def enclosure_mask(mapping,step,radius):
    frames,orientation = mapping["frames"],mapping["orientation"]
    n = len(frames);result = np.zeros(n,bool)
    if n <= 2*radius: return result
    for code in range(4):
        valid = (frames >= 0)&(orientation == code)
        complete = pd.Series(valid.astype(float)).rolling(2*radius+1,center=True,min_periods=2*radius+1).min().to_numpy() == 1
        center = np.arange(radius,n-radius)
        use = complete[center] & (frames[center]-frames[center-radius] >= step) & (frames[center+radius]-frames[center] >= step)
        result[center[use]] = True
    return result


def finish_motion(features,prefix):
    result = features.rename(columns={name:name[:3]+prefix+name[3:] for name in features})
    extra = {}
    for team in ("h__","a__"):
        kernel = np.minimum(result[team+prefix+"past5_control_kernel"],result[team+prefix+"future5_control_kernel"])
        result[team+prefix+"kernel"] = kernel
        for radius in (25,75,125):
            extra[team+prefix+f"kernel_past{radius}"] = kernel.rolling(radius+1,min_periods=1).mean()
            extra[team+prefix+f"kernel_future{radius}"] = kernel.iloc[::-1].rolling(radius+1,min_periods=1).mean().iloc[::-1]
    result[prefix+"observed"] = np.isfinite(result[[team+prefix+"kernel" for team in ("h__","a__")]]).all(axis=1).astype(float)
    return result.join(pd.DataFrame(extra)).astype(np.float32)


def pair_feature_views(tactical,broadcast,mapping,bc_features):
    n = len(tactical["positions"])
    tactical_features = native_pair_features(tactical,np.arange(n))
    frames = mapping["frames"]
    queries = np.unique(frames[frames >= 0])
    if not len(queries): raise ValueError("No aligned source queries")
    native = native_pair_features(broadcast,queries)
    aligned = pd.DataFrame(np.nan,index=np.arange(n),columns=native.columns,dtype=np.float32)
    valid = frames >= 0
    aligned.loc[valid] = native.iloc[np.searchsorted(queries,frames[valid])].to_numpy()
    for step,radius in ((5,10),(25,35)):
        allow = enclosure_mask(mapping,step,radius)
        columns = [name for name in aligned if f"past{step}_" in name or f"future{step}_" in name]
        aligned.loc[~allow,columns] = np.nan
    tc,bm = finish_motion(tactical_features,"tc_"),finish_motion(aligned,"bm_")
    taps = {}
    for name in ["bc_ball_seen"]+[prefix+"bc_same_"+suffix for prefix in ("h__","a__") for suffix in ("distance_1","control_share","nearest_speed")]:
        for lag in (-50,-25,-5,5,25,50): taps[name+f"_pair_lag{lag}"] = bc_features[name].shift(lag)
    base = bc_features.join(pd.DataFrame(taps)).astype(np.float32)
    views = dict(broadcast_taps=base,tactical_motion=tc,broadcast_motion=base.join(bm),fused_motion=base.join(tc).join(bm))
    info = tc.join(bm)
    return views,info
