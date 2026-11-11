"""Bounded, fallible pretrained action/actor features; no possession targets."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from provided_speed import bounded_statistics
from sports_action_sampling import clip_frames
from starter.load_homography import project

ACTION_NAMES = ['pass','drive','header','high_pass','out','cross','throw_in','shot',
                'ball_player_block','player_successful_tackle','free_kick','goal']
PROBABILITY = ['sa_background']+['sa_'+name for name in ACTION_NAMES]


def dense_actions(folder, records, n):
    records = sorted([r for r in records if clip_frames(r['center'])[-1] < n], key=lambda r:r['center'])
    frame = np.arange(n)
    output = {name:np.full(n,np.nan) for name in PROBABILITY+['sa_event_max','sa_entropy','sa_displacement','sa_team_right']}
    output['sa_seen'] = np.zeros(n)
    if not records:
        return pd.DataFrame(output).astype(np.float32)
    centers = np.array([r['center'] for r in records])
    right = np.minimum(np.searchsorted(centers,frame),len(centers)-1)
    left = np.maximum(right-1,0)
    chosen = np.where(np.abs(frame-centers[left]) <= np.abs(frame-centers[right]),left,right)
    valid = np.abs(frame-centers[chosen]) <= 25
    for i,row in enumerate(records):
        query = np.flatnonzero(valid & (chosen == i))
        if not len(query):
            continue
        data = np.load(Path(folder)/row['file'])
        np.testing.assert_array_equal(data['frames'],clip_frames(row['center']))
        logits = data['im_feat'][:,:13].astype(float)
        probability = np.exp(logits-logits.max(axis=1,keepdims=True))
        probability /= probability.sum(axis=1,keepdims=True)
        position = (query-data['frames'][0])/2
        lo,hi = np.floor(position).astype(int),np.ceil(position).astype(int)
        alpha = position-lo
        p = probability[lo]*(1-alpha[:,None])+probability[hi]*alpha[:,None]
        for j,name in enumerate(PROBABILITY):
            output[name][query] = p[:,j]
        output['sa_event_max'][query] = p[:,1:].max(axis=1)
        output['sa_entropy'][query] = -(p*np.log(np.maximum(p,1e-30))).sum(axis=1)
        for key,values in [('sa_displacement',data['displ_feat']),('sa_team_right',expit(data['team_feat'].astype(float)))]:
            output[key][query] = values[lo]*(1-alpha)+values[hi]*alpha
        output['sa_seen'][query] = 1
    return pd.DataFrame(output).astype(np.float32)


def actor_row(positions,teams,matrix,q,width=1920,height=1080):
    if not np.isfinite(q):
        return None
    good = np.isfinite(positions).all(axis=1)
    xy = positions[good]
    group = teams[good]
    if not len(xy):
        return None
    x,y = project(matrix,xy[:,0]/105.3,xy[:,1]/68.)
    visible = np.isfinite(x) & np.isfinite(y) & (x>=0) & (x<width) & (y>=0) & (y<height)
    medians = []
    for team in (0,1):
        values = x[visible & (group == team)]
        if len(values)<3:
            return None
        medians.append(float(np.median(values)))
    if abs(medians[0]-medians[1]) < 10:
        return None
    return [(medians[t]-medians[1-t])/width for t in (0,1)], [1-q if medians[t]<medians[1-t] else q for t in (0,1)]


def add_summaries(core,columns,blocks):
    result = core.copy()
    extras = {}
    for column in columns:
        for radius in (25,50):
            for direction in ('past','future'):
                mean,maximum = np.full(len(core),np.nan),np.full(len(core),np.nan)
                for left,right in blocks:
                    left,right = max(0,left),min(len(core),right)
                    if left>=right:
                        continue
                    values = core[column].iloc[left:right].to_numpy(float)
                    mean[left:right] = bounded_statistics(values,radius,direction)[0]
                    padded = np.pad(values,(radius,0) if direction=='past' else (0,radius),constant_values=np.nan)
                    windows = np.lib.stride_tricks.sliding_window_view(padded,radius+1)
                    maxima = np.where(np.isfinite(windows),windows,-np.inf).max(axis=1)
                    maxima[~np.isfinite(maxima)] = np.nan
                    maximum[left:right] = maxima
                extras[column+f'_{direction}{radius}_mean'] = mean
                extras[column+f'_{direction}{radius}_max'] = maximum
    return result.join(pd.DataFrame(extras)).astype(np.float32)


def feature_views(dense,source,matrices,error,valid,blocks):
    event_core = dense.drop(columns='sa_team_right')
    events = add_summaries(event_core,PROBABILITY[1:]+['sa_event_max'],blocks)
    n = len(dense)
    extra = {'sa_team_right':dense.sa_team_right.to_numpy(), 'sa_actor_seen':np.zeros(n)}
    for prefix in ('h__','a__'):
        for name in ['image_margin','actor_probability','actor_event']+ACTION_NAMES:
            extra[prefix+'sa_'+name] = np.full(n,np.nan)
    for frame in np.flatnonzero((dense.sa_seen.to_numpy()>0) & valid[:n] & (error[:n]<=1)):
        value = actor_row(source['positions'][frame],source['teams'],matrices[frame],float(dense.sa_team_right.iloc[frame]))
        if value is None:
            continue
        margins,shares = value
        extra['sa_actor_seen'][frame] = 1
        for team,prefix in enumerate(('h__','a__')):
            extra[prefix+'sa_image_margin'][frame] = margins[team]
            extra[prefix+'sa_actor_probability'][frame] = shares[team]
            extra[prefix+'sa_actor_event'][frame] = shares[team]*dense.sa_event_max.iloc[frame]
            for name in ACTION_NAMES:
                extra[prefix+'sa_'+name][frame] = shares[team]*dense['sa_'+name].iloc[frame]
    actor = add_summaries(pd.DataFrame(extra),['h__sa_actor_event','a__sa_actor_event'],blocks)
    return dict(events_only=events,events_actor=events.join(actor))


def support_summary(dense,views):
    public = dense.iloc[11185:18685]
    value = np.nan_to_num(public.sa_event_max.to_numpy(float),nan=0)
    candidates = np.flatnonzero((value>=.3) & (value>np.r_[-np.inf,value[:-1]]) & (value>=np.r_[value[1:],-np.inf]))
    peaks = []
    for index in sorted(candidates,key=lambda i:(-value[i],i)):
        if all(abs(index-old)>=25 for old in peaks):
            peaks.append(int(index))
    observed = int((public.sa_seen>0).sum())
    actor_coverage = float(views['events_actor'].sa_actor_seen.iloc[11185:18685].mean())
    passed = observed>=5000 and len(peaks)>=25
    eligible = ['events_only']+(['events_actor'] if actor_coverage>=.5 else [])
    return dict(observed_public_frames=observed,confident_local_peaks=len(peaks),
        event_max_std=float(value.std()),actor_coverage=actor_coverage,support_passed=passed,eligible_heads=eligible)
