"""Ball/player relative motion from 0.2 s displacements, without identity assumptions."""

import numpy as np
import pandas as pd

from games import GAME,GAMES,require
from starter.baseline import team_of_each_track


def build_relative_motion(n_frames):
    meta,raw=require(['atd_meta','atd_raw']);teams=team_of_each_track(meta)
    pitch=np.array(GAMES[GAME]['pitch']);positions=np.full((n_frames,len(teams),2),np.nan)
    ball=np.full((n_frames,2),np.nan)
    with open(raw) as handle:
        for frame,line in enumerate(handle):
            if frame>=n_frames:break
            frame_id,player_raw,ball_raw=line.strip().split(':')
            if int(frame_id)!=frame+1:raise ValueError('Bad frame index')
            positions[frame]=np.fromstring(player_raw.replace(';',','),sep=',').reshape(-1,3)[:,:2]*pitch
            ball[frame]=np.fromstring(ball_raw,sep=',')[:2]*pitch
    step=5;dt=.2
    pv=np.full_like(positions,np.nan);bv=np.full_like(ball,np.nan)
    pv[step:]=(positions[step:]-positions[:-step])/dt
    bv[step:]=(ball[step:]-ball[:-step])/dt
    pv[np.linalg.norm(pv,axis=2)>15]=np.nan
    bv[np.linalg.norm(bv,axis=1)>80]=np.nan
    columns={}
    for team,prefix in [(0,'h__'),(1,'a__')]:
        xy=positions[:,teams==team];v=pv[:,teams==team]
        distance=np.linalg.norm(xy-ball[:,None,:],axis=2)
        relative=np.linalg.norm(v-bv[:,None,:],axis=2)
        nearest=np.argmin(np.where(np.isfinite(distance),distance,np.inf),axis=1)
        d=distance[np.arange(n_frames),nearest]
        columns[prefix+'nearest_relative_velocity']=relative[np.arange(n_frames),nearest]
        columns[prefix+'distance_change_0.2s']=pd.Series(d).diff(step).to_numpy()/dt
        near=relative.copy();near[(distance>3)|(~np.isfinite(distance))]=np.nan
        has=np.isfinite(near).any(axis=1)
        minimum=np.min(np.where(np.isfinite(near),near,np.inf),axis=1)
        minimum[~has]=np.nan
        columns[prefix+'relative_velocity_within_3m']=minimum
        offset=xy[np.arange(n_frames),nearest]-ball
        dot=(offset*bv).sum(axis=1)
        norm=np.linalg.norm(offset,axis=1)*np.linalg.norm(bv,axis=1)
        columns[prefix+'ball_direction_alignment']=dot/np.maximum(norm,.01)
    return pd.DataFrame(columns).replace([np.inf,-np.inf],np.nan).astype(np.float32)
