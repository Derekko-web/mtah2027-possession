"""Bounded anonymous-player correspondence and label-free PathCRF input features."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter

RADIUS=150


def ordered_points(value):
    finite=np.asarray(value,float);finite=finite[np.isfinite(finite).all(axis=1)]
    return finite[np.lexsort((finite[:,1],finite[:,0]))]


def bounded_fill(path):
    result=np.array(path,float,copy=True);n,players,_=result.shape
    for player in range(players):
        seen=np.isfinite(result[:,player]).all(axis=1);indices=np.flatnonzero(seen)
        if seen.mean()<.8 or not len(indices):return None
        missing=np.flatnonzero(np.diff(np.r_[False,~seen,False]))
        for a,b in zip(missing[::2],missing[1::2]):
            if a==0 or b==n:
                if b-a>10:return None
            elif b-a>50:return None
        for axis in (0,1):result[:,player,axis]=np.interp(np.arange(n),indices,result[indices,player,axis])
    return result


def stitch_team(value):
    value=np.asarray(value,float);n=len(value);middle=n//2;seed=None
    for i in sorted(range(max(0,middle-25),min(n,middle+26)),key=lambda i:(abs(i-middle),i)):
        points=ordered_points(value[i])
        if len(points)!=11:continue
        d=np.linalg.norm(points[:,None]-points[None,:],axis=2);np.fill_diagonal(d,np.inf)
        if (d>.05).all():seed=i;break
    if seed is None:return None,'seed_roster'
    path=np.full((n,11,2),np.nan);path[seed]=ordered_points(value[seed])
    for direction in (-1,1):
        last=path[seed].copy();previous=last.copy();when=np.full(11,seed,int);previous_when=when.copy()
        for i in range(seed+direction,-1 if direction<0 else n,direction):
            points=ordered_points(value[i])
            if not len(points):continue
            delta=when-previous_when;velocity=np.divide(last-previous,delta[:,None]/25,out=np.zeros((11,2)),where=delta[:,None]!=0)
            speed=np.linalg.norm(velocity,axis=1);velocity*=np.minimum(1,12/np.maximum(speed,1e-12))[:,None]
            prediction=last+(i-when)[:,None]*velocity/25
            cost=np.linalg.norm(prediction[:,None]-points[None,:],axis=2)
            rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                gap=abs(i-when[row])
                if gap>50 or cost[row,col]>.75+12*gap/25:continue
                if np.linalg.norm(points[col]-last[row])>12*gap/25:continue
                row_other=np.delete(cost[row],col);col_other=np.delete(cost[:,col],row)
                if (len(row_other) and row_other.min()-cost[row,col]<.15) or (len(col_other) and col_other.min()-cost[row,col]<.15):continue
                path[i,row]=points[col];previous[row]=last[row];previous_when[row]=when[row];last[row]=points[col];when[row]=i
    filled=bounded_fill(path)
    return (None,'trajectory_gaps') if filled is None else (filled,'supported')


def running_features(path):
    n=len(path);features=np.zeros((n,11,6));features[:,:,:2]=path
    velocity=savgol_filter(np.diff(path,axis=0)*25,15,2,axis=0)
    acceleration=savgol_filter(np.diff(velocity,axis=0)*25,9,2,axis=0)
    features[1:,:,2:4]=velocity;features[0,:,2:4]=velocity[0]
    features[:,:,4]=np.linalg.norm(features[:,:,2:4],axis=2)
    features[1:-1,:,5]=np.linalg.norm(acceleration,axis=2)
    return features


def window(positions,teams,center):
    positions=np.asarray(positions);teams=np.asarray(teams)
    if center-RADIUS<0 or center+RADIUS>=len(positions):return None,None,'source_boundary'
    local=np.array(positions[center-RADIUS:center+RADIUS+1],float,copy=True)
    local[:,:,0]*=105/105.3;paths=[]
    for team in (0,1):
        path,reason=stitch_team(local[:,teams==team])
        if path is None:return None,None,reason
        paths.append(path)
    median=[np.median(p[:,:,0],axis=0) for p in paths]
    direct=median[0].min()+105-median[1].max();reverse=median[1].min()+105-median[0].max()
    if abs(direct-reverse)<10:return None,None,'orientation'
    left=0 if direct<reverse else 1;right=1-left
    value=np.zeros((51,26,8),np.float32);samples=np.arange(25,276,5)
    for role,team in enumerate((left,right)):
        a=role*11;b=a+11;value[:,a:b,0]=-1 if role==0 else 1
        keeper=np.argmin(median[team]) if role==0 else np.argmax(median[team])
        value[:,a+keeper,1]=1;value[:,a:b,2:]=running_features(paths[team])[samples]
    value[:,22:,2:4]=np.array([[0,34],[105,34],[52.5,0],[52.5,68]])
    if not np.isfinite(value).all():raise ValueError('Nonfinite model inputs after bounded correspondence')
    return value,left==0,'supported'
