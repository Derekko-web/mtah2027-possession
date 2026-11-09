"""Observation-only fixed corruption views and bounded anonymous-set windows."""
import numpy as np


def observation_views(positions,teams,pitch,seed,centered=True):
    positions=np.asarray(positions,np.float32);teams=np.asarray(teams);pitch=np.asarray(pitch,np.float32)
    n,m,_=positions.shape
    if m>46 or set(teams)!={0,1}:raise ValueError('Unusable player-set schema')
    absolute=positions+pitch/2 if centered else positions.copy()
    rng=np.random.default_rng(seed)
    dropout=np.repeat(rng.random(((n+9)//10,m))>=.2,10,axis=0)[:n]
    crop_centers=np.repeat(rng.uniform([0,0],pitch,size=((n+19)//20,2)),20,axis=0)[:n]
    burst=dropout.copy()
    for a in range(0,n,50):
        length=int(rng.integers(3,10));length=min(length,n-a)
        start=int(rng.integers(a,a+min(50,n-a)-length+1));burst[start:start+length]=False
    x=np.zeros((4,n,46,2),np.float32);mask=np.zeros((4,n,46),bool)
    group=np.zeros((n,46),np.int8);group[:,:m]=teams
    for view,retained in enumerate((np.ones((n,m),bool),dropout,None,burst)):
        value=absolute.copy()
        if view:value+=rng.normal(0,.2,value.shape).astype(np.float32)
        if view==2:retained=(np.abs(value-crop_centers[:,None])<=np.array([37.5,30])).all(axis=-1)
        valid=np.isfinite(value).all(axis=-1)&(value>=-5).all(axis=-1)&(value<=pitch+5).all(axis=-1)&retained
        normalized=value/pitch*2-1
        x[view,:,:m]=np.where(valid[:,:,None],normalized,0)
        mask[view,:,:m]=valid
    return x,mask,group


def assemble(x,mask,teams,half_id,centers,views,flip_x=None,flip_y=None,repeat=False,trim=True):
    centers=np.asarray(centers,int);views=np.asarray(views,int)
    indices=centers[:,None]+np.arange(-15,16)[None]
    if (indices<0).any() or (indices>=x.shape[1]).any() or not (half_id[indices]==half_id[centers,None]).all():raise ValueError('A model window crossed a half')
    values=np.array(x[views[:,None],indices],copy=True)
    observed=np.array(mask[views[:,None],indices],copy=True);groups=np.array(teams[indices],copy=True)
    if flip_x is not None:values[:,:,:,0]*=np.where(np.asarray(flip_x)[:,None,None],-1,1)
    if flip_y is not None:values[:,:,:,1]*=np.where(np.asarray(flip_y)[:,None,None],-1,1)
    if trim:
        keep=observed.any(axis=(0,1))
        if not keep.any():keep[0]=True
        values=values[:,:,keep];observed=observed[:,:,keep];groups=groups[:,:,keep]
    if repeat:
        values=np.repeat(values[:,15:16],31,axis=1);observed=np.repeat(observed[:,15:16],31,axis=1);groups=np.repeat(groups[:,15:16],31,axis=1)
    return values,observed,groups
