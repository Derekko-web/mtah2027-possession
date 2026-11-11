"""Target-independent ball perturbations alongside fixed R36 player observations."""
import numpy as np
from masked_player_data import assemble


def ball_views(ball,pitch,player_xy,player_seen,seed):
    pitch=np.asarray(pitch,np.float32);absolute=np.asarray(ball,np.float32)+pitch/2;n=len(absolute);rng=np.random.default_rng(seed)
    values=np.zeros((4,n,2),np.float32);observed=np.zeros((4,n),bool)
    for view in range(4):
        value=absolute.copy();keep=np.ones(n,bool)
        if view:
            p=.5 if view==3 else .35;keep=np.repeat(rng.random((n+9)//10)>=p,10)[:n]
            value+=rng.normal(0,.5 if view==1 else 1.,value.shape).astype(np.float32)
        if view==3:
            blocks=(n+9)//10;shifted=rng.random(blocks)<.2;angle=rng.uniform(0,2*np.pi,blocks);distance=rng.uniform(5,15,blocks)
            offset=np.stack([np.cos(angle),np.sin(angle)],axis=1)*distance[:,None]*shifted[:,None]
            value+=np.repeat(offset.astype(np.float32),10,axis=0)[:n]
        normalized=value/pitch*2-1
        if view==2:
            seen=player_seen[view];xy=player_xy[view]
            lower=np.where(seen[:,:,None],xy,np.inf).min(1)-.2;upper=np.where(seen[:,:,None],xy,-np.inf).max(1)+.2
            keep&=((normalized>=lower)&(normalized<=upper)).all(1)
        valid=keep&np.isfinite(value).all(1)&(value>=-5).all(1)&(value<=pitch+5).all(1)&player_seen[view].any(1)
        values[view]=np.where(valid[:,None],normalized,0);observed[view]=valid
    return values,observed


def assemble_ball(x,mask,teams,half,ball,ball_seen,centers,views,flip_x=None,flip_y=None,control=False,trim=True):
    values=list(assemble(x,mask,teams,half,centers,views,flip_x,flip_y,trim=trim));index=np.asarray(centers)[:,None]+np.arange(-15,16)[None]
    b=np.array(ball[np.asarray(views)[:,None],index],copy=True);seen=np.array(ball_seen[np.asarray(views)[:,None],index],copy=True)
    if flip_x is not None:b[:,:,0]*=np.where(np.asarray(flip_x)[:,None],-1,1)
    if flip_y is not None:b[:,:,1]*=np.where(np.asarray(flip_y)[:,None],-1,1)
    if control:b[:]=0;seen[:]=False
    return values+[b,seen]
