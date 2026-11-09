"""Native half-boundary padding; unavailable context is explicitly unobserved."""
import numpy as np


def assemble_native(x,mask,teams,half_id,centers,repeat=False):
    centers=np.asarray(centers,int)
    if (centers<0).any() or (centers>=x.shape[1]).any() or (half_id[centers]<0).any():raise ValueError('Native centers must be inside a known half')
    indices=centers[:,None]+np.arange(-15,16)[None]
    clipped=np.clip(indices,0,x.shape[1]-1)
    allowed=(indices>=0)&(indices<x.shape[1])&(half_id[clipped]==half_id[centers,None])
    value=np.array(x[0,clipped],copy=True);observed=np.array(mask[0,clipped],copy=True);groups=np.array(teams[clipped],copy=True)
    if repeat:
        value=np.repeat(value[:,15:16],31,axis=1);observed=np.repeat(observed[:,15:16],31,axis=1);groups=np.repeat(groups[:,15:16],31,axis=1)
    value[~allowed]=0;observed[~allowed]=False;groups[~allowed]=0
    return value,observed,groups
