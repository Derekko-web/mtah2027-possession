"""Bounded, training-only transforms for a public-label live/dead sequence head."""
import numpy as np

RADIUS=31
DILATIONS=(1,2,4,8,16)
WIDTH=16
STEPS=600
BATCH=128


def normalization(values,train):
    values=np.asarray(values,dtype=np.float64);train=np.asarray(train,int)
    if values.ndim!=2 or not len(train) or len(np.unique(train))!=len(train) or (train<0).any() or (train>=len(values)).any():
        raise ValueError('Expected unique eligible preprocessing rows')
    sample=values[train];finite=np.isfinite(sample);count=finite.sum(axis=0)
    mean=np.where(finite,sample,0).sum(axis=0)/np.maximum(count,1)
    variance=np.where(finite,(sample-mean)**2,0).sum(axis=0)/np.maximum(count,1)
    scale=np.sqrt(variance);scale[scale<1e-6]=1
    return mean,scale


def transform(values,mean,scale):
    values=np.asarray(values,dtype=np.float64);finite=np.isfinite(values)
    normalized=np.clip(np.where(finite,(values-mean)/scale,0),-8,8)
    return np.c_[normalized,~finite].astype(np.float32)


def windows(values,frames,half_blocks,repeat=False,radius=RADIUS):
    values=np.asarray(values,dtype=np.float32);frames=np.asarray(frames,int)
    if values.ndim!=2 or values.shape[1]%2 or not len(frames) or (frames<0).any() or (frames>=len(values)).any():
        raise ValueError('Invalid normalized frame window')
    indices=frames[:,None]+np.arange(-radius,radius+1)[None,:]
    if repeat:indices=np.broadcast_to(frames[:,None],indices.shape)
    allowed=np.zeros_like(indices,dtype=bool);assigned=np.zeros(len(frames),int)
    for left,right in half_blocks:
        centers=(frames>=left)&(frames<right);assigned+=centers
        allowed|=centers[:,None]&(indices>=left)&(indices<right)
    if not np.all(assigned==1):raise ValueError('Centers must belong to exactly one official half')
    allowed&=(indices>=0)&(indices<len(values))
    result=values[np.clip(indices,0,len(values)-1)].copy()
    result[~allowed]=np.r_[np.zeros(values.shape[1]//2),np.ones(values.shape[1]//2)]
    return result.transpose(0,2,1).copy()


def binary_targets(labels,train,validation):
    train=np.asarray(train,int);validation=np.asarray(validation,int)
    if len(np.unique(train))!=len(train) or not np.isin(labels[train],['H','A','D']).all():
        raise ValueError('Only eligible official public targets may train the model')
    if np.any((train>=validation.min()-250)&(train<=validation.max()+250)):raise ValueError('250-frame training exclusion violated')
    target=(labels[train]=='D').astype(np.float32)
    if set(target)!={0.,1.}:raise ValueError('Both live/dead classes required')
    weight=(len(target)/(2*np.bincount(target.astype(int))[target.astype(int)])).astype(np.float32)
    return target,weight


def replace_dead_probability(original,dead,frames):
    original=np.asarray(original,float);dead=np.asarray(dead,float);frames=np.asarray(frames,int)
    if original.shape[1]!=3 or dead.shape!=(len(frames),) or not np.isfinite(dead).all() or ((dead<0)|(dead>1)).any():
        raise ValueError('Invalid D probability')
    result=original.copy();mass=original[frames,:2].sum(axis=1)
    conditional=original[frames,:2]/mass[:,None]
    p=np.clip(dead,1e-6,1-1e-6)
    result[frames,:2]=conditional*(1-p[:,None]);result[frames,2]=p
    return result


def dead_overlay(reference,alternative,mode):
    reference=np.asarray(reference);alternative=np.asarray(alternative)
    if reference.shape!=alternative.shape:raise ValueError('Sequences differ')
    result=reference.copy()
    if mode=='trim':
        use=(reference=='D')&np.isin(alternative,['H','A']);result[use]=alternative[use]
    elif mode=='union':
        use=np.isin(reference,['H','A'])&(alternative=='D');result[use]='D'
    else:raise ValueError('Unknown D-only overlay')
    if not np.array_equal(result=='X',reference=='X'):raise AssertionError('Abstention changed')
    shared=np.isin(result,['H','A'])&np.isin(reference,['H','A'])
    if not np.array_equal(result[shared],reference[shared]):raise AssertionError('Existing ownership switched')
    return result
