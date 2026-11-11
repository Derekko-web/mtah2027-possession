"""Restricted unrotated WASB affine/normalization and connected-component readout.

Geometry follows WASB's MIT image.py. Integer bilinear sampling follows OpenCV's
INTER_LINEAR 5-bit interpolation table and 10-bit affine coordinate increments;
this adapter does not claim a measured OpenCV binary parity on this machine.
"""
import numpy as np


def inverse_affine(width,height):
    scale=max(width,height)/512
    return np.array([[scale,0,width/2-256*scale],[0,scale,height/2-144*scale]],np.float64)


def warp(image):
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('RGB uint8 image required')
    h,w,_=image.shape;m=inverse_affine(w,h)
    xq=(np.rint(np.arange(512)*m[0,0]*1024).astype(np.int64)+int(np.rint(m[0,2]*1024))+16)>>5
    yq=(np.rint((np.arange(288)*m[1,1]+m[1,2])*1024).astype(np.int64)+16)>>5
    x0=xq>>5;y0=yq>>5;fx=xq&31;fy=yq&31
    total=np.zeros((288,512,3),np.int64)
    for dx,wx in ((0,32-fx),(1,fx)):
        for dy,wy in ((0,32-fy),(1,fy)):
            xs=x0+dx;ys=y0+dy;valid=(ys[:,None]>=0)&(ys[:,None]<h)&(xs[None]>=0)&(xs[None]<w)
            pixels=image[np.clip(ys,0,h-1)[:,None],np.clip(xs,0,w-1)[None]].astype(np.int64)
            total+=pixels*(wy[:,None]*wx[None]*valid)[:,:,None]
    return np.clip((total+512)>>10,0,255).astype(np.uint8)


def normalize(image):
    value=warp(image).astype(np.float32)/np.float32(255)
    value=(value-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
    return value.transpose(2,0,1).copy()


def blobs(heatmap,width,height):
    from scipy.ndimage import label
    hm=np.asarray(heatmap,np.float32)
    if hm.shape!=(288,512) or not np.isfinite(hm).all():raise ValueError('Invalid WASB center heatmap')
    labels,count=label(hm>.5,structure=np.ones((3,3),int));matrix=inverse_affine(width,height);result=[]
    for i in range(1,count+1):
        ys,xs=np.where(labels==i);weights=hm[ys,xs];score=weights.sum()
        point=np.asarray([(xs*weights).sum()/score,(ys*weights).sum()/score,1.])
        xy=matrix@point;result.append((float(xy[0]),float(xy[1]),float(score)))
    return result
