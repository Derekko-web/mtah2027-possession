"""Dependency-free equivalent of the frozen eight-connected weighted blobs."""
import numpy as np
from wasb_adapter import inverse_affine


def components(hm,width,height):
    hm=np.asarray(hm,np.float32)
    if hm.shape!=(288,512) or not np.isfinite(hm).all():raise ValueError('Invalid heatmap')
    active=hm>.5;ys,xs=np.where(active);result=[];matrix=inverse_affine(width,height)
    for y,x in zip(ys,xs):
        if not active[y,x]:continue
        active[y,x]=False;todo=[(int(y),int(x))];points=[]
        while todo:
            cy,cx=todo.pop();points.append((cy,cx))
            a,b=max(0,cy-1),min(288,cy+2);c,d=max(0,cx-1),min(512,cx+2)
            yy,xx=np.where(active[a:b,c:d]);active[a:b,c:d]=False
            todo.extend((int(py+a),int(px+c)) for py,px in zip(yy,xx))
        points=np.asarray(sorted(points));yy,xx=points.T;weight=hm[yy,xx];score=weight.sum()
        xy=matrix@np.asarray([(xx*weight).sum()/score,(yy*weight).sum()/score,1.])
        result.append((float(xy[0]),float(xy[1]),float(score)))
    return result


def normalize_warped(image):
    x=np.asarray(image,np.float32)/np.float32(255)
    x=(x-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
    return x.transpose(2,0,1).copy()
