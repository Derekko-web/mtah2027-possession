"""Contact-triggered control continuation inside long reference live spells."""
import numpy as np


def repair(reference, fast, raw, motion, blocks, stable_frames=13):
    """Carry brief supported turnovers until kinematic control is reconfirmed.

    Only reference spells longer than250frames are eligible. A new owner must
    have at least3 exclusive observed contacts supported by fast probability
    >=.65 within13frames while reference-owner motion kernel is below.5.
    Release requires13or25 consecutive frames of reference-owner kernel>=.5
    and fast probability>=.65, within125frames. No fixed duration is imposed.
    """
    reference=np.asarray(reference);fast=np.asarray(fast)
    n=len(reference)
    if (reference.ndim!=1 or fast.shape!=(n,2) or len(raw)!=n or len(motion)!=n
            or stable_frames not in (13,25) or not np.isin(reference,['H','A','D','X']).all()):
        raise ValueError('Invalid control-continuation inputs')
    distance=raw[['h__distance_1','a__distance_1']].to_numpy()
    kernel=motion[['h__tc_kernel','a__tc_kernel']].to_numpy()
    contact=(distance<2)&(distance[:,::-1]-distance>.5)
    result=reference.copy();covered=np.zeros(n,bool);events=[]
    for left,right in blocks:
        if not 0<=left<right<=n or covered[left:right].any():raise ValueError('Invalid or overlapping decode blocks')
        covered[left:right]=True
        if not np.isfinite(fast[left:right]).all() or not np.allclose(fast[left:right].sum(1),1):
            raise ValueError('Missing fast ownership probabilities')
        local=reference[left:right]
        edges=np.r_[0,np.flatnonzero(local[1:]!=local[:-1])+1,len(local)]
        for a,b in zip(edges[:-1],edges[1:]):
            team=local[a]
            if team not in ('H','A') or b-a<=250:continue
            start,end=left+int(a),left+int(b);k=0 if team=='H' else 1;other=1-k
            seed=contact[start:end,other]&(fast[start:end,other]>=.65)&(kernel[start:end,k]<.5)
            indices=np.flatnonzero(seed)
            if not len(indices):continue
            groups=np.split(indices,np.flatnonzero(np.diff(indices)>13)+1)
            for group in groups:
                if len(group)<3:continue
                dense=[i for i in range(len(group)-2) if group[i+2]-group[i]<13]
                if not dense:continue
                onset=start+int(group[dense[0]])
                if onset-start<13:continue
                stop=min(end-13,onset+125)
                stable=(kernel[onset:stop,k]>=.5)&(fast[onset:stop,k]>=.65)
                run=0;finish=None
                for offset,value in enumerate(stable):
                    run=run+1 if value else 0
                    if run>=stable_frames:
                        finish=onset+offset+1;break
                if finish is None or not 8<=finish-onset<=125:continue
                label='H' if other==0 else 'A'
                result[onset:finish]=label
                events.append(dict(start=onset,end_exclusive=finish,team=label,
                                   trigger_contacts=int(np.sum(seed[onset-start:min(end-start,onset-start+13)]))))
    for state in ('D','X'):np.testing.assert_array_equal(result==state,reference==state)
    return result,events
