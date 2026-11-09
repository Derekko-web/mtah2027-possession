"""Two-camera contact recovery restricted to newly observed broadcast intervals."""
import numpy as np


def recover(reference, fast, raw, motion, broadcast, old_mapped, blocks, stable_frames=13):
    """Use bilateral contact evidence where the old camera branch was missing.

    New foreign control requires >=3tactical contacts with fast majority and
    >=3broadcast contacts in the same13-frame neighborhood. Only previously
    unmapped, newly observed intervals inside >250-frame reference live runs
    are eligible. Hold until13or25stable old-owner control frames, <=125frames.
    All proposals read unchanged inputs. D/X and prior short islands stay exact.
    """
    reference=np.asarray(reference);fast=np.asarray(fast);old_mapped=np.asarray(old_mapped,bool)
    n=len(reference)
    if (reference.ndim!=1 or fast.shape!=(n,2) or old_mapped.shape!=(n,)
            or any(len(x)!=n for x in (raw,motion,broadcast)) or stable_frames not in (13,25)
            or not np.isin(reference,['H','A','D','X']).all()):
        raise ValueError('Aligned reference, probabilities, observations and camera masks required')
    distances=raw[['h__distance_1','a__distance_1']].to_numpy()
    bc_distances=broadcast[['h__bc_same_distance_1','a__bc_same_distance_1']].to_numpy()
    kernel=motion[['h__tc_kernel','a__tc_kernel']].to_numpy()
    tactical_contact=(distances<2)&(distances[:,::-1]-distances>.5)
    new_observed=(broadcast.bc_matched.to_numpy()==1)&(broadcast.bc_same_ball_seen.to_numpy()==1)&np.isfinite(bc_distances).all(1)
    bc_contact=(bc_distances<2)&(bc_distances[:,::-1]-bc_distances>.5)&new_observed[:,None]&~old_mapped[:,None]
    result=reference.copy();events=[];covered=np.zeros(n,bool)
    for left,right in blocks:
        if not 0<=left<right<=n or covered[left:right].any():raise ValueError('Invalid or overlapping decode blocks')
        covered[left:right]=True
        if not np.isfinite(fast[left:right]).all() or (fast[left:right]<0).any() or not np.allclose(fast[left:right].sum(1),1):
            raise ValueError('Invalid or incomplete fast probability')
        values=reference[left:right]
        edges=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)]
        for a,b in zip(edges[:-1],edges[1:]):
            team=values[a]
            if team not in ('H','A') or b-a<=250:continue
            start,end=left+int(a),left+int(b);own=0 if team=='H' else 1;other=1-own
            seeds=(tactical_contact[start:end,other]&(fast[start:end,other]>.5)
                   &(kernel[start:end,own]<.5)&~old_mapped[start:end]&new_observed[start:end])
            indices=np.flatnonzero(seeds)
            if not len(indices):continue
            groups=np.split(indices,np.flatnonzero(np.diff(indices)>13)+1)
            for group in groups:
                dense=[i for i in range(len(group)-2) if group[i+2]-group[i]<13]
                if not dense:continue
                onset=start+int(group[dense[0]])
                if onset-start<13:continue
                contact_left=max(start,onset-13);contact_right=min(end,onset+13)
                contacts=int(bc_contact[contact_left:contact_right,other].sum())
                if contacts<3:continue
                stop=min(end-13,onset+125);run=0;finish=None
                stable=(kernel[onset:stop,own]>=.5)&(fast[onset:stop,own]>=.65)
                for offset,value in enumerate(stable):
                    run=run+1 if value else 0
                    if run>=stable_frames:finish=onset+offset+1;break
                if finish is None or not 8<=finish-onset<=125:continue
                if old_mapped[onset:finish].any() or not new_observed[onset:finish].all():continue
                if bc_contact[max(onset,finish-stable_frames):finish,own].sum()<3:continue
                label='H' if other==0 else 'A';result[onset:finish]=label
                events.append(dict(start=onset,end_exclusive=finish,team=label,
                    tactical_trigger_contacts=int(seeds[group[dense[0]]:min(end-start,group[dense[0]]+13)].sum()),
                    broadcast_trigger_contacts=contacts,release_confirmation_frames=stable_frames))
    for state in ('D','X'):np.testing.assert_array_equal(result==state,reference==state)
    return result,events
