"""Pure matching, ownership probes and prospective DFL continuation rules."""
import hashlib
import numpy as np
from evaluation import possession_metrics as pm
from control_refinement import smooth
from transition_penalty import decode


def native_prefix_lengths(raw_rows,observation_rows):
    if raw_rows!=18935 or observation_rows!=19060:
        raise ValueError('Pinned decoder/observation prefix schema differs')
    return dict(decoder_rows=raw_rows,short_rows=18935,long_rows=19060)


def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    equal=np.array_equal(a,b) if a.dtype.kind in 'OUS' else np.array_equal(a,b,equal_nan=True)
    if not equal:raise AssertionError('Diagnostic replay differs')


def matching(labels,prediction):
    """Official gap-aware greedy matching, including prediction identities."""
    gt,pred=pm.label_codes(labels),pm.label_codes(prediction)
    runs=pm.valid_runs(gt)
    pred=pm.fill_abstentions(pred,runs)
    truth=pm.possession_spells(gt,runs);predicted=pm.possession_spells(pred,runs)
    overlaps=pm.spell_overlaps(truth,predicted)
    gt_flags=np.zeros(len(truth),bool);pred_flags=np.zeros(len(predicted),bool);pairs=[]
    for iou,i,j in overlaps:
        if iou<.5:break
        if gt_flags[i] or pred_flags[j]:continue
        gt_flags[i]=True;pred_flags[j]=True;pairs.append((i,j,float(iou)))
    same(gt_flags,pm.matched_gt_flags(overlaps,len(truth),.5))
    return dict(truth=truth,predicted=predicted,overlaps=overlaps,
        gt_flags=gt_flags,pred_flags=pred_flags,pairs=pairs)


def best_iou(state,gt_id):
    return max((iou for iou,i,_ in state['overlaps'] if i==gt_id),default=0.)


def interval_iou(left,right):
    a,b=left[:2];c,d=right[:2]
    return max(0,min(b,d)-max(a,c))/float(max(b,d)-min(a,c))


def teacher_diagnostic(reference,home_probability,blocks):
    """Fixed smooth5/DP3, separately in reference-live/evaluation segments."""
    reference=np.asarray(reference);p=np.asarray(home_probability,float)
    if len(reference)!=len(p) or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid aligned teacher probabilities')
    output=reference.copy()
    for left,right in blocks:
        live=np.isin(reference[left:right],['H','A'])
        edges=np.flatnonzero(np.diff(np.r_[False,live,False]))
        for begin,end in zip(edges[::2],edges[1::2]):
            start,stop=left+begin,left+end
            probability=np.c_[p[start:stop],1-p[start:stop]]
            output[start:stop]=np.array(['H','A'])[decode(smooth(probability,5))]
    same(output=='D',reference=='D');same(output=='X',reference=='X')
    return output


def conditional_probability(original,new):
    original,new=np.asarray(original,float),np.asarray(new,float)
    if original.shape!=new.shape or original.ndim!=2 or original.shape[1]!=3:
        raise ValueError('Expected matching H/A/D arrays')
    out=original.copy();mass=original[:,:2].sum(axis=1);new_mass=new[:,:2].sum(axis=1)
    conditional=np.divide(new[:,:2],new_mass[:,None],out=np.full((len(new),2),.5),where=new_mass[:,None]>0)
    out[:,:2]=conditional*mass[:,None]
    same(out[:,2],original[:,2])
    return out


def oracle_probability(original,frames,target):
    """Target-aware diagnostic only; never usable as production inference."""
    output=np.asarray(original,float).copy();frames=np.asarray(frames,int);target=np.asarray(target,int)
    if len(frames)!=len(target) or not np.isin(target,[0,1]).all() or len(np.unique(frames))!=len(frames):
        raise ValueError('Oracle requires unique live frames and H/A targets')
    if (frames<0).any() or (frames>=len(output)).any() or not np.isfinite(output[frames]).all():
        raise ValueError('Oracle frames outside finite source domain')
    mass=output[frames,:2].sum(axis=1)
    output[frames,0]=mass*np.where(target==0,.99,.01)
    output[frames,1]=mass*np.where(target==1,.99,.01)
    same(output[:,2],np.asarray(original)[:,2])
    return output


def keep_reference_dead_unknown(reference,alternative):
    reference,alternative=np.asarray(reference),np.asarray(alternative)
    if reference.shape!=alternative.shape:raise ValueError('Overlay shapes differ')
    output=reference.copy();use=np.isin(reference,['H','A'])&np.isin(alternative,['H','A'])
    output[use]=alternative[use]
    same(output=='D',reference=='D');same(output=='X',reference=='X')
    return output


def confidence_summary(labels,reference,home_probability,threshold=.65):
    labels,reference=np.asarray(labels),np.asarray(reference);p=np.asarray(home_probability,float)
    if labels.shape!=reference.shape or p.shape!=labels.shape or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid confidence observations')
    live=np.isin(labels,['H','A']);decision=np.where(p>=.5,'H','A')
    recall={label:float((decision[labels==label]==label).mean()) if (labels==label).any() else None for label in ('H','A')}
    eligible=live&np.isin(reference,['H','A'])
    confident=np.maximum(p,1-p)>=threshold
    disagree=eligible&confident&(decision!=reference)
    correct=int((disagree&(decision==labels)).sum());harm=int((disagree&(reference==labels)).sum())
    if correct+harm!=int(disagree.sum()):raise AssertionError('H/A disagreement partition differs')
    return dict(public_live_frames=int(live.sum()),reference_live_true_live_frames=int(eligible.sum()),
        home_recall=recall['H'],away_recall=recall['A'],
        balanced_accuracy=sum(recall.values())/2 if all(v is not None for v in recall.values()) else None,
        confident_disagreements=correct+harm,confident_corrections=correct,confident_harms=harm,
        correction_precision=correct/(correct+harm) if correct+harm else None)


def oracle_effect(before,after,gt_id=None):
    lost=np.flatnonzero(before['gt_flags']&~after['gt_flags']).tolist()
    new=np.flatnonzero(~before['gt_flags']&after['gt_flags']).tolist()
    spurious_before=int((~before['pred_flags']).sum());spurious_after=int((~after['pred_flags']).sum())
    return dict(newly_matched_gt_ids=new,lost_matched_gt_ids=lost,
        spurious_before=spurious_before,spurious_after=spurious_after,
        target_gt_matched=bool(after['gt_flags'][gt_id]) if gt_id is not None else None,
        retained_all_previously_matched=not lost,no_extra_spurious=spurious_after<=spurious_before)


def continuation(protocols,rule):
    """Common fixed GT-ID across all protocols; deterministic slot4 then slot0."""
    common_checks={};eligible={}
    for name,data in protocols.items():
        m=data['confidence']
        common_checks[name]=dict(balanced_accuracy=m['balanced_accuracy'] is not None and m['balanced_accuracy']>=rule['minimum_balanced_accuracy'],
            confident_corrections=m['confident_corrections']>=rule['minimum_confident_corrections'],
            correction_precision=m['correction_precision'] is not None and m['correction_precision']>=rule['minimum_correction_precision'])
    for slot in (4,0):
        case_sets=[]
        for name,data in protocols.items():
            cases={case['gt_id'] for case in data['truth_cases'] if not case['reference_matched'] and case['frames']<=125
                and case['teacher_matched'] and case['teacher_correct_probability']>=rule['minimum_case_probability']
                and case['oracle'][str(slot)]['target_gt_matched'] and case['oracle'][str(slot)]['retained_all_previously_matched']
                and case['oracle'][str(slot)]['no_extra_spurious']}
            case_sets.append(cases)
        common=set.intersection(*case_sets) if case_sets else set()
        eligible[str(slot)]=dict(common_recoverable_gt_ids=sorted(common),common_cases=len(common),
            passed=all(all(v.values()) for v in common_checks.values()) and len(common)>=rule['minimum_common_cases'])
    selected=next((slot for slot in (4,0) if eligible[str(slot)]['passed']),None)
    return dict(passed=selected is not None,selected_slot=selected,confidence_checks=common_checks,slots=eligible,
        diagnostic_only=True,not_achievable_score_estimate=True)


def array_digest(value):
    array=np.ascontiguousarray(value)
    return hashlib.sha256(array.dtype.str.encode()+str(array.shape).encode()+array.tobytes()).hexdigest()
