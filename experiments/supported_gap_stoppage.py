"""Require a nontrivial live continuation before closing an unseen flip."""
import numpy as np
from dual_anchor_stoppage import refine as propose

VARIANTS={'supported_gap25':{}}


def refine(reference,heads,raw,broadcast,blocks):
    """Decline an unseen closure leaving <8frames of inferred live support.

    Eight frames is the existing minimum live-control span. It is applied to
    the unobserved continuation portion itself,rather than borrowing support
    from the preceding/following large live span. Observed stoppage decisions
    retain their original rule. All proposals use the unchanged reference.
    """
    reference=np.asarray(reference)
    _,proposals=propose(reference,heads,raw,broadcast,blocks)
    result=reference.copy();events=[]
    for event in proposals:
        start,end=event['start'],event['end_exclusive']
        if event['kind']=='dual_anchor_closure':
            dead_start,dead_end=event['dead_start'],event['dead_end_exclusive']
            remaining=(end-start)-(dead_end-dead_start)
            if remaining<8:continue
            result[start:end]=event['replacement'];result[dead_start:dead_end]='D'
            event=dict(event,live_continuation_frames=remaining)
        else:result[start:end]=event['replacement']
        events.append(event)
    np.testing.assert_array_equal(result=='X',reference=='X')
    np.testing.assert_array_equal(result[reference=='D'],reference[reference=='D'])
    return result,events
