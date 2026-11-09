"""Combine bounded prediction- and visibility-anchored stoppage closure."""
import numpy as np
from stoppage_gap_closure import refine as predicted_gap
from observed_gap_stoppage import refine as visibility_gap

VARIANTS={'dual_anchor25':{}}


def refine(reference,heads,raw,broadcast,blocks):
    """Accept either one-second gap anchor within the original unseen flip.

    Both anchors use identical eligibility and unchanged original inputs. Their
    union assigns D at the D-facing side;the remainder continues the opposite
    owner. Observed stationary fragments share their unchanged dead-head rule.
    No fitting,truth,protocol,frame-specific output or iterative chaining.
    """
    first,events=predicted_gap(reference,heads,raw,broadcast,blocks,dead_extension=25)
    second,alternatives=visibility_gap(reference,heads,raw,broadcast,blocks,allowance=25)
    result=first.copy();result[second=='D']='D'
    other={(e['start'],e['end_exclusive']):e for e in alternatives};accepted=[]
    for event in events:
        event=dict(event);alt=other[(event['start'],event['end_exclusive'])]
        if event['kind']=='invisible_gap_closure':
            event.update(kind='dual_anchor_closure',dead_start=min(event['dead_start'],alt['dead_start']),
                dead_end_exclusive=max(event['dead_end_exclusive'],alt['dead_end_exclusive']),
                visibility_anchor=alt['visibility_anchor'],allowance=25)
        accepted.append(event)
    np.testing.assert_array_equal(result=='X',np.asarray(reference)=='X')
    return result,accepted
