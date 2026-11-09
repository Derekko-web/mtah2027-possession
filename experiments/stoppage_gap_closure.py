"""Close an unseen owner-flip gap with bounded adjacent stoppage growth."""
import numpy as np
from spurious_stoppage_fragments import refine as propose

VARIANTS={
    'gap13':dict(dead_extension=13),
    'gap25':dict(dead_extension=25),
}


def refine(reference,heads,raw,broadcast,blocks,dead_extension=25):
    """Retain observed suppression;split an unseen flip at the D-facing edge.

    A wholly unobserved short flip between opposite live control and D is
    continued as that live owner,except up to13/25frames next to the original
    D boundary are assigned to D. This one-pass bounded allowance cannot cross
    the original flip,decode block,or any observed frame. The same rule applies
    at a restart or a stoppage and to either team. No target inputs.
    """
    if dead_extension not in (13,25):raise ValueError('Unsupported dead extension')
    reference=np.asarray(reference)
    result,events=propose(reference,heads,raw,broadcast,blocks,invisible=True,stoppage=True)
    accepted=[]
    for event in events:
        event=dict(event)
        if event['kind']=='invisible_continuation':
            start,end=event['start'],event['end_exclusive'];size=min(dead_extension,end-start)
            if start>0 and reference[start-1]=='D':a,b=start,start+size
            elif end<len(reference) and reference[end]=='D':a,b=end-size,end
            else:raise AssertionError('Unseen proposal lost original D boundary')
            result[a:b]='D'
            event.update(kind='invisible_gap_closure',dead_start=a,dead_end_exclusive=b,dead_extension=dead_extension)
        accepted.append(event)
    np.testing.assert_array_equal(result=='X',reference=='X')
    np.testing.assert_array_equal(result[reference=='D'],reference[reference=='D'])
    return result,accepted
