"""Keep selected ownership and only remove D under the new motion alternative."""

import numpy as np

from broadcast_refinement import decode_broadcast_closure
from pair_candidate import decode_pair_candidate


def decode_motion_dead_trim(current,previous,fast,heads,bc_probability,motion_probability,bc_features,info,raw,blocks):
    reference = decode_broadcast_closure(current,previous,fast,heads,bc_probability,bc_features,raw,blocks)
    alternative = decode_pair_candidate(current,previous,fast,heads,bc_probability,motion_probability,bc_features,info,raw,blocks,"broadcast_motion","replace_current")
    result = reference.copy()
    use = (reference == "D")&np.isin(alternative,["H","A"])
    result[use] = alternative[use]
    if not np.array_equal(result[reference != "D"],reference[reference != "D"]): raise AssertionError("Dead trim changed ownership")
    if ((result == "D")&(reference != "D")).any(): raise AssertionError("Dead trim addedD")
    return result
