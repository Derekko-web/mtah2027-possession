"""D-only changes must never replace a selected ownership or abstention decision."""

import unittest
from unittest.mock import patch

import numpy as np

from motion_dead_trim import decode_motion_dead_trim


class MotionDeadTrimTests(unittest.TestCase):
    def test_only_reference_dead_can_become_alternative_live(self):
        reference = np.array(["H","A","D","D","X","D","D"])
        alternative = np.array(["A","D","H","D","H","A","X"])
        with patch("motion_dead_trim.decode_broadcast_closure",return_value=reference),patch("motion_dead_trim.decode_pair_candidate",return_value=alternative):
            result = decode_motion_dead_trim(*([None]*10))
        np.testing.assert_array_equal(result,["H","A","H","D","X","A","D"])
        np.testing.assert_array_equal(reference,["H","A","D","D","X","D","D"])


if __name__ == "__main__": unittest.main()
