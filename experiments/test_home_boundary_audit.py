"""Compare arrival inference against an independently segmented implementation."""
import unittest
import numpy as np
import pandas as pd
from audit_home_boundaries import independent_refine
from source_aware_arrivals import refine as initial, VARIANTS as INITIAL
from leading_gap_arrivals import refine as guarded, VARIANTS as GUARDED


class IndependentArrivalTests(unittest.TestCase):
    def test_randomized_sources_gaps_teams_blocks_and_ties(self):
        rng = np.random.default_rng(20261107)
        changed = 0
        for case in range(100):
            n = 800
            reference = np.full(n, 'X', dtype='<U1')
            blocks = [(20, 360), (400, 780)]
            probability = rng.uniform(.01, .99, n)
            owner = np.column_stack((probability, 1-probability))
            if case % 3 == 0:
                owner = owner.astype(np.float32)
            tactical = rng.uniform(.1, 6, (n, 2))
            camera = rng.uniform(.1, 6, (n, 2))
            seen = np.ones(n)
            for block, (left, right) in enumerate(blocks):
                k = (case + block) % 2
                team, opponent = ('H', 'A')[k], ('A', 'H')[k]
                length = int(rng.integers(8, 251))
                start, end = left + 24, left + 24 + length
                reference[left:start] = opponent
                reference[start:end] = team
                reference[end:right] = 'D'
                first = int(rng.integers(0, max(1, length-3)))
                camera[start:end, k] = 1.
                camera[start:end, 1-k] = 4.
                camera[start:start+first, k] = 4.
                camera[start:start+first, 1-k] = 1.
                tactical[start:end, k] = 4.
                tactical[start:end, 1-k] = 1.
                prefix = int(rng.integers(1, 27))
                owner[start:end, k] = .85
                owner[start:end, 1-k] = .15
                owner[start:min(end, start+prefix), k] = .2
                owner[start:min(end, start+prefix), 1-k] = .8
                mode = (case + block) % 8
                if mode == 1:
                    seen[start:min(end, start+int(rng.integers(1, 14)))] = 0
                elif mode == 2:
                    seen[max(start, end-4):end] = 0
                elif mode == 3:
                    seen[start:min(end, start+30)] = 0
                elif mode == 4:
                    seen[start:end] = 0
                elif mode == 5:
                    camera[start:min(end, start+7)] = np.nan
                elif mode == 6:
                    seen[start:min(end, start+5)] = 0
                    camera[start:end, k] = 4.  # leading gap without owner control
                elif mode == 7:
                    seen[start:min(end, start+5)] = 0
                    owner[start:end] = .5  # exact zero-cost tie must stay put
            raw = pd.DataFrame(tactical, columns=['h__distance_1', 'a__distance_1'])
            bc = pd.DataFrame(camera, columns=['h__bc_same_distance_1', 'a__bc_same_distance_1'])
            bc['bc_matched'] = seen
            bc['bc_same_ball_seen'] = 1.
            original = reference.copy()
            for method, variants, leading in ((initial, INITIAL, False), (guarded, GUARDED, True)):
                for name, params in variants.items():
                    with self.subTest(case=case, variant=name):
                        actual, _ = method(reference, owner, raw, bc, blocks, **params)
                        expected = independent_refine(reference, owner, raw, bc, blocks,
                            params['camera'], params['posterior_radius'], leading)
                        np.testing.assert_array_equal(actual, expected)
                        np.testing.assert_array_equal(reference, original)
                        changed += bool(np.any(actual != original))
        self.assertGreater(changed, 100)


if __name__ == '__main__':
    unittest.main()
