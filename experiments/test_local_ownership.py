import unittest
import numpy as np
from local_ownership import parse_ownership, allowed_frame, sample_centers, local_json, NoRedirect


class LocalOwnershipTests(unittest.TestCase):
    def test_exact_parser_abstains_on_narrative_and_ambiguity(self):
        self.assertEqual(parse_ownership(' HOME. '), 'H')
        self.assertEqual(parse_ownership('away!'), 'A')
        self.assertEqual(parse_ownership('stopped'), 'D')
        for value in ('home or away', 'The home team', 'home\naway', '', None, 'unclear'):
            self.assertEqual(parse_ownership(value), 'X')

    def test_routes_and_redirects_are_rejected_before_network(self):
        for route in ('https://example.com', '//example.com', '/api/pull', '/api/generate?host=remote'):
            with self.assertRaises(ValueError):
                local_json(route)
        with self.assertRaises(ValueError):
            NoRedirect().redirect_request(None, None, 302, None, None, 'https://example.com')

    def test_schedule_excludes_actual_triplets_and_protected_context(self):
        valid = np.ones(20000, bool)
        reserved = [dict(start=11300, end=11320)]
        held = np.array([12060])
        centers = sample_centers(valid, [('half', 10000, 19000)], reserved, held)
        self.assertNotIn(11310, centers)
        self.assertNotIn(11560, centers)
        self.assertNotIn(12060, centers)
        for center in centers:
            self.assertTrue(all(allowed_frame(f, reserved, held) for f in (center-12, center, center+12)))
        self.assertFalse(allowed_frame(11185, [], [], exclude_public=True))


if __name__ == '__main__':
    unittest.main()
