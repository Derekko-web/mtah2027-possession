"""Check metric geometry, team symmetry and bounded temporal feature context."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import rich_features as rf


class RichFeatureTests(unittest.TestCase):
    def build(self, last_ball="0.5,0.5,NaN"):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "tracking.txt"
            lines = [f"{i+1}:0.51,0.5,1;0.6,0.5,2:0.5,0.5,NaN\n" for i in range(150)]
            lines[-1] = f"150:0.51,0.5,1;0.6,0.5,2:{last_ball}\n"
            raw.write_text("".join(lines))
            with patch.object(rf, "require", return_value=["metadata", str(raw)]), \
                 patch.object(rf, "team_of_each_track", return_value=np.array([0, 1])), \
                 patch.object(rf, "load_tags", return_value=[]):
                return rf.build_features(150)

    def test_distances_are_in_metres(self):
        features = self.build()
        self.assertAlmostEqual(features["h__distance_1"].iloc[0], 1.053, places=4)
        self.assertAlmostEqual(features["a__distance_1"].iloc[0], 10.53, places=4)
        self.assertAlmostEqual(features["ball_touchline_distance"].iloc[0], 34)

    def test_team_swap_is_an_involution(self):
        features = self.build()
        swapped = rf.swap_teams(features)
        pd.testing.assert_frame_equal(rf.swap_teams(swapped), features)
        np.testing.assert_array_equal(swapped["h__distance_1"], features["a__distance_1"])
        np.testing.assert_array_equal(swapped.ball_seen, features.ball_seen)

    def test_future_input_beyond_five_seconds_cannot_change_features(self):
        original = self.build()
        changed = self.build("0.9,0.5,NaN")
        # Last altered frame is 149; frames 0..23 are more than 125 frames earlier.
        pd.testing.assert_frame_equal(original.iloc[:24], changed.iloc[:24])
        self.assertNotEqual(original["a__distance_1"].iloc[-1], changed["a__distance_1"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
