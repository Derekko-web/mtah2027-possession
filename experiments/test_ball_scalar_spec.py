"""Guard unit conversion and third-field meaning using synthetic metadata."""

from pathlib import Path
import tempfile
import unittest

from inspect_ball_scalar import ball_scalar_spec


class BallScalarSpecTests(unittest.TestCase):
    def test_declared_speed_units_and_wrong_field_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'metadata.xml'
            template = '<Root><Channel id="s"><Unit>{unit}</Unit></Channel><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="{last}"/></Root>'
            path.write_text(template.format(unit=' Km/h ', last='s'))
            self.assertAlmostEqual(ball_scalar_spec(path)['raw_to_declared_meters_per_second'], 1/3.6)
            path.write_text(template.format(unit='m/s', last='s'))
            self.assertEqual(ball_scalar_spec(path)['raw_to_declared_meters_per_second'], 1.)
            path.write_text(template.format(unit='m/s', last='z'))
            with self.assertRaises(ValueError):
                ball_scalar_spec(path)
            path.write_text(template.format(unit='unknown', last='s'))
            with self.assertRaises(ValueError):
                ball_scalar_spec(path)


if __name__ == '__main__':
    unittest.main()
