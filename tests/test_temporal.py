import unittest
import numpy as np

from analytics.temporal import BallTrajectory
from analytics.pitch import ViewTransformer


def candidate(x, y, confidence):
    return {"centre": (x, y), "confidence": confidence}


class BallTrajectoryTests(unittest.TestCase):
    def test_prefers_continuity_over_distant_high_confidence_false_positive(self):
        trajectory = BallTrajectory(max_gap_frames=2, max_speed_fraction_s=.55)
        shape = (720, 1280, 3)
        trajectory.update([candidate(100, 100, .8)], 25, shape)
        observation = trajectory.update([candidate(112, 100, .45), candidate(900, 500, .99)], 25, shape)
        self.assertEqual(observation.point, (112.0, 100.0))

    def test_gap_fill_is_bounded_and_labelled(self):
        trajectory = BallTrajectory(max_gap_frames=2)
        shape = (720, 1280, 3)
        trajectory.update([candidate(100, 100, .8)], 25, shape)
        trajectory.update([candidate(110, 100, .8)], 25, shape)
        self.assertEqual(trajectory.update([], 25, shape).source, "predicted_gap_fill")
        self.assertEqual(trajectory.update([], 25, shape).source, "predicted_gap_fill")
        self.assertIsNone(trajectory.update([], 25, shape))

    def test_reacquires_after_expired_gap(self):
        trajectory = BallTrajectory(max_gap_frames=0)
        shape = (720, 1280, 3)
        trajectory.update([candidate(10, 10, .5)], 25, shape)
        self.assertIsNone(trajectory.update([], 25, shape))
        self.assertEqual(trajectory.update([candidate(900, 500, .6)], 25, shape).source, "detected")


class HomographyTests(unittest.TestCase):
    def test_round_trip(self):
        source = np.asarray(((0, 0), (100, 0), (100, 50), (0, 50)), np.float32)
        target = np.asarray(((0, 0), (12000, 0), (12000, 7000), (0, 7000)), np.float32)
        transformer = ViewTransformer(source, target)
        points = np.asarray(((20, 10), (75, 40)), np.float32)
        recovered = transformer.inverse_points(transformer.transform_points(points))
        np.testing.assert_allclose(points, recovered, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
