"""Deterministic unit tests for FeatureConstraints — no ART, no network.

Run: python -m unittest -v constraints.test_constraints  (from the harness dir)
"""

import unittest

import numpy as np

from constraints.constraints import FeatureConstraints, FeatureSpec


class TestFeatureConstraints(unittest.TestCase):
    def setUp(self):
        # col0 continuous [0,10], col1 ordinal {0,1,2}, col2 immutable
        self.fc = FeatureConstraints([
            FeatureSpec("amount", "continuous", lo=0.0, hi=10.0),
            FeatureSpec("grade", "ordinal", legal_values=[0.0, 1.0, 2.0]),
            FeatureSpec("age", "immutable"),
        ])

    def test_continuous_clipped_to_range(self):
        proj, changed, _ = self.fc.project([5.0, 1.0, 30.0], [99.0, 1.0, 30.0])
        self.assertEqual(proj[0], 10.0)
        self.assertTrue(changed[0])

    def test_continuous_lower_clip(self):
        proj, _, _ = self.fc.project([5.0, 1.0, 30.0], [-4.0, 1.0, 30.0])
        self.assertEqual(proj[0], 0.0)

    def test_ordinal_snaps_to_nearest_legal(self):
        proj, changed, _ = self.fc.project([5.0, 1.0, 30.0], [5.0, 1.7, 30.0])
        self.assertEqual(proj[1], 2.0)  # 1.7 nearest to 2.0
        self.assertTrue(changed[1])

    def test_immutable_restored(self):
        proj, changed, _ = self.fc.project([5.0, 1.0, 30.0], [5.0, 1.0, 99.0])
        self.assertEqual(proj[2], 30.0)
        self.assertFalse(changed[2])

    def test_noop_attack_has_zero_distance(self):
        proj, changed, dist = self.fc.project([5.0, 1.0, 30.0], [5.0, 1.0, 30.0])
        self.assertEqual(dist, 0.0)
        self.assertFalse(changed.any())

    def test_distance_is_l2_over_continuous(self):
        # only col0 moves by 3.0 -> l2 == 3.0
        _, _, dist = self.fc.project([5.0, 1.0, 30.0], [8.0, 1.0, 30.0])
        self.assertAlmostEqual(dist, 3.0)

    def test_fit_bounds_fills_continuous_range(self):
        fc = FeatureConstraints([FeatureSpec("x", "continuous")])
        fc.fit_bounds(np.array([[1.0], [4.0], [9.0]]))
        proj, _, _ = fc.project([4.0], [100.0])
        self.assertEqual(proj[0], 9.0)

    def test_batch_matches_row(self):
        X = np.array([[5.0, 1.0, 30.0], [2.0, 0.0, 40.0]])
        Xadv = np.array([[99.0, 1.7, 30.0], [-1.0, 0.0, 99.0]])
        Xp, changed, dist = self.fc.project_batch(X, Xadv)
        self.assertEqual(Xp[0, 0], 10.0)
        self.assertEqual(Xp[0, 1], 2.0)
        self.assertEqual(Xp[1, 0], 0.0)
        self.assertEqual(Xp[1, 2], 40.0)  # immutable restored
        self.assertEqual(dist.shape, (2,))

    def test_immutable_columns_reported(self):
        self.assertEqual(self.fc.immutable_columns(), [2])


if __name__ == "__main__":
    unittest.main()
