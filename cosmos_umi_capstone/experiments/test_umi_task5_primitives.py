import unittest

import numpy as np

from umi_task5_primitives import (
    ALPHAS,
    build_decoder_replay_plan,
    build_task5_call_plan,
    freeze_task5_directions,
    heldout_prediction_metrics,
    linear_formula_calibration,
    pair_additivity_metrics,
)


class Task5DirectionTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones((1, 2, 1, 2, 2), dtype=bool)
        bank = np.zeros((3,) + self.mask.shape, dtype=np.float32)
        bank[0, 0, 0, 0, 0] = 1
        bank[1, 0, 0, 0, 1] = 1
        bank[2, 0, 1, 0, 0] = 1
        for item in bank:
            item /= np.sqrt(np.mean(item[self.mask].astype(np.float64) ** 2))
        self.bank = bank

    def test_freeze_preserves_v0_v1_v2_and_combo_coefficients(self):
        frozen = freeze_task5_directions(self.bank, self.mask)
        self.assertEqual(tuple(frozen["ids"]), ("v0", "v1", "v2", "u01", "u12"))
        self.assertTrue(np.array_equal(frozen["directions"]["v0"], self.bank[0]))
        self.assertGreater(frozen["c01"], 0.0)
        self.assertGreater(frozen["c12"], 0.0)
        self.assertAlmostEqual(np.sqrt(np.mean(frozen["directions"]["u01"][self.mask] ** 2)), 1.0, places=6)
        self.assertAlmostEqual(np.sqrt(np.mean(frozen["directions"]["u12"][self.mask] ** 2)), 1.0, places=6)
        self.assertEqual(frozen["gram"].shape, (3, 3))

    def test_task5_full_generation_plan_is_exactly_32_calls(self):
        plan = build_task5_call_plan()
        self.assertEqual(ALPHAS, (1e-3, 3e-3, 1e-2))
        self.assertEqual(len(plan), 32)
        self.assertEqual(plan[0]["sample_id"], "baseline_pre")
        self.assertEqual(plan[-1]["sample_id"], "baseline_post")
        self.assertEqual(len({row["sample_id"] for row in plan}), 32)
        self.assertEqual(sum(row["kind"] == "perturbation" for row in plan), 30)

    def test_decoder_replay_plan_uses_task5_v0_coverage_only(self):
        plan = build_decoder_replay_plan()
        self.assertEqual(len(plan), 8)
        self.assertEqual(plan[0]["sample_id"], "baseline_pre")
        self.assertEqual(plan[-1]["sample_id"], "baseline_post")
        self.assertEqual(sum(row["kind"] == "perturbation" for row in plan), 6)
        self.assertTrue(all(row.get("direction_id", "v0") == "v0" for row in plan))


class Task5FormulaTests(unittest.TestCase):
    def setUp(self):
        self.v0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.v1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        self.v2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.jacobian = np.array([[2.0, -1.0, 0.5], [0.0, 3.0, 1.0]], dtype=np.float64)
        self.c01 = np.sqrt(np.mean((self.v0 + self.v1) ** 2))
        self.c12 = np.sqrt(np.mean((self.v1 + self.v2) ** 2))

    def test_linear_calibration_has_zero_additivity_and_holdout_residuals(self):
        result = linear_formula_calibration()
        self.assertEqual(result["status"], "PASS_EXACT")
        self.assertEqual(result["max_additivity_rms"], 0.0)
        self.assertEqual(result["max_prediction_rms"], 0.0)

    def test_additivity_keeps_empirical_combo_coefficient(self):
        d0 = self.jacobian @ self.v0
        d1 = self.jacobian @ self.v1
        u01 = (self.v0 + self.v1) / self.c01
        du = self.jacobian @ u01
        result = pair_additivity_metrics(du, d0, d1, self.c01)
        self.assertAlmostEqual(result["absolute_rms"], 0.0, places=12)
        self.assertAlmostEqual(result["relative_error"], 0.0, places=12)
        self.assertAlmostEqual(result["cosine"], 1.0, places=12)

    def test_holdout_prediction_uses_small_amplitude_derivatives_without_refitting(self):
        baseline = np.array([1.0, -2.0], dtype=np.float64)
        h0 = 0.01
        g = {"v0": self.jacobian @ self.v0, "v1": self.jacobian @ self.v1, "v2": self.jacobian @ self.v2}
        directions = {"v0": self.v0, "v1": self.v1, "v2": self.v2,
                      "u01": (self.v0 + self.v1) / self.c01,
                      "u12": (self.v1 + self.v2) / self.c12}
        actual = {}
        for direction_id, direction in directions.items():
            derivative = self.jacobian @ direction
            for alpha in (0.01, 0.03, 0.1):
                for sign in (1, -1):
                    actual[(direction_id, alpha, sign)] = baseline + sign * alpha * derivative
        result = heldout_prediction_metrics(actual, baseline, g, {"c01": self.c01, "c12": self.c12},
                                            h0=h0, holdout_alphas=(0.03, 0.1), combo_alphas=(0.01, 0.03, 0.1))
        self.assertTrue(all(row["status"] == "PASS" for row in result))
        self.assertTrue(all(row["relative_error"] == 0.0 for row in result))


if __name__ == "__main__":
    unittest.main()
