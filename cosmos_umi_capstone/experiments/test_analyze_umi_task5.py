"""Synthetic linear-map contracts for Task 5 analysis (no model substitute)."""
from __future__ import annotations

import unittest
import numpy as np


class Task5AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import analyze_umi_task5 as api
            import umi_task5_primitives as primitives
        except ModuleNotFoundError:
            api, primitives = None, None
        cls.api, cls.primitives = api, primitives

    def test_exact_linear_fixture_passes_direction_additivity_and_holdout(self):
        self.assertIsNotNone(self.api, "Task 5 analysis implementation is missing")
        carrier_shape = (1, 1, 3, 2, 2)
        mask = np.zeros(carrier_shape, dtype=bool)
        mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier_shape, dtype=np.float32)
        bank[0, :, :, 0, 0, 0] = 2.0
        bank[1, :, :, 0, 0, 1] = 2.0
        bank[2, :, :, 0, 1, 0] = 2.0
        frozen = self.primitives.freeze_task5_directions(bank, mask)
        directions = frozen["directions"]
        zbar = np.ones(carrier_shape, np.float32)
        y0 = np.zeros((1, 1, 2, 2, 2), np.float32)
        # A deterministic linear output map with nonzero responses for all five directions.
        weights = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 1, 2, 2)
        def response(direction):
            scalar = float(np.sum(direction[:, :, 0] * weights[:, :, 0]))
            return np.full_like(y0, scalar)
        records = {"baseline_pre": {"predicted_latent": y0, "decoded_final": np.zeros((3, 2, 2), np.float32)},
                   "baseline_post": {"predicted_latent": y0.copy(), "decoded_final": np.zeros((3, 2, 2), np.float32)}}
        for direction_id, direction in directions.items():
            for ordinal, alpha in enumerate((1e-3, 3e-3, 1e-2)):
                h = alpha
                for sign, label in ((1, "plus"), (-1, "minus")):
                    delta = sign * h * direction
                    records[f"{direction_id}_alpha_{ordinal:02d}_{label}"] = {
                        "predicted_latent": y0 + sign * h * response(direction),
                        "decoded_final": np.full((3, 2, 2), sign * h * float(response(direction).flat[0]), np.float32),
                        "actual_delta_fp32": delta, "target_delta_fp32": delta.copy(), "direction": direction,
                        "mask": mask, "z_bar": zbar,
                    }
        result = self.api.analyze_task5_records(records, plan_detail={"s_z": 1.0, "combination_coefficients": {
            "c01": frozen["c01"], "c12": frozen["c12"]}})
        self.assertTrue(all(row["candidate_status"] == "PASS" for row in result["fits"]))
        self.assertTrue(all(row["status"] == "PASS" for row in result["additivity"]))
        self.assertTrue(all(row["status"] == "PASS" for row in result["predictions"]))


if __name__ == "__main__":
    unittest.main()
