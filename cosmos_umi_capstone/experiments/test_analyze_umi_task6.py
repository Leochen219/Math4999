from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import analyze_umi_task6 as api
import umi_task5_primitives as p5


def _linear_records():
    shape = (1, 1, 3, 2, 2); mask = np.zeros(shape, bool); mask[:, :, 0] = True
    bank = np.zeros((3,) + shape, np.float32)
    bank[0, :, :, 0, 0, 0] = 2; bank[1, :, :, 0, 0, 1] = 2; bank[2, :, :, 0, 1, 0] = 2
    frozen = p5.freeze_task5_directions(bank, mask)
    y0 = np.zeros((1, 1, 2, 2, 2), np.float32)
    records = {"bridge_0__seed_0__baseline_pre": {"output_full": y0, "decoded_final": np.zeros((3,2,2), np.float32)},
               "bridge_0__seed_0__baseline_post": {"output_full": y0.copy(), "decoded_final": np.zeros((3,2,2), np.float32)}}
    weights = np.arange(1, 5, dtype=np.float32).reshape(1, 1, 1, 2, 2)
    for direction_id, direction in frozen["directions"].items():
        scalar = float(np.sum(direction[:, :, 0] * weights[:, :, 0]))
        for ordinal, alpha in enumerate((1e-3, 3e-3, 1e-2)):
            for sign, label in ((1,"plus"),(-1,"minus")):
                delta = sign * alpha * direction
                records[f"bridge_0__seed_0__{direction_id}_alpha_{ordinal:02d}_{label}"] = {
                    "output_full": y0 + sign * alpha * scalar,
                    "decoded_final": np.full((3,2,2), sign * alpha * scalar, np.float32),
                    "actual_delta_fp32": delta, "target_delta_fp32": delta.copy(), "direction": direction, "mask": mask, "z_bar": np.ones(shape, np.float32)}
    return records, frozen


class AnalysisTests(unittest.TestCase):
    def test_exact_linear_prefixed_task6_fixture_has_six_and_24_checks(self):
        records, frozen = _linear_records()
        result = api.analyze_task6_records(records, plan_detail={"s_z": 1.0, "combination_coefficients": {"c01": frozen["c01"], "c12": frozen["c12"]}}, strict=True)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(result["additivity"]), 6)
        self.assertEqual(len(result["predictions"]), 24)

    def test_missing_and_stopped_are_not_reported_as_scientific_failures(self):
        result = api.analyze_task6_records({"baseline_pre": {}}, strict=False)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["summary"]["prediction_total"], 24)

    def test_raw_manifest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "x.txt").write_text("x", encoding="utf-8")
            (root / "MANIFEST.sha256").write_text(f"{'0'*64}  x.txt\n", encoding="ascii")
            with self.assertRaises(ValueError): api.verify_raw_manifest(root)

    def test_float64_reduction_does_not_change_raw_fixture(self):
        records, frozen = _linear_records(); before = records["bridge_0__seed_0__baseline_pre"]["output_full"].copy()
        api.analyze_task6_records(records, plan_detail={"combination_coefficients": {"c01": frozen["c01"], "c12": frozen["c12"]}})
        np.testing.assert_array_equal(before, records["bridge_0__seed_0__baseline_pre"]["output_full"])


if __name__ == "__main__": unittest.main()
