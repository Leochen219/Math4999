import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import umi_precision_calibration as calibration_module
from umi_precision_calibration import (
    CalibrationConfig,
    ForwardEvaluation,
    analytic_jacobian_vector,
    build_calibration_inputs,
    centered_difference,
    evaluate_forward,
    generate_calibration_artifacts,
    one_sided_difference,
    relative_error_with_reason,
    scan_calibration,
)


class CalibrationFormulaTests(unittest.TestCase):
    def test_fixed_inputs_are_deterministic_and_have_required_geometry(self):
        left = build_calibration_inputs(seed=20260913)
        right = build_calibration_inputs(seed=20260913)

        self.assertEqual(left.weights.shape, (256, 128))
        self.assertEqual(left.weights.dtype, np.dtype(np.float32))
        self.assertEqual(left.x.shape, (128,))
        self.assertEqual(left.v.shape, (128,))
        self.assertAlmostEqual(float(np.linalg.norm(left.v)), 1.0, places=14)
        np.testing.assert_array_equal(left.weights, right.weights)
        np.testing.assert_array_equal(left.x, right.x)
        np.testing.assert_array_equal(left.v, right.v)

    def test_forward_and_jacobian_match_the_declared_cubic_formula(self):
        weights = np.array([[1.0, -2.0], [0.5, 3.0]], dtype=np.float32)
        x = np.array([0.2, -0.4], dtype=np.float64)
        v = np.array([-0.5, 0.75], dtype=np.float64)

        result = evaluate_forward(weights, x, compute_dtype="float64")
        wx = weights.astype(np.float64) @ x
        expected = wx + 0.1 * wx**3
        np.testing.assert_allclose(result.value, expected, rtol=0.0, atol=1e-15)

        expected_jv = (1.0 + 0.3 * wx**2) * (weights.astype(np.float64) @ v)
        np.testing.assert_allclose(
            analytic_jacobian_vector(weights, x, v), expected_jv, rtol=0.0, atol=1e-15
        )

    def test_differences_promote_each_evaluation_before_subtraction(self):
        x = np.array([1.0], dtype=np.float64)
        v = np.array([1.0], dtype=np.float64)

        def float32_response(point):
            return np.array([np.float32(1.0) + np.float32(2e-7) * np.float32(point[0])], dtype=np.float32)

        one_sided = one_sided_difference(float32_response, x, v, 2e-7)
        centered = centered_difference(float32_response, x, v, 2e-7)
        self.assertEqual(one_sided.dtype, np.dtype(np.float64))
        self.assertEqual(centered.dtype, np.dtype(np.float64))
        self.assertTrue(np.all(np.isfinite(one_sided)))
        self.assertTrue(np.all(np.isfinite(centered)))

    def test_hand_derived_cubic_fixture_checks_both_denominators_and_relative_error(self):
        x = np.array([2.0], dtype=np.float64)
        v = np.array([1.0], dtype=np.float64)
        h = 0.25

        def cubic(point):
            return np.asarray(point, dtype=np.float64) ** 3

        one_sided = one_sided_difference(cubic, x, v, h)
        centered = centered_difference(cubic, x, v, h)
        # These values are derived directly from (2 +/- 0.25)^3, not from
        # the production cubic calibration or its constants.
        expected_one_sided = np.array([(2.25**3 - 2.0**3) / 0.25])
        expected_centered = np.array([(2.25**3 - 1.75**3) / (2.0 * 0.25)])
        np.testing.assert_allclose(one_sided, expected_one_sided, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(centered, expected_centered, rtol=0.0, atol=1e-14)
        target = np.array([12.0])  # d/dx x^3 at x=2
        one_error = float(np.sqrt(np.mean((expected_one_sided - target) ** 2)) / np.sqrt(np.mean(target**2)))
        centered_error = float(np.sqrt(np.mean((expected_centered - target) ** 2)) / np.sqrt(np.mean(target**2)))
        one_metric = relative_error_with_reason(one_sided, target)
        centered_metric = relative_error_with_reason(centered, target)
        self.assertAlmostEqual(one_metric["value"], one_error, places=14)
        self.assertAlmostEqual(centered_metric["value"], centered_error, places=14)
        self.assertAlmostEqual(float(centered[0]), 12.0625, places=14)

    def test_precision_modes_hold_the_fp32_weight_representation_fixed(self):
        inputs = build_calibration_inputs(seed=20260913)
        reference = evaluate_forward(inputs.weights, inputs.x, compute_dtype="float64")
        low = evaluate_forward(inputs.weights, inputs.x, compute_dtype="float32")
        quantized = evaluate_forward(
            inputs.weights,
            inputs.x,
            compute_dtype="float64",
            input_quantization="bfloat16",
        )

        self.assertEqual(reference.weight_dtype, "float32")
        self.assertEqual(low.weight_dtype, "float32")
        self.assertEqual(quantized.weight_dtype, "float32")
        self.assertEqual(reference.execution_kind, "fp64_reference")
        self.assertEqual(low.execution_kind, "rounding_simulation")
        self.assertEqual(quantized.execution_kind, "input_quantization_rounding_simulation")
        self.assertEqual(reference.weight_sha256, low.weight_sha256)
        self.assertEqual(reference.weight_sha256, quantized.weight_sha256)
        self.assertEqual(reference.value.dtype, np.dtype(np.float64))
        self.assertEqual(low.value.dtype, np.dtype(np.float32))
        self.assertEqual(quantized.value.dtype, np.dtype(np.float64))
        self.assertNotEqual(reference.input_sha256, quantized.input_sha256)

    def test_zero_reference_norm_is_explicit_na(self):
        metric = relative_error_with_reason(np.array([1.0, 2.0]), np.zeros(2))
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["reason"], "reference denominator RMS is zero")

    def test_scan_uses_exact_grid_and_records_na_reasons(self):
        config = CalibrationConfig(seed=20260913)
        result = scan_calibration(config)
        self.assertEqual(result.h_grid.shape, (25,))
        self.assertEqual(result.h_grid.size, 25)
        self.assertAlmostEqual(float(result.h_grid[0]), 1e-7, places=15)
        self.assertAlmostEqual(float(result.h_grid[-1]), 1e-1, places=15)
        expected_log_step = np.log(1e-1 / 1e-7) / 24.0
        np.testing.assert_allclose(
            np.log(result.h_grid[1:]) - np.log(result.h_grid[:-1]),
            expected_log_step,
            rtol=1e-12,
            atol=1e-15,
        )
        self.assertEqual(len(result.rows), 25 * 3 * 2)
        self.assertEqual({row["scheme"] for row in result.rows}, {"one_sided", "centered"})
        self.assertEqual(
            {row["mode"] for row in result.rows},
            {"fp64_reference", "input_quantized_bf16", "low_precision_compute_fp32"},
        )
        self.assertTrue(any(row["error"] is not None for row in result.rows))
        self.assertIn("reason", relative_error_with_reason(np.zeros(2), np.zeros(2)))

    def test_nonfinite_forward_value_keeps_one_na_row_and_other_h_values(self):
        original = calibration_module.evaluate_forward
        calls = {"count": 0}

        def one_overflow(*args, **kwargs):
            result = original(*args, **kwargs)
            if calls["count"] == 0:
                calls["count"] += 1
                return ForwardEvaluation(
                    value=np.full_like(result.value, np.inf),
                    input_values=result.input_values,
                    compute_dtype=result.compute_dtype,
                    input_quantization=result.input_quantization,
                    execution_kind=result.execution_kind,
                    weight_dtype=result.weight_dtype,
                    weight_sha256=result.weight_sha256,
                    input_sha256=result.input_sha256,
                )
            calls["count"] += 1
            return result

        with mock.patch.object(calibration_module, "evaluate_forward", side_effect=one_overflow):
            result = scan_calibration(CalibrationConfig(seed=20260913))

        self.assertGreater(calls["count"], 1)
        unavailable = [row for row in result.rows if row["error"] is None]
        available = [row for row in result.rows if row["error"] is not None]
        self.assertTrue(unavailable)
        self.assertTrue(available)
        self.assertTrue(all(str(row["error_reason"]).startswith("N/A:") for row in unavailable))
        self.assertTrue(all(np.isfinite(float(row["h"])) for row in result.rows))
        strict_json = json.dumps({"rows": list(result.rows)}, allow_nan=False)
        self.assertNotIn("Infinity", strict_json)
        self.assertNotIn("NaN", strict_json)

    def test_nonfinite_difference_keeps_rows_with_explicit_na_reason(self):
        def nonfinite_difference(*args, **kwargs):
            return np.full(256, np.nan, dtype=np.float64)

        with mock.patch.object(calibration_module, "one_sided_difference", side_effect=nonfinite_difference):
            result = scan_calibration(CalibrationConfig(seed=20260913))

        one_sided_rows = [row for row in result.rows if row["scheme"] == "one_sided"]
        centered_rows = [row for row in result.rows if row["scheme"] == "centered"]
        self.assertEqual(len(one_sided_rows), 75)
        self.assertTrue(all(row["error"] is None for row in one_sided_rows))
        self.assertTrue(all(str(row["error_reason"]).startswith("N/A:") for row in one_sided_rows))
        self.assertTrue(all(row["derivative_rms"] is None for row in one_sided_rows))
        self.assertTrue(any(row["error"] is not None for row in centered_rows))


class CalibrationArtifactTests(unittest.TestCase):
    def test_checked_in_calibration_text_assets_are_lf_and_hash_bound(self):
        source = Path(__file__).with_name("artifacts") / "umi_precision_calibration"
        hashes = json.loads((source / "hashes.json").read_text(encoding="utf-8"))
        text_names = {
            name for name in hashes["files"]
            if name.endswith((".csv", ".json", ".md", ".svg"))
        } | {"hashes.json"}
        for name in sorted(text_names):
            payload = (source / name).read_bytes()
            self.assertNotIn(b"\r", payload, name)
            if name in hashes["files"]:
                self.assertEqual(hashlib.sha256(payload).hexdigest(), hashes["files"][name], name)

    def test_artifacts_include_scalar_data_plots_config_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "calibration"
            result = generate_calibration_artifacts(output)

            required = {
                "config.json",
                "metrics.csv",
                "metrics.json",
                "error_plot.png",
                "error_plot.svg",
                "hashes.json",
                "calibration_report.md",
            }
            self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
            self.assertEqual(result["status"], "COMPLETE")

            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            hashes = json.loads((output / "hashes.json").read_text(encoding="utf-8"))
            self.assertEqual(config["matrix_shape"], [256, 128])
            self.assertEqual(config["h_grid_count"], 25)
            self.assertEqual(metrics["row_count"], 150)
            self.assertEqual(config["weight_representation"], "fixed_fp32_values_reused_as_fp64")
            self.assertEqual(config["low_precision_execution"], "rounding_simulation")

            with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 150)
            self.assertIn("error_reason", rows[0])
            self.assertTrue(any(row["error"] == "" for row in rows) or all(row["error"] for row in rows))

            for name, digest in hashes["files"].items():
                payload = (output / name).read_bytes()
                self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_artifacts_are_byte_deterministic_across_distinct_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            generate_calibration_artifacts(first)
            generate_calibration_artifacts(second)
            first_names = sorted(path.name for path in first.iterdir() if path.is_file())
            second_names = sorted(path.name for path in second.iterdir() if path.is_file())
            self.assertEqual(first_names, second_names)
            for name in first_names:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)
            svg_text = (first / "error_plot.svg").read_text(encoding="utf-8")
            self.assertNotIn("dc:date", svg_text)
            svg_lines = svg_text.splitlines()
            self.assertTrue(all(line == line.rstrip(" \t") for line in svg_lines))

    def test_math_note_is_separate_from_mechanism_code_and_covers_bound(self):
        note = Path(__file__).with_name("umi_precision_calibration_math.md")
        self.assertTrue(note.is_file())
        text = note.read_text(encoding="utf-8")
        for required in (
            "Q(z_bar)=z_bar",
            "e_h",
            "unit v",
            "M*h/2",
            "L*||e_h||/h",
            "2*eta/h",
            "FP32 is not ground truth",
            "baseline zero does not imply eta=0",
            "one direction",
            "no-window result",
            "1/h",
        ):
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main()
