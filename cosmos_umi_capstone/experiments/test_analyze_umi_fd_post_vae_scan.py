import csv
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import analyze_umi_fd_post_vae_scan as analyzer_module
import republish_umi_fd_post_vae_analysis as republish_module
from republish_umi_fd_post_vae_analysis import republish

from umi_fd_post_vae_scan import ALPHAS, COMPATIBILITY_FIELDS, DIRECTION_COUNT, build_call_plan, generate_directions_for_mask, hash_predicted_noisy_region, write_sha256_manifest, sha256_array, sha256_file
from analyze_umi_fd_post_vae_scan import (
    _fit_loglog,
    _window_key,
    _window_index,
    analyze_records,
    analyze_run,
    finite_or_na,
    select_candidate_windows,
)


class AnalyzerTests(unittest.TestCase):
    def _png_rgb(self, path):
        payload = path.read_bytes()
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", payload[16:24])
        depth, color_type = payload[24], payload[25]
        assert depth == 8 and color_type == 2
        pos = 8
        compressed = bytearray()
        while pos < len(payload):
            length = struct.unpack(">I", payload[pos:pos + 4])[0]
            kind = payload[pos + 4:pos + 8]
            chunk = payload[pos + 8:pos + 8 + length]
            pos += 12 + length
            if kind == b"IDAT":
                compressed.extend(chunk)
            if kind == b"IEND":
                break
        raw = __import__("zlib").decompress(bytes(compressed))
        stride = width * 3
        rows = []
        offset = 0
        for _ in range(height):
            self_filter = raw[offset]
            assert self_filter == 0
            rows.append(np.frombuffer(raw[offset + 1:offset + 1 + stride], dtype=np.uint8).reshape(width, 3))
            offset += stride + 1
        return np.stack(rows), width, height

    def _records(self, *, nonlinear=False, high_floor=False, missing=False):
        mask = np.zeros((1, 1, 2, 1, 2), dtype=bool)
        mask[:, :, 0] = True
        base_latent = np.zeros(mask.shape, dtype=np.float32)
        base_rgb = np.zeros((3, 2, 2, 2), dtype=np.float32)
        records = {}
        network_hash = sha256_array(np.ones((2,), dtype=np.float32))
        common = {
            "carrier_fp32": mask.astype(np.float32), "condition_mask": mask,
            "initial_state": mask.astype(np.float32), "initial_condition_mask": mask,
            "network_condition_bf16": np.ones((2,), dtype=np.float32),
            "expected_network_condition_hash": network_hash, "network_condition_dtype": "torch.bfloat16",
            "initial_noise_hash": hash_predicted_noisy_region(mask.astype(np.float32), mask), "condition_step_hashes": [network_hash] * 30,
            "first_denoise_hash": network_hash, "final_denoise_hash": network_hash, "denoise_step_evidence": [{"step_index": i, "timestep": float(i)} for i in range(30)],
            "expected_denoise_steps": 30, "text_kv_lifecycle": [{"before": [1]}, {"after": [1]}],
            "cfg_branch_semantics": {"guidance": 1.0, "conditional_calls": 30, "unconditional_calls": 0, "expected_multiplicity": 1},
            "cfg_branch_observations": [{"branch": "conditional", "used": True, "step_index": i} for i in range(30)], "cfg_branch_calls": 30,
            "cache_lifecycle_valid": True,
        }
        for sample in build_call_plan():
            sid = sample["sample_id"]
            record = dict(common)
            record.update({"status": "success", "sample_id": sid, "spec": sample})
            latent = base_latent.copy()
            rgb = base_rgb.copy()
            if sid not in {"A", "B", "C", "D", "scan_pre", "scan_post"}:
                alpha = float(sample["alpha"])
                sign = int(sample["sign"])
                response_scale = alpha ** (2 if nonlinear else 1)
                latent[:, :, 1] += np.float32(sign * response_scale)
                rgb[:, -1] += np.float32(sign * response_scale)
                record["actual_delta_fp32"] = np.where(mask, sign * alpha, 0).astype(np.float32)
                record["actual_delta_bf16"] = record["actual_delta_fp32"].copy()
                record["target_delta_fp32"] = record["actual_delta_fp32"].copy()
                record["target_alpha"] = alpha
                record["target_epsilon"] = alpha
                record["target_rms"] = alpha
                record["s_z"] = 1.0
            else:
                record["actual_delta_fp32"] = np.zeros_like(mask, dtype=np.float32)
                record["actual_delta_bf16"] = record["actual_delta_fp32"].copy()
                record["target_delta_fp32"] = record["actual_delta_fp32"].copy()
                record["target_epsilon"] = 0.0
                record["target_rms"] = 0.0
                record["s_z"] = 0.0
                if sid == "scan_post" and high_floor:
                    latent += np.float32(1.0)
                    rgb += np.float32(1.0)
            record.update({
                "final_latent_full": latent, "predicted_latent": latent[:, :, 1:],
                "latent_slicing": {"predicted_indexes": [1], "axis": 2, "source_shape": list(latent.shape), "selected_shape": list(latent[:, :, 1:].shape)},
                "decoded_float": rgb, "decoded_final": rgb[:, -1],
                "image_slicing": {"frame_index": 1, "axis": 1, "source_shape": list(rgb.shape), "selected_shape": list(rgb[:, -1].shape)},
            })
            if missing and sid.startswith("dir_00"):
                record["actual_delta_fp32"] = np.full(mask.shape, np.nan, dtype=np.float32)
            records[sid] = record
        return records

    def test_linear_response_selects_window_with_slope_one(self):
        result = analyze_records(self._records(), output_dir=None)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertTrue(any(item["selected"] for item in result["window_decisions"]))
        slopes = [item["slope"] for item in result["fits"] if item["sign"] == 1 and item["space"] == "predicted_latent" and item["r2"] is not None]
        self.assertTrue(slopes)
        self.assertAlmostEqual(slopes[0], 1.0, places=6)
        self.assertAlmostEqual(next(item["r2"] for item in result["fits"] if item["sign"] == 1 and item["space"] == "predicted_latent"), 1.0, places=12)

    def test_nonlinear_and_high_floor_yield_explicit_no_candidate(self):
        nonlinear = analyze_records(self._records(nonlinear=True), output_dir=None)
        self.assertEqual(nonlinear["status"], "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL")
        high_floor = analyze_records(self._records(high_floor=True), output_dir=None)
        self.assertEqual(high_floor["status"], "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL")
        self.assertTrue(any("floor" in str(item["reasons"]).lower() for item in high_floor["window_decisions"]))

    def test_zero_denominators_are_na_and_json_safe(self):
        self.assertIsNone(finite_or_na(1.0, 0.0))
        result = analyze_records(self._records(missing=True), output_dir=None)
        encoded = json.dumps(result, allow_nan=False)
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)

    def test_constant_fit_and_zero_metric_nulls_have_field_specific_reasons(self):
        fit = _fit_loglog([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], ["a", "b", "c"], [])
        self.assertIsNone(fit["r2"])
        self.assertTrue(fit.get("r2_reason"))
        records = self._records()
        sample = next(key for key, value in records.items() if value.get("spec", {}).get("kind") == "perturbation")
        records[sample]["actual_delta_fp32"] = np.zeros_like(records[sample]["actual_delta_fp32"])
        records[sample]["actual_delta_bf16"] = np.zeros_like(records[sample]["actual_delta_bf16"])
        result = analyze_records(records)
        row = next(item for item in result["point_metrics"] if item["sample_id"] == sample)
        self.assertIsNone(row["gain"])
        self.assertIn("gain", row["undefined_reasons"])

    def test_fit_reason_metadata_is_not_attached_as_quantitative_null(self):
        fit = _fit_loglog([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], ["a", "b", "c"], [])
        self.assertIsNone(fit["reason"])
        result = analyze_records(self._records(), output_dir=None)
        for row in result["fits"]:
            if row.get("slope") is not None:
                self.assertNotIn("reason", row.get("null_reasons", {}))

    def test_outside_mask_and_realized_delta_mismatch_are_rejected(self):
        records = self._records()
        sample = next(key for key in records if key.startswith("dir_00") and key.endswith("plus"))
        records[sample]["actual_delta_fp32"] = records[sample]["actual_delta_fp32"].copy()
        records[sample]["actual_delta_fp32"][~records[sample]["condition_mask"]] = 1.0
        result = analyze_records(records)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("outside", result["reason"])

    def test_erased_bf16_point_cannot_select_candidate(self):
        records = self._records()
        for sample, record in records.items():
            if record.get("spec", {}).get("kind") == "perturbation":
                record["actual_delta_bf16"] = np.zeros_like(record["actual_delta_bf16"])
        result = analyze_records(records)
        self.assertEqual(result["status"], "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL")
        self.assertTrue(any("BF16" in " ".join(item["reasons"]) for item in result["window_decisions"]))
        row = next(item for item in result["point_metrics"] if item["sample_id"] == next(key for key in records if key.startswith("dir_00")) and item["space"] == "predicted_latent")
        self.assertEqual(row["bf16_survival_ratio"], 0.0)
        self.assertNotIn("bf16_survival_ratio undefined", str(row.get("undefined_reasons") or ""))

    def test_fabricated_bf16_delta_is_rejected_against_carrier_cast(self):
        records = self._records()
        sample = next(key for key, value in records.items() if value.get("spec", {}).get("kind") == "perturbation")
        record = records[sample]
        baseline = np.where(record["condition_mask"], 1.0, 0.0).astype(np.float32)
        realized = (baseline + record["actual_delta_fp32"]).astype(np.float32)
        record["carrier_fp32"] = baseline
        record["realized_carrier_fp32"] = realized
        record["actual_delta_fp32"] = (realized - baseline).astype(np.float32)
        record["actual_delta_bf16"] = np.where(record["condition_mask"], 1.0, 0.0).astype(np.float32)
        result = analyze_records(records)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("BF16", result["reason"])

    def test_negative_adjacent_direction_and_opposition_are_gated(self):
        records = self._records()
        for sample, record in records.items():
            if record.get("spec", {}).get("kind") == "perturbation" and int(record["spec"]["sign"]) == -1:
                bad = record["actual_delta_fp32"].copy()
                bad[record["condition_mask"]] = -float(record["spec"]["alpha"])
                first = np.flatnonzero(record["condition_mask"])[0]
                bad.reshape(-1)[first] *= 0.5
                record["actual_delta_fp32"] = bad
                record["actual_delta_bf16"] = bad.copy()
        result = analyze_records(records)
        self.assertEqual(result["status"], "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL")
        self.assertTrue(any("opposition" in " ".join(item["reasons"]) or "minus" in " ".join(item["reasons"]) for item in result["window_decisions"]))

    def test_target_delta_evidence_is_required(self):
        records = self._records()
        for record in records.values():
            if record.get("spec", {}).get("kind") == "perturbation":
                record.pop("target_delta_fp32", None)
        result = analyze_records(records)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("target", result["reason"])

    def test_target_scalar_nan_and_noise_recompute_are_rejected(self):
        records = self._records()
        sample = next(key for key, value in records.items() if value.get("spec", {}).get("kind") == "perturbation")
        records[sample]["target_epsilon"] = float("nan")
        result = analyze_records(records)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("target", result["reason"])
        records = self._records()
        records["scan_pre"]["initial_noise_hash"] = "0" * 64
        result = analyze_records(records)
        self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
        self.assertIn("noise", result["reason"])

    def test_input_rms_uses_condition_mask_domain(self):
        records = self._records()
        sample = next(key for key in records if key.startswith("dir_00") and key.endswith("plus"))
        result = analyze_records(records)
        row = next(item for item in result["point_metrics"] if item["sample_id"] == sample and item["space"] == "predicted_latent")
        self.assertAlmostEqual(row["actual_input_rms"], float(records[sample]["spec"]["alpha"]), places=7)

    def test_output_contract_is_idempotent_and_creates_plots_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"code_sha256":"' + ('a' * 64) + '"}\n', encoding="utf-8")
            (root / "status.json").write_text('{"status":"COMPLETE"}\n', encoding="utf-8")
            result = analyze_records(self._records(), output_dir=root)
            expected = {"scan_summary.json", "response_metrics.csv", "finite_difference_consistency.csv", "injection_diagnostics.csv", "window_decisions.csv", "implementation_notes.md", "experiment_report.md", "MANIFEST.sha256", "review_bundle.zip"}
            self.assertTrue(expected.issubset({path.name for path in root.iterdir()}))
            self.assertTrue(list(root.glob("*.png")))
            self.assertTrue(list(root.glob("*.svg")))
            png_header = (root / "response_magnitude.png").read_bytes()[16:24]
            self.assertGreaterEqual(struct.unpack(">II", png_header), (320, 180))
            first = (root / "scan_summary.json").read_bytes()
            analyze_records(self._records(), output_dir=root)
            self.assertEqual(first, (root / "scan_summary.json").read_bytes())
            bundle = (root / "review_bundle.zip").read_bytes()
            manifest = (root / "MANIFEST.sha256").read_bytes()
            analyze_records(self._records(), output_dir=root)
            self.assertEqual(bundle, (root / "review_bundle.zip").read_bytes())
            self.assertEqual(manifest, (root / "MANIFEST.sha256").read_bytes())
            report = (root / "experiment_report.md").read_text(encoding="utf-8")
            for heading in ("Technical summary", "Key findings with figures", "Scope/data/metric definitions", "Experiment/validation details", "Limitations/robustness", "Recommended next step", "Further questions"):
                self.assertIn(heading, report)
            self.assertIn("direction 0", report)
            self.assertIn("low-rank", report)
            self.assertIn("Offline re-analysis loaded no model; the inference run did", report)
            self.assertNotIn("No model was loaded, no SVD was run, and no real experiment result is claimed here", report)
            with (root / "response_metrics.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    self.assertNotIn(row.get("output_rms", ""), {"nan", "inf", "-inf"})
            with (root / "finite_difference_consistency.csv").open(newline="", encoding="utf-8") as stream:
                finite_rows = list(csv.DictReader(stream))
            self.assertTrue(finite_rows)
            self.assertTrue(any(row.get("input_cosine") not in {"", "None"} for row in finite_rows))
            self.assertTrue(any(row.get("relative_rms") not in {"", "None"} for row in finite_rows))
            svg_text = (root / "finite_difference_relative_rms.svg").read_text(encoding="utf-8")
            self.assertTrue("numeric" in svg_text.lower() or "<circle" in svg_text or "<path" in svg_text)
            response_svg = (root / "response_magnitude.svg").read_text(encoding="utf-8")
            self.assertIn("Actual input RMS", response_svg)
            self.assertIn("slope-1", response_svg)
            self.assertIn("predicted_latent", response_svg)
            self.assertIn("final_rgb", response_svg)
            self.assertIn("data-series='predicted_latent d0 +'", response_svg)
            self.assertIn("data-series='predicted_latent d0 -'", response_svg)
            heatmap_svg = (root / "finite_difference_cosine.svg").read_text(encoding="utf-8")
            self.assertIn("data-cell='predicted_latent|sign=1|direction=0|adjacent=0'", heatmap_svg)
            diagnostics_svg = (root / "diagnostics.svg").read_text(encoding="utf-8")
            for label in ("Actual / target", "BF16 survival", "Outside-mask exactness", "Response / float-floor ratio"):
                self.assertIn(label, diagnostics_svg)

    def test_png_fallback_has_readable_text_reference_ticks_legend_cells_and_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            response, width, height = self._png_rgb(root / "response_magnitude.png")
            background = np.array([248, 250, 252], dtype=np.uint8)
            changed = np.any(response != background, axis=2)
            self.assertGreater(changed[0:70, 100:890].sum(), 100, "title must be rendered into PNG pixels")
            self.assertGreater(changed[545:600, 100:890].sum(), 100, "axis labels and legend must be rendered into PNG pixels")
            # The neutral slope-1 reference has its own pixels in the central plot.
            self.assertGreater(np.all(response[100:500, 150:850] == np.array([101, 114, 126]), axis=2).sum(), 20)
            # At least one axis/tick region must contain text-like marks beyond the spine.
            self.assertGreater(changed[80:520, 95:145].sum(), 50)

            heatmap, _, _ = self._png_rgb(root / "finite_difference_relative_rms.png")
            cell_colors = {
                tuple(pixel)
                for pixel in heatmap[80:500, 110:875].reshape(-1, 3)
                if tuple(pixel) not in {tuple(background), (55, 65, 75), (38, 50, 56)}
            }
            self.assertGreaterEqual(len(cell_colors), 2, "fixed-scale heatmap must show at least data and background colors")
            # The first two adjacent-alpha cells are separated by a visible grid line.
            first = heatmap[85:105, 115:240]
            second = heatmap[85:105, 250:375]
            self.assertGreater(np.any(first != second, axis=2).sum(), 20)

            diagnostics, _, _ = self._png_rgb(root / "diagnostics.png")
            # Every one of the four explicitly labeled panels must carry marks/text.
            panel_regions = ((100, 80, 480, 275), (500, 80, 875, 275), (100, 310, 480, 505), (500, 310, 875, 505))
            for x0, y0, x1, y1 in panel_regions:
                region = diagnostics[y0:y1, x0:x1]
                self.assertGreater(np.any(region != background, axis=2).sum(), 200)

    def test_svg_contains_visible_mappings_ticks_bands_and_numeric_heatmap_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            response = (root / "response_magnitude.svg").read_text(encoding="utf-8")
            self.assertIn("log10", response)
            self.assertIn("tick", response.lower())
            self.assertIn("circle", response)
            self.assertIn("path", response)
            self.assertIn("slope-1 neutral reference", response)
            slope = (root / "slope_evidence.svg").read_text(encoding="utf-8")
            self.assertIn("0.8", slope)
            self.assertIn("1.2", slope)
            self.assertIn("acceptance band", slope.lower())
            heatmap = (root / "finite_difference_relative_rms.svg").read_text(encoding="utf-8")
            self.assertIn("1e-04", heatmap)
            self.assertIn("3e-04", heatmap)
            self.assertIn("min", heatmap.lower())
            self.assertIn("max", heatmap.lower())
            self.assertIn("predicted_latent d0 +", heatmap)
            diagnostics = (root / "diagnostics.svg").read_text(encoding="utf-8")
            for label in ("Actual / target", "BF16 survival", "Outside-mask exactness", "Response / float-floor ratio"):
                self.assertIn(label, diagnostics)
            with __import__("zipfile").ZipFile(root / "review_bundle.zip") as archive:
                self.assertEqual(archive.read("MANIFEST.sha256"), (root / "MANIFEST.sha256").read_bytes())

    def test_plot_svg_declares_series_identity_fixed_scales_and_ascii_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            response = (root / "response_magnitude.svg").read_text(encoding="utf-8")
            # Every plotted family needs a visible mapping, not just per-mark metadata.
            self.assertIn("Series map", response)
            self.assertIn("predicted_latent d0 +", response)
            self.assertIn("final_rgb d3 -", response)
            slope = (root / "slope_evidence.svg").read_text(encoding="utf-8")
            self.assertIn("candidate window", slope.lower())
            self.assertIn("predicted_latent d0 +", slope)
            self.assertIn("R2 ge 0.98", slope)
            cosine = (root / "finite_difference_cosine.svg").read_text(encoding="utf-8")
            self.assertIn("fixed scale", cosine.lower())
            self.assertIn("0.95", cosine)
            self.assertIn("1.00", cosine)
            relative = (root / "finite_difference_relative_rms.svg").read_text(encoding="utf-8")
            self.assertIn("0.25", relative)
            self.assertIn("color scale", relative.lower())

    def test_diagnostic_fixed_scales_make_all_zero_and_all_one_fixtures_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zero_rows = [{"actual_target_ratio": 0.0, "bf16_survival_ratio": 0.0, "outside_mask_exact": 0.0, "floor_multiple": 0.0}]
            one_rows = [{"actual_target_ratio": 1.0, "bf16_survival_ratio": 1.0, "outside_mask_exact": 1.0, "floor_multiple": 1.0}]
            zero_png, zero_svg = analyzer_module._render_fallback("diagnostics", zero_rows, [], [], [])
            one_png, one_svg = analyzer_module._render_fallback("diagnostics", one_rows, [], [], [])
            self.assertNotEqual(zero_png, one_png)
            self.assertNotEqual(zero_svg, one_svg)

    def test_public_cli_summary_is_compact_and_excludes_serialized_arrays(self):
        result = {
            "status": "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL",
            "candidate_count": 0,
            "baseline_floors": {"predicted_latent": 0.0, "final_rgb": 0.0},
            "point_metrics": [{"space": "predicted_latent", "response": [1, 2, 3]}] * 200,
            "fits": [{"slope": 1.0}] * 160,
        }
        summary = analyzer_module.format_cli_summary(result)
        decoded = json.loads(summary)
        self.assertEqual(decoded["status"], result["status"])
        self.assertEqual(decoded["candidate_count"], 0)
        self.assertNotIn("point_metrics", decoded)
        self.assertNotIn("fits", decoded)
        self.assertLess(len(summary), 1000)

    def test_run_timing_preserves_stage_scan_and_full_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {
                "A": ("2026-09-12T00:00:00+00:00", "2026-09-12T00:00:10+00:00"),
                "D": ("2026-09-12T00:00:30+00:00", "2026-09-12T00:00:40+00:00"),
                "scan_pre": ("2026-09-12T00:01:00+00:00", "2026-09-12T00:01:10+00:00"),
                "scan_post": ("2026-09-12T00:02:00+00:00", "2026-09-12T00:02:20+00:00"),
            }
            for sample_id, (started, finished) in values.items():
                sample_dir = root / "samples" / sample_id
                sample_dir.mkdir(parents=True)
                (sample_dir / "status.json").write_text(json.dumps({"status": "success", "started_utc": started, "finished_utc": finished}), encoding="utf-8")
            timing = analyzer_module._collect_run_timing(root)
            self.assertEqual(timing["stage_a"]["wall_seconds"], 40.0)
            self.assertEqual(timing["resumed_scan"]["wall_seconds"], 80.0)
            self.assertEqual(timing["full_formal"]["wall_seconds"], 140.0)

    def test_plot_layout_has_complete_keys_colorbars_and_bf16_overflow_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            response = (root / "response_magnitude.svg").read_text(encoding="utf-8")
            self.assertEqual(response.count("data-legend='"), 8)
            self.assertIn("plus marker", response.lower())
            self.assertIn("minus marker", response.lower())
            slope = (root / "slope_evidence.svg").read_text(encoding="utf-8")
            self.assertIn("window index", slope.lower())
            self.assertIn("window key", slope.lower())
            self.assertIn("Sign key: circle = plus marker; X = minus marker", slope)
            self.assertIn("data-panel='predicted_latent'", slope)
            self.assertIn("data-panel='final_rgb'", slope)
            cosine = (root / "finite_difference_cosine.svg").read_text(encoding="utf-8")
            self.assertIn("data-colorbar='", cosine)
            self.assertIn("colorbar tick", cosine.lower())
            self.assertIn("under", cosine.lower())
            self.assertIn("over", cosine.lower())
            diagnostics = (root / "diagnostics.svg").read_text(encoding="utf-8")
            self.assertIn("1.1", diagnostics)
            self.assertIn("overflow", diagnostics.lower())

    def test_slope_footer_roles_are_separated_and_inside_canvas(self):
        _, svg = analyzer_module._render_fallback("slope_evidence", [], [], [], [])
        positions = {}
        for role in ("slope-sign-key", "slope-acceptance", "slope-window-key", "slope-axis-label"):
            match = re.search(rf"<text(?=[^>]*data-role='{role}')(?=[^>]*y='([0-9.]+)')[^>]*>", svg)
            self.assertIsNotNone(match, role)
            positions[role] = float(match.group(1))
        self.assertTrue(all(0 <= value <= 660 for value in positions.values()))
        self.assertGreaterEqual(abs(positions["slope-acceptance"] - positions["slope-sign-key"]), 12)
        self.assertGreaterEqual(abs(positions["slope-window-key"] - positions["slope-acceptance"]), 12)
        self.assertGreaterEqual(abs(positions["slope-axis-label"] - positions["slope-sign-key"]), 12)

        bbox_pattern = r"<text(?=[^>]*data-role='(?P<role>slope-(?:sign-key|acceptance|axis-label|axis-tick))')(?=[^>]*data-bbox-y='(?P<top>[0-9.]+)')(?=[^>]*data-bbox-height='(?P<height>[0-9.]+)')[^>]*>"
        boxes = {role: [] for role in ("slope-sign-key", "slope-acceptance", "slope-axis-label", "slope-axis-tick")}
        for match in re.finditer(bbox_pattern, svg):
            boxes[match.group("role")].append((float(match.group("top")), float(match.group("height"))))
        self.assertGreaterEqual(len(boxes["slope-axis-tick"]), 10)
        sign_top, sign_height = boxes["slope-sign-key"][0]
        sign_bottom = sign_top + sign_height
        axis_bottom = max(top + height for role in ("slope-axis-label", "slope-axis-tick") for top, height in boxes[role])
        acceptance_top, _ = boxes["slope-acceptance"][0]
        self.assertGreaterEqual(sign_top, axis_bottom + 8)
        self.assertGreaterEqual(acceptance_top, sign_bottom + 7)
        self.assertTrue(all(top + height <= 660 for role_boxes in boxes.values() for top, height in role_boxes))

    def test_full_fit_fixture_uses_unique_window_ids_and_direction_legend(self):
        result = analyze_records(self._records(), output_dir=None)
        fits = result["fits"]
        keys = {_window_key(row) for row in fits}
        ids = {_window_index(row) for row in fits}
        self.assertEqual(len(fits), 160)
        self.assertEqual(len(keys), 10)
        self.assertEqual(len(ids), 10)
        png, svg = analyzer_module._render_fallback("slope_evidence", self._records() and [], [], fits, result["window_decisions"])
        self.assertEqual(svg.count("data-window-id='"), 160)
        self.assertEqual(svg.count("data-direction-legend='"), 4)

    def test_heatmap_marks_under_min_inside_max_over_and_missing_with_gap(self):
        values = (0.94, 0.95, 0.975, 1.0, 1.1)
        rows = [{"space": "predicted_latent", "direction": 0, "sign": 1, "adjacent_index": i, "input_cosine": value, "relative_rms": value} for i, value in enumerate(values)]
        rows.append({"space": "predicted_latent", "direction": 1, "sign": 1, "adjacent_index": 0, "input_cosine": None, "relative_rms": None})
        _, svg = analyzer_module._render_fallback("finite_difference_cosine", [], rows, [], [])
        for kind in ("under", "min", "inside", "max", "over", "missing"):
            self.assertIn(f"data-value-class='{kind}'", svg)
        self.assertIn("fill='rgb(230,120,20)'", svg)
        self.assertIn("fill='rgb(82,43,110)'", svg)
        self.assertIn("data-heatmap-gap='", svg)
        self.assertEqual(svg.count("data-colorbar='finite_difference_cosine'"), 26)

    def test_selected_plus_minus_boxes_and_shape_metadata_match_in_both_formats(self):
        records = self._records()
        result = analyze_records(records, output_dir=None)
        _, svg = analyzer_module._render_fallback("slope_evidence", [], [], result["fits"], result["window_decisions"])
        self.assertGreaterEqual(svg.count("data-selected='true'"), 2)
        self.assertGreaterEqual(svg.count("data-selected-box='true'"), 2)
        self.assertIn("data-shape='circle'", svg)
        self.assertIn("data-shape='x'", svg)
        _, diagnostic_svg = analyzer_module._render_fallback("diagnostics", [{"actual_target_ratio": 3.0, "bf16_survival_ratio": 2.0, "outside_mask_exact": 1.0, "floor_multiple": 1.0}], [], [], [])
        self.assertIn("data-shape='overflow'", diagnostic_svg)

    def test_implementation_notes_describe_manifest_stage_not_finalized_bundle_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            notes = (root / "implementation_notes.md").read_text(encoding="utf-8").lower()
            self.assertIn("manifest snapshot", notes)
            self.assertNotIn("exactly the finalized root manifest", notes)

    def test_presentation_revision_reads_frozen_csvs_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            revision = root / "revision"
            analyze_records(self._records(), output_dir=frozen)
            before = sha256_file(frozen / "scan_summary.json")
            provenance = republish(frozen, revision, source_commit="2e396a4" + "0" * 33)
            self.assertEqual(before, sha256_file(frozen / "scan_summary.json"))
            self.assertEqual(provenance["mode"], "presentation_only")
            self.assertFalse(provenance["metric_recomputation"])
            self.assertFalse(provenance["provenance_validator_called"])
            self.assertTrue((revision / "slope_evidence.png").is_file())
            self.assertTrue((revision / "analysis_revision_provenance.json").is_file())
            self.assertIn("presentation-only", (revision / "experiment_report.md").read_text(encoding="utf-8"))

    def test_presentation_revision_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "frozen"
            revision = root / "revision"
            analyze_records(self._records(), output_dir=frozen)
            revision.mkdir()
            (revision / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                republish(frozen, revision, source_commit="2e396a4" + "0" * 33)

    def _frozen_revision_fixture(self, root):
        frozen = root / "frozen"
        analyze_records(self._records(), output_dir=frozen)
        return frozen

    def test_presentation_revision_rejects_tampered_required_input_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = self._frozen_revision_fixture(root)
            path = frozen / "response_metrics.csv"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                republish(frozen, root / "revision", source_commit="2" * 40)
            self.assertFalse((root / "revision").exists())

    def test_presentation_revision_rejects_manifest_omission_malformed_and_duplicate(self):
        for mutation in ("omission", "malformed", "duplicate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                frozen = self._frozen_revision_fixture(root)
                manifest = frozen / "MANIFEST.sha256"
                lines = manifest.read_text(encoding="ascii").splitlines()
                if mutation == "omission":
                    lines = [line for line in lines if not line.endswith("  response_metrics.csv")]
                elif mutation == "malformed":
                    lines[0] = "not a sha256 entry"
                else:
                    lines.append(lines[0])
                manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
                with self.assertRaises(ValueError):
                    republish(frozen, root / "revision", source_commit="2" * 40)
                self.assertFalse((root / "revision").exists())

    def test_presentation_revision_rejects_equal_or_nested_input_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = self._frozen_revision_fixture(root)
            for output in (frozen, frozen / "nested", root):
                with self.subTest(output=output):
                    with self.assertRaises(ValueError):
                        republish(frozen, output, source_commit="2" * 40)

    def test_presentation_revision_cleans_sibling_stage_after_atomic_publish_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = self._frozen_revision_fixture(root)
            revision = root / "revision"
            original_replace = republish_module.os.replace
            state = {"failed": False}

            def fail_final(source, destination):
                if not state["failed"] and Path(destination) == revision and ".analysis-revision-stage-" in str(source):
                    state["failed"] = True
                    raise OSError("injected atomic publish failure")
                return original_replace(source, destination)

            republish_module.os.replace = fail_final
            try:
                with self.assertRaises(OSError):
                    republish(frozen, revision, source_commit="2" * 40)
            finally:
                republish_module.os.replace = original_replace
            self.assertFalse(revision.exists())
            self.assertFalse(any(root.glob(".analysis-revision-stage-*")))

    def test_presentation_revision_rejects_invalid_source_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = self._frozen_revision_fixture(root)
            for value in ("2e396a4", "A" * 40, "g" * 40, "2" * 41):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    republish(frozen, root / ("revision-" + str(len(value))), source_commit=value)

    def test_presentation_revision_records_verified_inputs_manifest_snapshot_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = self._frozen_revision_fixture(root)
            revision = root / "revision"
            source_analyzer = Path(republish_module.__file__).with_name("analyze_umi_fd_post_vae_scan.py")
            source_republisher = Path(republish_module.__file__)
            original_manifest = (frozen / "MANIFEST.sha256").read_bytes()
            provenance = republish(frozen, revision, source_commit="2" * 40)
            self.assertEqual(provenance["input_hashes"]["scan_summary.json"], sha256_file(frozen / "scan_summary.json"))
            self.assertEqual((revision / "original_run_manifest_snapshot.sha256").read_bytes(), original_manifest)
            self.assertEqual(provenance["original_run_manifest_snapshot_sha256"], sha256_file(revision / "original_run_manifest_snapshot.sha256"))
            self.assertEqual(provenance["source_hashes"]["analyze_umi_fd_post_vae_scan.py"], sha256_file(source_analyzer))
            self.assertEqual(provenance["source_hashes"]["republish_umi_fd_post_vae_analysis.py"], sha256_file(source_republisher))
            self.assertEqual((revision / "source" / source_analyzer.name).read_bytes(), source_analyzer.read_bytes())
            self.assertEqual((revision / "source" / source_republisher.name).read_bytes(), source_republisher.read_bytes())
            manifest_entries = (revision / "MANIFEST.sha256").read_text(encoding="ascii")
            self.assertIn("  original_run_manifest_snapshot.sha256\n", manifest_entries)
            self.assertIn("  source/analyze_umi_fd_post_vae_scan.py\n", manifest_entries)
            self.assertEqual(provenance["source_commit_status"], "declared")
            notes = (revision / "implementation_notes.md").read_text(encoding="utf-8").lower()
            self.assertNotIn("deterministic presentation-only renderer", notes)
            self.assertIn("generated timestamps", notes)

    def test_publication_preserves_unowned_png_svg_and_raw_inputs_bytewise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preserved = {
                "input_preview.png": b"raw-preview-png",
                "input_preview.svg": b"<svg>raw-preview</svg>",
                "raw_runner_input.bin": b"raw-runner-bytes",
            }
            for name, payload in preserved.items():
                path = root / name
                path.write_bytes(payload)
            analyze_records(self._records(), output_dir=root)
            for name, payload in preserved.items():
                self.assertEqual((root / name).read_bytes(), payload)

    def test_each_null_metric_has_a_field_specific_causal_reason(self):
        records = self._records()
        sample = next(key for key, value in records.items() if value.get("spec", {}).get("kind") == "perturbation")
        records[sample]["actual_delta_fp32"] = np.zeros_like(records[sample]["actual_delta_fp32"])
        records[sample]["actual_delta_bf16"] = np.zeros_like(records[sample]["actual_delta_bf16"])
        result = analyze_records(records, output_dir=None)
        for row in result["point_metrics"] + result["finite_difference"]:
            reasons = str(row.get("undefined_reasons") or "")
            for field, value in row.items():
                if value is None and field not in {"undefined_reasons", "null_reasons"} and not field.endswith("_reason"):
                    self.assertIn(field, reasons, f"missing causal reason for {field}: {row}")

    def test_zero_output_report_reports_finite_input_zero_response_and_na_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = self._records()
            for record in records.values():
                if record.get("spec", {}).get("kind") == "perturbation":
                    record["final_latent_full"] = np.zeros_like(record["final_latent_full"])
                    record["predicted_latent"] = np.zeros_like(record["predicted_latent"])
                    record["decoded_float"] = np.zeros_like(record["decoded_float"])
                    record["decoded_final"] = np.zeros_like(record["decoded_final"])
            result = analyze_records(records, output_dir=root)
            self.assertEqual(result["status"], "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL")
            report = (root / "experiment_report.md").read_text(encoding="utf-8")
            self.assertIn("output RMS: 0", report)
            self.assertIn("gain: 0", report)
            self.assertIn("slope", report)
            self.assertIn("N/A", report)

    def test_disk_validation_rejects_missing_sample_compatibility_even_for_a(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = self._records()
            z0 = records["A"]["carrier_fp32"].astype(np.float32)
            mask = records["A"]["condition_mask"].astype(bool)
            directions = generate_directions_for_mask(mask)
            np.save(root / "z0.npy", z0); np.save(root / "mask.npy", mask); np.save(root / "direction_bank.npy", directions)
            digest = "a" * 64
            config = {field: digest for field in COMPATIBILITY_FIELDS}
            config.update({"z0_sha256": sha256_file(root / "z0.npy"), "mask_sha256": sha256_file(root / "mask.npy"), "direction_sha256": sha256_file(root / "direction_bank.npy")})
            config.update({"runner_sha256": config["code_sha256"], "checkpoint_sha256": config["model_sha256"]})
            provenance = {"framework_root": "missing-framework", "checkpoint_path": "missing-checkpoint", "vae_path": "missing-vae", "input_path": "missing-input", "action_path": "missing-action"}
            provenance.update(config); (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (root / "call_plan.json").write_text(json.dumps(build_call_plan()), encoding="utf-8")
            for sid, record in records.items():
                record["provenance"] = dict(provenance)
                record["compatibility"] = dict(config)
                sample_dir = root / "samples" / sid; sample_dir.mkdir(parents=True)
                (sample_dir / "status.json").write_text(json.dumps({"status": "success", "required_artifacts": [], "artifact_sha256": {}}), encoding="utf-8")
            records["A"].pop("compatibility")
            valid, reason = analyzer_module._validate_run_bindings(root, config, records, build_call_plan())
            self.assertFalse(valid)
            self.assertIn("compatibility", reason)

    def test_diagnostic_rerun_removes_stale_science_and_emits_diagnostic_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            self.assertTrue((root / "response_metrics.csv").is_file())
            analyze_records({"A": self._records()["A"]}, output_dir=root)
            self.assertFalse((root / "response_metrics.csv").exists())
            self.assertTrue((root / "diagnostic_evidence.svg").is_file())

    def test_failed_publication_preserves_previous_complete_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            previous = (root / "scan_summary.json").read_bytes()
            original = analyzer_module._write_artifacts_impl
            def fail(*args, **kwargs):
                raise OSError("injected publication failure")
            analyzer_module._write_artifacts_impl = fail
            try:
                with self.assertRaises(OSError):
                    analyze_records(self._records(), output_dir=root)
            finally:
                analyzer_module._write_artifacts_impl = original
            self.assertEqual(previous, (root / "scan_summary.json").read_bytes())

    def test_replace_failure_restores_previous_tree_bytewise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analyze_records(self._records(), output_dir=root)
            before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            original = analyzer_module.os.replace
            state = {"failed": False}
            def fail_once(source, destination):
                if not state["failed"] and Path(destination).parent == root and ".analysis-stage-" in str(source):
                    state["failed"] = True
                    raise OSError("injected os.replace failure")
                return original(source, destination)
            analyzer_module.os.replace = fail_once
            try:
                with self.assertRaises(OSError):
                    analyze_records(self._records(), output_dir=root)
            finally:
                analyzer_module.os.replace = original
            after = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_run_refuses_unprovenance_and_writes_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.json").write_text(json.dumps({"status": "FAIL"}), encoding="utf-8")
            result = analyze_run(root)
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertTrue((root / "experiment_report.md").is_file())

    def test_disk_run_requires_explicit_gate_and_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "status.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
            (root / "config.json").write_text(json.dumps({field: "x" for field in analyzer_module.COMPATIBILITY_FIELDS}), encoding="utf-8")
            (root / "provenance.json").write_text("{}", encoding="utf-8")
            (root / "call_plan.json").write_text(json.dumps(build_call_plan()), encoding="utf-8")
            (root / "MANIFEST.sha256").write_text("", encoding="ascii")
            result = analyze_run(root)
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertIn("stage a", result["reason"].lower())
            (root / "status.json").write_text(json.dumps({"status": "COMPLETE", "gate": {"passed": True}}), encoding="utf-8")
            result = analyze_run(root)
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertIn("empty", result["reason"].lower())

    def test_disk_run_rejects_doubled_alpha_call_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = build_call_plan()
            plan[5]["alpha"] = float(plan[5]["alpha"]) * 2
            (root / "status.json").write_text(json.dumps({"status": "COMPLETE", "gate": {"passed": True}}), encoding="utf-8")
            (root / "config.json").write_text(json.dumps({}), encoding="utf-8")
            (root / "provenance.json").write_text(json.dumps({"source": "test"}), encoding="utf-8")
            (root / "call_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "MANIFEST.sha256").write_text("bad  call_plan.json\n", encoding="ascii")
            result = analyze_run(root)
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY")
            self.assertIn("call_plan", result["reason"])


if __name__ == "__main__":
    unittest.main()
