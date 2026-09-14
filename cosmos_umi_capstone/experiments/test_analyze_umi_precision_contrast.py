import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from umi_precision_primitives import fixed_noise_hash


class PrecisionContrastAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import analyze_umi_precision_contrast as module
        except ModuleNotFoundError:
            module = None
        cls.module = module

    def setUp(self):
        self.assertIsNotNone(self.module, "Task 4 precision contrast analyzer is missing")

    def _review_fixture(self, fixture, *, scope):
        """Build a production-shaped review source/run/sample tree."""
        source = fixture / "source"
        run = fixture / "raw_run"
        samples = run / "samples"
        source.mkdir(parents=True)
        samples.mkdir(parents=True)
        for name in self.module._REVIEW_SOURCE_FILES:
            (source / name).write_text(f"fixture:{name}\n", encoding="utf-8")
        (run / "config.json").write_text(json.dumps({"fixture": True}) + "\n", encoding="utf-8")
        (run / "status.json").write_text(json.dumps({"status": "complete", "scope": scope}) + "\n", encoding="utf-8")
        (run / "task4_execution.json").write_text(json.dumps({"source_dir": str(source)}) + "\n", encoding="utf-8")
        sample_dirs = {}
        names = self.module._REVIEW_REQUIRED_SAMPLE_ARRAY_FILES
        if scope == "full":
            names = (*names, "decoded_final.npy")
        for sample_id in self.module._REVIEW_SAMPLE_IDS:
            sample = samples / sample_id
            sample.mkdir()
            sample_dirs[sample_id] = str(sample)
            payload = {"sample_id": sample_id, "initial_noise_hash": "fixture", "noise_evidence": {}}
            (sample / "sample.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
            shape = (1, 1, 3, 1, 2)
            for name in names:
                array = np.zeros((3, 2, 2), dtype=np.float32) if name == "decoded_final.npy" else np.zeros(shape, dtype=np.float32)
                np.save(sample / name, array, allow_pickle=False)
            hashes = {"sample.json": self.module.sha256_file(sample / "sample.json")}
            hashes.update({name: self.module.sha256_file(sample / name) for name in names})
            (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": hashes}) + "\n", encoding="utf-8")
        return source, run, sample_dirs

    def _write_raw_manifest(self, run):
        entries = []
        for path in sorted(run.rglob("*")):
            if not path.is_file() or path.name == "MANIFEST.sha256" or "precision_analysis" in path.relative_to(run).parts:
                continue
            entries.append(f"{self.module.sha256_file(path)}  {path.relative_to(run).as_posix()}")
        (run / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="ascii")

    def _blocked_run(self, root):
        root.mkdir()
        (root / "status.json").write_text(json.dumps({
            "status": "blocked", "scope": "module", "diagnostic_attempts": 4,
            "formal_successful": 0, "identity": "blocked-id",
            "error": {"message": "hidden non-FP32 operation"},
        }), encoding="utf-8")
        (root / "config.json").write_text(json.dumps({"identity": "blocked-id"}) + "\n", encoding="utf-8")
        self._write_raw_manifest(root)

    def test_signed_response_decomposition_reconstructs_vector_identity(self):
        a = np.array([4.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 1.0], dtype=np.float32)
        c = np.array([1.0, 0.5], dtype=np.float32)
        result = self.module.decompose_precision_responses(a, b, c)
        np.testing.assert_array_equal(result["Delta_quant"], b - c)
        np.testing.assert_array_equal(result["Delta_compute"], a - b)
        np.testing.assert_array_equal(result["r_A_minus_r_C"], a - c)
        np.testing.assert_array_equal(result["reconstructed"], result["r_A_minus_r_C"])
        self.assertEqual(result["identity_error_rms"], 0.0)

    def test_window_thresholds_require_all_metrics_and_three_points(self):
        rows = [
            {"alpha": 1e-4, "input_cosine": 0.995, "plus_minus_input_cosine": 0.995,
             "plus_slope": 1.0, "plus_r2": 0.99, "minus_slope": 1.0, "minus_r2": 0.99,
             "paired_secant_cosine": 0.97, "paired_secant_relative_change": 0.1,
             "response_rms": 1.0, "group_repeat_floor": 0.01},
            {"alpha": 3e-4, "input_cosine": 0.995, "plus_minus_input_cosine": 0.995,
             "plus_slope": 1.0, "plus_r2": 0.99, "minus_slope": 1.0, "minus_r2": 0.99,
             "paired_secant_cosine": 0.97, "paired_secant_relative_change": 0.1,
             "response_rms": 1.0, "group_repeat_floor": 0.01},
            {"alpha": 1e-3, "input_cosine": 0.995, "plus_minus_input_cosine": 0.995,
             "plus_slope": 1.0, "plus_r2": 0.99, "minus_slope": 1.0, "minus_r2": 0.99,
             "paired_secant_cosine": 0.97, "paired_secant_relative_change": 0.1,
             "response_rms": 1.0, "group_repeat_floor": 0.01},
        ]
        decision = self.module.evaluate_window(rows, min_length=3)
        self.assertTrue(decision["selected"])
        self.assertEqual(decision["thresholds"]["input_cosine"], 0.99)
        rows[-1]["paired_secant_cosine"] = 0.94
        rejected = self.module.evaluate_window(rows, min_length=3)
        self.assertFalse(rejected["selected"])
        self.assertIn("paired_secant_cosine", rejected["failure_reasons"])

    def test_window_rejects_minus_side_adjacent_input_cosine_even_when_plus_passes(self):
        rows = [
            {"alpha": 1e-4, "plus_input_cosine": 0.995, "minus_input_cosine": 0.995,
             "plus_minus_input_cosine": 0.995, "plus_slope": 1.0, "plus_r2": 0.99,
             "minus_slope": 1.0, "minus_r2": 0.99, "paired_secant_cosine": 0.97,
             "paired_secant_relative_change": 0.1, "response_rms": 1.0, "group_repeat_floor": 0.01},
            {"alpha": 3e-4, "plus_input_cosine": 0.995, "minus_input_cosine": 0.80,
             "plus_minus_input_cosine": 0.995, "plus_slope": 1.0, "plus_r2": 0.99,
             "minus_slope": 1.0, "minus_r2": 0.99, "paired_secant_cosine": 0.97,
             "paired_secant_relative_change": 0.1, "response_rms": 1.0, "group_repeat_floor": 0.01},
            {"alpha": 1e-3, "plus_input_cosine": 0.995, "minus_input_cosine": 0.995,
             "plus_minus_input_cosine": 0.995, "plus_slope": 1.0, "plus_r2": 0.99,
             "minus_slope": 1.0, "minus_r2": 0.99, "paired_secant_cosine": 0.97,
             "paired_secant_relative_change": 0.1, "response_rms": 1.0, "group_repeat_floor": 0.01},
        ]
        decision = self.module.evaluate_window(rows, min_length=3)
        self.assertFalse(decision["selected"])
        self.assertTrue(any("minus_input_cosine" in reason for reason in decision["failure_reasons"]))

    def test_window_nonfinite_input_is_an_explicit_na_failure(self):
        rows = [
            {"alpha": alpha, "input_cosine": 0.995, "plus_input_cosine": 0.995, "minus_input_cosine": 0.995,
             "plus_minus_input_cosine": 0.995, "plus_slope": 1.0, "plus_r2": 0.99,
             "minus_slope": 1.0, "minus_r2": 0.99, "paired_secant_cosine": 0.97,
             "paired_secant_relative_change": 0.1, "response_rms": 1.0, "group_repeat_floor": 0.01}
            for alpha in (1e-4, 3e-4, 1e-3)
        ]
        rows[1]["minus_input_cosine"] = float("nan")
        decision = self.module.evaluate_window(rows, min_length=3)
        self.assertFalse(decision["selected"])
        self.assertTrue(any("N/A" in reason for reason in decision["failure_reasons"]))

    def test_windows_do_not_bridge_a_missing_alpha_from_the_fixed_grid(self):
        records = self._records(scope="module")
        records = {
            key: value for key, value in records.items()
            if not (value["spec"]["kind"] == "perturbation" and value["spec"]["alpha"] == 3e-4)
        }
        result = self.module.analyze_records(records)
        bridged = [
            decision for decision in result["window_decisions"]
            if decision.get("start_alpha") == 1e-4 and decision.get("end_alpha") == 3e-3
        ]
        self.assertTrue(bridged)
        self.assertTrue(all("missing" in " ".join(decision["failure_reasons"]).lower() for decision in bridged))

    def test_pair_metrics_report_target_derivative_actual_secant_and_one_sided_consistency(self):
        result = self.module.analyze_records(self._records(scope="module"))
        pair = next(row for row in result["paired_metrics"] if row["group"] == "A" and row["alpha"] == 1e-4)
        for key in ("target_centered_derivative_rms", "actual_paired_secant_rms",
                    "one_sided_derivative_cosine", "one_sided_relative_change"):
            self.assertIn(key, pair)
        self.assertIsNotNone(pair["target_centered_derivative_rms"])
        self.assertIsNotNone(pair["actual_paired_secant_rms"])

    def test_package_saves_signed_delta_tensors_outside_the_review_bundle(self):
        records = self._records(scope="full")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(records, root)
            tensors = sorted((root / "analysis_tensors" / "Delta_quant").glob("*.npy"))
            self.assertTrue(tensors)
            bundle_names = __import__("zipfile").ZipFile(root / "review_bundle.zip").namelist()
            self.assertFalse(any(name.endswith(".npy") for name in bundle_names))

    def test_analyze_records_marks_rgb_na_for_module_fallback_and_reports_deltas(self):
        records = self._records(scope="module")
        result = self.module.analyze_records(records)
        self.assertEqual(result["formal_call_count"], 42)
        self.assertEqual(result["scope"], "module")
        self.assertIn("predicted_latent", result["spaces"])
        self.assertIn("decoded_final_rgb", result["spaces"])
        self.assertEqual(result["spaces"]["decoded_final_rgb"]["status"], "N/A")
        self.assertIn("fallback", result["spaces"]["decoded_final_rgb"]["reason"].lower())
        self.assertTrue(result["vector_identity"]["passed"])
        self.assertTrue(result["input_identity"]["A_B_all_exact"])
        self.assertTrue(result["compute_identity"]["B_C_same_path"])

    def test_complete_is_blocked_when_recorded_run_status_is_blocked(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "blocked", "scope": "module"}})
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("recorded run status" in reason for reason in result["blocked_reasons"]))

    def test_complete_is_blocked_for_mixed_scopes(self):
        records = self._records(scope="module")
        records["B_pre"]["scope"] = "full"
        result = self.module.analyze_records(records)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["scope"], "mixed")

    def test_complete_is_blocked_when_full_rgb_output_is_missing(self):
        records = self._records(scope="full")
        records["C_pre"].pop("decoded_final")
        result = self.module.analyze_records(records)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("RGB" in reason or "decoded" in reason for reason in result["blocked_reasons"]))

    def test_complete_rejects_broadcasting_compatible_wrong_latent_shape(self):
        records = self._records(scope="module")
        records["A_pre"]["predicted_latent"] = np.zeros((1, 1, 1, 1, 2), dtype=np.float32)
        result = self.module.analyze_records(records)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("shape" in reason.lower() or "output" in reason.lower() for reason in result["blocked_reasons"]))

    def test_complete_recomputes_noise_hashes_instead_of_trusting_recorded_hash(self):
        records = self._records(scope="module")
        records["A_pre"]["initial_noise_hash"] = "forged"
        result = self.module.analyze_records(records)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("noise" in reason.lower() for reason in result["blocked_reasons"]))

    def test_package_writes_manifest_and_deterministic_review_bundle(self):
        records = self._records(scope="full")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.module.write_precision_artifacts(records, root)
            expected = {
                "precision_summary.json", "precision_metrics.csv", "window_decisions.csv",
                "vector_decomposition.csv", "precision_report_zh.md", "precision_path_note.md",
                "commands.md", "gpu_timing_summary.json", "precision_manifest.sha256",
                "review_bundle.zip", "response_rms.png", "response_rms.svg",
                "window_decisions.png", "window_decisions.svg",
            }
            self.assertTrue(expected.issubset({item.name for item in root.iterdir()}))
            self.assertEqual(result["status"], "COMPLETE")
            first_bundle = (root / "review_bundle.zip").read_bytes()
            first_manifest = (root / "precision_manifest.sha256").read_bytes()
            with self.assertRaises(FileExistsError):
                self.module.write_precision_artifacts(records, root)
            with tempfile.TemporaryDirectory() as revision_temp:
                revision = Path(revision_temp) / "revision-02"
                result_again = self.module.write_precision_artifacts(records, revision)
                self.assertEqual(result_again["status"], "COMPLETE")
                self.assertEqual(first_bundle, (revision / "review_bundle.zip").read_bytes())
                self.assertEqual(first_manifest, (revision / "precision_manifest.sha256").read_bytes())
            summary = json.loads((root / "precision_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["formal_call_count"], 42)
            self.assertIn("N/A", (root / "precision_report_zh.md").read_text(encoding="utf-8"))

    def test_package_commands_and_gpu_summary_bind_actual_execution_metadata(self):
        metadata = {
            "run_dir": "run-dir",
            "status": {"status": "module_complete", "scope": "module"},
            "config": {},
            "diagnostic_count": 4,
            "task4_execution": {
                "python": "python-bin",
                "framework_root": "framework-root",
                "checkpoint_path": "checkpoint-path",
                "old_run09": "old-run",
                "run_dir": "run-dir",
                "reanalysis_output": "reanalysis-output",
                "input_path": "input-path", "action_path": "action-path", "vae_path": "vae-path",
                "direction_seed": 20260912, "skip_old_reanalysis": True,
                "device": "cuda:0", "device_name": "NVIDIA Test GPU", "peak_memory_bytes": 123456,
                "requested_backend": "eager", "resolved_backend": "eager",
            },
        }
        result = self.module.analyze_records(self._records(scope="module"), metadata=metadata)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            commands = (root / "commands.md").read_text(encoding="utf-8")
            self.assertIn("python-bin", commands)
            self.assertIn("--old-run09 old-run", commands)
            self.assertIn("--direction-seed 20260912", commands)
            self.assertIn("--skip-old-reanalysis", commands)
            self.assertIn("--run-dir run-dir", commands)
            telemetry = json.loads((root / "gpu_timing_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(telemetry["device"], "cuda:0")
            self.assertEqual(telemetry["peak_memory_bytes"], 123456)
            self.assertEqual(telemetry["resolved_backend"], "eager")
            report = (root / "precision_report_zh.md").read_text(encoding="utf-8")
            self.assertIn("GPU/timing", report)
            self.assertIn("CPU 校准", report)

    def test_package_has_required_numeric_alpha_figures_and_na_annotations(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            expected = {
                "input_distortion.png", "input_distortion.svg",
                "latent_rgb_response.png", "latent_rgb_response.svg",
                "derivative_consistency.png", "derivative_consistency.svg",
                "delta_decomposition.png", "delta_decomposition.svg",
            }
            self.assertTrue(expected.issubset({item.name for item in root.iterdir()}))
            for name in ("input_distortion.svg", "latent_rgb_response.svg", "derivative_consistency.svg", "delta_decomposition.svg"):
                content = (root / name).read_text(encoding="utf-8")
                self.assertRegex(content, r"1e-04|0\.0001|alpha")
            from PIL import Image
            for name in ("input_distortion.png", "latent_rgb_response.png", "derivative_consistency.png", "delta_decomposition.png",
                         "response_rms.png", "window_decisions.png", "precision_response.png", "precision_windows.png"):
                with Image.open(root / name) as image:
                    self.assertEqual(image.size, (1200, 560))
            self.assertIn("N/A", (root / "latent_rgb_response.svg").read_text(encoding="utf-8"))

    def test_figures_reserve_legend_column_for_delta_labels(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            svg = (root / "delta_decomposition.svg").read_text(encoding="utf-8")
            self.assertIn("width='1200'", svg)
            self.assertIn("predicted_latent Delta_quant", svg)
            self.assertIn("predicted_latent Delta_compute", svg)
            self.assertIn("alpha (numeric; log10 position for positive alpha)", svg)

    def test_positive_amplitude_charts_use_log_y_and_slope_one_reference(self):
        result = self.module.analyze_records(self._records(scope="full"), metadata={"status": {"status": "complete", "scope": "full"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            for name in ("response_rms.svg", "latent_rgb_response.svg", "input_distortion.svg", "delta_decomposition.svg"):
                content = (root / name).read_text(encoding="utf-8")
                self.assertIn("log-y", content, name)
            for name in ("response_rms.svg", "latent_rgb_response.svg"):
                content = (root / name).read_text(encoding="utf-8")
                self.assertIn("slope=1 ref (visual only)", content, name)
                self.assertIn("target input RMS", content, name)
                self.assertIn("not an amplitude/fit comparison", content, name)

    def test_derivative_figure_uses_adjacent_paired_secants_and_both_thresholds(self):
        result = self.module.analyze_records(self._records(scope="full"), metadata={"status": {"status": "complete", "scope": "full"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            content = (root / "derivative_consistency.svg").read_text(encoding="utf-8")
            self.assertIn("adjacent paired-secant cosine", content)
            self.assertIn("adjacent paired-secant relative change", content)
            self.assertIn("threshold 0.95", content)
            self.assertIn("threshold 0.25", content)
            self.assertIn(">-1<", content)
            self.assertIn(">1<", content)
            self.assertNotIn("one-sided cosine", content)
            self.assertNotIn("one-sided relative change", content)

    def test_report_contains_numeric_group_space_conclusions_and_evidence_boundary(self):
        result = self.module.analyze_records(self._records(scope="full"), metadata={"status": {"status": "complete", "scope": "full"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            report = (root / "precision_report_zh.md").read_text(encoding="utf-8")
            for token in ("A / predicted_latent", "B / decoded_final_rgb", "C / predicted_latent",
                          "窗口数", "全幅拟合", "最小 alpha", "最大 alpha", "baseline floor",
                          "actual/target", "input_direction_cosine", "nonzero ratio",
                          "42+diagnostic", "耗时", "显存", "Delta 仅是范数", "证据边界"):
                self.assertIn(token, report)
            self.assertRegex(report, r"A / predicted_latent.*1e-04")

    def test_review_bundle_contains_whitelisted_runtime_and_representative_raw_evidence_only(self):
        result = self.module.analyze_records(self._records(scope="full"), metadata={"status": {"status": "complete", "scope": "full"}})
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            source, run, sample_dirs = self._review_fixture(fixture, scope="full")
            (source / "not_allowed_secret.py").write_text("secret\n", encoding="utf-8")
            result["review_evidence_sources"] = {"source_dir": str(source), "run_dir": str(run), "sample_dirs": sample_dirs}
            root = fixture / "analysis"
            self.module.write_precision_artifacts(result, root)
            names = set(__import__("zipfile").ZipFile(root / "review_bundle.zip").namelist())
            self.assertIn("review_evidence/source/run_umi_precision_experiment.py", names)
            self.assertIn("review_evidence/run/config.json", names)
            self.assertIn("review_evidence/samples/A_pre/predicted_latent.npy", names)
            self.assertIn("review_evidence/samples/C_alpha_05_minus/decoded_final.npy", names)
            self.assertNotIn("review_evidence/source/not_allowed_secret.py", names)
            self.assertFalse(any(name.startswith("analysis_tensors/") for name in names))
            self.assertIn("review_evidence_manifest.json", names)
            evidence_manifest = json.loads((root / "review_evidence_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence_manifest["status"], "COMPLETE")
            self.assertLess((root / "review_bundle.zip").stat().st_size, 2_000_000)

    def test_review_evidence_rejects_unsafe_source_path(self):
        result = self.module.analyze_records(self._records(scope="full"), metadata={"status": {"status": "complete", "scope": "full"}})
        result["review_evidence_sources"] = {"source_dir": "..\\outside", "sample_dirs": {}}
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "source|unsafe|evidence"):
                self.module.write_precision_artifacts(result, Path(temporary) / "analysis")

    def test_review_evidence_accepts_absolute_posix_style_source_dir(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            source, run, sample_dirs = self._review_fixture(Path(temporary), scope="module")
            result["review_evidence_sources"] = {"source_dir": source.as_posix(), "run_dir": run.as_posix(), "sample_dirs": sample_dirs}
            root = Path(temporary) / "analysis"
            self.module.write_precision_artifacts(result, root)
            manifest = json.loads((root / "review_evidence_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("review_evidence/source/run_umi_precision_experiment.py", {item["path"] for item in manifest["entries"]})

    def test_module_review_evidence_is_complete_without_rgb_or_optional_arrays(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary)
            source, run, sample_dirs = self._review_fixture(fixture, scope="module")
            (source / "__init__.py").unlink()
            result["review_evidence_sources"] = {"source_dir": source.as_posix(), "run_dir": run.as_posix(), "sample_dirs": sample_dirs}
            root = fixture / "analysis"
            self.module.write_precision_artifacts(result, root)
            manifest = json.loads((root / "review_evidence_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertIn("sample/A_pre/decoded_final.npy", manifest["optional_missing"])

    def test_runner_source_whitelist_tracks_real_bridge_scan_and_historical_hashes(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            source, run, sample_dirs = self._review_fixture(Path(temporary), scope="module")
            source = source.rename(Path(temporary) / "historical_source")
            expected_core = {
                "analyze_umi_precision_contrast.py", "run_umi_precision_experiment.py",
                "umi_fd_post_vae_bridge.py", "umi_fd_post_vae_scan.py", "umi_precision_official.py",
                "umi_precision_runtime.py", "umi_precision_storage.py", "umi_precision_primitives.py",
                "umi_precision_reanalysis.py", "umi_precision_identity.py", "umi_precision_calibration.py",
                "umi_precision_calibration_math.md",
            }
            for name in expected_core:
                (source / name).write_text(f"historical:{name}\n", encoding="utf-8")
            # This file is intentionally absent: it is not a required bridge.
            result["review_evidence_sources"] = {"source_dir": source.as_posix(), "run_dir": run.as_posix(), "sample_dirs": sample_dirs}
            root = Path(temporary) / "analysis"
            self.module.write_precision_artifacts(result, root)
            manifest = json.loads((root / "review_evidence_manifest.json").read_text(encoding="utf-8"))
            entries = {item["path"]: item for item in manifest["entries"]}
            for name in ("umi_fd_post_vae_bridge.py", "umi_fd_post_vae_scan.py"):
                relative = f"review_evidence/source/{name}"
                self.assertIn(relative, entries)
                self.assertEqual(entries[relative]["role"], "runner_source")
                self.assertEqual(entries[relative]["sha256"], self.module.sha256_file(source / name))
            self.assertNotIn("runner_source/umi_precision_torch.py", manifest["missing"])

    def test_runner_source_symlink_is_rejected(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlink is unavailable")
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "historical_source"
            source.mkdir()
            target = source / "real.py"
            target.write_text("not the runner\n", encoding="utf-8")
            try:
                os.symlink(target, source / "run_umi_precision_experiment.py")
            except OSError as error:
                self.skipTest(f"symlink creation unavailable: {error}")
            result["review_evidence_sources"] = {"source_dir": source.as_posix(), "sample_dirs": {}}
            with self.assertRaisesRegex(ValueError, "regular file|symlink|evidence"):
                self.module.write_precision_artifacts(result, Path(temporary) / "analysis")

    def test_terminal_blocked_commands_never_offer_resume(self):
        result = self.module.analyze_records(
            self._records(scope="module"),
            metadata={"status": {"status": "blocked", "scope": "module"},
                      "task4_execution": {"run_status": "blocked", "run_dir": "/runs/umi_precision_contrast_run09"}},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            commands = (root / "commands.md").read_text(encoding="utf-8")
            self.assertNotIn("--resume", commands)
            self.assertIn("not resumable", commands)

    def test_package_bundles_verified_calibration_and_math_note(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            calibration = root / "calibration"
            expected = {
                "calibration_report.md", "config.json", "error_plot.png", "error_plot.svg",
                "hashes.json", "metrics.csv", "metrics.json", "umi_precision_calibration_math.md",
            }
            self.assertTrue(expected.issubset({path.name for path in calibration.rglob("*") if path.is_file()}))
            manifest = (root / "calibration_manifest.sha256").read_text(encoding="ascii")
            self.assertIn("calibration/metrics.json", manifest)
            bundle_names = __import__("zipfile").ZipFile(root / "review_bundle.zip").namelist()
            self.assertIn("calibration/umi_precision_calibration_math.md", bundle_names)

    def test_package_rejects_tampered_calibration_artifact(self):
        result = self.module.analyze_records(self._records(scope="module"), metadata={"status": {"status": "module_complete", "scope": "module"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(result, root)
            (root / "calibration" / "metrics.json").write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash|manifest"):
                self.module._analysis_manifest(root)

    def test_real_runtime_writer_round_trips_top_level_arrays_and_reaches_complete(self):
        from run_umi_precision_experiment import _write_run_manifest
        from test_umi_precision_runtime import FakeRuntime
        import umi_precision_runtime as runtime_api

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = FakeRuntime(runtime_api)
            shape = (1, 1, 3, 2, 2)
            mask = np.zeros(shape, dtype=bool)
            mask[:, :, 0] = True
            direction = np.zeros(shape, dtype=np.float32)
            direction[mask] = 1.0
            inputs = runtime_api.PrecisionInputs(np.full(shape, 1.001, dtype=np.float32), [0], mask, np.stack([direction]))
            alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
            run_result = runtime_api.run_precision_experiment(runtime, inputs, root, alphas=alphas)
            self.assertEqual(run_result["status"], "complete")
            # Exercise the same production review-evidence path as a real
            # runner: task4_execution binds the run to this source snapshot.
            (root / "task4_execution.json").write_text(
                json.dumps({"source_dir": str(Path(__file__).resolve().parent)}) + "\n", encoding="utf-8")
            _write_run_manifest(root)
            records, metadata = self.module._load_precision_records(root)
            self.assertEqual(records["A_pre"]["predicted_latent"].shape, (1, 1, 2, 2, 2))
            analysis = self.module.analyze_run(root, root / "precision_analysis")
            self.assertEqual(analysis["status"], "COMPLETE")

    def test_persisted_signed_vectors_are_fp64_and_hashes_match_reload_with_cancellation(self):
        records = self._records(scope="module")
        records["B_alpha_00_plus"]["predicted_latent"] = records["B_alpha_00_plus"]["predicted_latent"].copy()
        records["B_alpha_00_plus"]["predicted_latent"].flat[0] += np.float32(0.125)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.module.write_precision_artifacts(records, root)
            row = next(row for row in json.loads((root / "precision_summary.json").read_text(encoding="utf-8"))["vector_decomposition"]
                       if row["space"] == "predicted_latent" and row["alpha"] == 1e-4 and row["sign"] == 1)
            delta = np.load(root / row["Delta_quant_path"], allow_pickle=False)
            compute = np.load(root / row["Delta_compute_path"], allow_pickle=False)
            total = np.load(root / row["r_A_minus_r_C_path"], allow_pickle=False)
            reconstructed = np.load(root / row["reconstructed_path"], allow_pickle=False)
            self.assertEqual(delta.dtype, np.float64)
            self.assertGreater(float(np.linalg.norm(delta)), 0.0)
            np.testing.assert_array_equal(delta + compute, reconstructed)
            np.testing.assert_array_equal(total, reconstructed)
            self.assertEqual(row["Delta_quant_sha256"], self.module._sha256_array(delta))
            self.assertEqual(row["reconstructed_sha256"], self.module._sha256_array(reconstructed))

    def test_nonempty_mixed_record_scopes_cannot_be_masked_by_root_scope(self):
        records = self._records(scope="module")
        records["B_pre"]["scope"] = "full"
        result = self.module.analyze_records(records, metadata={"status": {"status": "module_complete", "scope": "module"}})
        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any("scope" in reason.lower() for reason in result["blocked_reasons"]))

    def test_blocked_package_marks_unavailable_series_na_inside_each_requested_svg(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            output = run / "precision_analysis"
            self.module.analyze_run(run, output)
            for name in ("input_distortion.svg", "latent_rgb_response.svg", "derivative_consistency.svg", "delta_decomposition.svg"):
                self.assertIn("N/A", (output / name).read_text(encoding="utf-8"))

    def test_package_cli_has_no_overwrite_escape_hatch(self):
        import package_umi_precision_contrast as package
        args = package.parse_args(["--run-dir", "/run", "--output-dir", "/run/precision_analysis"])
        self.assertFalse(hasattr(args, "overwrite"))
        with self.assertRaises(SystemExit):
            package.parse_args(["--run-dir", "/run", "--output-dir", "/run/precision_analysis", "--overwrite"])

    def test_blocked_run_without_formal_samples_publishes_na_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            output = run / "precision_analysis"
            result = self.module.analyze_run(run, output)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertEqual(result["formal_call_count"], 0)
            self.assertEqual(result["diagnostic_call_count"], 4)
            self.assertEqual(result["scope"], "module")
            self.assertEqual(result["spaces"]["decoded_final_rgb"]["status"], "N/A")
            self.assertIn("fallback", result["spaces"]["decoded_final_rgb"]["reason"].lower())
            self.assertFalse(result["vector_identity"]["passed"])
            self.assertEqual(result["fit_metrics"], [])
            self.assertTrue((output / "review_bundle.zip").is_file())

    def test_analyze_run_rejects_extra_raw_file_outside_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            (run / "unexpected.bin").write_bytes(b"out of scope")
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.module.analyze_run(run, run / "precision_analysis")

    def test_analyze_run_rejects_raw_manifest_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            manifest = run / "MANIFEST.sha256"
            lines = manifest.read_text(encoding="ascii").splitlines()
            digest, relative = lines[0].split("  ", 1)
            lines[0] = "0" * 64 + "  " + relative
            manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "manifest"):
                self.module.analyze_run(run, run / "precision_analysis")

    def test_analyze_run_rejects_per_sample_artifact_hash_mismatch(self):
        from umi_precision_storage import PrecisionSampleStore

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            record = dict(self._records(scope="module")["A_pre"])
            record["identity"] = "blocked-id"
            store = PrecisionSampleStore(run / "samples")
            store.write_success("A_pre", record)
            artifact = next((run / "samples" / "A_pre").glob("*.npy"))
            artifact.write_bytes(artifact.read_bytes() + b"tamper")
            self._write_raw_manifest(run)
            with self.assertRaisesRegex(ValueError, "artifact|hash"):
                self.module.analyze_run(run, run / "precision_analysis")

    def test_analysis_output_must_be_under_run_precision_analysis_subtree(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            with self.assertRaisesRegex(ValueError, "output"):
                self.module.analyze_run(run, Path(temporary) / "outside")

    def test_completed_analysis_resume_returns_existing_matching_package_and_rejects_stale_extra(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            output = run / "precision_analysis"
            first = self.module.analyze_run(run, output)
            summary_before = (output / "precision_summary.json").read_bytes()
            second = self.module.analyze_run(run, output)
            self.assertEqual(first["status"], second["status"])
            self.assertEqual(summary_before, (output / "precision_summary.json").read_bytes())
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stale|analysis"):
                self.module.analyze_run(run, output)

    def test_completed_analysis_resume_rejects_old_package_missing_required_deliverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            self._blocked_run(run)
            output = run / "precision_analysis"
            self.module.analyze_run(run, output)
            (output / "delta_decomposition.svg").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete|analysis"):
                self.module.analyze_run(run, output)

    def _records(self, *, scope):
        shape = (1, 1, 3, 1, 2)
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, 0] = True
        z_bar = np.ones(shape, dtype=np.float32)
        direction = np.zeros(shape, dtype=np.float32)
        direction[mask] = 1.0
        records = {}
        alphas = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
        for group in "ABC":
            baseline = np.full((1, 1, 2, 1, 2), 0.25 + 0.01 * (group == "A"), dtype=np.float32)
            for kind, alpha, sign in [("pre", 0.0, 0), ("post", 0.0, 0)]:
                records[f"{group}_{kind}"] = self._record(
                    f"{group}_{kind}", group, kind, alpha, sign, z_bar, mask, direction,
                    baseline, scope, baseline_shift=(0.001 if kind == "post" else 0.0)
                )
            for index, alpha in enumerate(alphas):
                for sign in (1, -1):
                    response = baseline + np.float32(sign * alpha * 10.0)
                    records[f"{group}_alpha_{index:02d}_{'plus' if sign == 1 else 'minus'}"] = self._record(
                        f"{group}_alpha_{index:02d}_{'plus' if sign == 1 else 'minus'}",
                        group, "perturbation", alpha, sign, z_bar, mask, direction,
                        response, scope
                    )
        return records

    @staticmethod
    def _record(sample_id, group, kind, alpha, sign, z_bar, mask, direction, output, scope, baseline_shift=0.0):
        common = z_bar.copy()
        if sign:
            common[mask] += np.float32(sign * alpha * 10.0)
        expected_steps = 1 if scope == "module" else 2
        evidence = [
            {"role": "weights", "native_dtype": "bfloat16" if group == "A" else "float32"},
            {"role": "common_condition", "native_dtype": "float32"},
            {"role": "network_condition", "native_dtype": "bfloat16" if group == "A" else "float32"},
            {"role": "activation", "native_dtype": "bfloat16" if group == "A" else "float32", "step": 0},
            {"role": "denoiser_output", "native_dtype": "bfloat16" if group == "A" else "float32", "step": 0},
        ]
        if scope == "full":
            evidence.extend({"role": role, "native_dtype": "float32", "step": 0} for role in (
                "sampler_update", "sampler_velocity", "sampler_accumulator", "sampler_converted"))
        record = {
            "sample_id": sample_id,
            "spec": {"sample_id": sample_id, "group": group, "kind": kind, "alpha": alpha,
                      "sign": sign, "direction_index": 0, "model_seed": 0},
            "scope": scope,
            "common_input_fp32": common,
            "z_bar": z_bar.copy(),
            "mask": mask.copy(),
            "direction": direction.copy(),
            "initial_state": z_bar.copy(),
            "consumed_initial_state": z_bar.copy(),
            "sampler_input_state": z_bar.copy(),
            "consumed_initial_mask": mask.copy(),
            "condition_steps_fp32": np.stack([common] * expected_steps),
            "steps": [{"step": index, "timestep": 999 - index} for index in range(expected_steps)],
            "expected_steps": expected_steps,
            "noise_evidence": {"source": "first_velocity_input", "seed": 0, "prepare_seed": 0, "batch_size": 1},
            "request_initial_state": {}, "request_final_state": {},
            "cache": {"requested": False, "installed": False, "initial_empty": True, "final_empty": True},
            "execution": {"autocast": False, "tf32_matmul": False, "tf32_cudnn": False,
                           "backend": "fixture", "casts": [], "dispatch_observed": True, "operation_count": 1,
                           "operation_dtypes": {"float32": 1}},
            "tensor_evidence": evidence,
            "initial_noise_hash": fixed_noise_hash(z_bar, mask),
            "predicted_latent": output + np.float32(baseline_shift),
            "actual_delta_fp32": common - z_bar,
            "target_delta_fp32": common - z_bar,
            "primary_quantity": "predicted_predecode_latent" if scope == "full" else "step_0_predicted_denoiser_output",
        }
        if scope == "full":
            record["decoded_final"] = np.full((3, 2, 2), float(output.mean() + np.float32(baseline_shift)), dtype=np.float32)
        return record


if __name__ == "__main__":
    unittest.main()
