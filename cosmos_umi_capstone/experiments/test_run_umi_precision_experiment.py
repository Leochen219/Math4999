import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class PrecisionExperimentLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import run_umi_precision_experiment as module
        except ModuleNotFoundError:
            module = None
        cls.module = module

    def setUp(self):
        self.assertIsNotNone(self.module, "Task 4 remote launcher is missing")

    def test_lowest_unused_run_directory_never_reuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "umi_precision_contrast_run01").mkdir()
            (root / "umi_precision_contrast_run02").mkdir()
            selected = self.module.select_lowest_unused_run(root)
            self.assertEqual(selected.name, "umi_precision_contrast_run03")
            selected.mkdir()
            self.assertEqual(self.module.select_lowest_unused_run(root).name, "umi_precision_contrast_run04")

    def test_parser_requires_fixed_paths_and_preserves_resume_flag(self):
        args = self.module.parse_args([
            "--framework-root", "/framework", "--checkpoint-path", "/checkpoint",
            "--old-run09", "/old", "--run-root", "/runs", "--resume",
        ])
        self.assertTrue(args.resume)
        self.assertEqual(args.run_root, "/runs")
        self.assertEqual(args.alphas, list(self.module.ALPHAS))
        self.assertEqual(args.direction_seed, self.module.LEGACY_DIRECTION_SEED)

    def test_direction_bank_reuses_old_file_and_validates_hash_mask_and_rms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            carrier = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)
            mask = np.zeros_like(carrier, dtype=bool)
            mask[:, :, 0] = True
            bank = np.zeros((4,) + carrier.shape, dtype=np.float32)
            bank[0][mask] = 1.0
            np.save(root / "direction_bank.npy", bank, allow_pickle=False)
            with mock.patch.object(self.module, "sha256_file", return_value=self.module.LEGACY_DIRECTION_BANK_SHA256):
                loaded, metadata = self.module.load_legacy_direction_bank(root, carrier, mask)
            np.testing.assert_array_equal(loaded, bank)
            self.assertEqual(metadata["seed"], self.module.LEGACY_DIRECTION_SEED)
            self.assertEqual(metadata["bank_count"], 4)
            self.assertAlmostEqual(metadata["direction_0_masked_rms"], 1.0)

    def test_direction_seed_cannot_be_changed_to_generate_new_bank(self):
        with self.assertRaises(SystemExit):
            self.module.parse_args(["--direction-seed", "20260913"])

    def test_legacy_runtime_args_request_uniform_eager_backend(self):
        args = self.module.parse_args([])
        paths = {
            "framework_root": "/framework", "checkpoint_path": "/checkpoint",
            "vae_path": "/vae", "input_path": "/input", "action_path": "/action",
        }
        legacy = self.module._legacy_args(args, paths, Path("/setup"))
        self.assertFalse(legacy.use_torch_compile)

    def test_runtime_gate_rejects_compile_wrapper_or_unresolved_backend(self):
        class Model:
            def _get_velocity(self):
                return None
            def denoise(self):
                return None
            class Net:
                def forward(self):
                    return None
            net = Net()

        adapter = type("Adapter", (), {
            "runtime_setup": {"use_torch_compile": True}, "model": Model(),
        })()
        with self.assertRaisesRegex(self.module.BlockedExecution, "eager|compile"):
            self.module.verify_uniform_eager_runtime(adapter)

    def test_runtime_gate_accepts_explicit_eager_methods_and_records_backend(self):
        class Model:
            def _get_velocity(self):
                return None
            def denoise(self):
                return None
            class Net:
                def forward(self):
                    return None
                def _encode_text(self):
                    return None
                def _encode_vision(self):
                    return None
                def _encode_action(self):
                    return None
                def _decode_vision(self):
                    return None
                def _decode_action(self):
                    return None
            net = Net()

        adapter = type("Adapter", (), {
            "runtime_setup": {"use_torch_compile": False}, "model": Model(),
        })()
        evidence = self.module.verify_uniform_eager_runtime(adapter)
        self.assertEqual(evidence["requested_backend"], "eager")
        self.assertEqual(evidence["resolved_backend"], "eager")
        self.assertFalse(evidence["stale_wrappers"])
        self.assertIn("model.net.forward", evidence["checked_methods"])
        for name in ("_encode_text", "_encode_vision", "_encode_action", "_decode_vision", "_decode_action"):
            self.assertIn("model.net." + name, evidence["checked_methods"])
            self.assertEqual(evidence["owner_bindings"]["model.net." + name], "model.net")
        self.assertIn("model.net", evidence["checked_owners"])

    def test_runtime_gate_rejects_stale_encode_head_even_when_resolved_eager(self):
        class Net:
            def forward(self):
                return None
            def _encode_text(self):
                return None
            def _encode_vision(self):
                return None
            def _encode_action(self):
                return None
            def _decode_vision(self):
                return None
            def _decode_action(self):
                return None
        Net._encode_text._torchdynamo_orig_callable = object()
        class Model:
            net = Net()
            def _get_velocity(self):
                return None
            def denoise(self):
                return None
        adapter = type("Adapter", (), {"runtime_setup": {"use_torch_compile": False}, "model": Model()})()
        with self.assertRaisesRegex(self.module.BlockedExecution, "encode_text|compiled|dynamo"):
            self.module.verify_uniform_eager_runtime(adapter)

    def test_runtime_gate_rejects_encode_head_bound_to_another_network(self):
        class Net:
            def forward(self):
                return None
            def _encode_text(self):
                return None
            def _encode_vision(self):
                return None
            def _encode_action(self):
                return None
            def _decode_vision(self):
                return None
            def _decode_action(self):
                return None
        class Model:
            net = Net()
            def _get_velocity(self):
                return None
            def denoise(self):
                return None
        expected, wrong = Model.net, Net()
        expected._encode_text = wrong._encode_text
        adapter = type("Adapter", (), {"runtime_setup": {"use_torch_compile": False}, "model": Model()})()
        with self.assertRaisesRegex(self.module.BlockedExecution, "bound owner|model.net"):
            self.module.verify_uniform_eager_runtime(adapter)

    def test_runtime_gate_rejects_zero_required_network_methods(self):
        class Model:
            net = object()
        adapter = type("Adapter", (), {
            "runtime_setup": {"use_torch_compile": False}, "model": Model(),
        })()
        with self.assertRaisesRegex(self.module.BlockedExecution, "required|network"):
            self.module.verify_uniform_eager_runtime(adapter)

    def test_run_manifest_excludes_analysis_subtree_by_exact_relative_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.bin").write_bytes(b"raw")
            (root / "nested").mkdir()
            (root / "nested" / "MANIFEST.sha256").write_bytes(b"nested manifest is raw evidence")
            (root / "nested" / ".runner.lock").write_bytes(b"nested lock is raw evidence")
            (root / "precision_analysis").mkdir()
            (root / "precision_analysis" / "precision_summary.json").write_text("{}", encoding="utf-8")
            self.module._write_run_manifest(root)
            entries = (root / "MANIFEST.sha256").read_text(encoding="ascii")
            self.assertIn("raw.bin", entries)
            self.assertIn("nested/MANIFEST.sha256", entries)
            self.assertIn("nested/.runner.lock", entries)
            self.assertNotIn("precision_analysis/precision_summary.json", entries)

    def test_raw_manifest_reader_round_trips_nested_same_basename_evidence(self):
        import analyze_umi_precision_contrast as analyzer
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raw.bin").write_bytes(b"raw")
            (root / "nested").mkdir()
            (root / "nested" / "MANIFEST.sha256").write_bytes(b"nested manifest")
            (root / "nested" / ".runner.lock").write_bytes(b"nested lock")
            self.module._write_run_manifest(root)
            snapshot = analyzer._raw_manifest(root)
            self.assertIn("nested/MANIFEST.sha256", snapshot["entries"])
            self.assertIn("nested/.runner.lock", snapshot["entries"])
            self.assertNotIn("MANIFEST.sha256", snapshot["entries"])

    def _make_completed_run(self, root):
        import analyze_umi_precision_contrast as analyzer
        import umi_precision_runtime as runtime_api
        from run_umi_precision_experiment import _write_run_manifest
        from test_umi_precision_runtime import FakeRuntime

        shape = (1, 1, 3, 2, 2)
        mask = np.zeros(shape, dtype=bool)
        mask[:, :, 0] = True
        direction = np.zeros(shape, dtype=np.float32)
        direction[mask] = 1.0
        inputs = runtime_api.PrecisionInputs(np.full(shape, 1.001, dtype=np.float32), [0], mask, np.stack([direction]))
        runtime = FakeRuntime(runtime_api)
        result = runtime_api.run_precision_experiment(runtime, inputs, root, alphas=[.0001, .0003, .001, .003, .01, .03])
        self.assertEqual(result["status"], "complete")
        (root / "task4_execution.json").write_text(
            json.dumps({"source_dir": str(Path(__file__).resolve().parent)}) + "\n", encoding="utf-8")
        _write_run_manifest(root)
        analysis = analyzer.analyze_run(root, root / "precision_analysis")
        self.assertEqual(analysis["status"], "COMPLETE")
        return runtime

    def test_completed_launcher_resume_is_read_only_and_makes_zero_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "umi_precision_contrast_run17"
            runtime = self._make_completed_run(root)
            old = Path(temporary) / "old"
            old.mkdir()
            (old / "provenance.json").write_text("{}\n", encoding="utf-8")
            before = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            args = self.module.parse_args([
                "--framework-root", str(Path(temporary) / "framework"),
                "--checkpoint-path", str(Path(temporary) / "checkpoint"),
                "--old-run09", str(old), "--run-dir", str(root), "--resume", "--skip-old-reanalysis",
            ])
            paths = {"framework_root": "unused", "checkpoint_path": "unused", "vae_path": "unused",
                     "input_path": "unused", "action_path": "unused"}
            with mock.patch.object(self.module, "_resolve_inputs", return_value=paths), \
                 mock.patch.object(self.module, "require_torch_contract", side_effect=AssertionError("model gate must not run")):
                returned = self.module.execute_task4(args)
            self.assertTrue(returned["resume_read_only"])
            self.assertEqual(returned["analysis"]["status"], "COMPLETE")
            self.assertEqual(len(runtime.calls), 45)
            after = {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(before, after)

    def test_interrupted_resume_command_always_disables_old_reanalysis(self):
        import analyze_umi_precision_contrast as analyzer
        result = {"status": "BLOCKED", "metadata": {"task4_execution": {
            "run_status": "running", "python": "/env/bin/python", "source_dir": "/src",
            "framework_root": "/framework", "checkpoint_path": "/checkpoint", "old_run09": "/old",
            "run_dir": "/runs/umi_precision_contrast_run17", "skip_old_reanalysis": False,
        }}}
        commands = analyzer._commands_for_result(result)
        self.assertIn("--skip-old-reanalysis --resume", commands)

    def test_torch_gate_rejects_wrong_version_without_install_fallback(self):
        with self.assertRaisesRegex(self.module.BlockedExecution, "2.10.0\\+cu130"):
            self.module.require_torch_contract(type("Torch", (), {"__version__": "2.11.0+cu128", "version": type("Version", (), {"cuda": "12.8"})})())


if __name__ == "__main__":
    unittest.main()
