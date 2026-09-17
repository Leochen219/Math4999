import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


import run_umi_task6_decoder_only as entry


class DecoderOnlyEntryTests(unittest.TestCase):
    def _args(self, root, **overrides):
        values = dict(
            raw_root=Path(root) / "raw",
            decoder_root=Path(root) / "decoder",
            analysis_root=Path(root) / "analysis",
            control_root=Path(root) / "control",
            launch_contract=Path(root) / "contract.json",
            expected_raw_manifest=None,
            framework_root=Path(root) / "framework",
            checkpoint=Path(root) / "checkpoint.ckpt",
            vae=Path(root) / "vae.pt",
            loader="fixture:load",
            action=Path(root) / "action.json",
            video=Path(root) / "video.mp4",
            task5_root=Path(root) / "task5",
            gpu_index=0,
            resume=False,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def _raw(self, root):
        root.mkdir(parents=True)
        (root / "run_status.json").write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
        (root / "MANIFEST.sha256").write_text("a" * 64 + "  evidence.bin\n", encoding="ascii")
        (root / "evidence.bin").write_bytes(b"raw")

    def _contract(self, path):
        payload = {"code_bundle_sha256": "c" * 64, "task5": {}, "prompt": "decoder"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _safe_samplers(self):
        row = {"gpu_used_gib": 1.0, "gpu_free_gib": 100.0, "gpu_reserved_gib": 0.0,
               "ram_available_gib": 600.0, "rss_gib": 0.0, "swap_used_gib": 0.0,
               "disk_free_gib": 100.0}
        return {name: (lambda row=row: dict(row)) for name in ("gpu", "ram", "disk")}

    def test_preload_rejection_does_not_build_or_generate(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(temporary)
            self._raw(args.raw_root)
            contract = self._contract(args.launch_contract)
            raw_hashes = (entry.sha256_file(args.raw_root / "MANIFEST.sha256"),
                          entry.sha256_file(args.raw_root / "run_status.json"))
            factory = mock.Mock()
            monitor = mock.Mock()
            monitor.check.return_value = {"status": "HARD_STOP", "reason_code": "GPU_PRELOAD"}
            with mock.patch.object(entry, "load_json", return_value=contract), \
                 mock.patch.object(entry, "_verify_raw_task6", return_value=(raw_hashes[0], {"status": "COMPLETE"})), \
                 mock.patch.object(entry, "_validate_static_contract"), \
                 mock.patch.object(entry, "verify_pinned_bridge_assets", return_value={}), \
                 mock.patch.object(entry, "extract_task5_directions", return_value={"bank": "bank"}), \
                 mock.patch.object(entry, "OfficialRuntimeFactory", return_value=factory), \
                 mock.patch.object(entry, "resource_samplers", return_value=self._safe_samplers()), \
                 mock.patch.object(entry, "ResourceMonitor", return_value=monitor), \
                 mock.patch.object(entry, "run_task6_decoder_replays") as replay:
                result = entry.execute_decoder_only(args)
            self.assertEqual(result["status"], "RESOURCE_STOP")
            self.assertFalse(result["generation_started"])
            factory.build.assert_not_called()
            replay.assert_not_called()
            self.assertEqual(raw_hashes, (entry.sha256_file(args.raw_root / "MANIFEST.sha256"),
                                          entry.sha256_file(args.raw_root / "run_status.json")))

    def test_success_runs_sixteen_replays_before_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(temporary)
            self._raw(args.raw_root)
            contract = self._contract(args.launch_contract)
            factory = mock.Mock()
            runtime = mock.Mock()
            encoder = mock.Mock()
            factory.build.return_value = (runtime, mock.Mock(), encoder)
            monitor = mock.Mock()
            monitor.check.return_value = {"status": "OK"}
            order = []
            with mock.patch.object(entry, "load_json", return_value=contract), \
                 mock.patch.object(entry, "_verify_raw_task6", return_value=("a" * 64, {"status": "COMPLETE"})), \
                 mock.patch.object(entry, "_validate_static_contract"), \
                 mock.patch.object(entry, "verify_pinned_bridge_assets", return_value={}), \
                 mock.patch.object(entry, "extract_task5_directions", return_value={"bank": "bank"}), \
                 mock.patch.object(entry, "OfficialRuntimeFactory", return_value=factory), \
                 mock.patch.object(entry, "resource_samplers", return_value=self._safe_samplers()), \
                 mock.patch.object(entry, "ResourceMonitor", side_effect=[monitor, monitor]), \
                 mock.patch.object(entry, "run_task6_decoder_replays", side_effect=lambda *a, **k: order.append("decode") or {"status": "COMPLETE", "decoder_calls": 16}), \
                 mock.patch.object(entry, "analyze_task6_run", side_effect=lambda *a, **k: order.append("analysis") or {"status": "COMPLETE"}):
                result = entry.execute_decoder_only(args)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(order, ["decode", "analysis"])
            self.assertEqual(result["decoder_calls"], 16)
            self.assertFalse(result["generation_started"])
            runtime.cleanup.assert_called_once()
            factory.unload.assert_called_once()
            self.assertEqual(result["raw_manifest_sha256_before"], result["raw_manifest_sha256_after"])
            self.assertEqual(result["raw_status_sha256_before"], result["raw_status_sha256_after"])
            self.assertEqual(json.loads((args.control_root / "result.json").read_text(encoding="utf-8"))["decoder_calls"], 16)

    def test_exception_still_cleans_runtime_and_preserves_raw_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(temporary)
            self._raw(args.raw_root)
            contract = self._contract(args.launch_contract)
            before = (entry.sha256_file(args.raw_root / "MANIFEST.sha256"),
                      entry.sha256_file(args.raw_root / "run_status.json"))
            factory = mock.Mock(); runtime = mock.Mock(); encoder = mock.Mock()
            factory.build.return_value = (runtime, mock.Mock(), encoder)
            monitor = mock.Mock(); monitor.check.return_value = {"status": "OK"}
            with mock.patch.object(entry, "load_json", return_value=contract), \
                 mock.patch.object(entry, "_verify_raw_task6", return_value=(before[0], {"status": "COMPLETE"})), \
                 mock.patch.object(entry, "_validate_static_contract"), \
                 mock.patch.object(entry, "verify_pinned_bridge_assets", return_value={}), \
                 mock.patch.object(entry, "extract_task5_directions", return_value={"bank": "bank"}), \
                 mock.patch.object(entry, "OfficialRuntimeFactory", return_value=factory), \
                 mock.patch.object(entry, "resource_samplers", return_value=self._safe_samplers()), \
                 mock.patch.object(entry, "ResourceMonitor", side_effect=[monitor, monitor]), \
                 mock.patch.object(entry, "run_task6_decoder_replays", side_effect=RuntimeError("decode failed")):
                with self.assertRaises(RuntimeError):
                    entry.execute_decoder_only(args)
            runtime.cleanup.assert_called_once()
            factory.unload.assert_called_once()
            after = (entry.sha256_file(args.raw_root / "MANIFEST.sha256"),
                     entry.sha256_file(args.raw_root / "run_status.json"))
            self.assertEqual(before, after)
            control_result = json.loads((args.control_root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(control_result["status"], "BLOCKED")
            self.assertEqual(control_result["raw_generation_calls"], 0)

    def test_cleanup_failure_overwrites_success_terminal_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(temporary)
            self._raw(args.raw_root)
            contract = self._contract(args.launch_contract)
            factory = mock.Mock(); runtime = mock.Mock(); encoder = mock.Mock()
            runtime.cleanup.side_effect = RuntimeError("cleanup failed")
            factory.build.return_value = (runtime, mock.Mock(), encoder)
            monitor = mock.Mock(); monitor.check.return_value = {"status": "OK"}
            patches = [
                mock.patch.object(entry, "load_json", return_value=contract),
                mock.patch.object(entry, "_verify_raw_task6", return_value=("a" * 64, {"status": "COMPLETE"})),
                mock.patch.object(entry, "_validate_static_contract"),
                mock.patch.object(entry, "verify_pinned_bridge_assets", return_value={}),
                mock.patch.object(entry, "extract_task5_directions", return_value={"bank": "bank"}),
                mock.patch.object(entry, "OfficialRuntimeFactory", return_value=factory),
                mock.patch.object(entry, "resource_samplers", return_value=self._safe_samplers()),
                mock.patch.object(entry, "ResourceMonitor", side_effect=[monitor, monitor]),
                mock.patch.object(entry, "run_task6_decoder_replays", return_value={"status": "COMPLETE", "decoder_calls": 16}),
                mock.patch.object(entry, "analyze_task6_run", return_value={"status": "COMPLETE"}),
            ]
            started = [patcher.start() for patcher in patches]
            try:
                with self.assertRaises(RuntimeError):
                    entry.execute_decoder_only(args)
            finally:
                for patcher in patches:
                    patcher.stop()
            control_result = json.loads((args.control_root / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(control_result["status"], "RESOURCE_STOP")
            self.assertEqual(control_result["reason_code"], "FINALIZATION_FAILURE")

    def test_decoder_resource_stop_is_terminal_and_skips_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(temporary); self._raw(args.raw_root); contract = self._contract(args.launch_contract)
            factory = mock.Mock(); runtime = mock.Mock(); factory.build.return_value = (runtime, mock.Mock(), mock.Mock())
            monitor = mock.Mock(); monitor.check.return_value = {"status": "OK"}
            analysis_patcher = mock.patch.object(entry, "analyze_task6_run")
            analysis_mock = analysis_patcher.start()
            patches = [mock.patch.object(entry, "load_json", return_value=contract),
                       mock.patch.object(entry, "_verify_raw_task6", return_value=("a" * 64, {"status": "COMPLETE"})),
                       mock.patch.object(entry, "_validate_static_contract"),
                       mock.patch.object(entry, "verify_pinned_bridge_assets", return_value={}),
                       mock.patch.object(entry, "extract_task5_directions", return_value={"bank": "bank"}),
                       mock.patch.object(entry, "OfficialRuntimeFactory", return_value=factory),
                       mock.patch.object(entry, "resource_samplers", return_value=self._safe_samplers()),
                       mock.patch.object(entry, "ResourceMonitor", side_effect=[monitor, monitor]),
                       mock.patch.object(entry, "run_task6_decoder_replays", return_value={"status": "RESOURCE_STOP", "decoder_calls": 4})]
            for patcher in patches: patcher.start()
            try:
                result = entry.execute_decoder_only(args)
            finally:
                for patcher in patches: patcher.stop()
                analysis_patcher.stop()
            self.assertEqual(result["status"], "RESOURCE_STOP")
            self.assertEqual(result["decoder_calls"], 4)
            analysis_mock.assert_not_called()
            self.assertEqual(json.loads((args.control_root / "result.json").read_text())["status"], "RESOURCE_STOP")

    def test_load_monitor_hard_stop_skips_decoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(temporary); self._raw(args.raw_root); contract = self._contract(args.launch_contract)
            factory = mock.Mock(); factory.build.return_value = (mock.Mock(), mock.Mock(), mock.Mock())
            monitor = mock.Mock(); monitor.check.side_effect = [{"status": "OK"}, {"status": "HARD_STOP", "reason_code": "LOAD_GATE"}]
            with mock.patch.object(entry, "load_json", return_value=contract), \
                 mock.patch.object(entry, "_verify_raw_task6", return_value=("a" * 64, {"status": "COMPLETE"})), \
                 mock.patch.object(entry, "_validate_static_contract"), \
                 mock.patch.object(entry, "verify_pinned_bridge_assets", return_value={}), \
                 mock.patch.object(entry, "extract_task5_directions", return_value={"bank": "bank"}), \
                 mock.patch.object(entry, "OfficialRuntimeFactory", return_value=factory), \
                 mock.patch.object(entry, "resource_samplers", return_value=self._safe_samplers()), \
                 mock.patch.object(entry, "ResourceMonitor", return_value=monitor), \
                 mock.patch.object(entry, "run_task6_decoder_replays") as replay:
                result = entry.execute_decoder_only(args)
            self.assertEqual(result["reason_code"], "LOAD_GATE"); replay.assert_not_called()

    def test_analysis_and_control_roots_embedded_in_raw_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(temporary); self._raw(args.raw_root)
            args.analysis_root = args.raw_root / "analysis"
            with self.assertRaises(entry.OperationalEvidenceError):
                entry.execute_decoder_only(args)


if __name__ == "__main__":
    unittest.main()
