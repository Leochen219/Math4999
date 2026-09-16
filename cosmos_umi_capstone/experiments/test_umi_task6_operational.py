import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


class OperationalTask6Tests(unittest.TestCase):
    def setUp(self):
        import umi_task6_operational as op
        self.op = op

    def test_checkpoint_content_identity_preserves_file_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "checkpoint.bin"
            path.write_bytes(b"checkpoint")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(self.op.checkpoint_content_identity(path), expected)

    def test_checkpoint_directory_identity_is_deterministic_and_sensitive_to_name_and_content(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoint"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"weights")
            first = self.op.checkpoint_content_identity(root)
            self.assertEqual(first, self.op.checkpoint_content_identity(root))
            (root / "renamed.bin").write_bytes((root / "weights.bin").read_bytes())
            (root / "weights.bin").unlink()
            renamed = self.op.checkpoint_content_identity(root)
            self.assertNotEqual(first, renamed)
            (root / "renamed.bin").write_bytes(b"changed")
            self.assertNotEqual(renamed, self.op.checkpoint_content_identity(root))

    def test_checkpoint_content_identity_fails_closed_for_missing_path(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(self.op.OperationalEvidenceError):
                self.op.checkpoint_content_identity(Path(temp) / "missing-checkpoint")

    def test_checkpoint_content_identity_fails_closed_for_unsupported_path_type(self):
        with self.assertRaises(self.op.OperationalEvidenceError):
            self.op.checkpoint_content_identity(object())

    def test_factory_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "cosmos3-edge-model"
            checkpoint.mkdir(); (checkpoint / "model.bin").write_bytes(b"model")
            vae = root / "vae.pth"; vae.write_bytes(b"vae")
            contract = self._valid_factory_contract(checkpoint, vae)
            shutil_target = checkpoint
            for child in shutil_target.iterdir(): child.unlink()
            shutil_target.rmdir()
            with mock.patch.object(self.op, "_live_framework_commit", return_value=contract["framework_commit"]):
                with self.assertRaises(self.op.OperationalEvidenceError):
                    self.op.OfficialRuntimeFactory(
                        loader=lambda **kwargs: {}, framework_root=root, checkpoint=checkpoint, vae=vae,
                        contract=contract, direction_bank=np.zeros((3, 1, 48, 5, 16, 16), np.float32),
                        action=np.zeros((16, 10), np.float32), prompt=self.op.BRIDGE0_PROMPT,
                        video=root / "video.mp4",
                    )

    def test_factory_rejects_directory_vae(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "cosmos3-edge-model"
            checkpoint.mkdir(); (checkpoint / "model.bin").write_bytes(b"model")
            vae_file = root / "vae.pth"; vae_file.write_bytes(b"vae")
            contract = self._valid_factory_contract(checkpoint, vae_file)
            vae_file.unlink(); vae_file.mkdir(); (vae_file / "decoder.bin").write_bytes(b"vae")
            video = root / "video.mp4"; video.write_bytes(b"video")
            with mock.patch.object(self.op, "_live_framework_commit", return_value=contract["framework_commit"]):
                with self.assertRaises(self.op.OperationalEvidenceError):
                    self.op.OfficialRuntimeFactory(
                        loader=lambda **kwargs: {}, framework_root=root, checkpoint=checkpoint, vae=vae_file,
                        contract=contract, direction_bank=np.zeros((3, 1, 48, 5, 16, 16), np.float32),
                        action=np.zeros((16, 10), np.float32), prompt=self.op.BRIDGE0_PROMPT, video=video,
                    )

    def test_factory_accepts_valid_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "cosmos3-edge-model"
            checkpoint.mkdir()
            (checkpoint / "model.bin").write_bytes(b"model")
            vae = root / "vae.pth"; vae.write_bytes(b"vae")
            video = root / "video.mp4"; video.write_bytes(b"video")
            contract = self._valid_factory_contract(checkpoint, vae)
            with mock.patch.object(self.op, "_live_framework_commit", return_value=contract["framework_commit"]):
                try:
                    factory = self.op.OfficialRuntimeFactory(
                        loader=lambda **kwargs: {}, framework_root=root, checkpoint=checkpoint, vae=vae,
                        contract=contract, direction_bank=np.zeros((3, 1, 48, 5, 16, 16), np.float32),
                        action=np.zeros((16, 10), np.float32), prompt=self.op.BRIDGE0_PROMPT,
                        video=video,
                    )
                except self.op.OperationalEvidenceError as error:
                    self.fail(f"valid checkpoint directory was rejected: {error}")
                self.assertIsNotNone(factory)

    def test_catalog_bridge0_is_exactly_pinned(self):
        from umi_task6_primitives import STATE_CATALOG, SOURCE_COMMIT
        asset = STATE_CATALOG["bridge_0"]
        self.assertEqual(asset.source_commit, SOURCE_COMMIT)
        self.assertEqual(asset.action_path, "inputs/action/bridge_20260501_0.json")
        self.assertEqual(asset.video_path, "inputs/action/bridge_20260501_0.mp4")

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0" / "bridge_20260501_0.mp4").is_file(), "local reviewed bridge asset is not present")
    def test_local_bridge_pair_hash_and_action_shape(self):
        root = Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0"
        evidence = self.op.verify_pinned_bridge_assets(root / "bridge_20260501_0.json", root / "bridge_20260501_0.mp4")
        self.assertEqual(evidence["action_shape"], [16, 10]); self.assertEqual(evidence["fps"], 5)

    @unittest.skipUnless((Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0" / "bridge_20260501_0.mp4").is_file(), "local reviewed bridge asset is not present")
    def test_asset_bundle_refuses_overwrite(self):
        root = Path(__file__).resolve().parents[1] / "assets" / "task6_bridge0"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "bundle"
            self.op.prepare_bridge_upload_bundle(root / "bridge_20260501_0.json", root / "bridge_20260501_0.mp4", out)
            with self.assertRaises(FileExistsError):
                self.op.prepare_bridge_upload_bundle(root / "bridge_20260501_0.json", root / "bridge_20260501_0.mp4", out)

    def test_contract_never_accepts_self_attested_only(self):
        contract = self.contract()
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract)

    def test_contract_rejects_wrong_group_and_observed_identity(self):
        contract = self.contract(); observed = dict(contract)
        contract["group"] = {"state": "bridge_384", "seed": 0}
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=observed)

    def test_static_contract_requires_all_task5_and_bridge_hashes(self):
        contract = self.contract(); contract["task5"] = {"manifest_sha256": "a" * 64}
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=contract)
        contract = self.contract(); contract["bridge_asset_hashes"]["video"] = "0" * 64
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=contract)
        contract = self.contract(); observed = dict(contract); observed["cuda_version"] = "wrong"
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.validate_launch_contract(contract, observed=observed)

    def test_task5_direction_extraction_requires_manifest_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "samples").mkdir()
            for direction in ("v0", "v1", "v2"):
                sample = root / "samples" / f"{direction}_alpha_00_plus"; sample.mkdir()
                arr = np.zeros((1, 48, 5, 16, 16), np.float32)
                mask = np.zeros_like(arr, dtype=bool); mask[:, :, 0] = True; arr[mask] = 1.0
                np.save(sample / "direction.npy", arr)
                np.save(sample / "mask.npy", mask)
                (sample / "sample.json").write_text("{}")
                hashes = {name: hashlib.sha256((sample / name).read_bytes()).hexdigest() for name in ("direction.npy", "mask.npy", "sample.json")}
                (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": hashes}))
            plan = {"status": "complete", "plan": [{"sample_id": f"s{i}"} for i in range(32)]}; (root / "task5_plan.json").write_text(json.dumps(plan))
            (root / "status.json").write_text(json.dumps({"status": "complete", "formal_successful": 32}))
            entries = []
            for path in sorted(root.rglob("*")):
                if path.is_file(): entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
            (root / "MANIFEST.sha256").write_text("\n".join(entries) + "\n")
            result = self.op.extract_task5_directions(root)
            self.assertEqual(result["bank"].shape, (3, 1, 48, 5, 16, 16)); self.assertEqual(set(result["direction_sha256"]), {"v0", "v1", "v2", "u01", "u12"})

    def test_task5_manifest_rejects_nested_manifest_and_extra_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "task5_plan.json").write_text("{}")
            (root / "samples").mkdir(); (root / "samples" / "nested").mkdir()
            (root / "samples" / "nested" / "MANIFEST.sha256").write_text("bad")
            (root / "MANIFEST.sha256").write_text("\n")
            with self.assertRaises(self.op.OperationalEvidenceError): self.op.extract_task5_directions(root)

    def test_loader_spec_requires_module_function(self):
        with self.assertRaises(self.op.OperationalEvidenceError): self.op.import_callable("not-a-spec")

    def test_cosmos_loader_args_pin_bridge_fps_and_isolated_setup(self):
        import umi_task6_cosmos_loader as loader
        with tempfile.TemporaryDirectory() as temp:
            setup = Path(temp) / "run" / "loader_setup"
            args = loader.build_loader_args(framework_root=Path(temp), checkpoint=Path(temp) / "model",
                vae=Path(temp) / "vae", video=Path(temp) / "video.mp4", prompt="p",
                action=np.zeros((16, 10), np.float32), setup_dir=setup, phase="preflight", resume=False)
            self.assertEqual(args.fps, 5)
            self.assertEqual(Path(args.run_dir), setup)
            self.assertEqual(args.phase, "preflight")
            self.assertTrue(str(setup).endswith("loader_setup"))

    def test_official_sample_fps_preserves_default_and_bridge_override(self):
        import umi_fd_post_vae_scan as scan
        self.assertEqual(scan.resolved_sample_fps(type("Args", (), {})()), 20)
        self.assertEqual(scan.resolved_sample_fps(type("Args", (), {"fps": 5})()), 5)

    def test_loader_rejects_resolved_sample_fps_other_than_bridge_contract(self):
        import umi_task6_cosmos_loader as loader
        self.assertEqual(loader.resolve_bridge_fps(type("Sample", (), {"fps": 5})()), 5)
        with self.assertRaises(ValueError):
            loader.resolve_bridge_fps(type("Sample", (), {"fps": 20})())

    def test_official_preflight_failure_publishes_canonical_status(self):
        import run_umi_task6_official as official
        import run_umi_task6_experiment as runner
        with tempfile.TemporaryDirectory() as temp:
            original = official._execute_official_impl
            official._execute_official_impl = lambda args: (_ for _ in ()).throw(RuntimeError("loader resource failure"))
            try:
                with self.assertRaises(RuntimeError):
                    official.execute_official(type("Args", (), {"run_dir": temp, "phase": "preflight"})())
            finally:
                official._execute_official_impl = original
            status = json.loads((Path(temp) / "run_status.json").read_text())
            self.assertEqual(status["status"], "RESOURCE_STOP")
            self.assertFalse(status["generation_started"])

    def test_official_production_chain_preflight_then_smoke_preserves_six_hashes(self):
        import run_umi_task6_official as official
        import run_umi_task6_experiment as runner
        import umi_task6_runtime as runtime_api
        import umi_task6_operational as op
        carrier = np.ones((1, 48, 5, 16, 16), np.float32)
        mask = np.zeros_like(carrier, dtype=bool); mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, np.float32); bank[:, mask] = 1.0
        frozen = runtime_api._derive_frozen_directions_unpinned(bank, mask)
        direction_hashes = {name: runtime_api._array_sha(value) for name, value in frozen.items()}
        inputs = runtime_api.Task6Inputs(carrier, [0], mask, bank, action=np.zeros((16, 10), np.float32),
            prompt=official.BRIDGE0_PROMPT if hasattr(official, "BRIDGE0_PROMPT") else "Put the pot to the left of the purple item.",
            state="bridge_0", seed=0, direction_hashes=direction_hashes)
        class Runtime:
            model_seed = 0
            provenance = {"asset_source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d"}
            def actual_identity(self): return {"model": "fixture-content-v1"}
            def execute(self, spec, bound, *, scope="full"): return {"output_full": np.zeros((1, 48, 5, 16, 16), np.float32)}
            def cleanup(self): pass
        runtime = Runtime(); runtime.inputs = inputs
        safe = lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "gpu_reserved_gib": 0,
                        "ram_available_gib": 600, "rss_gib": 0, "swap_used_gib": 0, "disk_free_gib": 20}
        class Factory:
            phases = []
            def __init__(self, **kwargs): self.phases.append((kwargs["phase"], kwargs["run_dir"])); self.kwargs = kwargs
            def build(self): return runtime, inputs, object()
            def unload(self): pass
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); action = root / "action.json"; video = root / "video.mp4"; checkpoint = root / "model"; vae = root / "vae"
            action.write_text(json.dumps([[0.0] * 10 for _ in range(16)])); video.write_bytes(b"video"); checkpoint.write_bytes(b"model"); vae.write_bytes(b"vae")
            contract = {"framework_commit": "1" * 40, "asset_source_commit": op.SOURCE_COMMIT,
                "checkpoint_identity": {"sha256": "a" * 64}, "vae_sha256": "b" * 64, "torch_version": "fixture", "cuda_version": "fixture", "code_bundle_sha256": "c" * 64,
                "bridge_asset_hashes": dict(op.BRIDGE0_ASSET_SHA256), "fps": 5, "task5": {"manifest_sha256": "f" * 64, "plan_sha256": "0" * 64,
                    "direction_file_sha256": {k: "1" * 64 for k in ("v0", "v1", "v2")}, "direction_sha256": {k: "2" * 64 for k in direction_hashes}},
                "group": {"state": "bridge_0", "seed": 0}, "prompt": inputs.prompt, "action": [[0.0] * 10 for _ in range(16)],
                "settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "batch_size": 1, "autocast": False, "tf32": False, "diffusion_cache": False},
                "cache_flags": {"autocast": False, "tf32": False, "diffusion_cache": False}, "seed_routes": {"model": 0, "prepare": 0, "sampler": 0, "scheduler": 0},
                "geometry": {"carrier_shape": [1, 48, 5, 16, 16], "condition_indexes": [0], "predicted_indexes": [1, 2, 3, 4], "mask_shape": [1, 48, 5, 16, 16]}}
            launch = root / "launch.json"; launch.write_text(json.dumps(contract))
            task5 = {"bank": bank, "manifest_sha256": "f" * 64, "plan_sha256": "0" * 64,
                     "direction_file_sha256": {k: "1" * 64 for k in ("v0", "v1", "v2")}, "direction_sha256": {k: "2" * 64 for k in direction_hashes}}
            observed = dict(contract); observed.update({"checkpoint_identity": {"sha256": "a" * 64}, "vae_sha256": "b" * 64,
                "torch_version": "fixture", "cuda_version": "fixture", "runtime_identity": runtime.actual_identity()})
            old = {name: getattr(official, name) for name in ("OfficialRuntimeFactory", "verify_pinned_bridge_assets", "extract_task5_directions", "observe_live_launch", "validate_launch_contract", "resource_samplers", "preflight_task6", "build_task6_hash_binding")}
            old_runner_binding = runner.build_task6_hash_binding
            try:
                official.OfficialRuntimeFactory = Factory
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "d" * 64, "video_sha256": "e" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: task5
                official.observe_live_launch = lambda *args, **kwargs: observed
                official.validate_launch_contract = lambda *args, **kwargs: {"status": "PASS"}
                official.resource_samplers = lambda *args, **kwargs: {"gpu": safe, "ram": safe, "disk": safe}
                official.preflight_task6 = lambda *args, **kwargs: {"status": "PASS"}
                six = {name: name + "-hash" for name in ("code", "model", "config", "direction", "input", "noise")}
                official.build_task6_hash_binding = lambda *args, **kwargs: six
                runner.build_task6_hash_binding = lambda *args, **kwargs: six
                pre = official.execute_official(official.parse_args(["--phase", "preflight", "--run-dir", str(root), "--launch-contract", str(launch),
                    "--framework-root", str(root), "--checkpoint", str(checkpoint), "--vae", str(vae), "--action", str(action), "--video", str(video), "--task5-root", str(root)]))
                self.assertEqual(pre["status"], "PREFLIGHT_COMPLETE"); self.assertEqual(set(pre["hashes"]), set(six))
                smoke = official.execute_official(official.parse_args(["--phase", "resource-smoke", "--run-dir", str(root), "--launch-contract", str(launch),
                    "--framework-root", str(root), "--checkpoint", str(checkpoint), "--vae", str(vae), "--action", str(action), "--video", str(video), "--task5-root", str(root)]))
                self.assertEqual(smoke["hashes"], six); self.assertEqual([phase for phase, _ in Factory.phases], ["preflight", "resource-smoke"])
            finally:
                for name, value in old.items(): setattr(official, name, value)
                runner.build_task6_hash_binding = old_runner_binding

    def test_condition_encoder_adapter_uses_batched_torch_video_and_restores_cache(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("bundled CPU test runtime has no torch; official CUDA path exercises this seam")
        import umi_task6_cosmos_loader as loader
        class Encoder:
            def __init__(self): self.calls = []; self.cache = {"stale": 1}; self.weight = np.array([1, 2], np.float32)
            def actual_identity(self): return {"weights": self.weight.tolist()}
            def reset_cache(self): self.cache.clear()
            def encode(self, value):
                self.calls.append(value)
                self.cache["active"] = 1
                return value.float()
        encoder = Encoder(); adapter = loader.ConditionEncoderAdapter(encoder, device="cpu")
        frame = np.array([np.zeros((8, 8)), np.full((8, 8), 0.5), np.ones((8, 8))], np.float32)
        result = adapter(frame)
        self.assertEqual(tuple(result.shape), (1, 3, 1, 8, 8))
        self.assertEqual(str(result.dtype), "torch.float32")
        self.assertEqual(tuple(encoder.calls[0].shape), (1, 3, 1, 8, 8))
        self.assertEqual(float(encoder.calls[0].min()), -1.0)
        self.assertEqual(float(encoder.calls[0].max()), 1.0)
        self.assertEqual(float(encoder.calls[0][0, 1, 0, 0, 0]), 0.0)
        self.assertEqual(encoder.cache, {})
        adapter.reset_cache()
        self.assertEqual(encoder.cache, {})
        self.assertIn("weights", adapter.actual_identity()["encoder"]["content"])

    def test_raw_manifest_excludes_mutable_status_and_resource_outputs(self):
        import umi_task6_runtime as runtime_api
        import analyze_umi_task6 as analyzer
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "immutable.json").write_text("evidence")
            (root / "run_status.json").write_text('{"status":"AWAITING_REVIEW"}')
            runtime_api._write_manifest(root)
            manifest = (root / "MANIFEST.sha256").read_text()
            self.assertNotIn("run_status.json", manifest)
            safe = lambda: {"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600,
                            "rss_gib": 0, "swap_used_gib": 0, "disk_free_gib": 20}
            monitor = __import__("run_umi_task6_experiment").ResourceMonitor(root, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe)
            monitor.start(); monitor.stop()
            (root / "run_status.json").write_text('{"status":"RESOURCE_STOP"}')
            analyzer.verify_raw_manifest(root)

    def contract(self):
        return {"framework_commit": "1" * 40, "asset_source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d", "checkpoint_identity": {"sha256": "x"}, "fps": 5,
                "vae_sha256": "a" * 64, "torch_version": "2.10.0+cu130", "cuda_version": "13.0", "code_bundle_sha256": "b" * 64,
                "bridge_asset_hashes": {"action": "a" * 64, "video": "b" * 64}, "task5": {"manifest_sha256": "c" * 64},
                "group": {"state": "bridge_0", "seed": 0}, "prompt": self.op.BRIDGE0_PROMPT,
                "action": [[0.0] * 10 for _ in range(16)], "settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "batch_size": 1, "autocast": False, "tf32": False, "diffusion_cache": False},
                "cache_flags": {"autocast": False, "tf32": False, "diffusion_cache": False},
                "seed_routes": {"model": 0, "prepare": 0, "sampler": 0, "scheduler": 0},
                "geometry": {"carrier_shape": [1, 48, 5, 16, 16], "condition_indexes": [0], "predicted_indexes": [1, 2, 3, 4], "mask_shape": [1, 48, 5, 16, 16]}}

    def _valid_factory_contract(self, checkpoint, vae):
        contract = self.contract()
        import umi_fd_post_vae_scan as scan
        contract.update({
            "checkpoint_identity": {"sha256": scan.sha256_tree(checkpoint)},
            "vae_sha256": self.op.sha256_file(vae),
            "code_bundle_sha256": self.op.code_bundle_sha256(),
            "bridge_asset_hashes": dict(self.op.BRIDGE0_ASSET_SHA256),
            "task5": {
                "manifest_sha256": "c" * 64,
                "plan_sha256": "d" * 64,
                "direction_file_sha256": {key: "e" * 64 for key in ("v0", "v1", "v2")},
                "direction_sha256": {key: "f" * 64 for key in ("v0", "v1", "v2", "u01", "u12")},
            },
        })
        return contract


if __name__ == "__main__": unittest.main()
