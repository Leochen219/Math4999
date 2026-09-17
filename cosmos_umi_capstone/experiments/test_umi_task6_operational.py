import hashlib
import errno
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

    def test_checkpoint_content_identity_rejects_symlink_to_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "checkpoint.bin"; target.write_bytes(b"model")
            link = root / "checkpoint-link"; self._symlink_or_skip(link, target, target_is_directory=False)
            with self.assertRaisesRegex(self.op.OperationalEvidenceError, "symlink"):
                self.op.checkpoint_content_identity(link)

    def test_checkpoint_content_identity_rejects_symlink_to_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "checkpoint"; target.mkdir(); (target / "model.bin").write_bytes(b"model")
            link = root / "checkpoint-link"; self._symlink_or_skip(link, target, target_is_directory=True)
            with self.assertRaisesRegex(self.op.OperationalEvidenceError, "symlink"):
                self.op.checkpoint_content_identity(link)

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
            video = root / "video.mp4"; video.write_bytes(b"video")
            with mock.patch.object(self.op, "_live_framework_commit", return_value=contract["framework_commit"]):
                with self.assertRaisesRegex(self.op.OperationalEvidenceError, "checkpoint path is missing or unsupported"):
                    self.op.OfficialRuntimeFactory(
                        loader=lambda **kwargs: {}, framework_root=root, checkpoint=checkpoint, vae=vae,
                        contract=contract, direction_bank=np.zeros((3, 1, 48, 5, 16, 16), np.float32),
                        action=np.zeros((16, 10), np.float32), prompt=self.op.BRIDGE0_PROMPT,
                        video=video,
                    )

    def test_factory_rejects_symlink_to_file_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "checkpoint.bin"; target.write_bytes(b"model")
            link = root / "checkpoint-link"; self._symlink_or_skip(link, target, target_is_directory=False)
            vae = root / "vae.pth"; vae.write_bytes(b"vae"); video = root / "video.mp4"; video.write_bytes(b"video")
            contract = self._valid_factory_contract(target, vae)
            with mock.patch.object(self.op, "_live_framework_commit", return_value=contract["framework_commit"]):
                with self.assertRaisesRegex(self.op.OperationalEvidenceError, "symlink"):
                    self.op.OfficialRuntimeFactory(
                        loader=lambda **kwargs: {}, framework_root=root, checkpoint=link, vae=vae,
                        contract=contract, direction_bank=np.zeros((3, 1, 48, 5, 16, 16), np.float32),
                        action=np.zeros((16, 10), np.float32), prompt=self.op.BRIDGE0_PROMPT, video=video,
                    )

    def test_factory_rejects_symlink_to_directory_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "checkpoint"; target.mkdir(); (target / "model.bin").write_bytes(b"model")
            link = root / "checkpoint-link"; self._symlink_or_skip(link, target, target_is_directory=True)
            vae = root / "vae.pth"; vae.write_bytes(b"vae"); video = root / "video.mp4"; video.write_bytes(b"video")
            contract = self._valid_factory_contract(target, vae)
            with mock.patch.object(self.op, "_live_framework_commit", return_value=contract["framework_commit"]):
                with self.assertRaisesRegex(self.op.OperationalEvidenceError, "symlink"):
                    self.op.OfficialRuntimeFactory(
                        loader=lambda **kwargs: {}, framework_root=root, checkpoint=link, vae=vae,
                        contract=contract, direction_bank=np.zeros((3, 1, 48, 5, 16, 16), np.float32),
                        action=np.zeros((16, 10), np.float32), prompt=self.op.BRIDGE0_PROMPT, video=video,
                    )

    def test_factory_rejects_symlink_before_resolving_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "checkpoint.bin"; target.write_bytes(b"model")
            link = root / "checkpoint-link"; vae = root / "vae.pth"; vae.write_bytes(b"vae")
            video = root / "video.mp4"; video.write_bytes(b"video")
            contract = self._valid_factory_contract(target, vae)
            def reports_symlink(path):
                return path == link
            with mock.patch.object(Path, "is_symlink", autospec=True, side_effect=reports_symlink):
                with mock.patch.object(self.op, "_live_framework_commit", return_value=contract["framework_commit"]):
                    with self.assertRaisesRegex(self.op.OperationalEvidenceError, "symlink"):
                        self.op.OfficialRuntimeFactory(
                            loader=lambda **kwargs: {}, framework_root=root, checkpoint=link, vae=vae,
                            contract=contract, direction_bank=np.zeros((3, 1, 48, 5, 16, 16), np.float32),
                            action=np.zeros((16, 10), np.float32), prompt=self.op.BRIDGE0_PROMPT, video=video,
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

    def test_task5_direction_extraction_accepts_root_lock_in_manifest(self):
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
            (root / ".runner.lock").write_bytes(b"historical task5 lock")
            entries = []
            for path in sorted(root.rglob("*")):
                if path.is_file(): entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
            (root / "MANIFEST.sha256").write_text("\n".join(entries) + "\n")
            result = self.op.extract_task5_directions(root)
            self.assertEqual(result["bank"].shape, (3, 1, 48, 5, 16, 16)); self.assertEqual(set(result["direction_sha256"]), {"v0", "v1", "v2", "u01", "u12"})

    def test_task5_manifest_rejects_duplicate_root_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); lock = root / ".runner.lock"; lock.write_bytes(b"lock")
            digest = hashlib.sha256(lock.read_bytes()).hexdigest()
            (root / "MANIFEST.sha256").write_text(f"{digest}  .runner.lock\n{digest}  .runner.lock\n")
            with self.assertRaisesRegex(self.op.OperationalEvidenceError, "unsafe or duplicate"):
                self.op._verify_tree_manifest(root)

    def test_task5_manifest_rejects_root_lock_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / ".runner.lock").write_bytes(b"lock")
            (root / "MANIFEST.sha256").write_text(f"{'0' * 64}  .runner.lock\n")
            with self.assertRaisesRegex(self.op.OperationalEvidenceError, "manifest mismatch"):
                self.op._verify_tree_manifest(root)

    def test_task5_manifest_rejects_absolute_parent_and_manifest_paths(self):
        for invalid_kind in ("absolute", "parent", "manifest"):
            with self.subTest(invalid_kind=invalid_kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                invalid_path = {"absolute": str((root / "absolute").resolve()),
                                "parent": "../outside", "manifest": "MANIFEST.sha256"}[invalid_kind]
                (root / "MANIFEST.sha256").write_text(f"{'0' * 64}  {invalid_path}\n")
                with self.assertRaisesRegex(self.op.OperationalEvidenceError, "unsafe or duplicate"):
                    self.op._verify_tree_manifest(root)

    def test_task5_manifest_rejects_unlisted_extra_hidden_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); lock = root / ".runner.lock"; lock.write_bytes(b"lock")
            payload = root / "payload.txt"; payload.write_bytes(b"payload")
            (root / ".hidden").write_bytes(b"unlisted")
            entries = [
                f"{hashlib.sha256(lock.read_bytes()).hexdigest()}  .runner.lock",
                f"{hashlib.sha256(payload.read_bytes()).hexdigest()}  payload.txt",
            ]
            (root / "MANIFEST.sha256").write_text("\n".join(entries) + "\n")
            with self.assertRaisesRegex(self.op.OperationalEvidenceError, "inventory mismatch"):
                self.op._verify_tree_manifest(root)

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

    def test_loader_action_json_is_finite_exact_16_by_10_list_for_numpy_action(self):
        import umi_task6_cosmos_loader as loader
        action = (np.arange(160, dtype=np.float32).reshape(16, 10) / np.float32(17.0))
        encoded = loader._json_safe_action(action)
        self.assertIsInstance(encoded, list)
        self.assertEqual((len(encoded), len(encoded[0])), (16, 10))
        self.assertTrue(all(isinstance(item, float) for row in encoded for item in row))
        np.testing.assert_array_equal(np.asarray(encoded, dtype=np.float32), action)
        precise = np.zeros((16, 10), dtype=np.float64)
        precise[0, 0] = 0.12345678901234567
        self.assertEqual(loader._json_safe_action(precise)[0][0], float(precise[0, 0]))
        for invalid in (np.full((16, 10), np.nan, np.float32), np.zeros((15, 10), np.float32)):
            with self.assertRaises(ValueError): loader._json_safe_action(invalid)

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

    def test_official_derived_resume_failure_preserves_complete_raw_status(self):
        import run_umi_task6_official as official
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_status = {"status": "AWAITING_REVIEW", "phase": "PILOT", "generation_started": False,
                          "completed_samples": [f"sample-{i}" for i in range(32)], "hashes": {}}
            (root / "run_status.json").write_text(json.dumps(raw_status), encoding="utf-8")
            original = official._execute_official_impl
            official._execute_official_impl = lambda args: (_ for _ in ()).throw(RuntimeError("analysis stopped"))
            try:
                with self.assertRaises(RuntimeError):
                    official.execute_official(type("Args", (), {"run_dir": temp, "phase": "pilot", "resume": True})())
            finally:
                official._execute_official_impl = original
            self.assertEqual(json.loads((root / "run_status.json").read_text())["status"], "AWAITING_REVIEW")

    def test_official_cli_returns_zero_for_preflight_complete(self):
        import run_umi_task6_official as official
        original_parse, original_execute = official.parse_args, official.execute_official
        try:
            official.parse_args = lambda argv=None: object()
            official.execute_official = lambda args: {"status": "PREFLIGHT_COMPLETE"}
            self.assertEqual(official.main([]), 0)
        finally:
            official.parse_args, official.execute_official = original_parse, original_execute

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

    def test_official_pilot_rejects_unsafe_current_resources_before_factory_build(self):
        import run_umi_task6_official as official
        import run_umi_task6_experiment as runner
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence.json"; evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            events = []
            class Runtime:
                def cleanup(self): events.append("runtime_cleanup")
            runtime = Runtime()
            class Factory:
                def __init__(self, **kwargs): events.append("factory_init")
                def build(self): events.append("factory_build"); return runtime, object(), object()
                def unload(self): events.append("factory_unload")
            class Monitor:
                last_resources = {"gpu_used_gib": 2.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                                  "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
                def __init__(self, *args, **kwargs): events.append("monitor_init")
                def start(self): events.append("monitor_start"); return self
                def check(self, **kwargs): return {"status": "HARD_STOP", "reason_code": "GPU_START_USED_HIGH", "reason": "unsafe"}
                def stop(self): events.append("monitor_stop")
            unsafe = lambda: {"gpu_used_gib": 2.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                              "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
            (root / "action").write_text(json.dumps([[0.0] * 10 for _ in range(16)]), encoding="utf-8")
            (root / "video").write_bytes(b"video")
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets", "extract_task5_directions", "OfficialRuntimeFactory", "resource_samplers", "ResourceMonitor")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {"gpu": unsafe, "ram": unsafe, "disk": unsafe}
                official.ResourceMonitor = Monitor
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--launch-contract", str(evidence),
                    "--framework-root", str(root), "--checkpoint", str(root / "checkpoint"), "--vae", str(root / "vae"),
                    "--action", str(root / "action"), "--video", str(root / "video"), "--task5-root", str(root)])
                result = official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            self.assertNotIn("factory_build", events, events)
            self.assertEqual(result["status"], "RESOURCE_STOP")
            self.assertLess(events.index("monitor_start"), events.index("factory_build") if "factory_build" in events else len(events), events)
            self.assertIn("factory_unload", events)

    def test_official_preload_stop_preserves_accepted_smoke_binding(self):
        import run_umi_task6_official as official
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence.json"
            evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            (root / "action").write_text(json.dumps([[0.0] * 10 for _ in range(16)]), encoding="utf-8")
            (root / "video").write_bytes(b"video")
            six = {name: name + "-hash" for name in ("code", "model", "config", "direction", "input", "noise")}
            prior = {"status": "AWAITING_RESOURCE_REVIEW", "phase": "PILOT", "generation_started": False,
                     "smoke_run_id": "smoke-accepted-1", "hashes": six, "group": {"state": "bridge_0", "seed": 0},
                     "completed_samples": [], "failed_samples": [], "skipped_samples": [], "smoke_decision": {"status": "AWAITING_RESOURCE_REVIEW"}}
            (root / "run_status.json").write_text(json.dumps(prior), encoding="utf-8")
            (root / "smoke_acceptance.json").write_text(json.dumps({"status": "AWAITING_RESOURCE_REVIEW", "smoke_decision_accepted": True,
                "smoke_run_id": prior["smoke_run_id"], "hashes": six}), encoding="utf-8")
            class Factory:
                def __init__(self, **kwargs): pass
                def build(self): self.fail("preload hard-stop must not load the model")
                def unload(self): pass
            class Monitor:
                last_resources = {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                                  "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
                def __init__(self, *args, **kwargs): pass
                def start(self): return self
                def check(self, **kwargs): return {"status": "HARD_STOP", "reason_code": "GPU_START_USED_HIGH", "reason": "unsafe"}
                def stop(self): pass
            safe = lambda: dict(Monitor.last_resources)
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets",
                "extract_task5_directions", "OfficialRuntimeFactory", "resource_samplers", "ResourceMonitor")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {key: safe for key in ("gpu", "ram", "disk")}
                official.ResourceMonitor = Monitor
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--launch-contract", str(evidence),
                    "--framework-root", str(root), "--checkpoint", str(root / "checkpoint"), "--vae", str(root / "vae"),
                    "--action", str(root / "action"), "--video", str(root / "video"), "--task5-root", str(root)])
                result = official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            self.assertEqual(result["status"], "RESOURCE_STOP")
            self.assertEqual(result["smoke_run_id"], prior["smoke_run_id"])
            self.assertEqual(result["hashes"], six)
            self.assertEqual(result["group"], prior["group"])
            self.assertEqual(json.loads((root / "run_status.json").read_text())["smoke_run_id"], prior["smoke_run_id"])

    def test_official_raw_complete_preload_stop_preserves_complete_binding(self):
        import run_umi_task6_official as official
        from umi_task6_primitives import build_generation_plan
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence.json"
            evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            (root / "action").write_text(json.dumps([[0.0] * 10 for _ in range(16)]), encoding="utf-8")
            (root / "video").write_bytes(b"video")
            six = {name: name + "-hash" for name in ("code", "model", "config", "direction", "input", "noise")}
            expected = [item["sample_id"] for item in build_generation_plan("bridge_0", 0)]
            samples = root / "samples"
            samples.mkdir()
            for sample_id in expected:
                sample = samples / sample_id
                sample.mkdir()
                (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": {}}), encoding="utf-8")
            prior = {"status": "AWAITING_REVIEW", "phase": "PILOT", "generation_started": False,
                     "smoke_run_id": "smoke-accepted-complete", "hashes": six, "group": {"state": "bridge_0", "seed": 0},
                     "completed_samples": expected, "failed_samples": [], "skipped_samples": [], "smoke_decision": {"status": "AWAITING_REVIEW"}}
            (root / "run_status.json").write_text(json.dumps(prior), encoding="utf-8")
            (root / "smoke_acceptance.json").write_text(json.dumps({"status": "AWAITING_RESOURCE_REVIEW", "smoke_decision_accepted": True,
                "smoke_run_id": prior["smoke_run_id"], "hashes": six}), encoding="utf-8")
            manifest_entries = []
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name != "MANIFEST.sha256":
                    manifest_entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
            (root / "MANIFEST.sha256").write_text("\n".join(manifest_entries) + "\n", encoding="ascii")
            class Factory:
                def __init__(self, **kwargs): pass
                def build(self): self.fail("preload hard-stop must not load the model")
                def unload(self): pass
            class Monitor:
                last_resources = {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                                  "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
                def __init__(self, *args, **kwargs): pass
                def start(self): return self
                def check(self, **kwargs): return {"status": "HARD_STOP", "reason_code": "GPU_START_USED_HIGH", "reason": "unsafe"}
                def stop(self): pass
            safe = lambda: dict(Monitor.last_resources)
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets",
                "extract_task5_directions", "OfficialRuntimeFactory", "resource_samplers", "ResourceMonitor")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {key: safe for key in ("gpu", "ram", "disk")}
                official.ResourceMonitor = Monitor
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--launch-contract", str(evidence),
                    "--framework-root", str(root), "--checkpoint", str(root / "checkpoint"), "--vae", str(root / "vae"),
                    "--action", str(root / "action"), "--video", str(root / "video"), "--task5-root", str(root), "--resume"])
                result = official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            self.assertEqual(result["status"], "AWAITING_REVIEW")
            self.assertEqual(result["smoke_run_id"], prior["smoke_run_id"])
            self.assertEqual(result["hashes"], six)
            self.assertEqual(result["completed_samples"], expected)
            self.assertEqual(json.loads((root / "run_status.json").read_text())["status"], "AWAITING_REVIEW")

    def test_official_derived_resume_consumes_latched_loader_stop_before_decoder(self):
        import run_umi_task6_official as official
        from umi_task6_primitives import build_generation_plan
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence.json"
            evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            (root / "action").write_text(json.dumps([[0.0] * 10 for _ in range(16)]), encoding="utf-8")
            (root / "video").write_bytes(b"video")
            six = {name: name + "-hash" for name in ("code", "model", "config", "direction", "input", "noise")}
            expected = [item["sample_id"] for item in build_generation_plan("bridge_0", 0)]
            raw_status = {"status": "AWAITING_REVIEW", "phase": "PILOT", "generation_started": False,
                          "smoke_run_id": "smoke-derived", "hashes": six, "group": {"state": "bridge_0", "seed": 0},
                          "completed_samples": expected, "failed_samples": [], "skipped_samples": []}
            (root / "run_status.json").write_text(json.dumps(raw_status), encoding="utf-8")
            raw_status_before = ((root / "run_status.json").read_bytes(), (root / "run_status.json").stat().st_mtime_ns)
            class Runtime:
                provenance = {"fixture": "runtime"}
                def cleanup(self): pass
                def actual_identity(self): return {"fixture": "runtime"}
            runtime = Runtime()
            inputs = object()
            events = []
            class Factory:
                def __init__(self, **kwargs): pass
                def build(self):
                    events.append("loader_recovered")
                    Monitor.instances[0].latched = {"status": "HARD_STOP", "reason_code": "GPU_TRANSIENT", "reason": "latched during load"}
                    return runtime, inputs, object()
                def unload(self): events.append("factory_unload")
            class Monitor:
                instances = []
                def __init__(self, root, **kwargs):
                    self.root = Path(root); self._thread = object(); self.latched = None; self.check_calls = 0
                    Monitor.instances.append(self)
                def start(self): events.append(("monitor_start", self.root.name)); return self
                def check(self, **kwargs):
                    self.check_calls += 1
                    if self.latched is not None: return dict(self.latched)
                    return {"status": "OK"}
                def stop(self): events.append(("monitor_stop", self.root.name))
            safe = lambda: {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                            "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets",
                "extract_task5_directions", "OfficialRuntimeFactory", "resource_samplers", "ResourceMonitor", "run_pilot",
                "run_task6_decoder_replays")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {key: safe for key in ("gpu", "ram", "disk")}
                official.ResourceMonitor = Monitor
                official.run_pilot = lambda *args, **kwargs: dict(raw_status)
                official.run_task6_decoder_replays = lambda *args, **kwargs: self.fail("latched loader stop must block decoder")
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--launch-contract", str(evidence),
                    "--framework-root", str(root), "--checkpoint", str(root / "checkpoint"), "--vae", str(root / "vae"),
                    "--action", str(root / "action"), "--video", str(root / "video"), "--task5-root", str(root), "--resume"])
                with mock.patch.object(official, "_inspect_raw_complete", return_value=raw_status, create=True):
                    result = official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            self.assertEqual(result["status"], "AWAITING_REVIEW")
            self.assertIn("loader_recovered", events)
            self.assertEqual(Monitor.instances[0].check_calls, 2)
            self.assertNotEqual(Monitor.instances[0].root.resolve(), root.resolve())
            self.assertEqual(raw_status_before, ((root / "run_status.json").read_bytes(), (root / "run_status.json").stat().st_mtime_ns))

    def test_official_derived_resume_flushes_sibling_load_monitor_before_exact_decoder_resume(self):
        import run_umi_task6_official as official
        import umi_task6_decoder as decoder_api
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); decoder_root = root.parent / (root.name + "_decoder")
            decoder_root.mkdir()
            (decoder_root / "decoder_config.json").write_text("config\n", encoding="utf-8")
            (decoder_root / "replay.json").write_text("record\n", encoding="utf-8")
            decoder_api._decoder_manifest(decoder_root)
            manifest_before = ((decoder_root / "MANIFEST.sha256").read_bytes(),
                               (decoder_root / "MANIFEST.sha256").stat().st_mtime_ns)
            evidence = root / "evidence.json"
            evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            (root / "action").write_text(json.dumps([[0.0] * 10 for _ in range(16)]), encoding="utf-8")
            (root / "video").write_bytes(b"video")
            raw_status = {"status": "AWAITING_REVIEW", "phase": "PILOT", "generation_started": False,
                          "smoke_run_id": "raw-sibling-1", "hashes": {name: name + "-hash" for name in ("code", "model", "config", "direction", "input", "noise")},
                          "group": {"state": "bridge_0", "seed": 0}, "completed_samples": [], "failed_samples": [], "skipped_samples": []}
            events = []
            class Runtime:
                provenance = {"fixture": "runtime"}
                def actual_identity(self): return {"fixture": "runtime"}
                def cleanup(self): events.append("runtime_cleanup")
            runtime = Runtime()
            class Factory:
                def __init__(self, **kwargs): pass
                def build(self): events.append("factory_build"); return runtime, object(), object()
                def unload(self): events.append("factory_unload")
            class Monitor:
                instances = []
                def __init__(self, monitor_root, **kwargs):
                    self.root = Path(monitor_root); self._thread = object(); Monitor.instances.append(self)
                def start(self): events.append("load_monitor_start"); return self
                def check(self, **kwargs): return {"status": "OK"}
                def stop(self):
                    events.append("load_monitor_stop")
                    self.root.mkdir(parents=True, exist_ok=True)
                    (self.root / "gpu_samples.csv").write_bytes(b"flushed load telemetry\n")
            safe = lambda: {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                            "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
            decoder_calls = []
            def pilot(*args, **kwargs):
                # Simulate the generation runner's normal terminal monitor flush.
                kwargs["monitor"].stop()
                return dict(raw_status)
            def decoder_resume(*args, **kwargs):
                decoder_calls.append(Path(kwargs["decoder_root"]).resolve())
                decoder_api._verify_decoder_manifest(Path(kwargs["decoder_root"]).resolve())
                return {"status": "COMPLETE", "decoder_calls": 16}
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets",
                "extract_task5_directions", "OfficialRuntimeFactory", "resource_samplers", "ResourceMonitor", "run_pilot",
                "run_task6_decoder_replays")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {key: safe for key in ("gpu", "ram", "disk")}
                official.ResourceMonitor = Monitor
                official.run_pilot = pilot
                official.run_task6_decoder_replays = decoder_resume
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--decoder-root", str(decoder_root),
                    "--analysis-root", str(root / "analysis"), "--launch-contract", str(evidence), "--framework-root", str(root),
                    "--checkpoint", str(root / "checkpoint"), "--vae", str(root / "vae"), "--action", str(root / "action"),
                    "--video", str(root / "video"), "--task5-root", str(root), "--resume"])
                with mock.patch.object(official, "_inspect_raw_complete", return_value=raw_status, create=True), \
                     mock.patch("analyze_umi_task6.analyze_task6_run", return_value={"status": "COMPLETE"}):
                    result = official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            control_root = decoder_root.parent / (decoder_root.name + "_control")
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(decoder_calls, [decoder_root.resolve()])
            self.assertEqual(Monitor.instances[0].root.resolve(), (control_root / "load_monitor").resolve())
            self.assertTrue((control_root / "load_monitor" / "gpu_samples.csv").is_file())
            self.assertEqual(manifest_before, ((decoder_root / "MANIFEST.sha256").read_bytes(),
                                               (decoder_root / "MANIFEST.sha256").stat().st_mtime_ns))
            decoder_api._verify_decoder_manifest(decoder_root)

    def test_official_raw_complete_load_stop_marker_is_sibling_control_evidence(self):
        import run_umi_task6_official as official
        import umi_task6_decoder as decoder_api
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); decoder_root = root.parent / (root.name + "_decoder")
            decoder_root.mkdir()
            (decoder_root / "decoder_config.json").write_text("config\n", encoding="utf-8")
            (decoder_root / "replay.json").write_text("record\n", encoding="utf-8")
            decoder_api._decoder_manifest(decoder_root)
            evidence = root / "evidence.json"
            evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            (root / "action").write_text(json.dumps([[0.0] * 10 for _ in range(16)]), encoding="utf-8")
            (root / "video").write_bytes(b"video")
            raw_status = {"status": "AWAITING_REVIEW", "phase": "PILOT", "generation_started": False,
                          "smoke_run_id": "raw-stop-1", "hashes": {name: name + "-hash" for name in ("code", "model", "config", "direction", "input", "noise")},
                          "group": {"state": "bridge_0", "seed": 0}, "completed_samples": [], "failed_samples": [], "skipped_samples": []}
            class Runtime:
                def cleanup(self): pass
                def actual_identity(self): return {"fixture": "runtime"}
            class Factory:
                def __init__(self, **kwargs): pass
                def build(self): raise AssertionError("preload stop must block raw model load")
                def unload(self): pass
            class Monitor:
                def __init__(self, monitor_root, **kwargs): self.root = Path(monitor_root); self._thread = object()
                def start(self): return self
                def check(self, **kwargs): return {"status": "HARD_STOP", "reason_code": "LOAD_GATE", "reason": "load gate"}
                def stop(self): pass
            safe = lambda: {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                            "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets",
                "extract_task5_directions", "OfficialRuntimeFactory", "resource_samplers", "ResourceMonitor")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {key: safe for key in ("gpu", "ram", "disk")}
                official.ResourceMonitor = Monitor
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--decoder-root", str(decoder_root),
                    "--launch-contract", str(evidence), "--framework-root", str(root), "--checkpoint", str(root / "checkpoint"),
                    "--vae", str(root / "vae"), "--action", str(root / "action"), "--video", str(root / "video"),
                    "--task5-root", str(root), "--resume"])
                with mock.patch.object(official, "_inspect_raw_complete", return_value=raw_status, create=True):
                    result = official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            control_root = decoder_root.parent / (decoder_root.name + "_control")
            self.assertEqual(result["status"], "AWAITING_REVIEW")
            self.assertTrue((control_root / "load_resource_stop.json").is_file())
            self.assertFalse((decoder_root / "load_resource_stop.json").exists())
            decoder_api._verify_decoder_manifest(decoder_root)

    def test_official_raw_complete_monitor_close_failure_blocks_decoder_and_preserves_raw_evidence(self):
        import run_umi_task6_official as official
        import run_umi_task6_experiment as runner
        from test_run_umi_task6_experiment import Task6RunnerTests
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            api, runtime, inputs, counts, _ = Task6RunnerTests().public_chain(temp)
            safe = lambda: {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "gpu_reserved_gib": 0.0,
                            "ram_available_gib": 600.0, "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
            raw = api.run_pilot(runtime, inputs, temp,
                                monitor=api.ResourceMonitor(temp, gpu_sampler=safe, ram_sampler=safe, disk_sampler=safe))
            self.assertEqual(raw["status"], "AWAITING_REVIEW")
            status_path = root / "run_status.json"
            telemetry_names = ("gpu_samples.csv", "ram_samples.csv", "disk_samples.csv",
                               "sample_resource_snapshots.csv", "sample_resource_snapshots.jsonl")
            status_before = (status_path.read_bytes(), status_path.stat().st_mtime_ns)
            telemetry_before = {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns)
                                for name in telemetry_names}
            evidence = root / "official-evidence.json"
            evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            action = root / "official-action.json"; action.write_text(json.dumps(inputs.action.tolist()), encoding="utf-8")
            video = root / "official-video.mp4"; video.write_bytes(b"video")
            decoder_root = root.parent / (root.name + "_decoder")
            decoder_calls = []
            class Factory:
                def __init__(self, **kwargs): pass
                def build(self): return runtime, inputs, object()
                def unload(self): pass
            class ExplodingStop(runner.ResourceMonitor):
                def stop(self):
                    raise RuntimeError("derived load monitor join failed")
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets",
                "extract_task5_directions", "OfficialRuntimeFactory", "resource_samplers", "ResourceMonitor",
                "run_task6_decoder_replays")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {key: safe for key in ("gpu", "ram", "disk")}
                official.ResourceMonitor = ExplodingStop
                official.run_task6_decoder_replays = lambda *args, **kwargs: decoder_calls.append(True) or self.fail("decoder must be blocked by monitor close failure")
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--decoder-root", str(decoder_root),
                    "--launch-contract", str(evidence), "--framework-root", str(root), "--checkpoint", str(root / "checkpoint"),
                    "--vae", str(root / "vae"), "--action", str(action), "--video", str(video), "--task5-root", str(root), "--resume"])
                with mock.patch.object(official, "_inspect_raw_complete", return_value=raw, create=True):
                    with self.assertRaisesRegex(runner.ResourceStop, "derived load monitor join failed"):
                        official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            control_root = decoder_root.parent / (decoder_root.name + "_control")
            self.assertEqual(decoder_calls, [])
            self.assertTrue((control_root / "load_resource_stop.json").is_file())
            self.assertEqual((status_path.read_bytes(), status_path.stat().st_mtime_ns), status_before)
            self.assertEqual(telemetry_before, {name: ((root / name).read_bytes(), (root / name).stat().st_mtime_ns)
                                                for name in telemetry_names})

    def test_official_pilot_stops_preload_monitor_when_model_load_fails(self):
        import run_umi_task6_official as official
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            evidence = root / "evidence.json"
            evidence.write_text('{"task5": {}, "prompt": "Put the pot to the left of the purple item."}', encoding="utf-8")
            (root / "action").write_text(json.dumps([[0.0] * 10 for _ in range(16)]), encoding="utf-8")
            (root / "video").write_bytes(b"video")
            events = []
            class Factory:
                def __init__(self, **kwargs): events.append("factory_init")
                def build(self): events.append("factory_build"); raise RuntimeError("load failure")
                def unload(self): events.append("factory_unload")
            class Monitor:
                last_resources = {"gpu_used_gib": 0.0, "gpu_free_gib": 100.0, "ram_available_gib": 600.0,
                                  "rss_gib": 0.0, "swap_used_gib": 0.0, "disk_free_gib": 20.0}
                def __init__(self, *args, **kwargs): events.append("monitor_init")
                def start(self): events.append("monitor_start"); return self
                def check(self, **kwargs): events.append("monitor_check"); return {"status": "OK"}
                def stop(self): events.append("monitor_stop")
            safe = lambda: dict(Monitor.last_resources)
            old = {name: getattr(official, name) for name in ("_validate_static_contract", "verify_pinned_bridge_assets",
                                                               "extract_task5_directions", "OfficialRuntimeFactory",
                                                               "resource_samplers", "ResourceMonitor")}
            try:
                official._validate_static_contract = lambda contract: None
                official.verify_pinned_bridge_assets = lambda *args, **kwargs: {"action_sha256": "a" * 64, "video_sha256": "b" * 64}
                official.extract_task5_directions = lambda *args, **kwargs: {"bank": np.zeros((3, 1)), "manifest_sha256": "a" * 64,
                    "plan_sha256": "b" * 64, "direction_file_sha256": {}, "direction_sha256": {}}
                official.OfficialRuntimeFactory = Factory
                official.resource_samplers = lambda *args, **kwargs: {key: safe for key in ("gpu", "ram", "disk")}
                official.ResourceMonitor = Monitor
                args = official.parse_args(["--phase", "pilot", "--run-dir", str(root), "--launch-contract", str(evidence),
                    "--framework-root", str(root), "--checkpoint", str(root / "checkpoint"), "--vae", str(root / "vae"),
                    "--action", str(root / "action"), "--video", str(root / "video"), "--task5-root", str(root)])
                with self.assertRaisesRegex(RuntimeError, "load failure"):
                    official._execute_official_impl(args)
            finally:
                for name, value in old.items(): setattr(official, name, value)
            self.assertLess(events.index("monitor_start"), events.index("factory_build"), events)
            self.assertLess(events.index("factory_build"), events.index("monitor_stop"), events)
            self.assertEqual(events[-1], "factory_unload")

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
        checkpoint_hash = scan.sha256_tree(checkpoint) if checkpoint.is_dir() else scan.sha256_file(checkpoint)
        contract.update({
            "checkpoint_identity": {"sha256": checkpoint_hash},
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

    def _symlink_or_skip(self, link, target, *, target_is_directory):
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as error:
            winerror = getattr(error, "winerror", None)
            if winerror in (5, 1314) or error.errno in (errno.EACCES, errno.EPERM):
                self.skipTest(f"symlink creation is unavailable: {error}")
            raise


if __name__ == "__main__": unittest.main()
