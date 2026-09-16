import tempfile
import unittest
import json
from pathlib import Path

import numpy as np


class Task6RuntimeTests(unittest.TestCase):
    def setUp(self):
        import umi_task6_runtime as api
        self.api = api

    def inputs(self, **kwargs):
        carrier, indexes, mask, bank = self.fixture()
        from umi_fd_post_vae_bridge import sha256_array
        directions = self.api._derive_frozen_directions_unpinned(bank, mask)
        hashes = {key: sha256_array(value) for key, value in directions.items()}
        return self.api.Task6Inputs(carrier, indexes, mask, bank, z_bar=carrier,
                                    task4_reference={"fixture": True}, action=np.zeros((16, 10), np.float32),
                                    prompt="p", direction_hashes=hashes, **kwargs)

    def authorization(self, runtime, inputs):
        return {"status": "AWAITING_RESOURCE_REVIEW", "smoke_decision_accepted": True,
                "hashes": self.api.build_task6_hash_binding(runtime, inputs, self.api.task6_binding_config(inputs))}

    class Monitor:
        failure = None
        last_resources = {"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600, "disk_free_gib": 20}
        def start(self): return self
        def stop(self): return None
        def check(self, **kwargs): return {"status": "OK"}

    def fixture(self):
        carrier = np.ones((1, 48, 5, 16, 16), dtype=np.float32)
        mask = np.zeros_like(carrier, dtype=bool); mask[:, :, 0] = True
        bank = np.zeros((3,) + carrier.shape, dtype=np.float32)
        bank[:, mask] = 1.0
        return carrier, [0], mask, bank

    def reference_root(self, root, value=1):
        import hashlib, json
        fields = ("common_input_fp32", "initial_state", "consumed_initial_state", "sampler_input_state", "output_full")
        lines = []
        for name in ("baseline_pre", "baseline_post", "v0_alpha_00_plus", "v0_alpha_00_minus"):
            sample = Path(root, name); sample.mkdir()
            record = {}
            for field in fields:
                filename = field + ".npy"; np.save(sample / filename, np.array([value], np.float32)); record[field] = {"artifact": filename}
            (sample / "sample.json").write_text(json.dumps(record))
            artifacts = [sample / "sample.json", *(sample / (field + ".npy") for field in fields)]
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
            (sample / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": hashes}))
            for p in [*artifacts, sample / "status.json"]:
                lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}")
        manifest = Path(root, "MANIFEST.sha256"); manifest.write_text("\n".join(lines) + "\n")
        return Path(root), manifest

    def test_preflight_rejects_geometry_before_execute(self):
        carrier, indexes, mask, bank = self.fixture()
        with self.assertRaises(self.api.PreflightError):
            self.api.preflight_task6({"carrier": carrier, "condition_indexes": indexes,
                                      "mask": mask, "direction_bank": bank,
                                      "settings": {"autocast": True}})

    def test_strict_preflight_accepts_directory_checkpoint_and_file_vae(self):
        import hashlib, json
        import umi_task6_operational as operational
        inputs = self.inputs()
        class Runtime:
            def actual_identity(self): return {"runtime": "directory-checkpoint"}
        runtime = Runtime()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            action_path = root / "action.json"
            video_path = root / "video.mp4"
            checkpoint = root / "checkpoint"
            vae_path = root / "vae.bin"
            direction_path = root / "directions.npy"
            action = inputs.action.tolist()
            action_path.write_text(json.dumps(action), encoding="utf-8")
            video_path.write_bytes(b"video")
            checkpoint.mkdir(); (checkpoint / "model.bin").write_bytes(b"model")
            vae_path.write_bytes(b"vae")
            np.save(direction_path, np.stack([inputs.directions[key] for key in ("v0", "v1", "v2")]), allow_pickle=False)
            file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            direction_hashes = {key: self.api._array_sha(value) for key, value in inputs.directions.items()}
            config = {
                "environment": {"torch": "fixture", "cuda": "fixture"},
                "provenance": {"source_commit": operational.SOURCE_COMMIT},
                "asset_hashes": {"action": file_sha(action_path), "video": file_sha(video_path),
                                  "checkpoint": operational.checkpoint_content_identity(checkpoint), "vae": file_sha(vae_path)},
                "asset_paths": {"action": str(action_path), "video": str(video_path),
                                "checkpoint": str(checkpoint), "vae": str(vae_path)},
                "checkpoint_path": str(checkpoint), "vae_path": str(vae_path),
                "carrier_shape": list(inputs.z0.shape), "carrier_hash": inputs.identity()["z0"],
                "condition_indexes": [0], "predicted_indexes": [1, 2, 3, 4],
                "mask_shape": list(inputs.geometry.mask.shape), "mask_hash": inputs.geometry.metadata()["mask_sha256"],
                "action": action, "action_shape": [16, 10], "action_hash": self.api._array_sha(inputs.action),
                "prompt": inputs.prompt, "direction_hashes": direction_hashes,
                "direction_bank_path": str(direction_path), "direction_bank_file_hash": file_sha(direction_path),
                "settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "batch_size": 1,
                             "autocast": False, "tf32": False, "diffusion_cache": False},
                "seed_config": {"seed": 0, "prepare": 0, "sampler": 0, "scheduler": 0},
                "observed_runtime_identity": runtime.actual_identity(), "observed_input_identity": inputs.identity(),
            }
            result = self.api.preflight_task6(config, strict=True, runtime=runtime, inputs=inputs)
            self.assertEqual(result["status"], "PASS")

    def test_inputs_require_real_geometry_action_prompt_and_direction_manifest(self):
        carrier, indexes, mask, bank = self.fixture()
        with self.assertRaises(ValueError):
            self.api.Task6Inputs(carrier, indexes, mask, bank, action=np.zeros((15, 10), np.float32), prompt="p")
        with self.assertRaises(ValueError):
            self.api.Task6Inputs(carrier, indexes, mask, bank, action=np.zeros((16, 10), np.float32), prompt="")

    def test_direction_manifest_required(self):
        carrier, indexes, mask, bank = self.fixture()
        with self.assertRaises(ValueError): self.api.load_frozen_directions(bank, mask)

    def test_public_group_runner_and_authorization_helper_are_unavailable(self):
        self.assertFalse(hasattr(self.api, "run_task6_group"))
        self.assertFalse(hasattr(self.api, "build_task6_authorization"))

    def test_perturbation_realized_mask_rms_uses_s_z_and_preserves_exterior(self):
        inputs = self.inputs()
        for alpha in (0.001, 0.003, 0.01):
            for sign in (-1, 1):
                target = inputs.for_spec({"state": "bridge_0", "seed": 0, "kind": "perturbation",
                                          "direction_id": "v0", "alpha": alpha, "sign": sign})
                delta = target - inputs.z_bar
                self.assertAlmostEqual(float(np.sqrt(np.mean(delta[inputs.geometry.mask].astype(np.float64) ** 2))),
                                       alpha * inputs.s_z, places=7)
                self.assertTrue(np.array_equal(delta[~inputs.geometry.mask], np.zeros_like(delta[~inputs.geometry.mask])))

    def test_adapter_uses_official_full_scope_and_compatible_runtime_inputs(self):
        inputs = self.inputs()
        class StrictRuntime:
            provenance = {"seed": 0}
            def __init__(self): self.inputs = inputs; self.seen = None
            def actual_identity(self): return {"runtime": "strict"}
            def execute(self, spec, runtime_inputs, *, scope):
                if runtime_inputs is not self.inputs: raise ValueError("identity mismatch")
                if scope != "full": raise ValueError("scope mismatch")
                self.seen = (spec, scope); return {"output_full": runtime_inputs.for_spec(spec)}
        runtime = StrictRuntime(); adapter = self.api.Task6RuntimeAdapter(runtime, inputs)
        result = adapter.execute({"kind": "perturbation", "state": "bridge_0", "seed": 0,
                                  "alpha": 0.001, "sign": 1, "direction_id": "v0"}, scope="full")
        self.assertIn("output_full", result); self.assertEqual(runtime.seen[0]["group"], "C")
        self.assertTrue(np.array_equal(result["output_full"], inputs.for_spec(runtime.seen[0])))

    def test_adapter_publishes_immutable_task6_input_schema_for_baseline_and_perturbation(self):
        inputs = self.inputs()
        class StrictRuntime:
            def __init__(self): self.inputs = inputs
            def execute(self, spec, runtime_inputs, *, scope): return {"output_full": np.zeros((1, 48, 4, 16, 16), np.float32)}
        adapter = self.api.Task6RuntimeAdapter(StrictRuntime(), inputs)
        for spec in ({"state":"bridge_0","seed":0,"kind":"baseline","alpha":0.0,"sign":0}, {"state":"bridge_0","seed":0,"kind":"perturbation","direction_id":"v0","alpha":0.001,"sign":1}):
            record = adapter.execute(spec)
            for key in ("z_bar", "mask", "direction", "actual_delta_fp32", "target_delta_fp32", "s_z", "spec", "group", "predicted_latent"):
                self.assertIn(key, record)
            self.assertEqual(record["group"], {"state":"bridge_0","seed":0})
            self.assertEqual(record["predicted_latent"].dtype, np.float32)

    def test_adapter_saves_observed_common_input_and_delta_not_requested_target(self):
        inputs = self.inputs()
        observed = inputs.z_bar.copy()
        observed[inputs.geometry.mask] += np.float32(0.25)
        class StrictRuntime:
            def __init__(self): self.inputs = inputs
            def execute(self, spec, runtime_inputs, *, scope):
                return {"output_full": np.zeros((1, 48, 4, 16, 16), np.float32),
                        "common_input_fp32": observed.copy()}
        spec = {"state": "bridge_0", "seed": 0, "kind": "baseline", "alpha": 0.0, "sign": 0}
        record = self.api.Task6RuntimeAdapter(StrictRuntime(), inputs).execute(spec)
        np.testing.assert_array_equal(record["consumed_input_fp32"], observed)
        np.testing.assert_array_equal(record["actual_delta_fp32"], observed - inputs.z_bar)

    def test_adapter_slices_carrier_fallback_using_validated_prediction_indexes(self):
        inputs = self.inputs()
        class StrictRuntime:
            def __init__(self): self.inputs = inputs
            def execute(self, spec, runtime_inputs, *, scope): return {"output_full": runtime_inputs.z0.copy()}
        record = self.api.Task6RuntimeAdapter(StrictRuntime(), inputs).execute({"state": "bridge_0", "seed": 0, "kind": "baseline", "alpha": 0.0, "sign": 0})
        self.assertEqual(record["predicted_latent"].shape[2], 4)
        self.assertEqual(record["predicted_latent_source"], "runtime_carrier_sliced_by_predicted_indexes")

    def test_adapter_rejects_noncarrier_wrong_prediction_block_shape(self):
        inputs = self.inputs()
        class StrictRuntime:
            def __init__(self): self.inputs = inputs
            def execute(self, spec, runtime_inputs, *, scope): return {"output_full": np.zeros((1, 48, 3, 16, 16), np.float32)}
        with self.assertRaises(ValueError):
            self.api.Task6RuntimeAdapter(StrictRuntime(), inputs).execute({"state": "bridge_0", "seed": 0, "kind": "baseline", "alpha": 0.0, "sign": 0})

    def test_adapter_cleanup_uses_runtime_seam(self):
        inputs = self.inputs(); calls = []
        class StrictRuntime:
            def __init__(self): self.inputs = inputs; self.request_cache = {}
            def cleanup(self): calls.append("cleanup")
        adapter = self.api.Task6RuntimeAdapter(StrictRuntime(), inputs); adapter.cleanup()
        self.assertEqual(calls, ["cleanup"])

    def test_direction_bank_is_frozen_and_extra_direction_rejected(self):
        carrier, indexes, mask, bank = self.fixture()
        with self.assertRaises(ValueError):
            self.api.load_frozen_directions(np.concatenate([bank, bank[:1]]), mask)
        derived = self.api._derive_frozen_directions_unpinned(bank, mask)
        frozen = self.api.load_frozen_directions(bank, mask, expected_hashes={key: self.api._array_sha(value) for key, value in derived.items()})
        self.assertEqual(tuple(frozen), ("v0", "v1", "v2", "u01", "u12"))

    def test_runner_executes_exact_plan_and_resume_is_immutable(self):
        carrier, indexes, mask, bank = self.fixture()
        inputs = self.inputs()
        class FakeRuntime:
            provenance = {"seed": 0, "diffusion_cache": "off"}
            def __init__(self): self.calls = []
            def actual_identity(self): return {"fixture": "runtime"}
            def execute(self, spec, inputs, *, scope="full"):
                self.calls.append(spec["sample_id"])
                return {"output_full": inputs.for_spec(spec), "spec": dict(spec),
                        "scope": scope, "raw": np.array([len(self.calls)], np.float32)}
        with tempfile.TemporaryDirectory() as temp:
            runtime = FakeRuntime()
            result = self.api._run_task6_group(runtime, inputs, temp, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())
            self.assertEqual(result["status"], "AWAITING_REVIEW")
            self.assertEqual(result["successful_samples"], 32)
            self.assertEqual(len(runtime.calls), 32)
            self.assertEqual(self.api._run_task6_group(runtime, inputs, temp, resume=True, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())["successful_samples"], 32)
            self.assertEqual(len(runtime.calls), 32)

    def test_complete_raw_resume_skips_generation_and_preserves_status_bytes(self):
        inputs = self.inputs()
        class Runtime:
            provenance = {"seed": 0}
            def __init__(self, fail=False): self.calls = 0; self.fail = fail
            def actual_identity(self): return {"fixture": "raw-resume"}
            def execute(self, spec, inputs, *, scope="full"):
                self.calls += 1
                if self.fail: raise AssertionError("generation must be skipped for a complete raw resume")
                return {"output_full": inputs.for_spec(spec)}
            def cleanup(self): pass
        with tempfile.TemporaryDirectory() as temp:
            first_runtime = Runtime()
            first = self.api._run_task6_group(first_runtime, inputs, temp,
                authorization=self.authorization(first_runtime, inputs), monitor=self.Monitor())
            self.assertEqual(first["status"], "AWAITING_REVIEW"); self.assertEqual(first_runtime.calls, 32)
            status_path = Path(temp) / "run_status.json"
            before = (status_path.read_bytes(), status_path.stat().st_mtime_ns)
            resumed_runtime = Runtime(fail=True)
            resumed = self.api._run_task6_group(resumed_runtime, inputs, temp, resume=True,
                authorization=self.authorization(resumed_runtime, inputs), monitor=self.Monitor())
            self.assertEqual(resumed, json.loads(before[0].decode()))
            self.assertEqual(resumed_runtime.calls, 0)
            self.assertEqual((status_path.read_bytes(), status_path.stat().st_mtime_ns), before)

    def test_complete_raw_resume_does_not_stop_an_unstarted_monitor(self):
        inputs = self.inputs()
        class Runtime:
            provenance = {"seed": 0}
            def __init__(self): self.calls = 0
            def actual_identity(self): return {"fixture": "unstarted-monitor"}
            def execute(self, spec, inputs, *, scope="full"):
                self.calls += 1
                return {"output_full": inputs.for_spec(spec)}
            def cleanup(self): pass
        class UnstartedMonitor:
            _thread = None
            failure = None
            last_resources = {}
            def __init__(self): self.stopped = False
            def stop(self): self.stopped = True
        with tempfile.TemporaryDirectory() as temp:
            runtime = Runtime()
            first = self.api._run_task6_group(runtime, inputs, temp,
                authorization=self.authorization(runtime, inputs), monitor=self.Monitor())
            self.assertEqual(first["status"], "AWAITING_REVIEW")
            monitor = UnstartedMonitor()
            resumed = self.api._run_task6_group(runtime, inputs, temp, resume=True,
                authorization=self.authorization(runtime, inputs), monitor=monitor)
            self.assertEqual(resumed["status"], "AWAITING_REVIEW")
            self.assertFalse(monitor.stopped)

    def test_equivalence_requires_four_calls_and_reports_reuse(self):
        import hashlib
        class Runtime:
            def __init__(self, equal=True): self.calls = 0; self.equal = equal
            def execute(self, spec, inputs, *, scope="equivalence"):
                self.calls += 1
                value = 1 if self.equal else self.calls
                return {key: np.array([value], dtype=np.float32) for key in ("common_input_fp32", "initial_state", "consumed_initial_state", "sampler_input_state", "output_full")}
        with tempfile.TemporaryDirectory() as temp:
            root, manifest = self.reference_root(temp)
            result = self.api.verify_reference_reuse(Runtime(), None, reference_artifacts=root,
                                                     reference_manifest=manifest, execute=True)
        self.assertTrue(result["reusable"]); self.assertEqual(result["calls"], 4)

    def test_reuse_compares_each_corresponding_saved_artifact(self):
        class Runtime:
            def __init__(self): self.calls = 0
            def execute(self, spec, inputs, *, scope="full"):
                self.calls += 1
                return {key: np.array([self.calls], dtype=np.float32) for key in ("common_input_fp32", "initial_state", "consumed_initial_state", "sampler_input_state", "output_full")}
        with tempfile.TemporaryDirectory() as temp:
            root, manifest = self.reference_root(temp)
            result = self.api.verify_reference_reuse(Runtime(), None, reference_artifacts=root,
                                                     reference_manifest=manifest)
            self.assertFalse(result["reusable"]); self.assertEqual(result["calls"], 4)

    def test_reference_mapping_rejected(self):
        with self.assertRaises(ValueError): self.api.verify_reference_reuse(object(), None, reference_artifacts={"baseline_pre": object()})

    def test_root_manifest_tamper_rejected(self):
        class Runtime:
            def execute(self, spec, inputs, *, scope="full"):
                return {key: np.array([1], dtype=np.float32) for key in ("common_input_fp32", "initial_state", "consumed_initial_state", "sampler_input_state", "output_full")}
        with tempfile.TemporaryDirectory() as temp:
            root, manifest = self.reference_root(temp); manifest.write_text(manifest.read_text() + "tamper")
            with self.assertRaises(ValueError): self.api.verify_reference_reuse(Runtime(), None, reference_artifacts=root, reference_manifest=manifest)

    def test_resume_refuses_tampered_success(self):
        carrier, indexes, mask, bank = self.fixture()
        inputs = self.inputs()
        class Runtime:
            provenance = {"seed": 0}
            def actual_identity(self): return {"fixture": "tamper"}
            def execute(self, spec, inputs, *, scope="full"): return {"output_full": inputs.for_spec(spec)}
        with tempfile.TemporaryDirectory() as temp:
            runtime = Runtime(); self.api._run_task6_group(runtime, inputs, temp, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())
            sample = Path(temp, "samples", "bridge_0__seed_0__baseline_pre", "sample.json")
            sample.write_text(sample.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
            with self.assertRaises(ValueError): self.api._run_task6_group(runtime, inputs, temp, resume=True, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())

    def test_resume_refuses_tampered_npy_artifact(self):
        inputs = self.inputs()
        class Runtime:
            provenance = {"seed": 0}
            def actual_identity(self): return {"fixture": "tamper-npy"}
            def execute(self, spec, inputs, *, scope="full"): return {"output_full": inputs.for_spec(spec), "raw": np.array([1], np.float32)}
        with tempfile.TemporaryDirectory() as temp:
            runtime = Runtime(); self.api._run_task6_group(runtime, inputs, temp, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())
            artifact = Path(temp, "samples", "bridge_0__seed_0__baseline_pre", "raw.npy")
            artifact.write_bytes(artifact.read_bytes() + b"tampered")
            with self.assertRaises(ValueError): self.api._run_task6_group(runtime, inputs, temp, resume=True, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())

    def test_resume_requires_root_manifest(self):
        inputs = self.inputs()
        class Runtime:
            provenance = {"seed": 0}
            def actual_identity(self): return {"fixture": "manifest"}
            def execute(self, spec, inputs, *, scope="full"): return {"output_full": inputs.for_spec(spec)}
        with tempfile.TemporaryDirectory() as temp:
            runtime = Runtime(); self.api._run_task6_group(runtime, inputs, temp, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())
            Path(temp, "MANIFEST.sha256").unlink()
            with self.assertRaises(ValueError): self.api._run_task6_group(runtime, inputs, temp, resume=True, authorization=self.authorization(runtime, inputs), monitor=self.Monitor())


if __name__ == "__main__": unittest.main()
