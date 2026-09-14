import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from umi_fd_post_vae_scan import (
    ALPHAS,
    EXPECTED_CALL_COUNT,
    GateResult,
    OfficialPostVaeRuntimeAdapter,
    SampleStore,
    assert_resume_compatible,
    build_call_plan,
    evaluate_stage_a_gate,
    parse_args,
    prepare_sample_dir_for_run,
    local_vae_experiment_overrides,
    run_experiment,
    run_stage_a_and_scan,
    sha256_tree,
    validate_sample_evidence,
    _capture_network_condition_before_denoise,
    _reimpose_packed_condition,
)
from umi_fd_post_vae_bridge import build_broadcast_condition_mask, sha256_array


class PlanTests(unittest.TestCase):
    def test_plan_is_exactly_four_stage_calls_and_fifty_scan_calls(self):
        plan = build_call_plan()
        self.assertEqual(len(plan), EXPECTED_CALL_COUNT)
        self.assertEqual([item["sample_id"] for item in plan[:4]], ["A", "B", "C", "D"])
        self.assertEqual(plan[4]["sample_id"], "scan_pre")
        self.assertEqual(plan[-1]["sample_id"], "scan_post")
        scan = plan[5:-1]
        self.assertEqual(len(scan), 48)
        self.assertEqual([(item["direction_index"], item["alpha"], item["sign"]) for item in scan[:4]], [
            (0, ALPHAS[0], -1), (0, ALPHAS[0], 1),
            (0, ALPHAS[1], -1), (0, ALPHAS[1], 1),
        ])
        self.assertEqual(scan[-1]["direction_index"], 3)
        self.assertEqual(scan[-1]["alpha"], ALPHAS[-1])
        self.assertEqual(scan[-1]["sign"], 1)


class GateTests(unittest.TestCase):
    def _records(self):
        carrier = np.arange(32, dtype=np.float32).reshape(1, 1, 2, 2, 8)
        latent = np.arange(32, dtype=np.float32).reshape(1, 1, 2, 2, 8)
        decoded = np.arange(60, dtype=np.float32).reshape(3, 5, 2, 2)
        mask = np.zeros_like(carrier, dtype=bool)
        mask[:, :, 0, :, :] = True
        network_hash = sha256_array(carrier)
        common = {
            "carrier_fp32": carrier, "initial_state": carrier.copy(), "initial_condition_mask": mask.copy(), "network_condition_bf16": carrier,
            "predicted_latent": latent[:, :, 1:2].copy(), "decoded_float": decoded,
            "final_latent_full": latent.copy(), "decoded_final": decoded[:, 4], "latent_slicing": {"predicted_indexes": [1], "axis": 2, "source_shape": list(latent.shape), "selected_shape": list(latent[:, :, 1:2].shape)}, "image_slicing": {"frame_index": 4, "axis": 1, "source_shape": list(decoded.shape)},
            "initial_noise_hash": "noise", "condition_mask": mask,
            "condition_step_hashes": [network_hash] * 30, "first_denoise_hash": network_hash, "final_denoise_hash": network_hash, "expected_network_condition_hash": network_hash,
            "network_condition_dtype": "torch.bfloat16", "text_kv_lifecycle": [{"before": []}, {"before": ["request"]}],
            "cfg_branch_semantics": {"guidance": 1.0, "conditional_calls": 30, "unconditional_calls": 0, "expected_multiplicity": 1},
            "cfg_branch_observations": [{"branch": "conditional", "used": True, "step_index": index} for index in range(30)],
            "cfg_branch_calls": 30,
            "denoise_step_evidence": [{"step_index": index, "timestep": float(30 - index)} for index in range(30)],
            "expected_denoise_steps": 30, "cache_lifecycle_valid": True,
        }
        records = {name: dict(common) for name in "ABCD"}
        records["C"].update({
            "actual_delta_fp32": np.where(mask, 1.0, 0.0).astype(np.float32),
            "actual_delta_bf16": np.where(mask, 1.0, 0.0).astype(np.float32),
            "condition_step_hashes": [network_hash] * 30,
            "first_denoise_hash": network_hash, "final_denoise_hash": network_hash,
            "expected_network_condition_hash": network_hash, "network_condition_dtype": "torch.bfloat16",
            "predicted_latent_sliced": True, "decoded_final_frame": True,
            "predicted_latent": latent[:, :, 1:2].copy(), "decoded_final": decoded[:, 4].copy(),
            "final_latent_full": latent.copy(), "latent_slicing": {"predicted_indexes": [1], "axis": 2, "source_shape": list(latent.shape), "selected_shape": list(latent[:, :, 1:2].shape)}, "image_slicing": {"frame_index": 4, "axis": 1, "source_shape": list(decoded.shape)},
        })
        return records

    def test_gate_passes_and_reports_mismatch_without_running_scan(self):
        result = evaluate_stage_a_gate(self._records())
        self.assertIsInstance(result, GateResult)
        self.assertTrue(result.passed)
        records = GateTests()._records()
        records["B"]["decoded_float"] = records["B"]["decoded_float"].copy()
        records["B"]["decoded_float"][0] += 1
        failed = evaluate_stage_a_gate(records)
        self.assertFalse(failed.passed)
        self.assertTrue(any("A/B" in item for item in failed.failures))

    def test_c_evidence_failure_is_diagnostic(self):
        records = GateTests()._records()
        records["C"]["condition_step_hashes"] = ["first", "changed"]
        result = evaluate_stage_a_gate(records)
        self.assertFalse(result.passed)
        self.assertTrue(any("C" in item for item in result.failures))

    def test_gate_rejects_empty_nan_shape_or_dtype_mismatch_and_signed_zero(self):
        records = GateTests()._records()
        records["B"]["decoded_float"] = records["B"]["decoded_float"].astype(np.float64)
        result = evaluate_stage_a_gate(records)
        self.assertFalse(result.passed)
        records = self._records()
        records["B"]["decoded_float"] = np.empty((0,), dtype=np.float32)
        self.assertFalse(evaluate_stage_a_gate(records).passed)

    def test_gate_uses_fixed_thirty_steps_and_all_hashes(self):
        records = self._records()
        records["C"]["expected_denoise_steps"] = 2
        self.assertFalse(evaluate_stage_a_gate(records).passed)
        records = self._records()
        records["C"]["condition_step_hashes"][-1] = "wrong"
        self.assertFalse(evaluate_stage_a_gate(records).passed)
        records = self._records()
        records["C"]["network_condition_bf16"] = records["C"]["network_condition_bf16"].copy()
        records["C"]["network_condition_bf16"][0] += 1
        self.assertFalse(evaluate_stage_a_gate(records).passed)
        records = self._records()
        records["C"]["cfg_branch_observations"][0]["used"] = False
        self.assertFalse(evaluate_stage_a_gate(records).passed)
        records = self._records()
        records["C"]["actual_delta_fp32"] = np.array([np.nan] * 8, dtype=np.float32)
        self.assertFalse(evaluate_stage_a_gate(records).passed)
        records = self._records()
        records["B"]["network_condition_bf16"] = records["A"]["network_condition_bf16"].copy()
        records["B"]["network_condition_bf16"][0] = np.array(-0.0, dtype=np.float32)
        records["A"]["network_condition_bf16"][0] = np.array(0.0, dtype=np.float32)
        self.assertFalse(evaluate_stage_a_gate(records).passed)

    def test_predicted_indexes_are_derived_from_authoritative_mask(self):
        records = self._records()
        full = np.arange(32, dtype=np.float32).reshape(1, 1, 2, 2, 8)
        mask = np.zeros_like(full, dtype=bool)
        mask[:, :, 0, :, :] = True
        records["C"].update({
            "carrier_fp32": full.copy(),
            "condition_mask": mask.copy(),
            "initial_state": full.copy(),
            "initial_condition_mask": mask.copy(),
            "final_latent_full": full.copy(),
            "predicted_latent": full[:, :, 1:2, :, :].copy(),
            "latent_slicing": {"predicted_indexes": [0], "axis": 2, "source_shape": list(full.shape), "selected_shape": list(full[:, :, 1:2, :, :].shape)},
        })
        result = validate_sample_evidence(records["C"], stage_a=True)
        self.assertFalse(result.passed)
        self.assertTrue(any("predicted" in failure for failure in result.failures))

    def test_decoded_final_must_be_last_frame_not_any_matching_frame(self):
        records = self._records()
        records["C"]["decoded_final"] = records["C"]["decoded_float"][:, 1].copy()
        records["C"]["image_slicing"] = {"frame_index": 1, "axis": 1, "source_shape": list(records["C"]["decoded_float"].shape), "selected_shape": list(records["C"]["decoded_final"].shape)}
        result = validate_sample_evidence(records["C"], stage_a=True)
        self.assertFalse(result.passed)
        self.assertTrue(any("decoded final" in failure or "image slicing" in failure for failure in result.failures))

    def test_cfg_observations_must_cover_all_denoise_steps_and_step_identity(self):
        records = self._records()
        records["C"]["cfg_branch_observations"] = records["C"]["cfg_branch_observations"][:1]
        records["C"]["cfg_branch_calls"] = 1
        records["C"]["cfg_branch_semantics"]["conditional_calls"] = 1
        result = validate_sample_evidence(records["C"], stage_a=True)
        self.assertFalse(result.passed)
        self.assertTrue(any("CFG" in failure or "denoise" in failure for failure in result.failures))
        records = self._records()
        records["C"]["cfg_branch_observations"][3]["step_index"] = 99
        result = validate_sample_evidence(records["C"], stage_a=True)
        self.assertFalse(result.passed)


class AtomicAndResumeTests(unittest.TestCase):
    def test_atomic_success_is_preserved_and_incomplete_is_archived_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SampleStore(root)
            store.write_success("A", {"status": "success", "value": 1})
            self.assertEqual(store.prepare("A", resume=True), "skip")
            (root / "B").mkdir()
            (root / "B" / "partial.json").write_text("{}", encoding="utf-8")
            self.assertEqual(store.prepare("B", resume=True), "run")
            self.assertTrue((root / "B.attempt01" / "partial.json").is_file())

    def test_attempt_lifecycle_writes_tensor_artifacts_without_json_serialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SampleStore(tmp)
            store.prepare("A")
            started = json.loads((Path(tmp) / "A" / "status.json").read_text(encoding="utf-8"))["started_utc"]
            path = store.write_success("A", {"status": "success"}, artifacts={"latent.npy": np.arange(4, dtype=np.float32)})
            self.assertTrue((path / "latent.npy").is_file())
            np.testing.assert_array_equal(np.load(path / "latent.npy"), np.arange(4, dtype=np.float32))
            status = json.loads((path / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["started_utc"], started)
            self.assertIn("finished_utc", status)

    def test_invocation_history_is_append_only_across_stage_a_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            from umi_fd_post_vae_scan import append_invocation_history
            append_invocation_history(root, {"argv": ["stage-a"], "started_utc": "2026-09-12T00:00:00+00:00"})
            append_invocation_history(root, {"argv": ["resume"], "started_utc": "2026-09-12T01:00:00+00:00"})
            history = json.loads((root / "invocation_history.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["argv"] for entry in history], [["stage-a"], ["resume"]])

    def test_failed_attempt_preserves_partial_artifacts_and_success_requires_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = SampleStore(root)
            store.prepare("A")
            (root / "A" / "partial.npy").write_bytes(b"partial")
            store.write_failure("A", {"status": "fail"})
            self.assertTrue((root / "A.attempt01" / "partial.npy").is_file())
            store.write_success("B", {"status": "success"}, artifacts={"latent.npy": np.ones(2, dtype=np.float32)})
            (root / "B" / "latent.npy").write_bytes(b"corrupt")
            self.assertEqual(store.prepare("B", resume=True), "run")

    def test_resume_requires_every_compatibility_field_to_match(self):
        requested = {key: key for key in (
            "framework_sha256", "model_sha256", "vae_sha256", "code_sha256", "config_sha256",
            "bridge_sha256", "input_sha256", "action_sha256", "direction_sha256", "z0_sha256", "mask_sha256", "noise_policy_sha256",
        )}
        assert_resume_compatible(requested, dict(requested))
        with self.assertRaisesRegex(ValueError, "compatibility"):
            assert_resume_compatible(requested, {"framework": "a"})
        with self.assertRaisesRegex(ValueError, "compatibility"):
            assert_resume_compatible(requested, dict(requested, mask_sha256="changed"))

    def test_tree_provenance_changes_when_checkpoint_or_framework_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "a").write_text("1", encoding="utf-8")
            first = sha256_tree(root)
            (root / "a").write_text("2", encoding="utf-8")
            self.assertNotEqual(first, sha256_tree(root))

    def test_flat_official_mask_is_reshaped_only_after_exact_size_match(self):
        carrier_shape = (1, 2, 3, 2, 2)
        flat = np.zeros(np.prod(carrier_shape), dtype=bool)
        flat.reshape(carrier_shape)[:, :, 0] = True
        mask = build_broadcast_condition_mask([0], flat, carrier_shape)
        self.assertEqual(mask.shape, carrier_shape)
        with self.assertRaises(ValueError):
            build_broadcast_condition_mask([0], flat[:-1], carrier_shape)
        bad_indexes = flat.reshape(carrier_shape).copy()
        bad_indexes[:, :, 1] = True
        with self.assertRaises(ValueError):
            build_broadcast_condition_mask([0], bad_indexes, carrier_shape)

    def test_tensor_like_payload_is_detached_and_persisted_as_numpy_artifact(self):
        class TensorLike:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        with tempfile.TemporaryDirectory() as tmp:
            store = SampleStore(tmp)
            store.write_success("tensor", {"tensor": TensorLike(np.arange(3, dtype=np.float32))})
            loaded = store.load_record("tensor")
            np.testing.assert_array_equal(loaded["payload_tensor"], np.arange(3, dtype=np.float32))

    def test_bfloat_tensor_falls_back_through_float_before_numpy(self):
        class FloatTensor:
            def __init__(self, value): self.value = value
            def cpu(self): return self
            def numpy(self): return self.value
        class BFloatTensor:
            def __init__(self, value): self.value = value
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): raise TypeError("unsupported bfloat16 NumPy conversion")
            def float(self): return FloatTensor(self.value.astype(np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            store = SampleStore(tmp)
            store.write_success("bf16", {"tensor": BFloatTensor(np.arange(3, dtype=np.float32))})
            loaded = store.load_record("bf16")
            np.testing.assert_array_equal(loaded["payload_tensor"], np.arange(3, dtype=np.float32))


class AdapterAndCliTests(unittest.TestCase):
    def test_reimpose_packed_condition_changes_mask_only(self):
        reference = np.arange(8, dtype=np.float32).reshape(2, 2, 2, 1)
        tokens = reference.copy()
        tokens[0, 0] = 99.0
        tokens[1, 1] = 77.0
        mask = np.array([[[True]], [[False]]], dtype=bool)
        capture = {"packed_condition_reference": reference}
        _reimpose_packed_condition(capture, tokens, mask)
        np.testing.assert_array_equal(tokens[0], reference[0])
        self.assertTrue(np.all(tokens[1, 1] == 77.0))

    def test_network_condition_capture_is_pre_call_and_survives_in_place_mutation(self):
        tokens = np.array([[1.0, 2.0]], dtype=np.float32)
        packed = SimpleNamespace(
            vision=SimpleNamespace(
                tokens=[tokens],
                condition_mask=np.ones_like(tokens, dtype=bool),
                timesteps=np.array([3.0], dtype=np.float32),
            )
        )
        capture = {
            "denoise_step_evidence": [],
            "cfg_branch_calls": 0,
            "cfg_branch_observations": [],
            "denoise_step_hashes": [],
        }
        _capture_network_condition_before_denoise(capture, (packed,), {})
        expected_hash = capture["denoise_step_hashes"][0]
        expected_tokens = capture["network_condition_bf16"].copy()
        def original_denoise(*args, **kwargs):
            # Reproduce the official sampler's request-local in-place mutation
            # after the network boundary.
            args[0].vision.tokens[0][0, 0] = 9.0

        original_denoise(packed)
        # The mutation after the boundary must not rewrite copied evidence.
        self.assertEqual(capture["denoise_step_hashes"], [expected_hash])
        np.testing.assert_array_equal(capture["network_condition_bf16"], expected_tokens)

    def test_adapter_does_not_encode_twice_and_returns_fresh_clones(self):
        calls = []
        source = {"vision": np.arange(4, dtype=np.float32)}
        adapter = OfficialPostVaeRuntimeAdapter(lambda: calls.append(1) or source)
        first = adapter.condition_for_call("A")
        first["vision"][0] = 99
        second = adapter.condition_for_call("B", delta=np.zeros(4, dtype=np.float32), inject=lambda item, delta: item)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second["vision"][0], 0)

    def test_adapter_binds_model_owned_official_methods_and_restores_wrappers(self):
        class GenerationDataClean:
            __slots__ = ("x0_tokens_vision",)
            def __init__(self, tokens): self.x0_tokens_vision = [tokens]
        class SequencePlan:
            condition_frame_indexes_vision = [0]
        class Model:
            def __init__(self):
                self.calls = 0
                self.text_kv_cache = {}
                self.generate_kwargs = None
                self.tokenizer_vision_gen = SimpleNamespace(_keep_decoder_cache=False, _dec_cache={})
            def get_data_and_condition(self, batch, **kwargs):
                self.calls += 1
                return GenerationDataClean(np.ones((1, 2, 2, 1), dtype=np.float32))
            def _prepare_inference_data(self, batch, seed):
                data = self.get_data_and_condition(batch)
                initial = [np.ones((1, 2, 2, 1), dtype=np.float32)]
                masks = [np.array([[[True]], [[False]]], dtype=bool)]
                reference = [np.arange(4, dtype=np.float32)]
                return ([SequencePlan()], data, [], [], initial, reference, masks, False)
            def denoise(self, *args, **kwargs):
                self.text_kv_cache["request"] = 1
                return {"preds_vision": [np.zeros((1, 2, 2, 1), dtype=np.float32)]}
            def decode_vision(self, latent):
                return np.full((1, 3, 2, 1, 1), 0.5, dtype=np.float32)
            def generate_samples_from_batch(self, batch, **kwargs):
                self.generate_kwargs = kwargs
                self._prepare_inference_data(batch, [0])
                packed = SimpleNamespace(vision=SimpleNamespace(tokens=[np.ones((1, 2, 2, 1), dtype=np.float32)], condition_mask=[np.array([[[True]], [[False]]], dtype=bool)]))
                self.denoise(packed)
                self.denoise(packed)
                self.decode_vision(np.ones((1, 2, 2, 1), dtype=np.float32))
                return {"vision": [np.ones((1, 2, 2, 1), dtype=np.float32)]}
        class Pipeline:
            def __init__(self): self.model = Model()
        adapter = OfficialPostVaeRuntimeAdapter.from_pipeline(Pipeline())
        record = adapter.execute_call({"sample_id": "A", "injection": "normal"}, data_batch={})
        self.assertTrue(record["condition_capture"]["encode_call_count"] == 1)
        self.assertGreaterEqual(len(record["condition_capture"]["denoise_step_hashes"]), 2)
        self.assertEqual(record["condition_mask"].shape, (1, 2, 2, 1))
        self.assertEqual(adapter.model.generate_kwargs["guidance"], 1.0)
        self.assertEqual(adapter.model.generate_kwargs["shift"], 10.0)
        self.assertTrue(record["decoded_final_frame"])

    def test_real_adapter_records_pass_the_stage_a_gate(self):
        class PackedTensor:
            dtype = "torch.bfloat16"
            def __init__(self, value): self.value = value
            def __array__(self, dtype=None):
                value = np.asarray(self.value, dtype=np.float32)
                bits = value.view(np.uint32)
                rounded = ((bits + (((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF))) & np.uint32(0xFFFF0000)).view(np.float32)
                return np.asarray(rounded, dtype=dtype)
        class GenerationDataClean:
            __slots__ = ("x0_tokens_vision",)
            def __init__(self, tokens): self.x0_tokens_vision = [tokens]
        class SequencePlan:
            condition_frame_indexes_vision = [0]
        class Model:
            def __init__(self):
                self.text_kv_cache = {}
                self.tokenizer_vision_gen = SimpleNamespace(_keep_decoder_cache=False, _dec_cache={})
            def get_data_and_condition(self, batch, **kwargs):
                return GenerationDataClean(np.arange(4, dtype=np.float32).reshape(1, 2, 2, 1))
            def _prepare_inference_data(self, batch, seed):
                data = self.get_data_and_condition(batch)
                initial = [np.ones((1, 2, 2, 1), dtype=np.float32)]
                masks = [np.array([[[True]], [[False]]], dtype=bool)]
                reference = [data.x0_tokens_vision[0].copy()]
                return ([SequencePlan()], data, [], [], initial, reference, masks, False)
            def denoise(self, *args, **kwargs):
                self.text_kv_cache["request"] = 1
                return {"preds_vision": [np.zeros((1, 2, 2, 1), dtype=np.float32)]}
            def decode_vision(self, latent):
                return np.full((1, 3, 2, 1, 1), 0.5, dtype=np.float32)
            def generate_samples_from_batch(self, batch, **kwargs):
                data = self._prepare_inference_data(batch, [0])[1]
                packed = SimpleNamespace(vision=SimpleNamespace(tokens=[PackedTensor(data.x0_tokens_vision[0])], condition_mask=[np.array([[[True]], [[False]]], dtype=bool)], timesteps=np.array([0.5], dtype=np.float32)))
                for step in range(30):
                    packed.vision.timesteps = np.array([30.0 - step], dtype=np.float32)
                    self.denoise(packed)
                self.decode_vision(np.ones((1, 2, 2, 1), dtype=np.float32))
                return {"vision": [np.ones((1, 2, 2, 1), dtype=np.float32)]}
        class Pipeline:
            def __init__(self): self.model = Model()
        adapter = OfficialPostVaeRuntimeAdapter.from_pipeline(Pipeline())
        records = {}
        for spec in build_call_plan()[:4]:
            direction = np.ones((1, 2, 2, 1), dtype=np.float32) if spec["sample_id"] == "C" else None
            records[spec["sample_id"]] = adapter.execute_call(spec, data_batch={}, direction=direction)
        gate = evaluate_stage_a_gate(records)
        self.assertTrue(gate.passed, gate.failures)
        self.assertEqual(records["A"]["packed_condition_reference"].dtype, np.dtype(np.float32))
        self.assertNotEqual(records["A"]["packed_condition_reference"].dtype, np.dtype(bool))
        self.assertEqual(records["A"]["expected_network_condition_hash"], sha256_array(np.array([0.0, 1.0], dtype=np.float32)))
        self.assertEqual(len(set(records["C"]["condition_step_hashes"])), 1)
        self.assertEqual(records["C"]["condition_step_hashes"][0], records["C"]["expected_network_condition_hash"])
        self.assertIn("target_delta_fp32", records["C"])
        self.assertEqual(records["C"]["target_alpha"], 0.01)
        self.assertGreater(records["C"]["target_rms"], 0.0)

    def test_flat_initial_state_and_mask_map_exactly_to_five_dimensional_carrier(self):
        records = GateTests()._records()
        shape = (1, 2, 3, 2, 2)
        carrier = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        mask = np.zeros(shape, dtype=bool); mask[:, :, 0] = True
        records["C"].update({"carrier_fp32": carrier, "condition_mask": mask, "initial_state": carrier.reshape(1, -1), "initial_condition_mask": mask.reshape(1, -1), "actual_delta_fp32": np.where(mask, 1.0, 0.0).astype(np.float32), "actual_delta_bf16": np.where(mask, 1.0, 0.0).astype(np.float32), "final_latent_full": carrier.copy(), "predicted_latent": carrier[:, :, 1:, :, :].copy(), "latent_slicing": {"predicted_indexes": [1, 2], "axis": 2, "source_shape": list(carrier.shape), "selected_shape": list(carrier[:, :, 1:, :, :].shape)}})
        self.assertTrue(evaluate_stage_a_gate(records).passed, evaluate_stage_a_gate(records).failures)
        records["C"]["initial_condition_mask"] = np.roll(records["C"]["initial_condition_mask"], 1, axis=1)
        self.assertFalse(evaluate_stage_a_gate(records).passed)

    def test_decoded_final_is_exact_channel_frame_slice_for_rgb_t5(self):
        records = GateTests()._records()
        self.assertTrue(evaluate_stage_a_gate(records).passed)
        records["C"]["decoded_final"] = records["C"]["decoded_float"][:, 3]
        self.assertFalse(evaluate_stage_a_gate(records).passed)

    def test_prime_baseline_uses_fresh_encode_and_rejects_saved_mismatch(self):
        class GenerationDataClean:
            __slots__ = ("x0_tokens_vision",)
            def __init__(self, tokens): self.x0_tokens_vision = [tokens]
        class Model:
            def __init__(self): self.encode_count = 0
            def get_data_and_condition(self, batch, **kwargs):
                self.encode_count += 1
                return GenerationDataClean(np.ones((1, 2, 2, 1), dtype=np.float32))
            def _prepare_inference_data(self, batch, seed):
                self.get_data_and_condition(batch)
                raise AssertionError("resume prime must not re-encode through preparation")
        adapter = OfficialPostVaeRuntimeAdapter.from_pipeline(SimpleNamespace(model=Model()))
        saved_mask = np.zeros((1, 2, 2, 1), dtype=bool); saved_mask[:, 0] = True
        restored = adapter.prime_baseline({}, condition_indexes=[0], expected_carrier=np.ones((1, 2, 2, 1), dtype=np.float32), expected_mask=saved_mask, expected_direction_bank=None)
        self.assertEqual(restored.x0_tokens_vision[0].shape, (1, 2, 2, 1))
        self.assertEqual(adapter.model.encode_count, 1)
        with self.assertRaisesRegex(ValueError, "saved z0"):
            adapter.prime_baseline({}, condition_indexes=[0], expected_carrier=np.zeros((1, 2, 2, 1), dtype=np.float32), expected_mask=saved_mask, expected_direction_bank=None)

    def test_adapter_preserves_partial_capture_on_runtime_exception(self):
        class GenerationDataClean:
            __slots__ = ("x0_tokens_vision",)
            def __init__(self, tokens): self.x0_tokens_vision = [tokens]
        class SequencePlan:
            condition_frame_indexes_vision = [0]
        class Model:
            def get_data_and_condition(self, batch, **kwargs):
                return GenerationDataClean(np.ones((1, 2, 2, 1), dtype=np.float32))
            def _prepare_inference_data(self, batch, seed):
                return ([SequencePlan()], self.get_data_and_condition(batch), [], [], [np.ones((1, 2, 2, 1), dtype=np.float32)], [np.arange(4, dtype=np.float32)], [np.array([[[True]], [[False]]], dtype=bool)], False)
            def generate_samples_from_batch(self, batch, **kwargs):
                self._prepare_inference_data(batch, [0])
                raise RuntimeError("post-prepare failure")
        adapter = OfficialPostVaeRuntimeAdapter.from_pipeline(SimpleNamespace(model=Model()))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "post-prepare failure") as raised:
                adapter.execute_call({"sample_id": "A", "injection": "normal"}, data_batch={}, sample_dir=tmp)
            self.assertIn("carrier_fp32", raised.exception.capture)
            for name in ("carrier_fp32.npy", "initial_state.npy", "condition_mask.npy"):
                self.assertTrue((Path(tmp) / name).is_file(), name)
            self.assertTrue(callable(adapter.model.generate_samples_from_batch))

    def test_adapter_fails_if_decoder_cache_remains_after_decode(self):
        class Model:
            def get_data_and_condition(self, batch, **kwargs): return SimpleNamespace(x0_tokens_vision=[np.ones((1, 2, 2, 1), dtype=np.float32)])
        class Pipeline:
            def __init__(self): self.model = Model()
        adapter = OfficialPostVaeRuntimeAdapter.from_pipeline(Pipeline())
        adapter.model.tokenizer_vision_gen = SimpleNamespace(_keep_decoder_cache=False, _dec_cache={"leak": 1})
        with self.assertRaisesRegex(ValueError, "decoder cache"):
            adapter._assert_cache_lifecycle()

    def test_adapter_evidence_uses_explicit_top_level_schema_and_expected_bf16_hash(self):
        records = GateTests()._records()
        records["C"]["expected_network_condition_hash"] = records["C"]["condition_step_hashes"][0]
        self.assertTrue(evaluate_stage_a_gate(records).passed)
        records["C"]["first_denoise_hash"] = sha256_array(np.ones((2,), dtype=np.float32))
        self.assertFalse(evaluate_stage_a_gate(records).passed)

    def test_cli_parses_without_cosmos_or_torch(self):
        args = parse_args(["--run-dir", "run", "--stage-a-only"])
        self.assertTrue(args.stage_a_only)
        self.assertEqual(args.model_seed, 0)

    def test_local_vae_overrides_clear_remote_bucket_and_credentials(self):
        overrides = local_vae_experiment_overrides("/tmp/Wan2.2_VAE.pth")
        self.assertIn('model.config.tokenizer.bucket_name=""', overrides)
        self.assertIn('model.config.tokenizer.object_store_credential_path_pretrained=""', overrides)
        self.assertTrue(overrides[0].startswith("model.config.tokenizer.vae_path="))

    def test_run_experiment_loads_runtime_and_consumes_cli_paths_without_execute_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "framework").mkdir(); (root / "checkpoint").mkdir(); (root / "checkpoint" / "model.bin").write_bytes(b"model")
            (root / "vae.pth").write_bytes(b"vae"); (root / "input.png").write_bytes(b"input"); (root / "action.json").write_text("{}", encoding="utf-8")
            args = parse_args(["--run-dir", str(root / "run"), "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae.pth"), "--input-path", str(root / "input.png"), "--action-path", str(root / "action.json"), "--prompt", "move"])
            calls = []
            def execute(spec):
                calls.append(spec["sample_id"])
                return {"status": "success", "sample_id": spec["sample_id"]}
            result = run_experiment(args, runtime_factory=lambda parsed, run_dir: execute)
            self.assertEqual(result["status"], "FAIL")  # fake records do not satisfy the hard gate
            self.assertEqual(calls[:4], ["A", "B", "C", "D"])
            self.assertTrue((root / "run" / "config.json").is_file())

    def test_stage_a_only_then_resume_restores_records_and_runs_all_fifty_scan_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "framework").mkdir(); (root / "checkpoint").mkdir(); (root / "checkpoint" / "model.bin").write_bytes(b"model")
            (root / "vae.pth").write_bytes(b"vae"); (root / "input.png").write_bytes(b"input"); (root / "action.json").write_text("{}", encoding="utf-8")
            common = ["--run-dir", str(root / "run"), "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae.pth"), "--input-path", str(root / "input.png"), "--action-path", str(root / "action.json")]
            args_a = parse_args(common + ["--stage-a-only"])
            args_full = parse_args(common + ["--resume"])
            template = GateTests()._records()
            seen = []
            def execute(spec):
                seen.append(spec["sample_id"])
                record = dict(template.get(spec["sample_id"], template["A"]))
                record["status"] = "success"
                return record
            first = run_experiment(args_a, runtime_factory=lambda parsed, run_dir: execute)
            self.assertEqual(first["status"], "STAGE_A_ONLY")
            seen.clear()
            second = run_experiment(args_full, runtime_factory=lambda parsed, run_dir: execute)
            self.assertEqual(second["status"], "COMPLETE")
            self.assertEqual(len(seen), 50)

    def test_corrupt_scan_evidence_fails_integrated_fifty_call_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "framework").mkdir(); (root / "checkpoint").mkdir(); (root / "checkpoint" / "model.bin").write_bytes(b"model")
            (root / "vae.pth").write_bytes(b"vae"); (root / "input.png").write_bytes(b"input"); (root / "action.json").write_text("{}", encoding="utf-8")
            args = parse_args(["--run-dir", str(root / "run"), "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae.pth"), "--input-path", str(root / "input.png"), "--action-path", str(root / "action.json")])
            template = GateTests()._records()
            def execute(spec):
                record = dict(template[spec["sample_id"]] if spec["sample_id"] in template else template["A"])
                record["status"] = "success"
                if spec["sample_id"] == "scan_pre":
                    record["condition_step_hashes"] = ["corrupt"] * 30
                return record
            result = run_experiment(args, runtime_factory=lambda parsed, path: execute)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(json.loads((root / "run" / "status.json").read_text(encoding="utf-8"))["status"], "FAIL")
            self.assertEqual(json.loads((root / "run" / "samples" / "scan_pre" / "status.json").read_text(encoding="utf-8"))["status"], "fail")

    def test_integrated_run_rejects_supplied_conditioned_predicted_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "framework").mkdir(); (root / "checkpoint").mkdir(); (root / "checkpoint" / "model.bin").write_bytes(b"model")
            (root / "vae.pth").write_bytes(b"vae"); (root / "input.png").write_bytes(b"input"); (root / "action.json").write_text("{}", encoding="utf-8")
            args = parse_args(["--run-dir", str(root / "run"), "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae.pth"), "--input-path", str(root / "input.png"), "--action-path", str(root / "action.json")])
            template = GateTests()._records()
            def execute(spec):
                record = dict(template[spec["sample_id"]] if spec["sample_id"] in template else template["A"])
                record["status"] = "success"
                if spec["sample_id"] == "scan_pre":
                    record["latent_slicing"] = dict(record["latent_slicing"], predicted_indexes=[0])
                return record
            result = run_experiment(args, runtime_factory=lambda parsed, path: execute)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(json.loads((root / "run" / "samples" / "scan_pre" / "status.json").read_text(encoding="utf-8"))["status"], "fail")

    def test_failed_scan_record_prevents_complete_status(self):
        good = {"carrier_fp32": np.ones(2), "network_condition_bf16": np.ones(2), "predicted_latent": np.ones(2), "decoded_float": np.ones(2), "initial_noise_hash": "n", "condition_mask": np.array([True, False]), "actual_delta_fp32": np.array([1., 0.]), "actual_delta_bf16": np.array([1., 0.]), "condition_step_hashes": ["x", "x"], "predicted_latent_sliced": True, "decoded_final_frame": True}
        def execute(spec):
            if spec["sample_id"] == "scan_pre": return {"status": "fail"}
            return dict(good)
        result = run_stage_a_and_scan(execute)
        self.assertEqual(result["status"], "FAIL")

    def test_missing_denoise_or_final_output_fails_sample_evidence(self):
        good = GateTests()._records()["C"]
        self.assertTrue(validate_sample_evidence(good, stage_a=True).passed)
        missing = dict(good, denoise_step_hashes=[], condition_step_hashes=[])
        self.assertFalse(validate_sample_evidence(missing, stage_a=False).passed)
        missing = dict(good, decoded_final=np.array([np.nan], dtype=np.float32))
        self.assertFalse(validate_sample_evidence(missing, stage_a=False).passed)
        invalid_slice = dict(good, latent_slicing={"predicted_indexes": [999]})
        self.assertFalse(validate_sample_evidence(invalid_slice, stage_a=False).passed)

    def test_setup_failure_persists_diagnostic_status_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "framework").mkdir(); (root / "checkpoint").mkdir(); (root / "checkpoint" / "model").write_bytes(b"m")
            (root / "vae").write_bytes(b"v"); (root / "input").write_bytes(b"i"); (root / "action").write_text("{}", encoding="utf-8")
            run_dir = root / "run"
            args = parse_args(["--run-dir", str(run_dir), "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae"), "--input-path", str(root / "input"), "--action-path", str(root / "action")])
            with self.assertRaisesRegex(RuntimeError, "setup"):
                run_experiment(args, runtime_factory=lambda *_: (_ for _ in ()).throw(RuntimeError("setup failed")))
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"], "FAIL")
            self.assertTrue((run_dir / "MANIFEST.sha256").is_file())

    def test_rejected_reopen_preserves_existing_status_logs_and_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "framework").mkdir(); (root / "checkpoint").mkdir(); (root / "checkpoint" / "model").write_bytes(b"m")
            (root / "vae").write_bytes(b"v"); (root / "input").write_bytes(b"i"); (root / "action").write_text("{}", encoding="utf-8")
            run_dir = root / "run"; run_dir.mkdir()
            existing = {"status.json": b"{\"status\":\"COMPLETE\"}\n", "gpu_samples.csv": b"old,csv\n", "MANIFEST.sha256": b"old manifest\n"}
            for name, value in existing.items(): (run_dir / name).write_bytes(value)
            args = parse_args(["--run-dir", str(run_dir), "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae"), "--input-path", str(root / "input"), "--action-path", str(root / "action")])
            with self.assertRaises(FileExistsError): run_experiment(args, runtime_factory=lambda *_: None)
            self.assertEqual({name: (run_dir / name).read_bytes() for name in existing}, existing)
            resume_args = parse_args(["--run-dir", str(run_dir), "--resume", "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae"), "--input-path", str(root / "input"), "--action-path", str(root / "action")])
            with self.assertRaises((ValueError, FileNotFoundError)): run_experiment(resume_args, runtime_factory=lambda *_: None)
            self.assertEqual({name: (run_dir / name).read_bytes() for name in existing}, existing)

    def test_interrupt_after_stage_a_finalizes_compatibility_before_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "framework").mkdir(); (root / "checkpoint").mkdir(); (root / "checkpoint" / "model").write_bytes(b"m")
            (root / "vae").write_bytes(b"v"); (root / "input").write_bytes(b"i"); (root / "action").write_text("{}", encoding="utf-8")
            run_dir = root / "run"
            args = parse_args(["--run-dir", str(run_dir), "--framework-root", str(root / "framework"), "--checkpoint-path", str(root / "checkpoint"), "--vae-path", str(root / "vae"), "--input-path", str(root / "input"), "--action-path", str(root / "action")])
            template = GateTests()._records()
            def execute(spec):
                if spec["sample_id"] == "B":
                    raise KeyboardInterrupt("interrupt after A")
                return dict(template[spec["sample_id"] if spec["sample_id"] in template else "A"], status="success")
            with self.assertRaises(KeyboardInterrupt):
                run_experiment(args, runtime_factory=lambda parsed, path: execute)
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            self.assertNotEqual(config["z0_sha256"], "pending-stage-a")
            self.assertNotEqual(config["mask_sha256"], "pending-stage-a")
            self.assertTrue((run_dir / "direction_bank.npy").is_file())
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
