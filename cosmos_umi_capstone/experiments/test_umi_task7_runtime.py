"""Focused CPU contract tests for the Task 7 dynamic feedback seam."""
from __future__ import annotations

import tempfile
import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path

import numpy as np

from test_umi_precision_official import Model, Ops, Scheduler, Native


class _Task7Ops(Ops):
    """The approved fakeOps fixture plus explicit dtype telemetry for Task 2b."""

    @contextmanager
    def compute(self, dtype, evidence, execution, step):
        with super().compute(dtype, evidence, execution, step):
            execution.setdefault("operation_dtypes", {})["float32"] = execution.setdefault("operation_dtypes", {}).get("float32", 0) + 1
            yield


class _Encoder:
    def __init__(self, shape):
        self.calls = []
        self.shape = tuple(shape)

    def encode(self, frame, *, precision="native"):
        self.calls.append(np.array(frame, dtype=np.float32, copy=True))
        value = np.float32(np.asarray(frame, dtype=np.float32).mean() + 0.5)
        output = np.full(self.shape, value, dtype=np.float32)
        normalized = np.asarray(frame, dtype=np.float32)[None, :, None, :, :] * 2.0 - 1.0
        return {
            "arrays": {
                "input_rgb": np.array(frame, dtype=np.float32, copy=True),
                "encoder_input": normalized.copy(),
                "actual_encoder_input": normalized.copy(),
                "actual_output": output.copy(),
                "output": output.copy(),
            },
            "evidence": {
                "precision_path": precision,
                "actual_encoder_input_dtype": "float32",
                "scaled_latent_dtype": "float32",
                "actual_output_dtype": "float32",
                "operation_count": 3,
                "dispatch_observed": True,
            },
        }


def _fixture(seed=0):
    import umi_precision_official as official_api

    ops = _Task7Ops()
    model = Model(ops)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        checkpoint, decoder = root / "checkpoint", root / "decoder"
        checkpoint.write_bytes(b"fixture checkpoint")
        decoder.write_bytes(b"fixture decoder")
        direction = np.zeros((1, 1, 1, 3, 2, 2), dtype=np.float32)
        direction[:, :, :, 0] = 1.0
        runtime = official_api.OfficialPrecisionRuntime(
            model,
            {"prompt": "mouse", "action": [1, 2]},
            direction,
            provenance={"decode": "fixture-fixed", "seed": seed},
            ops=ops,
            scheduler_class=Scheduler,
            generation_settings={"num_steps": 30, "guidance": 1.0, "shift": 10.0},
            artifact_paths={"checkpoint": checkpoint, "decoder": decoder},
            model_seed=seed,
        )

    # The installed source uses this exact four-argument architecture-invariant
    # RNG.  The fixture's prepared outside-mask values make the seed-0 identity
    # assertion observable without inventing another random implementation.
    def arch_invariant_rand(shape, dtype, device, request_seed):
        values = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(tuple(shape))
        return Native(values + np.float32(request_seed), "float32")

    runtime.arch_invariant_rand = arch_invariant_rand
    # Mirrors the installed source's explicit FP32 RNG construction kwargs.
    model.tensor_kwargs_fp32 = {"dtype": "float32", "device": "cpu"}
    z0 = np.array(runtime.inputs.z0, dtype=np.float32, copy=True)
    mask = np.asarray(runtime.inputs.geometry.mask, dtype=bool)
    v0 = np.zeros_like(z0, dtype=np.float32)
    v0[mask] = 1.0
    condition_shape = (z0.shape[0], z0.shape[1], len(runtime.inputs.geometry.condition_indexes), *z0.shape[3:])
    return runtime, model, z0, mask, v0, condition_shape


class Task7FeedbackRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import umi_task7_runtime as api

        cls.api = api

    def test_extract_and_embed_preserve_authoritative_temporal_layout(self):
        runtime, _model, z0, mask, v0, condition_shape = _fixture()
        encoder = _Encoder(condition_shape)
        feedback = self.api.FeedbackRuntime(
            runtime,
            encoder=encoder,
            z0=z0,
            mask=mask,
            condition_indexes=runtime.inputs.geometry.condition_indexes,
            v0=v0,
            seed=0,
        )
        condition = feedback.extract_condition(z0)
        self.assertEqual(condition.shape, condition_shape)
        candidate = condition + np.float32(2.0)
        embedded = feedback.embed_condition(candidate)
        np.testing.assert_array_equal(feedback.extract_condition(embedded), candidate)
        np.testing.assert_array_equal(embedded[~mask], z0[~mask])
        with self.assertRaises(ValueError):
            feedback.step(np.zeros(z0.size, dtype=np.float32), 0)

    def test_step_runs_full_deferred_generation_then_one_decode_and_encode(self):
        runtime, model, z0, mask, v0, condition_shape = _fixture()
        decoder_inputs = []

        def decode_fp32(full_latent, *, precision):
            decoder_inputs.append(np.array(full_latent, dtype=np.float32, copy=True))
            return {
                "raw_output": np.zeros((1, 3, 3, 2, 2), dtype=np.float32),
                "operation_count": 4,
                "operation_dtypes": ["float32"],
                "invocations": 1,
                "precision": precision,
            }

        runtime.decode_fp32 = decode_fp32
        feedback = self.api.FeedbackRuntime(
            runtime,
            encoder=_Encoder(condition_shape),
            z0=z0,
            mask=mask,
            condition_indexes=runtime.inputs.geometry.condition_indexes,
            v0=v0,
            seed=0,
        )
        condition = feedback.extract_condition(z0) + np.float32(2.0)
        record = feedback.step(condition, 0)
        self.assertEqual(record["generation"]["decode_policy"], "deferred")
        self.assertEqual(record["dynamic_inputs_identity"]["binding"], "task7-request-local")
        from umi_fd_post_vae_bridge import sha256_array
        self.assertEqual(record["dynamic_inputs_identity"]["bound_full_sha256"], sha256_array(record["condition_input_full"]))
        self.assertEqual(record["generation"]["expected_steps"], 30)
        self.assertEqual(len(record["generation"]["steps"]), 30)
        self.assertEqual(record["evidence"]["operation_counts"], {"G": 1, "D": 1, "E": 1})
        self.assertEqual(record["denoiser_steps"], 30)
        self.assertEqual(model.decode_count, 0, "deferred generation must not decode")
        self.assertNotIn("_prepared_for_call", runtime.__dict__)
        self.assertEqual(len(decoder_inputs), 1)
        np.testing.assert_array_equal(decoder_inputs[0], record["full_latent"])
        self.assertEqual(tuple(record["full_latent"].shape), tuple(z0.shape))
        self.assertEqual(tuple(record["predicted_latent"].shape)[2], len(runtime.inputs.geometry.predicted_indexes))
        self.assertEqual(record["actual"]["condition_steps"].shape[0], 30)
        self.assertEqual(record["encoder"]["precision_path"], "temporary_fp32")
        self.assertIn("actual_encoder_input", record["encoder_arrays"])
        self.assertEqual(record["normalization"]["layout"], "CHW")
        self.assertEqual(record["next_consumption_check"], "deferred_to_next_step")

    def test_architecture_noise_is_prediction_only_seeded_and_stable(self):
        runtime, _model, z0, mask, v0, condition_shape = _fixture()
        runtime.decode_fp32 = lambda full_latent, *, precision: {
            "raw_output": np.zeros((1, 3, 3, 2, 2), dtype=np.float32),
            "operation_count": 1,
            "operation_dtypes": ["float32"],
            "invocations": 1,
        }
        feedback = self.api.FeedbackRuntime(
            runtime,
            encoder=_Encoder(condition_shape),
            z0=z0,
            mask=mask,
            condition_indexes=runtime.inputs.geometry.condition_indexes,
            v0=v0,
            seed=0,
        )
        condition = feedback.extract_condition(z0)
        first = feedback.step(condition, 0)
        repeat = feedback.step(condition, 0)
        other = feedback.for_seed(1).step(condition, 1)
        self.assertEqual(first["evidence"]["prediction_noise_hash"], repeat["evidence"]["prediction_noise_hash"])
        self.assertNotEqual(first["evidence"]["prediction_noise_hash"], other["evidence"]["prediction_noise_hash"])
        self.assertTrue(first["evidence"]["seed0_outside_mask_exact"])
        self.assertEqual(first["evidence"]["noise_hash_scope"], "prediction_region_only")
        self.assertEqual(first["generation"]["noise_evidence"]["seed"], 0)
        self.assertEqual(other["generation"]["noise_evidence"]["seed"], 1)

    def test_missing_official_noise_or_dtype_evidence_fails_closed(self):
        runtime, _model, z0, mask, v0, condition_shape = _fixture()
        del runtime.arch_invariant_rand
        feedback = self.api.FeedbackRuntime(
            runtime,
            encoder=_Encoder(condition_shape),
            z0=z0,
            mask=mask,
            condition_indexes=runtime.inputs.geometry.condition_indexes,
            v0=v0,
            seed=0,
        )
        fake_root = types.ModuleType("cosmos_framework")
        fake_utils = types.ModuleType("cosmos_framework.utils")
        fake_misc = types.ModuleType("cosmos_framework.utils.misc")
        fake_root.utils, fake_utils.misc = fake_utils, fake_misc
        with patch.dict(sys.modules, {"cosmos_framework": fake_root,
                                      "cosmos_framework.utils": fake_utils,
                                      "cosmos_framework.utils.misc": fake_misc}):
            with self.assertRaises(self.api.FeedbackRuntimeError):
                feedback.step(feedback.extract_condition(z0), 0)

    def test_missing_official_fp32_rng_kwargs_fails_closed(self):
        runtime, _model, z0, mask, v0, condition_shape = _fixture()
        runtime.model.tensor_kwargs_fp32 = {"device": "cpu"}
        feedback = self.api.FeedbackRuntime(
            runtime, encoder=_Encoder(condition_shape), z0=z0, mask=mask,
            condition_indexes=runtime.inputs.geometry.condition_indexes, v0=v0, seed=0)
        with self.assertRaises(self.api.FeedbackRuntimeError):
            feedback.step(feedback.extract_condition(z0), 0)

    def test_failure_restores_official_runtime_state_and_does_not_retry(self):
        runtime, model, z0, mask, v0, condition_shape = _fixture()
        calls = []

        def decode_fp32(full_latent, *, precision):
            calls.append(np.array(full_latent, dtype=np.float32, copy=True))
            raise RuntimeError("decoder fixture failure")

        runtime.decode_fp32 = decode_fp32
        original_net = model.net
        feedback = self.api.FeedbackRuntime(
            runtime,
            encoder=_Encoder(condition_shape),
            z0=z0,
            mask=mask,
            condition_indexes=runtime.inputs.geometry.condition_indexes,
            v0=v0,
            seed=0,
        )
        with self.assertRaisesRegex(RuntimeError, "decoder fixture failure"):
            feedback.step(feedback.extract_condition(z0), 0)
        self.assertIs(model.net, original_net)
        self.assertNotIn("_prepared_for_call", runtime.__dict__)
        self.assertEqual(model._sampler_state, {})
        self.assertEqual(model._text_kv_cache, {})
        self.assertEqual(len(calls), 1)


class Task7RealTorchDecoderTests(unittest.TestCase):
    """A tiny real-Torch CPU proof for conversion/observer/restoration."""

    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "real Torch fixture unavailable")
    def test_fp32_decoder_observes_compute_and_restores_bf16_state(self):
        import torch
        from types import SimpleNamespace as NS
        from umi_precision_official import TorchOps
        from umi_task7_runtime import _official_fp32_decode

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.bfloat16))
                self.register_buffer("scale", torch.tensor([1.0], dtype=torch.bfloat16))

            def decode(self, latent):
                return torch.tanh(latent[:, :3] * self.weight + self.scale)

        class Wan(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner
                self.dtype = torch.bfloat16
                self.mean = torch.zeros(3, dtype=torch.bfloat16)
                self.std = torch.ones(3, dtype=torch.bfloat16)

        class Tokenizer(torch.nn.Module):
            def __init__(self, wan):
                super().__init__()
                self.model = wan
                self._dec_cache = [None]

            def clear_cache(self):
                self._dec_cache = [None]

        class Model(torch.nn.Module):
            def __init__(self, tokenizer):
                super().__init__()
                self.tokenizer_vision_gen = tokenizer
                self.tensor_kwargs = {"dtype": torch.bfloat16}
                self.precision = torch.bfloat16

            def decode(self, latent):
                return self.tokenizer_vision_gen.model.model.decode(latent)

        inner = Inner()
        inner.train()
        tokenizer = Tokenizer(Wan(inner))
        model = Model(tokenizer)
        prepared = (None, NS(x0_tokens_vision=[torch.zeros((1, 48, 3, 2, 2), dtype=torch.float32)]))
        runtime = NS(model=model, ops=TorchOps(), prepared=prepared)
        before_dtype = inner.weight.dtype
        before_bytes = inner.weight.detach().view(torch.uint8).cpu().numpy().tobytes()
        result = _official_fp32_decode(runtime, np.zeros((1, 48, 3, 2, 2), dtype=np.float32))
        self.assertEqual(result["invocations"], 1)
        self.assertGreater(result["operation_count"], 0)
        self.assertEqual(set(result["operation_dtypes"]), {"float32"})
        self.assertEqual(result["inner_input_dtype"], "float32")
        self.assertEqual(inner.weight.dtype, before_dtype)
        self.assertEqual(inner.weight.detach().view(torch.uint8).cpu().numpy().tobytes(), before_bytes)
        self.assertTrue(inner.training)
        self.assertNotIn("decode", inner.__dict__)
        self.assertNotIn("decode", model.__dict__)


if __name__ == "__main__":
    unittest.main()
