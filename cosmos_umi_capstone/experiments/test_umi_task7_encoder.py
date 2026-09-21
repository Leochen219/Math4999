import unittest
from unittest import mock

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the bundled CPU runtime may omit torch
    torch = None

if torch is not None:
    try:
        from umi_task7_encoder import (EncoderPrecisionError, FeedbackEncoder,
                                       temporary_encoder_precision)
    except ImportError:  # pragma: no cover - red phase before implementation
        FeedbackEncoder = None
        EncoderPrecisionError = RuntimeError
        temporary_encoder_precision = None


if torch is not None:
  @unittest.skipUnless(torch is not None, "real Torch CPU tests require the optional torch dependency")
  class Task7EncoderTests(unittest.TestCase):
    class Inner(torch.nn.Module):
        def __init__(self, *, force_bf16=False, fail=False, fail_value=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))
            self.register_buffer("floating_buffer", torch.tensor(3.0, dtype=torch.bfloat16))
            self.register_buffer("integer_buffer", torch.tensor(7, dtype=torch.int64))
            self.force_bf16 = force_bf16
            self.fail = fail
            self.fail_value = fail_value
            self.calls = 0

        def encode(self, value, scale):
            self.calls += 1
            if self.fail:
                self.cache = {"active": 1}
                raise RuntimeError("encoder exploded")
            if self.fail_value:
                raise ValueError("encoder value failure")
            if self.force_bf16:
                value = value.to(torch.bfloat16)
            self.seen_scale_dtype = scale[0].dtype
            return value * self.weight * self.floating_buffer * scale[1]

    class VAE:
        def __init__(self, inner):
            super().__init__()
            self.model = inner
            self.dtype = torch.bfloat16
            self.mean = torch.tensor(0.5, dtype=torch.bfloat16)
            self.std = torch.tensor(2.0, dtype=torch.bfloat16)
            self.scale = (self.mean, torch.tensor(0.5, dtype=torch.bfloat16))
            self.integer_constant = torch.tensor(9, dtype=torch.int64)
            self.cache = ["stale", None, None]
            self.restore_failure = False

        @property
        def training(self):
            return self.model.training

        def eval(self):
            self.model.eval()
            return self

        def train(self, mode=True):
            self.model.train(mode)
            return self

        def clear_decoder_cache(self):
            self.cache[:] = [None, None, None]
            if hasattr(self.model, "cache"):
                self.model.cache.clear()

        def encode(self, value):
            original_dtype = value.dtype
            value = value.to(self.dtype)
            latent = self.model.encode(value, self.scale)
            return latent.to(original_dtype)

    class Tokenizer:
        def __init__(self, inner):
            self.model = Task7EncoderTests.VAE(inner)
            self.dtype = torch.bfloat16
            self.cache = {"stale": 1}
            self.training = True

        def reset_cache(self):
            self.cache.clear()
            self.model.clear_decoder_cache()

        def encode(self, value):
            return self.model.encode(value)

        def actual_identity(self):
            return {"weights": [self.model.model.weight.detach().float().item()]}

        def eval(self):
            self.model.eval()
            return self

        def train(self, mode=True):
            self.model.train(mode)
            self.training = bool(mode)
            return self

    class PlainWan:
        """Official-shaped plain wrapper beneath an outer interface."""

        def __init__(self, inner):
            self.model = inner
            self.dtype = torch.bfloat16
            self.mean = torch.tensor(0.5, dtype=torch.bfloat16)
            self.std = torch.tensor(2.0, dtype=torch.bfloat16)
            self.scale = (self.mean, torch.tensor(0.5, dtype=torch.bfloat16))

        def encode(self, value):
            value = value.to(self.dtype)
            return self.model.encode(value, self.scale)

    class ReadonlyDtypeWrapper:
        """Outer interface whose dtype is derived from the plain Wan model."""

        def __init__(self, plain_wan):
            self.model = plain_wan

        @property
        def dtype(self):
            return self.model.dtype

        def encode(self, value):
            return self.model.encode(value)

        def clear_decoder_cache(self):
            pass

    def setUp(self):
        self.frame = np.zeros((3, 8, 8), dtype=np.float32)
        self.frame[1] = 0.5
        self.frame[2] = 1.0

    def _encoder(self, **kwargs):
        return self.Tokenizer(self.Inner(**kwargs))

    def _readonly_dtype_encoder(self, **kwargs):
        return self.ReadonlyDtypeWrapper(self.PlainWan(self.Inner(**kwargs)))

    def test_actual_style_cache_clear_preserves_none_slot_structure_on_success_and_failure(self):
        tokenizer = self._encoder()
        vae = tokenizer.model
        vae.cache = ["stale", None, None]
        result = FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="native")
        self.assertTrue(result["evidence"]["cache_cleared_after"])
        self.assertEqual(vae.cache, [None, None, None])

        failing = self._encoder(fail=True)
        failing.model.cache = ["stale", None, None]
        with self.assertRaisesRegex(RuntimeError, "encoder exploded"):
            FeedbackEncoder(failing, device="cpu").encode(self.frame, precision="native")
        self.assertEqual(failing.model.cache, [None, None, None])

    def test_encoder_value_error_is_not_retried(self):
        tokenizer = self._encoder(fail_value=True)
        with self.assertRaisesRegex(ValueError, "encoder value failure"):
            FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="native")
        self.assertEqual(tokenizer.model.model.calls, 1)

    def test_fp32_restore_failure_still_cleans_hook_and_cache_and_does_not_publish(self):
        tokenizer = self._encoder()
        inner = tokenizer.model.model
        original_encode = inner.encode.__func__
        original_restore = __import__("umi_task7_encoder")._PrecisionSnapshot.restore
        old_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        def restore_then_fail(snapshot):
            original_restore(snapshot)
            raise RuntimeError("restore failed")

        with mock.patch("umi_task7_encoder._PrecisionSnapshot.restore", restore_then_fail):
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old_tf32
        self.assertIs(inner.encode.__func__, original_encode)
        self.assertEqual(tokenizer.cache, {})
        self.assertEqual(tokenizer.model.cache, [None, None, None])

    def test_fp32_fails_closed_when_dispatch_observer_is_unavailable(self):
        tokenizer = self._encoder()
        real_import = __import__("builtins").__import__

        def missing_observer(name, *args, **kwargs):
            if name == "torch.utils._python_dispatch":
                raise ImportError("observer unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=missing_observer):
            with self.assertRaisesRegex(EncoderPrecisionError, "dispatch observer"):
                FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")

    def test_fp32_evidence_records_per_call_state_dtypes(self):
        tokenizer = self._encoder()
        result = FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")
        state = result["evidence"]["state_dtypes"]
        self.assertTrue(state["parameters"])
        self.assertTrue(state["buffers"])
        self.assertTrue(state["constants"])
        self.assertTrue(all(dtype == "float32" for dtype in state["parameters"].values()))
        self.assertTrue(all(dtype == "float32" for dtype in state["buffers"].values()))
        self.assertTrue(all(dtype == "float32" for dtype in state["constants"].values()))

    def test_silent_precision_restore_noop_is_detected_before_publication(self):
        tokenizer = self._encoder()
        inner = tokenizer.model.model
        with mock.patch("umi_task7_encoder._PrecisionSnapshot.restore", lambda snapshot: None):
            with self.assertRaisesRegex(EncoderPrecisionError, "round-trip"):
                FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")
        self.assertEqual(tokenizer.cache, {})

    def test_native_path_preserves_native_inner_dtype_and_normalizes_rgb(self):
        tokenizer = self._encoder()
        result = FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="native")

        self.assertEqual(result["evidence"]["inner_input_dtype"], "bfloat16")
        self.assertEqual(result["evidence"]["inner_output_dtype"], "bfloat16")
        self.assertEqual(tuple(result["arrays"]["encoder_input"].shape), (1, 3, 1, 8, 8))
        self.assertEqual(float(result["arrays"]["encoder_input"][0, 0, 0, 0, 0]), -1.0)
        self.assertEqual(float(result["arrays"]["encoder_input"][0, 1, 0, 0, 0]), 0.0)
        self.assertEqual(float(result["arrays"]["encoder_input"][0, 2, 0, 0, 0]), 1.0)
        self.assertEqual(result["arrays"]["output"].dtype, np.float32)
        self.assertTrue(np.isfinite(result["arrays"]["output"]).all())

    def test_temporary_fp32_converts_compute_and_constants_then_restores_exact_state(self):
        tokenizer = self._encoder()
        tokenizer.train()
        inner = tokenizer.model.model
        original_encode = inner.encode.__func__
        original_weight = inner.weight.detach().clone()

        result = FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")

        self.assertEqual(result["evidence"]["inner_input_dtype"], "float32")
        self.assertEqual(result["evidence"]["inner_output_dtype"], "float32")
        self.assertIn("float32", result["evidence"]["operation_dtypes"])
        self.assertNotIn("bfloat16", result["evidence"]["operation_dtypes"])
        self.assertEqual(inner.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(inner.weight, original_weight))
        self.assertEqual(inner.floating_buffer.dtype, torch.bfloat16)
        self.assertEqual(inner.integer_buffer.dtype, torch.int64)
        self.assertEqual(tokenizer.model.mean.dtype, torch.bfloat16)
        self.assertEqual(tokenizer.model.scale[0].dtype, torch.bfloat16)
        self.assertEqual(tokenizer.model.integer_constant.dtype, torch.int64)
        self.assertTrue(tokenizer.training)
        self.assertIs(inner.encode.__func__, original_encode)
        self.assertEqual(tokenizer.cache, {})
        self.assertEqual(tokenizer.model.cache, [None, None, None])

    def test_temporary_fp32_rejects_hidden_bfloat16_even_if_public_output_is_float32(self):
        tokenizer = self._encoder(force_bf16=True)
        with self.assertRaises(EncoderPrecisionError):
            FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")
        self.assertEqual(tokenizer.model.model.weight.dtype, torch.bfloat16)
        self.assertEqual(tokenizer.model.cache, [None, None, None])

    def test_validation_rejects_integer_or_invalid_rgb_frames(self):
        api = FeedbackEncoder(self._encoder(), device="cpu")
        with self.assertRaises(ValueError):
            api.encode(np.zeros((3, 8, 8), dtype=np.uint8))
        with self.assertRaises(ValueError):
            api.encode(np.zeros((2, 8, 8), dtype=np.float32))
        with self.assertRaises(ValueError):
            api.encode(np.full((3, 8, 8), 1.1, dtype=np.float32))
        with self.assertRaises(ValueError):
            api.encode(np.full((3, 8, 8), np.nan, dtype=np.float32))

    def test_failure_restores_precision_train_state_and_cache(self):
        tokenizer = self._encoder(fail=True)
        tokenizer.train()
        inner = tokenizer.model.model
        original_encode = inner.encode.__func__
        with self.assertRaisesRegex(RuntimeError, "encoder exploded"):
            FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")
        self.assertIs(inner.encode.__func__, original_encode)
        self.assertEqual(inner.weight.dtype, torch.bfloat16)
        self.assertEqual(tokenizer.model.mean.dtype, torch.bfloat16)
        self.assertTrue(tokenizer.training)
        self.assertEqual(tokenizer.cache, {})
        self.assertEqual(tokenizer.model.cache, [None, None, None])

    def test_context_manager_restores_state_and_two_calls_do_not_retain_payloads(self):
        tokenizer = self._encoder()
        inner = tokenizer.model.model
        original_weight = inner.weight.detach().clone()
        with temporary_encoder_precision(tokenizer, precision="temporary_fp32"):
            self.assertEqual(inner.weight.dtype, torch.float32)
            self.assertEqual(tokenizer.model.scale[0].dtype, torch.float32)
        self.assertEqual(inner.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(inner.weight, original_weight))

        api = FeedbackEncoder(tokenizer, device="cpu")
        first = api.encode(self.frame, precision="temporary_fp32")
        second_frame = np.minimum(self.frame + 0.1, 1.0)
        second = api.encode(second_frame, precision="temporary_fp32")
        self.assertIsNot(first["arrays"]["output"], second["arrays"]["output"])
        self.assertIs(inner.encode.__func__, inner.__class__.encode)
        self.assertNotIn("arrays", api.__dict__)

    def test_readonly_derived_dtype_is_converted_and_restored(self):
        wrapper = self._readonly_dtype_encoder()
        inner = wrapper.model.model
        original_weight = inner.weight
        self.assertEqual(wrapper.dtype, torch.bfloat16)

        with temporary_encoder_precision(wrapper, precision="temporary_fp32"):
            self.assertEqual(wrapper.dtype, torch.float32)
            self.assertEqual(wrapper.model.dtype, torch.float32)
            self.assertEqual(inner.weight.dtype, torch.float32)

        self.assertEqual(wrapper.dtype, torch.bfloat16)
        self.assertEqual(wrapper.model.dtype, torch.bfloat16)
        self.assertIs(inner.weight, original_weight)
        self.assertEqual(inner.weight.dtype, torch.bfloat16)

    def test_readonly_derived_dtype_restores_after_encoder_failure(self):
        wrapper = self._readonly_dtype_encoder()
        inner = wrapper.model.model
        original_weight = inner.weight

        with self.assertRaisesRegex(RuntimeError, "body failure"):
            with temporary_encoder_precision(wrapper, precision="temporary_fp32"):
                self.assertEqual(wrapper.dtype, torch.float32)
                raise RuntimeError("body failure")

        self.assertEqual(wrapper.dtype, torch.bfloat16)
        self.assertEqual(wrapper.model.dtype, torch.bfloat16)
        self.assertIs(inner.weight, original_weight)
        self.assertEqual(inner.weight.dtype, torch.bfloat16)

else:
    class Task7EncoderTests(unittest.TestCase):
        @unittest.skip("real Torch CPU tests require the optional torch dependency")
        def test_torch_is_required_for_integration_fixtures(self):
            pass


if __name__ == "__main__":
    unittest.main()
