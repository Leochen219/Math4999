import unittest

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
        def __init__(self, *, force_bf16=False, fail=False):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.bfloat16))
            self.register_buffer("floating_buffer", torch.tensor(3.0, dtype=torch.bfloat16))
            self.register_buffer("integer_buffer", torch.tensor(7, dtype=torch.int64))
            self.force_bf16 = force_bf16
            self.fail = fail

        def encode(self, value, scale):
            if self.fail:
                self.cache = {"active": 1}
                raise RuntimeError("encoder exploded")
            if self.force_bf16:
                value = value.to(torch.bfloat16)
            self.seen_scale_dtype = scale[0].dtype
            return value * self.weight * self.floating_buffer * scale[1]

    class VAE(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner
            self.dtype = torch.bfloat16
            self.mean = torch.tensor(0.5, dtype=torch.bfloat16)
            self.std = torch.tensor(2.0, dtype=torch.bfloat16)
            self.scale = (self.mean, torch.tensor(0.5, dtype=torch.bfloat16))
            self.integer_constant = torch.tensor(9, dtype=torch.int64)
            self.cache = {"stale": 1}

        def reset_cache(self):
            self.cache.clear()
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
            self.model.reset_cache()

        def encode(self, value):
            return self.model.encode(value)

        def actual_identity(self):
            return {"weights": [float(self.model.model.weight.float())]}

        def eval(self):
            self.model.eval()
            return self

        def train(self, mode=True):
            self.model.train(mode)
            self.training = bool(mode)
            return self

    def setUp(self):
        self.frame = np.zeros((3, 8, 8), dtype=np.float32)
        self.frame[1] = 0.5
        self.frame[2] = 1.0

    def _encoder(self, **kwargs):
        return self.Tokenizer(self.Inner(**kwargs))

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
        self.assertEqual(tokenizer.model.cache, {})

    def test_temporary_fp32_rejects_hidden_bfloat16_even_if_public_output_is_float32(self):
        tokenizer = self._encoder(force_bf16=True)
        with self.assertRaises(EncoderPrecisionError):
            FeedbackEncoder(tokenizer, device="cpu").encode(self.frame, precision="temporary_fp32")
        self.assertEqual(tokenizer.model.model.weight.dtype, torch.bfloat16)
        self.assertEqual(tokenizer.model.cache, {})

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
        self.assertEqual(tokenizer.model.cache, {})

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

else:
    class Task7EncoderTests(unittest.TestCase):
        @unittest.skip("real Torch CPU tests require the optional torch dependency")
        def test_torch_is_required_for_integration_fixtures(self):
            pass


if __name__ == "__main__":
    unittest.main()
