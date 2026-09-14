"""Small CPU-only checks of the real Torch dispatch observer (no model/GPU)."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from umi_precision_runtime import EvidenceError, TensorEvidence, validate_capture, fingerprint
from umi_precision_official import TorchOps, OfficialPrecisionRuntime


@unittest.skipIf(torch is None, "local runtime has no Torch; also run in existing remote CPU runtime")
class TorchObservationTests(unittest.TestCase):
    def context(self):
        self.ops = TorchOps()
        self.evidence = TensorEvidence()
        self.execution = {"casts": [], "operation_count": 0}
        return self.ops.compute("float32", self.evidence, self.execution, 0)

    def test_cast_down_and_up_is_detected_at_the_actual_operator(self):
        with self.assertRaisesRegex(EvidenceError, "non-FP32"):
            with self.context():
                value = torch.tensor([1.0001], dtype=torch.float32)
                value.to(torch.bfloat16).float()
        self.assertTrue(self.execution["dispatch_observed"])
        self.assertGreater(self.execution["operation_count"], 0)
        self.assertEqual(self.execution["casts"][-1]["to"], "bfloat16")

    def test_tf32_enabled_inside_network_is_rejected_and_restored(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        with self.assertRaisesRegex(EvidenceError, "TF32"):
            with self.context():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.ones(2, dtype=torch.float32) + 1
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, previous)

    def test_bf16_native_evidence_and_float32_projection_are_distinct(self):
        evidence = TensorEvidence()
        values = evidence.record("network_condition", torch.tensor([1.25], dtype=torch.bfloat16), step=0)
        self.assertEqual(evidence.rows[0]["native_dtype"], "bfloat16")
        self.assertEqual(str(values.dtype), "float32")
        self.assertEqual(values.tolist(), [1.25])

    def test_fp32_execution_observes_intermediate_activation(self):
        with self.context():
            value = torch.tensor([1.5, 2.5], dtype=torch.float32)
            result = value.square() + value
        self.assertEqual(result.tolist(), [3.75, 8.75])
        self.assertTrue(any(row["role"] == "activation" and row["native_dtype"] == "float32"
                            for row in self.evidence.rows))

    def test_native_torch_full_abc_preserves_original_module_and_accepts_bf16_velocity(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.000123], dtype=torch.float32))
                self.register_buffer("buffer", torch.tensor([.125], dtype=torch.float32))
                self.register_buffer("buffer_low", torch.tensor([.25], dtype=torch.bfloat16))
                self.child = torch.nn.Identity().eval()
            def forward(self, packed_seq, memory=None):
                tokens = packed_seq.vision.tokens[0]
                return {"preds_vision": [tokens * self.weight + self.buffer]}
        class Scheduler:
            def __init__(self):
                self.model_outputs = [None]
            def convert_model_output(self, model_output, *, sample):
                return sample - .5 * model_output
            def step(self, model_output, timestep, sample, return_dict=False, generator=None):
                converted = self.convert_model_output(model_output, sample=sample)
                self.model_outputs = [converted]
                self.last_sample = sample
                return (sample - .01 * converted,)
        class Sampler(torch.nn.Module):
            def forward(self, velocity_fn, noise, *, num_steps, shift, seed):
                scheduler = Scheduler()
                generator = torch.Generator(device="cpu").manual_seed(seed[0])
                for i in range(num_steps):
                    timestep = torch.tensor([999-i], dtype=torch.float32)
                    velocity = velocity_fn(noise, timestep)
                    noise = [scheduler.step(velocity[0], timestep, noise[0], generator=generator)[0]]
                return noise
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = Net()
                self.sampler = Sampler()
                self.tokenizer_vision_gen = Net()
                self.config = {"mode": "UMI"}
                self.precision = torch.bfloat16
                self.tensor_kwargs = {"device": "cpu", "dtype": torch.bfloat16}
            def _prepare_inference_data(self, data_batch, seed, has_negative_prompt=False):
                value = torch.ones(1, 1, 3, 2, 2)
                mask = torch.zeros_like(value)
                mask[:, :, 0] = 1
                noise = torch.arange(12, dtype=torch.float32).reshape_as(value)
                noise[mask.bool()] = value[mask.bool()]
                return ([NS(condition_frame_indexes_vision=[0])], NS(x0_tokens_vision=[value]), [[1]], [[]],
                        [noise.flatten()], [value.flatten()], [mask.flatten()], False)
            def _get_velocity(self, *, noise_x, timestep, gen_data_clean):
                tokens = noise_x[0].reshape_as(gen_data_clean.x0_tokens_vision[0]).to(**self.tensor_kwargs)
                mask = torch.zeros_like(tokens, dtype=torch.float32)
                mask[:, :, 0] = 1
                packed = NS(vision=NS(tokens=[tokens], condition_mask=[mask], timesteps=timestep))
                return [self.denoise(data_batch_packed=packed)["preds_vision"][0].flatten()]
            def denoise(self, data_batch_packed=None, net=None, memory=None):
                return (net or self.net)(packed_seq=data_batch_packed, memory=memory)
            def generate_samples_from_batch(self, data_batch, seed, num_steps, **kwargs):
                prepared = self._prepare_inference_data(data_batch, seed)
                noise = prepared[4]
                def velocity_fn(noise, timestep):
                    return self._get_velocity(noise_x=noise, timestep=timestep, gen_data_clean=prepared[1])
                noise = self.sampler(velocity_fn, noise, num_steps=num_steps, shift=10., seed=seed)
                return {"vision": [noise[0].reshape_as(prepared[1].x0_tokens_vision[0])]}
            def decode(self, latent):
                return torch.zeros(1, 3, 5, 2, 2)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint, decoder = Path(temp) / "checkpoint", Path(temp) / "decoder"
            checkpoint.write_bytes(b"fixture checkpoint")
            decoder.write_bytes(b"fixture decoder")
            model = Model()
            original = model.net
            original_state = {name: value.clone() for name, value in original.state_dict().items()}
            kwargs = model.tensor_kwargs
            direction = np.zeros((1, 1, 1, 3, 2, 2), np.float32)
            direction[:, :, :, 0] = 1
            runtime = OfficialPrecisionRuntime(model, {"prompt": "mouse", "action": [1, 2]}, direction,
                provenance={}, scheduler_class=Scheduler, artifact_paths={"checkpoint": checkpoint, "decoder": decoder},
                generation_settings={"num_steps": 2})
            before_backend = runtime.actual_identity()
            with patch.object(torch, "__version__", "different-installed-backend"):
                self.assertNotEqual(before_backend, runtime.actual_identity())
            for group in "ABC":
                spec = {"sample_id": group, "group": group, "alpha": 0., "sign": 0}
                record = runtime.execute(spec, runtime.inputs, scope="full")
                validate_capture(record, spec, runtime.inputs, "full")
                self.assertIs(model.net, original)
                self.assertIs(model.tensor_kwargs, kwargs)
                self.assertTrue(original.training)
                self.assertFalse(original.child.training)
                self.assertTrue(all(torch.equal(original.state_dict()[name], value) for name, value in original_state.items()))
                self.assertTrue(all(original.state_dict()[name].dtype == value.dtype for name, value in original_state.items()))
                expected = "bfloat16" if group == "A" else "float32"
                self.assertEqual({row["native_dtype"] for row in record["tensor_evidence"] if row["role"] == "sampler_velocity"}, {expected})

    def test_identity_hashes_scalar_bf16_and_preserves_float64_low_bits(self):
        scalar = fingerprint(torch.tensor(1.25, dtype=torch.bfloat16))
        self.assertEqual(len(scalar), 64)
        self.assertNotEqual(scalar, fingerprint(torch.tensor(1.5, dtype=torch.bfloat16)))
        self.assertNotEqual(fingerprint(torch.tensor([1.], dtype=torch.float64)),
                            fingerprint(torch.tensor([1. + 1e-10], dtype=torch.float64)))


if __name__ == "__main__":
    unittest.main()
