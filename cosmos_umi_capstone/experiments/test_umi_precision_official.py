"""Execute the production seam wrappers against official-shaped CPU fixtures."""
import copy
import importlib
import unittest
import tempfile
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace as NS

import numpy as np

from test_umi_precision_runtime import Native
from umi_precision_runtime import EvidenceError, projection, validate_capture


class Ops:
    def __init__(self):
        self.dtype = None
        self.record = None

    def cast(self, value, dtype):
        if not hasattr(value, "dtype"):
            return value
        values = projection(value)
        if dtype == "bfloat16":
            # Independent truncation is exact for the fixtures used here.
            bits = values.view(np.uint32)
            bits[:] = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
        if self.record is not None:
            self.record["operation_count"] += 1
            self.record["casts"].append({"from": str(value.dtype), "to": dtype})
            if self.dtype == "float32" and dtype != "float32":
                raise EvidenceError("hidden low precision dispatch")
        return Native(values, dtype)

    def from_array(self, values, like, dtype="float32"):
        return self.cast(Native(values, "float32"), dtype)

    def set_weights(self, net, dtype):
        net.weight = self.cast(net.weight, dtype)
        net.buffer = self.cast(net.buffer, dtype)
        net.buffer_low = self.cast(net.buffer_low, dtype)
        net.training = False

    def weights(self, net):
        return [("weight", net.weight), ("buffer", net.buffer), ("buffer_low", net.buffer_low)]

    def ensure_independent(self, original, clone):
        for name in original.state_dict():
            if np.shares_memory(original.state_dict()[name].values, clone.state_dict()[name].values):
                raise EvidenceError("aliased network copy")

    def dtype(self, value):
        return str(value.dtype)

    @contextmanager
    def inference(self):
        yield

    @contextmanager
    def compute(self, dtype, evidence, execution, step):
        self.dtype, self.record = dtype, execution
        execution.update({"autocast": False, "tf32_matmul": False, "tf32_cudnn": False,
                          "backend": "numpy_fixture", "dispatch_observed": True})
        execution["operation_count"] += 1
        try:
            yield
        finally:
            self.dtype, self.record = None, None

    def telemetry(self):
        return {"available": False, "reason": "CPU fixture"}


class Scheduler:
    def __init__(self):
        self.model_outputs = []

    def step(self, model_output, timestep, sample, return_dict=False, generator=None):
        converted = self.convert_model_output(model_output, sample=sample)
        self.model_outputs.append(converted)
        self.last_sample = sample
        return (Native(projection(sample) + projection(converted) * .01, "float32"),)

    def convert_model_output(self, model_output, *, sample):
        # Like real UniPC: a scalar times BF16 velocity may be BF16, then
        # subtraction from FP32 sample produces the FP32 converted state.
        product = self.ops.cast(Native(.5 * projection(model_output), "float32"), model_output.dtype)
        return Native(projection(sample) - projection(product), "float32")


class Sampler:
    def __init__(self, model):
        self.model = model
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def forward(self, velocity_fn, noise, *, num_steps, shift, seed, step_callback=None):
        scheduler = Scheduler()
        scheduler.ops = self.model.ops
        for i in range(num_steps):
            timestep = Native([999-i], "float32")
            velocity = velocity_fn(noise, timestep)
            noise = [scheduler.step(velocity[0], timestep, noise[0], generator=NS(initial_seed=lambda: seed[0]))[0]]
        return noise


class Net:
    def __init__(self, ops):
        self.ops = ops
        self.weight = Native([2], "bfloat16")
        self.buffer = Native([1.000123], "float32")
        self.buffer_low = Native([.25], "bfloat16")
        self.training = True
        self.hidden_cast = False

    def named_modules(self):
        return [("", self)]

    def state_dict(self):
        return {"weight": self.weight, "buffer": self.buffer, "buffer_low": self.buffer_low}

    def forward(self, packed_seq, memory=None, **kwargs):
        value = packed_seq.vision.tokens[0]
        if self.hidden_cast:
            value = self.ops.cast(value, "bfloat16")
            value = self.ops.cast(value, "float32")
        # Global mean deliberately propagates condition perturbations into prediction.
        result = self.ops.cast(Native(np.full_like(projection(value), projection(value).mean()), "float32"), value.dtype)
        return {"preds_vision": [result]}

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class Model:
    def __init__(self, ops):
        self.ops = ops
        self.net = Net(ops)
        self.tensor_kwargs = {"device": "cpu", "dtype": "bfloat16"}
        self.precision = "bfloat16"
        self._sampler_state = {"old": 1}
        self._text_kv_cache = {"old": 1}
        self._diffusion_cache_installed = False
        self.decode_count = 0
        self.encode_count = 0
        self.seen = []
        self.config = {"precision": "bfloat16", "mode": "umi"}
        self.tokenizer_vision_gen = NS(weight=Native([.125], "float32"))
        self.mutate_noise = False
        self.sampler = Sampler(self)
        self.sampler_seed_override = None

    def _prepare_inference_data(self, data_batch, seed, has_negative_prompt=False):
        self.encode_count += 1
        shape = (1, 1, 3, 2, 2)
        baseline = np.ones(shape, np.float32)
        mask = np.zeros(shape, np.float32)
        mask[:, :, 0] = 1
        initial = np.arange(12, dtype=np.float32).reshape(shape)
        initial[mask.astype(bool)] = 1
        return ([NS(condition_frame_indexes_vision=[0])], NS(x0_tokens_vision=[Native(baseline, "float32")]),
                [[1]], [[]], [Native(initial.reshape(-1), "float32")],
                [Native(baseline.reshape(-1), "float32")], [Native(mask.reshape(-1), "float32")], False)

    def denoise(self, net=None, data_batch_packed=None, memory=None, **kwargs):
        self._text_kv_cache["used"] = 1
        return (net or self.net)(packed_seq=data_batch_packed, memory=memory)

    def generate_samples_from_batch(self, data_batch, seed, num_steps, **kwargs):
        prepared = self._prepare_inference_data(data_batch, seed)
        self.seen.append((data_batch, prepared))
        self._sampler_state["called"] = 1
        initial = prepared[4][0]
        shape = projection(prepared[1].x0_tokens_vision[0]).shape
        mask = projection(prepared[6][0]).reshape(shape)
        if self.mutate_noise:
            initial.values[-1] += 1
        def velocity_fn(noise, timestep):
            return self._get_velocity(noise_x=noise, timestep=timestep, gen_data_clean=prepared[1], mask=mask)
        sampler_seed = seed if self.sampler_seed_override is None else [self.sampler_seed_override]
        initial = self.sampler(velocity_fn, [initial], num_steps=num_steps, shift=10., seed=sampler_seed)[0]
        return {"vision": [Native(projection(initial).reshape(shape), "float32")]}

    def _get_velocity(self, *, noise_x, timestep, gen_data_clean, mask):
        shape = projection(gen_data_clean.x0_tokens_vision[0]).shape
        pack = NS(vision=NS(tokens=[self.ops.cast(Native(projection(noise_x[0]).reshape(shape), "float32"), self.tensor_kwargs["dtype"])],
                            condition_mask=[Native(mask, "float32")], timesteps=timestep))
        output = self.denoise(data_batch_packed=pack, memory=NS(_text_kv_cache=[]))["preds_vision"][0]
        return [Native(projection(output).reshape(-1), output.dtype)]

    def decode(self, value):
        self.decode_count += 1
        return Native(np.zeros((1, 3, 5, 2, 2)), "float32")


class OfficialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.api = importlib.import_module("umi_precision_official")
        except ModuleNotFoundError:
            cls.api = None

    def setUp(self):
        self.assertIsNotNone(self.api, "official precision seam implementation is missing")

    def fixture(self):
        ops = Ops()
        model = Model(ops)
        direction = np.zeros((1, 1, 1, 3, 2, 2), np.float32)
        direction[:, :, :, 0] = 1
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        checkpoint, decoder = Path(temp.name) / "checkpoint", Path(temp.name) / "decoder"
        checkpoint.write_bytes(b"fixture checkpoint")
        decoder.write_bytes(b"fixture decoder")
        runtime = self.api.OfficialPrecisionRuntime(model, {"prompt": "mouse", "action": [1, 2]}, direction,
            provenance={"decode": "fixture-fixed", "seed": 0}, ops=ops, scheduler_class=Scheduler,
            generation_settings={"num_steps": 2, "guidance": 1., "shift": 10.},
            artifact_paths={"checkpoint": checkpoint, "decoder": decoder})
        return runtime, model

    def test_true_seams_capture_dtype_noise_reset_baseline_and_output(self):
        runtime, model = self.fixture()
        records = {}
        noise_hash = None
        for group in "ABC":
            spec = {"sample_id": group, "group": group, "alpha": .0001, "sign": 1}
            record = runtime.execute(spec, runtime.inputs, scope="full")
            noise_hash = validate_capture(record, spec, runtime.inputs, "full", paired_noise=noise_hash)
            records[group] = record
        self.assertEqual(model.encode_count, 1)
        self.assertEqual(model.decode_count, 3)
        self.assertEqual(model._sampler_state, {})
        self.assertEqual(model._text_kv_cache, {})
        self.assertIsNot(model.seen[0][0], model.seen[1][0])
        self.assertIsNot(model.seen[0][1][1], model.seen[1][1][1])
        np.testing.assert_array_equal(records["A"]["common_input_fp32"], records["B"]["common_input_fp32"])
        self.assertFalse(np.array_equal(records["B"]["common_input_fp32"], records["C"]["common_input_fp32"]))
        self.assertEqual(records["A"]["predicted_latent"].shape, (1, 1, 2, 2, 2))
        self.assertEqual(records["C"]["decoded_final"].shape, (3, 2, 2))
        self.assertTrue(all(row["native_dtype"] == "float32" for row in records["B"]["tensor_evidence"]))
        self.assertIn("bfloat16", {row["native_dtype"] for row in records["A"]["tensor_evidence"]})

    def test_step_zero_module_stops_before_scheduler_update_and_decode(self):
        runtime, model = self.fixture()
        spec = {"sample_id": "module", "group": "B", "alpha": 0., "sign": 0}
        record = runtime.execute(spec, runtime.inputs, scope="module")
        validate_capture(record, spec, runtime.inputs, "module")
        self.assertEqual(record["steps"], [{"step": 0, "timestep": 999.}])
        self.assertEqual(model.decode_count, 0)
        self.assertFalse(any(row["role"] == "sampler_update" for row in record["tensor_evidence"]))
        self.assertEqual(record["primary_quantity"], "step_0_predicted_denoiser_output")

    def test_hidden_cast_then_upcast_is_rejected_and_hooks_restored(self):
        runtime, model = self.fixture()
        original = model.denoise
        scheduler_original = Scheduler.step
        model.net.hidden_cast = True
        with self.assertRaisesRegex(EvidenceError, "hidden"):
            runtime.execute({"sample_id": "bad", "group": "B", "alpha": 0., "sign": 0}, runtime.inputs, scope="full")
        self.assertEqual(model.denoise, original)
        self.assertIs(Scheduler.step, scheduler_original)
        self.assertEqual(model._sampler_state, {})
        self.assertEqual(model._text_kv_cache, {})

    def test_backend_flags_are_checked_again_at_each_compute_operation(self):
        self.assertTrue(callable(getattr(self.api, "assert_backend_disabled", None)), "dynamic backend assertion missing")
        backend = NS(is_autocast_enabled=lambda device: False,
                     backends=NS(cuda=NS(matmul=NS(allow_tf32=False)), cudnn=NS(allow_tf32=False)))
        self.api.assert_backend_disabled(backend)
        backend.backends.cuda.matmul.allow_tf32 = True
        with self.assertRaisesRegex(EvidenceError, "TF32"):
            self.api.assert_backend_disabled(backend)

    def test_official_bf16_velocity_is_valid_for_a_sampler(self):
        runtime, model = self.fixture()
        spec = {"sample_id": "A", "group": "A", "alpha": 0., "sign": 0}
        record = runtime.execute(spec, runtime.inputs, scope="full")
        validate_capture(record, spec, runtime.inputs, "full")
        self.assertEqual({r["native_dtype"] for r in record["tensor_evidence"] if r["role"] == "sampler_velocity"}, {"bfloat16"})
        self.assertEqual({r["native_dtype"] for r in record["tensor_evidence"] if r["role"] == "sampler_converted"}, {"float32"})

    def test_original_network_values_dtypes_training_and_attributes_survive_all_exits(self):
        for group, scope, fail in (("A", "full", False), ("B", "full", False), ("B", "full", True), ("B", "module", False)):
            with self.subTest(group=group, scope=scope, fail=fail):
                runtime, model = self.fixture()
                original_net = model.net
                model.net.weight = Native([2.000123], "float32")
                model.net.hidden_cast = fail
                before = {key: (value.dtype, value.values.tobytes()) for key, value in model.net.state_dict().items()}
                attributes = set(vars(model.net))
                model_attributes = set(vars(model))
                tensor_kwargs = model.tensor_kwargs
                try:
                    runtime.execute({"sample_id": "call", "group": group, "alpha": 0., "sign": 0}, runtime.inputs, scope=scope)
                except EvidenceError:
                    if not fail:
                        raise
                self.assertIs(model.net, original_net)
                self.assertEqual(before, {key: (value.dtype, value.values.tobytes()) for key, value in model.net.state_dict().items()})
                self.assertTrue(model.net.training)
                self.assertEqual(set(vars(model.net)), attributes)
                self.assertEqual(set(vars(model)), model_attributes)
                self.assertIs(model.tensor_kwargs, tensor_kwargs)

    def test_mutation_after_prepare_before_first_velocity_is_detected_in_both_scopes(self):
        for scope in ("full", "module"):
            with self.subTest(scope=scope):
                runtime, model = self.fixture()
                model.mutate_noise = True
                with self.assertRaisesRegex(EvidenceError, "consumed initial"):
                    runtime.execute({"sample_id": "bad", "group": "B", "alpha": 0., "sign": 0}, runtime.inputs, scope=scope)

    def test_actual_provenance_changes_without_caller_metadata_changes(self):
        self.assertTrue(callable(getattr(self.api.OfficialPrecisionRuntime, "actual_identity", None)), "actual identity missing")
        for changed in ("prompt", "action", "image", "model", "decoder", "config", "noise", "checkpoint_file", "decoder_file"):
            with self.subTest(changed=changed):
                runtime, model = self.fixture()
                before = runtime.actual_identity()
                if changed in ("prompt", "action", "image"):
                    runtime.data_batch[changed] = "changed behind unchanged caller metadata"
                elif changed == "model":
                    model.net.weight.values[0] += .1
                elif changed == "decoder":
                    model.tokenizer_vision_gen.weight.values[0] += .1
                elif changed == "config":
                    model.config["mode"] = "changed"
                elif changed.endswith("_file"):
                    runtime.artifact_paths[changed.removesuffix("_file")].write_bytes(b"actual file changed")
                else:
                    runtime.prepared[4][0].values[-1] += .1
                self.assertNotEqual(before, runtime.actual_identity())

    def test_dynamic_registered_net_attribute_is_restored_like_torch_modules(self):
        class RegisteredModel(Model):
            def __getattr__(self, name):
                if name == "net":
                    return self._modules[name]
                raise AttributeError(name)
            def __setattr__(self, name, value):
                if name == "net" and "_modules" in vars(self):
                    self._modules[name] = value
                else:
                    object.__setattr__(self, name, value)
            def __delattr__(self, name):
                if name == "net":
                    del self._modules[name]
                else:
                    object.__delattr__(self, name)
        runtime, model = self.fixture()
        original = model.net
        del model.net
        model._modules = {"net": original}
        model.__class__ = RegisteredModel
        runtime.execute({"sample_id": "B", "group": "B", "alpha": 0., "sign": 0}, runtime.inputs, scope="module")
        self.assertIn("net", model._modules)
        self.assertIs(model.net, original)

    def test_decoder_identity_is_derived_when_caller_omits_metadata(self):
        runtime, model = self.fixture()
        rebuilt = self.api.OfficialPrecisionRuntime(model, runtime.data_batch, np.stack([runtime.inputs.direction]),
            provenance={}, ops=runtime.ops, scheduler_class=Scheduler, artifact_paths=runtime.artifact_paths,
            generation_settings={"num_steps": 2})
        self.assertIn("decode", rebuilt.provenance)
        record = rebuilt.execute({"sample_id": "B", "group": "B", "alpha": 0., "sign": 0}, rebuilt.inputs, scope="module")
        self.assertEqual(record["decode_id"], rebuilt.provenance["decode"])

    def test_module_observes_actual_sampler_seed_before_first_velocity(self):
        runtime, model = self.fixture()
        model.sampler_seed_override = 1
        with self.assertRaisesRegex(EvidenceError, "sampler seed"):
            runtime.execute({"sample_id": "B", "group": "B", "alpha": 0., "sign": 0}, runtime.inputs, scope="module")


if __name__ == "__main__":
    unittest.main()
