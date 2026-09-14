"""Actual Cosmos run09 seams for the independent precision experiment.

Verified interfaces: OmniMoTModel._prepare_inference_data (8-tuple),
_get_velocity -> denoise -> net.forward(packed_seq=...), and
FlowUniPCMultistepScheduler.step(model_output=..., sample=...)[0].
No old experiment file is patched. Framework monkeypatches are request-local
and restored in finally, including on a deliberate step-zero stop.
"""
from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager

import numpy as np

try:
    from .umi_precision_runtime import EvidenceError, PrecisionCompatibilityError, PrecisionInputs, TensorEvidence, projection
    from .umi_fd_post_vae_scan import _clone_runtime, _cache_has_values
    from .umi_precision_identity import fingerprint, file_identity, source_identity, module_tensors, module_metadata
except ImportError:
    from umi_precision_runtime import EvidenceError, PrecisionCompatibilityError, PrecisionInputs, TensorEvidence, projection
    from umi_fd_post_vae_scan import _clone_runtime, _cache_has_values
    from umi_precision_identity import fingerprint, file_identity, source_identity, module_tensors, module_metadata


def _tensors(value):
    if hasattr(value, "dtype"):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def assert_backend_disabled(torch):
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise EvidenceError("TF32 was enabled inside compute")
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise EvidenceError("autocast was enabled inside compute")


class TorchOps:
    """Lazy Torch implementation; native dtype is read before any projection."""
    def __init__(self):
        import torch
        self.torch = torch

    def cast(self, value, dtype):
        return value.to(dtype=getattr(self.torch, dtype)) if self.torch.is_tensor(value) and value.is_floating_point() else value

    def from_array(self, values, like, dtype="float32"):
        return self.torch.as_tensor(np.array(values, copy=True), device=like.device, dtype=getattr(self.torch, dtype))

    def set_weights(self, net, dtype):
        net.to(dtype=getattr(self.torch, dtype))
        net.eval()

    def weights(self, net):
        # All floating weights AND buffers are observed, but copying a full
        # checkpoint back to CPU on every call is unnecessary and unbounded.
        for name, value in list(net.named_parameters()) + list(net.named_buffers()):
            if value.is_floating_point():
                yield name, value.reshape(-1)[:16]

    def ensure_independent(self, original, clone):
        originals = module_tensors(original)
        copies = module_tensors(clone)
        if originals.keys() != copies.keys():
            raise EvidenceError("independent precision network has different state keys")
        original_storage = {value.untyped_storage().data_ptr() for value in originals.values() if value.numel()}
        for name, value in originals.items():
            other = copies[name]
            if other.numel() and other.untyped_storage().data_ptr() in original_storage:
                raise EvidenceError("precision network copy aliases original storage")
            if (value.dtype != other.dtype or value.shape != other.shape or
                    not self.torch.equal(value.detach().contiguous().reshape(-1).view(self.torch.uint8),
                                         other.detach().contiguous().reshape(-1).view(self.torch.uint8))):
                raise EvidenceError("precision network copy changed original values before conversion")
        if fingerprint(module_metadata(original)) != fingerprint(module_metadata(clone)):
            raise EvidenceError("precision network copy changed training or extra state before conversion")

    @contextmanager
    def inference(self):
        with self.torch.inference_mode():
            yield

    @contextmanager
    def compute(self, dtype, evidence, execution, step):
        torch = self.torch
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        class Observe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                assert_backend_disabled(torch)
                result = func(*args, **(kwargs or {}))
                execution["operation_count"] += 1
                tensors = [x for x in tree_flatten((args, kwargs, result))[0]
                           if torch.is_tensor(x) and x.is_floating_point()]
                dtypes = {str(x.dtype).removeprefix("torch.") for x in tensors}
                counts = execution.setdefault("operation_dtypes", {})
                for native in dtypes:
                    counts[native] = counts.get(native, 0) + 1
                if "_to_copy" in str(func):
                    before = next((str(x.dtype).removeprefix("torch.") for x in tree_flatten(args)[0]
                                   if torch.is_tensor(x) and x.is_floating_point()), None)
                    after = next((str(x.dtype).removeprefix("torch.") for x in tree_flatten(result)[0]
                                  if torch.is_tensor(x) and x.is_floating_point()), None)
                    if before and after:
                        cast = {"op": str(func), "from": before, "to": after, "step": step}
                        if cast not in execution["casts"]:
                            execution["casts"].append(cast)
                if dtype == "float32" and dtypes - {"float32"}:
                    raise PrecisionCompatibilityError(f"hidden non-FP32 operation {func}: {sorted(dtypes)}")
                if dtype != "float32" and dtypes - {"float32", "bfloat16"}:
                    raise PrecisionCompatibilityError(f"unexpected BF16-path operation dtype {func}: {sorted(dtypes)}")
                if not any(r["role"] == "activation" and r["step"] == step for r in evidence.rows):
                    values = [x for x in tree_flatten(result)[0] if torch.is_tensor(x) and x.is_floating_point() and x.numel()]
                    if values:
                        evidence.record("activation", values[0].reshape(-1)[:16], step=step, name=str(func))
                return result
        try:
            with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False), Observe():
                execution.update({"autocast": bool(torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu")),
                                  "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                                  "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                                  "backend": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                              "cudnn": torch.backends.cudnn.version(),
                                              "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
                                              "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                              "math_sdp": torch.backends.cuda.math_sdp_enabled()},
                                  "dispatch_observed": True})
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def telemetry(self):
        torch = self.torch
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA unavailable"}
        torch.cuda.synchronize()
        return {"available": True, "device": torch.cuda.current_device(),
                "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "device_name": torch.cuda.get_device_name()}


class _StepZeroDone(BaseException):
    """Unwind the official generator immediately after the first denoiser."""


class OfficialPrecisionRuntime:
    def __init__(self, model, data_batch, direction_bank, *, provenance, ops=None,
                 scheduler_class=None, generation_settings=None, artifact_paths=None, inputs_factory=None):
        self.model = model
        self.ops = ops or TorchOps()
        if scheduler_class is None:
            from cosmos_framework.model.generator.diffusion.samplers.fm_solvers_unipc import FlowUniPCMultistepScheduler
            scheduler_class = FlowUniPCMultistepScheduler
        self.scheduler_class = scheduler_class
        self.provenance = copy.deepcopy(provenance)
        from pathlib import Path
        if artifact_paths is None or set(artifact_paths) != {"checkpoint", "decoder"}:
            raise ValueError("actual checkpoint and decoder artifact paths are required")
        self.artifact_paths = {name: Path(path).resolve(strict=True) for name, path in artifact_paths.items()}
        self.data_batch = _clone_runtime(data_batch)
        self.settings = {"num_steps": 30, "guidance": 1., "shift": 10.}
        self.settings.update(generation_settings or {})
        if self.settings["guidance"] != 1. or self.settings["shift"] != 10.:
            raise ValueError("UMI paired condition requires guidance=1 and shift=10")
        native_dtype = str(model.tensor_kwargs["dtype"]).removeprefix("torch.")
        if native_dtype != "bfloat16":
            raise ValueError("Q must derive from the existing BF16 velocity/network boundary")
        self.provenance["quantizer_source"] = {"observed_velocity_dtype": native_dtype,
            "source": "model.tensor_kwargs used by _get_velocity; checked again at actual A network read"}
        self.provenance["generation_settings"] = self.settings.copy()
        self.provenance["fp32_implementation"] = "same denoise/generation functions for B and C; no group-specific FP32 path"
        self.provenance["fallback"] = "one complete denoiser at actual sampler step 0; no scheduler update or decode"
        self.original_precision = model.precision
        self.original_tensor_kwargs = dict(model.tensor_kwargs)
        self._reset()

        self._check_caches()
        with self.ops.inference():
            prepared = model._prepare_inference_data(_clone_runtime(self.data_batch), [0], False)
        self._check_caches()
        if not isinstance(prepared, tuple) or len(prepared) != 8:
            raise ValueError("runtime preparation must provide the verified 8-field interface")
        if len(prepared[0]) != 1 or prepared[7]:
            raise ValueError("only batch-one vision forward dynamics without noisy actions is supported")
        carrier = projection(prepared[1].x0_tokens_vision[0])
        if projection(prepared[4][0]).size != carrier.size or projection(prepared[6][0]).size != carrier.size:
            raise ValueError("sampler state must contain exactly the runtime vision carrier")
        mask = projection(prepared[6][0]).reshape(carrier.shape)
        indexes = prepared[0][0].condition_frame_indexes_vision
        factory = inputs_factory or PrecisionInputs
        self.inputs = factory(carrier, indexes, mask, direction_bank)
        self.prepared = _clone_runtime(prepared)
        self._reset()
        self.provenance["decode"] = fingerprint({"decoder_state": self.model.tokenizer_vision_gen,
            "implementation": source_identity(self.model.decode), "artifact": file_identity(self.artifact_paths["decoder"])})

    def actual_identity(self):
        """Recompute actual content, independent of caller provenance claims."""
        decoder = getattr(self.model, "tokenizer_vision_gen", None)
        if decoder is None or not hasattr(self.model, "config"):
            raise ValueError("actual decoder and model config are required for provenance")
        torch = getattr(self.ops, "torch", None)
        backend = {"numpy": np.__version__}
        if torch is not None:
            backend.update({"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                            "cudnn_build": torch.backends.cudnn.version(), "default_dtype": str(torch.get_default_dtype()),
                            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "float32_matmul_precision": torch.get_float32_matmul_precision()})
        return {"data_batch": fingerprint(self.data_batch), "model_state": fingerprint(self.model.net),
                "decoder_state": fingerprint(decoder), "model_config": fingerprint(self.model.config),
                "prepared": fingerprint(self.prepared), "noise_policy": {"seed": 0, "batch_size": 1,
                "source": "actual _prepare_inference_data", "initial_noise": fingerprint(self.prepared[4]),
                "mask": fingerprint(self.prepared[6])}, "inputs": self.inputs.identity(),
                "settings": fingerprint(self.settings),
                "artifacts": {name: file_identity(path) for name, path in self.artifact_paths.items()},
                "code": [source_identity(obj) for obj in (self.model._prepare_inference_data,
                    self.model._get_velocity, self.model.denoise, self.model.decode,
                    self.model.generate_samples_from_batch, self.scheduler_class, type(self.model.sampler), type(self.ops))],
                "sampler_config": fingerprint(getattr(self.model.sampler, "cfg", None)), "backend": backend}

    def _reset(self):
        for name in ("_sampler_state", "_text_kv_cache", "text_kv_cache"):
            value = getattr(self.model, name, None)
            if hasattr(value, "clear"):
                value.clear()

    def _state(self):
        return {name: bool(_cache_has_values(getattr(self.model, name, None)))
                for name in ("_sampler_state", "_text_kv_cache", "text_kv_cache")}

    def _check_caches(self):
        for owner in (self.model, getattr(self.model, "tokenizer_vision_gen", None), getattr(self.model, "tokenizer_vision", None)):
            if owner is None:
                continue
            for obj in (owner, getattr(owner, "model", None)):
                if obj is None:
                    continue
                if bool(getattr(obj, "_keep_decoder_cache", False)):
                    raise EvidenceError("decoder cache keep policy is enabled")
                if any(_cache_has_values(getattr(obj, name, None)) for name in ("_enc_cache", "_dec_cache")):
                    raise EvidenceError("VAE cache escaped request")
        if any(bool(getattr(self.model, name, False)) for name in ("_diffusion_cache_installed", "diffusion_cache_installed")):
            raise EvidenceError("diffusion cache installed")

    def _prepared_for_call(self, target):
        prepared = list(_clone_runtime(self.prepared))
        clean = prepared[1]
        clean.x0_tokens_vision[0] = self.ops.from_array(target, clean.x0_tokens_vision[0])
        # Both the reference and conditioned initial slots must carry the same
        # quantized FP32 values. Predicted slots retain the cloned baseline noise.
        mask = self.inputs.geometry.mask
        noise = projection(prepared[4][0]).reshape(mask.shape)
        noise[mask] = target[mask]
        prepared[4][0] = self.ops.from_array(noise.reshape(projection(prepared[4][0]).shape), prepared[4][0])
        reference = projection(prepared[5][0]).reshape(mask.shape)
        reference[mask] = target[mask]
        prepared[5][0] = self.ops.from_array(reference.reshape(projection(prepared[5][0]).shape), prepared[5][0])
        return tuple(prepared)

    def execute(self, spec, inputs, *, scope):
        if scope not in ("full", "module") or inputs.identity() != self.inputs.identity():
            raise ValueError("runtime scope/input identity mismatch")
        model, ops = self.model, self.ops
        target = inputs.for_spec(spec) if hasattr(inputs, "for_spec") else inputs.for_call(spec["group"], spec["alpha"], spec["sign"])
        dtype = "bfloat16" if spec["group"] == "A" else "float32"
        evidence = TensorEvidence()
        record = {"scope": scope, "tensor_evidence": evidence.rows, "steps": [], "condition_steps_fp32": [],
                  "expected_steps": self.settings["num_steps"] if scope == "full" else 1,
                  "execution": {"casts": [], "operation_count": 0}, "decode_id": self.provenance["decode"],
                  "sampler_instances": [], "telemetry": {}}
        originals = []
        request_tensor_kwargs = model.tensor_kwargs
        request_precision = model.precision
        self._reset()
        self._check_caches()
        record["request_initial_state"] = self._state()
        record["cache"] = {"requested": bool(self.provenance.get("diffusion_cache_requested", False)),
                           "installed": bool(getattr(model, "_diffusion_cache_installed", False)),
                           "initial_empty": not any(self._state().values())}
        def patch(owner, name, value):
            missing = object()
            # nn.Module stores child modules in _modules and exposes them via
            # __getattr__; they must be reinstated, unlike an inherited method
            # shadow introduced only for the duration of a hook.
            owned = name in vars(owner) or inspect.getattr_static(owner, name, missing) is missing
            originals.append((owner, name, getattr(owner, name), owned))
            setattr(owner, name, value)
        try:
            if getattr(model.net, "_mixed_precision_runtime", None) is not None:
                raise RuntimeError("mixed/quantized network runtime cannot prove pure FP32")
            original_net = model.net
            working_net = copy.deepcopy(original_net, {id(ops): ops})
            if working_net is original_net:
                raise EvidenceError("independent precision network construction returned original")
            ops.ensure_independent(original_net, working_net)
            patch(model, "net", working_net)
            ops.set_weights(working_net, dtype)
            for name, weight in ops.weights(model.net):
                evidence.record("weights", weight, name=name)
            # All groups reach a shared FP32 interface; A alone casts packed
            # modality tokens immediately before the denoiser's network call.
            model.tensor_kwargs = dict(self.original_tensor_kwargs)
            model.tensor_kwargs["dtype"] = getattr(getattr(ops, "torch", None), "float32", "float32")
            model.precision = model.tensor_kwargs["dtype"]
            original_denoise = model.denoise
            original_forward = model.net.forward
            original_prepare = model._prepare_inference_data
            original_velocity = model._get_velocity
            original_sampler_forward = model.sampler.forward
            original_scheduler_step = self.scheduler_class.step
            original_convert = self.scheduler_class.convert_model_output
            scheduler_ids = []
            def prepare(*args, **kwargs):
                bound = inspect.signature(original_prepare).bind_partial(*args, **kwargs)
                if bound.arguments.get("seed") != [0]:
                    raise EvidenceError("actual preparation seed is not paired seed zero")
                prepared = self._prepared_for_call(target)
                record["initial_state"] = projection(prepared[4][0]).reshape(inputs.z_bar.shape)
                record["noise_evidence"] = {"source": "first_velocity_input", "seed": int(bound.arguments["seed"][0]),
                                            "prepare_seed": int(bound.arguments["seed"][0]), "batch_size": len(prepared[4])}
                return prepared
            patch(model, "_prepare_inference_data", prepare)
            def sampler_forward(*args, **kwargs):
                bound = inspect.signature(original_sampler_forward).bind_partial(*args, **kwargs)
                seeds = bound.arguments.get("seed")
                if seeds != [0]:
                    raise EvidenceError("actual sampler seed is not paired seed zero")
                noise = bound.arguments["noise"]
                if len(noise) != 1 or str(noise[0].dtype).removeprefix("torch.") != "float32":
                    raise EvidenceError("sampler consumed initial state must be one FP32 sample")
                consumed = projection(noise[0]).reshape(inputs.z_bar.shape)
                record["sampler_input_state"] = consumed
                if not np.array_equal(consumed, record["initial_state"]):
                    raise EvidenceError("sampler consumed initial state differs from preparation")
                record["noise_evidence"]["seed"] = int(seeds[0])
                return original_sampler_forward(*args, **kwargs)
            patch(model.sampler, "forward", sampler_forward)
            def velocity(*args, **kwargs):
                bound = inspect.signature(original_velocity).bind_partial(*args, **kwargs)
                noise_x = bound.arguments["noise_x"]
                if "consumed_initial_state" not in record:
                    if len(noise_x) != 1 or str(noise_x[0].dtype).removeprefix("torch.") != "float32":
                        raise EvidenceError("consumed initial noise must be one FP32 sample")
                    consumed = projection(noise_x[0]).reshape(inputs.z_bar.shape)
                    record["consumed_initial_state"] = consumed
                    if not np.array_equal(consumed, record["initial_state"]):
                        raise EvidenceError("consumed initial state differs from frozen preparation")
                return original_velocity(*args, **kwargs)
            patch(model, "_get_velocity", velocity)
            def network_forward(*args, **kwargs):
                packed = kwargs.get("packed_seq", args[0] if args else None)
                if packed is None:
                    raise EvidenceError("network packed_seq boundary missing")
                step = len(record["steps"]) - 1
                tokens = packed.vision.tokens[0]
                values = evidence.record("network_condition", tokens, step=step)
                if values.shape != target.shape:
                    raise EvidenceError("network condition layout changed")
                record["condition_steps_fp32"].append(values)
                result = original_forward(*args, **kwargs)
                # Also retain a real module output observation. Dispatch records
                # internal activation dtypes, including cast-then-upcast paths.
                for value in _tensors(result):
                    evidence.record("activation", value, step=step, name="net.forward")
                    break
                return result
            patch(model.net, "forward", network_forward)
            def denoise(*args, **kwargs):
                bound = inspect.signature(original_denoise).bind_partial(*args, **kwargs)
                packed = bound.arguments.get("data_batch_packed")
                if packed is None:
                    raise EvidenceError("denoise data_batch_packed boundary missing")
                step = len(record["steps"])
                if scope == "module" and step:
                    raise EvidenceError("module experiment attempted a second network call")
                tokens = packed.vision.tokens[0]
                if str(tokens.dtype).removeprefix("torch.") != "float32":
                    raise PrecisionCompatibilityError("hidden cast before common FP32 condition interface")
                actual = projection(tokens)
                if actual.shape != target.shape:
                    raise EvidenceError("packed vision shape does not match authoritative carrier")
                packed_mask = projection(packed.vision.condition_mask[0])
                try:
                    packed_mask = np.broadcast_to(packed_mask, actual.shape).astype(bool)
                except ValueError as error:
                    raise EvidenceError("packed network mask cannot map to runtime carrier") from error
                if not np.array_equal(packed_mask, inputs.geometry.mask):
                    raise EvidenceError("network mask differs from runtime prepare mask")
                if step == 0:
                    record["consumed_initial_mask"] = packed_mask.copy()
                exterior = actual[~packed_mask].copy()
                actual[packed_mask] = target[packed_mask]
                if not np.array_equal(actual[~packed_mask], exterior):
                    raise EvidenceError("reimposition changed predicted/noise positions")
                common = inputs.z_bar.copy()
                common[packed_mask] = actual[packed_mask]
                if "common_input_fp32" not in record:
                    record["common_input_fp32"] = evidence.record("common_condition", ops.from_array(common, tokens))
                packed.vision.tokens[0] = ops.from_array(actual, tokens)
                # Clone packed data so the generator's reusable template stays
                # FP32, including non-vision action/text request state.
                packed = _clone_runtime(packed)
                for modality in ("vision", "action", "sound", "lidar"):
                    item = getattr(packed, modality, None)
                    if item is not None and getattr(item, "tokens", None) is not None:
                        item.tokens = [ops.cast(value, dtype) for value in item.tokens]
                bound.arguments["data_batch_packed"] = packed
                timestep = float(projection(packed.vision.timesteps).reshape(-1)[0])
                record["steps"].append({"step": step, "timestep": timestep})
                with ops.compute(dtype, evidence, record["execution"], step):
                    result = original_denoise(*bound.args, **bound.kwargs)
                output = result["preds_vision"][0]
                evidence.record("denoiser_output", output, step=step)
                if scope == "module":
                    record["output_full"] = projection(output)
                    raise _StepZeroDone()
                return result
            patch(model, "denoise", denoise)
            def convert_model_output(scheduler, *args, **kwargs):
                result = original_convert(scheduler, *args, **kwargs)
                step = len([r for r in evidence.rows if r["role"] == "sampler_converted"])
                evidence.record("sampler_converted", result, step=step)
                if str(result.dtype).removeprefix("torch.") != "float32":
                    raise PrecisionCompatibilityError("UniPC converted model state is not FP32")
                return result
            patch(self.scheduler_class, "convert_model_output", convert_model_output)
            def scheduler_step(scheduler, *args, **kwargs):
                bound = inspect.signature(original_scheduler_step).bind_partial(scheduler, *args, **kwargs)
                if id(scheduler) not in scheduler_ids:
                    scheduler_ids.append(id(scheduler))
                    history = getattr(scheduler, "model_outputs", [])
                    if any(item is not None for item in history):
                        raise EvidenceError("new scheduler carries stale model history")
                    record["sampler_instances"].append({"identity": id(scheduler), "initial_history_empty": True})
                step = len([r for r in evidence.rows if r["role"] == "sampler_update"])
                sample = bound.arguments["sample"]
                velocity_value = bound.arguments["model_output"]
                evidence.record("sampler_velocity", velocity_value, step=step)
                evidence.record("sampler_accumulator", sample, step=step)
                generator = bound.arguments.get("generator")
                generator_seed = int(generator.initial_seed()) if generator is not None else None
                if generator_seed != 0:
                    raise EvidenceError("actual scheduler generator seed is not zero")
                record.setdefault("sampler_generator_seeds", []).append(generator_seed)
                if step == 0 and not np.array_equal(projection(sample).reshape(inputs.z_bar.shape), record["consumed_initial_state"]):
                    raise EvidenceError("scheduler consumed initial state changed after first velocity")
                if str(sample.dtype).removeprefix("torch.") != "float32":
                    raise PrecisionCompatibilityError("sampler accumulation input is not FP32")
                # Preserve real UniPC arithmetic. A permits its BF16 velocity
                # product before FP32 sample subtraction; no synthetic upcast.
                sampler_policy = "bf16_velocity_fp32_accumulator" if spec["group"] == "A" else "float32"
                with ops.compute(sampler_policy, evidence, record["execution"], step):
                    result = original_scheduler_step(*bound.args, **bound.kwargs)
                for value in getattr(scheduler, "model_outputs", []):
                    if value is not None and str(value.dtype).removeprefix("torch.") != "float32":
                        raise PrecisionCompatibilityError("UniPC model history is not FP32")
                evidence.record("sampler_update", result[0], step=step)
                return result
            patch(self.scheduler_class, "step", scheduler_step)
            with ops.inference():
                try:
                    result = model.generate_samples_from_batch(_clone_runtime(self.data_batch), seed=[0],
                        num_steps=self.settings["num_steps"], guidance=1., shift=10., has_negative_prompt=False,
                        skip_text_tokens_for_cfg=False, normalize_cfg=False, use_batched_cfg=False)
                    if scope == "module":
                        raise EvidenceError("module denoiser boundary was never reached")
                    latent = result["vision"][0]
                    record["output_full"] = projection(latent)
                    # Decode is always the same tokenizer and FP32 latent input,
                    # with original model settings; tokenizer weights never change.
                    model.tensor_kwargs = dict(self.original_tensor_kwargs)
                    model.precision = self.original_precision
                    decoded = projection(model.decode(ops.cast(latent, "float32")))
                    decoded = np.clip((1. + decoded) / 2., 0., 1.)
                    if decoded.ndim == 5 and decoded.shape[0] == 1:
                        decoded = decoded[0]
                    if decoded.ndim != 4 or decoded.shape[0] != 3:
                        raise EvidenceError("decoded image layout is not [3,T,H,W]")
                    record["decoded_final"] = decoded[:, -1].copy()
                    record["image_slicing"] = {"axis": 1, "frame_index": decoded.shape[1]-1,
                                                "source_shape": list(decoded.shape)}
                except _StepZeroDone:
                    pass
            record["telemetry"] = ops.telemetry()
        except Exception as error:
            error.capture = record
            raise
        finally:
            for owner, name, original, owned in reversed(originals):
                if owned:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
            model.tensor_kwargs = request_tensor_kwargs
            model.precision = request_precision
            record["request_final_state"] = self._state()
            self._reset()
            self._check_caches()
            record["cache"]["final_empty"] = not any(self._state().values())
        record["condition_steps_fp32"] = np.stack(record["condition_steps_fp32"])
        return record
