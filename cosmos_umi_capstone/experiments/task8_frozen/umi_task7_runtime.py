"""Bounded Task 7 dynamic feedback runtime.

This module is an adapter around one already-loaded
``OfficialPrecisionRuntime``.  It deliberately does not load a model, create
another resident copy, or use the old step-zero/module diagnostic.  Each call
binds one full carrier, runs the reviewed full/deferred generation, decodes
that full carrier once in FP32, and encodes the last frame once in FP32.
"""
from __future__ import annotations

import hashlib
import inspect
import numbers
import time
import gc
from collections.abc import Mapping
from typing import Any

import numpy as np

try:  # package import
    from .umi_precision_runtime import EvidenceError, projection
    from .umi_fd_post_vae_bridge import sha256_array
except ImportError:  # direct import from experiments/
    from umi_precision_runtime import EvidenceError, projection
    from umi_fd_post_vae_bridge import sha256_array


class FeedbackRuntimeError(RuntimeError):
    """A required installed/runtime seam is unavailable or incomplete."""


def _copy_array(value: Any, *, name: str, dtype=np.float32) -> np.ndarray:
    try:
        result = projection(value) if dtype is np.float32 else np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, AttributeError) as error:
        raise EvidenceError(f"{name} is not a numeric tensor") from error
    result = np.ascontiguousarray(result, dtype=dtype)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise EvidenceError(f"{name} must be finite and non-empty")
    return result.copy()


def _bool_array(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.size == 0:
        raise EvidenceError(f"{name} must be non-empty")
    return np.ascontiguousarray(result, dtype=bool).copy()


def _sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(int(dim) for dim in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _mapping(value: Any, *names: str) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
    for name in names:
        candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def _call_bound(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Bind before calling; never retry a method whose body raised TypeError."""
    try:
        inspect.signature(method).bind(*args, **kwargs)
    except (TypeError, ValueError) as error:
        raise FeedbackRuntimeError(f"runtime seam signature does not accept request: {error}") from error
    return method(*args, **kwargs)


def _clone_runtime(value: Any) -> Any:
    try:
        from .umi_fd_post_vae_scan import _clone_runtime
    except ImportError:
        from umi_fd_post_vae_scan import _clone_runtime
    return _clone_runtime(value)


def _dtype_name(value: Any) -> str:
    return str(getattr(value, "dtype", value)).removeprefix("torch.").lower()


def _source_identity(function: Any) -> dict[str, Any]:
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        source = None
    return {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", getattr(function, "__name__", type(function).__name__)),
        "source_sha256": None if source is None else hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        try:
            value = value.cpu().numpy()
        except (TypeError, ValueError):
            try:
                value = value.float().cpu().numpy()
            except (AttributeError, TypeError, ValueError):
                pass
    return np.asarray(value)


class _DynamicInputs:
    """Request-local full carrier with the resident input identity."""

    def __init__(self, base: Any, full: np.ndarray):
        self._base = base
        self._full = np.array(full, dtype=np.float32, copy=True)
        # OfficialPrecisionRuntime.validate_capture uses z_bar as the
        # authoritative exterior.  For Task 7 it must be the frozen source
        # carrier, not a stale quantized or previous request carrier.
        self.z_bar = self._full.copy()
        self.z0 = self._full.copy()

    def identity(self) -> Any:
        return {"base": self._base.identity(), "bound_full_sha256": sha256_array(self._full),
                "binding": "task7-request-local"}

    def for_spec(self, _spec: Mapping[str, Any]) -> np.ndarray:
        return self._full.copy()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def _resolve_arch_invariant_rand(runtime: Any) -> tuple[Any, dict[str, Any]]:
    function = getattr(runtime, "arch_invariant_rand", None)
    if not callable(function):
        try:
            from cosmos_framework.utils.misc import arch_invariant_rand as function
        except (ImportError, ModuleNotFoundError) as error:
            raise FeedbackRuntimeError("official arch_invariant_rand is unavailable") from error
    return function, _source_identity(function)


def _as_full_shape(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    result = _copy_array(value, name=name)
    if tuple(result.shape) != tuple(shape):
        raise EvidenceError(f"{name} shape {result.shape} differs from authoritative {shape}")
    return result


def _set_from_array(ops: Any, values: np.ndarray, like: Any) -> Any:
    method = getattr(ops, "from_array", None)
    if not callable(method):
        raise FeedbackRuntimeError("official ops.from_array is required")
    return _call_bound(method, np.array(values, dtype=np.float32, copy=True), like, dtype="float32")


def _seeded_prepared(runtime: Any, target: np.ndarray, mask: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    """Clone the official eight-tuple and rebuild only request-local state."""
    prepared = list(_clone_runtime(runtime.prepared))
    clean = prepared[1]
    full_template = clean.x0_tokens_vision[0]
    full_shape = tuple(int(dim) for dim in _to_numpy(full_template).shape)
    if full_shape != tuple(target.shape):
        raise EvidenceError("official clean carrier shape differs from frozen carrier")
    function, function_identity = _resolve_arch_invariant_rand(runtime)
    fp32_kwargs = getattr(getattr(runtime, "model", None), "tensor_kwargs_fp32", None)
    if not isinstance(fp32_kwargs, Mapping):
        raise FeedbackRuntimeError("official model.tensor_kwargs_fp32 is required for architecture noise")
    dtype = _mapping(fp32_kwargs, "dtype")
    device = _mapping(fp32_kwargs, "device")
    if dtype is None or device is None:
        raise FeedbackRuntimeError("official model.tensor_kwargs_fp32 must expose dtype and device")
    # This is the actual source construction used by the installed official
    # preparation path: draw at full carrier shape before packed flattening.
    pure = _call_bound(function, full_shape, dtype, device, int(seed))
    pure_values = _as_full_shape(pure, full_shape, name="arch_invariant_rand output")
    blended = np.array(pure_values, dtype=np.float32, copy=True)
    blended[mask] = target[mask]
    clean.x0_tokens_vision[0] = _set_from_array(runtime.ops, target, full_template)

    noise_template = prepared[4][0]
    noise_shape = tuple(int(dim) for dim in _to_numpy(noise_template).shape)
    if int(np.prod(noise_shape, dtype=np.int64)) != int(np.prod(full_shape, dtype=np.int64)):
        raise EvidenceError("official initial-noise layout is not the full carrier")
    prepared[4][0] = _set_from_array(runtime.ops, blended.reshape(noise_shape), noise_template)

    reference_template = prepared[5][0]
    reference_shape = tuple(int(dim) for dim in _to_numpy(reference_template).shape)
    if int(np.prod(reference_shape, dtype=np.int64)) != int(np.prod(full_shape, dtype=np.int64)):
        raise EvidenceError("official reference layout is not the full carrier")
    reference = _copy_array(reference_template, name="official reference").reshape(full_shape)
    reference[mask] = target[mask]
    prepared[5][0] = _set_from_array(runtime.ops, reference.reshape(reference_shape), reference_template)
    clean_readback = _copy_array(clean.x0_tokens_vision[0], name="prepared clean readback").reshape(full_shape)
    initial_readback = _copy_array(prepared[4][0], name="prepared initial readback").reshape(full_shape)
    reference_readback = _copy_array(prepared[5][0], name="prepared reference readback").reshape(full_shape)
    return tuple(prepared), {
        "pure_noise": pure_values,
        "blended_initial": blended,
        "reference": reference,
        "clean_readback": clean_readback,
        "initial_readback": initial_readback,
        "reference_readback": reference_readback,
        "condition_slot_hashes": {"clean": _sha(target), "reference": _sha(reference),
                                  "initial_mask": _sha(blended[mask])},
        "function": function_identity,
        "shape": list(full_shape),
        "dtype": _dtype_name(dtype),
        "device": None if device is None else str(device),
        "seed": int(seed),
    }


def _normalise_decoder_output(raw_value: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = _copy_array(raw_value, name="decoder raw output")
    float64 = raw.astype(np.float64, copy=False)
    unclipped = (1.0 + float64) / 2.0
    normalized = np.clip(unclipped, 0.0, 1.0).astype(np.float32)
    clamp_fraction = float(np.mean((unclipped < 0.0) | (unclipped > 1.0)))
    if normalized.ndim == 5 and normalized.shape[0] == 1:
        normalized = normalized[0]
    if normalized.ndim != 4 or normalized.shape[0] != 3:
        raise EvidenceError("decoder output must be [1,3,T,H,W] or [3,T,H,W]")
    frame = np.array(normalized[:, -1], dtype=np.float32, copy=True)
    return raw, frame, {
        "input_range": "[-1,1]",
        "output_range": "[0,1]",
        "arithmetic": "float64 then float32 storage",
        "clamp_fraction": clamp_fraction,
        "layout": "CHW",
        "source_shape": list(normalized.shape),
        "frame_axis": 1,
        "frame_index": int(normalized.shape[1] - 1),
        "resize": False,
    }


def _clear_decoder_cache(tokenizer: Any) -> None:
    clear = getattr(tokenizer, "clear_cache", None)
    if not callable(clear):
        clear = getattr(getattr(getattr(tokenizer, "model", None), "model", None), "clear_decoder_cache", None)
    if callable(clear):
        _call_bound(clear)


def _official_fp32_decode(runtime: Any, full_latent: np.ndarray) -> dict[str, Any]:
    """One actual decoder invocation, reusing Task 1 precision machinery."""
    try:
        import torch
    except (ImportError, ModuleNotFoundError) as error:
        raise FeedbackRuntimeError("Torch is required for an official FP32 decoder call") from error
    try:
        from .umi_task7_encoder import (_PrecisionSnapshot, _backend_guard,
                                        _clear_cache, _dispatch_observer,
                                        _state_dtype_evidence)
    except ImportError:  # direct import from experiments/
        from umi_task7_encoder import (_PrecisionSnapshot, _backend_guard,
                                       _clear_cache, _dispatch_observer,
                                       _state_dtype_evidence)
    model, ops = runtime.model, runtime.ops
    tokenizer = getattr(model, "tokenizer_vision_gen", None)
    wan = getattr(tokenizer, "model", None)
    inner = getattr(wan, "model", None)
    if tokenizer is None or wan is None or inner is None or not callable(getattr(inner, "decode", None)):
        raise EvidenceError("loaded decoder does not expose an inspectable inner decode")
    source = _copy_array(full_latent, name="decoder input full latent")
    like = runtime.prepared[1].x0_tokens_vision[0]
    old_model_kwargs, old_precision = model.tensor_kwargs, model.precision
    original_decode = inner.decode
    had_instance_decode = "decode" in getattr(inner, "__dict__", {})
    original_model_decode = getattr(model, "decode", None)
    had_instance_model_decode = "decode" in getattr(model, "__dict__", {})
    if not callable(original_model_decode):
        raise EvidenceError("loaded model does not expose decode")
    evidence: dict[str, Any] = {"operation_count": 0, "operation_dtypes": {}, "invocations": 0,
                                "model_decode_invocations": 0,
                                "precision_path": "temporary_fp32", "decoder_input_full": source.copy()}
    snapshot = _PrecisionSnapshot(tokenizer, torch)
    primary: BaseException | None = None
    try:
        _clear_cache(tokenizer)
        snapshot.convert()
        evidence["state_dtypes"] = _state_dtype_evidence(tokenizer, torch)
        floating_state = [dtype for bucket in evidence["state_dtypes"].values() for dtype in bucket.values()]
        if not floating_state or any(str(dtype) != "float32" for dtype in floating_state):
            raise EvidenceError("FP32 decoder state evidence is incomplete or non-FP32")
        for module in (tokenizer, wan, inner):
            eval_method = getattr(module, "eval", None)
            if callable(eval_method): _call_bound(eval_method)
        model.tensor_kwargs = dict(old_model_kwargs)
        model.tensor_kwargs["dtype"] = torch.float32
        model.precision = torch.float32

        def observed_decode(*args: Any, **kwargs: Any) -> Any:
            evidence["invocations"] += 1
            if not args:
                raise EvidenceError("decoder inner input is missing")
            evidence["inner_input_dtype"] = _dtype_name(args[0])
            result = original_decode(*args, **kwargs)
            evidence["inner_output_dtype"] = _dtype_name(result)
            if evidence["inner_input_dtype"] != "float32" or evidence["inner_output_dtype"] != "float32":
                raise EvidenceError("FP32 decoder boundary consumed or produced a non-FP32 tensor")
            return result

        def observed_model_decode(*args: Any, **kwargs: Any) -> Any:
            evidence["model_decode_invocations"] += 1
            return _call_bound(original_model_decode, *args, **kwargs)

        latent = _call_bound(ops.from_array, source.copy(), like, dtype="float32")
        with _backend_guard(torch, evidence), _dispatch_observer(torch, precision="temporary_fp32", evidence=evidence):
            with torch.inference_mode():
                inner.decode = observed_decode
                model.decode = observed_model_decode
                decoded = _call_bound(model.decode, ops.cast(latent, "float32"))
        if (evidence["invocations"] != 1 or evidence["model_decode_invocations"] != 1
                or evidence["operation_count"] <= 0):
            raise EvidenceError("FP32 decoder evidence did not prove exactly one actual decode")
        raw, frame, normalization = _normalise_decoder_output(decoded)
        evidence["decoder_raw_output"] = raw.copy()
        evidence["decoded_last_rgb"] = frame.copy()
        evidence["normalization"] = normalization
        return evidence
    except BaseException as error:
        primary = error
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        try:
            if had_instance_decode:
                inner.decode = original_decode
            elif "decode" in getattr(inner, "__dict__", {}):
                delattr(inner, "decode")
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            if had_instance_model_decode:
                model.decode = original_model_decode
            elif "decode" in getattr(model, "__dict__", {}):
                delattr(model, "decode")
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            snapshot.restore()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            snapshot.verify_restored()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            model.tensor_kwargs, model.precision = old_model_kwargs, old_precision
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            _clear_cache(tokenizer)
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary is not None:
                for error in cleanup_errors:
                    primary.add_note("decoder cleanup failed: " + repr(error))
            else:
                failure = RuntimeError("decoder cleanup failed")
                for error in cleanup_errors: failure.add_note(repr(error))
                raise failure from cleanup_errors[0]


class FeedbackRuntime:
    """One loaded official runtime with request-local G/D/E feedback state."""

    def __init__(self, runtime: Any, *, encoder: Any, z0: Any | None = None,
                 mask: Any | None = None, condition_indexes: Any | None = None,
                 v0: Any | None = None, seed: int = 0, reference: Any | None = None):
        if runtime is None or encoder is None:
            raise ValueError("one resident runtime and one feedback encoder are required")
        if z0 is None: z0 = reference
        if z0 is None or v0 is None:
            raise ValueError("frozen Task 6 z0 and v0 are required; no re-encoding fallback is allowed")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or int(seed) not in (0, 1):
            raise ValueError("feedback seed must be exactly 0 or 1")
        if not callable(getattr(runtime, "execute", None)) or getattr(runtime, "inputs", None) is None:
            raise FeedbackRuntimeError("resident runtime must expose OfficialPrecisionRuntime.execute and inputs")
        self.runtime, self.encoder, self.seed = runtime, encoder, int(seed)
        self.z0 = _copy_array(z0, name="frozen Task 6 z0")
        geometry = getattr(getattr(runtime, "inputs", None), "geometry", None)
        if mask is None: mask = getattr(geometry, "mask", None)
        if condition_indexes is None: condition_indexes = getattr(geometry, "condition_indexes", None)
        if mask is None or condition_indexes is None:
            raise ValueError("runtime mask and condition indexes are required")
        self.mask = _bool_array(mask, name="runtime mask")
        if self.mask.shape != self.z0.shape:
            raise ValueError("runtime mask must match frozen carrier shape")
        raw_indexes = tuple(int(index) for index in condition_indexes)
        if not raw_indexes or len(raw_indexes) != len(set(raw_indexes)):
            raise ValueError("condition indexes must be non-empty and unique")
        self.temporal_axis = self.z0.ndim - 3
        if any(index < 0 or index >= self.z0.shape[self.temporal_axis] for index in raw_indexes):
            raise ValueError("condition index is outside frozen carrier temporal extent")
        self.condition_indexes = tuple(sorted(raw_indexes))
        expected = np.zeros_like(self.mask, dtype=bool)
        for index in self.condition_indexes:
            selection = [slice(None)] * self.mask.ndim; selection[self.temporal_axis] = index
            if not np.all(self.mask[tuple(selection)]):
                raise ValueError("runtime mask is not framewise at each condition index")
            expected[tuple(selection)] = True
        if not np.array_equal(self.mask, expected):
            raise ValueError("runtime mask has coordinates outside authoritative condition indexes")
        self.predicted_indexes = tuple(index for index in range(self.z0.shape[self.temporal_axis])
                                       if index not in self.condition_indexes)
        if not self.predicted_indexes:
            raise ValueError("runtime mask leaves no predicted temporal coordinates")
        self.v0 = _copy_array(v0, name="frozen Task 6 v0")
        if self.v0.shape != self.z0.shape:
            raise ValueError("frozen v0 must match full carrier shape")
        if np.any(self.v0[~self.mask] != 0.0):
            raise ValueError("frozen v0 must be zero outside the runtime condition mask")
        self.z0.setflags(write=False); self.mask.setflags(write=False); self.v0.setflags(write=False)

    def for_seed(self, seed: int) -> "FeedbackRuntime":
        return type(self)(self.runtime, encoder=self.encoder, z0=self.z0, mask=self.mask,
                          condition_indexes=self.condition_indexes, v0=self.v0, seed=seed)

    bind_seed = for_seed

    def identity(self) -> dict[str, Any]:
        return {"runtime": f"{type(self.runtime).__module__}.{type(self.runtime).__qualname__}",
                "encoder": f"{type(self.encoder).__module__}.{type(self.encoder).__qualname__}",
                "z0": _sha(self.z0), "mask": _sha(self.mask), "v0": _sha(self.v0),
                "condition_indexes": list(self.condition_indexes), "seed": self.seed}

    def extract_condition(self, fullcarrier: Any) -> np.ndarray:
        full = _copy_array(fullcarrier, name="full carrier")
        if full.shape != self.z0.shape:
            raise ValueError("full carrier shape differs from frozen Task 6 carrier")
        return np.ascontiguousarray(np.take(full, self.condition_indexes, axis=self.temporal_axis), dtype=np.float32).copy()

    def embed_condition(self, condition_only: Any) -> np.ndarray:
        condition = _copy_array(condition_only, name="condition-only carrier")
        expected_shape = list(self.z0.shape)
        expected_shape[self.temporal_axis] = len(self.condition_indexes)
        if tuple(condition.shape) != tuple(expected_shape):
            raise ValueError(f"condition-only shape must be {tuple(expected_shape)}, got {condition.shape}")
        full = self.z0.copy()
        for ordinal, index in enumerate(self.condition_indexes):
            destination = [slice(None)] * full.ndim; destination[self.temporal_axis] = index
            source = [slice(None)] * condition.ndim; source[self.temporal_axis] = ordinal
            full[tuple(destination)] = condition[tuple(source)]
        if not np.array_equal(full[~self.mask], self.z0[~self.mask]):
            raise EvidenceError("condition embedding changed frozen exterior bytes")
        return full

    def _validate_condition(self, condition_only: Any) -> np.ndarray:
        value = np.asarray(condition_only)
        if not np.issubdtype(value.dtype, np.floating):
            raise ValueError("condition-only input must be floating point")
        expected_shape = list(self.z0.shape); expected_shape[self.temporal_axis] = len(self.condition_indexes)
        if tuple(value.shape) != tuple(expected_shape):
            raise ValueError("condition-only input shape differs from authoritative temporal layout")
        return _copy_array(value, name="condition-only input")

    def _cleanup_runtime(self) -> dict[str, Any]:
        errors: list[BaseException] = []
        reset = getattr(self.runtime, "_reset", None)
        if callable(reset):
            try: _call_bound(reset)
            except BaseException as error: errors.append(error)
        check = getattr(self.runtime, "_check_caches", None)
        if callable(check):
            try: _call_bound(check)
            except BaseException as error: errors.append(error)
        for name in ("release_generation_state", "cleanup_generation"):
            method = getattr(self.runtime, name, None)
            if callable(method):
                try: _call_bound(method)
                except BaseException as error: errors.append(error)
        collected = int(gc.collect())
        cuda_cleared = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_cleared = True
        except ImportError:
            pass
        except (AttributeError, RuntimeError) as error:
            errors.append(error)
        if errors:
            failure = RuntimeError("feedback runtime cleanup failed")
            for error in errors: failure.add_note(repr(error))
            raise failure from errors[0]
        return {"gc_collected": collected, "cuda_cache_cleared": cuda_cleared}

    def _validate_generation(self, record: Mapping[str, Any], target: np.ndarray,
                             noise_capture: Mapping[str, Any], seed: int) -> dict[str, Any]:
        if record.get("scope") != "full" or record.get("decode_policy") != "deferred":
            raise EvidenceError("feedback requires the full/deferred official generation path")
        if int(record.get("expected_steps", -1)) != 30 or len(record.get("steps", ())) != 30:
            raise EvidenceError("feedback generation must observe all 30 configured steps")
        if list(record.get("sampler_generator_seeds", ())) != [seed] * 30:
            raise EvidenceError("actual scheduler generator seed evidence is missing or mismatched")
        noise_evidence = record.get("noise_evidence")
        if not isinstance(noise_evidence, Mapping) or int(noise_evidence.get("seed", -1)) != seed:
            raise EvidenceError("actual prepare/sampler noise seed evidence is missing")
        execution = record.get("execution")
        if not isinstance(execution, Mapping) or int(execution.get("operation_count", 0)) <= 0:
            raise EvidenceError("actual G operation evidence is missing")
        if not execution.get("dispatch_observed") or not execution.get("backend"):
            raise EvidenceError("actual G backend/dispatch evidence is missing")
        observed_dtypes = execution.get("operation_dtypes")
        if not isinstance(observed_dtypes, Mapping) or not observed_dtypes:
            raise EvidenceError("actual G operation dtype evidence is missing")
        if set(str(key).removeprefix("torch.").lower() for key in observed_dtypes) != {"float32"}:
            raise EvidenceError("G evidence contains an unapproved non-FP32 operation")
        conditions = record.get("condition_steps_fp32")
        if conditions is None:
            raise EvidenceError("per-step consumed condition evidence is missing")
        conditions = _copy_array(conditions, name="per-step consumed conditions")
        if conditions.shape != (30,) + target.shape or not all(np.array_equal(item[self.mask], target[self.mask]) for item in conditions):
            raise EvidenceError("dynamic condition was not exactly consumed at all 30 denoiser steps")
        full = _as_full_shape(record.get("output_full"), self.z0.shape, name="full G latent")
        predicted = record.get("predicted_latent")
        expected_predicted = np.ascontiguousarray(np.take(full, self.predicted_indexes, axis=self.temporal_axis), dtype=np.float32)
        if predicted is None or not np.array_equal(_copy_array(predicted, name="predicted G latent"), expected_predicted):
            raise EvidenceError("predicted latent is not the authoritative full-latent slice")
        pure = _as_full_shape(noise_capture.get("pure_noise"), self.z0.shape, name="prediction noise")
        prediction_hash = sha256_array(pure[~self.mask])
        if record.get("initial_noise_hash") != prediction_hash:
            raise EvidenceError("official initial noise hash is not prediction-region-only")
        historical = getattr(getattr(self.runtime, "prepared", None), "__getitem__", lambda _index: None)(4)
        historical_array = None
        try:
            historical_array = _copy_array(self.runtime.prepared[4][0], name="historical Task 6 noise").reshape(self.z0.shape)
        except (AttributeError, IndexError, TypeError, ValueError):
            raise EvidenceError("frozen Task 6 prepared noise is unavailable")
        if seed == 0 and not np.array_equal(pure[~self.mask], historical_array[~self.mask]):
            raise EvidenceError("seed-0 prediction noise differs from frozen Task 6 outside-mask bytes")
        clean_readback = _as_full_shape(noise_capture.get("clean_readback"), self.z0.shape, name="prepared clean readback")
        initial_readback = _as_full_shape(noise_capture.get("initial_readback"), self.z0.shape, name="prepared initial readback")
        reference_readback = _as_full_shape(noise_capture.get("reference_readback"), self.z0.shape, name="prepared reference readback")
        if not np.array_equal(clean_readback, target):
            raise EvidenceError("official clean slot does not exactly contain the request carrier")
        if (not np.array_equal(initial_readback[self.mask], target[self.mask])
                or not np.array_equal(reference_readback[self.mask], target[self.mask])):
            raise EvidenceError("request condition was not written to all official condition slots")
        return {"full_latent": full, "predicted_latent": expected_predicted,
                "conditions": conditions, "prediction_noise": pure,
                "prediction_noise_hash": prediction_hash, "historical_noise": historical_array,
                "clean_readback": clean_readback, "initial_readback": initial_readback,
                "reference_readback": reference_readback}

    def _decode_once(self, full_latent: np.ndarray) -> dict[str, Any]:
        method = getattr(self.runtime, "decode_fp32", None)
        if callable(method):
            result = _call_bound(method, full_latent.copy(), precision="temporary_fp32")
            raw = _mapping(result, "raw_output", "decoder_raw_output", "raw")
            if raw is None:
                raise EvidenceError("decoder seam did not expose raw output")
            invocations = _mapping(result, "invocations", "decode_count")
            operation_count = _mapping(result, "operation_count")
            dtypes = _mapping(result, "operation_dtypes")
            if invocations is None or int(invocations) != 1 or operation_count is None or int(operation_count) <= 0:
                raise EvidenceError("decoder evidence does not prove exactly one actual invocation")
            if not isinstance(dtypes, (list, tuple, set)) or not dtypes or any(_dtype_name(dtype) != "float32" for dtype in dtypes):
                raise EvidenceError("decoder FP32 operation evidence is missing or non-FP32")
            raw_array, frame, normalization = _normalise_decoder_output(raw)
            return {"raw": raw_array, "frame": frame, "normalization": normalization,
                    "evidence": {str(key): value for key, value in (result.items() if isinstance(result, Mapping) else ())},
                    "operation_count": int(operation_count), "invocations": int(invocations),
                    "operation_dtypes": [str(dtype) for dtype in dtypes],
                    "decoder_input_full": full_latent.copy()}
        result = _official_fp32_decode(self.runtime, full_latent)
        return {"raw": result["decoder_raw_output"], "frame": result["decoded_last_rgb"],
                "normalization": result["normalization"], "evidence": result,
                "operation_count": int(result["operation_count"]), "invocations": int(result["invocations"]),
                "operation_dtypes": list(result["operation_dtypes"]), "decoder_input_full": full_latent.copy()}

    def _encode_once(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
        method = getattr(self.encoder, "encode", None)
        if not callable(method):
            raise FeedbackRuntimeError("Task 1 FeedbackEncoder.encode is required")
        result = _call_bound(method, frame.copy(), precision="temporary_fp32")
        arrays = _mapping(result, "arrays")
        evidence = _mapping(result, "evidence")
        output = _mapping(arrays, "actual_output", "output")
        if output is None or not isinstance(evidence, Mapping):
            raise EvidenceError("encoder did not expose actual output and precision evidence")
        actual_input = _mapping(arrays, "actual_encoder_input")
        if actual_input is None:
            raise EvidenceError("encoder did not expose its actual normalized FP32 input array")
        condition = _copy_array(output, name="actual encoded condition")
        expected_shape = list(self.z0.shape); expected_shape[self.temporal_axis] = len(self.condition_indexes)
        if tuple(condition.shape) != tuple(expected_shape):
            raise EvidenceError("encoder output is not the exact condition-only temporal shape")
        if evidence.get("actual_encoder_input_dtype") != "float32" or evidence.get("actual_output_dtype") != "float32":
            raise EvidenceError("encoder FP32 dtype evidence is missing")
        if int(evidence.get("operation_count", 0)) <= 0 or not evidence.get("dispatch_observed"):
            raise EvidenceError("encoder operation evidence is missing")
        arrays_copy: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            if isinstance(value, (np.ndarray, list, tuple)) or hasattr(value, "dtype"):
                arrays_copy[str(name)] = _copy_array(value, name=f"encoder array {name}")
        arrays_copy["actual_encoder_input"] = _copy_array(actual_input, name="actual encoder input")
        return condition, dict(evidence), arrays_copy

    def step(self, condition_only: Any, step_index: int) -> dict[str, Any]:
        # Task 10 reuses this proven request-local path with independent seed
        # schedules [0,1] and [2,3]. Constructor defaults and Task 8 calls
        # remain unchanged; only explicit request seeds are generalized.
        if isinstance(step_index, bool) or not isinstance(step_index, numbers.Integral) or int(step_index) < 0:
            raise ValueError("step_index must be a nonnegative integer seed")
        condition = self._validate_condition(condition_only)
        full = self.embed_condition(condition)
        started = time.perf_counter()
        capture: dict[str, Any] = {"step_index": int(step_index), "seed": int(step_index),
                                   "condition_input": condition.copy(), "condition_input_full": full.copy(),
                                   "operation_counts": {"G": 0, "D": 0, "E": 0}}
        original_model_seed = getattr(self.runtime, "model_seed", None)
        original_pair_hash = getattr(self.runtime, "_paired_noise_hash", None)
        original_prepare = getattr(self.runtime, "_prepared_for_call", None)
        had_instance_prepare = "_prepared_for_call" in getattr(self.runtime, "__dict__", {})
        original_bound_inputs = getattr(self.runtime, "inputs", None)
        noise_capture: dict[str, Any] = {}
        primary: BaseException | None = None
        try:
            if not callable(original_prepare):
                raise FeedbackRuntimeError("official runtime lacks request-local preparation seam")
            dynamic = _DynamicInputs(self.runtime.inputs, full)
            request_seed = int(step_index)
            spec = {"sample_id": f"task7_step_{step_index}_seed_{request_seed}", "group": "C", "kind": "baseline",
                    "alpha": 0.0, "sign": 0, "state": getattr(self.runtime.inputs, "state", "bridge_0"),
                    "seed": request_seed, "model_seed": request_seed}

            def prepare(target: Any) -> Any:
                prepared, observed = _seeded_prepared(self.runtime, _as_full_shape(target, self.z0.shape, name="prepared target"), self.mask, request_seed)
                noise_capture.clear(); noise_capture.update(observed)
                return prepared

            self.runtime.model_seed = request_seed
            self.runtime._paired_noise_hash = None
            self.runtime._prepared_for_call = prepare
            self.runtime.inputs = dynamic
            capture["dynamic_inputs_identity"] = dynamic.identity()
            capture["request_seed"] = request_seed
            # Count one G invocation before entering the official seam so a
            # failure still reports the attempted call without fabricating
            # successful step evidence.
            capture["operation_counts"]["G"] = 1
            generation = dict(_call_bound(self.runtime.execute, spec, dynamic, scope="full", decode_policy="deferred"))
            validated = self._validate_generation(generation, full, noise_capture, request_seed)
            capture.update({"generation": generation, **validated})
            capture["denoiser_steps"] = 30
            capture["generation_cleanup"] = self._cleanup_runtime()
            capture["operation_counts"]["D"] = 1
            decoded = self._decode_once(validated["full_latent"])
            capture.update({"decoder_raw_output": decoded["raw"].copy(), "decoded_last_rgb": decoded["frame"].copy(),
                            "normalization": decoded["normalization"], "decoder": decoded["evidence"],
                            "decoder_input_full": decoded["decoder_input_full"].copy()})
            if int(decoded["invocations"]) != 1:
                raise EvidenceError("decoder invocation count differs from one")
            capture["operation_counts"]["D"] = 1
            capture["operation_counts"]["E"] = 1
            encoded_condition, encoder_evidence, encoder_arrays = self._encode_once(decoded["frame"])
            encoded_carrier = self.embed_condition(encoded_condition)
            capture.update({"encoded_condition": encoded_condition.copy(), "encoded_carrier": encoded_carrier.copy(),
                            "next_condition_fp32": encoded_condition.copy(),
                            "encoder": encoder_evidence, "actual": {
                                "prepared_condition": validated["clean_readback"].copy(),
                                "initial_condition": validated["initial_readback"].copy(),
                                "reference_condition": validated["reference_readback"].copy(),
                                "first_condition": validated["conditions"][0].copy(),
                                "last_condition": validated["conditions"][-1].copy(),
                                "condition_steps": validated["conditions"].copy()},
                            "encoder_arrays": encoder_arrays,
                            "next_consumption_check": "deferred_to_next_step"})
            capture["condition_input_fp32"] = capture["condition_input"].copy()
            capture["condition_only_fp32"] = capture["condition_input"].copy()
            capture["decoded_final_frame"] = capture["decoded_last_rgb"].copy()
            capture["evidence"] = {
                "operation_counts": dict(capture["operation_counts"]),
                "prediction_noise_hash": validated["prediction_noise_hash"],
                "noise_hash_scope": "prediction_region_only",
                "seed0_outside_mask_exact": request_seed != 0 or np.array_equal(
                    validated["prediction_noise"][~self.mask], validated["historical_noise"][~self.mask]),
                "arch_invariant_rand": noise_capture["function"],
                "condition_slot_hashes": dict(noise_capture["condition_slot_hashes"]),
                "prepare_seed": int(generation["noise_evidence"]["prepare_seed"]),
                "sampler_seeds": list(generation["sampler_generator_seeds"]),
                "full_carrier_shape": list(self.z0.shape),
                "mask_dtype": "bool",
            }
            capture["elapsed_seconds"] = time.perf_counter() - started
            return capture
        except BaseException as error:
            primary = error
            error.capture = dict(capture)
            raise
        finally:
            if had_instance_prepare:
                self.runtime._prepared_for_call = original_prepare
            elif "_prepared_for_call" in getattr(self.runtime, "__dict__", {}):
                delattr(self.runtime, "_prepared_for_call")
            self.runtime.inputs = original_bound_inputs
            if original_model_seed is not None: self.runtime.model_seed = original_model_seed
            if hasattr(self.runtime, "_paired_noise_hash"): self.runtime._paired_noise_hash = original_pair_hash
            try:
                self._cleanup_runtime()
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note("feedback cleanup failed: " + repr(cleanup_error))
                else:
                    raise


__all__ = ["EvidenceError", "FeedbackRuntime", "FeedbackRuntimeError"]
