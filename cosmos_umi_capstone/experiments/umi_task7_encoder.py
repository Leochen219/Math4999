"""Request-local native/FP32 feedback encoding for the loaded UMI VAE.

The public seam is :class:`FeedbackEncoder`.  It accepts a floating-point RGB
frame in ``[0, 1]`` and returns independent CPU arrays together with evidence
about the actual encoder boundary and compute dtypes.  ``temporary_encoder_precision``
is the smaller context-manager API for callers that need to perform a single
request while the already-loaded VAE is temporarily converted to FP32.

This module deliberately does not load Cosmos, resize frames, write media, or
retain hooks.  All model mutations and observations are request-local and are
restored in ``finally`` blocks, including on encoder failure.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

try:
    from .umi_task6_cosmos_loader import _content_identity
except ImportError:  # pragma: no cover - direct ``experiments`` imports
    from umi_task6_cosmos_loader import _content_identity


_MISSING = object()
_CONSTANT_NAMES = ("mean", "std", "scale", "img_mean", "img_std", "video_mean", "video_std")
_CACHE_NAMES = ("_enc_cache", "_dec_cache", "cache", "_cache")


class EncoderPrecisionError(RuntimeError):
    """Raised when a requested FP32 call performs hidden non-FP32 compute."""


def _json_safe(value: Any) -> Any:
    """Keep identity evidence JSON-serializable without discarding content."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _owners(encoder: Any) -> list[Any]:
    """Return the wrapper/model chain without traversing arbitrary user data."""
    result: list[Any] = []
    seen: set[int] = set()
    current = encoder
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current)); result.append(current)
        current = getattr(current, "model", None)
    return result


def _resolve_encoder(encoder: Any) -> tuple[Any, Any, Any]:
    owners = _owners(encoder)
    wrapper = owners[0]
    vae = owners[1] if len(owners) > 1 else wrapper
    inner = owners[2] if len(owners) > 2 and callable(getattr(owners[2], "encode", None)) else vae
    return wrapper, vae, inner


def _iter_modules(value: Any) -> Iterator[Any]:
    seen: set[int] = set()
    for owner in _owners(value):
        if id(owner) in seen:
            continue
        seen.add(id(owner)); yield owner
        modules = getattr(owner, "modules", None)
        if callable(modules):
            try:
                for module in modules():
                    if id(module) not in seen:
                        seen.add(id(module)); yield module
            except (TypeError, RuntimeError):
                pass


def _torch_tensor(torch: Any, value: Any) -> bool:
    try:
        return bool(torch.is_tensor(value))
    except (AttributeError, TypeError):
        return hasattr(value, "detach") and hasattr(value, "dtype")


def _first_tensor(value: Any, torch: Any) -> Any:
    if _torch_tensor(torch, value) or isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        for item in value.values():
            try:
                return _first_tensor(item, torch)
            except TypeError:
                continue
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item, torch)
            except TypeError:
                continue
    for name in ("latent", "latents", "sample"):
        item = getattr(value, name, None)
        if item is not None:
            return _first_tensor(item, torch)
    raise TypeError("encoder returned no tensor")


def _array_copy(value: Any, torch: Any) -> np.ndarray:
    """Materialize a finite independent CPU array without losing dtype evidence."""
    if _torch_tensor(torch, value):
        detached = value.detach()
        try:
            array = detached.cpu().numpy()
        except (TypeError, RuntimeError):
            # NumPy has no native bfloat16 representation on many versions.
            array = detached.float().cpu().numpy()
        array = np.asarray(array)
    else:
        array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("encoder returned a non-numeric tensor")
    result = np.array(array, dtype=np.float32, copy=True)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError("encoder produced a nonfinite or empty latent")
    return result


def _input_copy(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(frame)
    if not np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError("RGB frame must be a floating-point array in [0,1]; uint8 is not accepted")
    array = np.array(array, dtype=np.float32, copy=True)
    if array.ndim == 4 and array.shape[0] == 1:
        array = np.array(array[0], dtype=np.float32, copy=True)
    if array.ndim != 3:
        raise ValueError("RGB frame must be CHW or HWC")
    # Prefer CHW when both dimensions happen to equal three; otherwise infer
    # HWC only from an explicit final RGB channel.
    if array.shape[0] == 3:
        chw = array
    elif array.shape[-1] == 3:
        chw = np.transpose(array, (2, 0, 1)).copy()
    else:
        raise ValueError("RGB frame must have exactly three channels")
    if min(chw.shape[1:]) <= 0 or not np.all(np.isfinite(chw)):
        raise ValueError("RGB frame must be finite and nonempty")
    if float(chw.min()) < 0.0 or float(chw.max()) > 1.0:
        raise ValueError("RGB frame values must be in [0,1]")
    # The source frame is kept independent from the normalized encoder input.
    source = np.array(chw, dtype=np.float32, copy=True)
    normalized = np.array(source[None, :, None, :, :] * 2.0 - 1.0, dtype=np.float32, copy=True)
    return source, normalized


def _convert_value(value: Any, torch: Any) -> Any:
    if _torch_tensor(torch, value):
        try:
            return value.to(dtype=torch.float32) if value.is_floating_point() else value
        except TypeError:
            return value.to(torch.float32) if value.is_floating_point() else value
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
        return np.array(value, dtype=np.float32, copy=True)
    if isinstance(value, tuple):
        return tuple(_convert_value(item, torch) for item in value)
    if isinstance(value, list):
        return [_convert_value(item, torch) for item in value]
    if isinstance(value, dict):
        return {key: _convert_value(item, torch) for key, item in value.items()}
    return value


def _call_to(module: Any, torch: Any) -> None:
    method = getattr(module, "to", None)
    if not callable(method):
        raise EncoderPrecisionError("encoder does not expose a dtype-conversion API")
    try:
        method(dtype=torch.float32)
    except TypeError:
        method(torch.float32)


class _PrecisionSnapshot:
    def __init__(self, encoder: Any, torch: Any):
        self.encoder = encoder
        self.torch = torch
        self.owners = _owners(encoder)
        self.modules = list(_iter_modules(encoder))
        # Keep the owning module and collection slot as well as the original
        # tensor.  ``Module.to`` commonly replaces buffer objects in
        # ``_buffers``; restoring only ``tensor.data`` would leak the FP32
        # replacement into the loaded model.
        self.tensors: list[tuple[Any, str, str, Any, Any]] = []
        seen: set[int] = set()
        for module in self.modules:
            for collection_name in ("_parameters", "_buffers"):
                collection = getattr(module, collection_name, None)
                if not isinstance(collection, dict):
                    continue
                for name, value in collection.items():
                    if value is not None and id(value) not in seen and hasattr(value, "data"):
                        seen.add(id(value)); self.tensors.append((module, collection_name, name, value, value.data))
        self.attributes: list[tuple[Any, str, Any]] = []
        for owner in self.owners:
            for name in ("dtype",) + _CONSTANT_NAMES:
                if hasattr(owner, name):
                    self.attributes.append((owner, name, getattr(owner, name)))
        self.training = [(module, bool(getattr(module, "training")))
                         for module in self.modules if hasattr(module, "training")]

    def convert(self) -> None:
        # Calling ``to`` on the VAE wrapper is important: it reaches floating
        # buffers registered outside the inner model while leaving integers.
        targets = [module for module in self.modules if callable(getattr(module, "to", None))]
        if not targets:
            raise EncoderPrecisionError("encoder has no module to convert to FP32")
        _call_to(targets[0], self.torch)
        for owner in self.owners:
            if hasattr(owner, "dtype"):
                setattr(owner, "dtype", self.torch.float32)
            for name in _CONSTANT_NAMES:
                if hasattr(owner, name):
                    setattr(owner, name, _convert_value(getattr(owner, name), self.torch))
        # Fail closed if a floating parameter/buffer escaped conversion.
        for module, collection_name, name, tensor, _data in self.tensors:
            collection = getattr(module, collection_name, None)
            current = collection.get(name, tensor) if isinstance(collection, dict) else tensor
            if getattr(current, "is_floating_point", lambda: False)() and _dtype_name(current) != "float32":
                raise EncoderPrecisionError("FP32 request left a floating parameter or buffer in non-FP32 dtype")

    def restore(self) -> None:
        for module, collection_name, name, tensor, data in self.tensors:
            collection = getattr(module, collection_name, None)
            if isinstance(collection, dict) and collection.get(name) is not tensor:
                collection[name] = tensor
            try:
                tensor.data = data
            except (AttributeError, RuntimeError):
                pass
        for owner, name, value in self.attributes:
            setattr(owner, name, value)
        for module, training in self.training:
            module.training = training


@contextmanager
def temporary_encoder_precision(encoder: Any, *, precision: str = "temporary_fp32") -> Iterator[dict[str, Any]]:
    """Temporarily convert the loaded encoder's floating state to FP32.

    The exact same parameter/buffer objects are restored on exit.  Integer
    buffers and non-floating constants are never converted.
    """
    if precision not in {"temporary_fp32", "fp32"}:
        raise ValueError("temporary_encoder_precision requires temporary_fp32")
    import torch
    snapshot = _PrecisionSnapshot(encoder, torch)
    primary_error: BaseException | None = None
    try:
        snapshot.convert()
        yield {"converted": True, "compute_dtype": "float32"}
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            snapshot.restore()
        except BaseException as restore_error:
            if primary_error is not None:
                primary_error.add_note("encoder precision restoration failed: " + repr(restore_error))
            else:
                raise


def _clear_cache(encoder: Any) -> None:
    for owner in _owners(encoder):
        for name in ("reset_cache", "clear_cache"):
            method = getattr(owner, name, None)
            if callable(method):
                method()
                break
        for name in _CACHE_NAMES:
            value = getattr(owner, name, None)
            if hasattr(value, "clear"):
                value.clear()
            if value is not None and hasattr(value, "__len__"):
                try:
                    if len(value) != 0:
                        raise RuntimeError("encoder cache did not clear")
                except TypeError:
                    pass


def _autocast_state(torch: Any, device: str) -> bool | None:
    method = getattr(torch, "is_autocast_enabled", None)
    if not callable(method):
        return None
    try:
        return bool(method(device))
    except TypeError:
        return bool(method()) if device == "cuda" else None


def _set_autocast(torch: Any, device: str, enabled: bool) -> None:
    method = getattr(torch, "set_autocast_enabled", None)
    if not callable(method):
        return
    try:
        method(device, enabled)
    except TypeError:
        if device == "cuda":
            method(enabled)


@contextmanager
def _backend_guard(torch: Any, evidence: dict[str, Any]) -> Iterator[None]:
    old_tf32: list[tuple[Any, str, Any]] = []
    backends = getattr(torch, "backends", None)
    cuda = getattr(backends, "cuda", None)
    cudnn = getattr(backends, "cudnn", None)
    for backend in (cuda, getattr(cuda, "matmul", None), cudnn):
        if backend is not None and hasattr(backend, "allow_tf32"):
            if not any(item[0] is backend and item[1] == "allow_tf32" for item in old_tf32):
                old_tf32.append((backend, "allow_tf32", backend.allow_tf32))
            backend.allow_tf32 = False
    old_auto = {device: _autocast_state(torch, device) for device in ("cuda", "cpu")}
    for device in old_auto:
        if old_auto[device] is not None:
            _set_autocast(torch, device, False)
    primary_error: BaseException | None = None
    try:
        evidence["autocast_disabled"] = all(
            value is None or not bool(_autocast_state(torch, device))
            for device, value in old_auto.items()
        )
        evidence["tf32_disabled"] = all(not bool(item[0].allow_tf32) for item in old_tf32)
        yield
    except BaseException as error:
        primary_error = error
        raise
    finally:
        restore_errors: list[BaseException] = []
        for backend, name, value in old_tf32:
            try:
                setattr(backend, name, value)
            except BaseException as error:
                restore_errors.append(error)
        for device, value in old_auto.items():
            if value is not None:
                try:
                    _set_autocast(torch, device, value)
                except BaseException as error:
                    restore_errors.append(error)
        if restore_errors:
            if primary_error is not None:
                primary_error.add_note("backend restoration failed: " + repr(restore_errors[0]))
            else:
                raise restore_errors[0]


@contextmanager
def _dispatch_observer(torch: Any, *, precision: str, evidence: dict[str, Any]) -> Iterator[None]:
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten
    except ImportError:  # pragma: no cover - old torch builds
        yield
        return

    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            result = func(*args, **kwargs)
            evidence["operation_count"] = evidence.get("operation_count", 0) + 1
            tensors = [item for item in tree_flatten((args, kwargs, result))[0]
                       if torch.is_tensor(item) and item.is_floating_point()]
            dtypes = {_dtype_name(item) for item in tensors}
            dtypes.discard(None)
            counts = evidence.setdefault("operation_dtypes", {})
            for dtype in dtypes:
                counts[dtype] = counts.get(dtype, 0) + 1
            if "_to_copy" in str(func):
                before = next((_dtype_name(item) for item in tree_flatten(args)[0]
                               if torch.is_tensor(item) and item.is_floating_point()), None)
                after = next((_dtype_name(item) for item in tree_flatten(result)[0]
                              if torch.is_tensor(item) and item.is_floating_point()), None)
                if before and after and before != after:
                    evidence.setdefault("casts", []).append({"op": str(func), "from": before, "to": after})
            if precision == "temporary_fp32" and dtypes - {"float32"}:
                raise EncoderPrecisionError("hidden non-FP32 encoder operation: " + repr(sorted(dtypes)))
            return result

    with Observe():
        yield


def _call_encoder(encoder: Any, value: Any) -> Any:
    method = getattr(encoder, "encode", None) or getattr(encoder, "encode_image", None)
    if method is None:
        method = encoder if callable(encoder) else None
    if method is None:
        raise ValueError("encoder must expose encode, encode_image, or __call__")
    try:
        return method(value)
    except (TypeError, ValueError) as first:
        try:
            return method(value, device=getattr(value, "device", None))
        except TypeError:
            raise first


class FeedbackEncoder:
    """Reusable actual-encoder seam for native and request-local FP32 calls."""

    def __init__(self, encoder: Any, *, device: str = "cuda"):
        if encoder is None:
            raise ValueError("encoder is required")
        self.encoder = encoder
        self.device = str(device)
        self._identity = _content_identity(encoder)
        if not any(callable(getattr(encoder, name, None)) for name in ("encode", "encode_image", "__call__")):
            raise ValueError("encoder must expose encode/encode_image/call")

    def actual_identity(self) -> dict[str, Any]:
        observed = _content_identity(self.encoder)
        return {"adapter": "FeedbackEncoder", "device": self.device,
                "encoder": _json_safe(observed), "bound": _json_safe(self._identity)}

    def encode(self, frame: Any, *, precision: str = "native") -> dict[str, Any]:
        if precision == "fp32":
            precision = "temporary_fp32"
        if precision not in {"native", "temporary_fp32"}:
            raise ValueError("encoder precision must be native or temporary_fp32")
        source, normalized = _input_copy(frame)
        import torch
        wrapper, _vae, inner = _resolve_encoder(self.encoder)
        evidence: dict[str, Any] = {
            "precision_path": precision,
            "input_dtype": "float32",
            "input_shape": list(source.shape),
            "encoder_input_shape": list(normalized.shape),
            "encoder_input_dtype": "float32",
            "operation_count": 0,
            "operation_dtypes": {},
            "casts": [],
            "encoder_identity": self.actual_identity(),
        }
        arrays: dict[str, np.ndarray] = {
            "input_rgb": source.copy(),
            "encoder_input": normalized.copy(),
        }
        snapshot = _PrecisionSnapshot(self.encoder, torch) if precision == "temporary_fp32" else None
        old_training = [(module, bool(getattr(module, "training")))
                        for module in _iter_modules(self.encoder) if hasattr(module, "training")]
        original_inner_encode = getattr(inner, "encode", _MISSING)
        had_instance_encode = hasattr(getattr(inner, "__dict__", {}), "get") and "encode" in inner.__dict__
        primary_error: BaseException | None = None
        captured: dict[str, Any] = {}

        def observed_inner_encode(*args: Any, **kwargs: Any) -> Any:
            if args:
                critical_input = _first_tensor(args[0], torch)
            else:
                critical_input = _first_tensor(next(iter(kwargs.values())), torch)
            captured["inner_input_dtype"] = _dtype_name(critical_input)
            captured["inner_input"] = _array_copy(critical_input, torch)
            result = original_inner_encode(*args, **kwargs)
            critical_output = _first_tensor(result, torch)
            captured["inner_output_dtype"] = _dtype_name(critical_output)
            captured["inner_output"] = _array_copy(critical_output, torch)
            if precision == "temporary_fp32" and captured["inner_output_dtype"] != "float32":
                raise EncoderPrecisionError("FP32 encoder produced a non-FP32 inner output")
            return result

        try:
            _clear_cache(self.encoder)
            for module in _iter_modules(self.encoder):
                eval_method = getattr(module, "eval", None)
                if callable(eval_method):
                    eval_method()
            if snapshot is not None:
                snapshot.convert()
            if callable(original_inner_encode):
                setattr(inner, "encode", observed_inner_encode)
            with _backend_guard(torch, evidence), _dispatch_observer(torch, precision=precision, evidence=evidence):
                with torch.inference_mode():
                    encoded = _call_encoder(wrapper, torch.from_numpy(normalized.copy()).to(device=self.device, dtype=torch.float32))
            public = _first_tensor(encoded, torch)
            evidence["output_dtype"] = _dtype_name(public)
            arrays["scaled_latent"] = captured.get("inner_output", _array_copy(public, torch)).copy()
            arrays["output"] = _array_copy(public, torch).copy()
            if captured:
                arrays["inner_input"] = captured["inner_input"].copy()
                evidence.update({name: captured[name] for name in ("inner_input_dtype", "inner_output_dtype")})
            else:
                evidence["inner_input_dtype"] = None
                evidence["inner_output_dtype"] = evidence["output_dtype"]
            if precision == "temporary_fp32" and evidence["inner_input_dtype"] != "float32":
                raise EncoderPrecisionError("FP32 encoder consumed a non-FP32 inner input")
            if not np.isfinite(arrays["output"]).all():
                raise ValueError("encoder produced a nonfinite output")
            # Explicit semantic aliases keep the contract readable for Stage A
            # callers while retaining the concise keys used by the evidence
            # code above.  Every returned array is a separate CPU copy.
            arrays["actual_encoder_input"] = arrays.get("inner_input", normalized).copy()
            arrays["actual_scaled_latent"] = arrays["scaled_latent"].copy()
            arrays["actual_output"] = arrays["output"].copy()
            evidence["actual_encoder_input_dtype"] = evidence["inner_input_dtype"]
            evidence["scaled_latent_dtype"] = evidence["inner_output_dtype"]
            evidence["actual_output_dtype"] = evidence["output_dtype"]
            evidence["cache_cleared_before"] = True
            evidence["cache_cleared_after"] = False
            return {"evidence": evidence, "arrays": arrays}
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if callable(original_inner_encode):
                try:
                    if had_instance_encode:
                        setattr(inner, "encode", original_inner_encode)
                    elif "encode" in getattr(inner, "__dict__", {}):
                        delattr(inner, "encode")
                except BaseException:
                    if primary_error is None:
                        raise
            if snapshot is not None:
                try:
                    snapshot.restore()
                except BaseException as error:
                    if primary_error is None:
                        raise
                    primary_error.add_note("encoder precision restoration failed: " + repr(error))
            for module, training in old_training:
                module.training = training
            try:
                _clear_cache(self.encoder)
                evidence["cache_cleared_after"] = True
            except BaseException as error:
                if primary_error is None:
                    raise
                primary_error.add_note("encoder cache cleanup failed: " + repr(error))

    __call__ = encode


def encode_feedback_frame(encoder: Any, frame: Any, *, precision: str = "native",
                          device: str = "cuda") -> dict[str, Any]:
    """Encode one frame through a reusable :class:`FeedbackEncoder` seam."""
    return FeedbackEncoder(encoder, device=device).encode(frame, precision=precision)


__all__ = ["EncoderPrecisionError", "FeedbackEncoder", "encode_feedback_frame",
           "temporary_encoder_precision"]
