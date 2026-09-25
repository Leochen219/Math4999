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
import inspect
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


def _cache_has_values(value: Any) -> bool:
    """Return whether a cache contains payloads, preserving None-slot lists."""
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_cache_has_values(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_cache_has_values(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return bool(value.size)
    try:
        return len(value) != 0
    except TypeError:
        return True


def _tensor_signature(value: Any) -> dict[str, Any]:
    detached = value.detach()
    flat = detached.reshape(-1)
    try:
        storage = int(detached.untyped_storage().data_ptr())
    except (AttributeError, RuntimeError):
        storage = id(detached)
    return {
        "dtype": _dtype_name(detached),
        "shape": tuple(int(item) for item in detached.shape),
        "numel": int(detached.numel()),
        "storage": storage,
        "requires_grad": bool(getattr(detached, "requires_grad", False)),
        "probe": flat[:16].clone(),
    }


def _value_signature(value: Any, torch: Any) -> Any:
    if _torch_tensor(torch, value):
        return ("tensor", _tensor_signature(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_signature(item, torch) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_value_signature(item, torch) for item in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(key), _value_signature(item, torch)) for key, item in value.items())))
    return ("object", id(value))


def _same_tensor_signature(value: Any, signature: dict[str, Any]) -> bool:
    observed = _tensor_signature(value)
    if any(observed[name] != signature[name] for name in ("dtype", "shape", "numel", "storage", "requires_grad")):
        return False
    probe = signature["probe"]
    return bool(value.detach().reshape(-1)[: probe.numel()].equal(probe))


def _same_value_signature(value: Any, signature: Any, torch: Any) -> bool:
    if signature[0] == "tensor":
        return _torch_tensor(torch, value) and _same_tensor_signature(value, signature[1])
    if signature[0] == "tuple":
        return isinstance(value, tuple) and len(value) == len(signature[1]) and all(
            _same_value_signature(observed, expected, torch) for observed, expected in zip(value, signature[1]))
    if signature[0] == "list":
        return isinstance(value, list) and len(value) == len(signature[1]) and all(
            _same_value_signature(observed, expected, torch) for observed, expected in zip(value, signature[1]))
    if signature[0] == "dict":
        if not isinstance(value, dict):
            return False
        expected = dict(signature[1])
        return set(str(key) for key in value) == set(expected) and all(
            _same_value_signature(value[key], expected[str(key)], torch) for key in value)
    return id(value) == signature[1]


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


def _training_items(modules: list[Any]) -> list[tuple[Any, bool]]:
    result: list[tuple[Any, bool]] = []
    for module in modules:
        if not hasattr(module, "training"):
            continue
        descriptor = inspect.getattr_static(type(module), "training", _MISSING)
        if isinstance(descriptor, property) and descriptor.fset is None:
            continue
        try:
            result.append((module, bool(getattr(module, "training"))))
        except (AttributeError, TypeError):
            continue
    return result


def _readonly_attribute(owner: Any, name: str) -> bool:
    """Identify derived properties that cannot accept an assignment."""
    descriptor = inspect.getattr_static(type(owner), name, _MISSING)
    return isinstance(descriptor, property) and descriptor.fset is None


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
        self.training = _training_items(self.modules)
        self.tensor_signatures = [
            (module, collection_name, name, _tensor_signature(data))
            for module, collection_name, name, _tensor, data in self.tensors
        ]
        self.original_slots = {
            (id(module), collection_name, name): tensor
            for module, collection_name, name, tensor, _data in self.tensors
        }
        self.attribute_signatures = [
            (owner, name, _value_signature(value, torch))
            for owner, name, value in self.attributes
        ]
        self.original_attributes = {(id(owner), name): value for owner, name, value in self.attributes}

    def convert(self) -> None:
        # Calling ``to`` on the VAE wrapper is important: it reaches floating
        # buffers registered outside the inner model while leaving integers.
        targets = [module for module in self.modules if callable(getattr(module, "to", None))]
        if not targets:
            raise EncoderPrecisionError("encoder has no module to convert to FP32")
        _call_to(targets[0], self.torch)
        for owner in self.owners:
            if hasattr(owner, "dtype"):
                if not _readonly_attribute(owner, "dtype"):
                    setattr(owner, "dtype", self.torch.float32)
            for name in _CONSTANT_NAMES:
                if hasattr(owner, name) and not _readonly_attribute(owner, name):
                    setattr(owner, name, _convert_value(getattr(owner, name), self.torch))
        # Fail closed if a floating parameter/buffer escaped conversion.
        for module, collection_name, name, tensor, _data in self.tensors:
            collection = getattr(module, collection_name, None)
            current = collection.get(name, tensor) if isinstance(collection, dict) else tensor
            if getattr(current, "is_floating_point", lambda: False)() and _dtype_name(current) != "float32":
                raise EncoderPrecisionError("FP32 request left a floating parameter or buffer in non-FP32 dtype")
        for owner, name, _value in self.attributes:
            if name == "dtype" and _readonly_attribute(owner, name):
                current = getattr(owner, name, _MISSING)
                if current is _MISSING or current != self.torch.float32:
                    raise EncoderPrecisionError("FP32 request left a read-only derived dtype unchanged")

    def restore(self) -> None:
        errors: list[BaseException] = []
        for module, collection_name, name, tensor, data in self.tensors:
            try:
                collection = getattr(module, collection_name, None)
                if isinstance(collection, dict) and collection.get(name) is not tensor:
                    collection[name] = tensor
                tensor.data = data
            except BaseException as error:
                errors.append(error)
        for owner, name, value in self.attributes:
            try:
                if not _readonly_attribute(owner, name):
                    setattr(owner, name, value)
            except BaseException as error:
                errors.append(error)
        for module, training in self.training:
            try:
                module.training = training
            except BaseException as error:
                errors.append(error)
        if errors:
            failure = RuntimeError("encoder state restoration failed")
            for error in errors:
                failure.add_note(repr(error))
            raise failure from errors[0]

    def verify_restored(self) -> None:
        """Fail closed if restore did not return the original state/slots."""
        errors: list[str] = []
        for module, collection_name, name, signature in self.tensor_signatures:
            collection = getattr(module, collection_name, None)
            current = collection.get(name) if isinstance(collection, dict) else None
            original = self.original_slots.get((id(module), collection_name, name))
            if current is not original or current is None or not _same_tensor_signature(current.data, signature):
                errors.append(f"tensor slot {collection_name}.{name} did not round-trip")
        for owner, name, signature in self.attribute_signatures:
            current = getattr(owner, name, _MISSING)
            original = self.original_attributes.get((id(owner), name), _MISSING)
            if current is _MISSING or current is not original or not _same_value_signature(current, signature, self.torch):
                errors.append(f"attribute {name} did not round-trip")
        for module, training in self.training:
            if getattr(module, "training", _MISSING) is not training:
                errors.append("training metadata did not round-trip")
        if errors:
            raise EncoderPrecisionError("encoder round-trip verification failed: " + "; ".join(errors))


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
        cleanup_errors: list[BaseException] = []
        try:
            snapshot.restore()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            snapshot.verify_restored()
        except BaseException as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note("encoder precision cleanup failed: " + repr(error))
            else:
                raise cleanup_errors[0]


def _clear_cache(encoder: Any) -> None:
    for owner in _owners(encoder):
        for name in ("clear_decoder_cache", "reset_cache", "clear_cache"):
            method = getattr(owner, name, None)
            if callable(method):
                method()
                break
        for name in _CACHE_NAMES:
            value = getattr(owner, name, None)
            # Wan's decoder cache is a fixed-length list of None slots after
            # clear_decoder_cache(); calling list.clear() would destroy its
            # required structure.  Only clear mutable map/set caches here.
            if isinstance(value, (dict, set)):
                value.clear()
            if _cache_has_values(value):
                raise RuntimeError("encoder cache did not clear")


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


def _state_dtype_evidence(encoder: Any, torch: Any) -> dict[str, dict[str, str]]:
    """Observe floating weights, buffers, and normalization constants per call."""
    state: dict[str, dict[str, str]] = {"parameters": {}, "buffers": {}, "constants": {}}
    used: set[str] = set()

    def put(bucket: str, name: str, value: Any) -> None:
        if not _torch_tensor(torch, value) or not getattr(value, "is_floating_point", lambda: False)():
            return
        key = name
        suffix = 1
        while key in used:
            suffix += 1
            key = f"{name}#{suffix}"
        used.add(key); state[bucket][key] = _dtype_name(value) or "unknown"

    def constants(name: str, value: Any) -> None:
        if _torch_tensor(torch, value):
            put("constants", name, value)
        elif isinstance(value, dict):
            for key, item in value.items():
                constants(f"{name}.{key}", item)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                constants(f"{name}[{index}]", item)

    for index, module in enumerate(_iter_modules(encoder)):
        prefix = f"module{index}:{type(module).__module__}.{type(module).__qualname__}"
        for collection_name, bucket in (("_parameters", "parameters"), ("_buffers", "buffers")):
            collection = getattr(module, collection_name, None)
            if isinstance(collection, dict):
                for name, value in collection.items():
                    put(bucket, f"{prefix}.{name}", value)
    for index, owner in enumerate(_owners(encoder)):
        for name in _CONSTANT_NAMES:
            if hasattr(owner, name):
                constants(f"owner{index}.{name}", getattr(owner, name))
    return state


@contextmanager
def _dispatch_observer(torch: Any, *, precision: str, evidence: dict[str, Any]) -> Iterator[None]:
    try:
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_flatten
    except ImportError as error:  # pragma: no cover - old torch builds
        evidence["dispatch_observed"] = False
        if precision == "temporary_fp32":
            raise EncoderPrecisionError("FP32 encoder dispatch observer unavailable") from error
        yield
        return

    evidence["dispatch_observed"] = True

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
    # Bind before execution so a keyword-only device parameter can be
    # supported without retrying an inference that already raised.
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method(value)
    try:
        signature.bind(value)
    except TypeError as first:
        try:
            signature.bind(value, device=getattr(value, "device", None))
        except TypeError:
            raise first
        return method(value, device=getattr(value, "device", None))
    return method(value)


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
        old_training = _training_items(list(_iter_modules(self.encoder)))
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
            evidence["state_dtypes"] = _state_dtype_evidence(self.encoder, torch)
            if precision == "temporary_fp32":
                floating_state = [dtype for bucket in evidence["state_dtypes"].values()
                                  for dtype in bucket.values()]
                if not floating_state or any(dtype != "float32" for dtype in floating_state):
                    raise EncoderPrecisionError("FP32 encoder state evidence is incomplete or non-FP32")
            if callable(original_inner_encode):
                setattr(inner, "encode", observed_inner_encode)
            with _backend_guard(torch, evidence), _dispatch_observer(torch, precision=precision, evidence=evidence):
                with torch.inference_mode():
                    encoded = _call_encoder(wrapper, torch.from_numpy(normalized.copy()).to(device=self.device, dtype=torch.float32))
            if precision == "temporary_fp32" and (
                not evidence.get("dispatch_observed") or evidence.get("operation_count", 0) <= 0
            ):
                raise EncoderPrecisionError("FP32 encoder dispatch observer produced no operation evidence")
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
            cleanup_errors: list[BaseException] = []
            if callable(original_inner_encode):
                try:
                    if had_instance_encode:
                        setattr(inner, "encode", original_inner_encode)
                    elif "encode" in getattr(inner, "__dict__", {}):
                        delattr(inner, "encode")
                except BaseException as error:
                    cleanup_errors.append(error)
            if snapshot is not None:
                try:
                    snapshot.restore()
                except BaseException as error:
                    cleanup_errors.append(error)
                try:
                    snapshot.verify_restored()
                except BaseException as error:
                    cleanup_errors.append(error)
            for module, training in old_training:
                try:
                    module.training = training
                except BaseException as error:
                    cleanup_errors.append(error)
            try:
                _clear_cache(self.encoder)
                evidence["cache_cleared_after"] = True
            except BaseException as error:
                cleanup_errors.append(error)
            if cleanup_errors:
                if primary_error is not None:
                    for error in cleanup_errors:
                        primary_error.add_note("encoder cleanup failed: " + repr(error))
                else:
                    failure = cleanup_errors[0]
                    for error in cleanup_errors[1:]:
                        failure.add_note("encoder cleanup failed: " + repr(error))
                    raise failure

    __call__ = encode


def encode_feedback_frame(encoder: Any, frame: Any, *, precision: str = "native",
                          device: str = "cuda") -> dict[str, Any]:
    """Encode one frame through a reusable :class:`FeedbackEncoder` seam."""
    return FeedbackEncoder(encoder, device=device).encode(frame, precision=precision)


__all__ = ["EncoderPrecisionError", "FeedbackEncoder", "encode_feedback_frame",
           "temporary_encoder_precision"]
