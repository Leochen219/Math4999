"""Content identities for actual tensors, inputs, config and loaded state."""
import hashlib
import inspect
import json
import math
import dataclasses
from pathlib import Path

import numpy as np


def file_identity(path):
    path = Path(path).resolve(strict=True)
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise ValueError("identity artifact directory is empty")
        return {"path": str(path), "files": {str(item.relative_to(path)): file_identity(item)["sha256"] for item in files}}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def source_identity(value):
    target = value if inspect.isfunction(value) or inspect.isclass(value) or inspect.ismethod(value) else type(value)
    try:
        source = inspect.getsourcefile(target)
    except TypeError:
        source = None
    result = {"type": target.__module__ + "." + target.__qualname__}
    if source:
        result["source"] = file_identity(source)
    return result


def module_tensors(module):
    """Complete compute tensor registry, including aliases/nonpersistent buffers."""
    return dict(sorted(
        [("parameter:" + name, value) for name, value in module.named_parameters(remove_duplicate=False)]
        + [("buffer:" + name, value) for name, value in module.named_buffers(remove_duplicate=False)]
    ))


def module_metadata(module):
    import torch
    modules = dict(module.named_modules(remove_duplicate=False))
    return {
        "training": {name: child.training for name, child in modules.items()},
        "extra_state": {name: child.get_extra_state() for name, child in modules.items()
                        if type(child).get_extra_state is not torch.nn.Module.get_extra_state},
    }


class _IdentityTraversal:
    """Deterministic graph traversal state for cycle/shared-object identity."""

    def __init__(self):
        self.seen = {}
        self.active = {}
        # Keep traversed containers alive so CPython cannot reuse an object id
        # for a later temporary dict (for example module tensors vs metadata).
        self.objects = {}

    def begin(self, value, path):
        marker = id(value)
        if marker in self.active:
            return {"relationship": "cycle", "node_ref": self.active[marker]}
        if marker in self.seen:
            return {"relationship": "shared", "node_ref": self.seen[marker]}
        self.seen[marker] = path
        self.active[marker] = path
        self.objects[marker] = value
        return None

    def end(self, value):
        self.active.pop(id(value), None)


def _reference_identity(value, reference):
    """Return a stable path-based marker for a cycle or shared object."""

    return {"source": source_identity(value), "reference": reference}


def _dict_path(path, key):
    return f"{path}[{json.dumps(str(key), ensure_ascii=True)}]"


def _set_order_key(value):
    """Canonical, address-independent ordering key for set members."""

    try:
        tree = content_tree(value, _IdentityTraversal(), "$set")
        return json.dumps(tree, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (RecursionError, TypeError, ValueError) as error:
        # A set has no semantic order. If a member cannot be represented
        # without process-specific state, fail closed instead of falling back
        # to repr(), which commonly embeds a memory address.
        raise TypeError("set member has no address-independent canonical identity") from error


def content_tree(value, _ctx=None, _path="$"):
    if _ctx is None:
        _ctx = _IdentityTraversal()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("nonfinite configuration value cannot be fingerprinted")
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray) or hasattr(value, "detach"):
        dtype = str(value.dtype)
        tensor = value.detach().cpu() if hasattr(value, "detach") else value
        try:
            array = tensor.numpy() if hasattr(tensor, "numpy") else tensor
        except TypeError:
            # BF16 has no NumPy dtype. Hash its native bytes, never an FP32
            # projection, so mixed precision and full-bit values remain distinct.
            import torch
            array = tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        array = np.asarray(array)
        shape = list(value.shape) if hasattr(value, "shape") else list(array.shape)
        return {"dtype": dtype, "shape": shape, "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}
    if isinstance(value, dict):
        reference = _ctx.begin(value, _path)
        if reference is not None:
            return _reference_identity(value, reference)
        try:
            return {str(key): content_tree(item, _ctx, _dict_path(_path, key)) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
        finally:
            _ctx.end(value)
    if isinstance(value, (list, tuple)):
        reference = _ctx.begin(value, _path)
        if reference is not None:
            return _reference_identity(value, reference)
        try:
            return [content_tree(item, _ctx, f"{_path}[{index}]") for index, item in enumerate(value)]
        finally:
            _ctx.end(value)
    if isinstance(value, (set, frozenset)):
        reference = _ctx.begin(value, _path)
        if reference is not None:
            return _reference_identity(value, reference)
        try:
            keyed = [(type(item).__module__, type(item).__qualname__, _set_order_key(item), item) for item in value]
            if len({key[:3] for key in keyed}) != len(keyed):
                # Equal canonical keys can still hide different shared/cycle
                # relationships once the enclosing set is traversed.  There
                # is no address-independent tie breaker, so fail closed.
                raise TypeError("set members have ambiguous canonical ordering")
            ordered = [item for _, _, _, item in sorted(keyed, key=lambda row: row[:3])]
            return [content_tree(item, _ctx, f"{_path}[{index}]") for index, item in enumerate(ordered)]
        finally:
            _ctx.end(value)
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value):
        return source_identity(value)
    if hasattr(value, "named_parameters") and hasattr(value, "named_buffers"):
        reference = _ctx.begin(value, _path)
        if reference is not None:
            return _reference_identity(value, reference)
        try:
            return {"source": source_identity(value), "tensors": content_tree(module_tensors(value), _ctx, f"{_path}.tensors"),
                    "metadata": content_tree(module_metadata(value), _ctx, f"{_path}.metadata")}
        finally:
            _ctx.end(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        reference = _ctx.begin(value, _path)
        if reference is not None:
            return _reference_identity(value, reference)
        try:
            fields = {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
            return {"source": source_identity(value), "fields": content_tree(fields, _ctx, f"{_path}.fields")}
        finally:
            _ctx.end(value)
    if hasattr(value, "state_dict"):
        reference = _ctx.begin(value, _path)
        if reference is not None:
            return _reference_identity(value, reference)
        try:
            training = {name: getattr(module, "training", None) for name, module in value.named_modules()}
            return {"source": source_identity(value), "state": content_tree(value.state_dict(), _ctx, f"{_path}.state"), "training": training}
        finally:
            _ctx.end(value)
    if type(value).__module__ == "torch" and type(value).__name__ in ("dtype", "device", "layout", "memory_format"):
        return str(value)
    if hasattr(value, "__dict__"):
        reference = _ctx.begin(value, _path)
        if reference is not None:
            return _reference_identity(value, reference)
        try:
            return {"source": source_identity(value), "attributes": content_tree(vars(value), _ctx, f"{_path}.attributes")}
        finally:
            _ctx.end(value)
    raise TypeError(f"cannot establish actual content identity for {type(value)}")


def fingerprint(value):
    canonical = json.dumps(content_tree(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
