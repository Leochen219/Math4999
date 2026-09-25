"""Task 8 runtime seams and the CPU-safe 256x256 input adapter.

The module deliberately does not import Torch or the Cosmos framework.  A live
caller supplies one already-loaded ``OfficialPrecisionRuntime`` and one
``FeedbackEncoder`` after the independent preflight/release gate.  The small
pure-Python seams here are also used by the CPU contract tests, which must
never imply model or GPU execution.
"""
from __future__ import annotations

import hashlib
import json
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np


class Task8RuntimeError(RuntimeError):
    """A required Task 8 runtime seam or evidence field is unavailable."""


class ActionRoutingError(Task8RuntimeError):
    """The prepared action did not reach the denoiser's packed action tokens."""


def _as_f32(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and hasattr(value, "cpu"):
        try:
            value = value.float().cpu().numpy()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    result = np.asarray(value, dtype=np.float32)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and non-empty")
    return np.ascontiguousarray(result, dtype=np.float32)


def _array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")); digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def action_token_hash(action: Any) -> str:
    """Hash the exact FP32 ``packed.action.tokens`` value consumed by G."""
    array = _as_f32(action, name="action tokens")
    return _array_hash(array)


@dataclass(frozen=True)
class Task8InputAdapter:
    """Validated NPZ input: 33 RGB observations and two normalized chunks."""

    rgb: np.ndarray
    actions: np.ndarray
    prompt: str
    provenance: Mapping[str, Any]
    raw_actions: np.ndarray | None = None

    def __post_init__(self) -> None:
        rgb = _as_f32(self.rgb, name="rgb")
        actions = _as_f32(self.actions, name="actions")
        if rgb.shape != (33, 3, 256, 256):
            raise ValueError(f"rgb must be float32 [33,3,256,256], got {rgb.shape}")
        if actions.shape != (2, 16, 10):
            raise ValueError(f"actions must be float32 [2,16,10], got {actions.shape}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(self.provenance, Mapping):
            raise ValueError("provenance must be a JSON object")
        raw = None if self.raw_actions is None else _as_f32(self.raw_actions, name="raw_actions")
        if raw is not None and raw.shape != actions.shape:
            raise ValueError("raw_actions must have the same [2,16,10] shape as actions")
        object.__setattr__(self, "rgb", rgb.copy())
        object.__setattr__(self, "actions", actions.copy())
        object.__setattr__(self, "provenance", json.loads(json.dumps(dict(self.provenance), sort_keys=True)))
        if raw is not None:
            object.__setattr__(self, "raw_actions", raw.copy())
        for value in (self.rgb, self.actions, self.raw_actions):
            if value is not None:
                value.setflags(write=False)

    @classmethod
    def from_npz(cls, path: str | Path, metadata_path: str | Path | None = None) -> "Task8InputAdapter":
        """Load the admitted preflight bundle without guessing representations.

        The production preflight contract stores ``rgb_float32`` and two flat
        32-step action arrays in ``inputs.npz``; language/provenance live in
        the companion ``task8_preflight.json``.  Reshaping is allowed only for
        the exact 32x10 representation.  The older inline-field form remains
        accepted for detached CPU fixtures, but production callers should use
        :meth:`from_preflight`, which requires the companion JSON explicitly.
        """
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            names = set(data.files)
            if {"rgb", "actions", "prompt", "provenance_json"}.issubset(names):
                prompt_value = data["prompt"]
                prompt = str(prompt_value.item() if prompt_value.ndim == 0 else prompt_value.tolist())
                provenance_value = data["provenance_json"]
                raw_provenance = provenance_value.item() if provenance_value.ndim == 0 else provenance_value.tolist()
                try:
                    provenance = json.loads(str(raw_provenance))
                except (TypeError, json.JSONDecodeError) as error:
                    raise ValueError("provenance_json is not valid JSON") from error
                raw_actions = data["raw_actions"] if "raw_actions" in names else None
                return cls(data["rgb"], data["actions"], prompt, provenance, raw_actions)
            required = {"rgb_float32", "actions_normalized", "actions_raw"}
            missing = sorted(required - names)
            if missing:
                raise ValueError(f"Task 8 preflight NPZ missing fields: {missing}")
            if metadata_path is None:
                raise ValueError("preflight NPZ requires companion task8_preflight.json")
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            if not isinstance(metadata, Mapping):
                raise ValueError("preflight metadata must be a JSON object")
            selected = metadata.get("selected", {})
            if not isinstance(selected, Mapping):
                raise ValueError("preflight metadata lacks selected provenance")
            prompt = selected.get("language", metadata.get("language"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("preflight metadata lacks selected.language")
            normalized = np.asarray(data["actions_normalized"])
            raw = np.asarray(data["actions_raw"])
            if normalized.shape == (32, 10):
                normalized = normalized.reshape(2, 16, 10)
            elif normalized.shape != (2, 16, 10):
                raise ValueError(f"actions_normalized must be exactly [32,10] or [2,16,10], got {normalized.shape}")
            if raw.shape == (32, 10):
                raw = raw.reshape(2, 16, 10)
            elif raw.shape != (2, 16, 10):
                raise ValueError(f"actions_raw must be exactly [32,10] or [2,16,10], got {raw.shape}")
            provenance = {"metadata_path": str(Path(metadata_path).resolve()), "selected": dict(selected),
                          "source_provenance": metadata.get("source_provenance", [])}
            return cls(data["rgb_float32"], normalized, prompt, provenance, raw)

    @classmethod
    def from_preflight(cls, npz_path: str | Path, metadata_path: str | Path) -> "Task8InputAdapter":
        """Strict production loader for the preflight NPZ plus report JSON."""
        return cls.from_npz(npz_path, metadata_path=metadata_path)


load_task8_batch = Task8InputAdapter.from_npz


def _replacement(parent: Any, key: Any, value: Any, *, ops: Any | None) -> Any:
    old_value = parent[key] if isinstance(parent, (Mapping, list, tuple)) else getattr(parent, key)
    if isinstance(old_value, (list, tuple)):
        if len(old_value) != 1:
            raise ActionRoutingError("Task 8 requires one batch-1 action token payload")
        if ops is not None and callable(getattr(ops, "from_array", None)):
            try:
                return [ops.from_array(value.copy(), old_value[0], dtype="float32")]
            except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                pass
        return [value.copy()]
    if ops is not None and callable(getattr(ops, "from_array", None)):
        try:
            return ops.from_array(value.copy(), old_value, dtype="float32")
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            pass
    return value.copy()


def _fit_action_to_template(values: np.ndarray, template: Any) -> np.ndarray:
    expected = _as_f32(template, name="action token template")
    if expected.shape == values.shape:
        return values.copy()
    if (expected.ndim == values.ndim == 2 and expected.shape[0] == values.shape[0]
            and expected.shape[1] >= values.shape[1]):
        padded = np.zeros(expected.shape, dtype=np.float32)
        padded[:, :values.shape[1]] = values
        return padded
    raise ActionRoutingError(f"normalized action shape {values.shape} cannot map to official token shape {expected.shape}")


def _set_child(parent: Any, key: Any, value: Any) -> None:
    if isinstance(parent, MutableMapping):
        parent[key] = value
    elif isinstance(parent, list):
        parent[key] = value
    elif isinstance(parent, tuple):
        raise ActionRoutingError("packed action tokens are inside an immutable tuple")
    else:
        setattr(parent, key, value)


def _iter_token_refs(root: Any, *, path: str = "", seen: set[int] | None = None):
    """Yield parent/key pairs for action token leaves in dicts or runtime objects."""
    seen = set() if seen is None else seen
    identity = id(root)
    if identity in seen or root is None or isinstance(root, (str, bytes, numbers.Number, np.ndarray)):
        return
    seen.add(identity)
    if isinstance(root, Mapping):
        for key, child in root.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).lower() in {"tokens", "action_tokens"} and "action" in path.lower():
                yield root, key, child_path
            else:
                yield from _iter_token_refs(child, path=child_path, seen=seen)
        return
    if isinstance(root, (list, tuple)):
        for index, child in enumerate(root):
            yield from _iter_token_refs(child, path=f"{path}[{index}]", seen=seen)
        return
    for key in ("packed", "packed_sequence", "sequence", "action", "tokens", "action_tokens", "packed_action"):
        if hasattr(root, key):
            child = getattr(root, key)
            child_path = f"{path}.{key}" if path else key
            if key in {"tokens", "action_tokens"} and "action" in path.lower():
                yield root, key, child_path
            else:
                yield from _iter_token_refs(child, path=child_path, seen=seen)


def refresh_prepared_action(prepared: Any, action: Any, *, ops: Any | None = None) -> dict[str, Any]:
    """Write one chunk into every actual packed action-token slot.

    Official preparation caches a packed action carrier.  Mutating an outer
    input array alone is therefore insufficient; this helper updates the
    prepared clone itself and returns the resulting token hash/path evidence.
    """
    values = _as_f32(action, name="normalized action")
    refs = list(_iter_token_refs(prepared))
    if not refs:
        raise ActionRoutingError("prepared state exposes no packed.action.tokens slot")
    paths: list[str] = []
    effective_values: np.ndarray | None = None
    for parent, key, path in refs:
        old = parent[key] if isinstance(parent, (Mapping, list, tuple)) else getattr(parent, key)
        template = old[0] if isinstance(old, (list, tuple)) and len(old) == 1 else old
        fitted = _fit_action_to_template(values, template)
        if effective_values is None:
            effective_values = fitted
        elif not np.array_equal(effective_values, fitted):
            raise ActionRoutingError("packed action token templates disagree in shape")
        _set_child(parent, key, _replacement(parent, key, fitted, ops=ops))
        paths.append(path)
    # Read back every slot; this proves the mutation was not a stale exterior.
    hashes = []
    for parent, key, _ in refs:
        current = parent[key] if isinstance(parent, (Mapping, list, tuple)) else getattr(parent, key)
        payload = current[0] if isinstance(current, (list, tuple)) and len(current) == 1 else current
        current_array = _as_f32(payload, name="prepared action tokens")
        if effective_values is None or current_array.shape != effective_values.shape or not np.array_equal(current_array, effective_values):
            raise ActionRoutingError("prepared action token readback differs from requested chunk")
        hashes.append(_array_hash(current_array))
    assert effective_values is not None
    return {"action_hash": action_token_hash(values), "effective_action_hash": action_token_hash(effective_values),
            "effective_action": effective_values.copy(), "token_hash": hashes[0],
            "token_hashes": hashes, "paths": paths, "shape": list(values.shape), "dtype": "float32"}


def refresh_official_prepared_action(prepared: Any, action: Any, *, ops: Any | None = None) -> dict[str, Any]:
    """Refresh GenerationDataClean FP32 action fields before official packing."""
    values = _as_f32(action, name="normalized action")
    try:
        clean = prepared[1]
    except (IndexError, KeyError, TypeError):
        clean = getattr(prepared, "clean", None)
    if clean is None:
        raise ActionRoutingError("official prepared state has no GenerationDataClean entry")
    existing = getattr(clean, "x0_tokens_action", None)
    template = existing[0] if isinstance(existing, (list, tuple)) and existing else None
    if template is None:
        existing = getattr(clean, "raw_state_action", None)
        template = existing[0] if isinstance(existing, (list, tuple)) and existing else None
    if template is None:
        raise ActionRoutingError("official prepared state has no action tensor template")
    effective = _fit_action_to_template(values, template)
    if ops is not None and callable(getattr(ops, "from_array", None)):
        payload = ops.from_array(effective.copy(), template, dtype="float32")
    else:
        payload = effective.copy()
    clean.x0_tokens_action = [payload]
    clean.raw_state_action = [payload]
    evidence = {"action_hash": action_token_hash(values), "effective_action_hash": action_token_hash(effective),
                "effective_action": effective.copy(), "clean_action_shape": list(effective.shape),
                "clean_action_dtype": str(getattr(payload, "dtype", "float32")),
                "clean_action_fields": ["raw_state_action", "x0_tokens_action"]}
    try:
        packed = refresh_prepared_action(prepared, values, ops=ops)
    except ActionRoutingError:
        packed = None
    if packed is not None:
        evidence.update({"token_hash": packed["token_hash"], "token_hashes": packed["token_hashes"],
                         "effective_action": packed["effective_action"], "effective_action_hash": packed["effective_action_hash"],
                         "packed_action_paths": packed["paths"]})
    return evidence


def _hash_sequence(value: Any) -> list[str]:
    if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
        if value.ndim >= 1 and value.shape[0] == 30:
            return [_array_hash(item.astype(np.float32, copy=False)) for item in value]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            else:
                result.append(_array_hash(_as_f32(item, name="action token")))
        return result
    return []


def validate_action_consumption(generation: Mapping[str, Any], action: Any, *, chunk_index: int) -> dict[str, Any]:
    """Require the denoiser to report matching packed action tokens at all 30 steps."""
    if isinstance(chunk_index, bool) or int(chunk_index) not in (0, 1):
        raise ValueError("chunk_index must be 0 or 1")
    expected = action_token_hash(action)
    candidates: list[Any] = []
    for key in ("packed_action_token_hashes", "action_token_hashes", "denoiser_action_token_hashes"):
        if key in generation:
            candidates.append(generation[key])
    nested = generation.get("action_evidence") if isinstance(generation, Mapping) else None
    if isinstance(nested, Mapping):
        for key in ("packed_action_token_hashes", "token_hashes", "action_token_hashes"):
            if key in nested:
                candidates.append(nested[key])
    if not candidates:
        raise ActionRoutingError("generation lacks actual packed action token hashes")
    hashes = _hash_sequence(candidates[0])
    if len(hashes) != 30:
        raise ActionRoutingError(f"expected 30 denoiser action-token hashes, got {len(hashes)}")
    if any(value != expected for value in hashes):
        raise ActionRoutingError("denoiser consumed an action token hash different from the requested chunk")
    return {"chunk_index": int(chunk_index), "expected_token_hash": expected,
            "consumed_token_hashes": list(hashes), "steps": 30, "all_steps_match": True}


@dataclass(frozen=True)
class Task8RuntimeConfig:
    """The single-runtime precision/sampler contract used by live callers."""

    device: str = "cuda:0"
    model_dtype: str = "float32"
    generator_dtype: str = "float32"
    denoiser_dtype: str = "float32"
    encoder_dtype: str = "float32"
    decoder_dtype: str = "float32"
    sampler: str = "UniPC"
    steps: int = 30
    guidance: float = 1.0
    shift: float = 10.0
    cache: bool = False
    autocast: bool = False
    tf32: bool = False
    batch_size: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {"device": self.device, "model_dtype": self.model_dtype,
                "generator_dtype": self.generator_dtype, "denoiser_dtype": self.denoiser_dtype,
                "encoder_dtype": self.encoder_dtype, "decoder_dtype": self.decoder_dtype,
                "sampler": self.sampler, "steps": self.steps, "guidance": self.guidance,
                "shift": self.shift, "cache": self.cache, "autocast": self.autocast,
                "tf32": self.tf32, "batch_size": self.batch_size}


class Task8Runtime:
    """Thin request adapter around one loaded official runtime.

    It owns no model state and never constructs a second runtime.  Its main
    responsibility is to route ``a0``/``a1`` into the request-local prepared
    state and to require denoiser-side action-token evidence before a record
    can be published.
    """

    def __init__(self, runtime: Any, batch: Task8InputAdapter, *, ops: Any | None = None,
                 config: Task8RuntimeConfig | None = None):
        if runtime is None or not callable(getattr(runtime, "execute", None)):
            raise Task8RuntimeError("one loaded OfficialPrecisionRuntime.execute is required")
        if not isinstance(batch, Task8InputAdapter):
            raise TypeError("batch must be a Task8InputAdapter")
        self.runtime, self.batch, self.ops = runtime, batch, ops or getattr(runtime, "ops", None)
        self.config = config or Task8RuntimeConfig()
        if self.config.batch_size != 1 or self.config.steps != 30 or self.config.model_dtype != "float32":
            raise ValueError("Task 8 requires batch=1, 30 steps, and FP32 model path")

    def action_for_chunk(self, chunk_index: int) -> np.ndarray:
        if isinstance(chunk_index, bool) or int(chunk_index) not in (0, 1):
            raise ValueError("chunk_index must be 0 or 1")
        return np.array(self.batch.actions[int(chunk_index)], copy=True)

    def prepare_action(self, prepared: Any, chunk_index: int) -> dict[str, Any]:
        return refresh_official_prepared_action(prepared, self.action_for_chunk(chunk_index), ops=self.ops)

    def validate_generation(self, generation: Mapping[str, Any], chunk_index: int) -> dict[str, Any]:
        return validate_action_consumption(generation, self.action_for_chunk(chunk_index), chunk_index=chunk_index)

    def execute_call(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        """Execute one request through the resident official full/deferred seam.

        ``condition_full`` must already be an FP32 encoded full carrier from
        the approved Task6/Task7 bridge.  Encoding is intentionally outside
        this method so G0/TF2/AR2 can record the exact FP32 encoder boundary
        and the caller can reuse G0's encoded last frame for AR2.
        """
        target = _as_f32(spec.get("condition_full"), name="condition_full")
        action = self.action_for_chunk(int(spec["chunk_index"]))
        inputs = getattr(self.runtime, "inputs", None)
        geometry = getattr(inputs, "geometry", None)
        mask = getattr(geometry, "mask", None)
        if mask is None or tuple(np.asarray(mask).shape) != tuple(target.shape):
            raise Task8RuntimeError("official runtime geometry mask does not match condition_full")
        try:
            from .umi_task7_runtime import _DynamicInputs, _seeded_prepared, _as_full_shape
            from .umi_fd_post_vae_scan import _clone_runtime
        except ImportError:  # pragma: no cover
            from umi_task7_runtime import _DynamicInputs, _seeded_prepared, _as_full_shape
            from umi_fd_post_vae_scan import _clone_runtime
        original_prepare = getattr(self.runtime, "_prepared_for_call", None)
        if not callable(original_prepare):
            raise Task8RuntimeError("official runtime lacks request-local preparation seam")
        original_inputs = getattr(self.runtime, "inputs", None)
        original_model_seed = getattr(self.runtime, "model_seed", None)
        original_pair_hash = getattr(self.runtime, "_paired_noise_hash", None)
        seed = int(spec["seed"])
        prepared_evidence: dict[str, Any] = {}
        action_step_hashes: list[str] = []
        model = getattr(self.runtime, "model", None)
        original_denoise = getattr(model, "denoise", None) if model is not None else None
        had_denoise = bool(model is not None and "denoise" in getattr(model, "__dict__", {}))
        try:
            def prepare(request_target: Any) -> Any:
                prepared, observed = _seeded_prepared(self.runtime, _as_full_shape(request_target, target.shape, name="prepared target"), np.asarray(mask, dtype=bool), seed)
                prepared_evidence.clear(); prepared_evidence.update(observed)
                prepared_evidence.update(self.prepare_action(prepared, int(spec["chunk_index"])))
                return prepared
            self.runtime._prepared_for_call = prepare
            self.runtime.model_seed = seed
            dynamic = _DynamicInputs(inputs, target)
            self.runtime.inputs = dynamic
            if model is not None and callable(original_denoise):
                def observed_denoise(*args: Any, **kwargs: Any) -> Any:
                    packed = kwargs.get("data_batch_packed")
                    if packed is None:
                        packed = next((value for value in args if getattr(value, "action", None) is not None), None)
                    action_modality = getattr(packed, "action", None)
                    tokens = getattr(action_modality, "tokens", None)
                    if isinstance(tokens, (list, tuple)) and len(tokens) == 1:
                        tokens = tokens[0]
                    if tokens is None:
                        raise ActionRoutingError("actual denoiser packed.action.tokens were not observed")
                    action_step_hashes.append(_array_hash(_as_f32(tokens, name="denoiser action tokens")))
                    return original_denoise(*args, **kwargs)
                model.denoise = observed_denoise
            call_spec = dict(spec)
            call_spec.update({"group": "C", "kind": "baseline", "alpha": 0.0, "sign": 0,
                              "state": call_spec.get("state", "bridge_0"), "seed": seed, "model_seed": seed})
            self.runtime._paired_noise_hash = None
            generation = dict(self.runtime.execute(call_spec, dynamic, scope="full", decode_policy="deferred"))
            generation["packed_action_token_hashes"] = list(action_step_hashes)
            effective_action = prepared_evidence.get("effective_action", action)
            action_evidence = validate_action_consumption(generation, effective_action, chunk_index=int(spec["chunk_index"]))
            generation["action_consumption"] = action_evidence
            generation["prepared_action"] = prepared_evidence
            return generation
        finally:
            if model is not None and callable(original_denoise):
                if had_denoise:
                    model.denoise = original_denoise
                elif "denoise" in getattr(model, "__dict__", {}):
                    delattr(model, "denoise")
            self.runtime.inputs = original_inputs
            if original_model_seed is not None:
                self.runtime.model_seed = original_model_seed
            if hasattr(self.runtime, "_paired_noise_hash"):
                self.runtime._paired_noise_hash = original_pair_hash
            if original_prepare is not None:
                self.runtime._prepared_for_call = original_prepare


GroundTruthRuntime = Task8Runtime


__all__ = [
    "ActionRoutingError", "GroundTruthRuntime", "Task8InputAdapter", "Task8Runtime", "Task8RuntimeConfig", "Task8RuntimeError",
    "action_token_hash", "load_task8_batch", "refresh_official_prepared_action",
    "refresh_prepared_action", "validate_action_consumption",
]
