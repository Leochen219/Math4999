"""Post-VAE UMI bridge primitives.

The numerical and boundary helpers in this module deliberately have no Cosmos
or Torch import at module import time.  The remote runner can therefore test
the experiment contract on a CPU host while the small runtime adapter uses the
same functions around the official pipeline.
"""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def _numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Convert NumPy/Torch-like values without importing Torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float") and hasattr(value, "cpu"):
        value = value.float().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = _numpy(value)
    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return array


def rms(value: Any) -> float:
    array = _finite_array(value, "value").astype(np.float64, copy=False)
    if array.size == 0:
        raise ValueError("RMS is undefined for an empty tensor")
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def cosine(left: Any, right: Any) -> float | None:
    a = _finite_array(left, "left").astype(np.float64, copy=False).reshape(-1)
    b = _finite_array(right, "right").astype(np.float64, copy=False).reshape(-1)
    if a.size != b.size:
        raise ValueError("cosine operands must have equal size")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return None
    return float(np.dot(a, b) / denominator)


def sha256_array(array: Any) -> str:
    value = np.ascontiguousarray(_numpy(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _flatten_indexes(indexes: Any) -> list[int]:
    if indexes is None:
        return []
    if isinstance(indexes, (int, np.integer)):
        return [int(indexes)]
    flattened: list[int] = []
    try:
        for item in indexes:
            if isinstance(item, (list, tuple, np.ndarray)):
                flattened.extend(_flatten_indexes(item))
            else:
                flattened.append(int(item))
    except (TypeError, ValueError) as error:
        raise ValueError("condition indexes must be an integer sequence") from error
    return flattened


def _carrier_temporal_axis(carrier_shape: tuple[int, ...]) -> int:
    if len(carrier_shape) < 3:
        raise ValueError(f"carrier shape must have at least 3 dimensions, got {carrier_shape}")
    return len(carrier_shape) - 3


def build_broadcast_condition_mask(
    condition_indexes: Iterable[int], packed_vision_condition_mask: Any, carrier_shape: Iterable[int]
) -> np.ndarray:
    """Build a full latent mask from runtime indexes and packed vision mask.

    The packed mask may be ``[T,H,W]``/``[B,T,H,W]`` or already have carrier
    rank.  Its temporal occupancy must exactly agree with the runtime indexes;
    this catches stale hard-coded masks before any latent is changed.
    """

    shape = tuple(int(dim) for dim in carrier_shape)
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"carrier shape must be non-empty and positive, got {shape}")
    indexes = _flatten_indexes(condition_indexes)
    if not indexes or len(indexes) != len(set(indexes)):
        raise ValueError("condition indexes must be non-empty and unique")
    temporal_axis = _carrier_temporal_axis(shape)
    temporal_size = shape[temporal_axis]
    if any(index < 0 or index >= temporal_size for index in indexes):
        raise ValueError(f"condition index is outside temporal extent {temporal_size}: {indexes}")

    packed = _finite_array(packed_vision_condition_mask, "packed vision condition mask")
    if packed.dtype != np.bool_:
        if not np.all(np.logical_or(packed == 0, packed == 1)):
            raise ValueError("packed vision condition mask must be boolean")
        packed = packed.astype(bool)
    if packed.ndim == 1:
        expected_size = int(np.prod(shape, dtype=np.int64))
        if packed.size != expected_size:
            raise ValueError(f"flattened packed mask has {packed.size} values; expected {expected_size}")
        packed = packed.reshape(shape)
    # Remove a singleton channel dimension when callers retain packed layout.
    if packed.ndim == len(shape) and packed.shape != shape:
        singleton = [axis for axis, dim in enumerate(packed.shape) if dim == 1]
        if len(singleton) == 1:
            packed = np.squeeze(packed, axis=singleton[0])
    if packed.shape == shape:
        full = packed.astype(bool, copy=True)
        active = np.any(full, axis=tuple(axis for axis in range(full.ndim) if axis != temporal_axis))
    else:
        expected_packed_shapes = {shape[temporal_axis:]}
        # Common packed layouts are [T,H,W] and [B,T,H,W].
        if len(shape) == 5:
            expected_packed_shapes.add((shape[0], shape[2], shape[3], shape[4]))
            expected_packed_shapes.add((shape[2], shape[3], shape[4]))
        # Packed masks emitted by framework versions can retain singleton
        # spatial dimensions (for example [B,T,1,1]); accept those only when
        # NumPy can broadcast them to the runtime carrier layout.
        packed_is_broadcastable = packed.shape in expected_packed_shapes
        if len(shape) == 5 and packed.ndim == 4:
            packed_is_broadcastable = packed.shape[0] in (1, shape[0]) and packed.shape[1] == shape[2] and packed.shape[2] in (1, shape[3]) and packed.shape[3] in (1, shape[4])
        if len(shape) == 5 and packed.ndim == 3:
            packed_is_broadcastable = packed.shape[0] == shape[2] and packed.shape[1] in (1, shape[3]) and packed.shape[2] in (1, shape[4])
        if not packed_is_broadcastable:
            raise ValueError(f"packed mask shape {packed.shape} does not match carrier shape {shape}")
        if packed.ndim == len(shape) - 2:
            # [T,H,W] -> add batch/channel dimensions at the carrier axes.
            packed = packed.reshape((1,) * temporal_axis + packed.shape)
        elif len(shape) == 5 and packed.ndim == 4:
            packed = packed[:, None, ...]
        full = np.broadcast_to(packed, shape).copy()
        active = np.any(full, axis=tuple(axis for axis in range(full.ndim) if axis != temporal_axis))
    expected = np.zeros(temporal_size, dtype=bool)
    expected[indexes] = True
    if not np.array_equal(active, expected):
        raise ValueError("runtime condition indexes and packed mask do not agree")
    if not np.any(full):
        raise ValueError("condition mask is empty")
    if not np.all(np.isfinite(full)):
        raise ValueError("condition mask must be finite")
    return np.ascontiguousarray(full, dtype=bool)


def predicted_positions(condition_mask: Any) -> np.ndarray:
    mask = _finite_array(condition_mask, "condition mask").astype(bool, copy=False)
    if mask.size == 0 or not np.any(mask) or np.all(mask):
        raise ValueError("condition mask must contain both condition and predicted positions")
    return np.ascontiguousarray(~mask)


def generate_direction_bank(
    count: int, shape: Iterable[int], *, mask: Any, seed: int = 20260912
) -> np.ndarray:
    """Generate independent CPU Gaussian directions normalized only on mask."""

    shape_tuple = tuple(int(dim) for dim in shape)
    if int(count) <= 0:
        raise ValueError("count must be positive")
    if not shape_tuple or any(dim <= 0 for dim in shape_tuple):
        raise ValueError(f"shape must be positive, got {shape_tuple}")
    mask_array = _finite_array(mask, "mask").astype(bool, copy=False)
    if mask_array.shape != shape_tuple or not np.any(mask_array):
        raise ValueError("mask must be non-empty and match shape")
    generator = np.random.default_rng(int(seed))
    bank = np.zeros((int(count),) + shape_tuple, dtype=np.float32)
    hashes: set[str] = set()
    for index in range(int(count)):
        direction = generator.standard_normal(shape_tuple).astype(np.float32)
        direction[~mask_array] = 0.0
        direction_rms = rms(direction[mask_array])
        if not np.isfinite(direction_rms) or direction_rms == 0.0:
            raise ValueError("generated direction has invalid mask RMS")
        direction[mask_array] /= np.float32(direction_rms)
        digest = sha256_array(direction)
        if digest in hashes:
            raise ValueError("generated direction hashes are not unique")
        hashes.add(digest)
        bank[index] = direction
    return bank


def _bf16_round(value: Any) -> np.ndarray:
    """Round float32 to BF16 and return the representable float32 value."""

    source = _finite_array(value, "value").astype(np.float32, copy=True)
    bits = source.view(np.uint32)
    rounding = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = (bits + rounding) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def validate_cache_off(diffusion_cache_requested: Any, diffusion_cache_installed: Any) -> None:
    if bool(diffusion_cache_requested) or bool(diffusion_cache_installed):
        raise ValueError("diffusion cache must be disabled: requested and installed must both be false")


def validate_runtime_setup(runtime: dict[str, Any], *, expected_checkpoint: str | Path | None = None) -> None:
    required = {"checkpoint_path", "sampler", "precision", "diffusion_cache_requested", "diffusion_cache_installed"}
    missing = required.difference(runtime)
    if missing:
        raise ValueError(f"runtime setup is missing fields: {sorted(missing)}")
    if expected_checkpoint is not None and Path(runtime["checkpoint_path"]).resolve() != Path(expected_checkpoint).resolve():
        raise ValueError("resolved checkpoint path differs from requested checkpoint")
    if str(runtime["sampler"]).lower() != "unipc":
        raise ValueError(f"sampler must resolve to unipc, got {runtime['sampler']!r}")
    if "bfloat16" not in str(runtime["precision"]).lower():
        raise ValueError(f"precision must resolve to BF16, got {runtime['precision']!r}")
    validate_cache_off(runtime["diffusion_cache_requested"], runtime["diffusion_cache_installed"])


@dataclass(frozen=True)
class DeltaConstruction:
    latent: np.ndarray
    delta: np.ndarray
    s_z: float
    target_rms: float


def construct_delta(z0: Any, mask: Any, direction: Any, *, alpha: float, sign: int) -> DeltaConstruction:
    baseline = _finite_array(z0, "z0").astype(np.float32, copy=True)
    mask_array = _finite_array(mask, "mask").astype(bool, copy=False)
    vector = _finite_array(direction, "direction").astype(np.float32, copy=False)
    if baseline.shape != mask_array.shape or baseline.shape != vector.shape:
        raise ValueError("z0, mask, and direction must have identical shapes")
    if not np.any(mask_array):
        raise ValueError("condition mask is empty")
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and non-negative")
    if int(sign) not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    s_z = rms(baseline[mask_array])
    if not np.isfinite(s_z) or s_z == 0.0:
        raise ValueError("s_z must be finite and non-zero")
    delta = np.zeros_like(baseline, dtype=np.float32)
    delta[mask_array] = np.float32(int(sign) * alpha * s_z) * vector[mask_array]
    latent = baseline + delta
    return DeltaConstruction(latent=latent, delta=delta, s_z=s_z, target_rms=float(alpha * s_z))


def perturbation_metrics(
    z0: Any, delta: Any, *, mask: Any, direction: Any, alpha: float, cross_amplitude_delta: Any | None = None
) -> dict[str, Any]:
    baseline = _finite_array(z0, "z0").astype(np.float32, copy=False)
    change = _finite_array(delta, "delta").astype(np.float32, copy=False)
    mask_array = _finite_array(mask, "mask").astype(bool, copy=False)
    vector = _finite_array(direction, "direction").astype(np.float32, copy=False)
    if not (baseline.shape == change.shape == mask_array.shape == vector.shape):
        raise ValueError("z0, delta, mask, and direction must have identical shapes")
    if not np.any(mask_array):
        raise ValueError("mask is empty")
    target_rms = float(float(alpha) * rms(baseline[mask_array]))
    # ``change`` is the requested float32 delta.  The experiment reports the
    # realized perturbation after the baseline is added and rounded to the
    # model-visible dtype, because a tiny request can disappear at large
    # baseline magnitudes (for example 1e8 + 1.0 in float32).
    realized = ((baseline + change) - baseline).astype(np.float32, copy=False)
    actual_rms = rms(realized[mask_array])
    baseline_bf16 = _bf16_round(baseline)
    perturbed_bf16 = _bf16_round(baseline + change)
    actual_bf16 = (perturbed_bf16 - baseline_bf16).astype(np.float32)
    result: dict[str, Any] = {
        "target_rms_fp32": target_rms,
        "actual_rms_fp32": actual_rms,
        "actual_rms_bf16": rms(actual_bf16[mask_array]),
        "relative_amplitude": None if target_rms == 0.0 else actual_rms / target_rms,
        "relative_amplitude_bf16": None if target_rms == 0.0 else rms(actual_bf16[mask_array]) / target_rms,
        "nonzero_ratio": float(np.count_nonzero(realized[mask_array]) / realized[mask_array].size),
        "bf16_nonzero_ratio": float(np.count_nonzero(actual_bf16[mask_array]) / actual_bf16[mask_array].size),
        "direction_cosine": cosine(realized[mask_array], vector[mask_array]),
        "bf16_direction_cosine": cosine(actual_bf16[mask_array], vector[mask_array]),
        "outside_mask_exact": bool(np.all(change[~mask_array] == 0) and np.all(realized[~mask_array] == 0)),
        "bf16_erased_by_cast": bool(target_rms > 0.0 and np.all(actual_bf16[mask_array] == 0)),
        "actual_delta_bf16": actual_bf16,
    }
    if cross_amplitude_delta is not None:
        cross = _finite_array(cross_amplitude_delta, "cross-amplitude delta").astype(np.float32, copy=False)
        if cross.shape != change.shape:
            raise ValueError("cross-amplitude delta shape mismatch")
        cross_realized = ((baseline + cross) - baseline).astype(np.float32, copy=False)
        result["cross_amplitude_direction_cosine"] = cosine(realized[mask_array], cross_realized[mask_array])
        result["cross_amplitude_relative_rms"] = None if actual_rms == 0 else rms((realized - cross_realized)[mask_array]) / actual_rms
    result["actual_delta_fp32"] = realized
    return result


def hash_predicted_noisy_region(initial_state: Any, condition_mask: Any) -> str:
    state = _finite_array(initial_state, "initial sampler state")
    mask = _finite_array(condition_mask, "condition mask").astype(bool, copy=False)
    if state.shape != mask.shape:
        raise ValueError("initial sampler state and condition mask must have equal shapes")
    if not np.any(~mask):
        raise ValueError("predicted/noisy region is empty")
    return sha256_array(np.ascontiguousarray(state[~mask].astype(np.float32, copy=False)))


# Names used by the scan runner and retained as explicit vocabulary for the
# evidence files.  They are aliases rather than wrappers so there is one
# implementation of every numerical primitive.
predicted_region_hash = hash_predicted_noisy_region
broadcast_condition_mask = build_broadcast_condition_mask
generate_directions = generate_direction_bank


def capture_final_latent(final_latent: Any, condition_mask: Any) -> dict[str, np.ndarray]:
    latent = _finite_array(final_latent, "final latent").astype(np.float32, copy=True)
    mask = _finite_array(condition_mask, "condition mask").astype(bool, copy=False)
    if latent.shape != mask.shape:
        raise ValueError("final latent and condition mask must have equal shapes")
    if not np.any(~mask):
        raise ValueError("predicted latent region is empty")
    temporal_axis = _carrier_temporal_axis(latent.shape)
    non_temporal_axes = tuple(axis for axis in range(mask.ndim) if axis != temporal_axis)
    frame_active = np.any(mask, axis=non_temporal_axes)
    frame_mask_shape = [1] * mask.ndim
    frame_mask_shape[temporal_axis] = mask.shape[temporal_axis]
    framewise_mask = np.broadcast_to(frame_active.reshape(frame_mask_shape), mask.shape)
    if not np.array_equal(mask, framewise_mask):
        raise ValueError("condition mask must be framewise for final latent slicing")
    indexes = np.where(np.any(mask, axis=tuple(axis for axis in range(mask.ndim) if axis != temporal_axis)))[0]
    predicted_indexes = [index for index in range(latent.shape[temporal_axis]) if index not in set(indexes.tolist())]
    predicted = np.take(latent, predicted_indexes, axis=temporal_axis)
    return {
        "full": np.ascontiguousarray(latent),
        "predicted_only": np.ascontiguousarray(predicted),
        "condition_indexes": np.asarray(indexes, dtype=np.int64),
        "predicted_indexes": np.asarray(predicted_indexes, dtype=np.int64),
    }


capture_predicted_latent = capture_final_latent


class ConditionTokenCapture:
    def __init__(self, expected: Any, *, cast_dtype: Any = np.float32) -> None:
        self.expected = self._cast(expected, cast_dtype)
        self.cast_dtype = cast_dtype
        self.steps: list[tuple[int, np.ndarray]] = []

    def _cast(self, value: Any, dtype: Any | None = None) -> np.ndarray:
        array = _finite_array(value, "condition tokens")
        if dtype in ("bfloat16", "bf16"):
            return _bf16_round(array)
        if dtype is np.float32 or dtype is None:
            return array.astype(np.float32, copy=True)
        return array.astype(dtype, copy=True)

    def record(self, step: int, tokens: Any) -> None:
        observed = self._cast(tokens, self.cast_dtype)
        if observed.shape != self.expected.shape:
            raise ValueError("conditioned tokens shape changed")
        self.steps.append((int(step), observed))

    def assert_persistent(self) -> None:
        if len(self.steps) < 2:
            raise ValueError("conditioned token observations must include first and final steps")
        for step, observed in self.steps:
            if not np.array_equal(observed, self.expected):
                raise ValueError(f"conditioned tokens changed at step {step}")

    @property
    def step_hashes(self) -> dict[int, str]:
        return {step: sha256_array(value) for step, value in self.steps}


def _clone(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    return copy.deepcopy(value)


class CachedConditioningData:
    """Call the official encoder once and expose only fresh baseline clones."""

    def __init__(self, get_data_and_condition: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._original = get_data_and_condition
        self._args = args
        self._kwargs = kwargs
        self._baseline: Any | None = None
        self._captured = False

    def _capture(self) -> None:
        if not self._captured:
            self._baseline = _clone(self._original(*self._args, **self._kwargs))
            self._captured = True

    def get(self, *, delta: Any | None = None, inject: Callable[[Any, Any], Any] | None = None) -> Any:
        self._capture()
        result = _clone(self._baseline)
        if delta is not None:
            if inject is None:
                raise ValueError("an injection function is required for explicit override calls")
            result = inject(result, _clone(delta))
        return result

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def baseline(self) -> Any:
        self._capture()
        return _clone(self._baseline)


class HookBoundary(AbstractContextManager):
    """Install a boundary hook and restore the exact original on all exits."""

    def __init__(self, target: Any, attribute: str, replacement: Any) -> None:
        self.target = target
        self.attribute = attribute
        self.replacement = replacement
        self.original: Any = None

    def __enter__(self) -> Any:
        self.original = getattr(self.target, self.attribute)
        replacement = self.replacement() if callable(self.replacement) else self.replacement
        setattr(self.target, self.attribute, replacement)
        return self.target

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        setattr(self.target, self.attribute, self.original)
        return False


class PostVaeCapture:
    """Boundary-level capture state shared by the official hook wrappers."""

    def __init__(self) -> None:
        self.initial_state: np.ndarray | None = None
        self.condition_mask: np.ndarray | None = None
        self.initial_state_hash: str | None = None
        self.final_latent: dict[str, np.ndarray] | None = None
        self.decoded_full: np.ndarray | None = None
        self.decoded_final: np.ndarray | None = None
        self.tokens: ConditionTokenCapture | None = None

    def record_initial_state(self, state: Any, condition_mask: Any) -> str:
        initial = _finite_array(state, "initial sampler state").astype(np.float32, copy=True)
        mask = _finite_array(condition_mask, "condition mask").astype(bool, copy=True)
        if initial.shape != mask.shape:
            raise ValueError("initial sampler state and condition mask must have equal shapes")
        self.initial_state = initial
        self.condition_mask = mask
        self.initial_state_hash = hash_predicted_noisy_region(initial, mask)
        return self.initial_state_hash

    def record_final_latent(self, latent: Any) -> dict[str, np.ndarray]:
        if self.condition_mask is None:
            raise ValueError("record the initial state and condition mask before the final latent")
        self.final_latent = capture_final_latent(latent, self.condition_mask)
        return self.final_latent

    def record_decoded(self, output: Any, *, expected_frames: int = 17) -> tuple[np.ndarray, np.ndarray]:
        full, final = validate_decoded_output_array(output, expected_frames=expected_frames)
        self.decoded_full, self.decoded_final = full, final
        return full, final


def run_with_hooks(operation: Callable[[], Any], hooks: Iterable[HookBoundary]) -> Any:
    """Run one official call and restore every hook on success or failure."""

    from contextlib import ExitStack

    with ExitStack() as stack:
        for hook in hooks:
            stack.enter_context(hook)
        return operation()


def validate_condition_tokens(expected: Any, observed_steps: Iterable[Any], *, cast_dtype: Any = "bfloat16") -> dict[int, str]:
    capture = ConditionTokenCapture(expected, cast_dtype=cast_dtype)
    for step, tokens in enumerate(observed_steps):
        capture.record(step, tokens)
    capture.assert_persistent()
    return capture.step_hashes


def validate_decoded_output_array(output: Any, *, expected_frames: int = 17) -> tuple[np.ndarray, np.ndarray]:
    array = _numpy(output)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"decoded output must be floating-point, got {array.dtype}")
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4 or array.shape[0] != 3:
        raise ValueError(f"decoded output must have shape [3,T,H,W], got {array.shape}")
    if array.shape[1] != int(expected_frames):
        raise ValueError(f"decoded output must contain {expected_frames} frames, got {array.shape[1]}")
    if not np.all(np.isfinite(array)):
        raise ValueError("decoded output must contain only finite values")
    if float(array.min()) < -1e-6 or float(array.max()) > 1.0 + 1e-6:
        raise ValueError("decoded output must be in [0,1]")
    full = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return full, np.ascontiguousarray(full[:, -1])


def build_setup_overrides_kwargs(
    *, checkpoint_path: str | Path, output_dir: str | Path, parallelism_preset: str = "latency", guardrails: bool = False, diffusion_cache: bool = False
) -> dict[str, Any]:
    return {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "output_dir": Path(output_dir).resolve(),
        "parallelism_preset": parallelism_preset,
        "guardrails": bool(guardrails),
        "diffusion_cache": bool(diffusion_cache),
    }


__all__ = [
    "CachedConditioningData", "ConditionTokenCapture", "DeltaConstruction", "HookBoundary", "PostVaeCapture",
    "broadcast_condition_mask", "build_broadcast_condition_mask", "build_setup_overrides_kwargs",
    "capture_final_latent", "capture_predicted_latent", "construct_delta", "cosine", "generate_direction_bank",
    "generate_directions", "hash_predicted_noisy_region", "perturbation_metrics", "predicted_positions",
    "predicted_region_hash", "rms", "run_with_hooks", "sha256_array", "validate_cache_off",
    "validate_condition_tokens", "validate_decoded_output_array", "validate_runtime_setup",
]
