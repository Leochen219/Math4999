"""Shared, audit-friendly primitives for the UMI precision contrast.

This module is intentionally CPU/NumPy only.  It is the common numerical
boundary used by the runner and the old-run reanalysis path: masks are derived
from runtime geometry, all comparisons are made on explicit FP32 values, and
differences are promoted to FP64 before subtraction.  No Cosmos or Torch
import is required at import time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

try:  # package import
    from .umi_fd_post_vae_bridge import build_broadcast_condition_mask, sha256_array
except ImportError:  # direct import from experiments/
    from umi_fd_post_vae_bridge import build_broadcast_condition_mask, sha256_array


def _array(value: Any, *, dtype: Any | None = None, name: str = "value") -> np.ndarray:
    """Detach tensor-like values without importing a framework."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except (AttributeError, TypeError, ValueError):
            pass
    result = np.asarray(value, dtype=dtype)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be non-empty and finite")
    return result


def _fp32(value: Any, *, name: str = "value") -> np.ndarray:
    return np.ascontiguousarray(_array(value, dtype=np.float32, name=name))


def _bf16_round(value: Any) -> np.ndarray:
    """Return the exact FP32 representation of round-to-nearest-even BF16."""

    source = _fp32(value)
    bits = source.view(np.uint32)
    rounding = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = (bits + rounding) & np.uint32(0xFFFF0000)
    return np.ascontiguousarray(rounded.view(np.float32))


def quantize_bf16_fp32(value: Any) -> np.ndarray:
    """Quantize an input through BF16, represented again as FP32.

    Keeping the result in FP32 makes bytewise A/B identity checks independent
    of whether a later runtime cast uses Torch, JAX, or another framework.
    """

    return _bf16_round(value)


quantize_fp32_to_bf16 = quantize_bf16_fp32


def effective_quantized_delta(baseline_fp32: Any, target_delta_fp32: Any) -> np.ndarray:
    """Compute the model-visible delta ``Q(x + d) - Q(x)`` in FP32."""

    baseline = _fp32(baseline_fp32, name="baseline_fp32")
    delta = _fp32(target_delta_fp32, name="target_delta_fp32")
    if baseline.shape != delta.shape:
        raise ValueError("baseline and target delta must have identical shapes")
    return np.ascontiguousarray((_bf16_round(baseline + delta) - _bf16_round(baseline)).astype(np.float32))


@dataclass(frozen=True)
class MaskGeometry:
    """Resolved runtime mask geometry and authoritative temporal slices."""

    carrier_shape: tuple[int, ...]
    temporal_axis: int
    condition_indexes: tuple[int, ...]
    predicted_indexes: tuple[int, ...]
    mask: np.ndarray

    def metadata(self) -> dict[str, Any]:
        return {
            "carrier_shape": list(self.carrier_shape),
            "temporal_axis": self.temporal_axis,
            "condition_indexes": list(self.condition_indexes),
            "predicted_indexes": list(self.predicted_indexes),
            "mask_sha256": sha256_array(self.mask),
        }


def mask_geometry(
    condition_indexes: Iterable[int],
    packed_vision_condition_mask: Any,
    carrier_shape: Iterable[int],
) -> MaskGeometry:
    """Resolve mask geometry from runtime indexes and the packed mask.

    ``build_broadcast_condition_mask`` performs the strict shape/layout and
    runtime-index agreement checks.  This wrapper makes the resulting temporal
    slices explicit so downstream code cannot silently use a hard-coded mask.
    """

    shape = tuple(int(value) for value in carrier_shape)
    if isinstance(condition_indexes, (int, np.integer)):
        raw_indexes = (int(condition_indexes),)
    else:
        raw_indexes = tuple(int(value) for value in condition_indexes)
    mask = np.ascontiguousarray(
        build_broadcast_condition_mask(raw_indexes, packed_vision_condition_mask, shape),
        dtype=bool,
    )
    temporal_axis = len(shape) - 3
    indexes = tuple(sorted(set(raw_indexes)))
    if len(indexes) != len(raw_indexes):
        # Repeated indexes are rejected by the underlying builder, but keeping
        # this guard here gives callers the same explicit geometry contract.
        raise ValueError("condition indexes must be unique")
    predicted = tuple(index for index in range(shape[temporal_axis]) if index not in indexes)
    if not predicted:
        raise ValueError("predicted temporal region is empty")
    return MaskGeometry(shape, temporal_axis, indexes, predicted, mask)


resolve_mask_geometry = mask_geometry


def slice_predicted_output(
    output: Any,
    condition_mask_or_geometry: Any,
    *,
    temporal_axis: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Slice the predicted temporal frames from a raw final tensor.

    The condition mask must be framewise: each temporal frame is either fully
    conditioned or fully predicted.  Returning metadata alongside the slice
    records the exact source shape and indexes used for audit/reanalysis.
    """

    if isinstance(condition_mask_or_geometry, MaskGeometry):
        geometry = condition_mask_or_geometry
        mask = geometry.mask
        axis = geometry.temporal_axis if temporal_axis is None else int(temporal_axis)
        indexes = geometry.predicted_indexes
    else:
        mask = np.asarray(condition_mask_or_geometry, dtype=bool)
        if mask.size == 0:
            raise ValueError("condition mask is empty")
        axis = len(mask.shape) - 3 if temporal_axis is None else int(temporal_axis)
        axis = axis if axis >= 0 else mask.ndim + axis
        if axis < 0 or axis >= mask.ndim:
            raise ValueError("temporal axis is outside mask rank")
        active = np.any(mask, axis=tuple(index for index in range(mask.ndim) if index != axis))
        inactive = np.all(mask, axis=tuple(index for index in range(mask.ndim) if index != axis))
        if not np.array_equal(active, inactive):
            raise ValueError("condition mask must be framewise")
        indexes = tuple(index for index, value in enumerate(active.tolist()) if not value)
        if not indexes:
            raise ValueError("predicted temporal region is empty")
    source = _fp32(output, name="output")
    if source.shape != mask.shape:
        raise ValueError("output and condition mask must have identical shapes")
    # Even for a MaskGeometry, re-check framewise layout at the final boundary.
    active = np.any(mask, axis=tuple(index for index in range(mask.ndim) if index != axis))
    inactive = np.all(mask, axis=tuple(index for index in range(mask.ndim) if index != axis))
    if not np.array_equal(active, inactive):
        raise ValueError("condition mask must be framewise")
    predicted = tuple(index for index, value in enumerate(active.tolist()) if not value)
    if predicted != tuple(indexes):
        raise ValueError("predicted indexes do not match the authoritative mask")
    selected = np.ascontiguousarray(np.take(source, predicted, axis=axis))
    metadata = {
        "axis": axis,
        "source_shape": list(source.shape),
        "selected_shape": list(selected.shape),
        "condition_indexes": [index for index, value in enumerate(active.tolist()) if value],
        "predicted_indexes": list(predicted),
        "mask_sha256": sha256_array(mask),
    }
    return selected, metadata


slice_predicted_latent = slice_predicted_output


def _identity_result(left: Any, right: Any, *, label: str, require_zero: bool = False) -> dict[str, Any]:
    try:
        a = _fp32(left, name=f"{label} left")
        b = _fp32(right, name=f"{label} right")
    except (TypeError, ValueError) as error:
        return {"passed": False, "label": label, "reason": str(error), "left_sha256": None, "right_sha256": None}
    if a.shape != b.shape:
        return {
            "passed": False,
            "label": label,
            "reason": "shape mismatch",
            "left_sha256": sha256_array(a),
            "right_sha256": sha256_array(b),
        }
    equal = a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")
    if require_zero and (not np.all(a == 0) or not np.all(b == 0)):
        equal = False
        reason = "zero-input identity requires both inputs to be exactly zero"
    else:
        reason = None if equal else "FP32 input values differ"
    return {
        "passed": bool(equal),
        "label": label,
        "reason": reason,
        "common_dtype": "float32",
        "left_sha256": sha256_array(a),
        "right_sha256": sha256_array(b),
    }


def validate_ab_exact_input(left: Any, right: Any) -> dict[str, Any]:
    """Validate A/B identity at the shared, quantized FP32 interface."""

    return _identity_result(left, right, label="A/B exact quantized input")


def validate_bc_zero_input(left: Any, right: Any) -> dict[str, Any]:
    """Validate that B/C share the exact all-zero input interface."""

    return _identity_result(left, right, label="B/C zero input", require_zero=True)


def build_precision_contract(
    common_input_fp32: Any | None = None,
    *,
    interface_tensors: Mapping[str, Any] | None = None,
    a_interface_fp32: Any | None = None,
    b_interface_fp32: Any | None = None,
    c_interface_fp32: Any | None = None,
    network_dtypes: Mapping[str, str] | None = None,
    quantize_input: bool = True,
    require_identities: bool = False,
) -> dict[str, Any]:
    """Build evidence-backed metadata for the A/B/C precision comparison.

    ``interface_tensors`` (or the explicit ``*_interface_fp32`` aliases) must
    be the observed A/B/C tensors at the common interface.  Each observation is
    converted to FP32 and, by default, BF16-rounded before the checks.  A
    common tensor without A/B/C observations is accepted only as source
    metadata; it never self-asserts an identity.  Native dtypes are required
    observations whenever interface tensors are supplied.
    """

    aliases = {"A": a_interface_fp32, "B": b_interface_fp32, "C": c_interface_fp32}
    supplied_aliases = {name: value for name, value in aliases.items() if value is not None}
    if supplied_aliases:
        if interface_tensors is not None:
            raise ValueError("provide interface_tensors or explicit A/B/C aliases, not both")
        interface_tensors = supplied_aliases
    if interface_tensors is not None:
        if set(interface_tensors) != {"A", "B", "C"}:
            raise ValueError("interface_tensors must contain exactly A, B, and C")
        if network_dtypes is None or set(network_dtypes) != {"A", "B", "C"}:
            raise ValueError("observed network_dtypes must contain exactly A, B, and C")
    dtypes = None if network_dtypes is None else {str(key): str(value) for key, value in network_dtypes.items()}
    source = None if common_input_fp32 is None else _fp32(common_input_fp32, name="common input")
    base = {
        "contract_version": "umi-input-quantization-compute-precision-v1",
        "common_interface_dtype": "float32",
        "quantizer": "bf16_round_to_nearest_even" if quantize_input else "none",
        "quantized_input_dtype": "float32",
        "source_input_sha256": None if source is None else sha256_array(source),
        "network_dtypes": dtypes,
    }
    if interface_tensors is None:
        base.update({
            "observed": False,
            "observation_reason": "actual A/B/C interface tensors and native dtype observations are required",
            "interface_hashes_fp32": None,
            "quantized_interface_hashes_fp32": None,
            "quantized_input_sha256": None,
            "A_B_input_sha256": None,
            "checks": {
                "A_B": {"passed": None, "reason": "A/B observation missing"},
                "B_C_zero": {"passed": None, "reason": "B/C observation missing"},
            },
            "A_B_exact_input_identity": None,
            "B_C_zero_input_identity": None,
        })
        return base

    observed = {name: _fp32(interface_tensors[name], name=f"{name} interface") for name in ("A", "B", "C")}
    quantized = {name: (quantize_bf16_fp32(value) if quantize_input else value.copy()) for name, value in observed.items()}
    ab = validate_ab_exact_input(quantized["A"], quantized["B"])
    bc = validate_bc_zero_input(quantized["B"], quantized["C"])
    if require_identities and not ab["passed"]:
        raise ValueError(f"A/B input identity failed: {ab.get('reason')}")
    if require_identities and not bc["passed"]:
        raise ValueError(f"B/C zero input identity failed: {bc.get('reason')}")
    base.update({
        "observed": True,
        "observation_reason": None,
        "interface_hashes_fp32": {name: sha256_array(value) for name, value in observed.items()},
        "quantized_interface_hashes_fp32": {name: sha256_array(value) for name, value in quantized.items()},
        "quantized_input_sha256": sha256_array(quantized["A"]),
        "A_B_input_sha256": sha256_array(quantized["B"]),
        "checks": {"A_B": ab, "B_C_zero": bc},
        "A_B_exact_input_identity": bool(ab["passed"]),
        "B_C_zero_input_identity": bool(bc["passed"]),
    })
    return base


def fixed_noise_hash(noise: Any, condition_mask: Any | None = None) -> str:
    """Hash the predicted/noisy region used by paired sampler calls."""

    array = _fp32(noise, name="sampler noise")
    if condition_mask is None:
        return sha256_array(array)
    mask = np.asarray(condition_mask, dtype=bool)
    if mask.shape != array.shape:
        raise ValueError("sampler noise and condition mask must have identical shapes")
    if not np.any(~mask):
        raise ValueError("predicted/noisy region is empty")
    return sha256_array(np.ascontiguousarray(array[~mask]))


hash_fixed_noise = fixed_noise_hash


def fixed_noise_identity(
    samples: Mapping[str, Any], condition_mask: Any | None = None, *, reference: str | None = None
) -> dict[str, Any]:
    """Compare paired sampler-noise hashes against one reference sample."""

    hashes = {str(name): fixed_noise_hash(value, condition_mask) for name, value in samples.items()}
    if not hashes:
        return {"passed": False, "reason": "no samples", "hashes": {}}
    ref = reference or next(iter(hashes))
    if ref not in hashes:
        raise ValueError(f"reference sample is missing: {ref}")
    passed = all(value == hashes[ref] for value in hashes.values())
    return {
        "passed": bool(passed),
        "reference": ref,
        "hashes": hashes,
        "reason": None if passed else "paired sampler-noise hashes differ",
    }


def _float64_pair(left: Any, right: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    a = _array(left, name=f"{name} left").astype(np.float64, copy=False)
    b = _array(right, name=f"{name} right").astype(np.float64, copy=False)
    if a.shape != b.shape:
        raise ValueError(f"{name} operands must have identical shapes")
    return a, b


def _rms64(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("RMS is undefined for an empty or non-finite tensor")
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def relative_derivative_change(left: Any, right: Any) -> float | None:
    """Return ``RMS(right-left) / RMS(left)`` with the left denominator."""

    try:
        a, b = _float64_pair(left, right, name="derivative")
        denominator = _rms64(a)
        if denominator == 0.0:
            return None
        return float(_rms64(b - a) / denominator)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def metric_with_reason(name: str, left: Any, right: Any) -> dict[str, Any]:
    """Return a scalar metric plus an explicit causal N/A reason."""

    try:
        a, b = _float64_pair(left, right, name=name)
        denominator = _rms64(a)
        if denominator == 0.0:
            return {"value": None, "reason": "left denominator RMS is zero"}
        value = _rms64(b - a) / denominator
        return {"value": float(value), "reason": None}
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        return {"value": None, "reason": f"{name} unavailable: {error}"}


def even_symmetry_residual(y_plus: Any, y_minus: Any, y0: Any) -> float:
    """Return unnormalised ``RMS(Y+ + Y- - 2Y0)``."""

    plus = _array(y_plus, name="Y_plus").astype(np.float64, copy=False)
    minus = _array(y_minus, name="Y_minus").astype(np.float64, copy=False)
    center = _array(y0, name="Y0").astype(np.float64, copy=False)
    if plus.shape != minus.shape or plus.shape != center.shape:
        raise ValueError("even-symmetry tensors must have identical shapes")
    return _rms64(plus + minus - (2.0 * center))


def decompose_response_vectors(r_A: Any, r_B: Any, r_C: Any) -> dict[str, Any]:
    """Decompose responses while making the A/C identity machine-checkable."""

    a = _array(r_A, name="r_A").astype(np.float64, copy=False)
    b = _array(r_B, name="r_B").astype(np.float64, copy=False)
    c = _array(r_C, name="r_C").astype(np.float64, copy=False)
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("response vectors must have identical shapes")
    direct = a - c
    b_to_c = b - c
    a_to_b = a - b
    reconstructed = b_to_c + a_to_b
    error = direct - reconstructed
    return {
        "r_A_minus_r_C": direct,
        "r_B_minus_r_C": b_to_c,
        "r_A_minus_r_B": a_to_b,
        "reconstructed": reconstructed,
        "identity_error": error,
        "identity_error_rms": _rms64(error),
        "identity_holds": bool(np.array_equal(direct, reconstructed)),
    }


vector_decomposition = decompose_response_vectors


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def canonical_json(value: Any) -> str:
    """Stable JSON representation for contract/hash metadata."""

    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


__all__ = [
    "MaskGeometry",
    "build_precision_contract",
    "canonical_json",
    "decompose_response_vectors",
    "effective_quantized_delta",
    "even_symmetry_residual",
    "fixed_noise_hash",
    "fixed_noise_identity",
    "hash_fixed_noise",
    "mask_geometry",
    "metric_with_reason",
    "quantize_bf16_fp32",
    "quantize_fp32_to_bf16",
    "relative_derivative_change",
    "resolve_mask_geometry",
    "slice_predicted_latent",
    "slice_predicted_output",
    "validate_ab_exact_input",
    "validate_bc_zero_input",
    "vector_decomposition",
]
