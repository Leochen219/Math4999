"""Pure NumPy numerical contracts and offline evidence analysis for Task7.

The numerical functions at the top of this module are intentionally small and
model-free.  The lower-level evidence adapter consumes only the reviewed
Task7 runner schema; it never loads a model or launches inference.  Runner
code and this analyzer are separate identities: the analyzer records its own
source digest and does not pretend to be the frozen live runner.
Inputs at the feedback interface are condition-carrier arrays and an explicit
boolean condition mask.  Differences are performed in FP32; scalar
reductions and fits are performed in FP64.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class EngineeringDataError(ValueError):
    """Raised when an input tensor cannot be used as scientific evidence."""


# A descriptive alias makes the distinction visible to callers that prefer a
# validation-oriented name.
Task7EngineeringError = EngineeringDataError


def _float32(value: Any, name: str, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Validate an already-materialized FP32 tensor without silently casting."""
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise EngineeringDataError(f"{name} must have dtype float32, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise EngineeringDataError(f"{name} shape {array.shape} does not match {shape}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise EngineeringDataError(f"{name} must be finite and nonempty")
    # Contiguity is not scientific content, but gives deterministic bytes and
    # avoids surprising views when a caller stores a returned residual.
    return np.ascontiguousarray(array)


def _mask(mask: Any, shape: tuple[int, ...], *, name: str = "condition mask") -> np.ndarray:
    result = np.asarray(mask)
    if result.dtype != np.dtype(bool):
        raise EngineeringDataError(f"{name} must have dtype bool, got {result.dtype}")
    if result.shape != shape:
        raise EngineeringDataError(f"{name} shape {result.shape} does not match {shape}")
    if result.size == 0 or not np.any(result):
        raise EngineeringDataError(f"{name} must select at least one coordinate")
    return np.ascontiguousarray(result)


def _masked(array: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    return array if mask is None else array[mask]


def fp32_difference(left: Any, right: Any) -> np.ndarray:
    """Return ``left - right`` using NumPy's FP32 subtraction exactly."""
    left_array = _float32(left, "left difference operand")
    right_array = _float32(right, "right difference operand", shape=left_array.shape)
    result = np.subtract(left_array, right_array, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise EngineeringDataError("FP32 difference overflowed to a non-finite value")
    return np.ascontiguousarray(result)


# Readable compatibility name for consumers that spell out the precision.
float32_difference = fp32_difference


def rms64(value: Any, mask: Any | None = None) -> float:
    """Return condition-mask RMS, with FP64 accumulation and a finite scalar."""
    array = _float32(value, "RMS value")
    selected_mask = None if mask is None else _mask(mask, array.shape)
    selected = _masked(array, selected_mask).astype(np.float64, copy=False)
    if selected.size == 0:
        raise EngineeringDataError("RMS selection is empty")
    result = float(np.sqrt(np.mean(np.square(selected, dtype=np.float64), dtype=np.float64)))
    if not math.isfinite(result):
        raise EngineeringDataError("RMS reduction is non-finite")
    return result


def cosine64(left: Any, right: Any, mask: Any | None = None) -> float | None:
    """Return FP64 cosine, or ``None`` for an exact zero denominator."""
    left_array = _float32(left, "left cosine value")
    right_array = _float32(right, "right cosine value", shape=left_array.shape)
    selected_mask = None if mask is None else _mask(mask, left_array.shape)
    # Flatten both full-carrier tensors before the inner product.  np.dot on
    # two rank>2 tensors is not a scalar inner product and can otherwise
    # return a matrix (or raise when coerced to float).
    left_selected = _masked(left_array, selected_mask).astype(np.float64, copy=False).reshape(-1)
    right_selected = _masked(right_array, selected_mask).astype(np.float64, copy=False).reshape(-1)
    denominator = float(np.linalg.norm(left_selected) * np.linalg.norm(right_selected))
    if denominator == 0.0:
        return None
    result = float(np.dot(left_selected, right_selected) / denominator)
    return result if math.isfinite(result) else None


def byte_equal(left: Any, right: Any) -> bool:
    """Compare dtype, shape, and contiguous bytes (so signed zero differs)."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
        return False
    return np.ascontiguousarray(left_array).tobytes() == np.ascontiguousarray(right_array).tobytes()


arrays_byte_equal = byte_equal


def input_geometry(actual_plus: Any, actual_minus: Any, direction: Any, mask: Any) -> dict[str, Any]:
    """Measure actual signed input lengths and direction/exterior gates.

    ``direction`` and actual deltas are in the input condition-carrier space;
    all geometric quantities are reduced only at ``mask`` coordinates.
    """
    plus = _float32(actual_plus, "positive actual input delta")
    minus = _float32(actual_minus, "negative actual input delta", shape=plus.shape)
    target = _float32(direction, "input direction", shape=plus.shape)
    selected_mask = _mask(mask, plus.shape)
    h_plus = rms64(plus, selected_mask)
    h_minus = rms64(minus, selected_mask)
    plus_cosine = cosine64(plus, target, selected_mask)
    minus_cosine = cosine64(minus, -target, selected_mask)
    opposite = cosine64(plus, minus, selected_mask)
    result: dict[str, Any] = {
        "h_plus": h_plus,
        "h_minus": h_minus,
        "plus_input_cosine": plus_cosine,
        "minus_input_cosine": minus_cosine,
        "opposite_cosine": opposite,
        "plus_minus_input_cosine": opposite,
        "plus_direction_cosine": plus_cosine,
        "minus_direction_cosine": minus_cosine,
        "plus_outside_exact": bool(np.all(plus[~selected_mask] == 0.0)),
        "minus_outside_exact": bool(np.all(minus[~selected_mask] == 0.0)),
        "plus_input_nonzero": h_plus > 0.0,
        "minus_input_nonzero": h_minus > 0.0,
    }
    return result


compute_input_geometry = input_geometry


def _fit_loglog(steps: Sequence[float], responses: Sequence[float]) -> dict[str, float | None]:
    """Fit log(response) against log(actual step), returning null on invalid data."""
    x = np.asarray(steps, dtype=np.float64).reshape(-1)
    y = np.asarray(responses, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2 or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        return {"slope": None, "intercept": None, "r2": None}
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        return {"slope": None, "intercept": None, "r2": None}
    log_x, log_y = np.log(x), np.log(y)
    centered_x = log_x - np.mean(log_x)
    centered_y = log_y - np.mean(log_y)
    xx = float(np.dot(centered_x, centered_x))
    if xx == 0.0:
        return {"slope": None, "intercept": None, "r2": None}
    slope = float(np.dot(centered_x, centered_y) / xx)
    intercept = float(np.mean(log_y) - slope * np.mean(log_x))
    predicted = slope * log_x + intercept
    total = float(np.dot(centered_y, centered_y))
    residual = float(np.dot(log_y - predicted, log_y - predicted))
    r2 = 1.0 if total == 0.0 and residual == 0.0 else (None if total == 0.0 else 1.0 - residual / total)
    return {"slope": slope, "intercept": intercept, "r2": r2 if r2 is None or math.isfinite(r2) else None}


fit_loglog = _fit_loglog


def _finite_nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EngineeringDataError(f"{name} must be a finite nonnegative scalar") from exc
    if not math.isfinite(result) or result < 0.0:
        raise EngineeringDataError(f"{name} must be a finite nonnegative scalar")
    return result


def _three(values: Iterable[Any], name: str) -> list[Any]:
    try:
        result = list(values)
    except TypeError as exc:
        raise EngineeringDataError(f"{name} must contain exactly three prespecified points") from exc
    if len(result) != 3:
        raise EngineeringDataError(f"{name} must contain exactly three prespecified points")
    return result


def _amplitudes(values: Iterable[Any]) -> tuple[float, float, float]:
    try:
        result = tuple(float(value) for value in _three(values, "amplitudes"))
    except (TypeError, ValueError) as exc:
        raise EngineeringDataError("amplitudes must be finite and positive") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise EngineeringDataError("amplitudes must be finite and positive")
    if not result[0] < result[1] < result[2]:
        raise EngineeringDataError("amplitudes must be strictly increasing")
    return result


def _threshold(value: float | None, *, low: float | None = None, high: float | None = None) -> bool:
    if value is None or not math.isfinite(value):
        return False
    return (low is None or value >= low) and (high is None or value <= high)


def one_direction_window(
    amplitudes: Sequence[float],
    direction: Any,
    plus_inputs: Sequence[Any],
    minus_inputs: Sequence[Any],
    plus_outputs: Sequence[Any],
    minus_outputs: Sequence[Any],
    baseline_output: Any,
    mask: Any,
    *,
    floor: float = 0.0,
    output_mask: Any | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed three-amplitude direction and all local reliability gates.

    Input geometry uses ``mask``; response RMS uses ``output_mask`` when
    supplied, otherwise the same mask (and therefore requires matching output
    shape).  The central quotient is always divided by each pair's actual
    ``h_plus + h_minus``.  No point may be dropped from this prespecified
    window.
    """
    alpha_values = _amplitudes(amplitudes)
    plus_input_values = _three(plus_inputs, "positive inputs")
    minus_input_values = _three(minus_inputs, "negative inputs")
    plus_output_values = _three(plus_outputs, "positive outputs")
    minus_output_values = _three(minus_outputs, "negative outputs")
    direction_array = _float32(direction, "input direction")
    input_mask = _mask(mask, direction_array.shape)
    baseline = _float32(baseline_output, "baseline output")
    response_mask = input_mask if output_mask is None else _mask(output_mask, baseline.shape, name="output mask")
    floor_value = _finite_nonnegative(floor, "response floor")

    points: list[dict[str, Any]] = []
    secants: list[dict[str, Any]] = []
    reasons: list[str] = []
    q_values: list[np.ndarray | None] = []
    for ordinal, alpha in enumerate(alpha_values):
        geometry = input_geometry(plus_input_values[ordinal], minus_input_values[ordinal], direction_array, input_mask)
        plus_output = _float32(plus_output_values[ordinal], f"positive output {ordinal}", shape=baseline.shape)
        minus_output = _float32(minus_output_values[ordinal], f"negative output {ordinal}", shape=baseline.shape)
        plus_response = fp32_difference(plus_output, baseline)
        minus_response = fp32_difference(minus_output, baseline)
        plus_rms = rms64(plus_response, response_mask)
        minus_rms = rms64(minus_response, response_mask)
        denominator = geometry["h_plus"] + geometry["h_minus"]
        quotient = None if denominator == 0.0 else np.divide(
            fp32_difference(plus_output, minus_output).astype(np.float64), denominator
        ).astype(np.float32)
        q_values.append(quotient)
        point = {
            "ordinal": ordinal,
            "alpha": alpha,
            **geometry,
            "plus_response_rms": plus_rms,
            "minus_response_rms": minus_rms,
            "plus_output_response_rms": plus_rms,
            "minus_output_response_rms": minus_rms,
            "central_q": quotient,
            "central_q_rms": None if quotient is None else rms64(quotient, response_mask),
            "response_reliable": {
                "plus": plus_rms > (0.0 if floor_value == 0.0 else 10.0 * floor_value),
                "minus": minus_rms > (0.0 if floor_value == 0.0 else 10.0 * floor_value),
            },
        }
        points.append(point)
        if geometry["plus_input_cosine"] is None or geometry["plus_input_cosine"] < 0.99:
            reasons.append(f"alpha={alpha}: plus input cosine below 0.99")
        if geometry["minus_input_cosine"] is None or geometry["minus_input_cosine"] < 0.99:
            reasons.append(f"alpha={alpha}: minus input cosine below 0.99")
        if geometry["opposite_cosine"] is None or geometry["opposite_cosine"] > -0.99:
            reasons.append(f"alpha={alpha}: plus/minus input is not opposite")
        if not geometry["plus_outside_exact"]:
            reasons.append(f"alpha={alpha}: positive input changes outside mask")
        if not geometry["minus_outside_exact"]:
            reasons.append(f"alpha={alpha}: negative input changes outside mask")
        threshold = 0.0 if floor_value == 0.0 else 10.0 * floor_value
        if plus_rms <= threshold:
            reasons.append(f"alpha={alpha}: positive response is not above reliability floor")
        if minus_rms <= threshold:
            reasons.append(f"alpha={alpha}: negative response is not above reliability floor")
        if denominator == 0.0:
            reasons.append(f"alpha={alpha}: actual central-step denominator is zero")

    plus_fit = _fit_loglog([point["h_plus"] for point in points], [point["plus_response_rms"] for point in points])
    minus_fit = _fit_loglog([point["h_minus"] for point in points], [point["minus_response_rms"] for point in points])
    if not _threshold(plus_fit["slope"], low=0.8, high=1.2):
        reasons.append("positive log-log slope outside [0.8,1.2]")
    if not _threshold(minus_fit["slope"], low=0.8, high=1.2):
        reasons.append("negative log-log slope outside [0.8,1.2]")
    if not _threshold(plus_fit["r2"], low=0.98):
        reasons.append("positive log-log R2 below 0.98")
    if not _threshold(minus_fit["r2"], low=0.98):
        reasons.append("negative log-log R2 below 0.98")
    for index in range(2):
        current, following = q_values[index], q_values[index + 1]
        if current is None or following is None:
            cosine, relative = None, None
            reasons.append(f"alpha={alpha_values[index]}: central secant is undefined")
        else:
            cosine = cosine64(current, following, response_mask)
            denominator = rms64(current, response_mask)
            relative = None if denominator == 0.0 else rms64(fp32_difference(following, current), response_mask) / denominator
            if not _threshold(cosine, low=0.95):
                reasons.append(f"alpha={alpha_values[index]}: adjacent secant cosine below 0.95")
            if relative is None or relative > 0.25:
                reasons.append(f"alpha={alpha_values[index]}: adjacent secant relative change above 0.25")
        secants.append({"from_alpha": alpha_values[index], "to_alpha": alpha_values[index + 1],
                        "cosine": cosine, "relative_change": relative,
                        "denominator": None if current is None else rms64(current, response_mask)})
    result = {
        "status": "PASS" if not reasons else "FAIL",
        "candidate_status": "PASS" if not reasons else "FAIL",
        "reliable": not reasons,
        "reasons": reasons,
        "failure_reasons": " | ".join(reasons),
        "point_count": 3,
        "window_alphas": alpha_values,
        "floor": floor_value,
        "points": points,
        "secants": secants,
        "plus_fit": plus_fit,
        "minus_fit": minus_fit,
        "additivity_status": "NOT_TESTED",
    }
    return result


analyze_one_direction_window = one_direction_window


def propagation_metrics(
    delta0: Any,
    delta1: Any,
    delta2: Any,
    mask: Any,
    *,
    floor0: float = 0.0,
    floor1: float = 0.0,
    floor2: float = 0.0,
) -> dict[str, Any]:
    """Compute two-step gains from FP32 condition-interface differences."""
    first = _float32(delta0, "delta0")
    second = _float32(delta1, "delta1", shape=first.shape)
    third = _float32(delta2, "delta2", shape=first.shape)
    selected_mask = _mask(mask, first.shape)
    floors = {"delta0": _finite_nonnegative(floor0, "floor0"),
              "delta1": _finite_nonnegative(floor1, "floor1"),
              "delta2": _finite_nonnegative(floor2, "floor2")}
    norms = {"delta0": rms64(first, selected_mask), "delta1": rms64(second, selected_mask),
             "delta2": rms64(third, selected_mask)}
    reasons: list[str] = []
    reliable: dict[str, bool] = {}
    for key, value in norms.items():
        threshold = 0.0 if floors[key] == 0.0 else 10.0 * floors[key]
        reliable[key] = value > threshold
        if not reliable[key]:
            reasons.append(f"{key} response is not above reliability floor")
    A1 = None if norms["delta0"] == 0.0 else norms["delta1"] / norms["delta0"]
    A2 = None if norms["delta0"] == 0.0 else norms["delta2"] / norms["delta0"]
    incremental = None if norms["delta1"] == 0.0 or A2 is None else A2 / A1 if A1 is not None else None
    if norms["delta0"] == 0.0:
        reasons.append("delta0 denominator is zero")
    if norms["delta1"] == 0.0:
        reasons.append("delta1 denominator is zero for incremental gain")
    return {
        "delta_rms": norms,
        "floor": floors,
        "reliability": reliable,
        "A1": A1,
        "A2": A2,
        "incremental": incremental,
        "reasons": reasons,
        "status": "PASS" if not reasons else "UNRELIABLE",
    }


compute_propagation = propagation_metrics


def _prediction_from_q(q: np.ndarray, delta1_rms: float, actual_delta2: np.ndarray,
                       output_mask: np.ndarray) -> tuple[np.ndarray, float | None]:
    predicted = np.multiply(q, np.float32(delta1_rms), dtype=np.float32)
    residual = fp32_difference(actual_delta2, predicted)
    denominator = rms64(actual_delta2, output_mask)
    error = None if denominator == 0.0 else rms64(residual, output_mask) / denominator
    return predicted, error


def fixed_beta_prediction(
    *,
    actual_delta1: Any,
    beta_plus_input: Any,
    beta_minus_input: Any,
    beta_plus_output: Any,
    beta_minus_output: Any,
    baseline_output: Any,
    actual_delta2: Any,
    mask: Any,
    beta: float = 0.1,
    local_window_pass: bool | None = None,
    response_reliable: bool | None = None,
    output_mask: Any | None = None,
) -> dict[str, Any]:
    """Predict ``delta2`` from the fixed beta=.1 actual-step central quotient.

    The supplied ``actual_delta1`` is normalized only through its own masked
    RMS.  No original direction (such as ``v0``) enters this calculation.
    """
    if not math.isclose(float(beta), 0.1, rel_tol=0.0, abs_tol=0.0):
        raise EngineeringDataError("Task7 prediction estimator beta is fixed at 0.1")
    delta1 = _float32(actual_delta1, "actual delta1")
    plus_input = _float32(beta_plus_input, "beta positive input", shape=delta1.shape)
    minus_input = _float32(beta_minus_input, "beta negative input", shape=delta1.shape)
    input_mask = _mask(mask, delta1.shape)
    baseline = _float32(baseline_output, "prediction baseline")
    plus_output = _float32(beta_plus_output, "beta positive output", shape=baseline.shape)
    minus_output = _float32(beta_minus_output, "beta negative output", shape=baseline.shape)
    observed_delta2 = _float32(actual_delta2, "actual delta2", shape=baseline.shape)
    selected_output_mask = input_mask if output_mask is None else _mask(output_mask, baseline.shape, name="output mask")
    delta1_rms = rms64(delta1, input_mask)
    geometry = input_geometry(plus_input, minus_input, delta1, input_mask)
    denominator = geometry["h_plus"] + geometry["h_minus"]
    reasons: list[str] = []
    if delta1_rms == 0.0:
        reasons.append("delta1 direction denominator is zero")
    if denominator == 0.0:
        reasons.append("beta=.1 actual-step denominator is zero")
    q = None if denominator == 0.0 else np.divide(
        fp32_difference(plus_output, minus_output).astype(np.float64), denominator
    ).astype(np.float32)
    if q is None:
        predicted, error = None, None
    else:
        predicted, error = _prediction_from_q(q, delta1_rms, observed_delta2, selected_output_mask)
    if error is None:
        reasons.append("zero evaluation denominator")
    if local_window_pass is not True:
        reasons.append("local window reliability flag was not true")
    if response_reliable is not True:
        reasons.append("measured response reliability flag was not true")
    if geometry["plus_input_cosine"] is None or geometry["plus_input_cosine"] < 0.99:
        reasons.append("beta=.1 positive input direction failed")
    if geometry["minus_input_cosine"] is None or geometry["minus_input_cosine"] < 0.99:
        reasons.append("beta=.1 negative input direction failed")
    if geometry["opposite_cosine"] is None or geometry["opposite_cosine"] > -0.99:
        reasons.append("beta=.1 inputs are not opposite")
    if not geometry["plus_outside_exact"] or not geometry["minus_outside_exact"]:
        reasons.append("beta=.1 input changed coordinates outside mask")
    if geometry["h_plus"] == 0.0 or geometry["h_minus"] == 0.0:
        reasons.append("beta=.1 input step is zero")
    if local_window_pass is False:
        reasons.append("local window reliability did not pass")
    if response_reliable is False:
        reasons.append("measured response reliability did not pass")
    geometry_reliable = (
        delta1_rms > 0.0
        and denominator > 0.0
        and geometry["plus_input_cosine"] is not None
        and geometry["plus_input_cosine"] >= 0.99
        and geometry["minus_input_cosine"] is not None
        and geometry["minus_input_cosine"] >= 0.99
        and geometry["opposite_cosine"] is not None
        and geometry["opposite_cosine"] <= -0.99
        and geometry["plus_outside_exact"]
        and geometry["minus_outside_exact"]
    )
    reliability = bool(geometry_reliable and local_window_pass is True and response_reliable is True and error is not None)
    if error is None:
        status = "N/A"
    elif reasons:
        status = "UNRELIABLE"
    else:
        status = "PASS" if error <= 0.10 else "FAIL"
        if status == "FAIL":
            reasons.append("Eprop above 0.10")
    return {
        "status": status,
        "Eprop": error,
        "q_actual": q,
        "predicted_delta2": predicted,
        "actual_delta2_rms": rms64(observed_delta2, selected_output_mask),
        "delta1_rms": delta1_rms,
        "beta": 0.1,
        "h_plus": geometry["h_plus"],
        "h_minus": geometry["h_minus"],
        "reasons": reasons,
        # Reliability means the local geometry/noise/evaluation gates were
        # defined and met; an Eprop above .10 is a reliable scientific FAIL,
        # not an UNRELIABLE measurement.
        "reliable": reliability,
    }


predict_fixed_beta = fixed_beta_prediction


def evaluate_fixed_beta_predictions(
    rays: Sequence[Mapping[str, Any]],
    *,
    q_by_ray: Sequence[Any],
    mask: Any,
    local_window_pass: bool | Sequence[bool] | None = None,
    response_reliable: bool | Sequence[bool] | None = None,
    output_mask: Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all six independent C rays without merging or projecting them."""
    if len(rays) != 6 or len(q_by_ray) != 6:
        raise EngineeringDataError("Task7 C evaluation requires exactly six rays")
    first_ray = _float32(rays[0]["ray"], "ray 0")
    input_mask = _mask(mask, first_ray.shape)
    selected_output_mask = input_mask if output_mask is None else output_mask
    rows: list[dict[str, Any]] = []
    def flag(global_value: bool | Sequence[bool] | None, record: Mapping[str, Any], key: str, index: int) -> bool | None:
        def strict(value: Any) -> bool | None:
            if value is None:
                return None
            if not isinstance(value, (bool, np.bool_)):
                raise EngineeringDataError(f"{key} must be bool or None")
            return bool(value)

        if key in record:
            return strict(record[key])
        if isinstance(global_value, (list, tuple, np.ndarray)):
            try:
                count = len(global_value)
            except TypeError as exc:
                raise EngineeringDataError(f"{key} must be bool, None, or six values") from exc
            if count != 6:
                raise EngineeringDataError(f"{key} must contain one value per ray")
            return strict(global_value[index])
        return strict(global_value)

    for index, (record, q_value) in enumerate(zip(rays, q_by_ray)):
        ray = _float32(record["ray"], f"ray {index}", shape=first_ray.shape)
        actual = _float32(record["actual_delta2"], f"actual delta2 ray {index}")
        q = _float32(q_value, f"q ray {index}", shape=actual.shape)
        d1_rms = rms64(ray, input_mask)
        ray_output_mask = _mask(selected_output_mask, actual.shape, name="output mask")
        pred, error = _prediction_from_q(q, d1_rms, actual, ray_output_mask)
        reasons: list[str] = []
        if d1_rms == 0.0:
            reasons.append("zero ray denominator")
        if error is None:
            reasons.append("zero evaluation denominator")
        ray_local_pass = flag(local_window_pass, record, "local_window_pass", index)
        ray_response_reliable = flag(response_reliable, record, "response_reliable", index)
        if error is None:
            status = "N/A"
        else:
            if ray_local_pass is not True:
                reasons.append("local window reliability flag was not true")
            if ray_response_reliable is not True:
                reasons.append("measured response reliability flag was not true")
            for reason in record.get("reasons", ()):
                reasons.append(str(reason))
            status = "UNRELIABLE" if reasons else ("PASS" if error <= 0.10 else "FAIL")
        reliable = bool(d1_rms > 0.0 and error is not None and
                        ray_local_pass is True and ray_response_reliable is True and not reasons)
        rows.append({"ray_index": index, "status": status, "Eprop": error,
                     "delta1_rms": d1_rms, "predicted_delta2": pred,
                     "reasons": reasons, "reliable": reliable})
    return rows


# ---------------------------------------------------------------------------
# Saved-evidence adapter (Task3b)

ANALYSIS_SCHEMA_VERSION = "umi-task7-analysis-v1"
ANALYSIS_VERSION = "task7-offline-evidence-v1"
TASK7_ALPHAS = (0.001, 0.003, 0.01)
TASK7_BETAS = (0.1, 0.2, 0.4)


def _runner_module() -> Any:
    """Import the released runner without duplicating its artifact schema."""
    try:
        from . import run_umi_task7_experiment as module
    except ImportError:  # direct execution from experiments/
        import run_umi_task7_experiment as module
    return module


def _json_safe(value: Any) -> Any:
    """Convert derived values to strict JSON without serializing tensors."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise EngineeringDataError("non-finite derived value cannot be serialized")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _analysis_source_sha256() -> str:
    """Digest this analyzer only; it is intentionally not the runner digest."""
    return _sha256_file(Path(__file__).resolve())


def _canonical(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_bool_mask(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(bool):
        raise EngineeringDataError(f"{name} must have dtype bool, got {array.dtype}")
    if shape is not None and array.shape != shape:
        raise EngineeringDataError(f"{name} shape {array.shape} does not match {shape}")
    if array.size == 0 or not np.any(array):
        raise EngineeringDataError(f"{name} must select at least one coordinate")
    return np.ascontiguousarray(array)


def _load_runner_array(path: Path, *, name: str, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    """Load through the runner helper, which copies and closes mmap handles."""
    module = _runner_module()
    try:
        value = module._load_array(path)
    except Exception as error:
        raise EngineeringDataError(f"unable to load {name}: {path}: {error}") from error
    array = np.asarray(value)
    if array.dtype != dtype:
        raise EngineeringDataError(f"{name} must have dtype {dtype}, got {array.dtype}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise EngineeringDataError(f"{name} must be finite and nonempty")
    return np.ascontiguousarray(array)


def _condition_indexes(mask: np.ndarray) -> tuple[int, ...]:
    """Return the runner's condition-frame coordinates from the full mask."""
    if mask.ndim < 4:
        raise EngineeringDataError("Task7 carrier mask must include temporal and spatial axes")
    temporal_axis = mask.ndim - 3
    indexes = tuple(index for index in range(mask.shape[temporal_axis])
                    if bool(np.all(np.take(mask, index, axis=temporal_axis))))
    if not indexes:
        raise EngineeringDataError("authoritative mask contains no complete condition frame")
    return indexes


def _condition_view(value: Any, mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32):
        raise EngineeringDataError(f"{name} must have dtype float32, got {array.dtype}")
    if tuple(array.shape) != tuple(mask.shape):
        raise EngineeringDataError(f"{name} shape {array.shape} differs from full carrier {mask.shape}")
    indexes = _condition_indexes(mask)
    temporal_axis = mask.ndim - 3
    selected = np.take(array, indexes, axis=temporal_axis)
    return np.ascontiguousarray(selected)


def _array_from_record(record: Mapping[str, Any], key: str, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    value = record.get(key)
    if not isinstance(value, np.ndarray):
        raise EngineeringDataError(f"record is missing tensor evidence: {key}")
    array = np.asarray(value)
    if array.dtype != dtype:
        raise EngineeringDataError(f"record {key} must have dtype {dtype}, got {array.dtype}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise EngineeringDataError(f"record {key} must be finite and nonempty")
    return np.ascontiguousarray(array)


def _record_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping) or not isinstance(payload.get("record"), Mapping):
        raise EngineeringDataError("Task7 sample payload lacks a runtime record")
    return payload["record"]


class _LazyStageRecords(Mapping[str, Mapping[str, Any]]):
    """Verified stage index whose large arrays are loaded one sample at a time."""

    def __init__(self, store: Any, sample_ids: Sequence[str], skipped: set[str]):
        self._store = store
        self._sample_ids = tuple(sample_ids)
        self._skipped = set(skipped)

    def __getitem__(self, sample_id: str) -> Mapping[str, Any]:
        if sample_id not in self._sample_ids:
            raise KeyError(sample_id)
        payload = self._store.load_record(sample_id)
        if sample_id in self._skipped:
            return payload
        return _record_payload(payload)

    def __iter__(self) -> Iterator[str]:
        return iter(self._sample_ids)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def get(self, sample_id: str, default: Any = None) -> Any:
        try:
            return self[sample_id]
        except KeyError:
            return default


def _baseline_floor(post: Any, pre: Any, *, name: str) -> float:
    """Compute a floor from the saved baseline pair, never from a caller flag."""
    post_array = _array_from_record(post, "encoded_condition") if isinstance(post, Mapping) else _float32(post, f"{name} post")
    pre_array = _array_from_record(pre, "encoded_condition") if isinstance(pre, Mapping) else _float32(pre, f"{name} pre")
    if post_array.shape != pre_array.shape:
        raise EngineeringDataError(f"{name} baseline pair shapes differ")
    return rms64(fp32_difference(post_array, pre_array))


def _operation_counts(record: Mapping[str, Any], stage: str) -> dict[str, int]:
    evidence = record.get("evidence")
    counts = evidence.get("operation_counts") if isinstance(evidence, Mapping) else None
    if not isinstance(counts, Mapping):
        raise EngineeringDataError(f"{stage} record lacks observed operation_counts")
    result: dict[str, int] = {}
    for key in ("G", "D", "E"):
        value = counts.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
            raise EngineeringDataError(f"{stage} operation count {key} is invalid")
        result[key] = int(value)
    return result


def _load_source_evidence(raw_root: str | Path, decoder_root: str | Path) -> dict[str, Any]:
    """Validate immutable Task6 pairs, then load only their approved artifacts."""
    module = _runner_module()
    try:
        source = module.validate_task7_sources(raw_root, decoder_root)
    except Exception as error:
        raise EngineeringDataError(f"Task7 source validation failed: {error}") from error
    rows: dict[str, dict[str, Any]] = {}
    v0: np.ndarray | None = None
    z0_view: np.ndarray | None = None
    mask_view: np.ndarray | None = None
    for row in source["samples"]:
        name = str(row["name"])
        raw = Path(row["raw"]); decoder = Path(row["decoder"])
        # The runner currently converts mask arrays to bool in its source
        # validator.  Re-check the stored dtype here: a casted mask cannot be
        # accepted as byte-exact engineering evidence.
        raw_mask = _load_runner_array(raw / "mask.npy", name=f"{name} mask", dtype=np.dtype(bool))
        mask = _strict_bool_mask(raw_mask, f"{name} mask")
        z0 = _load_runner_array(raw / "z_bar.npy", name=f"{name} z0")
        consumed = _load_runner_array(raw / "consumed_input_fp32.npy", name=f"{name} consumed input")
        direction_path = raw / "direction.npy"
        direction = _load_runner_array(direction_path, name=f"{name} direction") if direction_path.is_file() else None
        direct = _load_runner_array(decoder / "direct_condition_latent_float32.npy", name=f"{name} direct condition")
        if z0.shape != mask.shape or consumed.shape != z0.shape:
            raise EngineeringDataError(f"{name} source carrier/mask shape mismatch")
        delta = fp32_difference(consumed, z0)
        if np.any(delta[~mask] != 0.0):
            raise EngineeringDataError(f"{name} consumed condition changes coordinates outside authoritative mask")
        if not name.startswith("baseline") and direction is None:
            raise EngineeringDataError(f"{name} lacks the saved v0 direction artifact")
        if direction is not None:
            if direction.shape != z0.shape or np.any(direction[~mask] != 0.0):
                raise EngineeringDataError(f"{name} direction is not bound to the authoritative mask")
            if name.startswith("baseline") and np.any(direction != 0.0):
                raise EngineeringDataError(f"{name} baseline direction is not exactly zero")
            if not name.startswith("baseline"):
                if v0 is None:
                    v0 = direction.copy()
                elif not byte_equal(direction, v0):
                    raise EngineeringDataError(f"{name} direction differs bytewise from v0")
        condition_mask = np.ones_like(_condition_view(z0, mask, name=f"{name} z0"), dtype=bool)
        row_data = {
            "name": name, "raw": raw, "decoder": decoder, "direct": direct,
            # Keep only compact condition-coordinate tensors. Full chunks are
            # released at the end of this iteration; baseline_pre owns the
            # authoritative mask view used for later chain checks.
            "mask": mask if name == "baseline_pre" else None,
            "z0_condition": _condition_view(z0, mask, name=f"{name} z0"),
            "consumed_condition": _condition_view(consumed, mask, name=f"{name} consumed"),
            "condition_mask": condition_mask,
            "direction_condition": None if direction is None else _condition_view(direction, mask, name=f"{name} direction"),
            "delta_condition": _condition_view(delta, mask, name=f"{name} delta"),
        }
        if z0_view is None:
            z0_view, mask_view = row_data["z0_condition"], condition_mask
        elif not byte_equal(row_data["z0_condition"], z0_view):
            raise EngineeringDataError(f"{name} does not share the immutable z0 condition")
        rows[name] = row_data
    if v0 is None:
        raise EngineeringDataError("Task7 source contains no v0 direction")
    source["rows"] = rows
    source["mask"] = next(iter(rows.values()))["mask"] if next(iter(rows.values())).get("mask") is not None else None
    source["v0"] = v0
    source["v0_condition"] = _condition_view(v0, next(iter(rows.values()))["mask"], name="v0")
    source["z0_condition"] = z0_view
    source["condition_mask"] = mask_view
    return source


def load_task7_stage(run_dir: str | Path, stage: str) -> dict[str, Any]:
    """Load one complete stage through the runner's verified sample store.

    A partial, failed, skipped, tampered, or count-inconsistent stage is an
    engineering failure.  In particular, this function never substitutes a
    missing sample with zeros and never searches for guessed ``.npy`` names.
    """
    module = _runner_module()
    stage_name = str(stage).upper()
    if stage_name not in {"A", "B", "C"}:
        raise EngineeringDataError("stage must be A, B, or C")
    root = Path(run_dir).resolve() / "stages" / stage_name
    status_path, config_path = root / "run_status.json", root / "stage_config.json"
    if not status_path.is_file() or not config_path.is_file():
        raise EngineeringDataError(f"Task7 stage {stage_name} is missing run_status.json or stage_config.json")
    try:
        status = module._load_json(status_path)
        config = module._load_json(config_path)
        plan = module.build_stage_plan(stage_name)
        store = module.Task7SampleStore(root / "samples")
    except Exception as error:
        raise EngineeringDataError(f"unable to load Task7 stage {stage_name}: {error}") from error
    if not isinstance(status, Mapping) or status.get("status") != "COMPLETE":
        raise EngineeringDataError(f"Task7 stage {stage_name} is not COMPLETE (engineering status={status.get('status') if isinstance(status, Mapping) else None})")
    if (not isinstance(config, Mapping) or config.get("schema_version") != "umi-task7-stage-v2"
            or config.get("stage") != stage_name or _canonical(config.get("plan")) != _canonical(plan)
            or config.get("plan_sha256") != hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()):
        raise EngineeringDataError(f"Task7 stage {stage_name} config/plan binding is inconsistent")
    expected = [str(spec["sample_id"]) for spec in plan]
    completed, skipped, failed = status.get("completed_samples"), status.get("skipped_samples"), status.get("failed_samples")
    if not isinstance(completed, list) or not isinstance(skipped, list) or not isinstance(failed, list):
        raise EngineeringDataError(f"Task7 stage {stage_name} completion lists are malformed")
    if any(not isinstance(item, str) for item in (*completed, *skipped, *failed)):
        raise EngineeringDataError(f"Task7 stage {stage_name} completion list contains a non-string id")
    if (len(completed) != len(set(completed)) or len(skipped) != len(set(skipped))
            or len(failed) != len(set(failed))):
        raise EngineeringDataError(f"Task7 stage {stage_name} completion lists contain duplicates")
    if stage_name != "C" and (set(completed) != set(expected) or skipped or failed):
        raise EngineeringDataError(f"Task7 stage {stage_name} has incomplete sample IDs")
    if stage_name == "C" and ((set(completed) | set(skipped)) != set(expected)
                               or bool(set(completed) & set(skipped)) or bool(failed)):
        raise EngineeringDataError(f"Task7 stage C has inconsistent sample IDs")
    try:
        formal_counts = module.stage_counts(stage_name)
    except Exception as error:
        raise EngineeringDataError(f"Task7 stage {stage_name} count schema is unavailable") from error
    if (status.get("schema_version") != "umi-task7-run-v2"
            or status.get("planned_samples") != len(expected)
            or status.get("completed_count") != len(completed)
            or status.get("failed_count") != len(failed)
            or status.get("skipped_count") != len(skipped)
            or status.get("formal_counts") != formal_counts):
        raise EngineeringDataError(f"Task7 stage {stage_name} run-status counts/schema are inconsistent")
    records: dict[str, Mapping[str, Any]] = {}
    for sample_id in expected:
        try:
            # _verify checks every manifest-bound artifact and JSON reference
            # without materializing any .npy payload.  Arrays are loaded by
            # _LazyStageRecords only when the stage analysis reaches a sample.
            store._verify(store._path(sample_id))
            payload_json = module._load_json(store._path(sample_id) / "record.json")
        except Exception as error:
            raise EngineeringDataError(f"Task7 sample evidence failed verification: {sample_id}: {error}") from error
        if sample_id in skipped and payload_json.get("status") != "SKIPPED_C":
            raise EngineeringDataError(f"Task7 skipped C sample lacks explicit SKIPPED_C evidence: {sample_id}")
        payload_spec = payload_json.get("spec")
        spec = next(item for item in plan if item["sample_id"] == sample_id)
        if _canonical(payload_spec) != _canonical(spec):
            raise EngineeringDataError(f"Task7 sample spec differs from immutable plan: {sample_id}")
    lazy_records = _LazyStageRecords(store, expected, set(skipped))
    return {"run_dir": Path(run_dir).resolve(), "stage_root": root, "status": status,
            "config": config, "plan": plan, "records": lazy_records}


def _record_map(stage_data: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    records = stage_data["records"]
    if not isinstance(records, Mapping):
        raise EngineeringDataError("Task7 stage record index is not a mapping")
    return records  # type: ignore[return-value]


def _binding(stage_data: Mapping[str, Any]) -> Mapping[str, Any]:
    config = stage_data.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("binding"), Mapping):
        raise EngineeringDataError("Task7 stage config lacks immutable binding")
    return config["binding"]


def _validate_stage_binding(stage_data: Mapping[str, Any], source: Mapping[str, Any]) -> list[str]:
    """Bind stage config/status identities to the independently validated source."""
    reasons: list[str] = []
    config = stage_data.get("config")
    status = stage_data.get("status")
    binding = _binding(stage_data)
    status_binding = status.get("binding") if isinstance(status, Mapping) else None
    if not isinstance(status_binding, Mapping) or _canonical(status_binding) != _canonical(binding):
        reasons.append("stage run_status.binding differs from stage_config.binding")
    for key, expected in (("source", source.get("source_tree_sha256")),
                          ("mask", source.get("mask_sha256")),
                          ("z0", source.get("z0_sha256")),
                          ("v0", source.get("v0_sha256"))):
        if not isinstance(expected, str) or not expected or binding.get(key) != expected:
            reasons.append(f"stage binding.{key} does not match validated source identity")
    if not isinstance(binding.get("task7_code_sha256"), str) or not binding.get("task7_code_sha256"):
        reasons.append("stage binding.task7_code_sha256 is missing")
    config_sha = binding.get("config_sha256")
    if config_sha is None and isinstance(config, Mapping):
        config_sha = config.get("config_sha256")
    if config_sha is not None:
        config_payload = binding.get("config")
        if not isinstance(config_payload, Mapping) or not isinstance(config_sha, str):
            reasons.append("stage config_sha256 binding is malformed")
        else:
            observed = hashlib.sha256(_canonical(config_payload).encode("utf-8")).hexdigest()
            if observed != config_sha:
                reasons.append("stage config_sha256 does not match binding.config")
    return reasons


def _check_condition_chain(record: Mapping[str, Any], condition: np.ndarray, *, name: str,
                           full_mask: np.ndarray | None = None) -> list[str]:
    """Verify chain evidence at condition coordinates, returning diagnostics."""
    reasons: list[str] = []
    try:
        input_condition = _array_from_record(record, "condition_input_fp32")
        if not byte_equal(input_condition, condition):
            reasons.append(f"{name}: condition_input_fp32 differs from requested condition")
        encoded = _array_from_record(record, "encoded_condition")
        next_condition = record.get("next_condition_fp32")
        if not isinstance(next_condition, np.ndarray) or not byte_equal(encoded, next_condition):
            reasons.append(f"{name}: encoded_condition differs from next_condition_fp32")
    except EngineeringDataError as error:
        reasons.append(str(error))
    actual = record.get("actual")
    if not isinstance(actual, Mapping):
        reasons.append(f"{name}: missing actual condition-chain evidence")
        return reasons
    expected_keys = ("prepared_condition", "initial_condition", "reference_condition",
                     "first_condition", "last_condition", "condition_steps")
    for key in expected_keys:
        value = actual.get(key)
        if not isinstance(value, np.ndarray):
            reasons.append(f"{name}: missing actual.{key}")
            continue
        array = np.asarray(value)
        if array.dtype != np.dtype(np.float32) or not np.all(np.isfinite(array)):
            reasons.append(f"{name}: actual.{key} is not finite FP32")
            continue
        candidate = array
        if key == "condition_steps":
            if array.ndim < 1:
                reasons.append(f"{name}: actual.condition_steps shape differs from condition")
                continue
            if tuple(array.shape[1:]) == tuple(condition.shape):
                candidate = array
            elif full_mask is not None and tuple(array.shape[1:]) == tuple(full_mask.shape):
                candidate = np.stack([_condition_view(item, full_mask, name=f"{name} {key}") for item in array])
            else:
                reasons.append(f"{name}: actual.condition_steps shape differs from condition")
                continue
            if not all(byte_equal(item, condition) for item in candidate):
                reasons.append(f"{name}: actual.condition_steps differ from consumed condition")
            continue
        if tuple(candidate.shape) != tuple(condition.shape) and full_mask is not None and tuple(candidate.shape) == tuple(full_mask.shape):
            candidate = _condition_view(candidate, full_mask, name=f"{name} {key}")
        if tuple(candidate.shape) != tuple(condition.shape):
            # Full carrier evidence is accepted only when the condition-only
            # coordinates are exactly the requested input; no all-element
            # equality is inferred from a shape coincidence.
            reasons.append(f"{name}: actual.{key} shape differs from condition")
            continue
        if not byte_equal(candidate, condition):
            reasons.append(f"{name}: actual.{key} differs from consumed condition")
    return reasons


def _encoder_frame(row: Mapping[str, Any], *, name: str) -> np.ndarray:
    """Load one saved Task6 FP32-D frame without retaining a frame corpus."""
    frame = row.get("frame")
    if isinstance(frame, np.ndarray):
        value = np.ascontiguousarray(frame)
        if value.dtype != np.dtype(np.float32):
            raise EngineeringDataError(f"{name} saved frame must have dtype float32")
        return value
    decoder = row.get("decoder")
    if not isinstance(decoder, Path):
        decoder = Path(str(decoder))
    return _load_runner_array(decoder / "decoded_final_float32.npy", name=f"{name} decoded frame")


def _validate_encoder_evidence(record: Mapping[str, Any], source_row: Mapping[str, Any], *,
                               precision: str, encoded: np.ndarray, name: str) -> list[str]:
    """Validate the actual FeedbackEncoder evidence before A science.

    This deliberately checks the saved evidence contract rather than trusting
    the runner's summary flag. Native keeps its observed dtype path; the
    temporary FP32 path additionally requires all floating computation/state
    evidence to be float32 and its backend/cache guards to have been observed.
    """
    reasons: list[str] = []
    encoder = record.get("encoder")
    if not isinstance(encoder, Mapping):
        return [f"A {name} encoder evidence is missing"]
    evidence = encoder.get("evidence")
    arrays = encoder.get("arrays")
    if not isinstance(evidence, Mapping) or not isinstance(arrays, Mapping):
        return [f"A {name} encoder evidence/arrays mapping is missing"]
    required_evidence = (
        "precision_path", "input_dtype", "input_shape", "encoder_input_shape",
        "encoder_input_dtype", "operation_count", "operation_dtypes", "encoder_identity",
        "output_dtype", "inner_input_dtype", "inner_output_dtype", "state_dtypes",
        "actual_encoder_input_dtype", "scaled_latent_dtype", "actual_output_dtype",
        "dispatch_observed", "autocast_disabled", "tf32_disabled",
        "cache_cleared_before", "cache_cleared_after",
    )
    for key in required_evidence:
        if key not in evidence:
            reasons.append(f"A {name} encoder evidence missing {key}")
    if evidence.get("precision_path") != precision:
        reasons.append(f"A {name} precision_path is not {precision}")
    try:
        frame = _encoder_frame(source_row, name=name)
    except EngineeringDataError as error:
        reasons.append(str(error)); frame = None
    if frame is not None:
        input_rgb = arrays.get("input_rgb")
        if not isinstance(input_rgb, np.ndarray) or not byte_equal(input_rgb, frame):
            reasons.append(f"A {name} encoder input_rgb differs from saved Task6 decoded frame")
        if tuple(frame.shape) != tuple(evidence.get("input_shape", ())):
            reasons.append(f"A {name} encoder input_shape differs from saved frame")
        encoder_input = arrays.get("encoder_input")
        expected_shape = (1, 3, 1, frame.shape[1], frame.shape[2]) if frame.ndim == 3 and frame.shape[0] == 3 else None
        if not isinstance(encoder_input, np.ndarray):
            reasons.append(f"A {name} encoder_input array is missing")
        else:
            if expected_shape is None or tuple(encoder_input.shape) != expected_shape:
                reasons.append(f"A {name} encoder_input layout is not [1,3,1,H,W]")
            else:
                expected_input = np.subtract(np.multiply(frame, np.float32(2.0), dtype=np.float32),
                                             np.float32(1.0), dtype=np.float32)[None, :, None, :, :]
                if not byte_equal(encoder_input, expected_input):
                    reasons.append(f"A {name} encoder_input is not the FP32 [-1,1] conversion")
    actual_output = arrays.get("actual_output")
    if not isinstance(actual_output, np.ndarray) or not byte_equal(actual_output, encoded):
        reasons.append(f"A {name} actual_output differs bytewise from encoded_condition")
    for key in ("actual_encoder_input", "actual_output"):
        value = arrays.get(key)
        if not isinstance(value, np.ndarray):
            reasons.append(f"A {name} arrays.{key} is missing")
    state = evidence.get("state_dtypes")
    if not isinstance(state, Mapping) or any(not isinstance(state.get(bucket), Mapping) or not state[bucket]
                                             for bucket in ("parameters", "buffers", "constants")):
        reasons.append(f"A {name} state_dtypes is incomplete")
    else:
        for bucket in ("parameters", "buffers", "constants"):
            if any(not isinstance(dtype, str) for dtype in state[bucket].values()):
                reasons.append(f"A {name} state_dtypes contains an invalid dtype")
    operation_dtypes = evidence.get("operation_dtypes")
    if not isinstance(operation_dtypes, Mapping) or not operation_dtypes:
        reasons.append(f"A {name} operation_dtypes is incomplete")
    if not isinstance(evidence.get("operation_count"), (int, np.integer)) or int(evidence.get("operation_count", 0)) <= 0:
        reasons.append(f"A {name} operation_count is not positive")
    # All precision paths require the actual observed state; only FP32 mode
    # imposes the all-float32 values and backend/cache guards.
    dtype_fields = ("actual_encoder_input_dtype", "actual_output_dtype", "inner_input_dtype", "inner_output_dtype")
    if precision == "temporary_fp32":
        if evidence.get("dispatch_observed") is not True or evidence.get("operation_count", 0) <= 0:
            reasons.append(f"A {name} FP32 dispatch evidence is incomplete")
        if any(value != "float32" for value in evidence.get("operation_dtypes", {})):
            reasons.append(f"A {name} FP32 operation_dtypes contains a non-float32 dtype")
        if any(value != "float32" for bucket in ("parameters", "buffers", "constants")
               for value in (state.get(bucket, {}).values() if isinstance(state, Mapping) and isinstance(state.get(bucket), Mapping) else ())):
            reasons.append(f"A {name} FP32 state_dtypes contains a non-float32 dtype")
        if any(evidence.get(key) != "float32" for key in dtype_fields):
            reasons.append(f"A {name} FP32 inner/output dtype evidence is not float32")
        for key in ("autocast_disabled", "tf32_disabled", "cache_cleared_before", "cache_cleared_after"):
            if evidence.get(key) is not True:
                reasons.append(f"A {name} FP32 {key} evidence is not true")
    else:
        if any(not isinstance(evidence.get(key), str) or not evidence.get(key) for key in dtype_fields):
            reasons.append(f"A {name} native dtype evidence is incomplete")
    return reasons


def _a_stage_analysis(stage_data: Mapping[str, Any], source: Mapping[str, Any], *, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    records = _record_map(stage_data)
    rows = source["rows"]
    engineering_reasons: list[str] = _validate_stage_binding(stage_data, source)
    precision_results: dict[str, Any] = {}
    for precision in ("native", "temporary_fp32"):
        outputs: dict[str, np.ndarray] = {}
        floors: dict[str, float] = {}
        for name in ("baseline_pre", "v0_alpha_00_plus", "v0_alpha_00_minus",
                     "v0_alpha_01_plus", "v0_alpha_01_minus", "v0_alpha_02_plus",
                     "v0_alpha_02_minus", "baseline_post"):
            sample_id = f"A_{name}_{precision}"
            record = records.get(sample_id)
            if record is None:
                engineering_reasons.append(f"A missing record: {sample_id}")
                continue
            try:
                counts = _operation_counts(record, "A")
                if counts != {"G": 0, "D": 0, "E": 1}:
                    engineering_reasons.append(f"A {sample_id} operation counts differ: {counts}")
                output = _array_from_record(record, "encoded_condition")
            except EngineeringDataError as error:
                engineering_reasons.append(str(error)); continue
            outputs[name] = output
            if precision == "native" and not byte_equal(output, rows[name]["direct"]):
                engineering_reasons.append(f"A native output differs bytewise from saved direct condition: {name}")
            expected_shape = rows[name]["direct"].shape
            if tuple(output.shape) != tuple(expected_shape):
                engineering_reasons.append(f"A output shape differs from source direct condition: {name}")
            engineering_reasons.extend(_validate_encoder_evidence(
                record, rows[name], precision=precision, encoded=output, name=f"{name}/{precision}"))
        if len(outputs) != 8:
            continue
        baseline = outputs["baseline_pre"]
        direction = source["v0_condition"]
        # Each encoder's own baseline pair defines its measured FP32 floor.
        # It is deliberately computed only after both baseline tensors are
        # loaded; no record flag can manufacture a floor or a scientific pass.
        try:
            floor = rms64(fp32_difference(outputs["baseline_post"], outputs["baseline_pre"]),
                          np.ones_like(outputs["baseline_pre"], dtype=bool))
        except EngineeringDataError as error:
            engineering_reasons.append(f"A {precision} baseline floor invalid: {error}")
            continue
        plus_inputs = [rows[f"v0_alpha_{ordinal:02d}_plus"]["delta_condition"] for ordinal in range(3)]
        minus_inputs = [rows[f"v0_alpha_{ordinal:02d}_minus"]["delta_condition"] for ordinal in range(3)]
        plus_outputs = [outputs[f"v0_alpha_{ordinal:02d}_plus"] for ordinal in range(3)]
        minus_outputs = [outputs[f"v0_alpha_{ordinal:02d}_minus"] for ordinal in range(3)]
        try:
            window = one_direction_window(TASK7_ALPHAS, direction, plus_inputs, minus_inputs,
                                          plus_outputs, minus_outputs, baseline,
                                          source["condition_mask"], floor=floor,
                                          output_mask=np.ones_like(baseline, dtype=bool))
        except EngineeringDataError as error:
            engineering_reasons.append(f"A {precision} numerical input invalid: {error}")
            continue
        precision_results[precision] = {
            "status": window["status"], "floor": floor, "window": _json_safe(window),
            "points": [{key: value for key, value in point.items() if key != "central_q"}
                       for point in window["points"]],
        }
        for ordinal, point in enumerate(window["points"]):
            for sign in ("plus", "minus"):
                name = f"a_{precision}_{ordinal}_{sign}_response"
                base_name = f"v0_alpha_{ordinal:02d}_{sign}"
                tensors[name] = fp32_difference(outputs[base_name], baseline)
    # Native is a diagnostic comparison. The Task7 scientific release gate is
    # the saved temporary-FP32 path; a native scientific FAIL is descriptive
    # and must not block an otherwise valid FP32 result.
    fp32_result = precision_results.get("temporary_fp32")
    scientific_pass = bool(fp32_result is not None and not engineering_reasons
                           and fp32_result["status"] == "PASS")
    return {"engineering_pass": not engineering_reasons, "engineering_reasons": engineering_reasons,
            "scientific_pass": scientific_pass, "scientific_status": "PASS" if scientific_pass else "FAIL",
            "precision": precision_results,
            "native_scientific_status": precision_results.get("native", {}).get("status", "NOT_RUN"),
            "fp32_scientific_status": fp32_result["status"] if fp32_result is not None else "NOT_RUN",
            "a_scientific_pass": scientific_pass,
            "a_engineering_pass": not engineering_reasons}


def _b_stage_analysis(stage_data: Mapping[str, Any], source: Mapping[str, Any], a_result: Mapping[str, Any],
                      *, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    records = _record_map(stage_data)
    source_rows = source["rows"]
    engineering_reasons: list[str] = _validate_stage_binding(stage_data, source)
    noise_by_seed: dict[int, str] = {}
    trajectory_metrics: dict[str, Any] = {}
    condition_mask = source["condition_mask"]
    # Validate every record before deriving any deltas.  A scientific FAIL in
    # A does not suppress B's finite propagation measurements, but any wiring
    # or evidence failure does.
    for spec in stage_data["plan"]:
        sample_id = str(spec["sample_id"])
        record = records.get(sample_id)
        if record is None:
            engineering_reasons.append(f"B missing record: {sample_id}")
            continue
        try:
            counts = _operation_counts(record, "B")
            if counts != {"G": 1, "D": 1, "E": 1}:
                engineering_reasons.append(f"B {sample_id} operation counts differ: {counts}")
            condition = _array_from_record(record, "condition_input_fp32")
            encoded = _array_from_record(record, "encoded_condition")
            expected_condition = source_rows[str(spec["source_name"])] ["consumed_condition"] if int(spec["step_index"]) == 0 else None
            if expected_condition is not None and not byte_equal(condition, expected_condition):
                engineering_reasons.append(f"B step-0 condition differs from saved source: {sample_id}")
            chain_reasons = _check_condition_chain(record, condition, name=sample_id,
                                                   full_mask=source_rows["baseline_pre"]["mask"])
            engineering_reasons.extend(chain_reasons)
            evidence = record.get("evidence")
            noise = evidence.get("prediction_noise_hash") if isinstance(evidence, Mapping) else None
            seed = int(spec.get("seed", spec.get("step_index", -1)))
            if not isinstance(noise, str) or not noise:
                engineering_reasons.append(f"B {sample_id} lacks prediction_noise_hash")
            elif seed in noise_by_seed and noise_by_seed[seed] != noise:
                engineering_reasons.append(f"B seed-{seed} prediction noise is not paired: {sample_id}")
            elif isinstance(noise, str):
                noise_by_seed[seed] = noise
            if int(spec["step_index"]) == 0:
                full = _array_from_record(record, "full_latent")
                expected_full = _load_runner_array(source_rows[str(spec["source_name"])] ["raw"] / "output_full.npy",
                                                    name=f"B {sample_id} historical output")
                if not byte_equal(full, expected_full):
                    engineering_reasons.append(f"B step-0 full latent differs from historical output: {sample_id}")
        except EngineeringDataError as error:
            engineering_reasons.append(str(error))
    if set(noise_by_seed) >= {0, 1} and noise_by_seed[0] == noise_by_seed[1]:
        engineering_reasons.append("B seed-0 and seed-1 prediction noise identities must differ")
    if len(noise_by_seed) < 2:
        engineering_reasons.append("B does not provide paired seed-0/seed-1 noise identities")

    # A temporary-FP32 encoder output is the cross-stage first-step reference.
    # Native parity is checked by A independently; only this exact mapping is
    # allowed to carry A's local window into B/C.
    a_records: dict[str, Mapping[str, Any]] | None = None
    # The A records are not embedded in a_result; load them from the same run
    # so the adapter remains explicit and does not duplicate artifact paths.
    try:
        a_data = load_task7_stage(stage_data["run_dir"], "A")
        a_records = _record_map(a_data)
        for name in ("baseline_pre", "v0_alpha_00_plus", "v0_alpha_00_minus", "v0_alpha_01_plus",
                     "v0_alpha_01_minus", "v0_alpha_02_plus", "v0_alpha_02_minus", "baseline_post"):
            b_record = records.get(f"B_{name}_step_0")
            a_record = a_records.get(f"A_{name}_temporary_fp32")
            if b_record is None or a_record is None:
                engineering_reasons.append(f"B/A cross-stage record missing: {name}")
                continue
            b_encoded = _array_from_record(b_record, "encoded_condition")
            a_encoded = _array_from_record(a_record, "encoded_condition")
            if not byte_equal(b_encoded, a_encoded):
                engineering_reasons.append(f"B step-0 encoded condition differs from A FP32 encoder: {name}")
    except EngineeringDataError as error:
        engineering_reasons.append(f"B/A cross-stage check unavailable: {error}")

    baseline0 = records.get("B_baseline_pre_step_0")
    baseline1 = records.get("B_baseline_pre_step_1")
    baseline_post0 = records.get("B_baseline_post_step_0")
    baseline_post1 = records.get("B_baseline_post_step_1")
    baseline_repeatability_ok = False
    if baseline0 is None or baseline1 is None or baseline_post0 is None or baseline_post1 is None:
        engineering_reasons.append("B baseline-pre records are missing")
    else:
        try:
            b0_in = _array_from_record(baseline0, "condition_input_fp32")
            b0_out = _array_from_record(baseline0, "encoded_condition")
            b1_in = _array_from_record(baseline1, "condition_input_fp32")
            b1_out = _array_from_record(baseline1, "encoded_condition")
            # Floors are measured independently at z0/z1/z2 from the saved
            # baseline-pre/post pair, with FP32 subtraction and FP64 RMS.
            z0_floor = rms64(fp32_difference(
                _array_from_record(baseline_post0, "condition_input_fp32"), b0_in), condition_mask)
            z1_floor = rms64(fp32_difference(
                _array_from_record(baseline_post0, "encoded_condition"), b0_out), condition_mask)
            z2_floor = rms64(fp32_difference(
                _array_from_record(baseline_post1, "encoded_condition"), b1_out), condition_mask)
            if not byte_equal(b1_in, b0_out):
                engineering_reasons.append("B baseline step-1 input differs from baseline step-0 encoded condition")
            baseline_repeatability_ok = all((
                byte_equal(_array_from_record(baseline_post0, "condition_input_fp32"), b0_in),
                byte_equal(_array_from_record(baseline_post0, "encoded_condition"), b0_out),
                byte_equal(_array_from_record(baseline_post1, "condition_input_fp32"), b1_in),
                byte_equal(_array_from_record(baseline_post1, "encoded_condition"), b1_out),
            ))
            if not baseline_repeatability_ok:
                engineering_reasons.append("B baseline-pre/post replay is not byte-identical at z0/z1/z2")
            for name in ("v0_alpha_00_plus", "v0_alpha_00_minus", "v0_alpha_01_plus", "v0_alpha_01_minus",
                         "v0_alpha_02_plus", "v0_alpha_02_minus"):
                first = records.get(f"B_{name}_step_0")
                second = records.get(f"B_{name}_step_1")
                if first is None or second is None:
                    engineering_reasons.append(f"B trajectory records are missing: {name}")
                    continue
                d0 = fp32_difference(_array_from_record(first, "condition_input_fp32"), b0_in)
                d1 = fp32_difference(_array_from_record(first, "encoded_condition"), b0_out)
                d2 = fp32_difference(_array_from_record(second, "encoded_condition"), b1_out)
                tensors[f"b_{name}_delta0"] = d0
                tensors[f"b_{name}_delta1"] = d1
                tensors[f"b_{name}_delta2"] = d2
                metrics = propagation_metrics(d0, d1, d2, condition_mask,
                                              floor0=z0_floor, floor1=z1_floor, floor2=z2_floor)
                trajectory_metrics[name] = _json_safe(metrics)
        except EngineeringDataError as error:
            engineering_reasons.append(f"B propagation input invalid: {error}")
    finite_pass = bool(trajectory_metrics) and all(
        value.get("status") in {"PASS", "UNRELIABLE"} for value in trajectory_metrics.values())
    engineering_pass = not engineering_reasons
    repeatability_pass = engineering_pass and baseline_repeatability_ok and bool(trajectory_metrics) and len(trajectory_metrics) == 6
    return {"engineering_pass": engineering_pass, "engineering_reasons": engineering_reasons,
            "repeatability_pass": repeatability_pass, "b_engineering_pass": engineering_pass,
            "b_repeatability_pass": repeatability_pass, "finite_propagation": finite_pass,
            "trajectory": trajectory_metrics, "noise_by_seed": noise_by_seed,
            "floors": {"z0": locals().get("z0_floor", None), "z1": locals().get("z1_floor", None),
                       "z2": locals().get("z2_floor", None)}}


def _c_stage_analysis(stage_data: Mapping[str, Any], source: Mapping[str, Any], b_result: Mapping[str, Any],
                      *, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    records = _record_map(stage_data)
    reasons: list[str] = _validate_stage_binding(stage_data, source)
    condition_mask = source["condition_mask"]
    baseline_pre = records.get("C_baseline_pre")
    baseline_post = records.get("C_baseline_post")
    if baseline_pre is None or baseline_post is None:
        return {"engineering_pass": False, "engineering_reasons": ["C shared baseline records are missing"],
                "scientific_pass": False, "status": "ENGINEERING_FAIL", "rows": []}
    try:
        z1 = _array_from_record(baseline_pre, "condition_input_fp32")
        z2 = _array_from_record(baseline_pre, "encoded_condition")
        z2_post = _array_from_record(baseline_post, "encoded_condition")
        c_floor = rms64(fp32_difference(z2_post, z2), condition_mask)
        b_data = load_task7_stage(stage_data["run_dir"], "B")
        b_records = _record_map(b_data)
        b_z1 = _array_from_record(b_records["B_baseline_pre_step_0"], "encoded_condition")
        b_z2 = _array_from_record(b_records["B_baseline_pre_step_1"], "encoded_condition")
        if not byte_equal(z1, b_z1):
            reasons.append("C baseline input differs bytewise from B baseline step-0 encoded condition")
        if not byte_equal(z2, b_z2) or not byte_equal(z2_post, b_z2):
            reasons.append("C baseline output differs bytewise from B baseline step-1 encoded condition")
    except (EngineeringDataError, KeyError) as error:
        reasons.append(f"C/B baseline parity unavailable: {error}")
        z1 = z2 = None

    seed1_noise = None
    if isinstance(baseline_pre.get("evidence"), Mapping):
        seed1_noise = baseline_pre["evidence"].get("prediction_noise_hash")
    if not isinstance(seed1_noise, str) or not seed1_noise:
        reasons.append("C baseline lacks seed-1 prediction-noise identity")
    rows: list[dict[str, Any]] = []
    direction_metrics = b_result.get("trajectory", {}) if isinstance(b_result, Mapping) else {}
    for index in range(6):
        direction_name = f"delta1_{index:02d}"
        # B trajectories are ordered by source name, preserving all six signed
        # directions as independent rays; no original-v0 projection is used.
        source_name = ("v0_alpha_00_plus", "v0_alpha_00_minus", "v0_alpha_01_plus",
                       "v0_alpha_01_minus", "v0_alpha_02_plus", "v0_alpha_02_minus")[index]
        direction_record = b_records.get(f"B_{source_name}_step_0") if "b_records" in locals() else None
        if direction_record is None or z1 is None or z2 is None:
            reasons.append(f"C direction source is missing: {direction_name}")
            continue
        try:
            ray = fp32_difference(_array_from_record(direction_record, "encoded_condition"), b_z1)
            direction_floor = c_floor
            beta_records: dict[tuple[float, int], Mapping[str, Any]] = {}
            for beta in TASK7_BETAS:
                for sign, label in ((1, "plus"), (-1, "minus")):
                    sample_id = f"C_delta1_{index:02d}_beta_{beta:g}_{label}"
                    record = records.get(sample_id)
                    if record is None:
                        reasons.append(f"C missing record: {sample_id}"); continue
                    counts = _operation_counts(record, "C")
                    if counts != {"G": 1, "D": 1, "E": 1}:
                        reasons.append(f"C {sample_id} operation counts differ: {counts}")
                    evidence = record.get("evidence")
                    noise = evidence.get("prediction_noise_hash") if isinstance(evidence, Mapping) else None
                    if noise != seed1_noise:
                        reasons.append(f"C {sample_id} seed-1 prediction noise differs from B")
                    reasons.extend(_check_condition_chain(
                        record, _array_from_record(record, "condition_input_fp32"),
                        name=sample_id, full_mask=source["rows"]["baseline_pre"]["mask"]))
                    beta_records[(float(beta), int(sign))] = record
            plus_inputs: list[np.ndarray] = []
            minus_inputs: list[np.ndarray] = []
            plus_outputs: list[np.ndarray] = []
            minus_outputs: list[np.ndarray] = []
            for beta in TASK7_BETAS:
                plus = beta_records.get((beta, 1)); minus = beta_records.get((beta, -1))
                if plus is None or minus is None:
                    continue
                plus_inputs.append(fp32_difference(_array_from_record(plus, "condition_input_fp32"), z1))
                minus_inputs.append(fp32_difference(_array_from_record(minus, "condition_input_fp32"), z1))
                plus_outputs.append(_array_from_record(plus, "encoded_condition"))
                minus_outputs.append(_array_from_record(minus, "encoded_condition"))
            local_window = None
            if len(plus_inputs) == 3 and len(minus_inputs) == 3:
                local_window = one_direction_window(TASK7_BETAS, ray, plus_inputs, minus_inputs,
                                                     plus_outputs, minus_outputs, z2, condition_mask,
                                                     floor=direction_floor, output_mask=np.ones_like(z2, dtype=bool))
            beta1_plus = beta_records.get((0.1, 1)); beta1_minus = beta_records.get((0.1, -1))
            if beta1_plus is None or beta1_minus is None:
                continue
            actual_plus = fp32_difference(_array_from_record(beta1_plus, "encoded_condition"), z2)
            prediction = fixed_beta_prediction(
                actual_delta1=ray,
                beta_plus_input=fp32_difference(_array_from_record(beta1_plus, "condition_input_fp32"), z1),
                beta_minus_input=fp32_difference(_array_from_record(beta1_minus, "condition_input_fp32"), z1),
                beta_plus_output=_array_from_record(beta1_plus, "encoded_condition"),
                beta_minus_output=_array_from_record(beta1_minus, "encoded_condition"),
                baseline_output=z2, actual_delta2=actual_plus, mask=condition_mask,
                beta=0.1, local_window_pass=(local_window is not None and local_window["status"] == "PASS"),
                response_reliable=(local_window is not None and
                                   bool(local_window["points"][0]["response_reliable"]["plus"]) and
                                   bool(local_window["points"][0]["response_reliable"]["minus"])),
                output_mask=np.ones_like(z2, dtype=bool))
            tensors[f"c_{direction_name}_actual_delta2"] = actual_plus
            rows.append({"direction_id": direction_name, "status": prediction["status"],
                         "Eprop": prediction["Eprop"], "delta1_rms": prediction["delta1_rms"],
                         "local_window_status": None if local_window is None else local_window["status"],
                         "reliable": prediction["reliable"], "reasons": prediction["reasons"]})
        except EngineeringDataError as error:
            reasons.append(f"C {direction_name} numerical evidence invalid: {error}")
    engineering_pass = not reasons and len(rows) == 6
    scientific_pass = engineering_pass and all(row["status"] == "PASS" for row in rows)
    return {"engineering_pass": engineering_pass, "engineering_reasons": reasons,
            "scientific_pass": scientific_pass, "status": "PASS" if scientific_pass else "FAIL",
            "rows": rows, "c_engineering_pass": engineering_pass, "c_scientific_pass": scientific_pass}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(str(key))
        fieldnames = fields
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (Mapping, list, tuple)) else _json_safe(value)
                             for key, value in row.items()})


def _render_plots(root: Path, summary: Mapping[str, Any]) -> list[str]:
    """Render compact diagnostic charts when an existing matplotlib is usable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    created: list[str] = []
    precision = summary.get("A", {}).get("precision", {}) if isinstance(summary.get("A"), Mapping) else {}
    if precision:
        figure, axis = plt.subplots(figsize=(5.0, 3.5))
        for name, value in precision.items():
            points = value.get("points", []) if isinstance(value, Mapping) else []
            x = [float(point["h_plus"]) for point in points if point.get("h_plus") is not None]
            y = [float(point["plus_response_rms"]) for point in points if point.get("plus_response_rms") is not None]
            if x and y:
                axis.loglog(x, y, marker="o", label=name)
        axis.set_xlabel("actual condition step RMS")
        axis.set_ylabel("response RMS")
        axis.set_title("Task7 A response / precision (diagnostic)")
        axis.legend()
        figure.tight_layout()
        for extension in ("png", "svg"):
            filename = f"a_response_precision.{extension}"
            figure.savefig(root / filename, dpi=150 if extension == "png" else None)
            created.append(filename)
        plt.close(figure)
    b_summary = summary.get("B", {}).get("trajectory", {}) if isinstance(summary.get("B"), Mapping) else {}
    if b_summary:
        figure, axis = plt.subplots(figsize=(5.0, 3.5))
        names, a1, a2 = [], [], []
        for name, value in b_summary.items():
            names.append(str(name)); a1.append(value.get("A1")); a2.append(value.get("A2"))
        axis.plot(names, a1, "o-", label="A1")
        axis.plot(names, a2, "s-", label="A2")
        axis.set_xlabel("trajectory (positive/negative retained)")
        axis.set_ylabel("gain")
        axis.set_title("Task7 B propagation gains")
        axis.tick_params(axis="x", labelrotation=70)
        axis.legend(); figure.tight_layout()
        for extension in ("png", "svg"):
            filename = f"b_propagation_gains.{extension}"
            figure.savefig(root / filename, dpi=150 if extension == "png" else None)
            created.append(filename)
        plt.close(figure)
    c_rows = summary.get("C", {}).get("rows", []) if isinstance(summary.get("C"), Mapping) else []
    if c_rows:
        figure, axis = plt.subplots(figsize=(5.0, 3.5))
        axis.bar([str(row.get("direction_id")) for row in c_rows],
                 [float(row.get("Eprop") or 0.0) for row in c_rows])
        axis.axhline(0.10, color="red", linestyle="--", label="Eprop=.10")
        axis.set_xlabel("independent delta1 direction")
        axis.set_ylabel("Eprop")
        axis.set_title("Task7 C prediction error (diagnostic)")
        axis.tick_params(axis="x", labelrotation=70); axis.legend(); figure.tight_layout()
        for extension in ("png", "svg"):
            filename = f"c_prediction_errors.{extension}"
            figure.savefig(root / filename, dpi=150 if extension == "png" else None)
            created.append(filename)
        plt.close(figure)
    return created


def _write_report(root: Path, summary: Mapping[str, Any], *, plots: Sequence[str]) -> None:
    a = summary.get("A", {}) if isinstance(summary.get("A"), Mapping) else {}
    b = summary.get("B", {}) if isinstance(summary.get("B"), Mapping) else {}
    c = summary.get("C", {}) if isinstance(summary.get("C"), Mapping) else {}
    lines = [
        "# Task7 offline saved-evidence analysis",
        "",
        f"Analysis identity: `{summary.get('analysis_version')}`; analyzer source digest: `{summary.get('analysis_code_sha256')}`.",
        "This package is CPU-only re-analysis of saved tensors. It is not a model, GPU, accuracy, Jacobian, global-stability, or rank claim.",
        "",
        "## Stage status",
        "",
        f"- A engineering: `{a.get('engineering_pass')}`; scientific window: `{a.get('scientific_status')}`.",
        f"- B engineering: `{b.get('engineering_pass')}`; baseline repeatability: `{b.get('repeatability_pass')}`.",
        f"- C: `{c.get('status', 'SKIPPED')}`.",
        "",
        "## Precision and metrics",
        "",
        "Neural G/D/E computation is recorded as FP32. Approved P range normalization may use a float64 intermediate followed by FP32 storage; this report does not relabel that conversion as an all-FP32 scalar path.",
        "Differences use explicit FP32 subtraction. RMS, cosine, regression, and floors use FP64 reductions over the authoritative condition mask. Each three-amplitude window is fixed in advance; no point is dropped, and additivity is `NOT_TESTED`.",
        "A positive/negative input is measured with its actual signed step. Zero denominators remain null with an explicit reliability reason.",
        "",
        "## Controlled repeatability",
        "",
        "B uses the controlled repeated action with seeds 0 and 1. The analyzer requires paired prediction-noise identities, exact baseline-pre/post replay, and A/B first-step byte parity before permitting a C gate.",
        "",
        "## Mathematical note",
        "",
        "The central secant is `q=(F_plus-F_minus)/(h_plus+h_minus)` using actual masked steps. Under conditional differentiability, the local prediction error has the usual `O(h^2)+O(u/h)` structure; floor and mask gates decide whether that asymptotic expression is measurable. See `docs/task7-mathematical-interpretation.md`.",
        "",
        "## Figures",
        "",
    ]
    if plots:
        lines.extend(f"- `{plot}`" for plot in plots)
    else:
        lines.append("No plots were rendered because an existing matplotlib runtime was unavailable.")
    lines.extend(["", "Synthetic fixtures, when used by tests, are labelled test evidence and never presented as model results.", ""])
    text = "\n".join(lines)
    (root / "task7_report.md").write_text(text, encoding="utf-8")
    (root / "experiment_report.md").write_text(text, encoding="utf-8")


def _write_manifest(root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    excluded = {"MANIFEST.sha256", "review_bundle.zip"}
    lines = []
    for path in sorted((item for item in root.rglob("*") if item.is_file() and item.name not in excluded),
                       key=lambda item: item.relative_to(root).as_posix()):
        lines.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    return manifest


def _write_light_bundle(root: Path) -> Path:
    bundle = root / "review_bundle.zip"
    excluded = {"review_bundle.zip", "MANIFEST.sha256"}
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((item for item in root.rglob("*") if item.is_file() and item.name not in excluded),
                           key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            if path.suffix in {".mp4", ".mov", ".safetensors", ".ckpt", ".pt", ".pth"}:
                continue
            if "analysis_tensors" in path.parts and path.stat().st_size > 8 * 1024 * 1024:
                continue
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return bundle


def _stage_skip(reason: str) -> dict[str, Any]:
    return {"status": "SKIPPED", "engineering_pass": False, "scientific_pass": False,
            "engineering_reasons": [str(reason)]}


def _gate_for_c(run_dir: Path, source: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any] | None:
    """Create the exact runner-consumable C gate only when all prerequisites pass."""
    if not bool(a.get("a_scientific_pass")) or not bool(a.get("a_engineering_pass")):
        return None
    if not bool(b.get("b_engineering_pass")) or not bool(b.get("b_repeatability_pass")):
        return None
    module = _runner_module()
    try:
        a_digest = module._stage_completion_digests(run_dir, "A")
        b_digest = module._stage_completion_digests(run_dir, "B")
    except Exception:
        return None
    binding = _binding(load_task7_stage(run_dir, "B"))
    source_sha = binding.get("source", binding.get("source_tree_sha256"))
    code_sha = binding.get("task7_code_sha256")
    if not isinstance(source_sha, str) or not isinstance(code_sha, str) or not source_sha or not code_sha:
        return None
    if source.get("source_tree_sha256") != source_sha:
        return None
    return {"schema_version": "umi-task7-analysis-gate-v1", "analysis_version": ANALYSIS_VERSION,
            "analysis_code_sha256": _analysis_source_sha256(),
            "a_scientific_pass": True, "b_engineering_pass": True,
            "b_repeatability_pass": True, "source_sha256": source_sha,
            "code_sha256": code_sha,
            "a_run_status_sha256": a_digest["run_status_sha256"],
            "a_samples_manifest_sha256": a_digest["samples_manifest_sha256"],
            "b_run_status_sha256": b_digest["run_status_sha256"],
            "b_samples_manifest_sha256": b_digest["samples_manifest_sha256"]}


def analyze_task7_run(run_dir: str | Path, *, stage: str = "all",
                      raw_root: str | Path | None = None,
                      decoder_root: str | Path | None = None,
                      output_dir: str | Path | None = None) -> dict[str, Any]:
    """Analyze saved Task7 evidence, optionally stopping after A or B.

    ``stage='A'`` is the usable offline path while live B/C data are absent.
    A scientific FAIL is returned as a valid analysis result; engineering
    failures are explicit and never converted into zero-valued metrics.
    """
    requested = str(stage).upper()
    if requested not in {"A", "B", "C", "ALL"}:
        raise EngineeringDataError("stage must be A, B, C, or all")
    run = Path(run_dir).resolve()
    if raw_root is None or decoder_root is None:
        raise EngineeringDataError("raw_root and decoder_root are required for source-bound analysis")
    source = _load_source_evidence(raw_root, decoder_root)
    tensor_values: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION, "analysis_version": ANALYSIS_VERSION,
        "analysis_code_sha256": _analysis_source_sha256(), "run_dir": str(run),
        "source_tree_sha256": source.get("source_tree_sha256"),
        "source": {key: source.get(key) for key in ("raw_root", "decoder_root", "z0_sha256", "mask_sha256", "v0_sha256", "source_tree_sha256")},
    }
    try:
        a_data = load_task7_stage(run, "A")
        a = _a_stage_analysis(a_data, source, tensors=tensor_values)
    except EngineeringDataError as error:
        a = {"status": "ENGINEERING_FAIL", "a_engineering_pass": False,
             "a_scientific_pass": False, "engineering_pass": False,
             "engineering_reasons": [str(error)], "scientific_status": "NOT_RUN"}
    summary["A"] = a
    if requested == "A":
        summary["B"] = _stage_skip("not requested; A-only analysis")
        summary["C"] = _stage_skip("not requested; A-only analysis")
    else:
        try:
            b_data = load_task7_stage(run, "B")
            b = _b_stage_analysis(b_data, source, a, tensors=tensor_values)
        except EngineeringDataError as error:
            b = {"status": "ENGINEERING_FAIL", "b_engineering_pass": False,
                 "b_repeatability_pass": False, "engineering_pass": False,
                 "engineering_reasons": [str(error)]}
        summary["B"] = b
        gate = _gate_for_c(run, source, a, b)
        if gate is not None:
            summary["analysis_gate"] = gate
        if requested == "B":
            summary["C"] = _stage_skip("not requested; B-only analysis")
        elif gate is None:
            summary["C"] = _stage_skip("A scientific/engineering or B repeatability gate did not pass")
        else:
            try:
                gate_path = run / "analysis_gate.json"
                if not gate_path.is_file():
                    raise EngineeringDataError("C evidence exists without the analyzer gate consumed by the runner")
                existing_gate = json.loads(gate_path.read_text(encoding="utf-8"))
                if _canonical(existing_gate) != _canonical(gate):
                    raise EngineeringDataError("C analyzer gate lineage/source/code differs from current A/B evidence")
                c_data = load_task7_stage(run, "C")
                summary["C"] = _c_stage_analysis(c_data, source, b, tensors=tensor_values)
            except EngineeringDataError as error:
                c_root = run / "stages" / "C"
                if not c_root.exists():
                    summary["C"] = _stage_skip(f"C stage outputs are not present: {error}")
                else:
                    summary["C"] = {"status": "ENGINEERING_FAIL", "c_engineering_pass": False,
                                     "c_scientific_pass": False, "engineering_pass": False,
                                     "engineering_reasons": [str(error)]}

    destination = Path(output_dir).resolve() if output_dir is not None else run / "task7_analysis"
    for immutable in (Path(source["raw_root"]).resolve(), Path(source["decoder_root"]).resolve()):
        try:
            destination.relative_to(immutable)
        except ValueError:
            continue
        raise EngineeringDataError("analysis destination is inside an immutable source root")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"analysis destination already exists and is non-empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    tensor_root = destination / "analysis_tensors"
    tensor_root.mkdir()
    for name, value in sorted(tensor_values.items()):
        np.save(tensor_root / f"{name}.npy", np.ascontiguousarray(value, dtype=np.float32), allow_pickle=False)
    # CSVs consume only small scalar rows, never full tensors.
    a_rows: list[dict[str, Any]] = []
    a_geometry_rows: list[dict[str, Any]] = []
    a_secant_rows: list[dict[str, Any]] = []
    for precision, value in summary["A"].get("precision", {}).items():
        for point in value.get("points", []):
            a_rows.append({"precision": precision, "ordinal": point.get("ordinal"), "alpha": point.get("alpha"),
                           "h_plus": point.get("h_plus"), "h_minus": point.get("h_minus"),
                           "plus_response_rms": point.get("plus_response_rms"),
                           "minus_response_rms": point.get("minus_response_rms"),
                           "plus_input_cosine": point.get("plus_input_cosine"),
                           "minus_input_cosine": point.get("minus_input_cosine"),
                           "opposite_cosine": point.get("opposite_cosine")})
            a_geometry_rows.append({"precision": precision, "ordinal": point.get("ordinal"), "alpha": point.get("alpha"),
                                    "h_plus": point.get("h_plus"), "h_minus": point.get("h_minus"),
                                    "plus_input_cosine": point.get("plus_input_cosine"),
                                    "minus_input_cosine": point.get("minus_input_cosine"),
                                    "opposite_cosine": point.get("opposite_cosine"),
                                    "plus_outside_exact": point.get("plus_outside_exact"),
                                    "minus_outside_exact": point.get("minus_outside_exact")})
        for secant in value.get("window", {}).get("secants", []):
            a_secant_rows.append({"precision": precision, **secant})
    _write_csv(destination / "a_comparison.csv", a_rows)
    _write_csv(destination / "a_geometry.csv", a_geometry_rows)
    _write_csv(destination / "a_secants.csv", a_secant_rows)
    _write_csv(destination / "b_propagation.csv",
               [{"trajectory": name, **value} for name, value in summary.get("B", {}).get("trajectory", {}).items()])
    _write_csv(destination / "c_prediction.csv", summary.get("C", {}).get("rows", []))
    _write_json(destination / "analysis_summary.json", summary)
    if isinstance(summary.get("analysis_gate"), Mapping):
        _write_json(destination / "analysis_gate.json", summary["analysis_gate"])
        run_gate = run / "analysis_gate.json"
        if run_gate.exists():
            try:
                existing = json.loads(run_gate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise EngineeringDataError(f"existing run analysis_gate.json is unreadable: {error}") from error
            if _canonical(existing) != _canonical(summary["analysis_gate"]):
                raise EngineeringDataError("existing run analysis_gate.json is bound to different A/B evidence")
        else:
            _write_json(run_gate, summary["analysis_gate"])
    plots = _render_plots(destination, summary)
    _write_report(destination, summary, plots=plots)
    _write_light_bundle(destination)
    _write_manifest(destination)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline Task7 saved-evidence analysis")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--decoder-root", required=True)
    parser.add_argument("--stage", choices=("A", "B", "C", "all", "a", "b", "c"), default="all")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    try:
        result = analyze_task7_run(args.run_dir, stage=args.stage, raw_root=args.raw_root,
                                   decoder_root=args.decoder_root, output_dir=args.output_dir)
    except (EngineeringDataError, FileExistsError, OSError) as error:
        print(json.dumps({"status": "ENGINEERING_FAIL", "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(_json_safe(result), ensure_ascii=False, sort_keys=True))
    engineering_fail = any(isinstance(result.get(key), Mapping) and result[key].get("engineering_pass") is False
                            and result[key].get("status") == "ENGINEERING_FAIL" for key in ("A", "B", "C"))
    return 2 if engineering_fail else 0


__all__ = [
    "EngineeringDataError", "Task7EngineeringError", "arrays_byte_equal", "byte_equal",
    "compute_input_geometry", "compute_propagation", "cosine64", "evaluate_fixed_beta_predictions",
    "fit_loglog", "fixed_beta_prediction", "float32_difference", "fp32_difference",
    "input_geometry", "one_direction_window", "predict_fixed_beta", "propagation_metrics",
    "rms64", "analyze_one_direction_window", "load_task7_stage", "analyze_task7_run", "main",
]


if __name__ == "__main__":  # pragma: no cover - exercised by CLI smoke tests
    raise SystemExit(main())
