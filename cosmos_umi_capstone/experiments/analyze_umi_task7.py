"""Pure NumPy numerical contracts for the Task7 FP32 feedback analysis.

This module deliberately has no artifact loading, model, filesystem, or CLI
code.  Task3b consumes these small functions when it adds evidence handling.
Inputs at the feedback interface are condition-carrier arrays and an explicit
boolean condition mask.  Differences are performed in FP32; scalar
reductions and fits are performed in FP64.
"""
from __future__ import annotations

import math
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
        "reliable": status == "PASS",
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
        if key in record:
            value = record[key]
            return None if value is None else bool(value)
        if isinstance(global_value, (list, tuple, np.ndarray)):
            if len(global_value) != 6:
                raise EngineeringDataError(f"{key} must contain one value per ray")
            value = global_value[index]
            return None if value is None else bool(value)
        return global_value

    for index, (record, q_value) in enumerate(zip(rays, q_by_ray)):
        ray = _float32(record["ray"], f"ray {index}", shape=first_ray.shape)
        actual = _float32(record["actual_delta2"], f"actual delta2 ray {index}")
        q = _float32(q_value, f"q ray {index}", shape=actual.shape)
        d1_rms = rms64(ray, input_mask)
        pred, error = _prediction_from_q(q, d1_rms, actual,
                                         _mask(selected_output_mask, actual.shape, name="output mask"))
        reasons: list[str] = []
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
        rows.append({"ray_index": index, "status": status, "Eprop": error,
                     "delta1_rms": d1_rms, "predicted_delta2": pred,
                     "reasons": reasons})
    return rows


__all__ = [
    "EngineeringDataError", "Task7EngineeringError", "arrays_byte_equal", "byte_equal",
    "compute_input_geometry", "compute_propagation", "cosine64", "evaluate_fixed_beta_predictions",
    "fit_loglog", "fixed_beta_prediction", "float32_difference", "fp32_difference",
    "input_geometry", "one_direction_window", "predict_fixed_beta", "propagation_metrics",
    "rms64", "analyze_one_direction_window",
]
