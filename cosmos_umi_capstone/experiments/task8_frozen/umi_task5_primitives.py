"""Numerical contracts for the bounded UMI Task 5 direction experiment."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


ALPHAS = (1e-3, 3e-3, 1e-2)
DIRECTION_IDS = ("v0", "v1", "v2", "u01", "u12")


def _array(value: Any, *, dtype: Any = np.float64, name: str = "value") -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and nonempty")
    return array


def rms64(value: Any) -> float:
    array = _array(value, name="RMS value")
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def cosine64(left: Any, right: Any) -> float | None:
    left_array = _array(left, name="left cosine value").reshape(-1)
    right_array = _array(right, name="right cosine value").reshape(-1)
    if left_array.shape != right_array.shape:
        raise ValueError("cosine values must have the same shape")
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return None
    return float(np.dot(left_array, right_array) / denominator)


def _masked_rms(direction: np.ndarray, mask: np.ndarray) -> float:
    return rms64(direction[mask])


def freeze_task5_directions(direction_bank: Any, mask: Any) -> dict[str, Any]:
    bank = _array(direction_bank, dtype=np.float32, name="direction bank")
    mask_array = np.asarray(mask, dtype=bool)
    if bank.ndim != mask_array.ndim + 1 or tuple(bank.shape[1:]) != tuple(mask_array.shape):
        raise ValueError("direction bank must have three leading directions matching mask shape")
    if bank.shape[0] < 3 or not np.any(mask_array):
        raise ValueError("Task 5 requires directions 0, 1, 2 and a nonempty mask")
    original = {f"v{index}": np.array(bank[index], dtype=np.float32, copy=True) for index in range(3)}
    for direction_id, direction in original.items():
        if np.any(direction[~mask_array]):
            raise ValueError(f"{direction_id} is nonzero outside the condition mask")
        if not np.isclose(_masked_rms(direction, mask_array), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"{direction_id} does not have unit masked RMS")

    def combo(left: str, right: str, output: str) -> tuple[np.ndarray, float]:
        raw = original[left].astype(np.float64) + original[right].astype(np.float64)
        coefficient = _masked_rms(raw, mask_array)
        if not np.isfinite(coefficient) or coefficient == 0.0:
            raise ValueError(f"{output} normalization denominator is invalid")
        normalized = (raw / coefficient).astype(np.float32)
        normalized[~mask_array] = 0.0
        if not np.isclose(_masked_rms(normalized, mask_array), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(f"{output} normalization did not produce unit masked RMS")
        return normalized, float(coefficient)

    u01, c01 = combo("v0", "v1", "u01")
    u12, c12 = combo("v1", "v2", "u12")
    directions = {**original, "u01": u01, "u12": u12}
    gram = np.empty((3, 3), dtype=np.float64)
    for row in range(3):
        for column in range(3):
            gram[row, column] = float(np.mean(original[f"v{row}"][mask_array].astype(np.float64) *
                                                original[f"v{column}"][mask_array].astype(np.float64), dtype=np.float64))
    return {"ids": DIRECTION_IDS, "directions": directions, "c01": c01, "c12": c12,
            "gram": gram, "masked_rms": {key: _masked_rms(value, mask_array) for key, value in directions.items()}}


def build_task5_call_plan() -> list[dict[str, Any]]:
    plan = [{"sample_id": "baseline_pre", "kind": "baseline", "alpha": 0.0, "sign": 0,
             "direction_id": None, "model_seed": 0}]
    for direction_id in DIRECTION_IDS:
        for ordinal, alpha in enumerate(ALPHAS):
            for sign, label in ((1, "plus"), (-1, "minus")):
                plan.append({"sample_id": f"{direction_id}_alpha_{ordinal:02d}_{label}", "kind": "perturbation",
                             "alpha": alpha, "sign": sign, "direction_id": direction_id, "model_seed": 0})
    plan.append({"sample_id": "baseline_post", "kind": "baseline", "alpha": 0.0, "sign": 0,
                 "direction_id": None, "model_seed": 0})
    if len(plan) != 32 or len({entry["sample_id"] for entry in plan}) != 32:
        raise AssertionError("Task 5 formal plan must contain exactly 32 unique calls")
    return plan


def build_decoder_replay_plan() -> list[dict[str, Any]]:
    plan = [{"sample_id": "baseline_pre", "kind": "baseline", "alpha": 0.0, "sign": 0,
             "direction_id": "v0"}]
    for ordinal, alpha in enumerate(ALPHAS):
        for sign, label in ((1, "plus"), (-1, "minus")):
            plan.append({"sample_id": f"v0_alpha_{ordinal:02d}_{label}", "kind": "perturbation",
                         "alpha": alpha, "sign": sign, "direction_id": "v0"})
    plan.append({"sample_id": "baseline_post", "kind": "baseline", "alpha": 0.0, "sign": 0,
                 "direction_id": "v0"})
    if len(plan) != 8:
        raise AssertionError("Task 5 decoder replay plan must be baseline, six v0 perturbations, baseline")
    return plan


def pair_additivity_metrics(combo_derivative: Any, left_derivative: Any, right_derivative: Any,
                             coefficient: float) -> dict[str, Any]:
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("combination coefficient must be finite and positive")
    combo = _array(combo_derivative, name="combo derivative")
    left = _array(left_derivative, name="left derivative")
    right = _array(right_derivative, name="right derivative")
    if combo.shape != left.shape or combo.shape != right.shape:
        raise ValueError("additivity derivatives must have identical shape")
    actual = coefficient * combo
    predicted = left + right
    residual = actual - predicted
    absolute = rms64(residual)
    denominator = rms64(left) + rms64(right)
    return {"absolute_rms": absolute, "relative_error": None if denominator == 0.0 else absolute / denominator,
            "denominator_rms_sum": denominator, "cosine": cosine64(actual, predicted),
            "residual": residual.astype(np.float32), "actual_combo": actual.astype(np.float32),
            "predicted_combo": predicted.astype(np.float32)}


def _prediction_derivatives(original: Mapping[str, Any], coefficients: Mapping[str, float]) -> dict[str, np.ndarray]:
    required = {"v0", "v1", "v2"}
    if set(original) != required:
        raise ValueError("small-amplitude estimate must contain exactly v0, v1, and v2")
    g = {key: _array(value, name=f"derivative {key}") for key, value in original.items()}
    c01, c12 = float(coefficients["c01"]), float(coefficients["c12"])
    if c01 <= 0.0 or c12 <= 0.0 or not np.isfinite(c01 + c12):
        raise ValueError("combination coefficients must be finite and positive")
    return {**g, "u01": (g["v0"] + g["v1"]) / c01, "u12": (g["v1"] + g["v2"]) / c12}


def heldout_prediction_metrics(actual: Mapping[tuple[str, float, int], Any], baseline: Any,
                               small_amplitude_derivatives: Mapping[str, Any], coefficients: Mapping[str, float], *,
                               h0: float, holdout_alphas: tuple[float, ...], combo_alphas: tuple[float, ...]) -> list[dict[str, Any]]:
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError("small estimation step must be finite and positive")
    y0 = _array(baseline, name="prediction baseline")
    derivatives = _prediction_derivatives(small_amplitude_derivatives, coefficients)
    rows: list[dict[str, Any]] = []
    requested = {"v0": tuple(holdout_alphas), "v1": tuple(holdout_alphas), "v2": tuple(holdout_alphas),
                 "u01": tuple(combo_alphas), "u12": tuple(combo_alphas)}
    for direction_id, magnitudes in requested.items():
        for magnitude in magnitudes:
            if not np.isfinite(magnitude) or magnitude <= 0.0:
                raise ValueError("prediction magnitudes must be finite and positive")
            for sign in (1, -1):
                key = (direction_id, float(magnitude), sign)
                if key not in actual:
                    raise ValueError(f"missing heldout actual output: {key}")
                observed = _array(actual[key], name="heldout output")
                if observed.shape != y0.shape or derivatives[direction_id].shape != y0.shape:
                    raise ValueError("heldout prediction tensors must match baseline shape")
                predicted = y0 + sign * magnitude * derivatives[direction_id]
                response = observed - y0
                denominator = rms64(response)
                residual = observed - predicted
                relative = None if denominator == 0.0 else rms64(residual) / denominator
                status = "N/A" if relative is None else ("PASS" if relative <= 0.10 else "FAIL")
                rows.append({"direction_id": direction_id, "magnitude": float(magnitude), "sign": sign,
                             "estimate_magnitude": float(h0), "absolute_rms": rms64(residual),
                             "response_rms": denominator, "relative_error": relative, "status": status,
                             "residual": residual.astype(np.float32), "prediction": predicted.astype(np.float32)})
    return rows


def linear_formula_calibration() -> dict[str, Any]:
    baseline = np.array([1.0, -2.0], dtype=np.float64)
    derivatives = {"v0": np.array([2.0, 0.0]), "v1": np.array([0.0, 3.0]), "v2": np.array([1.0, -1.0])}
    coefficients = {"c01": 2.0, "c12": 2.0}
    combo = {"u01": (derivatives["v0"] + derivatives["v1"]) / 2.0,
             "u12": (derivatives["v1"] + derivatives["v2"]) / 2.0}
    additions = [pair_additivity_metrics(combo["u01"], derivatives["v0"], derivatives["v1"], 2.0),
                 pair_additivity_metrics(combo["u12"], derivatives["v1"], derivatives["v2"], 2.0)]
    all_derivatives = {**derivatives, **combo}
    actual: dict[tuple[str, float, int], np.ndarray] = {}
    for direction_id, derivative in all_derivatives.items():
        for magnitude in (0.01, 0.03, 0.1):
            for sign in (1, -1):
                actual[(direction_id, magnitude, sign)] = baseline + sign * magnitude * derivative
    predictions = heldout_prediction_metrics(actual, baseline, derivatives, coefficients, h0=0.01,
                                             holdout_alphas=(0.03, 0.1), combo_alphas=(0.01, 0.03, 0.1))
    max_add = max(item["absolute_rms"] for item in additions)
    max_pred = max(item["absolute_rms"] for item in predictions)
    return {"status": "PASS_EXACT" if max_add == 0.0 and max_pred == 0.0 else "FAIL",
            "max_additivity_rms": max_add, "max_prediction_rms": max_pred,
            "additivity": additions, "prediction": predictions}


__all__ = ["ALPHAS", "DIRECTION_IDS", "build_decoder_replay_plan", "build_task5_call_plan", "cosine64",
           "freeze_task5_directions", "heldout_prediction_metrics", "linear_formula_calibration",
           "pair_additivity_metrics", "rms64"]
