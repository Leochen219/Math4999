"""CPU-only analysis and deterministic packaging for the UMI precision contrast.

The runner in :mod:`umi_precision_runtime` owns model execution and evidence
capture.  This module deliberately consumes detached JSON/NumPy records only:
it never imports Cosmos, never loads a checkpoint, and never reruns a sample.
It computes latent and (when available) fixed-decode RGB responses separately,
keeps all undefined values as explicit ``None`` plus a reason, and publishes a
small review bundle without copying the remote tensor corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shlex
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

import numpy as np

try:
    from .umi_precision_primitives import fixed_noise_hash
    from .umi_precision_reanalysis import _atomic_rename_noreplace, is_task4_raw_manifest_excluded
except ImportError:  # direct import from experiments/
    from umi_precision_primitives import fixed_noise_hash
    from umi_precision_reanalysis import _atomic_rename_noreplace, is_task4_raw_manifest_excluded


ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
GROUPS = ("A", "B", "C")
SPACES = ("predicted_latent", "decoded_final_rgb")
MIN_WINDOW = 3
FORMAL_CALL_COUNT = 42
THRESHOLDS = {
    "adjacent_input_cosine": 0.99,
    "input_cosine": 0.99,
    "plus_minus_input_cosine": 0.99,
    "opposition_cosine": 0.99,
    "slope_min": 0.8,
    "slope_max": 1.2,
    "slope_r2": 0.98,
    "paired_secant_cosine": 0.95,
    "paired_secant_relative_change": 0.25,
    "response_floor_multiple": 10.0,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_PYTHON = "<recorded-python-path>"
_REMOTE_SOURCE = "<recorded-source-directory>"
_REMOTE_FRAMEWORK = "<recorded-framework-root>"
_REMOTE_CHECKPOINT = "<recorded-checkpoint-path>"
_REMOTE_OLD_RUN = "<recorded-old-run-directory>"
_CALIBRATION_SOURCE_FILES = (
    "calibration_report.md", "config.json", "error_plot.png", "error_plot.svg",
    "hashes.json", "metrics.csv", "metrics.json",
)
_CALIBRATION_FILES = set(_CALIBRATION_SOURCE_FILES) | {"umi_precision_calibration_math.md"}
# The review bundle is deliberately a small, auditable evidence slice.  These
# are source names that are part of the Task 4 runner/bridge/patch surface;
# arbitrary files from a source checkout are never copied.
_REVIEW_SOURCE_FILES = (
    "__init__.py", "analyze_umi_precision_contrast.py", "package_umi_precision_contrast.py",
    "run_umi_precision_experiment.py", "umi_fd_post_vae_bridge.py", "umi_fd_post_vae_scan.py",
    "umi_precision_official.py", "umi_precision_runtime.py", "umi_precision_storage.py",
    "umi_precision_primitives.py", "umi_precision_reanalysis.py", "umi_precision_identity.py",
    "umi_precision_calibration.py", "umi_precision_calibration_math.md",
)
_REVIEW_SOURCE_OPTIONAL_FILES = {"__init__.py", "package_umi_precision_contrast.py"}
_REVIEW_RUN_FILES = ("config.json", "status.json", "task4_execution.json", "provenance.json")
_REVIEW_RUN_OPTIONAL_FILES = {"provenance.json"}
_REVIEW_SAMPLE_ARRAY_FILES = (
    "common_input_fp32.npy", "z_bar.npy", "mask.npy", "direction.npy",
    "initial_state.npy", "consumed_initial_state.npy", "sampler_input_state.npy",
    "consumed_initial_mask.npy", "actual_delta_fp32.npy", "target_delta_fp32.npy",
    "predicted_latent.npy", "decoded_final.npy",
)
_REVIEW_REQUIRED_SAMPLE_ARRAY_FILES = (
    "common_input_fp32.npy", "z_bar.npy", "mask.npy", "direction.npy",
    "initial_state.npy", "consumed_initial_state.npy", "sampler_input_state.npy",
    "predicted_latent.npy",
)
_REVIEW_SAMPLE_IDS = tuple(
    [f"{group}_pre" for group in GROUPS]
    + [f"{group}_alpha_{index:02d}_{sign}" for group in GROUPS for index in (0, 5) for sign in ("plus", "minus")]
)
_REQUIRED_ANALYSIS_FILES = {
    "precision_summary.json", "candidate_decisions.json", "precision_metrics.csv", "paired_secant_metrics.csv",
    "adjacent_consistency.csv", "fit_metrics.csv", "window_decisions.csv", "vector_decomposition.csv",
    "precision_report_zh.md", "precision_path_note.md", "commands.md", "gpu_timing_summary.json", "gpu_timing_summary.csv",
    "calibration_manifest.sha256", "input_distortion.png", "input_distortion.svg", "latent_rgb_response.png", "latent_rgb_response.svg",
    "derivative_consistency.png", "derivative_consistency.svg", "delta_decomposition.png", "delta_decomposition.svg",
    "response_rms.png", "response_rms.svg", "window_decisions.png", "window_decisions.svg",
    "precision_response.png", "precision_response.svg", "precision_windows.png", "precision_windows.svg",
    "review_evidence_manifest.json",
}


def _array(value: Any, *, name: str = "value", dtype: Any | None = None) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} is missing")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value, dtype=dtype)
    if result.size == 0 or not np.all(np.isfinite(result.astype(np.float64, copy=False))):
        raise ValueError(f"{name} is empty or non-finite")
    return np.ascontiguousarray(result)


def _rms(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not np.all(np.isfinite(array)):
        return None
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def _cosine(left: Any, right: Any) -> float | None:
    try:
        a = np.asarray(left, dtype=np.float64).reshape(-1)
        b = np.asarray(right, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.size == 0 or a.size != b.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0.0 else float(np.dot(a, b) / denominator)


def _relative_change(left: Any, right: Any) -> tuple[float | None, str | None]:
    left_rms = _rms(left)
    right_array = np.asarray(right, dtype=np.float64)
    left_array = np.asarray(left, dtype=np.float64)
    if left_rms is None:
        return None, "left denominator is nonfinite or missing"
    if left_rms == 0.0:
        return None, "left denominator is zero"
    if left_array.shape != right_array.shape or not np.all(np.isfinite(right_array)):
        return None, "adjacent tensors are missing, nonfinite, or shape-mismatched"
    return float(_rms(right_array - left_array) / left_rms), None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(_canonical(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _spec(record: Mapping[str, Any]) -> dict[str, Any]:
    raw = record.get("spec") if isinstance(record.get("spec"), Mapping) else record
    group = str(raw.get("group", record.get("group", "")))
    kind = str(raw.get("kind", record.get("kind", "")))
    try:
        alpha = float(raw.get("alpha", record.get("alpha", 0.0)))
        sign = int(raw.get("sign", record.get("sign", 0)))
        direction = int(raw.get("direction_index", record.get("direction_index", 0)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid precision sample spec: {record.get('sample_id')}") from error
    sample_id = str(record.get("sample_id", raw.get("sample_id", "")))
    if group not in GROUPS or kind not in {"pre", "post", "perturbation"}:
        raise ValueError(f"invalid precision sample spec: {sample_id}")
    if not math.isfinite(alpha) or alpha < 0.0 or sign not in {-1, 0, 1}:
        raise ValueError(f"invalid precision sample alpha/sign: {sample_id}")
    if kind == "perturbation" and (alpha <= 0.0 or sign not in {-1, 1}):
        raise ValueError(f"perturbation sample must have positive alpha and signed direction: {sample_id}")
    if kind in {"pre", "post"} and (alpha != 0.0 or sign != 0):
        raise ValueError(f"baseline sample must have zero alpha/sign: {sample_id}")
    return {"sample_id": sample_id, "group": group, "kind": kind, "alpha": alpha, "sign": sign, "direction_index": direction}


def _output(record: Mapping[str, Any], space: str) -> np.ndarray | None:
    names = ("predicted_latent", "step_0_predicted_denoiser_output", "output_full") if space == "predicted_latent" else ("decoded_final",)
    for name in names:
        if record.get(name) is not None:
            try:
                return _array(record[name], name=f"{space}:{name}", dtype=np.float32)
            except (TypeError, ValueError):
                return None
    return None


def _input_parts(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    common = _array(record.get("common_input_fp32"), name="common_input_fp32", dtype=np.float32)
    z_bar = _array(record.get("z_bar", record.get("common_input_fp32")), name="z_bar", dtype=np.float32)
    if z_bar.shape != common.shape:
        raise ValueError("z_bar and common input shapes differ")
    mask_value = record.get("mask", record.get("consumed_initial_mask"))
    if mask_value is None:
        raise ValueError("authoritative condition mask is missing")
    mask = np.asarray(mask_value, dtype=bool)
    if mask.shape != common.shape or not np.any(mask) or not np.any(~mask):
        raise ValueError("authoritative condition mask must match input and contain both regions")
    direction_value = record.get("direction")
    direction = None if direction_value is None else _array(direction_value, name="direction", dtype=np.float32)
    if direction is not None and direction.shape != common.shape:
        raise ValueError("direction shape differs from common input")
    return common, z_bar, np.ascontiguousarray(mask), direction


def _compute_signature(record: Mapping[str, Any]) -> str | None:
    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        return None
    signature = {
        "backend": execution.get("backend"),
        "casts": execution.get("casts", []),
        "operation_count": execution.get("operation_count"),
        "operation_dtypes": execution.get("operation_dtypes", {}),
        "dispatch_observed": execution.get("dispatch_observed"),
        "autocast": execution.get("autocast"),
        "tf32_matmul": execution.get("tf32_matmul"),
        "tf32_cudnn": execution.get("tf32_cudnn"),
    }
    return hashlib.sha256(_canonical(signature).encode("utf-8")).hexdigest()


def _validate_record(record: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    try:
        common, z_bar, mask, _ = _input_parts(record)
        if not np.array_equal(common[~mask], z_bar[~mask]):
            reasons.append("common interface changed outside authoritative condition mask")
    except (TypeError, ValueError) as error:
        reasons.append(str(error))
        return reasons
    cache = record.get("cache")
    if not isinstance(cache, Mapping):
        reasons.append("cache evidence is missing")
    else:
        if bool(cache.get("requested")) or bool(cache.get("installed")):
            reasons.append("diffusion cache was requested or installed")
        if not bool(cache.get("initial_empty", False)) or not bool(cache.get("final_empty", False)):
            reasons.append("request-local cache was not empty at both boundaries")
    execution = record.get("execution")
    if not isinstance(execution, Mapping):
        reasons.append("execution evidence is missing")
    else:
        for key in ("autocast", "tf32_matmul", "tf32_cudnn"):
            if bool(execution.get(key)):
                reasons.append(f"{key} was enabled")
        if not execution.get("dispatch_observed") or int(execution.get("operation_count", 0) or 0) <= 0:
            reasons.append("actual compute dispatch evidence is missing")
    initial = record.get("initial_state")
    for name in ("consumed_initial_state", "sampler_input_state"):
        if initial is None or record.get(name) is None:
            reasons.append(f"actual consumed initial noise evidence is missing: {name}")
        else:
            try:
                if not np.array_equal(_array(initial, name="initial_state"), _array(record[name], name=name)):
                    reasons.append(f"actual consumed initial noise differs from frozen preparation: {name}")
            except (TypeError, ValueError) as error:
                reasons.append(str(error))
    noise = record.get("noise_evidence")
    if noise != {"source": "first_velocity_input", "seed": 0, "prepare_seed": 0, "batch_size": 1}:
        reasons.append("noise evidence is not the registered first-velocity seed-zero policy")
    if record.get("scope") == "module" and record.get("decoded_final") is not None:
        reasons.append("module fallback unexpectedly contains decoded RGB output")
    for name in ("common_input_fp32", "z_bar", "mask"):
        if record.get(name) is None:
            reasons.append(f"required tensor evidence is missing: {name}")
    return reasons


def decompose_precision_responses(r_A: Any, r_B: Any, r_C: Any) -> dict[str, Any]:
    """Return signed quantization/compute deltas and verify their vector sum."""

    a = np.asarray(r_A, dtype=np.float64)
    b = np.asarray(r_B, dtype=np.float64)
    c = np.asarray(r_C, dtype=np.float64)
    if a.shape != b.shape or a.shape != c.shape:
        raise ValueError("A/B/C response shapes differ")
    delta_quant = b - c
    delta_compute = a - b
    total = a - c
    reconstructed = delta_quant + delta_compute
    error = reconstructed - total
    total_rms = _rms(total)
    return {
        "Delta_quant": delta_quant,
        "Delta_compute": delta_compute,
        "r_A_minus_r_C": total,
        "reconstructed": reconstructed,
        "identity_error_rms": _rms(error) or 0.0,
        "identity_passed": bool(np.array_equal(reconstructed, total)),
        "Delta_quant_rms": _rms(delta_quant),
        "Delta_compute_rms": _rms(delta_compute),
        "r_A_minus_r_C_rms": total_rms,
        "Delta_quant_relative_norm": None if total_rms in (None, 0.0) else float(_rms(delta_quant) / total_rms),
        "Delta_compute_relative_norm": None if total_rms in (None, 0.0) else float(_rms(delta_compute) / total_rms),
        "relative_norm_reason": "r_A-r_C denominator is zero or undefined" if total_rms in (None, 0.0) else None,
    }


def _fit_loglog(points: Iterable[Mapping[str, Any]], sign: int | None = None) -> dict[str, Any]:
    selected = [row for row in points if sign is None or row.get("sign") == sign]
    x: list[float] = []
    y: list[float] = []
    exclusions: list[str] = []
    for row in selected:
        input_rms = row.get("input_rms")
        output_rms = row.get("output_rms")
        if (input_rms is None or output_rms is None or not math.isfinite(float(input_rms))
                or not math.isfinite(float(output_rms)) or input_rms <= 0.0 or output_rms <= 0.0):
            exclusions.append(str(row.get("sample_id", row.get("alpha"))))
            continue
        x.append(math.log10(float(input_rms)))
        y.append(math.log10(float(output_rms)))
    if len(x) < 2:
        return {"slope": None, "intercept": None, "r2": None, "points": len(x), "exclusions": exclusions,
                "reason": "fewer than two finite positive input/output points"}
    slope, intercept = np.polyfit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), 1)
    predicted = slope * np.asarray(x) + intercept
    residual = np.asarray(y) - predicted
    denominator = float(np.sum((np.asarray(y) - np.mean(y)) ** 2))
    r2 = None if denominator == 0.0 else float(1.0 - np.sum(residual ** 2) / denominator)
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "points": len(x), "exclusions": exclusions, "reason": None}


def evaluate_window(rows: Iterable[Mapping[str, Any]], *, min_length: int = MIN_WINDOW) -> dict[str, Any]:
    """Apply the pre-registered window thresholds to a compact evidence table."""

    values = list(rows)
    failures: list[str] = []
    if len(values) < min_length:
        failures.append(f"window contains {len(values)} points; at least {min_length} are required")

    def _metric(name: str, aliases: tuple[str, ...] = ()) -> list[float | None]:
        keys = (name,) + aliases
        result: list[float | None] = []
        for row in values:
            value = next((row.get(key) for key in keys if row.get(key) is not None), None)
            try:
                converted = None if value is None else float(value)
                result.append(converted)
            except (TypeError, ValueError):
                result.append(None)
        return result

    def _finite_or_na(value: float | None) -> bool:
        return value is not None and math.isfinite(float(value))

    plus_input_cosines = _metric("plus_input_cosine", ("input_cosine", "adjacent_input_cosine"))
    opposition = _metric("plus_minus_input_cosine", ("opposition_cosine",))
    paired_cosines = _metric("paired_secant_cosine",)
    paired_relative = _metric("paired_secant_relative_change", ("paired_secant_relative_rms",))
    for index, value in enumerate(plus_input_cosines):
        if index == len(plus_input_cosines) - 1 and value is None:
            continue  # no right-adjacent edge after the trailing alpha
        if not _finite_or_na(value) or value < THRESHOLDS["adjacent_input_cosine"]:
            failures.append(f"plus_input_cosine at index {index} is below 0.99 or N/A")
    minus_input_cosines = _metric("minus_input_cosine", ("input_cosine", "adjacent_input_cosine"))
    for index, value in enumerate(minus_input_cosines):
        if index == len(minus_input_cosines) - 1 and value is None:
            continue
        if not _finite_or_na(value) or value < THRESHOLDS["adjacent_input_cosine"]:
            failures.append(f"minus_input_cosine at index {index} is below 0.99 or N/A")
    for index, value in enumerate(opposition):
        if not _finite_or_na(value) or value < THRESHOLDS["plus_minus_input_cosine"]:
            failures.append(f"plus_minus_input_cosine at index {index} is below 0.99 or N/A")
    for index, value in enumerate(paired_cosines):
        if index == len(paired_cosines) - 1 and value is None:
            continue
        if not _finite_or_na(value) or value < THRESHOLDS["paired_secant_cosine"]:
            failures.append("paired_secant_cosine")
            failures.append(f"paired_secant_cosine at index {index} is below 0.95 or N/A")
    for index, value in enumerate(paired_relative):
        if index == len(paired_relative) - 1 and value is None:
            continue
        if not _finite_or_na(value) or value > THRESHOLDS["paired_secant_relative_change"]:
            failures.append(f"paired_secant_relative_change at index {index} is above 0.25 or N/A")

    slopes = []
    for label in ("plus", "minus"):
        slope_values = _metric(f"{label}_slope", ("slope" if label == "plus" else "",))
        r2_values = _metric(f"{label}_r2", ("r2" if label == "plus" else "",))
        slope = next((value for value in slope_values if value is not None), None)
        r2 = next((value for value in r2_values if value is not None), None)
        slopes.append({"sign": label, "slope": slope, "r2": r2})
        if not _finite_or_na(slope) or not THRESHOLDS["slope_min"] <= slope <= THRESHOLDS["slope_max"]:
            failures.append(f"{label} slope is outside [0.8,1.2] or N/A")
        if not _finite_or_na(r2) or r2 < THRESHOLDS["slope_r2"]:
            failures.append(f"{label} slope R2 is below 0.98 or N/A")

    response_values = _metric("response_rms")
    floor_values = _metric("group_repeat_floor")
    for index, (response, floor) in enumerate(zip(response_values, floor_values)):
        if not _finite_or_na(response):
            failures.append(f"response at index {index} is N/A")
        elif not _finite_or_na(floor):
            failures.append(f"group repeat floor at index {index} is N/A")
        elif floor == 0.0:
            if response == 0.0:
                failures.append(f"response at index {index} is zero while the repeat floor is zero")
        elif response <= THRESHOLDS["response_floor_multiple"] * floor:
            failures.append(f"response at index {index} is not above 10x group repeat floor")
    return {
        "selected": not failures,
        "thresholds": dict(THRESHOLDS),
        "failure_reasons": failures,
        "slopes": slopes,
        "point_count": len(values),
    }


def _record_point(record: Mapping[str, Any], *, baseline: np.ndarray, space: str) -> dict[str, Any] | None:
    spec = _spec(record)
    output = _output(record, space)
    if output is None:
        return None
    if output.shape != baseline.shape:
        return None
    if space == "decoded_final_rgb" and (output.ndim != 3 or output.shape[0] != 3):
        return None
    common, z_bar, mask, direction = _input_parts(record)
    input_delta = (common.astype(np.float64) - z_bar.astype(np.float64))[mask]
    target_rms = float(spec["alpha"] * (_rms(z_bar[mask]) or 0.0))
    input_rms = _rms(input_delta)
    response = output.astype(np.float64) - baseline.astype(np.float64)
    output_rms = _rms(response)
    direction_cosine = None
    if direction is not None and spec["sign"]:
        direction_cosine = _cosine(input_delta, (float(spec["sign"]) * direction[mask]).astype(np.float64))
    return {
        "sample_id": spec["sample_id"], "group": spec["group"], "space": space,
        "kind": spec["kind"], "alpha": spec["alpha"], "sign": spec["sign"],
        "direction_index": spec["direction_index"], "target_input_rms": target_rms,
        "input_rms": input_rms, "output_rms": output_rms,
        "gain": None if input_rms in (None, 0.0) else float(output_rms / input_rms),
        "gain_reason": "actual input RMS is zero or undefined" if input_rms in (None, 0.0) else None,
        "input_nonzero_ratio": None if input_delta.size == 0 else float(np.count_nonzero(input_delta) / input_delta.size),
        "input_direction_cosine": direction_cosine,
        "input_direction_cosine_reason": "direction or input norm is zero/missing" if direction_cosine is None else None,
        "target_step_rms": target_rms,
        "output_rms_reason": "response is zero" if output_rms == 0.0 else None,
        "_input": np.asarray(input_delta, dtype=np.float64),
        "_response": np.asarray(response, dtype=np.float64),
        "_target": np.asarray((common.astype(np.float64) - z_bar.astype(np.float64))[mask]),
    }


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def _pair_metrics(points: list[dict[str, Any]], floor: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_alpha = {float(row["alpha"]): row for row in points if row["sign"] == 1}
    by_minus = {float(row["alpha"]): row for row in points if row["sign"] == -1}
    pairs: list[dict[str, Any]] = []
    for alpha in sorted(set(by_alpha) & set(by_minus)):
        plus, minus = by_alpha[alpha], by_minus[alpha]
        input_plus = plus["_input"]
        input_minus = minus["_input"]
        response_plus = plus["_response"]
        response_minus = minus["_response"]
        paired_step = _rms(input_plus - input_minus)
        centered_input = (input_plus - input_minus) / 2.0
        centered_response = (response_plus - response_minus) / 2.0
        secant = None if paired_step in (None, 0.0) else (response_plus - response_minus) / paired_step
        target_step = plus.get("target_step_rms")
        target_derivative = None if target_step in (None, 0.0) else centered_response / float(target_step)
        target_derivative_reason = (
            "target alpha step is zero, nonfinite, or missing" if target_derivative is None else None
        )
        plus_step = _rms(input_plus)
        minus_step = _rms(-input_minus)
        plus_derivative = None if plus_step in (None, 0.0) else response_plus / float(plus_step)
        minus_derivative = None if minus_step in (None, 0.0) else -response_minus / float(minus_step)
        if plus_derivative is None or minus_derivative is None:
            one_sided_cosine = None
            one_sided_relative, one_sided_reason = None, "one-sided actual input step is zero, nonfinite, or missing"
        else:
            one_sided_cosine = _cosine(plus_derivative, minus_derivative)
            one_sided_relative, one_sided_reason = _relative_change(plus_derivative, minus_derivative)
            if one_sided_cosine is None and one_sided_reason is None:
                one_sided_reason = "one-sided derivative norm is zero or undefined"
        pair = {
            "space": plus["space"], "group": plus["group"], "alpha": alpha,
            "plus_minus_input_cosine": _cosine(input_plus, -input_minus),
            "plus_minus_input_cosine_reason": "one signed actual input norm is zero or undefined" if _cosine(input_plus, -input_minus) is None else None,
            "plus_minus_response_cosine": _cosine(response_plus, -response_minus),
            "target_centered_input_rms": _rms(centered_input),
            "target_step_rms": target_step,
            "target_centered_derivative_rms": _rms(target_derivative),
            "target_centered_derivative_reason": target_derivative_reason,
            "actual_centered_response_rms": _rms(centered_response),
            "even_symmetry_residual": _rms(response_plus + response_minus),
            "paired_step_rms": paired_step,
            "paired_secant_rms": _rms(secant),
            "paired_secant_reason": "paired plus/minus input step is zero, nonfinite, or missing" if secant is None else None,
            "actual_paired_secant_rms": _rms(secant),
            "actual_paired_secant_reason": "paired plus/minus input step is zero, nonfinite, or missing" if secant is None else None,
            "one_sided_plus_step_rms": plus_step,
            "one_sided_minus_step_rms": minus_step,
            "one_sided_derivative_cosine": one_sided_cosine,
            "one_sided_relative_change": one_sided_relative,
            "one_sided_consistency_reason": one_sided_reason,
            "response_rms": min(value for value in (plus.get("output_rms"), minus.get("output_rms")) if value is not None) if plus.get("output_rms") is not None and minus.get("output_rms") is not None else None,
            "group_repeat_floor": floor,
            "_target_centered_derivative": target_derivative,
            "_paired_secant": secant,
            "_one_sided_plus_derivative": plus_derivative,
            "_one_sided_minus_derivative": minus_derivative,
        }
        pairs.append(pair)
    adjacent: list[dict[str, Any]] = []
    for left, right in zip(pairs, pairs[1:]):
        cosine = _cosine(left.get("_paired_secant"), right.get("_paired_secant")) if left.get("_paired_secant") is not None and right.get("_paired_secant") is not None else None
        relative, relative_reason = _relative_change(left.get("_paired_secant"), right.get("_paired_secant")) if left.get("_paired_secant") is not None and right.get("_paired_secant") is not None else (None, "paired secant tensor is undefined")
        adjacent.append({
            "space": left["space"], "group": left["group"], "alpha_left": left["alpha"], "alpha_right": right["alpha"],
            "paired_secant_cosine": cosine,
            "paired_secant_cosine_reason": "one paired secant is undefined" if cosine is None else None,
            "paired_secant_relative_change": relative,
            "paired_secant_relative_change_reason": relative_reason,
        })
    return pairs, adjacent


def _make_windows(points: list[dict[str, Any]], pairs: list[dict[str, Any]], paired_adjacent: list[dict[str, Any]], floor: float | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    group = points[0]["group"] if points else (pairs[0]["group"] if pairs else "")
    space = points[0]["space"] if points else (pairs[0]["space"] if pairs else "")
    point_by_sign = {sign: sorted([row for row in points if row["sign"] == sign], key=lambda row: row["alpha"]) for sign in (-1, 1)}
    pair_by_alpha = {float(row["alpha"]): row for row in pairs}
    pair_adj_by_right = {float(row["alpha_right"]): row for row in paired_adjacent}
    for sign in (-1, 1):
        fit = _fit_loglog(point_by_sign[sign], sign=sign)
        fits.append({"space": space, "group": group, "sign": sign, **fit})
    # Candidate windows are slices of the pre-registered alpha grid, never of
    # the observed points.  A missing interior alpha therefore yields an
    # explicit N/A decision rather than a silently bridged window.
    for length in range(MIN_WINDOW, len(ALPHAS) + 1):
        for start in range(0, len(ALPHAS) - length + 1):
            window_alphas = list(ALPHAS[start:start + length])
            missing = [
                alpha for alpha in window_alphas
                if not all(any(row["alpha"] == alpha and row["sign"] == sign for row in point_by_sign[sign])
                           for sign in (-1, 1))
            ]
            if missing:
                decisions.append({
                    "selected": False, "thresholds": dict(THRESHOLDS),
                    "failure_reasons": ["missing required fixed alpha(s): " + ", ".join(f"{alpha:g}" for alpha in missing)],
                    "slopes": [], "point_count": 0, "space": space, "group": group,
                    "start_alpha": window_alphas[0], "end_alpha": window_alphas[-1],
                    "window_length": length, "plus_fit": None, "minus_fit": None,
                })
                continue
            window_points = [row for row in points if row["alpha"] in window_alphas]
            plus_fit = _fit_loglog(window_points, sign=1)
            minus_fit = _fit_loglog(window_points, sign=-1)
            eval_rows: list[dict[str, Any]] = []
            for index, alpha in enumerate(window_alphas):
                plus = next((row for row in window_points if row["alpha"] == alpha and row["sign"] == 1), None)
                minus = next((row for row in window_points if row["alpha"] == alpha and row["sign"] == -1), None)
                pair = pair_by_alpha.get(alpha)
                if index + 1 < len(window_alphas):
                    right = window_alphas[index + 1]
                    left_plus = next((row for row in window_points if row["alpha"] == alpha and row["sign"] == 1), None)
                    right_plus = next((row for row in window_points if row["alpha"] == right and row["sign"] == 1), None)
                    left_minus = next((row for row in window_points if row["alpha"] == alpha and row["sign"] == -1), None)
                    right_minus = next((row for row in window_points if row["alpha"] == right and row["sign"] == -1), None)
                    plus_input_cosine = _cosine(left_plus["_input"], right_plus["_input"]) if left_plus and right_plus else None
                    minus_input_cosine = _cosine(left_minus["_input"], right_minus["_input"]) if left_minus and right_minus else None
                    plus_input_relative, plus_input_reason = _relative_change(left_plus["_input"], right_plus["_input"]) if left_plus and right_plus else (None, "plus-side input is missing")
                    minus_input_relative, minus_input_reason = _relative_change(left_minus["_input"], right_minus["_input"]) if left_minus and right_minus else (None, "minus-side input is missing")
                    paired_adj = pair_adj_by_right.get(right)
                else:
                    plus_input_cosine, minus_input_cosine = None, None
                    plus_input_relative, minus_input_relative = None, None
                    plus_input_reason, minus_input_reason, paired_adj = None, None, None
                eval_rows.append({
                    "alpha": alpha,
                    "plus_input_cosine": plus_input_cosine,
                    "minus_input_cosine": minus_input_cosine,
                    "plus_minus_input_cosine": None if pair is None else pair.get("plus_minus_input_cosine"),
                    "plus_slope": plus_fit.get("slope"), "plus_r2": plus_fit.get("r2"),
                    "minus_slope": minus_fit.get("slope"), "minus_r2": minus_fit.get("r2"),
                    "paired_secant_cosine": None if paired_adj is None else paired_adj.get("paired_secant_cosine"),
                    "paired_secant_relative_change": None if paired_adj is None else paired_adj.get("paired_secant_relative_change"),
                    "response_rms": None if plus is None or minus is None else min(plus.get("output_rms"), minus.get("output_rms")),
                    "group_repeat_floor": floor,
                    "plus_input_relative_change": plus_input_relative,
                    "plus_input_relative_change_reason": plus_input_reason,
                    "minus_input_relative_change": minus_input_relative,
                    "minus_input_relative_change_reason": minus_input_reason,
                })
            # The last point has no right-adjacent pair; thresholds are defined
            # on adjacent edges, not on the trailing point.
            decision = evaluate_window(eval_rows, min_length=MIN_WINDOW)
            decision.update({"space": space, "group": group, "start_alpha": window_alphas[0], "end_alpha": window_alphas[-1],
                             "window_length": length, "plus_fit": plus_fit, "minus_fit": minus_fit})
            decisions.append(decision)
    return fits, decisions


def analyze_records(records: Mapping[str, Mapping[str, Any]], *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze loaded precision samples without touching a model or source run."""

    if not isinstance(records, Mapping):
        raise TypeError("records must be a mapping of sample_id to detached record")
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, record in records.items():
        if not isinstance(record, Mapping):
            continue
        try:
            spec = _spec(record)
        except ValueError:
            # Diagnostic records have names such as ``full_A_zero`` and are
            # intentionally outside the formal 42-call table.
            continue
        normalized[str(key)] = record
        if str(record.get("sample_id", key)) != spec["sample_id"]:
            raise ValueError(f"sample_id/spec mismatch: {key}")
    formal = [record for record in normalized.values() if _spec(record)["kind"] in {"pre", "post", "perturbation"}]
    formal_specs = [_spec(record) for record in formal]
    evidence_failures = {str(record.get("sample_id")): _validate_record(record) for record in formal}
    evidence_failures = {key: value for key, value in evidence_failures.items() if value}
    scope_values = sorted({str(record.get("scope", "unknown")) for record in formal})
    run_status = metadata.get("status", {}) if isinstance(metadata, Mapping) else {}
    recorded_scope = run_status.get("scope") if isinstance(run_status, Mapping) else None
    config = metadata.get("config", {}) if isinstance(metadata, Mapping) else {}
    config_scope = config.get("scope") if isinstance(config, Mapping) else None
    root_values = [value for value in (recorded_scope, config_scope) if value is not None]
    root_scopes = [str(value) for value in root_values if str(value) in {"full", "module"}]
    root_scope_invalid = any(str(value) not in {"full", "module"} for value in root_values)
    scope_binding_ok = True
    status_name = str(run_status.get("status", "")).lower() if isinstance(run_status, Mapping) else ""
    if formal:
        if len(scope_values) != 1 or scope_values[0] not in {"full", "module"}:
            # Non-empty formal evidence must have exactly one legal scope;
            # neither status nor config metadata may hide a mixed/unknown set.
            scope = scope_values[0] if len(scope_values) == 1 else "mixed"
            scope_binding_ok = False
        else:
            scope = scope_values[0]
            if root_scope_invalid or any(root_scope != scope for root_scope in root_scopes):
                scope_binding_ok = False
    elif status_name not in {"blocked", "terminal_blocked"}:
        # Root scope is descriptive only for an empty terminal BLOCKED run.
        scope = "unknown" if not root_scopes else (root_scopes[0] if len(set(root_scopes)) == 1 else "mixed")
        scope_binding_ok = False
    elif len(root_scopes) == 1 and not root_scope_invalid:
        scope = root_scopes[0]
    else:
        scope = "unknown" if not root_scopes else "mixed"
        scope_binding_ok = False
    groups = {group: {spec["kind"] if spec["kind"] in {"pre", "post"} else (spec["alpha"], spec["sign"]): record
                      for record, spec in ((record, _spec(record)) for record in formal) if spec["group"] == group}
              for group in GROUPS}
    input_identity: dict[str, Any] = {"A_B_all_exact": True, "A_B_mismatches": [], "checked_keys": 0}
    for key in sorted({(spec["kind"], spec["alpha"], spec["sign"]) for spec in formal_specs}):
        a = next((record for record in formal if (_spec(record)["group"], _spec(record)["kind"], _spec(record)["alpha"], _spec(record)["sign"]) == ("A", *key)), None)
        b = next((record for record in formal if (_spec(record)["group"], _spec(record)["kind"], _spec(record)["alpha"], _spec(record)["sign"]) == ("B", *key)), None)
        if a is None or b is None:
            input_identity["A_B_all_exact"] = False
            input_identity["A_B_mismatches"].append({"key": key, "reason": "A or B sample is missing"})
            continue
        input_identity["checked_keys"] += 1
        try:
            a_input = _array(a.get("common_input_fp32"), name="A common input", dtype=np.float32)
            b_input = _array(b.get("common_input_fp32"), name="B common input", dtype=np.float32)
            if a_input.shape != b_input.shape or not np.array_equal(a_input, b_input):
                input_identity["A_B_all_exact"] = False
                input_identity["A_B_mismatches"].append({"key": key, "reason": "common interface FP32 bytes differ"})
        except (TypeError, ValueError) as error:
            input_identity["A_B_all_exact"] = False
            input_identity["A_B_mismatches"].append({"key": key, "reason": str(error)})
    compute_signatures = {group: sorted({_compute_signature(record) for record in formal if _spec(record)["group"] == group}) for group in GROUPS}
    compute_identity = {"B_C_same_path": bool(compute_signatures["B"] and compute_signatures["B"] == compute_signatures["C"]),
                       "signatures": compute_signatures,
                       "reason": None if compute_signatures["B"] == compute_signatures["C"] and compute_signatures["B"] else "B/C execution signatures are missing or differ"}
    cache_identity = {"cache_off_all": (None if not formal else not any("cache was requested" in reason or "cache was installed" in reason for reasons in evidence_failures.values() for reason in reasons)),
                      "evidence_failures": evidence_failures}
    noise_hashes: dict[str, str] = {}
    noise_reasons: dict[str, str] = {}
    for record in formal:
        sample_id = str(record.get("sample_id"))
        try:
            _, _, mask, _ = _input_parts(record)
            recomputed = fixed_noise_hash(record["initial_state"], mask)
            recorded = record.get("initial_noise_hash")
            if recorded is not None and str(recorded) != recomputed:
                noise_reasons[sample_id] = "recorded initial noise hash differs from recomputed array hash"
            noise_hashes[sample_id] = recomputed
        except (TypeError, ValueError) as error:
            noise_reasons[sample_id] = str(error)
    unique_noise = sorted(set(noise_hashes.values()))
    noise_identity = {"all_equal": bool(noise_hashes) and len(unique_noise) == 1 and not noise_reasons,
                      "hashes": noise_hashes, "reasons": noise_reasons}

    point_metrics: list[dict[str, Any]] = []
    pair_metrics: list[dict[str, Any]] = []
    adjacent_metrics: list[dict[str, Any]] = []
    fit_metrics: list[dict[str, Any]] = []
    window_decisions: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    vector_tensors: list[dict[str, Any]] = []
    spaces: dict[str, dict[str, Any]] = {}
    baseline_floors: dict[str, dict[str, float | None]] = {group: {} for group in GROUPS}
    for space in SPACES:
        space_points: list[dict[str, Any]] = []
        space_reasons: list[str] = []
        for group in GROUPS:
            group_records = [record for record in formal if _spec(record)["group"] == group]
            pre_record = next((record for record in group_records if _spec(record)["kind"] == "pre"), None)
            post_record = next((record for record in group_records if _spec(record)["kind"] == "post"), None)
            pre_output = _output(pre_record, space) if pre_record is not None else None
            post_output = _output(post_record, space) if post_record is not None else None
            if space == "decoded_final_rgb":
                if pre_output is not None and (pre_output.ndim != 3 or pre_output.shape[0] != 3):
                    pre_output = None
                    space_reasons.append(f"{group}: pre baseline RGB shape is not [3,H,W]")
                if post_output is not None and (post_output.ndim != 3 or post_output.shape[0] != 3):
                    post_output = None
                    space_reasons.append(f"{group}: post baseline RGB shape is not [3,H,W]")
            floor = None
            floor_reason = None
            if pre_output is None or post_output is None:
                floor_reason = "pre/post baseline output is missing for this space"
            elif pre_output.shape != post_output.shape:
                floor_reason = "pre/post baseline output shapes differ"
            else:
                floor = _rms(post_output.astype(np.float64) - pre_output.astype(np.float64))
            baseline_floors[group][space] = floor
            group_points: list[dict[str, Any]] = []
            if pre_output is None:
                space_reasons.append(f"{group}: pre baseline output is missing")
            for record in group_records:
                spec = _spec(record)
                if spec["kind"] != "perturbation":
                    continue
                if pre_output is None:
                    continue
                point = _record_point(record, baseline=pre_output, space=space)
                if point is None:
                    space_reasons.append(f"{spec['sample_id']}: {space} output is missing")
                    continue
                point["group_repeat_floor"] = floor
                point["group_repeat_floor_reason"] = floor_reason
                group_points.append(point)
                space_points.append(point)
                point_metrics.append(point)
            pairs, paired_adjacent = _pair_metrics(group_points, floor)
            pair_metrics.extend(pairs)
            adjacent_metrics.extend(paired_adjacent)
            if not group_points:
                continue
            fits, decisions = _make_windows(group_points, pairs, paired_adjacent, floor)
            fit_metrics.extend(fits)
            window_decisions.extend(decisions)
            for alpha in ALPHAS:
                for sign in (1, -1):
                    key = (alpha, sign)
                    for group_record in (record for record in group_records if _spec(record)["kind"] == "perturbation"):
                        spec = _spec(group_record)
                        if math.isclose(spec["alpha"], alpha, rel_tol=0.0, abs_tol=1e-12) and spec["sign"] == sign:
                            row = next((item for item in group_points if item["sample_id"] == spec["sample_id"]), None)
                            if row is not None:
                                # Keep the raw response vectors local only; the
                                # compact vector CSV contains norms/hashes.
                                continue
        expected_points = len(GROUPS) * len(ALPHAS) * 2
        complete_space = len(space_points) == expected_points and all(
            baseline_floors[group].get(space) is not None for group in GROUPS
        )
        spaces.setdefault(space, {"status": "OK" if complete_space else "INCOMPLETE", "reason": None,
                                  "point_metrics": [], "paired_metrics": [], "adjacent_metrics": [],
                                  "fits": [], "window_decisions": [], "baseline_floors": baseline_floors})
        spaces[space]["point_metrics"] = [_public_row(row) for row in space_points]
        spaces[space]["paired_metrics"] = [_public_row(row) for row in pair_metrics if row["space"] == space]
        spaces[space]["adjacent_metrics"] = [_public_row(row) for row in adjacent_metrics if row["space"] == space]
        spaces[space]["fits"] = [row for row in fit_metrics if row["space"] == space]
        spaces[space]["window_decisions"] = [row for row in window_decisions if row["space"] == space]
        if not space_points:
            spaces[space]["status"] = "N/A"
            if space == "decoded_final_rgb" and scope == "module":
                spaces[space]["reason"] = "single-step denoiser fallback does not decode final RGB"
            else:
                spaces[space]["reason"] = "; ".join(dict.fromkeys(space_reasons)) or "no finite output tensors were available"
        elif space == "decoded_final_rgb" and scope == "module":
            spaces[space]["status"] = "N/A"
            spaces[space]["reason"] = "single-step denoiser fallback does not decode final RGB"
        elif space_reasons:
            spaces[space]["reason"] = "; ".join(dict.fromkeys(space_reasons))
        if space_points and not complete_space:
            spaces[space]["status"] = "INCOMPLETE"
            reasons = list(space_reasons)
            reasons.append(f"required output set is incomplete: {len(space_points)}/{expected_points} perturbation outputs")
            spaces[space]["reason"] = "; ".join(dict.fromkeys(reasons))

    by_space_key: dict[tuple[str, str, float, int], dict[str, Any]] = {(row["space"], row["group"], float(row["alpha"]), int(row["sign"])): row for row in point_metrics}
    vector_identity = {"passed": True, "checked": 0, "failures": [], "max_identity_error_rms": 0.0}
    for space in SPACES:
        for alpha in ALPHAS:
            for sign in (1, -1):
                rows = [by_space_key.get((space, group, alpha, sign)) for group in GROUPS]
                if any(row is None for row in rows):
                    continue
                decomposition = decompose_precision_responses(rows[0]["_response"], rows[1]["_response"], rows[2]["_response"])
                vector_identity["checked"] += 1
                vector_identity["max_identity_error_rms"] = max(vector_identity["max_identity_error_rms"], float(decomposition["identity_error_rms"] or 0.0))
                if not decomposition["identity_passed"]:
                    vector_identity["passed"] = False
                    vector_identity["failures"].append({"space": space, "alpha": alpha, "sign": sign, "identity_error_rms": decomposition["identity_error_rms"]})
                vector_rows.append({"space": space, "alpha": alpha, "sign": sign,
                                    "identity_passed": decomposition["identity_passed"],
                                    "identity_error_rms": decomposition["identity_error_rms"],
                                    "Delta_quant_rms": decomposition["Delta_quant_rms"],
                                    "Delta_compute_rms": decomposition["Delta_compute_rms"],
                                    "r_A_minus_r_C_rms": decomposition["r_A_minus_r_C_rms"],
                                    "Delta_quant_relative_norm": decomposition["Delta_quant_relative_norm"],
                                    "Delta_compute_relative_norm": decomposition["Delta_compute_relative_norm"],
                                    "relative_norm_reason": decomposition["relative_norm_reason"],
                                    "Delta_quant_sha256": _sha256_array(decomposition["Delta_quant"]),
                                    "Delta_compute_sha256": _sha256_array(decomposition["Delta_compute"]),
                                    "r_A_minus_r_C_sha256": _sha256_array(decomposition["r_A_minus_r_C"]),
                                    "reconstructed_sha256": _sha256_array(decomposition["reconstructed"]),})
                vector_tensors.append({"space": space, "alpha": alpha, "sign": sign,
                                       "Delta_quant": decomposition["Delta_quant"].astype(np.float64, copy=False),
                                       "Delta_compute": decomposition["Delta_compute"].astype(np.float64, copy=False),
                                       "r_A_minus_r_C": decomposition["r_A_minus_r_C"].astype(np.float64, copy=False),
                                       "reconstructed": decomposition["reconstructed"].astype(np.float64, copy=False)})

    if not vector_rows:
        vector_identity["passed"] = False
        vector_identity["reason"] = "no complete A/B/C formal response vectors were available"

    formal_keys = {(spec["group"], spec["kind"], spec["alpha"], spec["sign"]) for spec in formal_specs}
    expected_keys = {(group, kind, alpha, sign) for group in GROUPS for kind, alpha, sign in
                     [("pre", 0.0, 0), ("post", 0.0, 0)] + [("perturbation", alpha, sign) for alpha in ALPHAS for sign in (1, -1)]}
    formal_complete = formal_keys == expected_keys and len(formal) == FORMAL_CALL_COUNT
    if not formal_complete:
        input_identity["A_B_all_exact"] = False
        if not any(item.get("reason") == "formal call set is incomplete" for item in input_identity["A_B_mismatches"]):
            input_identity["A_B_mismatches"].append({"key": None, "reason": "formal call set is incomplete"})
    latent_complete = spaces.get("predicted_latent", {}).get("status") == "OK"
    rgb_complete = spaces.get("decoded_final_rgb", {}).get("status") == "OK"
    module_rgb_na = scope == "module" and spaces.get("decoded_final_rgb", {}).get("status") == "N/A"
    vector_complete = vector_identity["passed"] and vector_identity["checked"] >= len(ALPHAS) * 2
    recorded_status_ok = True
    if isinstance(run_status, Mapping) and run_status.get("status") is not None:
        expected_status = "complete" if scope == "full" else "module_complete" if scope == "module" else None
        recorded_status_ok = run_status.get("status") == expected_status
    required_outputs_ok = latent_complete and (rgb_complete if scope == "full" else module_rgb_na)
    status = "COMPLETE" if (
        formal_complete and not evidence_failures and input_identity["A_B_all_exact"]
        and compute_identity["B_C_same_path"] and noise_identity["all_equal"]
        and scope in {"full", "module"} and scope_binding_ok and required_outputs_ok and vector_complete and recorded_status_ok
    ) else "BLOCKED"
    blocked_reasons: list[str] = []
    if not formal_complete:
        blocked_reasons.append(f"formal call count/spec is incomplete: {len(formal)}/{FORMAL_CALL_COUNT}")
    if not scope_binding_ok:
        blocked_reasons.append("formal record scopes are mixed or do not match root scope metadata")
    if evidence_failures:
        blocked_reasons.append("one or more sample evidence checks failed")
    if not input_identity["A_B_all_exact"]:
        blocked_reasons.append("A/B common interface FP32 input identity failed")
    if not compute_identity["B_C_same_path"]:
        blocked_reasons.append("B/C observed compute path identity failed")
    if not noise_identity["all_equal"]:
        blocked_reasons.append("actual consumed initial sampler noise is not paired")
    if not latent_complete:
        blocked_reasons.append("predicted latent output set is missing, nonfinite, or shape-inconsistent")
    if scope == "full" and not rgb_complete:
        blocked_reasons.append("full scope requires a complete fixed-decode RGB output set")
    if scope == "module" and not module_rgb_na:
        blocked_reasons.append("module_complete must report RGB as N/A without decoded outputs")
    if not vector_complete:
        blocked_reasons.append("A/B/C vector identity is missing or failed")
    if isinstance(run_status, Mapping) and run_status.get("status") is not None and not recorded_status_ok:
        reason = run_status.get("error", {}).get("message") if isinstance(run_status.get("error"), Mapping) else None
        blocked_reasons.append("recorded run status: " + str(run_status.get("status")) + (f" ({reason})" if reason else ""))
    diagnostic_count = metadata.get("diagnostic_count") if isinstance(metadata, Mapping) else None
    if diagnostic_count is None and isinstance(run_status, Mapping):
        diagnostic_count = run_status.get("diagnostic_attempts")
    execution_metadata = metadata.get("task4_execution") if isinstance(metadata, Mapping) else None
    if not isinstance(execution_metadata, Mapping) and isinstance(metadata, Mapping):
        execution_metadata = metadata.get("execution")
    review_sample_dirs = {
        str(record.get("sample_id")): str(record.get("_sample_dir"))
        for record in formal
        if record.get("sample_id") in _REVIEW_SAMPLE_IDS and record.get("_sample_dir")
    }
    review_evidence_sources = {
        "source_dir": execution_metadata.get("source_dir") if isinstance(execution_metadata, Mapping) else None,
        "run_dir": metadata.get("run_dir") if isinstance(metadata, Mapping) else None,
        "sample_dirs": review_sample_dirs,
    }
    result = {
        "schema_version": "umi-input-quantization-compute-precision-analysis-v1",
        "status": status,
        "scope": scope,
        "formal_call_count": len(formal),
        "expected_formal_call_count": FORMAL_CALL_COUNT,
        "diagnostic_call_count": int(diagnostic_count) if diagnostic_count is not None else max(0, len(records) - len(formal)),
        "formal_complete": formal_complete,
        "blocked_reasons": blocked_reasons,
        "metadata": dict(metadata or {}),
        "review_evidence_sources": review_evidence_sources,
        "thresholds": dict(THRESHOLDS),
        "input_identity": input_identity,
        "compute_identity": compute_identity,
        "cache_identity": cache_identity,
        "noise_identity": noise_identity,
        "vector_identity": vector_identity,
        "baseline_floors": baseline_floors,
        "spaces": spaces,
        "point_metrics": [_public_row(row) for row in point_metrics],
        "paired_metrics": [_public_row(row) for row in pair_metrics],
        "_paired_tensors": pair_metrics,
        "adjacent_metrics": adjacent_metrics,
        "fit_metrics": fit_metrics,
        "window_decisions": window_decisions,
        "vector_decomposition": vector_rows,
        "_vector_tensors": vector_tensors,
        "timing": {str(record.get("sample_id")): record.get("elapsed_seconds") for record in formal if record.get("elapsed_seconds") is not None},
    }
    return result


def _safe_raw_relative(relative: str) -> str:
    pure = PurePosixPath(relative)
    if (
        not relative or "\\" in relative or pure.is_absolute() or pure.as_posix() != relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"unsafe raw manifest path: {relative!r}")
    if pure.parts[0] == "precision_analysis":
        raise ValueError("raw manifest must not cover the precision_analysis subtree")
    if is_task4_raw_manifest_excluded(relative):
        raise ValueError(f"raw manifest must not cover excluded path: {relative}")
    return relative


def _raw_manifest(root: Path) -> dict[str, Any]:
    """Verify exact raw-run coverage, excluding only the analysis subtree."""

    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError(f"raw manifest is missing: {manifest}")
    entries: dict[str, str] = {}
    lines = manifest.read_text(encoding="ascii").splitlines()
    if not lines:
        raise ValueError("raw manifest is empty")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"malformed raw manifest line: {line!r}")
        digest, relative = match.groups()
        relative = _safe_raw_relative(relative)
        if relative in entries:
            raise ValueError(f"duplicate raw manifest entry: {relative}")
        if relative == "MANIFEST.sha256":
            raise ValueError("raw manifest must not self-reference")
        path = root / Path(relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"raw manifest hash mismatch: {relative}")
        entries[relative] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not is_task4_raw_manifest_excluded(path.relative_to(root).as_posix())
    }
    if set(entries) != actual:
        missing = sorted(actual - set(entries))
        extra = sorted(set(entries) - actual)
        raise ValueError(f"raw manifest inventory mismatch; missing={missing}, extra={extra}")
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest),
        "entry_count": len(entries),
        "entries": dict(sorted(entries.items())),
        "coverage": "all raw run files except exact precision_analysis/** subtree and .runner.lock",
    }


def _verify_raw_manifest_snapshot(root: Path, snapshot: Mapping[str, Any]) -> None:
    current = _raw_manifest(root)
    if current.get("sha256") != snapshot.get("sha256") or current.get("entries") != snapshot.get("entries"):
        raise ValueError("raw evidence changed during analysis (TOCTOU manifest mismatch)")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _expected_formal_keys() -> set[tuple[str, str, float, int]]:
    return {
        (group, kind, float(alpha), int(sign))
        for group in GROUPS
        for kind, alpha, sign in [("pre", 0.0, 0), ("post", 0.0, 0)]
        + [("perturbation", alpha, sign) for alpha in ALPHAS for sign in (1, -1)]
    }


def _validate_sample_artifacts(sample_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate exact sample inventory and hashes before resolving arrays."""

    status = _load_json(sample_dir / "status.json")
    hashes = status.get("artifact_sha256")
    required = status.get("required_artifacts")
    if not isinstance(hashes, dict) or not isinstance(required, list):
        raise ValueError(f"sample artifact hashes are missing: {sample_dir}")
    if set(required) != set(hashes) or sorted(required) != required:
        raise ValueError(f"sample artifact inventory is not exact: {sample_dir}")
    files = {
        path.relative_to(sample_dir).as_posix()
        for path in sample_dir.rglob("*")
        if path.is_file() and path.name != "status.json"
    }
    if files != set(hashes):
        raise ValueError(f"sample artifact inventory mismatch: {sample_dir}")
    for name, digest in hashes.items():
        if not isinstance(name, str) or PurePosixPath(name).as_posix() != name or PurePosixPath(name).name != name:
            raise ValueError(f"unsafe sample artifact name: {name!r}")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid sample artifact digest: {name}")
        path = sample_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"sample artifact hash mismatch: {sample_dir / name}")
    record_json = _load_json(sample_dir / "sample.json")
    return status, record_json


def _resolve_artifact_refs(value: Any, sample_dir: Path, artifact_names: set[str]) -> Any:
    if isinstance(value, Mapping) and "artifact" in value:
        name = value.get("artifact")
        if not isinstance(name, str) or name not in artifact_names or not name.endswith(".npy"):
            raise ValueError(f"sample JSON references an unknown artifact: {name!r}")
        path = sample_dir / name
        array = np.load(path, allow_pickle=False)
        declared_dtype = value.get("dtype")
        declared_shape = value.get("shape")
        if declared_dtype is not None and str(array.dtype) != str(declared_dtype):
            raise ValueError(f"sample artifact dtype mismatch: {name}")
        if declared_shape is not None and list(array.shape) != list(declared_shape):
            raise ValueError(f"sample artifact shape mismatch: {name}")
        return np.ascontiguousarray(array)
    if isinstance(value, Mapping):
        return {str(key): _resolve_artifact_refs(item, sample_dir, artifact_names) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_artifact_refs(item, sample_dir, artifact_names) for item in value]
    return value


def _validate_record_binding(record: Mapping[str, Any], sample_dir: Path, identity: str | None, *, section: str) -> dict[str, Any]:
    raw_spec = record.get("spec")
    sample_id = record.get("sample_id")
    if sample_id is None and isinstance(raw_spec, Mapping):
        sample_id = raw_spec.get("sample_id")
    if sample_id != sample_dir.name:
        raise ValueError(f"sample/spec identity binding mismatch: {sample_id!r} != {sample_dir.name!r}")
    if identity is not None and record.get("identity") != identity:
        raise ValueError(f"sample identity binding mismatch: {sample_id}")
    spec = raw_spec
    if not isinstance(spec, Mapping) or spec.get("sample_id") != sample_id:
        raise ValueError(f"sample spec binding mismatch: {sample_id}")
    group = spec.get("group")
    kind = spec.get("kind")
    try:
        key = (str(group), str(kind), float(spec.get("alpha", 0.0)), int(spec.get("sign", 0)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid sample spec binding: {sample_id}") from error
    if section == "samples" and key not in _expected_formal_keys():
        raise ValueError(f"out-of-scope formal sample spec: {sample_id}")
    if section == "diagnostics" and kind != "diagnostic":
        raise ValueError(f"out-of-scope diagnostic sample spec: {sample_id}")
    result = dict(record)
    result["sample_id"] = sample_id
    return result


def _load_precision_records(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = run_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"precision run is not a directory: {root}")
    raw_snapshot = _raw_manifest(root)
    status = _load_json(root / "status.json") if (root / "status.json").is_file() else {"status": "UNKNOWN"}
    config = _load_json(root / "config.json") if (root / "config.json").is_file() else {}
    status_identity = status.get("identity")
    config_identity = config.get("identity")
    if status_identity is not None and config_identity is not None and status_identity != config_identity:
        raise ValueError("run status/config identity binding mismatch")
    identity = status_identity or config_identity
    records: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for section in ("samples", "diagnostics"):
        section_root = root / section
        if not section_root.is_dir():
            continue
        for sample_dir in sorted(section_root.iterdir()):
            if not sample_dir.is_dir():
                raise ValueError(f"unexpected file in {section}: {sample_dir.name}")
            if ".attempt." in sample_dir.name or not (sample_dir / "sample.json").is_file() or not (sample_dir / "status.json").is_file():
                raise ValueError(f"incomplete or stale {section} sample directory: {sample_dir}")
            sample_status, raw_record = _validate_sample_artifacts(sample_dir)
            record = _validate_record_binding(raw_record, sample_dir, identity, section=section)
            sample_id = str(record["sample_id"])
            artifact_names = set(sample_status["artifact_sha256"])
            record = _resolve_artifact_refs(record, sample_dir, artifact_names)
            # PrecisionSampleStore deliberately strips top-level ndarrays from
            # JSON and writes them as named .npy artifacts. Restore those
            # artifacts only after the exact status inventory/hash checks above.
            for artifact_name in sorted(artifact_names):
                if not artifact_name.endswith(".npy"):
                    continue
                artifact_path = sample_dir / artifact_name
                stem = Path(artifact_name).stem
                if stem in record:
                    raise ValueError(f"sample JSON/artifact conflict for top-level array: {sample_id}/{stem}")
                try:
                    array = np.load(artifact_path, allow_pickle=False)
                except (OSError, ValueError) as error:
                    raise ValueError(f"sample artifact cannot be loaded: {artifact_path}") from error
                if array.size == 0 or not np.all(np.isfinite(array.astype(np.float64, copy=False))):
                    raise ValueError(f"sample artifact is empty or nonfinite: {artifact_path}")
                record[stem] = np.ascontiguousarray(array)
            if sample_id in seen_ids:
                raise ValueError(f"duplicate precision sample id: {sample_id}")
            seen_ids.add(sample_id)
            if section == "samples" and sample_status.get("status") == "success":
                record["_sample_dir"] = str(sample_dir)
                records[sample_id] = record
    samples_root = root / "samples"
    if not samples_root.is_dir() and status.get("status") not in {"blocked", "BLOCKED"}:
        raise ValueError(f"precision samples directory is missing: {samples_root}")
    _verify_raw_manifest_snapshot(root, raw_snapshot)
    execution_path = root / "task4_execution.json"
    execution = _load_json(execution_path) if execution_path.is_file() else {}
    metadata = {"run_dir": str(root), "status": status, "config": config,
                "diagnostic_count": status.get("diagnostic_attempts"), "raw_manifest": raw_snapshot,
                "task4_execution": execution, "task4_execution_path": str(execution_path) if execution_path.is_file() else None}
    return records, metadata


def _resolve_analysis_destination(root: Path, output_dir: str | Path | None) -> Path:
    analysis_root = (root / "precision_analysis").resolve()
    candidate = analysis_root if output_dir is None else Path(output_dir)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(analysis_root)
    except ValueError as error:
        raise ValueError(f"analysis output must be bound under {analysis_root}") from error
    if candidate == root or candidate == root.parent:
        raise ValueError("analysis output must be a child of precision_analysis")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


_ANALYSIS_FILES = {
    "precision_summary.json", "candidate_decisions.json", "precision_metrics.csv", "paired_secant_metrics.csv",
    "adjacent_consistency.csv", "fit_metrics.csv", "window_decisions.csv", "vector_decomposition.csv",
    "precision_report_zh.md", "precision_path_note.md", "commands.md", "gpu_timing_summary.json",
    "gpu_timing_summary.csv", "precision_manifest.sha256", "MANIFEST.sha256", "review_bundle.zip",
    "calibration_manifest.sha256",
    "response_rms.png", "response_rms.svg", "window_decisions.png", "window_decisions.svg",
    "precision_response.png", "precision_response.svg", "precision_windows.png", "precision_windows.svg",
    "input_distortion.png", "input_distortion.svg", "latent_rgb_response.png", "latent_rgb_response.svg",
    "derivative_consistency.png", "derivative_consistency.svg", "delta_decomposition.png", "delta_decomposition.svg",
    "review_evidence_manifest.json",
}
_ANALYSIS_TENSOR_DIRS = {
    "Delta_quant", "Delta_compute", "r_A_minus_r_C", "reconstructed", "target_centered_derivative",
    "actual_paired_secant", "one_sided_plus_derivative", "one_sided_minus_derivative",
}
_BUNDLE_ROOT_FILES = _ANALYSIS_FILES - {"review_bundle.zip"}


def _review_evidence_relative_allowed(relative: str) -> bool:
    """Return whether a nested review-evidence path is in the fixed whitelist."""

    pure = PurePosixPath(relative)
    if not relative or "\\" in relative or pure.is_absolute() or pure.as_posix() != relative:
        return False
    parts = pure.parts
    if any(part in {"", ".", ".."} for part in parts) or len(parts) < 2 or parts[0] != "review_evidence":
        return False
    if parts[1] in {"source", "analysis_source"}:
        return len(parts) == 3 and parts[2] in _REVIEW_SOURCE_FILES
    if parts[1] == "run":
        return len(parts) == 3 and parts[2] in _REVIEW_RUN_FILES
    if parts[1] == "samples":
        return (
            len(parts) == 4
            and re.fullmatch(r"[A-Za-z0-9_-]+", parts[2]) is not None
            and parts[2] in _REVIEW_SAMPLE_IDS
            and (parts[3] in {"sample.json", "status.json"} or parts[3] in _REVIEW_SAMPLE_ARRAY_FILES)
        )
    return False


def _validate_calibration_manifest(output: Path) -> None:
    path = output / "calibration_manifest.sha256"
    if not path.is_file():
        raise ValueError("calibration manifest is missing")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"malformed calibration manifest line: {line!r}")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if relative in entries or pure.parts[:1] != ("calibration",) or len(pure.parts) != 2 or pure.parts[1] not in _CALIBRATION_FILES:
            raise ValueError(f"unsafe or duplicate calibration manifest path: {relative!r}")
        file_path = output / Path(relative)
        if not file_path.is_file() or sha256_file(file_path) != digest:
            raise ValueError(f"calibration artifact hash mismatch: {relative}")
        entries[relative] = digest
    actual = {path.relative_to(output).as_posix() for path in (output / "calibration").rglob("*") if path.is_file()}
    if set(entries) != actual or set(entries) != {f"calibration/{name}" for name in _CALIBRATION_FILES}:
        raise ValueError(f"calibration manifest inventory mismatch; missing={sorted(actual - set(entries))}, extra={sorted(set(entries) - actual)}")


def _analysis_manifest(output: Path) -> dict[str, Any]:
    manifest = output / "precision_manifest.sha256"
    if not manifest.is_file() or not (output / "MANIFEST.sha256").is_file():
        raise ValueError("analysis manifest is missing")
    precision_text = manifest.read_text(encoding="ascii")
    if precision_text != (output / "MANIFEST.sha256").read_text(encoding="ascii"):
        raise ValueError("analysis manifest aliases differ")
    entries: dict[str, str] = {}
    for line in precision_text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"malformed analysis manifest line: {line!r}")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if not relative or "\\" in relative or pure.is_absolute() or pure.as_posix() != relative or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(f"unsafe analysis manifest path: {relative!r}")
        if relative in entries:
            raise ValueError(f"duplicate analysis manifest entry: {relative}")
        path = output / Path(relative)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"analysis artifact hash mismatch: {relative}")
        if relative in {"precision_manifest.sha256", "MANIFEST.sha256", "review_bundle.zip"}:
            raise ValueError(f"analysis manifest self/generated entry is not allowed: {relative}")
        entries[relative] = digest
    actual = {
        relative
        for path in output.rglob("*")
        if path.is_file()
        for relative in [path.relative_to(output).as_posix()]
        if relative not in {"precision_manifest.sha256", "MANIFEST.sha256", "review_bundle.zip"}
    }
    if set(entries) != actual:
        raise ValueError(f"analysis manifest inventory mismatch; missing={sorted(actual - set(entries))}, extra={sorted(set(entries) - actual)}")
    missing_required = sorted(_REQUIRED_ANALYSIS_FILES - set(entries))
    if missing_required:
        raise ValueError(f"analysis package is incomplete; missing required artifacts: {missing_required}")
    _validate_calibration_manifest(output)
    for relative in entries:
        pure = PurePosixPath(relative)
        if len(pure.parts) == 1:
            if relative not in _ANALYSIS_FILES - {"precision_manifest.sha256", "MANIFEST.sha256", "review_bundle.zip"}:
                raise ValueError(f"stale analysis extra: {relative}")
        elif pure.parts[0] == "analysis_tensors" and len(pure.parts) == 3 and pure.parts[1] in _ANALYSIS_TENSOR_DIRS and pure.suffix == ".npy":
            continue
        elif pure.parts[0] == "calibration" and len(pure.parts) == 2 and pure.parts[1] in _CALIBRATION_FILES:
            continue
        elif pure.parts[0] == "review_evidence" and _review_evidence_relative_allowed(relative):
            continue
        else:
            raise ValueError(f"stale analysis extra: {relative}")
    return {"sha256": sha256_file(manifest), "entries": dict(sorted(entries.items()))}


def _verify_analysis_package(output: Path, raw_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _analysis_manifest(output)
    summary = _load_json(output / "precision_summary.json")
    recorded = summary.get("metadata", {}).get("raw_manifest", {}) if isinstance(summary.get("metadata"), Mapping) else {}
    if recorded.get("sha256") != raw_snapshot.get("sha256") or recorded.get("entries") != raw_snapshot.get("entries"):
        raise ValueError("analysis summary is not bound to the current raw manifest")
    if not (output / "review_bundle.zip").is_file():
        raise ValueError("review bundle is missing")
    return manifest


def _load_existing_analysis(output: Path, root: Path, raw_snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if not output.is_dir():
        raise ValueError("analysis destination is not a directory")
    _verify_analysis_package(output, raw_snapshot)
    _verify_raw_manifest_snapshot(root, raw_snapshot)
    result = _load_json(output / "precision_summary.json")
    result["output_dir"] = str(output)
    result["artifact_paths"] = sorted(path.name for path in output.iterdir() if path.is_file())
    return result


def analyze_run(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    records, metadata = _load_precision_records(root)
    result = analyze_records(records, metadata=metadata)
    destination = _resolve_analysis_destination(root, output_dir)
    if destination.exists():
        try:
            return _load_existing_analysis(destination, root, metadata["raw_manifest"])
        except ValueError as error:
            raise ValueError(f"stale or invalid analysis output: {destination}: {error}") from error
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=str(root.parent)))
    try:
        publication = write_precision_artifacts(result, stage)
        _verify_analysis_package(stage, metadata["raw_manifest"])
        _verify_raw_manifest_snapshot(root, metadata["raw_manifest"])
        try:
            _atomic_rename_noreplace(stage, destination)
        except FileExistsError as error:
            raise FileExistsError(f"analysis output appeared during publish: {destination}") from error
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
    result = dict(result)
    result["output_dir"] = str(destination)
    result["artifact_paths"] = publication.get("artifact_paths", [])
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    keys: list[str] = []
    for row in values:
        for key, value in row.items():
            if str(key).startswith("_") or isinstance(value, (np.ndarray, list, tuple, dict)):
                continue
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["status"])
        writer.writeheader()
        for row in values:
            writer.writerow({key: "" if row.get(key) is None else _json_safe(row.get(key)) for key in (keys or ["status"])})


def _chart_x_layout(series: Mapping[str, list[tuple[float, float | None]]], left: float, plot_w: float) -> tuple[Any, list[float], bool]:
    """Return a deterministic numeric-alpha x transform and tick values."""

    values = sorted({float(x_value) for points in series.values() for x_value, _ in points if math.isfinite(float(x_value))})
    log_alpha = bool(values) and all(value > 0.0 for value in values)
    transformed = [math.log10(value) for value in values] if log_alpha else values
    low = min(transformed) if transformed else 0.0
    high = max(transformed) if transformed else 1.0
    if high <= low:
        high = low + 1.0
    def transform(value: float, _index: int = 0) -> float:
        numeric = math.log10(float(value)) if log_alpha and float(value) > 0.0 else float(value)
        return left + plot_w * (numeric - low) / (high - low)
    if len(values) > 7:
        ticks = [values[index] for index in np.linspace(0, len(values) - 1, 7, dtype=int)]
        ticks = list(dict.fromkeys(ticks))
    else:
        ticks = values
    return transform, ticks, log_alpha


def _format_x_tick(value: float, *, log_alpha: bool) -> str:
    return f"{value:.0e}" if log_alpha else f"{value:.3g}"


def _chart_y_layout(series: Mapping[str, list[tuple[float, float | None]]], *, low: float,
                    top: float, plot_h: float, horizontal_lines: Mapping[str, float] | None,
                    log_y: bool | None, y_limits: tuple[float, float] | None = None) -> tuple[Any, list[float], bool, float, float, bool]:
    values = [float(value) for points in series.values() for _, value in points
              if value is not None and math.isfinite(float(value))]
    reference_values = [float(value) for value in (horizontal_lines or {}).values()
                        if value is not None and math.isfinite(float(value))]
    positive = [value for value in values + reference_values if value > 0.0]
    use_log = bool(log_y) and bool(positive)
    if use_log:
        low_exp = math.floor(math.log10(min(positive)))
        high_exp = math.ceil(math.log10(max(positive)))
        if high_exp <= low_exp:
            low_exp -= 1
            high_exp += 1
        def transform(value: float) -> float | None:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                return None
            return top + plot_h * (1.0 - (math.log10(float(value)) - low_exp) / (high_exp - low_exp))
        ticks = [10.0 ** (low_exp + (high_exp - low_exp) * index / 4.0) for index in range(5)]
        return transform, ticks, True, float(low_exp), float(high_exp), any(value <= 0.0 for value in values)
    if y_limits is not None:
        low_value, high_value = (float(y_limits[0]), float(y_limits[1]))
    else:
        low_value = 0.0 if not values else min(0.0, min(values))
        high_value = 1.0 if not values else max(1.0, max(values))
    if high_value <= low_value:
        high_value = low_value + 1.0
    def transform(value: float) -> float | None:
        if not math.isfinite(float(value)):
            return None
        return top + plot_h * (1.0 - (float(value) - low_value) / (high_value - low_value))
    ticks = [low_value + (high_value - low_value) * index / 4.0 for index in range(5)]
    return transform, ticks, False, low_value, high_value, False


def _chart_segments(points: list[tuple[float, float | None]], x: Any, y: Any) -> list[list[tuple[float, float]]]:
    segments: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for x_value, value in points:
        try:
            xx = x(float(x_value))
            yy = None if value is None else y(float(value))
        except (TypeError, ValueError, OverflowError):
            yy = None
            xx = None
        if xx is None or yy is None or not math.isfinite(float(yy)):
            if current:
                segments.append(current)
                current = []
            continue
        current.append((float(xx), float(yy)))
    if current:
        segments.append(current)
    return segments


def _chart_svg(title: str, series: Mapping[str, list[tuple[float, float | None]]], *, y_label: str,
               na_reasons: Mapping[str, str] | None = None, log_y: bool | None = None,
               horizontal_lines: Mapping[str, float] | None = None) -> str:
    # Reserve a wide right-hand legend column.  Delta labels and explicit N/A
    # annotations must remain readable instead of being clipped at 900px.
    width, height = 1200, 560
    left, top, right, bottom = 90, 60, 300, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ("#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#c62828")
    x, x_ticks, log_alpha = _chart_x_layout(series, left, plot_w)
    y, y_ticks, use_log_y, y_low, y_high, omitted_nonpositive = _chart_y_layout(
        series, low=0.0, top=top, plot_h=plot_h, horizontal_lines=horizontal_lines, log_y=log_y)
    lines = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
             "<rect width='100%' height='100%' fill='white'/>", f"<text x='{left}' y='28' font-size='20' font-family='Arial'>{html.escape(title)}</text>",
             f"<text x='18' y='{top + plot_h/2:.1f}' font-size='13' font-family='Arial' transform='rotate(-90 18 {top + plot_h/2:.1f})'>{html.escape(y_label)}</text>",
             f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#263238'/>",
             f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#263238'/>"]
    for value in y_ticks:
        yy = y(value)
        if yy is None:
            continue
        label = f"{value:.0e}" if use_log_y else f"{value:.3g}"
        lines.append(f"<line x1='{left-4}' y1='{yy:.2f}' x2='{left + plot_w}' y2='{yy:.2f}' stroke='#e0e0e0'/>")
        lines.append(f"<text x='{left-10}' y='{yy+4:.2f}' text-anchor='end' font-size='11' font-family='Arial'>{label}</text>")
    for name, value in (horizontal_lines or {}).items():
        yy = y(float(value))
        if yy is None:
            continue
        lines.append(f"<line x1='{left}' y1='{yy:.2f}' x2='{left + plot_w}' y2='{yy:.2f}' stroke='#455a64' stroke-width='2' stroke-dasharray='7,5'/>")
        lines.append(f"<text x='{left + plot_w - 4}' y='{yy-5:.2f}' text-anchor='end' font-size='11' font-family='Arial' fill='#455a64'>{html.escape(name)}</text>")
    for value in x_ticks:
        xx = x(value)
        lines.append(f"<line x1='{xx:.2f}' y1='{top + plot_h}' x2='{xx:.2f}' y2='{top + plot_h + 5}' stroke='#263238'/>")
        lines.append(f"<text x='{xx:.2f}' y='{height-48}' text-anchor='middle' font-size='11' font-family='Arial'>{html.escape(_format_x_tick(value, log_alpha=log_alpha))}</text>")
    for index, (name, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        segments = _chart_segments(points, x, y)
        for coords in segments:
            if len(coords) >= 2:
                lines.append("<polyline fill='none' stroke='%s' stroke-width='2' points='%s'/>" % (color, " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in coords)))
            for xx, yy in coords:
                lines.append(f"<circle cx='{xx:.2f}' cy='{yy:.2f}' r='3.5' fill='{color}'/>")
        ly = top + 18 + index * 18
        lines.append(f"<line x1='{width-280}' y1='{ly}' x2='{width-260}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        lines.append(f"<text x='{width-252}' y='{ly+4}' font-size='12' font-family='Arial'>{html.escape(name)}</text>")
        reason = (na_reasons or {}).get(name)
        if reason is None and not any(value is not None and math.isfinite(float(value)) for _, value in points):
            reason = "no finite values were recorded"
        if reason:
            lines.append(f"<text x='{left + 8}' y='{top + 18 + index * 18:.2f}' font-size='12' font-family='Arial' fill='#c62828'>N/A: {html.escape(str(reason))}</text>")
    if use_log_y:
        scale_note = "log-y (positive values only; zero/N/A omitted)"
    else:
        scale_note = "linear y"
    axis_label = "alpha (numeric; log10 position for positive alpha)" if log_alpha else "numeric x-axis"
    lines.append(f"<text x='{left + 8}' y='{height-32}' font-size='11' font-family='Arial' fill='#455a64'>{scale_note}</text>")
    lines.append(f"<text x='{left + plot_w/2}' y='{height-18}' text-anchor='middle' font-size='13' font-family='Arial'>{axis_label}</text>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _chart_png(title: str, series: Mapping[str, list[tuple[float, float | None]]], *, y_label: str,
               na_reasons: Mapping[str, str] | None = None, log_y: bool | None = None,
               horizontal_lines: Mapping[str, float] | None = None) -> bytes:
    from PIL import Image, ImageDraw
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 60, 300, 85
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ((21, 101, 192), (46, 125, 50), (239, 108, 0), (106, 27, 154), (0, 131, 143), (198, 40, 40))
    x, x_ticks, log_alpha = _chart_x_layout(series, left, plot_w)
    y, y_ticks, use_log_y, y_low, y_high, omitted_nonpositive = _chart_y_layout(
        series, low=0.0, top=top, plot_h=plot_h, horizontal_lines=horizontal_lines, log_y=log_y)
    draw.text((left, 25), title, fill=(20, 30, 35))
    draw.text((18, top + plot_h // 2), y_label, fill=(20, 30, 35))
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(38, 50, 56), width=2)
    draw.line((left, top, left, top + plot_h), fill=(38, 50, 56), width=2)
    for value in y_ticks:
        yy = y(value)
        if yy is None:
            continue
        yy = int(round(yy))
        label = f"{value:.0e}" if use_log_y else f"{value:.3g}"
        draw.line((left, yy, left + plot_w, yy), fill=(224, 224, 224), width=1)
        draw.text((left - 70, yy - 7), label, fill=(20, 30, 35))
    for name, value in (horizontal_lines or {}).items():
        yy = y(float(value))
        if yy is None:
            continue
        yy = int(round(yy))
        draw.line((left, yy, left + plot_w, yy), fill=(69, 90, 100), width=2)
        draw.text((left + plot_w - 180, yy - 16), name, fill=(69, 90, 100))
    for value in x_ticks:
        xx = int(round(x(value)))
        draw.line((xx, top + plot_h, xx, top + plot_h + 5), fill=(38, 50, 56), width=1)
        draw.text((xx - 18, height - 52), _format_x_tick(value, log_alpha=log_alpha), fill=(20, 30, 35))
    for index, (name, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        for coords in _chart_segments(points, x, y):
            coords_int = [(int(round(xx)), int(round(yy))) for xx, yy in coords]
            if len(coords_int) >= 2:
                draw.line(coords_int, fill=color, width=3)
            for xx, yy in coords_int:
                draw.ellipse((xx - 4, yy - 4, xx + 4, yy + 4), fill=color)
        ly = top + 12 + index * 18
        draw.line((width - 280, ly, width - 260, ly), fill=color, width=3)
        draw.text((width - 252, ly - 7), name, fill=(20, 30, 35))
        reason = (na_reasons or {}).get(name)
        if reason is None and not any(value is not None and math.isfinite(float(value)) for _, value in points):
            reason = "no finite values were recorded"
        if reason:
            draw.text((left + 8, top + 12 + index * 18), f"N/A: {reason}", fill=(198, 40, 40))
    if use_log_y:
        draw.text((left + 8, height - 32), "log-y (positive values only; zero/N/A omitted)", fill=(69, 90, 100))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        image.save(temporary, format="PNG", optimize=False)
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def _chart_svg_faceted(title: str, facets: Iterable[Mapping[str, Any]]) -> str:
    """Render independent y-axis facets into one deterministic 1200px SVG."""

    width, height = 1200, 560
    facet_list = list(facets)
    panel_width = 550
    colors = ("#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#c62828")
    lines = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
             "<rect width='100%' height='100%' fill='white'/>",
             f"<text x='45' y='28' font-size='20' font-family='Arial'>{html.escape(title)}</text>"]
    for facet_index, facet in enumerate(facet_list[:2]):
        panel_left = 20 + facet_index * panel_width
        left, top, right, bottom = panel_left + 58, 70, 175, 60
        plot_w, plot_h = panel_width - 58 - right, height - top - bottom
        series = facet.get("series", {})
        horizontal_lines = facet.get("horizontal_lines", {})
        na_reasons = facet.get("na_reasons", {})
        x, x_ticks, log_alpha = _chart_x_layout(series, left, plot_w)
        y, y_ticks, use_log_y, _, _, _ = _chart_y_layout(
            series, low=0.0, top=top, plot_h=plot_h,
            horizontal_lines=horizontal_lines, log_y=facet.get("log_y"), y_limits=facet.get("y_limits"))
        facet_title = str(facet.get("title", ""))
        y_label = str(facet.get("y_label", ""))
        lines.append(f"<text x='{panel_left + 20}' y='52' font-size='15' font-family='Arial'>{html.escape(facet_title)}</text>")
        lines.append(f"<text x='{panel_left + 12}' y='{top + plot_h/2:.1f}' font-size='11' font-family='Arial' transform='rotate(-90 {panel_left + 12} {top + plot_h/2:.1f})'>{html.escape(y_label)}</text>")
        lines.append(f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#263238'/>")
        lines.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#263238'/>")
        for value in y_ticks:
            yy = y(value)
            if yy is None:
                continue
            label = f"{value:.0e}" if use_log_y else f"{value:.3g}"
            lines.append(f"<line x1='{left-3}' y1='{yy:.2f}' x2='{left + plot_w}' y2='{yy:.2f}' stroke='#e0e0e0'/>")
            lines.append(f"<text x='{left-8}' y='{yy+4:.2f}' text-anchor='end' font-size='10' font-family='Arial'>{label}</text>")
        for name, value in horizontal_lines.items():
            yy = y(float(value))
            if yy is None:
                continue
            lines.append(f"<line x1='{left}' y1='{yy:.2f}' x2='{left + plot_w}' y2='{yy:.2f}' stroke='#455a64' stroke-width='2' stroke-dasharray='7,5'/>")
            lines.append(f"<text x='{left + plot_w - 2}' y='{yy-5:.2f}' text-anchor='end' font-size='10' font-family='Arial' fill='#455a64'>{html.escape(str(name))}</text>")
        for value in x_ticks:
            xx = x(value)
            lines.append(f"<line x1='{xx:.2f}' y1='{top + plot_h}' x2='{xx:.2f}' y2='{top + plot_h + 5}' stroke='#263238'/>")
            lines.append(f"<text x='{xx:.2f}' y='{height-38}' text-anchor='middle' font-size='10' font-family='Arial'>{html.escape(_format_x_tick(value, log_alpha=log_alpha))}</text>")
        legend_x = panel_left + panel_width - 162
        for index, (name, points) in enumerate(series.items()):
            color = colors[index % len(colors)]
            for coords in _chart_segments(points, x, y):
                if len(coords) >= 2:
                    lines.append("<polyline fill='none' stroke='%s' stroke-width='2' points='%s'/>" % (color, " ".join(f"{xx:.2f},{yy:.2f}" for xx, yy in coords)))
                for xx, yy in coords:
                    lines.append(f"<circle cx='{xx:.2f}' cy='{yy:.2f}' r='3' fill='{color}'/>")
            ly = top + 16 + index * 17
            lines.append(f"<line x1='{legend_x}' y1='{ly}' x2='{legend_x+16}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
            lines.append(f"<text x='{legend_x+21}' y='{ly+4}' font-size='10' font-family='Arial'>{html.escape(name)}</text>")
            reason = na_reasons.get(name) if isinstance(na_reasons, Mapping) else None
            if reason is None and not any(value is not None and math.isfinite(float(value)) for _, value in points):
                reason = "no finite values were recorded"
            if reason:
                lines.append(f"<text x='{left + 6}' y='{top + 16 + index * 17:.2f}' font-size='10' font-family='Arial' fill='#c62828'>N/A: {html.escape(str(reason))}</text>")
        lines.append(f"<text x='{left + plot_w/2}' y='{height-14}' text-anchor='middle' font-size='11' font-family='Arial'>{'alpha (log10 position)' if log_alpha else 'alpha'}</text>")
        lines.append(f"<text x='{panel_left + 20}' y='{height-14}' font-size='10' font-family='Arial' fill='#455a64'>{'log-y' if use_log_y else 'linear y'}</text>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _chart_png_faceted(title: str, facets: Iterable[Mapping[str, Any]]) -> bytes:
    """Render independent y-axis facets into one deterministic 1200px PNG."""

    from PIL import Image, ImageDraw
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 25), title, fill=(20, 30, 35))
    facet_list = list(facets)
    panel_width = 550
    colors = ((21, 101, 192), (46, 125, 50), (239, 108, 0), (106, 27, 154), (0, 131, 143), (198, 40, 40))
    for facet_index, facet in enumerate(facet_list[:2]):
        panel_left = 20 + facet_index * panel_width
        left, top, right, bottom = panel_left + 58, 70, 175, 60
        plot_w, plot_h = panel_width - 58 - right, height - top - bottom
        series = facet.get("series", {})
        horizontal_lines = facet.get("horizontal_lines", {})
        na_reasons = facet.get("na_reasons", {})
        x, x_ticks, log_alpha = _chart_x_layout(series, left, plot_w)
        y, y_ticks, use_log_y, _, _, _ = _chart_y_layout(series, low=0.0, top=top, plot_h=plot_h,
                                                          horizontal_lines=horizontal_lines, log_y=facet.get("log_y"),
                                                          y_limits=facet.get("y_limits"))
        draw.text((panel_left + 20, 49), str(facet.get("title", "")), fill=(20, 30, 35))
        draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=(38, 50, 56), width=2)
        draw.line((left, top, left, top + plot_h), fill=(38, 50, 56), width=2)
        for value in y_ticks:
            yy = y(value)
            if yy is None:
                continue
            yy = int(round(yy))
            label = f"{value:.0e}" if use_log_y else f"{value:.3g}"
            draw.line((left, yy, left + plot_w, yy), fill=(224, 224, 224), width=1)
            draw.text((left - 48, yy - 7), label, fill=(20, 30, 35))
        for name, value in horizontal_lines.items():
            yy = y(float(value))
            if yy is None:
                continue
            yy = int(round(yy))
            draw.line((left, yy, left + plot_w, yy), fill=(69, 90, 100), width=2)
            draw.text((left + plot_w - 155, yy - 15), str(name), fill=(69, 90, 100))
        for value in x_ticks:
            xx = int(round(x(value)))
            draw.line((xx, top + plot_h, xx, top + plot_h + 5), fill=(38, 50, 56), width=1)
            draw.text((xx - 18, height - 42), _format_x_tick(value, log_alpha=log_alpha), fill=(20, 30, 35))
        legend_x = panel_left + panel_width - 162
        for index, (name, points) in enumerate(series.items()):
            color = colors[index % len(colors)]
            for coords in _chart_segments(points, x, y):
                coords_int = [(int(round(xx)), int(round(yy))) for xx, yy in coords]
                if len(coords_int) >= 2:
                    draw.line(coords_int, fill=color, width=3)
                for xx, yy in coords_int:
                    draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=color)
            ly = top + 10 + index * 17
            draw.line((legend_x, ly, legend_x + 16, ly), fill=color, width=3)
            draw.text((legend_x + 21, ly - 7), name, fill=(20, 30, 35))
            reason = na_reasons.get(name) if isinstance(na_reasons, Mapping) else None
            if reason is None and not any(value is not None and math.isfinite(float(value)) for _, value in points):
                reason = "no finite values were recorded"
            if reason:
                draw.text((left + 6, top + 10 + index * 17), f"N/A: {reason}", fill=(198, 40, 40))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        image.save(temporary, format="PNG", optimize=False)
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def write_sha256_manifest(root: str | Path) -> Path:
    directory = Path(root).resolve()
    excluded = {"precision_manifest.sha256", "MANIFEST.sha256", "review_bundle.zip"}
    entries = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file() and item.relative_to(directory).as_posix() not in excluded):
        entries.append(f"{sha256_file(path)}  {path.relative_to(directory).as_posix()}")
    if not entries:
        raise ValueError("cannot write an empty precision manifest")
    content = "\n".join(entries) + "\n"
    precision = directory / "precision_manifest.sha256"
    precision.write_text(content, encoding="ascii")
    (directory / "MANIFEST.sha256").write_text(content, encoding="ascii")
    return precision


def _review_source_mapping(result: Mapping[str, Any]) -> Mapping[str, Any]:
    explicit = result.get("review_evidence_sources")
    if isinstance(explicit, Mapping):
        return explicit
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    nested = metadata.get("review_evidence_sources") if isinstance(metadata, Mapping) else None
    return nested if isinstance(nested, Mapping) else {}


def _copy_review_file(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"review evidence source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _reject_symlinked_path_components(value: str | Path, *, label: str) -> Path:
    """Parse an evidence path without ever resolving through a link.

    ``Path.resolve`` is intentionally delayed until after this check: resolving
    first would make a symlinked evidence root indistinguishable from the real
    directory it points at.  On Windows, junctions and other reparse points
    are covered by ``st_file_attributes`` in addition to ordinary symlinks.
    """

    raw = Path(str(value)).expanduser()
    if any(part in {".", ".."} for part in raw.parts):
        raise ValueError(f"unsafe review evidence {label}: {value}")
    candidate = raw if raw.is_absolute() else Path.cwd() / raw
    current = Path(candidate.anchor) if candidate.anchor else Path.cwd()
    parts = candidate.parts[1:] if candidate.anchor else candidate.parts
    for part in parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            # A missing tail is handled by the caller; existing ancestors have
            # already been checked and no untrusted resolution occurred.
            continue
        attributes = getattr(info, "st_file_attributes", 0)
        if current.is_symlink() or (attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)):
            raise ValueError(f"review evidence {label} contains symlink/reparse component: {current}")
    try:
        return candidate.resolve(strict=False)
    except OSError as error:
        raise ValueError(f"cannot resolve review evidence {label}: {value}") from error


def _review_dir(value: Any, *, label: str) -> Path | None:
    if value is None:
        return None
    candidate = _reject_symlinked_path_components(str(value), label=label)
    return candidate if candidate.is_dir() else None


def _runner_source_hashes(result: Mapping[str, Any]) -> dict[str, list[tuple[str, str]]]:
    """Return config-recorded source hashes keyed by basename.

    The run config stores absolute source paths.  Review evidence is copied by
    the fixed basename whitelist, so bind every basename that the real config
    records and reject malformed digest claims rather than silently treating
    them as unavailable.
    """

    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    config = metadata.get("config") if isinstance(metadata, Mapping) else {}
    recorded = config.get("source_sha256") if isinstance(config, Mapping) else None
    if recorded is None:
        return {}
    if not isinstance(recorded, Mapping):
        raise ValueError("metadata.config.source_sha256 must be an object")
    expected: dict[str, list[tuple[str, str]]] = {}
    for raw_path, digest in recorded.items():
        path = str(raw_path).replace("\\", "/")
        windows_absolute = bool(re.match(r"^[A-Za-z]:/", path))
        if not PurePosixPath(path).is_absolute() and not windows_absolute:
            raise ValueError(f"recorded runner source path must be absolute: {raw_path}")
        name = PurePosixPath(path).name
        if name not in _REVIEW_SOURCE_FILES:
            continue
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError(f"invalid recorded runner source hash: {raw_path}")
        expected.setdefault(name, []).append((str(raw_path), digest))
    return expected


def _write_review_evidence(root: Path, result: Mapping[str, Any]) -> None:
    """Copy only the reviewed runtime/source/sample evidence into the package.

    The raw run remains immutable.  This function never copies checkpoints,
    caches, or the full analysis tensor corpus; each source path and sample
    filename is checked against the fixed whitelist before copying.
    """

    mapping = _review_source_mapping(result)
    source_dir_value = mapping.get("source_dir")
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    execution = _execution_metadata(result)
    if source_dir_value is None:
        source_dir_value = execution.get("source_dir") if isinstance(execution, Mapping) else None
    source_dir = _review_dir(source_dir_value, label="source directory")
    analysis_source_dir = Path(__file__).resolve().parent
    run_dir_value = mapping.get("run_dir")
    if run_dir_value is None and isinstance(metadata, Mapping):
        run_dir_value = metadata.get("run_dir")
    run_dir = _review_dir(run_dir_value, label="run directory")
    sample_dirs = mapping.get("sample_dirs")
    if sample_dirs is None and isinstance(metadata, Mapping):
        sample_dirs = metadata.get("sample_dirs")
    if not isinstance(sample_dirs, Mapping):
        sample_dirs = {}
    # Detached synthetic/unit-test records have no real run provenance and may
    # still exercise chart generation without a review bundle.  A real GPU
    # execution always records task4_execution/source/sample bindings; once
    # any of those bindings is present, incomplete evidence is a hard error.
    strict_evidence = bool(source_dir_value or sample_dirs)
    if (isinstance(metadata, Mapping) and isinstance(metadata.get("task4_execution"), Mapping)
            and "source_dir" in metadata.get("task4_execution", {})):
        strict_evidence = True

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    optional_missing: list[str] = []
    required_sample_arrays = set(_REVIEW_REQUIRED_SAMPLE_ARRAY_FILES)
    if str(result.get("scope", "")).lower() == "full":
        required_sample_arrays.add("decoded_final.npy")
    source_hashes = _runner_source_hashes(result)

    def copy_named(source: Path, relative: str, *, role: str, recorded_sha256: str | None = None) -> None:
        if not _review_evidence_relative_allowed(relative):
            raise ValueError(f"unsafe review evidence path: {relative}")
        if recorded_sha256 is not None and sha256_file(source) != recorded_sha256:
            raise ValueError(f"runner source hash mismatch before copy: {source}")
        target = root / Path(relative)
        _copy_review_file(source, target)
        copied_sha256 = sha256_file(target)
        if recorded_sha256 is not None and copied_sha256 != recorded_sha256:
            raise ValueError(f"runner source hash mismatch after copy: {relative}")
        entries.append({"role": role, "path": relative,
                        "sha256": copied_sha256, "recorded_sha256": recorded_sha256,
                        "verified": recorded_sha256 is not None and copied_sha256 == recorded_sha256,
                        "size": target.stat().st_size})

    # analysis_source is always the currently executing analyzer/packager
    # tree.  The separate source/ role, when available, identifies the exact
    # runner/bridge source used by the GPU run (and may be an older commit).
    for name in _REVIEW_SOURCE_FILES:
        current = analysis_source_dir / name
        if current.is_file():
            copy_named(current, f"review_evidence/analysis_source/{name}", role="analysis_source")
    if source_dir is not None:
        for name in _REVIEW_SOURCE_FILES:
            source = source_dir / name
            if source.is_file():
                expected = source_hashes.get(name, [])
                expected_digest = expected[0][1] if expected else None
                # A basename can occur more than once in a source snapshot.
                # Accept it only when the copied bytes match one of the
                # recorded absolute-path claims.
                if len(expected) > 1:
                    observed = sha256_file(source)
                    if observed not in {digest for _, digest in expected}:
                        raise ValueError(f"runner source hash mismatch before copy: {source}")
                    expected_digest = observed
                copy_named(source, f"review_evidence/source/{name}", role="runner_source",
                           recorded_sha256=expected_digest)
            elif name not in _REVIEW_SOURCE_OPTIONAL_FILES:
                missing.append(f"runner_source/{name}")
    else:
        missing.append("runner_source/source_dir")

    if run_dir is not None:
        for name in _REVIEW_RUN_FILES:
            source = run_dir / name
            if source.is_file():
                copy_named(source, f"review_evidence/run/{name}", role="run_provenance")
            elif name not in _REVIEW_RUN_OPTIONAL_FILES:
                missing.append(f"run_provenance/{name}")
    else:
        missing.append("run_provenance/run_dir")

    if sample_dirs and run_dir is None:
        raise ValueError("sample evidence requires a real run directory")
    for sample_id in _REVIEW_SAMPLE_IDS:
        raw = sample_dirs.get(sample_id)
        if raw is None:
            missing.append(f"sample/{sample_id}")
            continue
        sample_dir = _review_dir(raw, label=f"sample directory {sample_id}")
        if run_dir is None or sample_dir is None:
            missing.append(f"sample/{sample_id}")
            continue
        expected_dir = (run_dir / "samples" / sample_id).resolve(strict=False)
        if sample_dir != expected_dir:
            raise ValueError(f"sample directory mapping escapes run samples: {sample_id}")
        if not sample_dir.is_dir():
            missing.append(f"sample/{sample_id}")
            continue
        try:
            sample_status = _load_json(sample_dir / "status.json")
        except ValueError as error:
            raise ValueError(f"sample {sample_id} status evidence is invalid") from error
        artifact_hashes = sample_status.get("artifact_sha256")
        if not isinstance(artifact_hashes, Mapping):
            raise ValueError(f"sample {sample_id} status lacks artifact_sha256")
        for name in ("sample.json", "status.json", *_REVIEW_SAMPLE_ARRAY_FILES):
            source = sample_dir / name
            if source.is_file():
                if name == "status.json":
                    # status.json necessarily cannot hash itself; its
                    # artifact_sha256 map must cover every other copied file.
                    copy_named(source, f"review_evidence/samples/{sample_id}/{name}", role="representative_sample")
                    continue
                recorded_digest = artifact_hashes.get(name)
                if not isinstance(recorded_digest, str) or not _SHA256.fullmatch(recorded_digest):
                    raise ValueError(f"sample {sample_id} lacks valid artifact hash: {name}")
                copy_named(source, f"review_evidence/samples/{sample_id}/{name}", role="representative_sample",
                           recorded_sha256=recorded_digest)
            elif name in required_sample_arrays:
                missing.append(f"sample/{sample_id}/{name}")
            elif name in {"sample.json", "status.json"}:
                missing.append(f"sample/{sample_id}/{name}")
            elif name in _REVIEW_SAMPLE_ARRAY_FILES:
                optional_missing.append(f"sample/{sample_id}/{name}")
    if missing and strict_evidence:
        raise ValueError("required review evidence is missing: " + ", ".join(sorted(set(missing))))
    manifest = {
        "schema_version": "umi-task4-review-evidence-v1",
        "status": "COMPLETE" if not missing else "PARTIAL",
        "roles": {
            "analysis_source": "current post-processing source used to produce this package",
            "runner_source": "source_dir recorded by the GPU run, if available",
            "run_provenance": "config/status/task4_execution/provenance from the recorded run",
            "representative_sample": "A/B/C baselines and minimum/maximum alpha positive/negative samples",
        },
        "entries": sorted(entries, key=lambda item: (item["role"], item["path"])),
        "missing": sorted(set(missing)),
        "optional_missing": sorted(set(optional_missing)),
        "sample_ids": list(_REVIEW_SAMPLE_IDS),
        "excluded": ["weights", "diffusion caches", "full analysis_tensors/**", "arbitrary source files"],
    }
    _write_json(root / "review_evidence_manifest.json", manifest)


def _write_bundle(root: Path) -> Path:
    destination = root / "review_bundle.zip"
    allowed = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == destination.name:
            continue
        relative = path.relative_to(root).as_posix()
        pure = PurePosixPath(relative)
        if not pure.parts:
            continue
        if len(pure.parts) == 1:
            if pure.parts[0] not in _BUNDLE_ROOT_FILES:
                continue
        elif pure.parts[:1] == ("calibration",):
            if len(pure.parts) != 2 or pure.parts[1] not in _CALIBRATION_FILES:
                continue
        elif pure.parts[:1] == ("review_evidence",):
            if not _review_evidence_relative_allowed(relative):
                continue
        else:
            # No implicit recursive directory allowance: analysis tensors,
            # arbitrary nested files, and unknown top-level directories stay
            # out of the lightweight review bundle.
            continue
        if path.suffix.lower() in {".pt", ".pth", ".mp4"}:
            continue
        if path.suffix.lower() == ".npy" and not (
            len(pure.parts) == 4 and pure.parts[:2] == ("review_evidence", "samples")
            and _review_evidence_relative_allowed(relative)
        ):
            continue
        if pure.parts[:1] == ("review_evidence",) and not _review_evidence_relative_allowed(relative):
            continue
        allowed.append(path)
    temporary = root / ".review_bundle.zip.tmp"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in allowed:
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    os.replace(temporary, destination)
    return destination


def _save_analysis_tensor(root: Path, relative: str, value: Any) -> str:
    """Save a raw analysis tensor in the remote package, never in the review zip."""

    # Keep the exact FP64 subtraction/derivative evidence that produced the
    # public norms and hashes; the review ZIP still excludes these raw arrays.
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"analysis tensor is empty or nonfinite: {relative}")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return relative


def _write_analysis_tensors(root: Path, result: dict[str, Any]) -> None:
    """Persist signed decomposition and derivative vectors outside the review bundle."""

    def label(space: str, group: str, alpha: float, sign: int) -> str:
        sign_name = "plus" if sign == 1 else "minus"
        return f"{space}_{group}_alpha_{alpha:.0e}_{sign_name}"

    public_pairs = result.get("paired_metrics", [])
    for pair in result.get("_paired_tensors", []):
        stem = label(str(pair["space"]), str(pair["group"]), float(pair["alpha"]), 1)
        key = (pair["space"], pair["group"], float(pair["alpha"]))
        public = next((row for row in public_pairs if (row.get("space"), row.get("group"), float(row.get("alpha"))) == key), None)
        if public is None:
            raise ValueError(f"paired public metric is missing: {key}")
        for field, directory, source in (
            ("target_centered_derivative_path", "target_centered_derivative", "_target_centered_derivative"),
            ("actual_paired_secant_path", "actual_paired_secant", "_paired_secant"),
            ("one_sided_plus_derivative_path", "one_sided_plus_derivative", "_one_sided_plus_derivative"),
            ("one_sided_minus_derivative_path", "one_sided_minus_derivative", "_one_sided_minus_derivative"),
        ):
            value = pair.get(source)
            if value is not None:
                relative = f"analysis_tensors/{directory}/{stem}.npy"
                public[field] = _save_analysis_tensor(root, relative, value)

    public_vectors = result.get("vector_decomposition", [])
    for vector in result.get("_vector_tensors", []):
        stem = label(str(vector["space"]), "ABC", float(vector["alpha"]), int(vector["sign"]))
        key = (vector["space"], float(vector["alpha"]), int(vector["sign"]))
        public = next((row for row in public_vectors if (row.get("space"), float(row.get("alpha")), int(row.get("sign"))) == key), None)
        if public is None:
            raise ValueError(f"vector public metric is missing: {key}")
        for field, directory, source in (
            ("Delta_quant_path", "Delta_quant", "Delta_quant"),
            ("Delta_compute_path", "Delta_compute", "Delta_compute"),
            ("r_A_minus_r_C_path", "r_A_minus_r_C", "r_A_minus_r_C"),
            ("reconstructed_path", "reconstructed", "reconstructed"),
        ):
            relative = f"analysis_tensors/{directory}/{stem}.npy"
            public[field] = _save_analysis_tensor(root, relative, vector[source])


def _execution_metadata(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    execution = metadata.get("task4_execution") if isinstance(metadata, Mapping) else None
    if not isinstance(execution, Mapping):
        execution = metadata.get("execution") if isinstance(metadata, Mapping) else None
    return execution if isinstance(execution, Mapping) else {}


def _write_calibration_artifacts(root: Path) -> None:
    """Copy and verify the checked-in CPU calibration evidence into the package."""

    source = Path(__file__).resolve().parent / "artifacts" / "umi_precision_calibration"
    math_note = Path(__file__).resolve().parent / "umi_precision_calibration_math.md"
    if not source.is_dir() or not math_note.is_file():
        raise ValueError("verified CPU calibration artifacts are missing from the source tree")
    try:
        hashes = json.loads((source / "hashes.json").read_text(encoding="utf-8"))
        expected = hashes["files"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"calibration hashes are invalid: {error}") from error
    if set(expected) != set(_CALIBRATION_SOURCE_FILES) - {"hashes.json"}:
        # hashes.json describes the reviewed files and is copied as evidence,
        # but intentionally does not hash itself.
        raise ValueError("calibration hashes do not cover the reviewed artifact set")
    destination = root / "calibration"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in _CALIBRATION_SOURCE_FILES:
        source_path = source / name
        if not source_path.is_file() or (name != "hashes.json" and sha256_file(source_path) != str(expected.get(name))):
            raise ValueError(f"calibration artifact hash mismatch: {name}")
        target = destination / name
        shutil.copyfile(source_path, target)
        copied.append(target)
    target_math = destination / "umi_precision_calibration_math.md"
    shutil.copyfile(math_note, target_math)
    copied.append(target_math)
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(copied)]
    (root / "calibration_manifest.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def _commands_for_result(result: Mapping[str, Any]) -> str:
    execution = _execution_metadata(result)
    metadata = result.get("metadata") if isinstance(result.get("metadata"), Mapping) else {}
    python = str(execution.get("python") or _REMOTE_PYTHON)
    source = str(execution.get("source_dir") or _REMOTE_SOURCE)
    framework = str(execution.get("framework_root") or _REMOTE_FRAMEWORK)
    checkpoint = str(execution.get("checkpoint_path") or _REMOTE_CHECKPOINT)
    old_run = str(execution.get("old_run09") or _REMOTE_OLD_RUN)
    run_dir = str(execution.get("run_dir") or (metadata.get("run_dir") if isinstance(metadata, Mapping) else "") or "<recorded-run-directory>")
    reanalysis = str(execution.get("reanalysis_output") or f"{old_run}_precision_reanalysis_task4")
    status_name = str(execution.get("run_status") or (metadata.get("status", {}).get("status") if isinstance(metadata.get("status"), Mapping) else None) or result.get("status") or "").lower()
    def q(value: Any) -> str:
        return shlex.quote(str(value))
    def source_script(name: str) -> str:
        return source.rstrip("/") + "/" + name
    old_command = f"{q(python)} {q(source_script('umi_precision_reanalysis.py'))} --source-dir {q(old_run)} --output-dir {q(reanalysis)}"
    run_parts = [q(python), q(source_script("run_umi_precision_experiment.py")),
                 "--framework-root", q(framework), "--checkpoint-path", q(checkpoint),
                 "--old-run09", q(old_run), "--run-dir", q(run_dir), "--direction-seed", "20260912"]
    for flag, key in (("--vae-path", "vae_path"), ("--input-path", "input_path"), ("--action-path", "action_path"), ("--prompt", "prompt")):
        if execution.get(key):
            run_parts.extend((flag, q(execution[key])))
    skip_old_reanalysis = bool(execution.get("skip_old_reanalysis", True))
    run_command = " ".join(run_parts + (["--skip-old-reanalysis"] if skip_old_reanalysis else []))
    # Reanalysis is a one-time pre-run action.  A formal run that was
    # interrupted must always resume with it disabled, even when the initial
    # invocation deliberately performed it.
    resume_command = " ".join(run_parts + ["--skip-old-reanalysis", "--resume"])
    analyze_command = f"{q(python)} {q(source_script('analyze_umi_precision_contrast.py'))} --run-dir {q(run_dir)} --output-dir {q(run_dir.rstrip('/') + '/precision_analysis')}"
    if status_name in {"blocked", "terminal_blocked"}:
        resume_section = "Terminal `BLOCKED` is not resumable. Preserve this evidence and allocate a fresh run revision; no resume command is permitted."
    elif status_name in {"complete", "module_complete"}:
        resume_section = "This run is terminal and complete. The launcher only supports a read-only hash-verified resume check; it performs zero model calls and writes no files. Do not rerun the model in this revision."
    else:
        resume_section = f"```bash\n{resume_command}\n```"
    invocation_evidence = ""
    if execution.get("argv") is not None or execution.get("effective_args") is not None:
        invocation_evidence = "\n## Recorded invocation evidence\n\n```json\n" + json.dumps(
            {"argv": execution.get("argv"), "effective_args": execution.get("effective_args"), "prompt": execution.get("prompt")},
            ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n```\n"
    return f"""# Exact commands

All commands below are the recorded cu130 interpreter and resolved source paths. The formal Task 4 table is fixed at 42 calls; diagnostic attempts are counted separately. The old run is reanalyzed from immutable raw tensors and is never rerun.

## Old run09 CPU-only reanalysis (exit 0 required)

```bash
{old_command}
```

## New formal run (fresh lowest unused runNN only)

```bash
{run_command}
```

## Resume after a blocked/interrupted run (hash-verified samples only)

{resume_section}

## Offline analysis and package publication (no model load/GPU call)

```bash
{analyze_command}
```
""" + invocation_evidence


def _gpu_timing_summary(result: Mapping[str, Any], timing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    execution = _execution_metadata(result)
    torch_info = execution.get("torch") if isinstance(execution.get("torch"), Mapping) else {}
    device = execution.get("device") or execution.get("gpu_device") or torch_info.get("device")
    device_name = execution.get("device_name") or execution.get("gpu_name")
    peak = execution.get("peak_memory_bytes")
    if peak is None and isinstance(execution.get("gpu"), Mapping):
        peak = execution["gpu"].get("peak_memory_bytes")
    try:
        peak = None if peak is None else int(peak)
    except (TypeError, ValueError):
        peak = None
    return {
        "scope": result.get("scope"), "sample_count": len(timing_rows), "rows": timing_rows,
        "device": device, "device_name": device_name, "peak_memory_bytes": peak,
        "peak_memory_gib": None if peak is None else peak / float(1024 ** 3),
        "peak_memory_reason": None if peak is not None else "runner did not record torch.cuda.max_memory_allocated",
        "requested_backend": execution.get("requested_backend") or execution.get("model_compile", {}).get("requested") if isinstance(execution.get("model_compile"), Mapping) else execution.get("requested_backend"),
        "resolved_backend": execution.get("resolved_backend") or "unknown",
        "telemetry_claim": "only recorded run telemetry is reported; missing GPU fields remain N/A",
    }


def _report_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if not math.isfinite(number) else f"{number:.3e}"


def _report_fit(fits: Iterable[Mapping[str, Any]], sign: int) -> str:
    fit = next((row for row in fits if int(row.get("sign", 0)) == sign), None)
    if not fit:
        return "N/A"
    return f"slope={_report_number(fit.get('slope'))}, R²={_report_number(fit.get('r2'))}, n={fit.get('points', 'N/A')}"


def _report_range(values: Iterable[Any]) -> str:
    finite: list[float] = []
    for value in values:
        try:
            converted = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(converted):
            finite.append(converted)
    if not finite:
        return "N/A"
    low, high = min(finite), max(finite)
    return _report_number(low) if low == high else f"{_report_number(low)}–{_report_number(high)}"


def _report_input_edge(points: Iterable[Mapping[str, Any]], alpha: float | None) -> str:
    if alpha is None:
        return "N/A"
    edge = [row for row in points if row.get("alpha") is not None and math.isclose(float(row["alpha"]), alpha, rel_tol=0.0, abs_tol=1e-12)]
    ratios: list[float] = []
    actual_rms: list[float] = []
    cosine: list[float] = []
    nonzero: list[float] = []
    for row in edge:
        actual = row.get("input_rms")
        target = row.get("target_input_rms")
        if actual is not None and target not in (None, 0.0):
            try:
                if math.isfinite(float(actual)) and math.isfinite(float(target)):
                    ratios.append(float(actual) / float(target))
            except (TypeError, ValueError):
                pass
        if actual is not None:
            actual_rms.append(actual)
        if row.get("input_direction_cosine") is not None:
            cosine.append(row["input_direction_cosine"])
        if row.get("input_nonzero_ratio") is not None:
            nonzero.append(row["input_nonzero_ratio"])
    return f"actual/target={_report_range(ratios)}; actual RMS={_report_range(actual_rms)}; input_direction_cosine={_report_range(cosine)}; nonzero ratio={_report_range(nonzero)}"


def _report_group_space_rows(result: Mapping[str, Any]) -> list[str]:
    point_metrics = result.get("point_metrics", [])
    fit_metrics = result.get("fit_metrics", [])
    decisions = result.get("window_decisions", [])
    floors = result.get("baseline_floors", {})
    rows: list[str] = []
    for group in GROUPS:
        for space in SPACES:
            points = [row for row in point_metrics if row.get("group") == group and row.get("space") == space]
            values = [float(row["input_rms"]) for row in points if row.get("input_rms") is not None and math.isfinite(float(row["input_rms"]))]
            responses = [float(row["output_rms"]) for row in points if row.get("output_rms") is not None and math.isfinite(float(row["output_rms"]))]
            alpha_values = [float(row["alpha"]) for row in points if row.get("alpha") is not None and math.isfinite(float(row["alpha"]))]
            min_alpha = f"{min(alpha_values):.0e}" if alpha_values else "N/A"
            max_alpha = f"{max(alpha_values):.0e}" if alpha_values else "N/A"
            min_alpha_value = min(alpha_values) if alpha_values else None
            max_alpha_value = max(alpha_values) if alpha_values else None
            windows = [row for row in decisions if row.get("group") == group and row.get("space") == space]
            selected = sum(1 for row in windows if row.get("selected"))
            fits = [row for row in fit_metrics if row.get("group") == group and row.get("space") == space]
            floor = floors.get(group, {}).get(space) if isinstance(floors, Mapping) else None
            rows.append(
                f"| {group} / {space} | 窗口数 {selected}/{len(windows)} | 全幅拟合 (+: {_report_fit(fits, 1)}; −: {_report_fit(fits, -1)}) | "
                f"最小 alpha={min_alpha} 输入失真（{_report_input_edge(points, min_alpha_value)}） | 最大 alpha={max_alpha} 输入失真（{_report_input_edge(points, max_alpha_value)}） | "
                f"响应范围 {_report_number(min(responses) if responses else None)}–{_report_number(max(responses) if responses else None)} | baseline floor {_report_number(floor)} |")
    return rows


def _write_report(root: Path, result: Mapping[str, Any]) -> None:
    scope = str(result.get("scope"))
    rgb = result.get("spaces", {}).get("decoded_final_rgb", {})
    rgb_line = "可用并单独分析。" if rgb.get("status") == "OK" else f"N/A：{rgb.get('reason') or '未提供固定解码 RGB 张量。'}"
    blocked = "；".join(result.get("blocked_reasons", [])) or "无"
    decisions = result.get("window_decisions", [])
    selected = sum(1 for item in decisions if item.get("selected"))
    execution = _execution_metadata(result)
    timing = result.get("timing") if isinstance(result.get("timing"), Mapping) else {}
    finite_timings = [float(value) for value in timing.values()
                      if isinstance(value, (int, float)) and math.isfinite(float(value))]
    elapsed = sum(finite_timings) if finite_timings else None
    elapsed_text = f"{elapsed:.3f} s" if elapsed is not None else "N/A"
    peak = execution.get("peak_memory_bytes")
    peak_text = "N/A" if peak is None else f"{peak} bytes ({float(peak) / 1024 ** 3:.3f} GiB)"
    report_rows = "\n".join(_report_group_space_rows(result))
    report = f"""# UMI 输入量化/计算精度对照 Task 4 报告

## 结论摘要

- 分析状态：`{result.get('status')}`；scope：`{scope}`；formal 调用：`{result.get('formal_call_count')}/{result.get('expected_formal_call_count')}`，diagnostic 单独计数：`{result.get('diagnostic_call_count')}`；固定预算记为 **42+diagnostic**，不可把诊断计入正式调用。
- 预测 latent：`{result.get('spaces', {}).get('predicted_latent', {}).get('status')}`；固定 decoder 最终 RGB：{rgb_line}
- A/B 共享输入逐字节恒等：`{result.get('input_identity', {}).get('A_B_all_exact')}`；B/C 计算路径恒等：`{result.get('compute_identity', {}).get('B_C_same_path')}`；cache off：`{result.get('cache_identity', {}).get('cache_off_all')}`；实际初始噪声配对：`{result.get('noise_identity', {}).get('all_equal')}`。
- 通过的候选连续 alpha 窗口：`{selected}/{len(decisions)}`。门限严格使用 adjacent input cosine ≥0.99、正负输入 cosine ≥0.99、两侧 slope ∈[0.8,1.2] 且 R²≥0.98、跨幅度 paired-secant cosine ≥0.95、相对变化 ≤0.25，以及 response > 10× group repeat floor（floor=0 时仅要求非零）。
- 阻断/N/A 原因：{blocked}

## A/B/C、latent/RGB 数值结论

下表来自实际 `point_metrics`、`fit_metrics`、`window_decisions` 与 `baseline_floors`，不是示意数值；输入失真使用实际消费接口的 RMS，最小/最大 alpha 均在正负样本中取范围。`RGB` 为固定 decoder 最终输出，fallback 时明确 N/A。

| group / space | 窗口数 | 全幅拟合 | 最小 alpha 输入失真 | 最大 alpha 输入失真 | 响应范围 | baseline floor |
|---|---:|---|---:|---:|---:|---:|
{report_rows}

## GPU、调用与量的边界

- formal/diagnostic：`{result.get('formal_call_count')} + {result.get('diagnostic_call_count')}`；记录样本累计耗时：`{elapsed_text}`；GPU：`{execution.get('device') or 'N/A'}` / `{execution.get('device_name') or 'N/A'}`；峰值显存：`{peak_text}`。缺失 timing/telemetry 保持 N/A，不由离线分析推断。
- `r_A`、`r_B`、`r_C` 是各组 perturbation 输出减去本组 pre baseline 的 response。签名向量定义为 `Delta_quant = r_B-r_C`、`Delta_compute = r_A-r_B`，并逐项验证 `Delta_quant+Delta_compute = r_A-r_C`。**Delta 仅是范数/向量差异，不是贡献百分比**；相对范数有分母时也不能解释为因果贡献百分比。
- full scope：预测 latent 与同一固定 decoder 的最终 RGB 分开分析；module fallback 只运行真实 sampler step 0 的完整 denoiser，不 scheduler update、不 decode，因此 RGB 全部 N/A，不能借用 latent 结果替代。

## 证据边界、N/A 与复现

- 每个样本均检查 cache 请求/安装状态、实际输入、实际 compute dispatch、消费的初始状态/噪声、finite/shape 与 dtype 证据。任何零分母、零响应、缺失 tensor、缺失 pre/post floor、未满足 paired secant 定义或 fallback 无 RGB 的情况均保留 `None` 和原因，不用 0 伪装。
- **证据边界**：本报告只声称 `metadata`/原始张量/哈希实际记录的内容；离线 analyzer 不加载模型、不重跑样本、不把 CPU 校准当作 Cosmos 结果。review bundle 的 source/run/sample 白名单和 SHA256 见 `review_evidence_manifest.json`；`analysis_tensors/` 不进入 ZIP。

## 校准与数学边界

CPU 校准与条件误差界说明是方法校准；它们不估计 Cosmos 的 L/M/η，也不替代本次真实模型证据。请单独阅读 `calibration/` 中的 calibration report 与 math note，勿把校准曲线当作 Cosmos 结论。

## 复现与归档

精确 run/resume 命令写入 `commands.md`；GPU/timing 汇总写入 `gpu_timing_summary.json`，设备/峰值显存若未被 runner 记录则明确为 N/A。`calibration/` 与 `calibration_manifest.sha256` 是已核验的 CPU 校准证据，和 Cosmos GPU 结果分开陈述。`precision_manifest.sha256` 和 `MANIFEST.sha256` 覆盖本地轻量产物，`review_bundle.zip` 固定顺序/时间戳，未复制权重/缓存或全量 analysis tensor。
"""
    root.joinpath("precision_report_zh.md").write_text(report, encoding="utf-8")
    path_note = f"""# Precision path note

- scope: `{scope}`
- A: BF16 model/network path after the shared quantized FP32 interface.
- B: strict FP32 input + FP32 compute path.
- C: strict FP32 input + the same FP32 compute path as B; C differs only at the requested interface value.
- Cache is required off and is checked from actual request state.
- If full FP32 compatibility fails, the only allowed fallback is one complete denoiser at sampler step 0 with frozen state/time/action/noise and no scheduler update/decode. This note does not claim RGB evidence for fallback.
- No model was loaded by offline analysis; model claims come only from the recorded run evidence.
"""
    root.joinpath("precision_path_note.md").write_text(path_note, encoding="utf-8")
    root.joinpath("commands.md").write_text(_commands_for_result(result), encoding="utf-8")


def write_precision_artifacts(data: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Analyze (if needed) and publish immutable deterministic artifacts.

    Publication is deliberately no-replace.  A new analysis revision gets a
    new directory; silently deleting an existing package would invalidate the
    raw-evidence binding and review trail.
    """

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    existing = [path for path in root.iterdir() if path.name not in {".runner.lock"}]
    if existing:
        raise FileExistsError(f"precision artifact directory is not empty: {root}")
    if "spaces" in data and "point_metrics" in data:
        result = dict(data)
    else:
        result = analyze_records(data)
    _write_analysis_tensors(root, result)
    _write_json(root / "precision_summary.json", result)
    _write_json(root / "candidate_decisions.json", result.get("window_decisions", []))
    _write_csv(root / "precision_metrics.csv", result.get("point_metrics", []))
    _write_csv(root / "paired_secant_metrics.csv", result.get("paired_metrics", []))
    _write_csv(root / "adjacent_consistency.csv", result.get("adjacent_metrics", []))
    _write_csv(root / "fit_metrics.csv", result.get("fit_metrics", []))
    _write_csv(root / "window_decisions.csv", result.get("window_decisions", []))
    _write_csv(root / "vector_decomposition.csv", result.get("vector_decomposition", []))
    _write_report(root, result)
    timing_rows = [{"sample_id": key, "elapsed_seconds": value} for key, value in sorted((result.get("timing") or {}).items())]
    _write_json(root / "gpu_timing_summary.json", _gpu_timing_summary(result, timing_rows))
    _write_csv(root / "gpu_timing_summary.csv", timing_rows)

    def alpha_series(rows: Iterable[Mapping[str, Any]], field: str, *, space: str | None = None,
                     group: str | None = None) -> list[tuple[float, float | None]]:
        selected = [row for row in rows if (space is None or row.get("space") == space) and (group is None or row.get("group") == group)]
        output: list[tuple[float, float | None]] = []
        for alpha in ALPHAS:
            values = []
            for row in selected:
                try:
                    if float(row.get("alpha")) == alpha and row.get(field) is not None and math.isfinite(float(row[field])):
                        values.append(float(row[field]))
                except (TypeError, ValueError):
                    continue
            output.append((alpha, max(values) if values else None))
        return output

    response_series: dict[str, list[tuple[float, float | None]]] = {}
    for group in GROUPS:
        response_series[group] = alpha_series(result.get("point_metrics", []), "output_rms", space="predicted_latent", group=group)
    slope_reference = alpha_series(result.get("point_metrics", []), "target_input_rms", space="predicted_latent", group="A")
    if not any(value is not None for _, value in slope_reference):
        slope_reference = alpha_series(result.get("point_metrics", []), "target_input_rms", space="predicted_latent")
    response_series["slope=1 ref (visual only)"] = slope_reference
    window_series: dict[str, list[tuple[float, float | None]]] = {}
    for space in SPACES:
        values = [1.0 if row.get("selected") else 0.0 for row in result.get("window_decisions", []) if row.get("space") == space]
        window_series[space] = [(float(index), value) for index, value in enumerate(values)] or [(0.0, None)]
    rgb_na_reason = result.get("spaces", {}).get("decoded_final_rgb", {}).get("reason") or "decoded RGB output is unavailable"
    latent_rgb_series: dict[str, list[tuple[float, float | None]]] = {}
    latent_rgb_na: dict[str, str] = {}
    for group in GROUPS:
        latent_rgb_series[f"latent {group}"] = alpha_series(result.get("point_metrics", []), "output_rms", space="predicted_latent", group=group)
        rgb_values = alpha_series(result.get("point_metrics", []), "output_rms", space="decoded_final_rgb", group=group)
        latent_rgb_series[f"RGB {group}"] = rgb_values
        if not any(value is not None for _, value in rgb_values):
            latent_rgb_na[f"RGB {group}"] = rgb_na_reason
    latent_rgb_series["slope=1 ref (visual only)"] = slope_reference

    derivative_cosine_series: dict[str, list[tuple[float, float | None]]] = {}
    derivative_relative_series: dict[str, list[tuple[float, float | None]]] = {}
    for group in GROUPS:
        adjacent = sorted(
            [row for row in result.get("adjacent_metrics", [])
             if row.get("space") == "predicted_latent" and row.get("group") == group],
            key=lambda row: float(row.get("alpha_right", 0.0)),
        )
        derivative_cosine_series[f"{group} adjacent paired-secant cosine"] = [
            (float(row["alpha_right"]), row.get("paired_secant_cosine")) for row in adjacent]
        derivative_relative_series[f"{group} adjacent paired-secant relative change"] = [
            (float(row["alpha_right"]), row.get("paired_secant_relative_change")) for row in adjacent]
    delta_series: dict[str, list[tuple[float, float | None]]] = {}
    for space in SPACES:
        delta_series[f"{space} Delta_quant"] = alpha_series(result.get("vector_decomposition", []), "Delta_quant_rms", space=space)
        delta_series[f"{space} Delta_compute"] = alpha_series(result.get("vector_decomposition", []), "Delta_compute_rms", space=space)

    chart_specs = {
        "response_rms": ("Response RMS by group (max over ±; slope=1 ref visual only, normalized by target input RMS; not an amplitude/fit comparison)", response_series, "response RMS", {}, True),
        "window_decisions": ("Candidate window decisions (1=selected)", window_series, "decision", {}, False),
        "input_distortion": ("Actual input distortion versus fixed alpha", {f"{group} actual input RMS": alpha_series(result.get("point_metrics", []), "input_rms", space="predicted_latent", group=group) for group in GROUPS} | {f"{group} target input RMS": alpha_series(result.get("point_metrics", []), "target_input_rms", space="predicted_latent", group=group) for group in GROUPS}, "input RMS", {}, True),
        "latent_rgb_response": ("Latent and fixed-decoder RGB response (max over ±; slope=1 ref visual only, normalized by target input RMS; not an amplitude/fit comparison)", latent_rgb_series, "response RMS", latent_rgb_na, True),
        "delta_decomposition": ("Signed Delta decomposition (max over ±)", delta_series, "Delta RMS", {}, True),
    }
    for name, (title, series, ylabel, na_reasons, use_log_y) in chart_specs.items():
        root.joinpath(f"{name}.svg").write_text(_chart_svg(title, series, y_label=ylabel, na_reasons=na_reasons, log_y=use_log_y), encoding="utf-8")
        root.joinpath(f"{name}.png").write_bytes(_chart_png(title, series, y_label=ylabel, na_reasons=na_reasons, log_y=use_log_y))
    derivative_facets = [
        {"title": "Adjacent paired-secant cosine", "series": derivative_cosine_series,
         "y_label": "cosine", "y_limits": (-1.0, 1.0),
         "horizontal_lines": {"threshold 0.95": THRESHOLDS["paired_secant_cosine"]}},
        {"title": "Adjacent paired-secant relative change", "series": derivative_relative_series,
         "y_label": "relative change", "horizontal_lines": {"threshold 0.25": THRESHOLDS["paired_secant_relative_change"]}},
    ]
    root.joinpath("derivative_consistency.svg").write_text(
        _chart_svg_faceted("Adjacent paired-secant consistency (independent y axes)", derivative_facets), encoding="utf-8")
    root.joinpath("derivative_consistency.png").write_bytes(_chart_png_faceted(
        "Adjacent paired-secant consistency (independent y axes)", derivative_facets))
    # Stable, descriptive aliases make the artifact self-explanatory to
    # reviewers while retaining the short names used by the test contract.
    aliases = {"precision_response": "response_rms", "precision_windows": "window_decisions"}
    for alias, source in aliases.items():
        for suffix in (".png", ".svg"):
            root.joinpath(alias + suffix).write_bytes(root.joinpath(source + suffix).read_bytes())
    _write_calibration_artifacts(root)
    _write_review_evidence(root, result)
    write_sha256_manifest(root)
    _write_bundle(root)
    artifact_paths = sorted(path.name for path in root.iterdir() if path.is_file())
    return {"status": result.get("status", "COMPLETE"), "output_dir": str(root), "artifact_paths": artifact_paths}


package_precision_artifacts = write_precision_artifacts
analyze_precision_run = analyze_run
load_precision_run = _load_precision_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline UMI precision contrast analyzer/packager")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = analyze_run(args.run_dir, args.output_dir)
    print(json.dumps(_json_safe({key: value for key, value in result.items() if key not in {"point_metrics", "paired_metrics", "adjacent_metrics", "fit_metrics", "window_decisions", "vector_decomposition"}}), sort_keys=True, ensure_ascii=False))
    return 0 if result.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALPHAS", "FORMAL_CALL_COUNT", "GROUPS", "MIN_WINDOW", "SPACES", "THRESHOLDS",
    "analyze_records", "analyze_run", "analyze_precision_run", "decompose_precision_responses",
    "evaluate_window", "load_precision_run", "package_precision_artifacts", "sha256_file",
    "write_precision_artifacts", "write_sha256_manifest",
]
