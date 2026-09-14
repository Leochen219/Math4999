"""Offline CPU analysis for the UMI post-VAE scan.

This module intentionally has no Cosmos or Torch dependency.  It consumes only
the runner's detached NumPy evidence and refuses quantitative work when the
provenance, manifest, or per-sample evidence contract is incomplete.
"""

from __future__ import annotations

import csv
import html
import hashlib
import io
import json
import math
import os
import shutil
import struct
import tempfile
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import re

import numpy as np

try:
    from .umi_fd_post_vae_scan import ALPHAS, COMPATIBILITY_FIELDS, DIRECTION_COUNT, DIRECTION_SEED, build_call_plan, construct_delta, generate_directions_for_mask, hash_predicted_noisy_region, sha256_file, sha256_tree, validate_sample_evidence, write_sha256_manifest
    from .umi_fd_post_vae_bridge import _bf16_round
except ImportError:
    from umi_fd_post_vae_scan import ALPHAS, COMPATIBILITY_FIELDS, DIRECTION_COUNT, DIRECTION_SEED, build_call_plan, construct_delta, generate_directions_for_mask, hash_predicted_noisy_region, sha256_file, sha256_tree, validate_sample_evidence, write_sha256_manifest
    from umi_fd_post_vae_bridge import _bf16_round


SPACES = ("predicted_latent", "final_rgb")
SIGNS = (-1, 1)
MIN_WINDOW = 3
WINDOW_INDEX_PAIRS = tuple(
    (start, end)
    for start in range(len(ALPHAS))
    for end in range(start + MIN_WINDOW - 1, len(ALPHAS))
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PROVENANCE_KEYS = {
    "framework_root", "checkpoint_path", "vae_path", "input_path", "action_path",
    "framework_sha256", "checkpoint_sha256", "model_sha256", "vae_sha256",
    "runner_sha256", "code_sha256", "bridge_sha256", "input_sha256", "action_sha256",
    "config_sha256", "direction_sha256", "noise_policy_sha256", "z0_sha256", "mask_sha256",
}


def _array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _finite(value: Any) -> bool:
    try:
        array = _array(value)
        return array.size > 0 and bool(np.all(np.isfinite(array)))
    except (TypeError, ValueError):
        return False


def _rms(value: Any) -> float | None:
    try:
        array = _array(value).astype(np.float64, copy=False)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        return float(np.sqrt(np.mean(array * array, dtype=np.float64)))
    except (TypeError, ValueError):
        return None


def _cosine(left: Any, right: Any) -> float | None:
    try:
        a = _array(left).astype(np.float64, copy=False).reshape(-1)
        b = _array(right).astype(np.float64, copy=False).reshape(-1)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if not a.size or a.size != b.size or denominator == 0.0 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return None
        return float(np.dot(a, b) / denominator)
    except (TypeError, ValueError):
        return None


def finite_or_na(value: Any, denominator: Any | None = None) -> float | None:
    """Return a finite scalar or explicit ``None`` for undefined arithmetic."""
    try:
        numerator = float(value)
        if denominator is not None:
            denominator = float(denominator)
            if denominator == 0.0 or not math.isfinite(denominator):
                return None
            numerator /= denominator
        return numerator if math.isfinite(numerator) else None
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None


def _alpha_index(value: Any) -> int | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return min(range(len(ALPHAS)), key=lambda index: abs(float(ALPHAS[index]) - value)) if any(math.isclose(value, float(alpha), rel_tol=0.0, abs_tol=1e-12) for alpha in ALPHAS) else None


def _window_key(row: Mapping[str, Any]) -> tuple[int, int] | None:
    """Return the full consecutive alpha window identity, not only its start."""
    start = _alpha_index(row.get("start_alpha")); end = _alpha_index(row.get("end_alpha"))
    if start is None or end is None or (start, end) not in WINDOW_INDEX_PAIRS:
        return None
    return (start, end)


def _window_index(row: Mapping[str, Any]) -> int:
    key = _window_key(row)
    return WINDOW_INDEX_PAIRS.index(key) if key in WINDOW_INDEX_PAIRS else -1


def _window_label(key: tuple[int, int]) -> str:
    return f"{ALPHAS[key[0]]:.0e}..{ALPHAS[key[1]]:.0e}"


def _heatmap_cell_style(value: Any, vmin: float, vmax: float) -> tuple[tuple[int, int, int], str]:
    """Map values to fixed-scale colors while keeping missing distinct."""
    missing = value is None
    try:
        numeric = float(value)
        missing = missing or not math.isfinite(numeric)
    except (TypeError, ValueError):
        missing = True
        numeric = 0.0
    if missing:
        return (238, 242, 245), "missing"
    if numeric < vmin:
        return (82, 43, 110), "under"
    if numeric > vmax:
        return (230, 120, 20), "over"
    norm = (numeric - vmin) / (vmax - vmin) if vmax > vmin else 0.0
    color = (round(245 - 150 * norm), round(220 - 70 * norm), round(245 - 20 * norm))
    return color, "min" if numeric == vmin else "max" if numeric == vmax else "inside"


def _relative(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    try:
        a = _rms(np.asarray(left) - np.asarray(right))
    except (TypeError, ValueError):
        return None
    b = _rms(right)
    return finite_or_na(a, b)


def _attach_null_reasons(row: dict[str, Any], extra: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Attach a causal reason to every undefined quantitative field in a row."""
    reasons: dict[str, str] = dict(extra or {})
    for field, value in row.items():
        if value is not None or field in {"reason", "exclusions", "points", "undefined_reasons", "null_reasons"} or field.endswith("_reason"):
            continue
        reasons.setdefault(field, {
            "relative_rms": "reference vector RMS is zero, nonfinite, or missing",
            "floor_multiple": "float baseline floor is zero, nonfinite, or missing",
            "paired_secant": "paired plus/minus step distance is zero, nonfinite, or missing",
            "paired_secant_rms": "paired secant tensor is undefined",
            "intercept": "log-log fit has fewer than three valid points or constant input scale",
            "slope": "log-log fit has fewer than three valid points or constant input scale",
            "r2": "log-log response values are constant or insufficient",
        }.get(field, "required source tensor or denominator is zero, nonfinite, or missing"))
    if reasons:
        row["null_reasons"] = reasons
        joined = "; ".join(f"{field} undefined: {reason}" for field, reason in sorted(reasons.items()))
        row["undefined_reasons"] = "; ".join(filter(None, [str(row.get("undefined_reasons") or ""), joined]))
    return row


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe_json(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(_safe_json(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _atomic_npy(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        with open(temp_name, "wb") as stream:
            np.save(stream, np.asarray(value), allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    keys: list[str] = []
    for row in values:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    lines: list[str] = []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent), prefix=f".{path.name}.") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["status"], extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({key: "" if row.get(key) is None else _safe_json(row.get(key)) for key in (keys or ["status"])})
        stream.flush(); os.fsync(stream.fileno()); temp_name = stream.name
    os.replace(temp_name, path)


def _space_value(record: Mapping[str, Any], space: str) -> np.ndarray | None:
    name = "predicted_latent" if space == "predicted_latent" else "decoded_final"
    value = record.get(name)
    try:
        array = _array(value).astype(np.float32, copy=False)
        return array if array.size and np.all(np.isfinite(array)) else None
    except (TypeError, ValueError):
        return None


def _validated_delta(record: Mapping[str, Any], name: str = "actual_delta_fp32") -> tuple[np.ndarray | None, str | None]:
    value = record.get(name)
    if value is None:
        value = record.get("realized_delta_fp32") if name == "actual_delta_fp32" else None
    try:
        array = _array(value).astype(np.float32, copy=False)
        mask = _array(record.get("condition_mask", np.ones_like(array, dtype=bool))).astype(bool)
        if array.shape != mask.shape or not np.all(np.isfinite(array)):
            return None, f"{name} is missing, non-finite, or shape-mismatched"
        outside = array[~mask]
        zero = np.zeros_like(outside)
        if outside.tobytes(order="C") != zero.tobytes(order="C"):
            return None, f"{name} has nonzero outside-mask values"
        if name in {"actual_delta_fp32", "actual_delta_bf16"}:
            baseline = record.get("carrier_fp32")
            realized = record.get("realized_carrier_fp32")
            if baseline is not None and realized is not None and name == "actual_delta_fp32":
                expected = (_array(realized).astype(np.float32, copy=False) - _array(baseline).astype(np.float32, copy=False)).astype(np.float32)
                if expected.shape != array.shape or expected.tobytes(order="C") != array.tobytes(order="C"):
                    return None, "actual_delta_fp32 is not bytewise realized_carrier-baseline_carrier"
            elif baseline is not None and realized is not None and name == "actual_delta_bf16":
                expected = (_bf16_round(_array(realized).astype(np.float32, copy=False)) - _bf16_round(_array(baseline).astype(np.float32, copy=False))).astype(np.float32)
                if expected.shape != array.shape or expected.tobytes(order="C") != array.tobytes(order="C"):
                    return None, "actual_delta_bf16 is not bytewise BF16(realized_carrier)-BF16(baseline_carrier)"
            elif record.get("provenance"):
                return None, "official sample is missing saved carrier pair for realized delta binding"
        return array.copy(), None
    except (TypeError, ValueError):
        return None, f"{name} is not a numeric tensor"


def _input_delta(record: Mapping[str, Any], name: str = "actual_delta_fp32") -> np.ndarray | None:
    value, _ = _validated_delta(record, name)
    return value


def _masked_rms(value: Any, record: Mapping[str, Any]) -> float | None:
    try:
        array = _array(value)
        mask = _array(record["condition_mask"]).astype(bool)
        if array.shape != mask.shape:
            return None
        return _rms(array[mask])
    except (KeyError, TypeError, ValueError):
        return None


def _validate_target_evidence(record: Mapping[str, Any]) -> str | None:
    """Validate the requested geometry and all scalar API evidence."""
    spec = record.get("spec") if isinstance(record.get("spec"), Mapping) else {}
    try:
        alpha = float(record.get("target_alpha"))
        epsilon = float(record.get("target_epsilon"))
        target_rms = float(record.get("target_rms"))
        s_z = float(record.get("s_z"))
    except (TypeError, ValueError):
        return "target_alpha/target_epsilon/target_rms/s_z are required finite scalars"
    if not all(math.isfinite(value) for value in (alpha, epsilon, target_rms, s_z)):
        return "target_alpha/target_epsilon/target_rms/s_z contain NaN or infinity"
    if spec.get("kind") == "perturbation" and not math.isclose(alpha, float(spec.get("alpha")), rel_tol=0.0, abs_tol=0.0):
        return "target_alpha does not equal the call-plan alpha"
    expected = alpha * s_z
    if not math.isclose(epsilon, expected, rel_tol=1e-6, abs_tol=1e-12) or not math.isclose(target_rms, expected, rel_tol=1e-6, abs_tol=1e-12):
        return "target_epsilon/target_rms are inconsistent with alpha*s_z"
    target = record.get("target_delta_fp32")
    mask = record.get("condition_mask")
    if target is None or mask is None or _masked_rms(target, record) is None:
        return "target delta evidence is missing or non-finite"
    if spec.get("kind") == "perturbation" and record.get("z0") is not None and record.get("direction_bank") is not None:
        try:
            expected_delta = construct_delta(record["z0"], mask, _array(record["direction_bank"])[int(spec["direction_index"])], alpha=alpha, sign=int(spec["sign"])).delta
            target_array = _array(target).astype(np.float32, copy=False)
            if target_array.shape != expected_delta.shape or target_array.tobytes(order="C") != expected_delta.astype(np.float32).tobytes(order="C"):
                return "target_delta_fp32 does not match reconstructed z0/mask/direction geometry"
        except (TypeError, ValueError, KeyError, IndexError):
            return "root direction-bank target geometry cannot be reconstructed"
    if not math.isclose(float(_masked_rms(target, record)), target_rms, rel_tol=1e-6, abs_tol=1e-12):
        return "target_delta_fp32 RMS does not equal target_rms"
    return None


def _validate_noise_identity(record: Mapping[str, Any]) -> str | None:
    state = record.get("initial_state")
    mask = record.get("initial_condition_mask", record.get("condition_mask"))
    saved = record.get("initial_noise_hash", record.get("predicted_noise_hash"))
    if state is None or mask is None or not isinstance(saved, str) or not _SHA256_RE.fullmatch(saved):
        return "initial_noise_hash is missing or not a valid sha256 identity"
    try:
        expected = hash_predicted_noisy_region(state, mask)
    except (TypeError, ValueError):
        return "initial_state/initial_condition_mask cannot be hashed"
    return None if saved == expected else "initial_noise_hash does not match saved initial_state and initial_condition_mask"


def _fit_loglog(x: list[float], y: list[float], points: list[str], exclusions: list[str]) -> dict[str, Any]:
    valid: list[tuple[float, float, str]] = []
    for x_value, y_value, point in zip(x, y, points):
        if x_value is None or y_value is None or x_value <= 0 or y_value <= 0 or not math.isfinite(x_value) or not math.isfinite(y_value):
            exclusions.append(f"{point}: nonpositive/nonfinite log point")
        else:
            valid.append((math.log(x_value), math.log(y_value), point))
    if len(valid) < MIN_WINDOW:
        reason = "fewer than three finite positive points"
        return {"slope": None, "intercept": None, "r2": None, "slope_reason": reason, "intercept_reason": reason, "r2_reason": reason, "points": [item[2] for item in valid], "exclusions": exclusions, "reason": reason}
    lx = np.asarray([item[0] for item in valid], dtype=np.float64)
    ly = np.asarray([item[1] for item in valid], dtype=np.float64)
    if float(np.sum((lx - np.mean(lx)) ** 2, dtype=np.float64)) == 0.0:
        reason = "constant log-x values make slope/intercept undefined"
        return {"slope": None, "intercept": None, "r2": None, "slope_reason": reason, "intercept_reason": reason, "r2_reason": "constant log-x values make R² undefined", "points": [item[2] for item in valid], "exclusions": exclusions, "reason": reason}
    slope, intercept = np.polyfit(lx, ly, 1)
    fitted = slope * lx + intercept
    residual = float(np.sum((ly - fitted) ** 2, dtype=np.float64))
    total = float(np.sum((ly - np.mean(ly)) ** 2, dtype=np.float64))
    r2 = None if total == 0.0 else float(1.0 - residual / total)
    r2_reason = "constant log-y values make R² undefined" if r2 is None else None
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "slope_reason": None, "intercept_reason": None, "r2_reason": r2_reason, "points": [item[2] for item in valid], "exclusions": exclusions, "reason": r2_reason}


def _record_point(record: Mapping[str, Any], sample_id: str, space: str, pre: np.ndarray, floor: float, direction: int, alpha: float, sign: int) -> dict[str, Any]:
    output = _space_value(record, space)
    response = None if output is None else (output.astype(np.float32) - pre.astype(np.float32)).astype(np.float32)
    delta = _input_delta(record)
    delta_bf16 = _input_delta(record, "actual_delta_bf16")
    target = _input_delta(record, "target_delta_fp32")
    _, bf16_reason = _validated_delta(record, "actual_delta_bf16")
    actual_rms = _masked_rms(delta, record)
    target_rms = _masked_rms(target, record)
    bf16_rms = _masked_rms(delta_bf16, record)
    output_rms = _rms(response)
    outside_exact = None
    input_mask = None
    try:
        mask = _array(record["condition_mask"]).astype(bool)
        input_mask = mask
        outside_exact = bool(delta is not None and np.array_equal(delta[~mask].tobytes(order="C"), np.zeros_like(delta[~mask]).tobytes(order="C")))
    except (KeyError, TypeError, ValueError):
        pass
    try:
        mask_values = _array(record["condition_mask"]).astype(bool)
        nonzero_ratio = None if delta is None else float(np.count_nonzero(delta[mask_values]) / delta[mask_values].size)
        bf16_nonzero_ratio = None if delta_bf16 is None else float(np.count_nonzero(delta_bf16[mask_values]) / delta_bf16[mask_values].size)
        direction_cosine = _cosine(delta[mask_values], target[mask_values]) if delta is not None and target is not None else None
        bf16_direction_cosine = _cosine(delta_bf16[mask_values], target[mask_values]) if delta_bf16 is not None and target is not None else None
    except (KeyError, TypeError, ValueError):
        nonzero_ratio = bf16_nonzero_ratio = direction_cosine = bf16_direction_cosine = None
    undefined_reasons = []
    for name, value in (("actual_input_rms", actual_rms), ("target_delta_rms", target_rms), ("bf16_input_rms", bf16_rms), ("output_rms", output_rms)):
        if value is None:
            undefined_reasons.append(f"{name} undefined")
    if floor == 0.0:
        undefined_reasons.append("floor_multiple undefined: zero baseline floor")
    if outside_exact is None:
        undefined_reasons.append("outside_mask_exact undefined: condition mask or input delta is missing")
    metrics_with_reasons = {
        "gain": finite_or_na(output_rms, actual_rms),
        "actual_target_ratio": finite_or_na(actual_rms, target_rms),
        "bf16_survival_ratio": finite_or_na(bf16_rms, actual_rms),
        "direction_cosine": direction_cosine,
        "bf16_direction_cosine": bf16_direction_cosine,
        "target_epsilon_derivative": finite_or_na(output_rms, target_rms),
        "actual_step_derivative": finite_or_na(output_rms, actual_rms),
    }
    for metric, value in metrics_with_reasons.items():
        if value is None:
            denominator = "target_delta_rms" if metric in {"actual_target_ratio", "target_epsilon_derivative"} else "actual_input_rms" if metric in {"gain", "bf16_survival_ratio", "actual_step_derivative"} else "vector norm"
            undefined_reasons.append(f"{metric} undefined: {denominator} is zero, nonfinite, or missing")
    return {
        "sample_id": sample_id, "space": space, "direction": direction, "alpha": alpha, "sign": sign,
        "target_alpha": record.get("target_alpha", alpha), "target_delta_rms": target_rms, "actual_input_rms": actual_rms,
        "bf16_input_rms": bf16_rms, "actual_target_ratio": metrics_with_reasons["actual_target_ratio"], "bf16_survival_ratio": metrics_with_reasons["bf16_survival_ratio"],
        "bf16_valid": bf16_reason is None and bf16_rms is not None, "bf16_reason": bf16_reason,
        "nonzero_ratio": nonzero_ratio, "bf16_nonzero_ratio": bf16_nonzero_ratio, "direction_cosine": direction_cosine, "bf16_direction_cosine": bf16_direction_cosine,
        "undefined_reasons": "; ".join(undefined_reasons) if undefined_reasons else None,
        "output_rms": output_rms, "gain": metrics_with_reasons["gain"], "floor": floor, "floor_multiple": finite_or_na(output_rms, floor),
        "outside_mask_exact": outside_exact, "response": response, "input": delta, "input_bf16": delta_bf16, "target_input": target,
        "input_mask": input_mask,
        "target_epsilon_derivative": metrics_with_reasons["target_epsilon_derivative"], "actual_step_derivative": metrics_with_reasons["actual_step_derivative"],
        "target_derivative": None if response is None or target_rms in (None, 0.0) else (response.astype(np.float64) / float(target_rms)).astype(np.float32),
        "actual_derivative": None if response is None or actual_rms in (None, 0.0) else (response.astype(np.float64) / float(actual_rms)).astype(np.float32),
    }


def _diagnostic(reason: str, output_dir: Path | None = None, *, status: str = "DIAGNOSTIC_ONLY") -> dict[str, Any]:
    result = {"status": status, "reason": reason, "scientific_scope": "four-direction interface/scale evidence only; no low-rank conclusion"}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".analysis-stage-", dir=str(output_dir.parent)))
        try:
            _atomic_json(stage / "scan_summary.json", result)
            _atomic_text(stage / "implementation_notes.md", "# Implementation notes\n\nNo quantitative analysis was published.\n\n## Manifest/bundle contract\n\nThe root manifest excludes `MANIFEST.sha256`, `MANIFEST.analysis.sha256`, and `review_bundle.zip`; generated diagnostic plots are included. Unowned raw inputs, including arbitrary PNG/SVG previews, are preserved bytewise and excluded from the lightweight bundle.\n\n## Chart map\n\nNo charts: evidence is absent or invalid.\n")
            _atomic_text(stage / "experiment_report.md", f"# UMI post-VAE scan report\n\n## Technical summary\n\nDiagnostic-only result: {reason}\n\n## Key findings with figures\n\nNo quantitative findings were published. See [diagnostic evidence](diagnostic_evidence.svg).\n\n## Scope/data/metric definitions\n\nThe fixed four-direction scan spaces are predicted latent and final float RGB; no tensors were accepted for quantitative analysis.\n\n## Experiment/validation details\n\nRoot/sample provenance, manifest, call-plan, and hardened evidence validation did not establish a complete scientific input.\n\n## Limitations/robustness\n\nThis is four-direction interface/scale evidence only and cannot support a low-rank conclusion.\n\n## Recommended next step\n\nRepair the reported provenance/evidence defect and rerun the offline analyzer.\n\n## Further questions\n\nWhich required sample, hash, or evidence boundary should be repaired first?\n")
            _diagnostic_plot(stage, reason)
            _write_analysis_manifest(stage, context_root=output_dir if output_dir.is_dir() else None)
            _make_bundle(stage, {}, context_root=output_dir if output_dir.is_dir() else None)
            _publish_stage(stage, output_dir)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return result


def _diagnostic_plot(output_dir: Path, reason: str) -> None:
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360'><rect width='100%' height='100%' fill='#f5f7fa'/><line x1='50' y1='310' x2='590' y2='310' stroke='#455a64'/><line x1='50' y1='50' x2='50' y2='310' stroke='#455a64'/><circle cx='320' cy='180' r='10' fill='#c62828'/><text x='20' y='32' fill='#263238'>Diagnostic-only: numeric evidence absent or invalid</text><text x='20' y='345' fill='#263238'>{reason[:180]}</text></svg>\n"
    _atomic_text(output_dir / "diagnostic_evidence.svg", svg)
    width, height = 640, 360
    pixels = bytearray(bytes((245, 247, 250)) * width * height)
    for x in range(45, 595):
        for y in (310, 309, 50, 51):
            pixels[(y * width + x) * 3:(y * width + x + 1) * 3] = bytes((69, 90, 100))
    for y in range(50, 311):
        for x in (50, 51):
            pixels[(y * width + x) * 3:(y * width + x + 1) * 3] = bytes((69, 90, 100))
    for x in range(312, 329):
        for y in range(172, 189):
            pixels[(y * width + x) * 3:(y * width + x + 1) * 3] = bytes((198, 40, 40))
    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        import struct, zlib
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    (output_dir / "diagnostic_evidence.png").write_bytes(png)


def select_candidate_windows(data: Any, *, min_length: int = MIN_WINDOW) -> list[dict[str, Any]]:
    """Evaluate every consecutive alpha window and return decisions."""
    if isinstance(data, Mapping) and "points" in data:
        points = data["points"]
    elif isinstance(data, Mapping) and "point_metrics" in data:
        points = data["point_metrics"]
    else:
        points = data
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in points or []:
        if isinstance(row, Mapping) and row.get("space") in SPACES and row.get("sign") in SIGNS:
            grouped.setdefault((str(row["space"]), int(row["direction"])), []).append(row)
    decisions: list[dict[str, Any]] = []
    for space in SPACES:
        for direction in range(DIRECTION_COUNT):
            plus = sorted(grouped.get((space, direction), []), key=lambda row: float(row.get("alpha", 0.0)))
            plus = [row for row in plus if int(row.get("sign", 0)) == 1]
            minus = {float(row.get("alpha")): row for row in grouped.get((space, direction), []) if int(row.get("sign", 0)) == -1}
            for start in range(max(0, len(plus) - min_length + 1)):
                for end in range(start + min_length, len(plus) + 1):
                    window = plus[start:end]
                    reasons: list[str] = []
                    indexes = [float(row["alpha"]) for row in window]
                    paired: list[dict[str, Any]] = []
                    for row in window:
                        other = minus.get(float(row["alpha"]))
                        if other is None:
                            reasons.append("missing plus/minus evidence"); continue
                        plus_input, minus_input = row.get("input"), other.get("input")
                        pair = {"alpha": row["alpha"], "input_cosine": _cosine(plus_input, -np.asarray(minus_input)) if plus_input is not None and minus_input is not None else None, "secant": None, "response_consistency": _cosine(row.get("response"), -np.asarray(other.get("response"))) if row.get("response") is not None and other.get("response") is not None else None, "even_residual": _rms((np.asarray(row.get("response")) + np.asarray(other.get("response"))) / 2.0) if row.get("response") is not None and other.get("response") is not None else None}
                        if pair["input_cosine"] is None or pair["input_cosine"] < 0.99:
                            reasons.append("plus/minus input opposition below 0.99")
                        if row.get("outside_mask_exact") is not True or other.get("outside_mask_exact") is not True:
                            reasons.append("outside-mask exactness missing")
                        if row.get("bf16_valid") is not True or other.get("bf16_valid") is not True or row.get("bf16_input_rms") in (None, 0.0) or other.get("bf16_input_rms") in (None, 0.0):
                            reasons.append("BF16 delta missing, nonfinite, or erased")
                        if row.get("response_rms", row.get("output_rms")) is None or other.get("response_rms", other.get("output_rms")) is None:
                            reasons.append("missing response")
                        floor = float(row.get("floor") or 0.0)
                        for value in (row.get("output_rms"), other.get("output_rms")):
                            if value is None or (floor == 0.0 and value <= 0.0) or (floor > 0.0 and value <= 10.0 * floor):
                                reasons.append("response does not clear float baseline floor")
                        if plus_input is not None and minus_input is not None:
                            difference = np.asarray(plus_input) - np.asarray(minus_input)
                            mask_for_distance = row.get("input_mask")
                            distance = _rms(difference[np.asarray(mask_for_distance, dtype=bool)]) if mask_for_distance is not None and np.asarray(mask_for_distance).shape == difference.shape else _rms(difference)
                        else:
                            distance = None
                        if distance and row.get("response") is not None and other.get("response") is not None:
                            pair["secant"] = (np.asarray(row["response"], dtype=np.float32) - np.asarray(other["response"], dtype=np.float32)) / np.float32(distance)
                        paired.append(pair)
                    for left, right in zip(window, window[1:]):
                        cosine = _cosine(left.get("input"), right.get("input"))
                        if cosine is None or cosine < 0.99:
                            reasons.append("adjacent realized input cosine below 0.99")
                    minus_window = [minus.get(float(row["alpha"])) for row in window]
                    for left, right in zip(minus_window, minus_window[1:]):
                        if left is None or right is None:
                            continue
                        cosine = _cosine(left.get("input"), right.get("input"))
                        if cosine is None or cosine < 0.99:
                            reasons.append("adjacent minus realized input cosine below 0.99")
                    slopes = data.get("fits", []) if isinstance(data, Mapping) else []
                    for sign in SIGNS:
                        fit = next((item for item in slopes if item.get("space") == space and int(item.get("direction", -1)) == direction and int(item.get("sign", 0)) == sign and item.get("start_alpha") == indexes[0] and item.get("end_alpha") == indexes[-1]), None)
                        if fit is None or fit.get("slope") is None or not (0.8 <= float(fit["slope"]) <= 1.2) or fit.get("r2") is None or float(fit["r2"]) < 0.98:
                            reasons.append(f"sign {sign} log-log fit fails slope/R2")
                    for left, right in zip(paired, paired[1:]):
                        if left.get("secant") is None or right.get("secant") is None:
                            reasons.append("paired secant undefined"); continue
                        cosine = _cosine(left["secant"], right["secant"])
                        relative = _relative(left["secant"], right["secant"])
                        if cosine is None or cosine < 0.95 or relative is None or relative > 0.25:
                            reasons.append("paired secant consistency fails")
                    decisions.append({"space": space, "direction": direction, "start_alpha": indexes[0], "end_alpha": indexes[-1], "alphas": indexes, "selected": not reasons, "reasons": sorted(set(reasons))})
    return decisions


_BITMAP_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"), "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"), "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"), "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"), "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"), "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"), "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"), "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"), "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"), "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"), "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"), "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"), "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"), "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"), "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"), "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"), "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"), "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"), "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    " ": ("00000",) * 7, ".": ("00000", "00000", "00000", "00000", "00000", "00110", "00110"), ":": ("00000", "00110", "00110", "00000", "00110", "00110", "00000"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"), "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"), "=": ("00000", "11111", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"), "(": ("00010", "00100", "01000", "01000", "01000", "00100", "00010"), ")": ("01000", "00100", "00010", "00010", "00010", "00100", "01000"),
    "[": ("01110", "01000", "01000", "01000", "01000", "01000", "01110"), "]": ("01110", "00010", "00010", "00010", "00010", "00010", "01110"), "<": ("00010", "00100", "01000", "10000", "01000", "00100", "00010"), ">": ("01000", "00100", "00010", "00001", "00010", "00100", "01000"), "|": ("00100",) * 7,
}


def _fallback_text(model: list[dict[str, Any]], text: str, x: int, y: int, *, size: int = 2, color: tuple[int, int, int] = (38, 50, 56), role: str | None = None) -> None:
    item = {"type": "text", "x": x, "y": y, "text": str(text), "size": max(1, int(size)), "color": color}
    if role is not None:
        item["role"] = role
    model.append(item)


def _render_fallback(name: str, rows: list[dict[str, Any]], finite_difference: list[dict[str, Any]], fits: list[dict[str, Any]], decisions: list[dict[str, Any]] | None = None) -> tuple[bytes, str]:
    """Build one explicit mark/transform model, then render it to PNG and SVG."""
    # Keep the plotting rectangle unchanged while reserving a larger footer
    # for complete legends, keys, and captions in both bitmap and SVG output.
    width, height, left, top, right, bottom = 900, 660, 145, 75, 25, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    background = (248, 250, 252); dark = (55, 65, 75)
    model: list[dict[str, Any]] = []
    titles = {"response_magnitude": "Response magnitude by space / direction / sign", "slope_evidence": "Four-direction slope evidence (dot / range)", "finite_difference_cosine": "Adjacent-alpha direction cosine heatmap", "finite_difference_relative_rms": "Adjacent-alpha direction relative-RMS heatmap", "paired_secant_cosine": "Adjacent-alpha paired-secant cosine heatmap", "paired_secant_relative_rms": "Adjacent-alpha paired-secant relative-RMS heatmap", "diagnostics": "Diagnostics: actual / target, BF16 survival, outside-mask exactness, response / float-floor"}
    colors = [(21, 101, 192), (0, 121, 107), (123, 31, 162), (230, 81, 0), (66, 133, 244), (0, 150, 136), (142, 36, 170), (239, 108, 0)]
    def add_text(text: str, x: int, y: int, size: int = 1, *, role: str | None = None) -> None: _fallback_text(model, text, x, y, size=size, role=role)
    def add_line(x1: float, y1: float, x2: float, y2: float, color: tuple[int, int, int] = dark, **attrs: Any) -> None: model.append({"type": "line", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "color": color, **attrs})
    def add_rect(x: float, y: float, w: float, h: float, color: tuple[int, int, int], **attrs: Any) -> None: model.append({"type": "rect", "x": x, "y": y, "w": w, "h": h, "color": color, **attrs})
    def add_mark(x: float, y: float, color: tuple[int, int, int], shape: str, label: str, **attrs: Any) -> None: model.append({"type": "mark", "x": x, "y": y, "color": color, "shape": shape, "label": label, **attrs})
    def scale(value: float, low: float, high: float, start: float, end: float) -> float: return start if high <= low else start + (end - start) * (value - low) / (high - low)
    def series_color(row: Mapping[str, Any]) -> tuple[int, int, int]: return colors[(SPACES.index(str(row.get("space"))) * DIRECTION_COUNT + int(row.get("direction", 0))) % len(colors)]
    decisions = decisions or []
    add_text(titles[name], left, 14, 1 if len(titles[name]) > 60 else 2); add_text("CPU evidence only | missing values are marked N/A", left, 45)
    if name == "response_magnitude":
        valid = [(r, math.log10(float(r["actual_input_rms"])), math.log10(float(r["output_rms"]))) for r in rows if r.get("actual_input_rms") and r.get("output_rms") and float(r["actual_input_rms"]) > 0 and float(r["output_rms"]) > 0]
        xs = [v[1] for v in valid] or [-4, 0]; ys = [v[2] for v in valid] or [-4, 0]; low, high = min(xs + ys) - .2, max(xs + ys) + .2
        add_line(left, top + plot_h, width - right, top + plot_h); add_line(left, top, left, top + plot_h)
        add_line(left, top + plot_h, width - right, top, (101, 114, 126), dash="7 5", label="slope-1 neutral reference")
        for value, label in ((low, f"{low:.1f}"), ((low + high) / 2, f"{(low + high) / 2:.1f}"), (high, f"{high:.1f}")):
            xx = round(scale(value, low, high, left, width - right)); yy = round(scale(value, low, high, top + plot_h, top)); add_line(xx, top + plot_h, xx, top + plot_h + 5); add_line(left - 5, yy, left, yy); add_text(label, xx - 16, top + plot_h + 10); add_text(label, left - 55, yy - 5)
        add_text("Actual input RMS (log10; float32 units)", 300, height - 104); add_text("Output RMS (log10; space units)", 300, 60); add_text("Tick labels: log10 x and y", left + 8, top + 8); add_text("slope-1 neutral reference", width - 250, top + 8); add_text("Series map: color = space/direction", left, height - 66); add_text("Sign key: circle = plus marker; X = minus marker", left, height - 52)
        for direction in range(DIRECTION_COUNT):
            xx = left + direction * 180
            for space_index, space in enumerate(SPACES):
                yy = height - 35 + space_index * 22
                add_mark(xx, yy, colors[(space_index * DIRECTION_COUNT + direction) % len(colors)], "circle", f"{space} d{direction}", legend=True)
                add_text(f"{space} d{direction}", xx + 8, yy - 5, 1)
        for row, x, y in sorted(valid, key=lambda item: -int(item[0]["sign"])): add_mark(scale(x, low, high, left, width - right), scale(y, low, high, top + plot_h, top), series_color(row), "circle" if int(row["sign"]) == 1 else "x", f"{row['space']} d{row['direction']} {'+' if int(row['sign']) == 1 else '-'}")
    elif name == "slope_evidence":
        valid = [r for r in fits if r.get("slope") is not None]; values = [float(r["slope"]) for r in valid] or [.8, 1.2]; low, high = min(0.0, min(values) - .1), max(1.2, max(values) + .1)
        panel_gap, panel_w = 24, (plot_w - 24) / 2.0; data_top, data_bottom = top + 28, top + plot_h - 42
        # Reserve a dedicated caption band between the plot and the two-column
        # window key.  Keeping this line above the key prevents it from
        # crossing the fourth/fifth key rows at the original output size.
        add_text("Slope (log10 fit); two panels use compact consecutive-window indices", left, 60); add_text("Acceptance band: 0.8 to 1.2; R2 ge 0.98; boxed = selected candidate window", left, 520, role="slope-acceptance"); add_text("Sign key: circle = plus marker; X = minus marker", left, 506, role="slope-sign-key")
        for space_index, space in enumerate(SPACES):
            x0 = left + space_index * (panel_w + panel_gap); add_rect(x0, data_top, panel_w, data_bottom - data_top, (255, 255, 255), panel=space, label=f"{space} panel")
            add_text(space, x0 + 6, data_top - 20); add_line(x0 + 18, data_bottom, x0 + panel_w - 8, data_bottom); add_line(x0 + 18, data_top, x0 + 18, data_bottom)
            band_y = scale(1.2, low, high, data_bottom, data_top); band_h = scale(.8, low, high, data_bottom, data_top) - band_y; add_rect(x0 + 18, band_y, panel_w - 26, band_h, (215, 232, 245), label="acceptance band 0.8 to 1.2")
            for tick in (0.0, .8, 1.0, 1.2, high):
                yy = round(scale(tick, low, high, data_bottom, data_top)); add_line(x0 + 12, yy, x0 + 18, yy); add_text(f"{tick:.1f}", x0 - 28, yy - 5, 1)
            max_window = len(WINDOW_INDEX_PAIRS) - 1
            for window_index in range(max_window + 1):
                xx = scale(window_index, 0, max(1, max_window), x0 + 24, x0 + panel_w - 12); add_line(xx, data_bottom, xx, data_bottom + 5); add_text(str(window_index), xx - 3, data_bottom + 9, 1, role="slope-axis-tick")
            add_text("window index", x0 + panel_w / 2 - 30, data_bottom + 23, 1, role="slope-axis-label")
            space_rows = [r for r in valid if r.get("space") == space]
            for row in space_rows:
                selected = any(d.get("space") == row.get("space") and int(d.get("direction", -1)) == int(row.get("direction", -2)) and d.get("start_alpha") == row.get("start_alpha") and d.get("end_alpha") == row.get("end_alpha") and d.get("selected") for d in decisions)
                window_index = _window_index(row); jitter = (int(row.get("direction", 0)) * 2 + (0 if int(row.get("sign", 0)) == 1 else 1)) * 3.0
                xx = scale(window_index, 0, max(1, max_window), x0 + 24, x0 + panel_w - 12) + jitter - 10; yy = scale(float(row["slope"]), low, high, data_bottom, data_top)
                add_mark(xx, yy, series_color(row), "circle" if int(row["sign"]) == 1 else "x", f"{space} d{row['direction']} {'+' if int(row['sign']) == 1 else '-'} window {window_index}", panel=space, selected=selected, window_id=window_index, window_key=_window_label(_window_key(row)) if _window_key(row) is not None else "unknown")
        add_text("Window key (index: alpha range)", left, 548, 1, role="slope-window-key")
        for index, key in enumerate(WINDOW_INDEX_PAIRS):
            add_text(f"{index}:{_window_label(key)}", left + (index % 2) * 320, 562 + (index // 2) * 12, 1)
        add_text("Direction legend:", width - 205, 548, 1)
        for direction in range(DIRECTION_COUNT):
            add_mark(width - 205 + direction * 42, 568, colors[direction], "circle", f"d{direction}", legend=True, direction_legend=True)
            add_text(f"d{direction}", width - 195 + direction * 42, 563, 1)
    elif name in {"finite_difference_cosine", "finite_difference_relative_rms", "paired_secant_cosine", "paired_secant_relative_rms"}:
        metric = {"finite_difference_cosine": "input_cosine", "finite_difference_relative_rms": "relative_rms", "paired_secant_cosine": "paired_secant_cosine", "paired_secant_relative_rms": "paired_secant_relative_rms"}[name]
        row_keys = [(space, sign, direction) for space in SPACES for sign in SIGNS for direction in range(DIRECTION_COUNT)]
        if metric in {"input_cosine", "paired_secant_cosine"}:
            vmin, vmax = 0.95, 1.0
        else:
            vmin, vmax = 0.0, 0.25
        colorbar_x = width - right - 75
        heatmap_right = colorbar_x - 24
        cell_w, cell_h = (heatmap_right - left) / max(1, len(ALPHAS) - 1), 21
        add_rect(left, top, heatmap_right - left, len(row_keys) * cell_h, background, heatmap_layout="plot", data_right=heatmap_right, colorbar_x=colorbar_x, gap=colorbar_x - heatmap_right)
        for row_index, (space, sign, direction) in enumerate(row_keys):
            yy = top + row_index * cell_h; add_text(f"{space} d{direction} {'+' if sign == 1 else '-'}", 2, round(yy + 4))
            for adjacent in range(len(ALPHAS) - 1):
                item = next((q for q in finite_difference if q.get("space") == space and int(q.get("sign", 0)) == sign and int(q.get("direction", -1)) == direction and int(q.get("adjacent_index", -1)) == adjacent), None); value = None if item is None else item.get(metric); color, value_class = _heatmap_cell_style(value, vmin, vmax); xx, ww, hh = round(left + adjacent * cell_w), max(1, round(cell_w) - 2), max(1, round(cell_h) - 2); add_rect(xx, round(yy), ww, hh, color, cell=f"{space}|sign={sign}|direction={direction}|adjacent={adjacent}", value=value, value_class=value_class); add_line(xx, round(yy), xx + ww, round(yy), (170, 180, 190))
        colorbar_y, colorbar_h, colorbar_w = top, min(plot_h, len(row_keys) * cell_h), 12
        for step in range(24):
            norm = step / 23.0; color = (round(245 - 150 * norm), round(220 - 70 * norm), round(245 - 20 * norm)); add_rect(colorbar_x, colorbar_y + (23 - step) * colorbar_h / 24.0, colorbar_w, colorbar_h / 24.0 + 1, color, colorbar=name, colorbar_position="gradient")
        add_rect(colorbar_x, colorbar_y - 8, colorbar_w, 6, (230, 120, 20), colorbar=name, colorbar_position="over", value=vmax); add_rect(colorbar_x, colorbar_y + colorbar_h + 2, colorbar_w, 6, (82, 43, 110), colorbar=name, colorbar_position="under", value=vmin)
        for value in (vmin, (vmin + vmax) / 2, vmax):
            yy = scale(value, vmin, vmax, colorbar_y + colorbar_h, colorbar_y); add_line(colorbar_x + colorbar_w, yy, colorbar_x + colorbar_w + 5, yy); add_text(f"{value:.3f}" if vmax - vmin < .1 else f"{value:.2f}", colorbar_x + colorbar_w + 8, yy - 5, 1)
        add_text("over", colorbar_x - 2, colorbar_y - 22, 1); add_text("under", colorbar_x - 2, colorbar_y + colorbar_h + 12, 1); add_text("colorbar tick", colorbar_x - 12, colorbar_y + colorbar_h + 28, 1)
        for adjacent in range(len(ALPHAS) - 1): add_text(f"{ALPHAS[adjacent]:.0e} to {ALPHAS[adjacent + 1]:.0e}", round(left + adjacent * cell_w), round(top + len(row_keys) * cell_h + 8))
        add_text(f"Numeric fixed scale / color scale: min={vmin:.2f} max={vmax:.2f}; ticks {vmin:.2f}, {(vmin + vmax) / 2:.3f}, {vmax:.2f}", left, height - 24)
    else:
        panels = (("actual_target_ratio", "Actual / target", 0.0, 2.0), ("bf16_survival_ratio", "BF16 survival", 0.0, 1.1), ("outside_mask_exact", "Outside-mask exactness", 0.0, 1.0), ("floor_multiple", "Response / float-floor ratio", 0.0, 20.0)); panel_w, panel_h = plot_w // 2 - 22, plot_h // 2 - 24
        for index, (metric, label, fixed_low, fixed_high) in enumerate(panels):
            x0, y0 = left + (index % 2) * (plot_w // 2), top + (index // 2) * (plot_h // 2); add_rect(x0, y0, panel_w, panel_h, (255, 255, 255), panel=label); add_text(label, x0 + 4, y0 + 4); add_line(x0 + 4, y0 + panel_h - 8, x0 + panel_w, y0 + panel_h - 8); add_line(x0 + 4, y0 + 22, x0 + 4, y0 + panel_h - 8)
            for tick in (fixed_low, (fixed_low + fixed_high) / 2.0, fixed_high):
                yy = round(scale(tick, fixed_low, fixed_high, y0 + panel_h - 12, y0 + 28)); add_line(x0 - 2, yy, x0 + 4, yy); add_text(f"{tick:.2g}", x0 - 34, yy - 5, 1)
            values = [(i, float(r[metric])) for i, r in enumerate(rows) if r.get(metric) is not None and isinstance(r.get(metric), (int, float)) and math.isfinite(float(r[metric]))]
            for i, value in values: add_mark(scale(i, 0, max(1, len(rows) - 1), x0 + 8, x0 + panel_w - 5), scale(max(fixed_low, min(fixed_high, value)), fixed_low, fixed_high, y0 + panel_h - 12, y0 + 28), (21, 101, 192), "overflow" if value < fixed_low or value > fixed_high else "circle", f"{metric}[{i}] value={value:g}", overflow=value < fixed_low or value > fixed_high)
            add_text(f"fixed scale [{fixed_low:g}, {fixed_high:g}] | samples={len(values)}", x0 + 6, y0 + panel_h - 21, 1)
            if metric == "bf16_survival_ratio": add_text("overflow marker for values >1.1", x0 + 6, y0 + panel_h - 34, 1)
        add_text("Four independent diagnostic panels; fixed scales; identity is sample index; overflow markers are explicit", left, height - 20)
    def rgb(color: tuple[int, int, int]) -> str: return f"rgb({color[0]},{color[1]},{color[2]})"
    def draw_text(canvas: bytearray, x: int, y: int, text: str, size: int, color: tuple[int, int, int]) -> None:
        cursor = x
        for char in str(text):
            glyph = _BITMAP_FONT.get(char.upper(), _BITMAP_FONT[" "])
            for gy, bits in enumerate(glyph):
                for gx, bit in enumerate(bits):
                    if bit == "1":
                        for dx in range(size):
                            for dy in range(size):
                                xx, yy = cursor + gx * size + dx, y + gy * size + dy
                                if 0 <= xx < width and 0 <= yy < height: canvas[(yy * width + xx) * 3:(yy * width + xx + 1) * 3] = bytes(color)
            cursor += 6 * size
    pixels = bytearray(bytes(background) * width * height)
    for item in model:
        if item["type"] == "text": draw_text(pixels, int(item["x"]), int(item["y"]), item["text"], int(item["size"]), tuple(item["color"]))
        elif item["type"] == "line":
            steps = max(abs(round(item["x2"] - item["x1"])), abs(round(item["y2"] - item["y1"])), 1)
            dash_values = [int(float(part)) for part in str(item.get("dash", "")).replace(",", " ").split() if part]
            pattern = dash_values or [steps + 1]; pattern_total = max(1, sum(pattern))
            for i in range(steps + 1):
                position = i % pattern_total; cursor = 0; draw = True
                for index, length in enumerate(pattern):
                    if position < cursor + length:
                        draw = index % 2 == 0
                        break
                    cursor += length
                if not draw: continue
                xx, yy = round(item["x1"] + (item["x2"] - item["x1"]) * i / steps), round(item["y1"] + (item["y2"] - item["y1"]) * i / steps)
                if 0 <= xx < width and 0 <= yy < height: pixels[(yy * width + xx) * 3:(yy * width + xx + 1) * 3] = bytes(item["color"])
        elif item["type"] == "rect":
            for yy in range(max(0, round(item["y"])), min(height, round(item["y"] + item["h"]))):
                for xx in range(max(0, round(item["x"])), min(width, round(item["x"] + item["w"]))): pixels[(yy * width + xx) * 3:(yy * width + xx + 1) * 3] = bytes(item["color"])
        elif item["type"] == "mark":
            if item.get("selected"):
                for xx in range(round(item["x"] - 7), round(item["x"] + 8)):
                    for yy in (round(item["y"] - 7), round(item["y"] + 7)):
                        if 0 <= xx < width and 0 <= yy < height: pixels[(yy * width + xx) * 3:(yy * width + xx + 1) * 3] = bytes((20, 20, 20))
                for yy in range(round(item["y"] - 7), round(item["y"] + 8)):
                    for xx in (round(item["x"] - 7), round(item["x"] + 7)):
                        if 0 <= xx < width and 0 <= yy < height: pixels[(yy * width + xx) * 3:(yy * width + xx + 1) * 3] = bytes((20, 20, 20))
            segments = ((-5, 0, 5, 0), (0, -5, 0, 5)) if item["shape"] == "overflow" else (((-4, -4, 4, 4),) if item["shape"] == "circle" else ((-5, -5, 5, 5), (-5, 5, 5, -5)))
            if item["shape"] == "circle":
                segments = tuple((dx, dy, dx, dy) for dx in range(-4, 5) for dy in range(-4, 5) if dx * dx + dy * dy <= 20)
            for x1, y1, x2, y2 in segments:
                for i in range(11):
                    xx, yy = round(item["x"] + x1 + (x2 - x1) * i / 10), round(item["y"] + y1 + (y2 - y1) * i / 10)
                    if 0 <= xx < width and 0 <= yy < height: pixels[(yy * width + xx) * 3:(yy * width + xx + 1) * 3] = bytes(item["color"])
    def chunk(kind: bytes, payload: bytes) -> bytes: return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height)); png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'><rect width='100%' height='100%' fill='#f8fafc'/>"]
    for item in model:
        if item["type"] == "text": svg.append(f"<text x='{item['x']}' y='{item['y'] + 7 * item['size']}' font-size='{8 * item['size']}' fill='{rgb(tuple(item['color']))}'" + (f" data-role='{html.escape(item['role'])}'" if item.get("role") else "") + f" data-bbox-y='{item['y']}' data-bbox-height='{7 * item['size']}'>{html.escape(str(item['text']))}</text>")
        elif item["type"] == "line": svg.append(f"<line x1='{item['x1']}' y1='{item['y1']}' x2='{item['x2']}' y2='{item['y2']}' stroke='{rgb(item['color'])}'" + (f" stroke-dasharray='{item['dash']}'" if item.get("dash") else "") + (f" data-reference='{html.escape(item['label'])}'" if item.get("label") else "") + "/>" )
        elif item["type"] == "rect": svg.append(f"<rect x='{item['x']}' y='{item['y']}' width='{item['w']}' height='{item['h']}' fill='{rgb(item['color'])}'" + (f" data-cell='{html.escape(item['cell'])}'" if item.get("cell") else "") + (f" data-panel='{html.escape(item['panel'])}'" if item.get("panel") else "") + (f" data-label='{html.escape(item['label'])}'" if item.get("label") else "") + (f" data-colorbar='{html.escape(item['colorbar'])}'" if item.get("colorbar") else "") + (f" data-colorbar-position='{html.escape(item['colorbar_position'])}'" if item.get("colorbar_position") else "") + (f" data-value='{item['value']}'" if item.get("value") is not None else "") + (f" data-value-class='{item['value_class']}'" if item.get("value_class") else "") + (f" data-heatmap-layout='{item['heatmap_layout']}' data-heatmap-gap='{item['gap']}' data-data-right='{item['data_right']}' data-colorbar-x='{item['colorbar_x']}'" if item.get("heatmap_layout") else "") + "/>" )
        elif item["type"] == "mark":
            metadata = f"data-series='{html.escape(item['label'])}' data-shape='{item['shape']}'" + (" data-legend='true'" if item.get("legend") else "") + (" data-direction-legend='true'" if item.get("direction_legend") else "") + (f" data-panel='{html.escape(item['panel'])}'" if item.get("panel") else "") + (" data-selected='true'" if item.get("selected") else "") + (" data-overflow='true'" if item.get("overflow") else "") + (f" data-window-id='{item['window_id']}' data-window-key='{html.escape(item['window_key'])}'" if item.get("window_id") is not None else "")
            if item.get("selected"):
                svg.append(f"<rect data-selected-box='true' x='{item['x']-7}' y='{item['y']-7}' width='14' height='14' fill='none' stroke='rgb(20,20,20)' stroke-width='2'/>")
            if item["shape"] == "circle": svg.append(f"<circle {metadata} cx='{item['x']}' cy='{item['y']}' r='5' fill='{rgb(item['color'])}'/>")
            elif item["shape"] == "overflow": svg.append(f"<path {metadata} d='M{item['x']-5},{item['y']}L{item['x']+5},{item['y']}M{item['x']},{item['y']-5}L{item['x']},{item['y']+5}' stroke='{rgb(item['color'])}' fill='none'/>")
            else: svg.append(f"<path {metadata} d='M{item['x']-5},{item['y']-5}L{item['x']+5},{item['y']+5}M{item['x']-5},{item['y']+5}L{item['x']+5},{item['y']-5}' stroke='{rgb(item['color'])}' fill='none'/>")
    svg.append("</svg>")
    return png, "".join(svg)


def _plots(output_dir: Path, rows: list[dict[str, Any]], finite_difference: list[dict[str, Any]], fits: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[str]:
    names = ["response_magnitude", "slope_evidence", "finite_difference_cosine", "finite_difference_relative_rms", "paired_secant_cosine", "paired_secant_relative_rms", "diagnostics"]
    # Keep one evidence model for both formats; this avoids a readable SVG and
    # an unrelated/metadata-only PNG when optional plotting libraries differ.
    for name in names:
        png, svg = _render_fallback(name, rows, finite_difference, fits, decisions)
        (output_dir / f"{name}.png").write_bytes(png)
        _atomic_text(output_dir / f"{name}.svg", svg)
    return [item for name in names for item in (f"{name}.png", f"{name}.svg")]

def _write_artifacts_impl(result: dict[str, Any], output_dir: Path, records: Mapping[str, Mapping[str, Any]], *, context_root: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "scan_summary.json", {key: value for key, value in result.items() if key not in {"point_metrics", "fits", "window_decisions"}})
    scalar_rows = [{key: value for key, value in row.items() if not isinstance(value, np.ndarray)} for row in result.get("point_metrics", [])]
    _write_csv(output_dir / "response_metrics.csv", scalar_rows)
    _write_csv(output_dir / "finite_difference_consistency.csv", result.get("finite_difference", []))
    _write_csv(output_dir / "injection_diagnostics.csv", scalar_rows)
    _write_csv(output_dir / "fit_metrics.csv", result.get("fits", []))
    _atomic_json(output_dir / "fit_metrics.json", result.get("fits", []))
    _write_csv(output_dir / "window_decisions.csv", result.get("window_decisions", []))
    tensor_root = output_dir / "analysis_tensors"
    for row in result.get("point_metrics", []):
        if row.get("response") is None:
            continue
        path = tensor_root / "responses" / str(row["space"]) / f"direction_{int(row['direction']):02d}" / f"alpha_{str(row['alpha']).replace('.', 'p')}_{'plus' if row['sign'] == 1 else 'minus'}.npy"
        _atomic_npy(path, np.asarray(row["response"], dtype=np.float32))
        for key, label in (("target_derivative", "target_epsilon"), ("actual_derivative", "actual_step")):
            if row.get(key) is not None:
                derivative_path = tensor_root / "derivatives" / str(row["space"]) / f"direction_{int(row['direction']):02d}" / f"alpha_{str(row['alpha']).replace('.', 'p')}_{'plus' if row['sign'] == 1 else 'minus'}_{label}.npy"
                _atomic_npy(derivative_path, row[key])
        if int(row.get("sign", 0)) == 1:
            opposite = next((other for other in result.get("point_metrics", []) if other.get("space") == row.get("space") and other.get("direction") == row.get("direction") and other.get("alpha") == row.get("alpha") and other.get("sign") == -1), None)
            if opposite is not None and opposite.get("response") is not None:
                center_path = tensor_root / "paired" / str(row["space"]) / f"direction_{int(row['direction']):02d}" / f"alpha_{str(row['alpha']).replace('.', 'p')}_center.npy"
                _atomic_npy(center_path, ((np.asarray(row["response"], dtype=np.float32) + np.asarray(opposite["response"], dtype=np.float32)) / 2.0).astype(np.float32))
            if row.get("paired_secant") is not None:
                secant_path = tensor_root / "paired" / str(row["space"]) / f"direction_{int(row['direction']):02d}" / f"alpha_{str(row['alpha']).replace('.', 'p')}_secant.npy"
                _atomic_npy(secant_path, row["paired_secant"])
    chart_paths = _plots(output_dir, result.get("point_metrics", []), result.get("finite_difference", []), result.get("fits", []), result.get("window_decisions", []))
    chart_map = "\n".join(f"- `{path}` — neutral descriptive chart for the corresponding metric family." for path in chart_paths)
    _atomic_text(output_dir / "implementation_notes.md", "# Implementation notes\n\nAnalysis is CPU-only and rerunnable. Differences are float32; reductions and fits are float64.\n\n## Manifest/bundle contract\n\n`MANIFEST.sha256` is the final root manifest written after staged publication; `MANIFEST.analysis.sha256` is the analysis-owned subset. The review bundle is a deterministic analysis-stage manifest snapshot: its embedded manifest is not promised to be byte-identical to a later root manifest that also includes post-analysis provenance or execution logs. Verify the final root and analysis manifests separately. The bundle includes generated plots and compact status/source evidence, and excludes raw tensor corpora, model files, and unowned input previews. ZIP timestamps/order are fixed. Publication owns only the explicit generated artifact allowlist, so arbitrary `input_preview.png`/`.svg` and raw runner/sample inputs remain bytewise unchanged across success, failure, and diagnostic reruns.\n\n## Chart map\n\n" + (chart_map or "No charts: evidence is absent." ) + "\n")
    selected = [item for item in result.get("window_decisions", []) if item.get("selected")]
    finding = "A qualifying local-linear interval was found." if selected else "No candidate local-linear interval qualifies under the pre-registered gates."
    interval_lines = "\n".join(f"- `{item['space']}`, direction {item['direction']}: alpha {item['start_alpha']}–{item['end_alpha']}" for item in selected) or "- None"
    finite_outputs = [float(row["output_rms"]) for row in result.get("point_metrics", []) if row.get("output_rms") is not None]
    finite_slopes = [float(row["slope"]) for row in result.get("fits", []) if row.get("slope") is not None]
    finite_r2 = [float(row["r2"]) for row in result.get("fits", []) if row.get("r2") is not None]
    finite_gains = [float(row["gain"]) for row in result.get("point_metrics", []) if row.get("gain") is not None]
    finite_survival = [float(row["bf16_survival_ratio"]) for row in result.get("point_metrics", []) if row.get("bf16_survival_ratio") is not None]
    negative_count = len(result.get("window_decisions", [])) - len(selected)
    zero_response = sum(1 for row in result.get("point_metrics", []) if row.get("output_rms") == 0.0)
    def _range_or_na(values: list[float]) -> str:
        return "N/A" if not values else f"{min(values):.8g}-{max(values):.8g}"
    family_lines: list[str] = []
    for space in SPACES:
        space_rows = [row for row in result.get("point_metrics", []) if row.get("space") == space]
        space_fits = [row for row in result.get("fits", []) if row.get("space") == space]
        outputs = [float(row["output_rms"]) for row in space_rows if row.get("output_rms") is not None]
        gains = [float(row["gain"]) for row in space_rows if row.get("gain") is not None]
        slopes = [float(row["slope"]) for row in space_fits if row.get("slope") is not None]
        r2s = [float(row["r2"]) for row in space_fits if row.get("r2") is not None]
        floors = [float(row["floor_multiple"]) for row in space_rows if row.get("floor_multiple") is not None]
        family_lines.append(f"{space}: output RMS: {_range_or_na(outputs)}; gain: {_range_or_na(gains)}; slope: {_range_or_na(slopes)}; R²: {_range_or_na(r2s)}; response/floor ratio: {_range_or_na(floors)}.")
    numerical = ("Per-space metric families (each independently reported): " + " ".join(family_lines) + " "
                 + f"BF16 survival range: `{_range_or_na(finite_survival)}`; candidate windows selected/checked: `{len(selected)}/{len(result.get('window_decisions', []))}` (negative results `{negative_count}`); zero-response points `{zero_response}`. Finite input with zero output is reported as zero gain/response, while only zero/nonfinite denominators or floors are N/A.")
    figure_lines = "\n".join(f"- [ `{path}` ]({path}) — This figure shows the corresponding pre-registered response/consistency diagnostic; interpret missing or null points as unavailable evidence." for path in chart_paths)
    timing = result.get("timing", {})
    timing_lines = "\n".join(f"- {label}: `{_safe_json(value)}`" for label, value in timing.items()) or "- Timing unavailable in sample status evidence."
    _atomic_text(output_dir / "experiment_report.md", f"# UMI post-VAE latent perturbation scan\n\n## Technical summary\n\n{finding}\n\nCandidate intervals actually selected:\n{interval_lines}\n\nPer-space float baseline floors: `{_safe_json(result.get('baseline_floors', {}))}`. Candidate count: `{result.get('candidate_count')}`.\n\nConcrete numerical findings: {numerical}\n\n## Preserved execution intervals\n\n{timing_lines}\n\nThe `stage_a` interval is A start through D finish; `resumed_scan` is scan_pre start through scan_post finish; `full_formal` is the first formal A start through scan_post finish. These intervals are derived from per-sample status timestamps, not from a single process segment.\n\n## Key findings with figures\n\n{figure_lines or '- No figures were published.'}\n\nThe response-magnitude figure shows the measured response range and slope-1 reference; inspect the plotted points against the CSV values. The slope figure shows the actual slope/R² intervals summarized above. The finite-difference cosine and relative-RMS figures consume adjacent-index rows for both spaces and both signs; paired-secant metrics are reported in the same tensor/CSV evidence. The diagnostics figure shows gain, BF16 survival, mask exactness, and floor ratios; nulls include their explicit undefined reasons.\n\n## Scope/data/metric definitions\n\nThe reference is `scan_pre`; the float baseline floor is `scan_post - scan_pre`. Quantitative spaces are predicted latent and final float RGB. Differences are float32, while RMS reductions, cosine values, fits, and window decisions use float64. Input RMS, paired step, gain, and derivative denominators are computed only inside the authoritative condition mask. Undefined denominators are null with a specific reason in metric rows.\n\n## Experiment/validation details\n\nSaved tensors were validated with the hardened runner evidence validator before analysis. Root/sample provenance, compatibility bindings, artifact hashes, fixed call-plan identity, fixed predicted-noise identity, BF16 realized-carrier binding, and target epsilon geometry were checked before quantitative loading.\n\n## Limitations/robustness\n\nThis is four-direction interface/scale evidence only and cannot support a low-rank conclusion. Offline re-analysis loaded no model; the inference run did. No SVD was run and no low-rank conclusion is made. The result is sensitive to the cache-off baseline floor and to erased/nonfinite BF16 evidence.\n\n## Recommended next step\n\nPre-register a latent-scale extension/refinement pilot that tests whether directional consistency becomes stable over a narrower or shifted latent alpha range before increasing direction count or running SVD. This is a follow-up proposal only; it is not run here and does not claim low rank.\n\n## Further questions\n\nHow stable are the slopes and cache-off floors across seeds, prompts, and an independent direction bank? Which negative windows fail because of floor, BF16 survival, adjacency, or opposition?\n")
    _write_analysis_manifest(output_dir, context_root=context_root)
    _make_bundle(output_dir, records, context_root=context_root or output_dir.parent)


_GENERATED_PLOT_NAMES = {
    "response_magnitude.png", "response_magnitude.svg", "slope_evidence.png", "slope_evidence.svg",
    "finite_difference_cosine.png", "finite_difference_cosine.svg",
    "finite_difference_relative_rms.png", "finite_difference_relative_rms.svg",
    "paired_secant_cosine.png", "paired_secant_cosine.svg",
    "paired_secant_relative_rms.png", "paired_secant_relative_rms.svg",
    "diagnostics.png", "diagnostics.svg", "diagnostic_evidence.png", "diagnostic_evidence.svg",
}
_ANALYSIS_ROOT_NAMES = {"scan_summary.json", "response_metrics.csv", "finite_difference_consistency.csv", "injection_diagnostics.csv", "window_decisions.csv", "fit_metrics.csv", "fit_metrics.json", "implementation_notes.md", "experiment_report.md", "review_bundle.zip", "MANIFEST.analysis.sha256", "MANIFEST.sha256", *_GENERATED_PLOT_NAMES}


def _write_analysis_manifest(root: Path, context_root: Path | None = None) -> Path:
    manifest = root / "MANIFEST.analysis.sha256"
    excluded = {"MANIFEST.analysis.sha256", "MANIFEST.sha256", "review_bundle.zip"}
    analysis_files = [item for item in root.rglob("*") if item.is_file() and item.name not in excluded and ".analysis-stage-" not in item.parts and (item.name in _ANALYSIS_ROOT_NAMES or "analysis_tensors" in item.parts)]
    analysis_lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(analysis_files, key=lambda p: p.relative_to(root).as_posix())]
    _atomic_text(manifest, "\n".join(analysis_lines) + "\n")
    snapshot: dict[str, str] = {}
    all_files = [item for item in root.rglob("*") if item.is_file() and item.name not in excluded and ".analysis-stage-" not in item.parts]
    for path in all_files:
        snapshot[path.relative_to(root).as_posix()] = sha256_file(path)
    if context_root is not None and context_root != root:
        for path in context_root.rglob("*"):
            if not path.is_file() or path.name in excluded or path.name in _ANALYSIS_ROOT_NAMES or "analysis_tensors" in path.parts:
                continue
            snapshot[path.relative_to(context_root).as_posix()] = sha256_file(path)
    root_lines = [f"{digest}  {relative}" for relative, digest in sorted(snapshot.items())]
    _atomic_text(root / "MANIFEST.sha256", "\n".join(root_lines) + "\n")
    return manifest


def _publish_stage(stage: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    owned = [child for child in output_dir.iterdir() if child.name in _ANALYSIS_ROOT_NAMES or child.name == "analysis_tensors"]
    backup = Path(tempfile.mkdtemp(prefix=".analysis-backup-", dir=str(output_dir.parent)))
    moved_old: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        # Move complete old artifacts aside; they remain byte-identical until
        # the staged set and its final manifest have succeeded.
        for child in owned:
            target = backup / child.name
            os.replace(child, target)
            moved_old.append((child, target))
        for source in sorted(stage.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(stage)
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            published.append(destination)
        _write_analysis_manifest(output_dir)
    except BaseException:
        # Remove every newly published owned path, including partially-created
        # nested tensor trees, before restoring the old byte-identical set.
        for path in reversed(published):
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
        for directory in sorted((path for path in output_dir.rglob("*") if path.is_dir()), key=lambda p: len(p.parts), reverse=True):
            if directory.name == "analysis_tensors" or any(part in {"responses", "derivatives", "paired"} for part in directory.parts):
                try:
                    directory.rmdir()
                except OSError:
                    pass
        for original, saved in reversed(moved_old):
            if saved.exists():
                os.replace(saved, original)
        raise
    else:
        shutil.rmtree(backup, ignore_errors=True)


def _write_artifacts(result: dict[str, Any], output_dir: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".analysis-stage-", dir=str(output_dir.parent)))
    try:
        _write_artifacts_impl(result, stage, records, context_root=output_dir)
        _publish_stage(stage, output_dir)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _make_bundle(output_dir: Path, records: Mapping[str, Mapping[str, Any]], *, context_root: Path | None = None) -> None:
    bundle = output_dir / "review_bundle.zip"
    with tempfile.NamedTemporaryFile(delete=False, dir=str(output_dir), prefix=".review_bundle.") as temp:
        temp_name = temp.name
    try:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            entries: dict[str, bytes] = {}
            def add_bytes(name: str, payload: bytes) -> None:
                entries[name] = bytes(payload)
            allowed = {"scan_summary.json", "response_metrics.csv", "finite_difference_consistency.csv", "injection_diagnostics.csv", "window_decisions.csv", "fit_metrics.csv", "fit_metrics.json", "implementation_notes.md", "experiment_report.md", "MANIFEST.sha256", "MANIFEST.analysis.sha256", "config.json", "provenance.json", "call_plan.json", "status.json", "invocation.json"}
            for path in sorted(output_dir.rglob("*")):
                if path.is_file() and (path.name in allowed or path.name in _GENERATED_PLOT_NAMES or path.name == "sample.json" and "samples" in path.parts):
                    add_bytes(path.relative_to(output_dir).as_posix(), path.read_bytes())
            if context_root is not None and context_root != output_dir:
                for name in ("config.json", "provenance.json", "call_plan.json", "status.json", "invocation.json", "gpu_samples.csv"):
                    source = context_root / name
                    if source.is_file():
                        add_bytes(name, source.read_bytes())
                sample_root = context_root / "samples"
                if sample_root.is_dir():
                    for source in sorted(sample_root.rglob("*.json")):
                        if source.name in {"status.json", "sample.json"}:
                            add_bytes(source.relative_to(context_root).as_posix(), source.read_bytes())
            source_root = Path(__file__).resolve().parent
            for source in (source_root / "umi_fd_post_vae_scan.py", source_root / "umi_fd_post_vae_bridge.py", source_root / "analyze_umi_fd_post_vae_scan.py"):
                if source.is_file():
                    add_bytes(f"source/{source.name}", source.read_bytes())
            for sid, record in records.items():
                if sid.startswith("dir_00") and ("alpha_0p0001" in sid or "alpha_0p03" in sid):
                    add_bytes(f"representative/{sid}.json", json.dumps(_safe_json({key: value for key, value in record.items() if not isinstance(value, np.ndarray)}), allow_nan=False).encode("utf-8"))
                    for name in ("predicted_latent", "decoded_final", "actual_delta_fp32", "actual_delta_bf16"):
                        if isinstance(record.get(name), np.ndarray):
                            buffer = io.BytesIO(); np.save(buffer, record[name], allow_pickle=False)
                            add_bytes(f"representative/{sid}_{name}.npy", buffer.getvalue())
            for name in sorted(entries):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 0
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, entries[name])
    finally:
        os.replace(temp_name, bundle)


def analyze_records(records: Mapping[str, Mapping[str, Any]], *, output_dir: str | Path | None = None, timing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    output_path = Path(output_dir) if output_dir is not None else None
    required = [item["sample_id"] for item in build_call_plan()]
    missing = [sample for sample in required if sample not in records]
    if missing:
        return _diagnostic("missing required sample states: " + ", ".join(missing), output_path)
    validation_failures: list[str] = []
    for sample_id in required:
        record = records[sample_id]
        evidence = validate_sample_evidence(record, stage_a=sample_id in {"A", "B", "C", "D"})
        if not evidence.passed:
            validation_failures.append(f"{sample_id}: " + "; ".join(evidence.failures))
        noise_reason = _validate_noise_identity(record)
        if noise_reason:
            validation_failures.append(f"{sample_id}: {noise_reason}")
    if validation_failures:
        return _diagnostic("per-sample evidence validation failed: " + " | ".join(validation_failures), output_path)
    integrity_failures: list[str] = []
    target_failures: list[str] = []
    for sample_id, record in records.items():
        if record.get("spec", {}).get("kind") != "perturbation":
            continue
        _, actual_reason = _validated_delta(record, "actual_delta_fp32")
        if actual_reason:
            integrity_failures.append(f"{sample_id}: {actual_reason}")
        _, bf16_reason = _validated_delta(record, "actual_delta_bf16")
        if bf16_reason:
            integrity_failures.append(f"{sample_id}: {bf16_reason}")
        _, target_reason = _validated_delta(record, "target_delta_fp32")
        target_contract_reason = _validate_target_evidence(record)
        if target_reason or target_contract_reason:
            target_failures.append(f"{sample_id}: {target_reason or target_contract_reason}")
    if integrity_failures:
        return _diagnostic("quantitative delta integrity failed: " + " | ".join(integrity_failures), output_path)
    if target_failures:
        return _diagnostic("target perturbation evidence failed: " + " | ".join(target_failures), output_path)
    pre_values = {space: _space_value(records["scan_pre"], space) for space in SPACES}
    post_values = {space: _space_value(records["scan_post"], space) for space in SPACES}
    if any(value is None for value in pre_values.values()) or any(value is None for value in post_values.values()):
        return _diagnostic("baseline tensors are missing or non-finite", output_path)
    floors = {space: _rms((post_values[space].astype(np.float32) - pre_values[space].astype(np.float32)).astype(np.float32)) for space in SPACES}
    point_metrics: list[dict[str, Any]] = []
    for sample_id, record in records.items():
        spec = record.get("spec", {})
        if spec.get("kind") != "perturbation":
            continue
        for space in SPACES:
            row = _record_point(record, sample_id, space, pre_values[space], float(floors[space] or 0.0), int(spec["direction_index"]), float(spec["alpha"]), int(spec["sign"]))
            point_metrics.append(row)
    for space in SPACES:
        for direction in range(DIRECTION_COUNT):
            for alpha in ALPHAS:
                plus = next((row for row in point_metrics if row["space"] == space and row["direction"] == direction and row["sign"] == 1 and row["alpha"] == alpha), None)
                minus = next((row for row in point_metrics if row["space"] == space and row["direction"] == direction and row["sign"] == -1 and row["alpha"] == alpha), None)
                if plus is None or minus is None:
                    continue
                plus["plus_minus_input_cosine"] = _cosine(plus.get("input"), -np.asarray(minus["input"])) if plus.get("input") is not None and minus.get("input") is not None else None
                plus["plus_minus_response_cosine"] = _cosine(plus.get("response"), -np.asarray(minus["response"])) if plus.get("response") is not None and minus.get("response") is not None else None
                plus["even_symmetry_residual"] = _rms((np.asarray(plus["response"]) + np.asarray(minus["response"])) / 2.0) if plus.get("response") is not None and minus.get("response") is not None else None
                if plus.get("input") is not None and minus.get("input") is not None:
                    difference = np.asarray(plus["input"]) - np.asarray(minus["input"])
                    mask_values = np.asarray(plus.get("input_mask"), dtype=bool)
                    distance = _rms(difference[mask_values]) if mask_values.shape == difference.shape else _rms(difference)
                else:
                    distance = None
                plus["paired_secant"] = None if distance in (None, 0.0) or plus.get("response") is None or minus.get("response") is None else ((np.asarray(plus["response"], dtype=np.float32) - np.asarray(minus["response"], dtype=np.float32)) / np.float32(distance)).astype(np.float32)
                plus["paired_secant_rms"] = _rms(plus.get("paired_secant"))
                pair_reasons = []
                for metric, value in (("plus_minus_input_cosine", plus.get("plus_minus_input_cosine")), ("plus_minus_response_cosine", plus.get("plus_minus_response_cosine")), ("even_symmetry_residual", plus.get("even_symmetry_residual")), ("paired_secant_rms", plus.get("paired_secant_rms"))):
                    if value is None:
                        pair_reasons.append(f"{metric} undefined: paired vector is missing, zero, or nonfinite")
                if plus.get("paired_secant") is None:
                    pair_reasons.append("paired_secant undefined: paired step denominator is zero or missing")
                if pair_reasons:
                    plus["undefined_reasons"] = "; ".join(filter(None, [plus.get("undefined_reasons"), *pair_reasons]))
                _attach_null_reasons(plus)
    for row in point_metrics:
        _attach_null_reasons(row)
    fits: list[dict[str, Any]] = []
    for space in SPACES:
        for direction in range(DIRECTION_COUNT):
            for sign in SIGNS:
                rows = sorted([row for row in point_metrics if row["space"] == space and row["direction"] == direction and row["sign"] == sign], key=lambda row: row["alpha"])
                for start in range(max(0, len(rows) - MIN_WINDOW + 1)):
                    for end in range(start + MIN_WINDOW, len(rows) + 1):
                        window = rows[start:end]
                        fit = _fit_loglog([row.get("actual_input_rms") for row in window], [row.get("output_rms") for row in window], [row["sample_id"] for row in window], [])
                        fit_row = {"space": space, "direction": direction, "sign": sign, "start_alpha": window[0]["alpha"], "end_alpha": window[-1]["alpha"], **fit}
                        _attach_null_reasons(fit_row)
                        fits.append(fit_row)
    finite_difference: list[dict[str, Any]] = []
    for space in SPACES:
        for direction in range(DIRECTION_COUNT):
            for sign in SIGNS:
                rows = sorted([row for row in point_metrics if row["space"] == space and row["direction"] == direction and row["sign"] == sign], key=lambda row: row["alpha"])
                for index, (left, right) in enumerate(zip(rows, rows[1:])):
                    previous_pair = next((item for item in point_metrics if item["space"] == space and item["direction"] == direction and item["sign"] == 1 and item["alpha"] == left["alpha"]), None)
                    current_pair = next((item for item in point_metrics if item["space"] == space and item["direction"] == direction and item["sign"] == 1 and item["alpha"] == right["alpha"]), None)
                    derivative_cosine = _cosine(left.get("actual_derivative"), right.get("actual_derivative"))
                    derivative_relative = _relative(left.get("actual_derivative"), right.get("actual_derivative"))
                    paired_cosine = _cosine(previous_pair.get("paired_secant"), current_pair.get("paired_secant")) if previous_pair and current_pair else None
                    paired_relative = _relative(previous_pair.get("paired_secant"), current_pair.get("paired_secant")) if previous_pair and current_pair else None
                    input_cosine = _cosine(left.get("input"), right.get("input"))
                    relative_rms = _relative(left.get("input"), right.get("input"))
                    undefined = []
                    for label, value in (("input_cosine", input_cosine), ("relative_rms", relative_rms), ("one_sided_derivative_cosine", derivative_cosine), ("one_sided_derivative_relative_rms", derivative_relative), ("paired_secant_cosine", paired_cosine), ("paired_secant_relative_rms", paired_relative)):
                        if value is None:
                            undefined.append(f"{label} undefined")
                    finite_row = {"space": space, "direction": direction, "sign": sign, "adjacent_index": index, "alpha_left": left["alpha"], "alpha_right": right["alpha"], "input_cosine": input_cosine, "relative_rms": relative_rms, "one_sided_derivative_cosine": derivative_cosine, "one_sided_derivative_relative_rms": derivative_relative, "paired_secant_cosine": paired_cosine, "paired_secant_relative_rms": paired_relative, "undefined_reasons": "; ".join(undefined) if undefined else None}
                    _attach_null_reasons(finite_row)
                    finite_difference.append(finite_row)
    decisions = select_candidate_windows({"points": point_metrics, "fits": fits})
    selected = [item for item in decisions if item.get("selected")]
    status = "COMPLETE" if selected else "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL"
    result: dict[str, Any] = {"status": status, "baseline_floors": floors, "candidate_count": len(selected), "scientific_scope": "four-direction interface/scale evidence only; no low-rank conclusion", "point_metrics": point_metrics, "finite_difference": finite_difference, "fits": fits, "window_decisions": decisions}
    if timing is not None:
        result["timing"] = dict(timing)
    if output_path is not None:
        _write_artifacts(result, output_path, records)
    return _safe_json(result)


def _verify_manifest(root: Path) -> tuple[bool, str]:
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file():
        return False, "MANIFEST.sha256 is missing"
    try:
        entries: dict[str, str] = {}
        lines = manifest.read_text(encoding="ascii").splitlines()
        if not lines:
            return False, "MANIFEST.sha256 is empty"
        for line in lines:
            digest, relative = line.split("  ", 1)
            if not _SHA256_RE.fullmatch(digest) or not relative or relative in entries or Path(relative).is_absolute() or ".." in Path(relative).parts:
                return False, f"invalid manifest entry: {relative}"
            entries[relative] = digest
            path = root / relative
            if not path.is_file() or sha256_file(path) != digest:
                return False, f"manifest mismatch: {relative}"
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name not in {"MANIFEST.sha256", "MANIFEST.analysis.sha256", "review_bundle.zip"}}
        if set(entries) != actual:
            return False, "MANIFEST.sha256 does not enumerate every run file exactly once"
        analysis_manifest = root / "MANIFEST.analysis.sha256"
        if analysis_manifest.is_file():
            analysis_entries = analysis_manifest.read_text(encoding="ascii").splitlines()
            for line in analysis_entries:
                digest, relative = line.split("  ", 1)
                path = root / relative
                if not _SHA256_RE.fullmatch(digest) or not path.is_file() or sha256_file(path) != digest:
                    return False, f"analysis manifest mismatch: {relative}"
    except (OSError, ValueError):
        return False, "MANIFEST.sha256 is malformed"
    return True, "ok"


def _canonical(value: Any) -> str:
    return json.dumps(_safe_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_run_bindings(root: Path, config: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]], expected_plan: list[Mapping[str, Any]]) -> tuple[bool, str]:
    required_arrays = ("z0.npy", "mask.npy", "direction_bank.npy")
    if any(not (root / name).is_file() for name in required_arrays):
        return False, "root z0/mask/direction artifacts are missing"
    try:
        z0 = np.load(root / "z0.npy", allow_pickle=False)
        mask = np.load(root / "mask.npy", allow_pickle=False).astype(bool)
        directions = np.load(root / "direction_bank.npy", allow_pickle=False)
    except (OSError, ValueError) as error:
        return False, f"root quantitative artifacts cannot be read: {error}"
    if z0.shape != mask.shape or directions.ndim != z0.ndim + 1 or directions.shape[1:] != z0.shape or directions.shape[0] != DIRECTION_COUNT:
        return False, "root z0/mask/direction shapes are incompatible"
    for field, filename in (("z0_sha256", "z0.npy"), ("mask_sha256", "mask.npy"), ("direction_sha256", "direction_bank.npy")):
        if config.get(field) != sha256_file(root / filename):
            return False, f"root compatibility hash binding mismatch: {field}"
    for field in COMPATIBILITY_FIELDS:
        value = config.get(field)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            return False, f"config provenance field is not a valid sha256: {field}"
    if config.get("runner_sha256", config.get("code_sha256")) != config.get("code_sha256"):
        return False, "runner_sha256 does not equal finalized code_sha256"
    if config.get("checkpoint_sha256", config.get("model_sha256")) != config.get("model_sha256"):
        return False, "checkpoint_sha256 does not equal finalized model_sha256"
    root_provenance: Mapping[str, Any] = {}
    try:
        root_provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False, "root provenance is not valid JSON"
    if not isinstance(root_provenance, Mapping) or not _PROVENANCE_KEYS.issubset(root_provenance):
        return False, "root provenance schema is incomplete"
    for field in _PROVENANCE_KEYS - {"framework_root", "checkpoint_path", "vae_path", "input_path", "action_path"}:
        value = root_provenance.get(field)
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            return False, f"root provenance field is not a valid sha256: {field}"
    if any(root_provenance.get(alias) != config.get(source) for alias, source in (("framework_sha256", "framework_sha256"), ("model_sha256", "model_sha256"), ("vae_sha256", "vae_sha256"), ("code_sha256", "code_sha256"), ("bridge_sha256", "bridge_sha256"), ("config_sha256", "config_sha256"), ("input_sha256", "input_sha256"), ("action_sha256", "action_sha256"), ("direction_sha256", "direction_sha256"), ("z0_sha256", "z0_sha256"), ("mask_sha256", "mask_sha256"), ("noise_policy_sha256", "noise_policy_sha256"))):
        return False, "root provenance does not equal finalized config bindings"
    if root_provenance.get("runner_sha256") != root_provenance.get("code_sha256") or root_provenance.get("checkpoint_sha256") != root_provenance.get("model_sha256"):
        return False, "root provenance aliases are contradictory"
    if not np.all(np.isfinite(directions)):
        return False, "direction bank contains nonfinite values"
    try:
        regenerated = generate_directions_for_mask(mask, count=DIRECTION_COUNT, seed=DIRECTION_SEED).astype(np.float32, copy=False)
    except (TypeError, ValueError):
        return False, "approved CPU direction bank could not be regenerated"
    if regenerated.shape != directions.shape or regenerated.tobytes(order="C") != directions.astype(np.float32, copy=False).tobytes(order="C"):
        return False, "direction bank does not match approved CPU seed/count/mask generation"
    for index, direction in enumerate(directions):
        values = direction[mask]
        if values.size == 0 or not math.isclose(_rms(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            return False, f"direction {index} is not mask-only RMS=1"
        if np.any(direction[~mask] != 0.0):
            return False, f"direction {index} has nonzero values outside mask"
    if len({direction.astype(np.float32, copy=False).tobytes(order="C") for direction in directions}) != DIRECTION_COUNT:
        return False, "direction bank contains duplicate directions"
    for path_key, hash_key in (("framework_root", "framework_sha256"), ("checkpoint_path", "model_sha256"), ("vae_path", "vae_sha256"), ("input_path", "input_sha256"), ("action_path", "action_sha256")):
        source = Path(str(root_provenance[path_key]))
        if source.exists():
            try:
                observed = sha256_tree(source) if source.is_dir() else sha256_file(source)
            except OSError:
                observed = None
            if observed is not None and observed != root_provenance[hash_key]:
                return False, f"root provenance path/hash mismatch: {path_key}"
    expected_by_id = {str(spec["sample_id"]): spec for spec in expected_plan}
    provenance_fields = set(COMPATIBILITY_FIELDS)
    noise_values: set[str] = set()
    for sample_id, spec in expected_by_id.items():
        record = records.get(sample_id)
        if record is None:
            return False, f"missing sample record: {sample_id}"
        if _canonical(record.get("spec")) != _canonical(spec):
            return False, f"sample spec identity mismatch: {sample_id}"
        sample_provenance = record.get("provenance")
        sample_compatibility = record.get("compatibility") if isinstance(record, Mapping) else None
        if not isinstance(sample_provenance, Mapping) or not isinstance(sample_compatibility, Mapping) or _canonical(sample_compatibility) != _canonical({field: config.get(field) for field in provenance_fields}) or _canonical(sample_provenance) != _canonical(root_provenance) or any(sample_provenance.get(field) != config.get(field) for field in provenance_fields):
            return False, f"sample compatibility binding mismatch: {sample_id}"
        noise = record.get("initial_noise_hash", record.get("predicted_noise_hash"))
        noise_reason = _validate_noise_identity(record)
        if noise_reason:
            return False, f"{sample_id}: {noise_reason}"
        noise_values.add(noise)
        sample_dir = root / "samples" / sample_id
        try:
            status = json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
            required = [str(item) for item in status.get("required_artifacts", [])]
            hashes = status.get("artifact_sha256", {})
            actual_artifacts = {path.relative_to(sample_dir).as_posix() for path in sample_dir.rglob("*.npy")}
            if status.get("status") != "success" or set(required) != actual_artifacts or set(hashes) != actual_artifacts:
                return False, f"sample artifact inventory mismatch: {sample_id}"
            for name in actual_artifacts:
                if sha256_file(sample_dir / name) != hashes.get(name):
                    return False, f"sample artifact hash mismatch: {sample_id}/{name}"
            carrier = _array(record.get("carrier_fp32")).astype(np.float32, copy=False)
            if carrier.shape != z0.shape or carrier.tobytes(order="C") != z0.astype(np.float32, copy=False).tobytes(order="C"):
                return False, f"sample baseline carrier does not equal root z0: {sample_id}"
        except (OSError, json.JSONDecodeError, TypeError):
            return False, f"sample status/artifact metadata is invalid: {sample_id}"
        if spec.get("kind") == "perturbation":
            try:
                expected = construct_delta(z0, mask, directions[int(spec["direction_index"])], alpha=float(spec["alpha"]), sign=int(spec["sign"]))
                target = _array(record.get("target_delta_fp32"))
                if target.dtype != np.dtype(np.float32) or target.shape != expected.delta.shape or target.tobytes(order="C") != expected.delta.tobytes(order="C"):
                    return False, f"target delta binding mismatch: {sample_id}"
                if _validate_target_evidence(record) is not None:
                    return False, f"target epsilon/rms binding mismatch: {sample_id}"
            except (TypeError, ValueError, KeyError):
                return False, f"target perturbation evidence is missing: {sample_id}"
    if len(noise_values) != 1:
        return False, "predicted-noise identity differs across samples"
    return True, "ok"


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    status_path = root / "status.json"
    if not status_path.is_file():
        return _diagnostic("status.json is missing", root)
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return _diagnostic(f"status.json cannot be read: {error}", root)
    if status.get("status") != "COMPLETE" or not isinstance(status.get("gate"), Mapping) or status.get("gate", {}).get("passed") is not True:
        return _diagnostic("Stage A did not pass; diagnostic report only", root)
    required_root = ("config.json", "provenance.json", "call_plan.json", "MANIFEST.sha256")
    missing = [name for name in required_root if not (root / name).is_file()]
    if missing:
        return _diagnostic("missing root provenance: " + ", ".join(missing), root)
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        expected_plan = build_call_plan()
        stored_plan = json.loads((root / "call_plan.json").read_text(encoding="utf-8"))
        if stored_plan != expected_plan:
            return _diagnostic("call_plan.json does not match the fixed 54-call identity", root)
        missing_config = [field for field in COMPATIBILITY_FIELDS if field not in config or not config[field] or config[field] == "pending-stage-a"]
        if missing_config:
            return _diagnostic("config.json is missing provenance hashes: " + ", ".join(missing_config), root)
        if not isinstance(provenance, Mapping) or not provenance:
            return _diagnostic("provenance.json is empty", root)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        return _diagnostic(f"root provenance cannot be read: {error}", root)
    valid, reason = _verify_manifest(root)
    if not valid:
        return _diagnostic(reason, root)
    records: dict[str, dict[str, Any]] = {}
    expected_plan = build_call_plan()
    for spec in expected_plan:
        sample_id = spec["sample_id"]
        sample_dir = root / "samples" / sample_id
        try:
            sample_status = json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
            if sample_status.get("status") != "success":
                return _diagnostic(f"sample {sample_id} is not successful", root)
            record = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
            if not isinstance(record.get("provenance"), Mapping) or not record.get("provenance"):
                return _diagnostic(f"sample {sample_id} is missing provenance", root)
            for artifact in sample_dir.rglob("*.npy"):
                record[artifact.stem] = np.load(artifact, allow_pickle=False)
            records[sample_id] = record
        except (OSError, json.JSONDecodeError, ValueError) as error:
            return _diagnostic(f"sample {sample_id} cannot be loaded: {error}", root)
    valid_bindings, binding_reason = _validate_run_bindings(root, config, records, expected_plan)
    if not valid_bindings:
        return _diagnostic(binding_reason, root)
    return analyze_records(records, output_dir=root, timing=_collect_run_timing(root))


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _collect_run_timing(root: Path) -> dict[str, Any]:
    timestamps: dict[str, dict[str, Any]] = {}
    for sample_id in ("A", "D", "scan_pre", "scan_post"):
        path = root / "samples" / sample_id / "status.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        started, finished = _parse_timestamp(payload.get("started_utc")), _parse_timestamp(payload.get("finished_utc"))
        if started is not None and finished is not None:
            timestamps[sample_id] = {"started_utc": started.isoformat(), "finished_utc": finished.isoformat(), "wall_seconds": max(0.0, (finished - started).total_seconds())}
    def interval(start_id: str, finish_id: str) -> dict[str, Any] | None:
        start, finish = timestamps.get(start_id), timestamps.get(finish_id)
        if not start or not finish:
            return None
        start_time, finish_time = _parse_timestamp(start["started_utc"]), _parse_timestamp(finish["finished_utc"])
        if start_time is None or finish_time is None:
            return None
        return {"start_utc": start["started_utc"], "finish_utc": finish["finished_utc"], "wall_seconds": max(0.0, (finish_time - start_time).total_seconds()), "start_sample": start_id, "finish_sample": finish_id}
    return {"stage_a": interval("A", "D"), "resumed_scan": interval("scan_pre", "scan_post"), "full_formal": interval("A", "scan_post"), "sample_calls": timestamps}


def format_cli_summary(result: Mapping[str, Any]) -> str:
    """Return the public CLI contract without serializing point/tensor arrays."""
    summary = {key: result.get(key) for key in ("status", "candidate_count", "baseline_floors", "scientific_scope", "timing") if key in result}
    return json.dumps(_safe_json(summary), sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Offline UMI post-VAE scan analysis")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    result = analyze_run(args.run_dir)
    print(format_cli_summary(result))
    return 0 if result.get("status") in {"COMPLETE", "NO_CANDIDATE_LOCAL_LINEAR_INTERVAL", "DIAGNOSTIC_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
