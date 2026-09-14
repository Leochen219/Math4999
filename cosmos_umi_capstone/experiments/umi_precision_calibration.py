"""CPU-only analytic calibration for the UMI precision contrast.

This module is deliberately independent of Cosmos and Torch.  It reuses the
Task 1 BF16 rounding and array-hash definitions, but keeps the calibration
mechanism separate from any Cosmos runtime evidence.  The represented weight
values are generated once as FP32 and reused (exactly, after a lossless cast)
by both the FP64 reference and the FP32 rounding-simulation paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

try:  # package import
    from .umi_precision_primitives import quantize_bf16_fp32, sha256_array
except ImportError:  # direct import from experiments/
    from umi_precision_primitives import quantize_bf16_fp32, sha256_array


DEFAULT_SEED = 20260913
MATRIX_ROWS = 256
MATRIX_COLS = 128
DEFAULT_WEIGHT_SCALE = 1.0 / math.sqrt(MATRIX_COLS)
DEFAULT_H_GRID = np.logspace(-7, -1, 25, dtype=np.float64)
SCHEMA_VERSION = "umi-input-quantization-compute-precision-calibration-v1"


@dataclass(frozen=True)
class CalibrationInputs:
    """Deterministic calibration tensors.

    ``weights`` is the authoritative represented matrix: it is always FP32.
    FP64 computations consume ``weights.astype(np.float64)`` so that a wider
    compute dtype cannot silently introduce a different weight representation.
    """

    weights: np.ndarray
    x: np.ndarray
    v: np.ndarray
    seed: int
    weight_scale: float

    @property
    def weight_sha256(self) -> str:
        return sha256_array(self.weights)

    @property
    def x_sha256(self) -> str:
        return sha256_array(self.x)

    @property
    def v_sha256(self) -> str:
        return sha256_array(self.v)


@dataclass(frozen=True)
class CalibrationConfig:
    """Fixed CPU calibration configuration."""

    seed: int = DEFAULT_SEED
    matrix_rows: int = MATRIX_ROWS
    matrix_cols: int = MATRIX_COLS
    weight_scale: float = DEFAULT_WEIGHT_SCALE
    h_grid: tuple[float, ...] = tuple(float(value) for value in DEFAULT_H_GRID)
    input_quantization: str = "bfloat16"
    low_precision_compute_dtype: str = "float32"
    low_precision_execution: str = "rounding_simulation"

    def __post_init__(self) -> None:
        if int(self.matrix_rows) != MATRIX_ROWS or int(self.matrix_cols) != MATRIX_COLS:
            raise ValueError(f"calibration matrix must be exactly {MATRIX_ROWS}x{MATRIX_COLS}")
        if not math.isfinite(float(self.weight_scale)) or float(self.weight_scale) <= 0.0:
            raise ValueError("weight_scale must be positive and finite")
        values = np.asarray(self.h_grid, dtype=np.float64)
        if values.shape != (25,) or not np.all(np.isfinite(values)) or not np.all(values > 0.0):
            raise ValueError("h_grid must contain exactly 25 positive finite values")
        if not np.array_equal(values, DEFAULT_H_GRID):
            raise ValueError("h_grid is fixed to logspace(-7, -1, 25)")
        if self.input_quantization != "bfloat16":
            raise ValueError("input_quantization is fixed to bfloat16")
        if self.low_precision_compute_dtype != "float32":
            raise ValueError("low_precision_compute_dtype is fixed to float32")
        if self.low_precision_execution != "rounding_simulation":
            raise ValueError("CPU low-precision execution must be labeled rounding_simulation")

    def to_dict(self, inputs: CalibrationInputs) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "matrix_shape": [MATRIX_ROWS, MATRIX_COLS],
            "matrix_seed": int(self.seed),
            "weight_scale": float(self.weight_scale),
            "weight_representation": "fixed_fp32_values_reused_as_fp64",
            "weight_dtype": "float32",
            "weight_sha256": inputs.weight_sha256,
            "x_dtype": str(inputs.x.dtype),
            "x_sha256": inputs.x_sha256,
            "v_dtype": str(inputs.v.dtype),
            "v_sha256": inputs.v_sha256,
            "v_l2_norm": float(np.linalg.norm(inputs.v.astype(np.float64, copy=False))),
            "h_grid": [float(value) for value in self.h_grid],
            "h_grid_count": len(self.h_grid),
            "input_quantization": "bfloat16_rounding_simulation",
            "low_precision_compute_dtype": "float32",
            "low_precision_execution": "rounding_simulation",
            "native_low_precision_execution": False,
            "native_low_precision_reason": (
                "NumPy CPU calibration uses explicit FP32 rounding; no native BF16 kernel or Cosmos runtime was loaded"
            ),
            "cosmos_loaded": False,
            "gpu_generation_started": False,
            "modes": [
                {
                    "name": "fp64_reference",
                    "compute_dtype": "float64",
                    "input_quantization": "none",
                    "execution_kind": "fp64_reference",
                },
                {
                    "name": "input_quantized_bf16",
                    "compute_dtype": "float64",
                    "input_quantization": "bfloat16_rounding_simulation",
                    "execution_kind": "input_quantization_rounding_simulation",
                },
                {
                    "name": "low_precision_compute_fp32",
                    "compute_dtype": "float32",
                    "input_quantization": "none",
                    "execution_kind": "rounding_simulation",
                },
            ],
        }


@dataclass(frozen=True)
class ForwardEvaluation:
    value: np.ndarray
    input_values: np.ndarray
    compute_dtype: str
    input_quantization: str
    execution_kind: str
    weight_dtype: str
    weight_sha256: str
    input_sha256: str


@dataclass(frozen=True)
class CalibrationResult:
    config: CalibrationConfig
    inputs: CalibrationInputs
    h_grid: np.ndarray
    target_jv: np.ndarray
    rows: tuple[dict[str, Any], ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)


def _finite_array(value: Any, *, name: str, dtype: Any | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and finite")
    return np.ascontiguousarray(array)


def build_calibration_inputs(
    *,
    seed: int = DEFAULT_SEED,
    matrix_rows: int = MATRIX_ROWS,
    matrix_cols: int = MATRIX_COLS,
    weight_scale: float = DEFAULT_WEIGHT_SCALE,
) -> CalibrationInputs:
    """Build the fixed 256x128 FP32 matrix and deterministic x/v pair."""

    if int(matrix_rows) != MATRIX_ROWS or int(matrix_cols) != MATRIX_COLS:
        raise ValueError(f"calibration matrix must be exactly {MATRIX_ROWS}x{MATRIX_COLS}")
    if not math.isfinite(float(weight_scale)) or float(weight_scale) <= 0.0:
        raise ValueError("weight_scale must be positive and finite")
    rng = np.random.default_rng(int(seed))
    raw_weights = rng.standard_normal((MATRIX_ROWS, MATRIX_COLS)).astype(np.float32)
    weights = np.ascontiguousarray(raw_weights * np.float32(weight_scale), dtype=np.float32)
    x = np.ascontiguousarray(rng.standard_normal(MATRIX_COLS).astype(np.float64))
    v = np.ascontiguousarray(rng.standard_normal(MATRIX_COLS).astype(np.float64))
    norm = float(np.linalg.norm(v))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError("generated direction is not a finite nonzero vector")
    v = np.ascontiguousarray(v / norm, dtype=np.float64)
    return CalibrationInputs(weights=weights, x=x, v=v, seed=int(seed), weight_scale=float(weight_scale))


make_calibration_inputs = build_calibration_inputs


def _fixed_weights(weights: Any) -> np.ndarray:
    array = _finite_array(weights, name="weights", dtype=np.float32)
    if array.ndim != 2 or any(int(dim) <= 0 for dim in array.shape):
        raise ValueError("weights must be a non-empty two-dimensional matrix")
    return array


def _input_vector(value: Any, *, name: str = "input") -> np.ndarray:
    array = _finite_array(value, name=name)
    if array.ndim != 1 or array.shape[0] <= 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    return array


def _input_representation(value: np.ndarray, input_quantization: str) -> np.ndarray:
    if input_quantization == "none":
        return np.ascontiguousarray(value)
    if input_quantization == "bfloat16":
        # Task 1's exact BF16 representation is kept in FP32 for a portable
        # hash and for an explicit, framework-independent rounding simulation.
        return np.ascontiguousarray(quantize_bf16_fp32(value), dtype=np.float32)
    if input_quantization in {"float32", "fp32"}:
        return np.ascontiguousarray(value, dtype=np.float32)
    raise ValueError(f"unsupported input_quantization: {input_quantization}")


def evaluate_forward(
    weights: Any,
    x: Any,
    *,
    compute_dtype: str = "float64",
    input_quantization: str = "none",
) -> ForwardEvaluation:
    """Evaluate ``F(x)=W*x + 0.1*(W*x)^3`` with explicit dtype controls."""

    fixed_weights = _fixed_weights(weights)
    source_x = _input_vector(x)
    if source_x.shape[0] != fixed_weights.shape[1]:
        raise ValueError("input width must equal the weight matrix column count")
    represented_x = _input_representation(source_x, input_quantization)
    if compute_dtype == "float64":
        # Casting the fixed FP32 values to FP64 is exact; no new weight values
        # are introduced by the reference path.
        w = fixed_weights.astype(np.float64, copy=False)
        input_for_compute = represented_x.astype(np.float64, copy=False)
        wx = np.matmul(w, input_for_compute).astype(np.float64, copy=False)
        cubic = np.multiply(np.multiply(wx, wx, dtype=np.float64), wx, dtype=np.float64)
        value = np.add(wx, np.multiply(np.float64(0.1), cubic, dtype=np.float64), dtype=np.float64)
        execution_kind = (
            "fp64_reference" if input_quantization == "none" else "input_quantization_rounding_simulation"
        )
    elif compute_dtype == "float32":
        # Every primitive is explicitly FP32 so this path is reproducible as a
        # CPU rounding simulation rather than an implicit native low-precision
        # claim.
        w = fixed_weights
        input_for_compute = represented_x.astype(np.float32, copy=False)
        wx = np.matmul(w, input_for_compute).astype(np.float32, copy=False)
        squared = np.multiply(wx, wx, dtype=np.float32)
        cubic = np.multiply(squared, wx, dtype=np.float32)
        value = np.add(wx, np.multiply(np.float32(0.1), cubic, dtype=np.float32), dtype=np.float32)
        execution_kind = "rounding_simulation"
    else:
        raise ValueError("compute_dtype must be float64 or float32")
    return ForwardEvaluation(
        value=np.ascontiguousarray(value),
        input_values=np.ascontiguousarray(represented_x),
        compute_dtype=compute_dtype,
        input_quantization=input_quantization,
        execution_kind=execution_kind,
        weight_dtype="float32",
        weight_sha256=sha256_array(fixed_weights),
        input_sha256=sha256_array(represented_x),
    )


def analytic_jacobian_vector(weights: Any, x: Any, v: Any) -> np.ndarray:
    """Return ``[1+0.3*(W*x)^2] o (W*v)`` using the fixed FP32 W values."""

    fixed_weights = _fixed_weights(weights).astype(np.float64, copy=False)
    point = _input_vector(x, name="x").astype(np.float64, copy=False)
    direction = _input_vector(v, name="v").astype(np.float64, copy=False)
    if point.shape != direction.shape:
        raise ValueError("x and v must have identical shapes")
    if point.shape[0] != fixed_weights.shape[1] or direction.shape[0] != fixed_weights.shape[1]:
        raise ValueError("x and v widths must equal the weight matrix column count")
    wx = np.matmul(fixed_weights, point)
    wv = np.matmul(fixed_weights, direction)
    return np.ascontiguousarray((1.0 + 0.3 * (wx**2)) * wv, dtype=np.float64)


def _difference_inputs(x: Any, v: Any, h: float) -> tuple[np.ndarray, np.ndarray, float]:
    point = _input_vector(x, name="x").astype(np.float64, copy=False)
    direction = _input_vector(v, name="v").astype(np.float64, copy=False)
    step = float(h)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("h must be positive and finite")
    return point, direction, step


def _as_fp64(value: Any, *, name: str) -> np.ndarray:
    array = _finite_array(value, name=name, dtype=np.float64)
    return np.ascontiguousarray(array, dtype=np.float64)


def one_sided_difference(function: Callable[[np.ndarray], Any], x: Any, v: Any, h: float) -> np.ndarray:
    """Compute ``(F(x+h*v)-F(x))/h`` after FP64 promotion of both values."""

    point, direction, step = _difference_inputs(x, v, h)
    plus = _as_fp64(function(point + np.float64(step) * direction), name="one-sided plus")
    base = _as_fp64(function(point), name="one-sided base")
    if plus.shape != base.shape:
        raise ValueError("one-sided evaluations must have identical shapes")
    return np.ascontiguousarray((plus - base) / np.float64(step), dtype=np.float64)


def centered_difference(function: Callable[[np.ndarray], Any], x: Any, v: Any, h: float) -> np.ndarray:
    """Compute ``(F(x+h*v)-F(x-h*v))/(2h)`` after FP64 promotion."""

    point, direction, step = _difference_inputs(x, v, h)
    plus = _as_fp64(function(point + np.float64(step) * direction), name="centered plus")
    minus = _as_fp64(function(point - np.float64(step) * direction), name="centered minus")
    if plus.shape != minus.shape:
        raise ValueError("centered evaluations must have identical shapes")
    return np.ascontiguousarray((plus - minus) / np.float64(2.0 * step), dtype=np.float64)


def _rms(value: Any) -> float:
    array = _as_fp64(value, name="RMS value")
    return float(np.sqrt(np.mean(array * array, dtype=np.float64)))


def relative_error_with_reason(estimate: Any, reference: Any) -> dict[str, Any]:
    """Return relative RMS error, or an explicit N/A reason for zero denominator."""

    left = _as_fp64(estimate, name="estimate")
    right = _as_fp64(reference, name="reference")
    if left.shape != right.shape:
        raise ValueError("estimate and reference must have identical shapes")
    denominator = _rms(right)
    if denominator == 0.0:
        return {"value": None, "reason": "reference denominator RMS is zero"}
    numerator = _rms(left - right)
    return {"value": float(numerator / denominator), "reason": None}


def _absolute_error_with_reason(estimate: Any, reference: Any) -> dict[str, Any]:
    left = _as_fp64(estimate, name="estimate")
    right = _as_fp64(reference, name="reference")
    if left.shape != right.shape:
        raise ValueError("estimate and reference must have identical shapes")
    return {"value": float(_rms(left - right)), "reason": None}


def _mode_specs() -> tuple[dict[str, str], ...]:
    return (
        {
            "name": "fp64_reference",
            "compute_dtype": "float64",
            "input_quantization": "none",
            "execution_kind": "fp64_reference",
        },
        {
            "name": "input_quantized_bf16",
            "compute_dtype": "float64",
            "input_quantization": "bfloat16",
            "execution_kind": "input_quantization_rounding_simulation",
        },
        {
            "name": "low_precision_compute_fp32",
            "compute_dtype": "float32",
            "input_quantization": "none",
            "execution_kind": "rounding_simulation",
        },
    )


def scan_calibration(config: CalibrationConfig | None = None) -> CalibrationResult:
    """Scan both finite-difference schemes over the fixed 25-point h grid."""

    cfg = config or CalibrationConfig()
    inputs = build_calibration_inputs(
        seed=cfg.seed,
        matrix_rows=cfg.matrix_rows,
        matrix_cols=cfg.matrix_cols,
        weight_scale=cfg.weight_scale,
    )
    h_grid = np.asarray(cfg.h_grid, dtype=np.float64)
    target = analytic_jacobian_vector(inputs.weights, inputs.x, inputs.v)
    rows: list[dict[str, Any]] = []
    for spec in _mode_specs():
        def evaluate(point: np.ndarray, _spec: Mapping[str, str] = spec) -> np.ndarray:
            return evaluate_forward(
                inputs.weights,
                point,
                compute_dtype=_spec["compute_dtype"],
                input_quantization=_spec["input_quantization"],
            ).value

        for scheme, difference in (("one_sided", one_sided_difference), ("centered", centered_difference)):
            for h in h_grid:
                derivative: np.ndarray | None = None
                derivative_reason: str | None = None
                try:
                    derivative = difference(evaluate, inputs.x, inputs.v, float(h))
                    if not np.all(np.isfinite(derivative)):
                        raise ValueError(f"{scheme} derivative is non-finite")
                    relative = relative_error_with_reason(derivative, target)
                    absolute = _absolute_error_with_reason(derivative, target)
                except (FloatingPointError, OverflowError, TypeError, ValueError) as error:
                    derivative_reason = f"N/A: {type(error).__name__}: {error}"
                    derivative = None
                    relative = {"value": None, "reason": derivative_reason}
                    absolute = {"value": None, "reason": derivative_reason}
                rows.append(
                    {
                        "mode": spec["name"],
                        "scheme": scheme,
                        "h": float(h),
                        "compute_dtype": spec["compute_dtype"],
                        "input_quantization": spec["input_quantization"],
                        "execution_kind": spec["execution_kind"],
                        "weight_dtype": "float32",
                        "weight_sha256": inputs.weight_sha256,
                        "target_jv_rms": _rms(target),
                        "derivative_rms": None if derivative is None else _rms(derivative),
                        "derivative_reason": derivative_reason,
                        "error": relative["value"],
                        "error_reason": relative["reason"],
                        "absolute_error_rms": absolute["value"],
                        "absolute_error_reason": absolute["reason"],
                    }
                )
    return CalibrationResult(
        config=cfg,
        inputs=inputs,
        h_grid=np.ascontiguousarray(h_grid),
        target_jv=np.ascontiguousarray(target),
        rows=tuple(rows),
    )


calibrate = scan_calibration


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


CSV_FIELDS = (
    "mode",
    "scheme",
    "h",
    "compute_dtype",
    "input_quantization",
    "execution_kind",
    "weight_dtype",
    "weight_sha256",
    "target_jv_rms",
    "derivative_rms",
    "derivative_reason",
    "error",
    "error_reason",
    "absolute_error_rms",
    "absolute_error_reason",
)


def _write_metrics_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else _json_safe(row.get(field)) for field in CSV_FIELDS})


def _write_error_plot(path_png: Path, path_svg: Path, result: CalibrationResult) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment-specific fallback
        raise RuntimeError("matplotlib is required to write PNG+SVG calibration plots") from error

    old_hashsalt = matplotlib.rcParams.get("svg.hashsalt")
    matplotlib.rcParams["svg.hashsalt"] = "umi_precision_calibration_v1"
    try:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for axis, scheme in zip(axes, ("one_sided", "centered")):
            for mode in ("fp64_reference", "input_quantized_bf16", "low_precision_compute_fp32"):
                selected = [row for row in result.rows if row["scheme"] == scheme and row["mode"] == mode]
                h_values = np.asarray([row["h"] for row in selected], dtype=np.float64)
                error_values = np.asarray(
                    [np.nan if row["error"] is None else max(float(row["error"]), np.finfo(np.float64).tiny) for row in selected],
                    dtype=np.float64,
                )
                finite = np.isfinite(error_values)
                if np.any(finite):
                    axis.loglog(h_values[finite], error_values[finite], marker="o", linewidth=1.2, markersize=3, label=mode)
            axis.set_title(f"{scheme.replace('_', ' ').title()} derivative")
            axis.set_xlabel("h")
            axis.set_ylabel("relative RMS error")
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(fontsize=8)
        figure.suptitle("UMI CPU analytic precision calibration")
        figure.savefig(
            path_png,
            dpi=160,
            metadata={"Date": None, "Software": "UMI CPU analytic precision calibration v1"},
        )
        # SVG accepts a narrower metadata schema than PNG; Date=None removes
        # the backend's otherwise volatile dc:date element.
        figure.savefig(path_svg, metadata={"Date": None})
        # Matplotlib's SVG backend emits trailing spaces in path continuation
        # lines. Normalize whitespace and line endings so generated artifacts
        # are byte-identical across output directories and runs.
        svg_text = path_svg.read_text(encoding="utf-8")
        normalized_svg = "\n".join(line.rstrip(" \t") for line in svg_text.splitlines()) + "\n"
        with path_svg.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(normalized_svg)
    finally:
        matplotlib.rcParams["svg.hashsalt"] = old_hashsalt
        plt.close("all")


def _summary(result: CalibrationResult) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode in ("fp64_reference", "input_quantized_bf16", "low_precision_compute_fp32"):
        for scheme in ("one_sided", "centered"):
            values = [row["error"] for row in result.rows if row["mode"] == mode and row["scheme"] == scheme and row["error"] is not None]
            key = f"{mode}:{scheme}"
            summary[key] = {
                "finite_points": len(values),
                "min_relative_error": None if not values else float(min(values)),
                "max_relative_error": None if not values else float(max(values)),
                "undefined_points": sum(
                    row["error"] is None for row in result.rows if row["mode"] == mode and row["scheme"] == scheme
                ),
            }
    return summary


def _report_text(result: CalibrationResult) -> str:
    summary = _summary(result)
    lines = [
        "# UMI analytic precision calibration",
        "",
        "This is a CPU-only method-calibration layer; it does not load Cosmos, SSH, start GPU generation, or download/upgrade dependencies.",
        "",
        "## Fixed setup",
        "",
        f"- Matrix: fixed `{MATRIX_ROWS}x{MATRIX_COLS}` FP32 values, seed `{result.config.seed}`, scale `{result.config.weight_scale:.17g}`.",
        f"- Input/direction: deterministic `x` and unit-L2 `v`; `v` norm is `{np.linalg.norm(result.inputs.v):.17g}`.",
        "- The same represented FP32 weight values are used in FP64 and FP32 paths; the FP64 cast is lossless.",
        "- Input-quantized mode is Task 1 BF16 round-to-nearest-even represented again as FP32.",
        "- Low-precision compute is explicitly `float32` rounding simulation, not native BF16 execution.",
        "- Differences are promoted to FP64 before subtraction and RMS reduction.",
        "",
        "## Scan",
        "",
        "The scan uses `h=logspace(-7,-1,25)` for one-sided and centered differences. Values and N/A reasons are in `metrics.csv` and `metrics.json`.",
        "",
        "| mode | scheme | finite points | min relative error | max relative error |",
        "|---|---|---:|---:|---:|",
    ]
    for key, values in summary.items():
        mode, scheme = key.split(":", 1)
        lines.append(
            f"| `{mode}` | `{scheme}` | {values['finite_points']} | {values['min_relative_error']} | {values['max_relative_error']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This calibration measures a known cubic CPU function and is not evidence about Cosmos L/M/eta values. FP32 is not ground truth; the FP64 path is a declared numerical reference with the same FP32 represented W. A zero baseline response does not imply eta=0. One direction does not establish a full Jacobian or low-rank structure. A missing finite-difference window does not prove non-differentiability. The conditional bound in the separate math note does not require a 1/h or U-shaped curve.",
            "",
            "See `experiments/umi_precision_calibration_math.md` for the step-by-step conditional error-bound proof and assumptions.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_calibration_artifacts(
    output_dir: str | Path,
    config: CalibrationConfig | None = None,
) -> dict[str, Any]:
    """Run the scan and write the complete lightweight artifact set."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = scan_calibration(config)
    inputs = result.inputs
    config_payload = result.config.to_dict(inputs)
    _write_json(output / "config.json", config_payload)
    metrics_payload = {
        "status": "COMPLETE",
        "schema_version": SCHEMA_VERSION,
        "row_count": result.row_count,
        "target_jv_sha256": sha256_array(result.target_jv),
        "target_jv_rms": _rms(result.target_jv),
        "summary": _summary(result),
        "rows": list(result.rows),
    }
    _write_json(output / "metrics.json", metrics_payload)
    _write_metrics_csv(output / "metrics.csv", result.rows)
    _write_error_plot(output / "error_plot.png", output / "error_plot.svg", result)
    (output / "calibration_report.md").write_text(_report_text(result), encoding="utf-8")

    files = {
        path.name: _sha256_file(path)
        for path in sorted(output.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "hashes.json"
    }
    _write_json(
        output / "hashes.json",
        {
            "schema_version": SCHEMA_VERSION,
            "manifest_scope": "all calibration artifact files except hashes.json itself",
            "files": files,
        },
    )
    return {
        "status": "COMPLETE",
        "output_dir": str(output),
        "row_count": result.row_count,
        "files": sorted(list(files) + ["hashes.json"]),
        "weight_sha256": inputs.weight_sha256,
    }


write_calibration_artifacts = generate_calibration_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only UMI analytic precision calibration")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "artifacts" / "umi_precision_calibration"),
    )
    args = parser.parse_args(argv)
    print(json.dumps(generate_calibration_artifacts(args.output_dir), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CalibrationConfig",
    "CalibrationInputs",
    "CalibrationResult",
    "DEFAULT_H_GRID",
    "DEFAULT_SEED",
    "ForwardEvaluation",
    "analytic_jacobian_vector",
    "build_calibration_inputs",
    "calibrate",
    "centered_difference",
    "evaluate_forward",
    "generate_calibration_artifacts",
    "make_calibration_inputs",
    "one_sided_difference",
    "relative_error_with_reason",
    "scan_calibration",
    "write_calibration_artifacts",
]
