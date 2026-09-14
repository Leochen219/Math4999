"""Production launcher for the bounded, real UMI Task 5 experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

try:
    from .analyze_umi_task5 import analyze_task5_run
    from .run_umi_precision_experiment import (DEFAULT_CHECKPOINT, DEFAULT_FRAMEWORK_ROOT, DEFAULT_OLD_RUN,
        BlockedExecution, _legacy_args, _load_old_provenance, _resolve_inputs, load_legacy_direction_bank,
        require_torch_contract, select_lowest_unused_run, verify_uniform_eager_runtime)
    from .umi_precision_primitives import quantize_bf16_fp32
    from .umi_task5_decoder import run_task5_decoder_replays
    from .umi_task5_primitives import linear_formula_calibration
    from .umi_task5_runtime import Task5Inputs, run_task5_experiment
except ImportError:
    from analyze_umi_task5 import analyze_task5_run
    from run_umi_precision_experiment import (DEFAULT_CHECKPOINT, DEFAULT_FRAMEWORK_ROOT, DEFAULT_OLD_RUN,
        BlockedExecution, _legacy_args, _load_old_provenance, _resolve_inputs, load_legacy_direction_bank,
        require_torch_contract, select_lowest_unused_run, verify_uniform_eager_runtime)
    from umi_precision_primitives import quantize_bf16_fp32
    from umi_task5_decoder import run_task5_decoder_replays
    from umi_task5_primitives import linear_formula_calibration
    from umi_task5_runtime import Task5Inputs, run_task5_experiment


DEFAULT_RUN_ROOT = None
DEFAULT_TASK4_RUN = None
RUN_PREFIX = "umi_task5_directional_linearity_run"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlockedExecution(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise BlockedExecution(f"expected JSON object: {path}")
    return value


def _check_sample_hashes(sample: Path, status: Mapping[str, Any]) -> None:
    try:
        from umi_fd_post_vae_scan import sha256_file
    except ImportError:
        from .umi_fd_post_vae_scan import sha256_file
    for filename, digest in status.get("artifact_sha256", {}).items():
        path = sample / str(filename)
        if not path.is_file() or sha256_file(path) != digest:
            raise BlockedExecution(f"Task 4 C_pre raw artifact hash mismatch: {path}")


def load_task4_c_reference(task4_run: str | Path) -> dict[str, Any]:
    """Read the actual successful C_pre tensors, never a redrawn Task 4 report."""
    root = Path(task4_run).resolve()
    status = _load_json(root / "status.json")
    if status.get("status") != "complete" or int(status.get("formal_successful", 0)) != 42 or status.get("scope") != "full":
        raise BlockedExecution("Task 4 is not the completed 42-call full-scope precision run")
    sample = root / "samples" / "C_pre"
    sample_status = _load_json(sample / "status.json")
    if sample_status.get("status") != "success":
        raise BlockedExecution("Task 4 C_pre is not a successful formal record")
    _check_sample_hashes(sample, sample_status)
    required = {name: sample / f"{name}.npy" for name in ("z_bar", "common_input_fp32", "direction", "mask", "predicted_latent")}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise BlockedExecution("Task 4 C_pre is missing raw tensors: " + ", ".join(missing))
    arrays = {name: np.load(path, allow_pickle=False) for name, path in required.items()}
    if arrays["z_bar"].dtype != np.float32 or not np.array_equal(arrays["z_bar"], arrays["common_input_fp32"]):
        raise BlockedExecution("Task 4 C_pre z_bar/common FP32 condition evidence is inconsistent")
    if arrays["direction"].shape != arrays["z_bar"].shape or arrays["mask"].shape != arrays["z_bar"].shape:
        raise BlockedExecution("Task 4 C_pre input tensors do not share carrier geometry")
    sample_json = _load_json(sample / "sample.json")
    slicing = sample_json.get("latent_slicing", {})
    if not slicing or slicing.get("selected_shape") == slicing.get("source_shape") or not slicing.get("condition_indexes") or not slicing.get("predicted_indexes"):
        raise BlockedExecution("Task 4 C_pre predicted latent does not prove fixed condition positions were excluded")
    metadata = {"task4_run": str(root), "task4_status": status.get("status"), "task4_formal_successful": int(status["formal_successful"]),
                "task4_scope": status.get("scope"), "c_pre_sample": str(sample), "c_pre_slicing": slicing,
                "z_bar_sha256": _array_hash(arrays["z_bar"]), "direction_sha256": _array_hash(arrays["direction"]),
                "mask_sha256": _array_hash(arrays["mask"])}
    return {"z_bar": arrays["z_bar"].astype(np.float32, copy=True), "direction": arrays["direction"].astype(np.float32, copy=True),
            "mask": arrays["mask"].astype(bool, copy=True), "metadata": metadata}


def _array_hash(value: np.ndarray) -> str:
    import hashlib
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(); digest.update(str(array.dtype).encode("ascii")); digest.update(str(array.shape).encode("ascii")); digest.update(array.tobytes())
    return digest.hexdigest()


def _task4_preflight(task4_root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the required Task 4 B/C report evidence into the new raw run."""
    candidates = [task4_root / "precision_analysis" / "revision02", task4_root / "precision_analysis"]
    files: dict[str, Path] = {}
    for candidate in candidates:
        if candidate.is_dir():
            for name in ("fit_metrics.csv", "window_decisions.csv", "difference_metrics.csv", "baseline_floors.csv", "precision_summary.json"):
                path = candidate / name
                if path.is_file() and name not in files:
                    files[name] = path
    rows: dict[str, list[dict[str, str]]] = {}
    for name, path in files.items():
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as stream:
                rows[name] = list(csv.DictReader(stream))
    b_windows = [row for row in rows.get("window_decisions.csv", []) if str(row.get("group", "")) == "B"]
    c_fit = [row for row in rows.get("fit_metrics.csv", []) if str(row.get("group", "")) == "C"]
    c_differences = [row for row in rows.get("difference_metrics.csv", []) if str(row.get("group", "")) == "C"]
    summary = ("Task 4 C full scope confirmed; B's strict windows are retained as failures due to effective input-direction "
               "and paired-secant criteria. C's predicted-latent slopes and adjacent secant diagnostics are copied verbatim below.")
    return {"summary": summary, "reference": dict(reference["metadata"]),
            "analysis_files": {name: str(path) for name, path in files.items()}, "B_window_failures": b_windows,
            "C_fit_metrics": c_fit, "C_difference_metrics": c_differences,
            "known_reported_C_values": {"plus_slope": 1.0071393692023047, "minus_slope": 1.0021137094048815,
                "adjacent_predicted_latent_cosines": [0.9999923, 0.9999993, 0.9999998, 0.9999839, 0.9990868],
                "maximum_adjacent_relative_change": 0.0427993},
            "known_reported_B_failure_categories": ["effective input cosine below 0.99", "paired secant cosine below 0.95", "paired secant relative change above 0.25"]}


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"immutable metadata differs on resume: {path.name}")
        return
    path.write_text(text, encoding="utf-8")


def _write_gpu_samples(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for sample in sorted((root / "samples").iterdir()):
        path = sample / "sample.json"
        if not path.is_file(): continue
        record = _load_json(path)
        telemetry = record.get("telemetry", {})
        rows.append({"sample_id": sample.name, "elapsed_seconds": record.get("elapsed_seconds"), **{str(k): v for k, v in telemetry.items() if not isinstance(v, (dict, list))}})
    with (root / "gpu_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        keys = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def _write_manifest(root: Path) -> None:
    try:
        from umi_fd_post_vae_scan import sha256_file
    except ImportError:
        from .umi_fd_post_vae_scan import sha256_file
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in sorted(root.rglob("*"))
             if path.is_file() and path.name != "MANIFEST.sha256"]
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded Task 5 on the existing verified Cosmos UMI environment")
    parser.add_argument("--framework-root", default=DEFAULT_FRAMEWORK_ROOT)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--old-run09", default=DEFAULT_OLD_RUN)
    parser.add_argument("--task4-run", default=DEFAULT_TASK4_RUN)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir")
    parser.add_argument("--vae-path")
    parser.add_argument("--input-path")
    parser.add_argument("--action-path")
    parser.add_argument("--prompt")
    parser.add_argument("--direction-seed", type=int, default=20260912)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def execute_task5(args: argparse.Namespace) -> dict[str, Any]:
    required = ["framework_root", "checkpoint_path", "old_run09", "task4_run"]
    if not args.run_dir:
        required.append("run_root")
    missing = [name.replace("_", "-") for name in required if not getattr(args, name, None)]
    if missing:
        raise BlockedExecution("explicit required path arguments are missing: " + ", ".join(missing))
    run_dir = Path(args.run_dir).resolve() if args.run_dir else select_lowest_unused_run(args.run_root, prefix=RUN_PREFIX).resolve()
    task4_root = Path(args.task4_run).resolve()
    if run_dir.exists() and not args.resume: raise FileExistsError(f"refusing to reuse Task 5 run directory: {run_dir}")
    reference = load_task4_c_reference(task4_root)
    preflight = _task4_preflight(task4_root, reference)
    calibration_full = linear_formula_calibration()
    calibration = {"status": calibration_full["status"], "max_additivity_rms": calibration_full["max_additivity_rms"],
                   "max_prediction_rms": calibration_full["max_prediction_rms"]}
    if calibration["status"] != "PASS_EXACT":
        raise BlockedExecution("CPU exact-arithmetic formula calibration failed")
    old_provenance = _load_old_provenance(Path(args.old_run09))
    paths = _resolve_inputs(args, old_provenance)
    import torch
    require_torch_contract(torch)
    if paths["framework_root"] not in sys.path: sys.path.insert(0, paths["framework_root"])
    from umi_fd_post_vae_scan import _clone_runtime, load_official_data_batch, load_official_runtime
    from umi_precision_official import OfficialPrecisionRuntime, TorchOps, projection
    setup_dir = run_dir.parent / f".{run_dir.name}.framework_setup"
    legacy = _legacy_args(args, paths, setup_dir)
    adapter = load_official_runtime(legacy, setup_dir)
    backend = verify_uniform_eager_runtime(adapter)
    data_batch, _ = load_official_data_batch(adapter, legacy, setup_dir)
    ops = TorchOps()
    with ops.inference():
        prepared = adapter.model._prepare_inference_data(_clone_runtime(data_batch), [0], False)
    carrier = projection(prepared[1].x0_tokens_vision[0])
    mask = projection(prepared[6][0]).reshape(carrier.shape).astype(bool)
    bank, bank_meta = load_legacy_direction_bank(args.old_run09, carrier, mask)
    if not np.array_equal(bank[0], reference["direction"]):
        raise BlockedExecution("Task 4 v0 is not byte-identical to the frozen old bank direction 0")
    if not np.array_equal(mask, reference["mask"]):
        raise BlockedExecution("current runtime condition mask differs from Task 4 C")
    if not np.array_equal(quantize_bf16_fp32(carrier), reference["z_bar"]):
        raise BlockedExecution("current runtime z_bar differs from the Task 4 C baseline")
    def inputs_factory(current_carrier, indexes, current_mask, current_bank):
        return Task5Inputs(current_carrier, indexes, current_mask, current_bank, z_bar=reference["z_bar"], task4_reference=reference["metadata"])
    provenance = {"framework_root": paths["framework_root"], "checkpoint_path": paths["checkpoint_path"], "vae_path": paths["vae_path"],
                  "input_path": paths["input_path"], "action_path": paths["action_path"], "sampler": "unipc", "precision": "bfloat16",
                  "diffusion_cache_requested": False, "diffusion_cache_installed": False, "seed": 0, "prompt": legacy.prompt,
                  "direction_bank": bank_meta, "task4_run": str(task4_root), "model_compile": {"requested": False, "resolved": False, **backend}}
    runtime = OfficialPrecisionRuntime(adapter.model, data_batch, bank, provenance=provenance, ops=ops,
        generation_settings={"num_steps": 30, "guidance": 1.0, "shift": 10.0}, artifact_paths={"checkpoint": paths["checkpoint_path"], "decoder": paths["vae_path"]},
        inputs_factory=inputs_factory)
    run = run_task5_experiment(runtime, runtime.inputs, run_dir, resume=bool(args.resume), cpu_calibration=calibration)
    _write_json_once(run_dir / "task4_preflight.json", preflight)
    _write_json_once(run_dir / "cpu_formula_calibration.json", calibration)
    decoder = run_task5_decoder_replays(runtime, run_dir, resume=bool(args.resume)) if run.get("status") == "complete" else {"status": "NOT_RUN", "reason": "Task 5A incomplete"}
    _write_gpu_samples(run_dir)
    analysis_dir = run_dir / "task5_analysis"
    if run.get("status") == "complete" and decoder.get("status") == "complete":
        analysis = (json.loads((analysis_dir / "task5_summary.json").read_text(encoding="utf-8")) if args.resume and analysis_dir.exists()
                    else analyze_task5_run(run_dir, analysis_dir))
    else:
        analysis = {"status": "BLOCKED", "reason": "formal generation or decoder replays incomplete"}
    _write_manifest(run_dir)
    return {"run_dir": str(run_dir), "task4_preflight": preflight, "run": run, "decoder": decoder, "analysis": analysis}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = execute_task5(args)
    except BlockedExecution as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, ensure_ascii=False)); return 3
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result["analysis"].get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
