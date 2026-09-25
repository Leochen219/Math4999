"""Stage-gated six-scene, two-seed Bridge feedback experiment.

Preflight never loads a model. Smoke and formal stages require --release.
Only one Task 8 model context exists at a time; every sample is published by
the verified atomic Task8SampleStore, and no completed sample is overwritten.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from umi_task10_real_scenes import (LOCKED_RECORDS, SEED_PAIRS, build_call_plan,
                                    forecast_disk_after_calls, locked_preflight_ranks,
                                    select_locked_records)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    from task8_frozen.task8_formal import _atomic_json
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)


def validate_preflight_report(report: Mapping[str, Any], record_index: int) -> None:
    if report.get("status") != "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED":
        raise ValueError(f"record {record_index} preflight did not complete")
    if report.get("record_count") != 52 or report.get("manifest_sha256_match") is not True:
        raise ValueError(f"record {record_index} dataset count/hash gate failed")
    selected = report.get("selected") or {}
    if selected.get("record_index") != record_index:
        raise ValueError(f"record {record_index} was substituted during preflight")
    input_info = selected.get("inputs_npz") or {}
    if (input_info.get("actions_normalized_shape") != [2, 16, 10]
            or input_info.get("rgb_float32_shape") != [33, 3, 256, 256]
            or not isinstance(input_info.get("sha256"), str)):
        raise ValueError(f"record {record_index} action/frame interface failed")
    camera = report.get("camera") or {}
    if (camera.get("feature_key") != "steps/observation/image_0"
            or camera.get("all_window_frames_rgb_256x256") is not True
            or camera.get("all_window_frames_nonconstant") is not True
            or int(camera.get("unique_encoded_frames", 0)) <= 1):
        raise ValueError(f"record {record_index} camera proof failed")
    if not (selected.get("language") or "").strip():
        raise ValueError(f"record {record_index} language is empty")
    parity = report.get("official_parity_artifact")
    if (not isinstance(parity, Mapping) or parity.get("status") != "PASS"
            or parity.get("action_exact_equal") is not True
            or parity.get("initial_pose_exact_equal") is not True
            or (parity.get("normalization") or {}).get("saved_exact_equal") is not True):
        raise ValueError(f"record {record_index} official action parity artifact missing")
    temporal = report.get("temporal_alignment") or {}
    if any(temporal.get(key) is not True for key in
           ("sequence_index_order_proven", "frequency_continuity_proven_from_official_source",
            "flags_prove_first_last_boundaries")):
        raise ValueError(f"record {record_index} temporal alignment proof failed")


def validate_run_identity(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if dict(saved) != dict(expected):
        raise ValueError("run identity mismatch; resume would mix source, data, or seeds")


def _source_identity(args: argparse.Namespace) -> dict[str, Any]:
    from task8_frozen.task8_formal import _framework_commit, _task8_source_paths, _sha256_path
    from umi_task9_bridge_preflight import _verify_manifest
    source_paths = _task8_source_paths() + [
        Path(__file__).resolve(),
        Path(__file__).with_name("umi_task10_real_scenes.py"),
        Path(__file__).with_name("umi_task9_bridge_preflight.py"),
        Path(__file__).with_name("analyze_umi_task10_real_scenes.py"),
    ]
    source_hashes = {path.name: _sha256_file(path) for path in source_paths}
    manifest = _verify_manifest(Path(args.dataset_root))
    if not all(row.get("match") for row in manifest["files"].values()):
        raise ValueError("dataset manifest SHA256 gate failed")
    return {"schema": "umi-task10-v1", "dataset_root": str(Path(args.dataset_root).resolve()),
            "dataset_sha256": manifest["files"], "code_sha256": source_hashes,
            "framework_commit": _framework_commit(args.framework_root),
            "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": _sha256_path(args.checkpoint),
            "vae": str(Path(args.vae).resolve()),
            "vae_sha256": _sha256_file(args.vae), "record_indices": list(LOCKED_RECORDS),
            "normalization_sha256": _sha256_file(Path(args.framework_root) / "cosmos_framework/data/generator/action/normalizer_stats/bridge_orig_lerobot_stats.json"),
            "official_parity_sha256": _sha256_file(args.official_parity_report),
            "seed_pairs": [list(pair) for pair in SEED_PAIRS], "fps": 5.0,
            "sampler": "UniPC", "steps": 30, "guidance": 1.0, "shift": 10.0,
            "precision": "FP32 G/D/E", "cache": False, "autocast": False, "tf32": False}


def _lock_identity(root: Path, expected: Mapping[str, Any], *, resume: bool) -> None:
    path = root / "run_identity.json"
    if path.exists():
        if not resume:
            raise FileExistsError("run already exists; use --resume only for the same identity")
        validate_run_identity(json.loads(path.read_text(encoding="utf-8")), expected)
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("nonempty run directory lacks run identity")
        root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(path, expected)


def _report_paths(root: Path, record_index: int) -> tuple[Path, Path]:
    folder = root / "preflight" / f"record_{record_index:02d}"
    return folder / "task8_preflight.json", folder / "inputs.npz"


def _load_preflight(root: Path, record_index: int) -> tuple[dict[str, Any], Path]:
    report_path, npz_path = _report_paths(root, record_index)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_preflight_report(report, record_index)
    if _sha256_file(npz_path) != report["selected"]["inputs_npz"]["sha256"]:
        raise ValueError(f"record {record_index} preflight NPZ hash mismatch")
    return report, npz_path


def preflight_all(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    from umi_task9_bridge_preflight import run_preflight
    existing = [(_report_paths(root, index)[0]).exists() for index in LOCKED_RECORDS]
    if any(existing) and not all(existing):
        if not args.resume:
            raise ValueError("partial preflight requires --resume")
    reports = []
    for index, rank in zip(LOCKED_RECORDS, locked_preflight_ranks()):
        report_path, _ = _report_paths(root, index)
        if report_path.exists():
            report, _ = _load_preflight(root, index)
        else:
            report = run_preflight(args.dataset_root, report_path.parent,
                                   normalizer_stats_path=Path(args.framework_root) / "cosmos_framework/data/generator/action/normalizer_stats/bridge_orig_lerobot_stats.json",
                                   official_parity_report=args.official_parity_report,
                                   eligible_rank=rank)
            validate_preflight_report(report, index)
            _load_preflight(root, index)
        reports.append(report)
    select_locked_records(reports[0]["records"])
    return reports


def _completed_sample_count(root: Path) -> int:
    count = 0
    for index in LOCKED_RECORDS:
        for pair in SEED_PAIRS:
            folder = root / "records" / f"record_{index:02d}" / f"seeds_{pair[0]}_{pair[1]}" / "formal" / "samples"
            for call in ("G0", "G0_repeat", "TF2", "AR2"):
                if (folder / call / "status.json").is_file():
                    count += 1
    return count


def run_smoke(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    if not args.release:
        raise ValueError("model smoke requires --release")
    from task8_frozen.task8_live import main as task8_smoke_main
    from task8_frozen.task8_formal import validate_smoke_run
    report_path, npz_path = _report_paths(root, LOCKED_RECORDS[0])
    _load_preflight(root, LOCKED_RECORDS[0])
    smoke_dir = root / "smoke"
    if not (smoke_dir / "run_status.json").is_file():
        invocation = ["--inputs-npz", str(npz_path), "--metadata-json", str(report_path),
                      "--run-dir", str(smoke_dir), "--framework-root", args.framework_root,
                      "--checkpoint", args.checkpoint, "--vae", args.vae, "--release"]
        if task8_smoke_main(invocation) != 0:
            raise RuntimeError("engineering smoke failed")
    smoke = validate_smoke_run(smoke_dir)
    import shutil
    free_gib = shutil.disk_usage(root).free / 2**30
    forecast = forecast_disk_after_calls(free_gib, int(smoke["measured_sample_bytes"]),
                                         48 - _completed_sample_count(root))
    result = {"status": "SMOKE_ACCEPTED", "measured_sample_bytes": smoke["measured_sample_bytes"],
              "disk_free_gib": free_gib, "forecast_after_formal_gib": forecast}
    _write_json_atomic(root / "smoke_gate.json", result)
    return result


def _formal_executor(executor: Any, spec: Mapping[str, Any]) -> Mapping[str, Any]:
    # On resume G0 may be skipped; refresh the live adapter from the saved
    # authoritative decoded frame rather than its potentially stale state.
    if spec.get("call") == "AR2":
        frame = spec.get("condition_rgb")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.float32:
            raise ValueError("AR2 lacks saved G0 FP32 decoded feedback frame")
        executor._last_g0_frame = np.array(frame, copy=True)
    return executor(spec)


def _verified_truth_conditions(context: Any, batch: Any, record_root: Path, *,
                               record_index: int, input_sha256: str, vae_sha256: str) -> Path:
    """Publish GT encodes only after both arrays and their evidence are complete."""
    from task8_frozen.task8_formal import _save_gt_conditions
    import shutil
    destination = record_root / "ground_truth_conditions.npz"
    evidence = record_root / "ground_truth_condition_evidence.json"
    receipt = record_root / "ground_truth_conditions.sha256.json"
    if receipt.exists():
        bound = json.loads(receipt.read_text(encoding="utf-8"))
        if (not destination.is_file() or not evidence.is_file()
                or _sha256_file(destination) != bound.get("npz_sha256")
                or _sha256_file(evidence) != bound.get("evidence_sha256")
                or bound.get("record_index") != record_index
                or bound.get("input_sha256") != input_sha256
                or bound.get("vae_sha256") != vae_sha256):
            raise ValueError("ground truth condition receipt/hash mismatch")
        with np.load(destination, allow_pickle=False) as saved:
            if not {"gt_condition_x16", "gt_condition_x32", "condition_mask"} <= set(saved.files):
                raise ValueError("ground truth conditions are incomplete")
        return destination
    if destination.exists() or evidence.exists():
        raise ValueError("unreceipted ground truth condition artifact; preserve and inspect before resume")
    attempt = record_root / "gt_attempts" / "attempt_001"
    ordinal = 1
    while attempt.exists():
        ordinal += 1
        attempt = record_root / "gt_attempts" / f"attempt_{ordinal:03d}"
    attempt.mkdir(parents=True)
    _save_gt_conditions(context, batch, attempt)
    staged_npz = attempt / destination.name
    staged_evidence = attempt / evidence.name
    with np.load(staged_npz, allow_pickle=False) as saved:
        if saved["gt_condition_x16"].dtype != np.float32 or saved["gt_condition_x32"].dtype != np.float32:
            raise ValueError("ground truth conditions are not float32")
    shutil.copy2(staged_npz, destination.with_suffix(".npz.tmp"))
    shutil.copy2(staged_evidence, evidence.with_suffix(".json.tmp"))
    os.replace(destination.with_suffix(".npz.tmp"), destination)
    os.replace(evidence.with_suffix(".json.tmp"), evidence)
    _write_json_atomic(receipt, {"npz_sha256": _sha256_file(destination),
                                "evidence_sha256": _sha256_file(evidence),
                                "record_index": record_index, "input_sha256": input_sha256,
                                "vae_sha256": vae_sha256})
    return destination


def _completed_pair(formal_dir: Path, plan: Sequence[Mapping[str, Any]], binding: Mapping[str, str]) -> bool:
    status_path = formal_dir / "run_status.json"
    if not status_path.is_file():
        return False
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("binding") != dict(binding) or status.get("plan") != [dict(row) for row in plan]:
        raise ValueError("completed pair identity differs from current binding or plan")
    if status.get("status") != "COMPLETE":
        return False
    from task8_frozen.run_umi_task8_experiment import Task8SampleStore
    store = Task8SampleStore(formal_dir / "samples")
    for row in plan:
        record = store.load_record(str(row["sample_id"]))
        if record.get("call") != row["call"]:
            raise ValueError("completed sample differs from locked call plan")
        del record
    return True


def _matrix_complete(root: Path) -> bool:
    """Never promote a sample count to a complete scientific matrix."""
    identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
    for index in LOCKED_RECORDS:
        _, input_path = _load_preflight(root, index)
        for pair in SEED_PAIRS:
            formal = root / "records" / f"record_{index:02d}" / f"seeds_{pair[0]}_{pair[1]}" / "formal"
            status_path = formal / "run_status.json"
            if not status_path.is_file():
                return False
            status = json.loads(status_path.read_text(encoding="utf-8"))
            binding = status.get("binding") or {}
            if (binding.get("model") != identity["checkpoint_sha256"]
                    or binding.get("vae") != identity["vae_sha256"]
                    or binding.get("data") != _sha256_file(input_path)):
                raise ValueError(f"matrix pair {index}/{pair} has mismatched model/VAE/input binding")
            if not _completed_pair(formal, build_call_plan(*pair), binding):
                return False
    return True


def run_formal(args: argparse.Namespace, root: Path) -> None:
    if not args.release:
        raise ValueError("formal model execution requires --release")
    from task8_frozen.run_umi_task8_experiment import ResourceStop, evaluate_task8_resources, run_task8
    from task8_frozen.task8_formal import (_framework_commit, _task8_source_paths,
                                          build_binding, validate_smoke_run)
    from task8_frozen.task8_live import Task8LiveExecutor, Task8ResourceMonitor, load_task8_live
    from task8_frozen.umi_task8_runtime import Task8InputAdapter, Task8RuntimeConfig
    smoke = validate_smoke_run(root / "smoke")
    if args.record_index not in LOCKED_RECORDS:
        raise ValueError("formal stage requires one preregistered --record-index per process")
    measured = int(smoke["measured_sample_bytes"])
    run_identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
    code_paths = _task8_source_paths() + [Path(__file__).resolve(),
                                           Path(__file__).with_name("umi_task10_real_scenes.py"),
                                           Path(__file__).with_name("umi_task9_bridge_preflight.py")]
    framework_commit = _framework_commit(args.framework_root)
    successful_sizes: list[int] = []
    for index in (args.record_index,):
        report, input_path = _load_preflight(root, index)
        batch = Task8InputAdapter.from_preflight(input_path, _report_paths(root, index)[0])
        record_root = root / "records" / f"record_{index:02d}"
        record_root.mkdir(parents=True, exist_ok=True)
        bindings = {}
        for pair in SEED_PAIRS:
            bindings[pair] = build_binding(code_paths=code_paths, model_path=args.checkpoint,
                                           vae_path=args.vae, input_path=input_path, actions=batch.actions,
                                           metadata=report, framework_commit=framework_commit,
                                           runtime_config=Task8RuntimeConfig().as_dict(), seed_pair=pair)
            if bindings[pair]["model"] != run_identity["checkpoint_sha256"]:
                raise ValueError("model changed after run identity was locked")
        pending = [pair for pair in SEED_PAIRS if not _completed_pair(
            record_root / f"seeds_{pair[0]}_{pair[1]}" / "formal", build_call_plan(*pair), bindings[pair])]
        if not pending:
            del batch
            continue
        monitor_base = record_root / "monitor_attempts"
        ordinal = 1
        while (monitor_base / f"attempt_{ordinal:03d}").exists():
            ordinal += 1
        monitor_root = monitor_base / f"attempt_{ordinal:03d}"
        monitor_root.mkdir(parents=True)
        monitor = Task8ResourceMonitor(monitor_root, gpu_index=0)
        context = None
        executor = None
        cleanup_error = None
        try:
            monitor.start()
            before = monitor.check(phase="preload", starting_new_sample=True)
            evaluate_task8_resources(before, phase="preload", starting_new_sample=True,
                                     remaining_samples=48 - _completed_sample_count(root),
                                     mean_success_sample_bytes=measured)
            context = load_task8_live(framework_root=args.framework_root, checkpoint=args.checkpoint,
                                      vae=args.vae, run_dir=record_root / "setup", batch=batch,
                                      release=True, resume=args.resume)
            _verified_truth_conditions(context, batch, record_root, record_index=index,
                                       input_sha256=_sha256_file(input_path),
                                       vae_sha256=run_identity["vae_sha256"])
            for pair in pending:
                pair_root = record_root / f"seeds_{pair[0]}_{pair[1]}"
                formal_dir = pair_root / "formal"
                plan = build_call_plan(*pair)
                binding = bindings[pair]
                executor = Task8LiveExecutor(context, resource_monitor=monitor, resource_phase="formal")
                def after_sample(sample_id: str, remaining_in_pair: int) -> None:
                    del remaining_in_pair
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
                    remaining = 48 - _completed_sample_count(root)
                    capture = getattr(monitor._monitor, "capture_sample", None)
                    if not callable(capture):
                        raise ResourceStop("MONITOR_FAILURE: post-cleanup capture is unavailable")
                    row = capture(sample_id, "post_cleanup", remaining, run_dir=formal_dir)
                    if row.get("decision_status") == "HARD_STOP":
                        raise ResourceStop(f"{row.get('reason_code')}: {row.get('reason')}")
                    actual = row.get("mean_success_sample_bytes")
                    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or actual <= 0:
                        raise ResourceStop("DISK_FORECAST_UNAVAILABLE: sample size not measured")
                    successful_sizes.append(int(actual))
                    gate = evaluate_task8_resources(row, phase="formal", starting_new_sample=False,
                                                    remaining_samples=remaining,
                                                    mean_success_sample_bytes=int(actual))
                    _write_json_atomic(monitor_root / "post_cleanup_latest.json",
                                       {"sample_id": sample_id, "resource_row": row, "gate": gate})
                    monitor.check(phase="formal", starting_new_sample=False)
                snapshot = monitor.check(phase="formal", starting_new_sample=True)
                evaluate_task8_resources(snapshot, phase="formal", starting_new_sample=True,
                                         remaining_samples=48 - _completed_sample_count(root),
                                         mean_success_sample_bytes=successful_sizes[-1] if successful_sizes else measured)
                result = run_task8(formal_dir, execute_call=lambda spec: _formal_executor(executor, spec),
                                   release=True, resume=args.resume, binding=binding,
                                   resource_monitor=monitor, input_batch=batch,
                                   post_sample_callback=after_sample, plan=plan, allow_custom_plan=True)
                if result["status"] != "COMPLETE":
                    raise RuntimeError(f"record {index} seeds {pair} did not complete")
                del executor, result
                executor = None
                gc.collect()
        except BaseException as error:
            _write_json_atomic(root / "run_status.json", {"status": "RESOURCE_STOP" if isinstance(error, ResourceStop) else "FAILED",
                                                       "record_index": index, "error": repr(error),
                                                       "completed_samples": _completed_sample_count(root)})
            raise
        finally:
            if context is not None:
                try:
                    context.cleanup()
                except BaseException as error:
                    cleanup_error = error
            try:
                monitor.stop()
            except BaseException as error:
                cleanup_error = cleanup_error or error
            executor = None
            context = None
            batch = None
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            if cleanup_error is not None:
                _write_json_atomic(root / "run_status.json", {"status": "CLEANUP_FAILED", "record_index": index,
                                                            "error": repr(cleanup_error),
                                                            "completed_samples": _completed_sample_count(root)})
                raise cleanup_error
    completed = _completed_sample_count(root)
    matrix_complete = _matrix_complete(root)
    _write_json_atomic(root / "run_status.json", {"status": "GENERATION_COMPLETE" if matrix_complete else "PARTIAL_GENERATION",
                                                "completed_samples": completed,
                                                "record_indices": list(LOCKED_RECORDS),
                                                "seed_pairs": [list(pair) for pair in SEED_PAIRS]})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "formal"), required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--framework-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--official-parity-report", required=True)
    parser.add_argument("--record-index", type=int, choices=LOCKED_RECORDS,
                        help="formal only: one episode/model lifetime per OS process")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--release", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage != "preflight" and not args.release:
        raise ValueError("generation stage requires explicit --release")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if os.environ["CUDA_VISIBLE_DEVICES"] != "0" or os.environ["HF_HUB_OFFLINE"] != "1":
        raise ValueError("GPU/offline environment differs from frozen policy")
    root = Path(args.run_root).resolve()
    identity = _source_identity(args)
    _lock_identity(root, identity, resume=args.resume)
    if args.stage == "preflight":
        preflight_all(args, root)
    elif args.stage == "smoke":
        run_smoke(args, root)
    else:
        for index in LOCKED_RECORDS:
            _load_preflight(root, index)
        run_formal(args, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
