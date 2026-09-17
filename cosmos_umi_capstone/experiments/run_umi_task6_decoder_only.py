"""Generation-free Task 6 decoder recovery entry point.

This command derives a new decoder tree from a completed raw run.  It never
opens the raw tree for writing and never calls the generation/pilot path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    from .run_umi_task6_official import load_json, resource_samplers
    from .umi_task6_operational import (OfficialRuntimeFactory, OperationalEvidenceError,
        extract_task5_directions, verify_pinned_bridge_assets, _validate_static_contract)
    from .umi_task6_primitives import evaluate_resources
    from .run_umi_task6_experiment import ResourceMonitor
    from .umi_task6_decoder import _verify_raw_task6, run_task6_decoder_replays, sha256_file
    from .analyze_umi_task6 import analyze_task6_run
except ImportError:  # pragma: no cover
    from run_umi_task6_official import load_json, resource_samplers
    from umi_task6_operational import (OfficialRuntimeFactory, OperationalEvidenceError,
        extract_task5_directions, verify_pinned_bridge_assets, _validate_static_contract)
    from umi_task6_primitives import evaluate_resources
    from run_umi_task6_experiment import ResourceMonitor
    from umi_task6_decoder import _verify_raw_task6, run_task6_decoder_replays, sha256_file
    from analyze_umi_task6 import analyze_task6_run


def _raw_identity(root: Path) -> tuple[str, str]:
    manifest_sha, _ = _verify_raw_task6(root)
    return manifest_sha, sha256_file(root / "run_status.json")


def _expected_manifest(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    return sha256_file(path) if path.is_file() else value.strip()


def _stop(monitor: Any) -> None:
    if monitor is None:
        return
    stop = getattr(monitor, "stop", None)
    if callable(stop):
        stop()


def _write_control_result(root: Path, result: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_text(json.dumps(dict(result), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _base_result(status: str, *, raw_before: tuple[str, str], raw_after: tuple[str, str] | None = None,
                 **extra: Any) -> dict[str, Any]:
    raw_verification_error = extra.get("raw_verification_error")
    after = (None, None) if raw_verification_error else (raw_before if raw_after is None else raw_after)
    result = {"status": status, "generation_started": False, "raw_generation_calls": 0,
              "raw_manifest_sha256_before": raw_before[0], "raw_manifest_sha256_after": after[0],
              "raw_status_sha256_before": raw_before[1], "raw_status_sha256_after": after[1]}
    result.update(extra)
    return result


def execute_decoder_only(args: argparse.Namespace) -> dict[str, Any]:
    if bool(getattr(args, "resume", False)):
        raise OperationalEvidenceError("decoder-only recovery does not accept --resume")
    raw_root = Path(args.raw_root).resolve()
    decoder_root = Path(args.decoder_root).resolve()
    control_root = Path(args.control_root).resolve()
    analysis_root = Path(args.analysis_root).resolve()
    if decoder_root == raw_root or raw_root in decoder_root.parents:
        raise OperationalEvidenceError("decoder root must be outside immutable raw root")
    roots = {"raw": raw_root, "decoder": decoder_root, "control": control_root, "analysis": analysis_root}
    for left_name, left in roots.items():
        for right_name, right in roots.items():
            if left_name != right_name and (left == right or left in right.parents):
                raise OperationalEvidenceError(f"{left_name} root must not contain {right_name} root")
    if decoder_root.exists() and any(decoder_root.iterdir()):
        raise OperationalEvidenceError("decoder root must be a new empty directory")
    raw_before = _raw_identity(raw_root)
    expected = _expected_manifest(args.expected_raw_manifest)
    if expected is not None and expected != raw_before[0]:
        raise OperationalEvidenceError("raw manifest does not match --expected-raw-manifest")

    contract = load_json(args.launch_contract)
    _validate_static_contract(contract)
    assets = verify_pinned_bridge_assets(args.action, args.video)
    task5_spec = contract.get("task5", {})
    task5 = extract_task5_directions(
        args.task5_root,
        expected_manifest_sha256=task5_spec.get("manifest_sha256"),
        expected_plan_sha256=task5_spec.get("plan_sha256"),
        expected_direction_file_hashes=task5_spec.get("direction_file_sha256"),
        expected_direction_hashes=task5_spec.get("direction_sha256"),
    )
    factory = OfficialRuntimeFactory(
        loader=args.loader, framework_root=args.framework_root, checkpoint=args.checkpoint,
        vae=args.vae, contract=contract, direction_bank=task5["bank"], action=load_json(args.action),
        prompt=contract["prompt"], video=args.video, phase="decoder-only", run_dir=control_root, resume=False,
    )
    samplers = resource_samplers(control_root, gpu_index=args.gpu_index)
    load_monitor = ResourceMonitor(control_root, gpu_sampler=samplers["gpu"], ram_sampler=samplers["ram"],
                                   disk_sampler=samplers["disk"])
    runtime = None
    decoder_monitor = None
    try:
        load_monitor.start()
        preload = load_monitor.check(phase="preload", starting_new_sample=True, remaining_samples=16,
                                     run_dir=control_root)
        preload = dict(preload)
        if preload.get("status") != "HARD_STOP":
            snapshot = {**samplers["gpu"](), **samplers["ram"](), **samplers["disk"]()}
            preload = evaluate_resources(snapshot, phase="preload", starting_new_sample=True)
        if preload.get("status") == "HARD_STOP":
            after = _raw_identity(raw_root)
            result = _base_result("RESOURCE_STOP", raw_before=raw_before, raw_after=after,
                                  phase="DECODER_ONLY", reason_code=preload.get("reason_code"),
                                  reason=preload.get("reason"), contract_sha256=sha256_file(args.launch_contract),
                                  code_sha256=sha256_file(Path(__file__)))
            _write_control_result(control_root, result)
            return result
        built = factory.build()
        runtime = built[0]
        encoder = built[2] if len(built) > 2 else getattr(runtime, "encoder", None)
        loaded = load_monitor.check(phase="pilot", starting_new_sample=False, remaining_samples=16,
                                    run_dir=control_root)
        if loaded.get("status") == "HARD_STOP":
            after = _raw_identity(raw_root)
            result = _base_result("RESOURCE_STOP", raw_before=raw_before, raw_after=after,
                                  phase="DECODER_ONLY", reason_code=loaded.get("reason_code"),
                                  reason=loaded.get("reason"), contract_sha256=sha256_file(args.launch_contract),
                                  code_sha256=sha256_file(Path(__file__)))
            _write_control_result(control_root, result)
            return result
        _stop(load_monitor)
        load_monitor = None
        decoder_monitor = ResourceMonitor(control_root / "decoder_monitor", gpu_sampler=samplers["gpu"],
                                          ram_sampler=samplers["ram"], disk_sampler=samplers["disk"])
        decoder = run_task6_decoder_replays(runtime, raw_root, encoder=encoder, resume=False,
                                            decoder_root=decoder_root, monitor=decoder_monitor,
                                            main_status_path=None)
        if decoder.get("status") != "COMPLETE" or int(decoder.get("decoder_calls", 0)) != 16:
            after = _raw_identity(raw_root)
            evidence = dict(decoder)
            evidence.pop("status", None)
            result = _base_result(decoder.get("status", "BLOCKED"), raw_before=raw_before, raw_after=after,
                                  phase="DECODER_ONLY", **evidence)
            _write_control_result(control_root, result)
            return result
        analysis = analyze_task6_run(raw_root, decoder_root=decoder_root, output_root=analysis_root)
        after = _raw_identity(raw_root)
        if after != raw_before:
            raise OperationalEvidenceError("immutable raw manifest or status changed during recovery")
        result = _base_result("COMPLETE", raw_before=raw_before, raw_after=after, phase="DECODER_ONLY",
                              decoder_calls=16, analysis=analysis, contract_sha256=sha256_file(args.launch_contract),
                              code_sha256=sha256_file(Path(__file__)), assets=assets)
        _write_control_result(control_root, result)
        return result
    except BaseException as error:
        try:
            after = _raw_identity(raw_root)
            verify_error = None
        except BaseException as verify_exception:
            after = None
            verify_error = f"{type(verify_exception).__name__}: {verify_exception}"
        _write_control_result(control_root, _base_result("BLOCKED", raw_before=raw_before, raw_after=after,
                                                         phase="DECODER_ONLY", reason_code=type(error).__name__,
                                                         reason=str(error), contract_sha256=sha256_file(args.launch_contract),
                                                         code_sha256=sha256_file(Path(__file__),),
                                                         **({"raw_verification_error": verify_error} if verify_error else {})))
        raise
    finally:
        active_error = sys.exc_info()[1]
        finalization_errors = []
        for label, callback in (
            ("runtime.cleanup", getattr(runtime, "cleanup", None) if runtime is not None else None),
            ("decoder_monitor.stop", lambda: _stop(decoder_monitor)),
            ("load_monitor.stop", lambda: _stop(load_monitor)),
            ("factory.unload", getattr(factory, "unload", None)),
        ):
            if not callable(callback):
                continue
            try:
                callback()
            except BaseException as finalization_error:
                finalization_errors.append({"stage": label, "type": type(finalization_error).__name__,
                                            "message": str(finalization_error)})
        if finalization_errors:
            try:
                after = _raw_identity(raw_root)
                verify_error = None
            except BaseException as verify_exception:
                after = None
                verify_error = f"{type(verify_exception).__name__}: {verify_exception}"
            payload = _base_result("RESOURCE_STOP", raw_before=raw_before, raw_after=after,
                                   phase="DECODER_ONLY", reason_code="FINALIZATION_FAILURE",
                                   reason="decoder-only cleanup or unload failed",
                                   secondary_errors=finalization_errors,
                                   contract_sha256=sha256_file(args.launch_contract),
                                   code_sha256=sha256_file(Path(__file__)),
                                   **({"raw_verification_error": verify_error} if verify_error else {}))
            _write_control_result(control_root, payload)
            final_error = RuntimeError("decoder-only finalization failed")
            if active_error is not None:
                setattr(active_error, "secondary_errors", finalization_errors)
                raise final_error from active_error
            raise final_error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6 generation-free decoder recovery")
    parser.add_argument("--raw-root", required=True); parser.add_argument("--decoder-root", required=True)
    parser.add_argument("--analysis-root", required=True); parser.add_argument("--control-root", required=True)
    parser.add_argument("--launch-contract", required=True); parser.add_argument("--expected-raw-manifest", required=True)
    parser.add_argument("--framework-root", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--loader", default="umi_task6_cosmos_loader:load_task6_cosmos_runtime")
    parser.add_argument("--action", required=True); parser.add_argument("--video", required=True)
    parser.add_argument("--task5-root", required=True); parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        result = execute_decoder_only(parse_args(argv))
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if result.get("status") == "COMPLETE" else 2
    except (OperationalEvidenceError, ValueError, OSError) as error:
        print(json.dumps({"status": "BLOCKED", "generation_started": False, "reason": str(error)}))
        return 3
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "generation_started": False, "reason": str(error)}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
