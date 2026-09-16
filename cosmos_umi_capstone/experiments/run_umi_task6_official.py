"""Production Task 6 command path.

The command intentionally requires an approved observed launch contract and an
explicit installed-environment loader.  It has no model defaults and no test
dependency injection in its CLI path.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np

try:
    from .umi_task6_operational import (OfficialRuntimeFactory, OperationalEvidenceError, extract_task5_directions,
        prepare_bridge_upload_bundle, validate_launch_contract, verify_pinned_bridge_assets, observe_live_launch,
        _validate_static_contract)
    from .umi_task6_primitives import STATE_CATALOG, build_generation_plan, evaluate_resources
    from .umi_task6_runtime import Task6Inputs, build_task6_hash_binding, preflight_task6, task6_binding_config
    from .run_umi_task6_experiment import (_atomic_json, BlockedExecution, ResourceMonitor, ResourceStop,
        accept_resource_smoke, run_pilot, run_resource_smoke)
    from .umi_task6_decoder import _verify_raw_task6, run_task6_decoder_replays
    from .umi_fd_post_vae_bridge import sha256_array
except ImportError:  # pragma: no cover
    from umi_task6_operational import OfficialRuntimeFactory, OperationalEvidenceError, extract_task5_directions, prepare_bridge_upload_bundle, validate_launch_contract, verify_pinned_bridge_assets, observe_live_launch, _validate_static_contract
    from umi_task6_primitives import STATE_CATALOG, build_generation_plan, evaluate_resources
    from umi_task6_runtime import Task6Inputs, build_task6_hash_binding, preflight_task6, task6_binding_config
    from run_umi_task6_experiment import (_atomic_json, BlockedExecution, ResourceMonitor, ResourceStop,
        accept_resource_smoke, run_pilot, run_resource_smoke)
    from umi_task6_decoder import _verify_raw_task6, run_task6_decoder_replays
    from umi_fd_post_vae_bridge import sha256_array


def load_json(path: str | os.PathLike[str]) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationalEvidenceError(f"invalid JSON evidence: {path}") from error


def resource_samplers(run_dir: str | os.PathLike[str], *, gpu_index: int = 0):
    """Create fail-closed NVML, psutil and disk samplers for one process."""
    root = Path(run_dir)
    def gpu():
        try:
            import pynvml
            pynvml.nvmlInit(); handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index)); info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used, free = info.used / 2**30, info.free / 2**30
            try:
                import torch
                allocated, reserved = torch.cuda.memory_allocated(gpu_index) / 2**30, torch.cuda.memory_reserved(gpu_index) / 2**30
                peak_allocated = torch.cuda.max_memory_allocated(gpu_index) / 2**30
            except Exception:
                allocated = reserved = peak_allocated = 0.0
            return {"gpu_used_gib": used, "gpu_free_gib": free, "gpu_allocated_gib": allocated,
                    "gpu_reserved_gib": reserved, "gpu_peak_allocated_gib": peak_allocated,
                    "gpu_peak_nvml_used_gib": used}
        except Exception as error:
            return {"monitor_failure": f"NVML sampler failed: {error}"}
    def ram():
        try:
            import psutil
            memory, swap = psutil.virtual_memory(), psutil.swap_memory()
            return {"ram_available_gib": memory.available / 2**30, "rss_gib": psutil.Process(os.getpid()).memory_info().rss / 2**30,
                    "swap_used_gib": swap.used / 2**30}
        except Exception as error:
            return {"monitor_failure": f"RAM sampler failed: {error}"}
    def disk():
        try:
            return {"disk_free_gib": shutil.disk_usage(root).free / 2**30}
        except Exception as error:
            return {"monitor_failure": f"disk sampler failed: {error}"}
    return {"gpu": gpu, "ram": ram, "disk": disk}


def build_inputs(factory: OfficialRuntimeFactory, runtime: Any, *, asset: Mapping[str, Any], direction: Mapping[str, Any]):
    """Build Task6Inputs from the runtime's prepared official geometry."""
    official = getattr(runtime, "inputs", None)
    if official is None:
        raise OperationalEvidenceError("official runtime did not expose prepared inputs")
    geometry = official.geometry
    action = factory.action
    return Task6Inputs(official.z0, list(geometry.condition_indexes), geometry.mask, direction["bank"],
                       z_bar=official.z_bar, action=action, prompt=factory.prompt, state="bridge_0", seed=0,
                       direction_hashes={k: sha256_array(v) for k, v in official.directions.items()})


_TASK6_HASH_KEYS = frozenset(("code", "model", "config", "direction", "input", "noise"))


def _status_claims_raw_complete(status: Mapping[str, Any], expected_ids: set[str]) -> bool:
    """Validate the mutable status portion of a completed raw run."""
    if not isinstance(status, Mapping) or status.get("status") not in {"AWAITING_REVIEW", "COMPLETE"}:
        return False
    completed = status.get("completed_samples", [])
    failed = status.get("failed_samples", [])
    skipped = status.get("skipped_samples", [])
    try:
        completed_ids = set(completed) if isinstance(completed, list) else set()
        skipped_ids = set(skipped) if isinstance(skipped, list) else set()
    except TypeError:
        return False
    return (isinstance(completed, list) and len(completed) == 32 and len(completed_ids) == 32 and
            completed_ids == expected_ids and isinstance(failed, list) and not failed and
            isinstance(skipped, list) and len(skipped) == len(skipped_ids) and
            skipped_ids.issubset(expected_ids))


def _inspect_raw_complete(run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    """Return a hash/manifest-bound raw status for decoder-only resumes."""
    root = Path(run_dir)
    try:
        manifest_sha, status = _verify_raw_task6(root)
        expected = {item["sample_id"] for item in build_generation_plan(
            status.get("group", {}).get("state", "bridge_0"), int(status.get("group", {}).get("seed", 0)))}
    except Exception:
        return None
    hashes = status.get("hashes")
    if not _status_claims_raw_complete(status, expected) or not isinstance(hashes, Mapping) or set(hashes) != _TASK6_HASH_KEYS:
        return None
    if any(not isinstance(value, str) or not value for value in hashes.values()):
        return None
    result = dict(status)
    result["raw_manifest_sha256"] = manifest_sha
    return result


def _execute_official_impl(args: argparse.Namespace) -> dict[str, Any]:
    contract = load_json(args.launch_contract)
    expected_observed = load_json(args.observed_evidence) if args.observed_evidence else None
    # Static contract checks happen before any installed framework import.
    _validate_static_contract(contract)
    assets = verify_pinned_bridge_assets(args.action, args.video)
    task5 = extract_task5_directions(args.task5_root,
        expected_manifest_sha256=contract["task5"].get("manifest_sha256"),
        expected_plan_sha256=contract["task5"].get("plan_sha256"),
        expected_direction_file_hashes=contract["task5"].get("direction_file_sha256"),
        expected_direction_hashes=contract["task5"].get("direction_sha256"))
    factory = OfficialRuntimeFactory(loader=args.loader, framework_root=args.framework_root, checkpoint=args.checkpoint,
                                     vae=args.vae, contract=contract, direction_bank=task5["bank"], action=load_json(args.action),
                                     prompt=contract["prompt"], video=args.video, phase=args.phase, run_dir=args.run_dir,
                                     resume=bool(args.resume))
    if args.phase == "preflight":
        root = Path(args.run_dir); root.mkdir(parents=True, exist_ok=True)
        samplers = resource_samplers(args.run_dir, gpu_index=args.gpu_index)
        resource_sample = {"stage": "pre_load", **dict(samplers["gpu"]()), **dict(samplers["ram"]()), **dict(samplers["disk"]())}
        resource_decision = evaluate_resources(resource_sample, phase="preload")
        (root / "preflight_resource_samples.json").write_text(json.dumps({"samples": [resource_sample], "decision": resource_decision}, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        if resource_decision.get("status") == "HARD_STOP":
            raise BlockedExecution(f"preflight resource gate stopped before model load: {resource_decision.get('reason')}")
        runtime = inputs = None
        try:
            runtime, inputs, _ = factory.build()
            live = observe_live_launch(
                contract, runtime=runtime, inputs=inputs, assets=assets, task5=task5,
                checkpoint=args.checkpoint, vae=args.vae, framework_root=args.framework_root)
            validate_launch_contract(contract, observed=live)
            if expected_observed is not None and expected_observed != live:
                raise OperationalEvidenceError("observed-evidence JSON differs from independently observed environment")
            bank_path = root / "task5_direction_bank.npy"
            if bank_path.exists():
                raise OperationalEvidenceError("preflight destination already contains a direction bank")
            with tempfile.NamedTemporaryFile(prefix=".task5_direction_bank.", suffix=".npy", dir=root,
                                             delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                np.save(temporary_path, task5["bank"], allow_pickle=False)
                os.replace(temporary_path, bank_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            identity = inputs.identity()
            binding_config = task6_binding_config(inputs)
            strict_config = {"environment": {"torch": live["torch_version"], "cuda": live["cuda_version"]},
                "fps": live["fps"],
                "provenance": dict(getattr(runtime, "provenance", {})), "asset_hashes": {"action": assets["action_sha256"], "video": assets["video_sha256"],
                    "checkpoint": live["checkpoint_identity"]["sha256"], "vae": live["vae_sha256"]},
                "asset_paths": {"action": args.action, "video": args.video, "checkpoint": args.checkpoint, "vae": args.vae},
                "checkpoint_path": args.checkpoint, "vae_path": args.vae, "carrier_shape": list(inputs.z0.shape), "carrier_hash": identity["z0"],
                "condition_indexes": list(inputs.geometry.condition_indexes), "predicted_indexes": list(inputs.geometry.predicted_indexes),
                "mask_shape": list(inputs.geometry.mask.shape), "mask_hash": inputs.geometry.metadata()["mask_sha256"],
                "action": np.asarray(inputs.action).tolist(), "action_shape": list(inputs.action.shape), "action_hash": sha256_array(inputs.action),
                "prompt": inputs.prompt, "direction_hashes": dict(identity["directions"]), "direction_bank_path": str(bank_path),
                "direction_bank_file_hash": __import__("hashlib").sha256(bank_path.read_bytes()).hexdigest(), "settings": binding_config["settings"],
                "seed_config": {"seed": inputs.seed, "prepare": inputs.seed, "sampler": inputs.seed, "scheduler": inputs.seed},
                "observed_runtime_identity": runtime.actual_identity(), "observed_input_identity": inputs.identity()}
            preflight_task6(strict_config, strict=True, runtime=runtime, inputs=inputs)
            hashes = build_task6_hash_binding(runtime, inputs, binding_config)
            result = {"status": "PREFLIGHT_COMPLETE", "phase": "RESOURCE_SMOKE", "generation_started": False,
                      "group": dict(contract["group"]), "hashes": hashes, "asset_evidence": assets,
                      "task5_evidence": {k: v for k, v in task5.items() if k != "bank"}, "live_evidence": live,
                      "preflight_config": strict_config}
            (root / "run_status.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            return result
        finally:
            if runtime is not None and callable(getattr(runtime, "cleanup", None)):
                runtime.cleanup()
            factory.unload()
    if args.phase == "resource-smoke":
        samplers = resource_samplers(args.run_dir, gpu_index=args.gpu_index)
        state = {}
        def load():
            built = factory.build(); state["runtime"], state["inputs"] = built[0], built[1]; state["encoder"] = built[2] if len(built) > 2 else None
            return state["runtime"], state["inputs"]
        def cleanup():
            if state.get("runtime") is not None and callable(getattr(state["runtime"], "cleanup", None)): state["runtime"].cleanup()
        def unload():
            factory.unload()
            state.clear()
        return run_resource_smoke(args.run_dir, samplers=samplers,
                                  lifecycle={"pre_load": lambda: None, "load": load, "cleanup": cleanup, "unload": unload})
    if args.phase != "pilot": raise BlockedExecution("unknown official Task 6 phase")
    samplers = resource_samplers(args.run_dir, gpu_index=args.gpu_index)
    decoder_root = (Path(args.decoder_root) if args.decoder_root else
                    Path(args.run_dir).parent / (Path(args.run_dir).name + "_decoder")).resolve()
    status_path = Path(args.run_dir) / "run_status.json"
    try:
        prior_status = load_json(status_path) if status_path.is_file() else {}
    except OperationalEvidenceError:
        prior_status = {}
    acceptance_path = Path(args.run_dir) / "smoke_acceptance.json"
    if acceptance_path.is_file():
        try:
            prior_acceptance = load_json(acceptance_path)
        except OperationalEvidenceError:
            prior_acceptance = {}
        if isinstance(prior_acceptance, Mapping):
            # The status is authoritative for sample progress, while an
            # accepted-smoke sidecar is authoritative for its identity if a
            # previous interrupted writer did not copy those fields over.
            prior_status = dict(prior_status)
            for key in ("smoke_run_id", "hashes"):
                if key not in prior_status and key in prior_acceptance:
                    prior_status[key] = prior_acceptance[key]
    raw_binding = _inspect_raw_complete(args.run_dir) if bool(args.resume) else None
    raw_status = dict(raw_binding) if raw_binding is not None else None
    raw_manifest_sha = None if raw_binding is None else raw_binding.get("raw_manifest_sha256")
    if raw_status is not None:
        raw_status.pop("raw_manifest_sha256", None)
        raw_root = Path(args.run_dir).resolve()
        control_root = decoder_root.with_name(decoder_root.name + "_control").resolve()
        for candidate, label in ((decoder_root, "derived decoder"), (control_root, "derived control")):
            try:
                candidate.relative_to(raw_root)
            except ValueError:
                pass
            else:
                raise OperationalEvidenceError(f"{label} root must be outside the immutable raw run")
    else:
        control_root = None
    monitor_root = (control_root / "load_monitor") if raw_status is not None else Path(args.run_dir)
    monitor = ResourceMonitor(monitor_root, gpu_sampler=samplers["gpu"], ram_sampler=samplers["ram"], disk_sampler=samplers["disk"])
    monitor_started = False
    runtime = None

    def monitor_needs_close() -> bool:
        event = getattr(monitor, "_stop", None)
        if event is not None and callable(getattr(event, "is_set", None)) and event.is_set():
            return False
        return getattr(monitor, "_thread", None) is not None

    def close_monitor() -> list[dict[str, str]]:
        nonlocal monitor_started
        errors: list[dict[str, str]] = []
        if monitor_started or monitor_needs_close():
            try:
                monitor.stop()
            except BaseException as error:
                errors.append({"type": type(error).__name__, "message": str(error)})
        monitor_started = False
        return errors

    def resource_stop_payload(decision: Mapping[str, Any]) -> dict[str, Any]:
        base = dict(raw_status or prior_status or {})
        reason_code = decision.get("reason_code") or "RESOURCE_STOP"
        reason = decision.get("reason") or "preload resource gate stopped"
        if raw_status is not None:
            # A complete raw run is immutable.  Persist a derived control
            # record, while returning the original completion/status binding.
            payload = dict(base)
            payload.update({"resource_stop": True, "resource_stop_phase": "PILOT",
                            "resource_stop_reason_code": reason_code, "resource_stop_reason": reason})
            if raw_manifest_sha:
                payload["raw_manifest_sha256"] = raw_manifest_sha
            control_root.mkdir(parents=True, exist_ok=True)
            _atomic_json(control_root / "load_resource_stop.json", payload)
            return payload
        payload = dict(base)
        payload.update({"status": "RESOURCE_STOP", "phase": "PILOT", "generation_started": False,
                        "reason_code": reason_code, "reason": reason})
        _atomic_json(status_path, payload)
        return payload

    try:
        # Start telemetry and enforce the current preload limits before the
        # loader can import or construct any model object. The monitor stays
        # alive through the load and is reused by the generation runner.
        monitor.start(); monitor_started = True
        preload = monitor.check(phase="preload", starting_new_sample=True,
                                remaining_samples=32, run_dir=Path(args.run_dir))
        if preload.get("status") == "HARD_STOP":
            payload = resource_stop_payload(preload)
            secondary = close_monitor()
            if secondary:
                payload["secondary_errors"] = secondary
                if raw_status is None:
                    _atomic_json(status_path, payload)
                else:
                    _atomic_json(control_root / "load_resource_stop.json", payload)
            factory.unload()
            return payload
        factory_result = factory.build(); runtime, inputs = factory_result[0], factory_result[1]
        encoder = factory_result[2] if len(factory_result) > 2 else getattr(runtime, "encoder", None)
        generation = run_pilot(runtime, inputs, args.run_dir, resume=bool(args.resume), monitor=monitor)
        # run_pilot normally owns the monitor through its terminal status
        # write.  A derived raw resume still needs one final latched check
        # after model load and before any decoder call.
        monitor_started = False
        if raw_status is not None:
            post_load = monitor.check(phase="pilot", starting_new_sample=False,
                                      remaining_samples=0, run_dir=Path(args.run_dir))
            if post_load.get("status") == "HARD_STOP":
                payload = resource_stop_payload(post_load)
                secondary = close_monitor()
                if secondary:
                    payload["secondary_errors"] = secondary
                    _atomic_json(control_root / "load_resource_stop.json", payload)
                try: runtime.cleanup()
                finally: factory.unload()
                return payload
    except BaseException as error:
        if raw_status is not None:
            # A failed derived load/monitor close is control evidence only.
            # Never rewrite the completed raw status or place this marker in
            # the exact-inventory decoder tree.
            failure_payload = dict(raw_status)
            failure_payload.update({"resource_stop": True, "resource_stop_phase": "PILOT",
                                    "resource_stop_reason_code": type(error).__name__,
                                    "resource_stop_reason": str(error)})
            if raw_manifest_sha:
                failure_payload["raw_manifest_sha256"] = raw_manifest_sha
            try:
                control_root.mkdir(parents=True, exist_ok=True)
                _atomic_json(control_root / "load_resource_stop.json", failure_payload)
            except BaseException:
                # Preserve the original fail-closed exception if control
                # evidence itself cannot be persisted.
                pass
        if monitor_started:
            try: monitor.stop()
            except BaseException: pass
            monitor_started = False
        try:
            if runtime is not None: runtime.cleanup()
        finally: factory.unload()
        raise
    if generation.get("status") not in {"AWAITING_REVIEW", "COMPLETE"}:
        try: runtime.cleanup()
        finally: factory.unload()
        return generation
    if encoder is None:
        try: runtime.cleanup()
        finally: factory.unload()
        raise OperationalEvidenceError("official loader must expose a condition encoder for decoder replay")
    # Decoder telemetry is derived evidence and must not mutate the raw
    # generation monitor files or its immutable manifest.
    decoder_monitor = ResourceMonitor(decoder_root, gpu_sampler=samplers["gpu"], ram_sampler=samplers["ram"], disk_sampler=samplers["disk"])
    decoder = run_task6_decoder_replays(runtime, args.run_dir, encoder=encoder, resume=bool(args.resume), decoder_root=decoder_root,
                                        monitor=decoder_monitor, main_status_path=Path(args.run_dir) / "run_status.json")
    if decoder.get("status") != "COMPLETE":
        try: runtime.cleanup()
        finally: factory.unload()
        return decoder
    # Analysis and packaging are intentionally called by the validated public
    # API only after both raw and decoder evidence are complete.
    try:
        from .analyze_umi_task6 import analyze_task6_run
    except ImportError:  # pragma: no cover
        from analyze_umi_task6 import analyze_task6_run
    try:
        return analyze_task6_run(args.run_dir, decoder_root=decoder_root, output_root=args.analysis_root)
    finally:
        try: runtime.cleanup()
        finally: factory.unload()


def execute_official(args: argparse.Namespace) -> dict[str, Any]:
    """Run one official phase and leave a canonical terminal status on failure."""
    try:
        return _execute_official_impl(args)
    except BaseException as error:
        root = Path(args.run_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        existing = root / "run_status.json"
        try:
            prior = load_json(existing) if existing.is_file() else {}
        except Exception:
            prior = {}
        # Never replace a completed status with a later setup error.  A failed
        # preflight/smoke/pilot gets an explicit canonical status instead.
        completed = prior.get("completed_samples", [])
        try:
            completed_ids = set(completed) if isinstance(completed, list) else set()
        except TypeError:
            completed_ids = set()
        raw_complete = (prior.get("status") in {"AWAITING_REVIEW", "COMPLETE"} and
                        isinstance(completed, list) and len(completed) == 32 and len(completed_ids) == 32)
        preserve = raw_complete or (
            str(getattr(args, "phase", "unknown")) == "preflight" and
            prior.get("status") == "PREFLIGHT_COMPLETE")
        if not preserve:
            text = str(error).lower()
            status = "RESOURCE_STOP" if any(token in text for token in ("resource", "oom", "cuda out of memory", "monitor")) else "BLOCKED"
            payload = {"status": status, "phase": str(getattr(args, "phase", "unknown")),
                       "generation_started": False, "reason_code": type(error).__name__, "reason": str(error)}
            temporary = root / ".run_status.json.tmp"
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, existing)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Official Task 6 operational driver")
    parser.add_argument("--phase", choices=("preflight", "resource-smoke", "pilot"), required=True)
    parser.add_argument("--run-dir", required=True); parser.add_argument("--decoder-root"); parser.add_argument("--analysis-root")
    parser.add_argument("--launch-contract", required=True); parser.add_argument("--observed-evidence", help="optional expected snapshot; never trusted as live evidence")
    parser.add_argument("--framework-root", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--vae", required=True)
    parser.add_argument("--loader", default="umi_task6_cosmos_loader:load_task6_cosmos_runtime", help="installed official adapter module:function")
    parser.add_argument("--action", required=True); parser.add_argument("--video", required=True); parser.add_argument("--task5-root", required=True)
    parser.add_argument("--gpu-index", type=int, default=0); parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    try:
        result = execute_official(parse_args(argv)); print(json.dumps(result, indent=2, default=str)); return 0 if result.get("status") in {"PASS", "PREFLIGHT_COMPLETE", "AWAITING_RESOURCE_REVIEW", "AWAITING_REVIEW", "COMPLETE"} else 2
    except (OperationalEvidenceError, BlockedExecution, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)})); return 3


if __name__ == "__main__": raise SystemExit(main())
