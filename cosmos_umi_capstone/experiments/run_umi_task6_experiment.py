"""Launcher, resource smoke, monitors and pilot state machine for Task 6.

All model/framework work is deliberately injected.  The default command is a
generation-free preflight and no phase silently advances to a later phase.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from .umi_task6_primitives import build_run_status, evaluate_resources
    from .umi_precision_storage import PrecisionSampleStore
    from .umi_task6_runtime import (PreflightError, ResourceStop, Task6Inputs, build_task6_hash_binding, task6_binding_config, preflight_task6, _run_task6_group,
                                    verify_reference_reuse)
except ImportError:
    from umi_task6_primitives import build_run_status, evaluate_resources
    from umi_precision_storage import PrecisionSampleStore
    from umi_task6_runtime import PreflightError, ResourceStop, Task6Inputs, build_task6_hash_binding, task6_binding_config, preflight_task6, _run_task6_group, verify_reference_reuse


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    stage = path.with_name("." + path.name + ".stage")
    stage.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(stage, path)


def _atomic_text(path: Path, value: str) -> None:
    stage = path.with_name("." + path.name + ".stage")
    stage.write_text(value, encoding="utf-8")
    os.replace(stage, path)


class AtomicSampleStore:
    """Independent temp directory publication with immutable successful samples."""
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, sample_id: str, values: Mapping[str, Any], metadata: Mapping[str, Any] | None = None) -> Path:
        if not sample_id or any(part in sample_id for part in ("/", "\\", "..")): raise ValueError("unsafe sample id")
        destination = self.root / sample_id
        if destination.exists(): raise FileExistsError(destination)
        stage = Path(tempfile.mkdtemp(prefix=f".{sample_id}.", dir=self.root))
        try:
            encoded: dict[str, Any] = {}
            for key, value in values.items():
                safe = str(key).replace("/", "_").replace("\\", "_")
                if isinstance(value, np.ndarray):
                    filename = safe + ".npy"; np.save(stage / filename, value, allow_pickle=False); encoded[str(key)] = {"artifact": filename, "shape": list(value.shape), "dtype": str(value.dtype)}
                elif isinstance(value, (str, int, float, bool)) or value is None:
                    encoded[str(key)] = value
                else: encoded[str(key)] = value
            (stage / "sample.json").write_text(json.dumps({"metadata": dict(metadata or {}), "values": encoded}, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
            hashes = {path.name: _sha_file(path) for path in sorted(stage.iterdir()) if path.is_file()}
            (stage / "MANIFEST.sha256").write_text("\n".join(f"{value}  {key}" for key, value in sorted(hashes.items())) + "\n", encoding="ascii")
            hashes["MANIFEST.sha256"] = _sha_file(stage / "MANIFEST.sha256")
            (stage / "status.json").write_text(json.dumps({"status": "success", "artifact_sha256": hashes}, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(stage, destination)
            return destination
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    def prepare(self, sample_id: str, *, resume: bool = False) -> str:
        destination = self.root / sample_id
        if destination.exists():
            status = destination / "status.json"
            state = json.loads(status.read_text(encoding="utf-8")) if status.is_file() else {}
            if state.get("status") == "success":
                if not resume: raise FileExistsError(destination)
                return "skip"
            if not resume: raise FileExistsError(destination)
            attempt = self.root / (sample_id + ".attempt.1")
            index = 1
            while attempt.exists():
                index += 1; attempt = self.root / f"{sample_id}.attempt.{index}"
            destination.rename(attempt)
        return "run"


def monitor_decision(snapshot: Mapping[str, Any], *, phase: str = "pilot", starting_new_sample: bool = False) -> dict[str, Any]:
    return evaluate_resources(snapshot, phase=phase, starting_new_sample=starting_new_sample)


class ResourceMonitor:
    """Background polling monitor with injectable clocks and samplers."""
    def __init__(self, root: str | Path, *, gpu_sampler: Callable[[], Mapping[str, Any]] | None = None,
                 ram_sampler: Callable[[], Mapping[str, Any]] | None = None,
                 disk_sampler: Callable[[], Mapping[str, Any]] | None = None, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if not all(callable(value) for value in (gpu_sampler, ram_sampler, disk_sampler)):
            raise ValueError("ResourceMonitor requires GPU, RAM, and disk samplers")
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.gpu_sampler = gpu_sampler
        self.ram_sampler = ram_sampler
        self.disk_sampler = disk_sampler
        self.cadence_seconds = {"gpu": 1.0, "ram": 5.0, "disk": 5.0}
        self.clock = clock; self.sleep = sleep; self._stop = threading.Event(); self.failure: Exception | None = None
        self._thread: threading.Thread | None = None; self._gpu: list[dict[str, Any]] = []; self._ram: list[dict[str, Any]] = []; self._disk: list[dict[str, Any]] = []
        self._sample_rows: list[dict[str, Any]] = []
        self._cleanup_gpu_baseline: float | None = None; self._cleanup_rss_baseline: float | None = None
        self._cleanup_gpu_consecutive = 0; self._cleanup_ram_consecutive = 0

    def _poll(self):
        next_gpu = self.clock(); next_ram = next_gpu
        while not self._stop.is_set():
            now = self.clock()
            try:
                if now >= next_gpu:
                    self._gpu.append({"timestamp": now, **dict(self.gpu_sampler())}); next_gpu += 1.0
                if now >= next_ram:
                    self._ram.append({"timestamp": now, **dict(self.ram_sampler())}); self._disk.append({"timestamp": now, **dict(self.disk_sampler())}); next_ram += 5.0
            except Exception as error:
                self.failure = error; self._stop.set(); break
            self.sleep(min(max(0.0, next_gpu - now), 0.1))

    def start(self):
        if self._thread is not None: raise RuntimeError("monitor already started")
        # Establish a synchronous baseline so a fast pilot cannot start a
        # sample before the background cadence has produced evidence.
        try:
            now = self.clock()
            self._gpu.append({"timestamp": now, **dict(self.gpu_sampler())})
            self._ram.append({"timestamp": now, **dict(self.ram_sampler())})
            self._disk.append({"timestamp": now, **dict(self.disk_sampler())})
        except Exception as error:
            self.failure = error
        self._thread = threading.Thread(target=self._poll, name="task6-resource-monitor", daemon=True); self._thread.start(); return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            if self._thread.is_alive(): self.failure = RuntimeError("resource monitor did not terminate")
        self._write("gpu_samples.csv", self._gpu); self._write("ram_samples.csv", self._ram); self._write("disk_samples.csv", self._disk)
        self._write("sample_resource_snapshots.csv", self._sample_rows)
        _atomic_text(self.root / "sample_resource_snapshots.jsonl", "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in self._sample_rows))
        if self.failure is not None: raise RuntimeError("resource monitor failed") from self.failure

    @property
    def last_resources(self) -> dict[str, Any]:
        latest = {}
        if self._gpu: latest.update(self._gpu[-1])
        if self._ram: latest.update(self._ram[-1])
        if self._disk: latest.update(self._disk[-1])
        return latest

    def check(self, *, phase: str = "pilot", starting_new_sample: bool = False,
              remaining_samples: int | None = None, run_dir: Path | None = None) -> dict[str, Any]:
        if self.failure is not None: return {"status": "HARD_STOP", "reason_code": "MONITOR_FAILURE", "reason": str(self.failure)}
        if not self.last_resources: return {"status": "HARD_STOP", "reason_code": "MONITOR_NO_SNAPSHOT", "reason": "resource monitor has not produced a sample"}
        snapshot = dict(self.last_resources)
        if remaining_samples is not None: snapshot["remaining_samples"] = int(remaining_samples)
        if run_dir is not None:
            sizes = self._validated_sample_sizes(run_dir)
            if sizes: snapshot["mean_success_sample_bytes"] = float(sum(sizes) / len(sizes))
        return evaluate_resources(snapshot, phase=phase, starting_new_sample=starting_new_sample)

    def capture_sample(self, sample_id: str, phase: str, remaining: int, run_dir: Path | None = None) -> dict[str, Any]:
        """Synchronously persist per-sample resource evidence from all samplers."""
        row = {"sample_id": str(sample_id), "phase": str(phase), "remaining_samples": int(remaining),
               "timestamp": float(self.clock())}
        row.update(dict(self.gpu_sampler())); row.update(dict(self.ram_sampler())); row.update(dict(self.disk_sampler()))
        if phase == "post_cleanup":
            gpu = row.get("gpu_used_gib", row.get("gpu_allocated_gib")); rss = row.get("rss_gib", row.get("process_rss_gib"))
            if gpu is not None:
                if self._cleanup_gpu_baseline is None: self._cleanup_gpu_baseline = float(gpu)
                growth = float(gpu) - self._cleanup_gpu_baseline; row["gpu_cleanup_growth_gib"] = growth
                self._cleanup_gpu_consecutive = self._cleanup_gpu_consecutive + 1 if growth > 2.0 else 0
                row["gpu_consecutive_growth_samples"] = self._cleanup_gpu_consecutive
            if rss is not None:
                if self._cleanup_rss_baseline is None: self._cleanup_rss_baseline = float(rss)
                growth = float(rss) - self._cleanup_rss_baseline; row["ram_cleanup_growth_gib"] = growth
                self._cleanup_ram_consecutive = self._cleanup_ram_consecutive + 1 if growth > 10.0 else 0
                row["ram_consecutive_growth_samples"] = self._cleanup_ram_consecutive
        if run_dir is not None:
            sizes = self._validated_sample_sizes(run_dir)
            if sizes: row["mean_success_sample_bytes"] = float(sum(sizes) / len(sizes))
            row["remaining_samples"] = int(remaining)
            row["consecutive_growth_samples"] = max(row.get("gpu_consecutive_growth_samples", 0), row.get("ram_consecutive_growth_samples", 0))
            decision = evaluate_resources(row, phase="pilot", starting_new_sample=(phase == "pre_sample"))
            row.update({"decision_status": decision.get("status"), "reason_code": decision.get("reason_code")})
        self._sample_rows.append(row)
        return row

    def _validated_sample_sizes(self, run_dir: Path) -> list[int]:
        sizes = []
        sample_root = Path(run_dir) / "samples"
        if not sample_root.is_dir(): return sizes
        for path in sorted(sample_root.iterdir()):
            status_path = path / "status.json"
            if not path.is_dir() or not status_path.is_file(): continue
            try:
                state = json.loads(status_path.read_text(encoding="utf-8"))
                hashes = state.get("artifact_sha256", {})
                if state.get("status") == "success" and all((path / name).is_file() and _sha_file(path / name) == digest for name, digest in hashes.items()):
                    sizes.append(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))
            except (OSError, ValueError, json.JSONDecodeError): continue
        return sizes

    def _write(self, filename: str, rows: list[dict[str, Any]]):
        path = self.root / filename
        keys = sorted({key for row in rows for key in row}) or ["timestamp"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)

    def __enter__(self): return self.start()
    def __exit__(self, exc_type, exc, tb):
        try: self.stop()
        except RuntimeError:
            if exc_type is None: raise


class Task6StateMachine:
    PHASES = ("PREFLIGHT", "RESOURCE_SMOKE", "AWAITING_RESOURCE_REVIEW", "PILOT", "ANALYSIS", "AWAITING_REVIEW")
    TRANSITIONS = {"PREFLIGHT": "RESOURCE_SMOKE", "RESOURCE_SMOKE": "AWAITING_RESOURCE_REVIEW",
                   "AWAITING_RESOURCE_REVIEW": "PILOT", "PILOT": "ANALYSIS", "ANALYSIS": "AWAITING_REVIEW"}
    def __init__(self, phase: str = "PREFLIGHT", *, state_path: str | Path | None = None):
        if phase not in self.PHASES: raise ValueError("unknown Task 6 phase")
        self.state_path = Path(state_path) if state_path is not None else None
        if self.state_path is not None and self.state_path.is_file():
            saved = json.loads(self.state_path.read_text(encoding="utf-8")); phase = saved.get("phase", phase)
            if phase not in self.PHASES: raise ValueError("invalid persisted Task 6 phase")
        self.phase = phase
        self._persist()
    def _persist(self):
        if self.state_path is not None: _atomic_json(self.state_path, {"phase": self.phase})
    def advance(self, *, accepted: bool = False):
        if self.phase in ("AWAITING_RESOURCE_REVIEW", "AWAITING_REVIEW") and not accepted: raise RuntimeError("explicit review decision required")
        self.phase = self.TRANSITIONS[self.phase]; self._persist(); return self.phase


def run_resource_smoke(run_dir: str | Path, *, lifecycle: Mapping[str, Callable[[], Any]] | None = None,
                       samplers: Mapping[str, Callable[[], Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Run exactly one baseline through the real preload/load/call/cleanup lifecycle."""
    if not isinstance(samplers, Mapping) or set(samplers) != {"gpu", "ram", "disk"} or not isinstance(lifecycle, Mapping) or set(lifecycle) != {"pre_load", "load", "cleanup", "unload"}:
        raise BlockedExecution("resource smoke requires lifecycle factory and GPU/RAM/disk samplers")
    if not all(callable(samplers[key]) for key in samplers): raise BlockedExecution("resource smoke samplers must be callable")
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    prior_path = root / "run_status.json"
    if not prior_path.is_file(): raise BlockedExecution("resource smoke requires prior PREFLIGHT_COMPLETE status")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior.get("status") != "PREFLIGHT_COMPLETE" or prior.get("phase") != "RESOURCE_SMOKE": raise BlockedExecution("resource smoke requires matching preflight")
    prior_hashes = prior.get("hashes")
    if not isinstance(prior_hashes, Mapping) or set(prior_hashes) != {"code", "model", "config", "direction", "input", "noise"}:
        raise BlockedExecution("preflight status has incomplete hash binding")
    background = ResourceMonitor(root, gpu_sampler=samplers["gpu"], ram_sampler=samplers["ram"], disk_sampler=samplers["disk"])
    background.start()
    rows: list[dict[str, Any]] = []
    stages = ("pre-load", "loaded", "call", "post-call", "post-cleanup", "unloaded")
    by_stage: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    def sample(stage, phase="resource-smoke"):
        try:
            row = {"stage": stage, "phase": phase, **dict(samplers["gpu"]()), **dict(samplers["ram"]()), **dict(samplers["disk"]())}
        except BaseException as error:
            row = {"stage": stage, "phase": phase, "status": "ERROR", "error": str(error)}
            errors.append({"stage": stage, "type": type(error).__name__, "message": str(error)})
        by_stage[stage] = row; rows.append(row); return row
    runtime = inputs = None; baseline: Mapping[str, Any] = {}; loaded_ok = False; load_attempted = False; baseline_calls = 0; hashes = prior_hashes; stopped = False
    try:
        lifecycle["pre_load"](); pre = sample("pre-load", "preload")
        pre_decision = evaluate_resources(pre, phase="preload")
        if pre_decision["status"] == "HARD_STOP":
            errors.append({"stage": "pre-load", "type": "ResourceStop", "message": pre_decision.get("reason", "unsafe preload")})
            raise ResourceStop(pre_decision.get("reason", "unsafe preload"))
        load_attempted = True
        loaded = lifecycle["load"]()
        if not isinstance(loaded, (tuple, list)) or len(loaded) != 2 or loaded[0] is None or loaded[1] is None: raise RuntimeError("lifecycle load must return non-None (runtime, inputs)")
        runtime, inputs = loaded; loaded_ok = True; sample("loaded")
        binding_config = task6_binding_config(inputs)
        hashes = build_task6_hash_binding(runtime, inputs, binding_config)
        if dict(hashes) != dict(prior_hashes): raise RuntimeError("loaded runtime/input binding differs from preflight")
        baseline_calls = 1
        baseline = runtime.execute({"sample_id": "baseline_smoke", "kind": "baseline", "alpha": 0.0, "sign": 0,
                                   "group": "C", "model_seed": inputs.seed}, inputs, scope="full")
        sample("call"); sample("post-call")
        try: lifecycle["cleanup"]()
        except BaseException as error: errors.append({"stage": "cleanup", "type": type(error).__name__, "message": str(error)})
        sample("post-cleanup")
        try: lifecycle["unload"]()
        except BaseException as error: errors.append({"stage": "unload", "type": type(error).__name__, "message": str(error)})
        sample("unloaded")
    except KeyboardInterrupt as error:
        errors.append({"stage": "lifecycle", "type": "KeyboardInterrupt", "message": str(error)})
    except BaseException as error:
        errors.append({"stage": "lifecycle", "type": type(error).__name__, "message": str(error)})
    finally:
        if load_attempted and "post-cleanup" not in by_stage:
            try: lifecycle["cleanup"]()
            except BaseException as error: errors.append({"stage": "cleanup", "type": type(error).__name__, "message": str(error)})
            sample("post-cleanup")
        if load_attempted and "unloaded" not in by_stage:
            try: lifecycle["unload"]()
            except BaseException as error: errors.append({"stage": "unload", "type": type(error).__name__, "message": str(error)})
            sample("unloaded")
        for stage in stages: by_stage.setdefault(stage, {"stage": stage, "status": "N/A", "reason": "not reached"})
        try: background.stop()
        except BaseException as error: errors.append({"stage": "monitor", "type": type(error).__name__, "message": str(error)})
        stopped = True
    if baseline and not errors and isinstance(baseline, Mapping):
        smoke_store = PrecisionSampleStore(root / "smoke_samples")
        sample_dir = root / "smoke_samples" / "baseline_smoke"
        if not sample_dir.exists(): smoke_store.write_success("baseline_smoke", {"scope": "resource-smoke", "binding": dict(hashes), "record": baseline})
    sample_dir = root / "smoke_samples" / "baseline_smoke"
    sample_bytes = sum(path.stat().st_size for path in sample_dir.rglob("*") if path.is_file()) if sample_dir.is_dir() else 0
    forecast = sample_bytes * 31 * 1.3
    disk_free = by_stage.get("unloaded", {}).get("disk_free_gib")
    peak_rows = list(getattr(background, "_gpu", [])) + rows
    peak_alloc = max((float(row.get("gpu_peak_allocated_gib", row.get("gpu_allocated_gib", 0))) for row in peak_rows), default=0.0)
    peak_reserved = max((float(row.get("gpu_peak_reserved_gib", row.get("gpu_reserved_gib", 0))) for row in peak_rows), default=0.0)
    peak_nvml = max((float(row.get("gpu_peak_nvml_used_gib", row.get("gpu_used_gib", row.get("nvml_used_gib", 0)))) for row in peak_rows), default=0.0)
    by_stage.setdefault("call", {"stage": "call", "status": "N/A"})
    by_stage["call"].update({"gpu_peak_allocated_gib": peak_alloc, "gpu_peak_reserved_gib": peak_reserved, "gpu_peak_nvml_used_gib": peak_nvml})
    actual_rows = [row for row in peak_rows if row.get("status") != "N/A"]
    worst = dict(max(actual_rows, key=lambda row: float(row.get("gpu_used_gib", row.get("nvml_used_gib", 0)))) if actual_rows else {})
    if actual_rows:
        for key in ("gpu_free_gib", "ram_available_gib", "rss_gib", "swap_used_gib", "disk_free_gib"):
            vals = [float(row[key]) for row in actual_rows if row.get(key) is not None]
            if vals: worst[key] = min(vals)
    worst.update({"gpu_used_gib": peak_nvml, "gpu_peak_allocated_gib": peak_alloc, "gpu_peak_nvml_used_gib": peak_nvml})
    if disk_free is not None: worst["disk_free_gib"] = float(disk_free)
    worst["mean_success_sample_bytes"] = sample_bytes; worst["remaining_samples"] = 31
    gate = evaluate_resources(worst, phase="resource-smoke")
    if gate.get("status") == "HARD_STOP":
        errors.append({"stage": "call", "type": gate.get("reason_code", "RESOURCE_STOP"), "message": gate.get("reason", "smoke peak gate failed")})
    status = "AWAITING_RESOURCE_REVIEW" if not errors else "RESOURCE_STOP"
    reason_code = "SMOKE_REVIEW_REQUIRED" if not errors else errors[0]["type"]
    reason = "baseline smoke complete; explicit review required" if not errors else errors[0]["message"]
    payload = build_run_status(status, reason_code=reason_code, reason=reason, completed=[], failed=[], skipped=[],
        resource_snapshots={"by_stage": by_stage, "background_gpu": background._gpu, "background_ram": background._ram, "background_disk": background._disk, "last": background.last_resources},
        hashes=dict(hashes), smoke_decision={"status": status, "reason_code": reason_code}, baseline_calls=baseline_calls,
        disk_size_forecast={"mean_success_sample_bytes": sample_bytes, "remaining_samples": 31, "forecast_bytes": forecast,
                            "forecast_free_gib": None if disk_free is None else float(disk_free) - forecast / 2**30}, errors=errors)
    _atomic_json(root / "smoke_stage_snapshots.json", {"stages": list(stages), "by_stage": by_stage})
    _atomic_json(root / "run_status.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bounded Task 6 pilot")
    parser.add_argument("--phase", choices=("preflight", "resource-smoke", "pilot"), default="preflight")
    parser.add_argument("--run-dir"); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--state", default="bridge_0"); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight-json")
    return parser.parse_args(argv)


class BlockedExecution(RuntimeError):
    """User-facing fail-closed launcher refusal."""


def execute_task6(args: argparse.Namespace, *, runtime: Any | None = None, inputs: Task6Inputs | None = None,
                  monitor: ResourceMonitor | None = None, lifecycle: Mapping[str, Callable[[], Any]] | None = None,
                  samplers: Mapping[str, Callable[[], Mapping[str, Any]]] | None = None) -> dict[str, Any]:
    """Execute only the explicitly requested phase; pilot needs injected runtime inputs."""
    if args.phase == "preflight":
        if runtime is None or inputs is None: raise BlockedExecution("preflight requires injected runtime and validated inputs")
        config = {}
        if args.preflight_json:
            path = Path(args.preflight_json)
            try: config = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error: raise BlockedExecution(f"invalid preflight JSON: {error}") from error
        if not args.preflight_json: raise BlockedExecution("explicit preflight evidence JSON is required")
        result = preflight_task6(config, strict=True, runtime=runtime, inputs=inputs)
        if args.run_dir:
            root = Path(args.run_dir); root.mkdir(parents=True, exist_ok=True)
            binding_config = task6_binding_config(inputs)
            payload = {"status": "PREFLIGHT_COMPLETE", "phase": "RESOURCE_SMOKE", "reason_code": "PREFLIGHT_COMPLETE",
                       "reason": "preflight complete; resource smoke requires explicit phase", "completed_samples": [], "failed_samples": [], "skipped_samples": [],
                       "hashes": build_task6_hash_binding(runtime, inputs, binding_config)}
            _atomic_json(root / "run_status.json", payload)
            result["run_status"] = payload
        return result
    if not args.run_dir: raise BlockedExecution("--run-dir is required for resource-smoke or pilot")
    if args.phase == "resource-smoke":
        if lifecycle is None or samplers is None: raise BlockedExecution("resource smoke requires injected lifecycle and samplers")
        return run_resource_smoke(args.run_dir, lifecycle=lifecycle, samplers=samplers)
    if args.state != "bridge_0" or args.seed != 0: raise BlockedExecution("pilot is restricted to bridge_0 / seed 0")
    if runtime is None or inputs is None: raise BlockedExecution("pilot runtime and validated inputs must be injected after review")
    return run_pilot(runtime, inputs, args.run_dir, resume=bool(args.resume), monitor=monitor)


def run_pilot(runtime: Any, inputs: Task6Inputs, run_dir: str | Path, *, resume: bool = False, monitor: Any | None = None) -> dict[str, Any]:
    """Explicit pilot entry point, restricted to bridge_0 / seed 0."""
    if inputs.state != "bridge_0" or inputs.seed != 0:
        raise BlockedExecution("pilot is restricted to bridge_0 / seed 0")
    if monitor is None: raise BlockedExecution("pilot requires an integrated resource monitor")
    binding_config = {"schema_version": "umi-task6-v1", "group": {"state": inputs.state, "seed": inputs.seed},
                      "settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "autocast": False, "tf32": False, "diffusion_cache": False, "batch_size": 1}}
    authorization = authorize_pilot(run_dir, expected_hashes=build_task6_hash_binding(runtime, inputs, binding_config))
    if not isinstance(monitor, ResourceMonitor):
        raise BlockedExecution("pilot requires a ResourceMonitor instance")
    return _run_task6_group(runtime, inputs, run_dir, resume=resume, authorization=authorization, monitor=monitor)


def authorize_pilot(run_dir: str | Path, *, expected_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Require an explicit, hash-bound acceptance of the prior smoke run."""
    path = Path(run_dir) / "run_status.json"
    if not path.is_file(): raise BlockedExecution("pilot requires a prior resource-smoke status")
    status = json.loads(path.read_text(encoding="utf-8"))
    acceptance_path = Path(run_dir) / "smoke_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8")) if acceptance_path.is_file() else status
    if acceptance.get("status") != "AWAITING_RESOURCE_REVIEW" or acceptance.get("smoke_decision_accepted") is not True:
        raise BlockedExecution("pilot requires an explicit accepted resource-smoke decision")
    hashes = acceptance.get("hashes", {})
    if set(hashes) != {"code", "model", "config", "direction", "input", "noise"}:
        raise BlockedExecution("resource-smoke status has incomplete hash binding")
    if expected_hashes is not None and dict(expected_hashes) != hashes:
        raise BlockedExecution("pilot hashes do not match accepted resource smoke")
    if status.get("status") not in {"AWAITING_RESOURCE_REVIEW", "RESOURCE_STOP"}:
        raise BlockedExecution("pilot status is not resumable")
    return {**status, **acceptance, "status": "AWAITING_RESOURCE_REVIEW"}


def accept_resource_smoke(run_dir: str | Path, *, hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Persist the controller's explicit smoke acceptance decision."""
    path = Path(run_dir) / "run_status.json"
    if not path.is_file(): raise BlockedExecution("resource-smoke status is missing")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "AWAITING_RESOURCE_REVIEW": raise BlockedExecution("only a reviewable smoke run can be accepted")
    if hashes is None: raise BlockedExecution("complete expected hash binding is required to accept smoke")
    if dict(hashes) != status.get("hashes", {}): raise BlockedExecution("smoke acceptance hashes do not match status")
    status["smoke_decision_accepted"] = True
    status["smoke_accepted_at_unix"] = time.time()
    acceptance_path = path.with_name("smoke_acceptance.json")
    _atomic_json(acceptance_path, {"status": "AWAITING_RESOURCE_REVIEW", "smoke_decision_accepted": True,
                                   "smoke_accepted_at_unix": status["smoke_accepted_at_unix"], "hashes": dict(hashes)})
    stage = path.with_name("." + path.name + ".accept")
    stage.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(stage, path)
    return status


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try: result = execute_task6(args)
    except (BlockedExecution, PreflightError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)})); return 3
    print(json.dumps(result, default=str)); return 0 if result.get("status") in {"PASS", "AWAITING_RESOURCE_REVIEW", "AWAITING_REVIEW"} else 2


__all__ = ["AtomicSampleStore", "BlockedExecution", "ResourceMonitor", "Task6StateMachine", "accept_resource_smoke", "authorize_pilot", "execute_task6", "main", "monitor_decision", "parse_args", "run_pilot", "run_resource_smoke"]


if __name__ == "__main__": raise SystemExit(main())
