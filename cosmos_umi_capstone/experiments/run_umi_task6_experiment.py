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
    from .umi_task6_runtime import (PreflightError, Task6Inputs, preflight_task6, run_task6_group,
                                    verify_reference_reuse)
except ImportError:
    from umi_task6_primitives import build_run_status, evaluate_resources
    from umi_task6_runtime import PreflightError, Task6Inputs, preflight_task6, run_task6_group, verify_reference_reuse


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


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
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.gpu_sampler = gpu_sampler or (lambda: {})
        self.ram_sampler = ram_sampler or (lambda: {})
        self.disk_sampler = disk_sampler or (lambda: {})
        self.cadence_seconds = {"gpu": 1.0, "ram": 5.0, "disk": 5.0}
        self.clock = clock; self.sleep = sleep; self._stop = threading.Event(); self.failure: Exception | None = None
        self._thread: threading.Thread | None = None; self._gpu: list[dict[str, Any]] = []; self._ram: list[dict[str, Any]] = []; self._disk: list[dict[str, Any]] = []

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
        self._thread = threading.Thread(target=self._poll, name="task6-resource-monitor", daemon=True); self._thread.start(); return self

    def stop(self):
        self._stop.set()
        if self._thread is not None: self._thread.join(timeout=5)
        self._write("gpu_samples.csv", self._gpu); self._write("ram_samples.csv", self._ram); self._write("disk_samples.csv", self._disk)
        if self.failure is not None: raise RuntimeError("resource monitor failed") from self.failure

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
    def __init__(self, phase: str = "PREFLIGHT"): 
        if phase not in self.PHASES: raise ValueError("unknown Task 6 phase")
        self.phase = phase
    def advance(self, *, accepted: bool = False):
        if self.phase in ("AWAITING_RESOURCE_REVIEW", "AWAITING_REVIEW") and not accepted: raise RuntimeError("explicit review decision required")
        self.phase = self.TRANSITIONS[self.phase]; return self.phase


def _status_hashes() -> dict[str, str]:
    return {key: hashlib.sha256(key.encode()).hexdigest() for key in ("code", "model", "config", "direction", "input", "noise")}


def run_resource_smoke(run_dir: str | Path, *, snapshots: list[Mapping[str, Any]] | None = None,
                       baseline_call: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Record baseline-only smoke evidence and stop for explicit review."""
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    rows = list(snapshots or [{"gpu_used_gib": 0, "gpu_free_gib": 100, "ram_available_gib": 600, "disk_free_gib": 20}])
    stages = ["pre-load", "loaded"]
    if baseline_call is not None: baseline_call()
    stages += ["call", "post-call", "post-cleanup", "unloaded"]
    decisions = [evaluate_resources(row, phase="resource-smoke") for row in rows]
    hard = next((item for item in decisions if item["status"] == "HARD_STOP"), None)
    status = "RESOURCE_STOP" if hard else "AWAITING_RESOURCE_REVIEW"
    stage_snapshots = {stage: dict(rows[min(index, len(rows) - 1)]) for index, stage in enumerate(stages)}
    payload = build_run_status(status, reason_code=hard["reason_code"] if hard else "SMOKE_REVIEW_REQUIRED",
                               reason=hard["reason"] if hard else "baseline smoke complete; explicit review required",
                               completed=[], failed=[], skipped=[], resource_snapshots={"stages": stages, "snapshots": rows, "by_stage": stage_snapshots}, hashes=_status_hashes(), smoke_decision=decisions[-1], disk_size_forecast={"mean_success_sample_bytes": None, "remaining_samples": None})
    (root / "run_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def execute_task6(args: argparse.Namespace, *, runtime: Any | None = None, inputs: Task6Inputs | None = None) -> dict[str, Any]:
    """Execute only the explicitly requested phase; pilot needs injected runtime inputs."""
    if args.phase == "preflight":
        config = {}
        if args.preflight_json:
            path = Path(args.preflight_json)
            try: config = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error: raise BlockedExecution(f"invalid preflight JSON: {error}") from error
        result = preflight_task6(config, strict=False)
        if args.run_dir:
            root = Path(args.run_dir); root.mkdir(parents=True, exist_ok=True)
            payload = build_run_status("AWAITING_RESOURCE_REVIEW", reason_code="PREFLIGHT_COMPLETE",
                reason="preflight complete; resource smoke requires explicit phase", hashes=_status_hashes())
            (root / "run_status.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            result["run_status"] = payload
        return result
    if not args.run_dir: raise BlockedExecution("--run-dir is required for resource-smoke or pilot")
    if args.phase == "resource-smoke": return run_resource_smoke(args.run_dir)
    if args.state != "bridge_0" or args.seed != 0: raise BlockedExecution("pilot is restricted to bridge_0 / seed 0")
    if runtime is None or inputs is None: raise BlockedExecution("pilot runtime and validated inputs must be injected after review")
    return run_task6_group(runtime, inputs, args.run_dir, resume=bool(args.resume))


def run_pilot(runtime: Any, inputs: Task6Inputs, run_dir: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Explicit pilot entry point, restricted to bridge_0 / seed 0."""
    if inputs.state != "bridge_0" or inputs.seed != 0:
        raise BlockedExecution("pilot is restricted to bridge_0 / seed 0")
    authorize_pilot(run_dir)
    return run_task6_group(runtime, inputs, run_dir, resume=resume)


def authorize_pilot(run_dir: str | Path, *, expected_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Require an explicit, hash-bound acceptance of the prior smoke run."""
    path = Path(run_dir) / "run_status.json"
    if not path.is_file(): raise BlockedExecution("pilot requires a prior resource-smoke status")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "AWAITING_RESOURCE_REVIEW" or status.get("smoke_decision_accepted") is not True:
        raise BlockedExecution("pilot requires an explicit accepted resource-smoke decision")
    hashes = status.get("hashes", {})
    if set(hashes) != {"code", "model", "config", "direction", "input", "noise"}:
        raise BlockedExecution("resource-smoke status has incomplete hash binding")
    if expected_hashes is not None and dict(expected_hashes) != hashes:
        raise BlockedExecution("pilot hashes do not match accepted resource smoke")
    return status


def accept_resource_smoke(run_dir: str | Path, *, hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Persist the controller's explicit smoke acceptance decision."""
    path = Path(run_dir) / "run_status.json"
    if not path.is_file(): raise BlockedExecution("resource-smoke status is missing")
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("status") != "AWAITING_RESOURCE_REVIEW": raise BlockedExecution("only a reviewable smoke run can be accepted")
    if hashes is not None and dict(hashes) != status.get("hashes", {}): raise BlockedExecution("smoke acceptance hashes do not match status")
    status["smoke_decision_accepted"] = True
    status["smoke_accepted_at_unix"] = time.time()
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try: result = execute_task6(args)
    except (BlockedExecution, PreflightError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)})); return 3
    print(json.dumps(result, default=str)); return 0 if result.get("status") in {"PASS", "AWAITING_RESOURCE_REVIEW", "AWAITING_REVIEW"} else 2


__all__ = ["AtomicSampleStore", "BlockedExecution", "ResourceMonitor", "Task6StateMachine", "accept_resource_smoke", "authorize_pilot", "execute_task6", "main", "monitor_decision", "parse_args", "run_pilot", "run_resource_smoke"]


if __name__ == "__main__": raise SystemExit(main())
