"""Serial decoder replays and condition round-trip helpers for Task 6.

This module deliberately consumes only completed generation samples.  It does
not run generation and never keeps more than one decoded sample in memory.
The decoder is restored in a ``finally`` block even when a replay fails.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from .umi_precision_runtime import EvidenceError, projection
    from .umi_task6_primitives import ALPHAS, RAW_MUTABLE_FILES, build_decoder_replay_plan, build_generation_plan
except ImportError:  # pragma: no cover
    from umi_precision_runtime import EvidenceError, projection
    from umi_task6_primitives import ALPHAS, RAW_MUTABLE_FILES, build_decoder_replay_plan, build_generation_plan

try:
    from .umi_precision_storage import ProcessLock
except ImportError:  # pragma: no cover
    from umi_precision_storage import ProcessLock


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text(path: Path, text: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="ascii") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decoder_code_sha256() -> str:
    """Bind both decoder implementations used by the public adapter seam."""
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).with_name("umi_task5_decoder.py")):
        digest.update(path.name.encode("utf-8")); digest.update(path.read_bytes())
    return digest.hexdigest()


def _bound_identity(value: Any) -> bool:
    """Reject absent or class-only runtime/encoder identities."""
    if value is None: return False
    if isinstance(value, Mapping):
        # The concrete encoder adapter wraps its observed identity as
        # ``encoder -> content -> fingerprint``.  Keep class/type/device
        # metadata insufficient on its own while accepting that reviewed
        # nested fingerprint shape alongside the established identity keys.
        content_keys = {"content_sha256", "weights_sha256", "fingerprint", "model_state", "decoder_state", "encoder_state", "artifacts", "code", "inputs"}
        return (any(key in value and value[key] not in (None, "", {}, []) for key in content_keys)
                or any(_bound_identity(nested) for nested in value.values() if isinstance(nested, Mapping)))
    return False


def _runtime_binding_identity(runtime: Any) -> Any:
    """Return a stable, non-class-only identity for config/resume binding."""
    actual = getattr(runtime, "actual_identity", None)
    if callable(actual):
        value = actual()
        if value is not None: return value
    value = _decoder_identity(runtime)
    if isinstance(value, Mapping): return value
    if value is not None: return None
    model, ops = getattr(runtime, "model", None), getattr(runtime, "ops", None)
    if model is not None or ops is not None: return None
    return None


def select_decoder_latents(state: str = "bridge_0", seed: int = 0) -> list[dict[str, Any]]:
    """Return eight and only eight logical replay selections."""
    plan = build_decoder_replay_plan(state, seed)
    logical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in plan:
        sample_id = str(item["sample_id"])
        if sample_id not in seen:
            seen.add(sample_id)
            logical.append(dict(item))
    if len(logical) != 8 or len(plan) != 16:
        raise AssertionError("Task 6 decoder plan must contain 8 logical latents and 16 calls")
    return logical


def decoder_replay_plan(state: str = "bridge_0", seed: int = 0) -> list[dict[str, Any]]:
    plan = build_decoder_replay_plan(state, seed)
    if len(plan) != 16 or len({item["replay_id"] for item in plan}) != 16:
        raise AssertionError("Task 6 decoder replay plan is not exactly 16 serial calls")
    return plan


def simulate_uint8(frame: Any) -> np.ndarray:
    """Explicit image quantization simulation, before any visualization I/O."""
    value = np.asarray(frame, dtype=np.float32)
    if not value.size or not np.all(np.isfinite(value)):
        raise ValueError("frame must be finite and non-empty")
    return (np.round(np.clip(value, 0.0, 1.0) * 255.0) / 255.0).astype(np.float32)


def _final_frame(decoded: Any) -> np.ndarray:
    value = np.asarray(projection(decoded), dtype=np.float32)
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 4 and value.shape[0] == 3:
        return np.ascontiguousarray(value[:, -1], dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 3:
        return np.ascontiguousarray(value, dtype=np.float32)
    raise EvidenceError("decoded output must have [3,T,H,W] or [3,H,W] layout")


def _call_encoder(encoder: Any, frame: np.ndarray) -> np.ndarray:
    if callable(encoder):
        result = encoder(frame)
    elif hasattr(encoder, "encode"):
        result = encoder.encode(frame)
    else:
        raise EvidenceError("condition encoder is not callable")
    result = np.asarray(projection(result), dtype=np.float32)
    if not result.size or not np.all(np.isfinite(result)):
        raise EvidenceError("condition encoder returned invalid latent")
    return np.ascontiguousarray(result)


def reencode_frame(frame: Any, encoder: Any, *, quantize: bool = False) -> dict[str, Any]:
    """Re-encode a saved float RGB frame directly or after uint8 simulation."""
    original = np.asarray(frame, dtype=np.float32).copy()
    input_frame = simulate_uint8(original) if quantize else original
    before = _encoder_identity(encoder)
    reset = getattr(encoder, "reset_cache", None) or getattr(encoder, "clear_cache", None)
    primary = None; latent = None; restore_error = None
    try:
        if callable(reset): reset()
        latent = _call_encoder(encoder, input_frame)
    except BaseException as error:
        primary = error
    finally:
        try:
            if callable(reset): reset()
        except BaseException as error:
            restore_error = error
    after = _encoder_identity(encoder)
    if before != after and restore_error is None: restore_error = EvidenceError("condition encoder state was not restored")
    if primary is not None:
        if restore_error is not None: setattr(primary, "restoration_error", restore_error)
        raise primary
    if restore_error is not None: raise restore_error
    assert latent is not None
    return {"input_float32": original, "input_after_uint8_simulation": input_frame,
            "condition_latent_float32": latent, "quantized": bool(quantize), "encoder_identity_before": before, "encoder_identity_after": after}


def _decode(runtime: Any, latent: np.ndarray, precision: str) -> tuple[Any, Callable[[], None] | None]:
    """Call the narrow decoder seam used by the official runtime or a test fake."""
    method = getattr(runtime, "decode_prediction_latent", None)
    if callable(method):
        result = method(latent, precision=precision)
        return result, None
    method = getattr(runtime, "decode", None)
    if callable(method):
        result = method(latent, precision=precision)
        return result, None
    # Task 5's verified runtime exposes the same low-level model layout.  The
    # import is local so importing this module remains CPU-only.
    try:
        from .umi_task5_decoder import replay_decode
    except ImportError:  # pragma: no cover
        from umi_task5_decoder import replay_decode
    record = replay_decode(runtime, latent, precision="native" if precision == "native_bf16" else "fp32")
    return record["decoder_normalized_full_output"], None


def replay_one(runtime: Any, latent: Any, *, precision: str) -> dict[str, Any]:
    """Replay one latent and restore dtype/cache state on success or failure."""
    if precision not in {"native_bf16", "temporary_fp32"}:
        raise ValueError("precision must be native_bf16 or temporary_fp32")
    before = _decoder_identity(runtime)
    cleanup = getattr(runtime, "restore_decoder_state", None)
    started = time.perf_counter()
    result = None; primary = None; restore_error = None
    try:
        decoded, _ = _decode(runtime, np.asarray(latent, dtype=np.float32), precision)
        output = np.asarray(decoded, dtype=np.float32).copy()
        if not output.size or not np.all(np.isfinite(output)):
            raise EvidenceError("decoder output is nonfinite or empty")
        result = {"status": "success", "precision": precision, "elapsed_seconds": time.perf_counter() - started,
                "decoded_full_float32": output, "decoded_final_float32": _final_frame(output),
                "decoder_state_before": before}
    except BaseException as error:
        primary = error
    finally:
        try:
            if callable(cleanup):
                cleanup()
            else:
                clear = getattr(runtime, "clear_decoder_cache", None)
                if callable(clear): clear()
        except BaseException as error:
            restore_error = error
    after = _decoder_identity(runtime)
    if before is not None and after is not None and before != after:
        restore_error = EvidenceError("decoder state identity was not restored")
    if primary is not None:
        if restore_error is not None: setattr(primary, "restoration_error", restore_error)
        raise primary
    if restore_error is not None: raise restore_error
    assert result is not None
    result["decoder_state_after"] = after
    return result


def _decoder_identity(runtime: Any) -> Any:
    for name in ("decoder_state_identity", "decoder_identity", "state_identity"):
        value = getattr(runtime, name, None)
        if callable(value): return value()
    value = getattr(runtime, "decoder_state", None)
    if isinstance(value, (str, int, float, bool)): return value
    return None


def _encoder_identity(encoder: Any) -> Any:
    for name in ("actual_identity", "identity", "state_identity", "encoder_identity"):
        value = getattr(encoder, name, None)
        if callable(value): return value()
    typ = type(encoder)
    value = {"type": typ.__name__, "module": typ.__module__, "qualname": typ.__qualname__}
    for name in ("dtype", "device", "model_seed"):
        if hasattr(encoder, name): value[name] = str(getattr(encoder, name))
    return value


def _load_sample(root: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sample = root / "samples" / sample_id
    status_path = sample / "status.json"
    if not status_path.is_file():
        raise EvidenceError(f"missing status for generation sample {sample_id}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "success":
        raise EvidenceError(f"generation sample is not successful: {sample_id}")
    for name, digest in status.get("artifact_sha256", {}).items():
        path = sample / name
        if not path.is_file() or sha256_file(path) != digest:
            raise EvidenceError(f"generation artifact hash mismatch: {sample_id}/{name}")
    for name in ("output_full.npy", "predicted_latent.npy", "sample.json"):
        if not (sample / name).is_file():
            raise EvidenceError(f"generation sample lacks {name}: {sample_id}")
    predicted = np.load(sample / "predicted_latent.npy", allow_pickle=False).astype(np.float32, copy=True)
    if predicted.shape != (1, 48, 4, 16, 16) or not np.all(np.isfinite(predicted)):
        raise EvidenceError(f"generation predicted_latent is not an exact four-frame block: {sample_id}")
    return np.load(sample / "output_full.npy", allow_pickle=False).astype(np.float32, copy=True), predicted, json.loads((sample / "sample.json").read_text(encoding="utf-8"))


def _verify_raw_task6(root: Path) -> tuple[str, Mapping[str, Any]]:
    status_path = root / "run_status.json"
    if not status_path.is_file(): raise EvidenceError("Task 6 raw status is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = [item["sample_id"] for item in build_generation_plan(status.get("group", {}).get("state", "bridge_0"), int(status.get("group", {}).get("seed", 0)))]
    completed = status.get("completed_samples", [])
    failed = status.get("failed_samples", [])
    skipped = status.get("skipped_samples", [])
    try:
        completed_ids = set(completed) if isinstance(completed, list) else set()
        skipped_ids = set(skipped) if isinstance(skipped, list) else set()
    except TypeError:
        completed_ids = skipped_ids = set()
    if (status.get("status") not in {"AWAITING_REVIEW", "COMPLETE"} or
            not isinstance(completed, list) or len(completed) != 32 or len(completed_ids) != 32 or completed_ids != set(expected) or
            not isinstance(failed, list) or failed or not isinstance(skipped, list) or
            len(skipped) != len(skipped_ids) or not skipped_ids.issubset(set(expected))):
        raise EvidenceError("decoder requires a completed 32-sample Task 6 raw run")
    manifest = root / "MANIFEST.sha256"
    if not manifest.is_file(): raise EvidenceError("Task 6 raw manifest is missing")
    seen: set[str] = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in seen:
            raise EvidenceError("malformed raw manifest")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in {"MANIFEST.sha256", ".runner.lock"}:
            raise EvidenceError("unsafe raw manifest path")
        if parts[1] in RAW_MUTABLE_FILES:
            continue
        target = root / relative
        if not target.is_file() or sha256_file(target) != parts[0]: raise EvidenceError(f"raw manifest mismatch: {parts[1]}")
        seen.add(parts[1])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in {"MANIFEST.sha256", ".runner.lock"} and p.relative_to(root).as_posix() not in RAW_MUTABLE_FILES}
    if seen != actual: raise EvidenceError("raw manifest inventory mismatch")
    for sample_id in expected:
        sample = root / "samples" / sample_id; state_path = sample / "status.json"
        if not state_path.is_file(): raise EvidenceError(f"raw sample is missing: {sample_id}")
        sample_status = json.loads(state_path.read_text(encoding="utf-8"))
        if sample_status.get("status") != "success": raise EvidenceError(f"raw sample is not successful: {sample_id}")
        for name, digest in sample_status.get("artifact_sha256", {}).items():
            if not (sample / name).is_file() or sha256_file(sample / name) != digest: raise EvidenceError(f"raw sample hash mismatch: {sample_id}/{name}")
    samples_root = root / "samples"
    actual_dirs = {p.name for p in samples_root.iterdir() if p.is_dir() and ".attempt." not in p.name} if samples_root.is_dir() else set()
    if actual_dirs != set(expected):
        raise EvidenceError("raw sample directory inventory mismatch")
    return sha256_file(manifest), status


def _decoder_manifest(root: Path) -> Path:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.relative_to(root).as_posix() not in {"MANIFEST.sha256", ".decoder.lock"}:
            entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n")
    path = root / "MANIFEST.sha256"; _atomic_text(path, "".join(entries)); return path


def _verify_decoder_manifest(root: Path) -> str:
    path = root / "MANIFEST.sha256"
    if not path.is_file(): raise EvidenceError("decoder manifest is missing")
    seen = set()
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1); relative = Path(parts[1]) if len(parts) == 2 else Path(".")
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in seen or relative.is_absolute() or ".." in relative.parts:
            raise EvidenceError("malformed decoder manifest")
        seen.add(parts[1]); target = root / relative
        if not target.is_file() or sha256_file(target) != parts[0]: raise EvidenceError(f"decoder manifest mismatch: {parts[1]}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in {"MANIFEST.sha256", ".decoder.lock"}}
    if actual != seen: raise EvidenceError("decoder manifest inventory mismatch")
    return sha256_file(path)


def run_task6_decoder_replays(runtime: Any, run_root: str | Path, *, state: str = "bridge_0", seed: int = 0,
                              encoder: Any | None = None, resume: bool = False,
                              decoder_root: str | Path | None = None, monitor: Any | None = None,
                              main_status_path: str | Path | None = None) -> dict[str, Any]:
    """Run exactly 16 serial replays in a derived tree, never in raw_root."""
    raw_root = Path(run_root).resolve()
    if encoder is None: raise EvidenceError("Task 6 decoder requires a condition encoder")
    raw_manifest_sha, raw_status = _verify_raw_task6(raw_root)
    root = Path(decoder_root).resolve() if decoder_root is not None else raw_root.parent / (raw_root.name + "_decoder")
    root.mkdir(parents=True, exist_ok=True)
    plan = decoder_replay_plan(state, seed); status_path = root / "status.json"
    runtime_identity = _runtime_binding_identity(runtime); encoder_identity = _encoder_identity(encoder)
    if not _bound_identity(runtime_identity): raise EvidenceError("decoder runtime identity is absent or class-only")
    if not _bound_identity(encoder_identity): raise EvidenceError("condition encoder identity is absent or class-only")
    config = {"schema_version": "umi-task6-decoder-v2", "state": state, "seed": seed, "plan": plan,
              "runtime_identity": runtime_identity, "encoder_identity": encoder_identity, "raw_manifest_sha256": raw_manifest_sha,
              "raw_group": raw_status.get("group"),
              "decoder_code_sha256": _decoder_code_sha256(),
              "source": str(Path(__file__).resolve())}
    config_text = json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    config_path = root / "decoder_config.json"
    if config_path.is_file() and not resume: raise FileExistsError("decoder output exists; use resume")
    if resume:
        if not config_path.is_file() or config_path.read_text(encoding="utf-8") != config_text: raise EvidenceError("decoder config mismatch")
        _verify_decoder_manifest(root)
    elif any(root.iterdir()): raise FileExistsError("decoder destination must be new")
    else: config_path.write_text(config_text, encoding="utf-8")
    summary = {"status": "running", "state": state, "seed": seed, "decoder_calls": 0,
               "selected_latents": 8, "failures": 0, "records": [], "config_sha256": sha256_file(config_path)}
    monitor_started = False
    monitor_stopped = False
    def stop_monitor() -> None:
        nonlocal monitor_stopped
        if monitor is not None and monitor_started and not monitor_stopped and callable(getattr(monitor, "stop", None)):
            monitor_stopped = True
            try:
                monitor.stop()
            except BaseException as error:
                summary.setdefault("secondary_errors", []).append({"type": type(error).__name__, "message": str(error)})
    def canonical_resource_stop(code: str, reason: str) -> dict[str, Any]:
        summary.update({"status": "RESOURCE_STOP", "reason_code": code, "reason": reason,
                        "resource_stop": True, "completed_decoder_calls": int(summary.get("decoder_calls", 0))})
        stop_monitor(); _atomic_json(status_path, summary); _decoder_manifest(root)
        if main_status_path is not None:
            main_path = Path(main_status_path)
            if main_path.is_file():
                try:
                    main = json.loads(main_path.read_text(encoding="utf-8"))
                    # Keep the generation terminal status authoritative.  The
                    # decoder is derived work; only its namespace is mutable,
                    # so raw completion remains analyzable after a decoder
                    # resource stop.
                    main.update({"decoder_reason_code": code, "decoder_reason": reason,
                                 "decoder_calls": int(summary.get("decoder_calls", 0)),
                                 "decoder_status": "RESOURCE_STOP"})
                    _atomic_json(main_path, main)
                except (OSError, ValueError, json.JSONDecodeError):
                    # Preserve the decoder evidence; a malformed mutable main
                    # status is itself reported in the decoder summary.
                    summary["main_status_update_error"] = "invalid or unwritable main status"
                    _atomic_json(status_path, summary); _decoder_manifest(root)
        return summary
    def monitor_guard(remaining: int) -> dict[str, Any] | None:
        if monitor is None: return None
        if getattr(monitor, "failure", None) is not None:
            return canonical_resource_stop("MONITOR_FAILURE", "decoder resource monitor failed")
        checker = getattr(monitor, "check", None)
        if callable(checker):
            decision = checker(phase="pilot", starting_new_sample=True, remaining_samples=remaining, run_dir=root)
            if decision.get("status") == "HARD_STOP":
                return canonical_resource_stop(str(decision.get("reason_code") or "RESOURCE_STOP"), str(decision.get("reason") or "decoder resource gate stopped"))
        return None
    @contextmanager
    def monitor_lifecycle():
        """Flush monitor evidence and terminal status on every exit path."""
        try:
            yield
        finally:
            stop_monitor()
            if summary.get("status") == "running":
                summary.update({"status": "INTERRUPTED", "reason_code": "UNHANDLED_EXIT",
                                "reason": "decoder exited before publishing a terminal result"})
                _atomic_json(status_path, summary)
                _decoder_manifest(root)

    with monitor_lifecycle(), ProcessLock(root / ".decoder.lock"):
        if monitor is not None and callable(getattr(monitor, "start", None)):
            # A monitor may create a worker and then raise while publishing its
            # initial sample. Mark it started before invocation so the outer
            # lifecycle guard always joins and flushes it.
            monitor_started = True
            try:
                monitor.start()
            except BaseException as error:
                return canonical_resource_stop("MONITOR_FAILURE", str(error))
        _atomic_json(status_path, summary)
        _decoder_manifest(root)
        for spec in plan:
            replay_dir = root / spec["replay_id"]
            if replay_dir.is_dir() and resume and (replay_dir / "record.json").is_file():
                try:
                    record = json.loads((replay_dir / "record.json").read_text(encoding="utf-8"))
                    if record.get("config_sha256") != sha256_file(config_path): raise EvidenceError("decoder success config binding mismatch")
                    for name, digest in record.get("artifact_sha256", {}).items():
                        if not (replay_dir / name).is_file() or sha256_file(replay_dir / name) != digest: raise EvidenceError("successful decoder artifact was tampered")
                except BaseException:
                    stop_monitor()
                    raise
                summary["records"].append(spec["replay_id"]); _atomic_json(status_path, summary); _decoder_manifest(root); continue
            try:
                stopped = monitor_guard(len(plan) - len(summary["records"]))
            except BaseException as error:
                return canonical_resource_stop("MONITOR_FAILURE", str(error))
            if stopped is not None: return stopped
            if monitor is not None and callable(getattr(monitor, "capture_sample", None)):
                try:
                    row = monitor.capture_sample(spec["replay_id"], "pre_sample", len(plan) - len(summary["records"]), root)
                except BaseException as error:
                    return canonical_resource_stop("MONITOR_FAILURE", str(error))
                if row.get("decision_status") == "HARD_STOP":
                    return canonical_resource_stop(str(row.get("reason_code") or "RESOURCE_STOP"), "decoder resource gate stopped before replay")
            if replay_dir.exists():
                attempt = 1
                while (root / f"{spec['replay_id']}.attempt.{attempt}").exists(): attempt += 1
                replay_dir.rename(root / f"{spec['replay_id']}.attempt.{attempt}")
            stage = Path(tempfile.mkdtemp(prefix=f".{spec['replay_id']}.", dir=str(root)))
            try:
                latent, predicted_latent, source_meta = _load_sample(raw_root, spec["sample_id"])
                record = replay_one(runtime, latent, precision=spec["decode_precision"])
                direct = reencode_frame(record["decoded_final_float32"], encoder, quantize=False)
                quant = reencode_frame(record["decoded_final_float32"], encoder, quantize=True)
                # Re-encoding can leave VAE/request allocations resident even
                # though replay_one restored decoder dtype/cache state.  Run
                # the validated runtime cleanup seam before taking the
                # post-cleanup resource sample; otherwise the monitor measures
                # a pre-cleanup plateau and can stop a healthy replay series.
                cleanup = getattr(runtime, "cleanup", None) or getattr(runtime, "reset_cache", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except BaseException as error:
                        return canonical_resource_stop("RUNTIME_CLEANUP_FAILURE", f"decoder runtime cleanup failed: {error}")
                arrays = {"decoder_input_full_latent.npy": latent, "predicted_latent.npy": predicted_latent, "decoded_full_float32.npy": record["decoded_full_float32"],
                          "decoded_final_float32.npy": record["decoded_final_float32"], "direct_float_input.npy": direct["input_float32"],
                          "uint8_simulated_input.npy": quant["input_after_uint8_simulation"], "direct_condition_latent_float32.npy": direct["condition_latent_float32"],
                          "uint8_condition_latent_float32.npy": quant["condition_latent_float32"]}
                for name, value in arrays.items(): np.save(stage / name, value, allow_pickle=False)
                metadata = {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}
                metadata.update({"status": "success", "spec": spec, "source_sample": source_meta, "config_sha256": sha256_file(config_path),
                                 "artifact_sha256": {name: sha256_file(stage / name) for name in arrays}})
                _atomic_json(stage / "record.json", metadata); os.replace(stage, replay_dir); stage = None
                summary["decoder_calls"] += 1; summary["records"].append(spec["replay_id"]); _atomic_json(status_path, summary); _decoder_manifest(root)
                if monitor is not None and callable(getattr(monitor, "capture_sample", None)):
                    try:
                        row = monitor.capture_sample(spec["replay_id"], "post_cleanup", len(plan) - len(summary["records"]), root)
                    except BaseException as error:
                        return canonical_resource_stop("MONITOR_FAILURE", str(error))
                    if row.get("decision_status") == "HARD_STOP":
                        return canonical_resource_stop(str(row.get("reason_code") or "RESOURCE_STOP"), "decoder resource gate stopped after cleanup")
            except Exception as error:
                text_error = str(error).lower()
                if isinstance(error, MemoryError) or "out of memory" in text_error or "cuda oom" in text_error:
                    return canonical_resource_stop("CUDA_OOM", str(error))
                summary["failures"] += 1; summary.update({"status": "BLOCKED", "error": {"type": type(error).__name__, "message": str(error)}}); stop_monitor(); _atomic_json(status_path, summary); _decoder_manifest(root); return summary
            finally:
                if stage is not None and stage.exists():
                    import shutil; shutil.rmtree(stage, ignore_errors=True)
            _atomic_json(status_path, summary); _decoder_manifest(root)
        summary.update({"status": "COMPLETE", "decoder_calls": 16}); stop_monitor()
        if summary.get("secondary_errors"):
            summary.update({"status": "RESOURCE_STOP", "reason_code": "MONITOR_FAILURE", "reason": "decoder resource monitor failed during shutdown"})
        _atomic_json(status_path, summary); _decoder_manifest(root); return summary


__all__ = ["decoder_replay_plan", "reencode_frame", "replay_one", "run_task6_decoder_replays",
           "select_decoder_latents", "sha256_file", "simulate_uint8", "_verify_decoder_manifest"]
