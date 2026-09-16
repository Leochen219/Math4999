"""Serial decoder replays and condition round-trip helpers for Task 6.

This module deliberately consumes only completed generation samples.  It does
not run generation and never keeps more than one decoded sample in memory.
The decoder is restored in a ``finally`` block even when a replay fails.
"""
from __future__ import annotations

import base64
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
    from .umi_task6_primitives import ALPHAS, build_decoder_replay_plan
except ImportError:  # pragma: no cover
    from umi_precision_runtime import EvidenceError, projection
    from umi_task6_primitives import ALPHAS, build_decoder_replay_plan


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    latent = _call_encoder(encoder, input_frame)
    return {"input_float32": original, "input_after_uint8_simulation": input_frame,
            "condition_latent_float32": latent, "quantized": bool(quantize)}


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
    before = getattr(runtime, "decoder_state", None)
    cleanup = getattr(runtime, "restore_decoder_state", None)
    started = time.perf_counter()
    try:
        decoded, _ = _decode(runtime, np.asarray(latent, dtype=np.float32), precision)
        output = np.asarray(decoded, dtype=np.float32).copy()
        if not output.size or not np.all(np.isfinite(output)):
            raise EvidenceError("decoder output is nonfinite or empty")
        return {"status": "success", "precision": precision, "elapsed_seconds": time.perf_counter() - started,
                "decoded_full_float32": output, "decoded_final_float32": _final_frame(output),
                "decoder_state_before": before}
    finally:
        try:
            if callable(cleanup):
                cleanup()
            else:
                clear = getattr(runtime, "clear_decoder_cache", None)
                if callable(clear): clear()
        except Exception:
            # Restoration failures must be visible to the caller, including
            # when the decode itself raised.  Never silently continue.
            raise


def _load_sample(root: Path, sample_id: str) -> tuple[np.ndarray, dict[str, Any]]:
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
    for name in ("output_full.npy", "sample.json"):
        if not (sample / name).is_file():
            raise EvidenceError(f"generation sample lacks {name}: {sample_id}")
    return np.load(sample / "output_full.npy", allow_pickle=False).astype(np.float32, copy=True), json.loads((sample / "sample.json").read_text(encoding="utf-8"))


def run_task6_decoder_replays(runtime: Any, run_root: str | Path, *, state: str = "bridge_0", seed: int = 0,
                              encoder: Any | None = None, resume: bool = False) -> dict[str, Any]:
    """Run exactly 16 serial replays and optional float/uint8 re-encodes."""
    root = Path(run_root); decoder_root = root / "decoder"
    decoder_root.mkdir(parents=True, exist_ok=True)
    plan = decoder_replay_plan(state, seed)
    status_path = decoder_root / "status.json"
    if status_path.is_file() and not resume:
        raise FileExistsError("Task 6 decoder output exists; use resume explicitly")
    summary = {"status": "running", "state": state, "seed": seed, "decoder_calls": 0,
               "selected_latents": len(select_decoder_latents(state, seed)), "failures": 0, "records": []}
    _atomic_json(status_path, summary)
    for spec in plan:
        replay_dir = decoder_root / spec["replay_id"]
        if replay_dir.is_dir() and resume and (replay_dir / "record.json").is_file():
            summary["records"].append(spec["replay_id"]); continue
        replay_dir.mkdir(parents=True, exist_ok=True)
        try:
            latent, source_meta = _load_sample(root, spec["sample_id"])
            record = replay_one(runtime, latent, precision=spec["decode_precision"])
            arrays: dict[str, np.ndarray] = {"decoder_input_full_latent.npy": latent,
                "decoded_full_float32.npy": record["decoded_full_float32"],
                "decoded_final_float32.npy": record["decoded_final_float32"]}
            if encoder is not None:
                direct = reencode_frame(record["decoded_final_float32"], encoder, quantize=False)
                quant = reencode_frame(record["decoded_final_float32"], encoder, quantize=True)
                arrays.update({"roundtrip_direct_condition_latent.npy": direct["condition_latent_float32"],
                               "roundtrip_uint8_condition_latent.npy": quant["condition_latent_float32"],
                               "roundtrip_uint8_input.npy": quant["input_after_uint8_simulation"]})
            for name, value in arrays.items(): np.save(replay_dir / name, value, allow_pickle=False)
            metadata = {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}
            metadata.update({"status": "success", "spec": spec, "source_sample": source_meta,
                             "artifact_sha256": {name: sha256_file(replay_dir / name) for name in arrays}})
            _atomic_json(replay_dir / "record.json", metadata)
            summary["decoder_calls"] += 1; summary["records"].append(spec["replay_id"])
        except Exception as error:
            summary["failures"] += 1; summary.update({"status": "BLOCKED", "error": {"type": type(error).__name__, "message": str(error)}})
            _atomic_json(status_path, summary)
            return summary
        _atomic_json(status_path, summary)
    summary.update({"status": "COMPLETE", "decoder_calls": 16})
    _atomic_json(status_path, summary)
    return summary


__all__ = ["decoder_replay_plan", "reencode_frame", "replay_one", "run_task6_decoder_replays",
           "select_decoder_latents", "sha256_file", "simulate_uint8"]
