"""Bounded real decoder replays for Task 5B (never invokes generation)."""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .umi_precision_runtime import EvidenceError, projection
    from .umi_precision_storage import PrecisionSampleStore as SampleStore, ProcessLock
    from .umi_task5_primitives import build_decoder_replay_plan
    from .umi_fd_post_vae_scan import sha256_file
except ImportError:
    from umi_precision_runtime import EvidenceError, projection
    from umi_precision_storage import PrecisionSampleStore as SampleStore, ProcessLock
    from umi_task5_primitives import build_decoder_replay_plan
    from umi_fd_post_vae_scan import sha256_file


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def decoder_support_info(runtime: Any) -> dict[str, Any]:
    """Prove the specific loaded Task 4 decoder can be restored after FP32 replay."""
    tokenizer = getattr(runtime.model, "tokenizer_vision_gen", None)
    wan = getattr(tokenizer, "model", None)
    inner = getattr(wan, "model", None)
    torch = getattr(runtime.ops, "torch", None)
    native = str(getattr(wan, "dtype", "")).removeprefix("torch.")
    required = (tokenizer, wan, inner, torch)
    # Wan2.1 exposes ``is_amp=False``; Wan2.2 implements the same pure-dtype
    # path without that attribute.  Both are eligible only when the loaded
    # native dtype is BF16 and the actual inner decoder is inspectable.
    amp = getattr(wan, "is_amp", False)
    supported = all(value is not None for value in required) and native == "bfloat16" and not bool(amp) and callable(getattr(inner, "decode", None))
    return {"supported_fp32_replay": bool(supported), "native_wan_dtype": native or None,
            "wan_is_amp": getattr(wan, "is_amp", None), "tokenizer_type": type(tokenizer).__name__ if tokenizer is not None else None,
            "wan_type": type(wan).__name__ if wan is not None else None, "inner_type": type(inner).__name__ if inner is not None else None,
            "reason": None if supported else "loaded decoder is not the verified non-AMP BF16 WanVAE path"}


def _clear_cache(tokenizer: Any) -> None:
    if bool(getattr(tokenizer, "keep_decoder_cache", getattr(tokenizer, "_keep_decoder_cache", False))):
        raise EvidenceError("Task 5B requires keep_decoder_cache=false")
    clear = getattr(tokenizer, "clear_cache", None)
    if not callable(clear):
        clear = getattr(getattr(getattr(tokenizer, "model", None), "model", None), "clear_decoder_cache", None)
    if not callable(clear): raise EvidenceError("decoder does not expose cache clearing")
    clear()


def _normalized_frame(decoded: Any) -> tuple[np.ndarray, np.ndarray]:
    raw = projection(decoded)
    normalized = np.clip((1.0 + raw.astype(np.float64)) / 2.0, 0.0, 1.0).astype(np.float32)
    if normalized.ndim == 5 and normalized.shape[0] == 1:
        normalized = normalized[0]
    if normalized.ndim != 4 or normalized.shape[0] != 3:
        raise EvidenceError("decoder did not return [3,T,H,W] after batch removal")
    return raw, normalized[:, -1].copy()


def replay_decode(runtime: Any, full_latent: Any, *, precision: str) -> dict[str, Any]:
    """Decode an already-generated full latent once, with cache reset and observed input."""
    if precision not in {"native", "fp32"}:
        raise ValueError("decoder precision must be native or fp32")
    info = decoder_support_info(runtime)
    if precision == "fp32" and not info["supported_fp32_replay"]:
        raise EvidenceError("FP32 decoder replay is unsupported: " + str(info["reason"]))
    model, ops = runtime.model, runtime.ops
    tokenizer, wan, inner = model.tokenizer_vision_gen, model.tokenizer_vision_gen.model, model.tokenizer_vision_gen.model.model
    torch = ops.torch
    source = projection(full_latent)
    like = runtime.prepared[1].x0_tokens_vision[0]
    record: dict[str, Any] = {"precision_path": precision, "decoder_support": info, "decoder_source_full_latent": source.copy()}
    original_inner_decode = inner.decode
    original_dtype = wan.dtype
    original_tensors = {name: getattr(wan, name) for name in ("mean", "std", "scale", "img_mean", "img_std", "video_mean", "video_std") if hasattr(wan, name)}
    original_tensor_kwargs, original_precision = model.tensor_kwargs, model.precision
    changed = False
    try:
        _clear_cache(tokenizer)
        if precision == "fp32":
            # Convert the same loaded weights in-place for this request only;
            # BF16 -> FP32 is exact and the original BF16 tensors are restored.
            inner.to(dtype=torch.float32)
            wan.dtype = torch.float32
            for name in ("mean", "std", "img_mean", "img_std", "video_mean", "video_std"):
                if hasattr(wan, name):
                    setattr(wan, name, getattr(wan, name).to(dtype=torch.float32))
            if hasattr(wan, "scale"):
                wan.scale = tuple(value.to(dtype=torch.float32) if hasattr(value, "to") else value for value in wan.scale)
            if "mean" in original_tensors and "std" in original_tensors:
                wan.scale = [wan.mean, 1.0 / wan.std]
            changed = True
        def observed_decode(*args, **kwargs):
            if not args:
                raise EvidenceError("decoder latent argument is absent")
            latent = args[0]
            record["decoder_consumed_scaled_latent"] = projection(latent)
            record["decoder_consumed_scaled_dtype"] = str(latent.dtype).removeprefix("torch.")
            output = original_inner_decode(*args, **kwargs)
            record["decoder_raw_output"] = projection(output)
            record["decoder_raw_output_dtype"] = str(output.dtype).removeprefix("torch.")
            return output
        inner.decode = observed_decode
        model.tensor_kwargs = dict(original_tensor_kwargs)
        model.precision = original_precision
        latent = ops.from_array(source, like, dtype="float32")
        with ops.inference():
            decoded = model.decode(ops.cast(latent, "float32"))
        raw, frame = _normalized_frame(decoded)
        if "decoder_raw_output" not in record:
            raise EvidenceError("decoder inner boundary was not observed")
        record["decoder_normalized_full_output"] = np.clip((1.0 + raw.astype(np.float64)) / 2.0, 0.0, 1.0).astype(np.float32)
        record["decoded_final"] = frame
        record["decoder_weight_dtype"] = str(next(inner.parameters()).dtype).removeprefix("torch.")
        record["decoder_cache_policy"] = {"keep_decoder_cache": bool(getattr(tokenizer, "keep_decoder_cache", getattr(tokenizer, "_keep_decoder_cache", False))), "cleared_before": True}
        return record
    finally:
        inner.decode = original_inner_decode
        if changed:
            inner.to(dtype=original_dtype)
            wan.dtype = original_dtype
            for name, value in original_tensors.items():
                setattr(wan, name, value)
        model.tensor_kwargs, model.precision = original_tensor_kwargs, original_precision
        _clear_cache(tokenizer)
        record["decoder_cache_cleared_after"] = True


def _load_generation_sample(run_root: Path, sample_id: str) -> dict[str, Any]:
    sample = run_root / "samples" / sample_id
    status = json.loads((sample / "status.json").read_text(encoding="utf-8"))
    if status.get("status") != "success":
        raise EvidenceError(f"generation sample is not successful: {sample_id}")
    for name, digest in status.get("artifact_sha256", {}).items():
        if sha256_file(sample / name) != digest:
            raise EvidenceError(f"generation artifact hash mismatch: {sample_id}/{name}")
    return {"full": np.load(sample / "decoder_input_full_latent.npy", allow_pickle=False),
            "original_rgb": np.load(sample / "decoded_final.npy", allow_pickle=False)}


def run_task5_decoder_replays(runtime: Any, run_root: str | Path, *, resume: bool = False) -> dict[str, Any]:
    """Perform exactly 8 native plus 8 FP32 decode-only calls from Task 5A data."""
    root = Path(run_root); decoder_root = root / "decoder"
    decoder_root.mkdir(parents=True, exist_ok=True)
    with ProcessLock(decoder_root / ".decoder.lock"):
        support = decoder_support_info(runtime)
        plan = build_decoder_replay_plan()
        config = {"schema_version": "umi-task5-decoder-replay-v1", "support": support, "plan": plan,
                  "paths": ["native", "fp32"], "source": str(Path(__file__).resolve())}
        config_text = json.dumps(config, ensure_ascii=False, sort_keys=True)
        config_path = decoder_root / "decoder_plan.json"
        if config_path.exists():
            if not resume: raise FileExistsError("decoder replay exists; explicit resume required")
            if config_path.read_text(encoding="utf-8") != config_text + "\n": raise ValueError("decoder replay config mismatch")
        else:
            _atomic_json(config_path, config)
        status_path = decoder_root / "status.json"
        if resume and status_path.exists():
            old = json.loads(status_path.read_text(encoding="utf-8"))
            if old.get("status") == "complete": return old
        summary: dict[str, Any] = {"status": "running", "generation_calls": 0, "decoder_calls": 0, "failures": 0,
                                   "retries": 0, "diagnostic_attempts": 0, "support": support}
        for precision in ("native", "fp32"):
            store = SampleStore(decoder_root / precision)
            for spec in plan:
                disposition = store.prepare(spec["sample_id"], resume=resume, required_files=("sample.json", "decoded_final.npy"))
                if disposition == "skip":
                    continue
                source = _load_generation_sample(root, spec["sample_id"])
                started = time.perf_counter(); summary["decoder_calls"] += 1; _atomic_json(status_path, summary)
                try:
                    record = replay_decode(runtime, source["full"], precision=precision)
                    exact = bool(np.array_equal(record["decoded_final"], source["original_rgb"]))
                    record.update({"status": "success", "spec": spec, "elapsed_seconds": time.perf_counter() - started,
                                   "original_task5_rgb_exact": exact,
                                   "original_task5_rgb_rms_difference": rms_diff(record["decoded_final"], source["original_rgb"])})
                    arrays = {key + ".npy": value for key, value in record.items() if isinstance(value, np.ndarray)}
                    store.write_success(spec["sample_id"], {key: value for key, value in record.items() if not isinstance(value, np.ndarray)}, artifacts=arrays)
                except Exception as error:
                    summary["failures"] += 1
                    store.write_failure(spec["sample_id"], {"status": "fail", "spec": spec,
                        "error": {"type": type(error).__name__, "message": str(error)}, "traceback": traceback.format_exc()})
                    summary.update({"status": "blocked", "blocked_path": precision, "blocked_sample": spec["sample_id"]})
                    _atomic_json(status_path, summary); return summary
        summary.update({"status": "complete", "generation_calls": 0, "decoder_calls": 16})
        _atomic_json(status_path, summary)
        return summary


def rms_diff(left: Any, right: Any) -> float:
    value = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(value * value, dtype=np.float64)))


__all__ = ["decoder_support_info", "replay_decode", "run_task5_decoder_replays"]
