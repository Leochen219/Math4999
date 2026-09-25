"""Concrete installed-Cosmos loader for the Task 6 official driver.

This adapter reuses the already-validated Task 5/precision environment setup;
it is intentionally not a second implementation of model construction.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _json_safe_action(action: Any) -> list[list[float]]:
    """Validate the official action shape and convert it to JSON primitives."""
    try:
        raw = np.asarray(action)
        if (raw.shape != (16, 10) or not np.issubdtype(raw.dtype, np.number) or
                np.issubdtype(raw.dtype, np.bool_) or np.iscomplexobj(raw)):
            raise ValueError
    except (TypeError, ValueError) as error:
        raise ValueError("official action must be finite numeric values with shape (16, 10)") from error
    if not np.all(np.isfinite(raw)):
        raise ValueError("official action must be finite numeric values with shape (16, 10)")
    # ``tolist`` converts NumPy scalars to JSON-native Python numbers while
    # retaining the factory's values; no FP32 narrowing is allowed here.
    return [[item for item in row] for row in raw.tolist()]


def _content_identity(value: Any) -> dict[str, Any]:
    """Return a non-class-only identity for a loaded encoder."""
    method = getattr(value, "actual_identity", None) or getattr(value, "content_identity", None)
    if callable(method):
        observed = method()
        if not isinstance(observed, dict) or not observed:
            raise ValueError("condition encoder identity must contain content evidence")
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "content": observed}
    if hasattr(value, "state_dict") or hasattr(value, "named_parameters"):
        try:
            from .umi_precision_identity import fingerprint
        except ImportError:  # pragma: no cover
            from umi_precision_identity import fingerprint
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "fingerprint": fingerprint(value)}
    attrs = getattr(value, "__dict__", None)
    if not isinstance(attrs, dict) or not attrs:
        raise ValueError("condition encoder does not expose a content identity")
    try:
        from .umi_precision_identity import fingerprint
    except ImportError:  # pragma: no cover
        from umi_precision_identity import fingerprint
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "fingerprint": fingerprint(value)}


class ConditionEncoderAdapter:
    """Strict VAE-condition encoder seam used by decoder round trips.

    The official UMI encoder consumes a batched five-dimensional video tensor;
    callers of the decoder store a single CHW float32 frame.  This adapter is
    the only conversion boundary and clears encoder caches for every request.
    """
    def __init__(self, encoder: Any, *, device: str = "cuda"):
        if encoder is None:
            raise ValueError("official model has no tokenizer_vision encoder")
        self.encoder = encoder
        self.device = str(device)
        self._identity = _content_identity(encoder)
        if not any(callable(getattr(encoder, name, None)) for name in ("encode", "encode_image", "__call__")):
            raise ValueError("condition encoder must expose encode/encode_image/call")
        self.reset_cache()

    def actual_identity(self) -> dict[str, Any]:
        # Recompute after reset so a changed model or VAE cannot pass resume.
        observed = _content_identity(self.encoder)
        return {"adapter": "ConditionEncoderAdapter", "device": self.device,
                "encoder": observed, "bound": self._identity}

    def _clear_one(self, owner: Any) -> None:
        for name in ("reset_cache", "clear_cache", "clear"):
            method = getattr(owner, name, None)
            if callable(method):
                method()
                return

    def reset_cache(self) -> None:
        self._clear_one(self.encoder)
        for name in ("_enc_cache", "_dec_cache", "cache", "_cache"):
            value = getattr(self.encoder, name, None)
            if hasattr(value, "clear"):
                value.clear()
            if value is not None and hasattr(value, "__len__") and len(value) != 0:
                raise RuntimeError("condition encoder cache did not clear")

    @staticmethod
    def _first_tensor(value: Any) -> Any:
        if hasattr(value, "detach") and hasattr(value, "shape"):
            return value
        if isinstance(value, dict):
            for item in value.values():
                try:
                    return ConditionEncoderAdapter._first_tensor(item)
                except TypeError:
                    continue
        if isinstance(value, (list, tuple)):
            for item in value:
                try:
                    return ConditionEncoderAdapter._first_tensor(item)
                except TypeError:
                    continue
        for name in ("latent", "latents", "sample"):
            item = getattr(value, name, None)
            if item is not None:
                return ConditionEncoderAdapter._first_tensor(item)
        if isinstance(value, np.ndarray):
            return value
        raise TypeError("condition encoder returned no tensor")

    def __call__(self, frame: Any) -> Any:
        import torch
        array = np.asarray(frame, dtype=np.float32)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 3:
            raise ValueError("condition frame must be CHW or HWC")
        if array.shape[0] not in (1, 3) and array.shape[-1] in (1, 3):
            array = np.transpose(array, (2, 0, 1))
        if array.shape[0] not in (1, 3) or not np.all(np.isfinite(array)) or np.min(array) < 0 or np.max(array) > 1:
            raise ValueError("condition frame must be finite [0,1] CHW")
        value = torch.from_numpy(np.ascontiguousarray(array)).to(device=self.device, dtype=torch.float32)[None, :, None, :, :]
        # Cosmos normalizes RGB immediately before VAE encoding. Keep report
        # inputs in [0,1], but prove the actual encoder boundary receives the
        # corresponding [-1,1] tensor.
        value = value.mul(2.0).sub(1.0)
        self.reset_cache()
        primary_error = None
        try:
            method = getattr(self.encoder, "encode", None) or getattr(self.encoder, "encode_image", None)
            if method is None:
                method = self.encoder
            try:
                encoded = method(value)
            except (TypeError, ValueError) as first:
                # A few installed tokenizer wrappers expose the same encoder
                # with a keyword-only device argument.
                try:
                    encoded = method(value, device=self.device)
                except TypeError:
                    primary_error = first
                    raise
            result = self._first_tensor(encoded)
            if isinstance(result, np.ndarray):
                result = torch.from_numpy(result)
            result = result.detach().to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(result).all():
                raise ValueError("condition encoder produced nonfinite latent")
            return result
        finally:
            try:
                self.reset_cache()
            except BaseException:
                if primary_error is None:
                    raise



def build_loader_args(*, framework_root: str | Path, checkpoint: str | Path, vae: str | Path,
                      video: str | Path, prompt: str, action: Any, setup_dir: str | Path,
                      phase: str, resume: bool) -> SimpleNamespace:
    """Build the exact sample namespace without importing Cosmos."""
    setup = Path(setup_dir).resolve()
    return SimpleNamespace(framework_root=str(Path(framework_root).resolve()), checkpoint_path=str(Path(checkpoint).resolve()),
        vae_path=str(Path(vae).resolve()), input_path=str(Path(video).resolve()) if video else None,
        action_path=None, prompt=str(prompt), action_chunk_index=0, gpu_index=0, direction_seed=20260912,
        model_seed=0, alphas=[0.001, 0.003, 0.01], num_steps=30, sampler="unipc", precision="bfloat16",
        parallelism_preset="latency", diffusion_cache=False, batch_size=1, fps=5,
        run_dir=str(setup), resume=bool(resume), phase=str(phase), stage_a_only=False, use_torch_compile=False)


def _make_setup_dir(run_dir: str | Path | None, phase: str, *, resume: bool) -> Path:
    if not run_dir:
        raise ValueError("official loader requires the current run_dir")
    root = Path(run_dir).resolve()
    setup = root.parent / f".{root.name}.task6_setup" / str(phase)
    if setup.exists() and not resume:
        raise FileExistsError(f"phase setup already exists: {setup}")
    setup.mkdir(parents=True, exist_ok=bool(resume))
    return setup


def _framework_commit(framework: Path) -> str:
    try:
        head = subprocess.run(["git", "-C", str(framework), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip().lower()
        dirty = subprocess.run(["git", "-C", str(framework), "status", "--porcelain"], check=True,
                               capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("framework root is not a readable git checkout") from error
    if len(head) != 40 or dirty:
        raise ValueError("framework checkout is dirty or has no valid HEAD")
    return head


def resolve_bridge_fps(sample_args: Any) -> int:
    """Validate the fps resolved by OmniSampleOverrides, not a literal."""
    try:
        value = int(getattr(sample_args, "fps"))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("resolved official sample args do not expose fps") from error
    if value != 5:
        raise ValueError(f"Bridge sample fps must resolve to 5, got {value}")
    return value


def _scalar_int(value: Any) -> int:
    """Read the scalar domain id stored in the official batch list."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("official domain_id must contain exactly one value")
        return _scalar_int(value[0])
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value)


def _sha256_bytes(value: Any) -> str:
    raw = value.detach().cpu().contiguous().numpy() if hasattr(value, "detach") else np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()


def _int_list(value: Any) -> list[int]:
    """Convert an official tensor/list metadata field to JSON-safe integers."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    values = np.asarray(value).reshape(-1).tolist()
    if not values:
        raise ValueError("official image_size metadata is empty")
    return [int(item) for item in values]


def _load_task6_bridge_data_batch(adapter: Any, args: Any, run_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Build Task 6's Bridge batch without changing the shared Task 4/5 seam.

    The official generic loader reads the raw media and selects a rectangular
    target bucket.  Task 6's Bridge contract instead uses frame zero only,
    preserves uint8 values, and applies the official reflection padding to a
    fixed square before calling the official action batch builder.
    """
    from cosmos_framework.inference.action import _load_actions, build_action_batch
    from cosmos_framework.inference.args import ModelMode, OmniSampleOverrides
    from cosmos_framework.inference.vision import read_media_frames
    from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
    from cosmos_framework.data.generator.action.utils.transforms import reflection_pad_to_target

    if not args.input_path or not args.action_path:
        raise ValueError("--input-path and --action-path are required for official execution")
    sample = {
        "name": "umi_task6_bridge", "model_mode": "forward_dynamics", "domain_name": "bridge_orig_lerobot",
        "view_point": "ego_view", "fps": int(getattr(args, "fps", 5)), "image_size": 256, "action_chunk_size": 16,
        "prompt": args.prompt, "vision_path": str(Path(args.input_path).resolve()),
        "action_path": str(Path(args.action_path).resolve()), "seed": 0, "guidance": 1.0, "shift": 10.0,
    }
    overrides = OmniSampleOverrides.model_validate(sample)
    overrides.output_dir = run_dir / "inputs"
    overrides.download(run_dir / "inputs")
    sample_args = overrides.build_sample(model_config=adapter.pipeline.model_config)
    adapter.sample_args = sample_args
    adapter.sample_settings = {
        "guidance": float(getattr(sample_args, "guidance", 1.0)), "shift": float(getattr(sample_args, "shift", 10.0)),
        "guidance_interval": getattr(sample_args, "guidance_interval", None), "has_negative_prompt": bool(getattr(sample_args, "has_negative_prompt", False)),
        "skip_text_tokens_for_cfg": bool(getattr(sample_args, "skip_text_tokens_for_cfg", False)),
        "normalize_cfg": bool(getattr(sample_args, "normalize_cfg", False)),
        "use_batched_cfg": bool(getattr(sample_args, "use_batched_cfg", False)), "sampler": getattr(adapter.model, "fixed_step_sampler", None),
    }
    if adapter.sample_settings["guidance"] != 1.0 or adapter.sample_settings["shift"] != 10.0:
        raise ValueError("resolved official sample settings must be guidance=1.0 and shift=10.0")

    frames, _ = read_media_frames(Path(args.input_path), max_frames=17)
    first_frame = frames[:, :1]
    if tuple(first_frame.shape) != (3, 1, 480, 640):
        raise ValueError(f"official Bridge first frame must be (3, 1, 480, 640), got {tuple(first_frame.shape)}")
    if getattr(first_frame, "dtype", None) is not None and str(first_frame.dtype) not in {"torch.uint8", "uint8"}:
        raise ValueError("official Bridge input frame must remain uint8")
    pad_dict = {"video": first_frame}
    reflection_pad_to_target(pad_dict, ["video"], keep_aspect_ratio=True, target_w=256, target_h=256)
    processed = pad_dict["video"]
    if tuple(processed.shape) != (3, 1, 256, 256):
        raise ValueError(f"official Bridge processed frame must be (3, 1, 256, 256), got {tuple(processed.shape)}")
    if str(processed.dtype) not in {"torch.uint8", "uint8"}:
        raise ValueError("official Bridge processed frame must remain uint8")
    action, raw_action_dim = _load_actions(Path(args.action_path), ModelMode.FORWARD_DYNAMICS, 16,
                                            int(adapter.model.config.max_action_dim), 10)
    if int(raw_action_dim) != 10:
        raise ValueError(f"official Bridge raw action dim must be 10, got {raw_action_dim}")
    data_batch = build_action_batch(
        video=processed, action=action, raw_action_dim=raw_action_dim, prompt=args.prompt,
        view_point="ego_view", domain_name="bridge_orig_lerobot", model_mode=ModelMode.FORWARD_DYNAMICS,
        action_chunk_size=16, fps=int(getattr(args, "fps", 5)), resolution="256",
        input_video_key=adapter.model.input_video_key, batch_size=1, device="cuda",
    )
    expected_domain_id = int(get_domain_id("bridge_orig_lerobot"))
    if expected_domain_id != 7:
        raise ValueError(f"official Bridge domain mapping must resolve to 7, got {expected_domain_id}")
    actual_domain_id = _scalar_int(data_batch.get("domain_id"))
    if actual_domain_id != expected_domain_id:
        raise ValueError(f"official Bridge data batch domain_id must be {expected_domain_id}, got {actual_domain_id}")
    final_image_size = _int_list(data_batch.get("image_size"))
    if final_image_size != [256, 256, 256, 256]:
        raise ValueError(f"official Bridge final image_size must be [256, 256, 256, 256], got {final_image_size}")
    pad_image_size = _int_list(pad_dict.get("image_size"))
    if pad_image_size != [256, 256, 192, 256]:
        raise ValueError(f"official Bridge preprocessed image_size must be [256, 256, 192, 256], got {pad_image_size}")
    evidence = {
        "input_frame_sha256": _sha256_bytes(first_frame), "processed_frame_sha256": _sha256_bytes(processed),
        "input_frame_shape": list(first_frame.shape), "processed_frame_shape": list(processed.shape),
        "source_hw": [int(first_frame.shape[-2]), int(first_frame.shape[-1])],
        "resized_content_hw": pad_image_size[2:], "preprocessed_image_size": pad_image_size,
        "official_final_image_size": final_image_size,
        "domain_name": "bridge_orig_lerobot", "domain_id": expected_domain_id,
        "data_batch_domain_id": actual_domain_id, "raw_action_dim": int(raw_action_dim),
    }
    return data_batch, sample_args, evidence


def load_task6_cosmos_runtime(*, framework_root: str, checkpoint: str, vae: str, device: str,
                              model_seed: int, prompt: str, action: Any, video: str | None = None,
                              contract: dict[str, Any] | None = None, run_dir: str | Path | None = None,
                              phase: str = "unknown", resume: bool = False) -> dict[str, Any]:
    if device != "cuda:0" or int(model_seed) != 0:
        raise ValueError("Task 6 official loader is pinned to cuda:0/model seed 0")
    framework = Path(framework_root).resolve()
    for path in (framework, Path(checkpoint), Path(vae)):
        if not path.exists():
            raise FileNotFoundError(path)
    if str(framework) not in sys.path:
        sys.path.insert(0, str(framework))
    # These imports are the exact Task 4/5 verified setup and data seams.
    try:
        from .umi_fd_post_vae_scan import load_official_runtime
        from .umi_precision_official import OfficialPrecisionRuntime, TorchOps
    except ImportError:  # pragma: no cover
        from umi_fd_post_vae_scan import load_official_runtime
        from umi_precision_official import OfficialPrecisionRuntime, TorchOps
    setup_dir = _make_setup_dir(run_dir, phase, resume=resume)
    args = build_loader_args(framework_root=framework, checkpoint=checkpoint, vae=vae, video=video,
                             prompt=prompt, action=action, setup_dir=setup_dir, phase=phase, resume=resume)
    # The official sample loader reads the paired action from a path.  Materialize
    # the exact action passed by the factory in the temporary setup directory.
    import json
    action_path = setup_dir / "action.json"
    if action_path.exists() and not resume:
        raise FileExistsError(f"phase setup action already exists: {action_path}")
    if not action_path.exists():
        temporary = setup_dir / ".action.json.tmp"
        temporary.write_text(json.dumps(_json_safe_action(action), sort_keys=True, allow_nan=False), encoding="utf-8")
        os.replace(temporary, action_path)
    args.action_path = str(action_path)
    post_adapter = load_official_runtime(args, setup_dir)
    data_batch, sample_args, input_evidence = _load_task6_bridge_data_batch(post_adapter, args, setup_dir)
    resolved_fps = resolve_bridge_fps(sample_args)
    ops = TorchOps()
    provenance = {"framework_root": str(framework), "framework_commit": _framework_commit(framework), "checkpoint_path": str(Path(checkpoint).resolve()),
        "vae_path": str(Path(vae).resolve()), "sampler": "unipc", "precision": "bfloat16",
        "asset_source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d",
        "diffusion_cache_requested": False, "diffusion_cache_installed": False, "seed": 0, "prompt": prompt,
        "fps": resolved_fps, **input_evidence}
    runtime_model = post_adapter.model
    # The verified UMI VAE path owns both encode/decode on tokenizer_vision_gen;
    # bind that object first so round-trip conditions use the same artifact as
    # the generated latent.  tokenizer_vision is only a compatibility fallback
    # for older framework revisions and must still pass the adapter checks.
    condition_encoder = getattr(runtime_model, "tokenizer_vision_gen", None)
    if condition_encoder is None:
        condition_encoder = getattr(runtime_model, "tokenizer_vision", None)
    runtime = {"model": runtime_model, "data_batch": data_batch,
        "fps": resolved_fps,
        "generation_settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0},
        "artifact_paths": {"checkpoint": str(Path(checkpoint).resolve()), "decoder": str(Path(vae).resolve())},
        "provenance": {**provenance, "condition_encoder": "tokenizer_vision_gen" if getattr(runtime_model, "tokenizer_vision_gen", None) is not None else "tokenizer_vision"},
        "encoder": ConditionEncoderAdapter(condition_encoder, device=device)}
    def unload():
        cleanup = getattr(post_adapter, "cleanup", None)
        if callable(cleanup): cleanup()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception:
            pass
    runtime["unload"] = unload
    return runtime


__all__ = ["ConditionEncoderAdapter", "build_loader_args", "load_task6_cosmos_runtime", "resolve_bridge_fps"]
