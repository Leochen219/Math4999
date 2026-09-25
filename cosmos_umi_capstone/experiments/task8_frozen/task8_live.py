"""Guarded Task 8 production loader and one-call baseline executor.

The module has no Torch/Cosmos imports at import time.  ``load_task8_live`` is
the sole model-construction seam and is reachable only with ``release=True``.
It loads one official runtime, builds a 256-square Bridge batch from the
admitted float32 preflight bundle, and delegates request-local G/D/E evidence
to the reviewed Task 7 ``FeedbackRuntime``.  The CLI is therefore safe to
inspect or test without starting a model process.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import traceback
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .umi_task8_runtime import (ActionRoutingError, Task8InputAdapter,
                                    action_token_hash, refresh_official_prepared_action,
                                    validate_action_consumption)
except ImportError:  # pragma: no cover - direct experiments import
    from umi_task8_runtime import (ActionRoutingError, Task8InputAdapter,
                                   action_token_hash, refresh_official_prepared_action,
                                   validate_action_consumption)


class LiveBlocked(RuntimeError):
    """The live path was requested without an explicit release gate."""


class LiveResourceStop(RuntimeError):
    """A fail-closed resource sample stopped the live path."""


def _sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(repr(tuple(int(dim) for dim in array.shape)).encode("ascii")); digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": _sha(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite evidence cannot be serialized")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, marker: str) -> None:
    """Write a status/evidence object atomically, including failure states."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{marker}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _required_path(value: str | Path, name: str, *, directory: bool = False,
                   allow_directory: bool = False) -> Path:
    path = Path(value).resolve()
    valid = path.is_dir() if directory else (path.is_file() or (allow_directory and path.is_dir()))
    if not valid:
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _loader_args(*, framework_root: Path, checkpoint: Path, vae: Path,
                 prompt: str, setup_dir: Path, resume: bool) -> SimpleNamespace:
    """Build the exact loader namespace without importing the framework."""
    return SimpleNamespace(
        framework_root=str(framework_root), checkpoint_path=str(checkpoint), vae_path=str(vae),
        input_path=None, action_path=None, prompt=str(prompt), action_chunk_index=0, gpu_index=0,
        direction_seed=20260912, model_seed=0, alphas=[0.001, 0.003, 0.01], num_steps=30,
        sampler="unipc", precision="bfloat16", parallelism_preset="latency", diffusion_cache=False,
        batch_size=1, fps=5, run_dir=str(setup_dir), resume=bool(resume), phase="task8",
        stage_a_only=False, use_torch_compile=False,
    )


def _check_runtime_environment(torch: Any) -> dict[str, Any]:
    version = str(getattr(torch, "__version__", ""))
    cuda = str(getattr(getattr(torch, "version", None), "cuda", ""))
    if not version.startswith("2.10") or not cuda.startswith("13.0"):
        raise LiveBlocked(f"Task 8 requires Torch 2.10/cu130, observed torch={version!r}, cuda={cuda!r}")
    if not bool(torch.cuda.is_available()) or int(torch.cuda.current_device()) != 0:
        raise LiveBlocked("Task 8 requires CUDA device 0")
    return {"torch": version, "cuda": cuda, "device": int(torch.cuda.current_device()),
            "device_name": str(torch.cuda.get_device_name(0))}


def load_task8_official_model(*, framework_root: str | Path, checkpoint: str | Path,
                              vae: str | Path, run_dir: str | Path,
                              prompt: str, release: bool = False, resume: bool = False) -> Any:
    """Load exactly one official model adapter after explicit release.

    This function intentionally does not build a data batch.  That keeps model
    construction one-shot while allowing each request to use the 256x256
    condition frame selected by the formal call plan.
    """
    if not release:
        raise LiveBlocked("Task 8 model loading requires main's explicit release")
    framework = _required_path(framework_root, "framework root", directory=True)
    checkpoint_path = _required_path(checkpoint, "checkpoint", allow_directory=True)
    vae_path = _required_path(vae, "VAE")
    root = Path(run_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
    setup_dir = root.parent / f".{root.name}.task8_setup"
    if setup_dir.exists() and not resume and any(setup_dir.iterdir()):
        raise FileExistsError(f"official loader setup already exists: {setup_dir}")
    setup_dir.mkdir(parents=True, exist_ok=True)
    if str(framework) not in sys.path:
        sys.path.insert(0, str(framework))
    import torch  # lazy: no model/GPU import before release
    environment = _check_runtime_environment(torch)
    try:
        from .umi_fd_post_vae_scan import load_official_runtime
    except ImportError:  # pragma: no cover
        from umi_fd_post_vae_scan import load_official_runtime
    args = _loader_args(framework_root=framework, checkpoint=checkpoint_path, vae=vae_path,
                        prompt=prompt, setup_dir=setup_dir, resume=resume)
    adapter = load_official_runtime(args, setup_dir)
    model = getattr(adapter, "model", None)
    if model is None:
        raise LiveBlocked("official loader returned no resident model")
    # These are contract evidence, not a second model conversion.  The
    # reviewed Task 7 C path temporarily uses FP32 for actual G/D/E calls and
    # restores the model's native wrapper after every request.
    adapter.task8_environment = environment
    adapter.task8_contract = {"device": "cuda:0", "batch_size": 1, "sampler": "UniPC",
                              "steps": 30, "guidance": 1.0, "shift": 10.0,
                              "diffusion_cache": False, "autocast": False, "tf32": False,
                              "GDE_precision": "float32"}
    if bool(getattr(model, "_diffusion_cache_installed", False)):
        raise LiveBlocked("official loader installed diffusion cache")
    return adapter


def build_task8_data_batch(adapter: Any, batch: Task8InputAdapter, *, frame_index: int,
                           chunk_index: int, device: str = "cuda") -> tuple[Any, dict[str, Any]]:
    """Build the official Bridge action batch directly from admitted 256 RGB."""
    if int(frame_index) not in (0, 16):
        raise ValueError("Task 8 real condition frame must be x0 or x16")
    if int(chunk_index) not in (0, 1):
        raise ValueError("Task 8 action chunk must be 0 or 1")
    frame = np.asarray(batch.rgb[int(frame_index)], dtype=np.float32)
    if frame.shape != (3, 256, 256) or not np.all(np.isfinite(frame)) or frame.min() < 0 or frame.max() > 1:
        raise ValueError("Task 8 RGB adapter must provide finite float32 CHW [0,1] 256-square frames")
    action = np.asarray(batch.actions[int(chunk_index)], dtype=np.float32)
    if action.shape != (16, 10) or not np.all(np.isfinite(action)):
        raise ValueError("Task 8 normalized action chunk must be finite float32 [16,10]")
    import torch
    try:
        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode
        from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
    except ImportError as error:  # pragma: no cover - installed remote only
        raise LiveBlocked("official action batch builder is unavailable") from error
    # The official nested-video path treats float input as already normalized
    # and skips its uint8 conversion. The admitted artifact is RGB [0,1], so
    # perform the one explicit [0,1] -> [-1,1] conversion here and never resize.
    video = torch.from_numpy(np.ascontiguousarray(frame * 2.0 - 1.0)).to(device=device, dtype=torch.float32)[:, None]
    model = adapter.model
    max_action_dim = int(getattr(getattr(model, "config", None), "max_action_dim", 10))
    if max_action_dim < 10:
        raise LiveBlocked(f"official max_action_dim={max_action_dim} cannot carry Bridge raw dim 10")
    padded_action = np.zeros((16, max_action_dim), dtype=np.float32)
    padded_action[:, :10] = action
    action_tensor = torch.from_numpy(np.ascontiguousarray(padded_action)).to(device=device, dtype=torch.float32)
    data_batch = build_action_batch(
        video=video, action=action_tensor, raw_action_dim=10, prompt=batch.prompt,
        view_point="ego_view", domain_name="bridge_orig_lerobot", model_mode=ModelMode.FORWARD_DYNAMICS,
        action_chunk_size=16, fps=5, resolution="256", input_video_key=model.input_video_key,
        batch_size=1, device=device,
    )
    video_key = str(model.input_video_key)
    # ``build_action_batch`` returns the legacy nested shape ``[[tensor]]``.
    # The official prepare seam treats a one-item nested list as a raw uint8
    # sample, even when its tensor is float32.  Normalize this boundary here
    # to the documented preprocessed shape ``[tensor[B,C,T,H,W]]`` and mark
    # it explicitly; this preserves the official pixels while selecting its
    # float32 [-1, 1] path.  Do not rebuild or cast any other batch fields.
    data_batch[video_key] = [_canonical_preprocessed_video(data_batch.get(video_key), video_key)]
    data_batch["is_preprocessed"] = True
    # Read back the official batch boundary so a future framework change
    # cannot silently resize or reinterpret the already-square float input.
    video_values: list[Any] = []
    def collect_video(value: Any) -> None:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            video_values.append(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                collect_video(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect_video(child)
    collect_video(data_batch.get(video_key))
    candidates = [tuple(int(dim) for dim in getattr(value, "shape", ())) for value in video_values]
    expected_video = np.repeat((frame * 2.0 - 1.0)[:, None], 17, axis=1)
    matching = []
    for value, shape in zip(video_values, candidates):
        if shape == (3, 17, 256, 256):
            observed = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            if np.array_equal(np.asarray(observed, dtype=np.float32), expected_video):
                matching.append(shape)
        elif shape == (1, 3, 17, 256, 256):
            observed = value.detach().cpu().numpy()[0] if hasattr(value, "detach") else np.asarray(value)[0]
            if np.array_equal(np.asarray(observed, dtype=np.float32), expected_video):
                matching.append(shape)
    if not matching:
        raise LiveBlocked(f"official batch did not preserve repeated 17-frame 256-square video under {video_key}: {candidates}")
    domain_id = int(get_domain_id("bridge_orig_lerobot"))
    observed_domain = data_batch.get("domain_id")
    if hasattr(observed_domain, "detach"):
        observed_domain = observed_domain.detach().cpu().reshape(-1).tolist()
    elif isinstance(observed_domain, (list, tuple)):
        observed_domain = list(observed_domain)
    else:
        observed_domain = [int(observed_domain)]
    if observed_domain != [domain_id]:
        raise LiveBlocked(f"official Bridge domain id differs: expected {[domain_id]}, got {observed_domain}")
    evidence = {"frame_index": int(frame_index), "chunk_index": int(chunk_index),
                "frame_shape": list(frame.shape), "frame_dtype": "float32", "frame_range": "[0,1]",
                "official_video_key": video_key, "official_video_shape": list(matching[0]),
                "official_video_range": "[-1,1]", "official_video_repeated_frames": 17,
                "action_shape": list(action.shape), "official_action_shape": list(padded_action.shape),
                "action_dtype": "float32", "action_hash": action_token_hash(action),
                "domain_name": "bridge_orig_lerobot", "domain_id": domain_id, "fps": 5,
                "resolution": "256", "batch_size": 1}
    return data_batch, evidence


def _canonical_preprocessed_video(value: Any, video_key: str) -> Any:
    """Return the official prepare seam's float32 ``[B,C,T,H,W]`` tensor.

    ``build_action_batch`` currently emits ``[[C,T,H,W]]``.  The model's
    ``_normalize_video_databatch_inplace`` only recognizes a preprocessed
    float tensor when the sample list contains the tensor itself, not another
    list.  This helper unwraps exactly that one-sample structure and adds the
    batch dimension without changing dtype or values.
    """
    item = value
    if isinstance(item, (list, tuple)):
        if len(item) != 1:
            raise LiveBlocked(f"official video batch has unexpected sample count under {video_key}: {len(item)}")
        item = item[0]
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise LiveBlocked(f"official video batch has unexpected nested sample count under {video_key}: {len(item)}")
            item = item[0]
    if not hasattr(item, "shape") or not hasattr(item, "unsqueeze"):
        raise LiveBlocked(f"official video batch under {video_key} is not a tensor")
    shape = tuple(int(dim) for dim in item.shape)
    if shape == (3, 17, 256, 256):
        item = item.unsqueeze(0)
    elif shape != (1, 3, 17, 256, 256):
        raise LiveBlocked(f"official video batch under {video_key} has unexpected shape {shape}")
    if str(getattr(item, "dtype", "")) not in {"torch.float32", "float32"}:
        raise LiveBlocked(f"official video batch under {video_key} has dtype {getattr(item, 'dtype', None)!r}")
    return item


def _prepared_geometry(runtime: Any) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    try:
        from .umi_fd_post_vae_scan import _clone_runtime
        from .umi_precision_official import projection
    except ImportError:  # pragma: no cover
        from umi_fd_post_vae_scan import _clone_runtime
        from umi_precision_official import projection
    with runtime.ops.inference():
        prepared = runtime.model._prepare_inference_data(_clone_runtime(runtime.data_batch), [0], False)
    carrier = np.ascontiguousarray(projection(prepared[1].x0_tokens_vision[0]), dtype=np.float32)
    mask = np.ascontiguousarray(projection(prepared[6][0]).reshape(carrier.shape), dtype=bool)
    indexes = tuple(int(index) for index in prepared[0][0].condition_frame_indexes_vision)
    return carrier, mask, indexes


@dataclass
class Task8LiveContext:
    adapter: Any
    runtime: Any
    feedback: Any
    encoder: Any
    batch: Task8InputAdapter
    environment: Mapping[str, Any]
    contract: Mapping[str, Any]

    def cleanup(self) -> None:
        errors: list[BaseException] = []
        for method_name in ("_reset", "_check_caches", "release_generation_state", "cleanup_generation"):
            method = getattr(self.runtime, method_name, None)
            if callable(method):
                try:
                    method()
                except BaseException as error:
                    errors.append(error)
        model = getattr(self.runtime, "model", None)
        for owner in (getattr(model, "tokenizer_vision_gen", None), getattr(model, "tokenizer_vision", None)):
            clear = getattr(owner, "clear_cache", None)
            if callable(clear):
                try:
                    clear()
                except BaseException as error:
                    errors.append(error)
        pipeline = getattr(self.adapter, "pipeline", None)
        for owner in (pipeline, self.adapter):
            for name in ("cleanup", "unload", "close"):
                method = getattr(owner, name, None)
                if callable(method):
                    try:
                        method()
                    except BaseException as error:
                        errors.append(error)
        gc.collect()
        try:
            import torch
            if bool(torch.cuda.is_available()):
                torch.cuda.empty_cache()
        except Exception:
            errors.append(sys.exc_info()[1] or RuntimeError("CUDA cleanup failed"))
        if errors:
            raise RuntimeError("Task 8 cleanup failed") from errors[0]


def load_task8_live(*, framework_root: str | Path, checkpoint: str | Path,
                    vae: str | Path, run_dir: str | Path, batch: Task8InputAdapter,
                    release: bool = False, resume: bool = False) -> Task8LiveContext:
    """Load one model and bind the reviewed Task 7 feedback runtime."""
    if not release:
        raise LiveBlocked("Task 8 live runtime requires main's explicit release")
    adapter = load_task8_official_model(framework_root=framework_root, checkpoint=checkpoint,
                                        vae=vae, run_dir=run_dir, prompt=batch.prompt,
                                        release=release, resume=resume)
    try:
        # The initial batch is only used to obtain the official 8-field carrier
        # and geometry.  Calls refresh actions request-locally below.
        data_batch, batch_evidence = build_task8_data_batch(adapter, batch, frame_index=0, chunk_index=0)
        try:
            from .umi_precision_official import OfficialPrecisionRuntime, TorchOps
            from .umi_task7_encoder import FeedbackEncoder
            from .umi_task7_runtime import FeedbackRuntime
        except ImportError:  # pragma: no cover
            from umi_precision_official import OfficialPrecisionRuntime, TorchOps
            from umi_task7_encoder import FeedbackEncoder
            from umi_task7_runtime import FeedbackRuntime
        ops = TorchOps()
        # OfficialPrecisionRuntime is the single resident runtime; its C
        # execution boundary is temporarily FP32 for G/D/E and restores state.
        carrier, mask, indexes = _prepared_geometry(SimpleNamespace(
            model=adapter.model, data_batch=data_batch, ops=ops,
        ))
        # PrecisionInputs requires a unit-RMS direction even though Task 8
        # never asks the runtime to perturb it. Keep the placeholder entirely
        # inside the condition mask and bind it as explicit setup evidence.
        directions = np.zeros((1,) + carrier.shape, dtype=np.float32)
        if not np.any(mask):
            raise LiveBlocked("official condition mask is empty")
        directions[0][mask] = np.float32(1.0)
        directions[0][mask] /= np.float32(np.sqrt(np.mean(directions[0][mask].astype(np.float64) ** 2)))
        model = adapter.model
        provenance = {"framework_root": str(Path(framework_root).resolve()),
                      "checkpoint_path": str(Path(checkpoint).resolve()), "vae_path": str(Path(vae).resolve()),
                      "prompt": batch.prompt, "sampler": "unipc", "precision": "float32",
                      "diffusion_cache_requested": False, "diffusion_cache_installed": False,
                      "task8_initial_batch": batch_evidence,
                      "direction_bank": "unit condition-mask placeholder; never used for formal perturbation"}
        runtime = OfficialPrecisionRuntime(
            model, data_batch, directions, provenance=provenance, ops=ops,
            generation_settings={"num_steps": 30, "guidance": 1.0, "shift": 10.0},
            artifact_paths={"checkpoint": str(Path(checkpoint).resolve()), "decoder": str(Path(vae).resolve())},
        )
        encoder_model = getattr(model, "tokenizer_vision_gen", None) or getattr(model, "tokenizer_vision", None)
        encoder = FeedbackEncoder(encoder_model, device="cuda")
        feedback = FeedbackRuntime(runtime, encoder=encoder, z0=carrier, mask=mask,
                                   condition_indexes=indexes, v0=np.zeros_like(carrier), seed=0)
        context = Task8LiveContext(adapter, runtime, feedback, encoder, batch,
                                   getattr(adapter, "task8_environment", {}),
                                   getattr(adapter, "task8_contract", {}))
        return context
    except BaseException as error:
        cleanup_errors: list[BaseException] = []
        for owner in (getattr(adapter, "pipeline", None), adapter):
            for name in ("cleanup", "unload", "close"):
                method = getattr(owner, name, None)
                if callable(method):
                    try:
                        method()
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
        for cleanup_error in cleanup_errors:
            error.add_note("live setup cleanup failed: " + repr(cleanup_error))
        raise


class Task8LiveExecutor:
    """Formal/smoke executor over one ``Task8LiveContext``."""

    def __init__(self, context: Task8LiveContext, *, resource_monitor: Any | None = None,
                 resource_phase: str = "formal"):
        self.context = context
        self.resource_monitor = resource_monitor
        self.resource_phase = str(resource_phase)
        self._last_g0_frame: np.ndarray | None = None

    def _condition_frame(self, spec: Mapping[str, Any]) -> np.ndarray:
        source = str(spec.get("condition_source", ""))
        if source == "real_x0":
            return np.array(self.context.batch.rgb[0], copy=True)
        if source == "real_x16":
            return np.array(self.context.batch.rgb[16], copy=True)
        if source == "g0_float_last_fp32":
            if self._last_g0_frame is None:
                raise LiveBlocked("AR2 requested before a verified G0 decoded frame")
            return np.array(self._last_g0_frame, copy=True)
        raise ValueError(f"unknown Task 8 condition source: {source}")

    def __call__(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        if self.resource_monitor is not None:
            self.resource_monitor.check(phase=self.resource_phase, starting_new_sample=True)
        frame = self._condition_frame(spec)
        encoded_input = self.context.encoder.encode(frame, precision="temporary_fp32")
        arrays = encoded_input.get("arrays", {}) if isinstance(encoded_input, Mapping) else {}
        condition = arrays.get("actual_output")
        if not isinstance(condition, np.ndarray) or condition.dtype != np.float32:
            raise LiveBlocked("FP32 condition encoder did not expose actual_output")
        chunk_index = int(spec["chunk_index"])
        action = np.asarray(self.context.batch.actions[chunk_index], dtype=np.float32)
        # FeedbackRuntime clones the official prepared tuple for each request.
        # Refresh the resident template to a request-local clone first; this is
        # the seam that prevents the cached Task 7 action from leaking into a1.
        try:
            from .umi_fd_post_vae_scan import _clone_runtime
        except ImportError:  # pragma: no cover
            from umi_fd_post_vae_scan import _clone_runtime
        base_prepared = self.context.runtime.prepared
        request_prepared = _clone_runtime(base_prepared)
        action_evidence = refresh_official_prepared_action(request_prepared, action, ops=self.context.runtime.ops)
        self.context.runtime.prepared = request_prepared
        model = self.context.runtime.model
        original_denoise = getattr(model, "denoise", None)
        if not callable(original_denoise):
            self.context.runtime.prepared = base_prepared
            raise LiveBlocked("official model denoiser is unavailable")
        consumed: list[str] = []
        had_denoise = "denoise" in getattr(model, "__dict__", {})

        @wraps(original_denoise)
        def observe_action(*args: Any, **kwargs: Any) -> Any:
            if self.resource_monitor is not None:
                self.resource_monitor.check(phase=self.resource_phase, starting_new_sample=False)
            packed = kwargs.get("data_batch_packed")
            if packed is None:
                packed = next((value for value in args if getattr(value, "action", None) is not None), None)
            modality = getattr(packed, "action", None)
            tokens = getattr(modality, "tokens", None)
            if isinstance(tokens, (list, tuple)):
                if len(tokens) != 1:
                    raise ActionRoutingError("official action token batch is not batch-one")
                tokens = tokens[0]
            if tokens is None:
                raise ActionRoutingError("actual packed.action.tokens were not observed")
            consumed.append(action_token_hash(tokens))
            return original_denoise(*args, **kwargs)

        try:
            model.denoise = observe_action
            capture = self.context.feedback.step(condition, int(spec["seed"]))
            expected = action_evidence.get("effective_action", action)
            action_check = validate_action_consumption({"packed_action_token_hashes": consumed}, expected, chunk_index=chunk_index)
        finally:
            if had_denoise:
                model.denoise = original_denoise
            elif "denoise" in getattr(model, "__dict__", {}):
                delattr(model, "denoise")
            self.context.runtime.prepared = base_prepared
        raw = np.asarray(capture["decoder_raw_output"], dtype=np.float32)
        normalized = np.clip((raw.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
        if normalized.ndim == 5 and normalized.shape[0] == 1:
            normalized = normalized[0]
        if normalized.ndim != 4 or normalized.shape[0] != 3 or normalized.shape[1] != 17:
            raise LiveBlocked(f"official decoder did not return full 17-frame RGB output: {normalized.shape}")
        last = np.array(normalized[:, -1], copy=True)
        if str(spec.get("call")) == "G0":
            self._last_g0_frame = last.copy()
        result = {
            "output_full": np.array(capture["full_latent"], copy=True),
            "decoded_rgb_full": normalized,
            "generated_rgb": np.array(normalized[:, 1:], copy=True),
            "decoded_raw": raw,
            "decoded_last_rgb": last,
            "generated_rgb_last": last.copy(),
            "condition_input": np.array(condition, copy=True),
            "condition_input_fp32": np.array(condition, copy=True),
            "condition_actual": np.array(capture["actual"]["last_condition"], copy=True),
            "encoded_condition": np.array(capture["encoded_condition"], copy=True),
            "prediction_noise_hash": capture["prediction_noise_hash"],
            "packed_action_token_hashes": list(consumed), "action_consumption": action_check,
            "action_hash": action_token_hash(action), "action": action.copy(),
            "precision": {"G": "float32", "D": "float32", "E": "float32",
                          "feedback": _json_safe(capture.get("evidence", {})),
                          "encoder_input": _json_safe(encoded_input.get("evidence", {}))},
            # Keep the complete boundary evidence in the atomic sample. The
            # nested arrays are serialized as .npy artifacts by Task8SampleStore.
            "generation": capture.get("generation", {}),
            "decoder": capture.get("decoder", {}),
            "encoder": capture.get("encoder", {}),
            "actual": capture.get("actual", {}),
            "condition_steps_fp32": np.array(capture["actual"]["condition_steps"], copy=True),
            "provenance": {"condition_source": spec.get("condition_source"), "action_source": spec.get("action_source"),
                           "condition_rgb_sha256": _sha(frame), "action_evidence": _json_safe(action_evidence)},
        }
        if self.resource_monitor is not None:
            self.resource_monitor.check(phase=self.resource_phase, starting_new_sample=False)
        return result


class Task8ResourceMonitor:
    def __init__(self, run_dir: str | Path, *, gpu_index: int = 0):
        self.run_dir, self.gpu_index = Path(run_dir), int(gpu_index)
        try:
            from .run_umi_task6_official import resource_samplers
            from .run_umi_task6_experiment import ResourceMonitor
            from .run_umi_task7_experiment import _strict_gpu_sampler
        except ImportError:  # pragma: no cover
            from run_umi_task6_official import resource_samplers
            from run_umi_task6_experiment import ResourceMonitor
            from run_umi_task7_experiment import _strict_gpu_sampler
        samplers = resource_samplers(self.run_dir, gpu_index=self.gpu_index)
        monitor_root = self.run_dir / "monitor" / "task8"
        self._monitor = ResourceMonitor(monitor_root, gpu_sampler=_strict_gpu_sampler(samplers["gpu"], self.gpu_index),
                                        ram_sampler=samplers["ram"], disk_sampler=samplers["disk"])
        self._monitor._task7_strict_gpu_sampler = True
        self.samples: list[dict[str, Any]] = []

    def start(self) -> "Task8ResourceMonitor":
        self._monitor.start()
        try:
            self.check(phase="preload", starting_new_sample=True)
        except BaseException as start_error:
            # ResourceMonitor.start() has already launched its background
            # poller.  If the synchronous preload gate rejects the run, clean
            # it up here: the CLI sets monitor_started only after this method
            # returns, so its outer finally cannot otherwise see the thread.
            try:
                self._monitor.stop()
            except BaseException as cleanup_error:
                add_note = getattr(start_error, "add_note", None)
                if callable(add_note):
                    add_note(f"resource monitor cleanup also failed: {cleanup_error!r}")
            raise
        return self

    def stop(self) -> None:
        self._monitor.stop()

    def abort(self) -> None:
        self._monitor.abort()

    def check(self, *, phase: str, starting_new_sample: bool) -> dict[str, Any]:
        decision = self._monitor.check(phase="pilot", starting_new_sample=starting_new_sample)
        if not isinstance(decision, Mapping) or decision.get("status") == "HARD_STOP":
            raise LiveResourceStop(f"resource monitor hard stop: {decision}")
        snapshot = dict(self._monitor.last_resources)
        if not snapshot:
            raise LiveResourceStop("MONITOR_NO_SNAPSHOT")
        if phase in {"preload", "startup"} and starting_new_sample and float(snapshot.get("gpu_used_gib", 0.0)) > 1.0:
            raise LiveResourceStop("GPU_START_USED_HIGH")
        if phase in {"preload", "startup"} and starting_new_sample and float(snapshot.get("ram_available_gib", 0.0)) < 500.0:
            raise LiveResourceStop("RAM_START_AVAILABLE_LOW")
        if phase == "resource-smoke":
            if float(snapshot.get("gpu_peak_allocated_gib", 0.0)) > 35.0:
                raise LiveResourceStop("GPU_SMOKE_PEAK_ALLOCATED_HIGH")
            if float(snapshot.get("gpu_peak_nvml_used_gib", 0.0)) > 45.0:
                raise LiveResourceStop("GPU_SMOKE_PEAK_USED_HIGH")
        disk = float(snapshot["disk_free_gib"])
        if disk < 5.0:
            raise LiveResourceStop("DISK_FREE_LOW: Task 8 retains a 5 GiB reserve")
        if float(snapshot["swap_used_gib"]) > 0:
            raise LiveResourceStop("SWAP_IN_USE")
        snapshot["phase"] = str(phase); snapshot["starting_new_sample"] = bool(starting_new_sample)
        self.samples.append(dict(snapshot))
        return snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-npz", required=True)
    parser.add_argument("--metadata-json", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--framework-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--release", action="store_true", help="main approval; without it no model is loaded")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        batch = Task8InputAdapter.from_preflight(args.inputs_npz, args.metadata_json)
        if not args.release:
            raise LiveBlocked("pass --release only after main approves the engineering smoke")
        if int(args.gpu_index) != 0:
            raise LiveBlocked("Task 8 is pinned to GPU 0")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible not in (None, "0"):
            raise LiveBlocked(f"CUDA_VISIBLE_DEVICES must be 0, observed {visible!r}")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        run_path = Path(args.run_dir).resolve()
        if run_path.exists() and any(run_path.iterdir()) and not args.resume:
            raise LiveBlocked(f"smoke run directory is non-empty; choose the lowest unused directory: {run_path}")
        run_path.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(run_path / "run_status.json", {
            "schema_version": "umi-task8-live-v1",
            "status": "LOADING",
            "generation_started": False,
            "sample_id": "engineering_smoke",
            "provenance": {
                "inputs_npz": str(Path(args.inputs_npz).resolve()),
                "metadata_json": str(Path(args.metadata_json).resolve()),
                "framework_root": str(Path(args.framework_root).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "vae": str(Path(args.vae).resolve()),
                "gpu_index": int(args.gpu_index),
                "release": bool(args.release),
                "resume": bool(args.resume),
                "code": str(Path(__file__).resolve()),
            },
        }, marker="loading")
        monitor = Task8ResourceMonitor(args.run_dir, gpu_index=args.gpu_index)
        context = None
        monitor_started = False
        try:
            monitor.start()
            monitor_started = True
            context = load_task8_live(framework_root=args.framework_root, checkpoint=args.checkpoint,
                                      vae=args.vae, run_dir=args.run_dir, batch=batch,
                                      release=True, resume=args.resume)
            running_status = {"schema_version": "umi-task8-live-v1", "status": "RUNNING",
                              "generation_started": True, "sample_id": "engineering_smoke",
                              "runtime_contract": dict(context.contract), "environment": dict(context.environment),
                              "resource_samples": list(monitor.samples)}
            _write_json_atomic(Path(args.run_dir) / "run_status.json", running_status, marker="running")
            executor = Task8LiveExecutor(context, resource_monitor=monitor, resource_phase="resource-smoke")
            result = executor({"call": "G0", "condition_source": "real_x0", "action_source": "a0",
                               "chunk_index": 0, "seed": 0})
            try:
                from .run_umi_task8_experiment import Task8SampleStore
            except ImportError:  # pragma: no cover
                from run_umi_task8_experiment import Task8SampleStore
            # Task8SampleStore writes arrays as immutable .npy artifacts and a
            # hash manifest; the JSON summary alone is never accepted as a
            # smoke result because it would discard the full tensors.
            Task8SampleStore(Path(args.run_dir) / "samples").write_success("engineering_smoke", result)
            sample_root = Path(args.run_dir) / "samples" / "engineering_smoke"
            measured_bytes = sum(path.stat().st_size for path in sample_root.rglob("*") if path.is_file())
            after = monitor.check(phase="resource-smoke", starting_new_sample=False)
            disk_free = float(after["disk_free_gib"])
            forecast_free = disk_free - 1.3 * 4 * measured_bytes / 2**30
            if forecast_free < 5.0:
                raise LiveResourceStop(f"DISK_FORECAST_LOW: measured smoke footprint leaves {forecast_free:.3f} GiB")
            status = {"schema_version": "umi-task8-live-v1", "status": "ENGINEERING_SMOKE_COMPLETE",
                      "generation_started": True, "sample_id": "engineering_smoke",
                      "runtime_contract": dict(context.contract), "environment": dict(context.environment),
                      "measured_sample_bytes": int(measured_bytes), "disk_free_gib": disk_free,
                      "forecast_free_gib_after_four_calls": forecast_free,
                      "resource_samples": list(monitor.samples)}
            _write_json_atomic(Path(args.run_dir) / "run_status.json", status, marker="complete")
            return 0
        finally:
            cleanup_error: BaseException | None = None
            try:
                if context is not None:
                    context.cleanup()
            except BaseException as error:
                cleanup_error = error
            if monitor_started:
                try:
                    monitor.stop()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if cleanup_error is not None:
                raise cleanup_error
    except Exception as error:
        try:
            status_path = Path(args.run_dir) / "run_status.json"
            prior = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            failure = {"schema_version": "umi-task8-live-v1", "status": "BLOCKED",
                       "generation_started": bool(prior.get("generation_started", False)),
                       "prior_status": prior.get("status"),
                       "error": {"type": type(error).__name__, "message": str(error),
                                 "traceback": traceback.format_exc()}}
            _write_json_atomic(status_path, failure, marker="failed")
        except Exception:
            pass
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 3


__all__ = ["LiveBlocked", "LiveResourceStop", "Task8LiveContext", "Task8LiveExecutor",
           "Task8ResourceMonitor", "build_task8_data_batch", "load_task8_live",
           "load_task8_official_model", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
