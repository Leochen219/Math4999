"""Gated production launcher for one-scene Task 9 FP32 response scans.

Importing this module is CPU-safe. A live Task 8 model is constructed only
after explicit ``--release`` and only from the integrity-checked Task 8
runtime snapshot named by ``--code-root``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .umi_task9_spectrum import build_call_plan, generate_direction_bank, run_samples
except ImportError:  # pragma: no cover - direct experiments import
    from umi_task9_spectrum import build_call_plan, generate_direction_bank, run_samples


TASK9_DISK_RESERVE_BYTES = 5 * 1024**3
DIRECTION_SEED = 20260923
TASK8_FRAMEWORK_COMMIT = "ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5"
TASK8_RUNTIME_MANIFEST_SHA256 = "a14a05755720f7ff5e7fbece1c60f99750853e15a574f0f12f8381559cfda9c6"
TASK8_SNAPSHOT_FILES = frozenset({
    "task8_live.py",
    "umi_task8_runtime.py",
    "umi_task7_runtime.py",
    "umi_task7_encoder.py",
})
TASK8_CONTRACT = {
    "device": "cuda:0",
    "batch_size": 1,
    "sampler": "UniPC",
    "steps": 30,
    "guidance": 1.0,
    "shift": 10.0,
    "diffusion_cache": False,
    "autocast": False,
    "tf32": False,
    "GDE_precision": "float32",
}


class LiveBlocked(RuntimeError):
    """A live run failed one of its explicit release or environment gates."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, help="verified Task 8 runtime snapshot directory")
    parser.add_argument("--data", required=True, help="one-scene Task 8 preflight NPZ")
    parser.add_argument("--metadata-json", required=True, help="companion task8_preflight.json")
    parser.add_argument("--framework", required=True, help="official Cosmos framework root")
    parser.add_argument("--checkpoint", required=True, help="official Cosmos model checkpoint")
    parser.add_argument("--vae", required=True, help="official VAE checkpoint file")
    parser.add_argument("--run-dir", required=True, help="new or explicitly resumed Task 9 run directory")
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1),
                        help="Task 7 paired diffusion seed; separate runs use 0 and 1")
    parser.add_argument("--smoke", action="store_true",
                        help="run a single pre-scan baseline for engineering validation")
    parser.add_argument("--resume", action="store_true", help="resume an identical immutable run")
    parser.add_argument("--release", action="store_true",
                        help="main approval gate; without this flag no model is loaded")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_task8_snapshot(code_root: str | Path) -> dict[str, Any]:
    """Verify every file listed in the immutable Task 8 snapshot manifest."""
    root = Path(code_root).resolve()
    manifest_path = root / "MANIFEST.sha256"
    if not root.is_dir() or not manifest_path.is_file():
        raise LiveBlocked(f"verified Task 8 code root or MANIFEST.sha256 is missing: {root}")
    verified: set[str] = set()
    try:
        lines = manifest_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        raise LiveBlocked(f"cannot read Task 8 runtime manifest: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+(.+?)\s*", line)
        if match is None:
            raise LiveBlocked(f"malformed Task 8 runtime manifest line {line_number}")
        expected, relative_name = match.groups()
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise LiveBlocked(f"unsafe path in Task 8 runtime manifest: {relative_name}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise LiveBlocked(f"Task 8 manifest path escapes code root: {relative_name}") from error
        if not path.is_file() or _sha256_file(path).lower() != expected.lower():
            raise LiveBlocked(f"Task 8 runtime manifest integrity check failed: {relative_name}")
        verified.add(Path(relative_name).as_posix())
    missing = sorted(TASK8_SNAPSHOT_FILES - verified)
    if missing:
        raise LiveBlocked("Task 8 runtime manifest is missing required modules: " + ", ".join(missing))
    manifest_hash = _sha256_file(manifest_path)
    if manifest_hash != TASK8_RUNTIME_MANIFEST_SHA256:
        raise LiveBlocked("Task 8 runtime manifest identity differs from the reviewed R4 snapshot")
    return {"code_root": str(root), "manifest_sha256": manifest_hash,
            "verified_files": len(verified)}


def _load_task8_api(code_root: str | Path) -> Any:
    """Load only the reviewed Task 8 runtime modules after manifest verification."""
    root = Path(code_root).resolve()
    verify_task8_snapshot(root)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    for source_path in root.glob("*.py"):
        existing = sys.modules.get(source_path.stem)
        if existing is None:
            continue
        existing_path = Path(getattr(existing, "__file__", "")).resolve()
        try:
            existing_path.relative_to(root)
        except ValueError as error:
            raise LiveBlocked(f"unverified {source_path.stem} module is already imported from {existing_path}") from error
    required_modules = ("umi_task8_runtime", "umi_task7_runtime", "umi_task7_encoder", "task8_live")
    loaded: dict[str, Any] = {}
    for name in required_modules:
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            raise LiveBlocked(f"Task 8 runtime module is unavailable: {name}: {error}") from error
        module_path = Path(getattr(module, "__file__", "")).resolve()
        try:
            module_path.relative_to(root)
        except ValueError as error:
            raise LiveBlocked(f"Task 8 imported an unverified module outside code root: {module_path}") from error
        loaded[name] = module
    api = loaded["task8_live"]
    for name in ("Task8InputAdapter", "Task8ResourceMonitor", "load_task8_live",
                 "action_token_hash", "refresh_official_prepared_action", "validate_action_consumption"):
        if not callable(getattr(api, name, None)):
            raise LiveBlocked(f"Task 8 live API is missing {name}")
    api.task9_snapshot_identity = verify_task8_snapshot(root)
    return api


def _validate_paths(args: argparse.Namespace) -> dict[str, Path]:
    values = {
        "code_root": Path(args.code_root).resolve(),
        "data": Path(args.data).resolve(),
        "metadata_json": Path(args.metadata_json).resolve(),
        "framework": Path(args.framework).resolve(),
        "checkpoint": Path(args.checkpoint).resolve(),
        "vae": Path(args.vae).resolve(),
        "run_dir": Path(args.run_dir).resolve(),
    }
    for name in ("code_root", "framework"):
        if not values[name].is_dir():
            raise LiveBlocked(f"{name.replace('_', '-')} directory does not exist: {values[name]}")
    for name in ("data", "metadata_json", "vae"):
        if not values[name].is_file():
            raise LiveBlocked(f"{name.replace('_', '-')} file does not exist: {values[name]}")
    if not values["checkpoint"].exists():
        raise LiveBlocked(f"checkpoint does not exist: {values['checkpoint']}")
    if values["run_dir"].exists() and not values["run_dir"].is_dir():
        raise LiveBlocked(f"run-dir exists but is not a directory: {values['run_dir']}")
    if values["run_dir"].is_dir() and any(values["run_dir"].iterdir()) and not args.resume:
        raise LiveBlocked(f"run-dir is nonempty; pass --resume only for an identical run: {values['run_dir']}")
    return values


def verify_framework_checkout(framework_root: str | Path) -> str:
    """Require the exact clean Cosmos checkout admitted by the Task 8 contract."""
    root = Path(framework_root).resolve()
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False, timeout=15)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise LiveBlocked(f"cannot verify official framework Git checkout: {error}") from error
    if head.returncode != 0 or status.returncode != 0:
        detail = (head.stderr or status.stderr).strip()
        raise LiveBlocked(f"framework must be a readable Git checkout: {detail or root}")
    observed = head.stdout.strip().lower()
    if observed != TASK8_FRAMEWORK_COMMIT:
        raise LiveBlocked(f"official framework commit mismatch: expected {TASK8_FRAMEWORK_COMMIT}, observed {observed!r}")
    if status.stdout.strip():
        raise LiveBlocked("official framework checkout has uncommitted changes")
    return observed


def _configure_offline_gpu0() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "0"):
        raise LiveBlocked(f"CUDA_VISIBLE_DEVICES must be unset or 0, observed {visible!r}")
    hub_offline = os.environ.get("HF_HUB_OFFLINE")
    if hub_offline not in (None, "1"):
        raise LiveBlocked(f"HF_HUB_OFFLINE must be 1, observed {hub_offline!r}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"


def _validate_torch_contract(torch_module: Any) -> dict[str, Any]:
    version = str(getattr(torch_module, "__version__", ""))
    cuda_version = str(getattr(getattr(torch_module, "version", None), "cuda", ""))
    if version != "2.10.0+cu130" or cuda_version != "13.0":
        raise LiveBlocked(f"Task 9 requires Torch 2.10.0+cu130 / CUDA 13.0, observed {version!r} / {cuda_version!r}")
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()) or int(cuda.current_device()) != 0:
        raise LiveBlocked("Task 9 requires an available CUDA device 0")
    try:
        torch_module.backends.cuda.matmul.allow_tf32 = False
        torch_module.backends.cudnn.allow_tf32 = False
    except AttributeError as error:
        raise LiveBlocked("Torch does not expose the required TF32 controls") from error
    if bool(torch_module.backends.cuda.matmul.allow_tf32) or bool(torch_module.backends.cudnn.allow_tf32):
        raise LiveBlocked("Task 9 requires TF32 disabled")
    return {"torch": version, "cuda": cuda_version, "device": 0,
            "device_name": str(cuda.get_device_name(0)), "tf32": False}


def _verify_context_contract(context: Any, torch_module: Any) -> dict[str, Any]:
    observed = getattr(context, "contract", None)
    if not isinstance(observed, Mapping):
        raise LiveBlocked("Task 8 live context did not expose its fixed runtime contract")
    mismatches = [f"{key}={observed.get(key)!r} (expected {expected!r})"
                  for key, expected in TASK8_CONTRACT.items() if observed.get(key) != expected]
    if mismatches:
        raise LiveBlocked("Task 8 FP32/cache-off contract mismatch: " + "; ".join(mismatches))
    if bool(torch_module.backends.cuda.matmul.allow_tf32) or bool(torch_module.backends.cudnn.allow_tf32):
        raise LiveBlocked("Task 8 loader re-enabled TF32")
    runtime = getattr(context, "runtime", None)
    model = getattr(runtime, "model", None)
    if model is not None and bool(getattr(model, "_diffusion_cache_installed", False)):
        raise LiveBlocked("Task 8 runtime installed diffusion cache")
    return dict(observed)


def _path_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "kind": "directory" if path.is_dir() else "file",
            "size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _content_identity(path: Path) -> dict[str, Any]:
    """Bind model/VAE inputs to bytes, streaming large checkpoint files once."""
    if path.is_file():
        return {"path": str(path), "kind": "file", "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path)}
    if not path.is_dir():
        raise LiveBlocked(f"content identity path does not exist: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise LiveBlocked(f"content identity directory is empty: {path}")
    entries = []
    total_bytes = 0
    root = path.resolve()
    for item in files:
        resolved = item.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise LiveBlocked(f"checkpoint symlink escapes the model directory: {item}") from error
        size = int(resolved.stat().st_size)
        total_bytes += size
        entries.append({"path": relative, "size_bytes": size, "sha256": _sha256_file(resolved)})
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"path": str(path), "kind": "directory", "file_count": len(entries),
            "total_size_bytes": total_bytes,
            "content_manifest_sha256": hashlib.sha256(encoded).hexdigest(), "files": entries}


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")); digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_safe_evidence(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": _array_sha256(value)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_evidence(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_evidence(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite Task 9 runtime evidence cannot be serialized")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if callable(getattr(value, "detach", None)) and callable(getattr(value, "cpu", None)):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": _array_sha256(array)}
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dict__"):
        return _json_safe_evidence(vars(value))
    return str(value)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    payload = json.dumps(_json_safe_evidence(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _preflight_identity(data_path: Path, metadata_path: Path) -> tuple[dict[str, Any], str, str]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LiveBlocked(f"invalid Task 8 preflight metadata: {error}") from error
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("selected"), Mapping):
        raise LiveBlocked("Task 8 preflight metadata lacks selected scene identity")
    if metadata.get("status") != "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED":
        raise LiveBlocked("Task 8 metadata is not an admitted generation-free preflight")
    selected = metadata["selected"]
    preflight_npz = selected.get("preflight_npz")
    if not isinstance(preflight_npz, Mapping):
        raise LiveBlocked("Task 8 preflight metadata lacks the NPZ SHA256 binding")
    expected_data_hash = str(preflight_npz.get("sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_data_hash) is None:
        raise LiveBlocked("Task 8 preflight NPZ SHA256 binding is malformed")
    data_hash = _sha256_file(data_path)
    if data_hash != expected_data_hash:
        raise LiveBlocked("Task 8 preflight NPZ SHA256 does not match the selected data file")
    keys = ("file_path", "episode_id", "record_index", "shard", "language")
    scene = {key: selected[key] for key in keys if key in selected}
    if not any(key in scene for key in ("file_path", "episode_id", "record_index")):
        raise LiveBlocked("Task 8 preflight metadata has no episode or record identity")
    return scene, data_hash, _sha256_file(metadata_path)


def _write_launcher_status(run_dir: Path, value: Mapping[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(run_dir / "launcher_status.json", value)


def _action_chunk0_identity(context: Any) -> tuple[str, dict[str, Any]]:
    batch = getattr(context, "batch", None)
    if batch is None:
        raise LiveBlocked("Task 8 context did not retain its admitted scene batch")
    prompt = getattr(batch, "prompt", None)
    if not isinstance(prompt, str) or not prompt.strip():
        raise LiveBlocked("Task 8 scene prompt is missing")
    normalized = np.asarray(getattr(batch, "actions", None))
    raw_values = getattr(batch, "raw_actions", None)
    if raw_values is None:
        raise LiveBlocked("Task 8 preflight did not retain raw action chunk 0")
    raw = np.asarray(raw_values)
    if normalized.shape != (2, 16, 10) or raw.shape != (2, 16, 10):
        raise LiveBlocked("Task 8 raw/normalized actions must both have shape [2,16,10]")
    if (normalized.dtype != np.float32 or raw.dtype != np.float32
            or not np.all(np.isfinite(normalized)) or not np.all(np.isfinite(raw))):
        raise LiveBlocked("Task 8 raw/normalized actions must be finite float32 arrays")
    hashes = {"raw_sha256": _array_sha256(raw[0]),
              "normalized_sha256": _array_sha256(normalized[0]),
              "raw_shape": list(raw[0].shape), "normalized_shape": list(normalized[0].shape),
              "raw_dtype": str(raw.dtype), "normalized_dtype": str(normalized.dtype)}
    return prompt, hashes


def _smoke_plan() -> list[dict[str, Any]]:
    return [{"sample_id": "baseline_pre", "kind": "baseline", "direction_id": None,
             "alpha": 0.0, "sign": 0}]


def execute_task9(args: argparse.Namespace, *, runtime_api: Any | None = None,
                  torch_module: Any | None = None) -> dict[str, Any]:
    """Execute one release-gated scene/seed response scan on Task 8's runtime."""
    paths = _validate_paths(args)
    if not bool(args.release):
        raise LiveBlocked("pass --release only after main approves this Task 9 run")
    _configure_offline_gpu0()
    framework_commit = verify_framework_checkout(paths["framework"])
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except ImportError as error:
            raise LiveBlocked("Torch is required for a released Task 9 run") from error
    environment = _validate_torch_contract(torch_module)
    if runtime_api is None:
        runtime_api = _load_task8_api(paths["code_root"])

    scene, data_hash, metadata_hash = _preflight_identity(paths["data"], paths["metadata_json"])
    batch = runtime_api.Task8InputAdapter.from_preflight(paths["data"], paths["metadata_json"])
    run_dir = paths["run_dir"]
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor = runtime_api.Task8ResourceMonitor(run_dir, gpu_index=0)
    monitor_started = False
    context = None
    primary_error: BaseException | None = None
    public_result: dict[str, Any] | None = None
    try:
        monitor_started = True
        monitor.start()
        context = runtime_api.load_task8_live(
            framework_root=paths["framework"], checkpoint=paths["checkpoint"],
            vae=paths["vae"], run_dir=run_dir, batch=batch,
            release=True, resume=bool(args.resume),
        )
        contract = _verify_context_contract(context, torch_module)
        feedback = getattr(context, "feedback", None)
        if feedback is None:
            raise LiveBlocked("Task 8 live context lacks FeedbackRuntime")
        try:
            base = np.ascontiguousarray(feedback.extract_condition(feedback.z0), dtype=np.float32)
            condition_indexes = tuple(int(index) for index in feedback.condition_indexes)
            predicted_indexes = tuple(int(index) for index in feedback.predicted_indexes)
            temporal_axis = int(feedback.temporal_axis)
            mask_full = np.ascontiguousarray(np.asarray(feedback.mask, dtype=bool))
            mask = np.ascontiguousarray(np.take(mask_full, condition_indexes, axis=temporal_axis), dtype=bool)
        except (AttributeError, TypeError, ValueError) as error:
            raise LiveBlocked(f"Task 8 FeedbackRuntime did not expose condition geometry: {error}") from error
        if base.shape != mask.shape or not base.size or not mask.any() or not np.all(np.isfinite(base)):
            raise LiveBlocked("Task 8 condition-only FP32 base/mask geometry is invalid")
        prompt, action_chunk0 = _action_chunk0_identity(context)

        if args.smoke:
            directions: dict[str, np.ndarray] = {}
            plan = _smoke_plan()
        else:
            bank = generate_direction_bank(mask, seed=DIRECTION_SEED,
                                           train_count=32, holdout_count=8)
            directions = {f"train_{index:02d}": bank[index].copy() for index in range(32)}
            directions.update({f"holdout_{index:02d}": bank[32 + index].copy() for index in range(8)})
            plan = build_call_plan(train_count=32, holdout_count=8, alpha=0.003,
                                   calibration_alpha=0.0015)
            if len(plan) != 162:
                raise LiveBlocked(f"Task 9 plan must contain 162 paired calls, observed {len(plan)}")

        snapshot_identity = getattr(runtime_api, "task9_snapshot_identity", None)
        identity = {
            "scene": scene,
            "prompt": prompt,
        "action_chunk0": action_chunk0,
            "condition_geometry": {"condition_indexes": list(condition_indexes),
                                   "predicted_indexes": list(predicted_indexes),
                                   "temporal_axis": temporal_axis,
                                   "full_mask_sha256": _array_sha256(mask_full),
                                   "condition_mask_sha256": _array_sha256(mask)},
            "data": {"path": str(paths["data"]), "sha256": data_hash},
            "metadata": {"path": str(paths["metadata_json"]), "sha256": metadata_hash},
            "code_root": snapshot_identity or {"path": str(paths["code_root"])},
            "launcher": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
            "framework": _path_identity(paths["framework"]),
            "framework_commit": framework_commit,
            "checkpoint": _content_identity(paths["checkpoint"]),
            "vae": _content_identity(paths["vae"]),
            "task8_contract": contract,
            "environment": environment,
            "direction_seed": DIRECTION_SEED,
            "mode": "smoke" if args.smoke else "formal",
        }
        scan_dir = run_dir / "scan"
        samples_root = scan_dir / "samples"
        pending_specs = iter([spec for spec in plan
                              if not (samples_root / str(spec["sample_id"])).exists()])

        def step(condition_only: np.ndarray, seed: int) -> Mapping[str, Any]:
            try:
                spec = next(pending_specs)
            except StopIteration as error:
                raise LiveBlocked("Task 9 step callback exceeded the frozen call plan") from error
            runtime = getattr(context, "runtime", None)
            model = getattr(runtime, "model", None)
            if runtime is None or model is None or not callable(getattr(model, "denoise", None)):
                raise LiveBlocked("Task 8 live context lacks its runtime action-consumption seam")
            clone = getattr(runtime_api, "clone_runtime", None)
            if not callable(clone):
                try:
                    clone = getattr(importlib.import_module("umi_fd_post_vae_scan"), "_clone_runtime")
                except (ImportError, AttributeError) as error:
                    raise LiveBlocked(f"Task 8 prepared-state clone helper is unavailable: {error}") from error
            try:
                action = np.asarray(context.batch.actions[0], dtype=np.float32).copy()
                request_prepared = clone(runtime.prepared)
                action_evidence = runtime_api.refresh_official_prepared_action(
                    request_prepared, action, ops=runtime.ops)
            except Exception as error:
                raise LiveBlocked(f"could not refresh admitted Task 8 action chunk 0: {error}") from error
            original_denoise = model.denoise
            instance_dict = getattr(model, "__dict__", {})
            had_instance_denoise = "denoise" in instance_dict
            original_prepared = runtime.prepared
            packed_action_token_hashes: list[str] = []

            @wraps(original_denoise)
            def observe_denoise(*positional: Any, **keywords: Any) -> Any:
                packed = keywords.get("data_batch_packed")
                if packed is None:
                    packed = next((item for item in positional
                                   if getattr(item, "action", None) is not None), None)
                action_pack = getattr(packed, "action", None) if packed is not None else None
                tokens = getattr(action_pack, "tokens", None)
                if tokens is None:
                    raise LiveBlocked("denoiser call did not expose packed action tokens")
                try:
                    tokens0 = tokens[0]
                except (IndexError, KeyError, TypeError) as error:
                    raise LiveBlocked("denoiser packed action tokens lack chunk 0") from error
                packed_action_token_hashes.append(str(runtime_api.action_token_hash(tokens0)))
                return original_denoise(*positional, **keywords)

            try:
                runtime.prepared = request_prepared
                model.denoise = observe_denoise
                capture = feedback.step(condition_only, seed)
            finally:
                if had_instance_denoise:
                    model.denoise = original_denoise
                else:
                    delattr(model, "denoise")
                runtime.prepared = original_prepared
            try:
                actual = capture["actual"]
                condition_steps = capture.get("condition_steps_fp32")
                if condition_steps is None:
                    condition_steps = actual["condition_steps"]
                if len(condition_steps) != 30:
                    raise ValueError(f"expected 30 condition steps, observed {len(condition_steps)}")
                consumed_full = np.asarray(condition_steps[-1], dtype=np.float32)
                condition_step_hashes = [_array_sha256(feedback.extract_condition(item))
                                         for item in condition_steps]
            except (KeyError, IndexError, TypeError, ValueError) as error:
                raise LiveBlocked(f"FeedbackRuntime lacks complete per-step FP32 condition readback: {error}") from error
            actual_consumed = feedback.extract_condition(consumed_full)
            if not isinstance(action_evidence, Mapping):
                raise LiveBlocked("Task 8 action refresh did not return evidence")
            try:
                action_check = runtime_api.validate_action_consumption(
                    {"packed_action_token_hashes": packed_action_token_hashes},
                    action_evidence.get("effective_action", action), chunk_index=0)
            except Exception as error:
                raise LiveBlocked(f"Task 8 action chunk 0 was not verified at all denoiser steps: {error}") from error
            if len(packed_action_token_hashes) != 30:
                raise LiveBlocked(f"Task 8 action was observed at {len(packed_action_token_hashes)} denoiser steps; expected 30")
            step_evidence = {
                "prompt": prompt,
                "condition_indexes": list(condition_indexes),
                "predicted_indexes": list(predicted_indexes),
                "temporal_axis": temporal_axis,
                "condition_step_sha256": condition_step_hashes,
                "actual_consumed_condition_sha256": _array_sha256(actual_consumed),
                "action_chunk0": action_chunk0,
                "action_token_evidence": {
                    "prepared": _json_safe_evidence(action_evidence),
                    "packed_action_token_hashes": packed_action_token_hashes,
                    "consumption": _json_safe_evidence(action_check),
                },
                "prediction_noise_hash": str(capture["prediction_noise_hash"]),
                "fp32_precision": {"runtime_contract": contract,
                                   "feedback": _json_safe_evidence(capture.get("evidence", {})),
                                   "decoder": _json_safe_evidence(capture.get("decoder", {})),
                                   "encoder": _json_safe_evidence(capture.get("encoder", {}))},
            }
            return {
                "predicted_latent": capture["predicted_latent"],
                "decoded_last_rgb": capture["decoded_last_rgb"],
                "next_condition_fp32": capture["next_condition_fp32"],
                "prediction_noise_hash": capture["prediction_noise_hash"],
                "actual_consumed_condition": actual_consumed,
                "step_evidence": step_evidence,
            }

        def resource_check(stage: str) -> None:
            monitor.check(phase="resource-smoke" if args.smoke else "formal",
                          starting_new_sample=(stage == "before"))

        result = run_samples(
            scan_dir, base, mask, directions, plan,
            seed=int(args.seed), identity=identity, step=step, resume=bool(args.resume),
            resource_check=resource_check,
            start_free_bytes=TASK9_DISK_RESERVE_BYTES,
            reserve_bytes=TASK9_DISK_RESERVE_BYTES,
            forecast_factor=1.3,
        )
        public_result = {**result, "run_dir": str(run_dir),
                         "scan_dir": str(scan_dir),
                         "mode": "smoke" if args.smoke else "formal",
                         "resource_samples": list(getattr(monitor, "samples", ())),
                         "runtime_contract": contract, "environment": environment}
    except BaseException as error:
        primary_error = error
        try:
            _write_launcher_status(run_dir, {"status": "BLOCKED", "generation_started": context is not None,
                                             "mode": "smoke" if args.smoke else "formal",
                                             "error": {"type": type(error).__name__, "message": str(error),
                                                       "traceback": traceback.format_exc()}})
        except Exception as status_error:
            add_note = getattr(error, "add_note", None)
            if callable(add_note):
                add_note(f"could not persist Task 9 failure status: {status_error!r}")
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if context is not None:
            try:
                context.cleanup()
            except BaseException as error:
                cleanup_errors.append(error)
        if monitor_started:
            try:
                monitor.stop()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note("Task 9 cleanup also failed: " + repr(error))
            else:
                failure = RuntimeError("Task 9 cleanup failed")
                for error in cleanup_errors:
                    failure.add_note(repr(error))
                try:
                    _write_launcher_status(run_dir, {"status": "BLOCKED", "generation_started": context is not None,
                                                     "mode": "smoke" if args.smoke else "formal",
                                                     "error": {"type": type(failure).__name__, "message": str(failure)}})
                except Exception as status_error:
                    failure.add_note(f"could not persist Task 9 cleanup failure: {status_error!r}")
                raise failure from cleanup_errors[0]
    if public_result is None:
        raise RuntimeError("Task 9 execution returned without a result")
    _write_launcher_status(run_dir, {"status": "COMPLETE", "generation_started": True,
                                     "mode": public_result["mode"], "completed": public_result["completed"],
                                     "total": public_result["total"],
                                     "runtime_contract": public_result["runtime_contract"],
                                     "environment": environment})
    return public_result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = execute_task9(args)
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 3
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0 if result.get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
