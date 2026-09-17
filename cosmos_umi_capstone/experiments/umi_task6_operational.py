"""Operational, generation-free contracts for the Task 6 pilot.

This module is deliberately small and explicit.  It owns the things that must
be pinned before a model is loaded (assets, Task 5 evidence and launch
contract); the actual Cosmos import is isolated in :class:`OfficialRuntimeFactory`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

try:
    from .umi_task6_primitives import SOURCE_COMMIT, STATE_CATALOG, parse_action, preprocess_frame
    from .umi_task6_runtime import Task6Inputs, load_frozen_directions
    from .umi_fd_post_vae_bridge import sha256_array
    from .umi_fd_post_vae_scan import content_identity as _content_identity
    from .umi_task5_primitives import freeze_task5_directions
    from .umi_precision_storage import ProcessLock
except ImportError:  # pragma: no cover
    from umi_task6_primitives import SOURCE_COMMIT, STATE_CATALOG, parse_action, preprocess_frame
    from umi_task6_runtime import Task6Inputs, load_frozen_directions
    from umi_fd_post_vae_bridge import sha256_array
    from umi_fd_post_vae_scan import content_identity as _content_identity
    from umi_task5_primitives import freeze_task5_directions
    from umi_precision_storage import ProcessLock


BRIDGE0_ASSET_SHA256 = {
    "action": "5c26b3cb84799812a70b534ad939551d2ac308fdc870ea0e66163bb52c9d61da",
    "video": "a86cfc81633b216891ca26dc58c72193a979c10ad72f123175fa8d61a67cdaec",
}
BRIDGE0_PROMPT = "Put the pot to the left of the purple item."
BRIDGE0_FPS = 5
PINNED_FRAME_SIZE = (256, 256)
PILOT_GROUP = {"state": "bridge_0", "seed": 0}
TASK6_CODE_SOURCES = ("umi_task6_runtime.py", "run_umi_task6_experiment.py", "umi_task6_primitives.py",
    "umi_fd_post_vae_bridge.py", "umi_fd_post_vae_scan.py", "umi_precision_runtime.py", "umi_precision_storage.py",
    "umi_precision_official.py", "umi_task5_runtime.py", "umi_task5_primitives.py",
    "umi_task6_operational.py", "run_umi_task6_official.py", "umi_task6_cosmos_loader.py",
    "umi_task6_decoder.py", "analyze_umi_task6.py")


class OperationalEvidenceError(ValueError):
    """Fail-closed error raised before or during an operational phase."""


def state_asset(state: str = "bridge_0") -> dict[str, Any]:
    """Return the immutable catalog entry used by operational callers."""
    if state not in STATE_CATALOG:
        raise OperationalEvidenceError(f"unknown Task 6 state: {state}")
    asset = STATE_CATALOG[state]
    return {"state": asset.state, "source_commit": asset.source_commit,
            "video_path": asset.video_path, "action_path": asset.action_path,
            "source_kind": asset.source_kind}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_content_identity(path: str | os.PathLike[str]) -> str:
    """Return the content hash for a regular-file or directory checkpoint."""
    try:
        return _content_identity(path)
    except OperationalEvidenceError:
        raise
    except (OSError, TypeError, ValueError) as error:
        if "symlink" in str(error).lower():
            raise OperationalEvidenceError(str(error)) from error
        if "missing or unsupported" in str(error).lower():
            raise OperationalEvidenceError(f"checkpoint path is missing or unsupported: {path}") from error
        label = str(path)
        raise OperationalEvidenceError(f"checkpoint content identity could not be observed: {label}") from error
    raise OperationalEvidenceError(f"checkpoint path is missing or unsupported: {path}")


def code_bundle_sha256(source_root: str | os.PathLike[str] | None = None) -> str:
    """Hash exactly the source bundle used by the Task 6 runtime binding."""
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in TASK6_CODE_SOURCES:
        path = root / name
        if path.is_file():
            digest.update(name.encode("utf-8")); digest.update(b"\0"); digest.update(path.read_bytes())
    return digest.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationalEvidenceError(f"invalid JSON: {path}") from error


def verify_pinned_bridge_assets(action_path: str | os.PathLike[str], video_path: str | os.PathLike[str], *,
                                state: str = "bridge_0", expected_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Verify the exact official bridge asset pair without decoding the video."""
    if state != "bridge_0":
        raise OperationalEvidenceError("only bridge_0 is pinned for the first pilot")
    action, video = Path(action_path).resolve(), Path(video_path).resolve()
    if not action.is_file() or not video.is_file():
        raise OperationalEvidenceError("bridge action/video asset is missing")
    expected = dict(expected_hashes or BRIDGE0_ASSET_SHA256)
    observed = {"action": sha256_file(action), "video": sha256_file(video)}
    if expected != observed:
        raise OperationalEvidenceError(f"pinned bridge asset hash mismatch: expected {expected}, observed {observed}")
    parsed = parse_action(_json(action))
    return {"state": state, "source_commit": SOURCE_COMMIT, "action_path": str(action),
            "video_path": str(video), "action_sha256": observed["action"],
            "video_sha256": observed["video"], "action_shape": list(parsed.shape),
            "prompt": BRIDGE0_PROMPT, "fps": BRIDGE0_FPS, "preprocess": "no_upscale_bicubic_antialias_bottom_right_reflection_256"}


def prepare_bridge_upload_bundle(action_path: str | os.PathLike[str], video_path: str | os.PathLike[str],
                                 destination: str | os.PathLike[str]) -> dict[str, Any]:
    """Copy the verified 1.6MB pair atomically into a small upload bundle."""
    evidence = verify_pinned_bridge_assets(action_path, video_path)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"upload bundle must be a new empty directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".bridge_upload.", dir=destination.parent))
    try:
        shutil.copy2(action_path, stage / Path(action_path).name)
        shutil.copy2(video_path, stage / Path(video_path).name)
        manifest = {"assets": evidence, "files": {p.name: sha256_file(p) for p in sorted(stage.iterdir())}}
        (stage / "upload_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Windows cannot atomically rename a directory over a new name.  Keep
        # the destination absent until all payload files are complete, then
        # publish each immutable file with no-replace semantics; the manifest
        # is written last and is the commit marker for upload consumers.
        destination.mkdir(parents=True, exist_ok=False)
        for payload in sorted(stage.iterdir(), key=lambda p: p.name == "upload_manifest.json"):
            os.replace(payload, destination / payload.name)
        stage.rmdir()
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise
    return manifest


def _verify_tree_manifest(root: Path, *, manifest_name: str = "MANIFEST.sha256", lock_name: str = ".runner.lock") -> str:
    def read_manifest_entries(base: Path) -> tuple[set[str], set[str]]:
        manifest = base / manifest_name
        if not manifest.is_file():
            raise OperationalEvidenceError(f"missing manifest: {manifest}")
        seen: set[str] = set()
        for line in manifest.read_text(encoding="ascii").splitlines():
            parts = line.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
                raise OperationalEvidenceError("malformed Task 5 manifest")
            relative = Path(parts[1])
            if relative.is_absolute() or ".." in relative.parts or relative.as_posix() == manifest_name or parts[1] in seen:
                raise OperationalEvidenceError("unsafe or duplicate Task 5 manifest path")
            target = base / relative
            if not target.is_file() or sha256_file(target) != parts[0]:
                raise OperationalEvidenceError(f"Task 5 manifest mismatch: {parts[1]}")
            seen.add(parts[1])
        actual = {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file() and p.relative_to(base).as_posix() not in {manifest_name, lock_name}}
        return seen, actual

    seen, actual = read_manifest_entries(root)
    listed = seen - {lock_name}
    if listed - actual:
        raise OperationalEvidenceError("Task 5 manifest inventory mismatch")
    unlisted = actual - listed
    for candidate in sorted(unlisted):
        relative = Path(candidate)
        if relative.name != Path(manifest_name).name or relative.parent == Path("."):
            raise OperationalEvidenceError("Task 5 manifest inventory mismatch")
        auxiliary_seen, auxiliary_actual = read_manifest_entries(root / relative.parent)
        if auxiliary_actual != auxiliary_seen - {lock_name}:
            raise OperationalEvidenceError("Task 5 manifest inventory mismatch")
    return sha256_file(root / manifest_name)


def extract_task5_directions(task5_root: str | os.PathLike[str], *, expected_manifest_sha256: str | None = None,
                             expected_plan_sha256: str | None = None, expected_direction_file_hashes: Mapping[str, str] | None = None,
                             expected_direction_hashes: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Extract v0/v1/v2 solely from successful Task 5 alpha-00-plus samples."""
    root = Path(task5_root).resolve()
    manifest_sha = _verify_tree_manifest(root)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise OperationalEvidenceError("Task 5 root manifest hash mismatch")
    plan_path = root / "task5_plan.json"
    if not plan_path.is_file():
        raise OperationalEvidenceError("Task 5 plan is missing")
    plan_sha = sha256_file(plan_path)
    if expected_plan_sha256 is not None and plan_sha != expected_plan_sha256:
        raise OperationalEvidenceError("Task 5 plan hash mismatch")
    plan = _json(plan_path)
    status_path = root / "status.json"
    if not status_path.is_file() or _json(status_path).get("status") != "complete" or int(_json(status_path).get("formal_successful", 0)) != 32:
        raise OperationalEvidenceError("Task 5 root status is not a completed 32-call run")
    plan_entries = plan.get("plan")
    if not isinstance(plan_entries, list) or len(plan_entries) != 32 or len({entry.get("sample_id") for entry in plan_entries if isinstance(entry, Mapping)}) != 32:
        raise OperationalEvidenceError("Task 5 plan is not an exact 32-call plan")
    found: dict[str, np.ndarray] = {}
    file_hashes: dict[str, str] = {}
    for direction in ("v0", "v1", "v2"):
        sample = root / "samples" / f"{direction}_alpha_00_plus"
        status_path, sample_json = sample / "status.json", sample / "sample.json"
        if not status_path.is_file() or not sample_json.is_file():
            raise OperationalEvidenceError(f"Task 5 direction sample missing: {direction}")
        status = _json(status_path)
        if status.get("status") != "success":
            raise OperationalEvidenceError(f"Task 5 direction sample is not successful: {direction}")
        hashes = status.get("artifact_sha256", {})
        direction_path = sample / "direction.npy"
        if hashes.get("direction.npy") != sha256_file(direction_path):
            raise OperationalEvidenceError(f"Task 5 direction artifact hash mismatch: {direction}")
        value = np.load(direction_path, allow_pickle=False).astype(np.float32, copy=True)
        if value.ndim != 5 or not np.all(np.isfinite(value)):
            raise OperationalEvidenceError(f"Task 5 direction has invalid shape/value: {direction}")
        found[direction] = value
        file_hashes[direction] = sha256_file(direction_path)
    expected = dict(expected_direction_file_hashes or {})
    for key, value in expected.items():
        if key in file_hashes and file_hashes[key] != value:
            raise OperationalEvidenceError(f"Task 5 direction file hash mismatch: {key}")
    mask_path = root / "samples" / "v0_alpha_00_plus" / "mask.npy"
    if not mask_path.is_file():
        raise OperationalEvidenceError("Task 5 direction evidence lacks the runtime condition mask")
    mask_status = _json(root / "samples" / "v0_alpha_00_plus" / "status.json")
    if mask_status.get("artifact_sha256", {}).get("mask.npy") != sha256_file(mask_path):
        raise OperationalEvidenceError("Task 5 condition mask artifact hash mismatch")
    mask = np.load(mask_path, allow_pickle=False).astype(bool, copy=False)
    if mask.shape != next(iter(found.values())).shape or not np.any(mask):
        raise OperationalEvidenceError("Task 5 runtime condition mask has the wrong shape")
    for direction, value in found.items():
        other_mask_path = root / "samples" / f"{direction}_alpha_00_plus" / "mask.npy"
        if not other_mask_path.is_file() or not np.array_equal(mask, np.load(other_mask_path, allow_pickle=False).astype(bool, copy=False)):
            raise OperationalEvidenceError(f"Task 5 direction mask differs: {direction}")
        if np.any(value[~mask]) or not np.isclose(np.sqrt(np.mean(value[mask].astype(np.float64) ** 2)), 1.0, atol=1e-6, rtol=0.0):
            raise OperationalEvidenceError(f"Task 5 direction geometry is invalid: {direction}")
    frozen = freeze_task5_directions(np.stack([found[k] for k in ("v0", "v1", "v2")]), mask)["directions"]
    frozen_hashes = {key: sha256_array(value) for key, value in frozen.items()}
    for key, value in dict(expected_direction_hashes or {}).items():
        if key in frozen_hashes and frozen_hashes[key] != value:
            raise OperationalEvidenceError(f"Task 5 frozen direction hash mismatch: {key}")
    bank = np.stack([found[key] for key in ("v0", "v1", "v2")])
    return {"bank": bank, "mask": mask, "directions": frozen, "manifest_sha256": manifest_sha,
            "plan_sha256": plan_sha, "direction_file_sha256": file_hashes, "direction_sha256": frozen_hashes,
            "task5_root": str(root)}


REQUIRED_CONTRACT_KEYS = frozenset({"framework_commit", "asset_source_commit", "checkpoint_identity", "vae_sha256", "torch_version", "cuda_version", "fps",
    "code_bundle_sha256", "bridge_asset_hashes", "task5", "group", "prompt", "action", "settings", "cache_flags", "seed_routes", "geometry"})


def validate_launch_contract(contract: Mapping[str, Any], *, observed: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate an approved launch contract; caller claims are never filled in."""
    if not isinstance(contract, Mapping) or not REQUIRED_CONTRACT_KEYS.issubset(contract):
        missing = sorted(REQUIRED_CONTRACT_KEYS - set(contract) if isinstance(contract, Mapping) else REQUIRED_CONTRACT_KEYS)
        raise OperationalEvidenceError(f"launch contract is missing required evidence: {missing}")
    _validate_static_contract(contract)
    group = contract["group"]
    if dict(group) != PILOT_GROUP:
        raise OperationalEvidenceError("launch contract is restricted to bridge_0 / seed 0")
    if contract["prompt"] != BRIDGE0_PROMPT or parse_action(contract["action"]).shape != (16, 10):
        raise OperationalEvidenceError("prompt/action contract mismatch")
    if contract.get("fps") != BRIDGE0_FPS:
        raise OperationalEvidenceError("Bridge sample fps must be exactly 5")
    settings = contract["settings"]
    expected_settings = {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "batch_size": 1,
                         "autocast": False, "tf32": False, "diffusion_cache": False}
    if any(settings.get(k) != v for k, v in expected_settings.items()):
        raise OperationalEvidenceError("generation settings do not match Task 4 C")
    flags = contract["cache_flags"]
    if flags != {"autocast": False, "tf32": False, "diffusion_cache": False}:
        raise OperationalEvidenceError("cache/precision flags do not match Task 4 C")
    routes = contract["seed_routes"]
    if routes != {"model": 0, "prepare": 0, "sampler": 0, "scheduler": 0}:
        raise OperationalEvidenceError("seed routes are not fully pinned")
    geometry = contract["geometry"]
    if geometry.get("carrier_shape") != [1, 48, 5, 16, 16] or geometry.get("condition_indexes") != [0] or geometry.get("predicted_indexes") != [1, 2, 3, 4] or geometry.get("mask_shape") != [1, 48, 5, 16, 16]:
        raise OperationalEvidenceError("runtime geometry does not match official UMI path")
    for key in ("vae_sha256", "code_bundle_sha256"):
        if not isinstance(contract[key], str) or len(contract[key]) != 64:
            raise OperationalEvidenceError(f"invalid launch identity: {key}")
    if observed is None:
        raise OperationalEvidenceError("approved observed launch evidence is required; self-attested contract is insufficient")
    if not isinstance(observed, Mapping) or not REQUIRED_CONTRACT_KEYS.issubset(observed):
        raise OperationalEvidenceError("observed launch evidence is incomplete")
    mismatches = [key for key in REQUIRED_CONTRACT_KEYS if key in observed and observed[key] != contract[key]]
    if mismatches:
        raise OperationalEvidenceError(f"launch contract differs from observed evidence: {mismatches}")
    return {"status": "PASS", "contract": dict(contract), "observed_keys": sorted(observed)}


def _validate_static_contract(contract: Mapping[str, Any]) -> None:
    """Validate immutable contract fields before any runtime is loaded."""
    framework_commit = contract.get("framework_commit")
    if not isinstance(framework_commit, str) or len(framework_commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in framework_commit):
        raise OperationalEvidenceError("framework git commit is missing or malformed")
    if contract.get("asset_source_commit") != SOURCE_COMMIT:
        raise OperationalEvidenceError("Bridge asset source commit is not pinned")
    assets = contract.get("bridge_asset_hashes")
    if assets != BRIDGE0_ASSET_SHA256:
        raise OperationalEvidenceError("bridge asset contract does not equal the pinned official hashes")
    task5 = contract.get("task5")
    required_task5 = {"manifest_sha256", "plan_sha256", "direction_file_sha256", "direction_sha256"}
    if not isinstance(task5, Mapping) or not required_task5.issubset(task5):
        raise OperationalEvidenceError("Task 5 contract evidence is incomplete")
    for key in ("manifest_sha256", "plan_sha256"):
        if not isinstance(task5[key], str) or len(task5[key]) != 64 or any(c not in "0123456789abcdef" for c in task5[key]):
            raise OperationalEvidenceError(f"invalid Task 5 hash: {key}")
    for section, names in (("direction_file_sha256", ("v0", "v1", "v2")), ("direction_sha256", ("v0", "v1", "v2", "u01", "u12"))):
        values = task5.get(section)
        if not isinstance(values, Mapping) or set(values) != set(names) or any(not isinstance(values[n], str) or len(values[n]) != 64 or any(c not in "0123456789abcdefABCDEF" for c in values[n]) for n in names):
            raise OperationalEvidenceError(f"Task 5 {section} evidence is incomplete")
    identity = contract.get("checkpoint_identity")
    if not isinstance(identity, Mapping) or set(identity) != {"sha256"} or not isinstance(identity["sha256"], str) or len(identity["sha256"]) != 64:
        raise OperationalEvidenceError("checkpoint content identity must be an exact SHA256")
    if not isinstance(contract.get("vae_sha256"), str) or len(contract["vae_sha256"]) != 64:
        raise OperationalEvidenceError("VAE content identity is missing")
    if not isinstance(contract.get("code_bundle_sha256"), str) or len(contract["code_bundle_sha256"]) != 64:
        raise OperationalEvidenceError("code bundle identity is missing")


def observe_live_launch(contract: Mapping[str, Any], *, runtime: Any, inputs: Any, assets: Mapping[str, Any],
                        task5: Mapping[str, Any], checkpoint: str | os.PathLike[str], vae: str | os.PathLike[str],
                        framework_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Build launch evidence from loaded objects and file/environment state."""
    try:
        import torch
        torch_version, cuda_version = str(torch.__version__), str(torch.version.cuda)
    except Exception as error:
        raise OperationalEvidenceError("live torch/CUDA observation failed") from error
    actual = runtime.actual_identity() if callable(getattr(runtime, "actual_identity", None)) else None
    if actual is None:
        raise OperationalEvidenceError("runtime did not expose actual content identity")
    geometry = inputs.geometry
    observation = {
        "framework_commit": _live_framework_commit(framework_root),
        "asset_source_commit": SOURCE_COMMIT,
        "checkpoint_identity": {"sha256": checkpoint_content_identity(checkpoint)},
        "vae_sha256": sha256_file(vae),
        "code_bundle_sha256": code_bundle_sha256(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "fps": _observed_fps(runtime, inputs),
        "bridge_asset_hashes": {"action": assets["action_sha256"], "video": assets["video_sha256"]},
        "task5": {"manifest_sha256": task5["manifest_sha256"], "plan_sha256": task5["plan_sha256"],
                  "direction_file_sha256": dict(task5["direction_file_sha256"]), "direction_sha256": dict(task5["direction_sha256"])},
        "group": {"state": inputs.state, "seed": inputs.seed}, "prompt": inputs.prompt,
        "action": np.asarray(inputs.action, dtype=np.float32).tolist(),
        "settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0, "batch_size": 1,
                     "autocast": False, "tf32": False, "diffusion_cache": False},
        "cache_flags": {"autocast": False, "tf32": False, "diffusion_cache": False},
        "seed_routes": {"model": int(getattr(runtime, "model_seed", inputs.seed)), "prepare": int(inputs.seed), "sampler": int(inputs.seed), "scheduler": int(inputs.seed)},
        "geometry": {"carrier_shape": list(inputs.z0.shape), "condition_indexes": list(geometry.condition_indexes),
                     "predicted_indexes": list(geometry.predicted_indexes), "mask_shape": list(inputs.geometry.mask.shape)},
        "runtime_identity": actual,
    }
    return observation


def _observed_fps(runtime: Any, inputs: Any) -> int:
    """Read the resolved sample/data-batch fps, never a caller default."""
    candidates = [getattr(runtime, "fps", None), getattr(runtime, "sample_fps", None),
                  getattr(getattr(runtime, "sample_args", None), "fps", None),
                  getattr(getattr(inputs, "sample_args", None), "fps", None)]
    for value in candidates:
        if value is not None:
            try: result = int(value)
            except (TypeError, ValueError): continue
            if result > 0: return result
    raise OperationalEvidenceError("live runtime did not expose resolved sample fps")


def _live_framework_commit(framework_root: str | os.PathLike[str] | None) -> str:
    """Read-only framework checkout identity; dirty/non-git trees fail closed."""
    if not framework_root:
        raise OperationalEvidenceError("framework_root is required for live commit observation")
    import subprocess
    path = Path(framework_root).resolve()
    try:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalEvidenceError("framework root is not a readable git checkout") from error
    commit = result.stdout.strip()
    if len(commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in commit) or dirty.stdout.strip():
        raise OperationalEvidenceError("framework checkout is dirty or has no valid HEAD")
    return commit.lower()


def import_callable(spec: str) -> Callable[..., Any]:
    if not isinstance(spec, str) or ":" not in spec:
        raise OperationalEvidenceError("loader must be an explicit module:function spec")
    module_name, function_name = spec.split(":", 1)
    if not module_name or not function_name:
        raise OperationalEvidenceError("loader must be an explicit module:function spec")
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise OperationalEvidenceError(f"loader is not callable: {spec}")
    return function


class OfficialRuntimeFactory:
    """Construct one validated official runtime from an explicit loader.

    The loader is an installed-environment adapter, not a test fake: it must
    return ``model`` and ``data_batch`` and may return the verified ops,
    scheduler and artifact paths required by ``OfficialPrecisionRuntime``.
    """
    def __init__(self, *, loader: str | Callable[..., Mapping[str, Any]], framework_root: str | os.PathLike[str],
                 checkpoint: str | os.PathLike[str], vae: str | os.PathLike[str], contract: Mapping[str, Any],
                 direction_bank: Any, action: Any, prompt: str, video: str | os.PathLike[str] | None = None,
                 phase: str = "unknown", run_dir: str | os.PathLike[str] | None = None, resume: bool = False):
        self.loader = import_callable(loader) if isinstance(loader, str) else loader
        if not callable(self.loader): raise OperationalEvidenceError("explicit official loader is required")
        checkpoint_source = Path(checkpoint)
        self.framework_root, self.checkpoint, self.vae = Path(framework_root).resolve(), checkpoint_source.resolve(), Path(vae).resolve()
        self._checkpoint_source = checkpoint_source
        if not self.framework_root.exists() or not self.vae.is_file():
            raise OperationalEvidenceError("framework/checkpoint/VAE path is missing")
        self.contract, self.direction_bank = dict(contract), np.asarray(direction_bank, dtype=np.float32)
        self.action, self.prompt = parse_action(action), str(prompt)
        self.video = None if video is None else Path(video).resolve()
        self.phase, self.run_dir, self.resume = str(phase), None if run_dir is None else Path(run_dir).resolve(), bool(resume)
        if self.video is None or not self.video.is_file():
            raise OperationalEvidenceError("pinned bridge video path is missing")
        observed_framework = _live_framework_commit(self.framework_root)
        if observed_framework != str(self.contract["framework_commit"]).lower():
            raise OperationalEvidenceError("framework checkout HEAD differs from launch contract")
        self._unload: Callable[[], Any] | None = None
        _validate_static_contract(self.contract)
        if self.contract["checkpoint_identity"]["sha256"] != checkpoint_content_identity(self._checkpoint_source):
            raise OperationalEvidenceError("checkpoint identity does not match the supplied path")
        if self.contract["vae_sha256"] != sha256_file(self.vae):
            raise OperationalEvidenceError("VAE identity does not match the supplied path")
        if self.contract["code_bundle_sha256"] != code_bundle_sha256():
            raise OperationalEvidenceError("code bundle identity does not match this checkout")

    def build(self):
        if self.video is None:
            raise OperationalEvidenceError("official loader requires the pinned bridge video path")
        payload = self.loader(framework_root=str(self.framework_root), checkpoint=str(self.checkpoint), vae=str(self.vae),
                              device="cuda:0", model_seed=0, prompt=self.prompt, action=self.action.copy(),
                              video=str(self.video), contract=dict(self.contract), run_dir=self.run_dir,
                              phase=self.phase, resume=self.resume)
        if not isinstance(payload, Mapping) or not {"model", "data_batch"}.issubset(payload):
            raise OperationalEvidenceError("official loader must return model and data_batch")
        self._unload = payload.get("unload") if callable(payload.get("unload")) else None
        artifact_paths = payload.get("artifact_paths")
        if not isinstance(artifact_paths, Mapping) or set(artifact_paths) != {"checkpoint", "decoder"}:
            raise OperationalEvidenceError("official loader must bind checkpoint and decoder artifacts")
        if Path(artifact_paths["checkpoint"]).resolve() != self.checkpoint:
            raise OperationalEvidenceError("official loader checkpoint path differs from contract")
        if not Path(artifact_paths["decoder"]).is_file():
            raise OperationalEvidenceError("official loader decoder artifact is missing")
        provenance = payload.get("provenance")
        if (not isinstance(provenance, Mapping) or provenance.get("asset_source_commit") != SOURCE_COMMIT or
                (provenance.get("framework_commit") is not None and provenance.get("framework_commit") != self.contract["framework_commit"]) or
                "diffusion_cache_requested" not in provenance or "diffusion_cache_installed" not in provenance or
                bool(provenance.get("diffusion_cache_requested")) or bool(provenance.get("diffusion_cache_installed"))):
            raise OperationalEvidenceError("official loader provenance does not prove diffusion cache is disabled")
        settings = dict(payload.get("generation_settings", {}))
        if not {"num_steps", "guidance", "shift"}.issubset(settings):
            raise OperationalEvidenceError("official loader must expose all generation settings")
        settings.update({key: getattr(payload.get("model"), key) for key in ("num_steps", "guidance", "shift") if hasattr(payload.get("model"), key)})
        if any(settings.get(key, expected) != expected for key, expected in (("num_steps", 30), ("guidance", 1.0), ("shift", 10.0))):
            raise OperationalEvidenceError("official loader generation settings differ from Task 4 C")
        try:
            from .umi_precision_official import OfficialPrecisionRuntime
        except ImportError:  # pragma: no cover
            from umi_precision_official import OfficialPrecisionRuntime
        base = payload.get("inputs_factory")
        if base is None:
            def base(carrier, indexes, mask, bank):
                frozen = freeze_task5_directions(bank, mask)["directions"]
                return Task6Inputs(carrier, indexes, mask, bank, action=self.action, prompt=self.prompt,
                                    state="bridge_0", seed=0,
                                    direction_hashes={k: sha256_array(v) for k, v in frozen.items()})
        runtime = OfficialPrecisionRuntime(payload["model"], payload["data_batch"], self.direction_bank,
            provenance=dict(provenance), ops=payload.get("ops"), scheduler_class=payload.get("scheduler_class"),
            generation_settings=payload.get("generation_settings"), artifact_paths=payload.get("artifact_paths"),
            inputs_factory=base, model_seed=0)
        if payload.get("fps") != BRIDGE0_FPS:
            raise OperationalEvidenceError("official loader must expose resolved Bridge fps=5")
        runtime.fps = int(payload["fps"])
        encoder = payload.get("encoder")
        if encoder is None:
            encoder = getattr(runtime, "encoder", None)
        if encoder is None or not callable(getattr(encoder, "actual_identity", None)):
            raise OperationalEvidenceError("official loader must expose a content-identified condition encoder")
        encoder.actual_identity()
        # The adapter is the public seam that adds the immutable Task 6
        # evidence schema (z_bar/mask/delta/predicted block) to every record.
        try:
            from .umi_task6_runtime import Task6RuntimeAdapter
        except ImportError:  # pragma: no cover
            from umi_task6_runtime import Task6RuntimeAdapter
        adapter = Task6RuntimeAdapter(runtime, runtime.inputs)
        return adapter, runtime.inputs, encoder

    def unload(self) -> None:
        """Release the loader-owned runtime without constructing another one."""
        if self._unload is not None:
            callback, self._unload = self._unload, None
            callback()


__all__ = ["BRIDGE0_ASSET_SHA256", "BRIDGE0_FPS", "BRIDGE0_PROMPT", "OfficialRuntimeFactory", "OperationalEvidenceError",
           "PINNED_FRAME_SIZE", "PILOT_GROUP", "extract_task5_directions", "import_callable", "prepare_bridge_upload_bundle",
           "checkpoint_content_identity", "code_bundle_sha256", "observe_live_launch", "sha256_file", "state_asset", "validate_launch_contract",
           "verify_pinned_bridge_assets"]
