"""Remote Task 4 launcher for the observed A/B/C precision experiment.

This script is intentionally explicit about the environment gate and paths.
It reuses the old run only for immutable CPU reanalysis, loads no model from a
remote URL, selects a never-used run directory, and delegates all model calls
to ``OfficialPrecisionRuntime`` plus the fail-closed 42-call runner.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

try:
    from .analyze_umi_precision_contrast import (_load_existing_analysis, _load_precision_records,
        analyze_run, sha256_file)
    from .umi_precision_reanalysis import is_task4_raw_manifest_excluded, reanalyze_old_run
    from .umi_precision_runtime import run_precision_experiment
except ImportError:  # direct import from experiments/
    from analyze_umi_precision_contrast import (_load_existing_analysis, _load_precision_records,
        analyze_run, sha256_file)
    from umi_precision_reanalysis import is_task4_raw_manifest_excluded, reanalyze_old_run
    from umi_precision_runtime import run_precision_experiment


ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
DEFAULT_RUN_ROOT = None
DEFAULT_FRAMEWORK_ROOT = None
DEFAULT_CHECKPOINT = None
DEFAULT_OLD_RUN = None
RUN_PREFIX = "umi_precision_contrast_run"
LEGACY_DIRECTION_SEED = 20260912
LEGACY_DIRECTION_BANK_SHA256 = "094096cd659dbfa7809be7a6c24cd0775fe6b548b6910a6ee55a41ca9f7f6d86"
_RUN_NAME = re.compile(r"^umi_precision_contrast_run([0-9]+)$")


class BlockedExecution(RuntimeError):
    """A hard environment/provenance blocker; never downgraded to a fake run."""


def select_lowest_unused_run(run_root: str | Path, *, prefix: str = RUN_PREFIX) -> Path:
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    used: set[int] = set()
    for path in root.iterdir():
        match = re.fullmatch(re.escape(prefix) + r"([0-9]+)", path.name)
        if match:
            used.add(int(match.group(1)))
    index = 1
    while index in used:
        index += 1
    return root / f"{prefix}{index:02d}"


def require_torch_contract(torch_module: Any) -> None:
    version = str(getattr(torch_module, "__version__", ""))
    cuda = str(getattr(getattr(torch_module, "version", None), "cuda", ""))
    if version != "2.10.0+cu130" or cuda != "13.0":
        raise BlockedExecution(f"remote torch contract requires 2.10.0+cu130 (CUDA 13.0), observed {version!r} / {cuda!r}")
    cuda_api = getattr(torch_module, "cuda", None)
    if cuda_api is None or not bool(cuda_api.is_available()):
        raise BlockedExecution("remote torch contract requires an available CUDA GPU")


def verify_uniform_eager_runtime(adapter: Any) -> dict[str, Any]:
    """Fail closed unless the complete denoiser resolves to one eager backend.

    The public adapter methods alone are not sufficient evidence: a compiled
    wrapper can close over the original BF16 network while reporting cloned
    FP32 weights.  Inspect the actual ``model.net`` owner and the model's
    denoiser dispatch methods, and persist exactly which objects were checked.
    """

    setup = getattr(adapter, "runtime_setup", {})
    if not isinstance(setup, Mapping) or setup.get("use_torch_compile") is not False:
        raise BlockedExecution("runtime did not resolve use_torch_compile=False; uniform eager A/B/C comparison is unavailable")
    model = getattr(adapter, "model", None)
    if model is None:
        raise BlockedExecution("resolved adapter has no model to inspect for the eager denoiser gate")
    net = getattr(model, "net", None)
    if net is None:
        raise BlockedExecution("resolved adapter.model.net is missing; the complete denoiser owner cannot be verified")
    required = (("model.net.forward", net, "forward", "model.net"),
                ("model.net._encode_text", net, "_encode_text", "model.net"),
                ("model.net._encode_vision", net, "_encode_vision", "model.net"),
                ("model.net._encode_action", net, "_encode_action", "model.net"),
                ("model.net._decode_vision", net, "_decode_vision", "model.net"),
                ("model.net._decode_action", net, "_decode_action", "model.net"),
                ("model._get_velocity", model, "_get_velocity", "model"),
                ("model.denoise", model, "denoise", "model"))
    stale: list[str] = []
    checked: list[str] = []
    checked_owners: list[str] = []
    checked_submodules: list[str] = []
    owner_bindings: dict[str, str] = {}

    def inspect_candidate(label: str, candidate: Any) -> None:
        if candidate is None:
            return
        cls = type(candidate)
        if "OptimizedModule" in cls.__name__ or "dynamo" in str(cls.__module__).lower():
            stale.append(label)
        if any(hasattr(candidate, attr) for attr in ("_torchdynamo_orig_callable", "_orig_mod", "_torchdynamo_inline")):
            stale.append(label)
        function = getattr(candidate, "__func__", None)
        if function is not None and function is not candidate:
            inspect_candidate(label, function)

    for label, owner_object, method_name, owner_label in required:
        method = getattr(owner_object, method_name, None)
        if not callable(method):
            raise BlockedExecution(f"required eager denoiser network method is missing or not callable: {label}")
        checked.append(label)
        owner_bindings[label] = owner_label
        if owner_label not in checked_owners:
            checked_owners.append(owner_label)
        bound_owner = getattr(method, "__self__", None)
        if bound_owner is not owner_object:
            raise BlockedExecution(f"required eager method has the wrong bound owner: {label} (expected {owner_label})")
        inspect_candidate(label, method)
        inspect_candidate(owner_label, owner_object)
        inspect_candidate(owner_label, bound_owner)
        inspect_candidate(label + ".forward", getattr(bound_owner, "forward", None))
    # Check the actual model/net classes as well as bound method objects; this
    # catches wrappers that do not expose Dynamo attributes on the function.
    inspect_candidate("model", model)
    inspect_candidate("model.net", net)
    model_cls = type(model)
    net_cls = type(net)
    inspect_candidate("model.class", model_cls)
    inspect_candidate("model.net.class", net_cls)
    named_modules = getattr(net, "named_modules", None)
    if callable(named_modules):
        try:
            for sub_name, submodule in named_modules():
                label = "model.net" if not sub_name else "model.net." + str(sub_name)
                checked_submodules.append(label)
                inspect_candidate(label, submodule)
        except (RuntimeError, TypeError, ValueError) as error:
            raise BlockedExecution(f"cannot inspect eager denoiser submodules: {error}") from error
    stale = sorted(set(stale))
    if stale:
        raise BlockedExecution("compiled/dynamo wrappers remain on the requested eager runtime: " + ", ".join(stale))
    return {"requested_backend": "eager", "resolved_backend": "eager", "stale_wrappers": [],
            "checked_methods": checked, "checked_owners": checked_owners,
            "owner_bindings": owner_bindings,
            "required_methods": [item[0] for item in required], "checked_submodules": checked_submodules}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed UMI precision contrast on the existing remote environment")
    parser.add_argument("--framework-root", default=DEFAULT_FRAMEWORK_ROOT)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--old-run09", default=DEFAULT_OLD_RUN)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-dir", default=None, help="explicit new run directory; must not already exist unless --resume")
    parser.add_argument("--reanalysis-output", default=None)
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--action-path", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--direction-seed", type=int, default=LEGACY_DIRECTION_SEED,
                        help="legacy direction seed; the old run09 direction bank is reused verbatim")
    parser.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-old-reanalysis", action="store_true")
    args = parser.parse_args(argv)
    if tuple(args.alphas) != ALPHAS:
        parser.error("the formal Task 4 alpha grid is fixed to 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2")
    if args.direction_seed != LEGACY_DIRECTION_SEED:
        parser.error(f"Task 4 must reuse old run09 direction bank seed {LEGACY_DIRECTION_SEED}")
    return args


def _load_old_provenance(old_run: Path) -> dict[str, Any]:
    path = old_run / "provenance.json"
    if not path.is_file():
        raise BlockedExecution(f"old run provenance is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BlockedExecution(f"old run provenance is invalid: {path}: {error}") from error
    if not isinstance(value, dict):
        raise BlockedExecution("old run provenance must be a JSON object")
    return value


def _resolve_inputs(args: argparse.Namespace, old_provenance: dict[str, Any]) -> dict[str, str]:
    values = {
        "framework_root": args.framework_root,
        "checkpoint_path": args.checkpoint_path,
        "vae_path": args.vae_path or old_provenance.get("vae_path"),
        "input_path": args.input_path or old_provenance.get("input_path"),
        "action_path": args.action_path or old_provenance.get("action_path"),
    }
    missing = [name for name, value in values.items() if not value or not Path(value).exists()]
    if missing:
        raise BlockedExecution("required local remote input paths are missing: " + ", ".join(missing))
    return {name: str(Path(value).resolve()) for name, value in values.items()}


def _require_explicit_launcher_paths(args: argparse.Namespace, *names: str) -> None:
    missing = [name.replace("_", "-") for name in names if not getattr(args, name, None)]
    if missing:
        raise BlockedExecution("explicit required path arguments are missing: " + ", ".join(missing))


def load_legacy_direction_bank(old_run: str | Path, carrier: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate old run09's direction bank without generating a new one."""

    old_run = Path(old_run).resolve()
    path = old_run / "direction_bank.npy"
    if not path.is_file():
        raise BlockedExecution(f"old run09 direction bank is missing: {path}")
    observed_hash = sha256_file(path)
    if observed_hash != LEGACY_DIRECTION_BANK_SHA256:
        raise BlockedExecution(
            f"old run09 direction bank hash mismatch: expected {LEGACY_DIRECTION_BANK_SHA256}, observed {observed_hash}"
        )
    try:
        bank = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise BlockedExecution(f"old run09 direction bank cannot be loaded: {path}: {error}") from error
    carrier = np.asarray(carrier, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if bank.dtype != np.float32 or bank.ndim != carrier.ndim + 1 or tuple(bank.shape[1:]) != tuple(carrier.shape):
        raise BlockedExecution(f"old run09 direction bank shape/dtype mismatch: {bank.dtype} {bank.shape}, carrier {carrier.shape}")
    direction = np.asarray(bank[0], dtype=np.float32)
    if not np.all(np.isfinite(direction)) or not np.any(mask):
        raise BlockedExecution("old run09 direction 0 is empty or non-finite")
    if np.any(direction[~mask]):
        raise BlockedExecution("old run09 direction 0 is nonzero outside the runtime condition mask")
    masked_rms = float(np.sqrt(np.mean(direction[mask].astype(np.float64) ** 2, dtype=np.float64)))
    if not np.isclose(masked_rms, 1.0, atol=1e-6, rtol=0.0):
        raise BlockedExecution(f"old run09 direction 0 masked RMS is not one: {masked_rms}")
    return np.array(bank, dtype=np.float32, copy=True), {
        "source": str(path), "sha256": observed_hash, "seed": LEGACY_DIRECTION_SEED,
        "bank_count": int(bank.shape[0]), "direction_0_masked_rms": masked_rms,
    }


def _legacy_args(args: argparse.Namespace, paths: dict[str, str], run_dir: Path) -> argparse.Namespace:
    return SimpleNamespace(
        framework_root=paths["framework_root"], checkpoint_path=paths["checkpoint_path"], vae_path=paths["vae_path"],
        input_path=paths["input_path"], action_path=paths["action_path"], prompt=args.prompt or "mouse arrangement",
        action_chunk_index=0, gpu_index=0, direction_seed=args.direction_seed, model_seed=0,
        alphas=list(ALPHAS), num_steps=30, sampler="unipc", precision="bfloat16",
        parallelism_preset="latency", diffusion_cache=False, batch_size=1, run_dir=str(run_dir),
        resume=bool(args.resume), stage_a_only=False, use_torch_compile=False,
    )


def _write_run_metadata(run_dir: Path, args: argparse.Namespace, paths: dict[str, str], *, torch_module: Any,
                        result: Mapping[str, Any], direction_metadata: Mapping[str, Any],
                        backend_evidence: Mapping[str, Any]) -> None:
    try:
        peak_memory_bytes = int(torch_module.cuda.max_memory_allocated())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        peak_memory_bytes = None
    payload = {
        "schema_version": "umi-input-quantization-compute-precision-run-v1",
        "torch": {"version": str(torch_module.__version__), "cuda": str(torch_module.version.cuda), "device": torch_module.cuda.get_device_name(0)},
        "framework_root": paths["framework_root"], "checkpoint_path": paths["checkpoint_path"], "vae_path": paths["vae_path"],
        "input_path": paths["input_path"], "action_path": paths["action_path"], "cache_requested": False,
        "alphas": list(ALPHAS), "formal_call_count": 42, "diagnostic_count": result.get("diagnostic_attempts"),
        "run_status": result.get("status"), "scope": result.get("scope"), "resume": bool(args.resume),
        "direction_bank": dict(direction_metadata),
        "python": sys.executable, "source_dir": str(Path(__file__).resolve().parent),
        # Keep the exact launcher invocation and effective defaults beside the
        # resolved paths; reports must not reconstruct a command from guesses.
        "argv": list(sys.argv), "effective_args": dict(vars(args)),
        "prompt": args.prompt or "mouse arrangement",
        "old_run09": str(Path(args.old_run09).resolve()), "run_dir": str(run_dir.resolve()),
        "reanalysis_output": str(Path(args.reanalysis_output).resolve()) if args.reanalysis_output else str(Path(args.old_run09).resolve().parent / f"{Path(args.old_run09).name}_precision_reanalysis"),
        "direction_seed": int(args.direction_seed), "skip_old_reanalysis": bool(args.skip_old_reanalysis),
        "device": "cuda:0", "device_name": torch_module.cuda.get_device_name(0),
        "peak_memory_bytes": peak_memory_bytes,
        "requested_backend": backend_evidence.get("requested_backend"), "resolved_backend": backend_evidence.get("resolved_backend"),
        "stale_wrappers": list(backend_evidence.get("stale_wrappers", [])),
        "eager_gate": {"checked_methods": list(backend_evidence.get("checked_methods", [])),
                       "checked_owners": list(backend_evidence.get("checked_owners", [])),
                       "owner_bindings": dict(backend_evidence.get("owner_bindings", {})),
                       "checked_submodules": list(backend_evidence.get("checked_submodules", [])),
                       "required_methods": list(backend_evidence.get("required_methods", []))},
        "model_compile": {"requested": False, "resolved": False, "requested_backend": backend_evidence.get("requested_backend"),
                          "resolved_backend": backend_evidence.get("resolved_backend"), "stale_wrappers": list(backend_evidence.get("stale_wrappers", [])),
                          "path": "uniform eager A/B/C backend; new run pre-registration required"},
    }
    (run_dir / "task4_execution.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_manifest(run_dir: Path) -> Path:
    files = [
        path for path in sorted(run_dir.rglob("*"))
        if path.is_file()
        and not is_task4_raw_manifest_excluded(path.relative_to(run_dir).as_posix())
    ]
    lines = [f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}" for path in files]
    manifest = run_dir / "MANIFEST.sha256"
    manifest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
    return manifest


def execute_task4(args: argparse.Namespace) -> dict[str, Any]:
    required = ["framework_root", "checkpoint_path", "old_run09"]
    if not args.run_dir:
        required.append("run_root")
    _require_explicit_launcher_paths(args, *required)
    old_run = Path(args.old_run09).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else select_lowest_unused_run(args.run_root).resolve()
    if run_dir == old_run or old_run in run_dir.parents or run_dir in old_run.parents:
        raise BlockedExecution("new run directory must be independent of old run09")
    if args.resume:
        if not run_dir.is_dir():
            raise BlockedExecution("--resume requires an existing run directory")
        # Terminal results are immutable.  Resolve and verify them before
        # touching the run parent, importing torch, checking input paths, or
        # loading a model; this is a read-only completed-run resume path.
        status_path = run_dir / "status.json"
        if status_path.is_file():
            try:
                terminal_status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise BlockedExecution(f"existing run status is invalid: {status_path}: {error}") from error
            status_name = str(terminal_status.get("status", "")).lower()
            if status_name in {"blocked", "terminal_blocked"}:
                raise BlockedExecution("terminal BLOCKED precision run is not resumable; use a new revision")
            if status_name in {"complete", "module_complete"}:
                if int(terminal_status.get("formal_successful", 0)) != 42:
                    raise BlockedExecution("terminal precision run lacks the required 42 formal calls")
                records, metadata = _load_precision_records(run_dir)
                if not records:
                    raise BlockedExecution("terminal precision run has no verified formal sample records")
                analysis_dir = run_dir / "precision_analysis"
                analysis = _load_existing_analysis(analysis_dir, run_dir, metadata["raw_manifest"])
                if str(analysis.get("status", "")).upper() != "COMPLETE":
                    raise BlockedExecution("terminal precision run has no COMPLETE immutable analysis package")
                return {"run_dir": str(run_dir), "run": terminal_status, "analysis": analysis,
                        "resume_read_only": True}
    elif run_dir.exists():
        raise FileExistsError(f"refusing to reuse existing precision run directory: {run_dir}")
    if not old_run.is_dir():
        raise BlockedExecution(f"old run09 directory is missing: {old_run}")
    old_provenance = _load_old_provenance(old_run)
    paths = _resolve_inputs(args, old_provenance)
    run_dir.parent.mkdir(parents=True, exist_ok=True)

    import torch
    require_torch_contract(torch)
    if str(paths["framework_root"]) not in sys.path:
        sys.path.insert(0, str(paths["framework_root"]))
    if not args.skip_old_reanalysis:
        reanalysis_output = Path(args.reanalysis_output).resolve() if args.reanalysis_output else old_run.parent / f"{old_run.name}_precision_reanalysis"
        if reanalysis_output.exists():
            raise FileExistsError(f"refusing to overwrite existing old-run reanalysis: {reanalysis_output}")
        reanalyze_old_run(old_run, reanalysis_output)

    from umi_fd_post_vae_scan import _clone_runtime, load_official_data_batch, load_official_runtime
    from umi_precision_official import OfficialPrecisionRuntime, TorchOps, projection

    # The official framework initializer writes logs/configuration to its
    # output directory.  Keep that setup directory beside (not inside) the
    # precision run because the atomic precision runner must start from an
    # otherwise empty destination and own its complete artifact tree.
    setup_dir = run_dir.parent / f".{run_dir.name}.framework_setup"
    legacy = _legacy_args(args, paths, setup_dir)
    adapter = load_official_runtime(legacy, setup_dir)
    backend_evidence = verify_uniform_eager_runtime(adapter)
    data_batch, _sample_args = load_official_data_batch(adapter, legacy, setup_dir)
    ops = TorchOps()
    with ops.inference():
        prepared = adapter.model._prepare_inference_data(_clone_runtime(data_batch), [0], False)
    if not isinstance(prepared, tuple) or len(prepared) != 8:
        raise BlockedExecution("official preparation did not expose the verified eight-field interface")
    carrier = projection(prepared[1].x0_tokens_vision[0])
    mask = projection(prepared[6][0]).reshape(carrier.shape).astype(bool)
    direction_bank, direction_metadata = load_legacy_direction_bank(old_run, carrier, mask)
    provenance = {
        "framework_root": paths["framework_root"], "checkpoint_path": paths["checkpoint_path"],
        "vae_path": paths["vae_path"], "input_path": paths["input_path"], "action_path": paths["action_path"],
        "sampler": "unipc", "precision": "bfloat16", "diffusion_cache_requested": False,
        "diffusion_cache_installed": False, "seed": 0, "direction_seed": args.direction_seed,
        "prompt": legacy.prompt, "runtime": "OfficialPrecisionRuntime model-owned boundaries",
        "direction_bank": direction_metadata,
        "model_compile": {"requested": False, "resolved": False, "requested_backend": backend_evidence["requested_backend"],
                          "resolved_backend": backend_evidence["resolved_backend"], "stale_wrappers": backend_evidence["stale_wrappers"],
                          "path": "uniform eager A/B/C backend"},
    }
    runtime = OfficialPrecisionRuntime(
        adapter.model, data_batch, direction_bank, provenance=provenance, ops=ops,
        generation_settings={"num_steps": 30, "guidance": 1.0, "shift": 10.0},
        artifact_paths={"checkpoint": paths["checkpoint_path"], "decoder": paths["vae_path"]},
    )
    result = run_precision_experiment(runtime, runtime.inputs, run_dir, alphas=ALPHAS, resume=bool(args.resume))
    np.save(run_dir / "z0.npy", runtime.inputs.z0.astype(np.float32), allow_pickle=False)
    np.save(run_dir / "mask.npy", runtime.inputs.geometry.mask.astype(bool), allow_pickle=False)
    np.save(run_dir / "direction_bank.npy", direction_bank.astype(np.float32), allow_pickle=False)
    _write_run_metadata(run_dir, args, paths, torch_module=torch, result=result, direction_metadata=direction_metadata,
                        backend_evidence=backend_evidence)
    _write_run_manifest(run_dir)
    analysis_dir = run_dir / "precision_analysis"
    analysis = analyze_run(run_dir, analysis_dir) if result.get("formal_successful") == 42 else {"status": "BLOCKED", "reason": "formal 42-call set is incomplete", "run_dir": str(run_dir)}
    return {"run_dir": str(run_dir), "run": result, "analysis": analysis}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = execute_task4(args)
    except BlockedExecution as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, ensure_ascii=False, sort_keys=True))
        return 3
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if result.get("analysis", {}).get("status") == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ALPHAS", "BlockedExecution", "DEFAULT_CHECKPOINT", "DEFAULT_FRAMEWORK_ROOT", "DEFAULT_OLD_RUN", "DEFAULT_RUN_ROOT", "execute_task4", "main", "parse_args", "require_torch_contract", "select_lowest_unused_run", "verify_uniform_eager_runtime"]
