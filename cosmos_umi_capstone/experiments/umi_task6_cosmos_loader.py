"""Concrete installed-Cosmos loader for the Task 6 official driver.

This adapter reuses the already-validated Task 5/precision environment setup;
it is intentionally not a second implementation of model construction.
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def load_task6_cosmos_runtime(*, framework_root: str, checkpoint: str, vae: str, device: str,
                              model_seed: int, prompt: str, action: Any, video: str | None = None,
                              contract: dict[str, Any] | None = None) -> dict[str, Any]:
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
        from .umi_fd_post_vae_scan import _clone_runtime, load_official_data_batch, load_official_runtime
        from .umi_precision_official import OfficialPrecisionRuntime, TorchOps
    except ImportError:  # pragma: no cover
        from umi_fd_post_vae_scan import _clone_runtime, load_official_data_batch, load_official_runtime
        from umi_precision_official import OfficialPrecisionRuntime, TorchOps
    setup_dir = Path(video).resolve().parent / ".task6_framework_setup" if video else framework / ".task6_framework_setup"
    setup_dir.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(framework_root=str(framework), checkpoint_path=str(Path(checkpoint).resolve()),
        vae_path=str(Path(vae).resolve()), input_path=str(Path(video).resolve()) if video else None,
        action_path=None, prompt=str(prompt), action_chunk_index=0, gpu_index=0, direction_seed=20260912,
        model_seed=0, alphas=[0.001, 0.003, 0.01], num_steps=30, sampler="unipc", precision="bfloat16",
        parallelism_preset="latency", diffusion_cache=False, batch_size=1, run_dir=str(setup_dir),
        resume=False, stage_a_only=False, use_torch_compile=False)
    # The official sample loader reads the paired action from a path.  Materialize
    # the exact action passed by the factory in the temporary setup directory.
    import json
    action_path = setup_dir / "action.json"
    action_path.write_text(json.dumps(action), encoding="utf-8")
    args.action_path = str(action_path)
    post_adapter = load_official_runtime(args, setup_dir)
    data_batch, _ = load_official_data_batch(post_adapter, args, setup_dir)
    ops = TorchOps()
    provenance = {"framework_root": str(framework), "checkpoint_path": str(Path(checkpoint).resolve()),
        "vae_path": str(Path(vae).resolve()), "sampler": "unipc", "precision": "bfloat16",
        "asset_source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d", "source_commit": "2b17a2413bd86b2cf9b03823637108851e4ddf2d",
        "diffusion_cache_requested": False, "diffusion_cache_installed": False, "seed": 0, "prompt": prompt}
    runtime_model = post_adapter.model
    runtime = {"model": runtime_model, "data_batch": data_batch, "ops": ops,
        "generation_settings": {"num_steps": 30, "guidance": 1.0, "shift": 10.0},
        "artifact_paths": {"checkpoint": str(Path(checkpoint).resolve()), "decoder": str(Path(vae).resolve())},
        "provenance": provenance, "encoder": getattr(runtime_model, "tokenizer_vision", None)}
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


__all__ = ["load_task6_cosmos_runtime"]
