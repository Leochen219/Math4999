"""Independent, fail-closed A/B/C runner for observed UMI precision contrasts.

No framework import or GPU execution occurs at import. ``OfficialPrecisionRuntime``
in umi_precision_official supplies the actual model seams; this module owns the
numeric inputs, evidence gate and durable call accounting.
"""
from __future__ import annotations

import hashlib
import inspect
import numbers
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

try:
    from .umi_precision_primitives import (canonical_json, fixed_noise_hash, mask_geometry,
        quantize_bf16_fp32, slice_predicted_output, validate_ab_exact_input)
    from .umi_fd_post_vae_bridge import construct_delta, sha256_array, validate_cache_off
    from .umi_fd_post_vae_scan import sha256_file
    from .umi_precision_storage import PrecisionSampleStore as SampleStore, ProcessLock, _atomic_rename_noreplace
    from .umi_precision_identity import fingerprint
except ImportError:
    from umi_precision_primitives import (canonical_json, fixed_noise_hash, mask_geometry,
        quantize_bf16_fp32, slice_predicted_output, validate_ab_exact_input)
    from umi_fd_post_vae_bridge import construct_delta, sha256_array, validate_cache_off
    from umi_fd_post_vae_scan import sha256_file
    from umi_precision_storage import PrecisionSampleStore as SampleStore, ProcessLock, _atomic_rename_noreplace
    from umi_precision_identity import fingerprint


def projection(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    result = np.ascontiguousarray(value, dtype=np.float32)
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError("tensor projection must be finite and nonempty")
    return result.copy()


class TensorEvidence:
    def __init__(self):
        self.rows = []
        self.casts = []

    def record(self, role, value, *, step=None, name=None):
        native = str(value.dtype).removeprefix("torch.")
        values = projection(value)
        self.rows.append({"role": role, "name": name, "step": step, "native_dtype": native,
                          "projection_dtype": "float32", "shape": list(values.shape),
                          "sha256_fp32": sha256_array(values)})
        return values


class PrecisionInputs:
    def __init__(self, z0, condition_indexes, packed_mask, direction_bank):
        self.z0 = projection(z0)
        self.geometry = mask_geometry(condition_indexes, packed_mask, self.z0.shape)
        bank = projection(direction_bank)
        if bank.ndim != self.z0.ndim + 1 or bank.shape[1:] != self.z0.shape:
            raise ValueError("old direction bank must match runtime carrier shape")
        self.direction = bank[0].copy()
        mask = self.geometry.mask
        if np.any(self.direction[~mask]) or not np.isclose(np.sqrt(np.mean(self.direction[mask].astype(np.float64)**2)), 1, atol=1e-6):
            raise ValueError("direction 0 must have unit mask RMS and zero exterior")
        self.z_bar = quantize_bf16_fp32(self.z0)
        self.bank_hash = sha256_array(bank)
        for value in (self.z0, self.z_bar, self.direction, self.geometry.mask):
            value.setflags(write=False)

    def for_call(self, group, alpha, sign):
        if group not in "ABC" or len(group) != 1:
            raise ValueError("unknown precision group")
        if alpha == 0:
            target = self.z_bar.copy()
        else:
            target = construct_delta(self.z_bar, self.geometry.mask, self.direction, alpha=alpha, sign=sign).latent
        if group in "AB":
            target = quantize_bf16_fp32(target)
        if not np.array_equal(target[~self.geometry.mask], self.z_bar[~self.geometry.mask]):
            raise ValueError("condition perturbation escaped mask")
        return target

    def identity(self):
        return {"z0": sha256_array(self.z0), "z_bar": sha256_array(self.z_bar),
                "direction_bank": self.bank_hash, "direction_0": sha256_array(self.direction),
                "geometry": self.geometry.metadata(), "quantizer": "bf16_round_to_nearest_even"}


def build_call_plan(alphas, model_seed=0):
    alphas = tuple(float(x) for x in alphas)
    if len(alphas) != 6 or any(not np.isfinite(x) or x <= 0 for x in alphas) or sorted(set(alphas)) != list(alphas):
        raise ValueError("exactly six distinct increasing positive finite alphas required")
    if isinstance(model_seed, bool) or not isinstance(model_seed, numbers.Integral):
        raise ValueError("model seed must be a non-negative integer")
    model_seed = int(model_seed)
    if model_seed < 0:
        raise ValueError("model seed must be a non-negative integer")
    result = []
    for group in "ABC":
        def entry(kind, alpha=0., sign=0, ordinal=None):
            suffix = kind if ordinal is None else f"alpha_{ordinal:02d}_{'plus' if sign == 1 else 'minus'}"
            return {"sample_id": f"{group}_{suffix}", "group": group, "kind": kind,
                    "alpha": alpha, "sign": sign, "direction_index": 0, "model_seed": model_seed}
        result.append(entry("pre"))
        for i, alpha in enumerate(alphas):
            for sign in (1, -1):
                result.append(entry("perturbation", alpha, sign, i))
        result.append(entry("post"))
    return result


class EvidenceError(ValueError):
    """Invalid experimental evidence: never reclassified as kernel incompatibility."""


class PrecisionCompatibilityError(EvidenceError):
    """Observed compute cannot satisfy requested precision; allow module gate."""


def validate_capture(record, spec, inputs, scope, *, paired_noise=None, require_decoded=True):
    if not isinstance(require_decoded, bool):
        raise ValueError("require_decoded must be a boolean")
    if not require_decoded and scope != "full":
        raise ValueError("deferred decode validation is only valid for full generation scope")
    mask = inputs.geometry.mask
    expected_seed_raw = spec.get("model_seed", spec.get("seed", 0))
    if isinstance(expected_seed_raw, bool) or not isinstance(expected_seed_raw, numbers.Integral):
        raise ValueError("model seed must be a non-negative integer")
    expected_seed = int(expected_seed_raw)
    if expected_seed < 0:
        raise ValueError("model seed must be non-negative")
    # Task 4 uses the fixed direction-0 ``for_call`` contract.  Later bounded
    # experiments may carry a frozen direction id in the call spec; retain the
    # same actual Cosmos observation boundary rather than duplicating it.
    expected = inputs.for_spec(spec) if hasattr(inputs, "for_spec") else inputs.for_call(spec["group"], spec["alpha"], spec["sign"])
    actual = projection(record["common_input_fp32"])
    if not validate_ab_exact_input(actual, expected)["passed"]:
        raise EvidenceError("observed common interface differs from requested values")
    if not np.array_equal(actual[~mask], inputs.z_bar[~mask]):
        raise EvidenceError("mask exterior changed")
    rows = record["tensor_evidence"]
    dtype = "bfloat16" if spec["group"] == "A" else "float32"
    required = {"weights", "common_condition", "network_condition", "activation", "denoiser_output"}
    if scope == "full":
        required.update({"sampler_update", "sampler_velocity", "sampler_accumulator", "sampler_converted"})
    if not required.issubset({row["role"] for row in rows}):
        raise EvidenceError("missing observed dtype roles")
    for row in rows:
        want = "float32" if row["role"] in {"common_condition", "sampler_update", "sampler_accumulator", "sampler_converted"} else dtype
        # BF16 models may accumulate normalization in FP32. The exact observations
        # are retained; B/C must have no lower precision anywhere in dispatch.
        valid = row["native_dtype"] == want or (spec["group"] == "A" and row["role"] == "activation" and row["native_dtype"] == "float32")
        if not valid:
            raise PrecisionCompatibilityError(f"hidden cast or wrong native dtype: {row['role']}={row['native_dtype']}")
    steps = record["steps"]
    count = int(record["expected_steps"])
    if count < 1 or len(steps) != count or (scope == "module" and count != 1):
        raise EvidenceError("incorrect denoiser step coverage")
    if [x["step"] for x in steps] != list(range(count)) or not all(np.isfinite(x["timestep"]) for x in steps):
        raise EvidenceError("invalid denoiser step/timestep observations")
    if scope == "full" and record.get("sampler_generator_seeds") != [expected_seed] * count:
        raise EvidenceError("actual scheduler generator seed evidence is missing or unpaired")
    for role in required - {"weights", "common_condition"}:
        if {r["step"] for r in rows if r["role"] == role} != set(range(count)):
            raise EvidenceError(f"missing per-step {role} evidence")
    conditions = np.asarray(record["condition_steps_fp32"])
    if conditions.shape != (count,) + actual.shape:
        raise EvidenceError("condition observation shape mismatch")
    if not all(np.array_equal(x[mask], expected[mask]) for x in conditions):
        raise EvidenceError("condition not persistent at network reads")
    cache = record["cache"]
    validate_cache_off(cache["requested"], cache["installed"])
    if not cache["initial_empty"] or not cache["final_empty"]:
        raise EvidenceError("request cache not empty")
    if any(record["request_initial_state"].values()):
        raise EvidenceError("request-local mutable state was not reset")
    execution = record["execution"]
    if any(execution[key] for key in ("autocast", "tf32_matmul", "tf32_cudnn")):
        raise EvidenceError("autocast/TF32 must be disabled")
    if not execution["dispatch_observed"] or execution["operation_count"] <= 0 or not execution["backend"]:
        raise EvidenceError("missing observed backend/operation evidence")
    if spec["group"] in "BC" and any(c["to"] != "float32" for c in execution["casts"]):
        raise PrecisionCompatibilityError("hidden non-FP32 cast in FP32 implementation")
    initial = projection(record["initial_state"]).reshape(mask.shape)
    consumed = projection(record["consumed_initial_state"])
    if consumed.shape != initial.shape or not np.array_equal(initial, consumed):
        raise EvidenceError("consumed initial state differs from frozen preparation")
    sampler_input = projection(record["sampler_input_state"])
    if sampler_input.shape != initial.shape or not np.array_equal(initial, sampler_input):
        raise EvidenceError("sampler consumed initial state differs from preparation")
    if not np.array_equal(np.asarray(record["consumed_initial_mask"], dtype=bool), mask):
        raise EvidenceError("consumed initial mask differs from frozen geometry")
    noise_evidence = record["noise_evidence"]
    if noise_evidence != {"source": "first_velocity_input", "seed": expected_seed, "prepare_seed": expected_seed, "batch_size": 1}:
        raise EvidenceError("missing or mismatched actual noise policy evidence")
    noise_hash = fixed_noise_hash(initial, mask)
    if paired_noise is not None and noise_hash != paired_noise:
        raise EvidenceError("paired initial sampler noise changed")
    if not np.array_equal(initial[mask], actual[mask]):
        raise EvidenceError("initial sampler condition differs from interface")
    selected, slicing = slice_predicted_output(record["output_full"], inputs.geometry)
    if scope == "full":
        if require_decoded:
            frame = projection(record["decoded_final"])
            if frame.ndim != 3 or frame.shape[0] != 3:
                raise EvidenceError("decoded final float frame must be [3,H,W]")
        else:
            if record.get("decode_policy") != "deferred":
                raise EvidenceError("deferred capture must identify decode_policy=deferred")
            if "decoded_final" in record:
                raise EvidenceError("deferred capture must not contain a decoded frame")
    elif "decoded_final" in record:
        raise EvidenceError("module fallback must never decode")
    direction = inputs.direction_for_spec(spec) if hasattr(inputs, "direction_for_spec") else inputs.direction
    target_delta = np.subtract(np.asarray(expected, dtype=np.float32), np.asarray(inputs.z_bar, dtype=np.float32), dtype=np.float32)
    if float(spec.get("alpha", 0.0)) == 0.0:
        theoretical_delta = np.zeros_like(target_delta, dtype=np.float32)
    else:
        s_z = float(getattr(inputs, "s_z", np.sqrt(np.mean(
            np.asarray(inputs.z_bar, dtype=np.float32)[mask].astype(np.float64) ** 2,
            dtype=np.float64))))
        theoretical_delta = np.multiply(
            np.float32(int(spec["sign"]) * float(spec["alpha"]) * s_z),
            np.asarray(direction, dtype=np.float32), dtype=np.float32)
        theoretical_delta[~mask] = 0.0
    actual_delta = np.subtract(np.asarray(actual, dtype=np.float32), np.asarray(inputs.z_bar, dtype=np.float32), dtype=np.float32)
    record.update({"initial_noise_hash": noise_hash, "predicted_latent": selected,
                   "latent_slicing": slicing, "actual_delta_fp32": actual_delta,
                   "target_delta_fp32": target_delta,
                   "theoretical_delta_fp32": theoretical_delta,
                   "mask": mask.copy(), "direction": direction.copy(), "z_bar": inputs.z_bar.copy(),
                   "primary_quantity": "predicted_predecode_latent" if scope == "full" else "step_0_predicted_denoiser_output"})
    return noise_hash


def _atomic_json(path, value):
    path = Path(path)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _strict_success(path, identity):
    status = json.loads((path / "status.json").read_text())
    if status.get("status") != "success":
        return
    hashes = status.get("artifact_sha256", {})
    if not hashes or "sample.json" not in hashes:
        raise ValueError("successful sample lacks strict artifact hashes")
    if any(not (path / name).is_file() or sha256_file(path / name) != digest for name, digest in hashes.items()):
        raise ValueError("successful sample integrity mismatch; refusing overwrite")
    if json.loads((path / "sample.json").read_text())["identity"] != identity:
        raise ValueError("sample resume identity mismatch")


def run_precision_experiment(runtime, inputs, run_dir, *, alphas, resume=False):
    """Run bounded diagnostics then 42 formal calls; never launch implicitly."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    with ProcessLock(root / ".runner.lock"):
        return _run_precision_experiment_locked(runtime, inputs, root, alphas=alphas, resume=resume)


def _run_precision_experiment_locked(runtime, inputs, run_dir, *, alphas, resume):
    plan = build_call_plan(alphas, model_seed=getattr(runtime, "model_seed", 0))
    root = Path(run_dir)
    source_paths = {Path(__file__).resolve()}
    for obj in (quantize_bf16_fp32, construct_delta, SampleStore, fingerprint, sha256_file, _atomic_rename_noreplace, type(runtime)):
        source = inspect.getsourcefile(obj)
        if source is None:
            raise ValueError("runner/runtime source identity cannot be established")
        source_paths.add(Path(source).resolve())
    config = {"version": 1, "runtime": runtime.provenance, "inputs": inputs.identity(), "plan": plan,
              "actual_runtime": runtime.actual_identity(),
              "source_sha256": {str(path): sha256_file(path) for path in sorted(source_paths)}}
    identity = hashlib.sha256(canonical_json(config).encode()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists():
        if not resume:
            raise FileExistsError("run already exists; explicit strict resume required")
        if json.loads(config_path.read_text()) != json.loads(canonical_json(config)):
            raise ValueError("strict resume config/hash mismatch")
    elif any(path.name != ".runner.lock" for path in root.iterdir()):
        raise ValueError("nonempty run directory without precision config")
    else:
        _atomic_json(config_path, config)
    def execute_locked():
        for dirname in ("samples", "diagnostics"):
            folder = root / dirname
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_dir() and not path.name.startswith(".") and ".attempt." not in path.name and (path / "status.json").exists():
                        _strict_success(path, identity)
        summary_path = root / "status.json"
        previous = json.loads(summary_path.read_text()) if resume and summary_path.exists() else {}
        if previous.get("status") in ("complete", "module_complete", "blocked"):
            return previous
        summary = {"status": "running", "scope": previous.get("scope", "full"),
                   "formal_successful": 0, "diagnostic_attempts": previous.get("diagnostic_attempts", 0),
                   "compatibility_attempts": previous.get("compatibility_attempts", []), "full_sampling_claim": False, "image_claim": False,
                   "identity": identity}
        for attempt in summary["compatibility_attempts"]:
            if attempt.get("status") == "running":
                attempt["status"] = "interrupted"
        paired_noise = None
        schedules = {}
        decode_id = runtime.provenance["decode"]
        def execute(spec, scope, diagnostic):
            nonlocal paired_noise
            store = SampleStore(root / ("diagnostics" if diagnostic else "samples"))
            sample_id = spec["sample_id"]
            disposition = store.prepare(sample_id, resume=resume, required_files=("sample.json",))
            if disposition == "skip":
                record = store.load_record(sample_id)
                # Top-level tensors are explicit artifacts, not nested JSON references.
                validate_capture(record, spec, inputs, scope, paired_noise=paired_noise)
                paired_noise = record["initial_noise_hash"]
                schedule = [step["timestep"] for step in record["steps"]]
                if scope in schedules and schedules[scope] != schedule:
                    raise EvidenceError("paired timestep schedule changed")
                schedules[scope] = schedule
                return record
            start = time.perf_counter()
            record = {}
            try:
                record = dict(runtime.execute(spec, inputs, scope=scope))
                paired_noise = validate_capture(record, spec, inputs, scope, paired_noise=paired_noise)
                schedule = [step["timestep"] for step in record["steps"]]
                if scope in schedules and schedules[scope] != schedule:
                    raise EvidenceError("paired timestep schedule changed")
                schedules[scope] = schedule
                if record["decode_id"] != decode_id:
                    raise EvidenceError("decode implementation changed across groups")
                record.update({"status": "success", "identity": identity, "spec": spec,
                               "elapsed_seconds": time.perf_counter() - start})
                arrays = {k + ".npy": v for k, v in record.items() if isinstance(v, np.ndarray)}
                store.write_success(sample_id, {k: v for k, v in record.items() if not isinstance(v, np.ndarray)}, artifacts=arrays)
                return record
            except Exception as error:
                payload = {"status": "fail", "identity": identity, "spec": spec, "scope": scope,
                           "elapsed_seconds": time.perf_counter() - start,
                           "error": {"type": type(error).__name__, "message": str(error)},
                           "traceback": traceback.format_exc(),
                           "partial_capture": getattr(error, "capture", record)}
                store.write_failure(sample_id, payload)
                raise
        def diagnostics(scope):
            records = {}
            for group in "ABC":
                spec = {"sample_id": f"{scope}_{group}_zero", "group": group, "alpha": 0., "sign": 0,
                        "kind": "diagnostic", "direction_index": 0,
                        "model_seed": int(getattr(runtime, "model_seed", 0))}
                attempt = next((item for item in summary["compatibility_attempts"]
                                if item["sample_id"] == spec["sample_id"] and item.get("status") == "success"), None)
                if attempt is None:
                    # The diagnostic budget is global to this run, including
                    # interrupted/resumed attempts.  Check before allocating
                    # another model call so a seventh diagnostic can never be
                    # launched after the six-call hard ceiling.
                    if int(summary["diagnostic_attempts"]) >= 6:
                        raise EvidenceError("diagnostic call maximum 6 exceeded")
                    summary["diagnostic_attempts"] += 1
                    attempt = {"scope": scope, "group": group, "sample_id": spec["sample_id"], "status": "running"}
                    summary["compatibility_attempts"].append(attempt)
                _atomic_json(summary_path, summary)
                try:
                    records[group] = execute(spec, scope, True)
                    attempt["status"] = "success"
                except Exception as error:
                    attempt.update({"status": "fail", "error": {"type": type(error).__name__, "message": str(error)}})
                    _atomic_json(summary_path, summary)
                    raise
                _atomic_json(summary_path, summary)
            for left, right in (("A", "B"), ("B", "C")):
                if not validate_ab_exact_input(records[left]["common_input_fp32"], records[right]["common_input_fp32"])["passed"]:
                    raise EvidenceError(f"{left}/{right} zero common input identity failed")
            if not np.array_equal(records["B"]["predicted_latent"], records["C"]["predicted_latent"]):
                raise EvidenceError("B/C zero output identity failed")
        try:
            if summary["scope"] == "full":
                previous_failure = next((item for item in summary["compatibility_attempts"]
                                         if item["scope"] == "full" and item.get("status") == "fail"), None)
                if previous_failure:
                    error_type = previous_failure.get("error", {}).get("type")
                    if error_type != "PrecisionCompatibilityError":
                        raise EvidenceError("saved full diagnostic failure cannot be retried or downgraded")
                    # A saved compatibility failure is the only legal reason
                    # to enter the complete-denoiser module fallback on resume.
                    summary["scope"] = "module"
                    _atomic_json(summary_path, summary)
                else:
                    try:
                        diagnostics("full")
                    except PrecisionCompatibilityError:
                        # Exactly one bounded fallback, preserving full failure metadata.
                        summary["scope"] = "module"
                        _atomic_json(summary_path, summary)
            if summary["scope"] == "module":
                diagnostics("module")
            scope = summary["scope"]
            common = {}
            zero_outputs = {}
            for spec in plan:
                record = execute(spec, scope, False)
                key = (spec["kind"], spec["alpha"], spec["sign"])
                if spec["group"] == "A":
                    common[key] = record["common_input_fp32"].copy()
                elif spec["group"] == "B" and not validate_ab_exact_input(common[key], record["common_input_fp32"])["passed"]:
                    raise EvidenceError("formal A/B common interface identity failed")
                if spec["alpha"] == 0:
                    if spec["group"] == "B":
                        zero_outputs[spec["kind"]] = record["predicted_latent"].copy()
                    elif spec["group"] == "C" and not np.array_equal(zero_outputs[spec["kind"]], record["predicted_latent"]):
                        raise EvidenceError("formal B/C zero output identity failed")
                summary["formal_successful"] += 1
                _atomic_json(summary_path, summary)
            summary.update({"status": "complete" if scope == "full" else "module_complete",
                            "full_sampling_claim": scope == "full", "image_claim": scope == "full"})
        except Exception as error:
            summary.update({"status": "blocked", "error": {"type": type(error).__name__, "message": str(error)}})
        _atomic_json(summary_path, summary)
        return summary
    return execute_locked()
