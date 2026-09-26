"""CPU contracts and live-call orchestration for Task 11's five-chunk run.

The module is deliberately lazy about Cosmos and Torch imports.  Its data
contracts, seed plan, result gates, metrics, and resume binding can all be
checked on CPU before a separately approved live stage loads a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


RECORD_INDEX = 15
CHUNK_LENGTH = 16
HORIZON_CHUNKS = 5
OBSERVATION_COUNT = HORIZON_CHUNKS * CHUNK_LENGTH + 1
TRANSITION_COUNT = HORIZON_CHUNKS * CHUNK_LENGTH
SEED_SCHEDULES: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
SHA256_LENGTH = 64


def _require_f32(value: Any, *, name: str, shape: tuple[int, ...] | None = None,
                 range01: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError(f"{name} must be float32")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and non-empty")
    if range01 and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        raise ValueError(f"{name} must be in [0,1]")
    return np.ascontiguousarray(array)


def task8_array_hash(value: Any) -> str:
    """Match the frozen Task 8 action/condition array identity convention."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _task8_live_array_hash(value: Any) -> str:
    """Match the frozen live executor's hash-only JSON descriptor."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(tuple(int(dim) for dim in array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class HorizonWindow:
    horizon: int
    input_window_start: int
    rgb: np.ndarray
    actions: np.ndarray
    raw_actions: np.ndarray
    global_frame_indexes: tuple[int, ...]
    global_action_chunks: tuple[int, int]
    local_chunk_index: int


@dataclass(frozen=True)
class HorizonTrajectory:
    """One admitted record with 81 real observations and 80 real actions."""

    record_index: int
    episode_id: str
    rgb: np.ndarray
    states: np.ndarray
    original_actions: np.ndarray
    raw_actions: np.ndarray
    normalized_actions: np.ndarray
    prompt: str

    def __post_init__(self) -> None:
        if isinstance(self.record_index, bool) or int(self.record_index) != RECORD_INDEX:
            raise ValueError(f"Task 11 is locked to record 15, got {self.record_index!r}")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("record 15 episode id must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("record 15 language prompt must be non-empty")
        rgb = _require_f32(self.rgb, name="rgb")
        if rgb.ndim != 4 or rgb.shape[0] != OBSERVATION_COUNT or rgb.shape[1] != 3:
            raise ValueError("Task 11 needs exactly 81 RGB observations with three channels")
        if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
            raise ValueError("RGB observations must be in [0,1]")
        states = _require_f32(self.states, name="states", shape=(OBSERVATION_COUNT, 7))
        original = _require_f32(self.original_actions, name="original_actions")
        if original.shape != (TRANSITION_COUNT, 7):
            raise ValueError("original_actions must contain exactly 80 transitions with shape [80,7]")
        raw = _require_f32(self.raw_actions, name="raw_actions", shape=(HORIZON_CHUNKS, CHUNK_LENGTH, 10))
        normalized = _require_f32(self.normalized_actions, name="normalized_actions",
                                  shape=(HORIZON_CHUNKS, CHUNK_LENGTH, 10))
        for array in (rgb, states, original, raw, normalized):
            array.setflags(write=False)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "original_actions", original)
        object.__setattr__(self, "raw_actions", raw)
        object.__setattr__(self, "normalized_actions", normalized)

    def window(self, horizon: int) -> HorizonWindow:
        """Map generated chunk 1..5 to its exact real 33-frame Task 8 window."""
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= HORIZON_CHUNKS:
            raise ValueError("horizon must be an integer from 1 through 5")
        start = 0 if horizon == 1 else (horizon - 2) * CHUNK_LENGTH
        local_chunk = 0 if horizon == 1 else 1
        first_action_chunk = start // CHUNK_LENGTH
        last_frame = start + 2 * CHUNK_LENGTH
        if last_frame >= OBSERVATION_COUNT or first_action_chunk + 1 >= HORIZON_CHUNKS:
            raise ValueError("requested horizon window would leave the selected real trajectory")
        indexes = tuple(range(start, last_frame + 1))
        if len(indexes) != 33 or indexes[-1] - indexes[0] != 32:
            raise ValueError("Task 11 windows must contain 33 contiguous real observations")
        chunks = (first_action_chunk, first_action_chunk + 1)
        return HorizonWindow(
            horizon=horizon,
            input_window_start=start,
            rgb=self.rgb[start : last_frame + 1],
            actions=np.stack([self.normalized_actions[index] for index in chunks], axis=0),
            raw_actions=np.stack([self.raw_actions[index] for index in chunks], axis=0),
            global_frame_indexes=indexes,
            global_action_chunks=chunks,
            local_chunk_index=local_chunk,
        )


def task8_input_for_window(trajectory: HorizonTrajectory, window: HorizonWindow) -> Any:
    """Build the frozen two-action adapter around one real 33-frame window."""
    try:
        from .task8_frozen.umi_task8_runtime import Task8InputAdapter
    except ImportError:  # pragma: no cover - direct experiments-cwd invocation
        from task8_frozen.umi_task8_runtime import Task8InputAdapter
    provenance = {
        "record_index": trajectory.record_index,
        "episode_id": trajectory.episode_id,
        "horizon": int(window.horizon),
        "input_window_start": int(window.input_window_start),
        "global_frame_indexes": list(window.global_frame_indexes),
        "global_action_chunks": list(window.global_action_chunks),
        "local_chunk_index": int(window.local_chunk_index),
        "adapter": "frozen Task8InputAdapter over observed contiguous frames",
    }
    return Task8InputAdapter(window.rgb, window.actions, trajectory.prompt, provenance, window.raw_actions)


def build_schedule_call_plan(schedule_index: int) -> tuple[dict[str, Any], ...]:
    """Return the locked ten-call plan for one of the two five-seed schedules."""
    if isinstance(schedule_index, bool) or not isinstance(schedule_index, int) or schedule_index not in (0, 1):
        raise ValueError("schedule index must be 0 or 1")
    schedule = SEED_SCHEDULES[schedule_index]
    rows: list[dict[str, Any]] = []
    for call, mode in (("G0", "common"), ("G0_repeat", "repeat")):
        rows.append({
            "sample_id": f"schedule{schedule_index}_{call}_h1_seed{schedule[0]}",
            "call": call, "horizon": 1, "mode": mode, "seed": schedule[0],
            "condition_source": "real_x0", "action_source": "global_chunk_1",
            "chunk_index": 0, "global_action_chunk": 0, "global_condition_frame": 0,
            "input_window_start": 0, "local_action_chunk_index": 0,
            "schedule_index": schedule_index,
        })
    for horizon in range(2, HORIZON_CHUNKS + 1):
        seed = schedule[horizon - 1]
        start = (horizon - 2) * CHUNK_LENGTH
        for call, mode, condition_source, condition_frame in (
            ("TF", "teacher_forced", "real_x16", (horizon - 1) * CHUNK_LENGTH),
            ("AR", "autoregressive", "g0_float_last_fp32", None),
        ):
            rows.append({
                "sample_id": f"schedule{schedule_index}_{call}_h{horizon}_seed{seed}",
                "call": call, "horizon": horizon, "mode": mode, "seed": seed,
                "condition_source": condition_source, "action_source": f"global_chunk_{horizon}",
                "chunk_index": 1, "global_action_chunk": horizon - 1,
                "global_condition_frame": condition_frame,
                "input_window_start": start, "local_action_chunk_index": 1,
                "schedule_index": schedule_index,
            })
    return tuple(rows)


def validate_call_result(spec: Mapping[str, Any], result: Mapping[str, Any], *,
                         expected_action: Any, expected_condition_input: Any,
                         expected_condition_carrier: Any, condition_mask: Any) -> dict[str, Any]:
    """Fail closed unless live result proves actual seed/action/condition routing."""
    if not isinstance(result, Mapping):
        raise ValueError("call result must be an object")
    seed = spec.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("planned model seed is invalid")
    action = _require_f32(expected_action, name="expected action", shape=(16, 10))
    observed_action = _require_f32(result.get("action"), name="observed action", shape=(16, 10))
    if not np.array_equal(action, observed_action):
        raise ValueError("requested action differs from the action returned by the executor")
    action_hash = task8_array_hash(action)
    if result.get("action_hash") != action_hash:
        raise ValueError("requested action hash does not match the executed action")
    provenance = result.get("provenance")
    action_evidence = provenance.get("action_evidence") if isinstance(provenance, Mapping) else None
    effective = action_evidence.get("effective_action") if isinstance(action_evidence, Mapping) else None
    if not isinstance(effective, Mapping) or effective.get("dtype") != "float32":
        raise ValueError("official padded action evidence is missing")
    try:
        effective_shape = tuple(int(value) for value in effective.get("shape", ()))
    except (TypeError, ValueError):
        effective_shape = ()
    if len(effective_shape) != 2 or effective_shape[0] != 16 or effective_shape[1] < 10:
        raise ValueError("official padded action shape is invalid")
    padded_action = np.zeros(effective_shape, dtype=np.float32)
    padded_action[:, :10] = action
    padded_hash = task8_array_hash(padded_action)
    if (effective.get("sha256") != _task8_live_array_hash(padded_action)
            or action_evidence.get("effective_action_hash") != padded_hash):
        raise ValueError("official padded action evidence differs from the requested chunk")
    hashes = result.get("packed_action_token_hashes")
    consumption = result.get("action_consumption")
    if not isinstance(hashes, (list, tuple)) or len(hashes) != 30 or any(value != padded_hash for value in hashes):
        raise ValueError("actual padded action tokens do not match the requested chunk at all 30 steps")
    if (not isinstance(consumption, Mapping) or consumption.get("all_steps_match") is not True
            or int(consumption.get("steps", 0)) != 30
            or tuple(consumption.get("consumed_token_hashes", ())) != tuple(hashes)
            or consumption.get("expected_token_hash") != padded_hash):
        raise ValueError("official all-step action-consumption evidence is invalid")
    noise_hash = result.get("prediction_noise_hash")
    if not isinstance(noise_hash, str) or len(noise_hash) != SHA256_LENGTH or any(c not in "0123456789abcdef" for c in noise_hash):
        raise ValueError("prediction-noise hash is missing or malformed")
    generation = result.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("actual generation evidence is missing")
    if list(generation.get("sampler_generator_seeds", ())) != [seed] * 30:
        raise ValueError("actual sampler seed evidence differs from the requested seed")
    noise = generation.get("noise_evidence")
    if not isinstance(noise, Mapping) or int(noise.get("seed", -1)) != seed or int(noise.get("prepare_seed", -1)) != seed:
        raise ValueError("actual prepared noise seed differs from the requested seed")

    wanted_input = _require_f32(expected_condition_input, name="expected condition input")
    observed_input = _require_f32(result.get("condition_input_fp32"), name="condition input")
    if not np.array_equal(wanted_input, observed_input):
        raise ValueError("condition input differs bitwise from its scheduled source")
    carrier = _require_f32(expected_condition_carrier, name="expected condition carrier")
    mask = np.asarray(condition_mask, dtype=bool)
    if mask.shape != carrier.shape or not np.any(mask):
        raise ValueError("condition mask does not match the expected carrier")
    steps = _require_f32(result.get("condition_steps_fp32"), name="per-step consumed conditions")
    if steps.shape != (30,) + carrier.shape:
        raise ValueError("per-step consumed condition geometry differs")
    if not all(np.array_equal(row[mask], carrier[mask]) for row in steps):
        raise ValueError("per-step consumed condition differs from the scheduled condition")
    return {"action_hash": action_hash, "effective_action_hash": padded_hash, "noise_hash": noise_hash,
            "condition_hash": task8_array_hash(observed_input), "seed": int(seed),
            "action_steps": len(hashes), "condition_steps": int(steps.shape[0])}


def compare_exact_repeat(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    """Require G0 repeat equality over full float tensors and routing evidence."""
    array_fields = ("output_full", "generated_rgb", "decoded_rgb_full", "decoded_last_rgb",
                    "encoded_condition", "condition_input_fp32", "action")
    for key in array_fields:
        left, right = reference.get(key), candidate.get(key)
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise ValueError(f"exact repeat lacks full {key} tensor evidence")
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"exact repeat differs in {key}")
    for key in ("action_hash", "prediction_noise_hash"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"exact repeat differs in {key}")
    if tuple(reference.get("packed_action_token_hashes", ())) != tuple(candidate.get("packed_action_token_hashes", ())):
        raise ValueError("exact repeat differs in consumed action hashes")


def execute_schedule(trajectory: HorizonTrajectory, schedule_index: int,
                     execute_call: Callable[[Mapping[str, Any], HorizonWindow], Mapping[str, Any]], *,
                     truth_conditions: Mapping[int, np.ndarray], condition_mask: Any,
                     embed_condition: Callable[[np.ndarray], np.ndarray], sample_store: Any | None = None,
                     resume: bool = False, future_formal_calls: int = 0,
                     after_sample: Callable[[str, int], Any] | None = None,
                     capture_results: bool | None = None) -> dict[str, Any]:
    """Run one locked schedule while keeping AR on its own prior float output.

    ``execute_call`` is the only live-runtime seam: the callback receives a
    normal Task 8 request spec and the selected real 33-observation window.
    CPU tests inject a deterministic callback; the runner supplies one
    ``Task8LiveExecutor`` bound to a single loaded context.
    """
    if not callable(execute_call) or not callable(embed_condition):
        raise ValueError("schedule execution requires a live-call and condition-embedding function")
    if isinstance(future_formal_calls, bool) or not isinstance(future_formal_calls, int) or future_formal_calls < 0:
        raise ValueError("future formal call count must be a nonnegative integer")
    plan = build_schedule_call_plan(schedule_index)
    required_truth_frames = (0, 16, 32, 48, 64, 80)
    if not isinstance(truth_conditions, Mapping) or any(frame not in truth_conditions for frame in required_truth_frames):
        raise ValueError("shared truth endpoints must be encoded for frames x0, x16, x32, x48, x64, and x80")
    shared_truth = {frame: _require_f32(truth_conditions[frame], name=f"truth condition x{frame}")
                    for frame in required_truth_frames}
    mask = np.asarray(condition_mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        raise ValueError("the admitted condition mask must be non-empty")
    if capture_results is None:
        capture_results = sample_store is None
    records: dict[str, Mapping[str, Any]] = {}
    sample_ids: list[str] = []
    call_evidence: dict[str, dict[str, Any]] = {}
    first_g0: Mapping[str, Any] | None = None
    previous_ar_frame: np.ndarray | None = None
    previous_ar_encoded: np.ndarray | None = None
    tf_evidence_by_horizon: dict[int, dict[str, Any]] = {}

    for ordinal, original_spec in enumerate(plan):
        spec = dict(original_spec)
        window = trajectory.window(int(spec["horizon"]))
        source = str(spec["condition_source"])
        if source == "real_x0":
            condition_rgb = np.array(window.rgb[0], copy=True)
            expected_input = shared_truth[0]
        elif source == "real_x16":
            condition_rgb = np.array(window.rgb[16], copy=True)
            global_frame = int(spec["global_condition_frame"])
            expected_input = shared_truth[global_frame]
        elif source == "g0_float_last_fp32":
            if previous_ar_frame is None or previous_ar_encoded is None:
                raise ValueError("autoregressive horizon requested before the previous AR float endpoint")
            condition_rgb = previous_ar_frame.copy()
        else:
            raise ValueError(f"unknown Task 11 condition source: {source}")
        spec["condition_rgb"] = condition_rgb.copy()
        spec["expected_action_hash"] = task8_array_hash(window.actions[int(spec["local_action_chunk_index"])])
        spec["window_provenance"] = {
            "record_index": trajectory.record_index,
            "episode_id": trajectory.episode_id,
            "global_frame_indexes": list(window.global_frame_indexes),
            "global_action_chunks": list(window.global_action_chunks),
            "local_action_chunk_index": int(spec["local_action_chunk_index"]),
            "input_window_start": int(window.input_window_start),
        }

        store_state = "run"
        if sample_store is not None:
            store_state = sample_store.prepare(str(spec["sample_id"]), resume=resume)
        if store_state == "skip":
            result = sample_store.load_record(str(spec["sample_id"]))
        else:
            result = execute_call(spec, window)
        if not isinstance(result, Mapping):
            raise ValueError(f"{spec['sample_id']} executor returned no result object")
        result = dict(result)
        result.setdefault("condition_rgb", condition_rgb.copy())
        provenance = result.get("provenance")
        if (not isinstance(provenance, Mapping)
                or provenance.get("condition_source") != source
                or provenance.get("condition_rgb_sha256") != _task8_live_array_hash(condition_rgb)):
            raise ValueError(f"{spec['sample_id']} actual encoded RGB condition provenance differs")

        expected_metadata = {
            "sample_id": str(spec["sample_id"]), "call": str(spec["call"]),
            "schedule_index": int(schedule_index), "seed": int(spec["seed"]),
            "horizon": int(spec["horizon"]), "mode": str(spec["mode"]),
            "global_action_chunk": int(spec["global_action_chunk"]),
            "global_condition_frame": spec.get("global_condition_frame"),
            "input_window_start": int(window.input_window_start),
            "local_action_chunk_index": int(spec["local_action_chunk_index"]),
            "window_provenance": dict(spec["window_provenance"]),
        }
        if store_state == "skip":
            for key, expected_value in expected_metadata.items():
                if result.get(key) != expected_value:
                    raise ValueError(f"resume sample metadata differs from the locked plan: {key}")

        if spec["call"] in {"G0", "G0_repeat"}:
            if spec["call"] == "G0":
                first_g0 = result
            else:
                if first_g0 is None:
                    raise ValueError("exact-repeat call is missing its G0 reference")
        elif spec["call"] == "TF":
            pass
        else:
            if previous_ar_encoded is None:
                raise ValueError("autoregressive condition lacks the preceding encoded float endpoint")
            expected_input = previous_ar_encoded

        expected_carrier = _require_f32(embed_condition(expected_input), name="embedded condition carrier")
        action = window.actions[int(spec["local_action_chunk_index"])]
        evidence = validate_call_result(spec, result, expected_action=action,
                                        expected_condition_input=expected_input,
                                        expected_condition_carrier=expected_carrier,
                                        condition_mask=mask)
        generated = _require_f32(result.get("generated_rgb"), name="generated RGB chunk", range01=True)
        full_decoded = _require_f32(result.get("decoded_rgb_full"), name="full decoded RGB", range01=True)
        decoded_last = _require_f32(result.get("decoded_last_rgb"), name="decoded final RGB", range01=True)
        encoded = _require_f32(result.get("encoded_condition"), name="encoded final condition")
        output = _require_f32(result.get("output_full"), name="full generated latent")
        if (generated.ndim != 4 or generated.shape[:2] != (3, CHUNK_LENGTH)
                or full_decoded.ndim != 4 or full_decoded.shape[:2] != (3, CHUNK_LENGTH + 1)
                or decoded_last.shape != generated[:, -1].shape
                or not np.array_equal(decoded_last, generated[:, -1])
                or not np.array_equal(full_decoded[:, 1:], generated)):
            raise ValueError(f"{spec['sample_id']} did not save the 16-frame float chunk and its exact endpoint")
        if encoded.size == 0 or output.size == 0:
            raise ValueError(f"{spec['sample_id']} has an empty latent or condition tensor")
        result.update(expected_metadata)

        if spec["call"] == "G0":
            previous_ar_frame = decoded_last.copy()
            previous_ar_encoded = encoded.copy()
        elif spec["call"] == "G0_repeat":
            assert first_g0 is not None
            compare_exact_repeat(first_g0, result)
            first_g0 = None
        elif spec["call"] == "TF":
            tf_evidence_by_horizon[int(spec["horizon"])] = evidence
        elif spec["call"] == "AR":
            tf = tf_evidence_by_horizon.get(int(spec["horizon"]))
            if tf is None:
                raise ValueError("AR call has no teacher-forced call for this horizon")
            for key in ("action_hash", "effective_action_hash", "noise_hash"):
                if tf.get(key) != evidence.get(key):
                    raise ValueError(f"paired TF/AR calls differ in {key} at horizon {spec['horizon']}")
            previous_ar_frame = decoded_last.copy()
            previous_ar_encoded = encoded.copy()

        call_evidence[str(spec["sample_id"])] = evidence
        sample_ids.append(str(spec["sample_id"]))
        if sample_store is not None and store_state != "skip":
            sample_store.write_success(str(spec["sample_id"]), result)
        if capture_results:
            records[str(spec["sample_id"])] = result
        if not capture_results:
            # Drop the large 16-frame/17-frame outputs before resource sampling;
            # only the small AR float endpoint/encoding and G0 repeat reference
            # intentionally survive across calls.
            del result, generated, full_decoded, decoded_last, encoded, output
            del expected_carrier, expected_input, condition_rgb, window, spec
        if after_sample is not None:
            remaining = future_formal_calls + (len(plan) - ordinal - 1)
            after_sample(str(original_spec["sample_id"]), remaining)
        if store_state == "skip" or not capture_results:
            if capture_results:
                del result

    if len(sample_ids) != 10 or len(set(sample_ids)) != 10:
        raise ValueError("one Task 11 seed schedule must complete exactly ten unique formal calls")
    return {"status": "COMPLETE", "schedule_index": int(schedule_index),
            "seed_schedule": list(SEED_SCHEDULES[schedule_index]), "formal_calls": len(sample_ids),
            "sample_ids": sample_ids, "call_evidence": call_evidence,
            "records": records if capture_results else {}}


def compute_endpoint_rgb_metrics(prediction: Any, truth: Any) -> dict[str, Any]:
    predicted = _require_f32(prediction, name="predicted endpoint", range01=True)
    target = _require_f32(truth, name="truth endpoint", range01=True)
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[0] != 3:
        raise ValueError("endpoint RGB arrays must share [3,H,W] geometry")
    difference = predicted.astype(np.float64) - target.astype(np.float64)
    rmse = float(np.sqrt(np.mean(difference * difference, dtype=np.float64)))
    mae = float(np.mean(np.abs(difference), dtype=np.float64))
    return {"rmse": rmse, "mae": mae,
            "psnr": None if rmse == 0.0 else float(-20.0 * np.log10(rmse)),
            "psnr_status": "infinite_exact_match" if rmse == 0.0 else "finite"}


def compute_error_geometry(teacher_forced: Any, autoregressive: Any, truth: Any) -> dict[str, Any]:
    tf = _require_f32(teacher_forced, name="teacher-forced endpoint", range01=True)
    ar = _require_f32(autoregressive, name="autoregressive endpoint", range01=True)
    target = _require_f32(truth, name="truth endpoint", range01=True)
    if tf.shape != ar.shape or tf.shape != target.shape or tf.ndim != 3 or tf.shape[0] != 3:
        raise ValueError("error geometry requires aligned [3,H,W] RGB endpoints")
    b = tf.astype(np.float64) - target.astype(np.float64)
    p = ar.astype(np.float64) - tf.astype(np.float64)
    e_tf = float(np.mean(b * b, dtype=np.float64))
    e_ar = float(np.mean((ar.astype(np.float64) - target.astype(np.float64)) ** 2, dtype=np.float64))
    cross = float(np.mean(2.0 * b * p, dtype=np.float64))
    change_squared = float(np.mean(p * p, dtype=np.float64))
    denominator = float(np.linalg.norm(b.reshape(-1)) * np.linalg.norm(p.reshape(-1)))
    cosine = None if denominator == 0.0 else float(np.sum(b * p, dtype=np.float64) / denominator)
    return {"tf_squared_error": e_tf, "ar_squared_error": e_ar,
            "cross_term_2dot_over_n": cross, "feedback_change_squared": change_squared,
            "squared_error_difference": e_ar - e_tf,
            "identity_residual": (e_ar - e_tf) - (cross + change_squared),
            "error_feedback_cosine": cosine,
            "feedback_change_rmse": float(np.sqrt(change_squared))}


def compute_masked_error_geometry(teacher_forced: Any, autoregressive: Any, truth: Any,
                                  condition_mask: Any) -> dict[str, Any]:
    """Compute the TF/AR squared-error identity only on authoritative latent slots."""
    tf = _require_f32(teacher_forced, name="teacher-forced latent")
    ar = _require_f32(autoregressive, name="autoregressive latent")
    target = _require_f32(truth, name="truth latent")
    mask = np.asarray(condition_mask, dtype=bool)
    if tf.shape != ar.shape or tf.shape != target.shape or mask.shape != tf.shape or not np.any(mask):
        raise ValueError("masked latent error geometry needs aligned arrays and a non-empty authoritative mask")
    b = tf[mask].astype(np.float64) - target[mask].astype(np.float64)
    p = ar[mask].astype(np.float64) - tf[mask].astype(np.float64)
    e_tf = float(np.mean(b * b, dtype=np.float64))
    e_ar = float(np.mean((ar[mask].astype(np.float64) - target[mask].astype(np.float64)) ** 2,
                          dtype=np.float64))
    cross = float(np.mean(2.0 * b * p, dtype=np.float64))
    change_squared = float(np.mean(p * p, dtype=np.float64))
    denominator = float(np.linalg.norm(b) * np.linalg.norm(p))
    cosine = None if denominator == 0.0 else float(np.sum(b * p, dtype=np.float64) / denominator)
    return {"tf_squared_error": e_tf, "ar_squared_error": e_ar,
            "cross_term_2dot_over_n": cross, "feedback_change_squared": change_squared,
            "squared_error_difference": e_ar - e_tf,
            "identity_residual": (e_ar - e_tf) - (cross + change_squared),
            "error_feedback_cosine": cosine,
            "feedback_change_rmse": float(np.sqrt(change_squared)),
            "masked_element_count": int(mask.sum())}


def masked_latent_metrics(prediction: Any, truth: Any, condition_mask: Any) -> dict[str, Any]:
    predicted = _require_f32(prediction, name="predicted latent")
    target = _require_f32(truth, name="truth latent")
    mask = np.asarray(condition_mask, dtype=bool)
    if predicted.shape != target.shape or mask.shape != target.shape or not np.any(mask):
        raise ValueError("condition latent and mask geometry differs")
    x, y = predicted[mask].astype(np.float64), target[mask].astype(np.float64)
    rms = float(np.sqrt(np.mean((x - y) ** 2, dtype=np.float64)))
    norms = float(np.linalg.norm(x) * np.linalg.norm(y))
    return {"rms": rms, "cosine": None if norms == 0.0 else float(np.dot(x, y) / norms)}


def adjacent_ar_change(previous: Any, current: Any, previous_truth: Any,
                       current_truth: Any) -> dict[str, float]:
    prev = _require_f32(previous, name="previous AR endpoint", range01=True)
    curr = _require_f32(current, name="current AR endpoint", range01=True)
    prev_target = _require_f32(previous_truth, name="previous truth endpoint", range01=True)
    curr_target = _require_f32(current_truth, name="current truth endpoint", range01=True)
    if not (prev.shape == curr.shape == prev_target.shape == curr_target.shape):
        raise ValueError("adjacent endpoints must be time-aligned and share geometry")
    prediction_change = curr.astype(np.float64) - prev.astype(np.float64)
    error_change = (curr.astype(np.float64) - curr_target.astype(np.float64)) - (
        prev.astype(np.float64) - prev_target.astype(np.float64))
    previous_error = prev.astype(np.float64) - prev_target.astype(np.float64)
    current_error = curr.astype(np.float64) - curr_target.astype(np.float64)
    previous_rmse = float(np.sqrt(np.mean(previous_error ** 2, dtype=np.float64)))
    current_rmse = float(np.sqrt(np.mean(current_error ** 2, dtype=np.float64)))
    return {"endpoint_prediction_change_rms": float(np.sqrt(np.mean(prediction_change ** 2, dtype=np.float64))),
            "endpoint_error_change_rms": float(np.sqrt(np.mean(error_change ** 2, dtype=np.float64))),
            "previous_endpoint_rmse": previous_rmse, "current_endpoint_rmse": current_rmse,
            "endpoint_rmse_delta": current_rmse - previous_rmse}


def evaluate_resource_gate(snapshot: Mapping[str, Any], *, phase: str,
                           remaining_calls: int | None = None,
                           sample_bytes: int | None = None) -> dict[str, Any]:
    """Apply the frozen Task 8/7 GPU, cgroup, swap, growth and disk policy."""
    try:
        from .task8_frozen.run_umi_task8_experiment import evaluate_task8_resources
    except ImportError:  # pragma: no cover - direct experiments-cwd invocation
        from task8_frozen.run_umi_task8_experiment import evaluate_task8_resources
    if remaining_calls is not None:
        if sample_bytes is None or isinstance(sample_bytes, bool) or int(sample_bytes) <= 0:
            raise RuntimeError("disk forecast needs a positive measured sample size")
    return evaluate_task8_resources(snapshot, phase=phase, starting_new_sample=phase not in {"post_cleanup"},
                                    remaining_samples=remaining_calls,
                                    mean_success_sample_bytes=sample_bytes)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def lock_run_identity(run_dir: str | Path, identity: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    """Create an immutable run identity or require an exact resume match."""
    root = Path(run_dir).resolve()
    identity_path = root / "run_identity.json"
    expected = json.loads(_canonical_json(dict(identity)))
    if identity_path.exists():
        if not resume:
            raise FileExistsError(f"run identity already exists; pass --resume to continue: {identity_path}")
        try:
            actual = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing run identity is invalid") from error
        if actual != expected:
            raise ValueError("resume identity differs from the locked source, model, code, or seed plan")
        return actual
    if resume:
        raise ValueError("resume requires an existing locked run identity")
    if root.exists() and any(root.iterdir()):
        raise ValueError("a non-empty run directory without identity cannot be reused")
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(identity_path, expected)
    return expected


__all__ = [
    "CHUNK_LENGTH", "HORIZON_CHUNKS", "HorizonTrajectory", "HorizonWindow",
    "OBSERVATION_COUNT", "RECORD_INDEX", "SEED_SCHEDULES", "TRANSITION_COUNT",
    "adjacent_ar_change", "build_schedule_call_plan", "compare_exact_repeat", "execute_schedule",
    "compute_endpoint_rgb_metrics", "compute_error_geometry", "evaluate_resource_gate",
    "compute_masked_error_geometry",
    "file_sha256", "lock_run_identity", "masked_latent_metrics", "task8_array_hash",
    "task8_input_for_window", "validate_call_result",
]
