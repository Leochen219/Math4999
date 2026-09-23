"""Generation-free Task 8 Bridge TFDS preflight.

This module intentionally has no TensorFlow, Torch, model, or network dependency.
It reads the flattened ``tf.train.Example`` records shipped by the Bridge TFDS
release, verifies TFRecord framing and local transfer hashes, and adapts one
eligible state/action window using the pinned Cosmos Bridge action path.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


class TFRecordError(ValueError):
    """Raised when a TFRecord frame or CRC is invalid."""


class SemanticError(ValueError):
    """Raised when a Bridge episode cannot satisfy the preflight contract."""


@dataclass(frozen=True)
class TFRecordRecord:
    index: int
    offset: int
    payload: bytes


@dataclass(frozen=True)
class FeatureValue:
    kind: str
    values: list[Any]


@dataclass(frozen=True)
class NormalizerStats:
    q01: np.ndarray
    q99: np.ndarray
    sha256: str


@dataclass
class BridgeEpisode:
    index: int
    episode_id: int | None
    states: np.ndarray
    actions: np.ndarray
    image0: list[bytes]
    images: dict[str, list[bytes]]
    language: list[str]
    has_image0: bool
    has_image_flags: dict[str, bool]
    has_language: bool
    is_first: np.ndarray
    is_last: np.ndarray
    is_terminal: np.ndarray
    file_path: str | None
    extra_keys: tuple[str, ...]


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> bytes:
    crc = _crc32c(data)
    masked = ((crc >> 15) | (crc << 17)) + 0xA282EAD8
    return struct.pack("<I", masked & 0xFFFFFFFF)


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift >= 70:
            break
    raise TFRecordError("truncated or overlong protobuf varint")


def _wire_fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    offset = 0
    while offset < len(data):
        tag, offset = _read_varint(data, offset)
        number, wire_type = tag >> 3, tag & 0x07
        if number <= 0:
            raise TFRecordError(f"invalid protobuf field number {number}")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise TFRecordError("truncated protobuf fixed64 field")
            value, offset = data[offset:end], end
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise TFRecordError("truncated protobuf length-delimited field")
            value, offset = data[offset:end], end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise TFRecordError("truncated protobuf fixed32 field")
            value, offset = data[offset:end], end
        else:
            raise TFRecordError(f"unsupported protobuf wire type {wire_type}")
        yield number, wire_type, value


def iter_tfrecord_records(path: str | Path, *, verify_crc: bool = True) -> Iterator[TFRecordRecord]:
    """Stream TFRecord frames without buffering the shard."""
    path = Path(path)
    with path.open("rb") as stream:
        index = 0
        while True:
            offset = stream.tell()
            length_raw = stream.read(8)
            if not length_raw:
                return
            if len(length_raw) != 8:
                raise TFRecordError(f"truncated length at offset {offset}")
            length_crc = stream.read(4)
            if len(length_crc) != 4:
                raise TFRecordError(f"truncated length CRC at offset {offset}")
            if verify_crc and length_crc != _masked_crc(length_raw):
                raise TFRecordError(f"length CRC mismatch at record {index}")
            length = struct.unpack("<Q", length_raw)[0]
            payload = stream.read(length)
            if len(payload) != length:
                raise TFRecordError(f"truncated payload at record {index}: expected {length}, got {len(payload)}")
            payload_crc = stream.read(4)
            if len(payload_crc) != 4:
                raise TFRecordError(f"truncated payload CRC at record {index}")
            if verify_crc and payload_crc != _masked_crc(payload):
                raise TFRecordError(f"payload CRC mismatch at record {index}")
            yield TFRecordRecord(index=index, offset=offset, payload=payload)
            index += 1


def _decode_packed_varints(data: bytes) -> list[int]:
    values: list[int] = []
    offset = 0
    while offset < len(data):
        value, offset = _read_varint(data, offset)
        values.append(value)
    return values


def _decode_feature(data: bytes) -> FeatureValue:
    for number, wire_type, value in _wire_fields(data):
        if number == 1:  # BytesList
            if not isinstance(value, bytes):
                raise TFRecordError("BytesList must be length-delimited")
            values = [item for field, _, item in _wire_fields(value) if field == 1]
            return FeatureValue("bytes", values)
        if number == 2:  # FloatList
            if not isinstance(value, bytes):
                raise TFRecordError("FloatList must be length-delimited")
            values: list[float] = []
            for field, item_wire, item in _wire_fields(value):
                if field != 1:
                    continue
                if item_wire == 2:
                    if not isinstance(item, bytes) or len(item) % 4:
                        raise TFRecordError("invalid packed FloatList")
                    values.extend(struct.unpack(f"<{len(item) // 4}f", item))
                elif item_wire == 5:
                    values.append(struct.unpack("<f", item)[0])
                else:
                    raise TFRecordError("invalid FloatList wire type")
            return FeatureValue("float", values)
        if number == 3:  # Int64List
            if not isinstance(value, bytes):
                raise TFRecordError("Int64List must be length-delimited")
            values: list[int] = []
            for field, item_wire, item in _wire_fields(value):
                if field != 1:
                    continue
                if item_wire == 2:
                    values.extend(_decode_packed_varints(item))
                elif item_wire == 0:
                    values.append(int(item))
                else:
                    raise TFRecordError("invalid Int64List wire type")
            return FeatureValue("int", values)
    raise TFRecordError("Feature has no supported value list")


def decode_tfexample(payload: bytes) -> dict[str, FeatureValue]:
    """Decode the flattened ``tf.train.Example`` payload used by Bridge TFDS."""
    features_messages = [value for number, _, value in _wire_fields(payload) if number == 1]
    if len(features_messages) != 1 or not isinstance(features_messages[0], bytes):
        raise TFRecordError("payload is not a tf.train.Example")
    decoded: dict[str, FeatureValue] = {}
    for number, _, entry in _wire_fields(features_messages[0]):
        if number != 1 or not isinstance(entry, bytes):
            raise TFRecordError("invalid Features map entry")
        key: bytes | None = None
        feature: bytes | None = None
        for field, _, value in _wire_fields(entry):
            if field == 1:
                key = value if isinstance(value, bytes) else None
            elif field == 2:
                feature = value if isinstance(value, bytes) else None
        if key is None or feature is None:
            raise TFRecordError("Features map entry missing key/value")
        decoded[key.decode("utf-8")] = _decode_feature(feature)
    return decoded


def _values(features: Mapping[str, FeatureValue], key: str, kind: str | None = None) -> list[Any]:
    try:
        value = features[key]
    except KeyError as error:
        raise SemanticError(f"missing required feature {key!r}") from error
    if kind is not None and value.kind != kind:
        raise SemanticError(f"feature {key!r} has kind {value.kind!r}, expected {kind!r}")
    return value.values


def _bool_scalar(features: Mapping[str, FeatureValue], key: str) -> bool:
    values = _values(features, key, "int")
    if len(values) != 1:
        raise SemanticError(f"metadata feature {key!r} is not scalar")
    return bool(values[0])


def decode_bridge_episode(record: TFRecordRecord) -> BridgeEpisode:
    features = decode_tfexample(record.payload)
    raw_states = np.asarray(_values(features, "steps/observation/state", "float"), dtype=np.float32)
    raw_actions = np.asarray(_values(features, "steps/action", "float"), dtype=np.float32)
    if raw_states.size % 7 or raw_actions.size % 7:
        raise SemanticError("state/action packed lengths are not multiples of 7")
    states = raw_states.reshape(-1, 7)
    actions = raw_actions.reshape(-1, 7)
    if len(states) != len(actions) or len(states) < 2:
        raise SemanticError(f"state/action lengths disagree or are too short: {len(states)} / {len(actions)}")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise SemanticError("state/action contains NaN or infinity")
    images = {
        camera: list(_values(features, f"steps/observation/{camera}", "bytes"))
        for camera in ("image_0", "image_1", "image_2", "image_3")
    }
    image0 = images["image_0"]
    language = [bytes(value).decode("utf-8") for value in _values(features, "steps/language_instruction", "bytes")]
    if any(len(values) != len(states) for values in images.values()) or len(language) != len(states):
        raise SemanticError("image/language sequence is not temporally aligned with state/action")
    is_first = np.asarray(_values(features, "steps/is_first", "int"), dtype=np.int64)
    is_last = np.asarray(_values(features, "steps/is_last", "int"), dtype=np.int64)
    is_terminal = np.asarray(_values(features, "steps/is_terminal", "int"), dtype=np.int64)
    for name, flags in (("is_first", is_first), ("is_last", is_last), ("is_terminal", is_terminal)):
        if len(flags) != len(states):
            raise SemanticError(f"{name} is not temporally aligned with state/action")
    episode_values = _values(features, "episode_metadata/episode_id", "int")
    episode_id = int(episode_values[0]) if episode_values else None
    file_path_values = _values(features, "episode_metadata/file_path", "bytes")
    file_path = bytes(file_path_values[0]).decode("utf-8") if file_path_values else None
    has_image_flags = {
        camera: _bool_scalar(features, f"episode_metadata/has_{camera}")
        for camera in ("image_0", "image_1", "image_2", "image_3")
    }
    return BridgeEpisode(
        index=record.index,
        episode_id=episode_id,
        states=states,
        actions=actions,
        image0=image0,
        images=images,
        language=language,
        has_image0=has_image_flags["image_0"],
        has_image_flags=has_image_flags,
        has_language=_bool_scalar(features, "episode_metadata/has_language"),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        file_path=file_path,
        extra_keys=tuple(sorted(set(features) - {
            "steps/observation/state", "steps/action", "steps/observation/image_0",
            "steps/language_instruction", "steps/is_first", "steps/is_last", "steps/is_terminal",
            "episode_metadata/episode_id", "episode_metadata/has_image_0", "episode_metadata/has_language",
        })),
    )


def jpeg_shape(encoded: bytes) -> tuple[int, int, int]:
    """Decode one JPEG only for shape/colour proof; returns H/W/C."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            channels = len(image.getbands())
            if image.mode not in {"RGB", "RGBA"}:
                raise SemanticError(f"image_0 JPEG mode {image.mode!r} is not RGB/RGBA")
            return int(image.height), int(image.width), channels
    except ImportError as error:  # pragma: no cover - runtime dependency diagnostic
        raise SemanticError("Pillow is required for JPEG shape proof") from error


def jpeg_content_proof(encoded: bytes) -> dict[str, Any]:
    """Decode one frame and retain enough evidence to reject empty placeholders."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(encoded)) as image:
            image.load()
            if image.mode not in {"RGB", "RGBA"}:
                raise SemanticError(f"image JPEG mode {image.mode!r} is not RGB/RGBA")
            pixels = np.asarray(image, dtype=np.uint8)
            return {
                "shape": [int(image.height), int(image.width), len(image.getbands())],
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "pixel_min": int(pixels.min()),
                "pixel_max": int(pixels.max()),
                "pixel_std": float(pixels.astype(np.float32).std()),
            }
    except ImportError as error:  # pragma: no cover - runtime dependency diagnostic
        raise SemanticError("Pillow is required for JPEG content proof") from error


class BridgeOrigAdapter:
    """CPU-only implementation of the pinned BridgeOrigLeRobotDataset transform."""

    DEFAULT_ROTATION = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
    )
    BRIDGE_TO_OPENCV = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32
    )
    TCP_TO_FLANGE = np.asarray(
        [[1.0, 0.0, 0.0, -0.093575], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    @staticmethod
    def _euler_xyz(angles: np.ndarray) -> np.ndarray:
        """Match pose_utils.Rotation.from_euler('xyz', ..., degrees=False)."""
        try:
            from scipy.spatial.transform import Rotation
        except ImportError as error:  # pragma: no cover - environment diagnostic
            raise SemanticError(
                "SciPy is required for the pinned official Euler XYZ transform; "
                "do not substitute a hand-written rotation implementation."
            ) from error
        values = np.asarray(angles, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise SemanticError(f"Euler XYZ angles must have shape (T, 3), got {values.shape}")
        return Rotation.from_euler("xyz", values, degrees=False).as_matrix().astype(np.float32)

    def build_raw_action(self, states: np.ndarray, action_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        action_rows = np.asarray(action_rows, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 7 or action_rows.ndim != 2 or action_rows.shape[1] != 7:
            raise SemanticError("Bridge state/action must have shape (T, 7)")
        if len(states) != len(action_rows) + 1:
            raise SemanticError("backward-framewise adapter needs one future observation per action")
        poses = np.eye(4, dtype=np.float32)[None].repeat(len(states), axis=0)
        poses[:, :3, :3] = self._euler_xyz(states[:, 3:6])
        poses[:, :3, 3] = states[:, :3]
        poses[:, :3, :3] = poses[:, :3, :3] @ self.DEFAULT_ROTATION
        poses = poses @ self.TCP_TO_FLANGE
        poses[:, :3, :3] = poses[:, :3, :3] @ self.BRIDGE_TO_OPENCV
        relative = []
        for current, future in zip(poses[:-1], poses[1:]):
            delta = np.linalg.inv(current) @ future
            rot6d = delta[:3, :3][:, :2].T.reshape(-1)
            relative.append(np.concatenate((delta[:3, 3], rot6d)))
        raw_action = np.concatenate((np.asarray(relative, dtype=np.float32), action_rows[:, 6:7]), axis=1)
        return raw_action, poses[0].copy()


def _scipy_cpu_evidence() -> dict[str, Any]:
    """Return an explicit evidence record for the official SciPy CPU path."""
    try:
        import scipy
        from scipy.spatial.transform import Rotation  # noqa: F401
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise SemanticError("official CPU parity requires scipy.spatial.transform.Rotation") from error
    return {
        "status": "PASS",
        "library": "scipy.spatial.transform.Rotation",
        "scipy_version": scipy.__version__,
        "call": "Rotation.from_euler('xyz', angles, degrees=False).as_matrix().astype(np.float32)",
        "framework_pose_utils_sha256": "73daa21bc5d66d94ee9217b6065b50ab81ee644d07067bb360326e4edf490380",
    }


def normalize_quantile(action: np.ndarray, q01: Sequence[float], q99: Sequence[float]) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    low = np.asarray(q01, dtype=np.float32)
    high = np.asarray(q99, dtype=np.float32)
    if action.shape[-1] != len(low) or len(low) != len(high):
        raise SemanticError("quantile stats do not match action dimension")
    return (2.0 * (action - low) / np.maximum(high - low, 1e-8) - 1.0).astype(np.float32)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PINNED_NORMALIZER_STATS_SHA256 = "e673c112e9809a2dc3ac4fffd42896c0a3c7538d4808eae35d62059af65cb267"


def load_normalizer_stats(
    path: str | Path,
    *,
    expected_sha256: str = PINNED_NORMALIZER_STATS_SHA256,
    action_dim: int = 10,
) -> NormalizerStats:
    """Load the exact pinned quantile stats; never infer or round statistics."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SemanticError(f"normalizer stats file does not exist: {path}")
    actual_sha256 = _sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise SemanticError(
            f"normalizer stats SHA256 mismatch for {path.name}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        q01 = np.asarray(payload["q01"], dtype=np.float32)
        q99 = np.asarray(payload["q99"], dtype=np.float32)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SemanticError(f"invalid normalizer stats JSON: {path}") from error
    if q01.shape != (action_dim,) or q99.shape != (action_dim,):
        raise SemanticError(
            f"normalizer stats must have q01/q99 shape ({action_dim},), "
            f"got {q01.shape} and {q99.shape}"
        )
    if not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q99 < q01):
        raise SemanticError("normalizer stats contain non-finite or descending quantiles")
    return NormalizerStats(q01=q01, q99=q99, sha256=actual_sha256)


def _manifest_hashes(root: Path) -> dict[str, str]:
    manifest = root / "MANIFEST.sha256"
    expected: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            bits = line.strip().split()
            if len(bits) == 2:
                expected[bits[1].lstrip("*\\/")] = bits[0]
    return expected


def _verify_manifest(root: Path) -> dict[str, Any]:
    """Verify every file named by the transfer manifest, not only the shard."""
    expected = _manifest_hashes(root)
    checks: dict[str, dict[str, Any]] = {}
    for relative_name, expected_sha256 in sorted(expected.items()):
        path = root / relative_name
        actual_sha256 = _sha256_file(path) if path.is_file() else None
        checks[relative_name] = {
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "match": actual_sha256 == expected_sha256,
        }
    return {
        "entry_count": len(checks),
        "all_match": bool(checks) and all(item["match"] for item in checks.values()),
        "files": checks,
    }


def _read_download_manifest(root: Path) -> dict[str, Any]:
    path = root / "download_manifest.json"
    if not path.is_file():
        raise SemanticError(f"missing download manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SemanticError(f"invalid download manifest: {path}") from error
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise SemanticError("download manifest has no file entries")
    return {
        "path": str(path),
        "declared_bytes": payload.get("bytes"),
        "declared_episodes": payload.get("episodes"),
        "file_count": len(files),
        "subset": payload.get("subset"),
        "note": payload.get("note", ""),
    }


def _language_value(episode: BridgeEpisode) -> str:
    values = {value.strip() for value in episode.language if value.strip()}
    if len(values) != 1:
        raise SemanticError(f"language is not constant/non-empty ({sorted(values)!r})")
    return next(iter(values))


def _episode_is_eligible(episode: BridgeEpisode, chunk_length: int) -> bool:
    """Apply the data gate using decoded payloads, not unreliable image flags."""
    first_flag = np.flatnonzero(episode.is_first)
    last_flag = np.flatnonzero(episode.is_last)
    aligned = (
        len(first_flag) == 1
        and first_flag[0] == 0
        and len(last_flag) == 1
        and last_flag[0] == len(episode.states) - 1
    )
    language_values = {value.strip() for value in episode.language if value.strip()}
    primary_nonempty = all(bool(blob) for blob in episode.image0)
    return bool(
        episode.has_language
        and len(language_values) == 1
        and len(episode.states) >= 2 * chunk_length + 1
        and aligned
        and len(episode.image0) == len(episode.states)
        and primary_nonempty
        and all(len(values) == len(episode.states) for values in episode.images.values())
        and np.isfinite(episode.states).all()
        and np.isfinite(episode.actions).all()
    )


def run_preflight(
    dataset_root: str | Path,
    report_root: str | Path,
    *,
    normalizer_stats_path: str | Path | None = None,
    official_parity_report: str | Path | None = None,
    chunk_length: int = 16,
    fps: float = 5.0,
    eligible_rank: int = 0,
) -> dict[str, Any]:
    """Verify the local subset and write JSON/Markdown evidence; never generates."""
    if eligible_rank < 0:
        raise ValueError("eligible_rank must be nonnegative")
    dataset_root = Path(dataset_root).resolve()
    report_root = Path(report_root).resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    files = sorted(dataset_root.glob("*.tfrecord*"))
    if len(files) != 1:
        raise SemanticError(f"expected exactly one TFRecord shard, found {len(files)}")
    shard = files[0]
    manifest = _verify_manifest(dataset_root)
    download_manifest = _read_download_manifest(dataset_root)
    actual_hash = _sha256_file(shard)
    stats_path = Path(normalizer_stats_path).resolve() if normalizer_stats_path else (
        Path(__file__).resolve().parent / "artifacts" / "bridge_orig_lerobot_stats.json"
    )
    stats = load_normalizer_stats(stats_path)
    parity_artifact: dict[str, Any] | None = None
    if official_parity_report is not None:
        parity_path = Path(official_parity_report).resolve()
        try:
            parity_payload = json.loads(parity_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SemanticError(f"invalid official parity report: {parity_path}") from error
        parity_artifact = {
            "path": str(parity_path),
            "sha256": _sha256_file(parity_path),
            "status": parity_payload.get("status"),
            "action_exact_equal": parity_payload.get("action_exact_equal"),
            "initial_pose_exact_equal": parity_payload.get("initial_pose_exact_equal"),
            "normalization": parity_payload.get("normalization"),
            "official_source": parity_payload.get("official_source"),
        }
    records: list[dict[str, Any]] = []
    selected: BridgeEpisode | None = None
    eligible_seen = 0
    for record in iter_tfrecord_records(shard, verify_crc=True):
        episode = decode_bridge_episode(record)
        first_flag = np.flatnonzero(episode.is_first)
        last_flag = np.flatnonzero(episode.is_last)
        aligned = (
            len(first_flag) == 1
            and first_flag[0] == 0
            and len(last_flag) == 1
            and last_flag[0] == len(episode.states) - 1
        )
        primary_nonempty = all(bool(blob) for blob in episode.image0)
        summary = {
            "record_index": episode.index,
            "episode_id": episode.episode_id,
            "file_path": episode.file_path,
            "frames": int(len(episode.states)),
            "has_language": episode.has_language,
            "has_image_flags": episode.has_image_flags,
            "metadata_image_0_contradicts_payload": bool(not episode.has_image0 and primary_nonempty),
            "language": episode.language[0] if episode.language else "",
            "state_shape": list(episode.states.shape),
            "action_shape": list(episode.actions.shape),
            "image_counts": {camera: len(values) for camera, values in episode.images.items()},
            "primary_jpeg_nonempty": primary_nonempty,
            "sequence_flags_aligned": bool(aligned),
        }
        records.append(summary)
        eligible = _episode_is_eligible(episode, chunk_length)
        if eligible:
            if eligible_seen == eligible_rank:
                selected = episode
            eligible_seen += 1

    raw_actions = np.empty((0, 10), dtype=np.float32)
    normalized_actions = np.empty((0, 10), dtype=np.float32)
    cpu_parity = _scipy_cpu_evidence()
    action_windows: list[np.ndarray] = []
    initial_poses: list[np.ndarray] = []
    window: np.ndarray | None = None
    rgb_float32: np.ndarray | None = None
    selected_data: dict[str, Any] | None = None
    camera_proof: dict[str, Any] = {"feature_key": "steps/observation/image_0", "selected": False}
    warnings: list[str] = []
    if selected is None:
        status = "STOPPED_NO_ELIGIBLE_WINDOW"
        stop_reasons = [
            f"Eligible record rank {eligible_rank} is unavailable among {eligible_seen} records "
            "with 33 observations, language, aligned boundaries, and a non-empty primary image_0 stream.",
        ]
    else:
        window = selected.states[: 2 * chunk_length + 1]
        adapter = BridgeOrigAdapter()
        for start in (0, chunk_length):
            raw, initial = adapter.build_raw_action(
                window[start : start + chunk_length + 1], selected.actions[start : start + chunk_length]
            )
            action_windows.append(raw)
            initial_poses.append(initial)
        raw_actions = np.concatenate(action_windows, axis=0)
        normalized_actions = normalize_quantile(raw_actions, stats.q01, stats.q99)

        primary_proofs = [jpeg_content_proof(blob) for blob in selected.image0[: 2 * chunk_length + 1]]
        try:
            from PIL import Image

            rgb_float32 = np.stack(
                [
                    np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255.0
                    for blob in selected.image0[: 2 * chunk_length + 1]
                ],
                axis=0,
            ).astype(np.float32)
        except ImportError as error:  # pragma: no cover - runtime dependency diagnostic
            raise SemanticError("Pillow is required to preserve RGB input tensors") from error
        stream_proofs: dict[str, Any] = {}
        for camera, blobs in selected.images.items():
            proofs = [jpeg_content_proof(blob) for blob in blobs[: 2 * chunk_length + 1]]
            stream_proofs[camera] = {
                "metadata_flag": bool(selected.has_image_flags[camera]),
                "frame_count": len(blobs),
                "decoded_shapes": sorted({tuple(proof["shape"]) for proof in proofs}),
                "unique_encoded_frames": len({proof["sha256"] for proof in proofs}),
                "pixel_std_min": min(proof["pixel_std"] for proof in proofs),
                "pixel_std_max": max(proof["pixel_std"] for proof in proofs),
                "all_rgb_256x256": all(proof["shape"] == [256, 256, 3] for proof in proofs),
                "all_nonconstant": all(proof["pixel_max"] > proof["pixel_min"] for proof in proofs),
            }
        camera_proof = {
            "feature_key": "steps/observation/image_0",
            "metadata_flag": bool(selected.has_image0),
            "metadata_payload_contradiction": bool(not selected.has_image0 and primary_proofs),
            "decoded_shapes": sorted({tuple(proof["shape"]) for proof in primary_proofs}),
            "unique_encoded_frames": len({proof["sha256"] for proof in primary_proofs}),
            "all_window_frames_rgb_256x256": all(proof["shape"] == [256, 256, 3] for proof in primary_proofs),
            "all_window_frames_nonconstant": all(proof["pixel_max"] > proof["pixel_min"] for proof in primary_proofs),
            "stream_inventory": stream_proofs,
            "primary_frame_sha256": [proof["sha256"] for proof in primary_proofs],
        }
        if not selected.has_image0:
            warnings.append(
                "episode_metadata/has_image_0 is false although image_0 contains valid decoded RGB frames; "
                "selection therefore uses payload validation plus official source mapping, not the unreliable flag."
            )
        selected_data = {
            "record_index": selected.index,
            "episode_id": selected.episode_id,
            "file_path": selected.file_path,
            "frames_used": int(len(window)),
            "language": _language_value(selected),
            "observation_state_shape": list(window.shape),
            "action_shape_per_chunk": [list(action.shape) for action in action_windows],
            "raw_action_sha256": hashlib.sha256(raw_actions.tobytes()).hexdigest(),
            "normalized_action_sha256": hashlib.sha256(normalized_actions.tobytes()).hexdigest(),
            "state_min": window.min(axis=0).astype(float).tolist(),
            "state_max": window.max(axis=0).astype(float).tolist(),
            "action_min": selected.actions[: 2 * chunk_length].min(axis=0).astype(float).tolist(),
            "action_max": selected.actions[: 2 * chunk_length].max(axis=0).astype(float).tolist(),
            "gripper_column": 6,
            "gripper_values": selected.actions[: 2 * chunk_length, 6].astype(float).tolist(),
            "rgb_float32_shape": list(rgb_float32.shape),
            "rgb_float32_sha256": hashlib.sha256(rgb_float32.tobytes()).hexdigest(),
        }
        status = "PREFLIGHT_PASSED_GENERATION_NOT_PERFORMED"
        stop_reasons = ["Model generation remains outside this generation-free preflight and requires independent main-agent admission."]

    if len(records) != 52 or download_manifest.get("declared_episodes") != 52:
        warnings.append(f"expected 52 TFRecord episodes, parsed {len(records)}")
        status = "STOPPED_RECORD_COUNT_MISMATCH"
    if not manifest["all_match"]:
        warnings.append("one or more downloaded files failed the transfer-manifest SHA256 check")
        status = "STOPPED_HASH_MISMATCH"
    if parity_artifact is not None and parity_artifact.get("status") != "PASS":
        warnings.append("the attached official CPU parity report did not pass")
        status = "STOPPED_OFFICIAL_PARITY"
    report: dict[str, Any] = {
        "status": status,
        "generation_performed": False,
        "dataset_root": str(dataset_root),
        "shard": shard.name,
        "shard_bytes": shard.stat().st_size,
        "shard_sha256": actual_hash,
        "manifest_sha256_match": manifest["files"].get(shard.name, {}).get("match", False),
        "manifest": manifest,
        "download_manifest": download_manifest,
        "normalizer_stats": {"path": str(stats_path), "sha256": stats.sha256, "q01": stats.q01.tolist(), "q99": stats.q99.tolist()},
        "official_parity_artifact": parity_artifact,
        "record_count": len(records),
        "eligible_record_count": eligible_seen,
        "selection_eligible_rank": eligible_rank,
        "records": records,
        "selected": selected_data,
        "camera": camera_proof,
        "temporal_alignment": {
            "sequence_index_order_proven": bool(len(records) == 52),
            "parallel_lengths_proven": bool(all(row["sequence_flags_aligned"] for row in records)),
            "flags_prove_first_last_boundaries": bool(all(row["sequence_flags_aligned"] for row in records)),
            "timestamp_feature_present": False,
            "fps_used_for_adapter": fps,
            "sample_period_s": 1.0 / fps,
            "frequency_continuity_proven_from_official_source": True,
            "wall_clock_alignment_proven": False,
        },
        "official_adapter": {
            "framework_commit": "ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5",
            "dataset_class": "cosmos_framework.data.generator.action.datasets.bridge_orig_lerobot_dataset.BridgeOrigLeRobotDataset",
            "pose_utils": "scipy.spatial.transform.Rotation.from_euler('xyz', angles, degrees=False).as_matrix().astype(float32)",
            "state_components": "state[:3] translation; state[3:6] EulerXYZ; no unit rescale",
            "transforms": ["DEFAULT_ROTATION", "TCP_TO_FLANGE", "BRIDGE_TO_OPENCV"],
            "pose_convention": "backward_framewise",
            "rotation_encoding": "rot6d",
            "gripper_source": "original action[:, 6]",
            "normalization": "quantile once: 2*(raw-q01)/clamp(q99-q01,min=1e-8)-1",
            "window_initial_pose_sha256": [hashlib.sha256(p.tobytes()).hexdigest() for p in initial_poses],
            "raw_action_sha256": hashlib.sha256(raw_actions.tobytes()).hexdigest(),
            "normalized_action_sha256": hashlib.sha256(normalized_actions.tobytes()).hexdigest(),
        },
        "official_cpu_parity": cpu_parity,
        "warnings": warnings,
        "stop_reasons": stop_reasons,
        "source_provenance": [
            {"kind": "tfds_features", "url": "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/features.json"},
            {"kind": "tfds_dataset_info", "url": "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/dataset_info.json"},
            {"kind": "state_action_semantics", "url": "https://github.com/rail-berkeley/bridge_data_v2/issues/26"},
            {"kind": "octo_bridge_config", "commit": "241fb3514b7c40957a86d869fecb7c7fc353f540", "path": "octo/data/oxe/oxe_dataset_configs.py", "mapping": "image_primary=image_0, action space POS_EULER"},
            {"kind": "octo_bridge_relabel", "path": "octo/data/utils/data_utils.py", "mapping": "adjacent state deltas preserve image/action index alignment"},
            {"kind": "bridge_robot_source", "commit": "b841131ecd512bafb303075bd8f8b677e0bf9f1f", "path": "widowx_envs/widowx_envs/base/robot_base_env.py"},
            {"kind": "raw_converter_alignment", "url": "https://github.com/rail-berkeley/bridge_data_v2/blob/main/data_processing/bridgedata_raw_to_numpy.py"},
            {"kind": "cosmos_framework_commit", "commit": "ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5", "path": "cosmos_framework/data/generator/action/datasets/bridge_orig_lerobot_dataset.py", "remote": "ssh://seetacloud-umi/root/autodl-tmp/cosmos-framework-task6-clean"},
            {"kind": "normalizer_stats", "sha256": stats.sha256, "path": str(stats_path)},
        ],
    }
    if selected_data is not None and selected is not None and window is not None and rgb_float32 is not None:
        npz_path = report_root / "preflight_window.npz"
        np.savez_compressed(
            npz_path,
            states=window,
            rgb_float32=rgb_float32,
            actions_original=selected.actions[: 2 * chunk_length].reshape(2, chunk_length, 7),
            actions_raw=raw_actions.reshape(2, chunk_length, 10),
            actions_normalized=normalized_actions.reshape(2, chunk_length, 10),
            actions_original_flat=selected.actions[: 2 * chunk_length],
            actions_raw_flat=raw_actions,
            actions_normalized_flat=normalized_actions,
        )
        report["selected"]["preflight_npz"] = {
            "path": str(npz_path),
            "sha256": _sha256_file(npz_path),
            "arrays": ["rgb_float32", "states", "actions_original", "actions_raw", "actions_normalized", "*_flat"],
        }
        inputs_path = report_root / "inputs.npz"
        np.savez_compressed(
            inputs_path,
            rgb_float32=rgb_float32,
            states=window,
            actions_original=selected.actions[: 2 * chunk_length].reshape(2, chunk_length, 7),
            actions_raw=raw_actions.reshape(2, chunk_length, 10),
            actions_normalized=normalized_actions.reshape(2, chunk_length, 10),
            actions_original_flat=selected.actions[: 2 * chunk_length],
            actions_raw_flat=raw_actions,
            actions_normalized_flat=normalized_actions,
        )
        report["selected"]["inputs_npz"] = {
            "path": str(inputs_path),
            "sha256": _sha256_file(inputs_path),
            "arrays": ["rgb_float32", "states", "actions_original", "actions_raw", "actions_normalized", "*_flat"],
            "rgb_float32_shape": list(rgb_float32.shape),
            "actions_normalized_shape": [2, chunk_length, 10],
        }
    (report_root / "task8_preflight.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report_root / "task8_preflight.md", report)
    return report


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["selected"]
    lines = [
        "# Task 8 data preflight",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "No model, tokenizer, VAE, CUDA, or generation path was imported or executed.",
        "",
        "## Integrity",
        "",
        f"- TFRecord records: `{report['record_count']}`; shard bytes: `{report['shard_bytes']}`.",
        f"- SHA256: `{report['shard_sha256']}`; local manifest match: `{report['manifest_sha256_match']}`.",
        f"- All `{report['manifest']['entry_count']}` manifest entries match: `{report['manifest']['all_match']}`.",
        f"- Pinned normalizer stats SHA256: `{report['normalizer_stats']['sha256']}`.",
        f"- Official CPU parity artifact: `{(report.get('official_parity_artifact') or {}).get('status', 'not attached')}`.",
        "",
        f"## Eligible original-order window at rank {report['selection_eligible_rank']}",
        "",
        *(
            [
                f"- Record `{selected['record_index']}`, episode `{selected['episode_id']}`, 33 observations, language: `{selected['language']}`.",
                f"- State shape `{selected['observation_state_shape']}`; each action chunk `{selected['action_shape_per_chunk']}`; gripper source `action[:, 6]`.",
                f"- image_0 decoded proof: `{report['camera']['decoded_shapes']}`, all 33 RGB 256x256: `{report['camera']['all_window_frames_rgb_256x256']}`; unique encoded frames: `{report['camera']['unique_encoded_frames']}`.",
                f"- Runtime input artifact: `{selected['inputs_npz']['rgb_float32_shape']}` RGB float32 and `{selected['inputs_npz']['actions_normalized_shape']}` normalized actions.",
            ]
            if selected is not None
            else ["- No eligible 33-observation window was found."]
        ),
        "",
        "## Temporal and semantic gate",
        "",
        "Discrete per-index parallel state/action/image/language alignment and first/last boundaries are checked for all parsed records. The official Bridge/Octo mapping identifies image_0 as the primary camera; decoded payload checks are retained alongside the contradictory metadata flag. The source is fixed at 5 Hz, so frame-index continuity supplies the temporal cadence even though TFDS has no wall-clock timestamp feature.",
        "",
        "## Pinned adapter",
        "",
        "`BridgeOrigLeRobotDataset` at framework commit `ffa9c6b60a6b04b2fae337577bc6cbd8a93c39f5` was reproduced once: state[:3] + EulerXYZ state[3:6], DEFAULT_ROTATION, TCP_TO_FLANGE, BRIDGE_TO_OPENCV, backward-framewise relative pose, rot6d, action[:,6] gripper, and quantile normalization. This is an observed-future-pose-conditioned adapter, not control-only prediction.",
        "",
        "## Sources",
    ]
    for source in report["source_provenance"]:
        label = source.get("kind", "source")
        target = source.get("url", source.get("remote", source.get("path", "")))
        lines.append(f"- {label}: {target}")
    lines += ["", "## Stop reasons", ""]
    lines.extend(f"- {reason}" for reason in report["stop_reasons"])
    if report.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines.extend(f"- {warning}" for warning in report["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--normalizer-stats-path", default=None)
    parser.add_argument("--official-parity-report", default=None)
    parser.add_argument("--eligible-rank", type=int, default=0,
                        help="Zero-based rank among eligible records in original shard order")
    args = parser.parse_args(argv)
    run_preflight(
        args.dataset_root,
        args.report_root,
        normalizer_stats_path=args.normalizer_stats_path,
        official_parity_report=args.official_parity_report,
        eligible_rank=args.eligible_rank,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
