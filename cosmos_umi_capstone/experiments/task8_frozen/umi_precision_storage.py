"""Process-owned locks and single-publication precision sample storage."""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path

import numpy as np

try:
    from .umi_precision_primitives import canonical_json
    from .umi_precision_reanalysis import _atomic_rename_noreplace
    from .umi_fd_post_vae_scan import sha256_file
except ImportError:
    from umi_precision_primitives import canonical_json
    from umi_precision_reanalysis import _atomic_rename_noreplace
    from umi_fd_post_vae_scan import sha256_file


class ProcessLock:
    """The OS releases ownership on exit or hard kill; the lock inode persists."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open("a+b")
        try:
            self.stream.seek(0, os.SEEK_END)
            if self.stream.tell() == 0:
                self.stream.write(b"0")
                self.stream.flush()
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise BlockingIOError("precision runner already owns this process lock") from error
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def _fsync_directory(path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def publish_directory(stage, destination):
    """One atomic no-replace rename after all staged files have been fsynced."""
    _atomic_rename_noreplace(stage, destination)
    _fsync_directory(destination.parent)


def _write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class PrecisionSampleStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sample_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError("invalid sample identifier")
        return self.root / sample_id

    def prepare(self, sample_id, *, resume=False, required_files=()):
        path = self._path(sample_id)
        if path.exists():
            state = json.loads((path / "status.json").read_text()) if (path / "status.json").exists() else {}
            if state.get("status") == "success":
                hashes = state.get("artifact_sha256", {})
                if "sample.json" not in hashes or any(not (path / name).is_file() or sha256_file(path / name) != digest
                                                      for name, digest in hashes.items()):
                    raise ValueError("successful sample hash mismatch; refusing overwrite")
                if any(not (path / name).is_file() for name in required_files):
                    raise ValueError("successful sample missing required artifact")
                if not resume:
                    raise FileExistsError(path)
                return "skip"
            if not resume:
                raise FileExistsError(path)
            path.rename(self.root / f"{sample_id}.attempt.{uuid.uuid4().hex}")
        # The final sample name remains absent while compute runs. _write
        # stages publication separately, so a kill cannot expose a partial success.
        return "run"

    def _write(self, sample_id, payload, artifacts, status):
        destination = self._path(sample_id)
        if destination.exists():
            raise FileExistsError(f"refusing overwrite: {destination}")
        stage = Path(tempfile.mkdtemp(prefix=f".{sample_id}.stage.", dir=self.root))
        arrays = dict(artifacts or {})
        def encode(value, name):
            if isinstance(value, np.ndarray):
                filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", name) + ".npy"
                if filename in arrays:
                    raise ValueError("duplicate array artifact")
                arrays[filename] = value
                return {"artifact": filename, "dtype": str(value.dtype), "shape": list(value.shape)}
            if isinstance(value, dict):
                return {str(key): encode(item, name + "_" + str(key)) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(item, name + "_" + str(i)) for i, item in enumerate(value)]
            return value
        safe = encode(payload, "payload")
        _write_json(stage / "sample.json", safe)
        for name, value in arrays.items():
            if Path(name).name != name or name in ("sample.json", "status.json"):
                raise ValueError("artifact must be a safe filename")
            with (stage / name).open("xb") as stream:
                if isinstance(value, np.ndarray):
                    np.save(stream, value, allow_pickle=False)
                elif isinstance(value, bytes):
                    stream.write(value)
                else:
                    raise TypeError("sample artifacts must be arrays or bytes")
                stream.flush()
                os.fsync(stream.fileno())
        hashes = {name: sha256_file(stage / name) for name in ["sample.json", *sorted(arrays)]}
        _write_json(stage / "status.json", {"status": status, "required_artifacts": sorted(hashes),
                                            "artifact_sha256": hashes})
        _fsync_directory(stage)
        publish_directory(stage, destination)
        return destination

    def write_success(self, sample_id, payload, *, artifacts=None):
        return self._write(sample_id, payload, artifacts, "success")

    def write_failure(self, sample_id, payload, *, artifacts=None):
        return self._write(sample_id, payload, artifacts, "fail")

    def load_record(self, sample_id):
        path = self._path(sample_id)
        record = json.loads((path / "sample.json").read_text())
        for artifact in path.glob("*.npy"):
            record[artifact.stem] = np.load(artifact, allow_pickle=False)
        return record
