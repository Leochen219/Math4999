"""Write a SHA256 manifest for a completed Task 9 scan after local figure rendering."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


def write_manifest(scan_dir: Path) -> Path:
    root = scan_dir.resolve()
    if not (root / "run_status.json").is_file():
        raise ValueError("Task 9 run status is missing")
    import json
    status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE":
        raise ValueError("Task 9 scan is not complete")
    manifest = root / "MANIFEST.sha256"
    lines = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item == manifest:
            continue
        resolved = item.resolve()
        resolved.relative_to(root)
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {item.relative_to(root).as_posix()}")
    temporary = manifest.with_name(".MANIFEST.sha256.tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", required=True, type=Path)
    args = parser.parse_args()
    print(write_manifest(args.scan_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
