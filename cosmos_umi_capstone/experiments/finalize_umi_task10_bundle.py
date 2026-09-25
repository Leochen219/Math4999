"""Refresh the Task 10 manifest and package a small audit bundle."""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finalize(root: Path, source_dir: Path) -> tuple[Path, str]:
    analysis = root / "analysis"
    required = (root / "run_status.json", analysis / "comparison.json",
                analysis / "independent_review.json", analysis / "independent_findings.md")
    if any(not path.is_file() for path in required):
        raise ValueError("finalization requires complete generation and both reviews")
    import json
    if json.loads(required[0].read_text(encoding="utf-8")).get("status") != "GENERATION_COMPLETE":
        raise ValueError("cannot finalize an incomplete generation matrix")
    if json.loads(required[2].read_text(encoding="utf-8")).get("status") != "INDEPENDENT_AUDIT_PASS":
        raise ValueError("independent review did not pass")
    bundle = analysis / "review_bundle.zip"
    manifest = root / "MANIFEST.sha256"
    bundle_hash = analysis / "review_bundle.sha256"
    with manifest.open("w", encoding="utf-8") as stream:
        for path in sorted(path for path in root.rglob("*") if path.is_file()
                           and path not in (manifest, bundle, bundle_hash)):
            stream.write(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n")
    files = [root / "run_identity.json", root / "run_status.json", root / "smoke_gate.json", manifest]
    files.extend(path for path in analysis.rglob("*") if path.is_file() and path not in (bundle, bundle_hash))
    files.extend(root.glob("preflight/record_*/task8_preflight.json"))
    files.extend(root.glob("records/record_*/monitor_attempts/attempt_*/monitor/task8/*samples.csv"))
    files.extend(root.glob("records/record_*/monitor_attempts/attempt_*/monitor/task8/sample_resource_snapshots.jsonl"))
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root).as_posix())
        for path in sorted(source_dir.glob("*task10*.py")):
            archive.write(path, f"code/{path.name}")
        archive.write(source_dir / "umi_task9_bridge_preflight.py", "code/umi_task9_bridge_preflight.py")
        for path in sorted((source_dir / "task8_frozen").glob("*.py")):
            archive.write(path, f"code/task8_frozen/{path.name}")
    digest = sha256(bundle)
    bundle_hash.write_text(f"{digest}  review_bundle.zip\n", encoding="utf-8")
    return bundle, digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    result, digest = finalize(Path(args.run_root), Path(__file__).resolve().parent)
    print(f"{result} {digest}")
