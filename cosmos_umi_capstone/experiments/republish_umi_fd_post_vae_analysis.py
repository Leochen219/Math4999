"""Presentation-only republish of a frozen post-VAE analysis.

This intentionally consumes the already-published scalar CSVs and summary.  It
does not load tensors, call the provenance validator, recompute metrics, or
modify the frozen input directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from analyze_umi_fd_post_vae_scan import _plots, _safe_json, sha256_file


_PRESENTATION_INPUTS = (
    "scan_summary.json",
    "response_metrics.csv",
    "finite_difference_consistency.csv",
    "injection_diagnostics.csv",
    "fit_metrics.csv",
    "fit_metrics.json",
    "window_decisions.csv",
)
_INT_FIELDS = {"direction", "sign", "adjacent_index", "start_index", "end_index", "window_id"}
_BOOL_FIELDS = {"selected", "overflow", "outside_mask_exact", "mask_outside_exact"}
_SHA256_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\s].*)$")
_SOURCE_FILES = {
    "analyze_umi_fd_post_vae_scan.py": "analyze_umi_fd_post_vae_scan.py",
    "republish_umi_fd_post_vae_analysis.py": "republish_umi_fd_post_vae_analysis.py",
}


def _coerce(field: str, value: str) -> Any:
    value = value.strip()
    if value == "" or value.lower() in {"null", "none", "n/a", "na"}:
        return None
    if field in _BOOL_FIELDS:
        return value.lower() in {"1", "true", "yes", "passed"}
    if field in _INT_FIELDS:
        try:
            return int(value)
        except ValueError:
            return value
    try:
        return float(value)
    except ValueError:
        return value


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{field: _coerce(field, value) for field, value in row.items()} for row in csv.DictReader(handle)]


def _parse_manifest(path: Path) -> dict[str, str]:
    """Parse a strict sha256sum manifest and reject ambiguous entries."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"original run manifest is unreadable: {path}") from exc
    if not lines or any(not line for line in lines):
        raise ValueError("original run manifest is empty or contains a blank line")
    entries: dict[str, str] = {}
    for line in lines:
        match = _SHA256_MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed original run manifest entry: {line!r}")
        digest, relative = match.groups()
        pure = PurePosixPath(relative)
        if (
            "\\" in relative
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in pure.parts)
            or pure.as_posix() != relative
            or relative.endswith(" ")
        ):
            raise ValueError(f"unsafe original run manifest path: {relative!r}")
        if relative in entries:
            raise ValueError(f"duplicate original run manifest entry: {relative}")
        entries[relative] = digest
    return entries


def _verify_frozen_inputs(frozen: Path, manifest: Path) -> tuple[dict[str, str], str]:
    """Verify required files against the original manifest before any reads/copies."""
    entries = _parse_manifest(manifest)
    input_hashes: dict[str, str] = {}
    for name in _PRESENTATION_INPUTS:
        expected = entries.get(name)
        if expected is None:
            raise ValueError(f"required presentation input is missing from original manifest: {name}")
        candidate = frozen / name
        if not candidate.is_file() or candidate.resolve().parent != frozen:
            raise FileNotFoundError(f"required frozen presentation input missing: {candidate}")
        actual = sha256_file(candidate)
        if actual != expected:
            raise ValueError(f"frozen presentation input hash mismatch for {name}: expected {expected}, got {actual}")
        input_hashes[name] = actual
    return input_hashes, sha256_file(manifest)


def _assert_safe_paths(frozen: Path, output: Path) -> None:
    if frozen == output or frozen in output.parents or output in frozen.parents:
        raise ValueError("frozen input and output directories must not be equal or nested")


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    if not entries:
        raise ValueError("cannot publish an empty revision")
    (root / "MANIFEST.sha256").write_text("\n".join(entries) + "\n", encoding="ascii")


def _verify_source_commit(source_commit: str, source_root: Path, source_hashes: dict[str, str]) -> dict[str, Any]:
    """Best-effort local binding of the declared commit to embedded source."""
    try:
        root_result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
            check=False, capture_output=True, timeout=5, text=True,
        )
        if root_result.returncode != 0:
            return {"status": "declared", "reason": "local git repository unavailable"}
        repo_root = Path(root_result.stdout.strip()).resolve()
        files: dict[str, str] = {}
        for name, expected in source_hashes.items():
            relative = source_root.resolve().relative_to(repo_root).as_posix() + "/" + name
            shown = subprocess.run(
                ["git", "-C", str(repo_root), "show", f"{source_commit}:{relative}"],
                check=False, capture_output=True, timeout=5,
            )
            if shown.returncode != 0:
                return {"status": "declared", "reason": f"{source_commit} does not contain {relative}"}
            actual = hashlib.sha256(shown.stdout).hexdigest()
            files[name] = actual
            if actual != expected:
                return {"status": "declared", "reason": f"working source differs from {source_commit}: {name}", "files": files}
        return {"status": "verified", "files": files}
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError):
        return {"status": "declared", "reason": "local git verification unavailable"}


def republish(input_dir: str | Path, output_dir: str | Path, *, source_commit: str) -> dict[str, Any]:
    frozen = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    if not frozen.is_dir():
        raise FileNotFoundError(f"frozen input directory does not exist: {frozen}")
    _assert_safe_paths(frozen, output)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be a full 40-character lowercase hexadecimal commit id")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing revision directory: {output}")
    root_manifest = frozen / "MANIFEST.sha256"
    if not root_manifest.is_file():
        raise FileNotFoundError(f"original run manifest missing: {root_manifest}")
    input_hashes, manifest_hash = _verify_frozen_inputs(frozen, root_manifest)
    source_root = Path(__file__).resolve().parent
    source_paths = {name: source_root / name for name in _SOURCE_FILES.values()}
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    source_commit_verification = _verify_source_commit(source_commit, source_root, source_hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".analysis-revision-stage-", dir=str(output.parent)))
    try:
        for name in _PRESENTATION_INPUTS:
            shutil.copyfile(frozen / name, stage / name)
        shutil.copyfile(root_manifest, stage / "original_run_manifest_snapshot.sha256")
        if sha256_file(stage / "original_run_manifest_snapshot.sha256") != manifest_hash:
            raise ValueError("original run manifest changed during publication")
        for name, expected in input_hashes.items():
            if sha256_file(stage / name) != expected:
                raise ValueError(f"frozen presentation input changed during publication: {name}")
        source_dir = stage / "source"
        source_dir.mkdir()
        for name, path in source_paths.items():
            shutil.copyfile(path, source_dir / name)
            if sha256_file(source_dir / name) != source_hashes[name]:
                raise ValueError(f"source file changed during publication: {name}")
        rows = _read_rows(stage / "response_metrics.csv")
        finite_difference = _read_rows(stage / "finite_difference_consistency.csv")
        fits = _read_rows(stage / "fit_metrics.csv")
        decisions = _read_rows(stage / "window_decisions.csv")
        chart_paths = _plots(stage, rows, finite_difference, fits, decisions)
        summary = json.loads((stage / "scan_summary.json").read_text(encoding="utf-8"))
        provenance = {
            "type": "analysis_revision",
            "mode": "presentation_only",
            "metric_recomputation": False,
            "provenance_validator_called": False,
            "source_commit": source_commit,
            "source_commit_status": source_commit_verification["status"],
            "source_commit_verification": source_commit_verification,
            "source_hashes": source_hashes,
            "input_summary_path": str(frozen / "scan_summary.json"),
            "input_summary_sha256": input_hashes["scan_summary.json"],
            "input_hashes": input_hashes,
            "original_run_manifest_path": str(root_manifest),
            "original_run_manifest_sha256": manifest_hash,
            "original_run_manifest_snapshot": "original_run_manifest_snapshot.sha256",
            "original_run_manifest_snapshot_sha256": sha256_file(stage / "original_run_manifest_snapshot.sha256"),
            "status": summary.get("status"),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "charts": chart_paths,
        }
        (stage / "analysis_revision_provenance.json").write_text(json.dumps(_safe_json(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        finding = "No candidate local-linear interval qualifies under the frozen numeric gates." if not summary.get("candidate_count") else "Frozen candidate decisions are reproduced for presentation only."
        figures = "\n".join(f"- `{path}`" for path in chart_paths)
        report = (
            "# UMI post-VAE analysis presentation revision\n\n"
            "This directory is a presentation-only revision of a frozen analysis. "
            "It reads the existing summary and scalar CSV evidence and regenerates charts; "
            "it does not recompute metrics, load tensors/models, invoke the provenance validator, or modify the frozen run.\n\n"
            f"Finding carried from frozen summary: {finding}\n\n"
            f"Status: `{summary.get('status')}`; candidate count: `{summary.get('candidate_count')}`.\n\n"
            "## Figures\n\n" + (figures or "- None") + "\n\n"
            "See `analysis_revision_provenance.json` for the declared source commit, source hashes, and input hashes. "
            "The original run manifest snapshot and this revision's manifest are separate evidence boundaries.\n"
        )
        (stage / "experiment_report.md").write_text(report, encoding="utf-8")
        notes = (
            "# Analysis revision notes\n\n"
            "This is a presentation-only renderer revision. Frozen scalar summary/CSV inputs are copied byte-for-byte; "
            "metrics are not recomputed and no provenance validator is called. Chart bytes are reproducible only when the "
            "verified inputs and rendering environment are held fixed; generated timestamps make the complete revision non-deterministic. "
            "The embedded source snapshots and chart presentation may differ from the original run while the original run manifest remains "
            "authoritative for execution evidence.\n"
        )
        (stage / "implementation_notes.md").write_text(notes, encoding="utf-8")
        _write_manifest(stage)
        os.replace(stage, output)
        stage = None
        return provenance
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Presentation-only UMI post-VAE analysis revision")
    parser.add_argument("--input-dir", required=True, help="frozen run09 analyzer output directory")
    parser.add_argument("--output-dir", required=True, help="new, empty analysis revision directory")
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    provenance = republish(args.input_dir, args.output_dir, source_commit=args.source_commit)
    print(json.dumps({"status": "ok", "mode": provenance["mode"], "metric_recomputation": False, "output_dir": str(Path(args.output_dir).resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
