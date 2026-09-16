"""Offline Task 6 analysis and deterministic review packaging.

Raw generation samples are treated as an immutable evidence boundary.  This
module only loads validated arrays, delegates the fixed Task 5 gate formulae,
and writes a staged analysis package; it never invokes a model.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import zlib
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from .analyze_umi_task5 import analyze_task5_records
    from .umi_task5_primitives import ALPHAS, DIRECTION_IDS
    from .umi_task6_decoder import sha256_file
except ImportError:  # pragma: no cover
    from analyze_umi_task5 import analyze_task5_records
    from umi_task5_primitives import ALPHAS, DIRECTION_IDS
    from umi_task6_decoder import sha256_file


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Mapping): return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def rms64(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32)
    if not array.size or not np.all(np.isfinite(array)): raise ValueError("metric input must be finite and non-empty")
    return float(np.sqrt(np.mean(np.square(array.astype(np.float64)), dtype=np.float64)))


def difference_metrics(left: Any, right: Any) -> dict[str, float | None]:
    """Float32 subtraction followed by float64 reductions in one space."""
    lhs = np.asarray(left, dtype=np.float32); rhs = np.asarray(right, dtype=np.float32)
    if lhs.shape != rhs.shape: return {"status": "N/A", "reason": "shape_mismatch", "rms": None, "max_abs": None, "mean_abs": None}
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)): return {"status": "N/A", "reason": "nonfinite", "rms": None, "max_abs": None, "mean_abs": None}
    diff = lhs.astype(np.float64) - rhs.astype(np.float64); absolute = np.abs(diff)
    return {"status": "OK", "reason": None, "rms": float(np.sqrt(np.mean(diff * diff, dtype=np.float64))),
            "max_abs": float(np.max(absolute)), "mean_abs": float(np.mean(absolute, dtype=np.float64))}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _rows_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in keys and not isinstance(value, (Mapping, list, tuple, np.ndarray)):
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys or ["status"]); writer.writeheader()
        for row in rows: writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in (keys or ["status"])})


def _normalize_records(records: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Strip the group prefix and normalize Task 6 ``output_full`` naming."""
    result: dict[str, dict[str, Any]] = {}
    for key, original in records.items():
        short = str(key).split("__")[-1]
        record = dict(original)
        if "predicted_latent" not in record and "output_full" in record:
            record["predicted_latent"] = record["output_full"]
        result[short] = record
    return result


def _required_ids() -> set[str]:
    ids = {"baseline_pre", "baseline_post"}
    for direction in DIRECTION_IDS:
        for ordinal in range(len(ALPHAS)):
            for sign in ("plus", "minus"):
                ids.add(f"{direction}_alpha_{ordinal:02d}_{sign}")
    return ids


def _status_incomplete(missing: list[str], *, reason: str = "raw group is incomplete") -> dict[str, Any]:
    return {"status": "INCOMPLETE", "reason": reason, "missing_samples": missing,
            "summary": {"direction_candidates_pass": 0, "direction_candidates_total": len(DIRECTION_IDS),
                        "additivity_total": 6, "prediction_total": 24}}


def analyze_task6_records(records: Mapping[str, Mapping[str, Any]], *, plan_detail: Mapping[str, Any] | None = None,
                          group: Mapping[str, Any] | None = None, strict: bool = False) -> dict[str, Any]:
    """Recompute the unchanged Task 5 metrics from one Task 6 group.

    Missing, stopped, or nonfinite evidence is a truthful incomplete result by
    default.  ``strict=True`` is useful for callers that want an exception.
    """
    normalized = _normalize_records(records)
    missing = sorted(_required_ids() - set(normalized))
    if missing:
        if strict: raise ValueError("incomplete Task 6 group: " + ", ".join(missing))
        return _status_incomplete(missing)
    detail = dict(plan_detail or {})
    detail.setdefault("s_z", 1.0)
    detail.setdefault("combination_coefficients", {"c01": 1.0, "c12": 1.0})
    try:
        result = analyze_task5_records(normalized, plan_detail=detail)
    except (ValueError, FloatingPointError) as error:
        if strict: raise
        return _status_incomplete([], reason=f"invalid raw evidence: {type(error).__name__}: {error}")
    result = dict(result)
    result.update({"status": "COMPLETE", "group": dict(group or {}), "raw_sample_count": len(normalized)})
    result["summary"] = {**dict(result.get("summary", {})), "additivity_total": 6, "prediction_total": 24,
                          "max_additivity_error": max((row.get("relative_error") or 0.0 for row in result.get("additivity", [])), default=None),
                          "max_holdout_error": max((row.get("relative_error") or 0.0 for row in result.get("predictions", [])), default=None)}
    return result


def _manifest_entries(root: Path, *, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = set(exclude or ()) | {"MANIFEST.sha256", "review_bundle.zip"}
    return {path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file() and path.name not in excluded}


def verify_raw_manifest(root: str | Path) -> dict[str, Any]:
    """Verify Task 6's immutable manifest and return its identity."""
    root = Path(root)
    path = root / "MANIFEST.sha256"
    if not path.is_file(): raise ValueError("Task 6 raw MANIFEST.sha256 is missing")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not parts[1] or parts[1] in entries:
            raise ValueError("malformed Task 6 raw manifest")
        rel = Path(parts[1]);
        if rel.is_absolute() or ".." in rel.parts: raise ValueError("unsafe raw manifest path")
        target = root / rel
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"raw manifest mismatch: {parts[1]}")
        entries[parts[1]] = parts[0]
    actual = _manifest_entries(root)
    if entries != actual: raise ValueError("raw manifest inventory mismatch")
    return {"sha256": sha256_file(path), "entries": entries}


def _load_records(root: Path) -> dict[str, dict[str, Any]]:
    if not (root / "samples").is_dir(): raise ValueError("Task 6 samples directory is missing")
    records: dict[str, dict[str, Any]] = {}
    for sample in sorted((root / "samples").iterdir()):
        if not sample.is_dir() or ".attempt." in sample.name: continue
        status_path = sample / "status.json"
        if not status_path.is_file(): raise ValueError(f"sample status missing: {sample.name}")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "success": continue
        for filename, digest in status.get("artifact_sha256", {}).items():
            path = sample / filename
            if not path.is_file() or sha256_file(path) != digest: raise ValueError(f"raw artifact mismatch: {sample.name}/{filename}")
        meta_path = sample / "sample.json"
        if not meta_path.is_file(): raise ValueError(f"sample metadata missing: {sample.name}")
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        for array_path in sample.glob("*.npy"):
            record[array_path.stem] = np.load(array_path, allow_pickle=False)
        records[sample.name] = record
    return records


def analyze_decoder_replays(run_root: str | Path) -> dict[str, Any]:
    """Read decoder replay records without touching the model."""
    root = Path(run_root) / "decoder"
    status_path = root / "status.json"
    if not status_path.is_file(): return {"status": "NOT_RUN", "metrics": [], "reason": "decoder status is absent"}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("status") != "COMPLETE": return {"status": "NOT_RUN", "metrics": [], "reason": "decoder stopped or incomplete", "status_record": status}
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "record.json").is_file(): continue
        record = json.loads((path / "record.json").read_text(encoding="utf-8"))
        for array_path in path.glob("*.npy"): record[array_path.stem] = np.load(array_path, allow_pickle=False)
        records[path.name] = record
    if len(records) != 16: return {"status": "INCOMPLETE", "metrics": [], "reason": f"expected 16 decoder records, found {len(records)}"}
    rows = []
    for name, record in sorted(records.items()):
        arrays = [value for key, value in record.items() if key.endswith("float32") and isinstance(value, np.ndarray)]
        row = {"replay_id": name, "precision": record.get("spec", {}).get("decode_precision", record.get("precision")),
               "status": record.get("status", "success"), "elapsed_seconds": record.get("elapsed_seconds")}
        if "decoded_final_float32" in record: row["decoded_final_rms"] = float(np.sqrt(np.mean(np.asarray(record["decoded_final_float32"], dtype=np.float64) ** 2)))
        if "roundtrip_direct_condition_latent" in record:
            row["roundtrip_direct_latent_rms"] = float(np.sqrt(np.mean(np.asarray(record["roundtrip_direct_condition_latent"], dtype=np.float64) ** 2)))
        if "roundtrip_uint8_condition_latent" in record:
            row["roundtrip_uint8_latent_rms"] = float(np.sqrt(np.mean(np.asarray(record["roundtrip_uint8_condition_latent"], dtype=np.float64) ** 2)))
        if "decoded_final_float32" in record and "roundtrip_uint8_input" in record:
            row["uint8_input_rms"] = rms64(record["roundtrip_uint8_input"])
        rows.append(row)
    by_id = {row["replay_id"]: row for row in rows}
    for row in rows:
        twin = row["replay_id"].replace("__native_bf16", "__temporary_fp32") if "native_bf16" in row["replay_id"] else row["replay_id"].replace("__temporary_fp32", "__native_bf16")
        row["paired_precision_replay_id"] = twin if twin in by_id else None
    return {"status": "COMPLETE", "metrics": rows, "decoder_calls": 16,
            "reason": "8 selected latents decoded serially at native BF16 and temporary FP32"}


def _png(path: Path, series: list[tuple[float, float]], title: str) -> None:
    """Small dependency-free PNG companion for every SVG plot."""
    try:
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (900, 520), "white"); draw = ImageDraw.Draw(image)
        draw.text((30, 20), title, fill="black"); draw.line((80, 460, 850, 460), fill="black", width=2); draw.line((80, 60, 80, 460), fill="black", width=2)
        if series:
            xs = [x for x, _ in series]; ys = [y for _, y in series]; xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
            if xmax == xmin: xmax += 1
            if ymax == ymin: ymax += 1
            pts = [(80 + 770*(x-xmin)/(xmax-xmin), 460 - 400*(y-ymin)/(ymax-ymin)) for x, y in series]
            if len(pts) > 1: draw.line(pts, fill=(37,99,235), width=3)
            for x, y in pts: draw.ellipse((x-4,y-4,x+4,y+4), fill=(37,99,235))
        image.save(path)
    except Exception:
        raw = zlib.compress(b"\x00\xff\xff\xff")
        data = b"\x89PNG\r\n\x1a\n"
        def chunk(kind: bytes, value: bytes) -> bytes: return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)
        data += chunk(b"IHDR", struct.pack(">IIBBBBB", 1,1,8,2,0,0,0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b"")
        path.write_bytes(data)


def _svg(path: Path, series: list[tuple[float, float]], title: str, y_label: str) -> None:
    pts = " ".join(f"{80 + 760*i/max(1,len(series)-1):.2f},{450 - 360*(y-min((v for _,v in series), default=0))/(max((v for _,v in series), default=1)-min((v for _,v in series), default=0) or 1):.2f}" for i, (_, y) in enumerate(series))
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520"><rect width="100%" height="100%" fill="white"/><text x="80" y="35" font-size="20">{title}</text><line x1="80" y1="450" x2="850" y2="450" stroke="black"/><line x1="80" y1="60" x2="80" y2="450" stroke="black"/><polyline fill="none" stroke="#2563eb" stroke-width="2" points="{pts}"/><text x="400" y="500">alpha</text><text x="15" y="260" transform="rotate(-90 15 260)">{y_label}</text></svg>', encoding="utf-8")


def _math_note() -> str:
    return r"""# Task 6 mathematical note

For state (k) and noise seed (s), the local expansion tested here is
[
G_{k,s}(z_k+delta)=G_{k,s}(z_k)+J_{k,s}(z_k)delta+R_{k,s}(delta).
]
If (G) is sufficiently smooth, a centered difference has truncation error
(O(h^2)). Floating point evaluation adds a term of order (O(u/h)), so an
arbitrarily small step is not automatically more accurate. FP32 differences
are reduced in float64 and all quantitative spaces are kept separate.

Passing the empirical finite-direction gates is evidence about the sampled
state, directions, and scales only. It proves neither existence of a Jacobian
nor a Lipschitz upper bound, and it does not establish low rank. Decoder and
re-encoding results describe the composite (E\circ D\circ G), not (G) alone.
"""


def _report(result: Mapping[str, Any], decoder: Mapping[str, Any]) -> str:
    if result.get("status") != "COMPLETE":
        return f"# UMI Task 6 report\n\nStatus: `{result.get('status')}`. {result.get('reason','')}\n"
    fits = result.get("fits", []); add = result.get("additivity", []); pred = result.get("predictions", [])
    return "\n".join(["# UMI Task 6 cross-context local-linearity pilot", "", "## Result",
        f"- Predicted-latent direction gates: {sum(x.get('candidate_status') == 'PASS' for x in fits)}/{len(fits)}.",
        f"- Additivity: {sum(x.get('status') == 'PASS' for x in add)}/{len(add)} (exactly six planned checks).",
        f"- Held-out prediction: {sum(x.get('status') == 'PASS' for x in pred)}/{len(pred)} (exactly 24 planned checks).",
        f"- Maximum additivity relative error: {result.get('summary',{}).get('max_additivity_error')}; maximum holdout relative error: {result.get('summary',{}).get('max_holdout_error')}.",
        f"- Decoder status: {decoder.get('status')}; decoder calls: {decoder.get('decoder_calls', 0)}.", "", "## Scope and limitations",
        "The pilot is bridge_0/seed 0 only. Other states and seeds are `not_run`, not failures. MP4/PNG are visualization-only; quantitative tensors are float32 with float64 reductions.", "", "## Interpretation", "Empirical passage does not prove Jacobian existence, a Lipschitz bound, or low rank. See `math_note.md`.", ""]) 


def _write_manifest(root: Path) -> Path:
    entries = _manifest_entries(root)
    _atomic_text(root / "MANIFEST.sha256", "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items())))
    return root / "MANIFEST.sha256"


def _package_valid(root: Path, raw_manifest_sha: str) -> bool:
    marker = root / "analysis_provenance.json"
    manifest = root / "MANIFEST.sha256"
    if not marker.is_file() or not manifest.is_file(): return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        verify_analysis_manifest(root)
        return value.get("raw_manifest_sha256") == raw_manifest_sha
    except Exception: return False


def verify_analysis_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root); manifest = root / "MANIFEST.sha256"
    if not manifest.is_file(): raise ValueError("analysis MANIFEST.sha256 is missing")
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in entries: raise ValueError("malformed analysis manifest")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts or relative.name in {"MANIFEST.sha256", "review_bundle.zip"}: raise ValueError("unsafe analysis manifest entry")
        target = root / relative
        if not target.is_file() or sha256_file(target) != parts[0]: raise ValueError(f"analysis artifact mismatch: {parts[1]}")
        entries[parts[1]] = parts[0]
    if entries != _manifest_entries(root): raise ValueError("analysis manifest inventory mismatch")
    return {"sha256": sha256_file(manifest), "entries": entries}


def write_task6_artifacts(result: Mapping[str, Any], output_dir: str | Path, *, raw_root: str | Path | None = None,
                          decoder: Mapping[str, Any] | None = None, source_dir: str | Path | None = None) -> dict[str, Any]:
    output = Path(output_dir)
    raw = Path(raw_root) if raw_root is not None else None
    raw_snapshot = verify_raw_manifest(raw) if raw is not None else {"sha256": None, "entries": {}}
    if output.exists() and _package_valid(output, raw_snapshot["sha256"]):
        return {"output_dir": str(output), "manifest": str(output / "MANIFEST.sha256"), "idempotent": True}
    if output.exists() and any(output.iterdir()): raise FileExistsError("analysis output exists but is stale or tampered")
    parent = output.parent; parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".task6-analysis-", dir=str(parent)))
    try:
        (stage / "figures").mkdir(); (stage / "analysis_tensors").mkdir()
        tensors = result.get("tensors", {})
        for name, value in tensors.items(): np.save(stage / "analysis_tensors" / Path(name).name, np.asarray(value, dtype=np.float32), allow_pickle=False)
        _rows_csv(stage / "point_metrics.csv", result.get("points", [])); _rows_csv(stage / "difference_metrics.csv", result.get("derivatives", [])); _rows_csv(stage / "fit_metrics.csv", result.get("fits", [])); _rows_csv(stage / "window_decisions.csv", result.get("fits", []) + result.get("image_fits", [])); _rows_csv(stage / "additivity_metrics.csv", result.get("additivity", [])); _rows_csv(stage / "prediction_metrics.csv", result.get("predictions", [])); _rows_csv(stage / "failure_amplitudes.csv", [{"direction_id": row.get("direction_id"), "alpha": row.get("alpha"), "status": row.get("candidate_status"), "failure_reasons": row.get("failure_reasons", "")} for row in result.get("fits", [])]); _rows_csv(stage / "decoder_metrics.csv", (decoder or {}).get("metrics", [])); _rows_csv(stage / "decoder_roundtrip_metrics.csv", (decoder or {}).get("metrics", []))
        _json(stage / "task6_summary.json", {key: value for key, value in result.items() if key != "tensors"}); _json(stage / "decoder_summary.json", decoder or {"status": "NOT_RUN"})
        provenance = {"schema_version": "umi-task6-analysis-v1", "raw_manifest_sha256": raw_snapshot["sha256"], "raw_root": str(raw) if raw else None}
        _json(stage / "analysis_provenance.json", provenance); _json(stage / "provenance.json", provenance); _json(stage / "config.json", {"alphas": list(ALPHAS), "directions": list(DIRECTION_IDS), "spaces": ["prediction_latent", "native_rgb", "fp32_rgb", "float_roundtrip_condition_latent", "uint8_simulated_roundtrip_condition_latent"]})
        _atomic_text(stage / "math_note.md", _math_note()); _atomic_text(stage / "experiment_report.md", _report(result, decoder or {"status": "NOT_RUN"})); _atomic_text(stage / "resource_report.md", "# Resource report\n\nResource evidence remains in the immutable raw run directory.\n")
        # A compact response chart; all raw tensors remain outside the zip.
        values = [(float(row.get("alpha", i)), float(row.get("response_rms", 0.0) or 0.0)) for i, row in enumerate(result.get("points", [])) if row.get("response_rms") is not None]
        _svg(stage / "figures" / "response.svg", values, "Task 6 response", "RMS response"); _png(stage / "figures" / "response.png", values, "Task 6 response")
        if source_dir is not None:
            source = Path(source_dir)
            for name in ("umi_task6_decoder.py", "analyze_umi_task6.py"):
                if (source / name).is_file(): shutil.copyfile(source / name, stage / name)
        bundle = stage / "review_bundle.zip"
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file() and path != bundle and "analysis_tensors" not in path.parts:
                    info = zipfile.ZipInfo(path.relative_to(stage).as_posix(), (1980,1,1,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
            archive.writestr("README.txt", "Large raw tensors remain in the immutable run directory.\n")
        _write_manifest(stage)
        if raw is not None and verify_raw_manifest(raw)["sha256"] != raw_snapshot["sha256"]: raise ValueError("raw evidence changed during analysis")
        if output.exists(): raise FileExistsError("analysis output appeared during publication")
        os.replace(stage, output); stage = None
    finally:
        if stage is not None: shutil.rmtree(stage, ignore_errors=True)
    return {"output_dir": str(output), "manifest": str(output / "MANIFEST.sha256"), "review_bundle_bytes": (output / "review_bundle.zip").stat().st_size, "idempotent": False}


def analyze_task6_run(run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir).resolve(); status_path = root / "run_status.json"
    if not status_path.is_file(): raise ValueError("Task 6 run_status.json is missing")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (root / "MANIFEST.sha256").is_file(): verify_raw_manifest(root)
    records = _load_records(root)
    result = analyze_task6_records(records, group=status.get("group"), plan_detail=status.get("plan_detail"))
    decoder = analyze_decoder_replays(root)
    destination = Path(output_dir) if output_dir is not None else root / "task6_analysis"
    published = write_task6_artifacts(result, destination, raw_root=root, decoder=decoder, source_dir=Path(__file__).parent)
    return {"status": result.get("status"), "decoder_status": decoder.get("status"), **published}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("run_dir"); parser.add_argument("--output-dir")
    args = parser.parse_args(argv); print(json.dumps(analyze_task6_run(args.run_dir, args.output_dir), ensure_ascii=False, sort_keys=True)); return 0


__all__ = ["analyze_decoder_replays", "analyze_task6_records", "analyze_task6_run", "difference_metrics", "rms64", "verify_analysis_manifest", "verify_raw_manifest", "write_task6_artifacts"]

if __name__ == "__main__": raise SystemExit(main())
