#!/usr/bin/env python3
"""Fail-closed verification and extraction for the audited PAPILA Kaggle candidate.

This is a provenance gate, not a general ZIP extractor.  It accepts only the
exact Kaggle archive whose byte size and GCS MD5 were recorded in
``papila_kaggle_download_status.md``.  It verifies those values and every ZIP
CRC before extracting to a new directory, rejects unsafe member paths, then
runs the official PAPILA clinical/image normalizer.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


EXPECTED_BYTES = 591_089_413
EXPECTED_MD5_BASE64 = "ZfKbYTi5JjuDIdhZo3N9jA=="


def _digest(path: Path, algorithm: str) -> bytes:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _find_papila_root(extraction_root: Path) -> Path:
    candidates = sorted(
        path.parent
        for path in extraction_root.rglob("ClinicalData")
        if path.is_dir() and (path.parent / "FundusImages").is_dir()
    )
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one PAPILA root with ClinicalData and FundusImages; found {candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument(
        "--reuse-extracted",
        action="store_true",
        help="Re-run schema normalization on an already extracted candidate after archive/ZIP checks; never overwrites it.",
    )
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"Candidate archive does not exist: {archive}")
    if archive.stat().st_size != EXPECTED_BYTES:
        raise SystemExit(f"Incomplete/wrong archive: {archive.stat().st_size} bytes, expected {EXPECTED_BYTES}")
    md5 = base64.b64encode(_digest(archive, "md5")).decode("ascii")
    if md5 != EXPECTED_MD5_BASE64:
        raise SystemExit(f"GCS MD5 mismatch: {md5}, expected {EXPECTED_MD5_BASE64}")
    sha256 = _digest(archive, "sha256").hex()

    if args.extract_dir.exists() and not args.reuse_extracted:
        raise SystemExit(f"Refusing to overwrite existing extraction directory: {args.extract_dir}")
    with zipfile.ZipFile(archive) as package:
        corrupted = package.testzip()
        if corrupted:
            raise SystemExit(f"ZIP CRC failure in member: {corrupted}")
        unsafe = [entry.filename for entry in package.infolist() if not _safe_member(entry.filename)]
        if unsafe:
            raise SystemExit(f"Unsafe ZIP paths rejected (first 10): {unsafe[:10]}")
        file_members = [entry for entry in package.infolist() if not entry.is_dir()]
        if not args.extract_dir.exists():
            package.extractall(args.extract_dir)

    papila_root = _find_papila_root(args.extract_dir)
    script = Path(__file__).with_name("normalize_papila_official.py")
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(script), str(papila_root), "--out", str(args.manifest_out)],
        check=True,
    )

    provenance = {
        "candidate": "Kaggle orvile/papila-retinal-fundus-images version 1",
        "archive": str(archive),
        "archive_bytes": EXPECTED_BYTES,
        "gcs_md5_base64": md5,
        "archive_sha256": sha256,
        "zip_file_members": len(file_members),
        "extraction_dir": str(args.extract_dir.resolve()),
        "papila_root": str(papila_root.resolve()),
        "canonical_manifest": str(args.manifest_out.resolve()),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_scope": "Distribution candidate passed transport, ZIP and official-schema checks; cite and describe it as PAPILA official data, not as a separate Kaggle dataset.",
    }
    provenance_path = args.manifest_out.with_suffix(".distribution_provenance.json")
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
