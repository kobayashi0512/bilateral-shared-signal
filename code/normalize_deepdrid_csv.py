#!/usr/bin/env python3
"""Add an explicit eye column to the official DeepDRiD CSV schema.

DeepDRiD's official image identifiers use ``<patient>_<l|r><view>`` (for
example ``1_l1``).  This adapter parses only that documented *image_id* field;
it never examines filesystem paths and never derives a patient ID from a
filename.  Any row that does not match the documented schema aborts the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


IMAGE_ID_PATTERN = re.compile(r"^[^_]+_([lLrR])[0-9]+$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"patient_id", "image_id"}
    missing = sorted(required - set(fields))
    if missing:
        raise SystemExit(f"Missing required DeepDRiD fields: {', '.join(missing)}")

    output_fields = list(fields)
    for extra in ("eye", "eye_source"):
        if extra not in output_fields:
            output_fields.append(extra)
    failures: list[dict[str, str]] = []
    output_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        image_id = (row.get("image_id") or "").strip()
        match = IMAGE_ID_PATTERN.fullmatch(image_id)
        if not match:
            failures.append({"row": str(row_number), "image_id": image_id})
            continue
        output = dict(row)
        output["eye"] = "L" if match.group(1).lower() == "l" else "R"
        output["eye_source"] = "DeepDRiD_documented_image_id_schema"
        output_rows.append(output)
    if failures:
        raise SystemExit(json.dumps({"unrecognized_rows": failures[:20], "count": len(failures)}, ensure_ascii=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    print(json.dumps({
        "input": str(args.csv_path),
        "output": str(args.out),
        "rows": len(output_rows),
        "eye_counts": {"L": sum(row["eye"] == "L" for row in output_rows), "R": sum(row["eye"] == "R" for row in output_rows)},
        "mapping_source": "documented image_id token only",
        "filesystem_paths_opened": 0,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

