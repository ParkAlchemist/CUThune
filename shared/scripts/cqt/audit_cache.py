from __future__ import annotations
from typing import Any, Dict, Optional

from pathlib import Path
import sys
import argparse
import csv

import torch
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-domain-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    files = sorted(args.cache_domain_root.glob("*.pt"))

    print(f"Cache root: {args.cache_domain_root}")
    print(f"Files: {len(files)}")

    records = []

    for path in tqdm(files, desc="Auditing cache"):
        record: Dict[str, Any] = {
            "cache_path": str(path),
            "status": "unknown",
        }

        try:
            data = torch.load(path, map_location="cpu")
            meta = data.get("metadata", {})

            cqt = data.get("cqt_mag", None)

            record.update({
                "status": "ok",
                "domain": meta.get("domain"),
                "source_path": meta.get("source_path"),
                "duration_seconds": meta.get("processed_stats", {}).get("duration_seconds"),
                "rms": meta.get("processed_stats", {}).get("rms"),
                "peak": meta.get("processed_stats", {}).get("peak"),
                "cqt_frames": meta.get("cqt_frames"),
                "has_phase": "cqt_phase" in data,
                "has_chroma": "chroma" in data,
                "cqt_shape": list(cqt.shape) if cqt is not None else None,
            })

        except Exception as exc:
            record.update({
                "status": "error",
                "error": repr(exc),
            })

        records.append(record)

    ok = [r for r in records if r["status"] == "ok"]
    errors = [r for r in records if r["status"] != "ok"]

    print("\nSummary:")
    print(f"  ok:     {len(ok)}")
    print(f"  errors: {len(errors)}")

    durations = [
        float(r["duration_seconds"])
        for r in ok
        if r.get("duration_seconds") is not None
    ]

    if durations:
        print(f"  total hours: {sum(durations) / 3600.0:.2f}")
        print(f"  min duration: {min(durations):.2f}s")
        print(f"  max duration: {max(durations):.2f}s")
        print(f"  mean duration: {sum(durations) / len(durations):.2f}s")

    out_csv = args.out_csv or (args.cache_domain_root / "audit.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for record in records for key in record.keys()})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
