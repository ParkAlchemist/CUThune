from __future__ import annotations

from pathlib import Path
import sys
import argparse
import csv
import traceback

import torch
from tqdm import tqdm

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.src.chiptune_shared.cqt.features import (
    list_audio_files,
    stable_id,
    config_hash,
    load_audio,
    trim_silence,
    peak_normalize,
    rms_normalize,
    audio_stats,
    compute_complex_cqt,
    complex_cqt_to_normalized_mag,
    compute_phase,
    compute_chroma_from_cqt_mag,
    write_jsonl,
)


CACHE_VERSION = 2


def build_feature_config(args: argparse.Namespace) -> dict:
    return {
        "cache_version": CACHE_VERSION,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_bins": args.n_bins,
        "bins_per_octave": args.bins_per_octave,
        "fmin": args.fmin,
        "db_min": args.db_min,
        "db_max": args.db_max,
        "trim_silence": args.trim_silence,
        "trim_top_db": args.trim_top_db,
        "normalization": args.normalization,
        "target_rms": args.target_rms,
        "store_phase": args.store_phase,
        "store_chroma": args.store_chroma,
        "cache_dtype": args.cache_dtype,
    }


def preprocess_file(
    audio_path: Path,
    source_root: Path,
    cache_root: Path,
    domain: str,
    args: argparse.Namespace,
    cfg_hash: str,
) -> dict:
    file_id = stable_id(audio_path, source_root)
    out_path = cache_root / f"{file_id}.pt"

    base_record = {
        "id": file_id,
        "domain": domain,
        "source_path": str(audio_path),
        "cache_path": str(out_path),
        "config_hash": cfg_hash,
        "cache_version": CACHE_VERSION,
    }

    if out_path.exists() and not args.rebuild:
        return {
            **base_record,
            "status": "skipped_existing",
        }

    try:
        y, sr = load_audio(
            audio_path,
            sample_rate=args.sample_rate,
            mono=True,
        )

        raw_stats = audio_stats(y, sr)

        if args.trim_silence:
            y = trim_silence(y, top_db=args.trim_top_db)

        if args.normalization == "peak":
            y = peak_normalize(y, target_peak=args.target_peak)
        elif args.normalization == "rms":
            y = rms_normalize(y, target_rms=args.target_rms)
        elif args.normalization == "none":
            pass
        else:
            raise ValueError(f"Unknown normalization: {args.normalization}")

        processed_stats = audio_stats(y, sr)

        if processed_stats["duration_seconds"] < args.min_duration:
            return {
                **base_record,
                "status": "too_short",
                "duration_seconds": processed_stats["duration_seconds"],
            }

        if processed_stats["rms"] < args.min_rms:
            return {
                **base_record,
                "status": "too_quiet",
                "duration_seconds": processed_stats["duration_seconds"],
                "rms": processed_stats["rms"],
            }

        cqt_complex = compute_complex_cqt(
            y,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            n_bins=args.n_bins,
            bins_per_octave=args.bins_per_octave,
            fmin=args.fmin,
        )

        cqt_mag, cqt_ref_mag = complex_cqt_to_normalized_mag(
            cqt_complex,
            db_min=args.db_min,
            db_max=args.db_max,
        )

        cqt_tensor = torch.from_numpy(cqt_mag).unsqueeze(0)

        if args.cache_dtype == "float16":
            cqt_tensor = cqt_tensor.half()
        elif args.cache_dtype == "float32":
            cqt_tensor = cqt_tensor.float()
        else:
            raise ValueError(f"Unsupported cache dtype: {args.cache_dtype}")

        payload = {
            "cache_version": CACHE_VERSION,
            "config_hash": cfg_hash,
            "metadata": {
                **base_record,
                "status": "ok",
                "sample_rate": args.sample_rate,
                "hop_length": args.hop_length,
                "n_bins": args.n_bins,
                "bins_per_octave": args.bins_per_octave,
                "fmin": args.fmin,
                "db_min": args.db_min,
                "db_max": args.db_max,
                "raw_stats": raw_stats,
                "processed_stats": processed_stats,
                "cqt_frames": int(cqt_tensor.shape[-1]),
                "cqt_shape": list(cqt_tensor.shape),
                "cqt_ref_mag": cqt_ref_mag,
            },
            "cqt_mag": cqt_tensor,
        }

        if args.store_phase:
            phase = compute_phase(cqt_complex)
            phase_tensor = torch.from_numpy(phase)

            if args.cache_dtype == "float16":
                phase_tensor = phase_tensor.half()
            else:
                phase_tensor = phase_tensor.float()

            payload["cqt_phase"] = phase_tensor

        if args.store_chroma:
            chroma = compute_chroma_from_cqt_mag(
                cqt_mag,
                bins_per_octave=args.bins_per_octave,
            )
            chroma_tensor = torch.from_numpy(chroma)

            if args.cache_dtype == "float16":
                chroma_tensor = chroma_tensor.half()
            else:
                chroma_tensor = chroma_tensor.float()

            payload["chroma"] = chroma_tensor

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out_path)

        return {
            **base_record,
            "status": "ok",
            "duration_seconds": processed_stats["duration_seconds"],
            "rms": processed_stats["rms"],
            "peak": processed_stats["peak"],
            "cqt_frames": int(cqt_tensor.shape[-1]),
            "cqt_shape": "x".join(str(x) for x in cqt_tensor.shape),
        }

    except Exception as exc:
        return {
            **base_record,
            "status": "error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def write_csv_index(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for record in records for key in record.keys()})

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--domain", required=True, choices=["poly", "chip"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)

    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--n-bins", type=int, default=84)
    parser.add_argument("--bins-per-octave", type=int, default=12)
    parser.add_argument("--fmin", type=float, default=None)

    parser.add_argument("--db-min", type=float, default=-80.0)
    parser.add_argument("--db-max", type=float, default=0.0)

    parser.add_argument("--trim-silence", action="store_true")
    parser.add_argument("--trim-top-db", type=float, default=60.0)

    parser.add_argument(
        "--normalization",
        choices=["none", "peak", "rms"],
        default="peak",
    )
    parser.add_argument("--target-peak", type=float, default=0.95)
    parser.add_argument("--target-rms", type=float, default=0.1)

    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--min-rms", type=float, default=1e-4)

    parser.add_argument("--store-phase", action="store_true")
    parser.add_argument("--store-chroma", action="store_true")

    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_config = build_feature_config(args)
    cfg_hash = config_hash(feature_config)

    source_root = args.source_root
    cache_domain_root = args.cache_root / args.domain
    cache_domain_root.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(source_root)

    if args.limit is not None:
        files = files[: args.limit]

    print(f"Domain:       {args.domain}")
    print(f"Source root:  {source_root}")
    print(f"Cache root:   {cache_domain_root}")
    print(f"Files:        {len(files)}")
    print(f"Config hash:  {cfg_hash}")
    print(f"Store phase:  {args.store_phase}")
    print(f"Store chroma: {args.store_chroma}")

    records = []

    for audio_path in tqdm(files, desc=f"Preprocessing {args.domain}"):
        record = preprocess_file(
            audio_path=audio_path,
            source_root=source_root,
            cache_root=cache_domain_root,
            domain=args.domain,
            args=args,
            cfg_hash=cfg_hash,
        )
        records.append(record)

    jsonl_path = cache_domain_root / "index.jsonl"
    csv_path = cache_domain_root / "index.csv"

    write_jsonl(jsonl_path, records)
    write_csv_index(csv_path, records)

    status_counts = {}
    for record in records:
        status = record.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nDone.")
    print("Status counts:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print(f"\nWrote: {jsonl_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
