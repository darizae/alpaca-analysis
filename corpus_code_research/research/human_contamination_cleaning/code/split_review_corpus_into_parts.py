#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import wave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a review corpus concat WAV + Raven selection table into deterministic parts "
            "without modifying the source corpus."
        )
    )
    parser.add_argument(
        "--corpus-dir",
        required=True,
        type=Path,
        help="Directory containing review_concat.wav, review_selection_table.txt, and manifest.json.",
    )
    parser.add_argument(
        "--parts",
        type=int,
        default=6,
        help="Number of deterministic parts to generate.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="split_into_6_parts",
        help="Name of output subdirectory created under --corpus-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def stable_float(value: float) -> str:
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def build_frame_ranges(total_frames: int, parts: int) -> list[tuple[int, int]]:
    base = total_frames // parts
    remainder = total_frames % parts
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(parts):
        width = base + (1 if idx < remainder else 0)
        end = start + width
        ranges.append((start, end))
        start = end
    return ranges


def read_selection_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"No header detected in selection table: {path}")
    return fieldnames, rows


def write_part_wav(
    src_wav: Path,
    dst_wav: Path,
    start_frame: int,
    frame_count: int,
) -> dict[str, int]:
    with wave.open(str(src_wav), "rb") as src:
        src_params = src.getparams()
        src.setpos(start_frame)
        with wave.open(str(dst_wav), "wb") as dst:
            dst.setnchannels(src_params.nchannels)
            dst.setsampwidth(src_params.sampwidth)
            dst.setframerate(src_params.framerate)
            dst.setcomptype(src_params.comptype, src_params.compname)

            remaining = frame_count
            chunk_frames = 65536
            while remaining > 0:
                take = min(remaining, chunk_frames)
                data = src.readframes(take)
                if not data:
                    raise RuntimeError("Unexpected EOF while splitting WAV.")
                dst.writeframes(data)
                remaining -= take

    return {
        "start_frame": start_frame,
        "end_frame_exclusive": start_frame + frame_count,
        "frame_count": frame_count,
    }


def main() -> int:
    args = parse_args()
    if args.parts < 1:
        raise ValueError("--parts must be >= 1")

    corpus_dir = args.corpus_dir.resolve()
    src_wav = corpus_dir / "review_concat.wav"
    src_table = corpus_dir / "review_selection_table.txt"
    src_manifest = corpus_dir / "manifest.json"

    for required in (src_wav, src_table, src_manifest):
        if not required.exists():
            raise FileNotFoundError(f"Required file missing: {required}")

    output_dir = corpus_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --force to proceed."
        )

    with wave.open(str(src_wav), "rb") as wav_in:
        total_frames = wav_in.getnframes()
        sample_rate = wav_in.getframerate()
        channels = wav_in.getnchannels()
        sampwidth = wav_in.getsampwidth()

    frame_ranges = build_frame_ranges(total_frames, args.parts)
    source_duration_s = total_frames / sample_rate

    original_fields, selection_rows = read_selection_rows(src_table)
    extra_fields = [
        "split_part_index",
        "split_part_start_s",
        "split_part_end_s",
        "split_original_begin_s",
        "split_original_end_s",
        "split_row_clipped",
    ]
    out_fields = original_fields + extra_fields

    run_manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": "split_review_corpus_into_parts.py",
        "source": {
            "corpus_dir": str(corpus_dir),
            "wav": str(src_wav),
            "selection_table": str(src_table),
            "manifest": str(src_manifest),
            "wav_sha256": sha256_file(src_wav),
            "selection_table_sha256": sha256_file(src_table),
            "manifest_sha256": sha256_file(src_manifest),
            "duration_s": source_duration_s,
            "total_frames": total_frames,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sample_width_bytes": sampwidth,
            "selection_rows": len(selection_rows),
        },
        "config": {
            "parts": args.parts,
            "output_subdir": args.output_subdir,
        },
        "parts": [],
    }

    total_written_rows = 0
    row_part_counts = [0] * len(selection_rows)

    for part_idx, (start_frame, end_frame) in enumerate(frame_ranges, start=1):
        part_name = f"part_{part_idx:02d}"
        part_dir = output_dir / part_name
        part_dir.mkdir(parents=True, exist_ok=True)

        part_frame_count = end_frame - start_frame
        part_start_s = start_frame / sample_rate
        part_end_s = end_frame / sample_rate

        part_wav = part_dir / f"review_concat_{part_name}.wav"
        part_table = part_dir / f"review_selection_table_{part_name}.txt"
        part_metadata = part_dir / f"metadata_{part_name}.json"

        wav_stats = write_part_wav(
            src_wav=src_wav,
            dst_wav=part_wav,
            start_frame=start_frame,
            frame_count=part_frame_count,
        )

        selected_rows: list[dict[str, str]] = []
        clipped_count = 0
        for row_idx, src_row in enumerate(selection_rows):
            original_begin = float(src_row["Begin Time (s)"])
            original_end = float(src_row["End Time (s)"])

            overlaps = (original_end > part_start_s) and (
                original_begin < part_end_s or (
                    part_idx == args.parts and original_begin <= part_end_s
                )
            )
            if not overlaps:
                continue

            local_begin = max(original_begin, part_start_s) - part_start_s
            local_end = min(original_end, part_end_s) - part_start_s
            local_begin = max(local_begin, 0.0)
            local_end = min(local_end, (part_end_s - part_start_s))

            clipped = (original_begin < part_start_s) or (original_end > part_end_s)
            if clipped:
                clipped_count += 1

            row = dict(src_row)
            row["Begin Time (s)"] = stable_float(local_begin)
            row["End Time (s)"] = stable_float(local_end)
            row["split_part_index"] = str(part_idx)
            row["split_part_start_s"] = stable_float(part_start_s)
            row["split_part_end_s"] = stable_float(part_end_s)
            row["split_original_begin_s"] = stable_float(original_begin)
            row["split_original_end_s"] = stable_float(original_end)
            row["split_row_clipped"] = "true" if clipped else "false"
            selected_rows.append(row)
            row_part_counts[row_idx] += 1

        # Re-number "Selection" for Raven convenience while keeping original row identity fields.
        for new_selection, row in enumerate(selected_rows, start=1):
            row["Selection"] = str(new_selection)

        with part_table.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=out_fields,
                delimiter="\t",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(selected_rows)

        total_written_rows += len(selected_rows)
        part_meta = {
            "part_index": part_idx,
            "part_name": part_name,
            "source_wav": str(src_wav),
            "source_selection_table": str(src_table),
            "source_manifest": str(src_manifest),
            "part_start_s": part_start_s,
            "part_end_s": part_end_s,
            "part_duration_s": part_end_s - part_start_s,
            "sample_rate_hz": sample_rate,
            "channels": channels,
            "sample_width_bytes": sampwidth,
            "wav": {
                "path": str(part_wav),
                "sha256": sha256_file(part_wav),
                **wav_stats,
            },
            "selection_table": {
                "path": str(part_table),
                "sha256": sha256_file(part_table),
                "rows_written": len(selected_rows),
                "rows_clipped_from_boundary_overlap": clipped_count,
                "fieldnames": out_fields,
            },
        }
        part_metadata.write_text(json.dumps(part_meta, indent=2), encoding="utf-8")

        run_manifest["parts"].append(
            {
                "part_index": part_idx,
                "part_dir": str(part_dir),
                "wav": str(part_wav),
                "selection_table": str(part_table),
                "metadata": str(part_metadata),
                "rows_written": len(selected_rows),
                "rows_clipped_from_boundary_overlap": clipped_count,
                "part_start_s": part_start_s,
                "part_end_s": part_end_s,
                "start_frame": start_frame,
                "end_frame_exclusive": end_frame,
                "frame_count": part_frame_count,
            }
        )

    rows_with_cross_part_overlap = sum(1 for count in row_part_counts if count > 1)
    extra_row_copies_from_overlap = sum(max(0, count - 1) for count in row_part_counts)

    run_manifest["summary"] = {
        "source_selection_rows": len(selection_rows),
        "sum_rows_written_across_parts": total_written_rows,
        "rows_with_cross_part_overlap": rows_with_cross_part_overlap,
        "extra_row_copies_from_overlap": extra_row_copies_from_overlap,
        "expected_sum_rows_written": len(selection_rows) + extra_row_copies_from_overlap,
    }

    out_manifest = output_dir / "manifest.json"
    out_manifest.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
