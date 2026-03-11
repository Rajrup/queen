#!/usr/bin/env python3
"""
DracoGS Compression + Decompression for QUEEN-trained Gaussian Splat Models.

For each frame:
  1. Read PLY from QUEEN output (own PLY reader)
  2. Encode via DracoGS (in-memory Draco bitstream)
  3. Decode via DracoGS (in-memory)
  4. Save decoded result as QUEEN-compatible PLY (own PLY writer with vertex_id)

Must be run in the queen conda environment.
"""

import os
import sys
import csv
import json
import time
import argparse
import re
from pathlib import Path
from typing import Any
from tqdm import tqdm

# --- sys.path setup: DracoGS build + compression dirs ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QUEEN_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_DRACOGS_ROOT = os.path.join(_QUEEN_ROOT, "DracoGS")
_DRACOGS_BUILD = os.path.join(_DRACOGS_ROOT, "build", "compression")
_DRACOGS_COMP = os.path.join(_DRACOGS_ROOT, "compression")

for p in (_DRACOGS_BUILD, _DRACOGS_COMP):
    if p not in sys.path:
        sys.path.insert(0, p)

from compression_decompression import encode_dracogs, decode_dracogs

if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from utils import read_gs_ply, save_gs_ply

DEFAULT_EG = 16
DEFAULT_EO = 16
DEFAULT_ET = 16
DEFAULT_ES = 16
DEFAULT_CL = 10

_FRAME_SPAN_TAG_RE = re.compile(r"^frames_\d+_\d+_int_\d+$")
_FRAME_DIR_TAG_RE = re.compile(r"^frame\d+$")

# ---------------------------------------------------------------------------
# PLY utilities
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def resolve_config_root(output_folder):
    output_path = Path(output_folder)
    if _FRAME_SPAN_TAG_RE.match(output_path.name) or _FRAME_DIR_TAG_RE.match(output_path.name):
        return output_path.parent
    return output_path


def write_single_frame_benchmark_csv(csv_path: Path, benchmark_row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_id",
            "total_encode_ms",
            "total_decode_ms",
            "original_points",
            "decoded_points",
            "uncompressed_size_bytes",
            "compressed_size_bytes",
        ])
        w.writerow([
            int(benchmark_row["frame"]),
            f"{float(benchmark_row['total_encode_ms']):.2f}",
            f"{float(benchmark_row['total_decode_ms']):.2f}",
            int(benchmark_row["original_points"]),
            int(benchmark_row["decoded_points"]),
            int(benchmark_row["uncompressed_size_bytes"]),
            int(benchmark_row["compressed_size_bytes"]),
        ])


def write_single_frame_config_json(config_path: Path, config_template: dict[str, Any], frame_id: int) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(config_template)
    out["frame_list"] = [int(frame_id)]
    out["frame_id"] = int(frame_id)
    with config_path.open("w") as f:
        json.dump(out, f, indent=4)


def find_queen_ply_path(frame_dir):
    """Find PLY file in a QUEEN frame directory."""
    canonical = os.path.join(frame_dir, "point_cloud.ply")
    if os.path.exists(canonical):
        return canonical
    iter_dir = os.path.join(frame_dir, "point_cloud")
    if os.path.exists(iter_dir):
        max_iter = searchForMaxIteration(iter_dir)
        per_iter = os.path.join(iter_dir, f"iteration_{max_iter}", "point_cloud.ply")
        if os.path.exists(per_iter):
            return per_iter
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DracoGS compress + decompress for QUEEN-trained models"
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to QUEEN model output dir (e.g. pretrained_output/.../queen_compressed_cook_spinach)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder", type=str, default=None,
                        help="Folder for decompressed PLY output (optional)")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=300)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--scene_name", type=str, required=True,
                        help="Neural_3D_Video sequence name (e.g. cook_spinach)")

    # LTS quantization parameters (0=lossless, higher=more bits=better quality)
    parser.add_argument("--eg", type=int, default=DEFAULT_EG, help="Quantization bits for position (0-30)")
    parser.add_argument("--eo", type=int, default=DEFAULT_EO, help="Quantization bits for opacity (0-30)")
    parser.add_argument("--et", type=int, default=DEFAULT_ET, help="Quantization bits for rotation/scales (0-30)")
    parser.add_argument("--es", type=int, default=DEFAULT_ES, help="Quantization bits for SH (0-30)")
    parser.add_argument("--cl", type=int, default=DEFAULT_CL, help="Compression level (0-10)")

    args = parser.parse_args()
    config_root = resolve_config_root(args.output_folder)

    os.makedirs(args.output_folder, exist_ok=True)
    if args.output_ply_folder is not None:
        os.makedirs(args.output_ply_folder, exist_ok=True)

    # Draco parameters
    qp = args.eg
    qfd = args.es
    qfr1 = args.es
    qfr2 = args.es
    qfr3 = args.es
    qo = args.eo
    qs = args.et
    qr = args.et
    cl = args.cl

    # --- Print configuration ---
    print("=" * 70)
    print("DracoGS Compress + Decompress Pipeline (QUEEN)")
    print("=" * 70)
    print(f"  PLY path:           {args.ply_path}")
    print(f"  Output folder:      {args.output_folder}")
    print(f"  Output PLY folder:  {args.output_ply_folder or '(skip)'}")
    print(f"  Frames:             {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  Scene:              {args.scene_name}")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Quantization:       qp={qp} qfd={qfd} qfr1={qfr1} qfr2={qfr2} qfr3={qfr3} qo={qo} qs={qs} qr={qr}")
    print(f"  Compression level:  {cl}")
    print("=" * 70)

    # --- Per-frame loop ---
    benchmark_rows = []

    for frame in tqdm(range(args.frame_start, args.frame_end + 1, args.interval), desc="Frames"):
        frame_str = str(frame).zfill(4)

        # --- 1. Locate and read PLY ---
        frame_dir = os.path.join(args.ply_path, "frames", frame_str)
        ply_file_path = find_queen_ply_path(frame_dir)
        if ply_file_path is None:
            print(f"Warning: PLY not found for frame {frame_str} at {frame_dir}, skipping")
            continue

        gs_data, uncompressed_size_bytes = read_gs_ply(ply_file_path, sh_degree=args.sh_degree)
        N_original = gs_data["positions"].shape[0]

        # --- 2. Encode (timed) ---
        t_enc_start = time.perf_counter()
        bitstream = encode_dracogs(
            gs_data,
            qp=qp, qfd=qfd,
            qfr1=qfr1, qfr2=qfr2, qfr3=qfr3,
            qo=qo, qs=qs, qr=qr,
            cl=cl,
        )
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000
        compressed_size_bytes = len(bitstream)

        # --- 3. Decode (timed) ---
        t_dec_start = time.perf_counter()
        gs_decoded = decode_dracogs(bitstream)
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000
        N_decoded = gs_decoded["positions"].shape[0]

        # --- 4. Save PLY (QUEEN-compatible with vertex_id) ---
        if args.output_ply_folder is not None:
            frame_ply_dir = os.path.join(args.output_ply_folder, "frames", frame_str)
            os.makedirs(frame_ply_dir, exist_ok=True)
            ply_out_path = os.path.join(frame_ply_dir, "point_cloud.ply")
            save_gs_ply(gs_decoded, ply_out_path)

            canonical_frame_ply_folder = config_root / f"frame{frame}" / "decompressed_ply" / str(frame) / "point_cloud"
            canonical_frame_ply_folder.mkdir(parents=True, exist_ok=True)
            canonical_ply_out_path = canonical_frame_ply_folder / "point_cloud.ply"
            if str(canonical_ply_out_path) != ply_out_path:
                save_gs_ply(gs_decoded, str(canonical_ply_out_path))

        benchmark_rows.append({
            "frame": frame_str,
            "total_encode_ms": encode_time_ms,
            "total_decode_ms": decode_time_ms,
            "original_points": N_original,
            "decoded_points": N_decoded,
            "uncompressed_size_bytes": uncompressed_size_bytes,
            "compressed_size_bytes": compressed_size_bytes,
        })

        tqdm.write(
            f"  Frame {frame_str}: N={N_original}→{N_decoded}, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        del gs_data, gs_decoded, bitstream

    # --- Benchmark CSV and summary ---
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_dracogs.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "total_encode_ms", "total_decode_ms",
                         "original_points", "decoded_points",
                         "uncompressed_size_bytes", "compressed_size_bytes"])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"],
                    f"{r['total_encode_ms']:.2f}",
                    f"{r['total_decode_ms']:.2f}",
                    r["original_points"],
                    r["decoded_points"],
                    r["uncompressed_size_bytes"],
                    r["compressed_size_bytes"],
                ])

        n = len(benchmark_rows)
        total_enc_ms = sum(r["total_encode_ms"] for r in benchmark_rows)
        total_dec_ms = sum(r["total_decode_ms"] for r in benchmark_rows)
        total_uncomp = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_comp = sum(r["compressed_size_bytes"] for r in benchmark_rows)

        config_out = {
            "scene_name": args.scene_name,
            "sh_degree": args.sh_degree,
            "qp": qp,
            "qfd": qfd,
            "qfr1": qfr1,
            "qfr2": qfr2,
            "qfr3": qfr3,
            "qo": qo,
            "qs": qs,
            "qr": qr,
            "cl": cl,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "interval": args.interval,
            "frame_list": [int(r["frame"]) for r in benchmark_rows],
        }
        with open(os.path.join(args.output_folder, "dracogs_config.json"), "w") as f:
            json.dump(config_out, f, indent=4)

        for row in benchmark_rows:
            frame_id = int(row["frame"])
            frame_root = config_root / f"frame{frame_id}"
            write_single_frame_benchmark_csv(frame_root / "benchmark_dracogs.csv", row)
            write_single_frame_config_json(frame_root / "dracogs_config.json", config_out, frame_id)

        print("\n" + "=" * 70)
        print("Benchmark Summary (DracoGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  (avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  (avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  CSV: {csv_path}")
        print(f"  Canonical frame root:      {config_root}")
        if args.output_ply_folder is not None:
            print(f"  Canonical PLY layout:      {config_root}/frame*/decompressed_ply")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
