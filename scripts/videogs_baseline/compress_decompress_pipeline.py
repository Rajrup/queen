#!/usr/bin/env python3
"""
VideoGS Combined Compress + Decompress Pipeline (no intermediate PNG).

Skips the intermediate PNG I/O by piping raw quantized pixel data directly
to/from ffmpeg. Quality and compression size are equivalent to the 4-step
PNG-based pipeline for the same QP, since PNG is a lossless format.

Pipeline per group:
  Compress:  Read PLY -> Quantize -> Pipe raw to ffmpeg -> H.264 MP4
  Decompress: MP4 -> Pipe from ffmpeg -> Dequantize -> Save PLY
"""

import os
import sys
import csv
import json
import time
import argparse
import subprocess
import numpy as np
from plyfile import PlyData
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QUEEN_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_VIDEOGS_COMPRESSION = os.path.join(_QUEEN_ROOT, "VideoGS", "compression")
_BACKUP_DIR = os.path.join(_THIS_DIR, "backup")

if _VIDEOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _VIDEOGS_COMPRESSION)
if _BACKUP_DIR not in sys.path:
    sys.path.insert(0, _BACKUP_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from compress_decompress import (
    calculate_image_size,
    quantize_videogs_image,
    dequantize_videogs_image,
    encode_channel_raw,
    decode_channel_raw,
    build_channel_qp_map,
)
from compress_to_png_full_sh import get_ply_matrix, find_queen_ply_path
from decompress_from_png_full_sh import save_ply

CAPPED_QP = 22

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VideoGS combined compress + decompress (no intermediate PNG)"
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Checkpoint dir with frame folders (0, 1, ...)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output for compressed videos, metadata, benchmark CSV")
    parser.add_argument("--output_ply_folder", type=str, required=True,
                        help="Output for decompressed PLY files")
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=300)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=2)
    parser.add_argument("--qp", type=int, default=25,
                        help="H.264 QP (0=lossless, 51=worst). Default: 25")
    args = parser.parse_args()

    video_folder = os.path.join(args.output_folder, "compressed_video")
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(args.output_ply_folder, exist_ok=True)

    channel_qp_map = build_channel_qp_map(args.sh_degree, args.qp)

    min_max_json = {}
    group_info_json = {}
    benchmark_rows = []

    num_frames_total = args.frame_end - args.frame_start + 1
    num_groups = (num_frames_total + args.group_size - 1) // args.group_size

    capped_qp = min(args.qp, CAPPED_QP)
    print("=" * 70)
    print("VideoGS Combined Compress + Decompress Pipeline")
    print("=" * 70)
    print(f"  PLY path:       {args.ply_path}")
    print(f"  Output folder:  {args.output_folder}")
    print(f"  Output PLY:     {args.output_ply_folder}")
    print(f"  Frames:         {args.frame_start} to {args.frame_end} "
          f"(interval={args.interval})")
    print(f"  Group size:     {args.group_size}")
    print(f"  SH degree:      {args.sh_degree}")
    print(f"  Groups:         {num_groups}")
    print(f"  --- QP per attribute ---")
    print(f"  Position LSB:   {channel_qp_map[0]}")
    print(f"  Position MSB:   {channel_qp_map[1]}")
    print(f"  Normals:        {channel_qp_map[6]}")
    print(f"  DC color:       {channel_qp_map[9]}")
    ch_offset = 12
    for band in range(1, args.sh_degree + 1):
        print(f"  SH band {band}:      {channel_qp_map[ch_offset]}")
        ch_offset += (2 * band + 1) * 3
    print(f"  Opacity:        {channel_qp_map[ch_offset]}")
    print(f"  Scale:          {channel_qp_map[ch_offset + 1]}")
    print(f"  Rotation:       {channel_qp_map[ch_offset + 4]}")
    print("=" * 70)

    for group_idx in tqdm(range(num_groups), desc="Groups"):
        g_frame_start = group_idx * args.group_size + args.frame_start
        g_frame_end = min(
            (group_idx + 1) * args.group_size - 1 + args.frame_start,
            args.frame_end,
        )
        frames = list(range(g_frame_start, g_frame_end + 1, args.interval))
        if not frames:
            continue

        group_info_json[str(group_idx)] = {
            "frame_index": [
                group_idx * args.group_size,
                (group_idx + 1) * args.group_size - 1,
            ],
            "name_index": [g_frame_start, g_frame_end],
        }

        group_video_dir = os.path.join(video_folder, f"group{group_idx}")
        os.makedirs(group_video_dir, exist_ok=True)

        # ==================================================================
        # COMPRESS — Phase 1: Read PLY + Quantize
        # ==================================================================
        quantized = {}
        quantize_ms = {}
        frame_meta = {}

        # Pre-scan: find max num_points in this group so all frames share one image_size
        group_max_points = 0
        for frame in frames:
            ply_path = find_queen_ply_path(args.ply_path, frame)
            if ply_path is not None:
                _pd = PlyData.read(ply_path)
                group_max_points = max(group_max_points, len(_pd['vertex']))
        image_size = calculate_image_size(group_max_points)

        for frame in frames:
            ply_file = find_queen_ply_path(args.ply_path, frame)
            if ply_file is None:
                tqdm.write(f"Warning: PLY not found for frame {str(frame).zfill(4)}")
                continue

            current_data, uncompressed_size = get_ply_matrix(ply_file)
            num_points = current_data.shape[0]

            t0 = time.perf_counter()
            images, frame_min_max = quantize_videogs_image(current_data, image_size)
            t1 = time.perf_counter()

            quantized[frame] = images
            quantize_ms[frame] = (t1 - t0) * 1000.0
            frame_meta[frame] = {
                "num_points": num_points,
                "uncompressed_size": uncompressed_size,
            }

            min_max_json[f"{frame}_num"] = num_points
            for k, v in frame_min_max.items():
                min_max_json[f"{frame}_{k}"] = v

            del current_data

        valid_frames = [f for f in frames if f in quantized]
        if not valid_frames:
            continue

        # ==================================================================
        # COMPRESS — Phase 2: Encode channels via raw-video pipe
        # ==================================================================
        channels = sorted(
            {int(k) for imgs in quantized.values() for k in imgs.keys()}
        )

        t_enc_start = time.perf_counter()
        for ch in channels:
            if ch not in channel_qp_map:
                raise RuntimeError(
                    f"Channel {ch} not found in QP map. "
                    f"Check --sh_degree ({args.sh_degree}) matches the PLY."
                )
            ch_qp = channel_qp_map[ch]

            raw_frames = [quantized[f][str(ch)] for f in valid_frames]
            mp4_path = os.path.join(group_video_dir, f"{ch}.mp4")
            encode_channel_raw(raw_frames, image_size, ch_qp, mp4_path)

        t_enc_end = time.perf_counter()
        group_encode_ms = (t_enc_end - t_enc_start) * 1000

        group_compressed_bytes = sum(
            os.path.getsize(os.path.join(group_video_dir, f))
            for f in os.listdir(group_video_dir)
            if f.endswith(".mp4")
        )

        del quantized

        # ==================================================================
        # DECOMPRESS — Phase 3: Decode channels via raw-video pipe
        # ==================================================================
        decoded_images = {f: {} for f in valid_frames}

        t_dec_start = time.perf_counter()
        for ch in channels:
            mp4_path = os.path.join(group_video_dir, f"{ch}.mp4")
            ch_frames = decode_channel_raw(mp4_path, image_size, len(valid_frames))
            for i, f in enumerate(valid_frames):
                decoded_images[f][str(ch)] = ch_frames[i]
        t_dec_end = time.perf_counter()
        group_decode_ms = (t_dec_end - t_dec_start) * 1000

        # ==================================================================
        # DECOMPRESS — Phase 4: Dequantize + Save PLY
        # ==================================================================
        encode_ms_per_frame = group_encode_ms / len(valid_frames)
        decode_ms_per_frame = group_decode_ms / len(valid_frames)
        compressed_per_frame = group_compressed_bytes / len(valid_frames)

        for frame in valid_frames:
            t0 = time.perf_counter()
            ply_data = dequantize_videogs_image(
                decoded_images[frame], frame, min_max_json
            )
            t1 = time.perf_counter()
            dequantize_ms = (t1 - t0) * 1000

            frame_str = str(frame).zfill(4)
            frame_ply_dir = os.path.join(
                args.output_ply_folder, "frames", frame_str
            )
            os.makedirs(frame_ply_dir, exist_ok=True)
            save_ply(
                ply_data,
                os.path.join(frame_ply_dir, "point_cloud.ply"),
                args.sh_degree,
            )

            benchmark_rows.append({
                "frame": frame,
                "quantize_ms": quantize_ms[frame],
                "encode_ms": encode_ms_per_frame,
                "decode_ms": decode_ms_per_frame,
                "dequantize_ms": dequantize_ms,
                "total_encode_ms": quantize_ms[frame] + encode_ms_per_frame,
                "total_decode_ms": decode_ms_per_frame + dequantize_ms,
                "uncompressed_size_bytes": frame_meta[frame]["uncompressed_size"],
                "compressed_size_bytes": int(compressed_per_frame),
                "original_points": frame_meta[frame]["num_points"],
            })

            tqdm.write(
                f"  Frame {frame}: N={frame_meta[frame]['num_points']}, "
                f"enc={quantize_ms[frame] + encode_ms_per_frame:.1f} ms, "
                f"dec={decode_ms_per_frame + dequantize_ms:.1f} ms, "
                f"comp={compressed_per_frame / 1024 / 1024:.2f} MB"
            )

        del decoded_images

    # ------------------------------------------------------------------
    # Save metadata
    # ------------------------------------------------------------------
    with open(os.path.join(args.output_folder, "min_max.json"), "w") as f:
        json.dump(min_max_json, f, indent=4)
    with open(os.path.join(args.output_folder, "group_info.json"), "w") as f:
        json.dump(group_info_json, f, indent=4)

    config_out = {
        "ply_path": args.ply_path,
        "frame_start": args.frame_start,
        "frame_end": args.frame_end,
        "group_size": args.group_size,
        "interval": args.interval,
        "sh_degree": args.sh_degree,
        "qp": args.qp,
        "channel_qp_map": {str(k): v for k, v in sorted(channel_qp_map.items())},
    }
    with open(os.path.join(args.output_folder, "videogs_config.json"), "w") as f:
        json.dump(config_out, f, indent=4)

    # ------------------------------------------------------------------
    # Benchmark CSV + summary
    # ------------------------------------------------------------------
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_videogs_pipeline.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame_id", "quantize_ms", "encode_ms", "decode_ms",
                "dequantize_ms", "total_encode_ms", "total_decode_ms",
                "uncompressed_size_bytes", "compressed_size_bytes",
                "original_points",
            ])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"],
                    f"{r['quantize_ms']:.2f}",
                    f"{r['encode_ms']:.2f}",
                    f"{r['decode_ms']:.2f}",
                    f"{r['dequantize_ms']:.2f}",
                    f"{r['total_encode_ms']:.2f}",
                    f"{r['total_decode_ms']:.2f}",
                    r["uncompressed_size_bytes"],
                    r["compressed_size_bytes"],
                    r["original_points"],
                ])

        n = len(benchmark_rows)
        total_enc = sum(r["total_encode_ms"] for r in benchmark_rows)
        total_dec = sum(r["total_decode_ms"] for r in benchmark_rows)
        total_uncomp = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_comp = sum(r["compressed_size_bytes"] for r in benchmark_rows)

        print("\n" + "=" * 70)
        print("Benchmark Summary (VideoGS combined pipeline)")
        print("=" * 70)
        print(f"  Frames processed:        {n}")
        print(f"  Total encode time:       {total_enc / 1000:.2f} s  "
              f"(avg {total_enc / n:.2f} ms/frame)")
        print(f"  Total decode time:       {total_dec / 1000:.2f} s  "
              f"(avg {total_dec / n:.2f} ms/frame)")
        print(f"  Total uncompressed:      {total_uncomp / 1024 / 1024:.2f} MB  "
              f"(avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed (MP4):  {total_comp / 1024 / 1024:.2f} MB  "
              f"(avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:       {total_uncomp / total_comp:.2f}x")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
