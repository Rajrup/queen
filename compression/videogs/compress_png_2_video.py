import os
import csv
import time
import shutil
import argparse
import json
import subprocess
from tqdm import tqdm


def get_qp_capped_channels(sh_degree):
    """Compute channel indices that should be capped at QP=22, matching the original
    compress_image_2_video.py logic: DC color, scale, and rotation channels.

    Channel layout from compress_to_png_full_sh.py:
      0,1  = x (low, high)
      2,3  = y (low, high)
      4,5  = z (low, high)
      6    = nx
      7    = ny
      8    = nz
      9    = f_dc_0
      10   = f_dc_1
      11   = f_dc_2
      12.. = f_rest_0 .. f_rest_{n_rest-1}
      ...  = opacity
      ...  = scale_0, scale_1, scale_2
      ...  = rot_0, rot_1, rot_2, rot_3
    """
    n_rest = (sh_degree + 1) ** 2 * 3 - 3
    dc_channels = [9, 10, 11]
    scale_channels = [12 + n_rest + 1, 12 + n_rest + 2, 12 + n_rest + 3]
    rot_channels = [12 + n_rest + 4, 12 + n_rest + 5, 12 + n_rest + 6, 12 + n_rest + 7]
    return set(dc_channels + scale_channels + rot_channels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress PNG attribute images to H.264 MP4 videos using ffmpeg")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="PNG output folder from compress_to_png_full_sh.py")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder for MP4 videos")
    parser.add_argument("--qp", type=int, default=22,
                        help="H.264 QP value (0=lossless, 51=worst quality). Default: 22")
    parser.add_argument("--sh_degree", type=int, default=2,
                        help="SH degree (used to determine QP capping for sensitive channels)")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    with open(os.path.join(args.input_folder, "group_info.json")) as f:
        group_info = json.load(f)

    capped_channels = get_qp_capped_channels(args.sh_degree)
    position_high_bytes = {1, 3, 5}

    total_compressed_size = 0
    total_frames = 0
    benchmark_rows = []

    for group_id, info in tqdm(sorted(group_info.items(), key=lambda x: int(x[0])), desc="Encoding groups"):
        frame_start, frame_end = info['name_index']
        group_size = frame_end - frame_start + 1

        input_group_path = os.path.join(args.input_folder, f"group{group_id}")
        output_group_path = os.path.join(args.output_folder, f"group{group_id}")
        os.makedirs(output_group_path, exist_ok=True)

        # Discover channel indices from PNG filenames
        channels = set()
        for fname in os.listdir(input_group_path):
            if fname.endswith('.png'):
                parts = fname.rsplit('_', 1)
                if len(parts) == 2:
                    ch_str = parts[1].replace('.png', '')
                    if ch_str.isdigit():
                        channels.add(int(ch_str))
        channels = sorted(channels)

        t0 = time.perf_counter()
        for ch in tqdm(channels, desc=f"Group {group_id} channels", leave=False):
            if ch in position_high_bytes:
                ch_qp = 0
            elif ch in capped_channels and args.qp > 22:
                ch_qp = 22
            else:
                ch_qp = args.qp

            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-start_number", str(frame_start),
                "-i", os.path.join(input_group_path, f"%d_{ch}.png"),
                "-vframes", str(group_size),
                "-c:v", "libx264",
                "-qp", str(ch_qp),
                "-pix_fmt", "yuvj444p",
                os.path.join(output_group_path, f"{ch}.mp4")
            ]
            subprocess.run(cmd, check=True)

        t1 = time.perf_counter()
        group_time_ms = (t1 - t0) * 1000

        # Measure compressed size for this group
        group_compressed_size = 0
        for fname in os.listdir(output_group_path):
            if fname.endswith('.mp4'):
                group_compressed_size += os.path.getsize(os.path.join(output_group_path, fname))

        avg_per_frame = group_compressed_size / group_size
        total_compressed_size += group_compressed_size
        total_frames += group_size
        benchmark_rows.append({
            "group_id": int(group_id),
            "time_ms": group_time_ms,
            "group_compressed_size_bytes": group_compressed_size,
            "num_frames": group_size,
            "avg_size_per_frame_bytes": avg_per_frame,
        })

        print(f"  Group {group_id}: {group_size} frames, "
              f"time={group_time_ms:.0f} ms, "
              f"total compressed size={group_compressed_size / 1024 / 1024:.2f} MB, "
              f"avg compressed size per frame={avg_per_frame / 1024 / 1024:.2f} MB")

    # Copy metadata JSONs
    for json_file in ["min_max.json", "viewer_min_max.json", "group_info.json"]:
        src = os.path.join(args.input_folder, json_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_folder, json_file))

    # Benchmark CSV
    csv_path = os.path.join(args.output_folder, "benchmark_compress_png_2_video.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "time_ms", "group_compressed_size_bytes", "num_frames", "avg_size_per_frame_bytes"])
        for r in benchmark_rows:
            w.writerow([r["group_id"], f"{r['time_ms']:.2f}", r["group_compressed_size_bytes"], r["num_frames"], int(r["avg_size_per_frame_bytes"])])

    total_time_ms = sum(r["time_ms"] for r in benchmark_rows)
    # Summary
    print("\n" + "=" * 60)
    print("Compression Summary (PNG to Video)")
    print("=" * 60)
    print(f"  Number of frames:      {total_frames}")
    print(f"  Number of groups:     {len(benchmark_rows)}")
    print(f"  Total encoding time:  {total_time_ms:.0f} ms")
    print(f"  Avg time per group:   {total_time_ms / len(benchmark_rows):.0f} ms")
    print(f"  Avg time per frame:   {total_time_ms / total_frames:.0f} ms")
    print(f"  Avg size per frame:   {total_compressed_size / total_frames / 1024 / 1024:.2f} MB")
    print(f"  Total compressed size: {total_compressed_size / 1024 / 1024:.2f} MB")
    print(f"  CSV: {csv_path}")
    print("=" * 60)
