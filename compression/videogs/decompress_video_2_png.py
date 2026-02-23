import os
import csv
import time
import shutil
import argparse
import json
import subprocess
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress H.264 MP4 videos back to PNG attribute images using ffmpeg")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder with MP4 videos from compress_png_2_video.py")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Output folder for decoded PNGs (input to decompress_from_png_full_sh.py)")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    with open(os.path.join(args.input_folder, "group_info.json")) as f:
        group_info = json.load(f)

    benchmark_rows = []
    for group_id, info in tqdm(sorted(group_info.items(), key=lambda x: int(x[0])), desc="Decoding groups"):
        frame_start, frame_end = info['name_index']
        num_frames = frame_end - frame_start + 1

        input_group_path = os.path.join(args.input_folder, f"group{group_id}")
        output_group_path = os.path.join(args.output_folder, f"group{group_id}")
        os.makedirs(output_group_path, exist_ok=True)

        # Discover channel indices from MP4 filenames
        channels = []
        for fname in os.listdir(input_group_path):
            if fname.endswith('.mp4'):
                ch_str = fname.replace('.mp4', '')
                if ch_str.isdigit():
                    channels.append(int(ch_str))
        channels.sort()

        t0 = time.perf_counter()
        for ch in tqdm(channels, desc=f"Group {group_id} channels", leave=False):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", os.path.join(input_group_path, f"{ch}.mp4"),
                "-pix_fmt", "gray",
                "-start_number", str(frame_start),
                os.path.join(output_group_path, f"%d_{ch}.png")
            ]
            subprocess.run(cmd, check=True)
        t1 = time.perf_counter()
        benchmark_rows.append({
            "group_id": int(group_id),
            "time_ms": (t1 - t0) * 1000,
            "num_frames": num_frames,
        })

    # Copy metadata JSONs
    for json_file in ["min_max.json", "viewer_min_max.json", "group_info.json"]:
        src = os.path.join(args.input_folder, json_file)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output_folder, json_file))

    # Benchmark CSV and summary
    csv_path = os.path.join(args.output_folder, "benchmark_decompress_video_2_png.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "time_ms", "num_frames"])
        for r in benchmark_rows:
            w.writerow([r["group_id"], f"{r['time_ms']:.2f}", r["num_frames"]])

    total_time_ms = sum(r["time_ms"] for r in benchmark_rows)
    total_frames = sum(r["num_frames"] for r in benchmark_rows)
    print("\n" + "=" * 60)
    print("Benchmark Summary (decompress video to PNG)")
    print("=" * 60)
    print(f"  Groups:           {len(benchmark_rows)}")
    print(f"  Total frames:     {total_frames}")
    print(f"  Total time:       {total_time_ms:.0f} ms")
    print(f"  Avg time/group:   {total_time_ms / len(benchmark_rows):.0f} ms")
    print(f"  Avg time/frame:   {total_time_ms / total_frames:.0f} ms")
    print(f"  CSV: {csv_path}")
    print("=" * 60)
    print("Video decompression complete.")
