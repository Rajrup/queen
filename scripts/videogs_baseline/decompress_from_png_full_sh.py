import os
import csv
import time
import numpy as np
import json
import argparse
import sys
from tqdm import tqdm

# --- Setup sys.path for LiVoGS imports ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_QUEEN_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))   
_VIDEOGS_COMPRESSION = os.path.join(_QUEEN_ROOT, "VideoGS", "compression")

if _VIDEOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _VIDEOGS_COMPRESSION)

from compress_decompress import decode_videogs_png, dequantize_videogs_image

def save_ply(data, output_file, sh_degree):
    """Write a QUEEN-compatible PLY with vertex_id from a flat data matrix.

    Attribute order: x,y,z, nx,ny,nz, f_dc_0..2, f_rest_0..n, opacity,
    scale_0..2, rot_0..3, vertex_id (int).
    """
    n, k = data.shape

    attribute_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(3):
        attribute_names.append(f'f_dc_{i}')

    # k = 3(pos) + 3(norm) + 3(dc) + n_rest + 1(op) + 3(scale) + 4(rot) = 17 + n_rest
    n_rest = k - 17
    for i in range(n_rest):
        attribute_names.append(f'f_rest_{i}')

    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append(f'scale_{i}')
    for i in range(4):
        attribute_names.append(f'rot_{i}')

    assert k == len(attribute_names), f"Shape mismatch: data has {k} cols, expected {len(attribute_names)}"

    vertex_ids = np.arange(n, dtype=np.int32)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as ply_file:
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(f"element vertex {n}\n".encode())

        for name in attribute_names:
            ply_file.write(f"property float {name}\n".encode())
        ply_file.write(b"property int vertex_id\n")

        ply_file.write(b"end_header\n")

        for i in range(n):
            ply_file.write(data[i].astype(np.float32).tobytes())
            ply_file.write(vertex_ids[i].tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compressed_folder", type=str, required=True, help="Folder containing min_max.json and group folders")
    parser.add_argument("--output_ply_folder", type=str, required=True)
    parser.add_argument("--sh_degree", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.output_ply_folder):
        os.makedirs(args.output_ply_folder)

    # Load Metadata
    with open(os.path.join(args.compressed_folder, "min_max.json"), "r") as f:
        min_max_info = json.load(f)
    
    with open(os.path.join(args.compressed_folder, "group_info.json"), "r") as f:
        group_info = json.load(f)

    benchmark_rows = []
    # Iterate Groups
    for group_id, info in tqdm(group_info.items(), desc="Decompressing Groups"):
        frame_start, frame_end = info['name_index']
        group_folder = os.path.join(args.compressed_folder, f"group{group_id}")
        
        for frame in tqdm(range(frame_start, frame_end + 1), desc=f"Group {group_id}", leave=False):
            
            # 1. Decode PNGs
            # num_attributes = 3(pos) + 3(norm) + 3(dc) + n_rest + 1(op) + 3(scale) + 4(rot)
            n_rest = (args.sh_degree + 1) ** 2 * 3 - 3
            num_attrs = 3 + 3 + 3 + n_rest + 1 + 3 + 4
            t0 = time.perf_counter()
            images = decode_videogs_png(group_folder, frame, num_attributes=num_attrs)
            if not images:
                print(f"No images found for frame {frame}")
                continue
            ply_data = dequantize_videogs_image(images, frame, min_max_info)
            t1 = time.perf_counter()
            time_ms = (t1 - t0) * 1000
            benchmark_rows.append({"frame": frame, "time_ms": time_ms})

            # Save PLY (not timed)
            frame_str = str(frame).zfill(4)
            frame_ply_folder = os.path.join(args.output_ply_folder, "frames", frame_str)
            ply_out_path = os.path.join(frame_ply_folder, "point_cloud.ply")
            save_ply(ply_data, ply_out_path, args.sh_degree)

    # Benchmark CSV and summary
    if benchmark_rows:
        csv_path = os.path.join(args.output_ply_folder, "benchmark_decompress_from_png.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "time_ms"])
            for r in benchmark_rows:
                w.writerow([r["frame"], f"{r['time_ms']:.2f}"])
        total_ms = sum(r["time_ms"] for r in benchmark_rows)
        n = len(benchmark_rows)
        print("\n" + "=" * 60)
        print("Benchmark Summary (decompress PNG to PLY, excl. save)")
        print("=" * 60)
        print(f"  Frames:           {n}")
        print(f"  Total time:       {total_ms / 1000:.2f} s")
        print(f"  Avg time/frame:   {total_ms / n:.2f} ms")
        print(f"  CSV: {csv_path}")
        print("=" * 60)

