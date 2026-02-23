import os
import csv
import time
import numpy as np
import cv2
from plyfile import PlyData
import json
import argparse
from tqdm import tqdm

def normalize_uint8(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8), min_val, max_val

def normalize_uint16(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint16), min_val, max_val
    normalized = (data - min_val) / (max_val - min_val) * (2 ** 16 - 1)
    return normalized.astype(np.uint16), min_val, max_val

def get_ply_matrix(file_path):
    plydata = PlyData.read(file_path)
    vertex = plydata['vertex']
    float_names = [p.name for p in vertex.properties if p.name != 'vertex_id']
    num_vertices = len(vertex)
    data_matrix = np.zeros((num_vertices, len(float_names)), dtype=np.float32)
    for i, name in enumerate(float_names):
        data_matrix[:, i] = vertex[name]
    # Exclude normals (nx, ny, nz) from uncompressed size
    n_float = len(float_names) - 3
    uncompressed_size_bytes = num_vertices * n_float * np.dtype(np.float32).itemsize
    return data_matrix, uncompressed_size_bytes

def calculate_image_size(num_points):
    image_size = 8
    while image_size * image_size < num_points:
        image_size += 8
    return image_size

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)

def find_queen_ply_path(ply_root, frame):
    """Locate QUEEN PLY: try canonical path first, then per-iteration."""
    frame_str = str(frame).zfill(4)
    frame_dir = os.path.join(ply_root, "frames", frame_str)
    canonical = os.path.join(frame_dir, "point_cloud.ply")
    if os.path.exists(canonical):
        return canonical
    ckpt_path = os.path.join(frame_dir, "point_cloud")
    if os.path.exists(ckpt_path):
        max_iter = searchForMaxIteration(ckpt_path)
        return os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")
    return None

def quantize_videogs_image(current_data, image_size):
    num_attributes = current_data.shape[1]
    images = {}
    min_max_info = {}
    
    for i in range(num_attributes):
        # Position attributes (0, 1, 2) -> uint16 split
        if i < 3:
            attribute_data, min_val, max_val = normalize_uint16(current_data[:, i])
            min_max_info[f'{i}_min'] = float(min_val)
            min_max_info[f'{i}_max'] = float(max_val)
            
            attribute_data_reshaped = attribute_data.reshape(-1, 1)
            image_odd = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            image_even = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            
            # Even = Low Byte, Odd = High Byte
            image_even[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped & 0xff)
            image_odd[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped >> 8)
            
            images[f"{2*i}"] = image_even.reshape((image_size, image_size))
            images[f"{2*i+1}"] = image_odd.reshape((image_size, image_size))
            
        else:
            attribute_data, min_val, max_val = normalize_uint8(current_data[:, i])
            min_max_info[f'{i}_min'] = float(min_val)
            min_max_info[f'{i}_max'] = float(max_val)
            
            attribute_data_reshaped = attribute_data.reshape(-1, 1)
            image = np.zeros((image_size * image_size, 1), dtype=np.uint8)
            image[:attribute_data_reshaped.shape[0], :] = attribute_data_reshaped
            
            # Offset index by +3 to match VideoGS convention (normals start at 6, etc.)
            # But wait, if we are just compressing generic attributes, we should just map i -> output_index
            # VideoGS convention: 
            # i=0 (x) -> 0, 1
            # i=1 (y) -> 2, 3
            # i=2 (z) -> 4, 5
            # i=3 (nx) -> 6
            # ...
            images[f"{i+3}"] = image.reshape((image_size, image_size))
            
    return images, min_max_info

def encode_videogs_png(images, output_path, frame_idx):
    for key, img in images.items():
        cv2.imwrite(os.path.join(output_path, f"{frame_idx}_{key}.png"), img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=300)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--ply_path", type=str, required=True, help="Path to training output containing frame folders (0, 1, ...)")
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--sh_degree", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    min_max_json = {}
    viewer_min_max_json = {}
    group_info_json = {}
    benchmark_rows = []  # frame_id, time_ms, compressed_size_bytes, num_points

    # Calculate number of groups (frame_end is inclusive)
    num_frames = args.frame_end - args.frame_start + 1
    num_groups = (num_frames + args.group_size - 1) // args.group_size

    for group in tqdm(range(num_groups), desc="Compressing Groups"):
        frame_start = group * args.group_size + args.frame_start
        frame_end = min((group + 1) * args.group_size - 1 + args.frame_start, args.frame_end)
        
        if frame_start > args.frame_end:
            break

        group_info_json[str(group)] = {}
        group_info_json[str(group)]['frame_index'] = [group * args.group_size, (group + 1) * args.group_size - 1]
        group_info_json[str(group)]['name_index'] = [frame_start, frame_end]

        output_path = os.path.join(args.output_folder, f"group{group}")
        os.makedirs(output_path, exist_ok=True)

        # Pre-scan: find max num_points in this group so all frames share one image_size
        # This is a hack to make all frames share the same image_size, otherwise decompression fails on this Neural_3D_Video dataset
        group_max_points = 0
        for frame in range(frame_start, frame_end + 1, args.interval):
            ply_path = find_queen_ply_path(args.ply_path, frame)
            if ply_path is not None:
                plydata = PlyData.read(ply_path)
                group_max_points = max(group_max_points, len(plydata['vertex']))
        image_size = calculate_image_size(num_points=group_max_points)

        for frame in tqdm(range(frame_start, frame_end + 1, args.interval), desc=f"Group {group} Frames", leave=False):
            
            ply_file_path = find_queen_ply_path(args.ply_path, frame)
            if ply_file_path is None:
                print(f"Warning: PLY not found for frame {str(frame).zfill(4)}")
                continue
            
            # Read PLY (not timed)
            current_data, uncompressed_size_bytes = get_ply_matrix(ply_file_path)
            original_points = current_data.shape[0]

            # Time only quantize + encode to PNG
            t0 = time.perf_counter()
            min_max_json[f'{frame}_num'] = original_points
            viewer_min_max_json[frame] = {}
            viewer_min_max_json[frame]['num'] = original_points
            viewer_min_max_json[frame]['info'] = []
            images, frame_min_max = quantize_videogs_image(current_data, image_size)
            encode_videogs_png(images, output_path, frame)
            t1 = time.perf_counter()
            time_ms = (t1 - t0) * 1000

            # Compressed size for this frame (sum of PNGs for this frame)
            frame_size = 0
            for fname in os.listdir(output_path):
                if fname.startswith(f"{frame}_") and fname.endswith(".png"):
                    frame_size += os.path.getsize(os.path.join(output_path, fname))

            benchmark_rows.append({"frame": frame, "time_ms": time_ms, "uncompressed_size_bytes": uncompressed_size_bytes, "compressed_size_bytes": frame_size, "original_points": original_points})

            # Update global min_max with frame info
            for k, v in frame_min_max.items():
                min_max_json[f'{frame}_{k}'] = v
                viewer_min_max_json[frame]['info'].append(v)

            tqdm.write(
                f"  Frame {frame}: N={original_points}, "
                f"enc={time_ms:.2f} ms, "
                f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
                f"comp={frame_size / 1024 / 1024:.2f} MB, "
                f"ratio={uncompressed_size_bytes / frame_size:.2f}x"
            )

    # Save Metadata
    with open(os.path.join(args.output_folder, "min_max.json"), "w") as f:
        json.dump(min_max_json, f, indent=4)

    with open(os.path.join(args.output_folder, "viewer_min_max.json"), "w") as f:
        json.dump(viewer_min_max_json, f, indent=4)

    with open(os.path.join(args.output_folder, "group_info.json"), "w") as f:
        json.dump(group_info_json, f, indent=4)

    # Benchmark CSV and summary
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_compress_to_png.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "time_ms", "uncompressed_size_bytes", "compressed_size_bytes", "original_points"])
            for r in benchmark_rows:
                w.writerow([r["frame"], f"{r['time_ms']:.2f}", r["uncompressed_size_bytes"], r["compressed_size_bytes"], r["original_points"]])
        total_time_ms = sum(r["time_ms"] for r in benchmark_rows)
        total_uncompressed_size = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_compressed_size = sum(r["compressed_size_bytes"] for r in benchmark_rows)
        n = len(benchmark_rows)
        print("\n" + "=" * 60)
        print("Benchmark Summary (compress to PNG)")
        print("=" * 60)
        print(f"  Frames processed:       {n}")
        print(f"  Total time (excl PLY):  {total_time_ms / 1000:.2f} s")
        print(f"  Avg time per frame:     {total_time_ms / n:.2f} ms")
        print(f"  Total uncompressed:     {total_uncompressed_size / 1024 / 1024:.2f} MB")
        print(f"  Avg uncompressed per frame: {total_uncompressed_size / n / 1024 / 1024:.2f} MB")
        print(f"  Total PNG size:         {total_compressed_size / 1024 / 1024:.2f} MB")
        print(f"  Avg size per frame:     {total_compressed_size / n / 1024 / 1024:.2f} MB")
        print(f"  CSV: {csv_path}")
        print("=" * 60)
        
    print("Compression Complete.")
