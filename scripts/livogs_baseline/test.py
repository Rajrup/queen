#!/usr/bin/env python3
"""
LiVoGS Compression + Decompression for VideoGS-trained Gaussian Splat Models.

For each frame:
  1. Load PLY from VideoGS checkpoint (not timed)
  2. encode_livogs(): Morton → Voxelize → Merge → Position encode → RAHT → Quantize → RLGR
  3. decode_livogs(): RLGR → Dequant → Position decode → RAHT prelude → iRAHT
  4. save_to_ply(): Save reconstructed model to disk (not timed)

The compressed bytestream stays on GPU (no GPU→CPU transfer).
"""

import os
import sys
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from plyfile import PlyData

# --- Setup sys.path for LiVoGS imports ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VIDEOGS_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_LIVOGS_COMPRESSION = os.path.join(_VIDEOGS_ROOT, "LiVoGS", "compression")
if _LIVOGS_COMPRESSION not in sys.path:
    sys.path.insert(0, _LIVOGS_COMPRESSION)

from compress_decompress import encode_livogs, decode_livogs

# ---------------------------------------------------------------------------
# PLY I/O (VideoGS-compatible)
# ---------------------------------------------------------------------------

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder) if "iteration_" in fname]
    return max(saved_iters)


def load_videogs_ply(ply_path, device='cuda'):
    """Load a VideoGS-trained PLY and return LiVoGS-compatible param dict on GPU.

    VideoGS PLY attribute order:
        x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..44, opacity,
        scale_0..2, rot_0..3

    Normals (nx, ny, nz) are always zero in VideoGS and are ignored.
    Opacities are converted from logit to [0,1], scales from log to positive.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    means = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1)
    sh_dc = np.stack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']], axis=1)

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith('f_rest_')],
        key=lambda x: int(x.split('_')[-1])
    )
    if rest_names:
        sh_rest = np.stack([vertex[name] for name in rest_names], axis=1)
    else:
        sh_rest = np.zeros((len(vertex), 0), dtype=np.float32)
    colors = np.concatenate([sh_dc, sh_rest], axis=1)

    opacities = np.asarray(vertex['opacity'])
    scales = np.stack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']], axis=1)
    quats = np.stack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']], axis=1)

    params = {
        'means': torch.from_numpy(means.copy()).float().to(device),
        'quats': torch.from_numpy(quats.copy()).float().to(device),
        'scales': torch.from_numpy(scales.copy()).float().to(device),
        'opacities': torch.from_numpy(opacities.copy()).float().to(device),
        'colors': torch.from_numpy(colors.copy()).float().to(device),
    }

    uncompressed_size_bytes = sum(
        v.numel() * v.element_size() for v in params.values()
    )

    # Normalize quaternions -> Refer to LiVoGS/compression/data_util.py
    params['quats'] = F.normalize(params['quats'], p=2, dim=1)
    # Logit → [0, 1]
    if params['opacities'].min() < 0 or params['opacities'].max() > 1:
        params['opacities'] = torch.sigmoid(params['opacities'])
    # Log → positive
    if params['scales'].min() < 0:
        params['scales'] = torch.exp(params['scales'])

    return params, uncompressed_size_bytes


def save_videogs_ply(params, output_path, sh_degree=3, eps=1e-6):
    """Save reconstructed params back to VideoGS-compatible PLY.

    Converts opacities back to logit space and scales back to log space so that
    GaussianModel.load_ply() can consume them directly.
    """
    means = params['means'].detach().cpu().float().numpy()
    quats = params['quats'].detach().cpu().float().numpy()
    scales = params['scales'].detach().cpu().float().numpy()
    opacities = params['opacities'].detach().cpu().float().numpy()
    colors = params['colors'].detach().cpu().float().numpy()

    N = means.shape[0]

    # Convert back to raw (logit / log) space for VideoGS compatibility
    opacities_c = np.clip(opacities, eps, 1.0 - eps)
    opacities_logit = np.log(opacities_c / (1.0 - opacities_c))
    scales_log = np.log(np.clip(scales, eps, None))

    # Build attribute names matching VideoGS convention
    attr_names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(3):
        attr_names.append(f'f_dc_{i}')
    n_rest = colors.shape[1] - 3
    for i in range(n_rest):
        attr_names.append(f'f_rest_{i}')
    attr_names.append('opacity')
    for i in range(3):
        attr_names.append(f'scale_{i}')
    for i in range(4):
        attr_names.append(f'rot_{i}')

    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate([
        means, normals, colors,
        opacities_logit.reshape(-1, 1),
        scales_log, quats,
    ], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for name in attr_names:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LiVoGS compress + decompress for VideoGS-trained models"
    )
    parser.add_argument("--ply_path", type=str, required=True,
                        help="Path to checkpoint dir containing frame folders (0, 1, ...)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder", type=str, required=True,
                        help="Folder for decompressed PLY output")
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=200)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--sh_degree", type=int, default=3)
    # LiVoGS-specific parameters
    parser.add_argument("--J", type=int, default=15,
                        help="Octree depth for voxelization (default: 15)")
    parser.add_argument("--quantize_step", type=float, default=0.0001,
                        help="Uniform quantization step for all attributes (default: 0.0001)")
    parser.add_argument("--quantize_step_quats", type=float, default=None,
                        help="Override quantization step for quaternions")
    parser.add_argument("--quantize_step_scales", type=float, default=None,
                        help="Override quantization step for scales")
    parser.add_argument("--quantize_step_opacity", type=float, default=None,
                        help="Override quantization step for opacity")
    parser.add_argument("--quantize_step_sh_dc", type=float, default=None,
                        help="Override quantization step for SH DC")
    parser.add_argument("--quantize_step_sh_rest", type=float, default=None,
                        help="Override quantization step for SH rest")
    parser.add_argument("--sh_color_space", type=str, default="rgb",
                        choices=["rgb", "yuv", "klt"],
                        help="Color space for SH coefficients (default: rgb)")
    parser.add_argument("--rlgr_block_size", type=int, default=4096,
                        help="RLGR parallel block size (default: 4096)")
    parser.add_argument("--quantize_config_json", type=str, default=None,
                        help="Path to JSON file with full quantize_config dict (overrides all --quantize_step_* args)")
    parser.add_argument("--nvcomp_algorithm", type=str, default="ANS",
                        choices=["None", "LZ4", "Snappy", "GDeflate", "Deflate",
                                 "zStandard", "Cascaded", "Bitcomp", "ANS"],
                        help="nvCOMP algorithm for octree position compression (default: ANS, 'None' to disable)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    nvcomp_algorithm = None if args.nvcomp_algorithm == "None" else args.nvcomp_algorithm

    # Build quantize_step dict — overridden entirely if --quantize_config_json is provided
    if args.quantize_config_json is not None:
        import json as _json
        with open(args.quantize_config_json) as _f:
            _qp_data = _json.load(_f)
        quantize_step = _qp_data["quantize_config"]
        print(f"  Loaded quantize config: {args.quantize_config_json} (label: {_qp_data.get('label', '?')})")
    else:
        qs = args.quantize_step
        quantize_step = {
            'quats': args.quantize_step_quats if args.quantize_step_quats is not None else qs,
            'scales': args.quantize_step_scales if args.quantize_step_scales is not None else qs,
            'opacity': args.quantize_step_opacity if args.quantize_step_opacity is not None else qs,
            'sh_dc': args.quantize_step_sh_dc if args.quantize_step_sh_dc is not None else qs,
            'sh_rest': [args.quantize_step_sh_rest if args.quantize_step_sh_rest is not None else qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
        }

    device = args.device
    if device.startswith('cuda:'):
        device_id = int(device.split(':')[1])
    else:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_ply_folder, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("LiVoGS Compress + Decompress Pipeline")
    print("=" * 70)
    print(f"  PLY path:           {args.ply_path}")
    print(f"  Output folder:      {args.output_folder}")
    print(f"  Output PLY folder:  {args.output_ply_folder}")
    print(f"  Frames:             {args.frame_start} to {args.frame_end} (interval={args.interval})")
    print(f"  SH degree:          {args.sh_degree}")
    print(f"  Device:             {device}")
    print(f"  J (octree depth):   {args.J}")
    print(f"  Quantize steps:     quats={quantize_step['quats']}, scales={quantize_step['scales']}, "
          f"opacity={quantize_step['opacity']}, sh_dc={quantize_step['sh_dc']}, sh_rest={quantize_step['sh_rest']}")
    print(f"  SH color space:     {args.sh_color_space}")
    print(f"  RLGR block size:    {args.rlgr_block_size}")
    print(f"  nvCOMP algorithm:   {nvcomp_algorithm if nvcomp_algorithm else 'none'}")
    print("=" * 70)

    # Warmup GPU
    print("Warmup GPU...")
    frame = args.frame_start
    ckpt_path = os.path.join(args.ply_path, str(frame), "point_cloud")
    if not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint not found: {ckpt_path}, skipping frame {frame}")
        raise ValueError(f"Checkpoint not found: {ckpt_path}")
    max_iter = searchForMaxIteration(ckpt_path)
    ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")

    params, _ = load_videogs_ply(ply_file_path, device=device)

    torch.cuda.synchronize(device_id)
    compressed_state = encode_livogs(
        params, J=args.J, device=device, device_id=device_id,
        sh_color_space=args.sh_color_space,
        quantize_step=quantize_step,
        rlgr_block_size=args.rlgr_block_size,
        nvcomp_algorithm=nvcomp_algorithm,
    )
    torch.cuda.synchronize(device_id)

    decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)

    torch.cuda.synchronize(device_id)
    print("Warmup GPU done.")

    benchmark_rows = []

    for frame in tqdm(range(args.frame_start, args.frame_end, args.interval), desc="Frames"):

        # --- 1. Load PLY (not timed) ---
        ckpt_path = os.path.join(args.ply_path, str(frame), "point_cloud")
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found: {ckpt_path}, skipping frame {frame}")
            continue
        max_iter = searchForMaxIteration(ckpt_path)
        ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")

        params, uncompressed_size_bytes = load_videogs_ply(ply_file_path, device=device)
        N_original = params['means'].shape[0]

        # --- 2. Encode (timed) ---
        torch.cuda.synchronize(device_id)
        t_enc_start = time.perf_counter()

        compressed_state = encode_livogs(
            params, J=args.J, device=device, device_id=device_id,
            sh_color_space=args.sh_color_space,
            quantize_step=quantize_step,
            rlgr_block_size=args.rlgr_block_size,
            nvcomp_algorithm=nvcomp_algorithm,
        )

        torch.cuda.synchronize(device_id)
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000

        Nvox = compressed_state['Nvox']
        compressed_size_bytes = compressed_state['total_compressed_bytes']
        position_compressed_bytes = compressed_state['position_compressed_bytes']
        attribute_compressed_bytes = compressed_state['attribute_compressed_bytes']

        # --- 3. Decode (timed) ---
        t_dec_start = time.perf_counter()

        decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)

        torch.cuda.synchronize(device_id)
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000

        # --- 4. Save PLY (not timed) ---
        frame_ply_folder = os.path.join(args.output_ply_folder, str(frame), "point_cloud")
        os.makedirs(frame_ply_folder, exist_ok=True)
        ply_out_path = os.path.join(frame_ply_folder, "point_cloud.ply")
        save_videogs_ply(decoded_params, ply_out_path, args.sh_degree)

        benchmark_rows.append({
            "frame": frame,
            "encode_time_ms": encode_time_ms,
            "decode_time_ms": decode_time_ms,
            "original_points": N_original,
            "voxelized_points": Nvox,
            "uncompressed_size_bytes": uncompressed_size_bytes,
            "compressed_size_bytes": compressed_size_bytes,
            "position_compressed_bytes": position_compressed_bytes,
            "attribute_compressed_bytes": attribute_compressed_bytes,
        })

        tqdm.write(
            f"  Frame {frame}: N={N_original}→{Nvox} voxels, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        # Clean up
        del params, compressed_state, decoded_params
        torch.cuda.empty_cache()

    # --- Benchmark CSV and summary ---
    if benchmark_rows:
        csv_path = os.path.join(args.output_folder, "benchmark_livogs.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_id", "encode_time_ms", "decode_time_ms",
                         "original_points", "voxelized_points", 
                         "uncompressed_size_bytes", "compressed_size_bytes",
                         "position_compressed_bytes", "attribute_compressed_bytes"])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"],
                    f"{r['encode_time_ms']:.2f}",
                    f"{r['decode_time_ms']:.2f}",
                    r["original_points"],
                    r["voxelized_points"],
                    r["uncompressed_size_bytes"],
                    r["compressed_size_bytes"],
                    r["position_compressed_bytes"],
                    r["attribute_compressed_bytes"],
                ])

        n = len(benchmark_rows)
        total_enc_ms = sum(r["encode_time_ms"] for r in benchmark_rows)
        total_dec_ms = sum(r["decode_time_ms"] for r in benchmark_rows)
        total_uncomp = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_comp = sum(r["compressed_size_bytes"] for r in benchmark_rows)
        total_position_comp = sum(r["position_compressed_bytes"] for r in benchmark_rows)
        total_attribute_comp = sum(r["attribute_compressed_bytes"] for r in benchmark_rows)
        total_orig_points = sum(r["original_points"] for r in benchmark_rows)
        total_vox_points = sum(r["voxelized_points"] for r in benchmark_rows)

        # Save config JSON for reproducibility
        import json
        config = {
            "J": args.J,
            "quantize_step": quantize_step,
            "sh_color_space": args.sh_color_space,
            "rlgr_block_size": args.rlgr_block_size,
            "sh_degree": args.sh_degree,
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "interval": args.interval,
            "nvcomp_algorithm": nvcomp_algorithm,
        }
        with open(os.path.join(args.output_folder, "livogs_config.json"), "w") as f:
            json.dump(config, f, indent=4)

        print("\n" + "=" * 70)
        print("Benchmark Summary (LiVoGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  (avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  (avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total position compressed size: {total_position_comp / 1024 / 1024:.2f} MB  (avg {total_position_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total attribute compressed size: {total_attribute_comp / 1024 / 1024:.2f} MB  (avg {total_attribute_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  Avg point reduction:       {total_orig_points / n:.0f} → {total_vox_points / n:.0f} "
              f"({total_orig_points / total_vox_points:.2f}x)")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
