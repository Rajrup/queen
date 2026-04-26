#!/usr/bin/env python3
"""LiVoGS Compression + Decompression for VideoGS-trained Gaussian Splat Models.

For each frame:
  1. Load PLY from VideoGS checkpoint (not timed)
  2. encode_livogs(): Morton → Voxelize → Merge → Position encode → RAHT → Quantize → RLGR
  3. decode_livogs(): RLGR → Dequant → Position decode → RAHT prelude → iRAHT
  4. save_to_ply(): Save reconstructed model to disk (not timed)

Usable as a library (``codec.compress_decompress(...)``) or standalone CLI::

    python scripts/livogs_baseline/rd_pipeline/codec.py --ply_path ... --output_folder ...
"""

import csv
import json
import os
import sys
import time
from typing import Any, Optional

# -- config import (must come before LiVoGS imports) --------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.setup_livogs_imports()

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from plyfile import PlyData

from compress_decompress import encode_livogs, decode_livogs
from utils.system_utils import searchForMaxIteration


# ---------------------------------------------------------------------------
# PLY I/O (VideoGS-compatible)
# ---------------------------------------------------------------------------

def load_videogs_ply(ply_path: str, device: str = "cuda") -> tuple[dict[str, torch.Tensor], int]:
    """Load a VideoGS-trained PLY and return LiVoGS-compatible param dict on GPU.

    VideoGS PLY attribute order::

        x, y, z, nx, ny, nz, f_dc_0..2, f_rest_0..44, opacity,
        scale_0..2, rot_0..3

    Normals (nx, ny, nz) are always zero in VideoGS and are ignored.
    Opacities are converted from logit to [0,1], scales from log to positive.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    means = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    sh_dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1)

    rest_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if rest_names:
        sh_rest = np.stack([vertex[name] for name in rest_names], axis=1)
    else:
        sh_rest = np.zeros((len(vertex), 0), dtype=np.float32)
    colors = np.concatenate([sh_dc, sh_rest], axis=1)

    opacities = np.asarray(vertex["opacity"])
    scales = np.stack([vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]], axis=1)
    quats = np.stack([vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]], axis=1)

    params = {
        "means":     torch.from_numpy(means.copy()).float().to(device),
        "quats":     torch.from_numpy(quats.copy()).float().to(device),
        "scales":    torch.from_numpy(scales.copy()).float().to(device),
        "opacities": torch.from_numpy(opacities.copy()).float().to(device),
        "colors":    torch.from_numpy(colors.copy()).float().to(device),
    }

    uncompressed_size_bytes = sum(v.numel() * v.element_size() for v in params.values())

    # Normalize quaternions
    params["quats"] = F.normalize(params["quats"], p=2, dim=1)
    # Logit → [0, 1]
    if params["opacities"].min() < 0 or params["opacities"].max() > 1:
        params["opacities"] = torch.sigmoid(params["opacities"])
    # Log → positive
    if params["scales"].min() < 0:
        params["scales"] = torch.exp(params["scales"])

    return params, uncompressed_size_bytes


def save_videogs_ply(
    params: dict[str, torch.Tensor],
    output_path: str,
    sh_degree: int = 3,
    eps: float = 1e-6,
) -> None:
    """Save reconstructed params back to VideoGS-compatible PLY.

    Converts opacities back to logit space and scales back to log space so that
    ``GaussianModel.load_ply()`` can consume them directly.
    """
    means = params["means"].detach().cpu().float().numpy()
    quats = params["quats"].detach().cpu().float().numpy()
    scales = params["scales"].detach().cpu().float().numpy()
    opacities = params["opacities"].detach().cpu().float().numpy()
    colors = params["colors"].detach().cpu().float().numpy()

    N = means.shape[0]

    opacities_c = np.clip(opacities, eps, 1.0 - eps)
    opacities_logit = np.log(opacities_c / (1.0 - opacities_c))
    scales_log = np.log(np.clip(scales, eps, None))

    attr_names = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attr_names.append(f"f_dc_{i}")
    n_rest = colors.shape[1] - 3
    for i in range(n_rest):
        attr_names.append(f"f_rest_{i}")
    attr_names.append("opacity")
    for i in range(3):
        attr_names.append(f"scale_{i}")
    for i in range(4):
        attr_names.append(f"rot_{i}")

    normals = np.zeros((N, 3), dtype=np.float32)
    data = np.concatenate([
        means, normals, colors,
        opacities_logit.reshape(-1, 1),
        scales_log, quats,
    ], axis=1).astype(np.float32)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {N}\n".encode())
        for name in attr_names:
            f.write(f"property float {name}\n".encode())
        f.write(b"end_header\n")
        f.write(data.tobytes())



def _per_channel_column_names(n_channels: int) -> list[str]:
    """Generate CSV column names for per-dimension compressed bytes.

    Channel layout: quats(0:4), scales(4:7), opacity(7), sh(8:).
    """
    names: list[str] = []
    for i in range(4):
        names.append(f"quats_dim{i}_compressed_bytes")
    for i in range(3):
        names.append(f"scales_dim{i}_compressed_bytes")
    names.append("opacity_dim0_compressed_bytes")
    num_sh = n_channels - 8
    for i in range(num_sh):
        names.append(f"sh_dim{i}_compressed_bytes")
    assert len(names) == n_channels
    return names

# ---------------------------------------------------------------------------
# Main compress + decompress pipeline
# ---------------------------------------------------------------------------

def compress_decompress(
    ply_path: str,
    output_folder: str,
    output_ply_folder: str,
    frame_start: int,
    frame_end: int,
    interval: int = 1,
    sh_degree: int = config.SH_DEGREE,
    J: int = config.J,
    quantize_step: Optional[dict[str, Any]] = None,
    sh_color_space: str = config.SH_COLOR_SPACE,
    rlgr_block_size: int = config.RLGR_BLOCK_SIZE,
    device: str = config.DEVICE,
    skip_save_ply: bool = False,
    nvcomp_algorithm: Optional[str] = config.NVCOMP_ALGORITHM,
) -> list[dict[str, Any]]:
    """Run LiVoGS compress + decompress for a range of frames.

    Writes ``benchmark_livogs.csv`` and ``livogs_config.json`` to *output_folder*.
    Returns list of per-frame benchmark dicts.
    """
    if quantize_step is None:
        qs = 0.0001
        quantize_step = {
            "quats": qs, "scales": qs, "opacity": qs,
            "sh_dc": qs,
            "sh_rest": [qs] * (3 * ((sh_degree + 1) ** 2 - 1)),
        }

    device_id: int
    if device.startswith("cuda:"):
        device_id = int(device.split(":")[1])
    else:
        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_ply_folder, exist_ok=True)

    # --- Print configuration ---
    print("=" * 70)
    print("LiVoGS Compress + Decompress Pipeline")
    print("=" * 70)
    print(f"  PLY path:           {ply_path}")
    print(f"  Output folder:      {output_folder}")
    print(f"  Output PLY folder:  {output_ply_folder}")
    print(f"  Frames:             {frame_start} to {frame_end} (interval={interval})")
    print(f"  SH degree:          {sh_degree}")
    print(f"  Device:             {device}")
    print(f"  J (octree depth):   {J}")
    print(f"  Quantize steps:     quats={quantize_step['quats']}, scales={quantize_step['scales']}, "
          f"opacity={quantize_step['opacity']}, sh_dc={quantize_step['sh_dc']}, sh_rest={quantize_step['sh_rest']}")
    print(f"  SH color space:     {sh_color_space}")
    print(f"  RLGR block size:    {rlgr_block_size}")
    print(f"  nvCOMP algorithm:   {nvcomp_algorithm if nvcomp_algorithm else 'none'}")
    print("=" * 70)

    # --- Warmup GPU ---
    print("Warmup GPU...")
    frame = frame_start
    ckpt_path = os.path.join(ply_path, str(frame), "point_cloud")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint folder not found for frame {frame}: {ckpt_path}")
    max_iter = searchForMaxIteration(ckpt_path)
    ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")

    params, _ = load_videogs_ply(ply_file_path, device=device)
    torch.cuda.synchronize(device_id)
    compressed_state = encode_livogs(
        params, J=J, device=device, device_id=device_id,
        sh_color_space=sh_color_space,
        quantize_step=quantize_step, rlgr_block_size=rlgr_block_size,
        nvcomp_algorithm=nvcomp_algorithm,
    )
    torch.cuda.synchronize(device_id)
    decode_livogs(compressed_state, device=device, device_id=device_id)
    torch.cuda.synchronize(device_id)
    print("Warmup GPU done.")

    # --- Frame loop ---
    benchmark_rows: list[dict[str, Any]] = []

    for frame in tqdm(range(frame_start, frame_end, interval), desc="Frames"):
        ckpt_path = os.path.join(ply_path, str(frame), "point_cloud")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint folder not found for frame {frame}: {ckpt_path}")
        max_iter = searchForMaxIteration(ckpt_path)
        ply_file_path = os.path.join(ckpt_path, f"iteration_{max_iter}", "point_cloud.ply")

        params, uncompressed_size_bytes = load_videogs_ply(ply_file_path, device=device)
        N_original = params["means"].shape[0]

        # Encode (timed)
        torch.cuda.synchronize(device_id)
        t_enc_start = time.perf_counter()
        compressed_state = encode_livogs(
            params, J=J, device=device, device_id=device_id,
            sh_color_space=sh_color_space,
            quantize_step=quantize_step, rlgr_block_size=rlgr_block_size,
            nvcomp_algorithm=nvcomp_algorithm,
        )
        torch.cuda.synchronize(device_id)
        t_enc_end = time.perf_counter()
        encode_time_ms = (t_enc_end - t_enc_start) * 1000

        Nvox = compressed_state["Nvox"]
        compressed_size_bytes = compressed_state["total_compressed_bytes"]
        position_compressed_bytes = compressed_state["position_compressed_bytes"]
        attribute_compressed_bytes = compressed_state["attribute_compressed_bytes"]
        per_channel_compressed_bytes = compressed_state["per_channel_compressed_bytes"]

        # Decode (timed)
        t_dec_start = time.perf_counter()
        decoded_params = decode_livogs(compressed_state, device=device, device_id=device_id)
        torch.cuda.synchronize(device_id)
        t_dec_end = time.perf_counter()
        decode_time_ms = (t_dec_end - t_dec_start) * 1000

        # Save PLY (not timed)
        if not skip_save_ply:
            frame_ply_folder = os.path.join(output_ply_folder, str(frame), "point_cloud")
            os.makedirs(frame_ply_folder, exist_ok=True)
            ply_out_path = os.path.join(frame_ply_folder, "point_cloud.ply")
            save_videogs_ply(decoded_params, ply_out_path, sh_degree)

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
            "per_channel_compressed_bytes": per_channel_compressed_bytes,
        })

        tqdm.write(
            f"  Frame {frame}: N={N_original}→{Nvox} voxels, "
            f"enc={encode_time_ms:.2f} ms, dec={decode_time_ms:.2f} ms, "
            f"uncomp={uncompressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"comp={compressed_size_bytes / 1024 / 1024:.2f} MB, "
            f"ratio={uncompressed_size_bytes / compressed_size_bytes:.2f}x"
        )

        del params, compressed_state, decoded_params
        torch.cuda.empty_cache()

    # --- Benchmark CSV and summary ---
    if benchmark_rows:
        csv_path = os.path.join(output_folder, "benchmark_livogs.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            per_ch_cols = _per_channel_column_names(len(benchmark_rows[0]["per_channel_compressed_bytes"]))
            w.writerow([
                "frame_id", "encode_time_ms", "decode_time_ms",
                "original_points", "voxelized_points",
                "uncompressed_size_bytes", "compressed_size_bytes",
                "position_compressed_bytes", "attribute_compressed_bytes",
                *per_ch_cols,
            ])
            for r in benchmark_rows:
                w.writerow([
                    r["frame"], f"{r['encode_time_ms']:.2f}", f"{r['decode_time_ms']:.2f}",
                    r["original_points"], r["voxelized_points"],
                    r["uncompressed_size_bytes"], r["compressed_size_bytes"],
                    r["position_compressed_bytes"], r["attribute_compressed_bytes"],
                    *r["per_channel_compressed_bytes"],
                ])

        # Save config JSON for reproducibility
        livogs_cfg = {
            "J": J,
            "quantize_step": quantize_step,
            "sh_color_space": sh_color_space,
            "rlgr_block_size": rlgr_block_size,
            "sh_degree": sh_degree,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "interval": interval,
            "nvcomp_algorithm": nvcomp_algorithm,
        }
        with open(os.path.join(output_folder, "livogs_config.json"), "w") as f:
            json.dump(livogs_cfg, f, indent=4)

        n = len(benchmark_rows)
        total_enc_ms  = sum(r["encode_time_ms"] for r in benchmark_rows)
        total_dec_ms  = sum(r["decode_time_ms"] for r in benchmark_rows)
        total_uncomp  = sum(r["uncompressed_size_bytes"] for r in benchmark_rows)
        total_comp    = sum(r["compressed_size_bytes"] for r in benchmark_rows)
        total_pos     = sum(r["position_compressed_bytes"] for r in benchmark_rows)
        total_attr    = sum(r["attribute_compressed_bytes"] for r in benchmark_rows)
        per_ch = [r["per_channel_compressed_bytes"] for r in benchmark_rows]
        total_quats   = sum(sum(ch[0:4]) for ch in per_ch)
        total_scales  = sum(sum(ch[4:7]) for ch in per_ch)
        total_opacity = sum(ch[7] for ch in per_ch)
        total_sh_dc   = sum(sum(ch[8:11]) for ch in per_ch)
        total_sh_rest = sum(sum(ch[11:]) for ch in per_ch)
        total_orig    = sum(r["original_points"] for r in benchmark_rows)
        total_vox     = sum(r["voxelized_points"] for r in benchmark_rows)

        print("\n" + "=" * 70)
        print("Benchmark Summary (LiVoGS compress + decompress)")
        print("=" * 70)
        print(f"  Frames processed:          {n}")
        print(f"  Total encode time:         {total_enc_ms / 1000:.2f} s  (avg {total_enc_ms / n:.2f} ms/frame)")
        print(f"  Total decode time:         {total_dec_ms / 1000:.2f} s  (avg {total_dec_ms / n:.2f} ms/frame)")
        print(f"  Total uncompressed size:   {total_uncomp / 1024 / 1024:.2f} MB  (avg {total_uncomp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total compressed size:     {total_comp / 1024 / 1024:.2f} MB  (avg {total_comp / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total position compressed: {total_pos / 1024 / 1024:.2f} MB  (avg {total_pos / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Total attribute compressed: {total_attr / 1024 / 1024:.2f} MB  (avg {total_attr / n / 1024 / 1024:.2f} MB/frame)")
        print(f"    - quats:   {total_quats / 1024 / 1024:.2f} MB  (avg {total_quats / n / 1024 / 1024:.2f} MB/frame)")
        print(f"    - scales:  {total_scales / 1024 / 1024:.2f} MB  (avg {total_scales / n / 1024 / 1024:.2f} MB/frame)")
        print(f"    - opacity: {total_opacity / 1024 / 1024:.2f} MB  (avg {total_opacity / n / 1024 / 1024:.2f} MB/frame)")
        print(f"    - sh_dc:   {total_sh_dc / 1024 / 1024:.2f} MB  (avg {total_sh_dc / n / 1024 / 1024:.2f} MB/frame)")
        print(f"    - sh_rest: {total_sh_rest / 1024 / 1024:.2f} MB  (avg {total_sh_rest / n / 1024 / 1024:.2f} MB/frame)")
        print(f"  Compression ratio:         {total_uncomp / total_comp:.2f}x")
        print(f"  Avg point reduction:       {total_orig / n:.0f} \u2192 {total_vox / n:.0f} "
              f"({total_orig / total_vox:.2f}x)")
        print(f"  CSV: {csv_path}")
        print("=" * 70)
    else:
        print("No frames were processed.")

    print("Done.")
    return benchmark_rows


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LiVoGS compress + decompress for VideoGS-trained models",
    )
    parser.add_argument("--ply_path",           type=str, required=True,
                        help="Path to checkpoint dir containing frame folders (0, 1, ...)")
    parser.add_argument("--output_folder",      type=str, required=True,
                        help="Folder for benchmark CSV and metadata")
    parser.add_argument("--output_ply_folder",  type=str, required=True,
                        help="Folder for decompressed PLY output")
    parser.add_argument("--frame_start",        type=int, default=0)
    parser.add_argument("--frame_end",          type=int, default=200)
    parser.add_argument("--interval",           type=int, default=1)
    parser.add_argument("--sh_degree",          type=int, default=config.SH_DEGREE)
    parser.add_argument("--J",                  type=int, default=config.J,
                        help="Octree depth for voxelization")
    parser.add_argument("--quantize_step",      type=float, default=0.0001,
                        help="Uniform quantization step for all attributes")
    parser.add_argument("--quantize_step_quats",   type=float, default=None)
    parser.add_argument("--quantize_step_scales",  type=float, default=None)
    parser.add_argument("--quantize_step_opacity", type=float, default=None)
    parser.add_argument("--quantize_step_sh_dc",   type=float, default=None)
    parser.add_argument("--quantize_step_sh_rest", type=float, default=None)
    parser.add_argument("--sh_color_space",     type=str, default=config.SH_COLOR_SPACE,
                        choices=["rgb", "yuv", "klt"])
    parser.add_argument("--rlgr_block_size",    type=int, default=config.RLGR_BLOCK_SIZE)
    parser.add_argument("--quantize_config_json", type=str, default=None,
                        help="Path to JSON with full quantize_config (overrides --quantize_step_*)")
    parser.add_argument("--device",             type=str, default=config.DEVICE)
    parser.add_argument("--skip_save_ply",      action="store_true",
                        help="Skip saving decompressed PLY files (fast mode)")
    args = parser.parse_args()

    # Build quantize_step dict
    if args.quantize_config_json is not None:
        with open(args.quantize_config_json) as _f:
            _qp_data = json.load(_f)
        qs_dict = _qp_data["quantize_config"]
        print(f"  Loaded quantize config: {args.quantize_config_json} (label: {_qp_data.get('label', '?')})")
    else:
        qs = args.quantize_step
        qs_dict = {
            "quats":   args.quantize_step_quats   or qs,
            "scales":  args.quantize_step_scales  or qs,
            "opacity": args.quantize_step_opacity or qs,
            "sh_dc":   args.quantize_step_sh_dc   or qs,
            "sh_rest": [args.quantize_step_sh_rest or qs] * (3 * ((args.sh_degree + 1) ** 2 - 1)),
        }

    compress_decompress(
        ply_path=args.ply_path,
        output_folder=args.output_folder,
        output_ply_folder=args.output_ply_folder,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        interval=args.interval,
        sh_degree=args.sh_degree,
        J=args.J,
        quantize_step=qs_dict,
        sh_color_space=args.sh_color_space,
        rlgr_block_size=args.rlgr_block_size,
        device=args.device,
        skip_save_ply=args.skip_save_ply,
    )
